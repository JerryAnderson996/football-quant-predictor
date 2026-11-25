"""Streamlit quantitative dashboard for Football Quant Predictor.

Run with:
    streamlit run src/football_predictor/dashboard/app.py
"""

from __future__ import annotations

import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from football_predictor.betting.devig import devig_odds, overround
from football_predictor.betting.strategy import evaluate_match_bets, select_best_value_bet
from football_predictor.config import load_settings
from football_predictor.db.duckdb_manager import DuckDBManager
from football_predictor.features.feature_store import PointInTimeFeatureStore
from football_predictor.ingestion.entity_resolution import TeamResolver

st.set_page_config(
    page_title="Football Quant Predictor | Serie A",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def load_resources() -> dict[str, object]:
    """Load settings, Silver data, trained Ensemble bundle, and backtest artifacts."""
    settings = load_settings()
    resolver = TeamResolver(settings.paths.team_mapping_path)
    feature_store = PointInTimeFeatureStore(config=settings.features)

    silver_df = pd.DataFrame()
    if settings.paths.duckdb_path.exists():
        db = DuckDBManager(settings.paths.duckdb_path)
        silver_df = db.read_silver_matches(league=settings.ingestion.league_code)

    bundle_path = settings.paths.artifacts_dir / "model_bundle.joblib"
    bundle = joblib.load(bundle_path) if bundle_path.exists() else None

    report_path = settings.paths.artifacts_dir / "backtest_report.json"
    backtest_report = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    )

    equity_path = settings.paths.artifacts_dir / "backtest_equity_curve.parquet"
    equity_df = pd.read_parquet(equity_path) if equity_path.exists() else pd.DataFrame()

    return {
        "settings": settings,
        "resolver": resolver,
        "feature_store": feature_store,
        "silver_df": silver_df,
        "bundle": bundle,
        "backtest_report": backtest_report,
        "equity_df": equity_df,
    }


def render_match_predictor_tab(res: dict[str, object]) -> None:
    """Tab 1: Match Predictor with Dixon-Coles heatmap, derived markets, and Kelly sizing."""
    bundle = res["bundle"]
    silver_df: pd.DataFrame = res["silver_df"]  # type: ignore[assignment]
    feature_store: PointInTimeFeatureStore = res["feature_store"]  # type: ignore[assignment]
    settings = res["settings"]

    if bundle is None or silver_df.empty:
        st.error(
            "Model bundle or Silver database not found. Run `scripts/run_ingestion.py` "
            "and `scripts/run_training.py` first."
        )
        return

    ensemble = bundle["ensemble"]
    ratings_df = ensemble.dc_model.get_team_ratings()
    team_map = dict(zip(ratings_df["team"], ratings_df["team_id"], strict=True))
    team_names = sorted(team_map.keys())

    st.subheader("Match Selection & Bookmaker Market")
    c1, c2, c3 = st.columns([2, 2, 1.5])
    default_home = team_names.index("Inter") if "Inter" in team_names else 0
    default_away = team_names.index("AC Milan") if "AC Milan" in team_names else 1

    home_name = c1.selectbox("Home Team", team_names, index=default_home)
    away_name = c2.selectbox("Away Team", team_names, index=default_away)
    as_of = c3.date_input("Match Date", value=date.today())

    if home_name == away_name:
        st.warning("Select two distinct teams.")
        return

    o1, o2, o3, o4 = st.columns([1.2, 1.2, 1.2, 1.4])
    odd_h = o1.number_input("Bookmaker Odd 1 (Home)", min_value=1.02, value=1.95, step=0.05)
    odd_d = o2.number_input("Bookmaker Odd X (Draw)", min_value=1.02, value=3.55, step=0.05)
    odd_a = o3.number_input("Bookmaker Odd 2 (Away)", min_value=1.02, value=4.10, step=0.05)
    bankroll = o4.number_input(
        "Bankroll (€)",
        min_value=10.0,
        value=float(settings.betting.initial_bankroll),
        step=100.0,
    )

    p1, p2, p3 = st.columns(3)
    devig_method = p1.selectbox("Devigging Method", ["power", "multiplicative"], index=0)
    min_ev_pct = p2.slider("Min EV Threshold (%)", 0.0, 15.0, 3.0, 0.5)
    kelly_mult = p3.slider("Fractional Kelly Multiplier", 0.05, 1.0, 0.25, 0.05)

    home_id = team_map[home_name]
    away_id = team_map[away_name]
    feats_row = feature_store.build_inference_row(
        silver_df=silver_df,
        home_team_id=home_id,
        away_team_id=away_id,
        as_of_date=as_of,
    )
    pred = ensemble.predict_match_full(
        home_team_id=home_id,
        away_team_id=away_id,
        features_row=feats_row,
        max_goals=8,
    )
    markets = pred["markets"]
    score_mat: np.ndarray = pred["score_matrix"]

    bet_cfg = settings.betting.model_copy(
        update={
            "devig_method": devig_method,
            "min_ev_threshold": min_ev_pct / 100.0,
            "kelly_fraction": kelly_mult,
        }
    )
    signals = evaluate_match_bets(
        model_probs=[markets.p_home, markets.p_draw, markets.p_away],
        bookmaker_odds=[odd_h, odd_d, odd_a],
        bankroll=bankroll,
        config=bet_cfg,
    )
    best_bet = select_best_value_bet(signals)
    vig_pct = overround([odd_h, odd_d, odd_a]) * 100.0
    fair_mkt = devig_odds([odd_h, odd_d, odd_a], method=devig_method)

    st.divider()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Home (1)", f"{markets.p_home * 100:.1f}%", f"Fair {1 / markets.p_home:.2f}")
    k2.metric("Draw (X)", f"{markets.p_draw * 100:.1f}%", f"Fair {1 / markets.p_draw:.2f}")
    k3.metric("Away (2)", f"{markets.p_away * 100:.1f}%", f"Fair {1 / markets.p_away:.2f}")
    k4.metric(
        "Over 2.5 / Under 2.5",
        f"{markets.p_over_25 * 100:.1f}% / {markets.p_under_25 * 100:.1f}%",
        f"xG {markets.expected_home_goals:.2f} - {markets.expected_away_goals:.2f}",
    )
    k5.metric(
        "BTTS (Goal / NoGoal)",
        f"{markets.p_btts_yes * 100:.1f}% / {markets.p_btts_no * 100:.1f}%",
        f"Bookie Vig {vig_pct:.2f}%",
    )

    if best_bet is not None:
        st.success(
            f"**VALUE BET SIGNAL: {best_bet.outcome}** @ **{best_bet.bookmaker_odds:.2f}** | "
            f"EV: **+{best_bet.expected_value * 100:.2f}%** | "
            f"Model: **{best_bet.model_prob * 100:.1f}%** vs Market Fair: "
            f"**{best_bet.market_fair_prob * 100:.1f}%** | "
            f"Fractional Kelly Stake: **€{best_bet.recommended_stake:.2f}** "
            f"({best_bet.stake_fraction * 100:.2f}% bankroll)"
        )
    else:
        st.info(
            f"No positive Value Bet above +{min_ev_pct:.1f}% EV threshold for the entered odds."
        )

    sig_df = pd.DataFrame(
        [
            {
                "Outcome": s.outcome,
                "Model Prob (%)": round(s.model_prob * 100, 2),
                "Market Fair Prob (%)": round(fair_mkt[idx] * 100, 2),
                "Bookmaker Odd": round(s.bookmaker_odds, 2),
                "Model Fair Odd": round(s.fair_odds, 2),
                "EV (%)": round(s.expected_value * 100, 2),
                "Edge (%)": round(s.edge * 100, 2),
                "Kelly Stake (€)": round(s.recommended_stake, 2),
                "Value Bet": "YES" if s.is_value_bet else "NO",
            }
            for idx, s in enumerate(signals)
        ]
    )
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

    col_left, col_right = st.columns([1.4, 1.0])
    with col_left:
        max_show = 6
        sub_mat = score_mat[: max_show + 1, : max_show + 1] * 100.0
        fig_hm = px.imshow(
            sub_mat,
            x=[str(g) for g in range(max_show + 1)],
            y=[str(g) for g in range(max_show + 1)],
            text_auto=".1f",
            color_continuous_scale="YlGnBu",
            labels={"x": f"{away_name} Goals", "y": f"{home_name} Goals", "color": "Prob (%)"},
            title="Dixon-Coles Exact Scoreline Matrix (%)",
        )
        fig_hm.update_layout(height=440)
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_right:
        top_df = pd.DataFrame(
            [
                {"Scoreline": s.score, "Probability (%)": round(s.probability * 100, 2)}
                for s in markets.top_scorelines
            ]
        )
        fig_bar = px.bar(
            top_df,
            x="Scoreline",
            y="Probability (%)",
            text="Probability (%)",
            title="Top 5 Most Likely Exact Scorelines",
            color="Probability (%)",
            color_continuous_scale="Teal",
        )
        fig_bar.update_layout(height=440, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


def render_backtest_tab(res: dict[str, object]) -> None:
    """Tab 2: Walk-Forward Backtest Equity Curve, Calibration Diagram, and KPI comparison."""
    report = res["backtest_report"]
    equity_df: pd.DataFrame = res["equity_df"]  # type: ignore[assignment]

    if report is None:
        st.warning("Backtest report not found. Run `scripts/run_backtest.py` first.")
        return

    cal = report["calibration_metrics"]
    bet = report["betting_metrics"]

    st.subheader(
        f"Walk-Forward Validation — Seasons {', '.join(report['test_seasons'])} "
        f"({report['n_matches']} OOS matches, {report['n_refits']} periodic refits)"
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Ensemble Log-Loss", f"{cal['log_loss_ensemble']:.4f}")
    m2.metric("Pinnacle Close Log-Loss", f"{cal['log_loss_pinnacle_close']:.4f}")
    m3.metric("Ensemble Brier", f"{cal['brier_ensemble']:.4f}")
    m4.metric("Ensemble ECE", f"{cal['ece_ensemble']:.4f}")
    m5.metric("ROI / Yield", f"{bet['roi_pct']:.1f}% / {bet['yield_pct']:.1f}%")
    clv_label = f"{bet['avg_clv_raw_pct']:.2f}% / {bet['avg_clv_novig_pct']:.2f}%"
    m6.metric("Avg CLV (Raw / No-Vig)", clv_label)

    c_left, c_right = st.columns(2)

    with c_left:
        if not equity_df.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=equity_df["date"],
                    y=equity_df["bankroll"],
                    mode="lines",
                    name="Bankroll (€)",
                    line={"color": "#1f77b4", "width": 2.5},
                )
            )
            fig_eq.add_hline(
                y=bet["initial_bankroll"],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Bankroll",
            )
            fig_eq.update_layout(
                title=f"Bankroll Equity Curve (Max Drawdown: {bet['max_drawdown_pct']:.1f}%)",
                xaxis_title="Date",
                yaxis_title="Bankroll (€)",
                height=420,
            )
            st.plotly_chart(fig_eq, use_container_width=True)

    with c_right:
        rel_ens = pd.DataFrame(report["reliability_diagram_ensemble"])
        rel_mkt = pd.DataFrame(report["reliability_diagram_market"])
        fig_cal = go.Figure()
        fig_cal.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line={"dash": "dash", "color": "gray"},
            )
        )
        if not rel_ens.empty:
            fig_cal.add_trace(
                go.Scatter(
                    x=rel_ens["mean_predicted"],
                    y=rel_ens["empirical_frequency"],
                    mode="lines+markers",
                    name=f"Ensemble (ECE={cal['ece_ensemble']:.4f})",
                    line={"color": "#008080", "width": 2.5},
                )
            )
        if not rel_mkt.empty:
            fig_cal.add_trace(
                go.Scatter(
                    x=rel_mkt["mean_predicted"],
                    y=rel_mkt["empirical_frequency"],
                    mode="lines+markers",
                    name=f"Market Open (ECE={cal['ece_market_open']:.4f})",
                    line={"color": "#ff7f0e", "width": 2.0},
                )
            )
        fig_cal.update_layout(
            title="Reliability Diagram (Probability Calibration)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Observed Outcome Frequency",
            height=420,
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    st.subheader("Model vs Market Calibration Comparison")
    comp_df = pd.DataFrame(
        [
            {
                "Model / Benchmark": "Ensemble (Dixon-Coles + Calibrated LGBM)",
                "Log-Loss (lower is better)": cal["log_loss_ensemble"],
                "Brier Score (lower is better)": cal["brier_ensemble"],
            },
            {
                "Model / Benchmark": "Dixon-Coles (Time-Decayed MLE)",
                "Log-Loss (lower is better)": cal["log_loss_dixon_coles"],
                "Brier Score (lower is better)": cal["brier_dixon_coles"],
            },
            {
                "Model / Benchmark": "Calibrated LightGBM (Gold Features)",
                "Log-Loss (lower is better)": cal["log_loss_gbm"],
                "Brier Score (lower is better)": cal["brier_gbm"],
            },
            {
                "Model / Benchmark": "Market Opening (Power Devigged)",
                "Log-Loss (lower is better)": cal["log_loss_market_open"],
                "Brier Score (lower is better)": cal["brier_market_open"],
            },
            {
                "Model / Benchmark": "Pinnacle Closing Line (Power Devigged)",
                "Log-Loss (lower is better)": cal["log_loss_pinnacle_close"],
                "Brier Score (lower is better)": cal["brier_pinnacle_close"],
            },
        ]
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.subheader("Per-Season Breakdown")
    st.dataframe(pd.DataFrame(report["per_season"]), use_container_width=True, hide_index=True)


def render_power_ratings_tab(res: dict[str, object]) -> None:
    """Tab 3: Dixon-Coles latent Attack (alpha) and Defense (beta) team power ratings."""
    bundle = res["bundle"]
    if bundle is None:
        st.warning("Model bundle not found. Run `scripts/run_training.py` first.")
        return

    dc = bundle["ensemble"].dc_model
    ratings_df = dc.get_team_ratings()

    st.subheader(
        f"Dixon-Coles Latent Team Power Ratings "
        f"(Home Advantage γ = {dc.home_adv_:.4f}, Low-Score ρ = {dc.rho_:.4f})"
    )

    fig_quad = px.scatter(
        ratings_df,
        x="attack_alpha",
        y="defense_beta",
        text="team",
        color="overall_power",
        color_continuous_scale="RdYlGn",
        labels={
            "attack_alpha": "Attack Strength (α — higher is stronger)",
            "defense_beta": "Defense Weakness (β — lower concedes fewer goals)",
            "overall_power": "Overall Power",
        },
        title="Attack vs Defense Quadrant (Top-Right = Strong Attack & Tight Defense)",
    )
    fig_quad.update_traces(textposition="top center", marker={"size": 11})
    fig_quad.update_yaxes(autorange="reversed")
    fig_quad.add_vline(x=1.0, line_dash="dash", line_color="gray")
    med_def = float(ratings_df["defense_beta"].median())
    fig_quad.add_hline(y=med_def, line_dash="dash", line_color="gray")
    fig_quad.update_layout(height=520)
    st.plotly_chart(fig_quad, use_container_width=True)

    st.dataframe(
        ratings_df.style.format(
            {
                "attack_alpha": "{:.3f}",
                "defense_beta": "{:.3f}",
                "log_attack": "{:+.3f}",
                "log_defense": "{:+.3f}",
                "overall_power": "{:+.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("⚽ Football Quant Predictor — Serie A Calibration & Value Betting")
    res = load_resources()

    tab1, tab2, tab3 = st.tabs(
        ["Match Predictor", "Backtest & Calibration", "Team Power Ratings"]
    )
    with tab1:
        render_match_predictor_tab(res)
    with tab2:
        render_backtest_tab(res)
    with tab3:
        render_power_ratings_tab(res)


if __name__ == "__main__":
    main()
