"""Expanding-window Walk-Forward Backtester with calibration and CLV analytics."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from football_predictor.betting.devig import devig_odds
from football_predictor.betting.strategy import evaluate_match_bets, select_best_value_bet
from football_predictor.config import AppSettings
from football_predictor.features.feature_store import get_feature_columns
from football_predictor.models.base import OUTCOME_TO_IDX
from football_predictor.models.dixon_coles import DixonColesModel
from football_predictor.models.ensemble import LogLossEnsembleModel

logger = logging.getLogger(__name__)


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute exact multiclass Brier Score: mean_n sum_c (p_{n,c} - y_{n,c})^2."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    """Compute class-pooled Expected Calibration Error (ECE) and reliability diagram bins.

    Evaluates all (match, outcome) predicted probabilities against their binary realizations,
    which directly measures calibration across all three 1X2 betting markets.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0

    pred_flat = p.ravel()
    true_flat = one_hot.ravel()
    total = len(pred_flat)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_info: list[dict[str, float | int]] = []

    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (pred_flat >= lo) & (pred_flat <= hi if i == n_bins - 1 else pred_flat < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        mean_pred = float(np.mean(pred_flat[mask]))
        emp_freq = float(np.mean(true_flat[mask]))
        ece += (cnt / total) * abs(mean_pred - emp_freq)
        bins_info.append(
            {
                "bin_lower": round(lo, 3),
                "bin_upper": round(hi, 3),
                "mean_predicted": round(mean_pred, 5),
                "empirical_frequency": round(emp_freq, 5),
                "count": cnt,
            }
        )
    return float(ece), bins_info


def compute_clv(
    taken_odds: float,
    closing_odds: float,
    closing_fair_prob: float | None = None,
) -> tuple[float, float]:
    """Compute Closing Line Value (CLV): (raw_clv, novig_clv).

    - raw_clv   = (taken_odds / closing_odds) - 1
    - novig_clv = (closing_fair_prob * taken_odds) - 1
    """
    raw_clv = float((taken_odds / closing_odds) - 1.0)
    novig_clv = (
        float(closing_fair_prob * taken_odds - 1.0) if closing_fair_prob is not None else raw_clv
    )
    return raw_clv, novig_clv


class WalkForwardBacktester:
    """Strict expanding-window walk-forward validation and bankroll simulation engine."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.feature_columns = get_feature_columns(settings.features.ema_spans)

    def _get_open_odds_cols(self) -> tuple[str, str, str]:
        if self.settings.backtest.odds_source == "market_avg_open":
            return ("avg_h", "avg_d", "avg_a")
        return ("psh", "psd", "psa")

    def run(
        self,
        gold_df: pd.DataFrame,
        save_artifacts: bool = True,
    ) -> dict[str, object]:
        """Execute walk-forward backtest over the last `n_test_seasons` seasons."""
        df = gold_df.copy()
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
        df = df.sort_values(["match_date", "match_timestamp", "match_id"]).reset_index(drop=True)

        seasons = sorted(df["season"].astype(str).unique())
        n_test = self.settings.backtest.n_test_seasons
        if len(seasons) <= n_test:
            raise ValueError(
                f"Need more than {n_test} seasons for walk-forward backtest; found {seasons}"
            )
        test_seasons = seasons[-n_test:]
        test_mask = df["season"].astype(str).isin(test_seasons)
        test_df = df.loc[test_mask].copy().reset_index(drop=True)

        logger.info(
            "Starting Walk-Forward Backtest on seasons=%s (%d OOS matches, retrain every %d days)",
            test_seasons,
            len(test_df),
            self.settings.backtest.retrain_interval_days,
        )

        open_cols = self._get_open_odds_cols()
        retrain_delta = timedelta(days=self.settings.backtest.retrain_interval_days)

        active_model: LogLossEnsembleModel | None = None
        prev_dc: DixonColesModel | None = None
        last_train_cutoff: date | None = None
        refit_history: list[dict[str, object]] = []

        bankroll = float(self.settings.betting.initial_bankroll)
        peak_bankroll = bankroll
        max_drawdown = 0.0

        prediction_records: list[dict[str, object]] = []
        bet_records: list[dict[str, object]] = []
        equity_records: list[dict[str, object]] = [
            {
                "date": str(test_df["match_date"].min()),
                "bankroll": bankroll,
                "drawdown_pct": 0.0,
                "n_bets_cumulative": 0,
            }
        ]

        # Iterate strictly day-by-day over test dates
        for current_date, day_df in test_df.groupby("match_date", sort=True):
            needs_retrain = (
                active_model is None
                or last_train_cutoff is None
                or (current_date - last_train_cutoff) >= retrain_delta
            )

            if needs_retrain:
                # CRITICAL ANTI-LEAKAGE GUARD: strictly match_date < current_date
                train_slice = df.loc[df["match_date"] < current_date]
                if len(train_slice) < self.settings.backtest.min_train_matches:
                    continue
                assert train_slice["match_date"].max() < current_date, (
                    "Temporal leakage detected in walk-forward split!"
                )

                model = LogLossEnsembleModel(
                    config=self.settings.models,
                    feature_columns=self.feature_columns,
                    random_seed=self.settings.project.random_seed,
                )
                if prev_dc is not None:
                    model.set_dc_warm_start(prev_dc)
                model.fit(train_slice)

                active_model = model
                prev_dc = model.dc_model
                last_train_cutoff = current_date
                refit_history.append(
                    {
                        "cutoff_date": str(current_date),
                        "n_train": int(len(train_slice)),
                        "weight_dc": round(model.weight_dc_, 4),
                        "dc_converged": bool(model.dc_model.converged_),
                        "dc_iters": int(model.dc_model.n_iter_),
                    }
                )

            assert active_model is not None
            p_ens_batch, p_dc_batch, p_gbm_batch = active_model.predict_components(day_df)

            # All bets on `current_date` are sized against start-of-day bankroll (no intra-day leak)
            start_of_day_bankroll = bankroll
            day_pnl = 0.0

            for idx, rec in enumerate(day_df.itertuples(index=False)):
                p_ens = p_ens_batch[idx]
                p_dc = p_dc_batch[idx]
                p_gbm = p_gbm_batch[idx]

                o_h = float(getattr(rec, open_cols[0]))
                o_d = float(getattr(rec, open_cols[1]))
                o_a = float(getattr(rec, open_cols[2]))
                c_h = float(rec.psch)
                c_d = float(rec.pscd)
                c_a = float(rec.psca)

                has_open = all(np.isfinite([o_h, o_d, o_a])) and min(o_h, o_d, o_a) > 1.0
                has_close = all(np.isfinite([c_h, c_d, c_a])) and min(c_h, c_d, c_a) > 1.0

                p_mkt_open = (
                    devig_odds([o_h, o_d, o_a], method=self.settings.betting.devig_method)
                    if has_open
                    else np.array([np.nan, np.nan, np.nan])
                )
                p_mkt_close = (
                    devig_odds([c_h, c_d, c_a], method=self.settings.betting.devig_method)
                    if has_close
                    else np.array([np.nan, np.nan, np.nan])
                )

                prediction_records.append(
                    {
                        "match_id": rec.match_id,
                        "season": str(rec.season),
                        "match_date": current_date,
                        "home_team": rec.home_team,
                        "away_team": rec.away_team,
                        "ftr": rec.ftr,
                        "target": int(rec.target),
                        "p_ens_h": float(p_ens[0]),
                        "p_ens_d": float(p_ens[1]),
                        "p_ens_a": float(p_ens[2]),
                        "p_dc_h": float(p_dc[0]),
                        "p_dc_d": float(p_dc[1]),
                        "p_dc_a": float(p_dc[2]),
                        "p_gbm_h": float(p_gbm[0]),
                        "p_gbm_d": float(p_gbm[1]),
                        "p_gbm_a": float(p_gbm[2]),
                        "p_mkt_open_h": float(p_mkt_open[0]),
                        "p_mkt_open_d": float(p_mkt_open[1]),
                        "p_mkt_open_a": float(p_mkt_open[2]),
                        "p_mkt_close_h": float(p_mkt_close[0]),
                        "p_mkt_close_d": float(p_mkt_close[1]),
                        "p_mkt_close_a": float(p_mkt_close[2]),
                        "open_h": o_h,
                        "open_d": o_d,
                        "open_a": o_a,
                        "close_h": c_h,
                        "close_d": c_d,
                        "close_a": c_a,
                    }
                )

                if not has_open or start_of_day_bankroll <= 1.0:
                    continue

                signals = evaluate_match_bets(
                    model_probs=p_ens,
                    bookmaker_odds=[o_h, o_d, o_a],
                    bankroll=start_of_day_bankroll,
                    config=self.settings.betting,
                )
                chosen = select_best_value_bet(signals)
                if chosen is None:
                    continue

                out_idx = OUTCOME_TO_IDX[chosen.outcome]
                won = int(rec.target) == out_idx
                stake = chosen.recommended_stake
                pnl = stake * (chosen.bookmaker_odds - 1.0) if won else -stake
                day_pnl += pnl

                closing_arr = [c_h, c_d, c_a]
                if has_close:
                    raw_clv, novig_clv = compute_clv(
                        taken_odds=chosen.bookmaker_odds,
                        closing_odds=closing_arr[out_idx],
                        closing_fair_prob=float(p_mkt_close[out_idx]),
                    )
                else:
                    raw_clv, novig_clv = float("nan"), float("nan")

                bet_records.append(
                    {
                        "match_id": rec.match_id,
                        "season": str(rec.season),
                        "match_date": current_date,
                        "home_team": rec.home_team,
                        "away_team": rec.away_team,
                        "outcome_bet": chosen.outcome,
                        "actual_ftr": rec.ftr,
                        "won": bool(won),
                        "model_prob": chosen.model_prob,
                        "market_fair_prob": chosen.market_fair_prob,
                        "odds_taken": chosen.bookmaker_odds,
                        "closing_odds": closing_arr[out_idx] if has_close else np.nan,
                        "expected_value": chosen.expected_value,
                        "stake_fraction": chosen.stake_fraction,
                        "stake": stake,
                        "pnl": pnl,
                        "clv_raw": raw_clv,
                        "clv_novig": novig_clv,
                    }
                )

            if day_pnl != 0.0:
                bankroll = max(0.0, bankroll + day_pnl)
                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0
                max_drawdown = max(max_drawdown, dd)
                equity_records.append(
                    {
                        "date": str(current_date),
                        "bankroll": round(bankroll, 2),
                        "drawdown_pct": round(dd * 100.0, 3),
                        "n_bets_cumulative": len(bet_records),
                    }
                )

        preds_df = pd.DataFrame(prediction_records)
        bets_df = pd.DataFrame(bet_records)
        equity_df = pd.DataFrame(equity_records)

        report = self._compile_report(
            preds_df=preds_df,
            bets_df=bets_df,
            equity_df=equity_df,
            test_seasons=test_seasons,
            refit_history=refit_history,
            final_bankroll=bankroll,
            max_drawdown=max_drawdown,
        )

        if save_artifacts:
            self._save_artifacts(report, preds_df, bets_df, equity_df)

        return report

    def _compile_report(
        self,
        preds_df: pd.DataFrame,
        bets_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        test_seasons: list[str],
        refit_history: list[dict[str, object]],
        final_bankroll: float,
        max_drawdown: float,
    ) -> dict[str, object]:
        y_true = preds_df["target"].to_numpy(dtype=np.int64)
        p_ens = preds_df[["p_ens_h", "p_ens_d", "p_ens_a"]].to_numpy(dtype=np.float64)
        p_dc = preds_df[["p_dc_h", "p_dc_d", "p_dc_a"]].to_numpy(dtype=np.float64)
        p_gbm = preds_df[["p_gbm_h", "p_gbm_d", "p_gbm_a"]].to_numpy(dtype=np.float64)

        valid_open = preds_df["p_mkt_open_h"].notna()
        valid_close = preds_df["p_mkt_close_h"].notna()
        p_open = preds_df.loc[
            valid_open, ["p_mkt_open_h", "p_mkt_open_d", "p_mkt_open_a"]
        ].to_numpy()
        p_close = preds_df.loc[
            valid_close, ["p_mkt_close_h", "p_mkt_close_d", "p_mkt_close_a"]
        ].to_numpy()

        ece_ens, reliability_ens = expected_calibration_error(y_true, p_ens, n_bins=10)
        ece_mkt, reliability_mkt = expected_calibration_error(
            y_true[valid_open.to_numpy()], p_open, n_bins=10
        )

        init_br = float(self.settings.betting.initial_bankroll)
        n_bets = len(bets_df)
        if n_bets > 0:
            total_staked = float(bets_df["stake"].sum())
            total_profit = float(bets_df["pnl"].sum())
            win_rate = float(bets_df["won"].mean())
            roi_pct = ((final_bankroll - init_br) / init_br) * 100.0
            yield_pct = (total_profit / total_staked) * 100.0 if total_staked > 0 else 0.0
            avg_ev = float(bets_df["expected_value"].mean()) * 100.0
            valid_clv = bets_df["clv_raw"].dropna()
            valid_novig_clv = bets_df["clv_novig"].dropna()
            avg_clv_raw = float(valid_clv.mean()) * 100.0 if len(valid_clv) else 0.0
            avg_clv_novig = float(valid_novig_clv.mean()) * 100.0 if len(valid_novig_clv) else 0.0
            pct_pos_clv = float((valid_clv > 0).mean()) * 100.0 if len(valid_clv) else 0.0
        else:
            total_staked = total_profit = win_rate = roi_pct = yield_pct = 0.0
            avg_ev = avg_clv_raw = avg_clv_novig = pct_pos_clv = 0.0

        per_season: list[dict[str, object]] = []
        for s in test_seasons:
            sub = preds_df.loc[preds_df["season"] == s]
            y_s = sub["target"].to_numpy(dtype=np.int64)
            p_s = sub[["p_ens_h", "p_ens_d", "p_ens_a"]].to_numpy()
            sub_open = sub["p_mkt_open_h"].notna()
            p_o = sub.loc[sub_open, ["p_mkt_open_h", "p_mkt_open_d", "p_mkt_open_a"]].to_numpy()
            b_s = bets_df.loc[bets_df["season"] == s] if n_bets > 0 else pd.DataFrame()
            staked_s = float(b_s["stake"].sum()) if len(b_s) else 0.0
            pnl_s = float(b_s["pnl"].sum()) if len(b_s) else 0.0
            per_season.append(
                {
                    "season": s,
                    "n_matches": int(len(sub)),
                    "log_loss_ensemble": round(float(log_loss(y_s, p_s, labels=[0, 1, 2])), 5),
                    "log_loss_market_open": round(
                        float(log_loss(y_s[sub_open.to_numpy()], p_o, labels=[0, 1, 2])), 5
                    ),
                    "brier_ensemble": round(multiclass_brier_score(y_s, p_s), 5),
                    "n_bets": int(len(b_s)),
                    "pnl": round(pnl_s, 2),
                    "yield_pct": round((pnl_s / staked_s) * 100.0, 2) if staked_s > 0 else 0.0,
                }
            )

        return {
            "test_seasons": test_seasons,
            "n_matches": int(len(preds_df)),
            "n_refits": len(refit_history),
            "retrain_interval_days": self.settings.backtest.retrain_interval_days,
            "calibration_metrics": {
                "log_loss_ensemble": round(float(log_loss(y_true, p_ens, labels=[0, 1, 2])), 5),
                "log_loss_dixon_coles": round(float(log_loss(y_true, p_dc, labels=[0, 1, 2])), 5),
                "log_loss_gbm": round(float(log_loss(y_true, p_gbm, labels=[0, 1, 2])), 5),
                "log_loss_market_open": round(
                    float(log_loss(y_true[valid_open.to_numpy()], p_open, labels=[0, 1, 2])), 5
                ),
                "log_loss_pinnacle_close": round(
                    float(log_loss(y_true[valid_close.to_numpy()], p_close, labels=[0, 1, 2])), 5
                ),
                "brier_ensemble": round(multiclass_brier_score(y_true, p_ens), 5),
                "brier_dixon_coles": round(multiclass_brier_score(y_true, p_dc), 5),
                "brier_gbm": round(multiclass_brier_score(y_true, p_gbm), 5),
                "brier_market_open": round(
                    multiclass_brier_score(y_true[valid_open.to_numpy()], p_open), 5
                ),
                "brier_pinnacle_close": round(
                    multiclass_brier_score(y_true[valid_close.to_numpy()], p_close), 5
                ),
                "ece_ensemble": round(ece_ens, 5),
                "ece_market_open": round(ece_mkt, 5),
            },
            "betting_metrics": {
                "initial_bankroll": init_br,
                "final_bankroll": round(final_bankroll, 2),
                "n_bets": n_bets,
                "win_rate_pct": round(win_rate * 100.0, 2),
                "total_staked": round(total_staked, 2),
                "total_profit": round(total_profit, 2),
                "roi_pct": round(roi_pct, 2),
                "yield_pct": round(yield_pct, 2),
                "max_drawdown_pct": round(max_drawdown * 100.0, 2),
                "avg_predicted_ev_pct": round(avg_ev, 2),
                "avg_clv_raw_pct": round(avg_clv_raw, 2),
                "avg_clv_novig_pct": round(avg_clv_novig, 2),
                "pct_positive_clv": round(pct_pos_clv, 2),
            },
            "per_season": per_season,
            "reliability_diagram_ensemble": reliability_ens,
            "reliability_diagram_market": reliability_mkt,
            "refit_history": refit_history,
            "equity_curve_points": len(equity_df),
        }

    def _save_artifacts(
        self,
        report: dict[str, object],
        preds_df: pd.DataFrame,
        bets_df: pd.DataFrame,
        equity_df: pd.DataFrame,
    ) -> None:
        out_dir: Path = self.settings.paths.artifacts_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        report_path = out_dir / "backtest_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        preds_df.to_parquet(out_dir / "backtest_predictions.parquet", index=False)
        bets_df.to_parquet(out_dir / "backtest_bets.parquet", index=False)
        equity_df.to_parquet(out_dir / "backtest_equity_curve.parquet", index=False)
        logger.info("Saved backtest artifacts -> %s", out_dir)
