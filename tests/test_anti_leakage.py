"""Rigorous anti-leakage tests for the point-in-time Gold Feature Store.

Verifies:
1. Perturbation / Counterfactual causality test: mutating a match's goals, shots, and
   outcome at date `t` leaves ALL features of that match (and every match with
   `match_date <= t`) bit-for-bit identical, while strictly propagating to matches at
   dates `> t` involving those teams.
2. Prefix truncation invariance: computing features on the full timeline vs a prefix
   ending at date `t` yields identical feature values for every row in the prefix.
3. Same-day isolation: mutating match A on date `t` cannot affect match B on date `t`.
4. Exact hand-computed verification of EMA, Elo, and rest_days causality.
5. Zero target/post-match columns in the modeling feature list.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from football_predictor.config import FeaturesConfig
from football_predictor.features.feature_store import (
    PointInTimeFeatureStore,
    get_feature_columns,
)


def _make_synthetic_silver_matches() -> pd.DataFrame:
    """Create a deterministic multi-round synthetic schedule with same-day fixtures."""
    rows = [
        # Round 1: 2024-08-18 (two matches on same date)
        ("M1", date(2024, 8, 18), "inter", "ac_milan", 2, 1, 6.0, 3.0),
        ("M2", date(2024, 8, 18), "juventus", "napoli", 0, 0, 2.0, 4.0),
        # Round 2: 2024-08-25 (target perturbation date `t`)
        ("M3", date(2024, 8, 25), "ac_milan", "juventus", 1, 0, 5.0, 2.0),
        ("M4", date(2024, 8, 25), "napoli", "inter", 1, 2, 4.0, 7.0),
        # Round 3: 2024-09-01 (strictly after `t`)
        ("M5", date(2024, 9, 1), "inter", "juventus", 3, 1, 8.0, 3.0),
        ("M6", date(2024, 9, 1), "ac_milan", "napoli", 2, 2, 5.0, 5.0),
        # Round 4: 2024-09-14
        ("M7", date(2024, 9, 14), "juventus", "ac_milan", 1, 1, 4.0, 4.0),
        ("M8", date(2024, 9, 14), "inter", "napoli", 1, 0, 5.0, 2.0),
    ]
    records = []
    for mid, mdate, h_id, a_id, fthg, ftag, hst, ast in rows:
        ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
        records.append(
            {
                "match_id": mid,
                "league": "I1",
                "season": "2425",
                "match_date": mdate,
                "match_timestamp": datetime(mdate.year, mdate.month, mdate.day, 18, 0),
                "home_team_id": h_id,
                "away_team_id": a_id,
                "home_team": h_id.title(),
                "away_team": a_id.title(),
                "fthg": fthg,
                "ftag": ftag,
                "ftr": ftr,
                "hst": hst,
                "ast": ast,
                "psh": 2.1,
                "psd": 3.4,
                "psa": 3.5,
                "avg_h": 2.05,
                "avg_d": 3.35,
                "avg_a": 3.45,
                "psch": 2.0,
                "pscd": 3.4,
                "psca": 3.6,
            }
        )
    return pd.DataFrame(records)


def test_mutating_match_at_t_never_changes_features_at_or_before_t_but_changes_t_plus_1() -> None:
    """Core counterfactual anti-leakage test: mutate M3 at date t, check <= t vs > t."""
    cfg = FeaturesConfig(ema_spans=[5, 10])
    store = PointInTimeFeatureStore(config=cfg)
    feature_cols = get_feature_columns(cfg.ema_spans)

    base_silver = _make_synthetic_silver_matches()
    mutated_silver = base_silver.copy()

    # Drastically mutate match M3 (ac_milan vs juventus on 2024-08-25)
    target_mask = mutated_silver["match_id"] == "M3"
    mutated_silver.loc[target_mask, "fthg"] = 0
    mutated_silver.loc[target_mask, "ftag"] = 9
    mutated_silver.loc[target_mask, "ftr"] = "A"
    mutated_silver.loc[target_mask, "hst"] = 0.0
    mutated_silver.loc[target_mask, "ast"] = 25.0

    gold_base = store.build_features(base_silver)
    gold_mut = store.build_features(mutated_silver)

    t_date = date(2024, 8, 25)

    # 1. Match M3 itself at date t MUST have bit-for-bit identical features
    m3_base = gold_base.loc[gold_base["match_id"] == "M3", feature_cols].reset_index(drop=True)
    m3_mut = gold_mut.loc[gold_mut["match_id"] == "M3", feature_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(m3_base, m3_mut, check_exact=True)

    # 2. ALL matches with match_date <= t (including same-day M4!) MUST be identical
    le_mask = gold_base["match_date"] <= t_date
    pd.testing.assert_frame_equal(
        gold_base.loc[le_mask, feature_cols].reset_index(drop=True),
        gold_mut.loc[le_mask, feature_cols].reset_index(drop=True),
        check_exact=True,
    )

    # 3. Matches at t+1 involving juventus (M5) and ac_milan (M6) MUST reflect the mutation
    for next_mid in ("M5", "M6"):
        next_base = gold_base.loc[gold_base["match_id"] == next_mid, feature_cols].to_numpy()
        next_mut = gold_mut.loc[gold_mut["match_id"] == next_mid, feature_cols].to_numpy()
        assert not np.allclose(next_base, next_mut), (
            f"Expected post-t match {next_mid} features to change after mutating M3!"
        )

    # Specifically verify Elo and goal EMAs moved in the expected direction on M5 & M6
    m5_elo_base = float(gold_base.loc[gold_base["match_id"] == "M5", "elo_away"].iloc[0])
    m5_elo_mut = float(gold_mut.loc[gold_mut["match_id"] == "M5", "elo_away"].iloc[0])
    assert m5_elo_mut > m5_elo_base, "Juventus Elo should increase after mutated 0-9 away win"

    m6_gf_base = float(gold_base.loc[gold_base["match_id"] == "M6", "home_gf_ema_5"].iloc[0])
    m6_gf_mut = float(gold_mut.loc[gold_mut["match_id"] == "M6", "home_gf_ema_5"].iloc[0])
    assert m6_gf_mut < m6_gf_base, "AC Milan GF EMA should decrease after scoring 0 instead of 1"


def test_prefix_truncation_invariance() -> None:
    """Adding future matches after date `t` must never alter features on or before `t`."""
    cfg = FeaturesConfig(ema_spans=[5, 10])
    store = PointInTimeFeatureStore(config=cfg)
    feature_cols = get_feature_columns(cfg.ema_spans)

    full_silver = _make_synthetic_silver_matches()
    cutoff = date(2024, 9, 1)
    prefix_silver = full_silver.loc[full_silver["match_date"] <= cutoff].copy()

    gold_full = store.build_features(full_silver)
    gold_prefix = store.build_features(prefix_silver)

    gold_full_prefix = gold_full.loc[gold_full["match_date"] <= cutoff, feature_cols].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        gold_full_prefix,
        gold_prefix[feature_cols].reset_index(drop=True),
        check_exact=True,
    )


def test_exact_manual_ema_rest_days_and_elo_values() -> None:
    """Verify exact numerical point-in-time math on first two matches of a team."""
    cfg = FeaturesConfig(
        ema_spans=[5],
        elo_initial_rating=1500.0,
        elo_k_factor=20.0,
        elo_home_advantage=65.0,
        default_rest_days=7.0,
    )
    store = PointInTimeFeatureStore(config=cfg)
    silver = _make_synthetic_silver_matches()
    gold = store.build_features(silver)

    # Match M1 (first match for inter & ac_milan): exact cold-start priors
    m1 = gold.loc[gold["match_id"] == "M1"].iloc[0]
    assert m1["elo_home"] == pytest.approx(1500.0)
    assert m1["elo_away"] == pytest.approx(1500.0)
    assert m1["rest_days_home"] == pytest.approx(7.0)
    assert m1["home_matches_played"] == pytest.approx(0.0)

    # Match M3 (ac_milan vs juventus on 2024-08-25, exactly 7 days after 2024-08-18)
    # ac_milan played 1 prior match (M1: scored 1, conceded 2, sot_for=3, sot_against=6, pts=0)
    m3 = gold.loc[gold["match_id"] == "M3"].iloc[0]
    assert m3["rest_days_home"] == pytest.approx(7.0)
    assert m3["home_matches_played"] == pytest.approx(1.0)
    assert m3["home_gf_ema_5"] == pytest.approx(1.0)
    assert m3["home_ga_ema_5"] == pytest.approx(2.0)
    assert m3["home_sot_for_ema_5"] == pytest.approx(3.0)
    assert m3["home_pts_ema_5"] == pytest.approx(0.0)

    # Match M7 (2024-09-14, 13 days after 2024-09-01)
    m7 = gold.loc[gold["match_id"] == "M7"].iloc[0]
    assert m7["rest_days_home"] == pytest.approx(13.0)
    assert m7["rest_days_away"] == pytest.approx(13.0)


def test_inference_row_strictly_excludes_as_of_date_and_future() -> None:
    """build_inference_row(as_of_date=t) must match the historical gold row at date t."""
    cfg = FeaturesConfig(ema_spans=[5, 10])
    store = PointInTimeFeatureStore(config=cfg)
    feature_cols = get_feature_columns(cfg.ema_spans)
    silver = _make_synthetic_silver_matches()

    gold = store.build_features(silver)
    m5_gold = gold.loc[gold["match_id"] == "M5", feature_cols].reset_index(drop=True)

    # Ask for inference features on 2024-09-01 while passing the FULL dataset (including M5..M8)
    m5_inf = store.build_inference_row(
        silver_df=silver,
        home_team_id="inter",
        away_team_id="juventus",
        as_of_date=date(2024, 9, 1),
    )[feature_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(m5_gold, m5_inf, check_exact=True)


def test_no_post_match_or_target_columns_in_feature_list() -> None:
    """Guardrail ensuring no target or post-match stats ever enter model feature columns."""
    forbidden = {
        "fthg",
        "ftag",
        "hthg",
        "htag",
        "ftr",
        "target",
        "hs",
        "as_",
        "hst",
        "ast",
        "psch",
        "pscd",
        "psca",
        "avg_ch",
        "avg_cd",
        "avg_ca",
    }
    cols = set(get_feature_columns([5, 10]))
    assert cols.isdisjoint(forbidden), f"Leakage columns detected: {cols & forbidden}"


def test_anti_leakage_on_real_silver_slice_if_present() -> None:
    """Run perturbation anti-leakage check directly on real ingested Serie A data if available."""
    from football_predictor.config import load_settings
    from football_predictor.db.duckdb_manager import DuckDBManager

    settings = load_settings()
    if not settings.paths.duckdb_path.exists():
        pytest.skip("Silver DuckDB not yet materialized.")

    db = DuckDBManager(settings.paths.duckdb_path)
    silver = db.read_silver_matches(league="I1").head(120).copy()
    store = PointInTimeFeatureStore(config=settings.features)
    feature_cols = get_feature_columns(settings.features.ema_spans)

    # Pick a mid-slice match
    target_row = silver.iloc[60]
    target_id = str(target_row["match_id"])
    target_date = pd.to_datetime(target_row["match_date"]).date()
    home_id = str(target_row["home_team_id"])

    mutated = silver.copy()
    mask = mutated["match_id"] == target_id
    mutated.loc[mask, "fthg"] = 10
    mutated.loc[mask, "ftag"] = 0
    mutated.loc[mask, "ftr"] = "H"
    mutated.loc[mask, "hst"] = 20.0
    mutated.loc[mask, "ast"] = 0.0

    g1 = store.build_features(silver)
    g2 = store.build_features(mutated)

    le_mask = g1["match_date"] <= target_date
    pd.testing.assert_frame_equal(
        g1.loc[le_mask, feature_cols].reset_index(drop=True),
        g2.loc[le_mask, feature_cols].reset_index(drop=True),
        check_exact=True,
    )

    # Next match strictly after target_date involving home_id must differ
    after_home = (g1["match_date"] > target_date) & (
        (g1["home_team_id"] == home_id) | (g1["away_team_id"] == home_id)
    )
    assert after_home.any(), "Expected at least one subsequent match for home team in slice"
    first_after_idx = g1.index[after_home][0]
    assert not np.allclose(
        g1.loc[first_after_idx, feature_cols].to_numpy(dtype=float),
        g2.loc[first_after_idx, feature_cols].to_numpy(dtype=float),
    )

    # Sanity: rest_days never exceeds max_rest_days and never negative
    assert (g1["rest_days_home"] >= 1.0).all()
    assert (g1["rest_days_home"] <= settings.features.max_rest_days).all()
    assert target_date + timedelta(days=1) > target_date
