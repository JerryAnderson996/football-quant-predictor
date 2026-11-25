"""Tests for Dixon-Coles mathematical correctness, convergence, GBM calibration, and Ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from sklearn.metrics import log_loss

from football_predictor.config import DixonColesConfig, load_settings
from football_predictor.features.feature_store import get_feature_columns
from football_predictor.models.dixon_coles import DixonColesModel
from football_predictor.models.ensemble import LogLossEnsembleModel
from football_predictor.models.gbm_model import CalibratedGBMModel


def _load_real_gold() -> pd.DataFrame:
    settings = load_settings()
    gold_path = settings.paths.gold_dir / "features.parquet"
    if not gold_path.exists():
        pytest.skip("Gold features.parquet not yet generated.")
    return pd.read_parquet(gold_path)


def test_dixon_coles_analytical_gradient_matches_finite_difference() -> None:
    """Verify exactness of vectorized analytical gradient against numerical finite differences."""
    rng = np.random.default_rng(42)
    n_teams = 6
    n_matches = 40
    home_idx = rng.integers(0, n_teams, size=n_matches)
    away_idx = (home_idx + rng.integers(1, n_teams, size=n_matches)) % n_teams
    fthg = rng.poisson(1.4, size=n_matches)
    ftag = rng.poisson(1.1, size=n_matches)
    weights = np.exp(-0.002 * np.linspace(100, 0, n_matches))

    model = DixonColesModel(DixonColesConfig(l2_reg=0.01))
    theta = rng.normal(0.0, 0.15, size=2 * n_teams + 1)
    theta[-1] = -0.04  # rho inside valid interior

    def func(x: np.ndarray) -> float:
        val, _ = model._neg_log_likelihood_and_grad(
            x, home_idx, away_idx, fthg, ftag, weights, n_teams
        )
        return val

    def grad(x: np.ndarray) -> np.ndarray:
        _, g = model._neg_log_likelihood_and_grad(
            x, home_idx, away_idx, fthg, ftag, weights, n_teams
        )
        return g

    err = check_grad(func, grad, theta)
    assert err < 1e-4, f"Analytical gradient mismatch: check_grad error = {err:.2e}"


def test_dixon_coles_convergence_identifiability_and_score_matrix_on_real_data() -> None:
    """Fit Dixon-Coles on real Serie A data and verify convergence, normalization, and sanity."""
    gold = _load_real_gold()
    train = gold.iloc[:1520].copy()
    val = gold.iloc[1520:1900].copy()

    model = DixonColesModel(DixonColesConfig(xi=0.0018, max_goals=8, l2_reg=0.0005))
    model.fit(train)

    assert model.converged_ is True
    # Identifiability constraint: sum(log(alpha_i)) == 0 -> geometric mean of alpha_i == 1
    assert np.sum(model.log_attack_) == pytest.approx(0.0, abs=1e-10)
    assert np.exp(np.mean(model.log_attack_)) == pytest.approx(1.0, abs=1e-10)
    # Home advantage should be strictly positive in Serie A
    assert model.home_adv_ > 0.05
    assert -0.25 <= model.rho_ <= 0.25

    # Score matrix properties
    mat = model.predict_score_matrix("inter", "hellas_verona", max_goals=8)
    assert mat.shape == (9, 9)
    assert np.all(mat >= 0.0)
    assert float(mat.sum()) == pytest.approx(1.0, abs=1e-12)

    # Derived market probabilities consistency
    markets, _ = model.predict_markets("inter", "hellas_verona", max_goals=8)
    assert markets.p_home + markets.p_draw + markets.p_away == pytest.approx(1.0, abs=1e-9)
    assert markets.p_over_25 is not None and markets.p_under_25 is not None
    assert markets.p_over_25 + markets.p_under_25 == pytest.approx(1.0, abs=1e-9)
    assert markets.p_btts_yes is not None and markets.p_btts_no is not None
    assert markets.p_btts_yes + markets.p_btts_no == pytest.approx(1.0, abs=1e-9)
    assert len(markets.top_scorelines) == 5
    # Top scorelines sorted descending
    top_probs = [s.probability for s in markets.top_scorelines]
    assert top_probs == sorted(top_probs, reverse=True)

    # Out-of-sample Log-Loss must strongly beat naive uniform 1/3 baseline (ln 3 = 1.0986)
    p_val = model.predict_proba(val)
    assert p_val.shape == (len(val), 3)
    assert np.allclose(p_val.sum(axis=1), 1.0, atol=1e-9)
    ll = log_loss(val["target"].to_numpy(), p_val, labels=[0, 1, 2])
    assert ll < 1.04, f"Expected Dixon-Coles OOS Log-Loss < 1.04, got {ll:.4f}"

    # Fallback on unseen/newly promoted club must still return valid normalized matrix
    mat_new = model.predict_score_matrix("new_promoted_fc", "inter")
    assert float(mat_new.sum()) == pytest.approx(1.0, abs=1e-12)


def test_calibrated_gbm_and_ensemble_fit_and_predict_cleanly() -> None:
    """Verify CalibratedGBMModel and LogLossEnsembleModel produce calibrated OOS probabilities."""
    settings = load_settings()
    gold = _load_real_gold()
    train = gold.iloc[:1520].copy()
    val = gold.iloc[1520:1900].copy()
    feat_cols = get_feature_columns(settings.features.ema_spans)

    gbm = CalibratedGBMModel(
        config=settings.models.gbm,
        feature_columns=feat_cols,
        random_seed=settings.project.random_seed,
    )
    gbm.fit(train)
    p_gbm = gbm.predict_proba(val)
    assert p_gbm.shape == (len(val), 3)
    assert np.allclose(p_gbm.sum(axis=1), 1.0, atol=1e-9)
    assert np.all((p_gbm > 0.0) & (p_gbm < 1.0))

    ens = LogLossEnsembleModel(
        config=settings.models,
        feature_columns=feat_cols,
        random_seed=settings.project.random_seed,
    )
    ens.fit(train)
    assert 0.0 <= ens.weight_dc_ <= 1.0

    p_ens = ens.predict_proba(val)
    assert p_ens.shape == (len(val), 3)
    assert np.allclose(p_ens.sum(axis=1), 1.0, atol=1e-9)
    ll_ens = log_loss(val["target"].to_numpy(), p_ens, labels=[0, 1, 2])
    assert ll_ens < 1.04, f"Expected Ensemble OOS Log-Loss < 1.04, got {ll_ens:.4f}"
