"""Unit tests for devigging, EV, Fractional Kelly, Brier/ECE, and CLV math."""

from __future__ import annotations

import numpy as np
import pytest

from football_predictor.betting.backtester import (
    compute_clv,
    expected_calibration_error,
    multiclass_brier_score,
)
from football_predictor.betting.devig import (
    devig_multiplicative,
    devig_odds,
    devig_power,
    overround,
)
from football_predictor.betting.strategy import (
    evaluate_match_bets,
    expected_value,
    fractional_kelly_stake,
    kelly_full_fraction,
    select_best_value_bet,
)
from football_predictor.config import BettingConfig


def test_devig_multiplicative_and_power_sum_to_one() -> None:
    odds = [1.80, 3.60, 4.75]
    assert overround(odds) > 0.0

    p_mult = devig_multiplicative(odds)
    p_pow = devig_power(odds)

    assert float(p_mult.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(p_pow.sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(p_mult > 0.0) and np.all(p_pow > 0.0)

    # Power method corrects favorite-longshot bias:
    # favorite gets HIGHER fair prob under Power than Multiplicative, longshot gets LOWER
    assert p_pow[0] > p_mult[0]
    assert p_pow[2] < p_mult[2]


def test_devig_zero_margin_odds_are_identity() -> None:
    fair_odds = [2.0, 4.0, 4.0]  # 0.50 + 0.25 + 0.25 = 1.00
    assert overround(fair_odds) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(devig_multiplicative(fair_odds), [0.5, 0.25, 0.25], atol=1e-12)
    np.testing.assert_allclose(devig_power(fair_odds), [0.5, 0.25, 0.25], atol=1e-12)


def test_devig_invalid_odds_raise() -> None:
    with pytest.raises(ValueError):
        devig_odds([1.0, 2.0, 3.0], method="power")
    with pytest.raises(ValueError):
        devig_odds([2.0, 3.0, 4.0], method="invalid")  # type: ignore[arg-type]


def test_ev_and_kelly_math_exactness() -> None:
    # p = 0.55, odd = 2.0 -> EV = 0.10, full Kelly = 0.10 / (2.0 - 1.0) = 0.10
    assert expected_value(0.55, 2.0) == pytest.approx(0.10, abs=1e-12)
    assert kelly_full_fraction(0.55, 2.0) == pytest.approx(0.10, abs=1e-12)

    # Fractional Kelly (0.25) -> 0.025 (< max cap 0.03), stake = 25 on 1000 bankroll
    frac, stake = fractional_kelly_stake(
        model_prob=0.55,
        decimal_odds=2.0,
        bankroll=1000.0,
        kelly_fraction=0.25,
        max_stake_fraction=0.03,
    )
    assert frac == pytest.approx(0.025, abs=1e-12)
    assert stake == pytest.approx(25.0, abs=1e-12)

    # Large edge capped strictly at max_stake_fraction = 0.03
    frac_cap, stake_cap = fractional_kelly_stake(
        model_prob=0.80,
        decimal_odds=2.5,
        bankroll=1000.0,
        kelly_fraction=0.25,
        max_stake_fraction=0.03,
    )
    assert frac_cap == pytest.approx(0.03, abs=1e-12)
    assert stake_cap == pytest.approx(30.0, abs=1e-12)

    # Negative EV -> zero stake
    assert kelly_full_fraction(0.40, 2.0) == 0.0
    frac_neg, stake_neg = fractional_kelly_stake(0.40, 2.0, bankroll=1000.0)
    assert frac_neg == 0.0 and stake_neg == 0.0


def test_evaluate_match_bets_filters_by_ev_threshold_and_selects_best() -> None:
    cfg = BettingConfig(
        devig_method="power",
        min_ev_threshold=0.03,
        kelly_fraction=0.25,
        max_stake_fraction=0.03,
    )
    # Home EV = 0.55 * 2.10 - 1 = +0.155 (value bet)
    # Draw EV = 0.25 * 3.40 - 1 = -0.15 (no bet)
    # Away EV = 0.20 * 3.50 - 1 = -0.30 (no bet)
    signals = evaluate_match_bets(
        model_probs=[0.55, 0.25, 0.20],
        bookmaker_odds=[2.10, 3.40, 3.50],
        bankroll=1000.0,
        config=cfg,
    )
    assert len(signals) == 3
    assert signals[0].is_value_bet is True
    assert signals[1].is_value_bet is False
    assert signals[2].is_value_bet is False

    best = select_best_value_bet(signals)
    assert best is not None
    assert best.outcome == "H"
    assert best.stake_fraction <= cfg.max_stake_fraction


def test_brier_ece_and_clv_metrics() -> None:
    # Perfect predictions -> Brier == 0.0, ECE == 0.0
    y = np.array([0, 1, 2])
    p_perfect = np.eye(3)
    assert multiclass_brier_score(y, p_perfect) == pytest.approx(0.0, abs=1e-12)
    ece_perf, _ = expected_calibration_error(y, p_perfect, n_bins=5)
    assert ece_perf == pytest.approx(0.0, abs=1e-12)

    # Uniform predictions [1/3, 1/3, 1/3] -> Brier == 2/3
    p_unif = np.full((3, 3), 1.0 / 3.0)
    assert multiclass_brier_score(y, p_unif) == pytest.approx(2.0 / 3.0, abs=1e-12)

    # CLV: took 2.20, closed at 2.00 -> raw CLV = +10%
    raw_clv, novig_clv = compute_clv(taken_odds=2.20, closing_odds=2.00, closing_fair_prob=0.49)
    assert raw_clv == pytest.approx(0.10, abs=1e-12)
    assert novig_clv == pytest.approx(0.49 * 2.20 - 1.0, abs=1e-12)
