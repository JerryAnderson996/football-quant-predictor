"""Expected Value (EV) computation and Fractional Kelly Criterion staking strategy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from football_predictor.betting.devig import devig_odds
from football_predictor.config import BettingConfig
from football_predictor.models.base import OUTCOME_LABELS


class ValueBetSignal(BaseModel):
    """Evaluated betting opportunity for a single 1X2 outcome."""

    outcome: str
    model_prob: float
    market_fair_prob: float
    bookmaker_odds: float
    fair_odds: float
    expected_value: float
    edge: float
    kelly_full_fraction: float
    stake_fraction: float
    recommended_stake: float
    is_value_bet: bool


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """Compute expected return per unit staked: EV = p * odd - 1."""
    if not (0.0 <= model_prob <= 1.0):
        raise ValueError(f"model_prob must be in [0, 1]; got {model_prob}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0; got {decimal_odds}")
    return float(model_prob * decimal_odds - 1.0)


def kelly_full_fraction(model_prob: float, decimal_odds: float) -> float:
    """Compute full Kelly Criterion fraction f* = max(0, (p * o - 1) / (o - 1))."""
    ev = expected_value(model_prob, decimal_odds)
    if ev <= 0.0:
        return 0.0
    return float(ev / (decimal_odds - 1.0))


def fractional_kelly_stake(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_stake_fraction: float = 0.03,
) -> tuple[float, float]:
    """Compute capped Fractional Kelly stake fraction and monetary stake amount.

    Returns:
        (stake_fraction, stake_amount)
    """
    if bankroll <= 0.0:
        return 0.0, 0.0
    if kelly_fraction <= 0.0 or max_stake_fraction <= 0.0:
        raise ValueError("kelly_fraction and max_stake_fraction must be strictly positive.")

    f_full = kelly_full_fraction(model_prob, decimal_odds)
    if f_full <= 0.0:
        return 0.0, 0.0

    f_scaled = min(kelly_fraction * f_full, max_stake_fraction)
    return float(f_scaled), float(f_scaled * bankroll)


def evaluate_match_bets(
    model_probs: Sequence[float],
    bookmaker_odds: Sequence[float],
    bankroll: float,
    config: BettingConfig | None = None,
) -> list[ValueBetSignal]:
    """Evaluate all 1X2 outcomes for a match against bookmaker odds."""
    cfg = config or BettingConfig()
    probs = np.asarray(model_probs, dtype=np.float64)
    odds = np.asarray(bookmaker_odds, dtype=np.float64)
    if probs.shape != (3,) or odds.shape != (3,):
        raise ValueError("model_probs and bookmaker_odds must each have length 3 (H, D, A).")

    fair_market = devig_odds(odds, method=cfg.devig_method)
    signals: list[ValueBetSignal] = []

    for idx, label in enumerate(OUTCOME_LABELS):
        p = float(probs[idx])
        o = float(odds[idx])
        p_mkt = float(fair_market[idx])
        ev = expected_value(p, o)
        f_full = kelly_full_fraction(p, o)
        in_odds_range = cfg.min_odds <= o <= cfg.max_odds
        is_val = (ev > cfg.min_ev_threshold) and in_odds_range

        if is_val:
            stake_frac, stake_amt = fractional_kelly_stake(
                model_prob=p,
                decimal_odds=o,
                bankroll=bankroll,
                kelly_fraction=cfg.kelly_fraction,
                max_stake_fraction=cfg.max_stake_fraction,
            )
        else:
            stake_frac, stake_amt = 0.0, 0.0

        signals.append(
            ValueBetSignal(
                outcome=label,
                model_prob=p,
                market_fair_prob=p_mkt,
                bookmaker_odds=o,
                fair_odds=float(1.0 / max(p, 1e-6)),
                expected_value=ev,
                edge=p - p_mkt,
                kelly_full_fraction=f_full,
                stake_fraction=stake_frac,
                recommended_stake=stake_amt,
                is_value_bet=is_val,
            )
        )
    return signals


def select_best_value_bet(signals: list[ValueBetSignal]) -> ValueBetSignal | None:
    """Select at most one value bet per match (highest positive EV) to avoid correlated bets."""
    qualifying = [s for s in signals if s.is_value_bet and s.stake_fraction > 0.0]
    if not qualifying:
        return None
    return max(qualifying, key=lambda s: s.expected_value)
