"""Bookmaker margin removal (devigging) utilities for 1X2 and multi-way odds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.optimize import brentq


def _validate_odds(odds: Sequence[float]) -> np.ndarray:
    arr = np.asarray(odds, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError("Odds must be a 1D sequence of at least 2 decimal odds.")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 1.0):
        raise ValueError(f"All decimal odds must be finite and strictly > 1.0; got {odds}")
    return arr


def overround(odds: Sequence[float]) -> float:
    """Return bookmaker overround (vig) = sum(1 / odd_i) - 1."""
    arr = _validate_odds(odds)
    return float(np.sum(1.0 / arr) - 1.0)


def devig_multiplicative(odds: Sequence[float]) -> np.ndarray:
    """Remove bookmaker margin proportionally via multiplicative normalization.

    p_i = (1 / o_i) / sum_j (1 / o_j)
    """
    arr = _validate_odds(odds)
    raw_probs = 1.0 / arr
    return raw_probs / float(np.sum(raw_probs))


def devig_power(odds: Sequence[float]) -> np.ndarray:
    """Remove bookmaker margin via the Power method (Clarke, Kovalchik & Ingram, 2017).

    Finds exponent k > 0 such that sum_i (1 / o_i)^k = 1.
    When the bookmaker has positive overround (sum(1/o_i) > 1), k > 1, which removes
    proportionally more margin from longshots than favorites (correcting favorite-longshot bias).
    """
    arr = _validate_odds(odds)
    raw_probs = 1.0 / arr
    total_implied = float(np.sum(raw_probs))

    if np.isclose(total_implied, 1.0, atol=1e-12):
        return raw_probs / total_implied

    def objective(k: float) -> float:
        return float(np.sum(np.power(raw_probs, k)) - 1.0)

    # For overround > 0 (total_implied > 1), root lies in (1, upper); for arbitrage < 1, in (low, 1)
    low, high = (1.0, 2.0) if total_implied > 1.0 else (1e-4, 1.0)
    if total_implied > 1.0:
        while objective(high) > 0.0 and high < 100.0:
            high *= 2.0
    else:
        while objective(low) < 0.0 and low > 1e-8:
            low *= 0.5

    k_star = float(brentq(objective, low, high, xtol=1e-12, rtol=1e-12))
    fair_probs = np.power(raw_probs, k_star)
    return fair_probs / float(np.sum(fair_probs))


def devig_odds(
    odds: Sequence[float],
    method: Literal["multiplicative", "power"] = "power",
) -> np.ndarray:
    """Convert raw decimal odds to fair (vig-free) probabilities summing to 1."""
    if method == "multiplicative":
        return devig_multiplicative(odds)
    if method == "power":
        return devig_power(odds)
    raise ValueError(f"Unknown devigging method '{method}'. Expected 'multiplicative' or 'power'.")
