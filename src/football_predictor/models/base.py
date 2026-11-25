"""Abstract base interface and prediction schemas for football match models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

OUTCOME_LABELS: tuple[str, str, str] = ("H", "D", "A")
OUTCOME_TO_IDX: dict[str, int] = {"H": 0, "D": 1, "A": 2}


class ScorelineProbability(BaseModel):
    """Single exact scoreline probability entry."""

    score: str
    home_goals: int
    away_goals: int
    probability: float


class MarketProbabilities(BaseModel):
    """Derived market probabilities from a match prediction / scoreline distribution."""

    p_home: float = Field(ge=0.0, le=1.0)
    p_draw: float = Field(ge=0.0, le=1.0)
    p_away: float = Field(ge=0.0, le=1.0)
    p_over_25: float | None = Field(default=None, ge=0.0, le=1.0)
    p_under_25: float | None = Field(default=None, ge=0.0, le=1.0)
    p_btts_yes: float | None = Field(default=None, ge=0.0, le=1.0)
    p_btts_no: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None
    top_scorelines: list[ScorelineProbability] = Field(default_factory=list)


def normalize_probabilities(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip and row-normalize a (..., 3) probability array so rows sum strictly to 1."""
    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0 - eps)
    if arr.ndim == 1:
        return arr / arr.sum()
    return arr / arr.sum(axis=1, keepdims=True)


class BaseMatchModel(ABC):
    """Abstract interface shared by Dixon-Coles, Calibrated GBM, and Ensemble models."""

    name: str = "base"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Self:
        """Fit the model strictly on historical training matches in `df`."""

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict 1X2 probabilities [P(H), P(D), P(A)] with shape (len(df), 3)."""
