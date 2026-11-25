"""Calibrated LightGBM multiclass (1X2) model with strictly chronological calibration."""

from __future__ import annotations

import logging
from typing import Self

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from football_predictor.config import GBMConfig
from football_predictor.features.feature_store import get_feature_columns
from football_predictor.models.base import BaseMatchModel, normalize_probabilities

logger = logging.getLogger(__name__)


class CalibratedGBMModel(BaseMatchModel):
    """LightGBM multiclass classifier calibrated on a strictly later temporal window."""

    name: str = "calibrated_gbm"

    def __init__(
        self,
        config: GBMConfig | None = None,
        feature_columns: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.config = config or GBMConfig()
        self.feature_columns = feature_columns or get_feature_columns()
        self.random_seed = random_seed
        self.base_estimator_: LGBMClassifier | None = None
        self.calibrator_: CalibratedClassifierCV | None = None
        self._is_calibrated: bool = False

    def _build_lgbm(self) -> LGBMClassifier:
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            subsample_freq=1,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.random_seed,
            n_jobs=1,
            verbosity=-1,
        )

    def _extract_X(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for CalibratedGBMModel: {missing[:5]}")
        return df[self.feature_columns].astype(np.float64)

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit LightGBM on earlier matches and calibrate strictly on the tail temporal window."""
        if df.empty or "target" not in df.columns:
            raise ValueError("CalibratedGBMModel.fit requires a non-empty DataFrame with 'target'.")

        ordered = (
            df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
            if {"match_date", "match_id"}.issubset(df.columns)
            else df.reset_index(drop=True)
        )

        X_all = self._extract_X(ordered)
        y_all = ordered["target"].to_numpy(dtype=np.int64)

        n = len(ordered)
        split_idx = int(round(n * (1.0 - self.config.val_fraction)))
        split_idx = max(50, min(split_idx, n - 30)) if n >= 80 else n

        X_train, y_train = X_all.iloc[:split_idx], y_all[:split_idx]
        X_cal, y_cal = X_all.iloc[split_idx:], y_all[split_idx:]

        base_clf = self._build_lgbm()
        base_clf.fit(X_train, y_train)
        self.base_estimator_ = base_clf

        # Calibrate strictly on future-of-train chronological split if all 3 classes exist
        if len(y_cal) >= 30 and len(np.unique(y_cal)) == 3:
            cal = CalibratedClassifierCV(
                estimator=FrozenEstimator(base_clf),
                method=self.config.calibration_method,
            )
            cal.fit(X_cal, y_cal)
            self.calibrator_ = cal
            self._is_calibrated = True
            logger.debug(
                "CalibratedGBM fitted (train=%d, cal=%d, method=%s)",
                len(y_train),
                len(y_cal),
                self.config.calibration_method,
            )
        else:
            self.calibrator_ = None
            self._is_calibrated = False
            logger.warning(
                "Insufficient chronological calibration slice (%d rows); using uncalibrated LGBM.",
                len(y_cal),
            )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict calibrated 1X2 probabilities [P(H), P(D), P(A)]."""
        if self.base_estimator_ is None:
            raise RuntimeError("CalibratedGBMModel must be fitted before predict_proba.")
        X = self._extract_X(df)
        if self._is_calibrated and self.calibrator_ is not None:
            raw = self.calibrator_.predict_proba(X)
        else:
            raw = self.base_estimator_.predict_proba(X)
        return normalize_probabilities(np.asarray(raw, dtype=np.float64))

    def feature_importances(self) -> pd.DataFrame:
        """Return LightGBM split feature importances sorted descending."""
        if self.base_estimator_ is None:
            return pd.DataFrame()
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": self.base_estimator_.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
