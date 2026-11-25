"""Log-Loss optimal stacking ensemble combining Dixon-Coles and Calibrated LightGBM."""

from __future__ import annotations

import logging
from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from football_predictor.config import ModelsConfig
from football_predictor.features.feature_store import get_feature_columns
from football_predictor.models.base import (
    BaseMatchModel,
    MarketProbabilities,
    normalize_probabilities,
)
from football_predictor.models.dixon_coles import DixonColesModel
from football_predictor.models.gbm_model import CalibratedGBMModel

logger = logging.getLogger(__name__)


class LogLossEnsembleModel(BaseMatchModel):
    """Convex combination p_ens = w * p_DC + (1 - w) * p_GBM optimized on temporal Log-Loss."""

    name: str = "ensemble"

    def __init__(
        self,
        config: ModelsConfig | None = None,
        feature_columns: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.config = config or ModelsConfig()
        self.feature_columns = feature_columns or get_feature_columns()
        self.random_seed = random_seed

        self.dc_model = DixonColesModel(config=self.config.dixon_coles)
        self.gbm_model = CalibratedGBMModel(
            config=self.config.gbm,
            feature_columns=self.feature_columns,
            random_seed=self.random_seed,
        )
        self.weight_dc_: float = 0.5
        self.val_log_loss_dc_: float = float("nan")
        self.val_log_loss_gbm_: float = float("nan")
        self.val_log_loss_ens_: float = float("nan")

    def set_dc_warm_start(self, prev_dc: DixonColesModel) -> None:
        """Warm-start internal Dixon-Coles optimization from a prior walk-forward model."""
        self.dc_model.set_warm_start(prev_dc)

    def fit(self, df: pd.DataFrame) -> Self:
        """Optimize blending weight w on a strict chronological holdout, then refit on full df."""
        if df.empty or "target" not in df.columns:
            raise ValueError(
                "LogLossEnsembleModel.fit requires a non-empty DataFrame with 'target'."
            )

        ordered = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
        n = len(ordered)
        val_frac = float(self.config.ensemble.val_fraction)
        split_idx = int(round(n * (1.0 - val_frac)))

        if n >= 160 and 80 <= split_idx <= n - 40:
            fit_df = ordered.iloc[:split_idx]
            val_df = ordered.iloc[split_idx:]
            y_val = val_df["target"].to_numpy(dtype=np.int64)

            val_dc = DixonColesModel(config=self.config.dixon_coles)
            if self.dc_model._warm_state is not None:
                val_dc._warm_state = self.dc_model._warm_state
            val_dc.fit(fit_df)

            val_gbm = CalibratedGBMModel(
                config=self.config.gbm,
                feature_columns=self.feature_columns,
                random_seed=self.random_seed,
            )
            val_gbm.fit(fit_df)

            p_dc = val_dc.predict_proba(val_df)
            p_gbm = val_gbm.predict_proba(val_df)

            self.val_log_loss_dc_ = float(log_loss(y_val, p_dc, labels=[0, 1, 2]))
            self.val_log_loss_gbm_ = float(log_loss(y_val, p_gbm, labels=[0, 1, 2]))

            def _objective(w: float) -> float:
                blended = normalize_probabilities(w * p_dc + (1.0 - w) * p_gbm)
                return float(log_loss(y_val, blended, labels=[0, 1, 2]))

            opt = minimize_scalar(_objective, bounds=(0.0, 1.0), method="bounded")
            self.weight_dc_ = float(np.clip(opt.x, 0.0, 1.0))
            self.val_log_loss_ens_ = float(opt.fun)

            # Warm-start final DC refit from val_dc for faster convergence
            self.dc_model.set_warm_start(val_dc)
            logger.info(
                "Ensemble weight optimized on chronological val (%d matches): "
                "w_DC=%.3f | LogLoss DC=%.4f, GBM=%.4f, ENS=%.4f",
                len(val_df),
                self.weight_dc_,
                self.val_log_loss_dc_,
                self.val_log_loss_gbm_,
                self.val_log_loss_ens_,
            )
        else:
            self.weight_dc_ = 0.5
            logger.warning(
                "Small dataset (%d rows); defaulting ensemble weight w_DC=0.50",
                n,
            )

        # Refit base models on the complete training window
        self.dc_model.fit(ordered)
        self.gbm_model.fit(ordered)
        return self

    def predict_components(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p_ensemble, p_dixon_coles, p_gbm) each of shape (len(df), 3)."""
        p_dc = self.dc_model.predict_proba(df)
        p_gbm = self.gbm_model.predict_proba(df)
        p_ens = normalize_probabilities(self.weight_dc_ * p_dc + (1.0 - self.weight_dc_) * p_gbm)
        return p_ens, p_dc, p_gbm

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ensemble 1X2 probabilities [P(H), P(D), P(A)]."""
        p_ens, _, _ = self.predict_components(df)
        return p_ens

    def predict_match_full(
        self,
        home_team_id: str,
        away_team_id: str,
        features_row: pd.DataFrame,
        max_goals: int | None = None,
    ) -> dict[str, object]:
        """Return full prediction bundle for a single upcoming fixture."""
        dc_markets, score_matrix = self.dc_model.predict_markets(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            max_goals=max_goals,
        )
        p_dc = np.array([dc_markets.p_home, dc_markets.p_draw, dc_markets.p_away], dtype=np.float64)
        p_gbm = self.gbm_model.predict_proba(features_row)[0]
        p_ens = normalize_probabilities(self.weight_dc_ * p_dc + (1.0 - self.weight_dc_) * p_gbm)

        ens_markets = MarketProbabilities(
            p_home=float(p_ens[0]),
            p_draw=float(p_ens[1]),
            p_away=float(p_ens[2]),
            p_over_25=dc_markets.p_over_25,
            p_under_25=dc_markets.p_under_25,
            p_btts_yes=dc_markets.p_btts_yes,
            p_btts_no=dc_markets.p_btts_no,
            expected_home_goals=dc_markets.expected_home_goals,
            expected_away_goals=dc_markets.expected_away_goals,
            top_scorelines=dc_markets.top_scorelines,
        )
        return {
            "markets": ens_markets,
            "score_matrix": score_matrix,
            "p_ensemble": p_ens,
            "p_dixon_coles": p_dc,
            "p_gbm": p_gbm,
            "weight_dc": self.weight_dc_,
        }
