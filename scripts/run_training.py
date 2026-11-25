"""CLI entrypoint for training and persisting the production Ensemble model."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import joblib
import pandas as pd

from football_predictor.config import load_settings, setup_logging
from football_predictor.features.feature_store import (
    build_and_persist_gold_features,
    get_feature_columns,
)
from football_predictor.models.ensemble import LogLossEnsembleModel

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Gold features (if needed) and train production Ensemble model."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional settings.yaml path.")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Force rebuild of Gold feature store from Silver DuckDB.",
    )
    return parser.parse_args()


def run_training(
    config_path: str | None = None,
    rebuild_features: bool = False,
) -> dict[str, object]:
    """Train Dixon-Coles + Calibrated LightGBM + Log-Loss Ensemble and save artifacts."""
    settings = load_settings(config_path)
    setup_logging(settings.log_level)

    gold_path = settings.paths.gold_dir / "features.parquet"
    if rebuild_features or not gold_path.exists():
        gold_df = build_and_persist_gold_features(settings)
    else:
        gold_df = pd.read_parquet(gold_path)

    feat_cols = get_feature_columns(settings.features.ema_spans)
    ensemble = LogLossEnsembleModel(
        config=settings.models,
        feature_columns=feat_cols,
        random_seed=settings.project.random_seed,
    )
    ensemble.fit(gold_df)

    artifacts_dir = settings.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = artifacts_dir / "model_bundle.joblib"
    joblib.dump(
        {
            "ensemble": ensemble,
            "feature_columns": feat_cols,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_matches": len(gold_df),
            "last_match_date": str(pd.to_datetime(gold_df["match_date"]).max().date()),
        },
        bundle_path,
    )

    ratings_df = ensemble.dc_model.get_team_ratings()
    ratings_path = artifacts_dir / "team_ratings.parquet"
    ratings_df.to_parquet(ratings_path, index=False)

    summary: dict[str, object] = {
        "n_matches": len(gold_df),
        "n_teams": len(ensemble.dc_model.idx_to_team),
        "weight_dc": round(ensemble.weight_dc_, 4),
        "weight_gbm": round(1.0 - ensemble.weight_dc_, 4),
        "dc_home_advantage": round(ensemble.dc_model.home_adv_, 4),
        "dc_rho": round(ensemble.dc_model.rho_, 4),
        "dc_converged": ensemble.dc_model.converged_,
        "val_log_loss_dc": round(ensemble.val_log_loss_dc_, 5),
        "val_log_loss_gbm": round(ensemble.val_log_loss_gbm_, 5),
        "val_log_loss_ensemble": round(ensemble.val_log_loss_ens_, 5),
        "bundle_path": str(bundle_path),
    }

    summary_path = artifacts_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Saved model bundle -> %s", bundle_path)
    logger.info("Training summary: %s", json.dumps(summary))
    logger.info(
        "Top 10 Serie A Power Ratings (Dixon-Coles):\n%s",
        ratings_df.head(10).to_string(index=False),
    )
    return summary


if __name__ == "__main__":
    args = parse_args()
    run_training(config_path=args.config, rebuild_features=args.rebuild_features)
