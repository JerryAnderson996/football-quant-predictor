"""CLI entrypoint for Phase 4 Walk-Forward Backtest on historical Serie A seasons."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from football_predictor.betting.backtester import WalkForwardBacktester
from football_predictor.config import load_settings, setup_logging
from football_predictor.features.feature_store import build_and_persist_gold_features

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict expanding-window Walk-Forward Backtest on Serie A."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional settings.yaml path.")
    return parser.parse_args()


def run_backtest(config_path: str | None = None) -> dict[str, object]:
    """Load Gold features, run Walk-Forward Backtester, and log metrics summary."""
    settings = load_settings(config_path)
    setup_logging(settings.log_level)

    gold_path = settings.paths.gold_dir / "features.parquet"
    gold_df = (
        pd.read_parquet(gold_path)
        if gold_path.exists()
        else build_and_persist_gold_features(settings)
    )

    backtester = WalkForwardBacktester(settings=settings)
    report = backtester.run(gold_df, save_artifacts=True)

    logger.info("=== WALK-FORWARD BACKTEST SUMMARY (Seasons: %s) ===", report["test_seasons"])
    logger.info("Calibration metrics:\n%s", json.dumps(report["calibration_metrics"], indent=2))
    logger.info("Betting metrics:\n%s", json.dumps(report["betting_metrics"], indent=2))
    logger.info("Per-season breakdown:\n%s", json.dumps(report["per_season"], indent=2))
    return report


if __name__ == "__main__":
    args = parse_args()
    run_backtest(config_path=args.config)
