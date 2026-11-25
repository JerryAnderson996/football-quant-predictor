"""CLI entrypoint for Phase 1 ingestion: Bronze CSV download -> Silver DuckDB & Parquet."""

from __future__ import annotations

import argparse
import logging

from football_predictor.config import load_settings, setup_logging
from football_predictor.db.duckdb_manager import DuckDBManager
from football_predictor.ingestion.entity_resolution import TeamResolver
from football_predictor.ingestion.football_data import FootballDataIngestor

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical Serie A CSVs and load Silver DuckDB / Parquet."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to settings.yaml override.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download raw Bronze CSV files even if already cached locally.",
    )
    return parser.parse_args()


def run_ingestion(config_path: str | None = None, force: bool = False) -> int:
    """Execute Bronze -> Silver ingestion pipeline and return total match count."""
    settings = load_settings(config_path)
    setup_logging(settings.log_level)

    logger.info(
        "Starting ingestion for league=%s seasons=%s",
        settings.ingestion.league_code,
        settings.ingestion.seasons,
    )
    resolver = TeamResolver(settings.paths.team_mapping_path)
    ingestor = FootballDataIngestor(settings=settings, resolver=resolver)
    matches_df = ingestor.load_all_normalized(force_download=force)

    db = DuckDBManager(settings.paths.duckdb_path)
    silver_parquet = settings.paths.silver_dir / "matches.parquet"
    total_count = db.upsert_silver_matches(matches_df, silver_parquet_path=silver_parquet)

    summary = db.get_season_summary()
    logger.info("Silver summary by season:\n%s", summary.to_string(index=False))
    return total_count


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(config_path=args.config, force=args.force)
