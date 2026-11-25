"""DuckDB Medallion database manager for Bronze, Silver, and Gold layers."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBManager:
    """Manages Bronze/Silver/Gold schemas, tables, views, and Parquet exports in DuckDB."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schemas()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_schemas(self) -> None:
        """Ensure Medallion tables exist with primary keys and clean types."""
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS silver_teams (
                    team_id VARCHAR PRIMARY KEY,
                    canonical_name VARCHAR NOT NULL
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS silver_matches (
                    match_id VARCHAR PRIMARY KEY,
                    league VARCHAR NOT NULL,
                    season VARCHAR NOT NULL,
                    match_date DATE NOT NULL,
                    match_timestamp TIMESTAMP NOT NULL,
                    home_team_id VARCHAR NOT NULL,
                    away_team_id VARCHAR NOT NULL,
                    home_team VARCHAR NOT NULL,
                    away_team VARCHAR NOT NULL,
                    fthg INTEGER NOT NULL,
                    ftag INTEGER NOT NULL,
                    hthg DOUBLE,
                    htag DOUBLE,
                    ftr VARCHAR NOT NULL,
                    hs DOUBLE,
                    as_ DOUBLE,
                    hst DOUBLE,
                    ast DOUBLE,
                    psh DOUBLE,
                    psd DOUBLE,
                    psa DOUBLE,
                    avg_h DOUBLE,
                    avg_d DOUBLE,
                    avg_a DOUBLE,
                    b365_h DOUBLE,
                    b365_d DOUBLE,
                    b365_a DOUBLE,
                    psch DOUBLE,
                    pscd DOUBLE,
                    psca DOUBLE,
                    avg_ch DOUBLE,
                    avg_cd DOUBLE,
                    avg_ca DOUBLE,
                    source_file VARCHAR,
                    ingested_at VARCHAR
                );
                """
            )

    def upsert_silver_matches(
        self,
        matches_df: pd.DataFrame,
        silver_parquet_path: Path | None = None,
    ) -> int:
        """Load normalized matches and dimension teams into Silver DuckDB & Parquet."""
        if matches_df.empty:
            logger.warning("Attempted to upsert an empty matches DataFrame.")
            return 0

        home_teams = matches_df[["home_team_id", "home_team"]].rename(
            columns={"home_team_id": "team_id", "home_team": "canonical_name"}
        )
        away_teams = matches_df[["away_team_id", "away_team"]].rename(
            columns={"away_team_id": "team_id", "away_team": "canonical_name"}
        )
        teams_df = (
            pd.concat([home_teams, away_teams], ignore_index=True)
            .drop_duplicates(subset=["team_id"], keep="last")
            .sort_values("team_id")
            .reset_index(drop=True)
        )

        with self._connect() as con:
            con.register("_incoming_teams", teams_df)
            con.execute(
                """
                INSERT OR REPLACE INTO silver_teams (team_id, canonical_name)
                SELECT team_id, canonical_name FROM _incoming_teams;
                """
            )
            con.unregister("_incoming_teams")

            con.register("_incoming_matches", matches_df)
            con.execute(
                """
                INSERT OR REPLACE INTO silver_matches
                SELECT * FROM _incoming_matches;
                """
            )
            con.unregister("_incoming_matches")

            # Clean analytical view ordered chronologically
            con.execute(
                """
                CREATE OR REPLACE VIEW v_silver_matches AS
                SELECT *
                FROM silver_matches
                ORDER BY match_timestamp ASC, match_id ASC;
                """
            )

            count = con.execute("SELECT COUNT(*) FROM silver_matches").fetchone()[0]

            if silver_parquet_path is not None:
                silver_parquet_path.parent.mkdir(parents=True, exist_ok=True)
                escaped = str(silver_parquet_path).replace("'", "''")
                con.execute(
                    f"""
                    COPY (
                        SELECT * FROM silver_matches
                        ORDER BY match_timestamp ASC, match_id ASC
                    ) TO '{escaped}' (FORMAT PARQUET);
                    """
                )
                logger.info("Exported Silver Parquet -> %s", silver_parquet_path)

        logger.info("Silver DuckDB updated: %d total matches across %d teams", count, len(teams_df))
        return int(count)

    def read_silver_matches(
        self,
        league: str | None = None,
        seasons: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read chronologically ordered matches from Silver DuckDB."""
        clauses: list[str] = []
        params: list[object] = []
        if league is not None:
            clauses.append("league = ?")
            params.append(league.upper())
        if seasons:
            placeholders = ", ".join("?" for _ in seasons)
            clauses.append(f"season IN ({placeholders})")
            params.extend(seasons)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT *
            FROM silver_matches
            {where_sql}
            ORDER BY match_timestamp ASC, match_id ASC
        """
        with self._connect() as con:
            df = con.execute(query, params).df()
        return df

    def read_silver_teams(self) -> pd.DataFrame:
        """Read canonical teams dimension table from Silver DuckDB."""
        with self._connect() as con:
            return con.execute("SELECT * FROM silver_teams ORDER BY canonical_name ASC").df()

    def register_gold_view(self, gold_parquet_path: Path | str) -> None:
        """Create or refresh a DuckDB Gold view pointing directly at the Gold Parquet store."""
        parquet_path = Path(gold_parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Gold feature store not found: {parquet_path}")
        escaped = str(parquet_path).replace("'", "''")
        with self._connect() as con:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW gold_features AS
                SELECT * FROM read_parquet('{escaped}');
                """
            )
        logger.info("Registered DuckDB view 'gold_features' -> %s", parquet_path)

    def get_season_summary(self) -> pd.DataFrame:
        """Return per-season match counts and date spans for data quality auditing."""
        query = """
            SELECT
                league,
                season,
                COUNT(*) AS n_matches,
                MIN(match_date) AS first_date,
                MAX(match_date) AS last_date,
                SUM(
                    CASE WHEN psh IS NOT NULL AND psd IS NOT NULL AND psa IS NOT NULL
                    THEN 1 ELSE 0 END
                ) AS n_with_open_odds,
                SUM(
                    CASE WHEN psch IS NOT NULL AND pscd IS NOT NULL AND psca IS NOT NULL
                    THEN 1 ELSE 0 END
                ) AS n_with_close_odds
            FROM silver_matches
            GROUP BY league, season
            ORDER BY league, season
        """
        with self._connect() as con:
            return con.execute(query).df()
