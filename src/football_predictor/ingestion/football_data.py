"""Downloader and normalizer for historical football-data.co.uk match and odds CSVs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from football_predictor.config import AppSettings
from football_predictor.ingestion.entity_resolution import TeamResolver, build_match_id

logger = logging.getLogger(__name__)


def _parse_match_date(series: pd.Series) -> pd.Series:
    """Parse football-data.co.uk date strings (DD/MM/YYYY or DD/MM/YY) deterministically."""
    parsed = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed_short = pd.to_datetime(series[missing], format="%d/%m/%y", errors="coerce")
        parsed = parsed.fillna(parsed_short)
    still_missing = parsed.isna()
    if still_missing.any():
        parsed_fallback = pd.to_datetime(series[still_missing], dayfirst=True, errors="coerce")
        parsed = parsed.fillna(parsed_fallback)
    return parsed.dt.date


def _parse_kickoff_time(series: pd.Series | None) -> pd.Series:
    """Normalize optional HH:MM kickoff time strings (defaults to 15:00 when absent)."""
    if series is None:
        return pd.Series(dtype="object")
    cleaned = series.astype(str).str.strip()
    valid = cleaned.str.match(r"^\d{1,2}:\d{2}$", na=False)
    return cleaned.where(valid, "15:00")


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Safely extract a numeric column, returning NaNs if absent."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _coalesce_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first non-null numeric value across candidate columns."""
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").astype("float64")
            result = result.fillna(vals)
    return result


class FootballDataIngestor:
    """Fetches raw seasonal CSVs into Bronze and produces normalized Silver DataFrames."""

    def __init__(self, settings: AppSettings, resolver: TeamResolver | None = None) -> None:
        self.settings = settings
        self.resolver = resolver or TeamResolver(settings.paths.team_mapping_path)
        self.bronze_dir = settings.paths.bronze_dir
        self.bronze_dir.mkdir(parents=True, exist_ok=True)

    def season_url(self, season: str, league_code: str | None = None) -> str:
        """Construct the football-data.co.uk CSV URL for a season and league."""
        league = league_code or self.settings.ingestion.league_code
        base = self.settings.ingestion.football_data_base_url.rstrip("/")
        return f"{base}/{season}/{league}.csv"

    def bronze_path(self, season: str, league_code: str | None = None) -> Path:
        """Return local Bronze CSV path for a given season and league."""
        league = league_code or self.settings.ingestion.league_code
        return self.bronze_dir / f"{league}_{season}.csv"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    def _fetch_bytes(self, url: str) -> bytes:
        """Download CSV content with redirect support and retries."""
        headers = {"User-Agent": "football-quant-predictor/0.1 (+https://github.com/quant-sports)"}
        timeout = self.settings.ingestion.request_timeout_seconds
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

    def download_season(
        self,
        season: str,
        league_code: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download raw CSV to data/bronze if absent (or when force=True)."""
        dest = self.bronze_path(season, league_code)
        if dest.exists() and not force and dest.stat().st_size > 100:
            logger.info("Bronze file already cached: %s", dest.name)
            return dest

        url = self.season_url(season, league_code)
        logger.info("Downloading %s -> %s", url, dest)
        content = self._fetch_bytes(url)
        if len(content) < 100 or b"HomeTeam" not in content[:2048]:
            raise ValueError(f"Unexpected CSV payload downloaded from {url} ({len(content)} bytes)")

        dest.write_bytes(content)
        logger.info("Saved Bronze CSV %s (%d bytes)", dest.name, len(content))
        return dest

    def download_all(self, force: bool = False) -> list[tuple[str, Path]]:
        """Download all configured seasons for the active league."""
        league = self.settings.ingestion.league_code
        paths: list[tuple[str, Path]] = []
        for season in self.settings.ingestion.seasons:
            csv_path = self.download_season(season, league_code=league, force=force)
            paths.append((season, csv_path))
        return paths

    def parse_bronze_csv(
        self,
        csv_path: Path,
        season: str,
        league_code: str | None = None,
    ) -> pd.DataFrame:
        """Parse a single Bronze CSV into a normalized Silver-schema DataFrame."""
        league = (league_code or self.settings.ingestion.league_code).upper()
        raw = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")

        # Drop empty trailing rows common in football-data.co.uk exports
        required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        missing_req = [c for c in required_cols if c not in raw.columns]
        if missing_req:
            raise ValueError(f"{csv_path.name} missing required columns: {missing_req}")

        df = raw.dropna(subset=required_cols).copy()
        df["match_date"] = _parse_match_date(df["Date"])
        df = df.dropna(subset=["match_date"]).copy()

        kickoff = (
            _parse_kickoff_time(df["Time"])
            if "Time" in df.columns
            else pd.Series("15:00", index=df.index)
        )
        match_ts = pd.to_datetime(
            df["match_date"].astype(str) + " " + kickoff.astype(str),
            errors="coerce",
        )
        # Fallback timestamp if any time failed to parse
        fallback_ts = pd.to_datetime(df["match_date"].astype(str) + " 15:00")
        df["match_timestamp"] = match_ts.fillna(fallback_ts)

        home_resolved = [self.resolver.resolve(str(name)) for name in df["HomeTeam"]]
        away_resolved = [self.resolver.resolve(str(name)) for name in df["AwayTeam"]]

        df["home_team_id"] = [t.team_id for t in home_resolved]
        df["away_team_id"] = [t.team_id for t in away_resolved]
        df["home_team"] = [t.canonical_name for t in home_resolved]
        df["away_team"] = [t.canonical_name for t in away_resolved]

        fthg = pd.to_numeric(df["FTHG"], errors="coerce")
        ftag = pd.to_numeric(df["FTAG"], errors="coerce")
        valid_goals = fthg.notna() & ftag.notna()
        df = df.loc[valid_goals].copy()
        fthg = fthg.loc[valid_goals].astype("int64")
        ftag = ftag.loc[valid_goals].astype("int64")

        result_series = np.where(fthg > ftag, "H", np.where(fthg == ftag, "D", "A"))

        match_ids = [
            build_match_id(league, season, m_date, h_id, a_id)
            for m_date, h_id, a_id in zip(
                df["match_date"], df["home_team_id"], df["away_team_id"], strict=True
            )
        ]

        silver = pd.DataFrame(
            {
                "match_id": match_ids,
                "league": league,
                "season": str(season),
                "match_date": df["match_date"].values,
                "match_timestamp": df["match_timestamp"].values,
                "home_team_id": df["home_team_id"].values,
                "away_team_id": df["away_team_id"].values,
                "home_team": df["home_team"].values,
                "away_team": df["away_team"].values,
                "fthg": fthg.values,
                "ftag": ftag.values,
                "hthg": _to_numeric(df, "HTHG").values,
                "htag": _to_numeric(df, "HTAG").values,
                "ftr": result_series,
                "hs": _to_numeric(df, "HS").values,
                "as_": _to_numeric(df, "AS").values,
                "hst": _to_numeric(df, "HST").values,
                "ast": _to_numeric(df, "AST").values,
                # Opening odds: Pinnacle (PSH/PSD/PSA), Market Average (AvgH/AvgD/AvgA), Bet365
                "psh": _coalesce_numeric(df, ["PSH", "AvgH", "B365H"]).values,
                "psd": _coalesce_numeric(df, ["PSD", "AvgD", "B365D"]).values,
                "psa": _coalesce_numeric(df, ["PSA", "AvgA", "B365A"]).values,
                "avg_h": _coalesce_numeric(df, ["AvgH", "B365H"]).values,
                "avg_d": _coalesce_numeric(df, ["AvgD", "B365D"]).values,
                "avg_a": _coalesce_numeric(df, ["AvgA", "B365A"]).values,
                "b365_h": _to_numeric(df, "B365H").values,
                "b365_d": _to_numeric(df, "B365D").values,
                "b365_a": _to_numeric(df, "B365A").values,
                # Closing odds: Pinnacle Closing (PSCH/PSCD/PSCA) & Market Avg Closing
                "psch": _coalesce_numeric(df, ["PSCH", "AvgCH", "B365CH"]).values,
                "pscd": _coalesce_numeric(df, ["PSCD", "AvgCD", "B365CD"]).values,
                "psca": _coalesce_numeric(df, ["PSCA", "AvgCA", "B365CA"]).values,
                "avg_ch": _to_numeric(df, "AvgCH").values,
                "avg_cd": _to_numeric(df, "AvgCD").values,
                "avg_ca": _to_numeric(df, "AvgCA").values,
                "source_file": csv_path.name,
                "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )

        # Deduplicate defensively on match_id keeping last record
        silver = silver.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)
        logger.info("Parsed %d valid matches from %s", len(silver), csv_path.name)
        return silver

    def load_all_normalized(self, force_download: bool = False) -> pd.DataFrame:
        """Download and normalize all configured seasons into a single chronological DataFrame."""
        downloaded = self.download_all(force=force_download)
        league = self.settings.ingestion.league_code
        frames = [
            self.parse_bronze_csv(csv_path, season=season, league_code=league)
            for season, csv_path in downloaded
        ]
        if not frames:
            raise RuntimeError("No seasonal data frames were ingested.")

        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=["match_id"], keep="last")
            .sort_values(["match_timestamp", "match_id"])
            .reset_index(drop=True)
        )
        return combined
