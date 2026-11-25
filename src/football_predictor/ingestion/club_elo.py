"""Resilient ClubElo API client with local disk caching and clean offline fallback."""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from football_predictor.config import ClubEloConfig
from football_predictor.ingestion.entity_resolution import TeamResolver

logger = logging.getLogger(__name__)


class ClubEloClient:
    """Fetches point-in-time Elo ratings from api.clubelo.com with disk cache and fallback.

    Point-in-time rule: when requesting pre-match Elo for match_date `t`, we query
    the ClubElo snapshot for `t - 1 day` to guarantee strict causality (`rating_date < t`).
    """

    def __init__(self, config: ClubEloConfig, resolver: TeamResolver) -> None:
        self.config = config
        self.resolver = resolver
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, float]] = {}
        self._offline_tripped = False

    def _cache_file(self, query_date: date) -> Path:
        return self.cache_dir / f"elo_{query_date.isoformat()}.csv"

    def _parse_snapshot(self, csv_text: str) -> dict[str, float]:
        """Parse ClubElo CSV payload into {team_id: elo_rating}."""
        if not csv_text or "Club" not in csv_text[:256]:
            return {}
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as exc:  # pragma: no cover - defensive parsing
            logger.warning("Failed to parse ClubElo snapshot CSV: %s", exc)
            return {}

        if "Club" not in df.columns or "Elo" not in df.columns:
            return {}

        ratings: dict[str, float] = {}
        for _, row in df.dropna(subset=["Club", "Elo"]).iterrows():
            team_id = self.resolver.resolve_id(str(row["Club"]))
            try:
                ratings[team_id] = float(row["Elo"])
            except (TypeError, ValueError):
                continue
        return ratings

    def get_snapshot(self, as_of_date: date) -> dict[str, float] | None:
        """Fetch ratings strictly as of `as_of_date` from cache or API.

        Returns None when disabled, offline, or rate-limited so downstream code can
        cleanly fall back to the internal incremental Elo engine.
        """
        date_key = as_of_date.isoformat()
        if date_key in self._memory_cache:
            return self._memory_cache[date_key]

        cache_path = self._cache_file(as_of_date)
        if cache_path.exists() and cache_path.stat().st_size > 20:
            parsed = self._parse_snapshot(cache_path.read_text(encoding="utf-8", errors="ignore"))
            if parsed:
                self._memory_cache[date_key] = parsed
                return parsed

        if not self.config.enabled or self._offline_tripped:
            return None

        url = f"{self.config.base_url.rstrip('/')}/{date_key}"
        try:
            with httpx.Client(timeout=self.config.timeout_seconds, follow_redirects=True) as client:
                resp = client.get(url)
                if resp.status_code == 429:
                    logger.warning("ClubElo rate-limited (HTTP 429); switching to fallback Elo.")
                    self._offline_tripped = True
                    return None
                resp.raise_for_status()
                text = resp.text
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("ClubElo unreachable (%s); switching to fallback Elo.", exc)
            self._offline_tripped = True
            return None

        cache_path.write_text(text, encoding="utf-8")
        parsed = self._parse_snapshot(text)
        if parsed:
            self._memory_cache[date_key] = parsed
            return parsed
        return None

    def get_pre_match_elo(self, team_id: str, match_date: date) -> float | None:
        """Return pre-match Elo strictly before `match_date` (using `match_date - 1 day`)."""
        snapshot = self.get_snapshot(match_date - timedelta(days=1))
        if snapshot is None:
            return None
        return snapshot.get(team_id)
