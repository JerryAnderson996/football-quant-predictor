"""Entity resolution for canonicalizing team names and generating stable IDs."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_key(name: str) -> str:
    """Normalize a raw team string for accent-, case-, and punctuation-insensitive lookup."""
    decomposed = unicodedata.normalize("NFKD", name.strip())
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = _NON_ALNUM_RE.sub(" ", ascii_only.lower()).strip()
    return cleaned


def slugify_team(name: str) -> str:
    """Create a stable snake_case identifier from a canonical team name."""
    key = normalize_key(name)
    return key.replace(" ", "_")


@dataclass(frozen=True)
class ResolvedTeam:
    """Canonical representation of a football club."""

    team_id: str
    canonical_name: str
    is_known: bool = True


class TeamResolver:
    """Resolves raw club names from heterogeneous sources to canonical IDs."""

    def __init__(self, mapping_path: Path | str | None = None) -> None:
        self._alias_to_team: dict[str, ResolvedTeam] = {}
        self._id_to_team: dict[str, ResolvedTeam] = {}
        self._unknown_seen: set[str] = set()

        if mapping_path is not None:
            self.load_mapping(Path(mapping_path))

    def load_mapping(self, mapping_path: Path) -> None:
        """Load canonical teams and aliases from a YAML configuration file."""
        if not mapping_path.exists():
            raise FileNotFoundError(f"Team mapping file not found: {mapping_path}")

        with mapping_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}

        teams_cfg: dict[str, dict[str, object]] = payload.get("teams", {})
        for canonical_name, meta in teams_cfg.items():
            team_id = str(meta.get("team_id") or slugify_team(canonical_name))
            resolved = ResolvedTeam(
                team_id=team_id,
                canonical_name=canonical_name,
                is_known=True,
            )
            self._id_to_team[team_id] = resolved
            self._register_alias(canonical_name, resolved)
            self._register_alias(team_id, resolved)

            aliases = meta.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    self._register_alias(str(alias), resolved)

        logger.debug(
            "Loaded %d canonical teams (%d aliases)",
            len(self._id_to_team),
            len(self._alias_to_team),
        )

    def _register_alias(self, alias: str, resolved: ResolvedTeam) -> None:
        key = normalize_key(alias)
        if not key:
            return
        existing = self._alias_to_team.get(key)
        if existing is not None and existing.team_id != resolved.team_id:
            logger.warning(
                "Alias collision for '%s': keeping '%s' over '%s'",
                alias,
                existing.team_id,
                resolved.team_id,
            )
            return
        self._alias_to_team[key] = resolved

    def resolve(self, raw_name: str, strict: bool = False) -> ResolvedTeam:
        """Resolve a raw team name to a canonical ResolvedTeam.

        Args:
            raw_name: Team name string as reported by an external source.
            strict: If True, raise KeyError when the team is missing from mapping.
        """
        if not raw_name or not raw_name.strip():
            raise ValueError("Team name cannot be empty.")

        key = normalize_key(raw_name)
        found = self._alias_to_team.get(key)
        if found is not None:
            return found

        if strict:
            raise KeyError(f"Unknown team name '{raw_name}' (normalized='{key}')")

        cleaned_name = " ".join(raw_name.strip().split())
        fallback = ResolvedTeam(
            team_id=slugify_team(cleaned_name),
            canonical_name=cleaned_name,
            is_known=False,
        )
        if key not in self._unknown_seen:
            self._unknown_seen.add(key)
            logger.warning(
                "Unmapped team '%s' -> fallback canonical='%s', team_id='%s'",
                raw_name,
                fallback.canonical_name,
                fallback.team_id,
            )
        return fallback

    def resolve_id(self, raw_name: str, strict: bool = False) -> str:
        """Return only the canonical team_id for a raw name."""
        return self.resolve(raw_name, strict=strict).team_id

    def resolve_name(self, raw_name: str, strict: bool = False) -> str:
        """Return only the canonical display name for a raw name."""
        return self.resolve(raw_name, strict=strict).canonical_name

    @property
    def canonical_teams(self) -> list[ResolvedTeam]:
        """Return all known canonical teams sorted by canonical_name."""
        return sorted(self._id_to_team.values(), key=lambda t: t.canonical_name)


def build_match_id(
    league: str,
    season: str,
    match_date: date,
    home_team_id: str,
    away_team_id: str,
) -> str:
    """Build a deterministic, unique primary key for a football match."""
    date_tag = match_date.strftime("%Y%m%d")
    return f"{league.upper()}_{season}_{date_tag}_{home_team_id}_vs_{away_team_id}"
