"""Unit tests for team entity resolution and deterministic ID generation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from football_predictor.config import load_settings
from football_predictor.ingestion.entity_resolution import (
    TeamResolver,
    build_match_id,
    normalize_key,
    slugify_team,
)


@pytest.fixture()
def resolver() -> TeamResolver:
    settings = load_settings()
    return TeamResolver(settings.paths.team_mapping_path)


def test_normalize_key_strips_accents_punctuation_and_case() -> None:
    assert normalize_key("  Atlético   Madrid!! ") == "atletico madrid"
    assert normalize_key("S.S.C. Napoli") == "s s c napoli"
    assert slugify_team("Hellas Verona FC") == "hellas_verona_fc"


@pytest.mark.parametrize(
    ("raw_name", "expected_id", "expected_canonical"),
    [
        ("Milan", "ac_milan", "AC Milan"),
        ("AC Milan", "ac_milan", "AC Milan"),
        ("A.C. Milan", "ac_milan", "AC Milan"),
        ("Inter", "inter", "Inter"),
        ("Internazionale", "inter", "Inter"),
        ("Inter Milan", "inter", "Inter"),
        ("Verona", "hellas_verona", "Hellas Verona"),
        ("Hellas Verona", "hellas_verona", "Hellas Verona"),
        ("Spal", "spal", "SPAL"),
        ("SPAL", "spal", "SPAL"),
        ("AS Roma", "roma", "Roma"),
        ("Roma", "roma", "Roma"),
        ("Atlético Madrid", "atletico_madrid", "Atletico Madrid"),
        ("Man City", "manchester_city", "Manchester City"),
    ],
)
def test_known_aliases_resolve_canonically(
    resolver: TeamResolver,
    raw_name: str,
    expected_id: str,
    expected_canonical: str,
) -> None:
    resolved = resolver.resolve(raw_name, strict=True)
    assert resolved.team_id == expected_id
    assert resolved.canonical_name == expected_canonical
    assert resolved.is_known is True


def test_unknown_team_fallback_and_strict_mode(resolver: TeamResolver) -> None:
    fallback = resolver.resolve("AC Nuova Squadra 2026", strict=False)
    assert fallback.is_known is False
    assert fallback.team_id == "ac_nuova_squadra_2026"
    assert fallback.canonical_name == "AC Nuova Squadra 2026"

    with pytest.raises(KeyError):
        resolver.resolve("AC Nuova Squadra 2026", strict=True)


def test_empty_team_name_raises_value_error(resolver: TeamResolver) -> None:
    with pytest.raises(ValueError):
        resolver.resolve("   ")


def test_build_match_id_is_deterministic() -> None:
    mid_1 = build_match_id("i1", "2324", date(2024, 5, 19), "inter", "lazio")
    mid_2 = build_match_id("I1", "2324", date(2024, 5, 19), "inter", "lazio")
    assert mid_1 == mid_2 == "I1_2324_20240519_inter_vs_lazio"


def test_missing_mapping_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TeamResolver(tmp_path / "does_not_exist.yaml")
