"""Point-in-time Gold Feature Store with strict anti-leakage guarantees.

All features for a match on date `t` are computed strictly from matches with
`match_date < t`. State updates (EMAs, internal Elo, last match date) happen
only after all matches on calendar date `t` have been featurized.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from football_predictor.config import AppSettings, FeaturesConfig
from football_predictor.db.duckdb_manager import DuckDBManager
from football_predictor.ingestion.club_elo import ClubEloClient

logger = logging.getLogger(__name__)

# Feature columns strictly safe for predictive modeling (100% pre-match, zero post-match leakage)
BASE_FEATURE_COLUMNS: list[str] = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "elo_expected_home",
    "rest_days_home",
    "rest_days_away",
    "rest_days_diff",
    "home_matches_played",
    "away_matches_played",
]


def get_feature_columns(ema_spans: list[int] | None = None) -> list[str]:
    """Return ordered list of all strictly pre-match feature column names."""
    spans = ema_spans or [5, 10]
    cols = list(BASE_FEATURE_COLUMNS)
    for span in spans:
        cols.extend(
            [
                f"home_gf_ema_{span}",
                f"home_ga_ema_{span}",
                f"home_sot_for_ema_{span}",
                f"home_sot_against_ema_{span}",
                f"home_pts_ema_{span}",
                f"away_gf_ema_{span}",
                f"away_ga_ema_{span}",
                f"away_sot_for_ema_{span}",
                f"away_sot_against_ema_{span}",
                f"away_pts_ema_{span}",
                f"home_venue_gf_ema_{span}",
                f"home_venue_ga_ema_{span}",
                f"home_venue_sot_for_ema_{span}",
                f"home_venue_sot_against_ema_{span}",
                f"away_venue_gf_ema_{span}",
                f"away_venue_ga_ema_{span}",
                f"away_venue_sot_for_ema_{span}",
                f"away_venue_sot_against_ema_{span}",
                f"gf_ema_{span}_diff",
                f"ga_ema_{span}_diff",
                f"sot_for_ema_{span}_diff",
                f"sot_against_ema_{span}_diff",
                f"pts_ema_{span}_diff",
            ]
        )
    return cols


@dataclass
class _TeamRollingState:
    """Mutable causal state for a single club strictly prior to the current match date."""

    elo: float
    last_match_date: date | None = None
    matches_played: int = 0
    overall_ema: dict[int, dict[str, float]] = field(default_factory=dict)
    home_ema: dict[int, dict[str, float]] = field(default_factory=dict)
    away_ema: dict[int, dict[str, float]] = field(default_factory=dict)

    def _update_ema_dict(
        self,
        store: dict[int, dict[str, float]],
        spans: list[int],
        obs: dict[str, float],
    ) -> None:
        for span in spans:
            alpha = 2.0 / (span + 1.0)
            if span not in store:
                store[span] = dict(obs)
            else:
                cur = store[span]
                for key, val in obs.items():
                    if not math.isnan(val):
                        prev = cur.get(key, val)
                        cur[key] = val if math.isnan(prev) else (alpha * val + (1.0 - alpha) * prev)

    def record_match(
        self,
        match_date: date,
        is_home: bool,
        gf: float,
        ga: float,
        sot_for: float,
        sot_against: float,
        points: float,
        spans: list[int],
    ) -> None:
        obs = {
            "gf": gf,
            "ga": ga,
            "sot_for": sot_for,
            "sot_against": sot_against,
            "pts": points,
        }
        self._update_ema_dict(self.overall_ema, spans, obs)
        if is_home:
            self._update_ema_dict(self.home_ema, spans, obs)
        else:
            self._update_ema_dict(self.away_ema, spans, obs)
        self.last_match_date = match_date
        self.matches_played += 1


def _to_date(value: object) -> date:
    """Convert Timestamp / datetime / str / date safely to datetime.date."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if hasattr(value, "date") and callable(value.date):
        return value.date()  # type: ignore[return-value]
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _elo_expected_score(rating_home: float, rating_away: float, home_advantage: float) -> float:
    """Logistic Elo win expectancy for the home team."""
    diff = (rating_home + home_advantage) - rating_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _elo_update(
    rating_home: float,
    rating_away: float,
    fthg: int,
    ftag: int,
    k_factor: float,
    home_advantage: float,
) -> tuple[float, float]:
    """Update internal Elo ratings post-match with goal-difference margin scaling."""
    actual_home = 1.0 if fthg > ftag else (0.5 if fthg == ftag else 0.0)
    expected_home = _elo_expected_score(rating_home, rating_away, home_advantage)
    gd = abs(fthg - ftag)
    # Mild logarithmic margin-of-victory multiplier (1.0 for draw/1-goal)
    if gd <= 1:
        mov = 1.0
    else:
        elo_gap = (rating_home - rating_away) if actual_home == 1.0 else (rating_away - rating_home)
        mov = math.log(gd + 1.0) * (2.2 / (elo_gap * 0.001 + 2.2))
    mov = max(1.0, min(mov, 2.5))
    delta = k_factor * mov * (actual_home - expected_home)
    return rating_home + delta, rating_away - delta


class PointInTimeFeatureStore:
    """Builds leak-free point-in-time Gold features from Silver matches."""

    # Prior league averages used strictly as cold-start defaults before a team has history
    _PRIOR_GF: float = 1.35
    _PRIOR_GA: float = 1.35
    _PRIOR_SOT: float = 4.30
    _PRIOR_PTS: float = 1.35

    def __init__(
        self,
        config: FeaturesConfig,
        club_elo_client: ClubEloClient | None = None,
    ) -> None:
        self.config = config
        self.club_elo_client = club_elo_client

    def _get_state(self, states: dict[str, _TeamRollingState], team_id: str) -> _TeamRollingState:
        if team_id not in states:
            states[team_id] = _TeamRollingState(elo=self.config.elo_initial_rating)
        return states[team_id]

    def _extract_ema(
        self,
        store: dict[int, dict[str, float]],
        fallback_store: dict[int, dict[str, float]],
        span: int,
        metric: str,
        prior: float,
    ) -> float:
        if span in store and not math.isnan(store[span].get(metric, math.nan)):
            return float(store[span][metric])
        if span in fallback_store and not math.isnan(fallback_store[span].get(metric, math.nan)):
            return float(fallback_store[span][metric])
        return prior

    def _compute_rest_days(self, last_date: date | None, current_date: date) -> float:
        if last_date is None:
            return float(self.config.default_rest_days)
        delta_days = (current_date - last_date).days
        if delta_days <= 0:
            return 1.0
        return float(min(delta_days, self.config.max_rest_days))

    def _resolve_pre_match_elo(
        self,
        team_id: str,
        match_date: date,
        internal_elo: float,
    ) -> float:
        """Prefer ClubElo snapshot at (match_date - 1 day) if available; else internal Elo."""
        if self.club_elo_client is not None:
            external = self.club_elo_client.get_pre_match_elo(team_id, match_date)
            if external is not None and not math.isnan(external):
                return float(external)
        return float(internal_elo)

    def _build_pair_features(
        self,
        home_id: str,
        away_id: str,
        match_date: date,
        states: dict[str, _TeamRollingState],
    ) -> dict[str, float]:
        """Compute strictly pre-match feature dictionary from current causal team states."""
        h_state = self._get_state(states, home_id)
        a_state = self._get_state(states, away_id)

        elo_home = self._resolve_pre_match_elo(home_id, match_date, h_state.elo)
        elo_away = self._resolve_pre_match_elo(away_id, match_date, a_state.elo)
        elo_diff = elo_home - elo_away
        elo_exp_home = _elo_expected_score(elo_home, elo_away, self.config.elo_home_advantage)

        rest_home = self._compute_rest_days(h_state.last_match_date, match_date)
        rest_away = self._compute_rest_days(a_state.last_match_date, match_date)

        row: dict[str, float] = {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_diff,
            "elo_expected_home": elo_exp_home,
            "rest_days_home": rest_home,
            "rest_days_away": rest_away,
            "rest_days_diff": rest_home - rest_away,
            "home_matches_played": float(h_state.matches_played),
            "away_matches_played": float(a_state.matches_played),
        }

        for span in self.config.ema_spans:
            h_gf = self._extract_ema(h_state.overall_ema, {}, span, "gf", self._PRIOR_GF)
            h_ga = self._extract_ema(h_state.overall_ema, {}, span, "ga", self._PRIOR_GA)
            h_sot_f = self._extract_ema(h_state.overall_ema, {}, span, "sot_for", self._PRIOR_SOT)
            h_sot_a = self._extract_ema(
                h_state.overall_ema, {}, span, "sot_against", self._PRIOR_SOT
            )
            h_pts = self._extract_ema(h_state.overall_ema, {}, span, "pts", self._PRIOR_PTS)

            a_gf = self._extract_ema(a_state.overall_ema, {}, span, "gf", self._PRIOR_GF)
            a_ga = self._extract_ema(a_state.overall_ema, {}, span, "ga", self._PRIOR_GA)
            a_sot_f = self._extract_ema(a_state.overall_ema, {}, span, "sot_for", self._PRIOR_SOT)
            a_sot_a = self._extract_ema(
                a_state.overall_ema, {}, span, "sot_against", self._PRIOR_SOT
            )
            a_pts = self._extract_ema(a_state.overall_ema, {}, span, "pts", self._PRIOR_PTS)

            hv_gf = self._extract_ema(
                h_state.home_ema, h_state.overall_ema, span, "gf", self._PRIOR_GF
            )
            hv_ga = self._extract_ema(
                h_state.home_ema, h_state.overall_ema, span, "ga", self._PRIOR_GA
            )
            hv_sot_f = self._extract_ema(
                h_state.home_ema, h_state.overall_ema, span, "sot_for", self._PRIOR_SOT
            )
            hv_sot_a = self._extract_ema(
                h_state.home_ema, h_state.overall_ema, span, "sot_against", self._PRIOR_SOT
            )

            av_gf = self._extract_ema(
                a_state.away_ema, a_state.overall_ema, span, "gf", self._PRIOR_GF
            )
            av_ga = self._extract_ema(
                a_state.away_ema, a_state.overall_ema, span, "ga", self._PRIOR_GA
            )
            av_sot_f = self._extract_ema(
                a_state.away_ema, a_state.overall_ema, span, "sot_for", self._PRIOR_SOT
            )
            av_sot_a = self._extract_ema(
                a_state.away_ema, a_state.overall_ema, span, "sot_against", self._PRIOR_SOT
            )

            row.update(
                {
                    f"home_gf_ema_{span}": h_gf,
                    f"home_ga_ema_{span}": h_ga,
                    f"home_sot_for_ema_{span}": h_sot_f,
                    f"home_sot_against_ema_{span}": h_sot_a,
                    f"home_pts_ema_{span}": h_pts,
                    f"away_gf_ema_{span}": a_gf,
                    f"away_ga_ema_{span}": a_ga,
                    f"away_sot_for_ema_{span}": a_sot_f,
                    f"away_sot_against_ema_{span}": a_sot_a,
                    f"away_pts_ema_{span}": a_pts,
                    f"home_venue_gf_ema_{span}": hv_gf,
                    f"home_venue_ga_ema_{span}": hv_ga,
                    f"home_venue_sot_for_ema_{span}": hv_sot_f,
                    f"home_venue_sot_against_ema_{span}": hv_sot_a,
                    f"away_venue_gf_ema_{span}": av_gf,
                    f"away_venue_ga_ema_{span}": av_ga,
                    f"away_venue_sot_for_ema_{span}": av_sot_f,
                    f"away_venue_sot_against_ema_{span}": av_sot_a,
                    f"gf_ema_{span}_diff": h_gf - a_gf,
                    f"ga_ema_{span}_diff": h_ga - a_ga,
                    f"sot_for_ema_{span}_diff": h_sot_f - a_sot_f,
                    f"sot_against_ema_{span}_diff": h_sot_a - a_sot_a,
                    f"pts_ema_{span}_diff": h_pts - a_pts,
                }
            )
        return row

    def _apply_date_batch_updates(
        self,
        day_matches: pd.DataFrame,
        states: dict[str, _TeamRollingState],
    ) -> None:
        """Apply post-match state updates ONLY after all matches on a date are featurized."""
        for rec in day_matches.itertuples(index=False):
            m_date = _to_date(rec.match_date)
            h_id = str(rec.home_team_id)
            a_id = str(rec.away_team_id)
            fthg = int(rec.fthg)
            ftag = int(rec.ftag)
            hst = float(rec.hst) if pd.notna(rec.hst) else math.nan
            ast = float(rec.ast) if pd.notna(rec.ast) else math.nan

            h_state = self._get_state(states, h_id)
            a_state = self._get_state(states, a_id)

            new_h_elo, new_a_elo = _elo_update(
                rating_home=h_state.elo,
                rating_away=a_state.elo,
                fthg=fthg,
                ftag=ftag,
                k_factor=self.config.elo_k_factor,
                home_advantage=self.config.elo_home_advantage,
            )
            h_state.elo = new_h_elo
            a_state.elo = new_a_elo

            h_pts = 3.0 if fthg > ftag else (1.0 if fthg == ftag else 0.0)
            a_pts = 3.0 if ftag > fthg else (1.0 if fthg == ftag else 0.0)

            h_state.record_match(
                match_date=m_date,
                is_home=True,
                gf=float(fthg),
                ga=float(ftag),
                sot_for=hst,
                sot_against=ast,
                points=h_pts,
                spans=self.config.ema_spans,
            )
            a_state.record_match(
                match_date=m_date,
                is_home=False,
                gf=float(ftag),
                ga=float(fthg),
                sot_for=ast,
                sot_against=hst,
                points=a_pts,
                spans=self.config.ema_spans,
            )

    def build_features(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Transform chronological Silver matches into point-in-time Gold features.

        Guarantees strict causality: for every match on date `t`, features depend
        ONLY on matches with `match_date < t`.
        """
        if silver_df.empty:
            raise ValueError("Cannot build features from an empty Silver DataFrame.")

        df = silver_df.copy()
        df["_parsed_date"] = [_to_date(v) for v in df["match_date"]]
        df = df.sort_values(["_parsed_date", "match_timestamp", "match_id"]).reset_index(drop=True)

        states: dict[str, _TeamRollingState] = {}
        feature_rows: list[dict[str, object]] = []

        # Group strictly by calendar date so no same-day match can leak into another
        for current_date, day_group in df.groupby("_parsed_date", sort=True):
            # Step 1: Read strictly pre-match state (< current_date) for ALL matches on this day
            for rec in day_group.itertuples(index=False):
                feats = self._build_pair_features(
                    home_id=str(rec.home_team_id),
                    away_id=str(rec.away_team_id),
                    match_date=current_date,
                    states=states,
                )
                row_meta: dict[str, object] = {
                    "match_id": rec.match_id,
                    "league": rec.league,
                    "season": rec.season,
                    "match_date": current_date,
                    "match_timestamp": rec.match_timestamp,
                    "home_team_id": rec.home_team_id,
                    "away_team_id": rec.away_team_id,
                    "home_team": rec.home_team,
                    "away_team": rec.away_team,
                    # Targets & market odds retained for training/backtesting (excluded from X)
                    "fthg": int(rec.fthg),
                    "ftag": int(rec.ftag),
                    "ftr": rec.ftr,
                    "target": 0 if rec.ftr == "H" else (1 if rec.ftr == "D" else 2),
                    "psh": getattr(rec, "psh", np.nan),
                    "psd": getattr(rec, "psd", np.nan),
                    "psa": getattr(rec, "psa", np.nan),
                    "avg_h": getattr(rec, "avg_h", np.nan),
                    "avg_d": getattr(rec, "avg_d", np.nan),
                    "avg_a": getattr(rec, "avg_a", np.nan),
                    "psch": getattr(rec, "psch", np.nan),
                    "pscd": getattr(rec, "pscd", np.nan),
                    "psca": getattr(rec, "psca", np.nan),
                }
                row_meta.update(feats)
                feature_rows.append(row_meta)

            # Step 2: Update team states ONLY after all matches on `current_date` are processed
            self._apply_date_batch_updates(day_group, states)

        gold_df = pd.DataFrame(feature_rows)
        gold_df = gold_df.sort_values(["match_date", "match_timestamp", "match_id"]).reset_index(
            drop=True
        )
        logger.info(
            "Built Gold feature store: %d matches x %d features",
            len(gold_df),
            len(get_feature_columns(self.config.ema_spans)),
        )
        return gold_df

    def build_inference_row(
        self,
        silver_df: pd.DataFrame,
        home_team_id: str,
        away_team_id: str,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute a single pre-match feature vector for an upcoming fixture on `as_of_date`.

        Strictly filters `silver_df` to matches with `match_date < as_of_date`.
        """
        df = silver_df.copy()
        df["_parsed_date"] = [_to_date(v) for v in df["match_date"]]
        history = df.loc[df["_parsed_date"] < as_of_date].sort_values(
            ["_parsed_date", "match_timestamp", "match_id"]
        )

        states: dict[str, _TeamRollingState] = {}
        for _, day_group in history.groupby("_parsed_date", sort=True):
            self._apply_date_batch_updates(day_group, states)

        feats = self._build_pair_features(
            home_id=home_team_id,
            away_id=away_team_id,
            match_date=as_of_date,
            states=states,
        )
        return pd.DataFrame([feats])

    def save_gold_parquet(self, gold_df: pd.DataFrame, output_path: Path | str) -> Path:
        """Persist Gold features to Parquet and return path."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        gold_df.to_parquet(out, index=False)
        logger.info("Saved Gold features -> %s (%d rows)", out, len(gold_df))
        return out


def build_and_persist_gold_features(settings: AppSettings) -> pd.DataFrame:
    """End-to-end helper: read Silver from DuckDB, build Gold Parquet, register Gold view."""
    from football_predictor.ingestion.entity_resolution import TeamResolver

    db = DuckDBManager(settings.paths.duckdb_path)
    silver_df = db.read_silver_matches(league=settings.ingestion.league_code)
    resolver = TeamResolver(settings.paths.team_mapping_path)
    club_elo = ClubEloClient(settings.ingestion.club_elo, resolver=resolver)

    store = PointInTimeFeatureStore(config=settings.features, club_elo_client=club_elo)
    gold_df = store.build_features(silver_df)

    gold_path = settings.paths.gold_dir / "features.parquet"
    store.save_gold_parquet(gold_df, gold_path)
    db.register_gold_view(gold_path)
    return gold_df
