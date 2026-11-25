"""FastAPI application exposing /predict, /value-bets, /backtest/summary, and /metrics."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from football_predictor.betting.devig import devig_odds, overround
from football_predictor.betting.strategy import (
    ValueBetSignal,
    evaluate_match_bets,
    select_best_value_bet,
)
from football_predictor.config import AppSettings, load_settings, setup_logging
from football_predictor.db.duckdb_manager import DuckDBManager
from football_predictor.features.feature_store import PointInTimeFeatureStore
from football_predictor.ingestion.entity_resolution import TeamResolver
from football_predictor.models.base import MarketProbabilities
from football_predictor.models.ensemble import LogLossEnsembleModel

logger = logging.getLogger(__name__)


@dataclass
class _AppState:
    settings: AppSettings
    resolver: TeamResolver
    feature_store: PointInTimeFeatureStore
    silver_df: pd.DataFrame
    ensemble: LogLossEnsembleModel | None = None
    trained_at: str | None = None
    last_match_date: str | None = None


_STATE: _AppState | None = None


def _load_runtime_state() -> _AppState:
    settings = load_settings()
    setup_logging(settings.log_level)

    resolver = TeamResolver(settings.paths.team_mapping_path)
    feature_store = PointInTimeFeatureStore(config=settings.features)

    silver_df = pd.DataFrame()
    if settings.paths.duckdb_path.exists():
        db = DuckDBManager(settings.paths.duckdb_path)
        silver_df = db.read_silver_matches(league=settings.ingestion.league_code)

    ensemble: LogLossEnsembleModel | None = None
    trained_at: str | None = None
    last_match_date: str | None = None

    bundle_path = settings.paths.artifacts_dir / "model_bundle.joblib"
    if bundle_path.exists():
        payload = joblib.load(bundle_path)
        ensemble = payload["ensemble"]
        trained_at = payload.get("trained_at")
        last_match_date = payload.get("last_match_date")
        logger.info("Loaded trained model bundle from %s", bundle_path)
    else:
        logger.warning("Model bundle not found at %s; run scripts/run_training.py.", bundle_path)

    return _AppState(
        settings=settings,
        resolver=resolver,
        feature_store=feature_store,
        silver_df=silver_df,
        ensemble=ensemble,
        trained_at=trained_at,
        last_match_date=last_match_date,
    )


def _get_state() -> _AppState:
    global _STATE
    if _STATE is None:
        _STATE = _load_runtime_state()
    return _STATE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _STATE
    _STATE = _load_runtime_state()
    yield


app = FastAPI(
    title="Football Quant Predictor API",
    version="0.1.0",
    description=(
        "Quantitative Serie A match probability calibration, Dixon-Coles scoreline matrix, "
        "and Value Bet / Fractional Kelly API."
    ),
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Input schema for single-match prediction and optional Value Bet evaluation."""

    home_team: str = Field(..., min_length=1, examples=["Inter"])
    away_team: str = Field(..., min_length=1, examples=["AC Milan"])
    match_date: date | None = Field(
        default=None,
        description="Target match date (defaults to today; strictly uses matches < match_date).",
    )
    odds_home: float | None = Field(default=None, gt=1.0, examples=[1.95])
    odds_draw: float | None = Field(default=None, gt=1.0, examples=[3.60])
    odds_away: float | None = Field(default=None, gt=1.0, examples=[4.10])
    bankroll: float | None = Field(default=None, gt=0.0, examples=[1000.0])
    devig_method: Literal["multiplicative", "power"] | None = None
    max_goals: int = Field(default=8, ge=4, le=12)

    @model_validator(mode="after")
    def _validate_odds_triplet(self) -> PredictRequest:
        odds = [self.odds_home, self.odds_draw, self.odds_away]
        if any(o is not None for o in odds) and not all(o is not None for o in odds):
            raise ValueError("Provide all three odds (odds_home, odds_draw, odds_away) or none.")
        return self


class PredictResponse(BaseModel):
    """Output schema for /predict."""

    home_team: str
    away_team: str
    home_team_id: str
    away_team_id: str
    as_of_date: date
    markets: MarketProbabilities
    fair_odds: dict[str, float]
    model_components: dict[str, object]
    score_matrix: list[list[float]]
    market_overround_pct: float | None = None
    market_fair_probs: dict[str, float] | None = None
    betting_signals: list[ValueBetSignal] = Field(default_factory=list)
    best_value_bet: ValueBetSignal | None = None


class ValueBetsBatchRequest(BaseModel):
    """Batch fixture request for /value-bets."""

    fixtures: list[PredictRequest] = Field(..., min_length=1)


class ValueBetMatchItem(BaseModel):
    """Qualified value bet attached to a fixture."""

    home_team: str
    away_team: str
    as_of_date: date
    signal: ValueBetSignal


def _predict_single(req: PredictRequest, state: _AppState) -> PredictResponse:
    if state.ensemble is None:
        raise HTTPException(
            status_code=503,
            detail="Model bundle not loaded. Run scripts/run_training.py first.",
        )

    home_res = state.resolver.resolve(req.home_team)
    away_res = state.resolver.resolve(req.away_team)
    if home_res.team_id == away_res.team_id:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")

    as_of = req.match_date or date.today()
    if state.silver_df.empty:
        raise HTTPException(
            status_code=503,
            detail="Silver match history not loaded. Run scripts/run_ingestion.py first.",
        )

    feats_row = state.feature_store.build_inference_row(
        silver_df=state.silver_df,
        home_team_id=home_res.team_id,
        away_team_id=away_res.team_id,
        as_of_date=as_of,
    )
    bundle = state.ensemble.predict_match_full(
        home_team_id=home_res.team_id,
        away_team_id=away_res.team_id,
        features_row=feats_row,
        max_goals=req.max_goals,
    )

    markets: MarketProbabilities = bundle["markets"]  # type: ignore[assignment]
    score_matrix = bundle["score_matrix"]
    p_ens = bundle["p_ensemble"]
    p_dc = bundle["p_dixon_coles"]
    p_gbm = bundle["p_gbm"]

    fair_odds = {
        "H": round(1.0 / max(markets.p_home, 1e-6), 4),
        "D": round(1.0 / max(markets.p_draw, 1e-6), 4),
        "A": round(1.0 / max(markets.p_away, 1e-6), 4),
    }

    betting_signals: list[ValueBetSignal] = []
    best_bet: ValueBetSignal | None = None
    overround_pct: float | None = None
    market_fair_probs: dict[str, float] | None = None

    if req.odds_home is not None and req.odds_draw is not None and req.odds_away is not None:
        odds_triplet = [req.odds_home, req.odds_draw, req.odds_away]
        bet_cfg = state.settings.betting.model_copy(
            update={"devig_method": req.devig_method or state.settings.betting.devig_method}
        )
        bankroll = req.bankroll or bet_cfg.initial_bankroll
        betting_signals = evaluate_match_bets(
            model_probs=[markets.p_home, markets.p_draw, markets.p_away],
            bookmaker_odds=odds_triplet,
            bankroll=bankroll,
            config=bet_cfg,
        )
        best_bet = select_best_value_bet(betting_signals)
        overround_pct = round(overround(odds_triplet) * 100.0, 3)
        fair_mkt = devig_odds(odds_triplet, method=bet_cfg.devig_method)
        market_fair_probs = {
            "H": round(float(fair_mkt[0]), 5),
            "D": round(float(fair_mkt[1]), 5),
            "A": round(float(fair_mkt[2]), 5),
        }

    return PredictResponse(
        home_team=home_res.canonical_name,
        away_team=away_res.canonical_name,
        home_team_id=home_res.team_id,
        away_team_id=away_res.team_id,
        as_of_date=as_of,
        markets=markets,
        fair_odds=fair_odds,
        model_components={
            "weight_dc": round(float(bundle["weight_dc"]), 4),
            "ensemble": [round(float(x), 5) for x in p_ens],
            "dixon_coles": [round(float(x), 5) for x in p_dc],
            "gbm": [round(float(x), 5) for x in p_gbm],
        },
        score_matrix=[[round(float(v), 6) for v in row] for row in score_matrix],
        market_overround_pct=overround_pct,
        market_fair_probs=market_fair_probs,
        betting_signals=betting_signals,
        best_value_bet=best_bet,
    )


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness and model readiness probe."""
    state = _get_state()
    return {
        "status": "ok",
        "model_loaded": state.ensemble is not None,
        "n_silver_matches": int(len(state.silver_df)),
        "trained_at": state.trained_at,
        "last_match_date": state.last_match_date,
    }


@app.get("/teams")
def list_teams() -> dict[str, object]:
    """Return available teams and current Dixon-Coles attack/defense ratings."""
    state = _get_state()
    if state.ensemble is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")
    ratings = state.ensemble.dc_model.get_team_ratings()
    return {
        "n_teams": int(len(ratings)),
        "home_advantage": round(state.ensemble.dc_model.home_adv_, 4),
        "rho": round(state.ensemble.dc_model.rho_, 4),
        "teams": ratings.to_dict(orient="records"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict calibrated probabilities, scoreline matrix, fair odds, and Value Bet / Kelly."""
    state = _get_state()
    return _predict_single(req, state)


@app.post("/value-bets")
def value_bets(req: ValueBetsBatchRequest) -> dict[str, object]:
    """Screen a slate of fixtures with bookmaker odds and return ranked positive-EV bets."""
    state = _get_state()
    found: list[ValueBetMatchItem] = []
    for fixture in req.fixtures:
        pred = _predict_single(fixture, state)
        for sig in pred.betting_signals:
            if sig.is_value_bet and sig.stake_fraction > 0.0:
                found.append(
                    ValueBetMatchItem(
                        home_team=pred.home_team,
                        away_team=pred.away_team,
                        as_of_date=pred.as_of_date,
                        signal=sig,
                    )
                )
    found.sort(key=lambda item: item.signal.expected_value, reverse=True)
    return {"n_fixtures": len(req.fixtures), "n_value_bets": len(found), "value_bets": found}


@app.get("/backtest/summary")
def backtest_summary() -> dict[str, object]:
    """Return the persisted Walk-Forward Backtest report."""
    state = _get_state()
    report_path = state.settings.paths.artifacts_dir / "backtest_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Backtest report not found. Run scripts/run_backtest.py first.",
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


@app.get("/metrics")
def metrics() -> dict[str, object]:
    """Return calibration, betting, and training metrics summary."""
    state = _get_state()
    artifacts = state.settings.paths.artifacts_dir
    out: dict[str, object] = {}

    train_path = artifacts / "training_summary.json"
    if train_path.exists():
        out["training"] = json.loads(train_path.read_text(encoding="utf-8"))

    bt_path = artifacts / "backtest_report.json"
    if bt_path.exists():
        report = json.loads(bt_path.read_text(encoding="utf-8"))
        out["backtest"] = {
            "test_seasons": report.get("test_seasons"),
            "n_matches": report.get("n_matches"),
            "calibration_metrics": report.get("calibration_metrics"),
            "betting_metrics": report.get("betting_metrics"),
        }
    if not out:
        raise HTTPException(status_code=404, detail="No training or backtest metrics found.")
    return out
