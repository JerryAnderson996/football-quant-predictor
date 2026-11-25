"""Typed configuration and structured logging for Football Quant Predictor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root(start: Path | None = None) -> Path:
    """Locate repository root by searching upward for pyproject.toml."""
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


class ProjectConfig(BaseModel):
    """Top-level project metadata and reproducibility settings."""

    name: str = "football-quant-predictor"
    random_seed: int = 42
    default_league: str = "I1"
    supported_leagues: dict[str, str] = Field(
        default_factory=lambda: {"I1": "Serie A", "CL": "UEFA Champions League"}
    )


class PathsConfig(BaseModel):
    """Filesystem layout for Medallion layers and artifacts."""

    data_dir: Path = Path("data")
    bronze_dir: Path = Path("data/bronze")
    silver_dir: Path = Path("data/silver")
    gold_dir: Path = Path("data/gold")
    artifacts_dir: Path = Path("artifacts")
    duckdb_path: Path = Path("data/silver/football.duckdb")
    team_mapping_path: Path = Path("config/team_mapping.yaml")

    def resolve_all(self, root: Path) -> PathsConfig:
        """Return a copy with all relative paths anchored at project root."""
        resolved: dict[str, Path] = {}
        for field_name in type(self).model_fields:
            val: Path = getattr(self, field_name)
            resolved[field_name] = val if val.is_absolute() else (root / val).resolve()
        return PathsConfig(**resolved)


class ClubEloConfig(BaseModel):
    """Configuration for the resilient ClubElo API client."""

    enabled: bool = False
    base_url: str = "http://api.clubelo.com"
    cache_dir: Path = Path("data/bronze/club_elo_cache")
    timeout_seconds: float = 8.0


class IngestionConfig(BaseModel):
    """Data ingestion configuration."""

    football_data_base_url: str = "https://www.football-data.co.uk/mmz4281"
    league_code: str = "I1"
    seasons: list[str] = Field(
        default_factory=lambda: ["2021", "2122", "2223", "2324", "2425", "2526"]
    )
    request_timeout_seconds: float = 20.0
    club_elo: ClubEloConfig = Field(default_factory=ClubEloConfig)


class FeaturesConfig(BaseModel):
    """Point-in-time feature engineering hyperparameters."""

    ema_spans: list[int] = Field(default_factory=lambda: [5, 10])
    elo_initial_rating: float = 1500.0
    elo_k_factor: float = 20.0
    elo_home_advantage: float = 65.0
    default_rest_days: float = 7.0
    max_rest_days: float = 30.0
    min_history_matches: int = 3


class DixonColesConfig(BaseModel):
    """Dixon-Coles model configuration."""

    xi: float = 0.0018
    max_goals: int = 8
    l2_reg: float = 0.0005
    max_iter: int = 400


class GBMConfig(BaseModel):
    """LightGBM + probability calibration configuration."""

    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    n_estimators: int = 200
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 5.0
    val_fraction: float = 0.20


class EnsembleConfig(BaseModel):
    """Log-Loss optimal stacking configuration."""

    val_fraction: float = 0.20


class ModelsConfig(BaseModel):
    """Aggregate configuration for all predictive models."""

    dixon_coles: DixonColesConfig = Field(default_factory=DixonColesConfig)
    gbm: GBMConfig = Field(default_factory=GBMConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)


class BettingConfig(BaseModel):
    """Devigging, EV filtering, and Fractional Kelly parameters."""

    devig_method: Literal["multiplicative", "power"] = "power"
    min_ev_threshold: float = 0.03
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.03
    initial_bankroll: float = 1000.0
    min_odds: float = 1.05
    max_odds: float = 15.0


class BacktestConfig(BaseModel):
    """Walk-forward backtesting configuration."""

    n_test_seasons: int = 2
    retrain_interval_days: int = 28
    min_train_matches: int = 600
    odds_source: Literal["pinnacle_open", "market_avg_open"] = "pinnacle_open"


class EnvOverrides(BaseSettings):
    """Environment variable overrides prefixed with FQP_."""

    model_config = SettingsConfigDict(
        env_prefix="FQP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    config_path: Path | None = None
    data_dir: Path | None = None
    artifacts_dir: Path | None = None
    club_elo_enabled: bool | None = None
    api_host: str = "127.0.0.1"
    api_port: int = 8000


class AppSettings(BaseModel):
    """Validated master application configuration."""

    root_dir: Path
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    betting: BettingConfig = Field(default_factory=BettingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


def load_settings(config_path: Path | str | None = None) -> AppSettings:
    """Load settings from YAML and apply environment overrides."""
    root = find_project_root()
    env = EnvOverrides()

    resolved_cfg = (
        Path(config_path)
        if config_path is not None
        else (env.config_path if env.config_path is not None else root / "config" / "settings.yaml")
    )
    if not resolved_cfg.is_absolute():
        resolved_cfg = (root / resolved_cfg).resolve()

    raw: dict[str, Any] = {}
    if resolved_cfg.exists():
        with resolved_cfg.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    settings = AppSettings(
        root_dir=root,
        log_level=env.log_level,
        api_host=env.api_host,
        api_port=env.api_port,
        **raw,
    )

    # Resolve paths relative to root and apply optional env overrides
    paths = settings.paths.resolve_all(root)
    if env.data_dir is not None:
        base_data = env.data_dir if env.data_dir.is_absolute() else (root / env.data_dir).resolve()
        paths = PathsConfig(
            data_dir=base_data,
            bronze_dir=base_data / "bronze",
            silver_dir=base_data / "silver",
            gold_dir=base_data / "gold",
            artifacts_dir=paths.artifacts_dir,
            duckdb_path=base_data / "silver" / "football.duckdb",
            team_mapping_path=paths.team_mapping_path,
        )
    if env.artifacts_dir is not None:
        art = (
            env.artifacts_dir
            if env.artifacts_dir.is_absolute()
            else (root / env.artifacts_dir).resolve()
        )
        paths = paths.model_copy(update={"artifacts_dir": art})

    club_elo_cache = settings.ingestion.club_elo.cache_dir
    if not club_elo_cache.is_absolute():
        club_elo_cache = (root / club_elo_cache).resolve()

    club_elo_enabled = (
        env.club_elo_enabled
        if env.club_elo_enabled is not None
        else settings.ingestion.club_elo.enabled
    )
    updated_club_elo = settings.ingestion.club_elo.model_copy(
        update={"cache_dir": club_elo_cache, "enabled": club_elo_enabled}
    )
    updated_ingestion = settings.ingestion.model_copy(update={"club_elo": updated_club_elo})

    return settings.model_copy(update={"paths": paths, "ingestion": updated_ingestion})


def setup_logging(level: str = "INFO") -> None:
    """Configure structured, human-readable logging for CLI, API, and pipelines."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(numeric_level)
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)
