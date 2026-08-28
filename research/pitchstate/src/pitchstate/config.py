"""Configuration loading and validation for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tomllib


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    frame_width: int
    frame_height: int
    sample_fps: float
    min_players_per_team: int


@dataclass(frozen=True)
class QualityConfig:
    minimum_calibration_confidence: float
    maximum_reprojection_error: float
    minimum_player_confidence: float
    minimum_team_confidence: float
    minimum_role_confidence: float


@dataclass(frozen=True)
class LoggingConfig:
    run_directory: str
    level: str


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: str
    project_name: str
    experiment_name: str
    seed: int
    runtime: RuntimeConfig
    quality: QualityConfig
    logging: LoggingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required config value: [{section}] {key}")
    return mapping[key]


def _bounded_float(value: Any, key: str, minimum: float, maximum: float) -> float:
    if not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
        raise ConfigError(f"{key} must be a number between {minimum} and {maximum}")
    return float(value)


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a TOML project configuration."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    schema_version = _require(raw, "schema_version", "root")
    project_name = _require(raw, "project_name", "root")
    experiment_name = _require(raw, "experiment_name", "root")
    seed = _require(raw, "seed", "root")
    if not isinstance(schema_version, str) or not schema_version:
        raise ConfigError("schema_version must be a non-empty string")
    if not isinstance(project_name, str) or not project_name:
        raise ConfigError("project_name must be a non-empty string")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ConfigError("experiment_name must be a non-empty string")
    if not isinstance(seed, int) or seed < 0:
        raise ConfigError("seed must be a non-negative integer")

    runtime = raw.get("runtime")
    quality = raw.get("quality")
    logging = raw.get("logging")
    if not isinstance(runtime, dict) or not isinstance(quality, dict) or not isinstance(logging, dict):
        raise ConfigError("runtime, quality, and logging sections are required")

    frame_width = _require(runtime, "frame_width", "runtime")
    frame_height = _require(runtime, "frame_height", "runtime")
    sample_fps = _require(runtime, "sample_fps", "runtime")
    min_players = _require(runtime, "min_players_per_team", "runtime")
    if not isinstance(frame_width, int) or frame_width <= 0:
        raise ConfigError("runtime.frame_width must be a positive integer")
    if not isinstance(frame_height, int) or frame_height <= 0:
        raise ConfigError("runtime.frame_height must be a positive integer")
    if not isinstance(sample_fps, (int, float)) or float(sample_fps) <= 0:
        raise ConfigError("runtime.sample_fps must be positive")
    if not isinstance(min_players, int) or min_players < 1:
        raise ConfigError("runtime.min_players_per_team must be at least 1")

    runtime_config = RuntimeConfig(frame_width, frame_height, float(sample_fps), min_players)
    quality_config = QualityConfig(
        _bounded_float(
            _require(quality, "minimum_calibration_confidence", "quality"),
            "quality.minimum_calibration_confidence",
            0.0,
            1.0,
        ),
        _bounded_float(
            _require(quality, "maximum_reprojection_error", "quality"),
            "quality.maximum_reprojection_error",
            0.0,
            1.0,
        ),
        _bounded_float(
            _require(quality, "minimum_player_confidence", "quality"),
            "quality.minimum_player_confidence",
            0.0,
            1.0,
        ),
        _bounded_float(
            _require(quality, "minimum_team_confidence", "quality"),
            "quality.minimum_team_confidence",
            0.0,
            1.0,
        ),
        _bounded_float(
            _require(quality, "minimum_role_confidence", "quality"),
            "quality.minimum_role_confidence",
            0.0,
            1.0,
        ),
    )
    run_directory = _require(logging, "run_directory", "logging")
    level = _require(logging, "level", "logging")
    if not isinstance(run_directory, str) or not run_directory:
        raise ConfigError("logging.run_directory must be a non-empty string")
    if not isinstance(level, str) or level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("logging.level must be DEBUG, INFO, WARNING, or ERROR")

    return ProjectConfig(
        schema_version=schema_version,
        project_name=project_name,
        experiment_name=experiment_name,
        seed=seed,
        runtime=runtime_config,
        quality=quality_config,
        logging=LoggingConfig(run_directory, level.upper()),
    )