"""Versioned data structures shared across pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Team = Literal["team_a", "team_b", "unknown"]
Role = Literal["player", "goalkeeper", "referee", "other", "unknown"]


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def footpoint(self) -> Point2D:
        return Point2D(self.x + self.width / 2.0, self.y + self.height)


@dataclass(frozen=True)
class Frame:
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    source: str = "synthetic"


@dataclass(frozen=True)
class Detection:
    detection_id: str
    category: Role
    bounding_box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class TrackObservation:
    track_id: str
    detection: Detection
    frame_index: int
    confidence: float
    team: Team = "unknown"
    role: Role = "unknown"
    team_confidence: float = 0.0
    role_confidence: float = 0.0
    pitch_point: Point2D | None = None


@dataclass(frozen=True)
class TeamRolePrediction:
    team: Team
    role: Role
    team_confidence: float
    role_confidence: float


@dataclass(frozen=True)
class CalibrationState:
    valid: bool
    confidence: float
    reprojection_error: float
    shot_id: str
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def project(self, point: Point2D, frame_width: int, frame_height: int) -> Point2D:
        """Project a pixel point into normalized pitch coordinates."""

        normalized_x = point.x / frame_width
        normalized_y = point.y / frame_height
        return Point2D(
            self.scale_x * normalized_x + self.offset_x,
            self.scale_y * normalized_y + self.offset_y,
        )


@dataclass(frozen=True)
class ShapeMetrics:
    team: Team
    player_count: int
    centroid: Point2D | None
    width: float | None
    depth: float | None
    mean_pairwise_spacing: float | None
    convex_hull_area: float | None
    compactness_proxy: float | None


@dataclass(frozen=True)
class MatchState:
    frame: Frame
    shot_id: str
    calibration: CalibrationState
    players: tuple[TrackObservation, ...]
    valid: bool
    abstention_reasons: tuple[str, ...] = ()
    team_shape: tuple[ShapeMetrics, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    schema_version: str
    run_id: str
    states: tuple[MatchState, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)