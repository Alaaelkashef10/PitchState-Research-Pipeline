"""Reference-annotation mapping and validation.

This module keeps raw reference labels separate from derived tactical metrics.
Ambiguous labels map to ``unknown``; malformed records are reported rather than
silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from pitchstate.schema import BoundingBox, Point2D, Role, Team


@dataclass(frozen=True)
class ReferenceAnnotation:
    game_id: str
    clip_id: str
    frame_index: int
    track_id: str
    identity: str
    role: Role
    team: Team
    bounding_box: BoundingBox | None
    pitch_point: Point2D | None


@dataclass(frozen=True)
class AnnotationIssue:
    record_index: int
    severity: str
    field: str
    message: str


@dataclass(frozen=True)
class AnnotationValidationReport:
    record_count: int
    valid_record_count: int
    issues: tuple[AnnotationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def _label(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ") if value is not None else ""


def map_team_label(value: Any) -> Team:
    normalized = _label(value)
    if normalized in {"team left", "left", "team a", "a"}:
        return "team_a"
    if normalized in {"team right", "right", "team b", "b"}:
        return "team_b"
    return "unknown"


def map_role_label(value: Any) -> Role:
    normalized = _label(value)
    if normalized in {"player", "outfield player"}:
        return "player"
    if normalized in {"goalkeeper", "goalie", "keeper"}:
        return "goalkeeper"
    if normalized in {"referee", "main referee", "side referee"}:
        return "referee"
    if normalized in {"other", "staff"}:
        return "other"
    return "unknown"


def map_identity_label(value: Any) -> str:
    if value is None or not str(value).strip():
        return "unknown"
    normalized = _label(value)
    return "unknown" if normalized in {"unknown", "ambiguous", "none", "nan"} else str(value)


def _number(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def map_annotation_record(record: Mapping[str, Any]) -> ReferenceAnnotation:
    """Map a verified raw record without inventing missing coordinates."""

    bbox = None
    box_values = [_number(record, key) for key in ("x", "y", "width", "height")]
    if all(value is not None for value in box_values):
        bbox = BoundingBox(*[float(value) for value in box_values])

    pitch_values = [_number(record, key) for key in ("pitch_x", "pitch_y")]
    pitch_point = (
        Point2D(float(pitch_values[0]), float(pitch_values[1]))
        if all(value is not None for value in pitch_values)
        else None
    )
    return ReferenceAnnotation(
        game_id=str(record.get("game_id", "")),
        clip_id=str(record.get("clip_id", "")),
        frame_index=int(record.get("frame_index", -1)),
        track_id=str(record.get("track_id", "")),
        identity=map_identity_label(record.get("identity")),
        role=map_role_label(record.get("role")),
        team=map_team_label(record.get("team")),
        bounding_box=bbox,
        pitch_point=pitch_point,
    )


def validate_annotation_records(records: Iterable[Mapping[str, Any]]) -> AnnotationValidationReport:
    """Return an issue report; callers decide whether warnings are acceptable."""

    records_list = list(records)
    issues: list[AnnotationIssue] = []
    valid_count = 0
    for index, record in enumerate(records_list):
        if not isinstance(record, Mapping):
            issues.append(AnnotationIssue(index, "error", "record", "record must be an object"))
            continue
        required = ("game_id", "clip_id", "frame_index", "track_id")
        missing = [key for key in required if key not in record or record[key] in ("", None)]
        for key in missing:
            issues.append(AnnotationIssue(index, "error", key, "required value is missing"))
        frame_index = record.get("frame_index")
        if not isinstance(frame_index, int) or frame_index < 0:
            issues.append(AnnotationIssue(index, "error", "frame_index", "must be a non-negative integer"))
        box_present = any(key in record for key in ("x", "y", "width", "height"))
        box_complete = all(key in record for key in ("x", "y", "width", "height"))
        if box_present and not box_complete:
            issues.append(AnnotationIssue(index, "error", "bounding_box", "box is partially specified"))
        if box_complete:
            values = [_number(record, key) for key in ("x", "y", "width", "height")]
            if any(value is None for value in values) or values[2] < 0 or values[3] < 0:
                issues.append(AnnotationIssue(index, "error", "bounding_box", "box values are invalid"))
        pitch_present = any(key in record for key in ("pitch_x", "pitch_y"))
        pitch_complete = all(key in record for key in ("pitch_x", "pitch_y"))
        if pitch_present and not pitch_complete:
            issues.append(AnnotationIssue(index, "error", "pitch_point", "pitch point is partially specified"))
        if map_team_label(record.get("team")) == "unknown":
            issues.append(AnnotationIssue(index, "warning", "team", "ambiguous or missing team mapped to unknown"))
        if map_role_label(record.get("role")) == "unknown":
            issues.append(AnnotationIssue(index, "warning", "role", "ambiguous or missing role mapped to unknown"))
        if map_identity_label(record.get("identity")) == "unknown":
            issues.append(AnnotationIssue(index, "warning", "identity", "ambiguous or missing identity mapped to unknown"))
        if not any(issue.record_index == index and issue.severity == "error" for issue in issues):
            valid_count += 1
    return AnnotationValidationReport(len(records_list), valid_count, tuple(issues))