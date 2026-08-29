"""Dataset manifests, split integrity, and provenance validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a dataset manifest is malformed."""


class LeakageError(ManifestError):
    """Raised when match or clip identities cross incompatible splits."""


#: Closed vocabulary for ``validation_status`` (schema v0.3+). Values are
#: intentionally distinct from ``status``: ``status`` describes source/access
#: state, ``validation_status`` describes whether the *local* manifest content
#: has been verified against a downloaded release. Keeping them separate
#: avoids a manifest silently reading as "ready" once access is granted but
#: before the local audit in ``dataset-audit.md`` has actually happened.
VALID_VALIDATION_STATUSES = frozenset(
    {
        "not_locally_verified",
        "locally_verified",
        "source_verified_access_pending",
        "source_verified",
        "invalid",
    }
)


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    partition: str
    games: tuple[str, ...]
    clips: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class AnnotationCoverage:
    """Evidence status for one annotation family.

    Values intentionally remain strings because ``source_verified`` and
    ``locally_verified`` are materially different states.
    """

    status: str
    fields: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    dataset_id: str
    dataset_version: str
    source_url: str
    license_or_access: str
    status: str
    local_root: str
    source_checked_on: str
    release_or_download_date: str | None
    clip_structure: dict[str, Any]
    annotations: dict[str, AnnotationCoverage]
    split_strategy: str
    splits: tuple[DatasetSplit, ...]
    notes: tuple[str, ...]
    validation_status: str = "not_locally_verified"
    preprocessing_version: str | None = None


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ManifestError(f"{key} must be null or a non-empty string")
    return value


def _string_tuple(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    values = raw.get(key, [])
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ManifestError(f"{key} must be a list of non-empty strings")
    return tuple(values)


def _validate_validation_status(raw: dict[str, Any]) -> str:
    value = raw.get("validation_status")
    if value not in VALID_VALIDATION_STATUSES:
        raise ManifestError(
            "validation_status must be one of "
            f"{sorted(VALID_VALIDATION_STATUSES)}, got {value!r}"
        )
    return value


def _validate_clip_structure(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError("clip_structure must be an object")
    for key in ("description", "duration_seconds", "frame_rate_fps", "resolution"):
        if key not in value:
            raise ManifestError(f"clip_structure is missing {key}")
    for key in ("description", "resolution"):
        if not isinstance(value[key], str) or not value[key]:
            raise ManifestError(f"clip_structure.{key} must be a non-empty string")
    for key in ("duration_seconds", "frame_rate_fps"):
        number = value[key]
        if number is not None and (
            not isinstance(number, (int, float)) or number <= 0
        ):
            raise ManifestError(f"clip_structure.{key} must be null or positive")
    return dict(value)


def _validate_annotations(value: Any) -> dict[str, AnnotationCoverage]:
    if not isinstance(value, dict) or not value:
        raise ManifestError("annotations must be a non-empty object")
    annotations: dict[str, AnnotationCoverage] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name:
            raise ManifestError("annotation names must be non-empty strings")
        if not isinstance(raw, dict):
            raise ManifestError(f"annotations.{name} must be an object")
        status = raw.get("status")
        if not isinstance(status, str) or not status:
            raise ManifestError(f"annotations.{name}.status must be a non-empty string")
        fields = raw.get("fields", [])
        if not isinstance(fields, list) or not all(
            isinstance(field, str) and field for field in fields
        ):
            raise ManifestError(f"annotations.{name}.fields must be a list of strings")
        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            raise ManifestError(f"annotations.{name}.notes must be a string")
        annotations[name] = AnnotationCoverage(status, tuple(fields), notes)
    return annotations


def audit_split_integrity(manifest: DatasetManifest) -> dict[str, int]:
    """Audit match/clip identities and fail on cross-split leakage."""

    game_owners: dict[str, str] = {}
    clip_owners: dict[str, str] = {}
    duplicate_games = 0
    duplicate_clips = 0
    for split in manifest.splits:
        for game_id in split.games:
            owner = game_owners.get(game_id)
            if owner is not None:
                duplicate_games += 1
                if owner != split.name:
                    raise LeakageError(
                        f"Game {game_id!r} appears in incompatible splits "
                        f"{owner!r} and {split.name!r}"
                    )
                raise LeakageError(f"Game {game_id!r} is duplicated in split {split.name!r}")
            game_owners[game_id] = split.name
        for clip_id in split.clips:
            owner = clip_owners.get(clip_id)
            if owner is not None:
                duplicate_clips += 1
                if owner != split.name:
                    raise LeakageError(
                        f"Clip {clip_id!r} appears in incompatible splits "
                        f"{owner!r} and {split.name!r}"
                    )
                raise LeakageError(f"Clip {clip_id!r} is duplicated in split {split.name!r}")
            clip_owners[clip_id] = split.name
    return {
        "split_count": len(manifest.splits),
        "game_count": len(game_owners),
        "clip_count": len(clip_owners),
        "duplicate_games": duplicate_games,
        "duplicate_clips": duplicate_clips,
    }


def load_manifest(path: str | Path) -> DatasetManifest:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ManifestError("Manifest root must be an object")
    splits_raw = raw.get("splits")
    if not isinstance(splits_raw, list) or not splits_raw:
        raise ManifestError("splits must be a non-empty list")
    splits: list[DatasetSplit] = []
    names: set[str] = set()
    for split_raw in splits_raw:
        if not isinstance(split_raw, dict):
            raise ManifestError("Each split must be an object")
        name = _string(split_raw, "name")
        if name in names:
            raise ManifestError(f"Duplicate split name: {name}")
        names.add(name)
        games = split_raw.get("games", [])
        if not isinstance(games, list) or not all(
            isinstance(game, str) and game for game in games
        ):
            raise ManifestError(f"{name}.games must be a list of strings")
        clips = _string_tuple(split_raw, "clips")
        splits.append(
            DatasetSplit(
                name=name,
                partition=_string(split_raw, "partition"),
                games=tuple(games),
                clips=clips,
                notes=split_raw.get("notes", ""),
            )
        )
        if not isinstance(split_raw.get("notes", ""), str):
            raise ManifestError(f"{name}.notes must be a string")
    notes = raw.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise ManifestError("notes must be a list of strings")
    manifest = DatasetManifest(
        schema_version=_string(raw, "schema_version"),
        dataset_id=_string(raw, "dataset_id"),
        dataset_version=_string(raw, "dataset_version"),
        source_url=_string(raw, "source_url"),
        license_or_access=_string(raw, "license_or_access"),
        status=_string(raw, "status"),
        local_root=_string(raw, "local_root"),
        source_checked_on=_string(raw, "source_checked_on"),
        release_or_download_date=_optional_string(raw, "release_or_download_date"),
        clip_structure=_validate_clip_structure(raw.get("clip_structure")),
        annotations=_validate_annotations(raw.get("annotations")),
        split_strategy=_string(raw, "split_strategy"),
        splits=tuple(splits),
        notes=tuple(notes),
        validation_status=_validate_validation_status(raw),
        preprocessing_version=_optional_string(raw, "preprocessing_version"),
    )
    audit_split_integrity(manifest)
    return manifest
