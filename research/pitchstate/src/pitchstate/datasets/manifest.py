"""Dataset manifest parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a dataset manifest is malformed."""


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    partition: str
    games: tuple[str, ...]
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
    splits: tuple[DatasetSplit, ...]
    notes: tuple[str, ...]


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} must be a non-empty string")
    return value


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
        if not isinstance(games, list) or not all(isinstance(game, str) for game in games):
            raise ManifestError(f"{name}.games must be a list of strings")
        splits.append(
            DatasetSplit(
                name=name,
                partition=_string(split_raw, "partition"),
                games=tuple(games),
                notes=split_raw.get("notes", ""),
            )
        )
    notes = raw.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise ManifestError("notes must be a list of strings")
    return DatasetManifest(
        schema_version=_string(raw, "schema_version"),
        dataset_id=_string(raw, "dataset_id"),
        dataset_version=_string(raw, "dataset_version"),
        source_url=_string(raw, "source_url"),
        license_or_access=_string(raw, "license_or_access"),
        status=_string(raw, "status"),
        local_root=_string(raw, "local_root"),
        splits=tuple(splits),
        notes=tuple(notes),
    )