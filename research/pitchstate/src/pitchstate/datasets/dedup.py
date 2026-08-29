"""Content-hash-based duplicate source-file detection.

Broadcast clips are sometimes re-exported, re-encoded with a different
container, or copied under a different filename during dataset assembly.
Filename- or size-based duplicate checks miss re-encoded copies and can
false-positive on same-sized-but-different clips. This module hashes file
*content* so duplicate detection is independent of filename and container.

This is a local file-integrity check, not a leakage check: leakage across
splits is a separate, higher-severity concern handled by
``datasets.manifest.audit_split_integrity``. A duplicate reported here may or
may not also be a leak; callers that care about leakage should cross-reference
duplicate groups against split membership themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

DEFAULT_CHUNK_SIZE = 1024 * 1024


def hash_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the sha256 hex digest of a file's content, streamed in chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DuplicateGroup:
    content_hash: str
    paths: tuple[Path, ...]


def detect_duplicate_source_files(
    paths: Iterable[str | Path],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[DuplicateGroup, ...]:
    """Group input files by content hash and return only groups with >1 file.

    Missing paths are skipped rather than raising, since dataset manifests may
    legitimately reference files that have not been downloaded yet (see
    ``dataset-audit.md``); a duplicate audit should not crash on that expected
    state. Non-existent inputs are simply excluded from grouping.
    """

    by_hash: dict[str, list[Path]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        content_hash = hash_file(path, chunk_size=chunk_size)
        by_hash.setdefault(content_hash, []).append(path)
    return tuple(
        DuplicateGroup(content_hash, tuple(group_paths))
        for content_hash, group_paths in sorted(by_hash.items())
        if len(group_paths) > 1
    )
