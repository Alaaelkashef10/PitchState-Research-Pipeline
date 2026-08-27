"""JSONL experiment logging with no dependency on a logging service."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Append structured events to a run-local JSONL file."""

    def __init__(self, run_directory: str | Path, run_id: str) -> None:
        directory = Path(run_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{run_id}.jsonl"
        self.run_id = run_id

    def event(self, name: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "run_id": self.run_id,
            "event": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")