"""Deterministic run identifiers and lightweight environment metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from typing import Any


def seed_everything(seed: int) -> None:
    """Seed standard-library randomness; model-specific seeds come later."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_run_id(project_name: str, experiment_name: str, config: dict[str, Any], seed: int) -> str:
    return stable_hash(
        {
            "project_name": project_name,
            "experiment_name": experiment_name,
            "config": config,
            "seed": seed,
        }
    )[:16]


def environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def code_revision() -> str:
    """Return the repository revision when available, never inventing one."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    revision = completed.stdout.strip()
    return revision or "unknown"