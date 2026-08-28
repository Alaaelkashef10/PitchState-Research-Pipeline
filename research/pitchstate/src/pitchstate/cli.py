"""Command-line entry point for Phase 0 and future experiment commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pitchstate.config import load_config
from pitchstate.datasets.manifest import load_manifest
from pitchstate.logging_utils import ExperimentLogger
from pitchstate.reproducibility import code_revision, environment_metadata
from pitchstate.smoke import run_smoke


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitchstate",
        description="Research CLI for confidence-aware football pitch-state reconstruction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run the deterministic Phase 0 synthetic pipeline.")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)

    manifest = subparsers.add_parser("manifest", help="Validate dataset metadata manifests.")
    manifest_subparsers = manifest.add_subparsers(dest="manifest_command", required=True)
    validate = manifest_subparsers.add_parser("validate", help="Validate one JSON manifest.")
    validate.add_argument("--path", type=Path, required=True)
    return parser


def _run_smoke(config_path: Path, output_path: Path) -> int:
    config = load_config(config_path)
    result = run_smoke(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger = ExperimentLogger(config.logging.run_directory, result.run_id)
    logger.event(
        "run_started",
        {
            "project_name": config.project_name,
            "experiment_name": config.experiment_name,
            "seed": config.seed,
            "config": config.to_dict(),
            "dataset_id": result.metadata.get("dataset_id"),
            "dataset_version": result.metadata.get("dataset_version"),
            "model_version": result.metadata.get("model_version"),
            "code_revision": code_revision(),
            "environment": environment_metadata(),
        },
    )
    logger.event(
        "run_completed",
        {
            "command": "smoke",
            "output": str(output_path),
            "state_count": len(result.states),
            "valid_state_count": sum(state.valid for state in result.states),
            "code_revision": code_revision(),
        },
    )
    print(f"Smoke run {result.run_id} wrote {output_path}")
    print(f"Valid states: {sum(state.valid for state in result.states)}/{len(result.states)}")
    return 0


def _validate_manifest(path: Path) -> int:
    manifest = load_manifest(path)
    print(
        f"Valid manifest: {manifest.dataset_id} "
        f"(version={manifest.dataset_version}, splits={len(manifest.splits)}, status={manifest.status})"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "smoke":
        return _run_smoke(args.config, args.output)
    if args.command == "manifest" and args.manifest_command == "validate":
        return _validate_manifest(args.path)
    raise RuntimeError("Unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())