# PitchState

PitchState is a research-oriented foundation for the question:

> Can confidence-aware temporal pitch calibration and tracklet association recover team-level pitch-space shape metrics from ordinary single-camera broadcast footage more accurately, and with better-calibrated abstention, than a conventional framewise detector-plus-tracker pipeline?

The first version is intentionally not a detector demo. It establishes the repository contracts, data/version boundaries, experiment metadata, and quality-gated state representation that later computer-vision models will use.

## Phase 0 status

Phase 0 contains:

- A standard-library-only Python package.
- TOML configuration loading with explicit schema validation.
- Dataset manifests that record dataset/version/access metadata without bundling data.
- JSONL experiment logging and deterministic run identifiers.
- Protocol interfaces for detection, tracking, team/role classification, calibration, projection, and tactical analysis.
- A synthetic end-to-end smoke pipeline that emits accepted team-shape measurements and abstention reasons.
- Unit tests for configuration, manifests, geometry, pipeline behavior, and reproducibility.

The smoke pipeline is a wiring test only. It is not a benchmark and does not demonstrate research performance.

## Quick start

From `research/pitchstate`:

```bash
make smoke
make test
make validate-manifest
```

Equivalent direct commands:

```bash
PYTHONPATH=src python3 -m pitchstate.cli --help
PYTHONPATH=src python3 -m pitchstate.cli smoke \
  --config configs/smoke.toml \
  --output outputs/smoke.json
```

The generated smoke JSON is safe to delete and is ignored by git.

## Repository map

```text
research/pitchstate/
├── configs/                  # Versioned experiment configuration
├── data/
│   ├── manifests/            # Dataset metadata and split declarations
│   ├── raw/                  # Local-only source data
│   ├── processed/            # Local-only normalized data
│   └── cache/                # Local-only reusable intermediate results
├── docs/                     # Research plan, architecture, and Phase 0 gate
├── experiments/runs/         # Local JSONL experiment logs
├── src/pitchstate/
│   ├── calibration/          # Field registration contracts
│   ├── classification/       # Team and role classification contracts
│   ├── datasets/              # Manifest and dataset adapter contracts
│   ├── detection/             # Object detection contracts
│   ├── evaluation/            # Evaluation extension point
│   ├── tracking/              # MOT contracts
│   ├── tactics/               # Descriptive shape metrics
│   ├── cli.py                 # User-facing command line
│   ├── config.py              # TOML configuration
│   ├── logging_utils.py       # JSONL experiment logging
│   ├── pipeline.py            # Dependency-injected orchestration
│   ├── reproducibility.py     # Seeds and stable run IDs
│   └── schema.py              # Versioned state data structures
└── tests/                    # Standard-library tests
```

## Data policy

Do not commit downloaded video, annotations, model weights, or generated caches. Add a manifest entry with the dataset name, version, source URL, license/access status, split names, and local path convention instead.

## Research scope

The complete research plan and explicit non-goals are in [`docs/research-plan.md`](docs/research-plan.md). The architecture and rationale are in [`docs/architecture.md`](docs/architecture.md).

The complete decision-oriented research brief, including dataset facts,
candidate comparisons, stage contracts, acceptance gates, roadmap, hardware
assumptions, risks, and source links is in
[`docs/research-brief.md`](docs/research-brief.md).