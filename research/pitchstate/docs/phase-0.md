# Phase 0 completion gate

## Included

- [x] Repository structure separates source, configs, data, docs, experiments,
      and tests.
- [x] TOML configuration with schema validation.
- [x] Dataset manifest validation with version/access metadata.
- [x] JSONL experiment logging and stable run IDs.
- [x] Protocols for future detection, tracking, classification, calibration,
      projection, tactical analysis, and evaluation.
- [x] Synthetic deterministic end-to-end smoke pipeline.
- [x] Unit tests using Python's standard library.
- [x] Documentation of the research question, scope, and decisions.

## Intentionally absent

- No detector, tracker, re-identification model, calibration model, ball model,
  or downloaded weights.
- No benchmark result or fabricated performance number.
- No database or web UI.
- No tactical intent or possession inference.

## Exit criteria before Phase 1

1. `make test` passes in a clean Python 3.11+ environment.
2. `make smoke` produces a deterministic JSON state artifact.
3. `make validate-manifest` rejects malformed manifests and accepts the fixture.
4. The dataset access/license decision is recorded before downloading data.
5. Phase 1 adds an adapter and annotations without changing the core state
   schema or pipeline orchestration.