# Architecture and decision register

## Stable stage contracts

The orchestration layer depends on small protocols rather than concrete model
libraries:

1. **Detector**: frame → detections with class, box, confidence, and provenance.
2. **Tracker**: frame + detections + shot context → track observations.
3. **Team/role classifier**: track observation → team/role predictions with
   independent confidence and an allowed `unknown` value.
4. **Calibrator**: frame + observations → shot-local calibration state,
   confidence, reprojection error, and validity.
5. **Projector**: image point + frame metadata + calibration → pitch point.
6. **Tactical analyzer**: valid player states → descriptive shape metrics.
7. **Evaluator**: predictions + references → metric records with dataset and
   schema provenance.

Each stage may be replaced without changing the state schema or CLI contract.

## State design

State is frame/shot scoped. Every player record carries:

- source frame and timestamp;
- shot ID;
- track ID;
- role/team and independent confidences;
- bounding box and image footpoint;
- pitch point when calibration is valid;
- detection/tracking/calibration provenance;
- validity and abstention reasons.

This prevents a downstream metric from looking valid when one of its upstream
inputs was only an unverified guess.

## Decision register

### Python standard library for Phase 0

**Decision:** use Python with no runtime dependencies in Phase 0.

**Alternatives:** start with PyTorch/Ultralytics/OpenCV, or make the foundation
TypeScript-only to match the surrounding workspace.

**Reason:** the target research ecosystem is Python, but model dependencies
would make the repository hard to run and would obscure whether the architecture
itself is sound. The model layer can be added later behind the same contracts.

### Planar normalized pitch coordinates

**Decision:** represent the first pitch space as normalized 2D coordinates.

**Alternatives:** full 3D camera calibration or learned field registration.

**Reason:** a planar model is sufficient for first-version team shape and keeps
the first claim testable. Full 3D calibration is postponed until planar error
is characterized.

### Shot-local identity

**Decision:** a cut or replay invalidates active shot state.

**Alternatives:** attempt cross-cut identity continuity.

**Reason:** ordinary broadcast footage does not provide reliable evidence for
cross-cut continuity. Resetting is scientifically safer and makes failure
visible.

### Descriptive geometry before tactics

**Decision:** start with centroid, width, depth, spacing, and a compactness
proxy.

**Alternatives:** formations, pressing structure, passing graphs, and territorial
control.

**Reason:** these later outputs depend on intent, phase, possession, or a
domain-specific influence model that cannot be validated from pixels alone in
the MVP.

### Explicit abstention

**Decision:** every metric can be invalid with a structured reason.

**Alternatives:** impute missing observations and always draw a result.

**Reason:** the research goal is trustworthy measurement. Silent imputation would
make visually persuasive but scientifically unsupported outputs.

### Manifest ``validation_status`` as a field distinct from ``status`` (2026-08-29)

**Decision:** bump the manifest schema to v0.3 and add a required
``validation_status`` field with a closed vocabulary (`not_locally_verified`,
`locally_verified`, `source_verified_access_pending`, `source_verified`,
`invalid`), enforced at load time in `datasets.manifest.load_manifest`.

**Alternatives:** overload the existing `status` field to also carry local
verification state, or leave local verification undeclared and rely on the
prose checklist in `dataset-audit.md` alone.

**Reason:** `status` already means source/access state (e.g.
`source_verified_access_pending`). Reusing it for "has this manifest's local
content actually been audited against a downloaded release" would let a
manifest read as ready the moment access is granted, before the nine-point
local audit in `dataset-audit.md` has actually run. Making local verification
its own closed-vocabulary field, enforced by the loader rather than only by
convention, prevents that gap from being silent. An optional
`preprocessing_version` field was added alongside it so a manifest can later
record which local preprocessing pass (if any) produced the files it
describes; it is not yet consumed by any pipeline stage.

### Runtime split-access guard as a separate layer from manifest leakage audit (2026-08-29)

**Decision:** add `datasets.split_policy.SplitAccessGuard`, a runtime object
built from a validated manifest that raises `TestSplitAccessError` when code
asks to use a frozen-split (default: `test`) game or clip identity for
`threshold_selection` or `hyperparameter_search`.

**Alternatives:** rely on `audit_split_integrity` alone, or document the
"never touch test for thresholds" rule as an engineering convention without
runtime enforcement.

**Reason:** `audit_split_integrity` verifies that manifest split membership
itself is leak-free (no game/clip in two splits). It cannot catch a
downstream mistake where evaluation or tuning code legitimately loads a
frozen-split identity and uses it to pick a threshold — the manifest stays
leak-free while the *experiment* leaks. Since the research question depends on
a frozen test set (see `research-plan.md`), this class of mistake is high-cost
and easy to make by accident during iterative experimentation. A guard object
that call sites must consult turns an easy silent mistake into an immediate,
loud `TestSplitAccessError`. It does not attempt to statically detect leakage
in arbitrary code; it only enforces the check where callers actually ask.

### Content-hash duplicate-file detection as a local integrity check, not a leakage check (2026-08-29)

**Decision:** add `datasets.dedup.detect_duplicate_source_files`, which groups
local files by sha256 content hash (not filename or size) and reports groups
of more than one file.

**Alternatives:** rely on filename/size heuristics, or defer duplicate
detection until an authorized SoccerNet release is downloaded.

**Reason:** dataset assembly can produce re-encoded or renamed copies of the
same source clip; filename/size checks miss re-encodes and can false-positive
on differently sized but unrelated clips. Content hashing is decoupled from
naming and container. This is explicitly scoped as a local file-integrity
check: it does not itself determine whether a duplicate crosses a split
boundary. A caller that cares about leakage must cross-reference a duplicate
group's paths against split membership (e.g. via `SplitAccessGuard`) — the two
checks are kept separate because they answer different questions and a
duplicate is not automatically a leak.