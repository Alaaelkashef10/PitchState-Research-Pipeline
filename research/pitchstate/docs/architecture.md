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