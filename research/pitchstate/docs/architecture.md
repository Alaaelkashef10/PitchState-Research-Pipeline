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

### Pure-Python planar homography (DLT) as the Phase 2 calibration foundation, not yet wired to the pipeline (2026-08-29)

**Decision:** add `calibration.homography`, implementing planar homography
estimation from point correspondences via a Direct-Linear-Transform-style
linear system, solved with a hand-written Gaussian-elimination solver
(`solve_linear_system`) — no numpy/scipy dependency. Exactly 4 correspondences
solve an 8x8 system directly; more than 4 fit via least squares over the
normal equations. The linearization fixes `h33 = 1`. The module is validated
only against synthetic, hand-constructed correspondences with a known
ground-truth transform, and is explicitly **not** wired into the `Calibrator`
protocol or `schema.CalibrationState` yet.

**Alternatives:** (a) use numpy/scipy for an SVD-based DLT solver; (b) wire a
homography-based `Calibrator` implementation into the pipeline immediately,
reusing the existing `CalibrationState` affine fields; (c) defer calibration
math entirely until real pitch-keypoint data exists.

**Reason for (a):** the Phase 0 decision register already commits this
repository to a dependency-free Python standard-library foundation so the
architecture's soundness is not obscured by model/library concerns; a
numpy/scipy solver would be more numerically robust (true SVD has no `h33 !=
0` assumption) but was judged not worth breaking that constraint for now. The
`h33 = 1` normalization and least-squares-via-normal-equations trade-off are
documented as known limitations in `calibration/homography.py`'s module
docstring rather than silently accepted.

**Reason for (b):** `CalibrationState` currently models a simple per-axis
affine projection (`scale_x`, `scale_y`, `offset_x`, `offset_y`), used by the
Phase 0 synthetic smoke pipeline (`smoke.py`, `test_smoke.py`). A real planar
homography is a full 3x3 projective transform, not representable in that
affine schema without either overloading `CalibrationState` (breaking the
existing, working smoke pipeline and its tests) or introducing a second,
inconsistent calibration representation into the schema. Wiring a concrete
`Calibrator` implementation into the pipeline also requires a real source of
image-to-pitch keypoint correspondences — i.e. a keypoint detector run
against actual broadcast frames — which does not exist in this repository.
Building that wiring now would mean shipping pipeline integration that cannot
be exercised against anything but synthetic stand-ins for a detector that
doesn't exist, which the project's "no fake progress" principle rules out.
The estimation math is therefore built and thoroughly tested as a standalone,
directly testable module first; pipeline integration is deferred to when a
concrete correspondence source exists.

**Reason for (c) not chosen:** the mathematics of homography estimation
(numerical stability, degenerate-input handling, coordinate conventions) does
not depend on real video and is fully testable now with synthetic
correspondences; deferring it until data access is granted would waste time
that can be spent validating this component in isolation today.

### Calibration evaluator scored on held-out error, not fit error, with ECE-style calibration error explicitly deferred (2026-08-29)

**Decision:** add `evaluation.calibration`, implementing
`valid_calibration_coverage`, `abstention_rate_by_reason`,
`reprojection_error_summary`, and a selective risk/coverage curve
(`selective_risk_coverage_curve` / `selective_risk_at_coverage`) over
`CalibrationEvaluationSample` records, each pairing a `HomographyEstimate`
with an optional held-out reprojection error. The risk/coverage curve ranks
samples by ascending *in-sample fit* error (the only signal available at
prediction time) and scores accepted samples by their *held-out* error where
available. `research-plan.md`'s reliability metric "calibration error" (ECE
sense) is explicitly not implemented here.

**Alternatives:** (a) summarize `HomographyEstimate.reprojection_error_mean`
directly, without a held-out/fit-error distinction; (b) rank the risk/coverage
curve by the same error being evaluated; (c) implement an ECE-style
calibration-error metric now using `valid` as a stand-in correctness signal.

**Reason for (a) not chosen:** for the minimal 4-correspondence case, fit
error is ~0 by construction (see `calibration/homography.py`) — treating it
as an accuracy number would silently report a perfect-looking metric that
measures curve-fitting, not calibration quality. `reprojection_error_summary`
therefore prefers a caller-supplied held-out error and counts how often it had
to fall back to the optimistic in-sample number
(`fit_error_fallback_count`), so the summary cannot misrepresent its own
reliability.

**Reason for (b) not chosen:** ranking a risk/coverage curve by the same
quantity used to compute risk lets the curve "cheat" — it would always look
monotonically well-behaved by construction, regardless of whether the
system's own confidence signal is any good. Ranking by fit error (known
before any ground truth) and scoring by held-out error (known only in
evaluation) is what makes the curve an honest test of the confidence signal
itself; `test_ranks_by_fit_confidence_not_true_error` in
`tests/test_calibration_evaluation.py` demonstrates this directly with a
deliberately misleading low-fit/high-true-error sample.

**Reason for (c) not chosen:** `HomographyEstimate` exposes a binary `valid`
flag and a continuous error, not a probability; computing ECE would require
either inventing a confidence score from nothing or misusing `valid` as a
correctness proxy in a way that doesn't map cleanly onto standard ECE
binning. Rather than build a metric on top of a signal this repository
doesn't actually produce, it is left as documented future work in
`evaluation/calibration.py`'s module docstring.