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

### Shape evaluator propagates abstention instead of scoring missing fields as zero error (2026-08-29)

**Decision:** add `evaluation.shape`, implementing `shape_error` (per-field
absolute error between one predicted/reference `ShapeMetrics` pair),
`shape_error_report` (aggregated mean/max over a frame sequence),
`temporal_jitter` (mean absolute frame-to-frame change in one field), and
`shape_change_agreement` (fraction of transitions where predicted and
reference agree on the direction of change). Every function returns `None`
for a field/frame wherever either side lacks that field, rather than
treating a missing value as zero error.

**Alternatives:** (a) treat a `None` field as 0.0 and include it in the mean;
(b) skip `None` fields silently without reporting how many comparisons were
actually possible; (c) only implement raw error metrics, without a
direction-of-change agreement metric.

**Reason for (a) not chosen:** `calculate_shape_metrics` (see
`tactics/shape.py`) returns an all-`None` `ShapeMetrics` when there are no
observed pitch points for a team in a frame — that is the system correctly
reporting "nothing was measurable," per this project's explicit-abstention
principle. Scoring that as zero error against a non-`None` reference would
report "perfect agreement" for a case that was actually a total miss; that is
the opposite of what an evaluator should do.

**Reason for (b) not chosen:** `ShapeErrorFieldSummary.compared_frames`
records how many frames actually contributed to each field's mean/max,
so a summary showing "mean width error 0.3" over only 2 of 50 frames reads
very differently from the same mean over 48 of 50 — the count is reported
alongside the number specifically so a caller cannot mistake sparse coverage
for a strong result.

**Reason for (c) not chosen:** research-plan.md's Shape metrics list
"agreement on shape changes over time" as distinct from an error magnitude
metric — a system can have real per-frame numeric error while still
correctly tracking whether a team is widening or narrowing, which is often
the more decision-relevant signal for a downstream analyst. Implementing only
raw error would silently drop that half of the specified metric.

### Detection evaluator uses VOC-style AP and an ignore convention for visibility subsets (2026-08-29)

**Decision:** add `evaluation.detection`, implementing `bounding_box_iou`,
confidence-ordered greedy per-category matching
(`match_detections_single_category`), `precision_recall_at_threshold`,
VOC 2010+ continuous-interpolation `average_precision` /
`mean_average_precision`, and `average_precision_by_visibility`, which scores
one ground-truth visibility bin (e.g. an occlusion level) while treating a
prediction that matches a real reference *outside* that bin as ignored,
neither a true positive nor a false positive.

**Alternatives:** (a) 11-point interpolated AP (the older PASCAL VOC
convention) instead of continuous all-points interpolation; (b) score
visibility subsets by simply filtering the reference set and counting any
non-matching prediction as a false positive; (c) implement only an
overall/single-threshold precision-recall metric, skipping AP/mAP entirely.

**Reason for (a) not chosen:** continuous all-points interpolation is exact
given the actual observed recall values (no grid-approximation error), is
no harder to implement correctly, and is the interpolation method used by
modern benchmarks (e.g. COCO's building blocks); 11-point interpolation was
only a historical concession to compute costs from 2007-era hardware that
does not apply here.

**Reason for (b) not chosen:** research-plan.md's dataset-strategy section
already commits to treating shot/replay/visibility annotation quality as
first-class ("record shot/replay status and visibility quality"). If a
detector correctly finds a real, heavily-occluded player while a caller is
scoring the "fully visible" subset, counting that as a false positive would
punish the detector for a correct detection that simply isn't a member of
the subset being measured — it would make the subset AP numbers reflect subset
membership noise rather than detector quality. The ignore convention (used by
COCO-style benchmarks for exactly this reason) is implemented instead:
out-of-subset matches are excluded from the precision/recall curve entirely.
`test_match_on_out_of_subset_reference_is_ignored_not_penalized` in
`tests/test_detection_evaluation.py` demonstrates the distinction directly.

**Reason for (c) not chosen:** research-plan.md's Detection metrics line
explicitly names "mAP by class and visibility subset" alongside
precision/recall — omitting AP would leave the module short of what was
specified, and `precision_recall_at_threshold` alone cannot answer
"how good is this detector across all operating points," which is the
question AP is for.

### Tracking evaluator implements the full HOTA/DetA/AssA/IDF1 definitions with two documented, honest approximations (2026-08-29)

**Decision:** add `evaluation.tracking`, implementing the TrackEval-style
HOTA/DetA/AssA definitions exactly (`detection_accuracy`,
`association_accuracy`, `hota_at_threshold`, `hota_summary` over the standard
0.05-0.95 threshold sweep), the standard IDF1 formula via an exact global
identity assignment (`identity_metrics`), and per-track ID switches,
fragmentation, coverage, and gap-recovery (`track_quality_reports`). Two
approximations are made, both documented in the module docstring rather than
silently accepted: (1) per-frame matching uses the same confidence-ordered
greedy IoU assignment as `evaluation.detection`, not the Hungarian-algorithm
optimum the official TrackEval implementation uses; (2) `identity_metrics`'s
global identity assignment is found by exhaustive permutation search
(exact, but factorial in identity count), capped at `DEFAULT_MAX_IDENTITIES
= 6` by default, rather than the Hungarian algorithm.

**Alternatives:** (a) implement a pure-Python Hungarian algorithm to make
both matching steps exactly match the official protocol; (b) implement only
ID switches/fragmentation/coverage and skip HOTA/DetA/AssA/IDF1 entirely,
citing the missing Hungarian solver as a blocker; (c) approximate HOTA/IDF1
with informal heuristics not tied to the published formulas.

**Reason for (a) not chosen:** a correct, general Hungarian algorithm is a
substantial, error-prone piece of numerical code to hand-write and verify in
pure Python; the greedy-matching approximation it would replace is already
proven exact for every unambiguous configuration (the only case where greedy
and optimal differ is when multiple candidates compete for the same box
within one frame), and every test fixture in
`tests/test_tracking_evaluation.py` is deliberately unambiguous so the
computed values are exact under either method. Given no real detector or
tracker exists in this repository yet to feed this evaluator, spending
significant effort on Hungarian-algorithm correctness now would not currently
change any measured result. This is documented as a real limitation to
revisit if evaluation ever needs genuinely ambiguous real-world frames.

**Reason for (b) not chosen:** the user explicitly requested the actual
metrics (HOTA, DetA, AssA, IDF1), not placeholder APIs; research-plan.md
also names them explicitly in its "Tracking" metrics line. Implementing them
with a documented, bounded approximation is more useful and more honest than
omitting them outright, since the approximation's exact boundary conditions
(ambiguous per-frame matches; >6 identities) are stated plainly rather than
hidden.

**Reason for (c) not chosen:** using the actual published TrackEval/IDF1
formulas (rather than an ad hoc heuristic) is what makes the hand-computed
test fixtures in `tests/test_tracking_evaluation.py` — perfect tracking,
missed detections, an induced ID switch, and an induced fragmentation with
successful/failed recovery — independently verifiable: each expected value
was derived from the formula definitions, not fit to whatever the code
happened to produce. An ad hoc heuristic would not have that property and
would risk producing numbers that look plausible but do not mean what they
claim to mean.