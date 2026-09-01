# PitchState research plan

## Research question

Can confidence-aware temporal pitch calibration and tracklet association recover
team-level pitch-space shape metrics from ordinary single-camera broadcast
footage more accurately, and with better-calibrated abstention, than a
conventional framewise detector-plus-tracker pipeline?

## Hypothesis

On main-camera broadcast segments with sufficient visible pitch evidence,
temporal field-registration smoothing, appearance-assisted association, and
explicit quality gating will reduce pitch-space shape error and temporal
instability relative to a conventional detector/tracker/framewise-homography
baseline. On replays, cuts, severe zooms, and heavy occlusion, the system will
not reliably recover tactical state and should abstain or reset.

This is a conditional hypothesis. It is not a claim that all football tactics
are identifiable from broadcast pixels.

## Why this is not a standard detection/tracking pipeline

A conventional pipeline ends at boxes, tracks, and a visualization. PitchState
adds a measured downstream claim:

1. Track and calibration provenance are preserved for every state.
2. Image observations are projected into a normalized pitch coordinate system.
3. Descriptive team-shape metrics are computed from pitch coordinates.
4. Confidence is propagated into the metric layer.
5. Camera cuts invalidate shot-local continuity.
6. The system can abstain with a reason instead of reporting a plausible-looking
   but unsupported tactical value.
7. Evaluation includes selective risk/coverage and temporal stability, not only
   detector or tracker scores.

The first contribution is a benchmarked system and evaluation protocol. It
should not be described as a novel model until a literature review and the
ablations establish that.

## MVP

The MVP accepts a continuous main-camera clip and emits, for accepted
timestamps:

- player/goalkeeper observations, track IDs, team labels, and confidence;
- image footpoints and normalized pitch coordinates;
- calibration quality and shot ID;
- team centroid, width, depth, pairwise spacing, and a compactness proxy;
- a validity flag and abstention reasons;
- machine-readable state plus overlays;
- detection, tracking, calibration, shape, and reliability metrics when
  reference annotations are available.

The MVP does not require ball possession, jersey numbers, cross-cut identity,
formations, pressing labels, or passing networks.

## What can be inferred

| Feature | Status in first version |
| --- | --- |
| Visible player image locations | Reliably measurable when detected |
| Short-shot tracks | Approximately measurable |
| Team assignment | Approximately measurable |
| Goalkeeper/referee role | Approximately measurable |
| Pitch registration | Approximately measurable on sufficiently visible shots |
| Team centroid/width/depth | Approximately measurable |
| Spacing/compactness proxy | Approximately measurable |
| Defensive lines and formation | Research-level/difficult |
| Free space, territorial control, passing lanes | Research-level/difficult |
| Passing networks and intent | Not realistically reliable for v1 |

No feature should be promoted without a reference annotation, metric, and
predefined abstention policy.

## Dataset strategy

### Primary: SoccerNet Game State Reconstruction

Use SoccerNet-GSR for end-to-end state reconstruction and derived shape
evaluation. Its task definition maps broadcast video to field locations, role,
team, and jersey identity, and its documented main metric is GS-HOTA:
<https://www.soccer-net.org/tasks/game-state-reconstruction>.

The task page currently documents 57 train, 59 validation, and 50 test
30-second main-camera clips. Those counts are treated as source facts to verify
against the downloaded release, not as hard-coded assumptions in the software.

Limitations are short clips, main-camera bias, and the fact that benchmark
labels do not automatically validate every tactical feature.

### Tracking: SoccerNet Tracking

Use the tracking benchmark to isolate association from calibration. It
documents association-only and full-tracking settings, HOTA with DetA/AssA,
and player/goalkeeper/referee/staff/ball object categories:
<https://www.soccer-net.org/tasks/tracking>.

### Calibration: SoccerNet Camera Calibration

Use the camera-calibration task for focused keypoint/field-registration
experiments before combining errors downstream:
<https://www.soccer-net.org/tasks/camera-calibration>.

### Supporting data

SoccerNet’s wider broadcast collection and camera-boundary/action tasks can
support cut/replay stress tests, but action labels are not tactical ground
truth:
<https://www.soccer-net.org/tasks/action-spotting>.

Tabular tracking datasets can validate shape formulas, not the video pipeline.
A small custom transfer set should later be stratified by stadium, lighting,
broadcast style, zoom, occlusion, and shot type.

### Annotation strategy

- Split by game/match, never by nearby frames.
- Use an `unknown` state for ambiguous team, role, or identity labels.
- Record shot/replay status and visibility quality.
- Record field landmarks, homography validity, and annotation confidence.
- Double-annotate a pilot subset to quantify disagreement.
- Keep reference coordinates and derived metrics separate.
- Freeze the test set before model tuning.

## Baselines

### B0: image-space

Pretrained detector → ByteTrack-style tracker → simple team labels → image-space
centroid/width/depth/spacing. No calibration and no meaningful abstention.

### B1: conventional pitch-space

Same detector/tracker → framewise field landmarks → robust homography →
footpoint projection → hard validity threshold → pitch-space shape metrics.

### P1: proposed

Same controlled detector → motion plus appearance association → shot boundary
reset → temporally filtered field registration → geometry/track/team confidence
propagation → selective shape outputs with abstention.

Detector weights, frame sampling, and evaluation splits must remain fixed when
comparing B1 and P1.

## Architecture

```text
Video
  → decode and metadata
  → shot/replay boundary detection
  → object detection
  → track management
  → role/team classification
  → field landmark/keypoint estimation
  → homography and temporal filtering
  → image footpoint extraction
  → pitch projection
  → versioned player/match state
  → quality/abstention gate
  → descriptive team-shape metrics
  → evaluation and visualization
```

The state model preserves frame index, timestamp, shot ID, track ID, role,
team, pixel location, pitch location, velocity when justified, confidence,
provenance, and tactical relationships only when those relationships are
explicitly supported.

## Metrics and acceptance

Detection: precision, recall, mAP by class and visibility subset.

Tracking: HOTA, DetA, AssA, IDF1, ID switches, fragmentation, track duration,
and recovery after occlusion.

Team/role: accuracy, macro-F1, frame-weighted team purity, track consistency,
unknown precision, and confusion matrix.

Calibration: keypoint error, homography reprojection error, pitch footpoint
error, and valid-calibration coverage.

Shape: centroid, width, depth, spacing, compactness error, temporal jitter,
and agreement on shape changes over time.

Reliability: risk/coverage curves, selective risk, calibration error, error at
fixed coverage, and abstention rates by failure category.

Engineering acceptance requires a clean deterministic smoke run, versioned
schemas, passing tests, valid manifest checks, and abstention on deliberately
bad synthetic geometry.

Research acceptance requires game-level split integrity, frozen baselines,
validation-only threshold selection, uncertainty intervals, and improvement on
the preregistered primary shape metric without merely reporting fewer frames.
No performance number is assumed before the baseline experiment.

## Ablations

1. Image-space versus pitch-space.
2. Framewise versus temporally filtered calibration.
3. Motion/IoU association versus appearance-assisted association.
4. Hard output versus confidence-aware abstention at matched coverage.
5. Ground-truth versus predicted detections.
6. Ground-truth versus predicted team labels.
7. State reset versus attempted continuity across cuts.
8. Sensitivity to missing and false players.
9. Frame-rate/sample-rate sensitivity.
10. Cross-game and cross-broadcast transfer.

## Failure policy

Cuts/replays reset state. Missing pitch lines cause calibration abstention.
Occluded tracks remain tentative only briefly. Similar kits may produce
`unknown`. Sideline footpoints receive higher uncertainty. Missing players are
reported as observed-player shape, never silently treated as full-team shape.
Ball loss is missing data, not inferred possession.

## Explicit non-goals for v1

- jersey-number recognition as a dependency;
- cross-camera identity continuity;
- player intent or decision quality;
- passing networks from ordinary broadcast;
- formation labels as ground truth;
- automatic pressing/phase labels;
- full territorial-control claims;
- 3D pose as a prerequisite;
- tactical language generation;
- hard possession inference.

## Hardware and compute

Planning assumptions, not benchmarks:

- 8–12 GB NVIDIA VRAM for inference/small experiments;
- 16–24 GB VRAM for comfortable fine-tuning and ablations;
- 8 logical CPU cores and 32 GB RAM;
- at least 250 GB for selected clips, caches, and checkpoints, with more for
  full-game archives;
- cached detections and calibration candidates so downstream experiments do not
  repeatedly rerun expensive upstream stages.

## Roadmap

| Phase | Objective | Gate |
| --- | --- | --- |
| 0 | Repository, config, manifests, logs, interfaces, tests, smoke | Clean deterministic run |
| 1 | Dataset acquisition and normalization | No leakage; labels mapped |
| 2 | Detection baseline | Reproducible held-out predictions |
| 3 | Tracking baseline | Inspectable MOT failures |
| 4 | Team/role classification | Track consistency and unknown state |
| 5 | Pitch calibration | Invalid geometry rejected |
| 6 | Pitch trajectories | Confidence/provenance on coordinates |
| 7 | Tactical state representation | Replayable versioned schema |
| 8 | Shape analytics | Synthetic and annotated formula validation |
| 9 | Full main-camera games | Resumable, cut-safe pipeline |
| 10 | Reliability/robustness | Selective-risk report |
| 11 | Optimization | Documented speed/accuracy trade-off |
| 12 | Final demo/research package | Every output traceable to data/code |

## Current status against the roadmap (2026-08-29)

- **Phase 0** (repository, config, manifests, logs, interfaces, tests, smoke):
  satisfied — deterministic smoke run passes.
- **Phase 1** (dataset acquisition and normalization): partially satisfied.
  Leakage-safety *infrastructure* is implemented and tested (manifest schema
  v0.3 with `validation_status`, `datasets.split_policy.SplitAccessGuard`,
  `datasets.dedup` content-hash duplicate detection — see architecture.md
  decision register, 2026-08-29). Actual dataset acquisition remains blocked:
  no authorized SoccerNet release has been downloaded, so no real games or
  clips populate the manifest's splits yet.
- **Phases 2–4** (detection, tracking, team/role classification baselines):
  not started; blocked on the same data-access gate as Phase 1's acquisition
  step, since a "baseline" is only meaningful against real footage.
- **Phase 5** (pitch calibration, gate: invalid geometry rejected): the
  *mathematical foundation* — planar homography estimation via DLT, in
  `calibration.homography` — is implemented and unit-tested against synthetic
  correspondences (numerical correctness, degenerate-input rejection,
  coordinate conventions, abstention on near-infinite projections; see
  `tests/test_homography.py` and the architecture.md decision register,
  2026-08-29). This is **not** the same as satisfying Phase 5's gate: that
  requires the estimator to operate on keypoints actually detected in
  broadcast footage, which needs a keypoint detector and real video, neither
  of which exists yet. `calibration.homography` is intentionally not wired
  into the `Calibrator` protocol or the pipeline until that exists — see
  `calibration/interface.py` and `calibration/homography.py` module
  docstrings.
- **Phases 6–12:** not started; each depends on an earlier blocked phase.

### Addendum (2026-08-29, later same day): calibration evaluator

`evaluation.calibration` now implements the "Calibration" and "Reliability"
metrics from this document's "Metrics and acceptance" section (reprojection
error summary, valid-calibration coverage, selective risk/coverage curve,
abstention rate by failure reason) against synthetic
`HomographyEstimate`/held-out-error fixtures — see
`tests/test_calibration_evaluation.py` and the architecture.md decision
register. The ECE-style "calibration error" metric named in that section is
explicitly deferred; see the same decision register entry for why. This is
still evaluation-methodology infrastructure, not a real-footage calibration
accuracy result — no such result exists yet.

### Addendum (2026-08-29, later same day): shape evaluator

`evaluation.shape` now implements the remaining "Shape" metrics from this
document's "Metrics and acceptance" section (centroid/width/depth/spacing/
compactness error, temporal jitter, and agreement on shape changes over time)
against synthetic `ShapeMetrics` fixtures — see `tests/test_shape_evaluation.py`
and the architecture.md decision register. As with the calibration evaluator,
this is evaluation-methodology infrastructure only; no real-footage shape
accuracy result exists yet, since `tactics.shape.calculate_shape_metrics` has
not been run against anything but synthetic observations.

### Addendum (2026-08-29, later same day): detection evaluator

`evaluation.detection` now implements the "Detection" metrics from this
document's "Metrics and acceptance" section (precision, recall, mAP by class
and visibility subset) against synthetic `Detection` fixtures — see
`tests/test_detection_evaluation.py` and the architecture.md decision
register. As with the calibration and shape evaluators, this is
evaluation-methodology infrastructure only: no real-footage detection
precision/recall/mAP result exists yet, since there is no detector
implementation or real annotated footage in this repository. AP values in the
test suite are hand-computable, deterministic properties of synthetic boxes,
not benchmark results.