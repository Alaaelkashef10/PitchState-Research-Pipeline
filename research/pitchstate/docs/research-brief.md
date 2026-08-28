# PitchState research and engineering brief

This document is the decision-oriented companion to
[`research-plan.md`](research-plan.md). It is deliberately explicit about
what is a source fact, what is an engineering decision, and what remains a
hypothesis. No performance result is asserted before the corresponding
experiment has been run.

## Project identity

**Name:** PitchState

**Description:** PitchState is a confidence-aware system for turning ordinary
single-camera football broadcast video into auditable, normalized pitch-space
player states and descriptive team-shape measurements. It is designed for
researchers and analysts who need to inspect the evidence behind a tactical
measurement, including calibration quality, track provenance, uncertainty, and
the reasons a frame was rejected.

**Research question:** Can temporal field registration and appearance-assisted,
shot-local tracklet association recover team-level pitch-space shape metrics
more accurately, and with better-calibrated abstention, than a conventional
framewise detector/tracker pipeline on usable main-camera broadcast segments?

**Hypothesis:** On segments with enough visible field structure, temporal
calibration and appearance-assisted association will reduce pitch-space shape
error and temporal jitter at matched coverage. On cuts, replays, severe zooms,
and heavy occlusion, explicit shot resets and quality gates will be safer than
pretending that a continuous tactical state is available.

**Engineering objective:** Build a modular, reproducible, resumable pipeline
whose output can be traced from a tactical number back to its frame, track,
detector, calibration estimate, dataset version, configuration, and code
revision.

## Minimum defensible contribution

The first contribution is not a claim that a new detector is state of the art.
It is a benchmarked system and protocol for selective pitch-space shape
measurement from broadcast video:

1. compare image-space and pitch-space shape errors against a controlled
   detector/tracker baseline;
2. quantify the effect of temporally filtered calibration and appearance
   association;
3. report risk/coverage and failure-stratified results, not only averages;
4. preserve abstentions and provenance so omitted evidence is visible.

This becomes a research contribution only after a literature review,
pre-registered metrics, frozen splits, and ablations establish that the
protocol or result is meaningfully different from prior work.

## What is and is not measurable

The following is the first-version reliability contract:

| Feature | Status | Required evidence |
| --- | --- | --- |
| Visible player image location | Reliably measurable when detected | Box/footpoint annotation |
| Short-shot track continuity | Approximately measurable | MOT identity annotations |
| Team assignment | Approximately measurable | Team labels plus unknown class |
| Goalkeeper/referee role | Approximately measurable | Role labels and confusion matrix |
| Pitch registration | Approximately measurable on visible shots | Landmark and reprojection error |
| Team centroid, width, depth | Approximately measurable | Reference pitch coordinates |
| Pairwise spacing and compactness proxy | Approximately measurable | Derived reference geometry |
| Defensive/midfield lines | Research-level / difficult | Time-aligned tactical annotation |
| Formation estimation | Research-level / difficult | Agreement study and reference labels |
| Occupied/free space | Research-level / difficult | Explicit influence/space definition |
| Passing options and lanes | Research-level / difficult | Ball, body orientation, and event labels |
| Pressing structure | Research-level / difficult | Phase and tactical annotations |
| Overloads and numerical superiority | Research-level / difficult | Ball-relative regions and role labels |
| Ball progression | Research-level / difficult | Reliable ball trajectory and event labels |
| Territorial/space control | Research-level / difficult | Chosen influence model and validation |
| Potential passing network | Not reliable for v1 | Requires more than broadcast evidence |
| Player intent, decision quality, possession certainty | Not reliable for v1 | Not identifiable from pixels alone |

“Approximately measurable” does not mean automatically correct. It means a
metric can be evaluated with an error bar and an abstention policy. A frame
with missing players is reported as observed-player shape, never silently
promoted to full-team shape.

## Dataset strategy

The official SoccerNet task pages were checked on 2026-08-28. These are source
facts, not assumptions encoded as software constants.

| Dataset/task | What it provides | Useful phase | Limits and access policy |
| --- | --- | --- | --- |
| SoccerNet Game State Reconstruction | Main-camera 1080p broadcast clips; the official page describes 57 train, 59 validation, and 50 test clips of 30 seconds, with player/goalkeeper/referee/other roles, team side, jersey number, and field/court coordinates. GS-HOTA is the stated main metric. | End-to-end state and team/role evaluation | Short clips and main-camera bias; jersey identity is not a v1 dependency. Confirm current release, terms, and permitted use before download. |
| SoccerNet Tracking | 12 complete main-camera games for tracklets; the task also documents 100 30-second clips, with separate association-only and full-tracking settings. HOTA with DetA/AssA is the main metric. | Association, occlusion, long-sequence stress | Benchmark scope and challenge subsets differ; preserve the exact downloaded release and split in a manifest. |
| SoccerNet Camera Calibration | The official page describes 20,028 images for the data and 2,104 challenge images, including images from the wider SoccerNet/action-spotting collection and replays. Evaluation uses line reprojection error and completeness. | Calibration and field-registration ablations | Still-image task evidence does not validate temporal shape analytics; replay imagery is a deliberate stress case. |
| SoccerNet camera-shot/replay/action tasks | Broadcast boundaries, replay context, and event timing that can support segmentation and stress stratification. | Cut/replay detection and robustness | Action labels are not tactical ground truth and must not be used as formation or possession labels. |
| Small custom transfer set | A double-annotated, stratified set across stadium, lighting, broadcast style, zoom, occlusion, and shot type. | External validity and annotation disagreement | Must be collected and licensed explicitly; it is not a substitute for a public benchmark. |

**Annotation strategy**

- Split by game/match, never by neighboring frames.
- Keep train, validation, and frozen test identities disjoint.
- Use `unknown` for ambiguous team, role, and identity labels.
- Record shot/replay status, field visibility, occlusion, and annotation
  confidence.
- Keep reference coordinates separate from derived shape metrics.
- Double-annotate a pilot subset and report disagreement before scaling.
- Record the dataset release or download date, source URL, access/license
  status, local path convention, and hashes only when the local data policy
  permits them.
- Never put credentials, signed URLs, videos, weights, or private links in a
  manifest.

## Model and algorithm decisions

| Stage | Candidates | Selected starting point | Reason and trade-off |
| --- | --- | --- | --- |
| Player/role detection | YOLO-family detector, RT-DETR, segmentation | A fixed pretrained YOLO-family detector behind `Detector` | Practical pretrained baseline and easy controlled ablations; accuracy and speed must be benchmarked on the chosen hardware. RT-DETR remains a comparison, not an assumption of superiority. |
| Association | ByteTrack, BoT-SORT, StrongSORT/ReID | ByteTrack-style B0, then BoT-SORT-like motion plus appearance P1 | Separates association gains from detector gains; ReID adds compute and can fail on similar kits. |
| Team classification | Jersey-color clustering, crop classifier, spatial/context model | Temporal crop color features plus an explicit `unknown` state | Simple and auditable; vulnerable to lighting, similar kits, and goalkeeper/referee colors. |
| Re-identification | Appearance embeddings, jersey-number recognition | Appearance embeddings after the baseline; jersey number optional | Tracklet association is useful without making OCR a hard dependency. |
| Ball | Dedicated small-object detector/tracker | Separate optional module, not a v1 validity dependency | Broadcast ball visibility is intermittent; lost ball is missing data, not possession. |
| Camera cuts/replays | Pixel-change detector, learned shot classifier, metadata | Shot-boundary protocol with conservative reset | A false continuity claim is worse than a visible reset. |
| Field registration | Learned keypoints, line/goal landmark detector, manual calibration | Landmarks/lines plus robust homography and temporal filtering | Uses interpretable geometry and supports reprojection-based rejection. |
| Pose | 2D pose, 3D pose | Not a prerequisite | Extra occlusion and compute risk without evidence it improves the first claim. |

The detector, frame sample rate, and split remain fixed when comparing B1 and P1.
Any model replacement must be recorded as a new experiment configuration.

## Complete architecture and stage contracts

```text
Broadcast video
  -> decode/metadata
  -> shot and replay boundaries
  -> object detection
  -> shot-local tracking and optional ReID
  -> team and role classification
  -> field landmark/keypoint estimation
  -> robust homography and temporal filtering
  -> footpoint extraction and pitch projection
  -> versioned player/match state
  -> confidence and abstention gate
  -> descriptive team-shape metrics
  -> evaluation and visualization
```

| Stage | Input -> output | Frequency | Confidence/failure rule | Evaluation |
| --- | --- | --- | --- | --- |
| Decode | Video -> frames, timestamps, dimensions | Sampled frame rate | Preserve decode errors; never interpolate timestamps silently | Frame/timestamp integrity |
| Shot boundary | Adjacent frames -> shot ID/boundary | Every sampled frame | Cut/replay starts a new shot; reset active tracks | Boundary precision/recall and reset correctness |
| Detection | Frame -> boxes, class, confidence, provenance | Every sampled frame | Keep raw detections; threshold only at downstream gates | Precision, recall, mAP by visibility/class |
| Tracking | Detections + shot -> track observations | Every sampled frame | Tentative tracks may be short-lived; no cross-shot continuity | HOTA, DetA, AssA, IDF1, switches, fragmentation |
| Team/role | Crop/track -> team, role, independent confidences | New/updated track | Similar kit or ambiguous role becomes `unknown` | Accuracy, macro-F1, purity, consistency |
| Calibration | Frame + landmarks -> shot-local registration | Every frame or keyframe | Reject missing/degenerate geometry; retain reprojection error | Keypoint, homography, reprojection error, coverage |
| Projection | Footpoint + registration -> normalized pitch point | Every eligible observation | No pitch point when calibration invalid; flag edge uncertainty | Pitch-coordinate error and out-of-bounds rate |
| State | All stage outputs -> versioned match state | Every sampled frame | Preserve source frame, shot, track, confidences, provenance | Schema validation and replayability |
| Quality gate | State -> accepted/abstained state | Every sampled frame | Require calibration, player/team/role thresholds, minimum counts | Risk/coverage, abstention reasons |
| Shape | Accepted player states -> centroid/width/depth/spacing/compactness | Every accepted frame | Descriptive only; no intent or phase inference | Metric error and temporal jitter |
| Evaluation | Predictions + references -> metrics/report | Per run | Store dataset/config/code provenance with metrics | Detection, tracking, calibration, shape, reliability |

Phase 0 implements the state schema, dependency-injected protocols, explicit
quality gate, shot-boundary hook, and synthetic smoke components. It does not
claim to implement the model stages.

## Baselines and experiment plan

**B0 — image-space baseline:** fixed detector, ByteTrack-style association,
simple team labels, and image-space shape. No calibration and no meaningful
abstention.

**B1 — conventional pitch-space baseline:** same detector/tracker, framewise
landmarks, robust homography, footpoint projection, and a fixed validity
threshold.

**P1 — proposed system:** same controlled detector, motion plus appearance
association, shot reset, temporally filtered registration, independent
confidence propagation, and selective shape output.

Primary experiments:

1. image-space versus pitch-space;
2. framewise versus temporally filtered calibration;
3. motion/IoU versus appearance-assisted association;
4. hard output versus confidence-aware abstention at matched coverage;
5. ground-truth versus predicted detections;
6. ground-truth versus predicted team/role labels;
7. reset at cuts versus attempted continuity;
8. sensitivity to missing/false players;
9. frame-rate and sample-rate sensitivity;
10. cross-game and cross-broadcast transfer.

The primary shape result should be a predeclared error on team centroid and
spread metrics, reported with coverage and uncertainty intervals. Secondary
results include temporal jitter, shape-change agreement, and failure-stratified
performance. Improvement must not come solely from reporting fewer frames.

## Evaluation and acceptance criteria

**Detection:** precision, recall, mAP by class and visibility subset.

**Tracking:** HOTA, DetA, AssA, IDF1, identity switches, fragmentation,
track duration, and recovery after occlusion.

**Team/role:** accuracy, macro-F1, frame-weighted team purity, track-level
consistency, unknown precision, and confusion matrices.

**Calibration:** keypoint error, homography reprojection error, footpoint error,
valid-calibration coverage, and out-of-bounds rate.

**Shape:** centroid/width/depth/spacing/compactness error, temporal jitter,
correlation of shape changes, and error by player visibility/count.

**Reliability:** risk/coverage curves, selective risk, calibration error,
fixed-coverage error, abstention rate, and abstention reasons by failure type.

Engineering acceptance for Phase 0:

- clean Python 3.11+ run with no runtime dependencies;
- stable schema and manifest validation;
- deterministic smoke output and stable run identifier;
- experiment metadata for configuration, seed, code revision, dataset/model
  version, environment, and metrics/log events;
- deliberate bad calibration and uncertain classification both abstain;
- shot boundaries change context and invoke an available tracker reset.

Research acceptance is stricter: game-level split integrity, frozen baselines,
validation-only threshold selection, confidence intervals, and a preregistered
primary result at matched coverage.

## Hardware and storage

These are planning ranges, not measured performance claims:

- Minimum practical inference target: consumer NVIDIA GPU with roughly 8–12 GB
  VRAM, 8 logical CPU cores, and 32 GB RAM.
- Comfortable experimentation: 16–24 GB VRAM, especially for detector
  fine-tuning, ReID, and parallel ablations.
- Storage: reserve at least 250 GB for selected clips, decoded intermediates,
  detection/calibration caches, and checkpoints; full-game archives require
  more.
- Benchmark throughput, memory, and end-to-end latency on the actual GPU and
  sample rate. Do not substitute vendor claims for measurements.
- Cache upstream detections and calibration candidates so tactical ablations do
  not rerun expensive perception stages.

## Roadmap and gates

| Phase | Objective and deliverables | Experiment/metric | Acceptance and failure handling |
| --- | --- | --- | --- |
| 0 | Repository, schemas, config, manifests, logging, CLI, protocols, smoke | Determinism and invalid-geometry tests | Clean smoke; abstain instead of impute |
| 1 | Acquire/normalize selected data and freeze match-level splits | Manifest audit, label mapping, leakage scan | No leakage; exact release recorded |
| 2 | Establish fixed detection baseline | mAP/precision/recall by class and visibility | Held-out predictions reproducible |
| 3 | Establish tracking baseline | HOTA/DetA/AssA, IDF1, switches, fragmentation | Inspectable MOT failure report |
| 4 | Add team/role classification | Macro-F1, purity, unknown precision, consistency | Ambiguous cases remain unknown |
| 5 | Add pitch calibration | Reprojection, keypoint, coverage | Degenerate/insufficient geometry rejected |
| 6 | Add pitch trajectories and smoothing | Coordinate error, jitter, gap recovery | Confidence and provenance preserved |
| 7 | Freeze tactical state representation | Schema replay and version checks | Every field traceable to evidence |
| 8 | Implement descriptive shape analytics | Synthetic/reference formula tests | No full-team claim with missing players |
| 9 | Run continuous main-camera games | Resumability, cut/replay recovery, throughput | No cross-cut IDs without evidence |
| 10 | Robustness and reliability study | Risk/coverage and stratified metrics | Report failures, not only mean scores |
| 11 | Optimize | Speed/memory versus quality curves | Trade-offs documented and reproducible |
| 12 | Package final research demo | Reproducible run, report, overlays, artifacts | Claims trace to fixed data/code |

Before advancing a phase, its acceptance evidence is stored with the run. If a
dataset, model, or hardware dependency is unavailable, implement the contract
and synthetic/fixture test but label the blocked result as unvalidated.

## Risks, limitations, and future work

Major risks are camera cuts and replays, severe zooms, occlusion, similar kits,
goalkeeper/referee confusion, missing or false detections, field-line scarcity,
domain shift across stadiums and broadcasters, and the ecological gap between
short benchmark clips and full matches. The system must reset, abstain, or
preserve uncertainty in each case.

V1 explicitly excludes jersey-number recognition as a dependency, cross-camera
identity, possession certainty, player intent, passing networks, formation
labels as ground truth, automatic pressing/phase labels, full territorial
control, 3D pose as a prerequisite, and tactical language generation.

Future extensions may add a dedicated ball module, calibrated player influence
models, event-aligned tactical annotations, multi-camera identity, richer
evaluation on full games, and learned uncertainty calibration. Each extension
needs its own reference annotation, metric, and failure policy first.

## Definitions of done

**MVP:** a continuous main-camera clip can produce versioned, machine-readable
player states with track/team/role confidence, shot-local calibration,
normalized pitch coordinates, descriptive team-shape metrics, overlays, and
explicit abstention reasons. It is useful for measurement audits, not for
automated tactical advice.

**Final version:** a resumable full-match system with validated detector,
tracking, role/team, calibration, and reliability components; frozen
match-level evaluation; cached intermediate evidence; stratified robustness
reports; and a demo whose every tactical number links to source evidence and
the experiment provenance that produced it.

## Sources

- SoccerNet Game State Reconstruction:
  <https://www.soccer-net.org/tasks/game-state-reconstruction>
- SoccerNet Tracking:
  <https://www.soccer-net.org/tasks/tracking>
- SoccerNet Camera Calibration:
  <https://www.soccer-net.org/tasks/camera-calibration>
- SoccerNet task index:
  <https://www.soccer-net.org/tasks>