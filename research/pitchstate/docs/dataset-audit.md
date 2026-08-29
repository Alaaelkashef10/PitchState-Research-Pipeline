# SoccerNet dataset audit and Phase 1 feasibility gate

**Audit date:** 2026-08-28  
**Local release status:** not downloaded; video access is not authorized in this
environment  
**Progression status:** blocked at the acquisition/annotation-verification gate

## Source audit

The following official pages and development-kit records were checked:

- Game State Reconstruction:
  <https://www.soccer-net.org/tasks/game-state-reconstruction>
  and <https://github.com/SoccerNet/sn-gamestate>
- Tracking:
  <https://www.soccer-net.org/tasks/tracking>
  and <https://github.com/SoccerNet/sn-tracking>
- Camera Calibration:
  <https://www.soccer-net.org/tasks/camera-calibration>
  and <https://github.com/SoccerNet/sn-calibration>
- Data/access:
  <https://www.soccer-net.org/data>
  and <https://www.soccer-net.org/faq>

The official Game State development kit names the downloadable task
`gamestate-2024` and documents `train`, `valid`, `test`, and `challenge` folders.
The task page describes 57 train, 59 validation, and 50 test main-camera clips,
each 30 seconds at 1080p. It describes player, goalkeeper, referee, and other
classes, team-left/team-right association, jersey number, and field/court
localization, with GS-HOTA as the benchmark metric.

The official Tracking page describes 12 complete main-camera games for tracklets
and a 100-clip, 30-second, 1080p benchmark. It distinguishes association-only
from full tracking and lists players, goalkeepers, referees, staff, and ball.
HOTA is the main metric, decomposed into DetA and AssA.

The official Camera Calibration page describes 20,028 images and a 2,104-image
challenge set, including action/replay imagery. It evaluates line reprojection
error and completeness. This is calibration evidence, not automatically
time-aligned player or team-shape ground truth.

The general SoccerNet data page describes broadcast videos as `.mkv` at 25 fps
and 720p or 224p, while the task-specific GSR and Tracking pages describe their
benchmark clips as 1080p. The manifest keeps task-specific values separate and
does not generalize the 25 fps statement to GSR until the actual release is
inspected.

The data page also documents action/replay images, replay-grounding labels, and
camera-shot labels for the wider SoccerNet collection. Their existence does not
prove that a stable shot/replay join exists for the GSR clips, so the manifest
marks those joins as unverified.

## Access and license constraints

The official data page requires an NDA before downloading video. The FAQ states
that the videos contain league copyright, are intended for research rather than
commercial use, and should not be redistributed. Algorithm repositories have
their own licenses. No credentials, signed URLs, or private links are recorded
in the manifest. The `release_or_download_date` remains null until an
authorized download occurs.

## Local audit result

`data/manifests/example.json` is a source-audited, access-pending manifest:

| Item | Result |
| --- | --- |
| Exact task release | `gamestate-2024` named by the published GSR development kit |
| Local video/annotation files | Not available |
| Local match/game membership | Empty until authorized release audit |
| Split policy | Match-level disjoint; validation selects thresholds; test frozen |
| Player/role/team fields | Source-described; exact serialized values pending local inspection |
| Tracking IDs | Source task describes identity; exact serialized fields pending local inspection |
| Pitch/reference coordinates | Source-described for GSR; shape derivability not locally verified |
| Calibration annotations | Separate task; not claimed as GSR fields |
| Shot/replay metadata | Published for related tasks; GSR join not verified |
| Missing/ambiguous labels | Unknown-preserving mapper and issue report implemented |
| Leakage audit | Passes empty fixture: 0 games, 0 clips, 0 duplicates |

## Shape-ground-truth feasibility gate

The primary research question requires reference pitch coordinates aligned to
player/team records at frame or timestamp level. The source description makes
this plausible, but it is not enough to certify the released serialization,
coordinate convention, missingness, identity semantics, or alignment.

Therefore the project **must not progress to detector, tracking, tactical, or
model experiments** until an authorized local release passes all of the
following checks:

1. enumerate every match/game and clip;
2. confirm train/validation/test disjointness by game;
3. inspect representative raw annotation records;
4. confirm player/team/role fields and their ambiguous values;
5. confirm tracking identity fields and frame indexing;
6. confirm pitch-coordinate presence, reference convention, and missingness;
7. join player records to frame timestamps without silent interpolation;
8. report coverage of reference coordinates by frame, role, and team;
9. freeze the test membership before threshold selection.

Until then, the shape ground truth status is `not_locally_verified`, not a
synthetic proxy. Synthetic fixtures validate software contracts only.

## Implemented gate infrastructure

- `datasets.manifest` validates provenance, clip structure, annotation status,
  split policy, and match/clip membership. Schema v0.3 additionally requires a
  closed-vocabulary `validation_status` field (see architecture.md decision
  register, 2026-08-29) and accepts an optional `preprocessing_version`.
- `audit_split_integrity` fails loudly on repeated game or clip IDs.
- `datasets.split_policy.SplitAccessGuard` enforces at runtime that frozen
  test-split game/clip identities cannot be used for threshold selection or
  hyperparameter search, raising `TestSplitAccessError` immediately. This
  closes the gap left by `audit_split_integrity`, which only checks the
  manifest's own consistency and cannot see how downstream experiment code
  uses identities.
- `datasets.dedup.detect_duplicate_source_files` groups local files by
  content hash (sha256) rather than filename or size, to catch re-encoded or
  renamed duplicate clips once files exist locally. It is a local-integrity
  check, not a leakage check — a duplicate must still be cross-referenced
  against split membership to determine if it is also a leak.
- `datasets.annotations` maps known team/role values and preserves unknown
  team, role, and identity labels.
- `validate_annotation_records` reports missing or inconsistent records without
  repairing them.
- `make audit-manifest` runs the leakage audit.

The example manifest's `validation_status` is `not_locally_verified`: the
schema-level infrastructure above is implemented and unit-tested against
synthetic fixtures, but no authorized SoccerNet-GSR release has been
downloaded or locally audited. The nine-point feasibility gate in this
document still governs when `validation_status` may move to
`locally_verified`.

## Exact next experiment

After authorized data access, run a **read-only release audit** before any model
execution. Produce a locally verified manifest and an annotation coverage
report. The first permitted experiment after that gate is a reference-only
shape calculation from ground-truth pitch coordinates, with no detector,
tracker, or tactical inference, to establish whether centroid, width, depth,
spacing, and compactness can be derived without unsupported assumptions.