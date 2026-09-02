"""Tracking evaluation metrics: HOTA, DetA, AssA, IDF1, ID switches,
fragmentation, track duration, and recovery after occlusion.

Scores :class:`pitchstate.schema.TrackObservation` predictions against
reference (ground-truth) track observations, implementing
``research-plan.md``'s "Tracking" metric list.

Matching convention (shared limitation with evaluation.detection)
--------------------------------------------------------------------
Per-frame matching between predictions and references uses the same
confidence-ordered greedy IoU assignment as ``evaluation.detection`` (see
that module's docstring). The official HOTA/TrackEval reference
implementation instead solves each frame's assignment via the Hungarian
algorithm (a global optimum), which this dependency-free codebase does not
implement for the per-frame step (it would need to run once per frame per
threshold). Greedy assignment is a documented approximation: it can only
differ from the optimum in genuinely ambiguous configurations (multiple
overlapping candidates within one frame), which every test fixture in
``tests/test_tracking_evaluation.py`` is deliberately constructed to avoid,
so the metric values there are exact under either matching rule.

HOTA / DetA / AssA
-------------------
Implements the TrackEval definition directly:

- ``DetA`` = TP / (TP + FP + FN), aggregated over every frame at one IoU
  threshold.
- ``AssA`` = the occurrence-weighted mean, over every true-positive
  (predicted_id, reference_id) frame match, of
  TPA / (TPA + FPA + FNA), where for a given matched pair (p, g): TPA is how
  many TP frames had exactly that pair; FPA is how many TP frames had p
  matched to some other reference id; FNA is how many TP frames had g
  matched to some other predicted id.
- ``HOTA`` = sqrt(DetA * AssA) at one threshold; :func:`hota_summary`
  averages DetA/AssA/HOTA over the standard 0.05-0.95 (step 0.05) threshold
  set, matching the official protocol's threshold sweep.

IDF1
-----
:func:`identity_metrics` finds the single global one-to-one mapping between
predicted and reference track identities that maximizes total matched
occurrences (the standard IDF1 formulation), then reports
IDTP/IDFP/IDFN/IDP/IDR/IDF1 using total instance counts (TP + FN for the
reference side, TP + FP for the predicted side) so that genuinely missed or
spurious detections are counted, not just identity-assignment errors within
already-detected frames.

The optimal global mapping is found by exhaustive permutation search over
padded identity lists, not the Hungarian algorithm — exact for the identity
counts involved, but factorial in the larger side's identity count.
:data:`DEFAULT_MAX_IDENTITIES` caps this at 6 (720 permutations) by default;
:func:`identity_metrics` raises rather than silently truncating identities if
either side exceeds the caller-supplied cap, since silently dropping
identities would misrepresent the metric's own inputs.

Abstention convention
----------------------
Every function returns ``None`` for a metric that is undefined given its
inputs (e.g. no ground truth at all) rather than returning 0.0 or 1.0, for
the same reason as ``evaluation.detection`` and ``evaluation.calibration``:
"nothing to measure" and "measured and it was bad" must not collapse to the
same number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from math import sqrt
from typing import Sequence

from pitchstate.evaluation.detection import bounding_box_iou
from pitchstate.schema import TrackObservation

DEFAULT_HOTA_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
DEFAULT_MAX_IDENTITIES = 6


@dataclass(frozen=True)
class FrameMatchResult:
    frame_index: int
    matches: tuple[tuple[TrackObservation, TrackObservation, float], ...]
    unmatched_predictions: tuple[TrackObservation, ...]
    unmatched_references: tuple[TrackObservation, ...]


def match_frame(
    predictions: Sequence[TrackObservation],
    references: Sequence[TrackObservation],
    *,
    iou_threshold: float,
    frame_index: int,
) -> FrameMatchResult:
    """Greedily match one frame's predicted observations to reference observations.

    Same convention as ``evaluation.detection.match_detections_single_category``:
    predictions are processed in descending confidence order, each greedily
    claiming its highest-IoU unmatched reference at or above
    ``iou_threshold``.
    """

    sorted_predictions = sorted(predictions, key=lambda obs: obs.confidence, reverse=True)
    used = [False] * len(references)
    matches: list[tuple[TrackObservation, TrackObservation, float]] = []
    unmatched_predictions: list[TrackObservation] = []
    for prediction in sorted_predictions:
        best_iou = 0.0
        best_index = -1
        for index, reference in enumerate(references):
            if used[index]:
                continue
            iou = bounding_box_iou(prediction.detection.bounding_box, reference.detection.bounding_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            used[best_index] = True
            matches.append((prediction, references[best_index], best_iou))
        else:
            unmatched_predictions.append(prediction)
    unmatched_references = [reference for index, reference in enumerate(references) if not used[index]]
    return FrameMatchResult(frame_index, tuple(matches), tuple(unmatched_predictions), tuple(unmatched_references))


def match_all_frames(
    predictions: Sequence[TrackObservation], references: Sequence[TrackObservation], *, iou_threshold: float
) -> tuple[FrameMatchResult, ...]:
    """Run :func:`match_frame` independently for every frame present in either input."""

    frame_indices = sorted({obs.frame_index for obs in predictions} | {obs.frame_index for obs in references})
    results = []
    for frame_index in frame_indices:
        frame_predictions = [obs for obs in predictions if obs.frame_index == frame_index]
        frame_references = [obs for obs in references if obs.frame_index == frame_index]
        results.append(
            match_frame(frame_predictions, frame_references, iou_threshold=iou_threshold, frame_index=frame_index)
        )
    return tuple(results)


@dataclass(frozen=True)
class DetectionAccuracyResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    det_a: float | None


def detection_accuracy(frame_results: Sequence[FrameMatchResult]) -> DetectionAccuracyResult:
    tp = sum(len(r.matches) for r in frame_results)
    fp = sum(len(r.unmatched_predictions) for r in frame_results)
    fn = sum(len(r.unmatched_references) for r in frame_results)
    total = tp + fp + fn
    return DetectionAccuracyResult(tp, fp, fn, (tp / total) if total > 0 else None)


@dataclass(frozen=True)
class AssociationAccuracyResult:
    true_positive_occurrences: int
    ass_a: float | None


def association_accuracy(frame_results: Sequence[FrameMatchResult]) -> AssociationAccuracyResult:
    occurrences = [
        (prediction.track_id, reference.track_id)
        for result in frame_results
        for prediction, reference, _ in result.matches
    ]
    if not occurrences:
        return AssociationAccuracyResult(0, None)

    pair_counts = Counter(occurrences)
    predicted_totals = Counter(pred_id for pred_id, _ in occurrences)
    reference_totals = Counter(ref_id for _, ref_id in occurrences)

    total_tp = len(occurrences)
    weighted_sum = 0.0
    for (pred_id, ref_id), tpa in pair_counts.items():
        fpa = predicted_totals[pred_id] - tpa
        fna = reference_totals[ref_id] - tpa
        weighted_sum += tpa * (tpa / (tpa + fpa + fna))
    return AssociationAccuracyResult(total_tp, weighted_sum / total_tp)


def hota_at_threshold(
    predictions: Sequence[TrackObservation], references: Sequence[TrackObservation], *, iou_threshold: float
) -> tuple[DetectionAccuracyResult, AssociationAccuracyResult, float | None]:
    """DetA, AssA, and HOTA (``sqrt(DetA * AssA)``) at a single IoU threshold."""

    frame_results = match_all_frames(predictions, references, iou_threshold=iou_threshold)
    det = detection_accuracy(frame_results)
    ass = association_accuracy(frame_results)
    if det.det_a is None:
        return det, ass, None
    ass_a = ass.ass_a if ass.ass_a is not None else 0.0
    return det, ass, sqrt(det.det_a * ass_a)


@dataclass(frozen=True)
class HotaSummary:
    thresholds: tuple[float, ...]
    det_a: float | None
    ass_a: float | None
    hota: float | None


def hota_summary(
    predictions: Sequence[TrackObservation],
    references: Sequence[TrackObservation],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_HOTA_THRESHOLDS,
) -> HotaSummary:
    """Average DetA/AssA/HOTA over a threshold sweep (default: 0.05-0.95 step 0.05).

    Returns an all-``None`` summary if there is no ground truth at all (every
    threshold is then undefined), rather than averaging over an empty set.
    """

    det_as: list[float] = []
    ass_as: list[float] = []
    hotas: list[float] = []
    for threshold in iou_thresholds:
        det, ass, hota = hota_at_threshold(predictions, references, iou_threshold=threshold)
        if det.det_a is None:
            continue
        det_as.append(det.det_a)
        ass_as.append(ass.ass_a if ass.ass_a is not None else 0.0)
        hotas.append(hota if hota is not None else 0.0)

    if not det_as:
        return HotaSummary(tuple(iou_thresholds), None, None, None)
    return HotaSummary(
        tuple(iou_thresholds),
        sum(det_as) / len(det_as),
        sum(ass_as) / len(ass_as),
        sum(hotas) / len(hotas),
    )


@dataclass(frozen=True)
class IdentityMetrics:
    id_true_positives: int
    id_false_positives: int
    id_false_negatives: int
    idp: float | None
    idr: float | None
    idf1: float | None


def identity_metrics(
    frame_results: Sequence[FrameMatchResult], *, max_identities: int = DEFAULT_MAX_IDENTITIES
) -> IdentityMetrics:
    """IDF1: exact global one-to-one identity assignment maximizing matched occurrences.

    Raises :class:`ValueError` if either side has more than ``max_identities``
    distinct identities, rather than silently truncating and reporting a
    metric computed over an incomplete identity set.
    """

    occurrences = [
        (prediction.track_id, reference.track_id)
        for result in frame_results
        for prediction, reference, _ in result.matches
    ]
    pair_counts = Counter(occurrences)

    fn_counts = Counter(
        reference.track_id for result in frame_results for reference in result.unmatched_references
    )
    fp_counts = Counter(
        prediction.track_id for result in frame_results for prediction in result.unmatched_predictions
    )

    predicted_ids = sorted({pred_id for pred_id, _ in occurrences} | set(fp_counts))
    reference_ids = sorted({ref_id for _, ref_id in occurrences} | set(fn_counts))

    if len(predicted_ids) > max_identities or len(reference_ids) > max_identities:
        raise ValueError(
            f"identity_metrics supports at most {max_identities} identities per side "
            f"(got {len(predicted_ids)} predicted, {len(reference_ids)} reference); "
            "raise max_identities to accept the added permutation-search cost"
        )

    reference_totals = Counter(ref_id for _, ref_id in occurrences)
    predicted_totals = Counter(pred_id for pred_id, _ in occurrences)
    total_gt_instances = sum(reference_totals.values()) + sum(fn_counts.values())
    total_pred_instances = sum(predicted_totals.values()) + sum(fp_counts.values())

    if total_gt_instances == 0 and total_pred_instances == 0:
        return IdentityMetrics(0, 0, 0, None, None, None)

    size = max(len(predicted_ids), len(reference_ids))
    padded_predicted = predicted_ids + [None] * (size - len(predicted_ids))
    padded_reference = reference_ids + [None] * (size - len(reference_ids))

    best_total = -1
    for permutation in permutations(padded_reference):
        total = 0
        for pred_id, ref_id in zip(padded_predicted, permutation):
            if pred_id is not None and ref_id is not None:
                total += pair_counts.get((pred_id, ref_id), 0)
        if total > best_total:
            best_total = total

    id_true_positives = max(best_total, 0)
    id_false_negatives = total_gt_instances - id_true_positives
    id_false_positives = total_pred_instances - id_true_positives

    idp = id_true_positives / total_pred_instances if total_pred_instances > 0 else None
    idr = id_true_positives / total_gt_instances if total_gt_instances > 0 else None
    denominator = 2 * id_true_positives + id_false_positives + id_false_negatives
    idf1 = (2 * id_true_positives / denominator) if denominator > 0 else None

    return IdentityMetrics(id_true_positives, id_false_positives, id_false_negatives, idp, idr, idf1)


@dataclass(frozen=True)
class TrackQualityReport:
    """Per-reference-track quality over its own observed frame span."""

    reference_track_id: str
    span_frames: int
    matched_frames: int
    coverage: float
    id_switches: int
    fragmentations: int
    total_gaps: int
    recovered_after_gap: int


def _build_reference_timelines(
    frame_results: Sequence[FrameMatchResult],
) -> dict[str, list[tuple[int, str | None]]]:
    timelines: dict[str, list[tuple[int, str | None]]] = {}
    for result in frame_results:
        for prediction, reference, _ in result.matches:
            timelines.setdefault(reference.track_id, []).append((result.frame_index, prediction.track_id))
        for reference in result.unmatched_references:
            timelines.setdefault(reference.track_id, []).append((result.frame_index, None))
    for timeline in timelines.values():
        timeline.sort(key=lambda entry: entry[0])
    return timelines


def track_quality_reports(frame_results: Sequence[FrameMatchResult]) -> tuple[TrackQualityReport, ...]:
    """Per-reference-track ID switches, fragmentation, coverage, and gap recovery.

    - **ID switch**: a matched frame's predicted ID differs from the last
      *matched* predicted ID this reference track had, however many
      unmatched frames occurred in between (standard MOT convention).
    - **Fragmentation**: a transition from matched to unmatched within the
      track's own frame span (an interruption).
    - **Recovery after occlusion**: among fragmentation gaps, the fraction
      that resume with the *same* predicted ID the track had immediately
      before the gap, rather than a different one.
    """

    timelines = _build_reference_timelines(frame_results)
    reports = []
    for reference_track_id in sorted(timelines):
        timeline = timelines[reference_track_id]
        span_frames = timeline[-1][0] - timeline[0][0] + 1
        matched_frames = sum(1 for _, pred_id in timeline if pred_id is not None)

        id_switches = 0
        last_matched_id: str | None = None
        prev_matched = timeline[0][1] is not None
        fragmentations = 0
        total_gaps = 0
        recovered_after_gap = 0
        for index, (_, pred_id) in enumerate(timeline):
            currently_matched = pred_id is not None
            if currently_matched:
                if last_matched_id is not None and pred_id != last_matched_id:
                    id_switches += 1
            if index > 0:
                if prev_matched and not currently_matched:
                    fragmentations += 1
                    total_gaps += 1
                elif (not prev_matched) and currently_matched:
                    if last_matched_id is not None and pred_id == last_matched_id:
                        recovered_after_gap += 1
            if currently_matched:
                last_matched_id = pred_id
            prev_matched = currently_matched

        reports.append(
            TrackQualityReport(
                reference_track_id=reference_track_id,
                span_frames=span_frames,
                matched_frames=matched_frames,
                coverage=matched_frames / span_frames,
                id_switches=id_switches,
                fragmentations=fragmentations,
                total_gaps=total_gaps,
                recovered_after_gap=recovered_after_gap,
            )
        )
    return tuple(reports)
