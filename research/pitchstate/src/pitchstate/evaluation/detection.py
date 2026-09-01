"""Detection evaluation metrics: precision, recall, and mAP by class and
visibility subset.

Scores :class:`pitchstate.schema.Detection` predictions against reference
(ground-truth) detections, implementing ``research-plan.md``'s "Detection"
metric list: "precision, recall, mAP by class and visibility subset."

Matching convention
--------------------
Predictions are matched to references per category, processed in descending
confidence order; each prediction is greedily assigned to the
highest-remaining-IoU unmatched reference of the same category, if that IoU
is at or above ``iou_threshold``. This is the standard PASCAL VOC / COCO
matching convention: it means a lower-confidence prediction cannot "steal" a
reference box away from a higher-confidence prediction that also overlaps it,
which is what makes confidence-ordered precision/recall curves (and AP)
meaningful.

Average precision uses the VOC 2010+ continuous (all-points) interpolation:
the precision envelope (max precision at any recall >= r) is integrated
against actual observed recall values, not a fixed grid. This is exact for
the sample size involved and avoids the coarser 11-point interpolation.

Visibility-subset AP uses an "ignore" convention, not a false-positive
penalty
------------------------------------------------------------------------
A prediction that matches a reference *outside* the requested visibility
subset (e.g. scoring "fully visible" detections while a prediction actually
matched a "heavily occluded" one) is excluded from the precision/recall
curve entirely -- it is neither a true positive nor a false positive for that
subset's score. Counting it as a false positive would penalize a detector for
correctly finding a real player who simply isn't in the subset being
scored, which would make visibility-stratified AP numbers misleading. This
mirrors the "ignore region" convention used by COCO-style detection
benchmarks.

Abstention convention
----------------------
:func:`average_precision` and :func:`average_precision_by_visibility` return
``None`` when there is no ground truth for the requested category (subset),
since "average precision" is undefined without any positives to recall --
returning 0.0 in that case would misrepresent "nothing to measure" as "total
failure." When ground truth exists but there are zero predictions, AP is
0.0, the standard convention (nothing was ever recalled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pitchstate.schema import BoundingBox, Detection


def bounding_box_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two axis-aligned boxes in (x, y, w, h) form."""

    a_x2, a_y2 = a.x + a.width, a.y + a.height
    b_x2, b_y2 = b.x + b.width, b.y + b.height

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0.0, a.width) * max(0.0, a.height)
    area_b = max(0.0, b.width) * max(0.0, b.height)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


@dataclass(frozen=True)
class MatchOutcome:
    """One prediction's assignment result, in confidence-descending order."""

    prediction: Detection
    is_true_positive: bool
    matched_reference: Detection | None
    iou: float | None


def match_detections_single_category(
    predictions: Sequence[Detection], references: Sequence[Detection], *, iou_threshold: float
) -> tuple[MatchOutcome, ...]:
    """Greedily match same-category predictions to references.

    Callers must pre-filter both sequences to a single category; this
    function does not check category equality, so passing mixed categories
    will silently produce a cross-category matching, which the module-level
    functions below avoid by filtering before calling this.
    """

    sorted_predictions = sorted(predictions, key=lambda detection: detection.confidence, reverse=True)
    used = [False] * len(references)
    outcomes: list[MatchOutcome] = []
    for prediction in sorted_predictions:
        best_iou = 0.0
        best_index = -1
        for index, reference in enumerate(references):
            if used[index]:
                continue
            iou = bounding_box_iou(prediction.bounding_box, reference.bounding_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            used[best_index] = True
            outcomes.append(MatchOutcome(prediction, True, references[best_index], best_iou))
        else:
            outcomes.append(
                MatchOutcome(prediction, False, None, best_iou if best_index >= 0 else None)
            )
    return tuple(outcomes)


@dataclass(frozen=True)
class PrecisionRecallPoint:
    precision: float | None
    recall: float | None
    true_positives: int
    false_positives: int
    false_negatives: int


def precision_recall_at_threshold(
    predictions: Sequence[Detection],
    references: Sequence[Detection],
    *,
    category: str,
    iou_threshold: float,
    confidence_threshold: float,
) -> PrecisionRecallPoint:
    """Precision/recall for one category at one fixed confidence threshold.

    ``precision`` is ``None`` when no predictions clear the confidence
    threshold (undefined, not zero); ``recall`` is ``None`` when there is no
    ground truth for the category (also undefined).
    """

    preds = [p for p in predictions if p.category == category and p.confidence >= confidence_threshold]
    refs = [r for r in references if r.category == category]

    outcomes = match_detections_single_category(preds, refs, iou_threshold=iou_threshold)
    true_positives = sum(1 for outcome in outcomes if outcome.is_true_positive)
    false_positives = len(outcomes) - true_positives
    false_negatives = len(refs) - true_positives

    precision = true_positives / len(preds) if preds else None
    recall = true_positives / len(refs) if refs else None
    return PrecisionRecallPoint(precision, recall, true_positives, false_positives, false_negatives)


def _integrate_precision_recall(precisions: Sequence[float], recalls: Sequence[float]) -> float:
    """VOC 2010+ all-points interpolated average precision."""

    padded_recall = [0.0, *recalls, 1.0]
    padded_precision = [0.0, *precisions, 0.0]
    for index in range(len(padded_precision) - 2, -1, -1):
        padded_precision[index] = max(padded_precision[index], padded_precision[index + 1])
    area = 0.0
    for index in range(len(padded_recall) - 1):
        delta_recall = padded_recall[index + 1] - padded_recall[index]
        if delta_recall != 0.0:
            area += delta_recall * padded_precision[index + 1]
    return area


def average_precision(
    predictions: Sequence[Detection],
    references: Sequence[Detection],
    *,
    category: str,
    iou_threshold: float,
) -> float | None:
    """VOC-style average precision for one category.

    Returns ``None`` (undefined) if ``references`` contains no instance of
    ``category``; returns ``0.0`` if references exist but there are zero
    matching predictions.
    """

    refs = [r for r in references if r.category == category]
    if not refs:
        return None
    preds = [p for p in predictions if p.category == category]
    if not preds:
        return 0.0

    outcomes = match_detections_single_category(preds, refs, iou_threshold=iou_threshold)
    total_gt = len(refs)
    cumulative_tp = 0
    cumulative_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for outcome in outcomes:
        if outcome.is_true_positive:
            cumulative_tp += 1
        else:
            cumulative_fp += 1
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
        recalls.append(cumulative_tp / total_gt)
    return _integrate_precision_recall(precisions, recalls)


def mean_average_precision(
    predictions: Sequence[Detection],
    references: Sequence[Detection],
    *,
    iou_threshold: float,
    categories: Sequence[str] | None = None,
) -> dict[str, float | None]:
    """Per-category AP plus a macro-averaged ``"mAP"`` entry.

    ``categories`` defaults to every category present in ``references``. The
    ``"mAP"`` entry averages only categories with a defined (non-``None``)
    AP; a category note explaining why is not included here because the
    per-category dict already shows which entries were ``None`` and why (see
    :func:`average_precision`).
    """

    if categories is None:
        categories = sorted({reference.category for reference in references})

    per_category: dict[str, float | None] = {
        category: average_precision(predictions, references, category=category, iou_threshold=iou_threshold)
        for category in categories
    }
    defined = [value for value in per_category.values() if value is not None]
    per_category["mAP"] = sum(defined) / len(defined) if defined else None
    return per_category


@dataclass(frozen=True)
class VisibilityTaggedDetection:
    """A reference detection annotated with a visibility bin (e.g. occlusion level)."""

    detection: Detection
    visibility: str


def average_precision_by_visibility(
    predictions: Sequence[Detection],
    references: Sequence[VisibilityTaggedDetection],
    *,
    category: str,
    visibility: str,
    iou_threshold: float,
) -> float | None:
    """AP for one category, restricted to one ground-truth visibility bin.

    Predictions may still match references outside ``visibility`` (they
    compete for the same boxes); such matches are excluded from the curve
    entirely rather than counted as false positives -- see module docstring
    for why. Returns ``None`` if there is no reference in this category and
    visibility bin.
    """

    same_category = [ref for ref in references if ref.detection.category == category]
    in_subset_flags = [ref.visibility == visibility for ref in same_category]
    total_gt_in_subset = sum(in_subset_flags)
    if total_gt_in_subset == 0:
        return None

    preds = [p for p in predictions if p.category == category]
    if not preds:
        return 0.0

    gt_detections = [ref.detection for ref in same_category]
    sorted_predictions = sorted(preds, key=lambda detection: detection.confidence, reverse=True)
    used = [False] * len(gt_detections)

    cumulative_tp = 0
    cumulative_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for prediction in sorted_predictions:
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_detections):
            if used[index]:
                continue
            iou = bounding_box_iou(prediction.bounding_box, gt.bounding_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index >= 0 and best_iou >= iou_threshold:
            used[best_index] = True
            if in_subset_flags[best_index]:
                cumulative_tp += 1
            else:
                continue  # ignored: matched a real, out-of-subset reference
        else:
            cumulative_fp += 1

        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
        recalls.append(cumulative_tp / total_gt_in_subset)

    if not precisions:
        return 0.0
    return _integrate_precision_recall(precisions, recalls)
