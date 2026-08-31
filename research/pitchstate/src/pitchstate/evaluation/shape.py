"""Shape evaluation metrics: centroid/width/depth/spacing/compactness error,
temporal jitter, and agreement on shape changes over time.

Scores :func:`pitchstate.tactics.shape.calculate_shape_metrics` output against
``research-plan.md``'s "Shape" metric list: "centroid, width, depth, spacing,
compactness error, temporal jitter, and agreement on shape changes over time."

Design note: abstention propagates, it is not treated as zero error
------------------------------------------------------------------------
``ShapeMetrics`` fields are all ``| None`` — :func:`calculate_shape_metrics`
returns an all-``None`` metrics object when there are no observed points (see
its docstring/behavior). A naive error function that treated ``None`` as 0
would silently score "no players observed" as "perfect agreement with the
reference," which is backwards: it is a missing comparison, not a correct
one. Every function here returns ``None`` for a field wherever either the
prediction or the reference for that field is ``None``, and reports how many
comparisons were actually possible (see ``ShapeErrorReport.compared_frames``)
so a caller can distinguish "small error" from "mostly incomparable."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pitchstate.schema import ShapeMetrics


@dataclass(frozen=True)
class ShapeError:
    """Per-field absolute error between one predicted and one reference frame.

    A field is ``None`` if either side lacked that field (abstained), rather
    than being scored as zero error.
    """

    centroid_error: float | None
    width_error: float | None
    depth_error: float | None
    spacing_error: float | None
    compactness_error: float | None


def _abs_diff(predicted: float | None, reference: float | None) -> float | None:
    if predicted is None or reference is None:
        return None
    return abs(predicted - reference)


def shape_error(predicted: ShapeMetrics, reference: ShapeMetrics) -> ShapeError:
    """Compute per-field absolute error for one predicted/reference frame pair."""

    if predicted.centroid is not None and reference.centroid is not None:
        centroid_error: float | None = (
            (predicted.centroid.x - reference.centroid.x) ** 2
            + (predicted.centroid.y - reference.centroid.y) ** 2
        ) ** 0.5
    else:
        centroid_error = None

    return ShapeError(
        centroid_error=centroid_error,
        width_error=_abs_diff(predicted.width, reference.width),
        depth_error=_abs_diff(predicted.depth, reference.depth),
        spacing_error=_abs_diff(predicted.mean_pairwise_spacing, reference.mean_pairwise_spacing),
        compactness_error=_abs_diff(predicted.compactness_proxy, reference.compactness_proxy),
    )


@dataclass(frozen=True)
class ShapeErrorFieldSummary:
    compared_frames: int
    mean: float | None
    max: float | None


@dataclass(frozen=True)
class ShapeErrorReport:
    """Aggregate error report over a sequence of predicted/reference frame pairs."""

    total_frames: int
    centroid: ShapeErrorFieldSummary
    width: ShapeErrorFieldSummary
    depth: ShapeErrorFieldSummary
    spacing: ShapeErrorFieldSummary
    compactness: ShapeErrorFieldSummary


def _summarize_field(errors: Sequence[float | None]) -> ShapeErrorFieldSummary:
    present = [error for error in errors if error is not None]
    if not present:
        return ShapeErrorFieldSummary(compared_frames=0, mean=None, max=None)
    return ShapeErrorFieldSummary(
        compared_frames=len(present),
        mean=sum(present) / len(present),
        max=max(present),
    )


def shape_error_report(
    predicted_sequence: Sequence[ShapeMetrics], reference_sequence: Sequence[ShapeMetrics]
) -> ShapeErrorReport:
    """Aggregate :func:`shape_error` over paired predicted/reference sequences.

    Sequences must be the same length and already frame-aligned by the
    caller; this function does no timestamp matching of its own, since doing
    that silently would risk comparing misaligned frames without the caller
    knowing.
    """

    if len(predicted_sequence) != len(reference_sequence):
        raise ValueError(
            "predicted_sequence and reference_sequence must have equal length "
            f"(got {len(predicted_sequence)} and {len(reference_sequence)}); "
            "align frames before calling shape_error_report"
        )
    if not predicted_sequence:
        raise ValueError("predicted_sequence and reference_sequence must be non-empty")

    per_frame = [shape_error(p, r) for p, r in zip(predicted_sequence, reference_sequence)]
    return ShapeErrorReport(
        total_frames=len(per_frame),
        centroid=_summarize_field([e.centroid_error for e in per_frame]),
        width=_summarize_field([e.width_error for e in per_frame]),
        depth=_summarize_field([e.depth_error for e in per_frame]),
        spacing=_summarize_field([e.spacing_error for e in per_frame]),
        compactness=_summarize_field([e.compactness_error for e in per_frame]),
    )


def temporal_jitter(sequence: Sequence[ShapeMetrics], *, field: str = "width") -> float | None:
    """Mean absolute frame-to-frame change in one scalar field over a sequence.

    High jitter on a field that should be physically stable (e.g. team width
    over a short, uninterrupted shot) is evidence of tracking or calibration
    instability, not real tactical movement — this is a diagnostic signal,
    not a correctness metric on its own.

    ``field`` must name a scalar (non-``None``-safe-comparable) attribute of
    ``ShapeMetrics``: ``"width"``, ``"depth"``, ``"mean_pairwise_spacing"``,
    or ``"compactness_proxy"``. Consecutive frames where either value is
    ``None`` are skipped (not scored as zero jitter), consistent with this
    module's abstention-propagation convention.
    """

    allowed_fields = {"width", "depth", "mean_pairwise_spacing", "compactness_proxy"}
    if field not in allowed_fields:
        raise ValueError(f"field must be one of {sorted(allowed_fields)}, got {field!r}")
    if len(sequence) < 2:
        raise ValueError("sequence must contain at least 2 frames to compute jitter")

    diffs: list[float] = []
    for previous, current in zip(sequence, sequence[1:]):
        previous_value = getattr(previous, field)
        current_value = getattr(current, field)
        if previous_value is None or current_value is None:
            continue
        diffs.append(abs(current_value - previous_value))

    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def shape_change_agreement(
    predicted_sequence: Sequence[ShapeMetrics],
    reference_sequence: Sequence[ShapeMetrics],
    *,
    field: str = "width",
    change_threshold: float = 0.0,
) -> float | None:
    """Fraction of frame-to-frame transitions where predicted and reference
    agree on the *direction* of change (increase/decrease/no-change) in
    ``field``.

    This targets "agreement on shape changes over time" specifically, as
    distinct from raw error magnitude: a system can have real per-frame error
    on the field's value while still correctly tracking whether the team is
    widening or narrowing, which is often the more decision-relevant signal.
    A transition counts as "no change" if the absolute delta is
    ``<= change_threshold`` on that side; both sides must be classifiable
    (neither value missing) for a transition to count. Returns ``None`` if no
    transition was classifiable on both sides.
    """

    if len(predicted_sequence) != len(reference_sequence):
        raise ValueError(
            "predicted_sequence and reference_sequence must have equal length "
            f"(got {len(predicted_sequence)} and {len(reference_sequence)})"
        )
    if len(predicted_sequence) < 2:
        raise ValueError("sequences must contain at least 2 frames to compute change agreement")

    def _direction(previous: float | None, current: float | None) -> int | None:
        if previous is None or current is None:
            return None
        delta = current - previous
        if abs(delta) <= change_threshold:
            return 0
        return 1 if delta > 0 else -1

    agreements = 0
    classifiable = 0
    for index in range(len(predicted_sequence) - 1):
        predicted_direction = _direction(
            getattr(predicted_sequence[index], field), getattr(predicted_sequence[index + 1], field)
        )
        reference_direction = _direction(
            getattr(reference_sequence[index], field), getattr(reference_sequence[index + 1], field)
        )
        if predicted_direction is None or reference_direction is None:
            continue
        classifiable += 1
        if predicted_direction == reference_direction:
            agreements += 1

    if classifiable == 0:
        return None
    return agreements / classifiable
