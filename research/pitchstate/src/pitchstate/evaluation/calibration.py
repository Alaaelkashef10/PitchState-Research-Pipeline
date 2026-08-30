"""Calibration evaluation metrics: reprojection error, coverage, selective risk.

This module scores :class:`pitchstate.calibration.homography.HomographyEstimate`
results against ``research-plan.md``'s "Calibration" and "Reliability" metric
lists:

- **Calibration:** homography reprojection error (mean/median/max), and
  valid-calibration coverage — implemented here.
- **Reliability:** risk/coverage curves, selective risk, error at fixed
  coverage, and abstention rates by failure category — implemented here.

What is explicitly out of scope
--------------------------------
The research plan's reliability list also names "calibration error" in the
ECE (expected calibration error) sense — how well a predicted confidence
matches empirical accuracy. That requires either a probabilistic confidence
score or a binary correctness event with calibrated bins, and
``HomographyEstimate`` currently exposes neither (it exposes a binary
``valid`` flag and a continuous reprojection error, not a probability). Adding
an ECE-style metric on top of that would mean inventing a confidence
score that nothing in this repository actually produces yet. This is
deferred rather than faked; see the architecture.md decision register entry
for this module.

Why "held-out" reprojection error matters
-------------------------------------------
``HomographyEstimate.reprojection_error_mean`` is computed over the same
correspondences used to *fit* the homography. For the minimal 4-correspondence
case that fit is an exact solve of an 8x8 system, so that error is ~0 by
construction and says nothing about generalization — it is an optimistic,
in-sample number, not an accuracy claim. This module accepts an optional
``held_out_reprojection_error`` per sample (error measured against
correspondence(s) *not* used in the fit) and prefers it wherever available.
Where it is not available, the module falls back to the in-sample fit error
and explicitly counts how often it had to do so
(``ReprojectionErrorSummary.fit_error_fallback_count``), so a summary is never
silently more optimistic than its inputs actually support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pitchstate.calibration.homography import HomographyEstimate


@dataclass(frozen=True)
class CalibrationEvaluationSample:
    """One evaluation instance: an attempted fit plus optional held-out error."""

    estimate: HomographyEstimate
    held_out_reprojection_error: float | None = None


@dataclass(frozen=True)
class ReprojectionErrorSummary:
    valid_sample_count: int
    fit_error_fallback_count: int
    mean: float | None
    median: float | None
    max: float | None
    p90: float | None


@dataclass(frozen=True)
class CoveragePoint:
    """One point on the selective risk/coverage curve.

    ``coverage`` is the fraction of *all* samples (including abstentions)
    accepted at this operating point. ``selective_risk`` is the mean error
    over exactly the accepted samples, in the order they were accepted.
    """

    coverage: float
    selective_risk: float
    accepted_count: int


def _sample_error(sample: CalibrationEvaluationSample) -> float | None:
    """Return the best-available error estimate, or None if abstained."""

    if not sample.estimate.valid:
        return None
    if sample.held_out_reprojection_error is not None:
        return sample.held_out_reprojection_error
    return sample.estimate.reprojection_error_mean


def valid_calibration_coverage(samples: Sequence[CalibrationEvaluationSample]) -> float:
    """Fraction of samples where a homography was successfully estimated.

    Raises :class:`ValueError` on an empty input rather than returning a
    numerically meaningless 0/0 result disguised as a real coverage figure.
    """

    if not samples:
        raise ValueError("samples must be non-empty")
    valid_count = sum(1 for sample in samples if sample.estimate.valid)
    return valid_count / len(samples)


def abstention_rate_by_reason(samples: Sequence[CalibrationEvaluationSample]) -> dict[str, float]:
    """Fraction of *all* samples that abstained, broken down by reason.

    Denominator is the full sample count (not just abstained samples), so
    these rates are directly comparable to ``valid_calibration_coverage`` —
    together they partition to 1.0.
    """

    if not samples:
        raise ValueError("samples must be non-empty")
    counts: dict[str, int] = {}
    for sample in samples:
        if sample.estimate.valid:
            continue
        reason = sample.estimate.reason or "unknown_reason"
        counts[reason] = counts.get(reason, 0) + 1
    total = len(samples)
    return {reason: count / total for reason, count in sorted(counts.items())}


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile (no interpolation) over already-sorted values."""

    if not sorted_values:
        raise ValueError("sorted_values must be non-empty")
    index = max(0, min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def reprojection_error_summary(samples: Sequence[CalibrationEvaluationSample]) -> ReprojectionErrorSummary:
    """Summarize reprojection error over samples with a successful fit.

    Uses each sample's held-out error where available, and its in-sample fit
    error otherwise (see module docstring); ``fit_error_fallback_count``
    reports how many samples used the weaker, optimistic fallback so a caller
    can judge how much to trust the summary. Returns an all-``None`` summary
    (not an error) when no sample produced a valid estimate, consistent with
    this project's explicit-abstention convention: "nothing was measurable"
    is itself a valid, reportable outcome.
    """

    errors: list[float] = []
    fallback_count = 0
    for sample in samples:
        if not sample.estimate.valid:
            continue
        if sample.held_out_reprojection_error is None:
            fallback_count += 1
        error = _sample_error(sample)
        assert error is not None  # valid sample always yields a numeric error
        errors.append(error)

    if not errors:
        return ReprojectionErrorSummary(
            valid_sample_count=0,
            fit_error_fallback_count=fallback_count,
            mean=None,
            median=None,
            max=None,
            p90=None,
        )

    ordered = sorted(errors)
    return ReprojectionErrorSummary(
        valid_sample_count=len(errors),
        fit_error_fallback_count=fallback_count,
        mean=sum(errors) / len(errors),
        median=_percentile(ordered, 0.5),
        max=ordered[-1],
        p90=_percentile(ordered, 0.9),
    )


def selective_risk_coverage_curve(
    samples: Sequence[CalibrationEvaluationSample],
) -> tuple[CoveragePoint, ...]:
    """Build a selective risk/coverage curve.

    Samples are ranked by ascending *in-sample fit* reprojection error — the
    only confidence-like signal available at prediction time, before any
    held-out ground truth is known — and accepted most-confident-first. Risk
    at each coverage level is the mean *true* error (held-out where available,
    else the same fit error) over the accepted samples. Ranking by fit error
    and scoring by held-out error (where both exist) keeps the curve honest:
    it cannot cheat by sorting on the number being evaluated.

    Returns an empty tuple if every sample abstained (no accepted operating
    point exists) rather than raising, matching this project's abstention
    conventions; a caller can check ``len(curve) == 0`` as a signal that
    calibration failed universally on this evaluation set.
    """

    if not samples:
        raise ValueError("samples must be non-empty")

    total = len(samples)
    valid_samples = [sample for sample in samples if sample.estimate.valid]
    if not valid_samples:
        return ()

    ranked = sorted(valid_samples, key=lambda sample: sample.estimate.reprojection_error_mean)
    true_errors = [_sample_error(sample) for sample in ranked]

    points: list[CoveragePoint] = []
    running_sum = 0.0
    for k, error in enumerate(true_errors, start=1):
        assert error is not None
        running_sum += error
        points.append(CoveragePoint(coverage=k / total, selective_risk=running_sum / k, accepted_count=k))
    return tuple(points)


def selective_risk_at_coverage(
    curve: Sequence[CoveragePoint], target_coverage: float
) -> CoveragePoint | None:
    """Return the curve point at the largest coverage <= target_coverage.

    Returns ``None`` if no point satisfies that (e.g. the curve is empty, or
    even the single most-confident sample already exceeds
    ``target_coverage``), rather than silently picking the nearest available
    point regardless of direction — reporting "risk at coverage >= X" as if
    it were "risk at coverage <= X" would overstate how much was accepted.
    """

    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be within [0.0, 1.0]")
    eligible = [point for point in curve if point.coverage <= target_coverage]
    if not eligible:
        return None
    return max(eligible, key=lambda point: point.coverage)
