import unittest

from pitchstate.calibration.homography import HomographyEstimate
from pitchstate.evaluation.calibration import (
    CalibrationEvaluationSample,
    abstention_rate_by_reason,
    reprojection_error_summary,
    selective_risk_at_coverage,
    selective_risk_coverage_curve,
    valid_calibration_coverage,
)


def _valid_estimate(fit_error: float) -> HomographyEstimate:
    return HomographyEstimate(
        valid=True,
        homography=None,  # not needed for these metrics
        correspondence_count=4,
        reprojection_error_mean=fit_error,
        reprojection_error_max=fit_error,
        reason=None,
    )


def _invalid_estimate(reason: str) -> HomographyEstimate:
    return HomographyEstimate(
        valid=False,
        homography=None,
        correspondence_count=3,
        reprojection_error_mean=None,
        reprojection_error_max=None,
        reason=reason,
    )


class CoverageTests(unittest.TestCase):
    def test_coverage_fraction_is_correct(self) -> None:
        samples = [
            CalibrationEvaluationSample(_valid_estimate(0.0)),
            CalibrationEvaluationSample(_valid_estimate(0.1)),
            CalibrationEvaluationSample(_invalid_estimate("insufficient_correspondences")),
            CalibrationEvaluationSample(_invalid_estimate("singular_system")),
        ]
        self.assertAlmostEqual(valid_calibration_coverage(samples), 0.5)

    def test_raises_on_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            valid_calibration_coverage([])


class AbstentionByReasonTests(unittest.TestCase):
    def test_rates_are_fractions_of_all_samples_and_partition_with_coverage(self) -> None:
        samples = [
            CalibrationEvaluationSample(_valid_estimate(0.0)),
            CalibrationEvaluationSample(_invalid_estimate("insufficient_correspondences")),
            CalibrationEvaluationSample(_invalid_estimate("insufficient_correspondences")),
            CalibrationEvaluationSample(_invalid_estimate("singular_system")),
        ]
        rates = abstention_rate_by_reason(samples)
        self.assertAlmostEqual(rates["insufficient_correspondences"], 0.5)
        self.assertAlmostEqual(rates["singular_system"], 0.25)
        self.assertAlmostEqual(sum(rates.values()) + valid_calibration_coverage(samples), 1.0)

    def test_valid_samples_are_excluded_from_reason_counts(self) -> None:
        samples = [CalibrationEvaluationSample(_valid_estimate(0.0))]
        self.assertEqual(abstention_rate_by_reason(samples), {})


class ReprojectionErrorSummaryTests(unittest.TestCase):
    def test_prefers_held_out_error_over_fit_error(self) -> None:
        samples = [
            CalibrationEvaluationSample(_valid_estimate(0.0), held_out_reprojection_error=2.0),
            CalibrationEvaluationSample(_valid_estimate(0.0), held_out_reprojection_error=4.0),
        ]
        summary = reprojection_error_summary(samples)
        self.assertEqual(summary.fit_error_fallback_count, 0)
        self.assertAlmostEqual(summary.mean, 3.0)
        self.assertAlmostEqual(summary.max, 4.0)

    def test_falls_back_to_fit_error_and_flags_it(self) -> None:
        samples = [CalibrationEvaluationSample(_valid_estimate(1.5))]
        summary = reprojection_error_summary(samples)
        self.assertEqual(summary.fit_error_fallback_count, 1)
        self.assertAlmostEqual(summary.mean, 1.5)

    def test_returns_all_none_when_every_sample_abstained(self) -> None:
        samples = [CalibrationEvaluationSample(_invalid_estimate("singular_system"))]
        summary = reprojection_error_summary(samples)
        self.assertEqual(summary.valid_sample_count, 0)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.median)
        self.assertIsNone(summary.max)
        self.assertIsNone(summary.p90)

    def test_p90_and_median_over_a_known_distribution(self) -> None:
        samples = [
            CalibrationEvaluationSample(_valid_estimate(0.0), held_out_reprojection_error=e)
            for e in [1.0, 2.0, 3.0, 4.0, 5.0]
        ]
        summary = reprojection_error_summary(samples)
        self.assertAlmostEqual(summary.median, 3.0)
        self.assertAlmostEqual(summary.p90, 5.0)


class SelectiveRiskCoverageCurveTests(unittest.TestCase):
    def test_ranks_by_fit_confidence_not_true_error(self) -> None:
        # Sample A: low fit error (looks confident) but a large held-out
        # error (overfit/misleading). Sample B: higher fit error but a small
        # held-out error. A "cheating" curve sorted by true error would put B
        # first; this curve must put A first because only fit error is known
        # at prediction time.
        sample_a = CalibrationEvaluationSample(_valid_estimate(0.01), held_out_reprojection_error=10.0)
        sample_b = CalibrationEvaluationSample(_valid_estimate(0.5), held_out_reprojection_error=0.2)
        curve = selective_risk_coverage_curve([sample_a, sample_b])
        self.assertEqual(len(curve), 2)
        self.assertAlmostEqual(curve[0].coverage, 0.5)
        self.assertAlmostEqual(curve[0].selective_risk, 10.0)  # A accepted first
        self.assertAlmostEqual(curve[1].coverage, 1.0)
        self.assertAlmostEqual(curve[1].selective_risk, (10.0 + 0.2) / 2)

    def test_coverage_accounts_for_abstained_samples_in_denominator(self) -> None:
        samples = [
            CalibrationEvaluationSample(_valid_estimate(0.0), held_out_reprojection_error=1.0),
            CalibrationEvaluationSample(_invalid_estimate("singular_system")),
        ]
        curve = selective_risk_coverage_curve(samples)
        self.assertEqual(len(curve), 1)
        self.assertAlmostEqual(curve[0].coverage, 0.5)  # only 1 of 2 total samples accepted

    def test_empty_curve_when_all_samples_abstain(self) -> None:
        samples = [CalibrationEvaluationSample(_invalid_estimate("singular_system"))]
        self.assertEqual(selective_risk_coverage_curve(samples), ())

    def test_raises_on_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            selective_risk_coverage_curve([])


class SelectiveRiskAtCoverageTests(unittest.TestCase):
    def test_returns_largest_point_not_exceeding_target(self) -> None:
        samples = [
            CalibrationEvaluationSample(_valid_estimate(0.1 * i), held_out_reprojection_error=float(i))
            for i in range(1, 5)
        ]
        curve = selective_risk_coverage_curve(samples)  # coverages: 0.25, 0.5, 0.75, 1.0
        point = selective_risk_at_coverage(curve, 0.6)
        self.assertAlmostEqual(point.coverage, 0.5)

    def test_returns_none_when_target_is_below_every_point(self) -> None:
        samples = [CalibrationEvaluationSample(_valid_estimate(0.0), held_out_reprojection_error=1.0)]
        curve = selective_risk_coverage_curve(samples)  # only point at coverage=1.0
        self.assertIsNone(selective_risk_at_coverage(curve, 0.1))

    def test_rejects_out_of_range_target(self) -> None:
        with self.assertRaises(ValueError):
            selective_risk_at_coverage((), 1.5)
