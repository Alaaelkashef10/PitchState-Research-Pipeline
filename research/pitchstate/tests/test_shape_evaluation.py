import unittest

from pitchstate.evaluation.shape import (
    shape_change_agreement,
    shape_error,
    shape_error_report,
    temporal_jitter,
)
from pitchstate.schema import Point2D, ShapeMetrics


def _metrics(
    centroid=None, width=None, depth=None, spacing=None, compactness=None, count: int = 2
) -> ShapeMetrics:
    return ShapeMetrics(
        team="team_a",
        player_count=count,
        centroid=centroid,
        width=width,
        depth=depth,
        mean_pairwise_spacing=spacing,
        convex_hull_area=None,
        compactness_proxy=compactness,
    )


class ShapeErrorTests(unittest.TestCase):
    def test_computes_expected_errors(self) -> None:
        predicted = _metrics(Point2D(1.0, 1.0), width=10.0, depth=20.0, spacing=5.0, compactness=0.5)
        reference = _metrics(Point2D(4.0, 5.0), width=12.0, depth=18.0, spacing=4.0, compactness=0.6)
        error = shape_error(predicted, reference)
        self.assertAlmostEqual(error.centroid_error, 5.0)  # 3-4-5 triangle
        self.assertAlmostEqual(error.width_error, 2.0)
        self.assertAlmostEqual(error.depth_error, 2.0)
        self.assertAlmostEqual(error.spacing_error, 1.0)
        self.assertAlmostEqual(error.compactness_error, 0.1)

    def test_missing_field_on_either_side_yields_none_not_zero(self) -> None:
        predicted = _metrics(width=None, depth=5.0)
        reference = _metrics(width=8.0, depth=5.0)
        error = shape_error(predicted, reference)
        self.assertIsNone(error.width_error)
        self.assertAlmostEqual(error.depth_error, 0.0)


class ShapeErrorReportTests(unittest.TestCase):
    def test_aggregates_mean_and_max_over_frames(self) -> None:
        predicted = [_metrics(width=10.0), _metrics(width=8.0), _metrics(width=6.0)]
        reference = [_metrics(width=12.0), _metrics(width=10.0), _metrics(width=6.0)]
        report = shape_error_report(predicted, reference)
        self.assertEqual(report.total_frames, 3)
        self.assertEqual(report.width.compared_frames, 3)
        self.assertAlmostEqual(report.width.mean, (2.0 + 2.0 + 0.0) / 3)
        self.assertAlmostEqual(report.width.max, 2.0)

    def test_field_with_no_comparable_frames_is_all_none(self) -> None:
        predicted = [_metrics(spacing=None), _metrics(spacing=None)]
        reference = [_metrics(spacing=5.0), _metrics(spacing=5.0)]
        report = shape_error_report(predicted, reference)
        self.assertEqual(report.spacing.compared_frames, 0)
        self.assertIsNone(report.spacing.mean)
        self.assertIsNone(report.spacing.max)

    def test_raises_on_mismatched_sequence_lengths(self) -> None:
        with self.assertRaises(ValueError):
            shape_error_report([_metrics(width=1.0)], [_metrics(width=1.0), _metrics(width=2.0)])

    def test_raises_on_empty_sequences(self) -> None:
        with self.assertRaises(ValueError):
            shape_error_report([], [])


class TemporalJitterTests(unittest.TestCase):
    def test_mean_absolute_frame_to_frame_change(self) -> None:
        sequence = [_metrics(width=10.0), _metrics(width=13.0), _metrics(width=11.0)]
        # deltas: |13-10|=3, |11-13|=2 -> mean 2.5
        self.assertAlmostEqual(temporal_jitter(sequence, field="width"), 2.5)

    def test_skips_transitions_with_missing_values(self) -> None:
        sequence = [_metrics(width=10.0), _metrics(width=None), _metrics(width=14.0)]
        # Only one transition is unclassifiable on both ends because None is
        # adjacent to both neighbors; no comparable transition exists.
        self.assertIsNone(temporal_jitter(sequence, field="width"))

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValueError):
            temporal_jitter([_metrics(width=1.0), _metrics(width=2.0)], field="not_a_field")

    def test_requires_at_least_two_frames(self) -> None:
        with self.assertRaises(ValueError):
            temporal_jitter([_metrics(width=1.0)], field="width")


class ShapeChangeAgreementTests(unittest.TestCase):
    def test_full_agreement_when_directions_match(self) -> None:
        predicted = [_metrics(width=10.0), _metrics(width=14.0), _metrics(width=12.0)]
        reference = [_metrics(width=20.0), _metrics(width=25.0), _metrics(width=22.0)]
        # predicted deltas: +4, -2 ; reference deltas: +5, -3 -> same signs both transitions
        self.assertAlmostEqual(shape_change_agreement(predicted, reference, field="width"), 1.0)

    def test_partial_agreement_when_one_transition_disagrees(self) -> None:
        predicted = [_metrics(width=10.0), _metrics(width=14.0), _metrics(width=12.0)]
        reference = [_metrics(width=20.0), _metrics(width=25.0), _metrics(width=30.0)]
        # transition 1: predicted +4 (up), reference +5 (up) -> agree
        # transition 2: predicted -2 (down), reference +5 (up) -> disagree
        self.assertAlmostEqual(shape_change_agreement(predicted, reference, field="width"), 0.5)

    def test_change_threshold_treats_small_deltas_as_no_change(self) -> None:
        predicted = [_metrics(width=10.0), _metrics(width=10.4)]
        reference = [_metrics(width=20.0), _metrics(width=19.7)]
        # Both deltas are within +-0.5 threshold -> both classified as "no change" -> agree
        self.assertAlmostEqual(
            shape_change_agreement(predicted, reference, field="width", change_threshold=0.5), 1.0
        )

    def test_returns_none_when_no_transition_is_classifiable(self) -> None:
        predicted = [_metrics(width=None), _metrics(width=None)]
        reference = [_metrics(width=1.0), _metrics(width=2.0)]
        self.assertIsNone(shape_change_agreement(predicted, reference, field="width"))

    def test_raises_on_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            shape_change_agreement([_metrics(width=1.0)], [_metrics(width=1.0), _metrics(width=2.0)])
