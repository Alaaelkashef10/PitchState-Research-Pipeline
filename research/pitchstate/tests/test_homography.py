import unittest

from pitchstate.calibration.homography import (
    DEGENERATE_COLLINEAR_IMAGE_POINTS,
    DEGENERATE_COLLINEAR_PITCH_POINTS,
    DEGENERATE_DUPLICATE_IMAGE_POINTS,
    INSUFFICIENT_CORRESPONDENCES,
    Homography,
    SingularMatrixError,
    estimate_homography,
    solve_linear_system,
)
from pitchstate.schema import Point2D


def _apply_ground_truth(matrix, point: Point2D) -> Point2D:
    """Forward-project a point through a known 3x3 matrix (test helper only)."""

    denom = matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2]
    x = (matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2]) / denom
    y = (matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2]) / denom
    return Point2D(x, y)


class SolveLinearSystemTests(unittest.TestCase):
    def test_solves_simple_well_conditioned_system(self) -> None:
        # x + y = 3 ; x - y = 1  =>  x=2, y=1
        solution = solve_linear_system([[1.0, 1.0], [1.0, -1.0]], [3.0, 1.0])
        self.assertAlmostEqual(solution[0], 2.0, places=9)
        self.assertAlmostEqual(solution[1], 1.0, places=9)

    def test_raises_on_singular_matrix(self) -> None:
        with self.assertRaises(SingularMatrixError):
            solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])

    def test_raises_on_mismatched_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            solve_linear_system([[1.0, 2.0], [3.0, 4.0]], [1.0])


class EstimateHomographyCorrectnessTests(unittest.TestCase):
    def test_identity_mapping_is_recovered_exactly(self) -> None:
        image_points = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]
        correspondences = [(p, p) for p in image_points]
        result = estimate_homography(correspondences)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.reprojection_error_mean, 0.0, places=6)
        projected = result.homography.apply(Point2D(5, 5))
        self.assertAlmostEqual(projected.x, 5.0, places=6)
        self.assertAlmostEqual(projected.y, 5.0, places=6)

    def test_affine_scale_and_translation_recovered_from_minimal_points(self) -> None:
        # Known ground truth: pitch = 2 * image + (3, -1)
        image_points = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 20), Point2D(0, 20)]
        correspondences = [(p, Point2D(2 * p.x + 3, 2 * p.y - 1)) for p in image_points]
        result = estimate_homography(correspondences)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.reprojection_error_mean, 0.0, places=6)
        # Held-out point not used in the fit: generalization, not just exact refit.
        held_out = Point2D(4, 7)
        projected = result.homography.apply(held_out)
        self.assertAlmostEqual(projected.x, 2 * 4 + 3, places=6)
        self.assertAlmostEqual(projected.y, 2 * 7 - 1, places=6)

    def test_true_perspective_homography_is_recovered(self) -> None:
        # A genuine projective transform (nonzero h7, h8) mimicking a broadcast
        # camera view of a pitch: parallel sidelines converge in image space.
        ground_truth = (
            (1.0, 0.2, 5.0),
            (0.0, 1.3, 2.0),
            (0.001, 0.0007, 1.0),
        )
        image_points = [
            Point2D(0, 0),
            Point2D(50, 0),
            Point2D(50, 30),
            Point2D(0, 30),
            Point2D(32, 11),  # 5th point makes this an overdetermined fit
        ]
        correspondences = [(p, _apply_ground_truth(ground_truth, p)) for p in image_points]
        result = estimate_homography(correspondences)
        self.assertTrue(result.valid)
        self.assertLess(result.reprojection_error_mean, 1e-6)
        held_out = Point2D(40, 10)
        expected = _apply_ground_truth(ground_truth, held_out)
        projected = result.homography.apply(held_out)
        self.assertAlmostEqual(projected.x, expected.x, places=5)
        self.assertAlmostEqual(projected.y, expected.y, places=5)

    def test_overdetermined_least_squares_fits_consistent_noiseless_data(self) -> None:
        ground_truth = (
            (1.5, 0.0, 10.0),
            (0.0, 1.5, -4.0),
            (0.0002, 0.0003, 1.0),
        )
        image_points = [
            Point2D(0, 0),
            Point2D(60, 0),
            Point2D(60, 40),
            Point2D(0, 40),
            Point2D(20, 12),
            Point2D(47, 28),
            Point2D(13, 33),
        ]
        correspondences = [(p, _apply_ground_truth(ground_truth, p)) for p in image_points]
        result = estimate_homography(correspondences)
        self.assertTrue(result.valid)
        self.assertEqual(result.correspondence_count, 7)
        self.assertLess(result.reprojection_error_mean, 1e-6)
        self.assertLess(result.reprojection_error_max, 1e-5)


class EstimateHomographyDegeneracyTests(unittest.TestCase):
    def test_fewer_than_four_correspondences_abstains(self) -> None:
        points = [Point2D(0, 0), Point2D(1, 0), Point2D(1, 1)]
        result = estimate_homography([(p, p) for p in points])
        self.assertFalse(result.valid)
        self.assertIsNone(result.homography)
        self.assertEqual(result.reason, INSUFFICIENT_CORRESPONDENCES)

    def test_collinear_image_points_abstain(self) -> None:
        image_points = [Point2D(0, 0), Point2D(1, 0), Point2D(2, 0), Point2D(3, 0)]
        pitch_points = [Point2D(0, 0), Point2D(10, 0), Point2D(20, 5), Point2D(30, 8)]
        result = estimate_homography(list(zip(image_points, pitch_points)))
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, DEGENERATE_COLLINEAR_IMAGE_POINTS)

    def test_collinear_pitch_points_abstain(self) -> None:
        image_points = [Point2D(0, 0), Point2D(10, 0), Point2D(20, 5), Point2D(30, 8)]
        pitch_points = [Point2D(0, 0), Point2D(1, 0), Point2D(2, 0), Point2D(3, 0)]
        result = estimate_homography(list(zip(image_points, pitch_points)))
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, DEGENERATE_COLLINEAR_PITCH_POINTS)

    def test_duplicate_image_points_abstain(self) -> None:
        image_points = [Point2D(0, 0), Point2D(0, 0), Point2D(10, 10), Point2D(0, 10)]
        pitch_points = [Point2D(0, 0), Point2D(5, 5), Point2D(10, 10), Point2D(0, 10)]
        result = estimate_homography(list(zip(image_points, pitch_points)))
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, DEGENERATE_DUPLICATE_IMAGE_POINTS)


class HomographyApplyTests(unittest.TestCase):
    def test_apply_abstains_when_point_maps_near_infinity(self) -> None:
        # h[2] = (1, 0, -10): denominator is zero exactly at x=10.
        homography = Homography(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, -10.0)))
        self.assertIsNone(homography.apply(Point2D(10.0, 5.0)))

    def test_apply_returns_finite_point_away_from_singularity(self) -> None:
        homography = Homography(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, -10.0)))
        projected = homography.apply(Point2D(0.0, 5.0))
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected.x, 0.0)
        self.assertAlmostEqual(projected.y, -0.5)
