"""Planar homography estimation: the mathematical foundation for pitch calibration.

This module implements Direct-Linear-Transform-style (DLT) planar homography
estimation from point correspondences, in pure Python with no numpy/scipy
dependency — consistent with the "Python standard library for Phase 0"
decision in ``docs/architecture.md``, which this module extends rather than
revisits.

What this module is
--------------------
Given a set of ``(image_point, pitch_point)`` correspondences — i.e. pixel
coordinates paired with known real-world pitch coordinates, such as line
intersections, penalty-spot, or corner markers — :func:`estimate_homography`
solves for the 3x3 projective transform mapping image points to pitch points,
and :meth:`Homography.apply` projects arbitrary image points through it.

What this module is NOT
------------------------
- It is **not** wired into :class:`pitchstate.calibration.interface.Calibrator`
  or :class:`pitchstate.schema.CalibrationState` yet. Doing so would require a
  concrete source of pitch-keypoint correspondences (a keypoint detector run
  against real broadcast frames), which does not exist in this repository and
  cannot be built without authorized SoccerNet video access (see
  ``docs/dataset-audit.md``). Wiring this into the pipeline before that exists
  would mean shipping an untestable stub, which the project's own "no fake
  progress" discipline rules out.
- It makes **no claim of real-world calibration accuracy**. Every test in
  ``tests/test_homography.py`` uses synthetic, hand-constructed
  correspondences with a known ground-truth transform. That validates the
  *mathematics* (the estimator recovers the transform it was given, handles
  degenerate input, is numerically stable on the systems tested). It says
  nothing about how well pitch-keypoint detection or this estimator would
  perform against real broadcast footage, motion blur, lens distortion, or
  detector noise, none of which are modeled here.

Known mathematical limitation
------------------------------
The linear system below is solved under the normalization ``h33 = 1`` (see
``_design_rows``). This is the standard simplification for a minimal 4-point
solve and is valid as long as the true homography does not map any of the
input points to the line at infinity (``h33 == 0`` in the unnormalized
system) — a degenerate case that essentially never occurs for a pitch viewed
from a normal broadcast camera angle, but is a real (if narrow) limitation
compared to an SVD-based DLT solver, which has no such assumption. An
SVD-based solver was not used here because it requires either a numerical
linear algebra dependency or a hand-implemented eigen-solver; the simpler,
fully-inspectable normalization is used instead and is documented rather than
silently assumed away. For more than 4 correspondences, the same
normalization is used in a least-squares (normal equations) fit rather than a
minimal exact solve; normal equations square the condition number of the
underlying system relative to an SVD approach, which is a known, documented
numerical-stability trade-off, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from typing import Sequence

from pitchstate.schema import Point2D

Matrix = list[list[float]]

#: Reasons estimate_homography can abstain instead of returning a homography.
INSUFFICIENT_CORRESPONDENCES = "insufficient_correspondences"
DEGENERATE_COLLINEAR_IMAGE_POINTS = "degenerate_configuration_collinear_image_points"
DEGENERATE_COLLINEAR_PITCH_POINTS = "degenerate_configuration_collinear_pitch_points"
DEGENERATE_DUPLICATE_IMAGE_POINTS = "degenerate_configuration_duplicate_image_points"
DEGENERATE_DUPLICATE_PITCH_POINTS = "degenerate_configuration_duplicate_pitch_points"
SINGULAR_SYSTEM = "singular_system"
NEAR_INFINITE_PROJECTION = "near_infinite_projection_during_validation"


class SingularMatrixError(ValueError):
    """Raised when a linear system has no unique solution (degenerate input)."""


def solve_linear_system(a: Matrix, b: Sequence[float], *, pivot_tolerance: float = 1e-9) -> list[float]:
    """Solve ``A x = b`` via Gaussian elimination with partial pivoting.

    Pure-Python, square-system solver. Raises :class:`SingularMatrixError`
    rather than returning a numerically meaningless result when a pivot is
    below ``pivot_tolerance`` — degenerate correspondence sets (e.g. collinear
    points) are expected to route through here and must fail loudly rather
    than produce a homography that looks valid but is not.
    """

    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        raise ValueError("A must be a square n x n matrix matching len(b)")
    aug = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < pivot_tolerance:
            raise SingularMatrixError(f"matrix is singular or near-singular at column {col}")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if factor != 0.0:
                    aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(n + 1)]
    return [aug[i][n] for i in range(n)]


@dataclass(frozen=True)
class Homography:
    """A 3x3 projective transform, stored as a tuple of three row-tuples."""

    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def apply(self, point: Point2D, *, denom_tolerance: float = 1e-9) -> Point2D | None:
        """Project ``point`` through the homography.

        Returns ``None`` (abstains) rather than a huge or infinite coordinate
        when the homogeneous denominator is near zero — that condition means
        the point maps arbitrarily close to the line at infinity, which is
        not a physically meaningful pitch coordinate.
        """

        h = self.matrix
        denom = h[2][0] * point.x + h[2][1] * point.y + h[2][2]
        if abs(denom) < denom_tolerance:
            return None
        proj_x = (h[0][0] * point.x + h[0][1] * point.y + h[0][2]) / denom
        proj_y = (h[1][0] * point.x + h[1][1] * point.y + h[1][2]) / denom
        return Point2D(proj_x, proj_y)


@dataclass(frozen=True)
class HomographyEstimate:
    """Result of :func:`estimate_homography`, including abstention metadata."""

    valid: bool
    homography: Homography | None
    correspondence_count: int
    reprojection_error_mean: float | None
    reprojection_error_max: float | None
    reason: str | None = None


def _design_rows(image_point: Point2D, pitch_point: Point2D) -> tuple[list[float], list[float], float, float]:
    """Return the two DLT design-matrix rows and RHS values for one correspondence.

    Derivation (with h33 fixed to 1, see module docstring):
        X * (h7*x + h8*y + 1) = h1*x + h2*y + h3
        Y * (h7*x + h8*y + 1) = h4*x + h5*y + h6
    rearranged into linear rows over unknowns [h1..h8]:
        [x, y, 1, 0, 0, 0, -x*X, -y*X] . h = X
        [0, 0, 0, x, y, 1, -x*Y, -y*Y] . h = Y
    """

    x, y = image_point.x, image_point.y
    big_x, big_y = pitch_point.x, pitch_point.y
    row1 = [x, y, 1.0, 0.0, 0.0, 0.0, -x * big_x, -y * big_x]
    row2 = [0.0, 0.0, 0.0, x, y, 1.0, -x * big_y, -y * big_y]
    return row1, row2, big_x, big_y


def _triplet_is_collinear(a: Point2D, b: Point2D, c: Point2D, *, sin_angle_tolerance: float = 1e-6) -> bool:
    ab = (b.x - a.x, b.y - a.y)
    ac = (c.x - a.x, c.y - a.y)
    norm_ab = hypot(*ab)
    norm_ac = hypot(*ac)
    if norm_ab < 1e-12 or norm_ac < 1e-12:
        # Coincident points: treated as a degenerate (collinear) configuration
        # by this check; duplicate-point detection below gives a more specific
        # abstention reason for that case.
        return True
    cross = ab[0] * ac[1] - ab[1] * ac[0]
    sin_angle = abs(cross) / (norm_ab * norm_ac)
    return sin_angle < sin_angle_tolerance


def _has_collinear_triplet(points: Sequence[Point2D]) -> bool:
    return any(_triplet_is_collinear(a, b, c) for a, b, c in combinations(points, 3))


def _has_duplicate_point(points: Sequence[Point2D], *, distance_tolerance: float = 1e-9) -> bool:
    return any(
        hypot(a.x - b.x, a.y - b.y) < distance_tolerance for a, b in combinations(points, 2)
    )


def estimate_homography(correspondences: Sequence[tuple[Point2D, Point2D]]) -> HomographyEstimate:
    """Estimate a planar homography from ``(image_point, pitch_point)`` pairs.

    Requires at least 4 correspondences. With exactly 4, solves the minimal
    8x8 linear system directly. With more than 4, fits via least squares
    (normal equations) over the same linearization. Abstains (``valid=False``
    with a ``reason``) rather than returning a numerically unstable result for
    degenerate input: too few points, collinear or duplicate points in either
    plane, or a singular linear system.
    """

    n = len(correspondences)
    if n < 4:
        return HomographyEstimate(
            valid=False,
            homography=None,
            correspondence_count=n,
            reprojection_error_mean=None,
            reprojection_error_max=None,
            reason=INSUFFICIENT_CORRESPONDENCES,
        )

    image_points = [pair[0] for pair in correspondences]
    pitch_points = [pair[1] for pair in correspondences]

    if _has_duplicate_point(image_points):
        return HomographyEstimate(
            False, None, n, None, None, DEGENERATE_DUPLICATE_IMAGE_POINTS
        )
    if _has_duplicate_point(pitch_points):
        return HomographyEstimate(
            False, None, n, None, None, DEGENERATE_DUPLICATE_PITCH_POINTS
        )
    if _has_collinear_triplet(image_points):
        return HomographyEstimate(
            False, None, n, None, None, DEGENERATE_COLLINEAR_IMAGE_POINTS
        )
    if _has_collinear_triplet(pitch_points):
        return HomographyEstimate(
            False, None, n, None, None, DEGENERATE_COLLINEAR_PITCH_POINTS
        )

    rows: list[list[float]] = []
    rhs: list[float] = []
    for image_point, pitch_point in correspondences:
        row1, row2, b1, b2 = _design_rows(image_point, pitch_point)
        rows.append(row1)
        rhs.append(b1)
        rows.append(row2)
        rhs.append(b2)

    try:
        if n == 4:
            h = solve_linear_system(rows, rhs)
        else:
            # Normal equations: (A^T A) h = A^T b, reducing the 2n x 8
            # overdetermined system to an 8x8 square system solvable with the
            # same Gaussian-elimination routine used for the minimal case.
            unknowns = 8
            ata = [[0.0] * unknowns for _ in range(unknowns)]
            atb = [0.0] * unknowns
            for row, value in zip(rows, rhs):
                for i in range(unknowns):
                    atb[i] += row[i] * value
                    for j in range(unknowns):
                        ata[i][j] += row[i] * row[j]
            h = solve_linear_system(ata, atb)
    except SingularMatrixError:
        return HomographyEstimate(False, None, n, None, None, SINGULAR_SYSTEM)

    homography = Homography(
        (
            (h[0], h[1], h[2]),
            (h[3], h[4], h[5]),
            (h[6], h[7], 1.0),
        )
    )

    errors: list[float] = []
    for image_point, pitch_point in correspondences:
        projected = homography.apply(image_point)
        if projected is None:
            return HomographyEstimate(False, None, n, None, None, NEAR_INFINITE_PROJECTION)
        errors.append(hypot(projected.x - pitch_point.x, projected.y - pitch_point.y))

    return HomographyEstimate(
        valid=True,
        homography=homography,
        correspondence_count=n,
        reprojection_error_mean=sum(errors) / len(errors),
        reprojection_error_max=max(errors),
        reason=None,
    )
