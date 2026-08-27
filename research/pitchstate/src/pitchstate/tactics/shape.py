"""Small, transparent geometric primitives for the MVP shape layer."""

from __future__ import annotations

from math import hypot
from typing import Iterable, Sequence

from pitchstate.schema import Point2D, ShapeMetrics, Team, TrackObservation


def _cross(origin: Point2D, first: Point2D, second: Point2D) -> float:
    return (first.x - origin.x) * (second.y - origin.y) - (
        first.y - origin.y
    ) * (second.x - origin.x)


def convex_hull(points: Iterable[Point2D]) -> list[Point2D]:
    unique = sorted({(point.x, point.y) for point in points})
    if len(unique) <= 1:
        return [Point2D(*point) for point in unique]
    sorted_points = [Point2D(*point) for point in unique]
    lower: list[Point2D] = []
    for point in sorted_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[Point2D] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(points: Sequence[Point2D]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[index].x * points[(index + 1) % len(points)].y
            - points[(index + 1) % len(points)].x * points[index].y
            for index in range(len(points))
        )
    ) / 2.0


def calculate_shape_metrics(team: Team, observations: Sequence[TrackObservation]) -> ShapeMetrics:
    points = [observation.pitch_point for observation in observations if observation.pitch_point]
    if not points:
        return ShapeMetrics(team, 0, None, None, None, None, None, None)
    centroid = Point2D(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )
    width = max(point.x for point in points) - min(point.x for point in points)
    depth = max(point.y for point in points) - min(point.y for point in points)
    distances = [
        hypot(first.x - second.x, first.y - second.y)
        for index, first in enumerate(points)
        for second in points[index + 1 :]
    ]
    spacing = sum(distances) / len(distances) if distances else 0.0
    area = polygon_area(convex_hull(points))
    extent = width * depth
    compactness = area / extent if extent > 0 else 0.0
    return ShapeMetrics(
        team=team,
        player_count=len(points),
        centroid=centroid,
        width=width,
        depth=depth,
        mean_pairwise_spacing=spacing,
        convex_hull_area=area,
        compactness_proxy=compactness,
    )