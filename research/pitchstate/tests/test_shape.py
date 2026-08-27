import unittest

from pitchstate.tactics.shape import calculate_shape_metrics
from pitchstate.schema import (
    BoundingBox,
    Detection,
    TrackObservation,
)


def observation(track_id: str, team: str, x: float, y: float) -> TrackObservation:
    detection = Detection(
        detection_id=track_id,
        category="player",
        bounding_box=BoundingBox(x, y, 0, 0),
        confidence=1.0,
    )
    return TrackObservation(
        track_id=track_id,
        detection=detection,
        frame_index=0,
        confidence=1.0,
        team=team,
        role="player",
        pitch_point=detection.bounding_box.footpoint,
    )


class ShapeTests(unittest.TestCase):
    def test_shape_metrics_are_transparent(self) -> None:
        metrics = calculate_shape_metrics(
            "team_a",
            [observation("a", "team_a", 0, 0), observation("b", "team_a", 1, 1)],
        )
        self.assertEqual(metrics.player_count, 2)
        self.assertEqual(metrics.width, 1)
        self.assertEqual(metrics.depth, 1)
        self.assertAlmostEqual(metrics.mean_pairwise_spacing, 2**0.5)
        self.assertEqual(metrics.convex_hull_area, 0)