import unittest

from pitchstate.datasets.annotations import (
    map_annotation_record,
    validate_annotation_records,
)


class AnnotationTests(unittest.TestCase):
    def test_ambiguous_labels_are_preserved_as_unknown(self) -> None:
        annotation = map_annotation_record(
            {
                "game_id": "game-1",
                "clip_id": "clip-1",
                "frame_index": 4,
                "track_id": "track-2",
                "team": "ambiguous",
                "role": None,
                "identity": None,
            }
        )
        self.assertEqual(annotation.team, "unknown")
        self.assertEqual(annotation.role, "unknown")
        self.assertEqual(annotation.identity, "unknown")

    def test_missing_and_inconsistent_records_are_reported(self) -> None:
        records = iter(
            [
                {
                    "game_id": "game-1",
                    "clip_id": "clip-1",
                    "frame_index": 0,
                    "track_id": "track-1",
                    "x": 1,
                    "y": 2,
                    "width": 4,
                },
                {"game_id": "game-1", "clip_id": "clip-1", "frame_index": -1},
            ]
        )
        report = validate_annotation_records(records)
        self.assertEqual(report.record_count, 2)
        self.assertEqual(report.valid_record_count, 0)
        self.assertFalse(report.valid)
        self.assertTrue(any(issue.field == "bounding_box" for issue in report.issues))
        self.assertTrue(any(issue.field == "track_id" for issue in report.issues))