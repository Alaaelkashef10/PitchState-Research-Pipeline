import unittest

from pitchstate.evaluation.tracking import (
    detection_accuracy,
    association_accuracy,
    hota_at_threshold,
    hota_summary,
    identity_metrics,
    match_all_frames,
    track_quality_reports,
)
from pitchstate.schema import BoundingBox, Detection, TrackObservation


def _obs(track_id: str, frame_index: int, x: float, y: float, confidence: float = 1.0) -> TrackObservation:
    detection = Detection(
        detection_id=f"{track_id}-{frame_index}",
        category="player",
        bounding_box=BoundingBox(x, y, 10, 10),
        confidence=confidence,
    )
    return TrackObservation(track_id=track_id, detection=detection, frame_index=frame_index, confidence=confidence)


class PerfectTrackingTests(unittest.TestCase):
    def _perfect_sequence(self) -> tuple[list[TrackObservation], list[TrackObservation]]:
        # One reference track, 5 frames, always found by the same predicted ID at the same spot.
        refs = [_obs("gt-1", frame, 0, 0) for frame in range(5)]
        preds = [_obs("p-1", frame, 0, 0) for frame in range(5)]
        return preds, refs

    def test_det_a_is_one(self) -> None:
        preds, refs = self._perfect_sequence()
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        det = detection_accuracy(frame_results)
        self.assertEqual(det.true_positives, 5)
        self.assertEqual(det.false_positives, 0)
        self.assertEqual(det.false_negatives, 0)
        self.assertAlmostEqual(det.det_a, 1.0)

    def test_ass_a_is_one(self) -> None:
        preds, refs = self._perfect_sequence()
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        ass = association_accuracy(frame_results)
        self.assertAlmostEqual(ass.ass_a, 1.0)

    def test_hota_is_one(self) -> None:
        preds, refs = self._perfect_sequence()
        _, _, hota = hota_at_threshold(preds, refs, iou_threshold=0.5)
        self.assertAlmostEqual(hota, 1.0)

    def test_idf1_is_one(self) -> None:
        preds, refs = self._perfect_sequence()
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        identity = identity_metrics(frame_results)
        self.assertEqual(identity.id_true_positives, 5)
        self.assertEqual(identity.id_false_positives, 0)
        self.assertEqual(identity.id_false_negatives, 0)
        self.assertAlmostEqual(identity.idf1, 1.0)

    def test_no_id_switches_or_fragmentation(self) -> None:
        preds, refs = self._perfect_sequence()
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        reports = track_quality_reports(frame_results)
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual(report.id_switches, 0)
        self.assertEqual(report.fragmentations, 0)
        self.assertAlmostEqual(report.coverage, 1.0)


class MissedDetectionTests(unittest.TestCase):
    def test_missing_prediction_counts_as_false_negative(self) -> None:
        refs = [_obs("gt-1", frame, 0, 0) for frame in range(3)]
        preds = [_obs("p-1", 0, 0, 0), _obs("p-1", 2, 0, 0)]  # frame 1 missed entirely
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        det = detection_accuracy(frame_results)
        self.assertEqual(det.true_positives, 2)
        self.assertEqual(det.false_negatives, 1)
        self.assertEqual(det.false_positives, 0)
        self.assertAlmostEqual(det.det_a, 2 / 3)

    def test_spurious_prediction_counts_as_false_positive(self) -> None:
        refs = [_obs("gt-1", 0, 0, 0)]
        preds = [_obs("p-1", 0, 0, 0), _obs("p-2", 0, 500, 500)]  # p-2 matches nothing
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        det = detection_accuracy(frame_results)
        self.assertEqual(det.true_positives, 1)
        self.assertEqual(det.false_positives, 1)
        self.assertAlmostEqual(det.det_a, 0.5)

    def test_det_a_is_none_with_no_ground_truth_and_no_predictions(self) -> None:
        frame_results = match_all_frames([], [], iou_threshold=0.5)
        det = detection_accuracy(frame_results)
        self.assertIsNone(det.det_a)


class IdentitySwitchTests(unittest.TestCase):
    def test_id_switch_is_detected_and_penalizes_hota_and_idf1(self) -> None:
        refs = [_obs("gt-1", frame, 0, 0) for frame in range(4)]
        # Same GT track picked up by p-1 for frames 0-1, then p-2 for frames 2-3: one ID switch.
        preds = [
            _obs("p-1", 0, 0, 0),
            _obs("p-1", 1, 0, 0),
            _obs("p-2", 2, 0, 0),
            _obs("p-2", 3, 0, 0),
        ]
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        reports = track_quality_reports(frame_results)
        self.assertEqual(reports[0].id_switches, 1)

        # DetA is unaffected (every frame still has a spatial TP)...
        det = detection_accuracy(frame_results)
        self.assertAlmostEqual(det.det_a, 1.0)
        # ...but AssA is reduced: p-1 and p-2 each associate correctly only half the time.
        ass = association_accuracy(frame_results)
        self.assertAlmostEqual(ass.ass_a, 0.5)

        identity = identity_metrics(frame_results)
        # Global best mapping picks one of {p-1, p-2} to match gt-1 for 2/4 occurrences.
        self.assertEqual(identity.id_true_positives, 2)
        self.assertAlmostEqual(identity.idf1, 2 * 2 / (2 * 2 + 2 + 2))


class FragmentationAndRecoveryTests(unittest.TestCase):
    def test_fragmentation_and_successful_recovery_with_same_id(self) -> None:
        refs = [_obs("gt-1", frame, 0, 0) for frame in range(5)]
        # Frame 2 is a total miss (occlusion); the SAME predicted ID resumes after.
        preds = [
            _obs("p-1", 0, 0, 0),
            _obs("p-1", 1, 0, 0),
            _obs("p-1", 3, 0, 0),
            _obs("p-1", 4, 0, 0),
        ]
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        report = track_quality_reports(frame_results)[0]
        self.assertEqual(report.fragmentations, 1)
        self.assertEqual(report.total_gaps, 1)
        self.assertEqual(report.recovered_after_gap, 1)
        self.assertEqual(report.id_switches, 0)  # same ID before/after the gap: not a switch

    def test_fragmentation_with_failed_recovery_counts_as_id_switch_too(self) -> None:
        refs = [_obs("gt-1", frame, 0, 0) for frame in range(5)]
        preds = [
            _obs("p-1", 0, 0, 0),
            _obs("p-1", 1, 0, 0),
            _obs("p-2", 3, 0, 0),  # different ID resumes after the gap
            _obs("p-2", 4, 0, 0),
        ]
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        report = track_quality_reports(frame_results)[0]
        self.assertEqual(report.fragmentations, 1)
        self.assertEqual(report.recovered_after_gap, 0)
        self.assertEqual(report.id_switches, 1)

    def test_coverage_reflects_span_not_just_matched_count(self) -> None:
        refs = [_obs("gt-1", frame, 0, 0) for frame in range(5)]
        preds = [_obs("p-1", 0, 0, 0), _obs("p-1", 4, 0, 0)]  # only endpoints matched
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        report = track_quality_reports(frame_results)[0]
        self.assertEqual(report.span_frames, 5)
        self.assertEqual(report.matched_frames, 2)
        self.assertAlmostEqual(report.coverage, 2 / 5)


class HotaSummaryTests(unittest.TestCase):
    def test_summary_is_one_for_perfect_tracking_across_all_thresholds(self) -> None:
        refs = [_obs("gt-1", frame, 0, 0) for frame in range(3)]
        preds = [_obs("p-1", frame, 0, 0) for frame in range(3)]
        summary = hota_summary(preds, refs)
        self.assertEqual(len(summary.thresholds), 19)
        self.assertAlmostEqual(summary.det_a, 1.0)
        self.assertAlmostEqual(summary.ass_a, 1.0)
        self.assertAlmostEqual(summary.hota, 1.0)

    def test_summary_is_none_with_no_ground_truth(self) -> None:
        summary = hota_summary([], [])
        self.assertIsNone(summary.det_a)
        self.assertIsNone(summary.ass_a)
        self.assertIsNone(summary.hota)


class IdentityMetricsLimitTests(unittest.TestCase):
    def test_raises_when_identity_count_exceeds_cap(self) -> None:
        refs = [_obs(f"gt-{i}", 0, i * 20, 0) for i in range(8)]
        preds = [_obs(f"p-{i}", 0, i * 20, 0) for i in range(8)]
        frame_results = match_all_frames(preds, refs, iou_threshold=0.5)
        with self.assertRaises(ValueError):
            identity_metrics(frame_results, max_identities=6)
