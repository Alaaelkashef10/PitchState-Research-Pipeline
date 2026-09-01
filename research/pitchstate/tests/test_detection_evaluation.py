import unittest

from pitchstate.evaluation.detection import (
    VisibilityTaggedDetection,
    average_precision,
    average_precision_by_visibility,
    bounding_box_iou,
    match_detections_single_category,
    mean_average_precision,
    precision_recall_at_threshold,
)
from pitchstate.schema import BoundingBox, Detection


def _det(det_id: str, category: str, x: float, y: float, w: float, h: float, confidence: float = 1.0) -> Detection:
    return Detection(detection_id=det_id, category=category, bounding_box=BoundingBox(x, y, w, h), confidence=confidence)


class BoundingBoxIouTests(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self) -> None:
        box = BoundingBox(0, 0, 10, 10)
        self.assertAlmostEqual(bounding_box_iou(box, box), 1.0)

    def test_disjoint_boxes_have_iou_zero(self) -> None:
        a = BoundingBox(0, 0, 5, 5)
        b = BoundingBox(100, 100, 5, 5)
        self.assertAlmostEqual(bounding_box_iou(a, b), 0.0)

    def test_partial_overlap_known_fraction(self) -> None:
        # a: [0,10]x[0,10] area 100; b: [5,15]x[0,10] area 100
        # intersection: [5,10]x[0,10] area 50; union = 100+100-50=150
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 0, 10, 10)
        self.assertAlmostEqual(bounding_box_iou(a, b), 50 / 150)


class MatchDetectionsSingleCategoryTests(unittest.TestCase):
    def test_higher_confidence_prediction_gets_first_pick(self) -> None:
        gt = _det("gt", "player", 0, 0, 10, 10)
        low_conf_overlap = _det("low", "player", 0, 0, 10, 10, confidence=0.4)
        high_conf_overlap = _det("high", "player", 0, 0, 10, 10, confidence=0.9)
        outcomes = match_detections_single_category(
            [low_conf_overlap, high_conf_overlap], [gt], iou_threshold=0.5
        )
        # processed in confidence order: high first (matches gt), low second (no gt left)
        self.assertEqual(outcomes[0].prediction.detection_id, "high")
        self.assertTrue(outcomes[0].is_true_positive)
        self.assertEqual(outcomes[1].prediction.detection_id, "low")
        self.assertFalse(outcomes[1].is_true_positive)

    def test_below_threshold_iou_is_not_a_match(self) -> None:
        gt = _det("gt", "player", 0, 0, 10, 10)
        # shift so IoU is just under 0.5
        pred = _det("p", "player", 6, 0, 10, 10, confidence=0.9)
        outcomes = match_detections_single_category([pred], [gt], iou_threshold=0.5)
        self.assertFalse(outcomes[0].is_true_positive)


class AveragePrecisionTests(unittest.TestCase):
    def test_hand_computed_example(self) -> None:
        gt1 = _det("gt1", "player", 0, 0, 10, 10)
        gt2 = _det("gt2", "player", 100, 100, 10, 10)
        matching_pred = _det("p1", "player", 0, 0, 10, 10, confidence=0.9)
        non_matching_pred = _det("p2", "player", 500, 500, 10, 10, confidence=0.8)
        ap = average_precision(
            [matching_pred, non_matching_pred], [gt1, gt2], category="player", iou_threshold=0.5
        )
        self.assertAlmostEqual(ap, 0.5)

    def test_perfect_detector_has_ap_one(self) -> None:
        gt1 = _det("gt1", "player", 0, 0, 10, 10)
        gt2 = _det("gt2", "player", 100, 100, 10, 10)
        p1 = _det("p1", "player", 0, 0, 10, 10, confidence=0.95)
        p2 = _det("p2", "player", 100, 100, 10, 10, confidence=0.9)
        ap = average_precision([p1, p2], [gt1, gt2], category="player", iou_threshold=0.5)
        self.assertAlmostEqual(ap, 1.0)

    def test_returns_none_when_no_ground_truth(self) -> None:
        pred = _det("p1", "player", 0, 0, 10, 10)
        ap = average_precision([pred], [], category="player", iou_threshold=0.5)
        self.assertIsNone(ap)

    def test_returns_zero_when_ground_truth_but_no_predictions(self) -> None:
        gt = _det("gt", "player", 0, 0, 10, 10)
        ap = average_precision([], [gt], category="player", iou_threshold=0.5)
        self.assertAlmostEqual(ap, 0.0)


class MeanAveragePrecisionTests(unittest.TestCase):
    def test_averages_only_defined_categories(self) -> None:
        gt_player = _det("gt1", "player", 0, 0, 10, 10)
        pred_player = _det("p1", "player", 0, 0, 10, 10, confidence=0.9)
        gt_referee = _det("gt2", "referee", 50, 50, 10, 10)
        # No prediction at all for referee -> AP 0.0 for referee, player AP 1.0
        result = mean_average_precision(
            [pred_player], [gt_player, gt_referee], iou_threshold=0.5
        )
        self.assertAlmostEqual(result["player"], 1.0)
        self.assertAlmostEqual(result["referee"], 0.0)
        self.assertAlmostEqual(result["mAP"], 0.5)

    def test_map_is_none_when_no_categories_have_ground_truth(self) -> None:
        result = mean_average_precision([], [], iou_threshold=0.5, categories=["player"])
        self.assertIsNone(result["player"])
        self.assertIsNone(result["mAP"])


class PrecisionRecallAtThresholdTests(unittest.TestCase):
    def test_basic_counts(self) -> None:
        gt1 = _det("gt1", "player", 0, 0, 10, 10)
        gt2 = _det("gt2", "player", 100, 100, 10, 10)
        matched = _det("p1", "player", 0, 0, 10, 10, confidence=0.9)
        unmatched = _det("p2", "player", 500, 500, 10, 10, confidence=0.8)
        point = precision_recall_at_threshold(
            [matched, unmatched], [gt1, gt2], category="player", iou_threshold=0.5, confidence_threshold=0.5
        )
        self.assertEqual(point.true_positives, 1)
        self.assertEqual(point.false_positives, 1)
        self.assertEqual(point.false_negatives, 1)
        self.assertAlmostEqual(point.precision, 0.5)
        self.assertAlmostEqual(point.recall, 0.5)

    def test_none_precision_when_nothing_clears_threshold(self) -> None:
        gt = _det("gt", "player", 0, 0, 10, 10)
        low_conf = _det("p", "player", 0, 0, 10, 10, confidence=0.2)
        point = precision_recall_at_threshold(
            [low_conf], [gt], category="player", iou_threshold=0.5, confidence_threshold=0.5
        )
        self.assertIsNone(point.precision)
        self.assertAlmostEqual(point.recall, 0.0)


class AveragePrecisionByVisibilityTests(unittest.TestCase):
    def test_match_on_out_of_subset_reference_is_ignored_not_penalized(self) -> None:
        visible_gt = VisibilityTaggedDetection(_det("gt1", "player", 0, 0, 10, 10), visibility="visible")
        occluded_gt = VisibilityTaggedDetection(
            _det("gt2", "player", 100, 100, 10, 10), visibility="occluded"
        )
        # This prediction correctly finds the OCCLUDED player, not the visible one.
        pred = _det("p1", "player", 100, 100, 10, 10, confidence=0.9)
        ap = average_precision_by_visibility(
            [pred], [visible_gt, occluded_gt], category="player", visibility="visible", iou_threshold=0.5
        )
        # The prediction is ignored for the "visible" subset (not a false positive),
        # and the one visible GT was never found -> AP should be 0.0, not penalized further.
        self.assertAlmostEqual(ap, 0.0)

    def test_correct_subset_match_scores_normally(self) -> None:
        visible_gt = VisibilityTaggedDetection(_det("gt1", "player", 0, 0, 10, 10), visibility="visible")
        pred = _det("p1", "player", 0, 0, 10, 10, confidence=0.9)
        ap = average_precision_by_visibility(
            [pred], [visible_gt], category="player", visibility="visible", iou_threshold=0.5
        )
        self.assertAlmostEqual(ap, 1.0)

    def test_returns_none_when_no_reference_in_subset(self) -> None:
        occluded_gt = VisibilityTaggedDetection(
            _det("gt1", "player", 0, 0, 10, 10), visibility="occluded"
        )
        pred = _det("p1", "player", 0, 0, 10, 10, confidence=0.9)
        ap = average_precision_by_visibility(
            [pred], [occluded_gt], category="player", visibility="visible", iou_threshold=0.5
        )
        self.assertIsNone(ap)
