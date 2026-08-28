import json
import unittest
from pathlib import Path

from pitchstate.calibration.interface import Calibrator
from pitchstate.config import load_config
from pitchstate.pipeline import run_pipeline
from pitchstate.reproducibility import make_run_id
from pitchstate.schema import CalibrationState
from pitchstate.smoke import (
    SyntheticClassifier,
    SyntheticCalibrator,
    SyntheticDetector,
    SyntheticShapeAnalyzer,
    SyntheticTracker,
    run_smoke,
    synthetic_frames,
)


ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_smoke_is_valid_and_reproducible(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        first = run_smoke(config)
        second = run_smoke(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first.states), 3)
        self.assertTrue(all(state.valid for state in first.states))
        self.assertTrue(all(len(state.team_shape) == 2 for state in first.states))
        expected_id = make_run_id(
            config.project_name,
            config.experiment_name,
            config.to_dict(),
            config.seed,
        )
        self.assertEqual(first.run_id, expected_id)

    def test_serialized_output_is_json(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        payload = run_smoke(config).to_dict()
        encoded = json.dumps(payload)
        self.assertIn('"schema_version": "0.1"', encoded)

    def test_invalid_calibration_abstains(self) -> None:
        class InvalidCalibrator(Calibrator):
            def estimate(self, frame, observations, shot_id):
                del frame, observations
                return CalibrationState(
                    valid=False,
                    confidence=0.1,
                    reprojection_error=0.9,
                    shot_id=shot_id,
                )

        config = load_config(ROOT / "configs" / "smoke.toml")
        result = run_pipeline(
            frames=synthetic_frames(config),
            detector=SyntheticDetector(),
            tracker=SyntheticTracker(),
            classifier=SyntheticClassifier(),
            calibrator=InvalidCalibrator(),
            analyzer=SyntheticShapeAnalyzer(),
            config=config,
            run_id="invalid-calibration",
        )
        self.assertTrue(all(not state.valid for state in result.states))
        self.assertTrue(
            all("CALIBRATION_INVALID" in state.abstention_reasons for state in result.states)
        )

    def test_uncertain_team_assignment_abstains(self) -> None:
        class UncertainClassifier(SyntheticClassifier):
            def classify(self, observation):
                prediction = super().classify(observation)
                return prediction.__class__(
                    prediction.team,
                    prediction.role,
                    team_confidence=0.1,
                    role_confidence=prediction.role_confidence,
                )

        config = load_config(ROOT / "configs" / "smoke.toml")
        result = run_pipeline(
            frames=synthetic_frames(config),
            detector=SyntheticDetector(),
            tracker=SyntheticTracker(),
            classifier=UncertainClassifier(),
            calibrator=SyntheticCalibrator(),
            analyzer=SyntheticShapeAnalyzer(),
            config=config,
            run_id="uncertain-team",
        )
        self.assertTrue(all(not state.valid for state in result.states))
        self.assertTrue(
            all("TOO_FEW_PLAYERS_TEAM_A" in state.abstention_reasons for state in result.states)
        )

    def test_shot_boundary_changes_context_and_resets_tracker(self) -> None:
        class BoundaryDetector:
            def is_boundary(self, previous_frame, frame):
                return frame.frame_index == 1

        class ResettableTracker(SyntheticTracker):
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

        config = load_config(ROOT / "configs" / "smoke.toml")
        tracker = ResettableTracker()
        result = run_pipeline(
            frames=synthetic_frames(config),
            detector=SyntheticDetector(),
            tracker=tracker,
            classifier=SyntheticClassifier(),
            calibrator=SyntheticCalibrator(),
            analyzer=SyntheticShapeAnalyzer(),
            config=config,
            run_id="shot-boundary",
            shot_boundary_detector=BoundaryDetector(),
        )
        self.assertEqual(tracker.reset_count, 1)
        self.assertEqual([state.shot_id for state in result.states], ["shot-000", "shot-001", "shot-001"])
        self.assertFalse(result.states[0].shot_boundary)
        self.assertTrue(result.states[1].shot_boundary)