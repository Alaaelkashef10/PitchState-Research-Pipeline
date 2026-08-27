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