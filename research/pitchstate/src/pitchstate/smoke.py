"""Deterministic synthetic components used only to verify Phase 0 wiring."""

from __future__ import annotations

from typing import Sequence

from pitchstate.calibration.interface import Calibrator
from pitchstate.classification.interface import TeamRoleClassifier
from pitchstate.config import ProjectConfig
from pitchstate.detection.interface import Detector
from pitchstate.pipeline import run_pipeline
from pitchstate.reproducibility import code_revision, environment_metadata, make_run_id, seed_everything
from pitchstate.schema import (
    BoundingBox,
    CalibrationState,
    Detection,
    Frame,
    TeamRolePrediction,
    TrackObservation,
)
from pitchstate.tactics.shape import calculate_shape_metrics
from pitchstate.tracking.interface import Tracker


class SyntheticDetector:
    def detect(self, frame: Frame) -> Sequence[Detection]:
        del frame
        positions = (
            (0, 20, 30),
            (1, 36, 44),
            (2, 52, 36),
            (3, 20, 70),
            (4, 36, 84),
            (5, 52, 76),
        )
        return tuple(
            Detection(
                detection_id=f"detection-{index}",
                category="player",
                bounding_box=BoundingBox(x, y, 8, 12),
                confidence=0.98,
            )
            for index, x, y in positions
        )


class SyntheticTracker:
    def update(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        shot_id: str,
    ) -> Sequence[TrackObservation]:
        del shot_id
        return tuple(
            TrackObservation(
                track_id=detection.detection_id,
                detection=detection,
                frame_index=frame.frame_index,
                confidence=detection.confidence,
            )
            for detection in detections
        )


class SyntheticClassifier(TeamRoleClassifier):
    def classify(self, observation: TrackObservation) -> TeamRolePrediction:
        index = int(observation.track_id.rsplit("-", 1)[-1])
        team = "team_a" if index < 3 else "team_b"
        return TeamRolePrediction(team, "player", 0.99, 0.99)


class SyntheticCalibrator(Calibrator):
    def estimate(
        self,
        frame: Frame,
        observations: Sequence[TrackObservation],
        shot_id: str,
    ) -> CalibrationState:
        del frame, observations
        return CalibrationState(
            valid=True,
            confidence=0.99,
            reprojection_error=0.01,
            shot_id=shot_id,
        )


class SyntheticShapeAnalyzer:
    def analyze(self, observations: Sequence[TrackObservation]):
        return tuple(
            calculate_shape_metrics(
                team,
                [observation for observation in observations if observation.team == team],
            )
            for team in ("team_a", "team_b")
        )


def synthetic_frames(config: ProjectConfig) -> tuple[Frame, ...]:
    return tuple(
        Frame(
            frame_index=index,
            timestamp_seconds=index / config.runtime.sample_fps,
            width=config.runtime.frame_width,
            height=config.runtime.frame_height,
        )
        for index in range(3)
    )


def run_smoke(config: ProjectConfig):
    seed_everything(config.seed)
    run_id = make_run_id(
        config.project_name,
        config.experiment_name,
        config.to_dict(),
        config.seed,
    )
    return run_pipeline(
        frames=synthetic_frames(config),
        detector=SyntheticDetector(),
        tracker=SyntheticTracker(),
        classifier=SyntheticClassifier(),
        calibrator=SyntheticCalibrator(),
        analyzer=SyntheticShapeAnalyzer(),
        config=config,
        run_id=run_id,
        metadata={
            "project_name": config.project_name,
            "experiment_name": config.experiment_name,
            "seed": config.seed,
            "config": config.to_dict(),
            "dataset_id": "synthetic-phase0",
            "dataset_version": "synthetic-v1",
            "model_version": "synthetic-components",
            "code_revision": code_revision(),
            "environment": environment_metadata(),
        },
    )