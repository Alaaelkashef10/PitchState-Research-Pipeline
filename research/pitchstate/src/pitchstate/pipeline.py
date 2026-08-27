"""Dependency-injected orchestration for the PitchState state pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from pitchstate.calibration.interface import Calibrator
from pitchstate.classification.interface import TeamRoleClassifier
from pitchstate.config import ProjectConfig
from pitchstate.detection.interface import Detector
from pitchstate.logging_utils import ExperimentLogger
from pitchstate.schema import (
    Frame,
    MatchState,
    PipelineResult,
    TrackObservation,
)
from pitchstate.tactics.interface import TacticalAnalyzer
from pitchstate.tracking.interface import Tracker


def run_pipeline(
    frames: Sequence[Frame],
    detector: Detector,
    tracker: Tracker,
    classifier: TeamRoleClassifier,
    calibrator: Calibrator,
    analyzer: TacticalAnalyzer,
    config: ProjectConfig,
    run_id: str,
    logger: ExperimentLogger | None = None,
) -> PipelineResult:
    """Run one shot-local pipeline and preserve invalid states explicitly."""

    states: list[MatchState] = []
    current_shot_id = "shot-000"
    for frame in frames:
        detections = detector.detect(frame)
        observations = list(tracker.update(frame, detections, current_shot_id))
        classified: list[TrackObservation] = []
        for observation in observations:
            prediction = classifier.classify(observation)
            classified.append(
                replace(
                    observation,
                    team=prediction.team,
                    role=prediction.role,
                    team_confidence=prediction.team_confidence,
                    role_confidence=prediction.role_confidence,
                )
            )

        calibration = calibrator.estimate(frame, classified, current_shot_id)
        projected: list[TrackObservation] = []
        for observation in classified:
            point = observation.detection.bounding_box.footpoint
            pitch_point = (
                calibration.project(point, frame.width, frame.height)
                if calibration.valid
                else None
            )
            projected.append(replace(observation, pitch_point=pitch_point))

        reasons: list[str] = []
        if not calibration.valid:
            reasons.append("CALIBRATION_INVALID")
        if calibration.confidence < config.quality.minimum_calibration_confidence:
            reasons.append("CALIBRATION_LOW_CONFIDENCE")
        if calibration.reprojection_error > config.quality.maximum_reprojection_error:
            reasons.append("CALIBRATION_REPROJECTION_ERROR")

        eligible = [
            observation
            for observation in projected
            if observation.role in {"player", "goalkeeper"}
            and observation.team in {"team_a", "team_b"}
            and observation.confidence >= config.quality.minimum_player_confidence
            and observation.pitch_point is not None
        ]
        for team in ("team_a", "team_b"):
            if sum(observation.team == team for observation in eligible) < config.runtime.min_players_per_team:
                reasons.append(f"TOO_FEW_PLAYERS_{team.upper()}")

        valid = not reasons
        shape = tuple(analyzer.analyze(eligible)) if valid else ()
        state = MatchState(
            frame=frame,
            shot_id=current_shot_id,
            calibration=calibration,
            players=tuple(projected),
            valid=valid,
            abstention_reasons=tuple(reasons),
            team_shape=shape,
        )
        states.append(state)
        if logger:
            logger.event(
                "frame_processed",
                {
                    "frame_index": frame.frame_index,
                    "valid": valid,
                    "detection_count": len(detections),
                    "track_count": len(observations),
                    "abstention_reasons": reasons,
                },
            )
    return PipelineResult(schema_version="0.1", run_id=run_id, states=tuple(states))