"""Calibration and projection protocols."""

from __future__ import annotations

from typing import Protocol, Sequence

from pitchstate.schema import CalibrationState, Frame, TrackObservation


class Calibrator(Protocol):
    def estimate(
        self,
        frame: Frame,
        observations: Sequence[TrackObservation],
        shot_id: str,
    ) -> CalibrationState:
        """Estimate a shot-local field registration state."""