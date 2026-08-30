"""Calibration and projection protocols.

The estimation math itself lives in ``calibration.homography`` (planar
homography via DLT, pure Python, no numpy dependency). That module is not yet
wired into the :class:`Calibrator` protocol below: doing so requires a
concrete source of pitch-keypoint correspondences (a keypoint detector run
against real broadcast frames), which does not exist in this repository yet
and is blocked pending authorized SoccerNet access (see
``docs/dataset-audit.md``). See ``calibration/homography.py``'s module
docstring for the full scope and limitations of what has been validated so
far — synthetic, hand-constructed correspondences only, no real-footage
accuracy claim.
"""

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
