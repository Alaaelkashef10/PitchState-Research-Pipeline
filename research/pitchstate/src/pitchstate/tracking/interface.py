"""Tracker protocol."""

from __future__ import annotations

from typing import Protocol, Sequence

from pitchstate.schema import Detection, Frame, TrackObservation


class Tracker(Protocol):
    def update(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        shot_id: str,
    ) -> Sequence[TrackObservation]:
        """Associate detections with shot-local tracks."""