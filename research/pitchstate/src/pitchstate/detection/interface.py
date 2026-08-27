"""Detector protocol."""

from __future__ import annotations

from typing import Protocol, Sequence

from pitchstate.schema import Detection, Frame


class Detector(Protocol):
    def detect(self, frame: Frame) -> Sequence[Detection]:
        """Return object detections for one frame."""