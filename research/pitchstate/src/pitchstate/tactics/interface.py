"""Tactical-analysis protocol."""

from __future__ import annotations

from typing import Protocol, Sequence

from pitchstate.schema import ShapeMetrics, TrackObservation


class TacticalAnalyzer(Protocol):
    def analyze(self, observations: Sequence[TrackObservation]) -> Sequence[ShapeMetrics]:
        """Compute descriptive metrics from valid pitch-space observations."""