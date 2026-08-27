"""Team and role classification protocol."""

from __future__ import annotations

from typing import Protocol

from pitchstate.schema import TeamRolePrediction, TrackObservation


class TeamRoleClassifier(Protocol):
    def classify(self, observation: TrackObservation) -> TeamRolePrediction:
        """Return independent team and role predictions."""