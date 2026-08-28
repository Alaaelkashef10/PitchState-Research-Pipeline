"""Shot-boundary detection protocol.

Shot-local continuity is an explicit safety boundary. Implementations may use
visual change detection, broadcast metadata, or a learned classifier later.
"""

from __future__ import annotations

from typing import Protocol

from pitchstate.schema import Frame


class ShotBoundaryDetector(Protocol):
    def is_boundary(self, previous_frame: Frame, frame: Frame) -> bool:
        """Return whether ``frame`` starts a new shot or replay segment."""