"""Evaluation protocol kept model-agnostic in Phase 0."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class Evaluator(Protocol):
    def evaluate(self, predictions: Sequence[Any], references: Sequence[Any]) -> dict[str, float]:
        """Return named metrics with their evaluation provenance recorded upstream."""