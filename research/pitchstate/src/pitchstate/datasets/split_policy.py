"""Runtime enforcement of frozen-split access policy.

``datasets.manifest.audit_split_integrity`` checks that a manifest's split
membership is internally consistent (no game or clip crosses splits). It does
not, and cannot, stop *code elsewhere in the pipeline* from later reading a
frozen test-split identity while choosing a decision threshold or searching
hyperparameters. That is a separate, easy-to-make mistake: the manifest can be
perfectly leak-free and an experiment script can still leak by selecting a
threshold that happens to look best on test.

``SplitAccessGuard`` closes that gap by making the frozen identity set an
explicit runtime object that selection/search code must consult. It does not
try to detect leakage by inspecting arbitrary code; it only guarantees that
*if* calling code checks with the guard before using an identity, a frozen
identity used for selection or search raises immediately instead of silently
producing an optimistic number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pitchstate.datasets.manifest import DatasetManifest

#: Purposes that are never permitted against a frozen split identity.
RESTRICTED_PURPOSES = frozenset({"threshold_selection", "hyperparameter_search"})

DEFAULT_FROZEN_SPLIT_NAMES = ("test",)


class TestSplitAccessError(RuntimeError):
    """Raised when a frozen test-split identity is used for a restricted purpose."""

    def __init__(self, identity: str, identity_kind: str, purpose: str, split_name: str) -> None:
        self.identity = identity
        self.identity_kind = identity_kind
        self.purpose = purpose
        self.split_name = split_name
        super().__init__(
            f"{identity_kind} {identity!r} belongs to frozen split {split_name!r} "
            f"and cannot be used for {purpose!r}"
        )


@dataclass(frozen=True)
class SplitAccessGuard:
    """Enforces that frozen-split identities are never used for selection/search.

    Construct once per manifest via :meth:`from_manifest`, then call
    ``assert_game_allowed`` / ``assert_clip_allowed`` (or the bulk
    ``filter_allowed`` helpers) at every point where code chooses which
    identities to use for threshold selection or hyperparameter search.
    """

    frozen_game_ids: frozenset[str]
    frozen_clip_ids: frozenset[str]
    frozen_split_names: tuple[str, ...]

    @classmethod
    def from_manifest(
        cls,
        manifest: DatasetManifest,
        frozen_split_names: Iterable[str] = DEFAULT_FROZEN_SPLIT_NAMES,
    ) -> "SplitAccessGuard":
        names = tuple(frozen_split_names)
        games: set[str] = set()
        clips: set[str] = set()
        owning_split: dict[str, str] = {}
        for split in manifest.splits:
            if split.name not in names:
                continue
            games.update(split.games)
            clips.update(split.clips)
            for identity in (*split.games, *split.clips):
                owning_split[identity] = split.name
        guard = cls(frozenset(games), frozenset(clips), names)
        object.__setattr__(guard, "_owning_split", owning_split)
        return guard

    def _split_name_for(self, identity: str) -> str:
        owning: dict[str, str] = getattr(self, "_owning_split", {})
        return owning.get(identity, self.frozen_split_names[0] if self.frozen_split_names else "test")

    def _assert_allowed(self, identity: str, identity_kind: str, frozen_ids: frozenset[str], *, purpose: str) -> None:
        if purpose not in RESTRICTED_PURPOSES:
            raise ValueError(
                f"purpose must be one of {sorted(RESTRICTED_PURPOSES)}, got {purpose!r}"
            )
        if identity in frozen_ids:
            raise TestSplitAccessError(identity, identity_kind, purpose, self._split_name_for(identity))

    def assert_game_allowed(self, game_id: str, *, purpose: str) -> None:
        """Raise :class:`TestSplitAccessError` if ``game_id`` is frozen for ``purpose``."""

        self._assert_allowed(game_id, "game", self.frozen_game_ids, purpose=purpose)

    def assert_clip_allowed(self, clip_id: str, *, purpose: str) -> None:
        """Raise :class:`TestSplitAccessError` if ``clip_id`` is frozen for ``purpose``."""

        self._assert_allowed(clip_id, "clip", self.frozen_clip_ids, purpose=purpose)

    def filter_allowed_games(self, game_ids: Iterable[str], *, purpose: str) -> tuple[str, ...]:
        """Return ``game_ids`` unchanged, or raise on the first frozen identity.

        This is a fail-loud helper for call sites that build a candidate set
        for threshold selection or hyperparameter search: it never silently
        drops the frozen identity, because a silent drop could mask a bug
        that included it in the first place.
        """

        for game_id in game_ids:
            self.assert_game_allowed(game_id, purpose=purpose)
        return tuple(game_ids)

    def filter_allowed_clips(self, clip_ids: Iterable[str], *, purpose: str) -> tuple[str, ...]:
        for clip_id in clip_ids:
            self.assert_clip_allowed(clip_id, purpose=purpose)
        return tuple(clip_ids)

    def is_frozen_game(self, game_id: str) -> bool:
        return game_id in self.frozen_game_ids

    def is_frozen_clip(self, clip_id: str) -> bool:
        return clip_id in self.frozen_clip_ids
