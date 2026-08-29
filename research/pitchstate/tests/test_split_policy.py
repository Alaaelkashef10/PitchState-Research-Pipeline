import unittest

from pitchstate.datasets.manifest import DatasetManifest, DatasetSplit
from pitchstate.datasets.split_policy import SplitAccessGuard, TestSplitAccessError


def _manifest(splits: tuple[DatasetSplit, ...]) -> DatasetManifest:
    return DatasetManifest(
        schema_version="0.3",
        dataset_id="x",
        dataset_version="1",
        source_url="https://example.com",
        license_or_access="unknown",
        status="test",
        local_root="data/raw/x",
        source_checked_on="2026-08-28",
        release_or_download_date=None,
        clip_structure={"description": "x", "duration_seconds": 1, "frame_rate_fps": 1, "resolution": "x"},
        annotations={},
        split_strategy="match_level_disjoint",
        splits=splits,
        notes=(),
        validation_status="not_locally_verified",
    )


class SplitAccessGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest(
            (
                DatasetSplit(name="train", partition="official", games=("game-1", "game-2"), clips=("clip-1",)),
                DatasetSplit(name="validation", partition="official", games=("game-3",), clips=("clip-2",)),
                DatasetSplit(name="test", partition="official", games=("game-4", "game-5"), clips=("clip-3",)),
            )
        )
        self.guard = SplitAccessGuard.from_manifest(self.manifest)

    def test_train_game_allowed_for_threshold_selection(self) -> None:
        self.guard.assert_game_allowed("game-1", purpose="threshold_selection")  # should not raise

    def test_validation_game_allowed_for_hyperparameter_search(self) -> None:
        self.guard.assert_game_allowed("game-3", purpose="hyperparameter_search")  # should not raise

    def test_frozen_test_game_blocked_for_threshold_selection(self) -> None:
        with self.assertRaises(TestSplitAccessError) as ctx:
            self.guard.assert_game_allowed("game-4", purpose="threshold_selection")
        self.assertEqual(ctx.exception.identity, "game-4")
        self.assertEqual(ctx.exception.split_name, "test")

    def test_frozen_test_clip_blocked_for_hyperparameter_search(self) -> None:
        with self.assertRaises(TestSplitAccessError):
            self.guard.assert_clip_allowed("clip-3", purpose="hyperparameter_search")

    def test_filter_allowed_games_raises_on_first_frozen_identity(self) -> None:
        with self.assertRaises(TestSplitAccessError):
            self.guard.filter_allowed_games(["game-1", "game-4"], purpose="threshold_selection")

    def test_filter_allowed_games_passes_through_non_frozen_identities(self) -> None:
        result = self.guard.filter_allowed_games(["game-1", "game-2"], purpose="threshold_selection")
        self.assertEqual(result, ("game-1", "game-2"))

    def test_is_frozen_game_and_clip_helpers(self) -> None:
        self.assertTrue(self.guard.is_frozen_game("game-4"))
        self.assertFalse(self.guard.is_frozen_game("game-1"))
        self.assertTrue(self.guard.is_frozen_clip("clip-3"))
        self.assertFalse(self.guard.is_frozen_clip("clip-1"))

    def test_invalid_purpose_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.guard.assert_game_allowed("game-1", purpose="reporting")

    def test_custom_frozen_split_names_are_respected(self) -> None:
        guard = SplitAccessGuard.from_manifest(self.manifest, frozen_split_names=("validation", "test"))
        with self.assertRaises(TestSplitAccessError):
            guard.assert_game_allowed("game-3", purpose="threshold_selection")
        guard.assert_game_allowed("game-1", purpose="threshold_selection")  # train still allowed
