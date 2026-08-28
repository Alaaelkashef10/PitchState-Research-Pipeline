import unittest
from pathlib import Path

from pitchstate.datasets.manifest import LeakageError, ManifestError, load_manifest


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_fixture_is_valid(self) -> None:
        manifest = load_manifest(ROOT / "data" / "manifests" / "example.json")
        self.assertEqual(manifest.dataset_id, "soccernet-gsr")
        self.assertEqual(len(manifest.splits), 3)
        self.assertEqual(manifest.split_strategy.split(";", 1)[0], "match_level_disjoint")

    def test_duplicate_splits_are_rejected(self) -> None:
        path = ROOT / "tests" / "_invalid-manifest.json"
        path.write_text(
            '{"schema_version":"0.2","dataset_id":"x","dataset_version":"1",'
            '"source_url":"https://example.com","license_or_access":"unknown",'
            '"status":"test","local_root":"data/raw/x",'
            '"source_checked_on":"2026-08-28","release_or_download_date":null,'
            '"clip_structure":{"description":"x","duration_seconds":1,"frame_rate_fps":1,"resolution":"x"},'
            '"annotations":{"players":{"status":"x","fields":[],"notes":""}},'
            '"split_strategy":"match_level_disjoint","splits":['
            '{"name":"train","partition":"a","games":[],"clips":[]},'
            '{"name":"train","partition":"b","games":[],"clips":[]}]}',
            encoding="utf-8",
        )
        try:
            with self.assertRaises(ManifestError):
                load_manifest(path)
        finally:
            path.unlink()

    def test_cross_split_game_leakage_is_rejected(self) -> None:
        path = ROOT / "tests" / "_leaky-manifest.json"
        path.write_text(
            '{"schema_version":"0.2","dataset_id":"x","dataset_version":"1",'
            '"source_url":"https://example.com","license_or_access":"unknown",'
            '"status":"test","local_root":"data/raw/x",'
            '"source_checked_on":"2026-08-28","release_or_download_date":null,'
            '"clip_structure":{"description":"x","duration_seconds":1,"frame_rate_fps":1,"resolution":"x"},'
            '"annotations":{"players":{"status":"x","fields":[],"notes":""}},'
            '"split_strategy":"match_level_disjoint","splits":['
            '{"name":"train","partition":"a","games":["game-1"],"clips":[]},'
            '{"name":"test","partition":"b","games":["game-1"],"clips":[]}]}',
            encoding="utf-8",
        )
        try:
            with self.assertRaises(LeakageError):
                load_manifest(path)
        finally:
            path.unlink()

    def test_duplicate_clip_leakage_is_rejected(self) -> None:
        path = ROOT / "tests" / "_leaky-clips.json"
        path.write_text(
            '{"schema_version":"0.2","dataset_id":"x","dataset_version":"1",'
            '"source_url":"https://example.com","license_or_access":"unknown",'
            '"status":"test","local_root":"data/raw/x",'
            '"source_checked_on":"2026-08-28","release_or_download_date":null,'
            '"clip_structure":{"description":"x","duration_seconds":1,"frame_rate_fps":1,"resolution":"x"},'
            '"annotations":{"players":{"status":"x","fields":[],"notes":""}},'
            '"split_strategy":"match_level_disjoint","splits":['
            '{"name":"train","partition":"a","games":[],"clips":["clip-1"]},'
            '{"name":"test","partition":"b","games":[],"clips":["clip-1"]}]}',
            encoding="utf-8",
        )
        try:
            with self.assertRaises(LeakageError):
                load_manifest(path)
        finally:
            path.unlink()