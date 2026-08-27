import unittest
from pathlib import Path

from pitchstate.datasets.manifest import ManifestError, load_manifest


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_fixture_is_valid(self) -> None:
        manifest = load_manifest(ROOT / "data" / "manifests" / "example.json")
        self.assertEqual(manifest.dataset_id, "soccernet-gsr")
        self.assertEqual(len(manifest.splits), 3)

    def test_duplicate_splits_are_rejected(self) -> None:
        path = ROOT / "tests" / "_invalid-manifest.json"
        path.write_text(
            '{"schema_version":"0.1","dataset_id":"x","dataset_version":"1",'
            '"source_url":"https://example.com","license_or_access":"unknown",'
            '"status":"test","local_root":"data/raw/x","splits":['
            '{"name":"train","partition":"a","games":[]},'
            '{"name":"train","partition":"b","games":[]}]}',
            encoding="utf-8",
        )
        try:
            with self.assertRaises(ManifestError):
                load_manifest(path)
        finally:
            path.unlink()