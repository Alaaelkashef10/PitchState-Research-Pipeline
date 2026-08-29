import tempfile
import unittest
from pathlib import Path

from pitchstate.datasets.dedup import detect_duplicate_source_files, hash_file


class DedupTests(unittest.TestCase):
    def test_identical_content_is_grouped_regardless_of_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "clip_a.mp4"
            b = root / "reencoded_copy.mp4"
            a.write_bytes(b"same-bytes" * 1000)
            b.write_bytes(b"same-bytes" * 1000)
            groups = detect_duplicate_source_files([a, b])
            self.assertEqual(len(groups), 1)
            self.assertEqual(set(groups[0].paths), {a, b})

    def test_different_content_is_not_grouped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "clip_a.mp4"
            b = root / "clip_b.mp4"
            a.write_bytes(b"content-one")
            b.write_bytes(b"content-two")
            groups = detect_duplicate_source_files([a, b])
            self.assertEqual(groups, ())

    def test_missing_paths_are_skipped_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "clip_a.mp4"
            existing.write_bytes(b"content")
            missing = root / "not_downloaded_yet.mp4"
            groups = detect_duplicate_source_files([existing, missing])
            self.assertEqual(groups, ())

    def test_hash_file_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            path.write_bytes(b"deterministic-content")
            self.assertEqual(hash_file(path), hash_file(path))

    def test_three_way_duplicate_forms_single_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"clip_{i}.mp4" for i in range(3)]
            for path in paths:
                path.write_bytes(b"triplicate")
            groups = detect_duplicate_source_files(paths)
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0].paths), 3)
