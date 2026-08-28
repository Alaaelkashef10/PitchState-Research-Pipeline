import unittest
from pathlib import Path

from pitchstate.config import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_smoke_config_loads(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        self.assertEqual(config.project_name, "pitchstate")
        self.assertEqual(config.runtime.min_players_per_team, 2)
        self.assertEqual(config.quality.minimum_calibration_confidence, 0.75)
        self.assertEqual(config.quality.minimum_team_confidence, 0.75)

    def test_invalid_quality_threshold_is_rejected(self) -> None:
        path = ROOT / "tests" / "_invalid.toml"
        path.write_text(
            'schema_version = "0.1"\nproject_name = "x"\nexperiment_name = "y"\nseed = 1\n'
            '[runtime]\nframe_width = 1\nframe_height = 1\nsample_fps = 1\nmin_players_per_team = 1\n'
            '[quality]\nminimum_calibration_confidence = 2\nmaximum_reprojection_error = 0.1\nminimum_player_confidence = 0.1\nminimum_team_confidence = 0.1\nminimum_role_confidence = 0.1\n'
            '[logging]\nrun_directory = "runs"\nlevel = "INFO"\n',
            encoding="utf-8",
        )
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            path.unlink()