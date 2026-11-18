import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.swanlab_tracker import SwanLabTrackingMixin


class SwanLabTrackerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trainer = SwanLabTrackingMixin()
        self.trainer.config = wrap_config(
            OmegaConf.create(
                {
                    "output_dir": self.directory.name,
                    "run_id": "b3_test",
                    "swanlab_project": "starVLA_B3",
                    "swanlab_workspace": None,
                    "swanlab_mode": "offline",
                    "trainer": {"learning_rate": {"base": 1e-5}},
                }
            )
        )
        self.trainer.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=MagicMock())
        self.sdk = MagicMock()
        self.env = patch.dict("os.environ", {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        sdk_patch = patch(
            "starVLA.training.trainer_utils.swanlab_tracker.importlib.import_module",
            return_value=self.sdk,
        )
        self.load_sdk = sdk_patch.start()
        self.addCleanup(sdk_patch.stop)

    def test_main_process_records_config_steps_and_finishes(self):
        self.trainer._init_swanlab()
        kwargs = self.sdk.init.call_args.kwargs
        self.assertEqual(kwargs["name"], "b3_test")
        self.assertEqual(kwargs["project"], "starVLA_B3")
        self.assertIsNone(kwargs["workspace"])
        self.assertEqual(kwargs["mode"], "offline")
        self.assertEqual(kwargs["log_dir"], str(Path(self.directory.name) / "swanlog"))
        self.assertEqual(kwargs["config"]["trainer"]["learning_rate"]["base"], 1e-5)
        self.trainer.accelerator.wait_for_everyone.assert_called_once()
        self.trainer._log_swanlab({"loss": 0.5, "learning_rate/base": 1e-5}, step=10)
        self.sdk.log.assert_called_once_with({"loss": 0.5, "learning_rate/base": 1e-5}, step=10)
        self.trainer._finish_swanlab()
        self.trainer._finish_swanlab()
        self.sdk.finish.assert_called_once()

    def test_non_main_process_never_imports_or_logs(self):
        self.trainer.accelerator.is_main_process = False
        self.trainer._init_swanlab()
        self.trainer._log_swanlab({"loss": 0.5}, step=10)
        self.trainer._finish_swanlab()
        self.load_sdk.assert_not_called()
        self.sdk.log.assert_not_called()
        self.sdk.finish.assert_not_called()
        self.trainer.accelerator.wait_for_everyone.assert_called_once()

    def test_disabled_mode_does_not_require_sdk(self):
        with patch.dict("os.environ", {"SWANLAB_MODE": "disabled"}):
            self.trainer._init_swanlab()
        self.load_sdk.assert_not_called()
        self.assertFalse(self.trainer._swanlab_enabled)
        self.trainer.accelerator.wait_for_everyone.assert_called_once()

    def test_environment_mode_overrides_yaml(self):
        with patch.dict("os.environ", {"SWANLAB_MODE": "online"}):
            self.trainer._init_swanlab()
        self.assertEqual(self.sdk.init.call_args.kwargs["mode"], "online")

    def test_init_failure_still_releases_other_ranks(self):
        self.sdk.init.side_effect = RuntimeError("unavailable")
        with self.assertLogs(level="WARNING"):
            self.trainer._init_swanlab()
        self.assertFalse(self.trainer._swanlab_enabled)
        self.trainer.accelerator.wait_for_everyone.assert_called_once()
        self.trainer._log_swanlab({"loss": 1}, step=1)
        self.sdk.log.assert_not_called()

    def test_missing_sdk_still_releases_other_ranks(self):
        self.load_sdk.side_effect = ModuleNotFoundError("swanlab")
        with self.assertLogs(level="WARNING"):
            self.trainer._init_swanlab()
        self.assertFalse(self.trainer._swanlab_enabled)
        self.trainer.accelerator.wait_for_everyone.assert_called_once()

    def test_log_failure_disables_retries_and_still_finishes_run(self):
        self.trainer._init_swanlab()
        self.sdk.log.side_effect = RuntimeError("disk full")
        with self.assertLogs(level="WARNING"):
            self.trainer._log_swanlab({"loss": 1}, step=1)
        self.trainer._log_swanlab({"loss": 0.5}, step=2)
        self.sdk.log.assert_called_once()
        self.trainer._finish_swanlab()
        self.sdk.finish.assert_called_once()

    def test_finish_failure_is_nonfatal_and_not_retried(self):
        self.trainer._init_swanlab()
        self.sdk.finish.side_effect = RuntimeError("unavailable")
        with self.assertLogs(level="WARNING"):
            self.trainer._finish_swanlab()
        self.trainer._finish_swanlab()
        self.sdk.finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
