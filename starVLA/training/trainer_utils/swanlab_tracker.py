"""SwanLab lifecycle shared by the VLA, co-training and VLM trainers."""

import importlib
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class SwanLabTrackingMixin:
    """Record on the global main process and keep tracker failures nonfatal."""

    def _init_swanlab(self):
        self._swanlab_enabled = False
        self._swanlab_run_active = False
        self._swanlab_sdk = None
        mode = os.environ.get("SWANLAB_MODE", getattr(self.config, "swanlab_mode", "online"))
        try:
            if self.accelerator.is_main_process and mode != "disabled":
                sdk = importlib.import_module("swanlab")
                self._swanlab_sdk = sdk
                logdir = Path(self.config.output_dir) / "swanlog"
                logdir.mkdir(parents=True, exist_ok=True)
                config = self.config.unwrap() if hasattr(self.config, "unwrap") else self.config
                config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else vars(config)
                sdk.init(
                    name=self.config.run_id,
                    project=getattr(self.config, "swanlab_project", "starVLA"),
                    workspace=getattr(self.config, "swanlab_workspace", None),
                    log_dir=str(logdir),
                    mode=mode,
                    config=config,
                    group="vla-train",
                )
                self._swanlab_enabled = True
                self._swanlab_run_active = True
        except Exception as exc:
            logger.warning("SwanLab init failed; continuing without experiment tracking: %s", exc)
        finally:
            # All ranks rendezvous even when rank zero disables logging or fails
            # initialization, before another rank starts the next collective.
            self.accelerator.wait_for_everyone()

    def _log_swanlab(self, metrics, step):
        if not self.accelerator.is_main_process or not getattr(self, "_swanlab_enabled", False):
            return
        try:
            self._swanlab_sdk.log(metrics, step=step)
        except Exception as exc:
            self._swanlab_enabled = False
            logger.warning("SwanLab log failed; disabling further metric logging: %s", exc)

    def _finish_swanlab(self):
        # A logging error must not prevent the active run from being closed.
        if not self.accelerator.is_main_process or not getattr(self, "_swanlab_run_active", False):
            return
        try:
            self._swanlab_sdk.finish()
        except Exception as exc:
            logger.warning("SwanLab finish failed: %s", exc)
        finally:
            self._swanlab_enabled = False
            self._swanlab_run_active = False
