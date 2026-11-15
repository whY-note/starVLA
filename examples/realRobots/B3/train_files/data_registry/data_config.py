# ruff: noqa: RUF012
"""B3 dual-arm absolute Cartesian commands, preserving the source xyzw layout."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class B3DualEEFDataConfig:
    """Three current images, current proprioception and actions t through t+15.

    Each arm occupies eight dimensions: position_m(3), quaternion_xyzw(4),
    gripper_open_fraction(1). No rotation conversion or gripper thresholding is
    applied. The same ordered fields are used for state and action statistics.
    """

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.front", "video.left_wrist", "video.right_wrist"]
    field_dims = {
        "left_position": 3,
        "left_quaternion": 4,
        "left_gripper": 1,
        "right_position": 3,
        "right_quaternion": 4,
        "right_gripper": 1,
    }
    state_keys = [f"state.{key}" for key in field_dims]
    action_keys = [f"action.{key}" for key in field_dims]
    state_key_dims = {f"state.{key}": dim for key, dim in field_dims.items()}
    action_key_dims = {f"action.{key}": dim for key, dim in field_dims.items()}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "min_max" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "min_max" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {"b3_dual_eef": B3DualEEFDataConfig()}

DATASET_NAMED_MIXTURES = {
    "b3_stack_blocks": [("data_v5_eef_lerobot_v21", 1.0, "b3_dual_eef")],
}
