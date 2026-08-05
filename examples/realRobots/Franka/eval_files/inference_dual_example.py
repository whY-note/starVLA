#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StarVLA dual-arm Franka inference example.

This script connects a 14-D StarVLA policy server to the dual-arm Franka
stack used in the local `gello_franka` project:

    policy action = [left  6-D body-frame Cartesian delta, left  gripper,
                     right 6-D body-frame Cartesian delta, right gripper]

Each 7-D arm action is converted to a joint target with FR3 differential IK
and sent to that arm's Gello bridge server. The bridge server is the component
that talks to fairo/polymetis and streams joint + gripper commands to the real
Franka controller.

Prerequisites:
    1. Start both bridge servers from gello_franka, for example:
           bash /data1t/haowenYan/gello_franka/experiments/start_dual_arm.sh
    2. Start the StarVLA policy WebSocket server with a dual-arm checkpoint:
           bash examples/realRobots/Franka/eval_files/run_policy_server.sh
    3. Set ACTION_STATS_PATH to the dual-arm dataset_statistics.json.

The module is intentionally import-safe: hardware libraries such as cv2, zmq,
pybullet, scipy, pynput, and polymetis are imported only when main() creates
the runtime environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


IMG_SIZE = (224, 224)
CONTROL_HZ = float(os.environ.get("CONTROL_HZ", "15"))
CONTROL_PERIOD = 1.0 / CONTROL_HZ
DEFAULT_GRIPPER_MAX_WIDTH = 0.100
REALSENSE_USB_PRODUCTS = {"0b07", "0b5b"}

FRANKA_JOINT_LIMITS = {
    "q_max": [2.8, 1.66, 2.8, -0.17, 2.8, 3.65, 2.8],
    "q_min": [-2.8, -1.66, -2.8, -2.97, -2.8, 0.08, -2.8],
}


# =====================================================================
# Path setup
# =====================================================================
def _add_path_once(path: Path) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _find_starvla_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "deployment" / "model_server" / "tools" / "websocket_policy_client.py").is_file():
            return parent
    return Path.cwd()


def _find_sibling_root(name: str, env_var: str) -> Optional[Path]:
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser()

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return None


def configure_runtime_paths(
    gello_root: Optional[str] = None,
    fairo_root: Optional[str] = None,
) -> Dict[str, Optional[Path]]:
    """Put StarVLA, gello_franka, and fairo/polymetis imports on sys.path."""
    starvla_root = _find_starvla_root()
    _add_path_once(starvla_root)

    gello_path = Path(gello_root).expanduser() if gello_root else _find_sibling_root("gello_franka", "GELLO_FRANKA_ROOT")
    if gello_path is not None:
        _add_path_once(gello_path)
        _add_path_once(gello_path / "experiments")

    fairo_path = Path(fairo_root).expanduser() if fairo_root else _find_sibling_root("fairo", "FAIRO_ROOT")
    if fairo_path is not None:
        _add_path_once(fairo_path / "polymetis" / "polymetis" / "python")
        _add_path_once(fairo_path / "polymetis")

    return {"starvla": starvla_root, "gello": gello_path, "fairo": fairo_path}


def discover_fr3_urdf(gello_root: Optional[Path] = None) -> Path:
    env_value = os.environ.get("FR3_URDF")
    if env_value:
        return Path(env_value).expanduser()
    if gello_root is None:
        gello_root = _find_sibling_root("gello_franka", "GELLO_FRANKA_ROOT")
    if gello_root is None:
        raise FileNotFoundError(
            "Could not find gello_franka. Set GELLO_FRANKA_ROOT or pass --gello_root."
        )
    return (
        gello_root
        / "fr3-urdf-pybullet"
        / "franka_description_pybullet"
        / "robots"
        / "fr3"
        / "fr3_pybullet.urdf"
    )


def make_policy_client(host: str, port: int):
    """Create the StarVLA WebSocket client lazily."""
    configure_runtime_paths()
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    return WebsocketClientPolicy(host=host, port=port)


# =====================================================================
# Action normalization and policy request helpers
# =====================================================================
def load_action_norm_stats(json_path: str, embodiment_key: str = "new_embodiment") -> Dict[str, np.ndarray]:
    """Load action normalization statistics from dataset_statistics.json."""
    if not json_path:
        raise ValueError("action_stats_path is empty. Pass --action_stats_path or set ACTION_STATS_PATH.")

    with open(json_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    if embodiment_key in stats_data:
        stats_data = stats_data[embodiment_key]
    elif "franka" in stats_data:
        stats_data = stats_data["franka"]

    if "action" in stats_data:
        stats_data = stats_data["action"]

    norm_stats = {
        "min": np.asarray(stats_data.get("min", stats_data.get("low", [])), dtype=np.float64),
        "max": np.asarray(stats_data.get("max", stats_data.get("high", [])), dtype=np.float64),
    }
    if "mask" in stats_data:
        norm_stats["mask"] = np.asarray(stats_data["mask"], dtype=bool)
    return norm_stats


def validate_action_norm_stats(
    action_norm_stats: Dict[str, np.ndarray],
    expected_action_dim: int = 14,
) -> Dict[str, np.ndarray]:
    """Validate and coerce action stats to arrays with the expected length."""
    mins = np.asarray(action_norm_stats.get("min", []), dtype=np.float64)
    maxs = np.asarray(action_norm_stats.get("max", []), dtype=np.float64)
    if mins.shape != maxs.shape:
        raise ValueError(f"Action min/max shape mismatch: min={mins.shape}, max={maxs.shape}")
    if mins.ndim != 1 or mins.shape[0] != expected_action_dim:
        raise ValueError(
            f"Expected {expected_action_dim}-D action stats, got min shape {mins.shape}. "
            "Check that ACTION_STATS_PATH points to a dual-arm dataset_statistics.json."
        )

    validated = {"min": mins, "max": maxs}
    if "mask" in action_norm_stats:
        mask = np.asarray(action_norm_stats["mask"], dtype=bool)
        if mask.shape != mins.shape:
            raise ValueError(f"Action mask shape mismatch: mask={mask.shape}, min={mins.shape}")
        validated["mask"] = mask
    return validated


def unnormalize_actions(
    normalized_actions: np.ndarray,
    action_norm_stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """Convert normalized actions in [-1, 1] back to the real action space.

    The Gello dual-arm recorder stores gripper actions as continuous values in
    [0, 1]. For that reason this function does not binarize indices 6 and 13;
    dimensions with mask=False pass through unchanged.
    """
    normalized = np.asarray(normalized_actions, dtype=np.float64).copy()
    if normalized.ndim == 1:
        normalized = normalized.reshape(1, -1)
    if normalized.ndim != 2:
        raise ValueError(f"normalized_actions must be 2-D [T, D], got shape {normalized.shape}")

    stats = validate_action_norm_stats(action_norm_stats, expected_action_dim=normalized.shape[-1])
    mask = stats.get("mask", np.ones_like(stats["min"], dtype=bool))
    clipped = np.clip(normalized, -1.0, 1.0)
    return np.where(mask, 0.5 * (clipped + 1.0) * (stats["max"] - stats["min"]) + stats["min"], clipped)


def build_request(
    images: List[np.ndarray],
    task_instruction: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    example: Dict[str, Any] = {
        "image": images,
        "lang": str(task_instruction),
    }
    if metadata:
        example["metadata"] = metadata
    return {"examples": [example]}


def parse_response(result: dict, expected_action_dim: Optional[int] = None) -> np.ndarray:
    """Parse a policy-server response and return an action chunk [T, D]."""
    data = result.get("data", result)
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(
            f"Policy server error: status={data.get('status')} ok={data.get('ok')} "
            f"error={data.get('error')} keys={list(data.keys())}"
        )

    for key in ("normalized_actions", "actions", "action"):
        if key not in data:
            continue
        actions = np.asarray(data[key])
        if actions.ndim == 3:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise ValueError(f"Action chunk must be 2-D after parsing, got shape {actions.shape}")
        if expected_action_dim is not None and actions.shape[-1] != expected_action_dim:
            raise ValueError(f"Expected action_dim={expected_action_dim}, got shape {actions.shape}")
        return actions

    raise KeyError(f"Could not extract actions from response. Available keys: {list(data.keys())}")


def obs_to_policy_images(obs: Dict[str, Any]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for key in sorted(obs.keys()):
        if key in ("state", "left/state", "right/state"):
            continue
        value = obs[key]
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        image = value
        if image.dtype != np.uint8:
            max_value = float(np.nanmax(image)) if image.size else 1.0
            image = (image * 255).astype(np.uint8) if max_value <= 1.0 else image.astype(np.uint8)
        images.append(image)
    return images


def query_policy_14d(
    client,
    obs: Dict[str, Any],
    task_instruction: str,
    episode_id: int,
    step_id: int,
) -> np.ndarray:
    request = build_request(
        obs_to_policy_images(obs),
        task_instruction,
        metadata={
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "timestamp": time.time(),
        },
    )
    return parse_response(client.predict_action(request), expected_action_dim=14)


def split_dual_action(action: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if action_arr.shape[0] != 14:
        raise ValueError(f"Expected 14-D dual-arm action, got shape {action_arr.shape}")
    return action_arr[:7].copy(), action_arr[7:].copy()


def parse_init_joints(
    combined: str,
    left: str = "",
    right: str = "",
) -> Tuple[List[float], List[float]]:
    """Parse init joints as either 14 combined values or two 7-D strings."""

    def parse_csv(text: str) -> List[float]:
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    if left or right:
        if not left or not right:
            raise ValueError("Provide both --left_init_joints and --right_init_joints, or neither.")
        left_values = parse_csv(left)
        right_values = parse_csv(right)
        if len(left_values) != 7 or len(right_values) != 7:
            raise ValueError(
                "--left_init_joints and --right_init_joints must each contain 7 comma-separated floats, "
                f"got {len(left_values)} and {len(right_values)}."
            )
        return left_values, right_values

    values = parse_csv(combined)
    if len(values) == 14:
        return values[:7], values[7:]
    if len(values) == 7:
        return list(values), list(values)
    raise ValueError(
        "--init_joints must contain 14 comma-separated floats (left|right), "
        f"or 7 floats to reuse the same pose for both arms. Got {len(values)}."
    )


# =====================================================================
# Camera / observation helpers
# =====================================================================
def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(image, size)


def _state_subdict(state: dict) -> dict:
    if "tcp_pose" in state:
        tcp_pose = np.asarray(state["tcp_pose"], dtype=np.float64)
    else:
        tcp_pose = np.asarray(state["ee_pos"] + state["ee_quat"], dtype=np.float64)

    return {
        "tcp_pose": tcp_pose,
        "tcp_vel": np.asarray(state.get("tcp_vel", state.get("ee_vel", [0.0] * 6)), dtype=np.float64),
        "gripper_pose": np.asarray([state.get("w_cmd", state.get("gripper_pose", 0.0))], dtype=np.float64),
        "q": np.asarray(state["q"], dtype=np.float64),
        "dq": np.asarray(state["dq"], dtype=np.float64),
    }


def build_observation(
    left_state: dict,
    right_state: dict,
    camera_frames: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    img_size: Tuple[int, int] = IMG_SIZE,
) -> dict:
    """Build the dual-arm observation format used by record_demos_dual.py."""
    obs: Dict[str, Any] = {}
    for name, (rgb, depth) in camera_frames.items():
        obs[name] = _resize_image(rgb, img_size)
        if depth is not None:
            obs[f"{name}_depth"] = _resize_image(depth, img_size)

    obs["left/state"] = _state_subdict(left_state)
    obs["right/state"] = _state_subdict(right_state)
    return obs


def _read_text_stripped(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def ensure_realsense_autosuspend_off(
    sysfs_usb_root: Path = Path("/sys/bus/usb/devices"),
    emit_warning: bool = True,
    udev_rule_path: Optional[Path] = None,
    require_safe: bool = False,
) -> List[Dict[str, Any]]:
    """Disable USB autosuspend for connected RealSense D435/D405 devices when possible."""
    root = Path(sysfs_usb_root)
    try:
        usb_paths = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return []

    statuses: List[Dict[str, Any]] = []
    for usb_path in usb_paths:
        id_vendor = _read_text_stripped(usb_path / "idVendor").lower()
        id_product = _read_text_stripped(usb_path / "idProduct").lower()
        if id_vendor != "8086" or id_product not in REALSENSE_USB_PRODUCTS:
            continue

        control_path = usb_path / "power" / "control"
        initial_control = _read_text_stripped(control_path)
        write_error = ""
        if initial_control and initial_control != "on":
            try:
                control_path.write_text("on", encoding="utf-8")
            except OSError as exc:
                write_error = str(exc)

        current_control = _read_text_stripped(control_path)
        statuses.append(
            {
                "usb_path": usb_path,
                "id_product": id_product,
                "product": _read_text_stripped(usb_path / "product"),
                "initial_control": initial_control,
                "current_control": current_control,
                "runtime_status": _read_text_stripped(usb_path / "power" / "runtime_status"),
                "control_path": control_path,
                "write_error": write_error,
            }
        )

    unsafe = [status for status in statuses if status["current_control"] != "on"]
    if unsafe:
        details = ", ".join(
            f"{status['usb_path'].name}:{status['product'] or status['id_product']} "
            f"control={status['current_control'] or 'missing'} "
            f"runtime={status['runtime_status'] or 'unknown'}"
            for status in unsafe
        )
        fix_hint = ""
        if udev_rule_path is not None:
            fix_hint = (
                f" Install the udev rule, then replug/trigger cameras: "
                f"sudo cp {udev_rule_path} /etc/udev/rules.d/ && "
                "sudo udevadm control --reload && "
                "sudo udevadm trigger --action=add --subsystem-match=usb --settle."
            )
        if require_safe:
            raise RuntimeError(f"RealSense USB autosuspend is unsafe: {details}.{fix_hint}")
    if emit_warning and unsafe:
        print(f"[camera] Warning: RealSense USB autosuspend is still enabled/unset: {details}")
        if udev_rule_path is not None:
            print(f"[camera] Install the udev rule, then replug/trigger cameras: sudo cp {udev_rule_path} /etc/udev/rules.d/")
            print("[camera] Then run: sudo udevadm control --reload && sudo udevadm trigger --action=add --subsystem-match=usb --settle")

    return statuses


def build_cameras_from_gello(record_depth: bool = False):
    os.environ.setdefault("RR_ENABLE_LEFT_WRIST", "1")
    from camera_config import build_cameras

    return build_cameras(record_depth=record_depth)


# =====================================================================
# Differential IK
# =====================================================================
class DifferentialIK:
    """Jacobian-based differential IK using pybullet in DIRECT mode."""

    def __init__(self, urdf_path: Path):
        try:
            import pybullet as p
            import pybullet_data
            from scipy.spatial.transform import Rotation
        except ImportError as exc:
            raise ImportError(
                "DifferentialIK requires pybullet and scipy. Run this in the gello_franka runtime environment."
            ) from exc

        self._p = p
        self._rotation = Rotation
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(str(urdf_path.parent), physicsClientId=self.physics_client)
        self.robot_id = p.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.physics_client,
        )

        name_to_id = {}
        for joint_index in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            info = p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.physics_client)
            name_to_id[info[1].decode()] = joint_index
            name_to_id[info[12].decode()] = joint_index

        self.arm_joint_indices = [name_to_id[f"fr3_joint{i}"] for i in range(1, 8)]
        self.ee_link_index = name_to_id["fr3_link8"]
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        self.num_dof = sum(
            1
            for joint_index in range(self.num_joints)
            if p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.physics_client)[3] > -1
        )

    def close(self) -> None:
        try:
            self._p.disconnect(physicsClientId=self.physics_client)
        except Exception:
            pass

    def set_joints(self, q: Sequence[float]) -> None:
        for idx, joint_index in enumerate(self.arm_joint_indices):
            self._p.resetJointState(
                self.robot_id,
                joint_index,
                float(q[idx]),
                physicsClientId=self.physics_client,
            )

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        state = self._p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            physicsClientId=self.physics_client,
        )
        return np.asarray(state[4], dtype=np.float64), np.asarray(state[5], dtype=np.float64)

    def compute_jacobian(self, q: Sequence[float]) -> np.ndarray:
        self.set_joints(q)
        zero_vec = [0.0] * self.num_dof
        all_positions = []
        arm_dof_indices = []
        dof_counter = 0
        for joint_index in range(self.num_joints):
            info = self._p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.physics_client)
            if info[3] <= -1:
                continue
            joint_state = self._p.getJointState(self.robot_id, joint_index, physicsClientId=self.physics_client)
            all_positions.append(joint_state[0])
            if joint_index in self.arm_joint_indices:
                arm_dof_indices.append(dof_counter)
            dof_counter += 1

        jac_lin, jac_ang = self._p.calculateJacobian(
            self.robot_id,
            self.ee_link_index,
            [0, 0, 0],
            all_positions,
            zero_vec,
            zero_vec,
            physicsClientId=self.physics_client,
        )
        jacobian = np.vstack([np.asarray(jac_lin), np.asarray(jac_ang)])
        return jacobian[:, arm_dof_indices]

    def delta_to_joint_delta(self, q: Sequence[float], delta_body: np.ndarray) -> np.ndarray:
        self.set_joints(q)
        _, quat = self.get_ee_pose()
        rotation = self._rotation.from_quat(quat).as_matrix()
        delta_body = np.asarray(delta_body, dtype=np.float64).reshape(6)
        dx_world = np.concatenate([rotation @ delta_body[:3], rotation @ delta_body[3:]])
        jacobian = self.compute_jacobian(q)
        damping = 1e-4
        inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping * np.eye(6))
        return inverse @ dx_world


def clip_joints(q: Sequence[float]) -> List[float]:
    return [
        float(max(lo, min(hi, value)))
        for value, lo, hi in zip(q, FRANKA_JOINT_LIMITS["q_min"], FRANKA_JOINT_LIMITS["q_max"])
    ]


# =====================================================================
# Bridge client and robot environment
# =====================================================================
class BridgeClient:
    """ZMQ client for gello_franka/experiments/r3_bridge_server.py."""

    def __init__(self, host: str, rep_port: int, push_port: int, pub_port: int):
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("BridgeClient requires pyzmq. Run inside the gello_franka runtime environment.") from exc

        self._zmq = zmq
        ctx = zmq.Context.instance()

        self.req = ctx.socket(zmq.REQ)
        self.req.connect(f"tcp://{host}:{rep_port}")

        self.push = ctx.socket(zmq.PUSH)
        self.push.connect(f"tcp://{host}:{push_port}")

        self.sub = ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{host}:{pub_port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.setsockopt(zmq.CONFLATE, 1)

    def handshake(self, init_q: Sequence[float]) -> None:
        self.req.send_json({"type": "init", "virtual_init_arm": [float(x) for x in init_q]})
        reply = self.req.recv_json()
        if reply.get("status") != "READY":
            raise RuntimeError(f"Bridge server did not become ready: {reply}")

    def get_state(self) -> Optional[dict]:
        try:
            return self.sub.recv_json(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return None

    def send_command(self, arm: Sequence[float], gripper_width: float) -> None:
        self.push.send_json(
            {
                "type": "cmd",
                "arm": [float(x) for x in arm],
                "gripper": float(gripper_width),
            }
        )

    def close(self) -> None:
        for socket_name in ("req", "push", "sub"):
            socket = getattr(self, socket_name, None)
            if socket is not None:
                try:
                    socket.close(linger=0)
                except Exception:
                    pass


def wait_for_state(
    bridge: BridgeClient,
    timeout: float = 0.25,
    poll_interval: float = 0.01,
) -> Optional[dict]:
    deadline = time.time() + timeout
    latest = None
    while time.time() < deadline:
        state = bridge.get_state()
        if state is not None:
            latest = state
            break
        time.sleep(poll_interval)
    return latest


def drain_state(bridge: BridgeClient, max_drain: int = 2000) -> int:
    drained = 0
    for _ in range(max_drain):
        if bridge.get_state() is None:
            break
        drained += 1
    return drained


def hard_reset_to_init(
    bridge: BridgeClient,
    init_q: Sequence[float],
    gripper_width: float,
    fresh_state_timeout: float = 5.0,
) -> dict:
    drain_state(bridge)
    bridge.handshake(init_q)
    for _ in range(5):
        bridge.send_command(init_q, gripper_width)
        time.sleep(0.02)

    fresh = wait_for_state(bridge, timeout=fresh_state_timeout, poll_interval=0.02)
    if fresh is None:
        raise RuntimeError("Timed out waiting for fresh bridge state after reset.")

    if "q" in fresh:
        err = max(abs(float(a) - float(b)) for a, b in zip(fresh["q"], init_q))
        if err > 0.10:
            print(f"[reset] Warning: bridge state is {err:.3f} rad from init after reset.")
    return fresh


def halt_bridge(bridge: BridgeClient, hold_s: float = 0.5) -> None:
    deadline = time.time() + hold_s
    last_q = None
    last_w = DEFAULT_GRIPPER_MAX_WIDTH
    while time.time() < deadline:
        state = bridge.get_state()
        if state is not None and "q" in state:
            last_q = list(state["q"])
            last_w = float(state.get("w_cmd", last_w))
        if last_q is not None:
            bridge.send_command(last_q, last_w)
        time.sleep(0.005)


def step_one_arm(
    ik: DifferentialIK,
    action_7: np.ndarray,
    state: dict,
    q_cmd: Optional[List[float]],
    config: argparse.Namespace,
) -> Tuple[List[float], float, np.ndarray, np.ndarray]:
    action = np.asarray(action_7, dtype=np.float64).reshape(7)
    cart_delta = action[:6]
    gripper_value = float(np.clip(action[6], 0.0, 1.0))
    gripper_width = gripper_value * float(config.gripper_max_width)

    q_actual = [float(x) for x in state["q"]]
    q_actual_arr = np.asarray(q_actual, dtype=np.float64)
    q_ref = q_cmd if bool(config.command_accum) and q_cmd is not None else q_actual
    dq = ik.delta_to_joint_delta(q_ref, cart_delta)
    q_new = clip_joints([q + dq_i for q, dq_i in zip(q_ref, dq)])
    return q_new, gripper_width, q_actual_arr, np.asarray(q_new, dtype=np.float64)


class DualFrankaBridgeEnv:
    """Dual-arm Franka env backed by two Gello bridge servers."""

    def __init__(
        self,
        bridge_left: BridgeClient,
        bridge_right: BridgeClient,
        ik_left: DifferentialIK,
        ik_right: DifferentialIK,
        cameras: Dict[str, Any],
        init_q_left: Sequence[float],
        init_q_right: Sequence[float],
        config: argparse.Namespace,
    ):
        self.bridge_left = bridge_left
        self.bridge_right = bridge_right
        self.ik_left = ik_left
        self.ik_right = ik_right
        self.cameras = cameras
        self.init_q_left = [float(x) for x in init_q_left]
        self.init_q_right = [float(x) for x in init_q_right]
        self.config = config
        self.state_left: Optional[dict] = None
        self.state_right: Optional[dict] = None
        self.q_cmd_left: Optional[List[float]] = list(self.init_q_left)
        self.q_cmd_right: Optional[List[float]] = list(self.init_q_right)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DualFrankaBridgeEnv":
        paths = configure_runtime_paths(args.gello_root, args.fairo_root)
        init_q_left, init_q_right = parse_init_joints(args.init_joints, args.left_init_joints, args.right_init_joints)

        cameras = {}
        if not args.no_camera:
            if not args.skip_realsense_power_check:
                udev_rule_path = None
                if paths["gello"] is not None:
                    udev_rule_path = paths["gello"] / "scripts" / "99-realsense-no-autosuspend.rules"
                ensure_realsense_autosuspend_off(
                    udev_rule_path=udev_rule_path,
                    require_safe=not args.allow_unsafe_realsense_power,
                )
            cameras = build_cameras_from_gello(record_depth=False)
            expected = {"external_camera", "left_wrist_camera", "right_wrist_camera"}
            missing = expected - set(cameras.keys())
            if missing:
                print(f"[camera] Warning: missing cameras {sorted(missing)}. Policy image count may differ from training.")

        urdf_path = Path(args.fr3_urdf).expanduser() if args.fr3_urdf else discover_fr3_urdf(paths["gello"])
        if not urdf_path.is_file():
            raise FileNotFoundError(f"FR3 URDF not found: {urdf_path}")

        env = cls(
            bridge_left=BridgeClient(args.zmq_host, args.left_rep_port, args.left_push_port, args.left_pub_port),
            bridge_right=BridgeClient(args.zmq_host, args.right_rep_port, args.right_push_port, args.right_pub_port),
            ik_left=DifferentialIK(urdf_path),
            ik_right=DifferentialIK(urdf_path),
            cameras=cameras,
            init_q_left=init_q_left,
            init_q_right=init_q_right,
            config=args,
        )
        print(f"[env] Left init : {[round(x, 4) for x in init_q_left]}")
        print(f"[env] Right init: {[round(x, 4) for x in init_q_right]}")
        return env

    def reset(self) -> dict:
        print("[env] Resetting both arms to init.")
        self.state_left = hard_reset_to_init(
            self.bridge_left,
            self.init_q_left,
            gripper_width=self.config.gripper_max_width,
            fresh_state_timeout=self.config.reset_timeout,
        )
        self.state_right = hard_reset_to_init(
            self.bridge_right,
            self.init_q_right,
            gripper_width=self.config.gripper_max_width,
            fresh_state_timeout=self.config.reset_timeout,
        )
        self.q_cmd_left = list(self.state_left.get("q", self.init_q_left))
        self.q_cmd_right = list(self.state_right.get("q", self.init_q_right))
        return self.get_obs()

    def _refresh_states(self) -> None:
        fresh_left = self.bridge_left.get_state()
        fresh_right = self.bridge_right.get_state()
        if fresh_left is not None:
            self.state_left = fresh_left
        if fresh_right is not None:
            self.state_right = fresh_right

    def get_obs(self) -> dict:
        self._refresh_states()
        if self.state_left is None:
            self.state_left = wait_for_state(self.bridge_left, timeout=self.config.verify_timeout)
        if self.state_right is None:
            self.state_right = wait_for_state(self.bridge_right, timeout=self.config.verify_timeout)
        if self.state_left is None or self.state_right is None:
            raise RuntimeError("Missing bridge state from one or both arms.")

        camera_frames: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        for name, camera in self.cameras.items():
            print(f"[camera] reading {name} ...", flush=True) # debug
            rgb, depth = camera.read()
            camera_frames[name] = (rgb, depth)
        return build_observation(self.state_left, self.state_right, camera_frames)

    def step(self, action: np.ndarray):
        action_left, action_right = split_dual_action(action)
        self._refresh_states()
        if self.state_left is None or self.state_right is None:
            raise RuntimeError("Cannot step without fresh left/right bridge state.")

        q_new_left, gripper_left, q_actual_left, q_cmd_left = step_one_arm(
            self.ik_left, action_left, self.state_left, self.q_cmd_left, self.config
        )
        q_new_right, gripper_right, q_actual_right, q_cmd_right = step_one_arm(
            self.ik_right, action_right, self.state_right, self.q_cmd_right, self.config
        )

        self.bridge_left.send_command(q_new_left, gripper_left)
        self.bridge_right.send_command(q_new_right, gripper_right)
        self.q_cmd_left = q_new_left
        self.q_cmd_right = q_new_right

        truncated = False
        info = {
            "left_gripper_width": gripper_left,
            "right_gripper_width": gripper_right,
            "left_q_cmd": q_new_left,
            "right_q_cmd": q_new_right,
        }

        if bool(self.config.verify_actions):
            state_timeout = float(self.config.verify_timeout)
            if self.config.state_stale_ms > 0:
                state_timeout = max(state_timeout, self.config.state_stale_ms / 1000.0)
            actual_left = wait_for_state(self.bridge_left, timeout=state_timeout)
            actual_right = wait_for_state(self.bridge_right, timeout=state_timeout)

            if self.config.state_stale_ms > 0 and (actual_left is None or actual_right is None):
                truncated = True
                info["stop_reason"] = "stale_bridge_state"
            if actual_left is not None:
                self.state_left = actual_left
                track_left = q_cmd_left - np.asarray(actual_left["q"], dtype=np.float64)
                info["left_track_err"] = float(np.max(np.abs(track_left)))
            else:
                info["left_track_err"] = None
            if actual_right is not None:
                self.state_right = actual_right
                track_right = q_cmd_right - np.asarray(actual_right["q"], dtype=np.float64)
                info["right_track_err"] = float(np.max(np.abs(track_right)))
            else:
                info["right_track_err"] = None

            if self.config.max_track_err > 0 and info["left_track_err"] is not None and info["right_track_err"] is not None:
                if info["left_track_err"] > self.config.max_track_err or info["right_track_err"] > self.config.max_track_err:
                    truncated = True
                    info["stop_reason"] = "track_error_exceeded"

        info["left_q_actual_before"] = q_actual_left
        info["right_q_actual_before"] = q_actual_right
        obs = self.get_obs()
        reward = 0.0
        done = False
        return obs, reward, done, truncated, info

    def halt(self, hold_s: float = 0.5) -> None:
        threads = [
            threading.Thread(target=halt_bridge, args=(self.bridge_left, hold_s), daemon=True),
            threading.Thread(target=halt_bridge, args=(self.bridge_right, hold_s), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=hold_s + 0.5)

    def close(self) -> None:
        self.bridge_left.close()
        self.bridge_right.close()
        self.ik_left.close()
        self.ik_right.close()


YourDualArmRobotEnv = DualFrankaBridgeEnv


# =====================================================================
# Keyboard / preview
# =====================================================================
_quit_requested = False
_keyboard_listener = None


def _init_x11_threads() -> None:
    try:
        import ctypes

        ctypes.CDLL("libX11.so.6").XInitThreads()
    except Exception as exc:
        print(f"[x11] XInitThreads unavailable: {exc}")


def _start_keyboard_halt(env: DualFrankaBridgeEnv) -> None:
    global _keyboard_listener

    try:
        from pynput import keyboard
    except ImportError:
        print("[keyboard] pynput not available; use Ctrl-C to stop.")
        return

    def on_press(key):
        global _quit_requested
        try:
            if key == keyboard.Key.esc:
                if not _quit_requested:
                    print("\n[keyboard] Esc pressed: halting both arms.")
                    env.halt()
                _quit_requested = True
        except AttributeError:
            pass

    _keyboard_listener = keyboard.Listener(on_press=on_press)
    _keyboard_listener.start()
    print("[keyboard] Esc: halt both arms and quit.")


def _stop_keyboard() -> None:
    global _keyboard_listener
    if _keyboard_listener is not None:
        _keyboard_listener.stop()
        _keyboard_listener = None


def show_preview(obs: Dict[str, Any], mode: str) -> None:
    if mode == "0":
        return
    import cv2

    image_keys = [
        key
        for key, value in obs.items()
        if key not in ("left/state", "right/state") and isinstance(value, np.ndarray) and value.ndim == 3
    ]
    if not image_keys:
        return
    keys = image_keys if mode == "all" else [mode if mode in image_keys else image_keys[0]]
    for key in keys:
        cv2.imshow(key, obs[key][:, :, ::-1])
    cv2.waitKey(1)


def destroy_preview_windows() -> None:
    try:
        import cv2

        cv2.destroyAllWindows()
    except Exception:
        pass


# =====================================================================
# Main inference loop
# =====================================================================
def inference_loop(args: argparse.Namespace) -> None:
    global _quit_requested, CONTROL_HZ, CONTROL_PERIOD

    CONTROL_HZ = float(args.control_hz)
    CONTROL_PERIOD = 1.0 / CONTROL_HZ
    action_norm_stats = validate_action_norm_stats(
        load_action_norm_stats(args.action_stats_path, embodiment_key=args.embodiment_key),
        expected_action_dim=14,
    )
    print(f"[stats] action min: {action_norm_stats['min']}")
    print(f"[stats] action max: {action_norm_stats['max']}")
    if "mask" in action_norm_stats:
        print(f"[stats] action mask: {action_norm_stats['mask']}")

    _init_x11_threads()
    env = DualFrankaBridgeEnv.from_args(args)
    policy = None

    try:
        print(f"[policy] Connecting to ws://{args.policy_host}:{args.policy_port}")
        policy = make_policy_client(args.policy_host, args.policy_port)
        _start_keyboard_halt(env)

        for episode in range(args.max_episodes):
            if _quit_requested:
                break
            obs = env.reset()
            print(f"\n[episode {episode + 1}/{args.max_episodes}] task: {args.task_instruction}")
            step_count = 0
            done = False

            while step_count < args.max_steps and not done and not _quit_requested:
                show_preview(obs, args.preview)
                normalized_chunk = query_policy_14d(
                    policy,
                    obs,
                    args.task_instruction,
                    episode_id=episode,
                    step_id=step_count,
                )
                action_chunk = unnormalize_actions(normalized_chunk, action_norm_stats)
                n_exec = min(args.actions_per_chunk, len(action_chunk))

                for action in action_chunk[:n_exec]:
                    if _quit_requested:
                        break
                    t0 = time.time()
                    obs, reward, done, truncated, info = env.step(action)
                    step_count += 1

                    if step_count <= 5 or step_count % args.verify_every_n == 0:
                        left_err = info.get("left_track_err")
                        right_err = info.get("right_track_err")
                        print(
                            f"[step {step_count}] "
                            f"L_grip={info['left_gripper_width'] * 1000:.1f}mm "
                            f"R_grip={info['right_gripper_width'] * 1000:.1f}mm "
                            f"L_err={left_err if left_err is not None else 'n/a'} "
                            f"R_err={right_err if right_err is not None else 'n/a'}"
                        )

                    if truncated:
                        print(f"[episode] stopping early: {info.get('stop_reason', 'truncated')}")
                        done = True
                        break

                    elapsed = time.time() - t0
                    if elapsed < CONTROL_PERIOD:
                        time.sleep(CONTROL_PERIOD - elapsed)

            print(f"[episode {episode + 1}] finished at {step_count} steps.")

    except KeyboardInterrupt:
        print("\n[inference] Ctrl-C received.")
    finally:
        print("[shutdown] Halting both arms and closing clients.")
        try:
            env.halt()
        except Exception as exc:
            print(f"[shutdown] halt failed: {exc}")
        _stop_keyboard()
        destroy_preview_windows()
        if policy is not None:
            try:
                policy.close()
            except Exception:
                pass
        env.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StarVLA dual-arm Franka inference through Gello bridge servers.")

    parser.add_argument("--gello_root", type=str, default=os.environ.get("GELLO_FRANKA_ROOT", ""))
    parser.add_argument("--fairo_root", type=str, default=os.environ.get("FAIRO_ROOT", ""))
    parser.add_argument("--fr3_urdf", type=str, default=os.environ.get("FR3_URDF", ""))
    parser.add_argument("--no_camera", action="store_true", help="Disable camera capture for dry-run/debug.")
    parser.add_argument(
        "--skip_realsense_power_check",
        action="store_true",
        default=os.environ.get("SKIP_REALSENSE_POWER_CHECK", "0") == "1",
        help="Skip the RealSense USB autosuspend guard.",
    )
    parser.add_argument(
        "--allow_unsafe_realsense_power",
        action="store_true",
        default=os.environ.get("ALLOW_UNSAFE_REALSENSE_POWER", "0") == "1",
        help="Warn but continue if any RealSense USB autosuspend control is not on.",
    )
    parser.add_argument("--preview", type=str, default=os.environ.get("RR_PREVIEW", "external"))

    parser.add_argument("--zmq_host", type=str, default=os.environ.get("ZMQ_HOST", "127.0.0.1"))
    parser.add_argument("--left_rep_port", type=int, default=int(os.environ.get("LEFT_ZMQ_REP_PORT", "6000")))
    parser.add_argument("--left_push_port", type=int, default=int(os.environ.get("LEFT_ZMQ_PULL_PORT", "6001")))
    parser.add_argument("--left_pub_port", type=int, default=int(os.environ.get("LEFT_ZMQ_PUB_PORT", "6002")))
    parser.add_argument("--right_rep_port", type=int, default=int(os.environ.get("RIGHT_ZMQ_REP_PORT", "6003")))
    parser.add_argument("--right_push_port", type=int, default=int(os.environ.get("RIGHT_ZMQ_PULL_PORT", "6004")))
    parser.add_argument("--right_pub_port", type=int, default=int(os.environ.get("RIGHT_ZMQ_PUB_PORT", "6005")))

    parser.add_argument("--policy_host", type=str, default=os.environ.get("POLICY_HOST", "127.0.0.1"))
    parser.add_argument("--policy_port", type=int, default=int(os.environ.get("POLICY_PORT", "5694")))
    parser.add_argument("--task_instruction", type=str, default=os.environ.get("TASK_INSTRUCTION", "perform the task"))
    parser.add_argument("--action_stats_path", type=str, default=os.environ.get("ACTION_STATS_PATH", ""))
    parser.add_argument("--embodiment_key", type=str, default=os.environ.get("EMBODIMENT_KEY", "new_embodiment"))

    parser.add_argument(
        "--init_joints",
        type=str,
        default=os.environ.get("INIT_JOINTS", "0,0,0,-1.57,0,1.57,0,0,0,0,-1.57,0,1.57,0"),
        help="14 floats left|right, or 7 floats to reuse the same pose for both arms.",
    )
    parser.add_argument("--left_init_joints", type=str, default=os.environ.get("LEFT_INIT_JOINTS", ""))
    parser.add_argument("--right_init_joints", type=str, default=os.environ.get("RIGHT_INIT_JOINTS", ""))

    parser.add_argument("--gripper_max_width", type=float, default=float(os.environ.get("GRIPPER_MAX_WIDTH", "0.100")))
    parser.add_argument("--max_episodes", type=int, default=int(os.environ.get("MAX_EPISODES", "10")))
    parser.add_argument("--max_steps", type=int, default=int(os.environ.get("MAX_STEPS", "3000")))
    parser.add_argument("--actions_per_chunk", type=int, default=int(os.environ.get("ACTIONS_PER_CHUNK", "16")))
    parser.add_argument("--control_hz", type=float, default=float(os.environ.get("CONTROL_HZ", "15")))

    parser.add_argument("--command_accum", type=int, default=int(os.environ.get("COMMAND_ACCUM", "1")))
    parser.add_argument("--verify_actions", type=int, default=int(os.environ.get("VERIFY_ACTIONS", "1")))
    parser.add_argument("--verify_every_n", type=int, default=int(os.environ.get("VERIFY_EVERY_N", "10")))
    parser.add_argument("--verify_timeout", type=float, default=float(os.environ.get("VERIFY_TIMEOUT", "0.25")))
    parser.add_argument("--reset_timeout", type=float, default=float(os.environ.get("RESET_TIMEOUT", "5.0")))
    parser.add_argument(
        "--max_track_err",
        type=float,
        default=float(os.environ.get("MAX_TRACK_ERR", "0")),
        help="Max |q_cmd - q_actual| per joint before stopping. 0 disables.",
    )
    parser.add_argument(
        "--state_stale_ms",
        type=int,
        default=int(os.environ.get("STATE_STALE_MS", "500")),
        help="If verify_actions is enabled and either bridge publishes no state within verify_timeout, stop. 0 disables.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    print("=" * 60)
    print("StarVLA dual-arm Franka inference")
    print(f"Left bridge : {args.zmq_host}:{args.left_rep_port}/{args.left_push_port}/{args.left_pub_port}")
    print(f"Right bridge: {args.zmq_host}:{args.right_rep_port}/{args.right_push_port}/{args.right_pub_port}")
    print(f"Policy      : ws://{args.policy_host}:{args.policy_port}")
    print(f"Stats       : {args.action_stats_path}")
    print(f"Task        : {args.task_instruction}")
    print(f"Chunk exec  : {args.actions_per_chunk} actions/query")
    print("=" * 60)
    inference_loop(args)


if __name__ == "__main__":
    main()
