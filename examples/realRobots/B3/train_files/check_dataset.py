#!/usr/bin/env python3
"""Check the B3 schema, real loader, normalization and optional model backward."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_recipe(cfg, robot):
    data = cfg.datasets.vla_data
    require(cfg.framework.name == "QwenOFT", "This B3 recipe expects framework.name=QwenOFT")
    require(data.dataset_py == "lerobot_datasets", "Use dataset_py=lerobot_datasets")
    require(data.data_mix == "b3_stack_blocks", "Use data_mix=b3_stack_blocks")
    require(data.get("action_mode", "abs") == "abs", "B3 xyzw commands require action_mode=abs")
    require(data.get("include_state", False) is True, "B3 training requires include_state=true")
    require(int(cfg.framework.action_model.action_dim) == sum(robot.action_key_dims.values()), "action_dim must be 16")
    require(int(cfg.framework.action_model.state_dim) == sum(robot.state_key_dims.values()), "state_dim must be 16")
    require(
        int(cfg.framework.action_model.action_horizon) == len(robot.action_indices),
        "action_horizon must match B3 action_indices (16)",
    )
    require(int(data.per_device_batch_size) > 0, "per_device_batch_size must be positive")
    require(int(data.get("num_workers", 4)) >= 0, "num_workers must be nonnegative")
    require(list(data.image_size) == list(data.obs_image_size), "image_size and obs_image_size must match")
    require(len(data.image_size) == 2 and min(data.image_size) > 0, "image_size must contain two positive values")
    ds = json.loads((REPO_ROOT / "starVLA/config/deepseeds/ds_config.yaml").read_text())
    require(
        int(cfg.trainer.gradient_accumulation_steps) == ds["gradient_accumulation_steps"] == 1,
        "This launcher uses the existing ZeRO-2 config with gradient_accumulation_steps=1; "
        "changing only the training YAML does not enable accumulation",
    )
    require(
        int(os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1")) == 1,
        "Unset ACCELERATE_GRADIENT_ACCUMULATION_STEPS or set it to 1 for this recipe",
    )
    for key in ("max_train_steps", "save_interval", "eval_interval", "logging_frequency"):
        require(int(cfg.trainer[key]) > 0, f"trainer.{key} must be positive")


def prepare_metadata(dataset_path, install):
    """Install the tracked schema once; reject an existing incompatible mapping."""
    template = json.loads((SCRIPT_DIR / "modality.json").read_text())
    target = dataset_path / "meta/modality.json"
    if not target.exists():
        require(install, f"Missing {target}; rerun with --install-modality")
        with target.open("x") as stream:
            json.dump(template, stream, indent=2)
            stream.write("\n")
        print(f"Installed {target}", flush=True)
    require(json.loads(target.read_text()) == template, f"Existing {target} differs from the B3 schema")
    return template


def validate_source(dataset_path, full_scan):
    import numpy as np
    import pyarrow.parquet as pq

    info = json.loads((dataset_path / "meta/info.json").read_text())
    require(info["codebase_version"] == "v2.1", "Expected LeRobot v2.1")
    require(info["fps"] == 50, "Expected 50 Hz source data")
    for column, arm_field in (("observation.state", "eepose"), ("action", "pose")):
        expected_names = []
        for arm in ("left", "right"):
            expected_names.extend(f"{arm}_arm.{arm_field}.position_m.{i}" for i in range(3))
            expected_names.extend(f"{arm}_arm.{arm_field}.orientation_xyzw.{i}" for i in range(4))
            expected_names.append(f"{arm}_gripper.open_fraction.0")
        feature = info["features"][column]
        require(feature["shape"] == [16], f"{column} must have 16 dimensions")
        require(feature["names"] == expected_names, f"Unexpected {column} order / units / quaternion convention")
    tasks = [json.loads(line) for line in (dataset_path / "meta/tasks.jsonl").read_text().splitlines()]
    require(tasks == [{"task_index": 0, "task": "stack blocks"}], "Unexpected B3 task mapping")
    episodes = [json.loads(line) for line in (dataset_path / "meta/episodes.jsonl").read_text().splitlines()]
    require(len(episodes) == info["total_episodes"], "Episode count differs from info.json")
    require(len({ep["episode_index"] for ep in episodes}) == len(episodes), "Duplicate episode indices")
    require(sum(ep["length"] for ep in episodes) == info["total_frames"], "Frame counts differ from info.json")
    cameras = ("front_camera", "left_wrist_camera", "right_wrist_camera")
    for episode in episodes:
        index = episode["episode_index"]
        values = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
        parquet = dataset_path / info["data_path"].format(**values)
        require(parquet.is_file(), f"Missing {parquet}")
        require(pq.read_metadata(parquet).num_rows == episode["length"], f"Wrong length in {parquet}")
        for camera in cameras:
            video_key = f"observation.images.{camera}"
            require(info["features"][video_key]["dtype"] == "video", f"Missing video feature {video_key}")
            video = dataset_path / info["video_path"].format(**values, video_key=video_key)
            require(video.is_file(), f"Missing {video}")
        if full_scan:
            table = pq.read_table(parquet, columns=["observation.state", "action", "timestamp", "task_index"])
            for column in ("observation.state", "action"):
                array = np.asarray(table[column].to_pylist(), dtype=np.float32)
                require(array.shape == (episode["length"], 16), f"Wrong shape in {parquet}:{column}")
                require(np.isfinite(array).all(), f"Non-finite data in {parquet}:{column}")
                for start in (3, 11):
                    require(
                        np.allclose(np.linalg.norm(array[:, start : start + 4], axis=1), 1, atol=1e-4),
                        f"Non-unit quaternion in {parquet}:{column}",
                    )
                require(
                    ((array[:, [7, 15]] >= 0) & (array[:, [7, 15]] <= 1)).all(),
                    f"Gripper outside [0, 1] in {parquet}:{column}",
                )
            timestamps = np.asarray(table["timestamp"].to_pylist())
            require(np.isfinite(timestamps).all(), f"Non-finite timestamps in {parquet}")
            require(np.allclose(np.diff(timestamps), 1 / info["fps"], atol=1e-4), f"Wrong timestamps in {parquet}")
            require(set(table["task_index"].to_pylist()) == {0}, f"Unknown task in {parquet}")
    print(f"Source OK: {len(episodes)} episodes, {info['total_frames']} frames, 3 videos per episode", flush=True)
    return info


def validate_sample(sample, cfg):
    import numpy as np

    horizon = int(cfg.framework.action_model.action_horizon)
    for key, shape in (("state", (1, 16)), ("action", (horizon, 16))):
        require(sample[key].shape == shape, f"{key} shape {sample[key].shape} != {shape}")
        require(np.isfinite(sample[key]).all(), f"Non-finite normalized {key}")
        require(np.max(np.abs(sample[key])) <= 1.001, f"{key} outside min_max range")
    require(sample["lang"] == "stack blocks", "Language mapping failed")
    require(len(sample["image"]) == 3, "Expected front / left wrist / right wrist images")
    for image in sample["image"]:
        require(image.mode == "RGB", f"Expected RGB, got {image.mode}")
        require(image.size == tuple(cfg.datasets.vla_data.image_size), "Wrong resized image dimensions")


def check_loader(cfg, info, samples, num_workers, report_dir, model_smoke, full_scan):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader

    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    mixture = get_vla_dataset(cfg.datasets.vla_data)
    dataset = mixture.datasets[0]
    require(len(dataset.trajectory_ids) == info["total_episodes"], "Loader skipped episodes")
    require(len(dataset) == info["total_frames"], "Loader skipped frames")
    checks = []
    for i in range(samples):
        # Spread checks across episodes, including the last frame and its padding.
        ep_slot = round(i * (len(dataset.trajectory_ids) - 1) / max(samples - 1, 1))
        episode = int(dataset.trajectory_ids[ep_slot])
        length = int(dataset.trajectory_lengths[ep_slot])
        step = (0, length // 2, length - 1)[i % 3]
        raw = dataset.get_step_data(episode, step)
        transformed = dataset.transforms(copy.deepcopy(raw))
        sample = dataset._pack_sample(transformed)
        validate_sample(sample, cfg)
        # Undo normalization from the packed model input, checking concatenation
        # order as well as the transform's inverse against the source parquet.
        packed_fields = {}
        for modality in ("state", "action"):
            offset = 0
            for key in dataset.modality_keys[modality]:
                field = dataset.lerobot_modality_meta.get_key_meta(key)
                width = field.end - field.start
                packed_fields[key] = torch.as_tensor(sample[modality][:, offset : offset + width])
                offset += width
        restored = dataset.transforms.unapply(packed_fields)
        parquet = dataset.dataset_path / info["data_path"].format(
            episode_chunk=episode // info["chunks_size"],
            episode_index=episode,
        )
        table = pq.read_table(parquet, columns=["observation.state", "action"])
        for modality, column in (("state", "observation.state"), ("action", "action")):
            keys = dataset.modality_keys[modality]
            expected = np.asarray(table[column].to_pylist())
            offsets = dataset.delta_indices[keys[0]]
            expected = expected[np.clip(step + offsets, 0, length - 1)]
            recovered = np.concatenate([restored[key] for key in keys], axis=1)
            np.testing.assert_allclose(recovered, expected, rtol=1e-5, atol=1e-6)
        checks.append({"episode": episode, "step": step})

    # Exercise the exact list-of-dicts collator used by training.
    loader = DataLoader(mixture, batch_size=2, num_workers=num_workers, collate_fn=collate_fn)
    batch = next(iter(loader))
    require(isinstance(batch, list) and len(batch) == 2, "Training collator failed")
    for sample in batch:
        validate_sample(sample, cfg)
    report_dir.mkdir(parents=True, exist_ok=True)
    mixture.save_dataset_statistics(report_dir / "dataset_statistics.json")
    stats = json.loads((report_dir / "dataset_statistics.json").read_text())[dataset.tag]
    require(len(stats["action"]["min"]) == 16, "Saved action statistics have wrong dimensions")
    require(all(stats["action"]["mask"]), "Continuous grippers must be normalized, without binary masks")

    if model_smoke:
        require(torch.cuda.is_available(), "--model-smoke requires a CUDA GPU and available VLM weights")
        from starVLA.model.framework.base_framework import build_framework

        model = build_framework(cfg).cuda().train()
        loss = model(batch[:1])["action_loss"]
        require(bool(torch.isfinite(loss)), "Non-finite model loss")
        loss.backward()
        require(any(p.grad is not None for p in model.action_model.parameters()), "Action head has no gradients")
        model.eval()
        with torch.inference_mode():
            prediction = model.predict_action(batch[:1])["normalized_actions"]
        require(prediction.shape == (1, 16, 16), f"Wrong prediction shape: {prediction.shape}")
        require(np.isfinite(prediction).all(), "Non-finite predictions")

    report = {
        "dataset_path": str(dataset.dataset_path),
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "sample_checks": checks,
        "num_workers": num_workers,
        "state_shape": [1, 16],
        "action_shape": [16, 16],
        "camera_count": 3,
        "normalization": "min_max",
        "model_smoke": model_smoke,
        "full_scan": full_scan,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"B3 checks passed. Report: {report_dir / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", type=Path, default=SCRIPT_DIR / "starvla_b3.yaml")
    parser.add_argument("--install-modality", action="store_true")
    parser.add_argument("--full-scan", action="store_true", help="Validate every state/action row and timestamp")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0, help="Worker count for the check's DataLoader")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "results/B3/data_check")
    parser.add_argument(
        "--model-smoke",
        action="store_true",
        help="Load VLM weights and check forward/backward/inference",
    )
    args, overrides = parser.parse_known_args()
    require(args.samples > 0 and args.num_workers >= 0, "samples must be positive; num-workers must be nonnegative")

    from omegaconf import OmegaConf

    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    cfg = apply_config_compat(
        OmegaConf.merge(
            OmegaConf.load(args.config_yaml),
            OmegaConf.from_dotlist(normalize_dotlist_args(overrides)),
        )
    )
    robot = ROBOT_TYPE_CONFIG_MAP["b3_dual_eef"]
    validate_recipe(cfg, robot)
    dataset_name = DATASET_NAMED_MIXTURES[cfg.datasets.vla_data.data_mix][0][0]
    dataset_path = Path(cfg.datasets.vla_data.data_root_dir) / dataset_name
    info = validate_source(dataset_path, args.full_scan)
    prepare_metadata(dataset_path, args.install_modality)
    check_loader(cfg, info, args.samples, args.num_workers, args.report_dir, args.model_smoke, args.full_scan)


if __name__ == "__main__":
    main()
