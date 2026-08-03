#!/usr/bin/env python3
"""Benchmark-agnostic offline action evaluation on one LeRobot trajectory.

For every frame in an episode, the script runs one model inference and aligns
the first action in the predicted action chunk with that frame's ground-truth
action.  It saves a per-dimension plot, a CSV file, and error metrics as JSON.

Run this script from the repository root, for example::

```
python tools/evaluate_trajectory_actions.py \
    --checkpoint results/Checkpoints/{run_id}/checkpoints/steps_100000_pytorch_model.pt \
    --benchmark-name Franka \
    --trajectory-index 0 \
    --use-bf16
```
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.share_tools import read_mode_config


LOGGER = logging.getLogger("trajectory_action_eval")


def _safe_path_component(value: str, argument: str) -> str:
    value = value.strip().replace(" ", "_")
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{argument} must be a non-empty path component, got {value!r}")
    return value


def _checkpoint_run_and_step(checkpoint: Path, ckpt_name: str | None) -> tuple[str, str]:
    """Return the output identity for a standard RUN/checkpoints/steps_N file."""
    run_name = ckpt_name or checkpoint.parents[1].name
    match = re.search(r"(?:^|_)steps?_(\d+)(?:_|$)", checkpoint.stem)
    if match is None:
        match = re.search(r"(\d+)(?!.*\d)", checkpoint.stem)
    step = match.group(1) if match else checkpoint.stem
    return (
        _safe_path_component(run_name, "--ckpt-name"),
        _safe_path_component(step, "checkpoint step"),
    )


def _prediction_kwargs(values: Sequence[str]) -> dict[str, Any]:
    """Parse repeatable KEY=VALUE arguments with OmegaConf scalar typing."""
    if not values:
        return {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--predict-arg expects KEY=VALUE, got {value!r}")
    parsed = OmegaConf.from_dotlist(list(values))
    result = OmegaConf.to_container(parsed, resolve=True)
    if not isinstance(result, dict):
        raise ValueError("--predict-arg values did not produce a mapping")
    return result


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first_step_flat(raw_data: dict[str, Any], keys: Sequence[str]) -> np.ndarray:
    """Concatenate the first temporal element of a list of modality keys."""
    parts = []
    for key in keys:
        value = _as_numpy(raw_data[key])
        # Dataset modalities normally have shape (T, D). Scalars and (D,)
        # are accepted as well to keep this useful across LeRobot versions.
        if value.ndim >= 2:
            value = value[0]
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def _dimension_labels(dataset: Any) -> list[str]:
    labels: list[str] = []
    trajectory_id = dataset.trajectory_ids[0]
    raw = dataset.get_step_data(trajectory_id, 0)
    for key in dataset.modality_keys["action"]:
        width = _first_step_flat(raw, [key]).size
        clean_key = key.removeprefix("action.")
        if width == 1:
            labels.append(clean_key)
        else:
            labels.extend(f"{clean_key}[{i}]" for i in range(width))
    return labels


def _compute_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    absolute_error = np.abs(error)
    squared_error = error**2
    return {
        "num_frames": int(target.shape[0]),
        "action_dim": int(target.shape[1]),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "max_absolute_error": float(absolute_error.max()),
        "per_dimension": [
            {
                "mae": float(absolute_error[:, i].mean()),
                "rmse": float(np.sqrt(squared_error[:, i].mean())),
                "max_absolute_error": float(absolute_error[:, i].max()),
            }
            for i in range(target.shape[1])
        ],
    }


def _save_plot(
    target: np.ndarray,
    prediction: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
) -> None:
    # Keep plotting optional at import time so ``--help`` and metric utility
    # tests do not require the visualization dependency to be initialized.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim = target.shape[1]
    columns = 2 if dim > 1 else 1
    rows = (dim + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 3.2 * rows), squeeze=False)
    x = np.arange(target.shape[0])

    for i, axis in enumerate(axes.flat):
        if i >= dim:
            axis.set_visible(False)
            continue
        axis.plot(x, target[:, i], label="ground truth", linewidth=1.4)
        axis.plot(x, prediction[:, i], label="prediction", linewidth=1.1, alpha=0.85)
        axis.set_title(labels[i])
        axis.set_xlabel("trajectory frame")
        axis.set_ylabel("action")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle("Ground-truth vs predicted action (first action of each predicted chunk)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_trajectory(dataset: Any, trajectory_index: int, trajectory_id: str | None) -> tuple[Any, int]:
    ids = list(dataset.trajectory_ids)
    if trajectory_id is not None:
        matching = [i for i, value in enumerate(ids) if str(value) == trajectory_id]
        if not matching:
            raise ValueError(f"trajectory id {trajectory_id} not found; available ids: {ids[:20]}")
        trajectory_index = matching[0]
    if not 0 <= trajectory_index < len(ids):
        raise IndexError(f"trajectory-index must be in [0, {len(ids) - 1}], got {trajectory_index}")
    trajectory_value = ids[trajectory_index]
    if isinstance(trajectory_value, np.generic):
        trajectory_value = trajectory_value.item()
    return trajectory_value, int(dataset.trajectory_lengths[trajectory_index])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt-path", dest="checkpoint", required=True)
    parser.add_argument("--benchmark-name", required=True, help="Name used to group results, e.g. LIBERO")
    parser.add_argument("--ckpt-name", help="Override checkpoint run name used in the output directory")
    parser.add_argument("--dataset-index", type=int, default=0, help="Dataset within the configured mixture")
    trajectory = parser.add_mutually_exclusive_group()
    trajectory.add_argument("--trajectory-index", type=int, default=0, help="Zero-based trajectory position")
    trajectory.add_argument("--trajectory-id", help="LeRobot episode/trajectory id (numeric or string)")
    parser.add_argument("--data-root-dir", help="Override datasets.vla_data.data_root_dir")
    parser.add_argument("--data-mix", help="Override datasets.vla_data.data_mix")
    parser.add_argument("--unnorm-key", help="Checkpoint statistics key; inferred for a single dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Evaluate only the first N frames (useful for smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "offline_eval",
        help="Root; final path is {ckpt_name}_{steps}/{benchmark_name}/episode_{id}",
    )
    parser.add_argument(
        "--predict-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Model-specific predict_action kwarg, e.g. num_ddim_steps=20",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable model config override passed to checkpoint loading",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    benchmark_name = _safe_path_component(args.benchmark_name, "--benchmark-name")
    run_name, step = _checkpoint_run_and_step(checkpoint, args.ckpt_name)
    predict_kwargs = _prediction_kwargs(args.predict_arg)

    # Read the model config from the checkpoint
    model_cfg, _ = read_mode_config(str(checkpoint)) 
    cfg = OmegaConf.create(model_cfg)
    data_cfg = cfg.datasets.vla_data
    if args.data_root_dir:
        data_cfg.data_root_dir = str(Path(args.data_root_dir).expanduser().resolve())
    if args.data_mix:
        data_cfg.data_mix = args.data_mix

    LOGGER.info("Building training dataset mix %s from %s", data_cfg.data_mix, data_cfg.data_root_dir)
    mixture = get_vla_dataset(data_cfg=data_cfg, mode="eval", seed=args.seed)
    if not 0 <= args.dataset_index < len(mixture.datasets):
        raise IndexError(
            f"dataset-index must be in [0, {len(mixture.datasets) - 1}], got {args.dataset_index}"
        )
    dataset = mixture.datasets[args.dataset_index]
    trajectory_id, trajectory_length = _select_trajectory(
        dataset, args.trajectory_index, args.trajectory_id
    )
    frame_count = min(trajectory_length, args.max_frames or trajectory_length)
    LOGGER.info(
        "Selected dataset=%s trajectory_id=%s frames=%d/%d",
        dataset.dataset_name,
        trajectory_id,
        frame_count,
        trajectory_length,
    )

    policy = PolicyServerWrapper(
        ckpt_path=str(checkpoint),
        device=args.device,
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
        config_overrides=args.config_override,
    )

    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for frame_index in tqdm(range(frame_count), desc="Inferring trajectory"):
            raw_data = dataset.get_step_data(trajectory_id, frame_index)
            # Read the physical-scale target before transforms normalize (and
            # may mutate) the modality arrays used for model input.
            target = _first_step_flat(raw_data, dataset.modality_keys["action"])
            transformed_data = dataset.transforms(raw_data)
            example = dataset._pack_sample(transformed_data)

            result = policy.predict_action(
                examples=[example],
                unnorm_key=args.unnorm_key,
                **predict_kwargs,
            )
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
                raise ValueError(f"Expected predicted actions shaped (1, T, D), got {action_chunk.shape}")
            prediction = action_chunk[0, 0]
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Action dimension mismatch at frame {frame_index}: "
                    f"ground truth {target.shape}, prediction {prediction.shape}"
                )
            targets.append(target)
            predictions.append(prediction)

    target_array = np.stack(targets)
    prediction_array = np.stack(predictions)
    labels = _dimension_labels(dataset)
    metrics = _compute_metrics(target_array, prediction_array)
    for label, values in zip(labels, metrics["per_dimension"]):
        values["name"] = label
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "dataset": dataset.dataset_name,
            "trajectory_id": trajectory_id,
            "trajectory_index": (
                int(np.where(dataset.trajectory_ids == trajectory_id)[0][0])
            ),
            "robot_tag": str(dataset.tag),
            "benchmark_name": args.benchmark_name,
            "data_mix": str(data_cfg.data_mix),
            "prediction_kwargs": predict_kwargs,
            "prediction_alignment": "first action of each predicted chunk",
        }
    )

    output_dir = (
        args.output_root.expanduser().resolve()
        / f"{run_name}_{step}"
        / benchmark_name
        / f"episode_{_safe_path_component(str(trajectory_id), 'trajectory id')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "action_comparison.png"
    csv_path = output_dir / "actions.csv"
    metrics_path = output_dir / "metrics.json"
    _save_plot(target_array, prediction_array, labels, plot_path)

    columns = ["frame"] + [f"true/{name}" for name in labels] + [f"pred/{name}" for name in labels]
    table = np.column_stack((np.arange(frame_count), target_array, prediction_array))
    np.savetxt(csv_path, table, delimiter=",", header=",".join(columns), comments="")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    LOGGER.info("Saved plot: %s", plot_path)
    LOGGER.info("Saved actions: %s", csv_path)
    LOGGER.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main(build_argparser().parse_args())
