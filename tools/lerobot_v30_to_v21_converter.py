"""Convert a LeRobot 3.0 dataset to a LeRobot 2.1 layout.

Usage:

```bash
python tools/lerobot_v30_to_v21_converter.py \
  --input_dir <path_to_v30_dataset> \
  --output_dir <path_to_v21_dataset>
```

The converter keeps the existing modality metadata, rewrites the legacy
``info.json`` paths, splits aggregated data parquet files into one parquet per
episode, and trims per-episode videos with ffmpeg when source videos are found.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    length: int
    task_index: int | None
    task: str | None
    data_chunk_index: int | None
    data_file_index: int | None
    data_file_from_index: int | None
    from_timestamps: dict[str, float]
    video_file_indices: dict[str, dict[str, int]]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: dict[str, Any], indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _as_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _format_candidates(template: str | None, **kwargs: Any) -> list[str]:
    if not template:
        return []
    candidates = []
    try:
        candidates.append(template.format(**kwargs))
    except Exception:
        pass
    return candidates


def _existing_from_candidates(root: Path, candidates: list[str], suffix: str) -> Path | None:
    seen: set[Path] = set()
    for candidate in candidates:
        path = root / candidate
        if path not in seen and path.exists():
            return path
        seen.add(path)

    for candidate in candidates:
        needle = candidate.replace("**/", "").replace("*", "")
        if not needle:
            continue
        for path in root.rglob(suffix):
            if needle in path.as_posix() and path.exists():
                return path
    return None


def _resolve_data_path(
    input_dir: Path,
    source_info: dict[str, Any],
    chunk_index: int | None,
    file_index: int | None,
    episode_index: int | None = None,
) -> Path:
    data_template = source_info.get("data_path")
    candidates: list[str] = []
    if chunk_index is not None and file_index is not None:
        candidates.extend(
            _format_candidates(
                data_template,
                chunk_index=chunk_index,
                file_index=file_index,
                episode_chunk=chunk_index,
                episode_index=episode_index if episode_index is not None else file_index,
                file_from_index=file_index,
            )
        )
        candidates.extend(
            [
                f"data/chunk-{chunk_index:03d}/file-{file_index:06d}.parquet",
                f"data/chunk-{chunk_index}/file-{file_index}.parquet",
                f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet" if episode_index is not None else "",
                f"data/chunk-{chunk_index}/episode_{episode_index}.parquet" if episode_index is not None else "",
            ]
        )
    candidates = [c for c in candidates if c]
    path = _existing_from_candidates(input_dir, candidates, ".parquet")
    if path is None:
        raise FileNotFoundError(
            f"Unable to resolve source data parquet for chunk_index={chunk_index}, file_index={file_index}"
        )
    return path


def _resolve_video_path(
    input_dir: Path,
    source_info: dict[str, Any],
    video_key: str,
    chunk_index: int | None,
    file_index: int | None,
    episode_index: int | None = None,
) -> Path | None:
    video_template = source_info.get("video_path")
    candidates: list[str] = []
    if chunk_index is not None and file_index is not None:
        candidates.extend(
            _format_candidates(
                video_template,
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
                episode_chunk=chunk_index,
                episode_index=episode_index if episode_index is not None else file_index,
            )
        )
        candidates.extend(
            [
                f"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4",
                f"videos/{video_key}/chunk-{chunk_index}/file-{file_index}.mp4",
                f"videos/chunk-{chunk_index:03d}/{video_key}/file-{file_index:06d}.mp4",
                f"videos/chunk-{chunk_index}/[{video_key}]/file-{file_index}.mp4",
                f"videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4" if episode_index is not None else "",
                f"videos/chunk-{chunk_index}/{video_key}/episode_{episode_index}.mp4" if episode_index is not None else "",
            ]
        )
    if episode_index is not None:
        candidates.extend(
            [
                f"videos/**/episode_{episode_index:06d}.mp4",
                f"**/episode_{episode_index:06d}.mp4",
            ]
        )
    candidates = [c for c in candidates if c]
    path = _existing_from_candidates(input_dir, candidates, ".mp4")
    return path


def _probe_video_duration(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to trim videos")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip())


def _trim_video(source: Path, start_ts: float, end_ts: float, target: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to split videos")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    duration = max(end_ts - start_ts, 1e-3)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(start_ts, 0.0):.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target),
    ]
    subprocess.run(cmd, check=True)


def _copy_ancillary_entries(input_dir: Path, output_dir: Path) -> None:
    for entry in input_dir.iterdir():
        if entry.name in {"data", "meta", "videos"}:
            continue
        dst = output_dir / entry.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        _copy_path(entry, dst)


def _load_tasks(tasks_path: Path | None, episodes_meta: pd.DataFrame) -> list[dict[str, Any]]:
    if tasks_path is not None and tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        if "task_index" not in tasks_df.columns:
            tasks_df = tasks_df.reset_index().rename(columns={tasks_df.index.name or "index": "task_index"})
        if "task" not in tasks_df.columns:
            task_columns = [c for c in tasks_df.columns if c != "task_index"]
            if task_columns:
                tasks_df = tasks_df.rename(columns={task_columns[0]: "task"})
        return tasks_df[[c for c in ["task_index", "task"] if c in tasks_df.columns]].to_dict(orient="records")

    tasks: list[dict[str, Any]] = []
    if "task_index" in episodes_meta.columns:
        task_indices = sorted({int(v) for v in episodes_meta["task_index"].dropna().unique().tolist()})
        for task_index in task_indices:
            tasks.append({"task_index": task_index, "task": str(task_index)})
    return tasks


def _load_episode_meta(meta_dir: Path) -> pd.DataFrame:
    episode_paths = sorted(meta_dir.glob("episodes/**/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No v3.0 episode parquet files found under {meta_dir / 'episodes'}")
    frames = [pd.read_parquet(path) for path in episode_paths]
    meta = pd.concat(frames, ignore_index=True)
    if "episode_index" not in meta.columns:
        raise ValueError("Episode metadata is missing the 'episode_index' column")
    return meta


def _build_v2_info(
    source_info: dict[str, Any],
    chunk_size: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
) -> dict[str, Any]:
    info = dict(source_info)
    info["codebase_version"] = "v2.1"
    info["chunks_size"] = int(chunk_size)
    info["data_path"] = LEGACY_DATA_PATH
    info["video_path"] = LEGACY_VIDEO_PATH
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(total_tasks)
    info["total_videos"] = int(total_videos)
    info["total_chunks"] = int(total_chunks)
    info.setdefault("splits", {})
    info["splits"]["train"] = f"0:{total_episodes}"
    return info


def _copy_metadata_files(input_dir: Path, output_dir: Path) -> None:
    meta_dir = input_dir / "meta"
    out_meta = output_dir / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)

    for name in ["modality.json", "embodiment.json", "stats_gr00t.json"]:
        src = meta_dir / name
        if src.exists():
            _copy_path(src, out_meta / name)


def _write_tasks(output_dir: Path, tasks: list[dict[str, Any]]) -> None:
    tasks_path = output_dir / "meta" / "tasks.jsonl"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def _write_episode_lines(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_dataset(input_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir} (pass --overwrite)")
        shutil.rmtree(output_dir)

    source_info_path = input_dir / "meta" / "info.json"
    source_info = _load_json(source_info_path) if source_info_path.exists() else {}
    chunk_size = int(source_info.get("chunks_size", source_info.get("chunk_size", 1000)))

    episodes_meta = _load_episode_meta(input_dir / "meta")
    tasks = _load_tasks(input_dir / "meta" / "tasks.parquet", episodes_meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_ancillary_entries(input_dir, output_dir)
    _copy_metadata_files(input_dir, output_dir)

    # Keep the original modality metadata; the output data files keep the same columns.
    modality_path = input_dir / "meta" / "modality.json"
    if modality_path.exists():
        _copy_path(modality_path, output_dir / "meta" / "modality.json")

    if (input_dir / "meta" / "embodiment.json").exists():
        _copy_path(input_dir / "meta" / "embodiment.json", output_dir / "meta" / "embodiment.json")

    # Prepare episode records.
    episode_records: dict[int, EpisodeMeta] = {}
    for _, row in episodes_meta.iterrows():
        episode_index = int(row["episode_index"])
        task_index = _as_int(row["task_index"]) if "task_index" in row else None
        task = None
        if task_index is not None:
            matched = next((t for t in tasks if _as_int(t.get("task_index")) == task_index), None)
            if matched is not None:
                task = matched.get("task")

        from_timestamps: dict[str, float] = {}
        video_file_indices: dict[str, dict[str, int]] = {}
        for col, value in row.items():
            if not isinstance(col, str) or not col.startswith("videos/"):
                continue
            if col.endswith("/from_timestamp"):
                video_key = col[len("videos/") : -len("/from_timestamp")]
                ts = _as_float(value)
                if ts is not None:
                    from_timestamps[video_key] = ts
            elif col.endswith("/chunk_index"):
                video_key = col[len("videos/") : -len("/chunk_index")]
                chunk = _as_int(value)
                file_col = f"videos/{video_key}/file_index"
                file_index = _as_int(row[file_col]) if file_col in row and not pd.isna(row[file_col]) else None
                if chunk is not None and file_index is not None:
                    video_file_indices.setdefault(video_key, {})["chunk_index"] = chunk
                    video_file_indices[video_key]["file_index"] = file_index

        episode_records[episode_index] = EpisodeMeta(
            episode_index=episode_index,
            length=_as_int(row.get("length")) or 0,
            task_index=task_index,
            task=str(task) if task is not None else None,
            data_chunk_index=_as_int(row.get("data/chunk_index")),
            data_file_index=_as_int(row.get("data/file_index")),
            data_file_from_index=_as_int(row.get("data/file_from_index")),
            from_timestamps=from_timestamps,
            video_file_indices=video_file_indices,
        )

    # Convert source data files into per-episode parquet files.
    grouped_by_source: dict[tuple[int | None, int | None], list[EpisodeMeta]] = {}
    for record in episode_records.values():
        grouped_by_source.setdefault((record.data_chunk_index, record.data_file_index), []).append(record)

    episode_lines: list[dict[str, Any]] = []
    stats_lines: list[dict[str, Any]] = []
    total_frames = 0
    total_videos = 0

    # Cache source dataframes because multiple output episodes can come from the same file.
    source_data_cache: dict[Path, pd.DataFrame] = {}

    for (chunk_index, file_index), records in sorted(
        grouped_by_source.items(), key=lambda item: (item[0][0] if item[0][0] is not None else -1, item[0][1] if item[0][1] is not None else -1)
    ):
        if chunk_index is None or file_index is None:
            continue

        source_data_path = _resolve_data_path(input_dir, source_info, chunk_index, file_index)
        if source_data_path not in source_data_cache:
            source_data_cache[source_data_path] = _read_parquet(source_data_path)
        source_df = source_data_cache[source_data_path]
        if "episode_index" not in source_df.columns:
            raise ValueError(f"Source data parquet {source_data_path} is missing episode_index column")

        records_sorted = sorted(
            records,
            key=lambda rec: (
                rec.data_file_from_index if rec.data_file_from_index is not None else rec.episode_index,
                rec.episode_index,
            ),
        )

        source_videos: dict[str, Path] = {}
        source_video_durations: dict[str, float] = {}
        for record in records_sorted:
            for video_key in record.from_timestamps:
                if video_key in source_videos:
                    continue
                source_video = _resolve_video_path(
                    input_dir,
                    source_info,
                    video_key,
                    chunk_index,
                    file_index,
                    episode_index=record.episode_index,
                )
                if source_video is not None:
                    source_videos[video_key] = source_video
                    try:
                        source_video_durations[video_key] = _probe_video_duration(source_video)
                    except Exception:
                        source_video_durations[video_key] = 0.0

        for idx, record in enumerate(records_sorted):
            episode_df = source_df.loc[source_df["episode_index"] == record.episode_index].copy()
            if episode_df.empty:
                # Some v3.0 layouts store one episode per parquet file.
                episode_df = source_df.copy()

            if record.length <= 0:
                record = EpisodeMeta(
                    episode_index=record.episode_index,
                    length=int(len(episode_df)),
                    task_index=record.task_index,
                    task=record.task,
                    data_chunk_index=record.data_chunk_index,
                    data_file_index=record.data_file_index,
                    data_file_from_index=record.data_file_from_index,
                    from_timestamps=record.from_timestamps,
                    video_file_indices=record.video_file_indices,
                )

            out_chunk = record.episode_index // chunk_size
            out_data_path = output_dir / LEGACY_DATA_PATH.format(episode_chunk=out_chunk, episode_index=record.episode_index)
            out_data_path.parent.mkdir(parents=True, exist_ok=True)
            episode_df.to_parquet(out_data_path, index=False)

            episode_line = {
                "episode_index": record.episode_index,
                "length": int(len(episode_df)),
            }
            if record.task_index is not None:
                episode_line["task_index"] = record.task_index
            if record.task is not None:
                episode_line["task"] = record.task
            if record.data_chunk_index is not None:
                episode_line["data/chunk_index"] = record.data_chunk_index
            if record.data_file_index is not None:
                episode_line["data/file_index"] = record.data_file_index
            if record.data_file_from_index is not None:
                episode_line["data/file_from_index"] = record.data_file_from_index
            if record.from_timestamps:
                episode_line["videos/from_timestamps"] = record.from_timestamps
            if record.video_file_indices:
                episode_line["videos/file_indices"] = record.video_file_indices
            episode_lines.append(episode_line)

            stats_lines.append(
                {
                    "episode_index": record.episode_index,
                    "length": int(len(episode_df)),
                    "task_index": record.task_index,
                    "data/chunk_index": record.data_chunk_index,
                    "data/file_index": record.data_file_index,
                    "data/file_from_index": record.data_file_from_index,
                    "source_data_path": str(source_data_path.relative_to(input_dir)),
                }
            )

            total_frames += int(len(episode_df))

            # Split videos for this episode when we can resolve the source mp4.
            next_record = records_sorted[idx + 1] if idx + 1 < len(records_sorted) else None
            for video_key, start_ts in record.from_timestamps.items():
                source_video = source_videos.get(video_key)
                if source_video is None:
                    continue

                next_start = None
                if next_record is not None and video_key in next_record.from_timestamps:
                    next_start = next_record.from_timestamps[video_key]

                end_ts = next_start if next_start is not None and next_start > start_ts else source_video_durations.get(video_key, 0.0)
                if end_ts <= start_ts:
                    end_ts = start_ts + max(float(len(episode_df)), 1.0)

                out_video_path = output_dir / LEGACY_VIDEO_PATH.format(
                    episode_chunk=out_chunk,
                    episode_index=record.episode_index,
                    video_key=video_key,
                )
                _trim_video(source_video, start_ts, end_ts, out_video_path)
                total_videos += 1

    _write_tasks(output_dir, tasks)
    _write_episode_lines(output_dir / "meta" / "episodes.jsonl", episode_lines)
    _write_episode_lines(output_dir / "meta" / "episodes_stats.jsonl", stats_lines)

    total_episodes = len(episode_lines)
    total_chunks = 0 if total_episodes == 0 else int(math.floor((max(episode_records) if episode_records else 0) / chunk_size) + 1)

    info = _build_v2_info(
        source_info=source_info,
        chunk_size=chunk_size,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=len(tasks),
        total_videos=total_videos,
        total_chunks=total_chunks,
    )
    _dump_json(output_dir / "meta" / "info.json", info, indent=2)

    if not (output_dir / "meta" / "stats_gr00t.json").exists():
        # Best-effort fallback: preserve the source stats if available.
        src_stats = input_dir / "meta" / "stats_gr00t.json"
        if src_stats.exists():
            _copy_path(src_stats, output_dir / "meta" / "stats_gr00t.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a LeRobot v3.0 dataset to v2.1 layout")
    parser.add_argument("--input_dir", required=True, type=Path, help="Path to the v3.0 dataset root")
    parser.add_argument("--output_dir", required=True, type=Path, help="Path to the converted v2.1 dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists")
    parser.add_argument(
        "--replace-original",
        action="store_true",
        help="Convert into a temporary sibling directory and replace the original input directory in place",
    )
    args = parser.parse_args()

    if args.replace_original:
        input_dir = args.input_dir.expanduser().resolve()
        temp_output = args.output_dir.expanduser().resolve()
        convert_dataset(input_dir=input_dir, output_dir=temp_output, overwrite=True)
        if input_dir.exists():
            shutil.rmtree(input_dir)
        temp_output.rename(input_dir)
        print(f"Converted dataset in place: {input_dir}")
        return

    convert_dataset(args.input_dir, args.output_dir, overwrite=args.overwrite)
    print(f"Converted dataset written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()