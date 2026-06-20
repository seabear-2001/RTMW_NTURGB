from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .ntu import iter_video_files, output_path_for_video, parse_ntu_filename


DEFAULTS: dict[str, Any] = {
    "data_root": "data/ntu60",
    "output_root": "outputs/rtmw_ntu60",
    "pattern": "**/*_rgb.avi",
    "backend": "onnxruntime",
    "device": "cpu",
    "mode": "balanced",
    "to_openpose": False,
    "max_persons": 2,
    "score_threshold": 0.3,
    "every_n": 1,
    "limit": 0,
    "skip_existing": True,
    "summary": None,
    "num_keypoints": 0,
}


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data or {}

    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def build_arg_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract RTMW whole-body keypoints from NTU60 RGB videos."
    )
    parser.add_argument("--config", default=None, help="Optional JSON/YAML config.")
    parser.add_argument("--data-root", default=defaults["data_root"])
    parser.add_argument("--output-root", default=defaults["output_root"])
    parser.add_argument("--pattern", default=defaults["pattern"])
    parser.add_argument("--backend", default=defaults["backend"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument(
        "--mode",
        default=defaults["mode"],
        choices=["lightweight", "balanced", "performance"],
    )
    parser.add_argument("--det", default=defaults.get("det"))
    parser.add_argument("--pose", default=defaults.get("pose"))
    parser.add_argument(
        "--to-openpose",
        action="store_true",
        default=bool(defaults.get("to_openpose", False)),
        help="Use OpenPose-style whole-body keypoint ordering.",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=int(defaults["max_persons"]),
        help="Fixed person slots saved per frame.",
    )
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=int(defaults.get("num_keypoints", 0)),
        help="Override keypoint count. Default: 133, or 134 with --to-openpose.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=float(defaults["score_threshold"]),
        help="Score threshold used for person ranking and centers.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=int(defaults["every_n"]),
        help="Process every nth frame.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(defaults.get("limit", 0)),
        help="Process only the first N videos. 0 means all.",
    )
    parser.add_argument(
        "--summary",
        default=defaults.get("summary"),
        help="CSV summary path. Default: <output-root>/index.csv.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=bool(defaults.get("skip_existing", True)),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed video.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    known, _ = pre_parser.parse_known_args(argv)

    defaults = dict(DEFAULTS)
    defaults.update(load_config(known.config))

    parser = build_arg_parser(defaults)
    return parser.parse_args(argv)


def build_model(args: argparse.Namespace):
    try:
        from rtmlib import Wholebody
    except ImportError as exc:
        raise RuntimeError(
            "rtmlib is required. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    kwargs: dict[str, Any] = {
        "mode": args.mode,
        "to_openpose": args.to_openpose,
        "backend": args.backend,
        "device": args.device,
    }
    if args.det:
        kwargs["det"] = args.det
    if args.pose:
        kwargs["pose"] = args.pose

    return Wholebody(**kwargs)


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return cv2


def score_people(scores: np.ndarray, threshold: float) -> np.ndarray:
    if scores.size == 0:
        return np.empty((0,), dtype=np.float32)
    visible = np.where(scores >= threshold, scores, np.nan)
    mean_scores = np.nanmean(visible, axis=1)
    fallback = np.mean(scores, axis=1)
    return np.where(np.isnan(mean_scores), fallback, mean_scores)


def person_centers(
    keypoints: np.ndarray, scores: np.ndarray, threshold: float
) -> np.ndarray:
    centers = np.full((keypoints.shape[0], 2), np.nan, dtype=np.float32)
    for person_idx in range(keypoints.shape[0]):
        valid = (
            np.isfinite(keypoints[person_idx, :, 0])
            & np.isfinite(keypoints[person_idx, :, 1])
            & (scores[person_idx] >= threshold)
        )
        if valid.any():
            centers[person_idx] = keypoints[person_idx, valid, :2].mean(axis=0)
    return centers


class PersonSlotter:
    def __init__(self, max_persons: int, score_threshold: float) -> None:
        self.max_persons = max_persons
        self.score_threshold = score_threshold
        self.previous_centers = np.full((max_persons, 2), np.nan, dtype=np.float32)

    def assign(self, keypoints: np.ndarray, scores: np.ndarray) -> list[int | None]:
        if keypoints.size == 0 or keypoints.shape[0] == 0:
            self.previous_centers[:] = np.nan
            return [None] * self.max_persons

        person_scores = score_people(scores, self.score_threshold)
        centers = person_centers(keypoints, scores, self.score_threshold)
        assignments: list[int | None] = [None] * self.max_persons
        used: set[int] = set()

        for slot_idx, previous in enumerate(self.previous_centers):
            if not np.isfinite(previous).all():
                continue
            distances = np.linalg.norm(centers - previous, axis=1)
            ranked = sorted(
                (
                    (distance, idx)
                    for idx, distance in enumerate(distances)
                    if idx not in used and np.isfinite(distance)
                ),
                key=lambda item: item[0],
            )
            if ranked:
                _, person_idx = ranked[0]
                assignments[slot_idx] = person_idx
                used.add(person_idx)

        remaining = sorted(
            (idx for idx in range(keypoints.shape[0]) if idx not in used),
            key=lambda idx: float(person_scores[idx]),
            reverse=True,
        )
        for slot_idx, value in enumerate(assignments):
            if value is not None or not remaining:
                continue
            assignments[slot_idx] = remaining.pop(0)

        self.previous_centers[:] = np.nan
        for slot_idx, person_idx in enumerate(assignments):
            if person_idx is not None and np.isfinite(centers[person_idx]).all():
                self.previous_centers[slot_idx] = centers[person_idx]

        return assignments


def normalize_model_output(
    keypoints: Any, scores: Any
) -> tuple[np.ndarray, np.ndarray]:
    keypoints_array = np.asarray(keypoints, dtype=np.float32)
    scores_array = np.asarray(scores, dtype=np.float32)

    if keypoints_array.size == 0:
        return (
            np.empty((0, 0, 2), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )

    if keypoints_array.ndim == 2:
        keypoints_array = keypoints_array[None, ...]
    if scores_array.ndim == 1:
        scores_array = scores_array[None, ...]

    if keypoints_array.ndim != 3 or keypoints_array.shape[-1] < 2:
        raise ValueError(f"Unexpected keypoint shape: {keypoints_array.shape}")
    if scores_array.ndim != 2:
        raise ValueError(f"Unexpected score shape: {scores_array.shape}")

    return keypoints_array[..., :2], scores_array


def default_keypoint_count(args: argparse.Namespace) -> int:
    if args.num_keypoints > 0:
        return args.num_keypoints
    return 134 if args.to_openpose else 133


def extract_video(
    video_path: Path,
    output_path: Path,
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cv2 = import_cv2()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    num_keypoints = default_keypoint_count(args)
    slotter = PersonSlotter(args.max_persons, args.score_threshold)

    keypoint_frames: list[np.ndarray] = []
    score_frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    raw_keypoint_counts: set[int] = set()
    processed = 0
    frame_idx = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx % args.every_n != 0:
            continue

        keypoints, scores = model(frame)
        keypoints_array, scores_array = normalize_model_output(keypoints, scores)
        if keypoints_array.shape[1] > 0:
            raw_keypoint_counts.add(int(keypoints_array.shape[1]))

        frame_keypoints = np.full(
            (args.max_persons, num_keypoints, 2), np.nan, dtype=np.float32
        )
        frame_scores = np.zeros(
            (args.max_persons, num_keypoints), dtype=np.float32
        )

        assignments = slotter.assign(keypoints_array, scores_array)
        for slot_idx, person_idx in enumerate(assignments):
            if person_idx is None:
                continue
            copy_count = min(num_keypoints, keypoints_array.shape[1])
            frame_keypoints[slot_idx, :copy_count] = keypoints_array[
                person_idx, :copy_count, :2
            ]
            frame_scores[slot_idx, :copy_count] = scores_array[
                person_idx, :copy_count
            ]

        keypoint_frames.append(frame_keypoints)
        score_frames.append(frame_scores)
        frame_indices.append(frame_idx)
        processed += 1

    capture.release()

    if keypoint_frames:
        keypoint_data = np.stack(keypoint_frames, axis=0)
        score_data = np.stack(score_frames, axis=0)
    else:
        keypoint_data = np.empty(
            (0, args.max_persons, num_keypoints, 2), dtype=np.float32
        )
        score_data = np.empty(
            (0, args.max_persons, num_keypoints), dtype=np.float32
        )

    metadata = parse_ntu_filename(video_path)
    metadata.update(
        {
            "source_path": str(video_path),
            "output_path": str(output_path),
            "total_video_frames": total_frames,
            "processed_frames": processed,
            "fps": fps,
            "width": width,
            "height": height,
            "raw_keypoint_counts": sorted(raw_keypoint_counts),
        }
    )

    runtime_config = {
        "backend": args.backend,
        "device": args.device,
        "mode": args.mode,
        "to_openpose": args.to_openpose,
        "max_persons": args.max_persons,
        "num_keypoints": num_keypoints,
        "score_threshold": args.score_threshold,
        "every_n": args.every_n,
        "det": args.det,
        "pose": args.pose,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        keypoints=keypoint_data,
        scores=score_data,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        frame_shape=np.asarray([height, width], dtype=np.int32),
        source_path=np.asarray(str(video_path)),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=True)),
        config=np.asarray(json.dumps(runtime_config, ensure_ascii=True)),
    )

    return {
        "status": "ok",
        "video_path": str(video_path),
        "output_path": str(output_path),
        "frames": processed,
        "total_video_frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "persons": args.max_persons,
        "keypoints": num_keypoints,
        **{
            key: metadata.get(key, "")
            for key in (
                "setup",
                "camera",
                "subject",
                "replication",
                "action",
                "label",
                "xsub_split",
                "xview_split",
            )
        },
        "error": "",
    }


def summary_fields() -> list[str]:
    return [
        "status",
        "video_path",
        "output_path",
        "frames",
        "total_video_frames",
        "width",
        "height",
        "fps",
        "persons",
        "keypoints",
        "setup",
        "camera",
        "subject",
        "replication",
        "action",
        "label",
        "xsub_split",
        "xview_split",
        "error",
    ]


def get_progress(iterable: list[Path]):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc="Extracting RTMW", unit="video")


def run(args: argparse.Namespace) -> int:
    if args.every_n < 1:
        raise ValueError("--every-n must be >= 1")
    if args.max_persons < 1:
        raise ValueError("--max-persons must be >= 1")

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    summary_path = Path(args.summary) if args.summary else output_root / "index.csv"

    videos = iter_video_files(data_root, args.pattern)
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if not videos:
        print(f"No videos found under {data_root} with pattern {args.pattern}")
        return 0

    model = build_model(args)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields())
        writer.writeheader()

        for video_path in get_progress(videos):
            out_path = output_path_for_video(video_path, data_root, output_root)
            if args.skip_existing and out_path.exists():
                metadata = parse_ntu_filename(video_path)
                row = {
                    "status": "skipped",
                    "video_path": str(video_path),
                    "output_path": str(out_path),
                    "frames": "",
                    "total_video_frames": "",
                    "width": "",
                    "height": "",
                    "fps": "",
                    "persons": args.max_persons,
                    "keypoints": default_keypoint_count(args),
                    "error": "",
                }
                row.update(
                    {
                        key: metadata.get(key, "")
                        for key in (
                            "setup",
                            "camera",
                            "subject",
                            "replication",
                            "action",
                            "label",
                            "xsub_split",
                            "xview_split",
                        )
                    }
                )
                writer.writerow(row)
                handle.flush()
                continue

            try:
                row = extract_video(video_path, out_path, model, args)
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = {
                    "status": "error",
                    "video_path": str(video_path),
                    "output_path": str(out_path),
                    "frames": "",
                    "total_video_frames": "",
                    "width": "",
                    "height": "",
                    "fps": "",
                    "persons": args.max_persons,
                    "keypoints": default_keypoint_count(args),
                    "setup": "",
                    "camera": "",
                    "subject": "",
                    "replication": "",
                    "action": "",
                    "label": "",
                    "xsub_split": "",
                    "xview_split": "",
                    "error": str(exc),
                }
            writer.writerow(row)
            handle.flush()

    print(f"Wrote summary: {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)
