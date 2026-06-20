from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULTS: dict[str, Any] = {
    "keypoint_root": "outputs/rtmw_ntu60",
    "model_out": "outputs/models/ntu60_rtmw_softmax.npz",
    "metrics_out": "outputs/models/ntu60_rtmw_metrics.json",
    "predictions_out": "outputs/models/ntu60_rtmw_predictions.csv",
    "split": "random",
    "num_classes": 0,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.05,
    "weight_decay": 0.0001,
    "test_ratio": 0.2,
    "seed": 42,
    "score_threshold": 0.1,
    "limit": 0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight NTU60 action classifier from RTMW keypoints."
    )
    parser.add_argument("--keypoint-root", default=DEFAULTS["keypoint_root"])
    parser.add_argument("--model-out", default=DEFAULTS["model_out"])
    parser.add_argument("--metrics-out", default=DEFAULTS["metrics_out"])
    parser.add_argument("--predictions-out", default=DEFAULTS["predictions_out"])
    parser.add_argument(
        "--split",
        default=DEFAULTS["split"],
        choices=["xsub", "xview", "random", "all"],
        help="Use NTU cross-subject/cross-view metadata, random split, or all data.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=DEFAULTS["num_classes"],
        help="Class count. Use 0 to infer 60/120-style count from labels.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULTS["learning_rate"],
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULTS["weight_decay"],
    )
    parser.add_argument("--test-ratio", type=float, default=DEFAULTS["test_ratio"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULTS["score_threshold"],
        help="Keypoints below this score are ignored when building features.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULTS["limit"],
        help="Train from only the first N keypoint files. 0 means all.",
    )
    return parser


def safe_json_array_value(value: np.ndarray) -> dict[str, Any]:
    raw = value.item() if value.shape == () else value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def load_metadata(data: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
    if "metadata" in data:
        metadata = safe_json_array_value(data["metadata"])
    else:
        metadata = {}
    metadata.setdefault("source_path", str(path))
    return metadata


def discover_keypoint_files(root: str | Path, limit: int = 0) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    files = sorted(root_path.rglob("*.npz"), key=lambda item: item.as_posix().lower())
    if limit and limit > 0:
        files = files[:limit]
    return files


def weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    weights = weights.astype(np.float32)
    total = np.sum(weights, axis=axis, keepdims=True)
    total = np.maximum(total, 1.0)
    return np.sum(values * weights, axis=axis, keepdims=True) / total


def weighted_std(
    values: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    axis: int,
) -> np.ndarray:
    weights = weights.astype(np.float32)
    total = np.sum(weights, axis=axis, keepdims=True)
    total = np.maximum(total, 1.0)
    variance = np.sum(((values - mean) ** 2) * weights, axis=axis, keepdims=True)
    variance = variance / total
    return np.sqrt(np.maximum(variance, 0.0))


def normalize_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    valid = (
        np.isfinite(keypoints[..., 0])
        & np.isfinite(keypoints[..., 1])
        & (scores >= score_threshold)
    )

    if not valid.any():
        return np.zeros_like(keypoints, dtype=np.float32), valid

    valid_xy = keypoints[valid]
    center = np.mean(valid_xy, axis=0)
    scale = float(np.mean(np.std(valid_xy, axis=0)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    normalized = (keypoints - center) / scale
    normalized[~np.isfinite(normalized)] = 0.0
    normalized[~valid] = 0.0
    return normalized.astype(np.float32), valid


def feature_from_sample(
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
) -> np.ndarray:
    normalized, valid = normalize_keypoints(keypoints, scores, score_threshold)
    valid_weights = valid[..., None].astype(np.float32)

    mean = weighted_mean(normalized, valid_weights, axis=0)[0]
    std = weighted_std(normalized, valid_weights, mean[None, ...], axis=0)[0]

    if normalized.shape[0] > 1:
        velocity = normalized[1:] - normalized[:-1]
        velocity_valid = (valid[1:] & valid[:-1])[..., None].astype(np.float32)
        velocity_mean = weighted_mean(velocity, velocity_valid, axis=0)[0]
        velocity_std = weighted_std(
            velocity,
            velocity_valid,
            velocity_mean[None, ...],
            axis=0,
        )[0]
    else:
        velocity_mean = np.zeros_like(mean)
        velocity_std = np.zeros_like(std)

    clipped_scores = np.clip(scores, 0.0, 1.0)
    score_mean = np.mean(clipped_scores, axis=0)
    score_std = np.std(clipped_scores, axis=0)

    feature_parts = [
        mean.reshape(-1),
        std.reshape(-1),
        velocity_mean.reshape(-1),
        velocity_std.reshape(-1),
        score_mean.reshape(-1),
        score_std.reshape(-1),
    ]
    return np.concatenate(feature_parts).astype(np.float32)


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, Any]] = []

    for path in discover_keypoint_files(args.keypoint_root, args.limit):
        try:
            with np.load(path, allow_pickle=False) as data:
                if "keypoints" not in data or "scores" not in data:
                    continue
                metadata = load_metadata(data, path)
                label = metadata.get("label")
                if label is None:
                    continue
                label = int(label)
                if label < 0:
                    continue
                if args.num_classes > 0 and label >= args.num_classes:
                    continue
                feature = feature_from_sample(
                    data["keypoints"],
                    data["scores"],
                    args.score_threshold,
                )
        except Exception as exc:
            print(f"Skipping unreadable keypoint file {path}: {exc}")
            continue

        metadata["keypoint_path"] = str(path)
        features.append(feature)
        labels.append(label)
        metadata_rows.append(metadata)

    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    return np.stack(features).astype(np.float32), np.asarray(labels), metadata_rows


def random_split(
    labels: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices.tolist())
            continue
        test_count = max(1, int(round(len(indices) * test_ratio)))
        test_count = min(test_count, len(indices) - 1)
        test_indices.extend(indices[:test_count].tolist())
        train_indices.extend(indices[test_count:].tolist())

    return np.asarray(train_indices, dtype=np.int64), np.asarray(test_indices, dtype=np.int64)


def split_dataset(
    labels: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    split: str,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if split == "all":
        indices = np.arange(len(labels), dtype=np.int64)
        return indices, np.empty((0,), dtype=np.int64)

    if split in {"xsub", "xview"}:
        key = "xsub_split" if split == "xsub" else "xview_split"
        train = [
            index
            for index, row in enumerate(metadata_rows)
            if row.get(key) == "train"
        ]
        test = [
            index
            for index, row in enumerate(metadata_rows)
            if row.get(key) == "test"
        ]
        if train and test:
            return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
        print(f"Split metadata for {split} is incomplete; falling back to random.")

    return random_split(labels, test_ratio, seed)


def standardize_train_test(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features[train_indices], axis=0)
    std = np.std(features[train_indices], axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32), mean, std


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def accuracy(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> float:
    if features.shape[0] == 0:
        return 0.0
    predictions = np.argmax(features @ weights + bias, axis=1)
    return float(np.mean(predictions == labels))


def cross_entropy_loss(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    weight_decay: float,
) -> float:
    if features.shape[0] == 0:
        return 0.0
    probs = softmax(features @ weights + bias)
    loss = -np.log(probs[np.arange(features.shape[0]), labels] + 1e-12).mean()
    loss += 0.5 * weight_decay * float(np.sum(weights * weights))
    return float(loss)


def train_softmax(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(args.seed)
    feature_dim = features.shape[1]
    weights = rng.normal(
        loc=0.0,
        scale=0.01,
        size=(feature_dim, args.num_classes),
    ).astype(np.float32)
    bias = np.zeros((args.num_classes,), dtype=np.float32)

    history: list[dict[str, float]] = []
    batch_size = max(1, args.batch_size)
    learning_rate = float(args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            batch_features = features[batch_indices]
            batch_labels = labels[batch_indices]

            probs = softmax(batch_features @ weights + bias)
            probs[np.arange(batch_features.shape[0]), batch_labels] -= 1.0
            probs /= batch_features.shape[0]

            grad_weights = batch_features.T @ probs + args.weight_decay * weights
            grad_bias = np.sum(probs, axis=0)
            weights -= learning_rate * grad_weights.astype(np.float32)
            bias -= learning_rate * grad_bias.astype(np.float32)

        row = {
            "epoch": float(epoch),
            "train_loss": cross_entropy_loss(
                features[train_indices],
                labels[train_indices],
                weights,
                bias,
                args.weight_decay,
            ),
            "train_acc": accuracy(
                features[train_indices],
                labels[train_indices],
                weights,
                bias,
            ),
            "test_acc": accuracy(
                features[test_indices],
                labels[test_indices],
                weights,
                bias,
            )
            if len(test_indices)
            else 0.0,
        }
        history.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            print(
                "epoch "
                f"{epoch:03d}/{args.epochs} "
                f"loss={row['train_loss']:.4f} "
                f"train_acc={row['train_acc']:.3f} "
                f"test_acc={row['test_acc']:.3f}"
            )

    return weights, bias, history


def write_predictions(
    path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    weights: np.ndarray,
    bias: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    split_by_index = {int(index): "train" for index in train_indices}
    split_by_index.update({int(index): "test" for index in test_indices})
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = np.argmax(features @ weights + bias, axis=1)
    fields = ["split", "label", "prediction", "correct", "keypoint_path", "source_path"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(metadata_rows):
            writer.writerow(
                {
                    "split": split_by_index.get(index, ""),
                    "label": int(labels[index]),
                    "prediction": int(predictions[index]),
                    "correct": int(predictions[index] == labels[index]),
                    "keypoint_path": row.get("keypoint_path", ""),
                    "source_path": row.get("source_path", ""),
                }
            )


def run(args: argparse.Namespace) -> int:
    features, labels, metadata_rows = load_dataset(args)
    if features.shape[0] == 0:
        print(f"No RTMW keypoint .npz files found under {args.keypoint_root}.")
        print("Run extraction first, or put extracted keypoints in that directory.")
        return 0

    if args.num_classes <= 0:
        args.num_classes = int(labels.max()) + 1
        if args.num_classes <= 60:
            args.num_classes = 60
        elif args.num_classes <= 120:
            args.num_classes = 120
        print(f"Inferred num_classes={args.num_classes}")

    train_indices, test_indices = split_dataset(
        labels,
        metadata_rows,
        args.split,
        args.test_ratio,
        args.seed,
    )
    if len(train_indices) == 0:
        print("No training samples found after splitting. Nothing to train.")
        return 0

    features, feature_mean, feature_std = standardize_train_test(features, train_indices)
    print(
        f"Training samples: {len(train_indices)}, "
        f"test samples: {len(test_indices)}, "
        f"feature_dim: {features.shape[1]}"
    )

    weights, bias, history = train_softmax(
        features,
        labels,
        train_indices,
        test_indices,
        args,
    )

    metrics = {
        "samples": int(features.shape[0]),
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "feature_dim": int(features.shape[1]),
        "num_classes": int(args.num_classes),
        "split": args.split,
        "epochs": int(args.epochs),
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "history": history,
    }

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        model_path,
        weights=weights,
        bias=bias,
        feature_mean=feature_mean,
        feature_std=feature_std,
        num_classes=np.asarray(args.num_classes, dtype=np.int32),
        config=np.asarray(json.dumps(vars(args), ensure_ascii=True)),
        metrics=np.asarray(json.dumps(metrics, ensure_ascii=True)),
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_predictions(
        args.predictions_out,
        features,
        labels,
        metadata_rows,
        weights,
        bias,
        train_indices,
        test_indices,
    )

    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {args.predictions_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
