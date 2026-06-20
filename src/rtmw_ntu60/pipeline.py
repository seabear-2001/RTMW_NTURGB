from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULT_MANIFEST = "manifests/ntu60_rgb.json"
EXAMPLE_MANIFEST = "manifests/ntu60_rgb.example.json"


def repo_root() -> Path:
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "requirements.txt").exists():
        return package_root
    return Path.cwd()


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def is_default_manifest(args: argparse.Namespace) -> bool:
    manifest = str(args.manifest).replace("\\", "/")
    return manifest == DEFAULT_MANIFEST


def has_video_files(data_root: str | Path, pattern: str) -> bool:
    root = Path(data_root)
    if not root.exists():
        return False
    return any(
        path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
        for path in root.glob(pattern)
    )


def has_keypoint_files(keypoint_root: str | Path) -> bool:
    root = Path(keypoint_root)
    return root.exists() and any(path.is_file() for path in root.rglob("*.npz"))


def ensure_default_manifest_template(manifest: Path) -> None:
    example = resolve_project_path(EXAMPLE_MANIFEST)
    if manifest.exists() or example is None or not example.exists():
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(example, manifest)
    print(f"Created manifest template: {manifest}")


def env_ntu_urls() -> list[str]:
    value = os.environ.get("NTU60_RGB_URLS", "").strip()
    if not value:
        return []
    urls: list[str] = []
    for line in value.replace(";", "\n").splitlines():
        url = line.strip()
        if url:
            urls.append(url)
    return urls


def write_manifest_from_env(manifest: Path) -> bool:
    urls = env_ntu_urls()
    if not urls:
        return False

    manifest.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "description": "Generated from NTU60_RGB_URLS by main.py.",
        "files": [
            {
                "url": url,
                "sha256": "",
                "extract": True,
            }
            for url in urls
        ],
    }
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Generated manifest from NTU60_RGB_URLS: {manifest}")
    return True


def print_manifest_placeholder_hint(manifest: Path, data_root: str | Path) -> None:
    print("")
    print("NTU60 download is not configured yet.")
    print(f"  Manifest: {manifest}")
    print("  Reason: it still contains PUT_AUTHORIZED_DOWNLOAD_URL_HERE.")
    print("")
    print("Next step for automatic download:")
    print(f"  Option A: Open {manifest}")
    print("            Replace PUT_AUTHORIZED_DOWNLOAD_URL_HERE with an authorized NTU60 URL")
    print("            Run python main.py again")
    print("  Option B: Set environment variable NTU60_RGB_URLS to authorized URL(s)")
    print("            Run python main.py again")
    print("")
    print("If you already have NTU60 RGB videos:")
    print(f"  Put *_rgb.avi files under {data_root}")
    print("  Then run python main.py --skip-download")
    print("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download NTU60 archives, extract them, then run RTMW keypoints."
    )

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Download manifest with authorized NTU60 archive URLs.",
    )
    pipeline.add_argument(
        "--download-dir",
        default="data/downloads/ntu60",
        help="Directory for downloaded NTU60 archives.",
    )
    pipeline.add_argument(
        "--data-root",
        default="data/ntu60",
        help="Directory where NTU60 videos are extracted or already located.",
    )
    pipeline.add_argument(
        "--output-root",
        default="outputs/rtmw_ntu60",
        help="Directory for RTMW .npz outputs.",
    )
    pipeline.add_argument(
        "--install-deps",
        action="store_true",
        help="Force installing requirements.txt before running the pipeline.",
    )
    pipeline.add_argument(
        "--no-auto-install",
        action="store_true",
        help="Do not auto-install missing Python dependencies.",
    )
    pipeline.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip manifest download/extraction and only run RTMW extraction.",
    )
    pipeline.add_argument(
        "--download-only",
        action="store_true",
        help="Download/extract NTU60 archives, then stop before RTMW extraction.",
    )
    pipeline.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip RTMW extraction and train from existing keypoint .npz files.",
    )
    pipeline.add_argument(
        "--skip-train",
        action="store_true",
        help="Stop after download/extraction and do not train.",
    )
    pipeline.add_argument(
        "--train-only",
        action="store_true",
        help="Skip download and extraction, then train from existing keypoints.",
    )

    download = parser.add_argument_group("download")
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--no-resume", action="store_true")
    download.add_argument(
        "--no-archive-extract",
        action="store_true",
        help="Download archives but do not unpack them into --data-root.",
    )
    download.add_argument("--timeout", type=int, default=60)

    extract = parser.add_argument_group("rtmw extraction")
    extract.add_argument("--pattern", default="**/*_rgb.avi")
    extract.add_argument("--backend", default="onnxruntime")
    extract.add_argument("--device", default="cpu")
    extract.add_argument(
        "--mode",
        default="balanced",
        choices=["lightweight", "balanced", "performance"],
    )
    extract.add_argument("--det", default=None)
    extract.add_argument("--pose", default=None)
    extract.add_argument("--to-openpose", action="store_true")
    extract.add_argument("--max-persons", type=int, default=2)
    extract.add_argument("--num-keypoints", type=int, default=0)
    extract.add_argument("--score-threshold", type=float, default=0.3)
    extract.add_argument("--every-n", type=int, default=1)
    extract.add_argument("--limit", type=int, default=0)
    extract.add_argument(
        "--summary",
        default=None,
        help="CSV summary path. Default: <output-root>/index.csv.",
    )
    extract.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=True,
    )
    extract.add_argument("--fail-fast", action="store_true")

    train = parser.add_argument_group("training")
    train.add_argument(
        "--model-out",
        default="outputs/models/ntu60_rtmw_softmax.npz",
        help="Path for the trained lightweight classifier.",
    )
    train.add_argument(
        "--metrics-out",
        default="outputs/models/ntu60_rtmw_metrics.json",
        help="Path for training metrics JSON.",
    )
    train.add_argument(
        "--predictions-out",
        default="outputs/models/ntu60_rtmw_predictions.csv",
        help="Path for train/test prediction CSV.",
    )
    train.add_argument(
        "--train-split",
        default="random",
        choices=["xsub", "xview", "random", "all"],
        help="Training split for the lightweight classifier.",
    )
    train.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Class count. Use 0 to infer 60 or 120 from extracted labels.",
    )
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--test-ratio", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--train-score-threshold",
        type=float,
        default=0.1,
        help="Ignore keypoints below this score when building train features.",
    )
    train.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Train from only the first N keypoint files. 0 means all.",
    )
    return parser


def install_dependencies() -> None:
    root = repo_root()
    requirements = root / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(
            f"requirements.txt not found at {requirements}. "
            "Run without --install-deps or execute from the project root."
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )


def missing_runtime_packages() -> list[str]:
    package_imports = {
        "gdown": "gdown",
        "numpy": "numpy",
        "opencv-python": "cv2",
        "onnxruntime": "onnxruntime",
        "requests": "requests",
        "rtmlib": "rtmlib",
        "tqdm": "tqdm",
    }
    missing: list[str] = []
    for package, module_name in package_imports.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package)
    return missing


def ensure_runtime_dependencies(args: argparse.Namespace) -> None:
    if args.no_auto_install:
        return
    missing = missing_runtime_packages()
    if not missing:
        return
    print(f"Missing Python dependencies detected: {', '.join(missing)}")
    print("Installing requirements.txt before continuing...")
    install_dependencies()


def run_download(args: argparse.Namespace) -> int:
    from . import download

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Copy "
            f"{EXAMPLE_MANIFEST} to {DEFAULT_MANIFEST} "
            "and add authorized NTU60 download URLs."
        )

    download_args = argparse.Namespace(
        manifest=args.manifest,
        download_dir=args.download_dir,
        extract_dir=args.data_root,
        overwrite=args.overwrite,
        no_resume=args.no_resume,
        no_extract=args.no_archive_extract,
        timeout=args.timeout,
    )
    return download.run(download_args)


def should_download(args: argparse.Namespace) -> bool:
    from . import download

    manifest = Path(args.manifest)
    default_manifest = bool(getattr(args, "default_manifest", is_default_manifest(args)))
    if not manifest.exists():
        if default_manifest and write_manifest_from_env(manifest):
            return True
        if default_manifest and not args.download_only:
            ensure_default_manifest_template(manifest)
            print_manifest_placeholder_hint(manifest, args.data_root)
            return False
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Copy "
            f"{EXAMPLE_MANIFEST} to {DEFAULT_MANIFEST} "
            "and add authorized NTU60 download URLs."
        )

    try:
        entries = download.load_manifest(manifest)
        download.validate_manifest(entries)
    except ValueError as exc:
        if default_manifest and write_manifest_from_env(manifest):
            return True
        if default_manifest and not args.download_only:
            print_manifest_placeholder_hint(manifest, args.data_root)
            return False
        raise exc

    return True


def run_keypoint_extract(args: argparse.Namespace) -> int:
    from . import extract

    extract_args = argparse.Namespace(
        data_root=args.data_root,
        output_root=args.output_root,
        pattern=args.pattern,
        backend=args.backend,
        device=args.device,
        mode=args.mode,
        det=args.det,
        pose=args.pose,
        to_openpose=args.to_openpose,
        max_persons=args.max_persons,
        num_keypoints=args.num_keypoints,
        score_threshold=args.score_threshold,
        every_n=args.every_n,
        limit=args.limit,
        summary=args.summary,
        skip_existing=args.skip_existing,
        fail_fast=args.fail_fast,
    )
    return extract.run(extract_args)


def run_train(args: argparse.Namespace) -> int:
    from . import train

    train_args = argparse.Namespace(
        keypoint_root=args.output_root,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        predictions_out=args.predictions_out,
        split=args.train_split,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        test_ratio=args.test_ratio,
        seed=args.seed,
        score_threshold=args.train_score_threshold,
        limit=args.train_limit,
    )
    return train.run(train_args)


def run(args: argparse.Namespace) -> int:
    args.default_manifest = is_default_manifest(args)
    args.manifest = str(resolve_project_path(args.manifest))
    args.download_dir = str(resolve_project_path(args.download_dir))
    args.data_root = str(resolve_project_path(args.data_root))
    args.output_root = str(resolve_project_path(args.output_root))
    args.model_out = str(resolve_project_path(args.model_out))
    args.metrics_out = str(resolve_project_path(args.metrics_out))
    args.predictions_out = str(resolve_project_path(args.predictions_out))
    if args.summary:
        args.summary = str(resolve_project_path(args.summary))

    if args.install_deps:
        install_dependencies()
    else:
        ensure_runtime_dependencies(args)

    if args.skip_download and args.download_only:
        raise ValueError("--skip-download and --download-only cannot be used together.")
    if args.train_only and args.download_only:
        raise ValueError("--train-only and --download-only cannot be used together.")

    downloaded = False
    if not args.skip_download and not args.train_only and should_download(args):
        result = run_download(args)
        if result != 0:
            return result
        downloaded = True

    if args.download_only:
        if not downloaded:
            raise ValueError(
                "Download-only mode needs a real manifest. Fill "
                f"{args.manifest} with authorized NTU60 URLs first."
            )
        return 0

    if args.train_only:
        return run_train(args)

    videos_available = has_video_files(args.data_root, args.pattern)
    if not args.skip_extract and videos_available:
        result = run_keypoint_extract(args)
        if result != 0:
            return result
    elif not args.skip_extract and not videos_available:
        print(
            f"No local NTU60 RGB videos found under {args.data_root} "
            f"with pattern {args.pattern}. Nothing to extract yet."
        )
    elif args.skip_extract:
        print("Skipping RTMW extraction; training will use existing keypoint files.")

    if args.skip_train:
        return 0

    if not has_keypoint_files(args.output_root):
        print(
            f"No RTMW keypoint .npz files found under {args.output_root}. "
            "Nothing to train yet."
        )
        return 0

    return run_train(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
