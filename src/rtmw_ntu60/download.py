from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Any
from urllib.parse import unquote, urlparse
import zipfile


PLACEHOLDER_MARKERS = {
    "PUT_AUTHORIZED_DOWNLOAD_URL_HERE",
    "https://example.com/authorized/nturgbd_rgb_s001.zip",
}

GDRIVE_FILE_RE = re.compile(r"/file/d/([^/]+)")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix.lower() == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = data if isinstance(data, list) else data.get("files")
        if not isinstance(files, list):
            raise ValueError("Manifest JSON must contain a 'files' list.")
        return [dict(item) for item in files]

    entries: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append({"url": stripped, "extract": True})
    return entries


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    if "drive.google.com" in parsed.netloc:
        file_id = gdrive_id_from_url(url)
        if file_id:
            return f"{file_id}.download"
    name = Path(unquote(parsed.path)).name
    if not name:
        raise ValueError(f"Could not infer filename from URL: {url}")
    return name


def target_name(entry: dict[str, Any]) -> str:
    if entry.get("name"):
        return str(entry["name"])
    if entry.get("url"):
        return filename_from_url(str(entry["url"]))
    if entry.get("gdrive_id"):
        return f"{entry['gdrive_id']}.download"
    raise ValueError("Manifest entry must include 'name', 'url', or 'gdrive_id'.")


def gdrive_id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return None
    match = GDRIVE_FILE_RE.search(parsed.path)
    if match:
        return match.group(1)
    if parsed.path.endswith("/uc"):
        from urllib.parse import parse_qs

        query = parse_qs(parsed.query)
        ids = query.get("id")
        if ids:
            return ids[0]
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def validate_manifest(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        url = str(entry.get("url", ""))
        if url in PLACEHOLDER_MARKERS or "PUT_AUTHORIZED" in url:
            raise ValueError(
                "Manifest still contains placeholder URLs. Copy the example "
                "manifest and replace placeholders with authorized NTU60 links."
            )


def progress_bar(total: int | None):
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True,
    )


def download_http(
    url: str,
    target: Path,
    overwrite: bool,
    resume: bool,
    timeout: int,
) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "HTTP downloads require requests. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_suffix(target.suffix + ".part")

    if target.exists() and not overwrite:
        print(f"Already downloaded: {target}")
        return target

    headers: dict[str, str] = {}
    mode = "wb"
    existing_size = 0
    if resume and part_path.exists() and not overwrite:
        existing_size = part_path.stat().st_size
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=timeout) as response:
        if response.status_code == 416:
            part_path.replace(target)
            return target
        response.raise_for_status()

        if existing_size and response.status_code != 206:
            existing_size = 0
            mode = "wb"

        content_length = response.headers.get("Content-Length")
        total = int(content_length) + existing_size if content_length else None
        bar = progress_bar(total)
        if bar is not None and existing_size:
            bar.update(existing_size)

        with part_path.open(mode + "") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))

        if bar is not None:
            bar.close()

    part_path.replace(target)
    return target


def download_gdrive(entry: dict[str, Any], target: Path, overwrite: bool) -> Path:
    if target.exists() and not overwrite:
        print(f"Already downloaded: {target}")
        return target

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive entries require gdown. Install with: "
            "python -m pip install gdown"
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    file_id = entry.get("gdrive_id") or gdrive_id_from_url(str(entry.get("url", "")))
    url = entry.get("url")
    if file_id:
        result = gdown.download(id=str(file_id), output=str(target), quiet=False)
    else:
        result = gdown.download(url=str(url), output=str(target), quiet=False)
    if result is None:
        raise RuntimeError(f"Google Drive download failed: {target}")
    return target


def safe_target(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    base_resolved = base.resolve()
    try:
        target.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes target directory: {name}")
    return target


def extract_zip(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            safe_target(destination, member.filename)
        archive.extractall(destination)


def extract_tar(path: Path, destination: Path) -> None:
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            safe_target(destination, member.name)
        archive.extractall(destination)


def extract_with_7z(path: Path, destination: Path) -> None:
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if not seven_zip:
        raise RuntimeError(
            f"{path.suffix} extraction requires 7z/7za on PATH: {path}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [seven_zip, "x", str(path), f"-o{destination}", "-y"],
        check=True,
    )


def extract_archive(path: Path, destination: Path) -> None:
    suffixes = "".join(path.suffixes).lower()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {path} -> {destination}")

    if suffixes.endswith(".zip"):
        extract_zip(path, destination)
    elif suffixes.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        extract_tar(path, destination)
    elif suffixes.endswith((".7z", ".rar")):
        extract_with_7z(path, destination)
    else:
        print(f"Skipping extraction for unsupported archive type: {path}")


def should_extract(entry: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.no_extract:
        return False
    return bool(entry.get("extract", True))


def download_entry(entry: dict[str, Any], args: argparse.Namespace) -> Path:
    download_dir = Path(args.download_dir)
    target = download_dir / target_name(entry)

    if entry.get("gdrive_id") or "drive.google.com" in str(entry.get("url", "")):
        path = download_gdrive(entry, target, args.overwrite)
    else:
        url = entry.get("url")
        if not url:
            raise ValueError(f"Entry has no URL: {entry}")
        path = download_http(
            str(url),
            target,
            overwrite=args.overwrite,
            resume=not args.no_resume,
            timeout=args.timeout,
        )

    verify_checksum(path, entry.get("sha256") or None)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and extract authorized NTU RGB+D 60 archives."
    )
    parser.add_argument("--manifest", required=True, help="JSON or text manifest.")
    parser.add_argument(
        "--download-dir",
        default="data/downloads/ntu60",
        help="Directory for downloaded archives.",
    )
    parser.add_argument(
        "--extract-dir",
        default="data/ntu60",
        help="Directory for extracted data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    return parser


def run(args: argparse.Namespace) -> int:
    entries = load_manifest(args.manifest)
    validate_manifest(entries)
    if not entries:
        print("Manifest has no files.")
        return 0

    for index, entry in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] {target_name(entry)}")
        archive_path = download_entry(entry, args)
        if should_extract(entry, args):
            extract_archive(archive_path, Path(args.extract_dir))

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
