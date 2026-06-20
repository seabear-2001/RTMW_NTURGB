from __future__ import annotations

from pathlib import Path
import re
from typing import Any


NTU_FILENAME_RE = re.compile(
    r"S(?P<setup>\d{3})"
    r"C(?P<camera>\d{3})"
    r"P(?P<subject>\d{3})"
    r"R(?P<replication>\d{3})"
    r"A(?P<action>\d{3})"
    r"(?:_(?P<modality>[A-Za-z0-9]+))?",
    re.IGNORECASE,
)

NTU60_XSUB_TRAIN_SUBJECTS = {
    1,
    2,
    4,
    5,
    8,
    9,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    25,
    27,
    28,
    31,
    34,
    35,
    38,
}

NTU60_XVIEW_TRAIN_CAMERAS = {2, 3}


def parse_ntu_filename(path: str | Path) -> dict[str, Any]:
    """Parse standard NTU RGB+D filename metadata."""
    name = Path(path).stem
    match = NTU_FILENAME_RE.search(name)
    if match is None:
        return {"name": name, "is_ntu": False}

    metadata: dict[str, Any] = {"name": name, "is_ntu": True}
    for key in ("setup", "camera", "subject", "replication", "action"):
        metadata[key] = int(match.group(key))

    modality = match.group("modality")
    if modality:
        metadata["modality"] = modality.lower()

    metadata["label"] = metadata["action"] - 1
    metadata["xsub_split"] = (
        "train"
        if metadata["subject"] in NTU60_XSUB_TRAIN_SUBJECTS
        else "test"
    )
    metadata["xview_split"] = (
        "train"
        if metadata["camera"] in NTU60_XVIEW_TRAIN_CAMERAS
        else "test"
    )
    return metadata


def iter_video_files(root: str | Path, pattern: str) -> list[Path]:
    root_path = Path(root)
    videos = [
        path
        for path in root_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    ]
    return sorted(videos, key=lambda item: item.as_posix().lower())


def output_path_for_video(
    video_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    suffix: str = ".npz",
) -> Path:
    video = Path(video_path)
    root = Path(data_root)
    out_root = Path(output_root)
    try:
        relative = video.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(video.name)

    return out_root / relative.with_suffix(suffix)
