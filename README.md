# RTMW NTU60 Keypoint Extraction

This project extracts whole-body 2D keypoints from NTU RGB+D 60 RGB videos with
RTMW through `rtmlib`.

The repository does not include NTU60 data. NTU RGB+D is distributed by the
dataset owners and may require registration or authenticated links. Put the
authorized download URLs in a manifest, then run the provided automation script.

## What Gets Produced

For every NTU RGB video such as:

```text
S001C001P001R001A001_rgb.avi
```

the extractor writes:

```text
outputs/rtmw_ntu60/S001C001P001R001A001_rgb.npz
```

Each `.npz` contains:

- `keypoints`: `(frames, max_persons, keypoints, 2)` float32 array, x/y pixels.
- `scores`: `(frames, max_persons, keypoints)` float32 confidence array.
- `frame_indices`: original video frame indices that were processed.
- `frame_shape`: video frame height and width.
- `metadata`: JSON string parsed from the NTU filename.
- `config`: JSON string with the RTMW/runtime settings.

By default the extractor keeps the top 2 people per frame, which matches NTU60
single-person and two-person actions.

## Setup

```powershell
python -m pip install -r requirements.txt
```

For GPU inference, install a CUDA-compatible `onnxruntime-gpu` build instead of
the CPU `onnxruntime` package, then run with `--device cuda`.

## One-Command Main Pipeline

You can run the main entry directly:

```powershell
python main.py
```

The default pipeline is:

```text
download NTU60 archives -> extract RGB videos -> extract RTMW keypoints -> train classifier
```

`main.py` checks for missing Python packages and installs `requirements.txt`
automatically before the pipeline starts. Use `--no-auto-install` if you want to
manage the environment manually.

If `manifests\ntu60_rgb.json` is missing, `main.py` creates it from the example
template, skips download for that run, and checks `data\ntu60` for existing RGB
videos. This keeps local smoke runs from failing before you add authorized NTU60
links.

After filling `manifests\ntu60_rgb.json` with authorized NTU60 links, run:

```powershell
python main.py --manifest manifests\ntu60_rgb.json --device cpu
```

Alternatively, set authorized URL(s) in the environment and just run `main.py`.
Multiple URLs can be separated with semicolons:

```powershell
$env:NTU60_RGB_URLS="https://your-authorized-url/nturgbd_rgb_s001.zip"
python main.py
```

The default manifest is already shaped for the NTU60 RGB Google Drive archive.
The NTU120 supplement is intentionally disabled by default. This downloads the
NTU60 archive, extracts it into `data\ntu60`, writes RTMW keypoints to
`outputs\rtmw_ntu60`, then trains a lightweight NumPy softmax classifier and
writes:

```text
outputs\models\ntu60_rtmw_softmax.npz
outputs\models\ntu60_rtmw_metrics.json
outputs\models\ntu60_rtmw_predictions.csv
```

Useful variants:

```powershell
python main.py --install-deps --manifest manifests\ntu60_rgb.json
python main.py --skip-download --limit 2
python main.py --train-only --epochs 100
python main.py --skip-train
python main.py --force-extract
python main.py --download-only
```

`manifests\ntu60_rgb.json` is ignored by git so local download links are not
committed accidentally.

Archives are not re-extracted on every run. After a successful extraction,
`main.py` writes a small marker under `data\ntu60\.extract_markers`; later runs
reuse the extracted data unless the archive changes or you pass `--force-extract`.

## Run on Existing NTU60 RGB Videos

```powershell
python scripts\extract_ntu60_rtmw.py `
  --data-root data\ntu60 `
  --output-root outputs\rtmw_ntu60 `
  --device cpu `
  --mode balanced
```

If your RGB videos are nested under a different directory, set `--data-root` to
that root. The default pattern is `**/*_rgb.avi`.

## Download Then Extract

1. Copy the template manifest:

```powershell
Copy-Item manifests\ntu60_rgb.example.json manifests\ntu60_rgb.json
```

2. Replace each placeholder URL with authorized NTU RGB+D 60 RGB download links.
   You can add as many entries as needed.

3. Run the full pipeline:

```powershell
scripts\run_ntu60_rtmw.ps1 `
  -Manifest manifests\ntu60_rgb.json `
  -DataRoot data\ntu60 `
  -DownloadDir data\downloads\ntu60 `
  -OutputRoot outputs\rtmw_ntu60 `
  -Device cpu `
  -Mode balanced
```

To test the pipeline on a small subset after data is present:

```powershell
scripts\run_ntu60_rtmw.ps1 -SkipDownload -Limit 2
```

Linux/macOS users can use:

```bash
bash scripts/run_ntu60_rtmw.sh --manifest manifests/ntu60_rgb.json --device cpu
```

## Manifest Format

```json
{
  "files": [
    {
      "name": "nturgbd_rgb_s001.zip",
      "url": "https://example.com/authorized/nturgbd_rgb_s001.zip",
      "sha256": "",
      "extract": true
    }
  ]
}
```

Supported entries:

- `url`: direct HTTP(S) URL.
- `gdrive_id`: optional Google Drive file id, if your authorized link uses
  Google Drive.
- `name`: output filename in the download directory.
- `sha256`: optional checksum.
- `extract`: whether to extract the archive after download.

Archive extraction supports `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.7z`, and `.rar`.
For `.7z` or `.rar`, install `7z` and make sure it is on `PATH`.

The NTU120 supplement link can be added back later by adding another manifest
entry with Google Drive file id `1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w`.

## Useful Options

```powershell
python scripts\extract_ntu60_rtmw.py --help
python scripts\download_ntu60.py --help
python scripts\train_ntu60.py --help
```

Common extractor options:

- `--mode lightweight|balanced|performance`: RTMW model preset from `rtmlib`.
- `--device cpu|cuda`: inference device.
- `--max-persons 2`: fixed number of person slots per frame.
- `--every-n 2`: process every nth frame for quick experiments.
- `--limit 10`: process only the first N videos.
- `--det` and `--pose`: override detector/pose model with local paths or URLs.

Common training options:

- `--epochs 50`: number of lightweight classifier epochs.
- `--train-split xsub|xview|random|all`: split used for training metrics.
- `--num-classes 0`: infer the class count from extracted labels.
- `--train-only`: train from existing `.npz` keypoint files.
- `--skip-train`: run download/extraction without training.

## Notes

- RTMW model files are downloaded by `rtmlib` on first use if they are not
  already cached.
- The extractor stores whole-body keypoints. With the default MMPose convention,
  RTMW returns 133 keypoints.
- NTU60 metadata is parsed from filenames and included in each output file and
  in `outputs/rtmw_ntu60/index.csv`.
