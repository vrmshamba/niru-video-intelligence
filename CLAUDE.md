# NIRU MVP - Claude Code Guide

## Project Overview
AI-powered video analysis pipeline for Kenyan media content. Processes local video files through Whisper (speech-to-text) and YOLOv8 (object detection), then serves results via a Flask dashboard.

## Architecture
- `process_videos.py` — core pipeline (Whisper + YOLO + safety analysis + insertion points)
- `app.py` — Flask web server (port 5000), serves dashboard and result files
- `templates/dashboard.html` — frontend UI (vanilla JS, fetches `/api/results`)
- `download_videos.py` — stub for Google Drive downloads (currently non-functional)

## Key Directories
| Path | Purpose |
|------|---------|
| `videos/` | Default input folder for test videos |
| `results/` | Output: `*_analysis.json` + `web_annotated_*.mp4` |
| `models/` | AI model cache (Whisper tiny, yolov8n.pt) |
| `H:/` | Main video library (~153 files: .mp4, .avi, .mov, .mxf) |

## Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Process videos from default folder (videos/)
python process_videos.py

# Process videos from H: drive library
python process_videos.py --source-dir "H:/"

# Process a specific folder
python process_videos.py --source-dir "H:/some/subfolder"

# Start the dashboard
python app.py
# then open http://127.0.0.1:5000
```

## Output File Naming Convention
The pipeline outputs the following for each video `foo.mp4`:
- `results/foo_analysis.json` — full analysis (transcript, scenes, safety, insertion points)
- `results/web_annotated_foo.mp4` — YOLO-annotated video (browser-compatible naming)

The dashboard JS at `dashboard.html:448` constructs annotated video URLs as `web_annotated_{video_name}`. The pipeline (`process_videos.py:76`) must generate files with the `web_annotated_` prefix - these must stay in sync.

## Known Issues / TODO

### Active bugs
- **YOLO model save path**: `YOLO("yolov8n.pt")` (line 43) downloads to CWD, not `models/`. Pass a download path to keep things tidy.
- **`total_objects_detected` mislabeled**: counts scenes-with-objects, not total object count. *(low priority)*

### Format support
- `.mp4`, `.avi`, `.mov` are supported (case-insensitive match)
- `.mxf` (broadcast format) is NOT supported — OpenCV cannot decode MXF without ffmpeg. These files are skipped with a warning.

### Not yet built
- Dashboard UI to browse and trigger processing of H: drive library (`/api/library`, `/api/process` endpoints)
- Processing progress feedback in the UI
- `download_videos.py` Google Drive IDs (all commented out)

## Dependencies
See `requirements.txt`. Key versions:
- `numpy==1.24.3` — pinned; may conflict with newer ultralytics/torch
- `flask==3.0.0`
- `openai-whisper` (loads `tiny` model by default)
- `ultralytics` (yolov8n)
- `opencv-python`

## Environment
- Windows 11, H: drive is the working volume
- `.venv` virtual environment at project root (`.gitignore`d)
- Models and large video files are `.gitignore`d
