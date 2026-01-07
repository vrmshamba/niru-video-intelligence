# NIRU Video Intelligence Platform (MVP)

A prototype system for AI-powered video analysis, featuring speech-to-text, object detection, safety screening, and insertion point identification.

## Features
- **Speech-to-Text:** Generates transcripts using OpenAI Whisper.
- **Object Detection:** Identifies objects using YOLOv8.
- **Safety Screening:** Flags potential weapons, violence keywords, and large crowds.
- **Insertion Points:** Recommends safe 5-second windows for messaging.
- **Annotated Videos:** Generates "AI Vision" videos with bounding boxes.
- **Dashboard:** Web interface to view results.

## Requirements
- Python 3.8+
- Dependencies: `torch`, `torchvision`, `ultralytics`, `openai-whisper`, `flask`, `pillow`, `opencv-python`

## Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Place MP4 videos in the `videos` folder.

## Usage
1.  **Run Processing:**
    Double-click `run_processing.bat` (or run `python process_videos.py`).
    This will process all videos and save results to `results/`.

2.  **View Dashboard:**
    Double-click `run_demo.bat` (or run `python app.py`).
    Open `http://127.0.0.1:5000` to see the analysis.

## Online Demo (For Judges)
To show this to judges online while running it locally:
1. Run the app: `python app.py`
2. Expose it: `npx localtunnel --port 5000`

For more detailed deployment options, see [DEPLOYMENT.md](DEPLOYMENT.md).

## File Structure
- `videos/`: Input video files.
- `results/`: Output JSONs and Annotated Videos.
- `models/`: Downloaded AI models.
- `process_videos.py`: Main processing logic.
- `app.py`: Dashboard backend.
