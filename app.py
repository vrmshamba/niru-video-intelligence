from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import json
import threading
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ

app = Flask(__name__)

# Configuration for large video files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

VIDEOS_DIR = "videos"
RESULTS_DIR = "results"
CORRECTIONS_DIR = "training/corrections"
# Set NIRU_LIBRARY_DIR env var to override (e.g. to point at H:/)
LIBRARY_DIR = os.environ.get("NIRU_LIBRARY_DIR", "H:/")
SUPPORTED_EXTS = ('.mp4', '.avi', '.mov', '.mxf')

# Single-slot processing state — one video at a time
_processing_lock = threading.Lock()
_processing_state = {"running": False, "filename": None, "status": "idle", "error": None}

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/results')
def get_results():
    results = []
    if os.path.exists(RESULTS_DIR):
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('_analysis.json'):
                filepath = os.path.join(RESULTS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results.append(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return jsonify(results)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve original video files — checks LIBRARY_DIR (H:/) first, then VIDEOS_DIR."""
    for directory in [LIBRARY_DIR, VIDEOS_DIR]:
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            mimetype = 'video/quicktime' if ext in ('.mov', '.mxf') else 'video/mp4'
            return send_from_directory(directory, filename, mimetype=mimetype, as_attachment=False)
    print(f"Video not found in library or videos/: {filename}")
    return f"Video not found: {filename}", 404

@app.route('/results/<path:filename>')
def serve_result_file(filename):
    """Serve result files including annotated videos"""
    try:
        # Check if it's a video file
        if filename.endswith('.mp4'):
            return send_from_directory(
                RESULTS_DIR, 
                filename,
                mimetype='video/mp4',
                as_attachment=False
            )
        else:
            # For JSON and other files
            return send_from_directory(RESULTS_DIR, filename)
    except Exception as e:
        print(f"Error serving result file {filename}: {e}")
        return f"Error: {str(e)}", 404

@app.route('/api/library')
def get_library():
    """List all supported video files in LIBRARY_DIR with size and processed status."""
    files = []
    if os.path.exists(LIBRARY_DIR):
        for f in sorted(os.listdir(LIBRARY_DIR)):
            full_path = os.path.join(LIBRARY_DIR, f)
            if os.path.isfile(full_path) and f.lower().endswith(SUPPORTED_EXTS):
                stem = os.path.splitext(f)[0]
                processed = os.path.exists(os.path.join(RESULTS_DIR, f"{stem}_analysis.json"))
                files.append({
                    "name": f,
                    "size_mb": round(os.path.getsize(full_path) / 1024 / 1024, 1),
                    "processed": processed,
                })
    return jsonify({"source_dir": LIBRARY_DIR, "count": len(files), "files": files})


@app.route('/api/process', methods=['POST'])
def trigger_process():
    """Start background processing for a single file from LIBRARY_DIR.
    Body: {"filename": "some video.mp4"}
    Returns 409 if another video is already being processed.
    """
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({"error": "Missing 'filename' in request body"}), 400

    filename = data['filename']
    filepath = os.path.join(LIBRARY_DIR, filename)

    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found in library: {filename}"}), 404

    if not filename.lower().endswith(SUPPORTED_EXTS):
        return jsonify({"error": f"Unsupported format (must be .mp4, .avi, or .mov): {filename}"}), 400

    with _processing_lock:
        if _processing_state["running"]:
            return jsonify({"status": "busy", "current": _processing_state["filename"]}), 409
        _processing_state.update({"running": True, "filename": filename, "status": "processing", "error": None})

    def run():
        try:
            # Lazy import — avoids loading torch/whisper at Flask startup
            from process_videos import VideoProcessor
            whisper_size = os.environ.get("NIRU_WHISPER_MODEL", "small")
            processor = VideoProcessor(whisper_model_size=whisper_size)
            processor.load_models()
            processor.process_video(filepath)
            with _processing_lock:
                _processing_state.update({"running": False, "status": "done", "error": None})
        except Exception as e:
            with _processing_lock:
                _processing_state.update({"running": False, "status": "error", "error": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "filename": filename})


@app.route('/api/process/status')
def process_status():
    """Return the current processing state."""
    with _processing_lock:
        return jsonify(dict(_processing_state))


@app.route('/api/retranscribe', methods=['POST'])
def retranscribe():
    """Re-run Gemini transcription only on an already-processed video.
    Updates only the transcript field in the existing result JSON.
    Body: {"filename": "some video.mp4"}
    Returns 409 if a job is already running.
    """
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({"error": "Missing 'filename' in request body"}), 400

    filename = data['filename']
    stem = os.path.splitext(filename)[0]
    result_path = os.path.join(RESULTS_DIR, f"{stem}_analysis.json")

    if not os.path.exists(result_path):
        return jsonify({"error": f"No existing result for: {filename}"}), 404

    # Find the video file
    filepath = None
    for directory in [LIBRARY_DIR, VIDEOS_DIR]:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            filepath = candidate
            break
    if not filepath:
        return jsonify({"error": f"Video file not found: {filename}"}), 404

    with _processing_lock:
        if _processing_state["running"]:
            return jsonify({"status": "busy", "current": _processing_state["filename"]}), 409
        _processing_state.update({"running": True, "filename": filename, "status": "retranscribing", "error": None})

    def run():
        try:
            from process_videos import VideoProcessor
            from datetime import datetime
            whisper_size = os.environ.get("NIRU_WHISPER_MODEL", "small")
            processor = VideoProcessor(whisper_model_size=whisper_size)
            processor.load_models()

            transcript = processor.transcribe_audio(filepath)

            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)

            result["transcript"] = transcript
            result["summary"]["language_detected"] = transcript.get("language", "unknown")
            result["retranscribed_at"] = datetime.now().isoformat()

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            with _processing_lock:
                _processing_state.update({"running": False, "status": "done", "error": None})
            print(f"Retranscription complete: {filename}")
        except Exception as e:
            with _processing_lock:
                _processing_state.update({"running": False, "status": "error", "error": str(e)})
            print(f"Retranscription error ({filename}): {e}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "filename": filename})


@app.route('/api/corrections', methods=['GET'])
def get_corrections():
    """Return a dict of {video_name: correction} for all saved corrections."""
    corrections = {}
    if os.path.exists(CORRECTIONS_DIR):
        for fname in os.listdir(CORRECTIONS_DIR):
            if fname.endswith('_corrected.json'):
                try:
                    with open(os.path.join(CORRECTIONS_DIR, fname), encoding='utf-8') as f:
                        data = json.load(f)
                    corrections[data['video_name']] = data
                except Exception:
                    pass
    return jsonify(corrections)


@app.route('/api/corrections/save', methods=['POST'])
def save_correction():
    """Save a human-corrected transcript for a video.
    Body: {
      "video_name": "foo.mp4",
      "language": "Swahili/Sheng",
      "segments": [{"start": 0.0, "end": 4.2, "text": "corrected text"}, ...]
    }
    Writes to training/corrections/{stem}_corrected.json.
    """
    data = request.get_json()
    if not data or 'video_name' not in data:
        return jsonify({"error": "Missing video_name"}), 400

    video_name = data['video_name']
    stem = os.path.splitext(video_name)[0]

    from datetime import datetime
    os.makedirs(CORRECTIONS_DIR, exist_ok=True)

    correction = {
        "video_name": video_name,
        "language": data.get("language", ""),
        "corrected_at": datetime.now().isoformat(),
        "segments": data.get("segments", []),
        "full_text": " ".join(s.get("text", "") for s in data.get("segments", []))
    }

    path = os.path.join(CORRECTIONS_DIR, f"{stem}_corrected.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(correction, f, ensure_ascii=False, indent=2)

    return jsonify({"status": "saved", "path": path, "segments": len(correction["segments"])})


@app.route('/api/corrections/delete', methods=['POST'])
def delete_correction():
    """Delete a saved correction. Body: {"video_name": "foo.mp4"}"""
    data = request.get_json()
    if not data or 'video_name' not in data:
        return jsonify({"error": "Missing video_name"}), 400
    stem = os.path.splitext(data['video_name'])[0]
    path = os.path.join(CORRECTIONS_DIR, f"{stem}_corrected.json")
    if os.path.exists(path):
        os.remove(path)
    return jsonify({"status": "deleted"})


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run with host 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)