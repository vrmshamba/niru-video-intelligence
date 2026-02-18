from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import json
import threading

app = Flask(__name__)

# Configuration for large video files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

VIDEOS_DIR = "videos"
RESULTS_DIR = "results"
# Set NIRU_LIBRARY_DIR env var to override (e.g. to point at H:/)
LIBRARY_DIR = os.environ.get("NIRU_LIBRARY_DIR", "H:/")
SUPPORTED_EXTS = ('.mp4', '.avi', '.mov')

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
    """Serve original video files"""
    try:
        return send_from_directory(
            VIDEOS_DIR, 
            filename,
            mimetype='video/mp4',
            as_attachment=False
        )
    except Exception as e:
        print(f"Error serving video {filename}: {e}")
        return f"Error: {str(e)}", 404

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
            processor = VideoProcessor()
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


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run with host 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)