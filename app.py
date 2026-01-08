from flask import Flask, render_template, jsonify, send_from_directory
import os
import json

app = Flask(__name__)

# Configuration for large video files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

VIDEOS_DIR = "videos"
RESULTS_DIR = "results"

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

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run with host 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)