from flask import Flask, render_template, jsonify, send_from_directory
import os
import json

app = Flask(__name__)

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
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        results.append(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return jsonify(results)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEOS_DIR, filename)

@app.route('/results/<path:filename>')
def serve_result_file(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)