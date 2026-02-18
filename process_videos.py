import os
import json
import argparse
import subprocess
import torch
import cv2
from datetime import datetime
import whisper
import importlib.metadata

# Monkey patch for torchvision metadata missing in this env
try:
    importlib.metadata.version("torchvision")
except importlib.metadata.PackageNotFoundError:
    print("Patching torchvision version metadata...")
    original_version = importlib.metadata.version
    def patched_version(package):
        if package == "torchvision":
            return "0.16.0" 
        return original_version(package)
    importlib.metadata.version = patched_version

from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, models_dir="models", results_dir="results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.ensure_directories()
        
        self.whisper_model = None
        self.yolo_model = None
        
    def ensure_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def load_models(self):
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("tiny", download_root=self.models_dir)
        print("Whisper model loaded.")
        
        print("Loading YOLO model...")
        # Auto-downloads to current dir usually, let's verify path handling later
        self.yolo_model = YOLO("yolov8n.pt") 
        print("YOLO model loaded.")

    def process_video(self, video_path):
        filename = os.path.basename(video_path)
        print(f"Processing: {filename}")
        
        result = {
            "video_name": filename,
            "processed_at": datetime.now().isoformat(),
            "transcript": {},
            "scene_analysis": {},
            "safety_concerns": [],
            "insertion_points": [],
            "summary": {}
        }
        
        # 2. Extract Audio & Transcribe (Whisper)
        try:
            print("Starting transcription...")
            transcript = self.transcribe_audio(video_path)
            result["transcript"] = transcript
            result["summary"]["language_detected"] = transcript.get("language", "unknown")
            print("Transcription complete.")
        except Exception as e:
            print(f"Error in transcription: {e}")
            result["transcript"] = {"error": str(e)}

        # 3. Object Detection (YOLO) + Annotation
        try:
            print("Starting object detection and annotation...")
            # Create a path for the annotated video
            annotated_filename = f"web_annotated_{filename}"
            annotated_path = os.path.join(self.results_dir, annotated_filename)

            scene_data = self.detect_objects(video_path, output_path=annotated_path)
            result["scene_analysis"] = scene_data
            result["summary"]["total_objects_detected"] = len(scene_data["scenes"])
            result["annotated_video"] = annotated_filename # Store ref for dashboard
            print(f"Object detection complete. Annotated video saved to {annotated_path}")
        except Exception as e:
            print(f"Error in object detection: {e}")
            result["scene_analysis"] = {"error": str(e)}
        
        # 4. Safety Analysis
        try:
            print("Starting safety analysis...")
            safety_concerns = self.analyze_safety(result["transcript"], result["scene_analysis"])
            result["safety_concerns"] = safety_concerns
            result["summary"]["safety_flags"] = len(safety_concerns)
            print("Safety analysis complete.")
        except Exception as e:
            print(f"Error in safety analysis: {e}")
            result["safety_concerns"] = [{"error": str(e)}]
            
        # 5. Insertion Points
        try:
            print("Finding insertion points...")
            insertion_points = self.find_insertion_points(result["scene_analysis"])
            result["insertion_points"] = insertion_points
            result["summary"]["safe_insertion_windows"] = len(insertion_points)
            print("Insertion points found.")
        except Exception as e:
            print(f"Error in insertion points: {e}")

        # 6. Save Results
        self.save_result(filename, result)
        return result
    
    def transcribe_audio(self, video_path):
        if not self.whisper_model:
            raise ValueError("Whisper model not loaded.")
        result = self.whisper_model.transcribe(video_path)
        return {
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"]
        }

    def detect_objects(self, video_path, output_path=None):
        if not self.yolo_model:
            raise ValueError("YOLO model not loaded.")

        # Open source video to get properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Write annotated frames to a temp file with mp4v (reliably supported by
        # OpenCV on all platforms), then re-encode to H.264 for browser playback.
        video_writer = None
        tmp_path = None
        if output_path:
            tmp_path = output_path + ".tmp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

        # vid_stride=30 means process every 30th frame. 
        # BUT if we want a smooth annotated video, we probably want to process ALL frames 
        # or at least write input frames even if we don't detect on them.
        # YOLO track/predict usually handles this. stream=True yields results.
        # If we use stride, we skip frames in the output too unless we handle reading manually.
        # For MVP "Annotated Video", let's process every 5th frame to save time but keep it watchable?
        # Or just let YOLO handle it. If we want a watchable video, we should probably output at original FPS.
        # Let's try stride=5 for performant "demo" quality. 
        
        # NOTE: If we want the output video to be the full video, we can't use stride in the generate loop easily 
        # without duplicating frames or having a choppy video. 
        # Let's stick to stride=1 (process all) for the "Annotated" version so it looks good, 
        # OR stride=1 but only run detection every N frames (YOLO track mode?).
        # For simplicity in MVP: Process EVERY frame (stride=1) but maybe resize if too slow? 
        # Let's stick to default stride=1 for the demo video to be smooth.
        
        print(f"Running YOLO inference (generating annotated video)...")
        # Lower confidence for better detection, especially partial views
        results = self.yolo_model(video_path, stream=True, verbose=False, conf=0.25) # Lowered from default 0.25
        
        scenes = []
        frame_idx = 0
        
        for r in results:
            # Annotation with enhanced visibility
            if video_writer:
                # r.plot() with enhanced visibility settings
                # line_width=2 (thicker boxes), font_size=1.0 (larger labels), labels=True
                annotated_frame = r.plot(line_width=3, font_size=1.2, labels=True, conf=True) 
                video_writer.write(annotated_frame)
            
            # Data Extraction (Sampling for JSON to keep it small)
            # Only save data every 30 frames (1 sec) to avoid massive JSON
            if frame_idx % 30 == 0:
                frame_objects = []
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r.names[cls_id]
                    
                    # Lowered threshold to 0.25 to catch more objects (backs, partial views)
                    if conf > 0.25:
                        frame_objects.append({
                            "object": label,
                            "confidence": conf
                        })
                
                if frame_objects:
                    timestamp = frame_idx / fps if fps else 0
                    scenes.append({
                        "frame_index": frame_idx,
                        "timestamp": timestamp,
                        "objects": frame_objects
                    })
            
            frame_idx += 1
            
        if video_writer:
            video_writer.release()
            self._reencode_h264(tmp_path, output_path)

        return {
            "total_scenes_analyzed": len(scenes),
            "scenes": scenes
        }

    def _reencode_h264(self, tmp_path, output_path):
        """Re-encode tmp_path to H.264/yuv420p at output_path, then delete tmp_path.
        Falls back to renaming tmp_path if ffmpeg is not available."""
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                os.remove(tmp_path)
                print(f"Re-encoded to H.264: {output_path}")
            else:
                print(f"ffmpeg re-encode failed (returncode {result.returncode}): {result.stderr[-500:]}")
                os.replace(tmp_path, output_path)
                print(f"Kept mp4v fallback at: {output_path}")
        except FileNotFoundError:
            print("ffmpeg not found in PATH. Keeping mp4v-encoded file (may not play in all browsers).")
            os.replace(tmp_path, output_path)

    def analyze_safety(self, transcript, scene_analysis):
        concerns = []
        
        # 1. Keyword Check
        keywords = ["violence", "kill", "attack", "gun", "weapon", "danger"]
        text = transcript.get("text", "").lower()
        found_keywords = [k for k in keywords if k in text]
        if found_keywords:
            concerns.append({
                "type": "sensitive_keywords",
                "timestamp": 0, # global for text
                "description": f"Detected sensitive keywords: {', '.join(found_keywords)}",
                "severity": "medium"
            })
            
        # 2. Visual Check
        if "scenes" in scene_analysis:
            for scene in scene_analysis["scenes"]:
                objects = [o["object"] for o in scene["objects"]]
                
                # Check for Weapons
                weapons = ["knife", "gun", "sword"] # Standard COCO classes: knife, etc? YOLOv8 coco classes include 'knife', 'scissors'
                # Let's assume standard COCO: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
                detected_weapons = [w for w in weapons if w in objects]
                
                if detected_weapons:
                    concerns.append({
                        "type": "weapon_detection",
                        "timestamp": scene.get("timestamp", 0),
                        "description": f"Detected potential weapon: {', '.join(detected_weapons)}",
                        "severity": "high"
                    })
                
                # Check for Crowds
                person_count = objects.count("person")
                if person_count >= 10:
                    concerns.append({
                        "type": "crowd_detection",
                        "timestamp": scene.get("timestamp", 0),
                        "description": f"Large gathering detected: {person_count} people",
                        "severity": "medium"
                    })
                    
        return concerns

    def find_insertion_points(self, scene_analysis):
        points = []
        if "scenes" not in scene_analysis:
            return points
            
        scenes = scene_analysis["scenes"]
        # Look for windows of 5 seconds (5 scenes if 1 fps)
        window_size = 5 
        
        i = 0
        while i <= len(scenes) - window_size:
            window = scenes[i:i+window_size]

            avg_objects = sum(len(s["objects"]) for s in window) / window_size
            is_stable = all(len(s["objects"]) < 5 for s in window)

            if is_stable:
                points.append({
                    "timestamp": window[0]["timestamp"],
                    "duration": 5,
                    "reason": "stable_scene_low_objects",
                    "object_count": int(avg_objects)
                })
                i += window_size  # skip past this window to avoid overlapping results
            else:
                i += 1  # slide forward one scene and try again

        return points

    def save_result(self, video_filename, data):
        json_name = f"{os.path.splitext(video_filename)[0]}_analysis.json"
        path = os.path.join(self.results_dir, json_name)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved results to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIRU video processing pipeline")
    parser.add_argument(
        "--source-dir",
        default="videos",
        help="Directory containing video files to process (default: videos/)"
    )
    args = parser.parse_args()

    processor = VideoProcessor()
    processor.load_models()

    videos_dir = args.source_dir
    if not os.path.exists(videos_dir):
        print(f"Directory '{videos_dir}' not found.")
    else:
        print(f"Scanning '{videos_dir}' for videos...")
        SUPPORTED_EXTS = ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')
        video_files = [
            f for f in os.listdir(videos_dir)
            if os.path.isfile(os.path.join(videos_dir, f))
            and f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]

        if not video_files:
            print("No supported video files found (.mp4, .avi, .mov).")
            print("Note: .MXF files are not supported by this pipeline.")
        else:
            print(f"Found {len(video_files)} videos: {video_files}")
            for video_file in video_files:
                video_path = os.path.join(videos_dir, video_file)
                try:
                    processor.process_video(video_path)
                except Exception as e:
                    print(f"Failed to process {video_file}: {e}")
    print("Batch processing complete.")
