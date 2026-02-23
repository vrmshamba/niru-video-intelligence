import os
import json
import argparse
import subprocess
import torch
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ
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
    def __init__(self, models_dir="models", results_dir="results", whisper_model_size="small"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.whisper_model_size = whisper_model_size
        self.ensure_directories()

        self.whisper_model = None
        self.yolo_model = None
        
    def ensure_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def _get_ffmpeg(self):
        """Return a path to an ffmpeg binary named exactly 'ffmpeg' (or 'ffmpeg.exe' on Windows).
        Checks system PATH first, then falls back to the imageio_ffmpeg bundled binary,
        copying it to models/ffmpeg.exe so it can be found by name."""
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        try:
            import imageio_ffmpeg
            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            # imageio_ffmpeg names the binary 'ffmpeg-win-x86_64-vX.X.exe', not 'ffmpeg.exe'.
            # Copy it to models/ffmpeg.exe so subprocess calls to "ffmpeg" resolve correctly.
            dest = os.path.join(self.models_dir, "ffmpeg.exe")
            if not os.path.exists(dest):
                shutil.copy2(bundled, dest)
                print(f"Copied bundled ffmpeg to {dest}")
            return dest
        except Exception:
            return None

    def load_models(self):
        self.ffmpeg_path = self._get_ffmpeg()
        if self.ffmpeg_path:
            print(f"ffmpeg ready: {self.ffmpeg_path}")
            # Add models/ to PATH so whisper's subprocess("ffmpeg") call finds ffmpeg.exe
            ffmpeg_dir = os.path.dirname(self.ffmpeg_path)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        else:
            print("WARNING: ffmpeg not found. Transcription and H.264 encoding will be unavailable.")

        print(f"Loading Whisper {self.whisper_model_size} model...")
        self.whisper_model = whisper.load_model(self.whisper_model_size, download_root=self.models_dir)
        print("Whisper model loaded.")
        
        print("Loading YOLO model...")
        custom_model = os.path.join(self.models_dir, "niru_nairobi.pt")
        base_model   = os.path.join(self.models_dir, "yolov8n.pt")
        if os.path.exists(custom_model):
            self.yolo_model = YOLO(custom_model)
            print(f"Loaded custom Nairobi model: {custom_model}")
        else:
            self.yolo_model = YOLO(base_model)   # downloads to models/ if not present
            print(f"Loaded base YOLO model: {base_model}")
            print("  (train and place models/niru_nairobi.pt to use the custom model)")

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
    
    def _extract_audio(self, video_path):
        """Extract audio to a temporary MP3 using ffmpeg. Returns temp file path."""
        import tempfile, time as _time
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        ffmpeg = self.ffmpeg_path or "ffmpeg"
        result = subprocess.run(
            [ffmpeg, "-y", "-i", video_path, "-vn",
             "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
             tmp.name],
            capture_output=True
        )
        if result.returncode != 0:
            os.unlink(tmp.name)
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.decode()[:300]}")
        return tmp.name

    def _transcribe_with_gemini(self, video_path):
        """Transcribe using Gemini 2.5 Flash — uploads audio only (cheaper, free-tier friendly)."""
        import time
        from google import genai

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        # Extract audio only — much cheaper than uploading the full video
        print("  Extracting audio for Gemini upload...")
        audio_path = self._extract_audio(video_path)

        try:
            print("  Uploading audio to Gemini API...")
            uploaded = client.files.upload(file=audio_path)
        finally:
            os.unlink(audio_path)  # delete temp audio immediately after upload

        # Wait for Gemini to finish processing (state: PROCESSING → ACTIVE)
        while uploaded.state.name == "PROCESSING":
            print("  Waiting for file to become active...")
            time.sleep(3)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini file processing failed: {uploaded.state}")
        print("  File active, requesting transcription...")

        prompt = """You are a professional transcriptionist specialising in East African languages.
Transcribe this audio/video exactly as spoken, preserving the original language.
The content may be in any of these languages:
  English, Swahili, Sheng (Nairobi street slang — mix of Swahili/English/Kikuyu),
  Kikuyu (Gikuyu), Kalenjin (Nandi/Kipsigis/Tugen dialects), Maasai (Maa),
  Dholuo (Luo), Luhya (Luyia), Kisii (Gusii/Ekegusii).

Return ONLY a valid JSON object — no markdown, no code fences:
{
  "language": "full language name as detected",
  "language_code": "ISO 639 code or best approximation",
  "confidence": "high | medium | low",
  "notes": "optional — mixed languages, unclear speech, dialect details, or null",
  "text": "complete transcription of all speech",
  "segments": [
    {"start": 0.0, "end": 4.2, "text": "segment text"},
    ...
  ]
}
If there is no speech, return text as empty string and segments as [].
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded, prompt],
            )
        finally:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

        # Strip accidental code fences
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

        return json.loads(raw)

    def transcribe_audio(self, video_path):
        if not self.whisper_model:
            raise ValueError("Whisper model not loaded.")

        # ── Gemini path (preferred — handles all Kenyan languages) ───────────
        if os.environ.get("GOOGLE_API_KEY"):
            try:
                print("  Using Gemini 2.0 Flash for transcription...")
                g = self._transcribe_with_gemini(video_path)
                conf_map = {"high": 0.95, "medium": 0.70, "low": 0.40}
                return {
                    "text": g.get("text", ""),
                    "language": g.get("language", "unknown"),
                    "language_code": g.get("language_code", ""),
                    "language_confidence": conf_map.get(g.get("confidence", "low"), 0.40),
                    "language_note": g.get("notes"),
                    "segments": g.get("segments", []),
                    "source": "gemini",
                }
            except Exception as e:
                print(f"  Gemini transcription failed ({e}). Falling back to Whisper.")

        # ── Whisper fallback ─────────────────────────────────────────────────
        print("  Using Whisper for transcription...")
        audio = whisper.load_audio(video_path)
        audio_segment = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_segment).to(self.whisper_model.device)
        _, lang_probs = self.whisper_model.detect_language(mel)

        detected_lang = max(lang_probs, key=lang_probs.get)
        lang_confidence = round(lang_probs[detected_lang], 3)

        WHISPER_SUPPORTED = {"sw", "en", "fr", "de", "es", "ar", "pt", "hi", "zh"}
        if lang_confidence < 0.4 or detected_lang not in WHISPER_SUPPORTED:
            lang_hint = None
            language_note = (
                f"uncertain ({detected_lang}, {lang_confidence:.0%} conf) — "
                "possibly Kikuyu, Dholuo, Kalenjin, or other Kenyan language. "
                "Set GOOGLE_API_KEY for full language support."
            )
        else:
            lang_hint = detected_lang
            language_note = None

        result = self.whisper_model.transcribe(video_path, language=lang_hint)
        return {
            "text": result["text"],
            "language": result["language"],
            "language_code": result["language"],
            "language_confidence": lang_confidence,
            "language_note": language_note,
            "segments": result["segments"],
            "source": "whisper",
        }

    def detect_objects(self, video_path, output_path=None):
        if not self.yolo_model:
            raise ValueError("YOLO model not loaded.")

        # YOLO doesn't support MXF — convert to a temp MP4 first if needed
        yolo_path, mxf_tmp = self._to_mp4_if_needed(video_path)

        # Open source video to get properties
        cap = cv2.VideoCapture(yolo_path)
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
        results = self.yolo_model(yolo_path, stream=True, verbose=False, conf=0.25)
        
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

        if mxf_tmp and os.path.exists(mxf_tmp):
            os.remove(mxf_tmp)

        return {
            "total_scenes_analyzed": len(scenes),
            "scenes": scenes
        }

    def _to_mp4_if_needed(self, video_path):
        """If video_path is MXF, convert to a temp MP4 via ffmpeg so YOLO can read it.
        Returns (path_to_use, tmp_path_to_delete). tmp_path_to_delete is None if no
        conversion was needed."""
        if not video_path.lower().endswith('.mxf'):
            return video_path, None
        if not self.ffmpeg_path:
            raise RuntimeError("ffmpeg not available — cannot convert MXF for YOLO.")
        tmp_mp4 = video_path + ".yolo_tmp.mp4"
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", video_path,
            "-vcodec", "copy",   # stream-copy video (fast, no re-encode)
            "-an",               # drop audio (YOLO doesn't need it)
            tmp_mp4
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg MXF->MP4 conversion failed: {result.stderr[-500:]}")
        print(f"Converted MXF to temp MP4 for YOLO: {tmp_mp4}")
        return tmp_mp4, tmp_mp4

    def _reencode_h264(self, tmp_path, output_path):
        """Re-encode tmp_path to H.264/yuv420p at output_path, then delete tmp_path.
        Falls back to renaming tmp_path if ffmpeg is not available."""
        if not self.ffmpeg_path:
            print("ffmpeg not available. Keeping mp4v-encoded file (may not play in all browsers).")
            os.replace(tmp_path, output_path)
            return
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", tmp_path,
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.remove(tmp_path)
            print(f"Re-encoded to H.264: {output_path}")
        else:
            print(f"ffmpeg re-encode failed (returncode {result.returncode}): {result.stderr[-500:]}")
            os.replace(tmp_path, output_path)
            print(f"Kept mp4v fallback at: {output_path}")

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
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small). Use medium for best Swahili/Dholuo accuracy."
    )
    args = parser.parse_args()

    processor = VideoProcessor(whisper_model_size=args.whisper_model)
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
            and f.lower().endswith(('.mp4', '.avi', '.mov', '.mxf'))
        ]

        if not video_files:
            print("No supported video files found (.mp4, .avi, .mov, .mxf).")
        else:
            print(f"Found {len(video_files)} videos: {video_files}")
            for video_file in video_files:
                video_path = os.path.join(videos_dir, video_file)
                try:
                    processor.process_video(video_path)
                except Exception as e:
                    print(f"Failed to process {video_file}: {e}")
    print("Batch processing complete.")
