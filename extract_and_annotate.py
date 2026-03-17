"""
extract_and_annotate.py

Extracts frames from H: drive videos for Roboflow labeling.
Optionally runs YOLOv8 auto-annotation to pre-populate labels (speeds up manual correction).

Default behaviour: scene-change detection across ALL videos, no frame cap.
A frame is saved only when the scene shifts meaningfully — avoids thousands of
near-identical frames from static shots.

Modes:
  Scene change (default): extract frames only when scene changes significantly
  Uniform:                extract one frame every --interval seconds (--uniform)

Output:
  Flat (default): annotation_frames/{video_stem}_NNNNN.jpg  ← best for Roboflow drag-and-drop
  Subfolders:     annotation_frames/{video_stem}/frame_NNNNN.jpg  (--subfolders)

Usage:
    python extract_and_annotate.py                  # all videos, scene-change, no cap
    python extract_and_annotate.py --n 10           # first 10 videos only
    python extract_and_annotate.py --threshold 15   # more sensitive (more frames)
    python extract_and_annotate.py --threshold 40   # less sensitive (fewer frames)
    python extract_and_annotate.py --uniform        # uniform sampling every 3s instead
    python extract_and_annotate.py --annotate       # also run YOLO pre-annotation
"""

import os
import argparse
import cv2
from pathlib import Path

FRAMES_DIR  = "annotation_frames"
LABELS_DIR  = "annotation_labels"
LIBRARY_DIR = "H:/"
SUPPORTED   = ('.mp4', '.avi', '.mov', '.mxf')

# COCO-80 class names (YOLOv8 default) — used only when --annotate is set
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]


def get_videos(library_dir, n=None):
    files = sorted([
        f for f in os.listdir(library_dir)
        if os.path.isfile(os.path.join(library_dir, f))
        and f.lower().endswith(SUPPORTED)
        and ".yolo_tmp" not in f          # skip pipeline temp files
    ])
    return files if n is None else files[:n]


def extract_frames_uniform(cap, fps, total_frames, interval_sec, max_frames, out_dir, stem):
    """Extract one frame every interval_sec seconds, up to max_frames (None = unlimited)."""
    frame_step = max(1, int(fps * interval_sec))
    extracted = 0
    frame_idx = 0
    saved_paths = []

    while (max_frames is None or extracted < max_frames) and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        path = _save_frame(frame, out_dir, stem, extracted)
        saved_paths.append(path)
        extracted += 1
        frame_idx += frame_step

    return saved_paths


def extract_frames_scene_change(cap, fps, total_frames, threshold, max_frames, out_dir, stem):
    """Extract frames whenever the scene changes by more than threshold (mean pixel diff).
    max_frames=None means no limit."""
    check_step = max(1, int(fps * 0.5))  # check every 0.5s
    extracted = 0
    frame_idx = 0
    last_gray = None
    saved_paths = []

    while (max_frames is None or extracted < max_frames) and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is None or cv2.absdiff(gray, last_gray).mean() >= threshold:
            path = _save_frame(frame, out_dir, stem, extracted)
            saved_paths.append(path)
            last_gray = gray
            extracted += 1

        frame_idx += check_step

    return saved_paths


def _save_frame(frame, out_dir, stem, index):
    """Save a frame as JPEG. Returns the saved path."""
    filename = f"{stem}_{index:05d}.jpg"
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def annotate_frames(frame_paths, label_dir, yolo_model):
    """Run YOLO on each frame and write YOLO-format .txt label files."""
    os.makedirs(label_dir, exist_ok=True)
    for frame_path in frame_paths:
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        label_path = os.path.join(label_dir, f"{stem}.txt")
        if os.path.exists(label_path):
            continue

        results = yolo_model(frame_path, verbose=False, conf=0.25)
        lines = []
        for r in results:
            img_w, img_h = r.orig_shape[1], r.orig_shape[0]
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w  = (x2 - x1) / img_w
                h  = (y2 - y1) / img_h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="Frame extraction for Roboflow labeling")
    parser.add_argument("--library-dir", default=LIBRARY_DIR,
                        help="Directory to scan for videos (default: H:/)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N videos (default: all videos)")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between frames in uniform mode (default: 3.0)")
    parser.add_argument("--max", type=int, default=None, dest="max_frames",
                        help="Max frames per video (default: unlimited)")
    parser.add_argument("--uniform", action="store_true",
                        help="Use uniform sampling every --interval seconds instead of scene-change")
    parser.add_argument("--threshold", type=float, default=25.0,
                        help="Scene change sensitivity — mean pixel diff to trigger save "
                             "(0-255, lower=more frames, default: 25)")
    parser.add_argument("--annotate", action="store_true",
                        help="Run YOLOv8 COCO-80 auto-annotation (useful as a labeling starting point)")
    parser.add_argument("--subfolders", action="store_true",
                        help="Save frames into per-video subfolders instead of flat directory")
    args = parser.parse_args()

    frames_root = Path(FRAMES_DIR)
    frames_root.mkdir(parents=True, exist_ok=True)

    model = None
    if args.annotate:
        from ultralytics import YOLO
        print("Loading YOLOv8n model for auto-annotation...")
        model = YOLO("yolov8n.pt")
        Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)
        classes_path = Path(LABELS_DIR) / "classes.txt"
        classes_path.write_text('\n'.join(COCO_CLASSES))
        print(f"Saved class list to {classes_path}\n")

    videos = get_videos(args.library_dir, args.n)
    if not videos:
        print(f"No supported videos found in {args.library_dir}")
        return

    mode = f"uniform ({args.interval}s interval)" if args.uniform else "scene-change"
    cap_str = f"max {args.max_frames} frames/video" if args.max_frames else "no frame cap"
    print(f"Found {len(videos)} videos · mode: {mode} · {cap_str}")
    print(f"Output: {frames_root}/\n")

    total_frames_saved = 0

    for i, filename in enumerate(videos, 1):
        video_path = os.path.join(args.library_dir, filename)
        stem = Path(filename).stem

        # Determine output directory
        if args.subfolders:
            out_dir = frames_root / stem
        else:
            out_dir = frames_root
        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[{i}/{len(videos)}] {filename}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("  SKIP — OpenCV could not open file")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_video_frames / fps

        if args.uniform:
            frame_paths = extract_frames_uniform(
                cap, fps, total_video_frames, args.interval, args.max_frames, out_dir, stem
            )
        else:
            frame_paths = extract_frames_scene_change(
                cap, fps, total_video_frames, args.threshold, args.max_frames, out_dir, stem
            )

        cap.release()
        total_frames_saved += len(frame_paths)
        print(f"  {len(frame_paths)} frames extracted  ({duration:.0f}s video)")

        if args.annotate and model and frame_paths:
            label_dir = os.path.join(LABELS_DIR, stem) if args.subfolders else LABELS_DIR
            print(f"  Running YOLO annotation on {len(frame_paths)} frames...")
            annotate_frames(frame_paths, label_dir, model)
            print(f"  Labels saved to {label_dir}")

    print(f"\nDone. {total_frames_saved} total frames saved to {FRAMES_DIR}/")
    if args.annotate:
        print(f"Labels saved to {LABELS_DIR}/")
    print("\nNext step: zip annotation_frames/ and upload to Roboflow.")


if __name__ == "__main__":
    main()
