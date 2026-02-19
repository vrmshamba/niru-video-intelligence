"""
extract_and_annotate.py

1. Extracts 1 frame per second from a list of videos using OpenCV.
2. Saves frames as JPGs to annotation_frames/<video_stem>/
3. Runs YOLOv8 on every extracted frame.
4. Saves YOLO-format .txt labels to annotation_labels/<video_stem>/
   Each label line: class_id cx_norm cy_norm w_norm h_norm
5. Saves classes.txt (COCO-80 names) alongside the labels.

Usage:
    python extract_and_annotate.py [--videos-dir H:/] [--n 10]
"""

import os
import sys
import argparse
import cv2

# MXF needs ffmpeg pre-conversion for YOLO but OpenCV can open it directly for frame extraction.
# We use OpenCV for frames (works on all formats), then YOLO on the saved JPGs.

FRAMES_DIR  = "annotation_frames"
LABELS_DIR  = "annotation_labels"
LIBRARY_DIR = "H:/"
N_VIDEOS    = 10
SUPPORTED   = ('.mp4', '.avi', '.mov', '.mxf')

# COCO-80 class names (YOLOv8 default)
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


def get_videos(library_dir, n):
    files = sorted([
        f for f in os.listdir(library_dir)
        if os.path.isfile(os.path.join(library_dir, f))
        and f.lower().endswith(SUPPORTED)
    ])
    return files[:n]


def extract_frames(video_path, out_dir):
    """Extract 1 frame per second. Returns list of saved frame paths."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: OpenCV could not open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = int(total_frames / fps)

    print(f"  Duration: ~{duration_s}s @ {fps:.1f}fps — extracting {duration_s} frames")

    saved = []
    for sec in range(duration_s):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"frame_{sec:05d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(out_path)

    cap.release()
    return saved


def annotate_frames(frame_paths, label_dir, yolo_model):
    """Run YOLO on each frame and write YOLO-format .txt labels."""
    os.makedirs(label_dir, exist_ok=True)
    for frame_path in frame_paths:
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        label_path = os.path.join(label_dir, f"{stem}.txt")

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


def save_classes(labels_root):
    classes_path = os.path.join(labels_root, "classes.txt")
    with open(classes_path, 'w') as f:
        f.write('\n'.join(COCO_CLASSES))
    print(f"Saved {classes_path}")


def main():
    parser = argparse.ArgumentParser(description="Frame extraction + YOLO auto-annotation")
    parser.add_argument("--library-dir", default=LIBRARY_DIR)
    parser.add_argument("--n", type=int, default=N_VIDEOS, help="Number of videos to process")
    args = parser.parse_args()

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    from ultralytics import YOLO
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded.\n")

    videos = get_videos(args.library_dir, args.n)
    print(f"Processing {len(videos)} videos from {args.library_dir}\n")

    save_classes(LABELS_DIR)

    for i, filename in enumerate(videos, 1):
        video_path = os.path.join(args.library_dir, filename)
        stem = os.path.splitext(filename)[0]
        frame_dir = os.path.join(FRAMES_DIR, stem)
        label_dir = os.path.join(LABELS_DIR, stem)

        print(f"[{i}/{len(videos)}] {filename}")

        # Step 1: Extract frames
        print("  Extracting frames...")
        frame_paths = extract_frames(video_path, frame_dir)
        print(f"  Saved {len(frame_paths)} frames to {frame_dir}")

        if not frame_paths:
            print("  Skipping annotation (no frames extracted).")
            continue

        # Step 2: Auto-annotate
        print("  Running YOLO annotation...")
        annotate_frames(frame_paths, label_dir, model)
        label_count = sum(1 for f in os.listdir(label_dir) if f.endswith('.txt'))
        print(f"  Saved {label_count} label files to {label_dir}\n")

    print("Done.")


if __name__ == "__main__":
    main()
