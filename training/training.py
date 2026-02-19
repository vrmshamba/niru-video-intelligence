"""
training/training.py

YOLOv8 fine-tuning script for NIRU — Nairobi street footage.

Two-phase workflow
------------------
1. PREPARE  (--prepare-only or auto-run before training)
   - Reads annotation_frames/ and annotation_labels/ (COCO-80 auto-labels).
   - Remaps the 8 COCO classes that overlap with the 17 Nairobi classes.
   - Discards label lines for COCO classes with no Nairobi equivalent.
   - Writes images and remapped labels to training/prepared/{images,labels}/{train,val}/.
   - Splits by VIDEO (not by frame) — 80% train / 20% val — to avoid data leakage.

2. TRAIN
   - Calls YOLO("yolov8n.pt").train() with training/dataset.yaml.
   - Results land in runs/detect/niru_nairobi/.

Usage
-----
    # Full prepare + train:
    python training/training.py

    # Prepare only (inspect data before training):
    python training/training.py --prepare-only

    # Train only (dataset already prepared):
    python training/training.py --train-only

    # Override hyperparameters:
    python training/training.py --epochs 100 --imgsz 1280 --batch 8

Notes
-----
- The 9 Nairobi-specific classes (hawker, police_officer, traffic_marshal, etc.)
  have no COCO equivalent. They will show 0 labels after remapping. Add manual
  annotations to annotation_labels/<video>/ for those classes and re-run prepare.
- matatu vs bus: initially both map from COCO "bus". After manual re-labeling,
  re-run prepare to get the proper split.
- bodaboda vs motorcycle: initially both map from COCO "motorcycle". Same caveat.
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root, two levels up from this file)
# ---------------------------------------------------------------------------
THIS_DIR    = Path(__file__).parent                          # training/
PROJECT_DIR = THIS_DIR.parent                                # H:/NIRU_MVP_Dev
FRAMES_DIR  = PROJECT_DIR / "annotation_frames"
LABELS_DIR  = PROJECT_DIR / "annotation_labels"
PREPARED    = THIS_DIR / "prepared"
DATASET_YAML = THIS_DIR / "dataset.yaml"

# ---------------------------------------------------------------------------
# COCO-80 class ID → Nairobi class ID
# -1 means discard (no meaningful equivalent)
# ---------------------------------------------------------------------------
COCO_TO_NAIROBI = {
    0:   6,   # person        → pedestrian
    1:   5,   # bicycle       → bicycle
    2:   2,   # car           → car
    3:   4,   # motorcycle    → motorcycle  (includes bodaboda; differentiate with manual labels)
    5:   0,   # bus           → matatu      (includes bus; differentiate with manual labels)
    7:   3,   # truck         → truck
    9:  11,   # traffic light → traffic_light
    11: 12,   # stop sign     → road_sign
}

VAL_SPLIT = 0.20   # fraction of videos held out for validation
RANDOM_SEED = 42


def remap_label_file(src_txt: Path) -> list[str]:
    """Read a COCO-80 label file and return remapped lines for Nairobi classes."""
    remapped = []
    if not src_txt.exists():
        return remapped
    for line in src_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        coco_cls = int(parts[0])
        nairobi_cls = COCO_TO_NAIROBI.get(coco_cls, -1)
        if nairobi_cls == -1:
            continue
        remapped.append(f"{nairobi_cls} {' '.join(parts[1:])}")
    return remapped


def prepare_dataset(force: bool = False) -> None:
    """
    Build training/prepared/ from annotation_frames/ + annotation_labels/.
    Creates:
        prepared/images/train/   prepared/images/val/
        prepared/labels/train/   prepared/labels/val/
    """
    if PREPARED.exists() and not force:
        print(f"[prepare] {PREPARED} already exists. Use --force-prepare to rebuild.")
        return

    if not FRAMES_DIR.exists():
        sys.exit(f"[prepare] ERROR: {FRAMES_DIR} not found. Run extract_and_annotate.py first.")
    if not LABELS_DIR.exists():
        sys.exit(f"[prepare] ERROR: {LABELS_DIR} not found. Run extract_and_annotate.py first.")

    # Collect video stems that have both frames and labels
    video_stems = sorted([
        d.name for d in FRAMES_DIR.iterdir()
        if d.is_dir() and (LABELS_DIR / d.name).is_dir()
    ])

    if not video_stems:
        sys.exit("[prepare] ERROR: No matching frame/label directories found.")

    print(f"[prepare] Found {len(video_stems)} annotated video(s).")

    # Train/val split by video (prevents frame-level leakage)
    random.seed(RANDOM_SEED)
    shuffled = video_stems[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_SPLIT))
    val_stems  = set(shuffled[:n_val])
    train_stems = set(shuffled[n_val:])
    print(f"[prepare] Split: {len(train_stems)} train videos / {len(val_stems)} val videos.")

    # Create output directories
    for split in ("train", "val"):
        (PREPARED / "images" / split).mkdir(parents=True, exist_ok=True)
        (PREPARED / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train_img": 0, "val_img": 0, "train_lbl": 0, "val_lbl": 0,
             "discarded_imgs": 0, "remapped_lines": 0}

    for stem in video_stems:
        split = "val" if stem in val_stems else "train"
        frame_src_dir = FRAMES_DIR / stem
        label_src_dir = LABELS_DIR / stem
        img_dst_dir   = PREPARED / "images" / split
        lbl_dst_dir   = PREPARED / "labels" / split

        jpg_files = sorted(frame_src_dir.glob("*.jpg"))
        for jpg in jpg_files:
            label_src = label_src_dir / (jpg.stem + ".txt")
            remapped = remap_label_file(label_src)

            # Use video stem as prefix to avoid filename collisions across videos
            safe_stem = stem.replace(" ", "_").replace(",", "")
            dst_name  = f"{safe_stem}__{jpg.name}"
            lbl_name  = f"{safe_stem}__{jpg.stem}.txt"

            # Copy image
            shutil.copy2(jpg, img_dst_dir / dst_name)
            stats[f"{split}_img"] += 1

            # Write remapped label (empty file if no detections → valid for YOLO)
            (lbl_dst_dir / lbl_name).write_text("\n".join(remapped))
            if remapped:
                stats[f"{split}_lbl"] += 1
                stats["remapped_lines"] += len(remapped)

    print(f"[prepare] Train: {stats['train_img']} images, {stats['train_lbl']} with labels.")
    print(f"[prepare] Val  : {stats['val_img']} images, {stats['val_lbl']} with labels.")
    print(f"[prepare] Total remapped label lines: {stats['remapped_lines']}")
    print(f"[prepare] Dataset written to: {PREPARED}")


def train_model(epochs: int, imgsz: int, batch: int, device: str) -> None:
    """Run YOLOv8 fine-tuning."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[train] ERROR: ultralytics not installed. Run: pip install ultralytics")

    # Verify prepared dataset exists
    train_imgs = PREPARED / "images" / "train"
    if not train_imgs.exists() or not any(train_imgs.glob("*.jpg")):
        sys.exit(
            f"[train] ERROR: No training images found at {train_imgs}.\n"
            "        Run with --prepare-only first, or omit --train-only."
        )

    print(f"\n[train] Starting fine-tuning: epochs={epochs} imgsz={imgsz} batch={batch} device={device}")
    print(f"[train] Base model : yolov8n.pt")
    print(f"[train] Dataset    : {DATASET_YAML}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name="niru_nairobi",
        exist_ok=True,
        pretrained=True,
        patience=15,          # early stopping: stop if no improvement for 15 epochs
        save=True,
        plots=True,
        verbose=True,
    )

    best_weights = PROJECT_DIR / "runs" / "detect" / "niru_nairobi" / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[train] Done. Best weights: {best_weights}")
        print(f"[train] Copy to models/ with: cp {best_weights} models/niru_nairobi.pt")
    else:
        print("\n[train] Done. Check runs/detect/niru_nairobi/ for outputs.")


def print_class_summary() -> None:
    """Print expected label counts per Nairobi class after remapping."""
    nairobi_names = {
        0: "matatu", 1: "bus", 2: "car", 3: "truck",
        4: "motorcycle", 5: "bicycle", 6: "pedestrian",
        7: "hawker", 8: "police_officer", 9: "traffic_marshal",
        10: "handcart", 11: "traffic_light", 12: "road_sign",
        13: "market_stall", 14: "matatu_conductor", 15: "crowd", 16: "bodaboda",
    }
    counts = {i: 0 for i in range(17)}

    if not LABELS_DIR.exists():
        print("[summary] annotation_labels/ not found — skipping class summary.")
        return

    for txt in LABELS_DIR.rglob("*.txt"):
        if txt.name == "classes.txt":
            continue
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            coco_cls = int(parts[0])
            nairobi_cls = COCO_TO_NAIROBI.get(coco_cls, -1)
            if nairobi_cls != -1:
                counts[nairobi_cls] += 1

    total = sum(counts.values())
    print(f"\n[summary] Remapped label distribution across all annotation_labels/ ({total} total):")
    for cls_id, name in nairobi_names.items():
        bar = "#" * min(40, counts[cls_id] // max(1, total // 400))
        note = "  <-- manual labels needed" if counts[cls_id] == 0 else ""
        print(f"  {cls_id:2d}  {name:<20s} {counts[cls_id]:6d}  {bar}{note}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="NIRU YOLOv8 fine-tuning — prepare dataset and/or train model."
    )
    parser.add_argument("--prepare-only",   action="store_true",
                        help="Prepare dataset only; do not train.")
    parser.add_argument("--train-only",     action="store_true",
                        help="Train only; skip dataset preparation.")
    parser.add_argument("--force-prepare",  action="store_true",
                        help="Rebuild training/prepared/ even if it exists.")
    parser.add_argument("--epochs",  type=int,   default=50,
                        help="Number of training epochs (default: 50).")
    parser.add_argument("--imgsz",   type=int,   default=640,
                        help="Input image size (default: 640).")
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Batch size (default: 16; reduce if OOM).")
    parser.add_argument("--device",  type=str,   default="0",
                        help="Device: '0' for GPU 0, 'cpu' for CPU (default: '0').")
    parser.add_argument("--summary", action="store_true",
                        help="Print class label distribution and exit.")
    args = parser.parse_args()

    # Change working directory to project root so relative paths in dataset.yaml resolve
    os.chdir(PROJECT_DIR)

    if args.summary:
        print_class_summary()
        return

    if not args.train_only:
        prepare_dataset(force=args.force_prepare)

    if not args.prepare_only:
        train_model(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )


if __name__ == "__main__":
    main()
