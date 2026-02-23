"""
training/training.py

YOLOv8 fine-tuning script for NIRU — Nairobi street footage.

RECOMMENDED WORKFLOW (Roboflow)
--------------------------------
1. Label frames on Roboflow with your 20 Nairobi classes.
2. Export dataset from Roboflow → YOLOv8 format → download zip.
3. Extract zip to a local folder, e.g. roboflow_export/.
4. Run:
       python training/training.py --roboflow-export roboflow_export/
   This uses Roboflow's data.yaml directly (train/valid splits already done).

FALLBACK WORKFLOW (COCO auto-labels only)
------------------------------------------
1. Run extract_and_annotate.py --annotate  to generate COCO-80 auto-labels.
2. Run:
       python training/training.py --prepare-only   # remap COCO → Nairobi, split dataset
       python training/training.py --train-only     # train on prepared/
   Or combined:
       python training/training.py

Other options
-------------
    python training/training.py --epochs 100 --imgsz 1280 --batch 8
    python training/training.py --summary    # show label counts per class

Notes
-----
- 15 of the 20 Nairobi classes need manual annotation (nganya, city_hoppa,
  school_bus, large_bus, bodaboda_no_helmet, bodaboda_stage, tuk_tuk, mkokoteni,
  street_hawkers, traffic_marshal, conductor, matatu_stage, illegal_dumping,
  garbage_pile, pothole). They show 0 labels until manually annotated.
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
    0:  14,   # person        → passenger      (street_hawkers/conductor/marshal need manual split)
    3:   5,   # motorcycle    → bodaboda        (bodaboda_no_helmet/tuk_tuk need manual split)
    5:   0,   # bus           → matatu          (nganya/city_hoppa/school_bus/large_bus need manual split)
    9:  17,   # traffic light → traffic_lights
    11: 18,   # stop sign     → road_sign
    # COCO 1 (bicycle), 2 (car), 7 (truck) → discarded (no equivalent in 20-class schema)
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

    # Support both flat layout (video_stem_NNNNN.jpg) and subfolder layout
    # Flat: annotation_frames/MATATU CBD NRB_00001.jpg
    # Subfolder: annotation_frames/MATATU CBD NRB/frame_00001.jpg
    flat_jpgs = sorted(FRAMES_DIR.glob("*.jpg"))

    if flat_jpgs:
        # Flat layout — derive video stem from filename prefix (everything before last _NNNNN)
        stem_map = {}  # stem → [jpg_paths]
        for jpg in flat_jpgs:
            # filename: "{video_stem}_{index:05d}.jpg"
            parts = jpg.stem.rsplit("_", 1)
            stem = parts[0] if len(parts) == 2 and parts[1].isdigit() else jpg.stem
            stem_map.setdefault(stem, []).append(jpg)
        video_stems = sorted(stem_map.keys())
        use_flat = True
    else:
        # Subfolder layout (legacy)
        video_stems = sorted([
            d.name for d in FRAMES_DIR.iterdir()
            if d.is_dir() and (LABELS_DIR / d.name).is_dir()
        ])
        stem_map = {}
        use_flat = False

    if not video_stems:
        sys.exit("[prepare] ERROR: No frames found in annotation_frames/. Run extract_and_annotate.py first.")

    print(f"[prepare] Found {len(video_stems)} video stem(s) ({'flat' if use_flat else 'subfolder'} layout).")

    # Train/val split by video stem (prevents frame-level leakage)
    random.seed(RANDOM_SEED)
    shuffled = video_stems[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_SPLIT))
    val_stems   = set(shuffled[:n_val])
    train_stems = set(shuffled[n_val:])
    print(f"[prepare] Split: {len(train_stems)} train videos / {len(val_stems)} val videos.")

    # Create output directories
    for split in ("train", "val"):
        (PREPARED / "images" / split).mkdir(parents=True, exist_ok=True)
        (PREPARED / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train_img": 0, "val_img": 0, "train_lbl": 0, "val_lbl": 0,
             "remapped_lines": 0}

    for stem in video_stems:
        split = "val" if stem in val_stems else "train"
        img_dst_dir = PREPARED / "images" / split
        lbl_dst_dir = PREPARED / "labels" / split

        if use_flat:
            jpg_files = sorted(stem_map[stem])
            label_src_dir = LABELS_DIR  # labels are also flat
        else:
            jpg_files = sorted((FRAMES_DIR / stem).glob("*.jpg"))
            label_src_dir = LABELS_DIR / stem

        for jpg in jpg_files:
            label_src = label_src_dir / (jpg.stem + ".txt")
            remapped = remap_label_file(label_src)

            # Sanitise stem for use in destination filename
            safe_stem = stem.replace(" ", "_").replace(",", "")
            dst_name = f"{safe_stem}__{jpg.name}"
            lbl_name = f"{safe_stem}__{jpg.stem}.txt"

            shutil.copy2(jpg, img_dst_dir / dst_name)
            stats[f"{split}_img"] += 1

            (lbl_dst_dir / lbl_name).write_text("\n".join(remapped))
            if remapped:
                stats[f"{split}_lbl"] += 1
                stats["remapped_lines"] += len(remapped)

    print(f"[prepare] Train: {stats['train_img']} images, {stats['train_lbl']} with labels.")
    print(f"[prepare] Val  : {stats['val_img']} images, {stats['val_lbl']} with labels.")
    print(f"[prepare] Total remapped label lines: {stats['remapped_lines']}")
    print(f"[prepare] Dataset written to: {PREPARED}")


def train_model(epochs: int, imgsz: int, batch: int, device: str,
                roboflow_export: str | None = None) -> None:
    """Run YOLOv8 fine-tuning.

    roboflow_export: path to folder containing Roboflow's exported data.yaml.
                     If given, that data.yaml is used directly instead of
                     training/data.yaml, bypassing the prepare step.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[train] ERROR: ultralytics not installed. Run: pip install ultralytics")

    if roboflow_export:
        data_yaml = Path(roboflow_export) / "data.yaml"
        if not data_yaml.exists():
            sys.exit(f"[train] ERROR: data.yaml not found in {roboflow_export}")
        print(f"[train] Using Roboflow export: {data_yaml}")
    else:
        data_yaml = DATASET_YAML
        train_imgs = PREPARED / "images" / "train"
        if not train_imgs.exists() or not any(train_imgs.glob("*.jpg")):
            sys.exit(
                f"[train] ERROR: No training images found at {train_imgs}.\n"
                "        Run --prepare-only first, or use --roboflow-export."
            )

    print(f"\n[train] Starting fine-tuning: epochs={epochs} imgsz={imgsz} batch={batch} device={device}")
    print(f"[train] Base model : yolov8n.pt")
    print(f"[train] Dataset    : {data_yaml}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
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
        0:  "matatu",
        1:  "nganya",
        2:  "city_hoppa",
        3:  "school_bus",
        4:  "large_bus",
        5:  "bodaboda",
        6:  "bodaboda_no_helmet",
        7:  "bodaboda_stage",
        8:  "tuk_tuk",
        9:  "mkokoteni",
        10: "street_hawkers",
        11: "traffic_marshal",
        12: "conductor",
        13: "matatu_stage",
        14: "passenger",
        15: "illegal_dumping",
        16: "garbage_pile",
        17: "traffic_lights",
        18: "road_sign",
        19: "pothole",
    }
    counts = {i: 0 for i in range(20)}

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
    parser.add_argument("--roboflow-export", type=str, default=None, metavar="DIR",
                        help="Path to Roboflow export folder containing data.yaml. "
                             "Skips prepare step and trains directly on Roboflow's splits.")
    parser.add_argument("--summary", action="store_true",
                        help="Print class label distribution and exit.")
    args = parser.parse_args()

    # Change working directory to project root so relative paths in dataset.yaml resolve
    os.chdir(PROJECT_DIR)

    if args.summary:
        print_class_summary()
        return

    if args.roboflow_export:
        # Roboflow path: skip prepare, train directly from export
        train_model(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            roboflow_export=args.roboflow_export,
        )
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
