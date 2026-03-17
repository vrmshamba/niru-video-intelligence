"""
training/rank_frames.py

Ranks annotation frames by likelihood of containing hard-to-label classes:
  nganya, city_hoppa, bodaboda_no_helmet, conductor, street_hawkers, pothole

Heuristics use existing COCO-80 auto-labels as proxy signals:
  COCO 0  (person)     → proxy for conductor / street_hawkers
  COCO 3  (motorcycle) → proxy for bodaboda_no_helmet
  COCO 5  (bus)        → proxy for nganya / city_hoppa
  video name keywords  → proxy for pothole (ground-level driving footage)

Output: top 50 frames with score breakdown, ready to paste into Label Studio.

Usage:
    python training/rank_frames.py
    python training/rank_frames.py --top 100 --out training/frames_to_label.txt
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field

THIS_DIR    = Path(__file__).parent
PROJECT_DIR = THIS_DIR.parent
LABELS_DIR  = PROJECT_DIR / "annotation_labels"
FRAMES_DIR  = PROJECT_DIR / "annotation_frames"

# Video name keywords that suggest ground-level road footage (pothole proxy)
GROUND_LEVEL_KEYWORDS = [
    "walking", "cbd", "cabro", "street", "backshot", "phone", "hawker",
]


@dataclass
class FrameScore:
    video:    str
    frame:    str
    img_path: str
    # per-class scores (0-20 each, total max 120)
    s_nganya:          int = 0
    s_city_hoppa:      int = 0
    s_bodaboda_helmet: int = 0
    s_conductor:       int = 0
    s_hawkers:         int = 0
    s_pothole:         int = 0

    @property
    def total(self) -> int:
        return (self.s_nganya + self.s_city_hoppa + self.s_bodaboda_helmet
                + self.s_conductor + self.s_hawkers + self.s_pothole)

    def breakdown(self) -> str:
        parts = []
        if self.s_nganya:          parts.append(f"nganya:{self.s_nganya}")
        if self.s_city_hoppa:      parts.append(f"city_hoppa:{self.s_city_hoppa}")
        if self.s_bodaboda_helmet: parts.append(f"no_helmet:{self.s_bodaboda_helmet}")
        if self.s_conductor:       parts.append(f"conductor:{self.s_conductor}")
        if self.s_hawkers:         parts.append(f"hawkers:{self.s_hawkers}")
        if self.s_pothole:         parts.append(f"pothole:{self.s_pothole}")
        return "  ".join(parts) if parts else "—"


def parse_label_file(txt: Path):
    """Return list of (class_id, cx, cy, w, h) tuples."""
    detections = []
    if not txt.exists():
        return detections
    for line in txt.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls  = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            detections.append((cls, cx, cy, w, h))
        except ValueError:
            continue
    return detections


def score_frame(video_stem: str, detections: list, is_ground_level: bool) -> dict:
    buses  = [(cx, cy, w, h) for cls, cx, cy, w, h in detections if cls == 5]
    motos  = [(cx, cy, w, h) for cls, cx, cy, w, h in detections if cls == 3]
    people = [(cx, cy, w, h) for cls, cx, cy, w, h in detections if cls == 0]
    cars   = [(cx, cy, w, h) for cls, cx, cy, w, h in detections if cls == 2]

    n_bus    = len(buses)
    n_moto   = len(motos)
    n_person = len(people)
    n_car    = len(cars)

    # Bus bounding-box areas (normalised w*h)
    bus_areas = [w * h for _, _, w, h in buses]
    max_bus_area = max(bus_areas) if bus_areas else 0.0

    # ── nganya  ──────────────────────────────────────────────────────────────
    # Medium-sized bus (area 0.04–0.20); more than one bus in frame is a bonus
    s_nganya = 0
    if n_bus >= 1:
        s_nganya += 8
        if 0.04 <= max_bus_area <= 0.20:
            s_nganya += 8    # medium frame size = matatu/nganya range
        if n_bus >= 2:
            s_nganya += 4    # multiple buses = stage / terminus

    # ── city_hoppa  ──────────────────────────────────────────────────────────
    # Large bus (area > 0.20); often accompanied by many people at a bus stop
    s_city_hoppa = 0
    if n_bus >= 1:
        s_city_hoppa += 5
        if max_bus_area > 0.20:
            s_city_hoppa += 10   # large = full-size bus
        if n_person >= 5:
            s_city_hoppa += 5    # crowd at a city hoppa stop

    # ── bodaboda_no_helmet  ──────────────────────────────────────────────────
    # Motorcycle in an active street scene; multiple motos = stage likely
    s_boda_helmet = 0
    if n_moto >= 1:
        s_boda_helmet += 8
        if n_person >= 2:
            s_boda_helmet += 6   # operating in traffic / carrying passengers
        if n_moto >= 2:
            s_boda_helmet += 6   # bodaboda stage

    # ── conductor  ───────────────────────────────────────────────────────────
    # Person(s) in close proximity to a bus = likely conductor / tout interaction
    s_conductor = 0
    if n_bus >= 1 and n_person >= 1:
        s_conductor += 10
        if n_person >= 3:
            s_conductor += 6    # several people = boarding / touting scene
        if n_bus >= 2:
            s_conductor += 4    # terminus with multiple matatus

    # ── street_hawkers  ──────────────────────────────────────────────────────
    # Dense pedestrian scene with few vehicles = open-air market / hawker zone
    s_hawkers = 0
    if n_person >= 5:
        s_hawkers += 8
        if n_bus == 0 and n_car <= 1:
            s_hawkers += 8       # pedestrian-dominated = hawker context
        if n_person >= 15:
            s_hawkers += 4       # very dense crowd

    # ── pothole  ─────────────────────────────────────────────────────────────
    # Ground-level footage + sparse detections = road surface more visible
    s_pothole = 0
    if is_ground_level:
        s_pothole += 10
        total_dets = len(detections)
        if total_dets <= 3:
            s_pothole += 6    # clear road with few objects = potholes visible
        if total_dets == 0:
            s_pothole += 4    # empty road frame

    return dict(
        s_nganya=min(20, s_nganya),
        s_city_hoppa=min(20, s_city_hoppa),
        s_bodaboda_helmet=min(20, s_boda_helmet),
        s_conductor=min(20, s_conductor),
        s_hawkers=min(20, s_hawkers),
        s_pothole=min(20, s_pothole),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50, help="Number of frames to output")
    parser.add_argument("--out", type=str, default="", help="Save list to file")
    args = parser.parse_args()

    if not LABELS_DIR.exists():
        sys.exit(f"ERROR: {LABELS_DIR} not found. Run extract_and_annotate.py first.")

    results: list[FrameScore] = []

    video_dirs = sorted([d for d in LABELS_DIR.iterdir() if d.is_dir()])
    print(f"Scanning {len(video_dirs)} video label directories...")

    for label_dir in video_dirs:
        video_stem = label_dir.name
        video_lower = video_stem.lower()
        is_ground_level = any(kw in video_lower for kw in GROUND_LEVEL_KEYWORDS)

        txt_files = sorted(label_dir.glob("*.txt"))
        for txt in txt_files:
            detections = parse_label_file(txt)
            scores = score_frame(video_stem, detections, is_ground_level)

            # Only keep frames with at least one positive signal
            if all(v == 0 for v in scores.values()):
                continue

            img_rel = str(FRAMES_DIR / video_stem / (txt.stem + ".jpg"))

            fs = FrameScore(
                video=video_stem,
                frame=txt.stem,
                img_path=img_rel,
                **scores,
            )
            results.append(fs)

    results.sort(key=lambda x: x.total, reverse=True)
    top = results[:args.top]

    header = f"\nTop {args.top} frames most likely to contain target classes\n"
    header += f"{'RANK':<5} {'SCORE':<6} {'VIDEO':<45} {'FRAME':<12} SIGNALS\n"
    header += "-" * 110

    lines = [header]
    for rank, fs in enumerate(top, 1):
        video_short = fs.video[:44]
        line = f"{rank:<5} {fs.total:<6} {video_short:<45} {fs.frame:<12} {fs.breakdown()}"
        lines.append(line)

    output = "\n".join(lines)
    print(output)

    # Also print absolute image paths for easy copy-paste into labeling tools
    print(f"\n{'-'*60}")
    print("Image paths (for Label Studio / Roboflow import):")
    print(f"{'-'*60}")
    for fs in top:
        print(fs.img_path)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(output + "\n\n")
            f.write("Image paths:\n")
            for fs in top:
                f.write(fs.img_path + "\n")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
