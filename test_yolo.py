from ultralytics import YOLO
import sys

print("Starting YOLO test...")
try:
    print("Imported ultralytics.")
    model = YOLO("yolov8n.pt")
    print("Loaded YOLO model.")
    results = model("https://ultralytics.com/images/bus.jpg")
    print(f"Inference successful. Detected {len(results[0].boxes)} objects.")
except Exception as e:
    print(f"Error: {e}")
