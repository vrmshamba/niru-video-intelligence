import sys
print("Starting debug...")
sys.stdout.flush()

print("Importing os/json...")
import os
import json
print("Done.")
sys.stdout.flush()

print("Importing torch...")
import torch
print(f"Done. Version: {torch.__version__}")
sys.stdout.flush()

print("Importing cv2...")
import cv2
print(f"Done. Version: {cv2.__version__}")
sys.stdout.flush()

print("Importing whisper...")
import whisper
print("Done.")
sys.stdout.flush()

print("Applying patch...")
import importlib.metadata
try:
    importlib.metadata.version("torchvision")
except importlib.metadata.PackageNotFoundError:
    print("Patching...")
    original_version = importlib.metadata.version
    def patched_version(package):
        if package == "torchvision":
            return "0.16.0" 
        return original_version(package)
    importlib.metadata.version = patched_version
print("Done.")
sys.stdout.flush()

print("Importing ultralytics...")
from ultralytics import YOLO
print("Done.")
sys.stdout.flush()

print("Debug complete.")
