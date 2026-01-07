import torch
import whisper
import os

def test_environment():
    print("Testing environment...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        print("Loading Whisper 'tiny' model...")
        model = whisper.load_model("tiny", download_root="models")
        print("✅ Whisper model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")

    try:
        import cv2
        print(f"✅ OpenCV imported: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found.")

    try:
        import ultralytics
        print(f"✅ Ultralytics imported: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not found.")
        
    try:
        import PIL
        print(f"✅ Pillow imported: {PIL.__version__}")
    except ImportError:
        print("❌ Pillow not found.")

if __name__ == "__main__":
    test_environment()
