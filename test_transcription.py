import os
from process_videos import VideoProcessor

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.load_models()
    
    video_path = r"d:\NIRU_MVP_Dev\videos\documentary isiolo.mp4"
    if os.path.exists(video_path):
        print(f"Testing transcription on: {video_path}")
        try:
            processor.process_video(video_path)
            print("Successfully processed documentary isiolo.mp4")
        except Exception as e:
            print(f"Failed: {e}")
    else:
        print("Video not found.")
