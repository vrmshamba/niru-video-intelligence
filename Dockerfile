FROM python:3.10-slim

# System deps for OpenCV, ffmpeg, and audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces runs on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
