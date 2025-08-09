import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import subprocess
import tempfile
import time

# Configurable parameters
MODEL = os.getenv("LLM7_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = "https://api.llm7.io/v1/chat/completions"
FONT_PATH = "VT323-Regular.ttf"

# Get YouTube stream key from environment variable
STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY")
if not STREAM_KEY:
    raise ValueError("YOUTUBE_STREAM_KEY environment variable is not set.")
RTMP_URL = f"rtmp://a.rtmp.youtube.com/live2/{STREAM_KEY}"

# Query LLM7 API
def query_llm7(prompt, model=MODEL):
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 150}
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Render a single frame with text
def render_frame(text, width=1920, height=1080):
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 48)
    draw.text((50, 50), text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Write frames to named pipe for ffmpeg
def stream_to_youtube():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_fifo:
        fifo_path = tmp_fifo.name

    os.unlink(fifo_path)
    os.mkfifo(fifo_path)

    # Start ffmpeg process
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "1920x1080",
        "-r", "30",
        "-i", fifo_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-maxrate", "3000k",
        "-bufsize", "6000k",
        "-pix_fmt", "yuv420p",
        "-g", "50",
        "-f", "flv",
        RTMP_URL
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd)

    # Open FIFO for writing frames
    with open(fifo_path, "wb") as fifo:
        while True:
            q = "What is consciousness?"
            a = query_llm7(q)
            frames = [
                render_frame(f"Q: {q}"),
                render_frame(f"A: {a}")
            ]
            for frame in frames:
                fifo.write(frame.tobytes())
                time.sleep(1/30)  # maintain ~30fps

if __name__ == "__main__":
    print("Starting YouTube Live stream worker...")
    stream_to_youtube()
