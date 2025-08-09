import os
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
from flask import Flask
import time

# Flask app for Render deployment
app = Flask(__name__)

# Configurable parameters
MODEL = os.getenv("LLM7_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = "https://api.llm7.io/v1/chat/completions"
FONT_PATH = os.path.join(os.path.dirname(__file__), "VT323-Regular.ttf")

# YouTube Live settings
YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY")
YOUTUBE_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"
FPS = 30
WIDTH = 1920
HEIGHT = 1080

# Query LLM7 API
def query_llm7(prompt, model=MODEL):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Render a CRT-style frame
def render_frame(text):
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 48)
    draw.text((50, 50), text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Start streaming to YouTube Live
def stream_to_youtube():
    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{WIDTH}x{HEIGHT}",
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'veryfast',
        '-f', 'flv',
        YOUTUBE_URL
    ]

    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    try:
        while True:
            q = "What is consciousness?"
            a = query_llm7(q)

            # Create a few frames for Q and A
            for _ in range(FPS * 3):  # Show Q for 3 seconds
                process.stdin.write(render_frame(f"Q: {q}").tobytes())
            for _ in range(FPS * 5):  # Show A for 5 seconds
                process.stdin.write(render_frame(f"A: {a}").tobytes())

            # Pause before next question
            time.sleep(1)

    except BrokenPipeError:
        print("FFmpeg pipe closed. Stream ended.")
    finally:
        process.stdin.close()
        process.wait()

@app.route('/')
def index():
    return "TerminalMind01 Render API running. Visit /stream to start YouTube Live."

@app.route('/stream')
def stream():
    if not YOUTUBE_STREAM_KEY:
        return "Missing YOUTUBE_STREAM_KEY environment variable.", 500
    stream_to_youtube()
    return "Streaming started to YouTube Live!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
