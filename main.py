import argparse
import os
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import subprocess
from flask import Flask, send_file

# Flask app for Render deployment
app = Flask(__name__)

# Configurable parameters
MODEL = os.getenv("LLM7_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = "https://api.llm7.io/v1/chat/completions"
API_KEY = os.getenv("LLM7_API_KEY")
FONT_PATH = "VT323-Regular.ttf"

# Function to query LLM7 API
def query_llm7(prompt, model=MODEL):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 150}
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# CRT effect renderer
def render_frame(text, width=1920, height=1080):
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 48)
    draw.text((50, 50), text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Generate video from frames
def generate_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

@app.route('/')
def index():
    return "TerminalMind01 Render API running. Access /generate to produce video."

@app.route('/generate')
def generate():
    frames = []
    q = "What is consciousness?"
    a = query_llm7(q)
    frames.append(render_frame(f"Q: {q}"))
    frames.append(render_frame(f"A: {a}"))
    output_path = "output.mp4"
    generate_video(frames, output_path)
    return send_file(output_path, mimetype='video/mp4')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
