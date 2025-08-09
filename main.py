import os
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
from flask import Flask, jsonify
import time
import threading
import uuid
import sys
from typing import Optional

app = Flask(__name__)

# -----------------------
# Configuration (via env)
# -----------------------
MODEL = os.getenv("LLM7_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = os.getenv("LLM7_API_URL", "https://api.llm7.io/v1/chat/completions")
# LLM7 does not require API key per your note, so we don't use one.

FONT_FILENAME = os.getenv("FONT_FILENAME", "VT323-Regular.ttf")
FONT_PATH = os.path.join(os.path.dirname(__file__), FONT_FILENAME)

YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY", "")
YOUTUBE_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"

# Video settings
FPS = int(os.getenv("FPS", "30"))
WIDTH = int(os.getenv("WIDTH", "1920"))
HEIGHT = int(os.getenv("HEIGHT", "1080"))
Q_SECONDS = int(os.getenv("Q_SECONDS", "3"))   # show question this many seconds
A_SECONDS = int(os.getenv("A_SECONDS", "5"))   # show answer this many seconds

# Globals to manage background streamer
STREAM_THREAD: Optional[threading.Thread] = None
STOP_EVENT = threading.Event()
FFMPEG_PROCESS: Optional[subprocess.Popen] = None
STREAM_LOCK = threading.Lock()
IS_STREAMING = False

# -----------------------
# Utility: query LLM7
# -----------------------
def query_llm7(prompt: str, model: str = MODEL) -> str:
    """Query LLM7 and return the assistant content text."""
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150
        }
        resp = requests.post(API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        # Defensive: navigate response structure that LLM7 uses
        return body["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[query_llm7] Error querying LLM7: {e}", file=sys.stderr)
        return f"(LLM7 error: {e})"

# -----------------------
# Utility: render frame
# -----------------------
def load_font(size=48):
    """Try to load the configured TTF. Fall back to PIL default if unavailable."""
    try:
        font = ImageFont.truetype(FONT_PATH, size)
        return font
    except Exception as e:
        print(f"[load_font] Could not load '{FONT_PATH}': {e}. Using default font.", file=sys.stderr)
        return ImageFont.load_default()

def render_frame(text: str, width=WIDTH, height=HEIGHT) -> np.ndarray:
    """Return a single BGR frame (numpy array) with CRT-like green text on black."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = load_font(48)

    # Simple text layout: left margin and top margin
    margin_x, margin_y = 50, 50
    draw.text((margin_x, margin_y), text, font=font, fill=(0, 255, 0))

    # Optionally: add a simple scanline effect (subtle)
    arr = np.array(img)
    # darken every other row slightly to simulate scanlines
    arr[::2] = (arr[::2] * 0.95).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# -----------------------
# Streaming worker
# -----------------------
def _start_ffmpeg_process() -> subprocess.Popen:
    """Start the ffmpeg subprocess that accepts rawvideo from stdin and pushes to YouTube RTMP."""
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-f", "flv",
        YOUTUBE_URL
    ]
    print(f"[ffmpeg] Starting ffmpeg with command: {' '.join(ffmpeg_command)}")
    proc = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc

def stream_to_youtube_loop():
    """Main background loop that queries LLM7 and pipes frames into ffmpeg until STOP_EVENT is set."""
    global FFMPEG_PROCESS, IS_STREAMING
    print("[stream] Background streamer starting...")
    try:
        FFMPEG_PROCESS = _start_ffmpeg_process()
    except Exception as e:
        print(f"[stream] Failed to start ffmpeg: {e}", file=sys.stderr)
        IS_STREAMING = False
        return

    try:
        while not STOP_EVENT.is_set():
            # Example content - you can swap out for any prompts or logic
            q = "What is consciousness?"
            a = query_llm7(q)

            # Produce Q frames
            frames_q = FPS * Q_SECONDS
            frame_q_bgr = render_frame(f"Q: {q}")
            for i in range(frames_q):
                if STOP_EVENT.is_set():
                    break
                try:
                    # write bytes
                    FFMPEG_PROCESS.stdin.write(frame_q_bgr.tobytes())
                except BrokenPipeError:
                    print("[stream] BrokenPipeError while writing Q frames. Exiting loop.", file=sys.stderr)
                    STOP_EVENT.set()
                    break

            if STOP_EVENT.is_set():
                break

            # Produce A frames
            frames_a = FPS * A_SECONDS
            frame_a_bgr = render_frame(f"A: {a}")
            for i in range(frames_a):
                if STOP_EVENT.is_set():
                    break
                try:
                    FFMPEG_PROCESS.stdin.write(frame_a_bgr.tobytes())
                except BrokenPipeError:
                    print("[stream] BrokenPipeError while writing A frames. Exiting loop.", file=sys.stderr)
                    STOP_EVENT.set()
                    break

            # small pause between Q/A cycles (non-blocking check)
            for _ in range(5):
                if STOP_EVENT.is_set():
                    break
                time.sleep(0.2)

        print("[stream] STOP_EVENT set or loop ended. Closing ffmpeg stdin.")
    except Exception as e:
        print(f"[stream] Unexpected error in streaming loop: {e}", file=sys.stderr)
    finally:
        # close ffmpeg stdin and wait a moment
        try:
            if FFMPEG_PROCESS and FFMPEG_PROCESS.stdin:
                try:
                    FFMPEG_PROCESS.stdin.close()
                except Exception:
                    pass
            # try to gracefully terminate
            if FFMPEG_PROCESS:
                print("[stream] Terminating ffmpeg process...")
                FFMPEG_PROCESS.terminate()
                try:
                    FFMPEG_PROCESS.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[stream] ffmpeg did not exit in time; killing", file=sys.stderr)
                    FFMPEG_PROCESS.kill()
        except Exception as e:
            print(f"[stream] Error cleaning ffmpeg process: {e}", file=sys.stderr)
        finally:
            FFMPEG_PROCESS = None
            IS_STREAMING = False
            STOP_EVENT.clear()
            print("[stream] Background streamer stopped.")

# -----------------------
# Flask endpoints
# -----------------------
@app.route("/")
def index():
    return jsonify({"status": "TerminalMind Render API running", "routes": ["/stream", "/stop", "/status"]})

@app.route("/stream")
def stream_start():
    """Start streamer in background thread and return immediately."""
    global STREAM_THREAD, IS_STREAMING

    if not YOUTUBE_STREAM_KEY:
        return jsonify({"error": "Missing YOUTUBE_STREAM_KEY environment variable."}), 500

    with STREAM_LOCK:
        if IS_STREAMING:
            return jsonify({"status": "already_streaming"})
        # reset any previous stop flags
        STOP_EVENT.clear()
        STREAM_THREAD = threading.Thread(target=stream_to_youtube_loop, daemon=True)
        STREAM_THREAD.start()
        IS_STREAMING = True
        print("[/stream] Started background streaming thread.")
        return jsonify({"status": "stream_started"})

@app.route("/stop")
def stream_stop():
    """Signal the background streamer to stop gracefully."""
    global STREAM_THREAD, IS_STREAMING
    with STREAM_LOCK:
        if not IS_STREAMING:
            return jsonify({"status": "not_streaming"})
        print("[/stop] Stop requested; signalling background streamer...")
        STOP_EVENT.set()
        # Wait a short while for thread to stop (non-blocking)
        timeout = 10.0
        start = time.time()
        while STREAM_THREAD and STREAM_THREAD.is_alive() and (time.time() - start) < timeout:
            time.sleep(0.2)
        if STREAM_THREAD and STREAM_THREAD.is_alive():
            print("[/stop] Background thread did not stop within timeout; ffmpeg should be terminating.")
        IS_STREAMING = False
        return jsonify({"status": "stop_requested"})

@app.route("/status")
def status():
    return jsonify({
        "is_streaming": IS_STREAMING,
        "ffmpeg_running": FFMPEG_PROCESS is not None and (FFMPEG_PROCESS.poll() is None),
        "youtube_url": "configured" if YOUTUBE_STREAM_KEY else "missing",
        "fps": FPS,
        "resolution": f"{WIDTH}x{HEIGHT}"
    })

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    # Note: keep debug off for production on Render
    port = int(os.environ.get("PORT", "5000"))
    print(f"[main] Starting Flask on 0.0.0.0:{port}. Press CTRL+C to quit.")
    app.run(host="0.0.0.0", port=port)
