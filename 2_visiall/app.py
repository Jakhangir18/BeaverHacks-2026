import subprocess
import sounddevice as sd
import numpy as np
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading
import re

app = Flask(__name__)
CORS(app)

XVF_HOST = "/home/rasberypi/Documents/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/rpi_64bit/xvf_host"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

RMS_MIN = 0.003
RMS_MAX = 0.05
DANGER_THRESHOLD = 0.7

_angle = 0.0
_volume = 0.0
_lock = threading.Lock()


def angle_to_direction(angle):
    boundaries = [
        (22.5,  'front'),
        (67.5,  'front-right'),
        (112.5, 'right'),
        (157.5, 'back-right'),
        (202.5, 'back'),
        (247.5, 'back-left'),
        (292.5, 'left'),
        (337.5, 'front-left'),
        (360.0, 'front'),
    ]
    for threshold, name in boundaries:
        if angle < threshold:
            return name
    return 'front'


def audio_callback(indata, frames, time, status):
    global _volume
    rms = float(np.sqrt(np.mean(indata ** 2)))
    vol = (rms - RMS_MIN) / (RMS_MAX - RMS_MIN)
    vol = max(0.0, min(1.0, vol))
    with _lock:
        _volume = vol


def doa_loop():
    global _angle
    while True:
        try:
            result = subprocess.run(
                ['sudo', XVF_HOST, 'AEC_AZIMUTH_VALUES'],
                capture_output=True, text=True, timeout=2
            )
            # Output format: "... (123.45 deg) ..."
            matches = re.findall(r'\((\d+\.\d+) deg\)', result.stdout)
            if matches:
                angle = float(matches[-1]) % 360
                with _lock:
                    _angle = angle
        except Exception:
            pass


threading.Thread(target=doa_loop, daemon=True).start()

try:
    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
    )
    _stream.start()
except Exception as e:
    print(f"[audio] {e}")


@app.route('/state')
def state():
    with _lock:
        angle = _angle
        vol = _volume
    direction = angle_to_direction(angle) if vol > 0.02 else None
    return jsonify({
        'direction': direction,
        'angle':     round(angle, 1),
        'volume':    round(vol, 3),
        'danger':    vol >= DANGER_THRESHOLD,
    })


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/manifest.json')
def manifest():
    return send_from_directory('.', 'manifest.json')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        ssl_context=('192.168.68.55.pem', '192.168.68.55-key.pem'),
    )
