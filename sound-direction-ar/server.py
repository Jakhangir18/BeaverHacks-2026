import subprocess
import re
import numpy as np
import sounddevice as sd
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time

app = Flask(__name__, static_folder=".")
CORS(app)

# Path to ReSpeaker XVF3800 host control
XVF_HOST = '/home/rasberypi/Documents/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/rpi_64bit/xvf_host'

# Config
VOLUME_THRESHOLD = 0.004
SAMPLE_RATE = 16000
SAMPLE_DURATION = 0.15

current_state = {
    "direction": "front",
    "volume": 0,
    "danger": False
}

def get_doa_angle():
    """Get Direction of Arrival (angle in degrees 0-360)"""
    try:
        result = subprocess.run(
            ['sudo', XVF_HOST, 'AEC_AZIMUTH_VALUES'],
            capture_output=True,
            text=True,
            timeout=1
        )
        matches = re.findall(r'\((\d+\.\d+) deg\)', result.stdout)
        if matches:
            return float(matches[-1])
    except:
        pass
    return None

def get_volume_level():
    """Get RMS loudness from microphone"""
    try:
        samples = int(SAMPLE_RATE * SAMPLE_DURATION)
        audio = sd.rec(samples, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        rms = np.sqrt(np.mean(audio ** 2))
        return float(rms)
    except:
        return 0.0

def angle_to_direction(angle):
    """Convert DOA angle (0-360) to 8 directions"""
    if angle is None:
        return "front"
    
    # Normalize angle to 0-360
    angle = angle % 360
    
    # Map angle to 8 directions (45° per direction)
    # 0° = front, 45° = front-right, 90° = right, etc.
    directions = ['front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left']
    idx = int((angle + 22.5) / 45.0) % 8
    return directions[idx]

def normalize_volume(rms):
    """Convert RMS (0.0-0.1) to volume 0-100"""
    # Adjust these thresholds based on your microphone
    if rms < 0.003:
        return 0
    elif rms > 0.05:
        return 100
    else:
        return int((rms - 0.003) / 0.047 * 100)

def poll_microphone():
    """Background thread: continuously poll microphone"""
    global current_state
    
    while True:
        try:
            # Get audio data
            volume_rms = get_volume_level()
            volume = normalize_volume(volume_rms)
            
            # Only process if sound detected
            if volume_rms >= VOLUME_THRESHOLD:
                angle = get_doa_angle()
                direction = angle_to_direction(angle)
            else:
                direction = "front"
                volume = 0
            
            # Update state
            current_state["direction"] = direction
            current_state["volume"] = volume
            current_state["danger"] = volume > 75
            
            print(f"📡 {direction} | volume={volume} | RMS={volume_rms:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(0.3)

# Flask routes
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/state")
def get_state():
    return jsonify(current_state)

if __name__ == "__main__":
    # Start microphone polling in background
    mic_thread = threading.Thread(target=poll_microphone, daemon=True)
    mic_thread.start()
    print("🎙️ Microphone polling started")
    
    # Start Flask server
    app.run(host="0.0.0.0", port=5000, debug=False, ssl_context=("192.168.68.55.pem", "192.168.68.55-key.pem"))
