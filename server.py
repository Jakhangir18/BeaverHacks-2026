import subprocess
import re
import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np

XVF_HOST = '/home/rasberypi/Documents/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/rpi_64bit/xvf_host'
CLIENTS = set()

# Tune these
VOLUME_THRESHOLD = 0.025       # increase this if it still reacts to noise
MIN_ANGLE_CHANGE = 8.0         # degrees; ignore tiny direction changes
POLL_INTERVAL = 0.3            # seconds
SAMPLE_RATE = 16000
SAMPLE_DURATION = 0.15         # seconds


def get_doa_angle():
    result = subprocess.run(
        ['sudo', XVF_HOST, 'AEC_AZIMUTH_VALUES'],
        capture_output=True,
        text=True
    )

    matches = re.findall(r'\((\d+\.\d+) deg\)', result.stdout)

    if matches:
        return float(matches[-1])

    return None


def get_volume_level():
    """
    Returns RMS loudness from the microphone.
    Higher number = louder sound.
    """

    samples = int(SAMPLE_RATE * SAMPLE_DURATION)

    audio = sd.rec(
        samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    rms = np.sqrt(np.mean(audio ** 2))
    return float(rms)


def angle_difference(a, b):
    """
    Smallest difference between two angles.
    Example: 359° and 1° are 2° apart, not 358°.
    """

    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


async def handle_client(websocket):
    CLIENTS.add(websocket)
    print(f"Client connected. Total: {len(CLIENTS)}")

    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        print(f"Client disconnected. Total: {len(CLIENTS)}")


async def poll_mic():
    last_angle = None

    while True:
        volume = get_volume_level()

        if volume < VOLUME_THRESHOLD:
            # Uncomment this while tuning:
            # print(f"quiet: volume={volume:.4f}")
            await asyncio.sleep(POLL_INTERVAL)
            continue

        angle = get_doa_angle()

        if angle is not None:
            should_send = (
                last_angle is None or
                angle_difference(angle, last_angle) >= MIN_ANGLE_CHANGE
            )

            if should_send:
                last_angle = angle

                if CLIENTS:
                    msg = json.dumps({
                        "angle": angle,
                        "volume": volume
                    })

                    await asyncio.gather(
                        *[c.send(msg) for c in CLIENTS.copy()]
                    )

                print(f"📡 {angle:.1f}° | volume={volume:.4f} → {len(CLIENTS)} client(s)")

        await asyncio.sleep(POLL_INTERVAL)


async def main():
    print("🎙️ DOA WebSocket server starting on ws://0.0.0.0:8765")

    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await poll_mic()


if __name__ == '__main__':
    asyncio.run(main())
