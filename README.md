# SPOOT: Sound Point Of Origin Tracker - [Demo](https://www.youtube.com/watch?v=r0NJpTgAblA)

SPOOT is built around **Gemini**. The Pi captures sound and direction; **Whisper** turns audio into text on-device; **Gemini** interprets transcript, angle, and loudness together and replies with tight, structured HUD copy so people who are deaf in one ear (or juggling noise) instantly know whether to look, and how urgent it is.

Local sensors answer *where*. **Gemini** answers *whether it matters* and *what to say*: name calls, warnings (“watch out”, “heads up”), and background chatter, distilled into single-line cues that mirror on the OLED and the phone AR overlay.

---

## Gemini in the pipeline

Gemini sits at the end of each enrichment pass: Whisper hands off a transcript, the server attaches **bearing (degrees)** and **volume RMS**, and **`gemini-2.x-flash`** completes small JSON objects the UI already understands (`event_type`, `importance`, `directed_at_user`, **`message`**). That keeps the OLED and `/state`/AR canvases wired to **one coherent voice** tuned for accessibility, not raw STT pasted on screen.

- **Structured output:** JSON-shaped responses constrained in the prompt (no prose drift: easy to animate and color-code).
- **Person-aware prompting:** Seed `USER_NAME` and aliases into both Whisper decoding and Gemini so “someone yelled *you* from the left” survives noisy rooms.
- **Rate-aware wiring:** configurable spacing, backoff, and timeout against `429`/quota so the HUD stays resilient on a Raspberry Pi tethered through a phone hotspot.

Gemini complements on-device Whisper: transcription stays **on the Pi**, while Gemini provides the contextual layer **that chooses** what earns a glance.

### Why Gemini in this ecosystem (speed)

Direction-of-arrival cues go stale quickly. You want the HUD to stabilize **soon after** the mic stops capturing, not after a second heavyweight model churns on the Pi. **Gemini Flash**-family models are tuned for that pattern: **one short `generateContent` round-trip**, a handful of tokens in and out, and you get labeled importance plus an 8-ish word **`message`** the OLED and AR can paint immediately.

SPOOT already spends Raspberry Pi cycles on capture, buffering, optional VAD gating, and **faster-whisper**. Running yet another large **local** semantic model would fight for RAM and thermal headroom and stretch tail latency. **Offloading semantic triage to Gemini** keeps the edge device focused on acoustics while the cloud finishes the “does this deserve a glance?” decision at Flash speed.

The wire format reinforces that pace: Gemini sees **compact context** (transcript snippet, numeric angle and volume) instead of streaming raw audio upward, so bandwidth and tokenizer work stay trivial. Returned **JSON-shaped** payloads map straight onto HUD channels, so no extra formatting pass sits between inference and `/state` mirrors.

Together, Flash latency + lean I/O keeps **Gemini** aligned with wearable, glanceable HUDs where **seconds matter**.

---

## Features

- **Semantic HUD (Gemini):** Single-line summaries and importance levels streamed to WebSocket + HTTP mirror for wearable + AR chrome.
- **Spatial audio:** Continuous azimuth from the reSpeaker XVF3800 `xvf_host` tool (runs under `sudo`; path is configured in `server.py`).
- **Live mic buffering:** RMS-based triggering with adaptive pre/post capture windows for phrase-level STT.
- **Speech pipeline:** Optional **Silero VAD** gate, **faster-whisper** transcription, fuzzy/phonetic name matching, transcript deduping, all feeding Gemini with cleaner text.
- **Outputs:**
  - **WebSocket JSON** on port **8765** (`ws://<host>:8765`) → `oled_hud.py` consumes frames with `stage`, `angle`, `message`, `importance`, etc.
  - **HTTP** (`python server.py` serves Flask on port **5000**): `/state` for the AR HUD, `/` + `/manifest.json` for the progressive web app (`index.html`).

---

## Hardware

| Part | Role |
|------|------|
| **reSpeaker USB 4‑Mic Array (XVF3800)** | Beamforming / DOA + capture |
| **Raspberry Pi** (recommended for realtime) | Runs `server.py` + optionally `oled_hud.py` |
| **SSD1306 128×64 OLED** (I²C `0x3C`) | Optional clip-on directional HUD |

Non-Pi installs can run portions of `server.py` for development but DOA relies on vendor `xvf_host` on Pi-class setups.

---

## Software setup

Use Python 3.11+ recommended (Pi OS / Debian).

### 1. Clone and virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

### 2. Dependencies used by `server.py` beyond the base file

The pipeline loads extra packages on demand; install explicitly:

```bash
pip install faster-whisper jellyfish
# Silero VAD (recommended): CPU PyTorch wheels for ARM/x86 vary; see https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you omit torch / faster-whisper, STT/VAD degrade gracefully (logged warnings).

### 3. OLED (optional)

On the Pi with I²C enabled and the display wired:

```bash
pip install adafruit-circuitpython-ssd1306 adafruit-blinka Pillow
```

Run **`oled_hud.py`** alongside **`server.py`** on the same host so it can open `ws://localhost:8765`.

### 4. Environment (Gemini first)

Create **`.env`** in the repo root (see `server.py`’s `_load_dotenv`; existing shell env vars win over the file). **Gemini ships the live HUD captions.** Put your key from [Google AI Studio](https://aistudio.google.com/apikey) in **`GEMINI_API_KEY`**, then tune **`GEMINI_MODEL`** (defaults are flash-class models for latency on Raspberry Pi hardware).

| Variable | Purpose |
|---------|---------|
| `GEMINI_API_KEY` | Google Gemini API key backing `generateContent` for each enrichment pass |
| `GEMINI_MODEL` | e.g. `gemini-2.0-flash`, `gemini-2.5-flash` to balance quality vs Raspberry Pi realtime |
| `GEMINI_TIMEOUT_SEC` | Stall budget per Gemini call |
| `GEMINI_MIN_INTERVAL_SEC`, `GEMINI_BACKOFF_BASE_SEC`, `GEMINI_BACKOFF_CAP_SEC` | Gentle throttling after bursty alerts or `429` responses |
| `USER_NAME`, `USER_NAME_ALIASES` | Passed into Gemini + Whisper biases so YOUR name boosts “directed-at-you” |
| `NAME_FUZZY_THRESHOLD` | Fuzzy cutoff for spotting your name variants before Gemini sees them |
| `AUDIO_DEVICE` | `sounddevice` input: index integer or substring of device name |
| `PREBUFFER_SEC`, `POSTBUFFER_SEC`, `POSTBUFFER_MAX_SEC` | Capture window |
| `WHISPER_MODEL`, `WHISPER_BEAM_SIZE`, `WHISPER_VAD_FILTER` | STT tuning |
| `USE_SILERO_VAD` | `1`/`0` |
| `ENRICH_COOLDOWN_SEC`, `TRANSCRIPT_DEDUP_THRESHOLD` | Event spacing / dedupe |
| `DOA_FLIP_LEFT_RIGHT` | `1` mirrors left/right if the mic is rotated (see below) |

**Security:** Never commit `.env` or TLS private keys.

---

## Running

Requires passwordless **`sudo`** for `/path/to/xvf_host AEC_AZIMUTH_VALUES` (or invoke the server from a sudo-capable automation you trust).

```bash
source .venv/bin/activate
python server.py
```

You should see:

- Whisper + Gemini warmup in the logs, then Gemini lines (`gemini: ATTEMPTING`, skips, backoff on `429`, etc.) as audio events enqueue.
- WebSockets on **`ws://0.0.0.0:8765`**
- Flask on **`http://0.0.0.0:5000`**, or **HTTPS** on **5000** if `192.168.68.55.pem` and `192.168.68.55-key.pem` live next to `server.py` (replace with your own cert filenames in `run_flask_server()` if needed).

Terminal two (OLED):

```bash
python oled_hud.py
```

**Phone AR:** open `https://<pi-ip>:5000` (or HTTP if you have no cert), allow camera access. The overlay polls `/state` for bearings and volume plus the **same short Gemini captions** the OLED renders; captions auto-hide on the handset after a few seconds for readability.

---

## DOA orientation

- **`DOA_FLIP_LEFT_RIGHT`** in `.env`: default in `server.py` is **on** (`1`/`true`/…); use `DOA_FLIP_LEFT_RIGHT=0` to disable. Mirrors hardware azimuth with `logical = (360 - raw) % 360` so left/right match when the mic board is rotated.
- **`oled_hud.py`** has separate **`FLIP_LEFT_RIGHT`** / **`ANGLE_OFFSET`** for display-only tweaks. Prefer fixing azimuth once in `server.py` + env so WebSocket, AR, and Gemini “where” strings stay consistent.

---

## Project layout

| Path | Role |
|------|------|
| `server.py` | Main process: DOA thread, audio, Whisper, Gemini, WebSocket broadcast, Flask `/state` + static PWA |
| `index.html` | SoundAR-style camera overlay + HUD |
| `manifest.json` | PWA manifest |
| `oled_hud.py` | Pi OLED client subscribing to WebSocket HUD frames |
| `requirements.txt` | Core pip deps (extend with faster-whisper, torch per above) |

---

## Customizing vendor tools

DOA **`xvf_host`** path is set by **`XVF_HOST`** near the top of `server.py`; change it if your SEEED / reSpeaker install lives elsewhere.

---

## Creators

- **[Carson Secrest](https://github.com/Carson274)** (`Carson274`): OLED HUD (`oled_hud.py`), Gemini integration in the server pipeline, and contributions to mapping and correcting DOA from the mic array.
- **[Jak Tynshimov](https://github.com/Jakhangir18)** (`Jakhangir18`): phone AR layer (`index.html` / `/state`), 3D-printed glasses hardware, and Raspberry Pi bring-up.

**SPOOT** means **S**ound **P**oint **O**f **O**rigin **T**racker.
