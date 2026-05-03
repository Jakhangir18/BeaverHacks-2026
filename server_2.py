import asyncio
import collections
import difflib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd
import websockets


# =========================
# .env auto-loader
# =========================

def _load_dotenv(path=".env"):
    """
    Tiny .env loader. Populates os.environ from a KEY=VALUE file before any
    config reads happen. Avoids needing python-dotenv as a hard dependency.
    Existing environment variables take precedence over .env values.
    """

    if not os.path.exists(path):
        return 0

    loaded = 0

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()

                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]

                if key and key not in os.environ:
                    os.environ[key] = val
                    loaded += 1
    except Exception as exc:
        print(f"warning: couldn't load {path}: {exc}")

    return loaded


_DOTENV_LOADED = _load_dotenv()


# =========================
# Hardware / device config
# =========================

XVF_HOST = '/home/rasberypi/Documents/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/rpi_64bit/xvf_host'

CLIENTS = set()


# =========================
# Detection tuning
# =========================

# Increase this if it still reacts to background noise.
VOLUME_THRESHOLD = 0.025

# Smallest direction change (degrees) we bother re-broadcasting.
MIN_ANGLE_CHANGE = 8.0

# How often we sample loudness when idle.
POLL_INTERVAL = 0.3

SAMPLE_RATE = 16000

# Block size of the continuous mic stream (~64 ms at 16 kHz).
BLOCKSIZE = 1024

# How much PRE-trigger audio to keep in the rolling buffer. This is what
# gives us the start of a phrase ("hey carson...") even though we only
# noticed the loudness spike partway through.
PREBUFFER_SEC = float(os.environ.get("PREBUFFER_SEC", "1.0"))

# How much MORE audio to wait for after a trigger before transcribing.
# Together with PREBUFFER_SEC this is the size of the chunk fed to STT.
POSTBUFFER_SEC = float(os.environ.get("POSTBUFFER_SEC", "1.4"))

# How many samples of recent audio to RMS-average for the loudness probe.
VOLUME_WINDOW_SEC = 0.2

# Don't fire enrichment more than once per this many seconds.
ENRICH_COOLDOWN_SEC = float(os.environ.get("ENRICH_COOLDOWN_SEC", "4.0"))


# =========================
# User / Gemini config
# =========================

# Default user name used when matching transcripts. Override with env var.
USER_NAME = os.environ.get("USER_NAME", "Carson")

# Comma-separated list of known STT misreads of USER_NAME. STT is heavily
# biased toward real English words; for made-up or rare names ("spoot")
# you'll see things like "spooked", "Foods", "spook" come back instead.
# Add those here so they still count as the user's name being called.
#   USER_NAME_ALIASES=spooked,spook,foods,sport
USER_NAME_ALIASES = tuple(
    a.strip().lower()
    for a in os.environ.get("USER_NAME_ALIASES", "").split(",")
    if a.strip()
)

# Word-level fuzzy match threshold for name detection (0.0–1.0). Lower = more
# tolerant of STT errors but more false positives. 0.72 catches things like
# "spooked"↔"spoot" without accepting random English words.
NAME_FUZZY_THRESHOLD = float(os.environ.get("NAME_FUZZY_THRESHOLD", "0.72"))

# Gemini API key MUST come from the environment. Never hardcode.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TIMEOUT_SEC = float(os.environ.get("GEMINI_TIMEOUT_SEC", "3.0"))

# Minimum spacing between Gemini calls (seconds). The free tier is ~15 RPM,
# so 4s is a safe default. Increase if you keep seeing 429s.
GEMINI_MIN_INTERVAL_SEC = float(os.environ.get("GEMINI_MIN_INTERVAL_SEC", "4.0"))

# Default pause when Gemini returns 429 with no Retry-After header.
GEMINI_BACKOFF_BASE_SEC = float(os.environ.get("GEMINI_BACKOFF_BASE_SEC", "30.0"))
GEMINI_BACKOFF_CAP_SEC = float(os.environ.get("GEMINI_BACKOFF_CAP_SEC", "300.0"))

# Phrases that should always be flagged as urgent in the local fallback.
URGENT_PHRASES = ("watch out", "look out", "careful", "heads up", "move")


# Circuit-breaker state for Gemini rate limiting. Module-level so it
# persists across enrichment tasks.
_gemini_pause_until = 0.0
_gemini_last_call = 0.0
_gemini_consecutive_429 = 0


# =========================
# DOA + audio capture
# =========================

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


def rms(audio_array):
    if len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_array ** 2)))


# ----- Rolling audio buffer fed by a continuous InputStream -----
#
# Why a rolling buffer? When the user shouts "hey carson watch out", the
# loudness threshold trips midway through the phrase. If we only captured
# audio AFTER the trigger we'd miss "hey" (and Google STT often returns
# nothing for a 1-word fragment). By keeping the last PREBUFFER_SEC of
# audio in memory at all times we can reach BACK in time once we trigger.

_BUFFER_TOTAL_SEC = PREBUFFER_SEC + POSTBUFFER_SEC + 0.5  # small slack
_MAX_BLOCKS = int(_BUFFER_TOTAL_SEC * SAMPLE_RATE / BLOCKSIZE) + 4

_audio_blocks = collections.deque(maxlen=_MAX_BLOCKS)
_audio_lock = threading.Lock()
_audio_stream = None


def _audio_callback(indata, frames, time_info, status):
    # Runs on sounddevice's audio thread. Keep this fast.
    block = indata[:, 0].copy()
    with _audio_lock:
        _audio_blocks.append(block)


def start_audio_stream():
    """Open the single continuous mic stream that feeds the rolling buffer."""

    global _audio_stream

    if _audio_stream is not None:
        return

    _audio_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=BLOCKSIZE,
        callback=_audio_callback,
    )
    _audio_stream.start()


def stop_audio_stream():
    global _audio_stream

    if _audio_stream is None:
        return

    try:
        _audio_stream.stop()
        _audio_stream.close()
    finally:
        _audio_stream = None


def _snapshot_buffer(seconds):
    """Return the last `seconds` of audio currently in the rolling buffer."""

    with _audio_lock:
        blocks = list(_audio_blocks)

    if not blocks:
        return np.zeros(0, dtype='float32')

    audio = np.concatenate(blocks)
    needed = int(seconds * SAMPLE_RATE)

    if len(audio) > needed:
        audio = audio[-needed:]

    return audio


def get_volume_level():
    """
    Recent RMS loudness from the rolling buffer. Non-blocking — the mic
    stream is always running, so this just inspects the last few blocks.
    """

    return rms(_snapshot_buffer(VOLUME_WINDOW_SEC))


async def capture_phrase():
    """
    Wait for the post-trigger window to roll into the buffer, then return
    PREBUFFER_SEC of pre-trigger context + POSTBUFFER_SEC of post-trigger
    audio.
    """

    await asyncio.sleep(POSTBUFFER_SEC)
    return _snapshot_buffer(PREBUFFER_SEC + POSTBUFFER_SEC)


def angle_difference(a, b):
    """
    Smallest difference between two angles.
    Example: 359° and 1° are 2° apart, not 358°.
    """

    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def direction_word(angle):
    """Coarse human-readable direction used in fallback messages."""

    if angle is None:
        return "?"

    a = angle % 360

    if a >= 337.5 or a < 22.5:
        return "front"
    if a < 67.5:
        return "front-right"
    if a < 112.5:
        return "right"
    if a < 157.5:
        return "back-right"
    if a < 202.5:
        return "behind"
    if a < 247.5:
        return "back-left"
    if a < 292.5:
        return "left"
    return "front-left"


# =========================
# Transcription (fast, local-ish)
# =========================

def transcribe_audio(audio_array):
    """
    Best-effort transcription of a short mono float32 chunk.

    Uses the SpeechRecognition library's free Google Web Speech endpoint
    with show_all=True so we get Google's full ranked list of hypotheses.
    For made-up names ("spoot") the top pick is often wrong but a lower
    alternative ("spook") is closer — having all of them in the transcript
    string lets fuzzy name matching still succeed.

    Returns a string (possibly empty) and never raises.
    """

    if len(audio_array) == 0:
        return ""

    try:
        import speech_recognition as sr
    except ImportError:
        print("speech_recognition not installed; skipping transcription")
        return ""

    audio_int16 = np.clip(audio_array * 32767.0, -32768, 32767).astype(np.int16)
    audio_data = sr.AudioData(audio_int16.tobytes(), SAMPLE_RATE, 2)

    recognizer = sr.Recognizer()

    try:
        result = recognizer.recognize_google(audio_data, show_all=True)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        print(f"STT request error: {exc}")
        return ""
    except Exception as exc:
        print(f"STT unexpected error: {exc}")
        return ""

    if not result or not isinstance(result, dict):
        return ""

    alternatives = result.get("alternative", [])
    texts = [
        alt.get("transcript", "").strip()
        for alt in alternatives
        if alt.get("transcript")
    ]

    if not texts:
        return ""

    top = texts[0]

    # Append distinct alternatives so fuzzy/keyword matching has more
    # candidate words. Display-wise this looks like:
    #     Foods (spooked / spook)
    # which is also a useful debugging hint for the user.
    seen = {top.lower()}
    others = []
    for t in texts[1:]:
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        others.append(t)
        if len(others) >= 3:
            break

    if others:
        return f"{top} ({' / '.join(others)})"
    return top


# =========================
# Local rule-based reasoning
# =========================

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _name_targets():
    """All canonical strings we treat as 'the user's name' for matching."""

    targets = [USER_NAME.lower()]
    targets.extend(a for a in USER_NAME_ALIASES if a)
    return targets


def detect_name(transcript):
    """
    Return True if the transcript appears to contain the user's name.

    Three strategies, in order of confidence:
      1. Direct substring match against USER_NAME or any USER_NAME_ALIASES.
      2. Word-level fuzzy match (difflib ratio) against the same targets.
         Catches close STT misreads like "spooked" vs "spoot" without
         requiring you to anticipate every variation.
    """

    if not transcript:
        return False

    text = transcript.lower()
    targets = _name_targets()

    for target in targets:
        if target and target in text:
            return True

    if NAME_FUZZY_THRESHOLD <= 0:
        return False

    words = _WORD_RE.findall(text)
    for word in words:
        for target in targets:
            if not target:
                continue
            ratio = difflib.SequenceMatcher(None, word, target).ratio()
            if ratio >= NAME_FUZZY_THRESHOLD:
                return True

    return False


def local_reason(transcript, angle, volume, name_detected):
    """
    Rule-based importance estimate. Used:
      1. As a fast pre-Gemini HUD update.
      2. As a fallback when Gemini fails / times out / returns bad JSON.
    """

    text = (transcript or "").lower()
    where = direction_word(angle)

    if name_detected:
        return {
            "event_type": "speech",
            "directed_at_user": True,
            "importance": "high",
            "message": f"{USER_NAME} called from {where}",
        }

    if any(phrase in text for phrase in URGENT_PHRASES):
        return {
            "event_type": "warning",
            "directed_at_user": True,
            "importance": "high",
            "message": f"Warning from {where}",
        }

    if text:
        return {
            "event_type": "speech",
            "directed_at_user": False,
            "importance": "low",
            "message": f"Voice from {where}",
        }

    return {
        "event_type": "unknown",
        "directed_at_user": False,
        "importance": "low",
        "message": f"Sound from {where}",
    }


# =========================
# Gemini reasoning layer
# =========================

# Build the prompt the way Gemini will see it. f-string-friendly.
GEMINI_PROMPT_TEMPLATE = """You are helping a user who is deaf in one ear understand whether a sound is worth looking at.

User name: {user_name}
Transcript: {transcript}
Sound angle in degrees: {angle}
Volume RMS: {volume}

Decide if this sound is likely directed at the user or important enough to alert them.

Return ONLY valid JSON with this exact shape:
{{
  "directed_at_user": true,
  "importance": "low|medium|high",
  "event_type": "speech|warning|background|unknown",
  "message": "short HUD-friendly message"
}}

Rules:
- If the transcript contains the user's name, directed_at_user should be true and importance should be high.
- If the transcript includes urgent phrases like 'watch out', 'look out', 'careful', or 'move', importance should be high.
- If it seems like background conversation, importance should be low.
- Keep message under 8 words.
- Do not include markdown, explanation, or extra keys.
"""


def _coerce_gemini_result(result):
    """Validate Gemini's JSON. Returns None if structurally bad."""

    if not isinstance(result, dict):
        return None

    importance = result.get("importance", "low")
    if importance not in ("low", "medium", "high"):
        importance = "low"

    event_type = result.get("event_type", "unknown")
    if event_type not in ("speech", "warning", "background", "unknown"):
        event_type = "unknown"

    return {
        "event_type": event_type,
        "directed_at_user": bool(result.get("directed_at_user", False)),
        "importance": importance,
        "message": str(result.get("message", "Sound detected"))[:80],
    }


def _parse_retry_after(headers):
    """Pull a numeric seconds value out of an HTTP Retry-After header."""

    if not headers:
        return None

    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None

    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _event_tag(angle, volume):
    """Short prefix tag so per-event logs can be grouped/grepped together."""

    angle_str = f"{angle:.1f}°" if angle is not None else "?°"
    return f"[a={angle_str}/v={volume:.4f}]"


def _fmt_classification(reasoning):
    """Single-line summary of a reasoning dict for log output."""

    return (
        f"importance={reasoning.get('importance', '?'):<6} "
        f"event={reasoning.get('event_type', '?'):<10} "
        f"directed={reasoning.get('directed_at_user', False)!s:<5} "
        f"msg=\"{reasoning.get('message', '')}\""
    )


def _gemini_skip_reason(transcript):
    """
    Decide whether to even attempt a Gemini call this event.

    Returns a human-readable reason string when we should skip,
    or "" when we should proceed.

    Skip cases:
      - No API key.
      - Empty transcript (nothing for Gemini to reason about).
      - Transcript has no keywords the user likely cares about
        (random background chatter → local fallback is enough).
      - Circuit breaker open from a recent 429.
      - Called Gemini too recently (RPM throttle).
    """

    if not GEMINI_API_KEY:
        return "no API key"

    if not transcript:
        return "no transcript"

    text = transcript.lower()

    # Name match uses fuzzy + alias logic so we still call Gemini when
    # STT returned a near-miss like "spooked" for "spoot".
    if detect_name(transcript):
        pass
    else:
        social_phrases = (
            "watch out",
            "look out",
            "careful",
            "heads up",
            "move",
            "excuse me",
            "hey",
        )
        if not any(phrase in text for phrase in social_phrases):
            return "no important keywords"

    now = time.monotonic()

    if now < _gemini_pause_until:
        remaining = _gemini_pause_until - now
        return f"paused {remaining:.0f}s after 429"

    since_last = now - _gemini_last_call
    if since_last < GEMINI_MIN_INTERVAL_SEC:
        wait = GEMINI_MIN_INTERVAL_SEC - since_last
        return f"throttled, wait {wait:.1f}s"

    return ""


def _gemini_should_skip(transcript):
    """Backward-compatible boolean form of _gemini_skip_reason."""

    return bool(_gemini_skip_reason(transcript))


def call_gemini_blocking(transcript, angle, volume):
    """
    Synchronous Gemini REST call.
    Returns dict (matching the local-reason shape) or None on any failure.
    Implements a 429 circuit breaker so we don't hammer the API while
    rate-limited.
    """

    global _gemini_pause_until, _gemini_last_call, _gemini_consecutive_429

    tag = _event_tag(angle, volume)

    skip = _gemini_skip_reason(transcript)
    if skip:
        print(f"{tag} gemini: SKIPPED ({skip})")
        return None

    print(f"{tag} gemini: ATTEMPTING transcript=\"{transcript}\"")

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        user_name=USER_NAME,
        transcript=transcript or "",
        angle=f"{angle:.1f}" if angle is not None else "?",
        volume=f"{volume:.4f}",
    )

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    _gemini_last_call = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=GEMINI_TIMEOUT_SEC) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            # Rate limited. Open the circuit breaker. Honor Retry-After when
            # present, otherwise fall back to exponential backoff.
            retry_after = _parse_retry_after(exc.headers)
            _gemini_consecutive_429 += 1
            backoff = GEMINI_BACKOFF_BASE_SEC * (2 ** (_gemini_consecutive_429 - 1))
            pause = max(retry_after or 0.0, backoff)
            pause = min(pause, GEMINI_BACKOFF_CAP_SEC)
            _gemini_pause_until = time.monotonic() + pause
            print(f"{tag} gemini: 429 rate-limited, pausing {pause:.0f}s")
        else:
            print(f"{tag} gemini: HTTP {exc.code} {exc.reason}")
        return None
    except urllib.error.URLError as exc:
        print(f"{tag} gemini: HTTP error {exc}")
        return None
    except Exception as exc:
        print(f"{tag} gemini: error {exc}")
        return None

    # Successful round-trip → reset the 429 streak.
    _gemini_consecutive_429 = 0

    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(f"{tag} gemini: bad JSON ({exc})")
        return None

    result = _coerce_gemini_result(parsed)
    if result is None:
        print(f"{tag} gemini: bad payload, ignoring")
    else:
        print(f"{tag} gemini: returned {_fmt_classification(result)}")

    return result


async def call_gemini(transcript, angle, volume):
    """
    Async wrapper around the blocking call. Adds a hard timeout
    so a hung HTTP request can never stall the pipeline.
    """

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(call_gemini_blocking, transcript, angle, volume),
            timeout=GEMINI_TIMEOUT_SEC + 0.5,
        )
    except asyncio.TimeoutError:
        print(f"{_event_tag(angle, volume)} gemini: TIMEOUT")
        return None


# =========================
# Payload + broadcasting
# =========================

def make_payload(angle, volume, *, stage, transcript="",
                 name_detected=False, reasoning=None):
    """
    Always produce a payload that matches the documented JSON shape so
    the HUD never has to guard against missing fields.
    """

    payload = {
        "stage": stage,
        "ts": time.time(),
        "angle": angle,
        "volume": volume,
        "transcript": transcript,
        "name_detected": name_detected,
        "event_type": "unknown",
        "directed_at_user": False,
        "importance": "low",
        "message": f"Sound from {direction_word(angle)}",
    }

    if reasoning:
        payload.update(reasoning)

    return payload


async def broadcast(payload):
    if not CLIENTS:
        return

    msg = json.dumps(payload)

    await asyncio.gather(
        *[c.send(msg) for c in CLIENTS.copy()],
        return_exceptions=True,
    )


# =========================
# Pipeline: enrichment task
# =========================

async def enrich_event(angle, volume):
    """
    Background task spawned per trigger. Runs OFF the polling loop so the
    main loop can keep watching for the next loud sound.

    Sequence:
      1. Wait for the post-trigger audio to fill into the rolling buffer,
         then snapshot pre+post audio and transcribe it.
      2. Local name + keyword detection -> push HUD update immediately.
      3. Call Gemini for higher-quality reasoning -> push final HUD update.
      4. On Gemini failure, the local reasoning becomes the final.
    """

    tag = _event_tag(angle, volume)

    audio_array = await capture_phrase()
    transcript = await asyncio.to_thread(transcribe_audio, audio_array)
    name_detected = detect_name(transcript)

    if transcript:
        print(f"{tag} transcript: \"{transcript}\" (name_detected={name_detected})")
    else:
        print(f"{tag} transcript: <empty>")

    local = local_reason(transcript, angle, volume, name_detected)
    print(f"{tag} local:  {_fmt_classification(local)}")

    await broadcast(make_payload(
        angle, volume,
        stage="transcribed",
        transcript=transcript,
        name_detected=name_detected,
        reasoning=local,
    ))

    gemini = await call_gemini(transcript, angle, volume)

    if gemini is None:
        # Local result becomes the final. Re-broadcast with stage=final
        # so the HUD knows no more updates are coming.
        print(f"{tag} FINAL (local):  {_fmt_classification(local)}")
        await broadcast(make_payload(
            angle, volume,
            stage="final",
            transcript=transcript,
            name_detected=name_detected,
            reasoning=local,
        ))
        return

    # Safety net: even if Gemini missed it, name detection always wins.
    if name_detected:
        gemini["directed_at_user"] = True
        if gemini.get("importance") != "high":
            gemini["importance"] = "high"

    print(f"{tag} FINAL (gemini): {_fmt_classification(gemini)}")
    await broadcast(make_payload(
        angle, volume,
        stage="final",
        transcript=transcript,
        name_detected=name_detected,
        reasoning=gemini,
    ))


# =========================
# Pipeline: poll loop
# =========================

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
    last_enrich = 0.0

    while True:
        # Volume is now a cheap read off the rolling buffer.
        volume = get_volume_level()

        if volume < VOLUME_THRESHOLD:
            await asyncio.sleep(POLL_INTERVAL)
            continue

        angle = await asyncio.to_thread(get_doa_angle)

        if angle is None:
            await asyncio.sleep(POLL_INTERVAL)
            continue

        should_send = (
            last_angle is None or
            angle_difference(angle, last_angle) >= MIN_ANGLE_CHANGE
        )

        if not should_send:
            await asyncio.sleep(POLL_INTERVAL)
            continue

        last_angle = angle

        # Step 1: IMMEDIATE directional alert.
        # No transcription, no Gemini. The HUD lights up before we
        # have any idea what was said or how important it is.
        await broadcast(make_payload(angle, volume, stage="directional"))
        print(f"{_event_tag(angle, volume)} 📡 directional alert → {len(CLIENTS)} client(s)")

        # Step 2: Spawn enrichment as a background task. It self-captures
        # from the rolling buffer (pre-trigger + post-trigger audio) so
        # poll_mic doesn't have to wait for any recording to finish.
        now = time.monotonic()

        if now - last_enrich >= ENRICH_COOLDOWN_SEC:
            last_enrich = now
            asyncio.create_task(enrich_event(angle, volume))

        await asyncio.sleep(POLL_INTERVAL)


async def main():
    print("🎙️  DOA WebSocket server starting on ws://0.0.0.0:8765")

    if _DOTENV_LOADED:
        print(f"    .env           = loaded {_DOTENV_LOADED} var(s)")
    else:
        print("    .env           = not loaded (file missing or empty)")

    print(f"    USER_NAME      = {USER_NAME}")

    if USER_NAME_ALIASES:
        print(f"    ALIASES        = {', '.join(USER_NAME_ALIASES)}")

    print(f"    FUZZY THRESH   = {NAME_FUZZY_THRESHOLD}")
    print(f"    GEMINI_MODEL   = {GEMINI_MODEL}")
    print(f"    GEMINI_KEY     = {'set' if GEMINI_API_KEY else 'NOT set (local fallback only)'}")
    print(f"    AUDIO WINDOW   = {PREBUFFER_SEC:.2f}s pre + {POSTBUFFER_SEC:.2f}s post")

    start_audio_stream()

    try:
        async with websockets.serve(handle_client, "0.0.0.0", 8765):
            await poll_mic()
    finally:
        stop_audio_stream()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stop_audio_stream()
        print("\nServer stopped.")
