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

# Optional audio device override. Accepts:
#   - integer index (e.g. AUDIO_DEVICE=1)
#   - device name substring (e.g. AUDIO_DEVICE="XVF3800")
# Leave blank to auto-pick the first input-capable device.
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "").strip()

# How much PRE-trigger audio to keep in the rolling buffer.
PREBUFFER_SEC = float(os.environ.get("PREBUFFER_SEC", "1.0"))

# Minimum post-trigger capture window (seconds).
POSTBUFFER_SEC = float(os.environ.get("POSTBUFFER_SEC", "1.0"))

# Hard cap on the post-trigger capture window.
POSTBUFFER_MAX_SEC = float(os.environ.get("POSTBUFFER_MAX_SEC", "2.5"))

# How often we re-check loudness while extending the post-trigger window.
POSTBUFFER_TICK_SEC = 0.15

# How many samples of recent audio to RMS-average for the loudness probe.
VOLUME_WINDOW_SEC = 0.2

# Don't fire enrichment more than once per this many seconds.
ENRICH_COOLDOWN_SEC = float(os.environ.get("ENRICH_COOLDOWN_SEC", "1.5"))

# Similarity threshold (0–1) above which a new transcript is considered a
# duplicate of the last one and suppressed. 0.85 = "nearly identical".
TRANSCRIPT_DEDUP_THRESHOLD = float(os.environ.get("TRANSCRIPT_DEDUP_THRESHOLD", "0.85"))


# =========================
# Whisper STT config
# =========================

# Which faster-whisper model to use. "tiny.en" is fastest on Pi 4;
# "base.en" is meaningfully more accurate if latency allows.
#   WHISPER_MODEL=tiny.en
#   WHISPER_MODEL=base.en
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny.en").strip()

# Beam size for Whisper decoding. Higher = more accurate, slower.
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

# Set to "1" to enable Whisper's built-in VAD filter (silences silent
# padding before decoding). Recommended on; saves latency on short captures.
WHISPER_VAD_FILTER = os.environ.get("WHISPER_VAD_FILTER", "1").strip() == "1"

# Lazy-initialised singleton so we pay the model-load cost only once.
_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    """Return the faster-whisper model, loading it on first call."""

    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            )

        print(f"    Loading Whisper model '{WHISPER_MODEL}' (first call) …")
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )
        print(f"    Whisper model ready.")

    return _whisper_model


# =========================
# Silero VAD config
# =========================

# Set to "0" to disable the Silero VAD gate (every trigger goes to STT).
USE_SILERO_VAD = os.environ.get("USE_SILERO_VAD", "1").strip() == "1"

_silero_model = None
_silero_utils = None
_silero_lock = threading.Lock()


def _get_silero():
    """Return (model, utils) for Silero VAD, loading on first call."""

    global _silero_model, _silero_utils

    if _silero_model is not None:
        return _silero_model, _silero_utils

    with _silero_lock:
        if _silero_model is not None:
            return _silero_model, _silero_utils

        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is required for Silero VAD. "
                "Run: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )

        print("    Loading Silero VAD model (first call) …")
        model, utils = torch.hub.load(
            'snakers4/silero-vad',
            'silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        _silero_model = model
        _silero_utils = utils
        print("    Silero VAD ready.")

    return _silero_model, _silero_utils


def is_speech(audio_array):
    """
    Return True if Silero VAD detects at least one speech segment in the
    given mono float32 array (16 kHz). Falls back to True on any error so
    we never silently drop audio when the model misbehaves.
    """

    if not USE_SILERO_VAD:
        return True

    try:
        import torch
        model, utils = _get_silero()
        get_speech_timestamps = utils[0]

        tensor = torch.from_numpy(audio_array.copy())
        timestamps = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.4,          # lower = more sensitive
            min_speech_duration_ms=150,
        )
        return len(timestamps) > 0

    except Exception as exc:
        print(f"Silero VAD error (passing through): {exc}")
        return True


# =========================
# User / Gemini config
# =========================

USER_NAME = os.environ.get("USER_NAME", "Spoot")

USER_NAME_ALIASES = tuple(
    a.strip().lower()
    for a in os.environ.get("USER_NAME_ALIASES", "").split(",")
    if a.strip()
)

NAME_FUZZY_THRESHOLD = float(os.environ.get("NAME_FUZZY_THRESHOLD", "0.72"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TIMEOUT_SEC = float(os.environ.get("GEMINI_TIMEOUT_SEC", "3.0"))
GEMINI_MIN_INTERVAL_SEC = float(os.environ.get("GEMINI_MIN_INTERVAL_SEC", "4.0"))
GEMINI_BACKOFF_BASE_SEC = float(os.environ.get("GEMINI_BACKOFF_BASE_SEC", "30.0"))
GEMINI_BACKOFF_CAP_SEC = float(os.environ.get("GEMINI_BACKOFF_CAP_SEC", "300.0"))

URGENT_PHRASES = ("watch out", "look out", "careful", "heads up", "move")

# Circuit-breaker state for Gemini rate limiting.
_gemini_pause_until = 0.0
_gemini_last_call = 0.0
_gemini_consecutive_429 = 0

# Last transcript seen, used for deduplication across enrichment calls.
_last_transcript = ""


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


_BUFFER_TOTAL_SEC = PREBUFFER_SEC + POSTBUFFER_MAX_SEC + 0.5
_MAX_BLOCKS = int(_BUFFER_TOTAL_SEC * SAMPLE_RATE / BLOCKSIZE) + 4

_audio_blocks = collections.deque(maxlen=_MAX_BLOCKS)
_audio_lock = threading.Lock()
_audio_stream = None


def _audio_callback(indata, frames, time_info, status):
    block = indata[:, 0].copy()
    with _audio_lock:
        _audio_blocks.append(block)


def _list_input_devices():
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"warning: couldn't query audio devices: {exc}")
        return []

    out = []
    for idx, info in enumerate(devices):
        if info.get("max_input_channels", 0) > 0:
            out.append((idx, info))
    return out


def _resolve_audio_device():
    inputs = _list_input_devices()

    if AUDIO_DEVICE:
        try:
            idx = int(AUDIO_DEVICE)
            return idx
        except ValueError:
            pass

        needle = AUDIO_DEVICE.lower()
        for idx, info in inputs:
            if needle in info.get("name", "").lower():
                return idx

        print(f"warning: AUDIO_DEVICE={AUDIO_DEVICE!r} did not match any input device")

    try:
        default = sd.default.device
        default_in = default[0] if isinstance(default, (list, tuple)) else default
        if isinstance(default_in, int) and default_in >= 0:
            return default_in
    except Exception:
        pass

    if inputs:
        return inputs[0][0]

    return None


def _print_device_list():
    inputs = _list_input_devices()
    if not inputs:
        print("  (no input-capable devices found)")
        return
    for idx, info in inputs:
        name = info.get("name", "?")
        ch = info.get("max_input_channels", 0)
        rate = int(info.get("default_samplerate", 0))
        print(f"  [{idx}] {name}  ({ch} ch, default {rate} Hz)")


def start_audio_stream():
    global _audio_stream

    if _audio_stream is not None:
        return

    device = _resolve_audio_device()

    try:
        _audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=BLOCKSIZE,
            device=device,
            callback=_audio_callback,
        )
        _audio_stream.start()
    except Exception as exc:
        print()
        print(f"ERROR: couldn't open audio stream on device {device!r}: {exc}")
        print("Available input devices:")
        _print_device_list()
        print()
        print("Fix: set AUDIO_DEVICE=<index or name substring> in your .env")
        print("     e.g.  AUDIO_DEVICE=1")
        print("     or    AUDIO_DEVICE=XVF3800")
        raise

    try:
        info = sd.query_devices(device) if device is not None else sd.query_devices(kind="input")
        name = info.get("name", str(device))
    except Exception:
        name = str(device)

    print(f"    AUDIO_DEVICE   = [{device}] {name}")


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
    return rms(_snapshot_buffer(VOLUME_WINDOW_SEC))


async def capture_phrase():
    """
    Wait for the post-trigger window to roll into the buffer, then return
    pre-trigger context + post-trigger audio.
    """

    await asyncio.sleep(POSTBUFFER_SEC)
    elapsed = POSTBUFFER_SEC

    while elapsed < POSTBUFFER_MAX_SEC:
        if get_volume_level() < VOLUME_THRESHOLD:
            break

        wait = min(POSTBUFFER_TICK_SEC, POSTBUFFER_MAX_SEC - elapsed)
        await asyncio.sleep(wait)
        elapsed += wait

    audio = _snapshot_buffer(PREBUFFER_SEC + elapsed)
    return audio, elapsed


def angle_difference(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def direction_word(angle):
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
# Transcription — Whisper
# =========================

def _whisper_initial_prompt():
    """
    Seed the Whisper decoder with vocabulary it should prefer. Including
    USER_NAME and aliases here biases the token probabilities toward our
    unusual name even when the acoustic signal is ambiguous.
    """
    parts = [USER_NAME]
    parts.extend(USER_NAME_ALIASES)
    parts += [
        f"Hey {USER_NAME}",
        "watch out", "look out", "careful", "heads up",
        "excuse me", "move",
    ]
    return ", ".join(dict.fromkeys(parts))  # deduplicated, order-preserving


def transcribe_audio(audio_array):
    """
    Transcribe a short mono float32 chunk using faster-whisper.

    Pipeline:
      1. Silero VAD gate — if no speech detected, return early.
      2. Whisper decode with USER_NAME-biased initial_prompt.
      3. Return (display_string, [alternatives]) — single-element list
         because Whisper doesn't produce ranked alternatives the way
         Google Web Speech does. The list wrapper keeps the downstream
         API identical.

    Never raises. Returns ("", []) on empty input or any failure.
    """

    if len(audio_array) == 0:
        return "", []

    # VAD gate: skip STT entirely if no speech detected in the chunk.
    if not is_speech(audio_array):
        return "", []

    try:
        model = _get_whisper()
    except RuntimeError as exc:
        print(f"Whisper unavailable: {exc}")
        return "", []

    try:
        segments, _info = model.transcribe(
            audio_array,
            language="en",
            initial_prompt=_whisper_initial_prompt(),
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=WHISPER_VAD_FILTER,
            # word_timestamps adds latency — skip unless you need word timing
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    except Exception as exc:
        print(f"Whisper transcription error: {exc}")
        return "", []

    if not text:
        return "", []

    return text, [text]


# =========================
# Transcript deduplication
# =========================

def is_new_transcript(text):
    """
    Return True if `text` is meaningfully different from the last seen
    transcript. Suppresses near-duplicate enrichment calls that arise
    when two loud events overlap and produce almost identical STT output.

    Updates the global _last_transcript on every non-empty, non-duplicate.
    """

    global _last_transcript

    if not text:
        return False

    ratio = difflib.SequenceMatcher(
        None, text.lower(), _last_transcript.lower()
    ).ratio()

    if ratio > TRANSCRIPT_DEDUP_THRESHOLD:
        return False

    _last_transcript = text
    return True


# =========================
# Local rule-based reasoning
# =========================

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _name_targets():
    targets = [USER_NAME.lower()]
    targets.extend(a for a in USER_NAME_ALIASES if a)
    return [t for t in targets if t]


# ---------------------------------------------------------------------------
# Phonetic matching (jellyfish)
# ---------------------------------------------------------------------------

def _phonetic_name_match(word):
    """
    Return True if `word` sounds like USER_NAME or any alias according to
    Metaphone or Soundex. Catches STT errors that are acoustically plausible
    but differ in spelling — e.g. "spook" vs "spoot".

    Falls back to False (no match) if jellyfish is not installed, so the
    rest of the pipeline still works without it.
    """

    try:
        import jellyfish
    except ImportError:
        return False

    word_l = word.lower()
    targets = _name_targets()

    for target in targets:
        try:
            if jellyfish.metaphone(word_l) == jellyfish.metaphone(target):
                return True
        except Exception:
            pass
        try:
            if jellyfish.soundex(word_l) == jellyfish.soundex(target):
                return True
        except Exception:
            pass

    return False


def _match_one_alternative(alt, targets):
    """
    Score a single STT alternative against every name target.

    Match priority:
      1. Substring (exact containment) — ratio 1.0, kind "substring"
      2. Phonetic (Metaphone / Soundex) — ratio 0.95, kind "phonetic"
      3. Fuzzy (SequenceMatcher)        — ratio varies, kind "fuzzy"

    Returns a match dict or None.
    """

    if not alt:
        return None

    text = alt.lower()
    words = _WORD_RE.findall(text)

    # 1. Substring
    for target in targets:
        if target in text:
            matched_word = next(
                (w for w in words if target in w),
                target,
            )
            return {
                "alternative": alt,
                "target": target,
                "word": matched_word,
                "ratio": 1.0,
                "kind": "substring",
            }

    # 2. Phonetic
    for word in words:
        if _phonetic_name_match(word):
            # Find which target it sounded like.
            target_hit = next(
                (t for t in targets if _phonetic_name_match_pair(word, t)),
                targets[0],
            )
            return {
                "alternative": alt,
                "target": target_hit,
                "word": word,
                "ratio": 0.95,
                "kind": "phonetic",
            }

    # 3. Fuzzy
    if NAME_FUZZY_THRESHOLD <= 0:
        return None

    best = None
    for word in words:
        for target in targets:
            ratio = difflib.SequenceMatcher(None, word, target).ratio()
            if ratio < NAME_FUZZY_THRESHOLD:
                continue
            if best is None or ratio > best["ratio"]:
                best = {
                    "alternative": alt,
                    "target": target,
                    "word": word,
                    "ratio": ratio,
                    "kind": "fuzzy",
                }

    return best


def _phonetic_name_match_pair(word, target):
    """Return True if `word` sounds like `target` via Metaphone or Soundex."""

    try:
        import jellyfish
        word_l = word.lower()
        target_l = target.lower()
        return (
            jellyfish.metaphone(word_l) == jellyfish.metaphone(target_l) or
            jellyfish.soundex(word_l) == jellyfish.soundex(target_l)
        )
    except Exception:
        return False


def name_match_details(text_or_alts):
    """
    Score the user's name across one or many STT hypotheses.

    Returns a match dict or None.
    """

    if not text_or_alts:
        return None

    alternatives = [text_or_alts] if isinstance(text_or_alts, str) else list(text_or_alts)
    targets = _name_targets()

    if not targets:
        return None

    best = None
    for alt in alternatives:
        match = _match_one_alternative(alt, targets)
        if match is None:
            continue
        if match["kind"] in ("substring", "phonetic"):
            return match
        if best is None or match["ratio"] > best["ratio"]:
            best = match

    return best


def detect_name(text_or_alts):
    return name_match_details(text_or_alts) is not None


def local_reason(transcript, angle, volume, name_detected):
    text = (transcript or "").lower()
    where = direction_word(angle)

    if name_detected:
        return {
            "event_type": "speech",
            "directed_at_user": True,
            "importance": "high",
            "message": f"Your name was called from {where}",
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
    angle_str = f"{angle:.1f}°" if angle is not None else "?°"
    return f"[a={angle_str}/v={volume:.4f}]"


def _fmt_classification(reasoning):
    return (
        f"importance={reasoning.get('importance', '?'):<6} "
        f"event={reasoning.get('event_type', '?'):<10} "
        f"directed={reasoning.get('directed_at_user', False)!s:<5} "
        f"msg=\"{reasoning.get('message', '')}\""
    )


def _gemini_skip_reason(transcript, alternatives=None):
    if not GEMINI_API_KEY:
        return "no API key"

    if not transcript:
        return "no transcript"

    text = transcript.lower()

    name_target = alternatives if alternatives else transcript

    if detect_name(name_target):
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


def call_gemini_blocking(transcript, angle, volume, alternatives=None):
    global _gemini_pause_until, _gemini_last_call, _gemini_consecutive_429

    tag = _event_tag(angle, volume)

    skip = _gemini_skip_reason(transcript, alternatives=alternatives)
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


async def call_gemini(transcript, angle, volume, alternatives=None):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                call_gemini_blocking,
                transcript, angle, volume, alternatives,
            ),
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
    Background task spawned per trigger.

    Sequence:
      1. Capture adaptive audio window (pre-trigger + post-trigger).
      2. Silero VAD gate inside transcribe_audio — skip if no speech.
      3. Whisper STT with USER_NAME-biased initial_prompt.
      4. Transcript deduplication — suppress near-identical repeats.
      5. Local name + keyword detection → push HUD "transcribed" update.
      6. Gemini for higher-quality reasoning → push "final" HUD update.
         On Gemini skip/failure the local result becomes final.
    """

    tag = _event_tag(angle, volume)

    audio_array, post_sec = await capture_phrase()
    transcript, alternatives = await asyncio.to_thread(transcribe_audio, audio_array)

    # Deduplication: if the transcript is nearly identical to the last one,
    # broadcast nothing new — just let the existing HUD state stand.
    if transcript and not is_new_transcript(transcript):
        print(f"{tag} transcript: DUPLICATE suppressed (\"{transcript}\")")
        return

    name_match = name_match_details(alternatives) if alternatives else None
    name_detected = name_match is not None

    if transcript:
        print(f"{tag} transcript: \"{transcript}\" (name_detected={name_detected}, post={post_sec:.2f}s)")
    else:
        print(f"{tag} transcript: <empty> (post={post_sec:.2f}s)")

    if name_match:
        print(
            f"{tag} name match: word={name_match['word']!r} "
            f"target={name_match['target']!r} ratio={name_match['ratio']:.2f} "
            f"({name_match['kind']}) in alt={name_match['alternative']!r}"
        )

    local = local_reason(transcript, angle, volume, name_detected)
    print(f"{tag} local:  {_fmt_classification(local)}")

    await broadcast(make_payload(
        angle, volume,
        stage="transcribed",
        transcript=transcript,
        name_detected=name_detected,
        reasoning=local,
    ))

    gemini = await call_gemini(transcript, angle, volume, alternatives=alternatives)

    if gemini is None:
        print(f"{tag} FINAL (local):  {_fmt_classification(local)}")
        await broadcast(make_payload(
            angle, volume,
            stage="final",
            transcript=transcript,
            name_detected=name_detected,
            reasoning=local,
        ))
        return

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
        volume = get_volume_level()

        if volume < VOLUME_THRESHOLD:
            await asyncio.sleep(POLL_INTERVAL)
            continue

        angle = await asyncio.to_thread(get_doa_angle)

        if angle is None:
            await asyncio.sleep(POLL_INTERVAL)
            continue

        now = time.monotonic()

        should_broadcast_directional = (
            last_angle is None or
            angle_difference(angle, last_angle) >= MIN_ANGLE_CHANGE
        )

        if should_broadcast_directional:
            last_angle = angle
            await broadcast(make_payload(angle, volume, stage="directional"))
            print(f"{_event_tag(angle, volume)} 📡 directional alert → {len(CLIENTS)} client(s)")

        if now - last_enrich >= ENRICH_COOLDOWN_SEC:
            last_enrich = now
            asyncio.create_task(enrich_event(angle, volume))

        await asyncio.sleep(POLL_INTERVAL)


async def main():
    print("🎙️  DOA WebSocket server starting on ws://0.0.0.0:8765")

    if _DOTENV_LOADED:
        print(f"    .env            = loaded {_DOTENV_LOADED} var(s)")
    else:
        print("    .env            = not loaded (file missing or empty)")

    print(f"    USER_NAME       = {USER_NAME}")

    if USER_NAME_ALIASES:
        print(f"    ALIASES         = {', '.join(USER_NAME_ALIASES)}")

    print(f"    FUZZY THRESH    = {NAME_FUZZY_THRESHOLD}")
    print(f"    WHISPER_MODEL   = {WHISPER_MODEL}  (beam={WHISPER_BEAM_SIZE}, vad={WHISPER_VAD_FILTER})")
    print(f"    SILERO_VAD      = {'enabled' if USE_SILERO_VAD else 'disabled'}")
    print(f"    GEMINI_MODEL    = {GEMINI_MODEL}")
    print(f"    GEMINI_KEY      = {'set' if GEMINI_API_KEY else 'NOT set (local fallback only)'}")
    print(f"    AUDIO WINDOW    = {PREBUFFER_SEC:.2f}s pre + {POSTBUFFER_SEC:.2f}–{POSTBUFFER_MAX_SEC:.2f}s post (adaptive)")
    print(f"    ENRICH_COOLDOWN = {ENRICH_COOLDOWN_SEC:.1f}s  |  DEDUP_THRESH = {TRANSCRIPT_DEDUP_THRESHOLD:.2f}")

    # Eagerly load heavy models at startup so the first real event isn't slow.
    print("    Pre-loading models …")
    try:
        await asyncio.to_thread(_get_whisper)
    except Exception as exc:
        print(f"    WARNING: Whisper pre-load failed: {exc}")

    if USE_SILERO_VAD:
        try:
            await asyncio.to_thread(_get_silero)
        except Exception as exc:
            print(f"    WARNING: Silero VAD pre-load failed: {exc}")

    try:
        start_audio_stream()
    except Exception:
        return

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
