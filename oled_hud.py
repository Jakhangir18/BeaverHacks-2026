import asyncio
import json
import os
import sys
import time

import board
import busio
import websockets
from PIL import Image, ImageDraw, ImageFont

import adafruit_ssd1306


# =========================
# Display config
# =========================

WIDTH = 128
HEIGHT = 64

# Elecrow / ELEGOO style 128x64 dual-color OLED: horizontal band ends at this row so
# small labels stay in one physical color zone (often blue narrow strip) while the
# main HUD (typically yellow zone) fills the remainder. Adjust if yours differs.
_DUAL_SPLIT = os.environ.get("OLED_DUAL_COLOR_SPLIT", "").strip()
if _DUAL_SPLIT:
    TITLE_BAND_HEIGHT = max(12, min(24, int(_DUAL_SPLIT)))
else:
    TITLE_BAND_HEIGHT = 16

# I2C address: most SSD1306 boards are 0x3C; some use 0x3D ( ADDR solder jumper ).
# Set env if needed, e.g. export OLED_I2C_ADDR=0x3d

# Physical mount: rotate the whole HUD 180° when the panel is upside down.
OLED_ROTATE_180 = os.environ.get("OLED_ROTATE_180", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Same Pi as server.py: localhost works. Separate machine: OLED_WS_URL=ws://192.168.x.x:8765.
WS_URL = os.environ.get("OLED_WS_URL", "ws://127.0.0.1:8765").strip()

# If the direction looks rotated, adjust this.
# Examples: 90, -90, 180
ANGLE_OFFSET = 0

# If left/right feel backwards, change this to True.
FLIP_LEFT_RIGHT = False

# Edge thickness in pixels for the directional border.
EDGE_THICKNESS = 5
EDGE_THICKNESS_HIGH = 9

# After a "your name" style HUD alert, ignore directional clears for this long (seconds).
NAME_CALL_HOLD_SEC = float(os.environ.get("OLED_NAME_CALL_HOLD_SEC", "5"))

# No I2C hardware: still subscribe to the server and print each frame to stdout (good over SSH).
OLED_HEADLESS = os.environ.get("OLED_HEADLESS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

i2c = None
oled = None


def _oled_i2c_addr() -> int:
    raw = os.environ.get("OLED_I2C_ADDR", "0x3c").strip().lower()
    if raw.startswith("0x"):
        return int(raw, 16)
    return int(raw)


def _print_oled_i2c_help(exc: BaseException, addr: int) -> None:
    bus_hint = "try: sudo i2cdetect -y 1   (on some boards also -y 10)"
    print(
        f"\nOLED I2C init failed at address 0x{addr:02x}: {exc!r}\n"
        "Things to check:\n"
        "  - raspi-config: Interface Options -> I2C enabled, then reboot\n"
        f"  - Scan the bus ({bus_hint})\n"
        "  - SSD1306 GND/VCC/SDA/SCL and 3v3/logic levels\n"
        "  - If the module has an ADDR jumper, export OLED_I2C_ADDR=0x3d and retry\n"
        "  - Another driver using the same pins (disable it or wire to bus 2)\n"
        "SSH without a panel: OLED_HEADLESS=1 python oled_hud.py\n",
        file=sys.stderr,
        flush=True,
    )


def init_oled_panel() -> None:
    global i2c, oled

    if OLED_HEADLESS:
        print("OLED_HEADLESS=1: skipping I2C; frames go to stdout as text lines.")
        return

    addr = _oled_i2c_addr()
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=addr)
    except Exception as exc:
        _print_oled_i2c_help(exc, addr)
        raise SystemExit(1) from exc


# =========================
# Fonts
# =========================

try:
    big_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        17
    )
    medium_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        14
    )
    small_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        10
    )
except Exception:
    big_font = ImageFont.load_default()
    medium_font = ImageFont.load_default()
    small_font = ImageFont.load_default()


# =========================
# Helpers
# =========================

def _rotate_for_panel(image: Image.Image) -> Image.Image:
    """Return image oriented for how the SSD1306 is mounted (default: 180°)."""

    if not OLED_ROTATE_180:
        return image
    try:
        return image.transpose(Image.Transpose.ROTATE_180)
    except AttributeError:
        return image.transpose(Image.ROTATE_180)


def oled_blit(image: Image.Image) -> None:
    if oled is None:
        return
    oled.image(_rotate_for_panel(image))
    oled.show()


def normalize_angle(angle):
    return angle % 360


def apply_angle_adjustments(angle):
    angle = normalize_angle(angle + ANGLE_OFFSET)

    if FLIP_LEFT_RIGHT:
        angle = normalize_angle(360 - angle)

    return angle


def direction_info(angle):
    """
    Returns:
      line1, line2, edges

    Edges:
      "top" = sound in front
      "right" = sound to the right
      "bottom" = sound behind
      "left" = sound to the left
    """

    angle = normalize_angle(angle)

    if angle >= 337.5 or angle < 22.5:
        return "FRONT", "", ["top"]

    if angle < 67.5:
        return "FRONT", "RIGHT", ["top", "right"]

    if angle < 112.5:
        return "RIGHT", "", ["right"]

    if angle < 157.5:
        return "BACK", "RIGHT", ["bottom", "right"]

    if angle < 202.5:
        return "BEHIND", "", ["bottom"]

    if angle < 247.5:
        return "BACK", "LEFT", ["bottom", "left"]

    if angle < 292.5:
        return "LEFT", "", ["left"]

    return "FRONT", "LEFT", ["top", "left"]


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _text_dims(draw, text, font):
    bx0, by0, bx1, by1 = draw.textbbox((0, 0), text, font=font)
    return bx1 - bx0, by1 - by0, bx0, by0


def draw_centered_in_rect(
    draw, text, font, left, top, right_ex, bot_ex, fill=255
) -> None:
    """Horizontal + vertical center inside half-open rectangle [left, right_ex) × [top, bot_ex)."""
    if not text or right_ex <= left or bot_ex <= top:
        return
    w, h, _bx0, by0 = _text_dims(draw, text, font)
    x = left + max(0, (right_ex - left - w) // 2)
    vert_span = bot_ex - top
    y = top + max(0, (vert_span - h) // 2) - by0
    draw.text((int(x), int(y)), text, font=font, fill=fill)


def _horizontal_insets(thickness: int) -> tuple[int, int]:
    inset = thickness + 2
    left, right_ex = inset, WIDTH - inset
    if left >= right_ex:
        return (0, WIDTH)
    return (left, right_ex)


def _pil_dual_band_rects() -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Narrow dual-color strip vs large zone in Pillow coordinates BEFORE _rotate_for_panel.

    After a default 180° rotation, Pillow row 63 maps to physical top; keep the slim
    caption strip near Pillow bottom so it aligns with the narrow physical band on
    ELEGOO-style SSD1306 modules.
    """
    t_h = TITLE_BAND_HEIGHT
    if OLED_ROTATE_180:
        return ((HEIGHT - t_h, HEIGHT), (0, HEIGHT - t_h))
    return ((0, t_h), (t_h, HEIGHT))


def _title_draw_span_pixels(
    edges: list[str],
    thickness: int,
    ttl_top: int,
    ttl_bot_ex: int,
) -> tuple[int, int] | None:
    """
    Half-open [lo, hi_ex) suitable for lettering inside the slim dual-color strip,
    clipped when thick directional TOP/BOTTOM edge bars collide with it.
    """
    lo = ttl_top + 1
    hi_ex = ttl_bot_ex

    if "top" in edges:
        cut = thickness + 2
        if ttl_top < cut:
            lo = max(lo, min(cut, ttl_bot_ex - 8))

    if "bottom" in edges:
        bar_first = HEIGHT - thickness
        if ttl_top < HEIGHT and ttl_bot_ex > bar_first:
            hi_ex = min(hi_ex, bar_first)

    if hi_ex - lo < 6:
        return None
    return (lo, hi_ex)


def _main_content_span_pixels(
    edges: list[str],
    thickness: int,
    main_top_nom: int,
    main_bot_nom_ex: int,
) -> tuple[int, int]:
    """Half-open drawing span for large yellow-zone content."""

    bot_ex = min(main_bot_nom_ex, _main_bottom_exclusive(edges, thickness))
    top = max(main_top_nom, 1)

    cut = thickness + 2 if "top" in edges else None
    if cut is not None and top < cut:
        top = cut

    return (top, bot_ex)


def _main_bottom_exclusive(edges: list, thickness: int) -> int:
    """Exclusive bottom for main-zone content so it clears bottom edge LEDs."""
    if "bottom" in edges:
        return HEIGHT - thickness - 2
    return HEIGHT - 3


def _stack_lines_center(
    draw,
    lines: list[str],
    font,
    left: int,
    top: int,
    right_ex: int,
    bot_ex: int,
    line_gap: int,
    fill=255,
) -> None:
    """Vertically stacked, each line centered; block centered between top and bot_ex."""
    if not lines or right_ex <= left or bot_ex <= top:
        return
    metrics = [_text_dims(draw, ln, font) for ln in lines]
    hs = [m[1] for m in metrics]
    total_h = sum(hs) + line_gap * max(0, len(lines) - 1)
    span = bot_ex - top
    y = float(top + max(0, (span - total_h) // 2))
    for ln, (w, h, _bx0, by0) in zip(lines, metrics, strict=False):
        x = left + max(0, ((right_ex - left) - w) // 2)
        yi = int(y - by0)
        draw.text((x, yi), ln, font=font, fill=fill)
        y += float(h + line_gap)


def draw_edge_highlights(draw, edges, thickness=EDGE_THICKNESS, fill=255):
    t = thickness

    if "top" in edges:
        draw.rectangle((0, 0, WIDTH, t), fill=fill)

    if "right" in edges:
        draw.rectangle((WIDTH - t, 0, WIDTH, HEIGHT), fill=fill)

    if "bottom" in edges:
        draw.rectangle((0, HEIGHT - t, WIDTH, HEIGHT), fill=fill)

    if "left" in edges:
        draw.rectangle((0, 0, t, HEIGHT), fill=fill)


def wrap_text(draw, text, font, max_width):
    """Greedy word-wrap. Returns a list of lines that each fit max_width."""

    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        w, _ = text_size(draw, candidate, font)

        if w <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def clear_display():
    if oled is None:
        return
    oled.fill(0)
    oled.show()


def show_status(message):
    if OLED_HEADLESS:
        print(f"[OLED HUD] {message}", flush=True)
        return

    image = Image.new("1", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)

    left_x, right_ex = _horizontal_insets(0)
    (title_top, title_bot_ex), (main_top0, main_bex_nom) = _pil_dual_band_rects()

    tit_span = _title_draw_span_pixels([], 0, title_top, title_bot_ex)
    main_top_eff, main_bex_eff = _main_content_span_pixels([], 0, main_top0, main_bex_nom)

    if tit_span:
        tsl, tst = tit_span
        hdr_font = medium_font if tst - tsl >= 14 else small_font
        draw_centered_in_rect(draw, "SOUND HUD", hdr_font, left_x, tsl, right_ex, tst, 255)

    stacked = wrap_text(draw, message, small_font, right_ex - left_x)[:6]
    if not tit_span:
        hdr_bits = wrap_text(draw, "SOUND HUD", small_font, right_ex - left_x)
        stacked = (hdr_bits or ["SOUND HUD"])[:1] + stacked

    caption_floor = max(main_top_eff + 2, tit_span[1] + 1 if tit_span else title_bot_ex)
    if caption_floor < main_bex_eff and stacked:
        _stack_lines_center(
            draw,
            stacked,
            small_font,
            left_x,
            caption_floor,
            right_ex,
            main_bex_eff,
            line_gap=10,
            fill=255,
        )

    oled_blit(image)


def show_frame(raw_angle, message="", importance="low", name_detected=False):
    """
    Render one HUD frame.

    Dual-color 128x64 panels: OLED_DUAL_COLOR_SPLIT sets the slim band thickness (often 16px).
    With OLED_ROTATE_180 enabled (default), that strip sits on the Pillow-bottom so it pins to
    the physical top after rotate (blue stripe on ELEGOO-style OLEDs).

    raw_angle    angle from server (degrees, 0=front, 90=right, ...)
    message      optional short text describing what was heard.
                 When empty we render the original "big direction word" UI.
    importance   "low" | "medium" | "high"
                 high   -> thick edges + inverted (white) background
                 medium -> normal edges
                 low    -> normal edges
    name_detected  from server JSON; when True (or message matches name-call phrasing),
                  shows "YOUR NAME" / "WAS CALLED" in the narrow dual-color strip.
    """

    if OLED_HEADLESS:
        try:
            ang = float(raw_angle) % 360.0
        except (TypeError, ValueError):
            ang = raw_angle
        body = (message or "").replace("\n", " ").strip()
        if body:
            tail = body[:100]
        else:
            tail = "(direction only)"
        alert = ui_is_name_alert(body, name_detected)
        print(
            f"[OLED HUD] ang={ang!s} imp={importance} name_alert={alert} msg={tail!r}",
            flush=True,
        )
        return

    high = importance == "high"

    bg = 255 if high else 0
    fg = 0 if high else 255

    image = Image.new("1", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(image)

    angle = apply_angle_adjustments(raw_angle)
    line1, line2, edges = direction_info(angle)

    thickness = EDGE_THICKNESS_HIGH if high else EDGE_THICKNESS
    draw_edge_highlights(draw, edges, thickness=thickness, fill=fg)

    left_x, right_ex = _horizontal_insets(thickness)
    inner_max = right_ex - left_x

    (title_top, title_bot_ex), (main_top_nom, main_bex_nom) = _pil_dual_band_rects()
    tit_span = _title_draw_span_pixels(edges, thickness, title_top, title_bot_ex)
    main_top_eff, main_bot_eff = _main_content_span_pixels(
        edges, thickness, main_top_nom, main_bex_nom
    )

    caption_floor = max(main_top_eff + 2, tit_span[1] + 1 if tit_span else title_bot_ex)

    named = ui_is_name_alert(message, name_detected)

    if named:
        name_band = tit_span
        if name_band is None:
            ft = main_top_eff + 1
            fb = min(ft + max(12, TITLE_BAND_HEIGHT - 2), main_bot_eff - 22)
            if fb - ft >= 10:
                name_band = (ft, fb)

        if name_band:
            _draw_name_call_in_band(
                draw, left_x, right_ex, name_band[0], name_band[1], fg
            )
        else:
            ft = main_top_eff + 2
            fb = min(ft + TITLE_BAND_HEIGHT, main_bot_eff - 18)
            if fb - ft >= 8:
                _draw_name_call_in_band(draw, left_x, right_ex, ft, fb, fg)
                name_band = (ft, fb)

        dir_top = main_top_eff + 4
        if name_band is not None and tit_span is None:
            dir_top = max(dir_top, name_band[1] + 2)

        if line2:
            _stack_lines_center(
                draw,
                [line1, line2],
                medium_font,
                left_x,
                dir_top,
                right_ex,
                main_bot_eff,
                line_gap=6,
                fill=fg,
            )
        else:
            draw_centered_in_rect(
                draw,
                line1,
                big_font,
                left_x,
                dir_top,
                right_ex,
                main_bot_eff,
                fill=fg,
            )
    elif message:
        header = line1 if not line2 else f"{line1} {line2}"

        if tit_span:
            tsl, tst = tit_span
            hdr_font = medium_font if tst - tsl >= 14 else small_font
            draw_centered_in_rect(draw, header, hdr_font, left_x, tsl, right_ex, tst, fill=fg)

        captions = wrap_text(draw, message, small_font, inner_max)[:4]

        stacked = list(captions)
        if not tit_span:
            hdr_pref = wrap_text(draw, header, small_font, inner_max)
            stacked = hdr_pref[:2] + stacked

        if caption_floor < main_bot_eff and stacked:
            _stack_lines_center(
                draw,
                stacked,
                small_font,
                left_x,
                caption_floor,
                right_ex,
                main_bot_eff,
                line_gap=10 if len(stacked) < 4 else 7,
                fill=fg,
            )
    elif line2:
        _stack_lines_center(
            draw,
            [line1, line2],
            medium_font,
            left_x,
            main_top_eff + 4,
            right_ex,
            main_bot_eff,
            line_gap=6,
            fill=fg,
        )
    else:
        draw_centered_in_rect(
            draw,
            line1,
            big_font,
            left_x,
            main_top_eff + 4,
            right_ex,
            main_bot_eff,
            fill=fg,
        )

    oled_blit(image)


def message_indicates_name_call(text: str) -> bool:
    msg = (text or "").lower()
    return "your name was called" in msg or "name was called from" in msg


def payload_triggers_name_call_hold(data) -> bool:
    """Payload likely shows the name-call HUD (suppress directional clears awhile)."""

    if bool(data.get("name_detected")):
        return True
    return message_indicates_name_call(data.get("message") or "")


def ui_is_name_alert(message: str, name_detected: bool) -> bool:
    return bool(name_detected) or message_indicates_name_call(message)


def _draw_name_call_in_band(
    draw,
    left_x: int,
    right_ex: int,
    band_top: int,
    band_bot_ex: int,
    fg: int,
) -> None:
    """Two-line 'your name was called' cue inside the narrow dual-color strip."""
    if band_bot_ex <= band_top or right_ex <= left_x:
        return
    span = band_bot_ex - band_top
    if span >= 12:
        _stack_lines_center(
            draw,
            ["YOUR NAME", "WAS CALLED"],
            small_font,
            left_x,
            band_top,
            right_ex,
            band_bot_ex,
            line_gap=0,
            fill=fg,
        )
    else:
        draw_centered_in_rect(
            draw,
            "NAME CALLED",
            small_font,
            left_x,
            band_top,
            right_ex,
            band_bot_ex,
            fill=fg,
        )


# =========================
# WebSocket client
# =========================

async def run_oled_client():
    show_status("Connecting...")

    while True:
        try:
            async with websockets.connect(WS_URL) as websocket:
                print(f"Connected to {WS_URL}")
                show_status("Connected")
                hold_directional_until_mono = 0.0

                async for raw in websocket:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"Bad JSON received: {raw}")
                        continue

                    angle = data.get("angle")
                    if angle is None:
                        continue

                    stage = data.get("stage", "directional")
                    importance = data.get("importance", "low")
                    message = data.get("message", "") or ""

                    # Directional pings clear message text unless we're holding after a name-call.
                    if stage == "directional":
                        if time.monotonic() < hold_directional_until_mono:
                            continue
                        show_frame(
                            angle, message="", importance="low", name_detected=False
                        )
                    else:
                        show_frame(
                            angle,
                            message=message,
                            importance=importance,
                            name_detected=bool(data.get("name_detected")),
                        )
                        if payload_triggers_name_call_hold(data):
                            hold_directional_until_mono = (
                                time.monotonic() + NAME_CALL_HOLD_SEC
                            )

        except Exception as error:
            print(f"Connection error: {error}")
            show_status("No server")
            await asyncio.sleep(1)


# =========================
# Main
# =========================

if __name__ == "__main__":
    print(f"OLED HUD: WebSocket target is {WS_URL!r}")

    try:
        init_oled_panel()
        clear_display()
        asyncio.run(run_oled_client())

    except KeyboardInterrupt:
        clear_display()
        print("OLED HUD stopped.")
