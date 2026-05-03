import asyncio
import json

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
OLED_ADDR = 0x3C

# If this runs on the same Raspberry Pi as server.py, keep localhost.
WS_URL = "ws://localhost:8765"

# If the direction looks rotated, adjust this.
# Examples: 90, -90, 180
ANGLE_OFFSET = 0

# If left/right feel backwards, change this to True.
FLIP_LEFT_RIGHT = False

# Edge thickness in pixels for the directional border.
EDGE_THICKNESS = 5
EDGE_THICKNESS_HIGH = 9

# How long (seconds) to keep showing a "final" message before reverting
# back to plain directional mode the next time we get a "directional" frame.
MESSAGE_HOLD_SEC = 3.0


# =========================
# OLED setup
# =========================

i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=OLED_ADDR)


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


def draw_centered_text(draw, text, y, font, fill=255):
    width, _ = text_size(draw, text, font)
    x = (WIDTH - width) // 2
    draw.text((x, y), text, font=font, fill=fill)


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
    oled.fill(0)
    oled.show()


def show_status(message):
    image = Image.new("1", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)

    draw_centered_text(draw, "SOUND HUD", 17, medium_font)
    draw_centered_text(draw, message, 36, small_font)

    oled.image(image)
    oled.show()


def show_frame(raw_angle, message="", importance="low"):
    """
    Render one HUD frame.

    raw_angle    angle from server (degrees, 0=front, 90=right, ...)
    message      optional short text describing what was heard.
                 When empty we render the original "big direction word" UI.
    importance   "low" | "medium" | "high"
                 high   -> thick edges + inverted (white) background
                 medium -> normal edges
                 low    -> normal edges
    """

    high = importance == "high"

    bg = 255 if high else 0
    fg = 0 if high else 255

    image = Image.new("1", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(image)

    angle = apply_angle_adjustments(raw_angle)
    line1, line2, edges = direction_info(angle)

    thickness = EDGE_THICKNESS_HIGH if high else EDGE_THICKNESS
    draw_edge_highlights(draw, edges, thickness=thickness, fill=fg)

    if message:
        # Compact direction header at top, message body below.
        header = line1 if not line2 else f"{line1} {line2}"
        draw_centered_text(draw, header, 6, medium_font, fill=fg)

        # Wrap message inside the inner area (clear of the edge bars).
        inner_max = WIDTH - 2 * (thickness + 2)
        lines = wrap_text(draw, message, small_font, inner_max)[:3]

        y = 28
        for line in lines:
            draw_centered_text(draw, line, y, small_font, fill=fg)
            y += 12
    else:
        # Original directional-only layout.
        if line2:
            draw_centered_text(draw, line1, 19, medium_font, fill=fg)
            draw_centered_text(draw, line2, 35, medium_font, fill=fg)
        else:
            draw_centered_text(draw, line1, 25, big_font, fill=fg)

    oled.image(image)
    oled.show()


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

                    # On the IMMEDIATE alert we deliberately drop any leftover
                    # message from the previous event so the user instantly
                    # sees a clean directional frame.
                    if stage == "directional":
                        show_frame(angle, message="", importance="low")
                    else:
                        show_frame(angle, message=message, importance=importance)

        except Exception as error:
            print(f"Connection error: {error}")
            show_status("No server")
            await asyncio.sleep(1)


# =========================
# Main
# =========================

if __name__ == "__main__":
    try:
        clear_display()
        asyncio.run(run_oled_client())

    except KeyboardInterrupt:
        clear_display()
        print("OLED HUD stopped.")
