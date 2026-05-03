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

# Edge thickness in pixels
EDGE_THICKNESS = 5


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


def draw_centered_text(draw, text, y, font):
    width, _ = text_size(draw, text, font)
    x = (WIDTH - width) // 2
    draw.text((x, y), text, font=font, fill=255)


def draw_edge_highlights(draw, edges):
    t = EDGE_THICKNESS

    if "top" in edges:
        draw.rectangle((0, 0, WIDTH, t), fill=255)

    if "right" in edges:
        draw.rectangle((WIDTH - t, 0, WIDTH, HEIGHT), fill=255)

    if "bottom" in edges:
        draw.rectangle((0, HEIGHT - t, WIDTH, HEIGHT), fill=255)

    if "left" in edges:
        draw.rectangle((0, 0, t, HEIGHT), fill=255)


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


def show_frame(raw_angle, connected=True):
    image = Image.new("1", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)

    angle = apply_angle_adjustments(raw_angle)
    line1, line2, edges = direction_info(angle)

    draw_edge_highlights(draw, edges)

    if line2:
        draw_centered_text(draw, line1, 19, medium_font)
        draw_centered_text(draw, line2, 35, medium_font)
    else:
        draw_centered_text(draw, line1, 25, big_font)

    # Small connection indicator, kept subtle.
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

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        angle = data.get("angle")

                        if angle is None:
                            continue

                        show_frame(angle, connected=True)

                    except json.JSONDecodeError:
                        print(f"Bad JSON received: {message}")

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
