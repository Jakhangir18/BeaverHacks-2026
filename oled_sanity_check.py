#!/usr/bin/env python3
"""
Temporary OLED sanity check (SSD1306 128x64 via I2C).

Does not touch server.py or websockets; only verifies wiring, address, Blinka,
and that pixels update.

Usage (on Raspberry Pi):

  source .venv/bin/activate   # optional
  python oled_sanity_check.py

Stops cleanly on Ctrl+C after blanking the display.
"""

import sys
import time

import board
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

WIDTH = 128
HEIGHT = 64
OLED_ADDR = 0x3C

try:
    _font_large = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
    )
    _font_small = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10
    )
except Exception:
    _font_large = ImageFont.load_default()
    _font_small = ImageFont.load_default()


def _paint(oled: adafruit_ssd1306.SSD1306_I2C, mono: Image.Image) -> None:
    oled.image(mono)
    oled.show()


def _center_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
    bx0, by0, bx1, by1 = draw.textbbox((0, 0), text, font=font)
    w, h = bx1 - bx0, by1 - by0
    x = max(0, (WIDTH - w) // 2)
    y = max(0, (HEIGHT - h) // 2)
    return x, y, w, h


def main() -> int:
    print(f"OLED sanity: initializing I2C + SSD1306_I2C (addr=0x{OLED_ADDR:02X})...")

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=OLED_ADDR)
    except Exception as exc:
        print(f"FAILED to open display: {exc}", file=sys.stderr)
        print("Tip: enable I2C, check Addr 0x3C wiring, adafruit-blinka installed?", file=sys.stderr)
        return 1

    def phase(name: str, delay: float) -> None:
        print(f"  {name}")
        time.sleep(delay)

    try:
        # 1) All off
        mono = Image.new("1", (WIDTH, HEIGHT), 0)
        _paint(oled, mono)
        phase("Canvas black", 0.25)

        # 2) All on (shows dead rows/columns clearly)
        mono = Image.new("1", (WIDTH, HEIGHT), 1)
        _paint(oled, mono)
        phase("Canvas white (full invert check)", 0.6)

        # 3) Black + hollow border + diagonal corners
        mono = Image.new("1", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(mono)
        draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), outline=255, width=2)
        draw.rectangle((1, 1, 5, 5), outline=255, fill=255)
        draw.rectangle((WIDTH - 6, HEIGHT - 6, WIDTH - 2, HEIGHT - 2), outline=255, fill=255)
        draw.line((24, HEIGHT - 1, WIDTH - 25, 0), fill=255)
        _paint(oled, mono)
        phase("Border + markers", 0.5)

        # 4) Text screen
        mono = Image.new("1", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(mono)
        t1 = "OLED SANITY OK"
        t2 = "SPOOT HUD"
        x1, _, _w1, h1 = _center_bbox(draw, t1, _font_large)
        x2, _, _w2, h2 = _center_bbox(draw, t2, _font_small)
        gap = 4
        block_h = h1 + gap + h2
        top_y = max(6, (HEIGHT - block_h) // 2)
        draw.text((x1, top_y), t1, font=_font_large, fill=255)
        draw.text((x2, top_y + h1 + gap), t2, font=_font_small, fill=255)
        _paint(oled, mono)
        phase(f'Holding "{t1}" 4 s', 4.0)

        # Clear
        oled.fill(0)
        oled.show()
        print("  Cleared. Done.")
        return 0

    except KeyboardInterrupt:
        print("\nStopped (interrupt). Clearing...")
        oled.fill(0)
        oled.show()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
