"""APA102 LED driver for the optional status indicator.

The ReSpeaker 2-Mics Pi HAT v2.0 does NOT in fact include APA102 LEDs (only
v1.0 did), so this module is currently a no-op on this hardware. It is kept
for future use when the user attaches an external APA102 / WS2812 strip on
the SPI header.

State machine: off / idle / listening / thinking / speaking / error.
Animations (e.g. 'thinking' = sinusoidal pulse) run as a background task.
"""
from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Optional

try:
    import spidev
    HAS_SPI = True
except ImportError:
    HAS_SPI = False

from . import config


def _find_spidev() -> Optional[tuple[int, int]]:
    """Return (bus, device) of the first /dev/spidev*.* node, or None."""
    try:
        names = sorted(os.listdir("/dev"))
    except OSError:
        return None
    for n in names:
        if not n.startswith("spidev"):
            continue
        rest = n[len("spidev"):]
        if "." not in rest:
            continue
        bus_s, dev_s = rest.split(".", 1)
        try:
            return int(bus_s), int(dev_s)
        except ValueError:
            continue
    return None


class APA102:
    def __init__(
        self,
        num_leds: int = config.LED_NUM_PIXELS,
        bus: Optional[int] = None,
        device: Optional[int] = None,
        max_speed_hz: int = 8_000_000,
    ):
        self.num_leds = num_leds
        self._pixels: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * num_leds
        self.error: Optional[str] = None
        self.spi = None

        if not HAS_SPI:
            self.error = "spidev python lib not installed"
            return
        if bus is None or device is None:
            found = _find_spidev()
            if found is None:
                self.error = "no /dev/spidev* device found (SPI not enabled?)"
                return
            bus, device = found
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(bus, device)
            self.spi.max_speed_hz = max_speed_hz
            self.spi.mode = 0
        except (OSError, PermissionError) as e:
            self.error = f"failed to open /dev/spidev{bus}.{device}: {e}"
            self.spi = None

    @property
    def available(self) -> bool:
        return self.spi is not None

    def fill(self, r: int, g: int, b: int, brightness: int = config.LED_DEFAULT_BRIGHTNESS) -> None:
        for i in range(self.num_leds):
            self._pixels[i] = (
                max(0, min(255, int(r))),
                max(0, min(255, int(g))),
                max(0, min(255, int(b))),
                max(0, min(31, int(brightness))),
            )

    def show(self) -> None:
        if not self.available:
            return
        data = [0x00, 0x00, 0x00, 0x00]
        for r, g, b, br in self._pixels:
            data += [0xE0 | (br & 0x1F), b, g, r]
        end_bytes = max(1, (self.num_leds + 15) // 16)
        data += [0xFF] * end_bytes
        try:
            self.spi.xfer2(data)
        except OSError as e:
            self.error = f"SPI write failed: {e}"

    def off(self) -> None:
        self.fill(0, 0, 0, 0)
        self.show()

    def close(self) -> None:
        if self.spi is None:
            return
        try:
            self.off()
            self.spi.close()
        except Exception:
            pass
        self.spi = None


class LEDStatus:
    """High-level state setter; animations run in a background task."""

    PALETTE = {
        # state    -> (r,   g,   b,  brightness, animation)
        "off":       (0,   0,   0,   0,    "solid"),
        "idle":      (10,  10,  30,  4,    "solid"),    # dim blue
        "listening": (0,   60,  100, 16,   "solid"),    # cyan
        "thinking":  (200, 140, 0,   None, "pulse"),    # yellow pulse
        "speaking":  (0,   150, 50,  16,   "solid"),    # green
        "error":     (200, 0,   0,   16,   "solid"),    # red
    }

    def __init__(self, num_leds: int = config.LED_NUM_PIXELS):
        self.led = APA102(num_leds=num_leds)
        self.state = "off"
        self._anim_task: Optional[asyncio.Task] = None

    @property
    def available(self) -> bool:
        return self.led.available

    @property
    def error(self) -> Optional[str]:
        return self.led.error

    async def set_state(self, state: str) -> None:
        if not self.available or state not in self.PALETTE:
            return
        if self._anim_task is not None and not self._anim_task.done():
            self._anim_task.cancel()
            try:
                await self._anim_task
            except asyncio.CancelledError:
                pass
            self._anim_task = None
        self.state = state
        r, g, b, br, anim = self.PALETTE[state]
        if anim == "solid":
            self.led.fill(r, g, b, br or 0)
            self.led.show()
        elif anim == "pulse":
            self._anim_task = asyncio.create_task(self._pulse(r, g, b))

    async def _pulse(self, r: int, g: int, b: int, period_s: float = 1.2) -> None:
        try:
            t0 = time.monotonic()
            while True:
                t = time.monotonic() - t0
                phase = (t % period_s) / period_s
                br_norm = 0.4 + 0.6 * (0.5 - 0.5 * math.cos(2 * math.pi * phase))
                self.led.fill(r, g, b, int(31 * br_norm))
                self.led.show()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise

    async def stop(self) -> None:
        if self._anim_task is not None and not self._anim_task.done():
            self._anim_task.cancel()
            try:
                await self._anim_task
            except asyncio.CancelledError:
                pass
        self.led.close()
