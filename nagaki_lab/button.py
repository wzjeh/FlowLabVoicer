"""Physical pushbutton input via Pi GPIO.

A momentary-contact pushbutton (e.g. Sanwa OBSF-30-K) wired between a GPIO
pin and ground. Internal pull-up; pressing the button drives the pin low.
Each press invokes a user-supplied async callback — by convention,
``ConversationLoop.on_user_action`` — so the button is functionally
identical to pressing ENTER in the terminal UI.

This is the third input surface alongside the keyboard ENTER and the wake
word, and is the only one that survives systemd-daemon mode (no TTY and
no on-mic person required).

Wiring (default BCM 17 = header pin 11, freed when the ReSpeaker HAT v2.0
was removed)::

    one terminal  -> GPIO 17 (header pin 11)
    other terminal -> GND     (header pin 9)

Use as an async context manager from a running event loop::

    async with ButtonListener(on_press=loop.on_user_action) as b:
        if not b.available:
            print('button disabled:', b.error)
        # ... button presses now fire on_press() coroutines ...
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from . import config


class ButtonListener:
    """Wraps gpiozero.Button, bridging its threaded callback into asyncio."""

    def __init__(
        self,
        gpio_pin: Optional[int] = None,
        on_press: Optional[Callable[[], Awaitable[None]]] = None,
        bounce_time: float = 0.05,
    ):
        self.gpio_pin = gpio_pin if gpio_pin is not None else config.BUTTON_GPIO_PIN
        self.on_press = on_press
        self.bounce_time = bounce_time
        self._button = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.available: bool = False
        self.error: Optional[str] = None

    async def __aenter__(self) -> "ButtonListener":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self) -> None:
        """Acquire the GPIO pin and arm the press callback."""
        if self.gpio_pin is None:
            self.error = "BUTTON_GPIO_PIN is None — button disabled"
            return
        try:
            from gpiozero import Button
            self._loop = asyncio.get_running_loop()
            self._button = Button(
                self.gpio_pin,
                pull_up=True,
                bounce_time=self.bounce_time,
            )
            self._button.when_pressed = self._on_pressed_thread
            self.available = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.available = False

    def _on_pressed_thread(self) -> None:
        """Called from gpiozero's worker thread on each button press.

        Schedules the user's async callback on the event loop. We can't
        directly ``await`` here — gpiozero invokes us synchronously from a
        non-asyncio thread.
        """
        if self.on_press is None or self._loop is None:
            return

        def _spawn() -> None:
            cb = self.on_press
            if cb is None:
                return
            try:
                asyncio.create_task(cb())
            except RuntimeError:
                # event loop is closing — silently drop the press
                pass

        try:
            self._loop.call_soon_threadsafe(_spawn)
        except Exception:
            pass

    async def stop(self) -> None:
        """Release the GPIO pin."""
        if self._button is not None:
            try:
                self._button.close()
            except Exception:
                pass
        self._button = None
        self.available = False
