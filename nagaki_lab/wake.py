"""Wake-word detector via OpenWakeWord (v0.4 API).

Pre-trained wakewords shipped with the package: alexa, hey_jarvis,
hey_mycroft, timer, weather. We default to `hey_jarvis` (Iron Man / JARVIS
vibe, fits a chemistry lab). A custom "Nagaki" model is a future training
task — when that .onnx is available, instantiate this class with
`wakeword="nagaki"` and OpenWakeWord will load it via its filename match
or you can pass the path directly via `model_path=`.

Usage:
    detector = WakeWordDetector("hey_jarvis")
    while ...:
        chunk_int16 = ...   # 50–100 ms of int16 mono audio at 16 kHz
        if detector.feed(chunk_int16):
            # wake detected!

Each call to feed() returns True only on the leading edge of a detection;
a short cooldown prevents the same wake utterance from re-firing.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from . import config


class WakeWordDetector:

    def __init__(
        self,
        wakeword: str = config.DEFAULT_WAKE_WORD,
        threshold: float = config.WAKE_WORD_THRESHOLD,
        cooldown_s: float = config.WAKE_COOLDOWN_S,
        model_path: Optional[str] = None,
    ):
        from openwakeword import get_pretrained_model_paths
        from openwakeword.model import Model

        if model_path is None:
            model_path = self._resolve_pretrained_path(
                wakeword, get_pretrained_model_paths()
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"wake-word model not found: {model_path}")

        # OpenWakeWord uses the basename (without .onnx) as the dict key in
        # predict() output — e.g. "hey_jarvis_v0.1".
        self._model_key = os.path.splitext(os.path.basename(model_path))[0]

        self.wakeword = wakeword
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.model = Model(wakeword_model_paths=[model_path])
        self.last_score = 0.0
        self._cooldown_until = 0.0

    @staticmethod
    def _resolve_pretrained_path(name: str, available_paths: list[str]) -> str:
        for p in available_paths:
            base = os.path.splitext(os.path.basename(p))[0]
            # match "hey_jarvis" against "hey_jarvis_v0.1"
            if base == name or base.startswith(f"{name}_"):
                return p
        names = sorted({
            os.path.splitext(os.path.basename(p))[0].split("_v")[0]
            for p in available_paths
        })
        raise ValueError(
            f"no pre-trained wake-word matching '{name}'. Available: {names}"
        )

    def feed(self, chunk_int16: np.ndarray) -> bool:
        """Returns True only on the rising edge of a detection."""
        if time.monotonic() < self._cooldown_until:
            # keep model state fresh during cooldown so it doesn't double-trigger
            self.model.predict(chunk_int16)
            return False
        scores = self.model.predict(chunk_int16)
        score = float(scores.get(self._model_key, 0.0))
        self.last_score = score
        if score >= self.threshold:
            self._cooldown_until = time.monotonic() + self.cooldown_s
            return True
        return False
