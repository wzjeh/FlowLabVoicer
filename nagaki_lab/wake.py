"""Wake-word detector via OpenWakeWord (v0.4 API).

Pre-trained wakewords shipped with the package: alexa, hey_jarvis,
hey_marvin, hey_mycroft, timer, weather. We default to `alexa` — chosen
over hey_jarvis because it has the largest training corpus among the
pre-trained models (lowest false-positive rate) and is easily pronounced
in Mandarin (啊里克莎) and Japanese (アレクサ, an everyday loanword). A
custom Chinese-native model (e.g. "小爱" / "永木") is a future training
task — when that .onnx is available, instantiate this class with the
appropriate `wakeword=` name (OpenWakeWord matches by filename stem) or
pass an explicit path via `model_path=`.

Usage:
    detector = WakeWordDetector("alexa")
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
        # Always run the model and refresh last_score, even when muted /
        # in cooldown — otherwise self.last_score gets frozen at whatever
        # value triggered the last wake and the dispatcher's monitoring
        # log shows stale "score = 0.71" lines for the whole mute window
        # (looks like a stuck score; actually the detector is fine, just
        # not reporting current numbers). The fire decision is then a
        # separate check below.
        scores = self.model.predict(chunk_int16)
        score = float(scores.get(self._model_key, 0.0))
        self.last_score = score
        if time.monotonic() < self._cooldown_until:
            return False
        if score >= self.threshold:
            self._cooldown_until = time.monotonic() + self.cooldown_s
            return True
        return False

    def mute_until(self, deadline_monotonic: float) -> None:
        """Postpone the next possible detection until at least the given
        monotonic timestamp. Chunks fed during the muted window still go
        through the model so its internal state stays consistent — they
        just can't fire a wake. Used by ConversationLoop to suppress
        false triggers caused by the model's TTS tail echoing back into
        the mic right after a response ends.
        """
        if deadline_monotonic > self._cooldown_until:
            self._cooldown_until = deadline_monotonic
