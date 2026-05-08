"""Central configuration. All paths, model names, audio rates and conversation
thresholds live here. Business code imports from this module rather than
hard-coding constants."""
from __future__ import annotations

from pathlib import Path

# ---------- paths ----------
HOME = Path.home()
PROJECT_ROOT = HOME / "voice"
DATA_DIR = PROJECT_ROOT
GEMINI_KEY_PATH = HOME / ".gemini_key"
MEMORY_DB_PATH = DATA_DIR / "memory.db"

# ---------- audio I/O ----------
# Live API expects 16 kHz mono S16 input, returns 24 kHz mono S16 output.
INPUT_RATE = 16_000
INPUT_CHANNELS = 1
INPUT_BLOCK_MS = 50               # mic callback granularity
OUTPUT_RATE = 24_000
OUTPUT_CHANNELS = 1
# Route playback through PipeWire's ALSA bridge ("default") rather than the
# raw hardware device ("plughw:UACDemoV10,0"). PipeWire mixes multiple
# clients, so aplay (model TTS) and ffplay (music tool) can share the USB
# speaker simultaneously instead of fighting for an exclusive hardware
# handle. Symptom this fixed: "aplay: audio open error: Device or resource
# busy" whenever the model tried to speak while ffplay was still alive
# (paused or otherwise) — SIGSTOP doesn't release ALSA handles.
SPEAKER_DEVICE = "default"

# ---------- Live API ----------
MODEL = "gemini-2.5-flash-native-audio-latest"
API_VERSION = "v1beta"
DEFAULT_VOICE = "Aoede"
AVAILABLE_VOICES = ("Aoede", "Puck", "Charon", "Kore", "Fenrir")

# Light translation requests use a smaller / cheaper model than the Live model.
TRANSLATE_MODEL = "gemini-flash-lite-latest"

# ---------- conversation tuning ----------
MIN_CAPTURE_SECONDS = 0.5         # below this we drop the capture
MIN_PEAK_RMS = 150.0              # int16 peak below this counted as silent
MAX_CAPTURE_SECONDS = 60.0        # safety cap if user forgets to press ENTER
TRAILING_SILENCE_S = 1.5          # zero-PCM padding so server VAD sees a clean end
SERVER_IDLE_TIMEOUT_S = 25.0      # abort if server is silent for this long
TOOL_CALL_TIMEOUT_S = 12.0        # abort a tool dispatch if it takes longer; send error back
UPLOAD_CHUNK_BYTES = 4096         # bytes per send_realtime_input call
RECONNECT_BACKOFF_INITIAL_S = 1.0
RECONNECT_BACKOFF_MAX_S = 30.0

# ---------- bluetooth ----------
BT_HEADSET_NAME_SUBSTRING = "Baseus"

# ---------- wake word ----------
# 'alexa' chosen over hey_jarvis: easier to pronounce in Mandarin (啊里克莎)
# and Japanese (アレクサ — already in daily usage as a loanword), shipped
# with the largest training corpus among the pre-trained models so its
# false-positive rate is lowest. To switch back, just change this string;
# any model in openwakeword's resources/models/ is selectable by name stem.
DEFAULT_WAKE_WORD = "alexa"
# 0.65 with post-response mute (see ConversationLoop._set_state) gives
# the cleanest UX: stricter than the original 0.5, comfortably above the
# noise-floor 0.57, and the mute window covers the dominant remaining
# false-positive source (model TTS tail echoing back through the BT mic).
# Trade-off: very-marginal real wakes (~0.62 — borderline mumbles) will
# miss; loud-clear "alexa" reliably scores 0.7+ even on CVSD. If you
# personally tend to under-articulate, drop to 0.6.
WAKE_WORD_THRESHOLD = 0.65
WAKE_COOLDOWN_S = 1.5
# Wake-mode captures that contain only background noise should NOT be
# uploaded — they waste an API turn and let the model hallucinate a
# response to garbage. Threshold needs to sit between "real speech"
# and "ambient noise / model TTS tail echo" peaks.
#
# Calibration data:
#   BT headset (CVSD 8 kHz):    real speech 8000-15000, noise <3000
#   USB mic (AB13X 16 kHz):     real speech 2000-4000,  noise/echo <300
#
# 1500 sits cleanly in the gap for the USB mic (covers speech, rejects
# echoes / silent OWW false-positives) and is still safe for BT (real
# speech easily clears it). If you swap to a louder mic later you can
# raise this; if speech keeps getting rejected lower it.
# ENTER-mode capture stays gated by the very-permissive MIN_PEAK_RMS
# below because the user explicitly pressed the key.
WAKE_MIN_PEAK_RMS = 1500.0

# ---------- LEDs (APA102 on SPI) ----------
LED_NUM_PIXELS = 3
LED_DEFAULT_BRIGHTNESS = 8

# ---------- physical button (e.g. Sanwa OBSF-30-K) ----------
# BCM pin number of the input — header pin 11 = BCM 17. Wire one terminal
# to this pin and the other to any GND pin. Internal pull-up, so the
# button only needs the two terminals (no resistor). Set to None to
# disable the button feature entirely.
# GPIO 17 was previously claimed by the ReSpeaker HAT v2.0's onboard
# button; with the HAT removed it's free for the external Sanwa.
BUTTON_GPIO_PIN = 17

# ---------- PubChem (chemistry tool backend) ----------
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest"
PUBCHEM_HTTP_TIMEOUT_S = 20.0


def read_api_key() -> str:
    """Read the Gemini API key once. Never echo or log this value."""
    return GEMINI_KEY_PATH.read_text().strip()
