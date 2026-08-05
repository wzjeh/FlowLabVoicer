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

# ---------- websocket I/O timeouts ----------
# Every network call in LiveSession must be time-bounded. Incident that
# motivated this (2026-06-09): during a Gemini-side 1011 outage, a
# reconnect attempt hung inside client.aio.live.connect() with no
# timeout. The run() loop blocked there for 20 HOURS — wake dispatcher
# and watchdog were already cancelled, so the device looked alive
# (service "running") but never responded to button or wake word again
# until the kernel OOM-killed it. With these timeouts every connect /
# send attempt fails fast and the existing backoff-reconnect path
# retries; combined with the systemd watchdog this makes any hang
# self-healing.
# Receive-side stalls are already covered by SERVER_IDLE_TIMEOUT_S
# above (the conversation watchdog cancels the turn and reconnects).
CONNECT_TIMEOUT_S = 30.0          # opening the Live API websocket
CLOSE_TIMEOUT_S = 10.0            # closing it (a hung close also wedges run())
SEND_TIMEOUT_S = 10.0             # any single send_* call (audio chunk, tool resp, …)

# ---------- bluetooth ----------
BT_HEADSET_NAME_SUBSTRING = "Baseus"

# ---------- wake word ----------
# 'alexa' chosen over hey_jarvis: easier to pronounce in Mandarin (啊里克莎)
# and Japanese (アレクサ — already in daily usage as a loanword), shipped
# with the largest training corpus among the pre-trained models so its
# false-positive rate is lowest. To switch back, just change this string;
# any model in openwakeword's resources/models/ is selectable by name stem.
DEFAULT_WAKE_WORD = "alexa"
# Threshold history:
#   BT headset:  0.65 (noise floor ~0.57, needed margin)
#   USB AB13X:   0.65 (near-field, real "alexa" scored 0.7-1.0 reliably)
#   XVF3800:     0.40 — measured wake scores by distance (normal voice):
#       0.5 m -> 0.81,  1.5 m -> 0.87,  3 m -> 0.07 (== noise floor!).
#     The far-field array's DSP (AGC / noise-suppression / de-reverb /
#     beamforming) reshapes the audio so OpenWakeWord (near-field-trained)
#     can't recognise "alexa" at 3 m AT ALL — the mic hears you fine
#     (RMS 3109 at 3 m) but the wake features are gone. No threshold fixes
#     that: 0.07 is indistinguishable from silence.
#   XVF3800 round 2 (8/05): 0.35 proved far too eager in a BUSY lab —
#     ordinary background conversation (colleagues chatting in English)
#     scored 0.38-0.69 and triggered 8 false wakes in an afternoon, each
#     recording bystanders' chatter and shipping it to the API (privacy
#     problem in a shared lab, not just an annoyance). Genuine "alexa"
#     clusters at 0.72-1.00 near-field. 0.70 sits exactly on the measured
#     separation line: everything background stayed <=0.69.
#   Trade-off accepted: soft/marginal wakes (the 0.4-0.6 mumbles) will NOT
#     fire — say "alexa" clearly within ~2 m, or use the physical BUTTON
#     (100% reliable). If clear wakes start missing, the fix is NOT a
#     lower threshold (background overlaps there) but the deferred options:
#     XVF3800 onboard KWS or a custom-trained wake model.
#   NOTE: shout LESS, not more — OWW wants a normal speaking voice.
WAKE_WORD_THRESHOLD = 0.70
WAKE_COOLDOWN_S = 1.5
# Wake-mode captures that contain only background noise should NOT be
# uploaded — they waste an API turn and let the model hallucinate a
# response to garbage. Threshold needs to sit between "real speech"
# and "ambient noise / model TTS tail echo" peaks.
#
# Calibration history (peak RMS of wake-driven captures, real sessions):
#   round 1 (BT headset, 5/06):  real speech 8000-15000, noise <3000 → gate 4000
#   round 2 (USB mic, 5/08):     real speech  822- 3971, echo  <300 → gate 500
#   round 3 (June logs, 6/10):   gate 500 rejected REAL users at
#                                186/195/234/390 (lab members standing at
#                                normal distance — quieter than the dev
#                                sitting next to the mic). Observed false
#                                triggers in the same period: 9-135.
#                                → gate 150 sits in the measured gap.
#
# The margin is thin (135 vs 186), so the occasional noise capture WILL
# get uploaded — the cost is one API turn and a "didn't catch that"
# reply, which is far better than silently ignoring a real user. The
# OWW score gate (>=0.65) and server-side STT are additional filters.
# Every capture now logs a "[capture-rms] decision=... peak=..." line
# (see conversation._end_capture) so the next recalibration has
# both-sided data; check acceptance stats via bin/health.py.
# ENTER-mode capture stays gated by the very-permissive MIN_PEAK_RMS
# below because the user explicitly pressed the key.
WAKE_MIN_PEAK_RMS = 150.0

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
