# Nagaki Lab Voice Assistant

Voice assistant for the **Nagaki Laboratory's flow chemistry group**, running on a Raspberry Pi 5 at the lab bench. Speaks Chinese / Japanese / English, calls flow-chemistry tools (residence time, tube volume, MW lookup, solution prep, multilingual term translation, timers), all via Gemini Live API.

## Hardware

| Item | Role |
|---|---|
| Raspberry Pi 5 (2 GB) | main board |
| ReSpeaker 2-Mics Pi HAT v2.0 | (currently unused — driver problem on Trixie kernel 6.12; mic capture via Bluetooth instead) |
| Bluetooth headset (Baseus Eli 2i Fit) | mic input via HFP profile |
| USB Speaker (Jieli UACDemoV1.0) | audio output |

## Project layout

```
voice/
├── README.md
├── .gemini_key                              # API key, mode 600 (NEVER print)
├── memory.db                                # SQLite turn log + tool caches
├── .venv/                                   # python 3.13 venv
│
├── nagaki_lab/                              # the package
│   ├── config.py                            # all constants
│   ├── prompts.py                           # SYSTEM_PROMPT
│   │
│   ├── audio/
│   │   ├── input.py                         # MicCapture (sounddevice)
│   │   ├── output.py                        # SpeakerPlayback (aplay)
│   │   └── bluetooth.py                     # ensure_hfp() (PipeWire)
│   ├── leds.py                              # APA102 driver + LEDStatus
│   │
│   ├── live.py                              # LiveSession wrapper (typed events)
│   ├── memory.py                            # TurnLog
│   ├── wake.py                              # WakeWordDetector (OpenWakeWord)
│   │
│   ├── tools/
│   │   ├── timer.py                         # set_timer, list_active_timers, ...
│   │   ├── chemistry.py                     # tube_volume, residence_time, ...
│   │   ├── translation.py                   # translate_term
│   │   └── _pubchem.py                      # internal REST client
│   │
│   ├── conversation.py                      # ConversationLoop (state machine)
│   └── ui_terminal.py                       # TerminalUI (stdin → loop)
│
├── bin/                                     # entry-point scripts
│   ├── chat.py                              # voice conversation
│   ├── say.py                               # one-shot text → speech
│   ├── boot_greeter.py                      # systemd: chime + LED idle
│   └── list_models.py                       # utility
│
├── tests/
│   ├── test_live_minimal.py                 # bare-SDK multi-turn diagnostic
│   ├── test_tools.py
│   └── test_mic.py
│
└── systemd/
    ├── voice-boot.service                    # boot-time chime + LED greeter
    └── voice-chat.service                    # auto-start chat in wake mode
```

## Physical pushbutton (Sanwa OBSF-30-K or any momentary switch)

A pushbutton wired to GPIO 17 (header pin 11) and GND (pin 9) gives a
hands-free, gloves-friendly trigger that's identical to pressing ENTER
in the terminal. Internal pull-up; no resistor needed.

```
   one terminal   -> GPIO 17  (header pin 11)
   other terminal -> GND       (header pin 9 or any GND)
```

Test the wiring before launching chat:

```bash
.venv/bin/python tests/test_button.py    # press button — terminal prints each press
```

## Boot-time greeter + chat auto-start (systemd, user-mode)

Two **user-mode** units together:

| Unit | Type | Purpose |
|---|---|---|
| `voice-boot.service` | oneshot | Triple-beep + LED idle on boot — confirms device alive |
| `voice-chat.service` | simple   | Long-running `bin/chat.py --wake alexa`, auto-restart on crash |

User-mode because PipeWire / mic / speaker live in the user session.
Install once:

```bash
# 1. enable user systemd at boot regardless of console / SSH login
sudo loginctl enable-linger $USER

# 2. install both units
mkdir -p ~/.config/systemd/user
cp ~/voice/systemd/voice-boot.service ~/.config/systemd/user/
cp ~/voice/systemd/voice-chat.service ~/.config/systemd/user/
systemctl --user daemon-reload

# 3. enable on boot (greeter chimes; chat then takes over)
systemctl --user enable --now voice-boot.service
systemctl --user enable --now voice-chat.service

# 4. verify
systemctl --user status voice-boot.service voice-chat.service
```

Run-time controls:
```bash
systemctl --user disable voice-chat.service        # stop auto-start
systemctl --user restart voice-chat.service        # restart now
journalctl --user-unit voice-chat.service -fb      # tail this-boot logs
journalctl --user-unit voice-chat.service -b -n50  # last 50 from this boot
```

In daemon (no-TTY) mode, ENTER is unreachable; trigger conversation via
the wake word ("alexa, ...") or the GPIO pushbutton instead.

## Maintenance / self-healing

Three layers keep the device alive unattended (added after the
2026-06-09 incident, where an untimeouted websocket connect hung the
process for 20 hours until the kernel OOM-killed it):

1. **I/O timeouts** — every Live-API connect (30 s), close (10 s) and
   send (10 s) is time-bounded (`config.CONNECT_TIMEOUT_S` etc.), so the
   reconnect loop always makes progress. Receive-side stalls are covered
   by the existing 25 s server-idle watchdog.
2. **systemd watchdog** — `Type=notify` + `WatchdogSec=300`. The main
   loop pings WATCHDOG=1 every 30 s (`nagaki_lab/sdnotify.py`); any hang
   class we didn't foresee gets killed + restarted within 5 minutes.
3. **MemoryMax=800M** — contains the known slow leak (~12 MB/day);
   hitting the cap restarts only this unit instead of inviting the
   kernel's global OOM killer.

When the assistant "feels broken", run the health report first:

```bash
.venv/bin/python bin/health.py            # last 24 h
.venv/bin/python bin/health.py --days 7   # weekly review
```

It aggregates service state, temperature, event counts (wakes, RMS
accept/reject, turns, reconnects), the hourly `[health]` RSS trend, and
the most recent error lines.

Monthly: glance at the health output for (a) RSS slope — if MemoryMax
restarts become frequent, root-cause the leak; (b) `capture-rms`
acceptance rate — recalibrate `WAKE_MIN_PEAK_RMS` if real speech is
being rejected (calibration history lives in `config.py`); (c) reconnect
count — if 1011 upload failures eat questions often, revisit the
deferred upload-retry design.

## How to run

```bash
cd ~/voice
source .venv/bin/activate

# voice conversation (default)
python bin/chat.py

# voice conversation with wake word (default: alexa)
python bin/chat.py --wake alexa

# pick a different TTS voice
python bin/chat.py --voice Puck

# one-shot say
python bin/say.py "你好，介绍一下你自己"

# list available Live API models
python bin/list_models.py
```

### Slash commands during chat

| | |
|---|---|
| (ENTER) | toggle: start record / send recording / abort response |
| `/exit` | quit |
| `/memory` | show recent persisted turns |
| `/reset` | wipe memory.db |
| `/timers` | active timers |
| `/handle` | current Live session resumption handle |
| `/state` | current state machine state |

### Conversation state machine

```
WAKE_LISTENING (only with --wake)
       ▼  wake or ENTER
  CAPTURING ──── ENTER ────▶ RESPONDING
       ▲                          │  turn_complete OR user_abort (ENTER)
       └──────────────────────────┘
                  ▼
                IDLE  (or back to WAKE_LISTENING)
```

## Adding a display (or any other consumer of live events)

`LiveSession` is event-driven. Anything that wants to observe the conversation
just subscribes — it does not need to know about, or be known by, the
`ConversationLoop`. Example: a future small OLED display.

```python
from nagaki_lab.live import LiveSession

# `live` is the LiveSession instance built in bin/chat.py.
def show_partial(ev): display.append_text(ev.text)
def show_done(ev):    display.flush_line()
def show_speaking(ev): display.set_indicator("🔊")

live.on(LiveSession.EVENT_ASSISTANT_TRANSCRIPT, show_partial)
live.on(LiveSession.EVENT_TURN_COMPLETE,        show_done)
live.on(LiveSession.EVENT_AUDIO_CHUNK,          show_speaking)
```

Available events (`LiveSession.EVENT_*`):

| Constant | Payload | When |
|---|---|---|
| `EVENT_USER_TRANSCRIPT` | `UserTranscript(text)` | Server-side STT of user audio (incremental) |
| `EVENT_ASSISTANT_TRANSCRIPT` | `AssistantTranscript(text)` | Model speech transcript (incremental) |
| `EVENT_AUDIO_CHUNK` | `AudioOut(pcm)` | One chunk of model audio to play |
| `EVENT_TOOL_CALL` | `ToolCallEvent(function_calls)` | Model wants to call tools |
| `EVENT_TURN_COMPLETE` | `TurnComplete()` | End of one model turn |
| `EVENT_GO_AWAY` | `GoAway(time_left)` | Server about to close |
| `EVENT_RESUMPTION_UPDATE` | `ResumptionUpdate(handle)` | New session-resumption handle |
| `EVENT_RECV_ERROR` | `RecvError(exception)` | Receive-loop exception |
| `EVENT_ANY` | `(name, event)` | Fires for every event above (for cross-cutting concerns) |

Handlers may be sync `def f(event)` or async `async def f(event)`. Exceptions
in a handler are caught and logged so one bad handler can't break the loop.

## Design principles

- **Layered**: hardware drivers know nothing of business logic; orchestrator knows nothing of stdin.
- **Configuration centralised** in `config.py`. No magic numbers in business code.
- **Each layer has a single responsibility**:
  - `audio/*` = hardware I/O
  - `tools/*` = pure business functions, callable by LLM
  - `live.py` = SDK protocol wrapper, emits typed events
  - `conversation.py` = orchestrator, state machine
  - `ui_terminal.py` = user interface adapter
  - `bin/*.py` = thin entry points, just wire dependencies
- **Fresh-session policy**: each program start is a clean conversation. Live API session preserves context within one run; `memory.db` records turns for the user's reference (not injected into model context).

## Known traps

See also long-term memory in `~/.claude/projects/.../memory/feedback_voice_traps.md`. Highlights:
1. ReSpeaker HAT v2.0 has no APA102 LEDs (v1.0 did). External strip required if LED status is wanted.
2. ReSpeaker HAT v2.0 mic doesn't work on Trixie kernel 6.12; would need Bookworm kernel 6.6.
3. Pi OS Trixie's PipeWire often leaves a connected BT headset at `bluez5.profile = "off"`. `wpctl set-profile` does NOT work; we use `pw-cli set-param`. Handled automatically by `audio/bluetooth.py`.
4. Baseus Eli 2i Fit HFP only supports CVSD (8 kHz), not mSBC (16 kHz). PipeWire upsamples; speech still works.
5. Gemini API key must NEVER appear in chat or logs; read via `config.read_api_key()`.
6. PEP 668: never `pip install` against system python — always venv.

## Diagnostics

When voice chat misbehaves, run the bare-SDK multi-turn test:

```bash
python tests/test_live_minimal.py
```

This sends the same 4-second clip three times in a row with ZERO orchestration code on top, and reports whether each turn got a response. Decisively answers whether a multi-turn bug is in the SDK / model / send pattern, or in our orchestration.
