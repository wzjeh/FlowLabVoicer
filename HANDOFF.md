# Nagaki Lab Voice Assistant — Handoff (v0.1, 2026-05-01)

State at end of session: **multi-turn voice chat works**, **13 tools live**, **wake mode partially working** (detection fires but server reply unreliable — defer until investigated). All bugs from earlier in the day are fixed.

---

## Quick start

```bash
cd ~/voice && source .venv/bin/activate
python bin/chat.py                    # voice chat (push-to-talk via ENTER)
python bin/chat.py --voice Puck       # change TTS voice
python bin/chat.py --quiet            # suppress timing logs
python bin/say.py "你好"               # one-shot text→speech
python bin/list_models.py             # list available Live API models
python tests/test_live_minimal.py     # bare-SDK multi-turn diagnostic
python tests/test_tools.py            # tool unit tests
python tests/test_mic.py              # mic loopback (record + play)
```

In-chat keys / commands:

| | |
|---|---|
| ENTER | toggle: start record / send / abort response |
| `/exit` `/quit` | leave |
| `/memory` | show recent persisted turns |
| `/reset` | wipe memory.db |
| `/timers` | active timers |
| `/handle` | session_resumption handle |
| `/state` | state machine state |

---

## Architecture (layered, no upward dependencies)

```
config.py + prompts.py         constants & system prompt
   │
   ├── audio/{input,output,bluetooth}.py    hardware drivers
   ├── leds.py                              hardware drivers
   ├── tools/*.py                           LLM-callable functions
   ├── memory.py                            SQLite persistence
   ├── live.py                              Live API session + event bus
   └── wake.py                              wake word (optional)
        │
        └── conversation.py                 ConversationLoop (state machine)
             │
             └── ui_terminal.py             stdin → loop dispatcher
                  │
                  └── bin/chat.py           assembly + entry
```

State machine in `ConversationLoop`:

```
WAKE_LISTENING (only with --wake) ──── wake or ENTER
       ▼
  CAPTURING ──── ENTER ────▶ RESPONDING
       ▲                          │  turn_complete OR user abort
       └──────────────────────────┘
                  ▼
                IDLE  (or back to WAKE_LISTENING)
```

---

## All 13 tools (LLM-callable)

Located in `nagaki_lab/tools/`, registered in `tools/__init__.py`.

### timer.py (4)
- `set_timer(seconds, label)` — countdown + 880 Hz triple beep
- `list_active_timers()` — list pending
- `cancel_timer(timer_id)`
- `current_time()` — ISO + epoch

### chemistry.py (4)
- `tube_volume(inner_diameter_mm, length_mm)` — π·(d/2)²·L
- `residence_time(inner_diameter_mm=1.0, length_cm, flow_rate_mL_per_min)` — τ = V/Q
- `chemical_info(name)` — MW + formula + mp + bp from PubChem (cached in `chem_cache` SQLite table)
- `solution_prep(chemical_name, concentration, concentration_unit, final_volume_mL)` — auto-looks-up MW; supports M / mM / uM / %w/v / g/L / mg/mL

### translation.py (1)
- `translate_term(term, domain="general")` — CN ↔ EN ↔ JP via `gemini-flash-lite-latest`. Returns zh / en / ja / ja_reading / notes. Cached in `translation_cache` SQLite table. 12 s outer timeout.

### system.py (1)
- `control_volume(action, percent=10)` — wpctl on `@DEFAULT_AUDIO_SINK@`
  - `action`: `up` / `down` / `set` / `mute` / `unmute` / `get`
  - Auto-unmutes if you raise volume from a muted state

### music.py (3)
- `play_music(title=DEFAULT)` — ffplay, library at `~/voice/music/`, default `一笑江湖`, fuzzy-substring match on filename stem
- `stop_music()`
- `list_music()`

Currently in `~/voice/music/`: 一笑江湖.mp3 + 游京.mp3.

---

## LiveSession event API (for future display / logger / web UI)

`live.py` exposes a pub-sub interface. Anything can subscribe without touching ConversationLoop:

```python
live.on(LiveSession.EVENT_AUDIO_CHUNK, lambda ev: speaker.write(ev.pcm))
live.on(LiveSession.EVENT_TURN_COMPLETE, lambda ev: print("done"))
```

| Event constant | Payload | When |
|---|---|---|
| `EVENT_USER_TRANSCRIPT` | `UserTranscript(text)` | Server STT of user audio (incremental) |
| `EVENT_ASSISTANT_TRANSCRIPT` | `AssistantTranscript(text)` | Model speech transcript (incremental) |
| `EVENT_AUDIO_CHUNK` | `AudioOut(pcm)` | One chunk of model PCM (24 kHz mono S16) |
| `EVENT_TOOL_CALL` | `ToolCallEvent(function_calls)` | Model wants to call tools |
| `EVENT_TURN_COMPLETE` | `TurnComplete()` | End of one model turn |
| `EVENT_GO_AWAY` | `GoAway(time_left)` | Server about to close |
| `EVENT_RESUMPTION_UPDATE` | `ResumptionUpdate(handle)` | New session-resumption handle |
| `EVENT_RECV_ERROR` | `RecvError(exception)` | Receive-loop exception |
| `EVENT_ANY` | `(name, event)` | Fires for every event above |

Handlers may be sync or async; exceptions are caught per-handler.

---

## Robustness layers (6 in total — no path crashes the program)

1. **Tool 12s timeout** (`config.TOOL_CALL_TIMEOUT_S`): hung tool → error response to model, model can recover
2. **Server idle 25s watchdog** (`config.SERVER_IDLE_TIMEOUT_S`): no server message for 25s → force session reconnect via session_resumption (handle preserved)
3. **Upload exception caught** in `_end_capture`: 1011 keepalive / 1008 policy → flag reconnect, return to idle
4. **`on_user_action` try/except** (conversation.py): unexpected error → reconnect, don't crash UI
5. **`TerminalUI` outer try/except** (ui_terminal.py): final safety net for prompt loop
6. **Recv error → event** (`EVENT_RECV_ERROR` instead of raise): no "Task exception was never retrieved" warnings

Plus: **fresh session policy** — each program start is a clean slate, no prior memory injected. SQLite turn log only for human reference.

---

## Configuration (`nagaki_lab/config.py`)

All constants in one place. Key knobs:

| Constant | Default | What |
|---|---|---|
| `MODEL` | `gemini-2.5-flash-native-audio-latest` | Live model |
| `TRANSLATE_MODEL` | `gemini-flash-lite-latest` | translate_term backend |
| `DEFAULT_VOICE` | `Aoede` | TTS voice |
| `INPUT_RATE` | 16000 | mic sample rate |
| `OUTPUT_RATE` | 24000 | speaker sample rate |
| `SPEAKER_DEVICE` | `plughw:UACDemoV10,0` | aplay device |
| `MIN_CAPTURE_SECONDS` | 0.5 | discard captures shorter than this |
| `MIN_PEAK_RMS` | 150.0 | discard captures quieter than this |
| `MAX_CAPTURE_SECONDS` | 45.0 | hard cap on recording length |
| `TRAILING_SILENCE_S` | 1.5 | zero-PCM padding after each capture |
| `SERVER_IDLE_TIMEOUT_S` | 25.0 | watchdog threshold |
| `TOOL_CALL_TIMEOUT_S` | 12.0 | per-tool wait_for |
| `RECONNECT_BACKOFF_INITIAL_S` | 1.0 | first reconnect delay |
| `RECONNECT_BACKOFF_MAX_S` | 30.0 | reconnect backoff cap |
| `BT_HEADSET_NAME_SUBSTRING` | `Baseus` | for HFP auto-setup |
| `DEFAULT_WAKE_WORD` | `hey_jarvis` | when --wake (no value) |
| `WAKE_WORD_THRESHOLD` | 0.5 | OpenWakeWord score |

---

## Known limitations / deferred

| | |
|---|---|
| **Wake mode (`--wake`) unstable** | Detection fires (score ≥ 0.3 with `--wake-threshold 0.3`) but server often does not reply on wake-driven turns. Cause unknown — possibly MicStream + send_realtime_input interaction. ENTER mode is the stable production path. |
| **ReSpeaker HAT v2.0 mic** | Doesn't work on Trixie kernel 6.12 (codec ENXIO). Bluetooth mic is the workaround. |
| **HAT v2.0 has no LEDs** | v1.0 had APA102 LEDs; v2.0 removed them. `voice_leds.py` driver kept for future external strip. |
| **systemd auto-start** | Unit at `systemd/voice-boot.service` written but not installed. |
| **Music + voice mix** | Music and model speech mix on the same speaker. No auto-pause yet. Workaround: say "stop music" before talking. |
| **Custom "永木" wake word** | Pre-trained models include hey_jarvis / alexa / hey_mycroft / hey_marvin. Custom training (~30 min via OpenWakeWord) deferred. |
| **Session-resumption handle lifetime** | Live API tokens expire after ~10–60 min. After that the next reconnect starts a fresh conversation. |

---

## Voice-command cheat sheet

| | What you say | Model calls |
|---|---|---|
| **Identity** | "你是谁" / "用日语介绍下你自己" | (no tool) — answers "Nagaki Laboratory" / "永木研究室" |
| **Chemistry** | "0.5M NaCl 250mL 怎么配" | `solution_prep` |
| | "乙醇的沸点" | `chemical_info` |
| | "1mm 50cm 1mL/min 停留时间" | `residence_time` |
| | "4.6mm × 25cm 体积" | `tube_volume` |
| **Timer** | "5 分钟煮面定时" | `set_timer` |
| | "现在几点" | `current_time` |
| **Translation** | "抽真空 用日语怎么说" | `translate_term` |
| **Volume** | "声音大一点" / "调到 30%" / "静音" | `control_volume` |
| **Music** | "放首歌" / "放游京" / "停" | `play_music` / `stop_music` |
| **Multilingual** | mid-conversation switch CN↔JP↔EN | (handled by language policy in prompt) |
| **Refusal** | Korean / Spanish / etc. | "Sorry, this lab assistant only handles Chinese, Japanese, or English." |

---

## File reference

```
~/voice/
├── README.md                                 ← layout, run, design, troubleshooting
├── HANDOFF.md                                ← this file
├── .gemini_key                               mode 600, NEVER print
├── memory.db                                 SQLite (turns, chem_cache, translation_cache)
├── .venv/                                    Python 3.13
├── music/                                    one-time setup: drop .mp3 here
│   ├── README.txt
│   ├── 一笑江湖.mp3
│   └── 游京.mp3
│
├── nagaki_lab/                               main package
│   ├── __init__.py
│   ├── config.py                             central constants
│   ├── prompts.py                            SYSTEM_PROMPT (~1750 chars, lean)
│   ├── memory.py                             TurnLog
│   ├── live.py                               LiveSession + event bus
│   ├── conversation.py                       ConversationLoop state machine
│   ├── ui_terminal.py                        stdin reader + slash commands
│   ├── leds.py                               APA102 driver (currently unused)
│   ├── wake.py                               WakeWordDetector (OpenWakeWord)
│   ├── audio/
│   │   ├── input.py                          MicCapture (one-shot) + MicStream (continuous)
│   │   ├── output.py                         SpeakerPlayback (200 ms prebuffer)
│   │   └── bluetooth.py                      ensure_hfp() via wpctl + pw-cli
│   └── tools/
│       ├── __init__.py                       DISPATCH + get_tool_declarations()
│       ├── timer.py                          set_timer / list / cancel / current_time
│       ├── chemistry.py                      tube_volume / residence_time / chemical_info / solution_prep
│       ├── _pubchem.py                       internal REST client
│       ├── translation.py                    translate_term
│       ├── system.py                         control_volume
│       └── music.py                          play_music / stop_music / list_music
│
├── bin/                                      entry scripts (~30–110 lines each)
│   ├── chat.py                               main voice chat
│   ├── say.py                                one-shot text→speech
│   ├── boot_greeter.py                       systemd boot chime + LED
│   └── list_models.py                        list models on this API key
│
├── tests/
│   ├── test_live_minimal.py                  bare-SDK multi-turn diagnostic (saved decisive root-cause for the per-turn-receive bug)
│   ├── test_tools.py                         all tools unit smoke
│   └── test_mic.py                           BT mic loopback
│
└── systemd/
    └── voice-boot.service                    not installed yet
```

---

## Things worth doing next time (in priority order)

1. **Stress test** — script that hits every tool 5× in a row, looks for regressions
2. **Auto-pause music when model speaks** — hook EVENT_AUDIO_CHUNK to call music.pause(), EVENT_TURN_COMPLETE to resume
3. **Fix wake-mode bug** — bisect MicStream + send_realtime_input interaction
4. **Train custom 永木 wake word** — OpenWakeWord synthetic-data pipeline, ~30 min CPU
5. **Install systemd boot service** — so device just-works after power-on
6. **Add small OLED display** — subscribe to the LiveSession event bus, show transcript incrementally
7. **Reflash to Bookworm + install Seeed HAT v2.0 driver** — gets rid of BT mic dependency, real HAT mic works
