"""nagaki_lab — voice assistant for the Nagaki Laboratory's flow chemistry group.

Layered architecture:

  config + prompts                       constants, the system prompt
  audio/{input,output,bluetooth}, leds   hardware drivers (no business logic)
  memory                                 SQLite turn log
  tools/                                 callable tools exposed to the LLM
  live                                   Live API session wrapper
  wake                                   wake-word detector (optional)
  conversation                           orchestrator state machine
  ui_terminal                            stdin/keyboard adapter

Entry points live in ../bin/.
"""
__version__ = "0.1.0"
