"""Minimal sd_notify client — lets systemd supervise us with a watchdog.

Implements just enough of the sd_notify(3) protocol (a datagram to the
socket named in $NOTIFY_SOCKET) to support Type=notify + WatchdogSec in
voice-chat.service. No external dependency.

Usage:
    sdnotify.ready()      # once, after startup completes (Type=notify)
    sdnotify.watchdog()   # periodically, from the main loop, to prove liveness

If NOTIFY_SOCKET is unset (running in a terminal, not under systemd),
every call is a silent no-op, so dev runs behave identically.
"""
from __future__ import annotations

import os
import socket


def _notify(message: str) -> None:
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):           # abstract-namespace socket
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.send(message.encode())
    except OSError:
        pass


def ready() -> None:
    _notify("READY=1")


def watchdog() -> None:
    _notify("WATCHDOG=1")
