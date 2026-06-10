#!/usr/bin/env python3
"""bin/health.py — one-shot health report for the voice assistant.

Run this FIRST whenever the assistant "feels broken". It aggregates every
check that was done by hand during the 2026-06-10 incident postmortem:

    .venv/bin/python bin/health.py            # last 24h focus
    .venv/bin/python bin/health.py --days 7   # longer window

Sections: service state / temperature / event counts from the journal
(wakes, RMS accept/reject, turns, reconnects, errors) / RSS trend /
recent error lines.
"""
import argparse
import re
import subprocess
import sys
from collections import Counter


def sh(cmd: list[str], timeout: float = 15.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def journal(since: str) -> list[str]:
    out = sh(["journalctl", "--user-unit", "voice-chat.service",
              "--since", since, "--no-pager", "-o", "short-iso"], timeout=60.0)
    return out.splitlines()


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1,
                    help="journal window in days (default 1)")
    args = ap.parse_args()
    since = f"-{args.days * 24}h"

    # --- service ---
    section("service")
    status = sh(["systemctl", "--user", "status", "voice-chat.service",
                 "--no-pager"])
    for line in status.splitlines():
        s = line.strip()
        if s.startswith(("Active:", "Main PID:", "Memory:", "Tasks:")):
            print(" ", s)
    failed = "Active: active (running)" not in status
    if failed:
        print("  !! SERVICE NOT RUNNING — try: "
              "systemctl --user restart voice-chat.service")

    # --- hardware ---
    section("hardware")
    print("  temp:     ", sh(["vcgencmd", "measure_temp"]).strip())
    throttled = sh(["vcgencmd", "get_throttled"]).strip()
    print("  throttled:", throttled,
          "" if throttled.endswith("0x0") else "  !! check power/cooling")

    # --- journal stats ---
    lines = journal(since)
    section(f"events (last {args.days}d, {len(lines)} journal lines)")
    counts = Counter()
    rms_accept, rms_reject = [], []
    rss_points = []
    errors = []
    for ln in lines:
        if "[wake! " in ln:
            counts["wake fired"] += 1
        if "<<< [you" in ln:
            counts["successful turns"] += 1
        if "reconnect requested" in ln:
            counts["reconnects"] += 1
        if "[tool: " in ln:
            counts["tool calls"] += 1
        if "Watchdog timeout" in ln or "watchdog" in ln and "systemd" in ln:
            counts["systemd watchdog kills"] += 1
        if "oom" in ln.lower() or "status=9/KILL" in ln:
            counts["OOM/SIGKILL"] += 1
        m = re.search(r"\[capture-rms\] decision=(\w+).*?peak=(\d+)", ln)
        if m:
            (rms_accept if m.group(1) == "accepted" else rms_reject).append(
                int(m.group(2)))
        m = re.search(r"\[health\] rss_mb=(\d+) uptime_h=([\d.]+)", ln)
        if m:
            rss_points.append((float(m.group(2)), int(m.group(1))))
        if re.search(r"Traceback|disconnected:|upload failed|recv error|"
                     r"Failed with result", ln):
            errors.append(ln)
    for k in ("wake fired", "successful turns", "reconnects", "tool calls",
              "systemd watchdog kills", "OOM/SIGKILL"):
        print(f"  {k:<24} x{counts[k]}")

    # --- RMS calibration ---
    section("capture-rms calibration")
    total = len(rms_accept) + len(rms_reject)
    if total == 0:
        print("  (no captures in window)")
    else:
        rate = 100.0 * len(rms_accept) / total
        print(f"  accepted {len(rms_accept)}/{total} ({rate:.0f}%)")
        if rms_accept:
            print(f"  accepted peaks: min={min(rms_accept)} "
                  f"max={max(rms_accept)}")
        if rms_reject:
            print(f"  rejected peaks: min={min(rms_reject)} "
                  f"max={max(rms_reject)}"
                  + ("   !! rejects close to gate — consider lowering"
                     if max(rms_reject) > 100 else ""))

    # --- RSS trend ---
    section("memory trend ([health] lines)")
    if not rss_points:
        print("  (no [health] lines yet — they appear hourly)")
    else:
        first, last = rss_points[0], rss_points[-1]
        print(f"  first: {first[1]} MB @ uptime {first[0]:.1f}h")
        print(f"  last:  {last[1]} MB @ uptime {last[0]:.1f}h")
        dh = last[0] - first[0]
        if dh >= 1.0:
            slope = (last[1] - first[1]) / dh * 24
            print(f"  slope: {slope:+.1f} MB/day"
                  + ("   !! leaking — watch MemoryMax restarts"
                     if slope > 20 else ""))

    # --- recent errors ---
    section("recent errors (last 5)")
    if not errors:
        print("  (none)")
    for ln in errors[-5:]:
        print(" ", ln[:160])

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
