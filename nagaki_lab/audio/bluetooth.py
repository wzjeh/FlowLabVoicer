"""Force a paired BT headset into the HFP (mic-enabled) profile.

Why we need this: on Pi OS Trixie, PipeWire frequently leaves a bluez5 device
at `bluez5.profile = "off"` even when bluez itself reports "Connected: yes".
A Source node still appears in `wpctl status`, but it is a stub that delivers
near-silence. `wpctl set-profile` does NOT work for SPA bluez profiles (the
real profile indices are huge, e.g. 196864), so we drive `pw-cli` directly.

The function is idempotent: if HFP is already active we no-op.
"""
from __future__ import annotations

import re
import subprocess
import time
from typing import Optional

from .. import config

# Profiles that expose a microphone, in preference order. mSBC (16 kHz) gives
# better quality than CVSD (8 kHz), but Baseus Eli 2i Fit only supports CVSD.
HFP_PROFILE_PREFS = (
    "headset-head-unit-msbc",
    "headset-head-unit-cvsd",
    "headset-head-unit",
)


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "") + (out.stderr or "")
    except Exception:
        return ""


def find_bt_device_id(name_substring: str) -> Optional[int]:
    """First WirePlumber device id whose name contains `name_substring` and
    is a bluez5 device. Returns None if not connected."""
    text = _run(["wpctl", "status"])
    in_devices = False
    for line in text.splitlines():
        if line.startswith("Audio"):
            in_devices = False
        if "Devices:" in line:
            in_devices = True
            continue
        if in_devices and "bluez5" in line and name_substring in line:
            m = re.search(r"(\d+)\.\s+", line)
            if m:
                return int(m.group(1))
    return None


def list_profiles(device_id: int) -> list[tuple[int, str]]:
    """Parse pw-cli enum-params output -> [(index, name), ...]."""
    text = _run(["pw-cli", "enum-params", str(device_id), "EnumProfile"])
    profiles: list[tuple[int, str]] = []
    cur_idx: Optional[int] = None
    in_index = False
    in_name = False
    for line in text.splitlines():
        s = line.strip()
        if "Profile:index" in s:
            in_index, in_name = True, False
            continue
        if "Profile:name" in s:
            in_index, in_name = False, True
            continue
        if in_index and s.startswith("Int "):
            try:
                cur_idx = int(s.split()[-1])
            except ValueError:
                cur_idx = None
            in_index = False
            continue
        if in_name and s.startswith("String "):
            name = s[len("String "):].strip().strip('"')
            if cur_idx is not None and name:
                profiles.append((cur_idx, name))
            cur_idx = None
            in_name = False
    return profiles


def choose_hfp_index(profiles: list[tuple[int, str]]) -> Optional[tuple[int, str]]:
    by_name = {name: idx for idx, name in profiles}
    for pref in HFP_PROFILE_PREFS:
        if pref in by_name:
            return by_name[pref], pref
    return None


def has_active_hfp_source(name_substring: str) -> bool:
    """When HFP is active, wpctl status lists a real Source named after the
    headset. When inactive, only stub Filter-class nodes exist."""
    text = _run(["wpctl", "status"])
    in_sources = False
    for line in text.splitlines():
        s = line.rstrip()
        if "Sources:" in s and "Filters" not in s:
            in_sources = True
            continue
        if "Filters:" in s or "Streams:" in s or s.startswith("Video"):
            in_sources = False
            continue
        if in_sources and name_substring in s:
            return True
    return False


def _set_profile(device_id: int, profile_index: int) -> bool:
    pod = f"{{ index: {profile_index}, save: true }}"
    out = _run(["pw-cli", "set-param", str(device_id), "Profile", pod])
    return "Object:" in out or "type Spa:Pod" in out


def ensure_hfp(
    name_substring: str = config.BT_HEADSET_NAME_SUBSTRING,
    verbose: bool = True,
) -> bool:
    """Ensure the named BT headset is in an HFP profile so its mic delivers
    real audio. Returns True if HFP is active afterwards."""
    if has_active_hfp_source(name_substring):
        if verbose:
            print(f"[bt] HFP source already active for {name_substring!r}")
        return True

    dev_id = find_bt_device_id(name_substring)
    if dev_id is None:
        if verbose:
            print(f"[bt] no bluez5 device matching {name_substring!r} — skipping HFP setup")
        return False

    profiles = list_profiles(dev_id)
    chosen = choose_hfp_index(profiles)
    if chosen is None:
        if verbose:
            avail = ", ".join(name for _, name in profiles)
            print(f"[bt] device {dev_id} has no HFP profile available. Profiles: {avail}")
        return False

    idx, name = chosen
    if verbose:
        print(f"[bt] setting device {dev_id} -> profile {name!r} (index {idx})")
    _set_profile(dev_id, idx)

    for _ in range(10):
        time.sleep(0.3)
        if has_active_hfp_source(name_substring):
            if verbose:
                print("[bt] HFP active.")
            return True

    if verbose:
        print("[bt] profile request sent but HFP source not yet visible.")
    return False
