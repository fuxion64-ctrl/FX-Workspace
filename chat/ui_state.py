"""
ui_state.py
===========
Where the page keeps its look-and-feel choices (1.4.8): theme mode, text size, sidebar, pinned items... They used
to live in the browser's localStorage, so a phone and the PC (or a reinstalled browser) each started from nothing.
Now the server holds them in data/ui_state.json:

    {"install": {...}, "pc": {...}, "phone": {...}}

`install` keys are shared by every device (welcome / tour done, last seen version); the rest belong to a profile -
the PC and the phone keep separate layouts because their screens differ. Only known keys are stored, each value
small, so a buggy page can't fill the file.
"""
from __future__ import annotations

import json
import os
import threading

import atomic_io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "ui_state.json")   # module-level so tests can patch it

INSTALL_KEYS = ("welcomed", "toured", "seenVersion")
PROFILE_KEYS = ("mode", "textScale", "motion", "contrast", "sidebarW", "sidebarCollapsed", "sbCollapsed",
                "sections", "pinned", "tools", "preset", "researchDepth", "compareModel", "animateReplies",
                "notify", "notifySound", "panelOpen2", "zoom")
PROFILES = ("pc", "phone")
MAX_VALUE_BYTES = 4096

_lock = threading.Lock()


def _load() -> dict:
    """The file's three sections; anything missing, unreadable or the wrong shape reads as empty."""
    try:
        with open(PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        raw = None
    if not isinstance(raw, dict):
        raw = {}
    return {name: (raw[name] if isinstance(raw.get(name), dict) else {}) for name in ("install", *PROFILES)}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    atomic_io.replace(tmp, PATH)   # a crash mid-write can't leave half a file


def _view(data: dict, profile: str) -> dict:
    return {**data["install"], **data[profile]}


def _ok_value(value) -> bool:
    try:
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8")) <= MAX_VALUE_BYTES
    except (TypeError, ValueError):
        return False


def get(profile: str) -> dict:
    profile = profile if profile in PROFILES else "pc"
    with _lock:
        return _view(_load(), profile)


def update(profile: str, changes: dict) -> dict:
    """Merges the known, small keys of `changes`; null removes a key. Returns the caller's merged state."""
    profile = profile if profile in PROFILES else "pc"
    with _lock:
        data = _load()
        touched = False
        for key, value in changes.items():
            if key in INSTALL_KEYS:
                section = data["install"]
            elif key in PROFILE_KEYS:
                section = data[profile]
            else:
                continue
            if value is None:
                touched = section.pop(key, None) is not None or touched
            elif _ok_value(value):
                section[key] = value
                touched = True
        if touched:
            _save(data)
        return _view(data, profile)
