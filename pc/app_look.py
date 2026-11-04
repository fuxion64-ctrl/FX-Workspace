"""
app_look.py
===========
The assistant's app_appearance tool (1.4.6): light, dark or follow Windows, and which appearance - the same
choices as Settings > Appearance. It changes only this app's own look, nothing outside the app, so it runs without
asking (like mini_window). The page notices through /api/state's settings_at and restyles itself.
"""
from __future__ import annotations

import time

import core.state as shared
from core import config

# What people call each appearance -> its id (the AI-inspired ones by the app they look like).
NAMES = {"classic": "classic", "sunrise": "sunrise", "terminal": "terminal", "eclipse": "eclipse", "aurora": "aurora",
         "claude": "clay", "clay": "clay", "chatgpt": "mono", "mono": "mono", "grok": "void", "void": "void",
         "gemini": "prism", "prism": "prism"}
SHOWN = {"classic": "Classic", "sunrise": "Sunrise", "terminal": "Terminal", "eclipse": "Eclipse", "aurora": "Aurora",
         "clay": "Claude", "mono": "ChatGPT", "void": "Grok", "prism": "Gemini"}
MODES = {"light": "light", "dark": "dark", "system": "system", "auto": "system", "windows": "system"}


def changed():
    """Tell the page the settings changed outside it (it reloads them on the next /api/state)."""
    shared.app_state["settings_at"] = time.time()


def apply(mode: str | None = None, appearance: str | None = None) -> dict:
    s = dict(shared.app_state["settings"])
    if mode is None and appearance is None:
        return {"ok": False, "message": "Say what to change: mode (light, dark or system) and/or appearance."}
    if mode is not None:
        if str(mode).lower() not in MODES:
            return {"ok": False, "message": "Mode is light, dark or system (follow Windows)."}
        s["color_mode"] = MODES[str(mode).lower()]
    if appearance is not None:
        key = NAMES.get(str(appearance).strip().lower())
        if not key:
            return {"ok": False, "message": f"Appearance is one of: {', '.join(SHOWN.values())}."}
        s["appearance"] = key
    s.update(config.clean_appearance(s))
    shared.app_state["settings"] = s
    config.save_settings(s)
    changed()
    mode_words = {"light": "light", "dark": "dark", "system": "following Windows (light or dark)"}[s["color_mode"]]
    return {"ok": True, "appearance": SHOWN.get(s["appearance"], s["appearance"]), "mode": s["color_mode"],
            "message": f"The app now looks {SHOWN.get(s['appearance'], s['appearance'])}, {mode_words}."}
