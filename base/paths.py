"""
paths.py
========
Where the big, downloaded things live - one place, so a fresh install needs no configuration.

  HOME                      <app>\\local, or the AGENTIC_HOME environment variable (e.g. a roomier drive)
  HOME\\llama.cpp            llama-server.exe (install.bat can download it); HOME\\llama.cpp\\prism: PrismML's fork
  HOME\\models               chat models (.gguf) - drop one in and press Settings > Models > Scan for models
  HOME\\Images, \\Videos,     the image / video / sound / music engines and their models (each feature's own
  \\Audio, \\Music, \\Voice    setup downloads into these; Settings can point a feature somewhere else)

Older installs kept these elsewhere (the original author's E:, A: and G: folders, or C:\\llama.cpp). Those
places are still found when they exist, after the ones above, so an existing setup keeps working unchanged.
Nothing here creates folders on import.
"""
from __future__ import annotations

import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.abspath(os.environ.get("AGENTIC_HOME") or os.path.join(APP_DIR, "local"))

# Earlier layouts, searched after HOME (only when they exist on this PC).
LEGACY_ROOTS = (r"E:\ClankerLocalCpp", r"A:\ClankerLocalCpp", r"G:\ClankerLocalCpp")


def home(*parts: str) -> str:
    return os.path.join(HOME, *parts)


def first_existing(*candidates: str, default: str | None = None) -> str:
    """The first candidate that exists, else default (else the first candidate)."""
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return default if default is not None else candidates[0]


def feature_root(name: str) -> str:
    """A feature's default folder (Images, Videos, Audio, Music, Voice): HOME\\name, or an older install's."""
    return first_existing(home(name), *(os.path.join(r, name) for r in LEGACY_ROOTS), default=home(name))


def llama_dirs() -> tuple:
    """Where llama-server.exe is looked for, stock llama.cpp first (llama_builds, run.bat, launcher.pyw)."""
    return (home("llama.cpp"), r"E:\ClankerLocalCpp", r"C:\llama.cpp")


def prism_dirs() -> tuple:
    return (home("llama.cpp", "prism"), r"E:\ClankerLocalCpp\prism")


def model_dirs(settings: dict | None = None) -> list:
    """Folders scanned for chat models, the main one first (new downloads go there): the folder chosen in
    Settings > Models (1.4.8, "models_folder") or HOME\\models, then HOME\\models when another was chosen (models
    already there keep working), then any more folders the user added ("model_dirs").
    settings None: the app's current settings (Find models' downloads used to ignore the choice)."""
    if settings is None:
        try:
            import core.state as shared
            settings = shared.app_state.get("settings") or {}
        except Exception:
            settings = {}
    main = settings.get("models_folder") if isinstance(settings.get("models_folder"), str) else ""
    extra = [d for d in (settings.get("model_dirs") or []) if isinstance(d, str) and d.strip()]
    return list(dict.fromkeys([main.strip() or home("models"), home("models"), *extra]))
