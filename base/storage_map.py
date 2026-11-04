"""
storage_map.py
==============
Settings > Storage (1.4.6, asked 4 Nov): every drive this app keeps something on, how full it is, and what of the
app's is on it - the chat models, picture / video / music models, the llama.cpp engine, what you made, the app itself.

Read-only: nothing is moved or deleted here. Each file is counted once, in the first part that claims it (chat models
by their models.ini entry first, then the feature folders, then the rest), so parts that share a folder - the models
folder next to llama-server.exe in older installs - never add up to more than the drive holds.
"""
from __future__ import annotations

import os
import shutil
import threading
import time

import paths

_cache = {"at": 0.0, "result": None}
_lock = threading.Lock()
CACHE_S = 600
MODEL_EXT = (".gguf", ".safetensors", ".bin", ".ckpt", ".onnx", ".pt", ".pth")


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _drive(p: str) -> str:
    return (os.path.splitdrive(os.path.abspath(p))[0] or os.path.abspath(p)[:1]).upper()


def _label(drive: str) -> str:
    """The volume's own name (what File Explorer shows), or "" off Windows."""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(261)
        if ctypes.windll.kernel32.GetVolumeInformationW(drive + "\\", buf, 261, None, None, None, None, 0):
            return buf.value
    except Exception:
        pass
    return ""


class _Counter:
    """Sizes files once each across all the parts."""

    def __init__(self):
        self.seen: set = set()

    def file(self, p: str) -> int:
        k = _norm(p)
        if k in self.seen:
            return -1
        try:
            n = os.path.getsize(p)
        except OSError:
            return -1
        self.seen.add(k)
        return n

    def tree(self, root: str, skip=()) -> tuple:
        """(bytes, files) under root, minus the folders in skip and files already counted."""
        total = count = 0
        skip = {_norm(s) for s in skip}
        if not root or not os.path.isdir(root):
            return 0, 0
        for folder, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if _norm(os.path.join(folder, d)) not in skip]
            for name in names:
                n = self.file(os.path.join(folder, name))
                if n >= 0:
                    total += n
                    count += 1
        return total, count


def _settings() -> dict:
    try:
        import core.state as shared
        return shared.app_state.get("settings") or {}
    except Exception:
        return {}


def _feature_roots() -> list:
    """(id, name, what, folder) for the feature engines and their models - each from its own setting."""
    out = []
    try:
        import image_gen
        out.append(("pictures", "Picture models", "stable-diffusion.cpp and the image models", image_gen._cfg()["image_root"]))
    except Exception:
        pass
    try:
        import video_gen
        out.append(("videos", "Video models", "the video engine and Wan models", video_gen._cfg()["video_root"]))
    except Exception:
        pass
    try:
        import music_gen
        out.append(("music", "Music engine", "ACE-Step and its models", music_gen._cfg()["music_root"]))
    except Exception:
        pass
    for name in ("Audio", "Voice"):
        out.append((name.lower(), f"{name} engine", f"the {name.lower()} models it downloaded", paths.feature_root(name)))
    return out


def build() -> dict:
    """The whole map (on the caller's thread; it walks folders, so call it through snapshot())."""
    c, parts = _Counter(), []
    data = os.path.join(paths.APP_DIR, "data")

    def add(pid, name, what, bytes_, files, where, items=None):
        if bytes_ <= 0:
            return
        parts.append({"id": pid, "name": name, "what": what, "bytes": bytes_, "files": files,
                      "path": where, "drive": _drive(where), **({"items": items} if items else {})})

    # 1. Chat models: what models.ini points at, model and vision file, one part per drive.
    by_drive: dict = {}
    model_dirs = set()
    try:
        import llama_local
        configs = llama_local.list_model_configs()
    except Exception:
        configs = []
    for cfg in configs:
        size = 0
        for p in (cfg.get("model_path"), cfg.get("mmproj_path")):
            if p and os.path.isfile(p):
                n = c.file(p)
                size += max(0, n)
                model_dirs.add(os.path.dirname(os.path.abspath(p)))
        if size:
            d = _drive(cfg["model_path"])
            row = by_drive.setdefault(d, {"bytes": 0, "items": [], "dir": os.path.dirname(os.path.abspath(cfg["model_path"]))})
            row["bytes"] += size
            row["items"].append({"name": cfg.get("name") or os.path.basename(cfg["model_path"]), "bytes": size})
    for d, row in by_drive.items():
        row["items"].sort(key=lambda i: -i["bytes"])
        add(f"chat-{d}", "Chat models", f"{len(row['items'])} in models.ini", row["bytes"], len(row["items"]), row["dir"], row["items"])
    # 2. Model files in the same folders that models.ini doesn't use - often old downloads you can remove.
    for folder in sorted(model_dirs | {os.path.abspath(p) for p in paths.model_dirs(_settings())}):
        if not os.path.isdir(folder):
            continue
        size = files = 0
        items = []
        for name in os.listdir(folder):
            p = os.path.join(folder, name)
            if os.path.isfile(p) and name.lower().endswith(MODEL_EXT):
                n = c.file(p)
                if n > 0:
                    size += n
                    files += 1
                    items.append({"name": name, "bytes": n})
        items.sort(key=lambda i: -i["bytes"])
        add(f"unused-{_norm(folder)}", "Model files not in models.ini", "in a models folder, but no model uses them", size, files, folder, items[:12])
    # 3. The feature engines and their models (each in its own folder).
    feature_dirs = []
    for pid, name, what, root in _feature_roots():
        if root and os.path.isdir(root) and not _norm(root).startswith(_norm(data)):
            b, f = c.tree(root)
            add(pid, name, what, b, f, root)
            feature_dirs.append(root)
    # 4. The llama.cpp engine: its own files (the models next to it were counted above).
    for d in dict.fromkeys(os.path.abspath(x) for x in (*paths.llama_dirs(), *paths.prism_dirs())):
        if not os.path.isfile(os.path.join(d, "llama-server.exe")):
            continue
        size = files = 0
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.isfile(p) and not name.lower().endswith(MODEL_EXT):
                n = c.file(p)
                if n > 0:
                    size += n
                    files += 1
        add(f"llama-{_norm(d)}", "llama.cpp engine", "llama-server and its libraries" + (" (PrismML's build)" if "prism" in d.lower() else ""), size, files, d)
    # 5. What you made and what the app keeps (data\).
    for pid, name, what, sub in (("made-pictures", "Pictures you made", "and their trash", "images"),
                                 ("made-videos", "Videos you made", "", "videos"), ("made-music", "Songs you made", "", "music"),
                                 ("voice-models", "Voice models", "listening and speaking", "voice")):
        b, f = c.tree(os.path.join(data, sub))
        add(pid, name, what, b, f, os.path.join(data, sub))
    b, f = c.tree(data)
    add("app-data", "Chats, memory and the rest", "everything else in data\\", b, f, data)
    # 6. The app itself, and its Python packages.
    venv = os.path.join(paths.APP_DIR, ".venv")
    b, f = c.tree(venv)
    add("app-python", "The app's Python", "its packages (.venv)", b, f, venv)
    b, f = c.tree(paths.APP_DIR, skip=(data, venv, *feature_dirs))
    add("app", "The app", "its own files", b, f, paths.APP_DIR)

    drives = []
    for d in sorted({p["drive"] for p in parts}):
        try:
            u = shutil.disk_usage(d + "\\")
        except OSError:
            continue
        mine = sorted((p for p in parts if p["drive"] == d), key=lambda p: -p["bytes"])
        app_bytes = sum(p["bytes"] for p in mine)
        drives.append({"drive": d, "label": _label(d), "total": u.total, "free": u.free, "used": u.total - u.free,
                       "app": app_bytes, "other": max(0, u.total - u.free - app_bytes), "parts": mine})
    return {"ok": True, "at": time.time(), "drives": drives}


_state = {"measuring": False}


def _measure():
    try:
        res = build()
        with _lock:
            _cache.update(at=time.time(), result=res)
    finally:
        _state["measuring"] = False


def snapshot(force: bool = False) -> dict:
    """The last map at once, measured again in the background when it is older than CACHE_S (or on force).
    Walking the engines' folders takes time - 36 s on this PC for ~95,000 files in the audio and music engines -
    so the page never waits for it: it shows the last map and "measuring" until the new one is in."""
    with _lock:
        res, age = _cache["result"], time.time() - _cache["at"]
        stale = force or not res or age > CACHE_S
        if stale and not _state["measuring"]:
            _state["measuring"] = True
            threading.Thread(target=_measure, daemon=True, name="storage-map").start()
    return {**(res or {"ok": True, "drives": [], "at": 0}), "measuring": _state["measuring"]}
