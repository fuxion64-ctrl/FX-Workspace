"""
model_scan.py
=============
Adding a chat model without editing models.ini by hand.

  - Drop a .gguf into local\\models (or a folder added in Settings > Models) and press "Scan for models" - or
    paste a .gguf's path into "Add a model file". Each new file gets its own [section] in models.ini, named
    after the file; a vision model's projector (a *mmproj* .gguf beside it) is attached as mmproj.
  - A fresh install has no models.ini: ensure_ini() makes it from models.example.ini (the [*] defaults every
    model inherits) and then scans, so run.bat / launcher.pyw start with whatever is already in local\\models.
  - New sections are APPENDED as text - comments and the user's own edits in models.ini stay as they are.
  - llama-server's router reads its presets when it starts: restart_router() restarts it (when one is running)
    so the new models show up at once.

Split models (name-00001-of-00003.gguf) are added once, by their first part, as llama.cpp loads them.
"""
from __future__ import annotations

import os
import re
import shutil
import atomic_io
import local_runtime

import llama_local
import paths

EXAMPLE = os.path.join(paths.APP_DIR, "models.example.ini")
_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
_SECTION = re.compile(r"[^A-Za-z0-9._+-]+")


def ini_path() -> str:
    return llama_local._ini_path()


@llama_local.serialized_ini
def ensure_ini() -> str:
    """models.ini exists (made from models.example.ini when missing), then local\\models is scanned."""
    path = ini_path()
    if not os.path.exists(path):
        if os.path.exists(EXAMPLE):
            shutil.copyfile(EXAMPLE, path)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write("[*]\nngl = auto\nctx-size = 16384\n")
    os.makedirs(paths.home("models"), exist_ok=True)
    try:
        scan()
    except OSError:
        pass
    return path


def _known_files() -> set:
    return {os.path.normcase(os.path.abspath(c["model_path"])) for c in llama_local.list_model_configs() if c.get("model_path")}


def _is_projector(name: str) -> bool:
    return "mmproj" in name.lower()


def _candidates(folder: str, depth: int = 2) -> list:
    """The model files under folder (and its subfolders, two levels): not projectors, split models once."""
    out = []
    base = folder.rstrip("\\/").count(os.sep)
    for root, dirs, files in os.walk(folder):
        if root.count(os.sep) - base >= depth:
            dirs[:] = []
        for n in sorted(files):
            if not n.lower().endswith(".gguf") or _is_projector(n):
                continue
            m = _SPLIT.search(n)
            if m and int(m.group(1)) != 1:
                continue
            out.append(os.path.join(root, n))
    return out


def projector_for(model_file: str) -> str | None:
    """A vision projector beside the model: the only *mmproj* .gguf in its folder, or the one sharing its name."""
    folder, stem = os.path.dirname(model_file), os.path.basename(model_file)[:-5].lower()
    try:
        projs = [n for n in os.listdir(folder) if n.lower().endswith((".gguf", ".mmproj")) and _is_projector(n)]
    except OSError:
        return None
    if len(projs) == 1:
        return os.path.join(folder, projs[0])
    base = re.split(r"[-_.](?:q\d|iq\d|f16|bf16|f32)", stem)[0]
    named = [n for n in projs if n.lower().startswith(base)] if base else []
    return os.path.join(folder, named[0]) if len(named) == 1 else None


def section_name(model_file: str, taken: set) -> str:
    name = _SPLIT.sub(".gguf", os.path.basename(model_file))[:-5]
    name = _SECTION.sub("-", name).strip("-") or "model"
    unique, n = name, 2
    while unique.lower() in taken:
        unique, n = f"{name}-{n}", n + 1
    return unique


def _own_options(model: str) -> str:
    """Options a model needs from the start: a hybrid (Qwen 3.5 - Gated DeltaNet layers) must stay whole on the GPU,
    or llama.cpp splits it and it crawls (diagnostics.keeps_whole) - so it gets ngl = 99 when it is added."""
    try:
        import diagnostics
        return "ngl = 99\n" if diagnostics.keeps_whole(diagnostics.meta_for(model)) else ""
    except Exception:
        return ""


def _append(entries: list) -> None:
    path = ini_path()
    original = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            original = f.read()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(original)
        for name, model, proj in entries:
            f.write(f"\n[{name}]\nmodel = {model}\n" + (f"mmproj = {proj}\n" if proj else "") + _own_options(model))
    atomic_io.replace(tmp, path)
    llama_local._ini_cache.update(key=None, configs=None)
    llama_local.invalidate_cache()


@llama_local.serialized_ini
def scan(settings: dict | None = None) -> dict:
    """Add every model file in the model folders that models.ini doesn't have yet."""
    known = _known_files()
    taken = {c["name"].lower() for c in llama_local.list_model_configs()}
    added, folders = [], paths.model_dirs(settings)
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for f in _candidates(folder):
            if os.path.normcase(os.path.abspath(f)) in known:
                continue
            name = section_name(f, taken)
            taken.add(name.lower())
            known.add(os.path.normcase(os.path.abspath(f)))
            added.append((name, f, projector_for(f)))
    if added:
        _append(added)
    return {"added": [{"name": n, "model": m, "mmproj": p} for n, m, p in added], "folders": folders}


@llama_local.serialized_ini
def add_file(path: str) -> dict:
    """One model by its file path (Settings > Models > Add a model file)."""
    path = os.path.abspath(os.path.expandvars(str(path or "").strip().strip('"')))
    if not path.lower().endswith(".gguf"):
        raise ValueError("That isn't a .gguf model file.")
    if _is_projector(os.path.basename(path)):
        raise ValueError("That's a vision projector (mmproj) - add the model file beside it instead.")
    if not os.path.isfile(path):
        raise ValueError("There's no file at that path.")
    if os.path.normcase(path) in _known_files():
        raise ValueError("That model is already in the list.")
    name = section_name(path, {c["name"].lower() for c in llama_local.list_model_configs()})
    proj = projector_for(path)
    _append([(name, path, proj)])
    return {"name": name, "model": path, "mmproj": proj}


def restart_router() -> bool:
    """New presets reach llama-server's router when it restarts; restarts it when it's running. True if it did."""
    with local_runtime.reserve(wait=False):
        import llama_builds
        run = llama_builds.running(force=True)
        if not run.get("pid"):
            return False
        llama_builds.switch(run.get("kind") or "latest", run)
        return True

def find_file(name: str, settings: dict | None = None, max_depth: int = 3) -> str | None:
    """A moved model file, by its name (1.4.7): the model folders, the older install folders and the folders the other
    presets' files are in, a few levels deep. None when it isn't on this PC - then the entry can be removed."""
    missing = llama_local.file_missing(name)
    if not missing or not missing.get("path"):
        return None
    want = os.path.basename(missing["path"]).lower()
    roots = list(paths.model_dirs(settings)) + [r for r in paths.LEGACY_ROOTS if os.path.isdir(r)]
    for c in llama_local.list_model_configs():
        for p in (c.get("model_path"), c.get("mmproj_path")):
            if p and os.path.isfile(p):
                roots.append(os.path.dirname(p))
    seen = set()
    for root in roots:
        root = os.path.normcase(os.path.abspath(root))
        if root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        base_depth = root.count(os.sep)
        for folder, dirs, files in os.walk(root):
            if folder.count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            for f in files:
                if f.lower() == want:
                    return os.path.join(folder, f)
    return None
