"""
load_failure.py - why a local model really failed to load (1.4.6 reliability).

llama-server's router answers every failed load with the same 500, "model name=X failed to load". The page read that
as "most often it doesn't fit in the GPU's free memory" and offered Run on RAM too - but on this PC (4 Nov 2026) the
real causes were a model file that had been moved away (Gemma-4-E4B-Tools: "failed to open GGUF file ... No such
file or directory") and a 35B model that couldn't reserve 17.5 GB of system RAM (Ornith-1.5-35B, load-mode = none:
"failed to allocate CUDA_Host buffer"). Neither is helped by moving layers to RAM.

So the cause is read from evidence: the preset's files on disk, then the lines the model's own llama-server wrote
to the router's log (data/logs/llama-server.log) for its last load. Out of GPU memory is only said when the log
says so. Each kind leads with a fixed phrase the page matches (index.html ERROR_KINDS) to offer the right action.
"""
from __future__ import annotations

import os
import re

import llama_local

TAIL_BYTES = 3 * 1024 * 1024          # the end of the log is enough: the last load is what failed

# (kind, pattern) - first match wins, most specific first.
PATTERNS = (
    ("missing_file", re.compile(r"failed to open GGUF file .*(?:No such file|cannot find)|No such file or directory", re.I)),
    ("ram", re.compile(r"failed to allocate CUDA_Host buffer|unable to allocate CUDA_Host|"
                       r"ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate|"
                       r"failed to allocate buffer of size \d+ .*(?:CPU|host)|not enough (?:system )?memory", re.I)),
    ("vram", re.compile(r"cudaMalloc failed|CUDA error: out of memory|out of memory|"
                        r"failed to allocate CUDA\d*(?:_Split)? buffer|unable to allocate CUDA\d+ buffer|"
                        r"OutOfDeviceMemory|failed to allocate Vulkan\d* buffer", re.I)),
    ("unsupported", re.compile(r"unknown model architecture|unknown pre-tokenizer|invalid ggml type|"
                               r"tensor type .* (?:is )?not supported|wrong number of tensors", re.I)),
    ("bad_file", re.compile(r"invalid magic|unexpectedly reached end of file|data is not within the file bounds|"
                            r"failed to read (?:tensor|key|header)|corrupt", re.I)),
    ("mmproj", re.compile(r"clip_model_loader: failed|failed to load (?:multimodal|mmproj)|mtmd_init_from_file: .*fail", re.I)),
    ("bad_option", re.compile(r"error while handling argument|invalid argument|unknown argument|error: invalid", re.I)),
)

_ALLOC_RE = re.compile(r"buffer of size (\d+)")


def _tail(path: str) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            data = f.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def last_load_lines(model: str, path: str | None = None) -> list[str]:
    """What the model's own llama-server wrote during its LAST load, from the router's log: the lines prefixed with
    the port the router spawned it on, from that spawn to its exit. [] when the log has no load of it."""
    lines = _tail(path or llama_local.router_log_path())
    spawn = re.compile(r"spawning server instance with name=" + re.escape(model) + r" on port (\d+)")
    start, port = None, None
    for i in range(len(lines) - 1, -1, -1):
        m = spawn.search(lines[i])
        if m:
            start, port = i, m.group(1)
            break
    if start is None:
        return []
    prefix = f"[{port}]"
    out = []
    for line in lines[start + 1:]:
        if line.startswith(prefix):
            out.append(line[len(prefix):].strip())
        elif f"instance name={model} exited" in line or "spawning server instance" in line:
            break
    return out


def _clean(line: str) -> str:
    # "0.01.216.574 E llama_model_load: error loading model: ..." -> "llama_model_load: error loading model: ..."
    line = re.sub(r"^[\d.]+\s+[EWI]\s+", "", line.strip())
    return line[:220]


def diagnose(model: str, error: str = "", log_path: str | None = None) -> dict:
    """{"kind", "evidence", "path"?, "bytes"?} - kind one of missing_file, missing_mmproj, ram, vram, unsupported,
    bad_file, mmproj, bad_option, prism, unknown."""
    cfg = llama_local._config_for(model) or {}
    path, mmproj = cfg.get("model_path") or "", cfg.get("mmproj_path") or ""
    if path and not os.path.exists(path):
        return {"kind": "missing_file", "path": path, "evidence": f"{path} does not exist"}
    if mmproj and not os.path.exists(mmproj):
        return {"kind": "missing_mmproj", "path": mmproj, "evidence": f"{mmproj} does not exist"}
    if llama_local.load_hint(model):
        return {"kind": "prism", "evidence": llama_local.load_hint(model).strip()}
    lines = last_load_lines(model, log_path)
    errors = [ln for ln in lines if re.match(r"^[\d.]+\s+E\s", ln)] or lines
    for kind, pattern in PATTERNS:
        hit = next((ln for ln in errors if pattern.search(ln)), None)
        if hit is None and kind in ("vram", "ram"):
            hit = next((ln for ln in lines if pattern.search(ln)), None)
        if hit is not None:
            if kind == "missing_file":
                return {"kind": kind, "path": path, "evidence": _clean(hit)}
            out = {"kind": kind, "evidence": _clean(hit)}
            m = _ALLOC_RE.search(hit) or next((_ALLOC_RE.search(ln) for ln in errors if _ALLOC_RE.search(ln)), None)
            if m:
                out["bytes"] = int(m.group(1))
            return out
    # Our own error text can still say it (a model call that ran out of memory mid-reply, a cloud-style message).
    for kind, pattern in PATTERNS:
        if kind in ("vram", "ram") and pattern.search(error or ""):
            return {"kind": kind, "evidence": _clean(error)}
    last = next((ln for ln in reversed(errors) if "failed to load model" not in ln and "exiting due to" not in ln), "")
    return {"kind": "unknown", "evidence": _clean(last) if last else ""}


def message(model: str, diag: dict) -> str:
    """The sentence for the chat. Each kind leads with a phrase the page recognises (index.html ERROR_KINDS)."""
    kind, ev = diag.get("kind"), diag.get("evidence") or ""
    gb = f" ({diag['bytes'] / 2 ** 30:.1f} GB)" if diag.get("bytes") else ""
    if kind == "missing_file":
        return (f"{model} can't load: its model file isn't there ({diag.get('path') or 'no path in models.ini'}). It was "
                "moved, renamed or deleted - pick another model, or Scan for models in Settings › Models.")
    if kind == "missing_mmproj":
        return (f"{model} can't load: its image projector (mmproj) file isn't there ({diag.get('path')}). Pick another "
                "model, or fix its mmproj line in Settings › Models.")
    if kind == "prism":
        return f"{model} can't load on this llama.cpp build.{' ' + ev if ev else ''}"
    if kind == "ram":
        return (f"{model} can't load: not enough free system memory (RAM){gb} - llama-server couldn't reserve it. "
                "Close big programs and try again, or pick a smaller model. (llama-server: " + ev + ")")
    if kind == "vram":
        return (f"{model} can't load: not enough GPU memory for it right now. Run on RAM too puts what doesn't fit "
                "in system RAM. (llama-server: " + ev + ")")
    if kind == "unsupported":
        return (f"{model} can't load: this llama.cpp build doesn't support its format. Update llama.cpp, or pick "
                "another model. (llama-server: " + ev + ")")
    if kind == "bad_file":
        return (f"{model} can't load: its model file looks damaged or incomplete - download it again. "
                "(llama-server: " + ev + ")")
    if kind == "mmproj":
        return (f"{model} can't load: its image projector (mmproj) was refused - it may belong to another model. "
                "(llama-server: " + ev + ")")
    if kind == "bad_option":
        return (f"{model} can't load: llama-server refused an option in its models.ini section. "
                "(llama-server: " + ev + ")")
    return f"{model} couldn't load (llama-server: {ev or 'its log gives no reason'})."


def is_load_failure(error: str) -> bool:
    """The router's answer when a model didn't come up (any cause)."""
    return bool(re.search(r"failed to load|exited with status|loading model .* failed|model .* not loaded", error or "", re.I))


def explain(model: str, error) -> str | None:
    """The plain sentence for a failed load, or None when `error` isn't a load failure."""
    text = str(error or "")
    if not is_load_failure(text):
        return None
    return message(model, diagnose(model, text))
