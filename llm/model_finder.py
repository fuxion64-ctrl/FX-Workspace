"""
model_finder.py
===============
Settings > Models > Find models: search Hugging Face for llama.cpp models (.gguf), see which files fit THIS PC,
and download one into the models folder - the idea of Odysseus' Cookbook / hwfit, done the FX way.

  search(query)   Hugging Face's model search, GGUF repos only, most downloaded first.
  files(repo)     the repo's .gguf files (split files grouped), each with its size, quant and how it fits:
                    "gpu"      weights + a working context fit in the GPU's free memory      -> full speed
                    "partial"  they fit in GPU + free RAM: part runs from RAM                 -> slower
                    "no"       they don't fit                                                 -> not offered
                  and the one to pick ("best": the largest Q4+ file that fits on the GPU; else the best partial).
  download(repo, path)
                  a background download through the network layer (Settings > Network's route and kill switch):
                  resumable (.part + HTTP Range), every part of a split model, checked against the SHA-256 the Hub
                  publishes, then added to models.ini (model_scan.add_file) - which also attaches a vision
                  projector (mmproj) beside it. status() follows it; cancel() stops it.

The fit is an estimate from file sizes (the header isn't downloaded yet): weights + KV cache and compute for an
8K context, ~ (0.10 GB per billion parameters + 0.6 GB). Once a model is on disk, Settings > Models' context
helpers (diagnostics.ctx_limits) measure it from its real header.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from urllib.parse import quote

import httpx

import fast_download
import netlayer
import paths

HF = "https://huggingface.co"
GB = 1024 ** 3
CTX_PLAN = 8192
# Bits per weight of llama.cpp's quants (incl. their scales) - used to read a file's parameter count from its size.
QUANT_BITS = {"F32": 32, "BF16": 16, "F16": 16, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q5_0": 5.5,
              "Q4_K_M": 4.85, "Q4_K_S": 4.6, "Q4_0": 4.5, "IQ4_XS": 4.25, "IQ4_NL": 4.5, "Q3_K_L": 4.3, "Q3_K_M": 3.9,
              "Q3_K_S": 3.5, "IQ3_M": 3.7, "IQ3_XS": 3.3, "Q2_K": 3.35, "IQ2_M": 2.7, "IQ2_XS": 2.4, "IQ1_M": 1.75,
              "MXFP4": 4.25, "TQ1_0": 1.7, "TQ2_0": 2.1}
_QUANT_RX = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted((re.escape(q) for q in QUANT_BITS), key=len, reverse=True))
                       + r"|UD-[A-Z0-9_]+)(?![A-Za-z0-9])", re.I)
_SPLIT_RX = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
_PARAMS_RX = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-z])")
_ACTIVE_RX = re.compile(r"[-_]A(\d+(?:\.\d+)?)B", re.I)

_lock = threading.Lock()
_state: dict = {"state": "idle", "repo": "", "file": "", "done": 0, "total": 0, "error": "", "added": None}
_cancel = threading.Event()
_paused = threading.Event()          # set with _cancel by pause(): the stop ends as "paused", not "cancelled"
_cache: dict = {}            # url -> (at, value): search and file lists for 10 minutes


def _get(url: str, ttl: float = 600.0):
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    with netlayer.client("other", timeout=httpx.Timeout(20.0, connect=10.0), follow_redirects=True, private=True) as c:
        r = c.get(url)
    if r.status_code == 404:
        raise ValueError("Hugging Face doesn't have that model (or it's private).")
    if r.status_code >= 400:
        raise RuntimeError(f"Hugging Face answered {r.status_code}.")
    value = r.json()
    _cache[url] = (time.time(), value)
    return value


# ---- this PC -------------------------------------------------------------------------------------------------
def hardware() -> dict:
    """{"gpu", "vram_gb", "free_vram_gb", "ram_gb", "free_ram_gb"} - free VRAM leaves the desktop's own share."""
    import diagnostics
    import system_metrics
    gpu = {}
    try:
        gpu = diagnostics.gpu_info() or {}
    except Exception:
        pass
    snap = {}
    try:
        snap = system_metrics.snapshot() or {}
    except Exception:
        pass
    ram_total = float(snap.get("ram_total_gb") or 0)
    ram_used = float(snap.get("ram_used_gb") or 0)
    usable = gpu.get("plan_usable_mb") or gpu.get("usable_mb") or 0
    return {"gpu": gpu.get("name") or "", "vram_gb": round((gpu.get("total_mb") or 0) / 1024, 1),
            "free_vram_gb": round(max(0, usable) / 1024, 1), "ram_gb": round(ram_total, 1),
            "free_ram_gb": round(max(0.0, ram_total - ram_used) * 0.8, 1)}


# ---- reading a repo ------------------------------------------------------------------------------------------
def quant_of(name: str) -> str:
    m = _QUANT_RX.search(name.replace(".gguf", ""))
    return m.group(1).upper() if m else ""


def params_b(repo: str, size_bytes: int, quant: str) -> float | None:
    """Parameters in billions: from the name ("Qwen3-8B"), else from size and quant."""
    m = _PARAMS_RX.findall(repo.split("/")[-1].replace("_", "-"))
    if m:
        return float(m[-1]) if not _ACTIVE_RX.search(repo) else max(float(x) for x in m)
    bits = QUANT_BITS.get(quant)
    return round(size_bytes * 8 / bits / 1e9, 1) if bits else None


def fit(size_bytes: int, params: float | None, hw: dict) -> dict:
    need = size_bytes / GB + (params or size_bytes / GB) * 0.10 + 0.6
    free_vram, free_ram = hw.get("free_vram_gb") or 0, hw.get("free_ram_gb") or 0
    level = "gpu" if free_vram and need <= free_vram else "partial" if need <= free_vram + free_ram else "no"
    return {"need_gb": round(need, 1), "fit": level}


# The first-run welcome's one-click starter (1.4.3): Qwen's own GGUF builds, biggest first. The sizes are the files'
# real sizes, so the pick needs no network - nothing goes to Hugging Face until the download is clicked.
STARTERS = (
    {"name": "Qwen3 8B", "repo": "Qwen/Qwen3-8B-GGUF", "path": "Qwen3-8B-Q4_K_M.gguf", "size": 5_027_783_488, "params_b": 8.2},
    {"name": "Qwen3 4B", "repo": "Qwen/Qwen3-4B-GGUF", "path": "Qwen3-4B-Q4_K_M.gguf", "size": 2_497_280_256, "params_b": 4.0},
    {"name": "Qwen3 1.7B", "repo": "Qwen/Qwen3-1.7B-GGUF", "path": "Qwen3-1.7B-Q8_0.gguf", "size": 1_834_426_016, "params_b": 1.7},
)


def starter(hw: dict | None = None) -> dict:
    """The starter that suits this PC: the biggest that fits on the GPU, else the smallest that runs partly from
    RAM, else none. {"hardware", "pick": {name, repo, path, gb, need_gb, fit} | None}."""
    hw = hw if hw is not None else hardware()
    rows = [{**s, "gb": round(s["size"] / GB, 1), **fit(s["size"], s["params_b"], hw)} for s in STARTERS]
    on_gpu = [r for r in rows if r["fit"] == "gpu"]
    partial = [r for r in rows if r["fit"] == "partial"]
    pick = on_gpu[0] if on_gpu else partial[-1] if partial else None
    return {"hardware": hw, "pick": pick, "have": bool(_on_disk_names())}


def search(query: str = "", limit: int = 30) -> list[dict]:
    q = (query or "").strip()[:80]
    url = (f"{HF}/api/models?filter=gguf&sort=downloads&direction=-1&limit={int(limit)}"
           + (f"&search={quote(q)}" if q else ""))
    rows = _get(url)
    out = []
    for r in rows if isinstance(rows, list) else []:
        rid = r.get("id") or r.get("modelId")
        if not rid:
            continue
        tags = r.get("tags") or []
        out.append({"repo": rid, "downloads": r.get("downloads") or 0, "likes": r.get("likes") or 0,
                    "updated": r.get("lastModified") or r.get("createdAt") or "",
                    "vision": any(t in ("image-text-to-text", "vision", "multimodal") for t in tags) or r.get("pipeline_tag") == "image-text-to-text",
                    "params_b": params_b(rid, 0, "")})
    return out


def files(repo: str) -> dict:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""):
        raise ValueError("A Hugging Face model is named like owner/name.")
    tree = _get(f"{HF}/api/models/{repo}/tree/main?recursive=true")
    hw = hardware()
    groups: dict[str, dict] = {}
    mmproj = []
    for f in tree if isinstance(tree, list) else []:
        p = f.get("path") or ""
        if f.get("type") != "file" or not p.lower().endswith(".gguf"):
            continue
        size = int((f.get("lfs") or {}).get("size") or f.get("size") or 0)
        sha = (f.get("lfs") or {}).get("oid") or ""
        if "mmproj" in p.lower():
            mmproj.append({"path": p, "size": size, "sha256": sha})
            continue
        key = _SPLIT_RX.sub(".gguf", p)
        g = groups.setdefault(key, {"path": p, "parts": [], "size": 0})
        g["parts"].append({"path": p, "size": size, "sha256": sha})
        g["size"] += size
        if _SPLIT_RX.search(p) and p.endswith("-00001-of-" + _SPLIT_RX.search(p).group(2) + ".gguf"):
            g["path"] = p
    out = []
    for key, g in groups.items():
        g["parts"].sort(key=lambda x: x["path"])
        g["path"] = g["parts"][0]["path"]
        quant = quant_of(os.path.basename(key))
        pb = params_b(repo, g["size"], quant)
        out.append({"path": g["path"], "name": os.path.basename(key), "size": g["size"], "gb": round(g["size"] / GB, 2),
                    "quant": quant, "parts": len(g["parts"]), "params_b": pb, **fit(g["size"], pb, hw)})
    out.sort(key=lambda x: x["size"])
    bits = lambda x: QUANT_BITS.get(x["quant"], 0)
    on_gpu = [x for x in out if x["fit"] == "gpu" and bits(x) >= 4]
    partial = [x for x in out if x["fit"] == "partial" and bits(x) >= 4]
    best = (on_gpu[-1] if on_gpu else partial[0] if partial else next((x for x in reversed(out) if x["fit"] != "no"), None))
    have = _on_disk_names()
    for x in out:
        x["best"] = best is not None and x["path"] == best["path"]
        x["on_disk"] = os.path.basename(x["path"]).lower() in have
    return {"repo": repo, "files": out, "mmproj": sorted(mmproj, key=lambda m: m["size"]), "hardware": hw}


def _on_disk_names() -> set:
    names = set()
    for d in paths.model_dirs():
        for root, _dirs, fs in os.walk(d) if os.path.isdir(d) else ():
            names.update(f.lower() for f in fs if f.lower().endswith(".gguf"))
    return names


# ---- downloading ---------------------------------------------------------------------------------------------
def status() -> dict:
    with _lock:
        return dict(_state)


def cancel() -> dict:
    _paused.clear()
    _cancel.set()
    return {"ok": True}


def pause() -> dict:
    """Stop for now, keeping every piece downloaded so far: Resume (download() again) carries on from there (1.4.6)."""
    with _lock:
        if _state.get("state") != "running":
            return {"ok": False, "message": "Nothing is downloading."}
    _paused.set()
    _cancel.set()
    return {"ok": True}


def download(repo: str, path: str, with_mmproj: bool = True) -> dict:
    listing = files(repo)
    item = next((f for f in listing["files"] if f["path"] == path), None)
    if not item:
        return {"ok": False, "message": "That file isn't in the model's repo."}
    if item["fit"] == "no":
        return {"ok": False, "message": f"{item['name']} needs about {item['need_gb']} GB - more than this PC has free."}
    tree = _get(f"{HF}/api/models/{repo}/tree/main?recursive=true")
    key = _SPLIT_RX.sub(".gguf", path)
    parts = sorted(({"path": f["path"], "size": int((f.get("lfs") or {}).get("size") or f.get("size") or 0),
                     "sha256": (f.get("lfs") or {}).get("oid") or ""}
                    for f in tree if f.get("type") == "file" and _SPLIT_RX.sub(".gguf", f.get("path") or "") == key),
                   key=lambda x: x["path"])
    proj = None
    if with_mmproj and listing["mmproj"]:
        prefer = [m for m in listing["mmproj"] if re.search(r"f16|bf16", m["path"], re.I)] or listing["mmproj"]
        proj = prefer[0]
    with _lock:
        if _state["state"] == "running":
            return {"ok": False, "message": f"Already downloading {_state['file']}."}
        total = sum(p["size"] for p in parts) + (proj["size"] if proj else 0)
        _state.update(state="running", repo=repo, file=os.path.basename(path), path=path, mmproj=bool(with_mmproj),
                      done=0, total=total, error="", added=None, started=time.time())
        _cancel.clear()
        _paused.clear()
    # Include owner and subfolder: repos (and variants within one repo) may
    # publish identical basenames, especially mmproj-F16.gguf.
    identity = repo + "/" + path.rpartition("/")[0]
    folder_name = re.sub(r"[^\w.-]+", "_", repo.split("/")[-1]) + "-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    folder = os.path.join(paths.model_dirs()[0], folder_name)
    threading.Thread(target=_run, args=(repo, folder, parts, proj), name="model-finder", daemon=True).start()
    return {"ok": True, **status()}


def _matches_hash(path: str, expected: str) -> bool:
    if not expected:
        return True
    with _lock:
        _state["checking"] = os.path.basename(path)
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                if _cancel.is_set():
                    raise InterruptedError("Cancelled while verifying the model.")
                h.update(chunk)
        return h.hexdigest() == expected.lower()
    finally:
        with _lock:
            _state.pop("checking", None)


def _fetch(c: httpx.Client, repo: str, part: dict, dest: str) -> None:
    tmp = dest + ".part"
    if _cancel.is_set():
        raise InterruptedError("Download cancelled.")
    if (os.path.exists(dest) and os.path.getsize(dest) == part["size"]
            and _matches_hash(dest, part.get("sha256") or "")):
        with _lock:
            _state["done"] += part["size"]
        return
    url = f"{HF}/{repo}/resolve/main/{quote(part['path'])}"
    if part["size"] >= fast_download.MIN_SPLIT:
        # In pieces over several connections, resumable piece by piece (1.4.6, fast_download.py).
        def add(n):
            with _lock:
                _state["done"] += n
        try:
            fast_download.fetch(url, tmp, part["size"], cancel=_cancel, progress=add)
        except fast_download.Cancelled as e:
            raise InterruptedError(str(e))
        return _finish(tmp, dest, part)
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if have > part["size"]:
        os.remove(tmp)
        have = 0
    with _lock:
        _state["done"] += have
    headers = {"Range": f"bytes={have}-"} if have else {}
    with c.stream("GET", url, headers=headers) as r:
        if r.status_code == 416:
            pass
        elif r.status_code >= 400:
            raise RuntimeError(f"Hugging Face answered {r.status_code} for {part['path']}")
        else:
            if r.status_code == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", ""))
                if (not match or int(match[1]) != have or int(match[3]) != part["size"]
                        or not have <= int(match[2]) < part["size"]):
                    raise RuntimeError("Invalid resume range from Hugging Face; partial file was kept unchanged.")
            if have and r.status_code != 206:          # the server ignored the range: start over
                with _lock:
                    _state["done"] -= have
                have = 0
            with open(tmp, "ab" if have else "wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    if _cancel.is_set():
                        raise InterruptedError("Cancelled - the part downloaded so far is kept and resumes next time.")
                    f.write(chunk)
                    with _lock:
                        _state["done"] += len(chunk)
    _finish(tmp, dest, part)


def _finish(tmp: str, dest: str, part: dict) -> None:
    if os.path.getsize(tmp) != part["size"]:
        raise RuntimeError(f"{part['path']} is incomplete ({os.path.getsize(tmp)} of {part['size']} bytes) - try again to resume.")
    if not _matches_hash(tmp, part.get("sha256") or ""):
        os.remove(tmp)
        if os.path.exists(tmp + ".json"):            # fast_download's piece record: never resume a bad file
            os.remove(tmp + ".json")
        raise RuntimeError(f"{part['path']} doesn't match the SHA-256 Hugging Face publishes - it was deleted.")
    if _cancel.is_set():
        raise InterruptedError("Download cancelled; partial file kept.")
    os.replace(tmp, dest)


def _run(repo: str, folder: str, parts: list, proj: dict | None) -> None:
    try:
        os.makedirs(folder, exist_ok=True)
        with netlayer.client("other", timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True, private=True) as c:
            for p in parts:
                _fetch(c, repo, p, os.path.join(folder, os.path.basename(p["path"])))
            if proj:
                _fetch(c, repo, proj, os.path.join(folder, os.path.basename(proj["path"])))
        import model_scan
        try:
            added = model_scan.add_file(os.path.join(folder, os.path.basename(parts[0]["path"])))
        except ValueError as e:                     # already in models.ini (downloaded before)
            added = {"name": None, "note": str(e)}
        restarted = False
        import core.state as shared
        if added.get("name") and not shared.state_lock.locked():
            try:
                restarted = model_scan.restart_router()   # the router reads models.ini only when it starts
            except Exception:
                pass
        with _lock:
            _state.update(state="done", done=_state["total"], added=added, restarted=restarted)
    except InterruptedError as e:
        with _lock:
            if _paused.is_set():
                _state.update(state="paused", error="")
            else:
                _state.update(state="cancelled", error=str(e))
    except Exception as e:
        with _lock:
            _state.update(state="failed", error=str(e)[:300])
