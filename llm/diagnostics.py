"""
diagnostics.py
==============
Checks this PC, llama-server, the models and the app; says what will fail or
run slowly and why; fixes what it can (Settings > Diagnostics, GET/POST
/api/diagnostics). It runs once at startup and on demand.

The heart of it is whether each model FITS ON THE GPU. A local model is fast
only while its weights, its KV cache (which grows with the context size) and
llama.cpp's compute buffers all fit in the VRAM the other apps leave free.
When they don't, Windows' NVIDIA driver quietly lends system RAM for the rest
("sysmem fallback"): it keeps working, several times slower. llama.cpp's own
--fit (on by default) would instead move whole layers to the CPU - but only
for options the preset leaves unset (or "auto"), and models.ini used to pin
ngl = 99 for every model. The policy (fit_policy) is [*] ngl = auto with a
256 MB margin (--fit's default 1 GB would move layers off needlessly): at each
load llama.cpp measures the VRAM that is free right then and puts everything
on the GPU when it fits, only the overflow on the CPU when another app took
some - it never spills. The context sizes are planned against the everyday
baseline of other apps (at most DEFAULT_OTHER_MB), not the moment.

  estimate(meta, opts, ctx, cache)   VRAM a model needs, in MB (weights, KV, compute)
  assess(name, cfg, gpu)             fits / tight / spills / partial (layers on the CPU by plan),
                                     and the change that makes it fit: the largest context (up
                                     to what it was trained on, at least 16k) with an f16 or
                                     q8_0 KV cache; the experts of as few layers as needed in
                                     RAM when a MoE model's weights are too big; else 16k with
                                     the layers that don't fit on the CPU (--fit)
  placement(model)                   for a loaded model: VRAM, RAM it spilled into, the
                                     share on the CPU - shown beside the model while it works
  run() / last()                     every check; fix(id) applies one ("all": the recommended set)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time

import gguf_meta
import llama_builds
import llama_local
import system_metrics

try:
    import psutil
except ImportError:
    psutil = None

MARGIN_MB = 256              # VRAM left free: the driver, the desktop growing, fragmentation
DEFAULT_OTHER_MB = 1100      # the desktop's and everyday apps' VRAM: plans assume at most this (and it when unmeasured)
EXPERT_SHARE = 0.8           # of a MoE model's weights: the routed experts
MIN_CTX = 16384              # the smallest context a plan picks (unless the model was configured smaller)
CTX_STEPS = (65536, 49152, 32768, 24576, 16384, 12288, 8192)
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
MB = 1024 * 1024

_meta_cache: dict = {}
_state = {"result": None, "running": False}
_lock = threading.Lock()


_DISK_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model_meta_cache.json")
_disk: dict | None = None
_meta_lock = threading.Lock()                # the header cache's own (_lock guards the checks)


def _disk_cache() -> dict:
    """GGUF headers read before, by file (path, size, modified time) - kept across starts: reading every model's header
    at startup (multi-GB files, some on a hard disk) held the server up for seconds."""
    global _disk
    if _disk is None:
        try:
            import json
            with open(_DISK_CACHE, encoding="utf-8") as f:
                _disk = json.load(f)
        except (OSError, ValueError):
            _disk = {}
    return _disk


def _disk_save() -> None:
    try:
        import json
        os.makedirs(os.path.dirname(_DISK_CACHE), exist_ok=True)
        tmp = _DISK_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_disk, f)
        os.replace(tmp, _DISK_CACHE)
    except OSError:
        pass


def meta_for(path: str) -> dict | None:
    """The model's GGUF header (cached per file version - in memory and on disk), or None."""
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return None
    hit = _meta_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    stamp = f"{key[1]}:{key[2]}"
    with _meta_lock:
        saved = _disk_cache().get(path)
    if saved and saved.get("stamp") == stamp:
        meta = saved.get("meta")
    else:
        try:
            meta = gguf_meta.read(path)
        except (OSError, gguf_meta.GGUFError):
            meta = None
        with _meta_lock:
            _disk_cache()[path] = {"stamp": stamp, "meta": meta}
            _disk_save()
    _meta_cache[path] = (key, meta)
    return meta


def _vocab(meta: dict) -> int:
    toks = meta["raw"].get("tokenizer.ggml.tokens")
    return len(toks) if isinstance(toks, list) else int(meta["raw"].get(f"{meta['arch']}.vocab_size") or 150_000)


def kv_bytes(meta: dict, ctx: int, cache_k: str = "f16", cache_v: str = "f16", ubatch: int = 512) -> float:
    """The KV cache at this context, in bytes - per architecture: hybrid models
    (Qwen 3.5) cache only their attention layers; Gemma caches its sliding-window
    layers for the window only and shares the cache of its last layers;
    DeepSeek 2 (MLA) caches one compressed latent per layer."""
    raw, arch = meta["raw"], meta["arch"]
    layers = int(meta["layers"] or 0)
    bk, bv = KV_BYTES.get(cache_k, 2.0), KV_BYTES.get(cache_v, 2.0)
    kvh = meta["kv_heads"]
    kv_of = (lambda i: int(kvh[i] if i < len(kvh) else kvh[-1])) if isinstance(kvh, list) else (lambda i: int(kvh or 0))
    k_len, v_len = int(meta["key_len"] or 0), int(meta["value_len"] or 0)
    if arch == "deepseek2" and raw.get(f"{arch}.attention.kv_lora_rank"):
        rank, rope = int(raw[f"{arch}.attention.kv_lora_rank"]), int(raw.get(f"{arch}.rope.dimension_count") or 64)
        return layers * ctx * ((rank + rope) * bk + rank * bv)
    total = 0.0
    interval = meta.get("full_attention_interval")
    pattern = raw.get(f"{arch}.attention.sliding_window_pattern")
    shared = int(raw.get(f"{arch}.attention.shared_kv_layers") or 0)
    window = int(meta.get("sliding_window") or 0)
    k_swa = int(raw.get(f"{arch}.attention.key_length_swa") or k_len)
    v_swa = int(raw.get(f"{arch}.attention.value_length_swa") or v_len)
    for i in range(max(0, layers - shared)):
        if interval and (i + 1) % int(interval):              # a recurrent layer: no KV cache
            continue
        swa = bool(pattern[i]) if isinstance(pattern, list) and i < len(pattern) else (bool(window) and not pattern and arch in ("gemma2", "gemma3"))
        if swa and window:
            total += kv_of(i) * (window + ubatch) * (k_swa * bk + v_swa * bv)
        else:
            total += kv_of(i) * ctx * (k_len * bk + v_len * bv)
    if interval:                                             # the recurrent layers' state
        total += 48 * MB
    return total


def estimate(meta: dict, cfg: dict, opts: dict, ctx: int, cache: str | None = None, cpu_moe: int = 0) -> dict:
    """VRAM (MB) the model needs with all its layers on the GPU (cpu_moe: how many
    layers keep their experts in RAM, like --n-cpu-moe)."""
    weights = os.path.getsize(cfg["model_path"]) / MB
    layers = int(meta.get("layers") or 0)
    if cpu_moe and layers and meta.get("expert_count"):
        # ~80% of a MoE file is experts; attention, shared experts and embeddings stay on the GPU
        weights *= 1 - EXPERT_SHARE * min(1.0, int(cpu_moe) / layers)
    mmproj = os.path.getsize(cfg["mmproj_path"]) / MB if cfg.get("mmproj_path") and os.path.exists(cfg["mmproj_path"]) else 0.0
    ck = cache or opts.get("cache-type-k") or "f16"
    cv = cache or opts.get("cache-type-v") or "f16"
    ub = int(opts.get("ubatch-size") or 512)
    kv = kv_bytes(meta, ctx, ck, cv, ub) / MB
    compute = 280 + ub * _vocab(meta) * 4 / MB
    if str(opts.get("flash-attn", "auto")).lower() == "off":     # without flash attention the scores are materialized
        compute += int(meta.get("heads") or 32) * ub * ctx * 4 / MB * 0.25
    total = weights + mmproj + kv + compute
    return {"weights_mb": round(weights), "mmproj_mb": round(mmproj), "kv_mb": round(kv), "compute_mb": round(compute),
            "total_mb": round(total), "ctx": ctx, "cache": ck}


def ctx_limits(name: str, cfg: dict, gpu: dict) -> dict:
    """Settings > Models' context helpers: the most the model was trained for ("max"), and the largest context
    that still fits in the GPU with its current settings (KV cache type, experts in RAM) - in steps of 1,024,
    never past the trained maximum. {} when the file or its header can't be read."""
    if not cfg.get("model_path") or not os.path.exists(cfg["model_path"]):
        return {}
    meta = meta_for(cfg["model_path"])
    if not meta:
        return {}
    opts = llama_local.model_options(name)
    trained = int(meta.get("trained_ctx") or 0)
    cpu_moe = int(opts.get("n-cpu-moe") or 0) if meta.get("expert_count") else 0
    usable = gpu.get("plan_usable_mb") or gpu.get("usable_mb") or 0
    out = {"trained": trained, "usable_mb": usable, "cache": opts.get("cache-type-k") or "f16", "cpu_moe": cpu_moe}
    if not usable:
        return out
    hi = trained or 262144
    if estimate(meta, cfg, opts, 1024, cpu_moe=cpu_moe)["total_mb"] > usable:
        out["fits"] = 0                                     # even the weights don't fit: no context does
        return out
    lo = 1                                                  # in units of 1,024 tokens: lo fits, hi may not
    hi_u = hi // 1024
    if estimate(meta, cfg, opts, hi_u * 1024, cpu_moe=cpu_moe)["total_mb"] <= usable:
        lo = hi_u
    else:
        while hi_u - lo > 1:
            mid = (lo + hi_u) // 2
            if estimate(meta, cfg, opts, mid * 1024, cpu_moe=cpu_moe)["total_mb"] <= usable:
                lo = mid
            else:
                hi_u = mid
    out["fits"] = lo * 1024
    return out


def gpu_info() -> dict:
    """{"name", "total_mb", "used_mb", "other_mb", "usable_mb", "model_mb", "spilled_mb"} or {} without an NVIDIA GPU."""
    snap = system_metrics.snapshot(force=True)
    gpu = snap.get("gpu") or {}
    if not gpu.get("vram_total_mb"):
        return {}
    proc = system_metrics.gpu_process_memory() or {}
    model_mb = float(proc.get("dedicated_mb") or 0)
    used = float(gpu.get("vram_used_mb") or 0)
    other = max(300.0, used - model_mb) if used else DEFAULT_OTHER_MB
    total = float(gpu["vram_total_mb"])
    # Plans assume the everyday baseline, not this moment: what other apps hold swings by ~1 GB within an
    # hour (a browser tab, a video). --fit absorbs the swings at load time; the context sizes stay put.
    plan_other = min(other, DEFAULT_OTHER_MB)
    return {"name": gpu.get("name"), "total_mb": round(total), "used_mb": round(used), "other_mb": round(other),
            "usable_mb": round(total - other - MARGIN_MB), "plan_other_mb": round(plan_other),
            "plan_usable_mb": round(total - plan_other - MARGIN_MB), "model_mb": round(model_mb),
            "spilled_mb": round(float(proc.get("shared_mb") or 0))}


def keeps_whole(meta: dict | None) -> bool:
    """Hybrid recurrent models (Qwen 3.5: Gated Delta Net layers between the attention layers) must stay whole
    on the GPU. Split between GPU and CPU, llama.cpp turns the fused Gated Delta Net kernel off for every layer
    and the model crawls - measured on this PC (build 10735): Huihui-Qwen3.5-9B split by --fit gave no reply
    to a 120-token request in 10 minutes. Spilling into shared memory is far less bad, so these keep
    ngl = 99 and --fit stays out of it."""
    return bool(meta and meta.get("full_attention_interval"))


def fit_policy() -> dict:
    """The [*] change that lets llama.cpp's --fit place every model at load time
    ({} when it already can): ngl = auto with a MARGIN_MB margin, and no model
    pinning its own layer count (except the ones that must stay whole)."""
    defaults = llama_local.model_options("*")
    change: dict = {}
    if str(defaults.get("ngl", "auto")).lower() != "auto":
        change.setdefault("*", {})["ngl"] = "auto"
    if str(defaults.get("fit", "on")).lower() == "off":
        change.setdefault("*", {})["fit"] = None
    if str(defaults.get("fit-target", "")) != str(MARGIN_MB):
        change.setdefault("*", {})["fit-target"] = MARGIN_MB
    for c in llama_local.list_model_configs():
        own = llama_local.model_own_options(c["name"])
        if str(own.get("ngl", "auto")).lower() != "auto" and not keeps_whole(meta_for(c["model_path"])):
            change[c["name"]] = {"ngl": None}
    return change


def assess(name: str, cfg: dict, gpu: dict) -> dict:
    """One model: does it fit, and what would make it fit (the recommended options)."""
    row = {"name": name, "path": cfg["model_path"], "issues": [], "change": {}}
    if not cfg["model_path"] or not os.path.exists(cfg["model_path"]):
        row.update(status="missing", issues=[f"Its file isn't there: {cfg['model_path'] or '(no path)'}"])
        return row
    if cfg.get("mmproj_path") and not os.path.exists(cfg["mmproj_path"]):
        row["issues"].append(f"Its image projector (mmproj) is missing: {cfg['mmproj_path']} - it can't read images.")
    meta = meta_for(cfg["model_path"])
    if not meta:
        row.update(status="unreadable", issues=row["issues"] + ["Its GGUF header couldn't be read."])
        return row
    opts = llama_local.model_options(name)
    ctx = int(opts.get("ctx-size") or 65536)
    trained = int(meta.get("trained_ctx") or 0)
    cpu_moe = int(opts.get("n-cpu-moe") or 0) if meta.get("expert_count") else 0
    cur = estimate(meta, cfg, opts, ctx, cpu_moe=cpu_moe)
    usable = gpu.get("plan_usable_mb") or gpu.get("usable_mb") or 0
    row.update(arch=meta["arch"], layers=meta["layers"], trained_ctx=trained, ctx=ctx, cache=cur["cache"], estimate=cur,
               usable_mb=usable, size_mb=round(os.path.getsize(cfg["model_path"]) / MB), moe=bool(meta.get("expert_count")),
               cpu_moe=cpu_moe)
    if trained and ctx > trained:
        row["issues"].append(f"Its context ({ctx:,}) is beyond what it was trained on ({trained:,}): answers degrade past that point.")
    if not usable:
        row["status"] = "no-gpu"
        return row
    over = cur["total_mb"] - usable
    whole = keeps_whole(meta)
    # Settings a model needs whatever its context: "extra" joins every change made below.
    extra = {"ngl": 99} if whole and str(opts.get("ngl", "auto")).lower() in ("auto", "") else {}
    if cpu_moe and str(opts.get("load-mode", "")).lower() != "none":
        extra["load-mode"] = "none"
        row["issues"].append("Its experts in RAM are memory-mapped, so each is read from disk the first time it is used - on "
                             "this PC's hard disk the first reply took ~105 s instead of ~13 s. load-mode = none reads them into "
                             "RAM when it loads.")
    if "ngl" in extra:
        row["issues"].append("Split between the GPU and the CPU this architecture (hybrid, like Qwen 3.5) becomes extremely "
                             "slow, and ngl = auto lets llama.cpp split it whenever VRAM runs short: keep it whole on the GPU "
                             "(ngl = 99).")
    # ngl = auto with --fit on: llama.cpp itself keeps the layers that don't fit on the CPU (planned, steady)
    fitted = not whole and str(opts.get("ngl", "")).lower() == "auto" and str(opts.get("fit", "on")).lower() != "off"
    row["status"] = "fits" if over <= 0 else "partial" if fitted else "tight" if over <= 300 else "spills"
    if row["status"] == "partial":
        row["cpu_share"] = round(min(1.0, over / max(1, cur["weights_mb"])), 2)
    elif row["status"] != "fits":
        row["issues"].append(f"Needs ~{cur['total_mb'] / 1024:.1f} GB of VRAM at {ctx:,} tokens ({cur['kv_mb'] / 1024:.1f} GB of it the "
                             f"context cache), ~{usable / 1024:.1f} GB is free: about {max(0, over) / 1024:.1f} GB would spill into RAM.")
    if row["status"] == "fits" and not (trained and ctx > trained):
        if extra:
            row.update(change=dict(extra), plan=" ".join(p for k, p in (("ngl", "Keep it whole on the GPU (ngl = 99) so it never gets split."),
                                                                         ("load-mode", "Read its experts into RAM when it loads (load-mode = none).")) if k in extra))
        return row
    cap = min(ctx, trained) if trained else ctx
    pick = None
    for step in [cap] + [s for s in CTX_STEPS if s < cap]:
        for cache in ("f16", "q8_0"):
            est = estimate(meta, cfg, opts, step, cache)
            if est["total_mb"] <= usable:
                pick = {"ctx": step, "cache": cache, "est": est}
                break
        if pick:
            break
    # Below 16k tokens the agent's tools and a few files no longer fit in the context: rather keep 16k and let
    # --fit put a few layers on the CPU (it measures the free VRAM at load time, so it also adapts to other apps).
    if pick and pick["ctx"] >= min(cap, MIN_CTX):
        row["change"] = {"ctx-size": pick["ctx"], **({"cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on"}
                                                      if pick["cache"] == "q8_0" else {})}
        if row["status"] == "partial" and "ngl" in llama_local.model_own_options(name):
            row["change"].update({"ngl": None, "fit-target": None})   # it fits whole now: the [*] default again
        row["change"].update(extra)
        row["after"] = pick["est"]
        row["plan"] = f"Run it at {pick['ctx']:,} tokens" + (" with a q8_0 context cache" if pick["cache"] == "q8_0" else "") + \
                      f": ~{pick['est']['total_mb'] / 1024:.1f} GB, all on the GPU."
        return row
    step = min(cap, 32768 if meta.get("expert_count") else MIN_CTX)
    if meta.get("expert_count") and meta.get("layers"):
        # Experts of the first layers go to RAM; the rest stay on the GPU as far as the free VRAM allows.
        layers = int(meta["layers"])
        per_layer = row["size_mb"] * EXPERT_SHARE / layers
        base = estimate(meta, cfg, opts, step, "q8_0", cpu_moe=layers)
        on_gpu = max(0, min(layers, int((usable - MARGIN_MB - base["total_mb"]) // per_layer)))
        est = estimate(meta, cfg, opts, step, "q8_0", cpu_moe=layers - on_gpu)
        # **extra: a hybrid MoE (Qwen 3.5 35B-A3B) keeps ngl = 99 - only its experts go to RAM, never whole layers.
        row["change"] = {"ctx-size": step, "cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on",
                         "n-cpu-moe": layers - on_gpu, "load-mode": "none", **extra}
        row["after"] = est
        row["plan"] = (f"Its weights alone ({row['size_mb'] / 1024:.1f} GB) are bigger than the free VRAM. It is a mixture-of-experts "
                       f"model: keep the experts of {layers - on_gpu} of its {layers} layers in RAM and everything else on the GPU "
                       f"(~{est['total_mb'] / 1024:.1f} GB), read into RAM when it loads - much faster than spilling.")
    elif whole:
        est = estimate(meta, cfg, opts, step, "q8_0")
        row["change"] = {"ctx-size": step, "cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on", **extra}
        row["after"] = est
        row["plan"] = (f"It can't fit entirely, and this architecture crawls when split between the GPU and the CPU, so it "
                       f"stays whole on the GPU at {step:,} tokens: ~{max(0, est['total_mb'] - usable) / 1024:.1f} GB will come from "
                       "system RAM (slower). A smaller quantization of this model (e.g. Q4_K_M) would fit.")
    else:
        est = estimate(meta, cfg, opts, step, "q8_0")
        share = min(1.0, max(0.0, (est["total_mb"] - usable) / max(1, est["weights_mb"])))
        row["change"] = {"ctx-size": step, "cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on"}
        if not fitted:                                       # the [*] policy isn't applied: this model gets it itself
            row["change"].update({"ngl": "auto", "fit-target": MARGIN_MB})
        row["after"] = est
        row["plan"] = (f"It can't fit entirely. At {step:,} tokens llama.cpp keeps ~{share:.0%} of its layers on the CPU "
                       "(--fit) - slower than all-GPU, but steady, unlike spilling into shared memory.")
    if row["status"] in ("partial", "tight", "spills") and all(str(opts.get(k, "")).lower() == str(v).lower()
                                                            for k, v in row["change"].items()):
        row["change"] = {}                                   # already the best it can do on this card
        row.pop("plan", None)
        row.pop("after", None)
    return row


def placement(model: str | None) -> dict | None:
    """Where the loaded model is: {"model", "gpu_mb", "spilled_mb", "cpu_share", "state": gpu|partial|spilled}."""
    if not model:
        return None
    proc = system_metrics.gpu_process_memory() or {}
    gpu_mb, spilled = float(proc.get("dedicated_mb") or 0), float(proc.get("shared_mb") or 0)
    out = {"model": model, "gpu_mb": round(gpu_mb), "spilled_mb": round(spilled), "cpu_share": 0.0, "state": "gpu"}
    cfg = next((c for c in llama_local.list_model_configs() if c["name"] == model), None)
    meta = meta_for(cfg["model_path"]) if cfg and cfg["model_path"] and os.path.exists(cfg["model_path"]) else None
    if meta and gpu_mb:
        opts = llama_local.model_options(model)
        need = estimate(meta, cfg, opts, int(llama_local.model_context_limit(model) or opts.get("ctx-size") or 65536),
                        cpu_moe=int(opts.get("n-cpu-moe") or 0))
        # "Shared Usage" also counts RAM llama.cpp pins on purpose (weights it runs on the CPU with load-mode =
        # none: 8.4 GB for DeepSeek-Coder-V2 here), so only what the GPU part lacks in VRAM counts as spilled.
        spilled = min(spilled, max(0.0, need["total_mb"] - gpu_mb))
        out["spilled_mb"] = round(spilled)
        missing = need["total_mb"] - gpu_mb - spilled
        if missing > 600:                                   # neither in VRAM nor lent RAM: layers on the CPU
            out["cpu_share"] = round(min(1.0, missing / max(1, need["weights_mb"])), 2)
            out["state"] = "partial"
    if spilled > 256:
        out["state"] = "spilled"
    return out


# ==============================================================================
# The checks
# ==============================================================================
def _check(area, level, title, detail="", fix=None) -> dict:
    return {"area": area, "level": level, "title": title, "detail": detail, **({"fix": fix} if fix else {})}


def _folder_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / MB


def _server_exe() -> str | None:
    import paths
    for d in (*paths.llama_dirs(), *paths.prism_dirs()):
        exe = os.path.join(d, "llama-server.exe")
        if os.path.exists(exe):
            return exe
    return shutil.which("llama-server")


def _version(exe: str) -> str:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=8,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        m = re.search(r"build (\d+)", out.stdout + out.stderr) or re.search(r"version: (\d+)", out.stdout + out.stderr)
        return f"build {m.group(1)}" if m else ""
    except Exception:
        return ""


def gpu_devices(exe: str) -> list | None:
    """(1.4.8) The GPUs llama-server can use ("CUDA0: NVIDIA GeForce RTX 3060 Ti ..."), [] for none, None if unknown."""
    try:
        out = subprocess.run([exe, "--list-devices"], capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return None
    text = (out.stdout or "") + (out.stderr or "")
    if "Available devices" not in text:
        return None                                       # an older build without --list-devices
    return re.findall(r"^\s*((?:CUDA|Vulkan|ROCm|SYCL|Metal)\d+:.*)$", text, re.M)


def drive_media(letter: str) -> dict:
    """{"media": "HDD" | "SSD" | ..., "name"} of the physical disk behind a drive letter (Windows), or {}."""
    if os.name != "nt" or not re.fullmatch(r"[A-Za-z]", letter or ""):
        return {}
    script = (f"$n = (Get-Partition -DriveLetter {letter} -ErrorAction Stop).DiskNumber; "
              "$d = Get-PhysicalDisk | Where-Object DeviceId -eq $n | Select-Object -First 1; "
              "\"$($d.MediaType)|$($d.FriendlyName)\"")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                             text=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    except Exception:
        return {}
    media, _, name = out.partition("|")
    return {"media": media, "name": name} if media else {}


def run(app_dir: str | None = None) -> dict:
    """Every check (a few seconds: GPU, disks, every model's header)."""
    app_dir = app_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checks, started = [], time.time()
    # ---- hardware
    gpu = gpu_info()
    if gpu:
        checks.append(_check("hardware", "ok", f"{gpu['name']}: {gpu['total_mb'] / 1024:.0f} GB of VRAM",
                             f"Other apps use {gpu['other_mb'] / 1024:.1f} GB; ~{gpu['usable_mb'] / 1024:.1f} GB is left for a model."))
        if gpu["other_mb"] > gpu["plan_other_mb"] + 300:
            checks.append(_check("hardware", "warn" if gpu["other_mb"] > 2000 else "info", "Other apps hold more VRAM than usual right now",
                                 f"{gpu['other_mb'] / 1024:.1f} GB is in use without a model (the plans assume ~{gpu['plan_other_mb'] / 1024:.1f} GB) - "
                                 "browsers, games and video apps use VRAM. A model loaded now keeps the part that no longer fits on "
                                 "the CPU until they close; closing them leaves more room."))
    else:
        checks.append(_check("hardware", "warn", "No NVIDIA GPU found (nvidia-smi)", "Local models run on the CPU - much slower."))
    if psutil:
        vm = psutil.virtual_memory()
        cores = psutil.cpu_count(logical=False) or 0
        checks.append(_check("hardware", "ok" if vm.available > 4 * 1024 ** 3 else "warn",
                             f"{vm.total / 1024 ** 3:.0f} GB RAM, {vm.available / 1024 ** 3:.0f} GB free · {cores} CPU cores"))
    for label, path in (("the app", app_dir), ("Windows (temp, the page file)", os.environ.get("SystemDrive", "C:") + "\\")):
        try:
            free = shutil.disk_usage(path).free / 1024 ** 3
        except OSError:
            continue
        level = "error" if free < 2 else "warn" if free < 8 else "ok"
        checks.append(_check("hardware", level, f"{free:.1f} GB free on {os.path.splitdrive(path)[0] or path} ({label})",
                             "" if level == "ok" else "Low disk space makes saving chats, installing updates and Windows' page "
                             "file fail. Free some space - for example empty the pip cache (pip cache purge) or Downloads."))
        if os.path.splitdrive(path)[0].upper() == os.path.splitdrive(os.environ.get("SystemDrive", "C:"))[0].upper():
            break
    # ---- llama-server
    exe = _server_exe()
    if exe:
        checks.append(_check("server", "ok", f"llama-server: {exe}", _version(exe)))
        # (1.4.8) The engine can see the GPU at all. On 10 Nov a llama.cpp update ran out of disk space mid-unpack:
        # NVIDIA's runtime files were missing or cut short, llama-server listed no device, and every model ran on
        # the CPU at 4 tokens/s with nothing saying why.
        devices = gpu_devices(exe)
        if gpu and devices is not None and not devices:
            checks.append(_check("server", "error", "llama.cpp can't use your GPU - every model runs on the CPU",
                                 f"llama-server lists no GPU although {gpu['name']} is there. Its GPU files are missing or "
                                 "damaged (an update that ran out of disk space does this). Free some space on the drive, then "
                                 "run the installer again (Settings > About > Update now, or the setup file): setup repairs it."))
        prism = llama_builds.exe_of("prism")
        if prism and prism != exe:
            checks.append(_check("server", "ok", f"PrismML's llama.cpp, for Ternary Bonsai 2: {prism}", _version(prism)))
    else:
        checks.append(_check("server", "error", "llama-server.exe wasn't found", "Run install.bat to download it into local\\llama.cpp "
                             "(or put llama-server.exe there yourself). Local mode needs it."))
    policy = fit_policy()
    if policy:
        pinned = [k for k in policy if k != "*"]
        checks.append(_check("server", "warn", "Models are pinned to the GPU (ngl = 99)" if "ngl" in policy.get("*", {}) or pinned
                             else "llama.cpp's --fit margin isn't set",
                             "With a fixed layer count llama.cpp can't react when another app takes VRAM: the model spills into "
                             "shared memory instead - several times slower. Apply sets ngl = auto with a 256 MB margin in [*]"
                             + (f" (and removes the fixed count from {len(pinned)} model{'s' if len(pinned) != 1 else ''})" if pinned else "")
                             + ": at each load everything still goes on the GPU when it fits, and only what doesn't fit then "
                             "runs on the CPU.", {"id": "fit-policy", "label": "Apply"}))
    up = llama_local.server_models(force=True)
    loaded = next((m for m in (up or []) if llama_local.router_status(m).get("loaded")), None)
    where = placement(loaded) if loaded and gpu else None
    if gpu:
        gpu["spilled_mb"] = where["spilled_mb"] if where else 0      # the counter alone also includes pinned RAM
    if where and where["spilled_mb"] > 256:
        checks.append(_check("hardware", "error", f"{loaded} spilled {where['spilled_mb'] / 1024:.1f} GB into system RAM",
                             "The GPU is reading part of it over PCIe, which is several times slower. The fixes below make it fit."))
    checks.append(_check("server", "ok" if up is not None else "info",
                         "llama-server is running" + (f" · {len(up)} models" if up else "") if up is not None else "llama-server isn't running",
                         "" if up is not None else "run.bat starts it. Cloud mode works without it."))
    rows = [assess(c["name"], c, gpu) for c in llama_local.list_model_configs()]
    slow = [r for r in rows if r.get("status") in ("spills", "tight")]
    # ---- models
    for r in rows:
        if r["status"] == "missing":
            checks.append(_check("models", "warn", f"{r['name']}: file missing", r["issues"][0],
                                 {"id": f"remove:{r['name']}", "label": "Remove it from models.ini"}))
        elif r["status"] == "unreadable":
            checks.append(_check("models", "warn", f"{r['name']}: can't be read", " ".join(r["issues"])))
        elif r["status"] == "partial":
            checks.append(_check("models", "warn" if r.get("change") else "info", f"{r['name']}: about {r['cpu_share']:.0%} of it runs on the CPU",
                                 "It's bigger than the free VRAM, so llama.cpp keeps some of its layers on the CPU (--fit): slower "
                                 "than all-GPU, but steady - nothing spills." + "".join(" " + i for i in r["issues"])
                                 + (f" Fix: {r['plan']}" if r.get("plan") else ""),
                                 {"id": f"fit:{r['name']}", "label": "Apply"} if r.get("change") else None))
        elif r["issues"]:
            level = "error" if r["status"] == "spills" and r.get("change") else "warn"   # no fix left: a warning
            checks.append(_check("models", level, f"{r['name']}: " + ({"spills": "doesn't fit on the GPU", "tight": "barely fits on the GPU"}
                                 .get(r["status"], "can run faster" if r["status"] == "fits" and r.get("change") else "check its context")),
                                 " ".join(r["issues"]) + (f" Fix: {r['plan']}" if r.get("plan") else ""),
                                 {"id": f"fit:{r['name']}", "label": "Apply"} if r.get("change") else None))
        else:
            checks.append(_check("models", "ok", f"{r['name']}: fits on the GPU",
                                 (f"~{r['estimate']['total_mb'] / 1024:.1f} GB at {r['ctx']:,} tokens" if r.get("estimate") else "")
                                 + (f", with the experts of {r['cpu_moe']} layers in RAM" if r.get("cpu_moe") else "")))
    if policy or any(r.get("change") for r in rows):
        todo = sum(1 for r in rows if r.get("change")) + (1 if policy else 0)
        checks.insert(0, _check("summary", "warn", f"{todo} setting{'s' if todo != 1 else ''} can be made faster or steadier",
                                "Apply all writes the recommended context sizes, cache types and placement to models.ini (the old "
                                "file is kept as models.ini.bak-<time>). Each model uses them the next time it loads.",
                                {"id": "all", "label": "Apply all"}))
    for letter in sorted({os.path.splitdrive(r["path"])[0].rstrip(":").upper() for r in rows if r["status"] != "missing"} - {""}):
        disk = drive_media(letter)
        if disk.get("media") == "HDD":
            checks.append(_check("hardware", "warn", f"The models are on a hard disk ({letter}: {disk['name']})".replace(" )", ")"),
                                 "Switching models reads several GB from a spinning disk: 30-60 s per load on this PC. Once "
                                 "loaded, a model that sits on the GPU is unaffected; weights kept on the CPU are read into RAM "
                                 "when it loads (load-mode = none) instead of from the disk as they are first used. An SSD "
                                 "would make switching several times faster."))
    if slow and gpu:
        checks.append(_check("hardware", "info", "Tip: stop Windows lending RAM to the GPU",
                             "NVIDIA Control Panel > Manage 3D settings > Program settings > llama-server.exe > CUDA - Sysmem Fallback "
                             "Policy: Prefer No Sysmem Fallback. A model that doesn't fit then fails to load instead of running "
                             "slowly - pair it with the fixes above."))
    # ---- the app
    data = os.path.join(app_dir, "data")
    if os.path.isdir(data):
        sizes = {name: _folder_mb(os.path.join(data, name)) for name in os.listdir(data) if os.path.isdir(os.path.join(data, name))}
        big = sum(sizes.values())
        top = ", ".join(f"{k} {v:,.0f} MB" for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:3] if v >= 1)
        checks.append(_check("app", "warn" if big > 3000 else "ok", f"App data: {big:,.0f} MB", top,
                             {"id": "cleanup", "label": "Remove uploads and recordings older than 30 days"} if big > 500 else None))
    probe = system_metrics._COUNTERS
    checks.append(_check("app", "ok", "GPU telemetry reads Windows counters directly",
                         "In-process (a few ms), not a PowerShell call every 1.5 s.") if os.name == "nt" and (probe is None or probe.ok)
                  else _check("app", "info", "GPU telemetry: per-process VRAM isn't available here", ""))
    result = {"at": time.time(), "seconds": round(time.time() - started, 1), "gpu": gpu, "models": rows, "checks": checks,
              "counts": {lvl: sum(1 for c in checks if c["level"] == lvl) for lvl in ("error", "warn", "info", "ok")}}
    _state["result"] = result
    return result


def last() -> dict | None:
    return _state["result"]


def run_in_background():
    """At startup: the checks run on their own thread; the page reads the result when it's there."""
    def go():
        with _lock:
            if _state["running"]:
                return
            _state["running"] = True
        try:
            run()
        except Exception:
            pass
        finally:
            _state["running"] = False
    threading.Thread(target=go, daemon=True, name="diagnostics").start()


# ==============================================================================
# Fixes
# ==============================================================================
def _prune(folder: str, days: int = 30) -> int:
    cutoff, removed = time.time() - days * 86400, 0
    if not os.path.isdir(folder):
        return 0
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        try:
            if os.path.getmtime(path) < cutoff:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def fix(fix_id: str, app_dir: str | None = None) -> dict:
    """Apply one fix: "all", "fit-policy", "fit:<model>", "remove:<model>", "cleanup"."""
    app_dir = app_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = _state["result"] or run(app_dir)
    rows = {r["name"]: r for r in result["models"]}
    changes: dict = {}
    first_backup = None
    if fix_id in ("all", "fit-policy") and fit_policy():
        # The [*] policy first: the per-model plans depend on it (they skip what it already covers).
        first_backup = llama_local.set_model_options(fit_policy())
        result = run(app_dir)
        rows = {r["name"]: r for r in result["models"]}
        if fix_id == "fit-policy":
            return {"ok": True, "backup": first_backup, "changed": ["*"],
                    "message": f"llama.cpp now fits every model to the free VRAM at load time (ngl = auto, 256 MB margin; the old "
                               f"file is {os.path.basename(first_backup)}). Restart llama-server (run.bat) or load a model again to apply."}
    if fix_id == "fit-policy":
        return {"ok": False, "message": "It is already set."}
    if fix_id == "all":
        for name, r in rows.items():
            if r.get("change"):
                changes.setdefault(name, {}).update(r["change"])
    elif fix_id.startswith("fit:"):
        r = rows.get(fix_id[4:])
        if not r or not r.get("change"):
            return {"ok": False, "message": "Nothing to change for that model any more."}
        changes[r["name"]] = dict(r["change"])
    elif fix_id.startswith("remove:"):
        name = fix_id[7:]
        path = llama_local._ini_path()
        cp = llama_local._load_ini(path)
        if name not in cp.sections():
            return {"ok": False, "message": "That model is no longer in models.ini."}
        backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, backup)
        cp.remove_section(name)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            cp.write(f)
        os.replace(path + ".tmp", path)
        llama_local._ini_cache.update(key=None, configs=None)
        run(app_dir)
        return {"ok": True, "message": f"Removed {name} from models.ini (the old file is {os.path.basename(backup)}). "
                                       "Restart llama-server (run.bat) so its model list updates."}
    elif fix_id == "cleanup":
        n = sum(_prune(os.path.join(app_dir, "data", d)) for d in ("uploads", "recordings"))
        run(app_dir)
        return {"ok": True, "message": f"Removed {n} old upload{'s' if n != 1 else ''} and recording{'s' if n != 1 else ''}."}
    if not changes and not first_backup:
        return {"ok": False, "message": "Nothing to change."}
    if changes:
        backup = llama_local.set_model_options(changes, backup=not first_backup)
        run(app_dir)
    backup = first_backup or backup                          # the file as it was before this fix
    names = [n for n in changes if n != "*"]
    what = (["llama.cpp now fits every model to the free VRAM at load time"] if first_backup else []) + \
           ([f"{len(names)} model{'s' if len(names) != 1 else ''} got new settings"] if names else [])
    return {"ok": True, "backup": backup, "changed": (["*"] if first_backup else []) + names,
            "message": f"{'; '.join(what)} (the old models.ini is {os.path.basename(backup)}). Each model uses its new "
                       "settings the next time it loads - Load it again from the model menu to apply now."}
