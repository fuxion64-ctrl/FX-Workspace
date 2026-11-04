"""
image_gen.py
============
Local image generation: stable-diffusion.cpp's sd-server (the image twin of llama.cpp), run on this PC
with one of the registered models - nothing is sent anywhere.

  fast      FLUX.2 klein 4B (Black Forest Labs, Apache-2.0): 4 steps; can also edit an attached image
  balanced  Z-Image-Turbo (Alibaba Tongyi, Apache-2.0): 8 steps; photoreal, readable text in images
  best      Krea 2 Turbo (Krea, Krea 2 Community License): 12B, 8 steps; the highest quality, slowest
  plus single-file checkpoints: DreamShaper 8 and Anything V5 (SD 1.5), Animagine XL 4.0 and Pony V6 XL (SDXL)

Files live in settings image_root (default local\\Images - paths.py; image_setup.py downloads and verifies
them by SHA-256); after a partial move (move_models) a model may still be in one of image_extra_roots. One
sd-server runs at a time, for the model in use; it stops after image_keep_minutes idle.

Speed, as measured on this PC (RTX 3060 Ti 8 GB, 48 GB RAM, models on a hard disk; 2026-10-01):
  * The engine keeps the weights in RAM and caches what fits on the GPU. With desktop apps holding ~2.3 GB of
    VRAM that cache left 0.7 GB for the 1.2 GB the drawing itself needs, and every picture failed. --max-vram
    -<image_vram_reserve> keeps that room free (1.5 GB works for all three models at 1024x1024); a picture that
    still runs out is retried once with 1 GB more.
  * The step cache (request cache_mode "easycache") made Fast 3x and Balanced 2.7x faster with the same
    picture (Balanced a touch softer in fine detail); on Krea 2 it turned the sky to grain, so Best never uses it.
  * The model files are memory-mapped, so the first step reads them from disk in scattered pieces: 248 s on the
    hard disk (the 290 s cat). warm() reads them front to back into Windows' cache first - as soon as a message
    asks for a picture, while the chat model is still writing the prompt.
The GPU (8 GB) can't hold a chat model and an image model together: with image_free_gpu on, the chat model is
unloaded first (llama_local.unload_all) and comes back on the next message.

Made images: data/images/<id>.png with <id>.json beside it (prompt, model, seed, steps, size, seconds, fav).
Trash is data/images/trash/. A picture is deleted for good only from the trash, on the user's click (purge,
empty_trash) - nothing deletes one by itself.

Prompts pass check_prompt first: sexual content involving minors is always refused; with image_filter on
(the default), sexual content of real named people and sexual violence are refused too (the Krea 2 licence
asks deployers to filter). The filter reads the words only - it never touches the image or its quality.
"""
from __future__ import annotations

import paths
import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "data", "images")
DEFAULT_ROOT = paths.feature_root("Images")
PORT = int(os.environ.get("SD_SERVER_PORT", "8095"))
BASE = f"http://127.0.0.1:{PORT}"

MODELS = {
    "fast": {"name": "FLUX.2 klein 4B", "by": "Black Forest Labs", "license": "Apache-2.0",
             "diffusion": "flux-2-klein-4b-Q8_0.gguf", "llm": "Qwen3-4B-Q4_K_M.gguf", "vae": "flux2-vae.safetensors",
             "steps": 4, "cfg": 1.0, "sampler": "euler", "edit": True, "step_cache": True,
             "about": "Fastest, and can edit an image you attach."},
    "balanced": {"name": "Z-Image-Turbo", "by": "Alibaba Tongyi", "license": "Apache-2.0",
                 "diffusion": "z_image_turbo-Q6_K.gguf", "llm": "Qwen3-4B-Q4_K_M.gguf", "vae": "z_image_ae.safetensors",
                 "steps": 8, "cfg": 1.0, "sampler": None, "edit": False, "step_cache": True,
                 "about": "Photoreal people and places; readable text in images."},
    "best": {"name": "Krea 2 Turbo", "by": "Krea", "license": "Krea 2 Community License",
             "diffusion": "Krea-2-Turbo-Q4_K_M.gguf", "llm": "Qwen3-VL-4B-Instruct-Q4_K_M.gguf", "vae": "krea2_vae.safetensors",
             "steps": 8, "cfg": 1.0, "sampler": None, "edit": False, "step_cache": False,   # the cache turns it to grain
             "about": "The highest quality and style; the slowest."},
    "dreamshaper8": {"name": "DreamShaper 8", "by": "Lykon", "license": "DreamShaper model license",
                    "checkpoint": "DreamShaper_8_pruned.safetensors", "clip_skip": 1,
                    "steps": 20, "cfg": 7.0, "sampler": "euler_a", "edit": False, "step_cache": False,
                    "about": "SD 1.5 artwork and illustration; works best near 512px."},
    "anythingv5": {"name": "Anything V5", "by": "Anything community", "license": "CreativeML OpenRAIL-M",
                   "checkpoint": "anything-v5.safetensors", "clip_skip": 2,
                   "steps": 20, "cfg": 7.0, "sampler": "euler_a", "edit": False, "step_cache": False,
                   "about": "SD 1.5 anime illustration; works best near 512px."},
    # Settings from its model card (2026-10-13): CFG 5, Euler a; tag prompts ("1girl, name, series, ...,
    # masterpiece, high score, great score, absurdres"). `negative` is used when a request brings none of its own.
    # Measured here at 832x1216 (2026-10-13): the published fp16 file needed 2.5 GB kept free and sampled 28 steps in
    # 49 s; this q8_0 copy (sd-cli -M convert --type q8_0, from the fp16 file beside it) fits at 1.5 GB and takes
    # 29 s with no visible difference; 20 steps instead of the card's 28 takes 21 s, a touch softer. The step
    # cache made no difference on SDXL.
    "animagine4": {"name": "Animagine XL 4.0", "by": "Cagliostro Lab", "license": "CreativeML Open RAIL++-M",
                   "checkpoint": "animagine-xl-4.0-opt-q8_0.gguf", "clip_skip": 2,
                   "steps": 20, "cfg": 5.0, "sampler": "euler_a", "edit": False, "step_cache": False,
                   "negative": "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, "
                               "cropped, worst quality, low quality, low score, bad score, average score, signature, "
                               "watermark, username, blurry",
                   "about": "SDXL anime; knows many anime characters by name. Best at 832x1216 or 1024x1024."},
    # Pony Diffusion V6 XL (the "start with this one" file, SHA-256 checked against Civitai's), q8_0 like Animagine.
    # Prompts start with the score tags. V7 (AuraFlow) isn't here: sd-server can't run AuraFlow.
    # The author's Euler a / CFG 7 at 20 steps came out soft and smeared here (2026-10-13, seed 1234); DPM++ 2M
    # Karras, 28 steps, CFG 6 with `suffix` (appended to every prompt) and the extra negatives gave clean lines.
    "pony6": {"name": "Pony Diffusion V6 XL", "by": "PurpleSmartAI", "license": "Fair AI Public License 1.0-SD",
              "checkpoint": "ponyDiffusionV6XL-q8_0.gguf", "clip_skip": 2,
              "steps": 28, "cfg": 6.0, "sampler": "dpm++2m", "scheduler": "karras", "edit": False, "step_cache": False,
              "suffix": "sharp focus, clean lineart, highly detailed",
              "negative": "score_4, score_5, score_6, lowres, bad anatomy, bad hands, watermark, signature, blurry, sketch",
              "about": "SDXL cartoon, anime, furry and western styles; start prompts with "
                       "'score_9, score_8_up, score_7_up'."},
    # The same Pony file, tuned toward Animagine XL 4.0's clean anime look (user request, 2026-10-13). Compared
    # side by side with Animagine on one prompt/seed: DPM++ 2M SDE Karras, CFG 5, 30 steps, "anime coloring /
    # official art / clean lineart / soft lighting" and a negative against painterly, 3D and oversaturated looks
    # came closest; "vibrant colors" overcooked it pink. `prefix` goes before the user's words (unless they already
    # wrote score tags), `suffix` after.
    "pony6_anime": {"name": "Pony V6 · Anime clean", "by": "PurpleSmartAI", "license": "Fair AI Public License 1.0-SD",
                    "checkpoint": "ponyDiffusionV6XL-q8_0.gguf", "clip_skip": 2,
                    "steps": 30, "cfg": 5.0, "sampler": "dpm++2m_sde", "scheduler": "karras", "edit": False,
                    "step_cache": False,
                    "prefix": "score_9, score_8_up, score_7_up, source_anime, rating_safe, anime coloring, official art",
                    "suffix": "clean lineart, cel shading, detailed eyes, soft lighting, natural lighting, "
                              "detailed background, depth of field",
                    "negative": "score_4, score_5, score_6, 3d, realistic, photorealistic, sketch, painting, watercolor, "
                                "painterly, blurry, lowres, bad anatomy, bad hands, monochrome, oversaturated, "
                                "high contrast, jpeg artifacts, watermark, signature, text",
                    "about": "Pony with an Animagine-like clean anime look - just describe the picture."},
    # The user's favourite of that comparison (CFG 4.5, warm backlight), made repeatable: the same base plus
    # backlighting / golden hour / rim light / light particles held its look over 3 seeds ("glowing hair" was left
    # out - it turned silver hair blond).
    "pony6_warm": {"name": "Pony V6 · Anime warm", "by": "PurpleSmartAI", "license": "Fair AI Public License 1.0-SD",
                   "checkpoint": "ponyDiffusionV6XL-q8_0.gguf", "clip_skip": 2,
                   "steps": 30, "cfg": 4.5, "sampler": "dpm++2m_sde", "scheduler": "karras", "edit": False,
                   "step_cache": False,
                   "prefix": "score_9, score_8_up, score_7_up, source_anime, rating_safe, anime coloring, official art",
                   "suffix": "clean lineart, cel shading, detailed eyes, backlighting, warm lighting, golden hour, "
                             "rim light, light particles, soft lighting, detailed background, depth of field",
                   "negative": "score_4, score_5, score_6, 3d, realistic, photorealistic, sketch, painting, watercolor, "
                               "painterly, blurry, lowres, bad anatomy, bad hands, monochrome, oversaturated, "
                               "high contrast, jpeg artifacts, watermark, signature, text",
                   "about": "Pony's clean anime look with warm, glowing backlight - just describe the picture."},
}
CATALOG = tuple(MODELS)              # the models this app knows and can set up; yours join them (add_model)
QUALITIES = tuple(MODELS)


# ---- your own models (1.4.6) --------------------------------------------------------------------------------------
# Images › Add a model: a single-file checkpoint already on this PC (SD 1.x, SD 2.x or SDXL; .safetensors or .gguf)
# joins the list with settings that suit its kind. The kind is read from the file's own tensor names, never guessed
# from its name. Kept in data/image_models.json; the file stays where it is, and removing it from the list leaves it.
CUSTOM_PATH = os.path.join(ROOT, "data", "image_models.json")
MODEL_EXT = (".safetensors", ".gguf", ".ckpt")
KINDS = {   # what each kind is good with, from the base models' cards (steps, CFG, sampler, clip skip, seconds here)
    "sd1": {"label": "SD 1.5", "steps": 25, "cfg": 7.0, "sampler": "euler_a", "clip_skip": 1, "seconds": 12,
            "about": "An SD 1.5 model; works best near 512 px."},
    "sd2": {"label": "SD 2.x", "steps": 25, "cfg": 7.0, "sampler": "euler_a", "clip_skip": 1, "seconds": 15,
            "about": "An SD 2.x model; works best near 768 px."},
    "sdxl": {"label": "SDXL", "steps": 25, "cfg": 6.0, "sampler": "euler_a", "clip_skip": 1, "seconds": 30,
             "about": "An SDXL model; best at 1024x1024 or 832x1216."},
}
_ANIME = re.compile(r"anime|anything|pony|animagine|illustrious|noob|counterfeit|waifu|hentai|toon", re.I)
_FAST = re.compile(r"turbo|lightning|hyper|lcm|\bdmd", re.I)


class ModelFileError(ValueError):
    """Why a file can't be added - said plainly (the page shows it)."""


def _safetensors_names(path: str) -> list:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if not 2 <= n < 200 * 2 ** 20:
            raise ModelFileError("This doesn't look like a .safetensors file - its header can't be read.")
        head = json.loads(f.read(n))
    return [k for k in head if k != "__metadata__"]


def _gguf_names(path: str) -> list:
    """The tensor names in a GGUF file's header (the format llama.cpp and sd.cpp share)."""
    import struct
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ModelFileError("This doesn't look like a .gguf file.")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version < 2 or n_tensors > 100_000 or n_kv > 100_000:
            raise ModelFileError("This .gguf file's header can't be read (an unknown version).")
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        string = lambda: f.read(struct.unpack("<Q", f.read(8))[0])          # noqa: E731

        def skip(t):
            if t in fixed:
                f.seek(fixed[t], 1)
            elif t == 8:
                string()
            elif t == 9:
                inner, count = struct.unpack("<IQ", f.read(12))
                if inner in fixed:
                    f.seek(fixed[inner] * count, 1)
                else:
                    for _ in range(count):
                        skip(inner)
            else:
                raise ModelFileError("This .gguf file's header can't be read.")
        for _ in range(n_kv):
            string()
            skip(struct.unpack("<I", f.read(4))[0])
        names = []
        for _ in range(n_tensors):
            names.append(string().decode("utf-8", "replace"))
            dims = struct.unpack("<I", f.read(4))[0]
            f.seek(8 * dims + 4 + 8, 1)                                       # the dims, its type, its offset
        return names


def model_kind(path: str) -> str:
    """"sd1" | "sd2" | "sdxl" for a whole checkpoint the engine runs from one file; ModelFileError for anything else,
    saying what it is when it can tell (a LoRA, a FLUX or SD3 part without its text encoder and VAE)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ckpt":
        raise ModelFileError("A .ckpt file can run code when it's opened - convert it to .safetensors first.")
    try:
        names = _safetensors_names(path) if ext == ".safetensors" else _gguf_names(path)
    except ModelFileError:
        raise
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise ModelFileError(f"The file can't be read: {str(e)[:120]}")
    has = lambda *parts: any(all(p in n for p in parts) for n in names)    # noqa: E731
    if has("lora_") or (has("lora_up") or has("lora_down") or has(".lora.")) and not has("input_blocks"):
        raise ModelFileError("This is a LoRA (a style add-on), not a model - LoRAs come with Labs' picture styles.")
    if has("double_blocks") or has("joint_blocks") or has("cap_embedder"):
        raise ModelFileError("This is a FLUX, SD3 or Z-Image part on its own - it needs its text encoder and VAE "
                             "beside it, which this app can't add from one file yet.")
    if not has("input_blocks"):
        if any(n.startswith("blk.") for n in names) and has("token_embd"):
            raise ModelFileError("This is a chat model - add it in Settings › Models, not here.")
        raise ModelFileError("This file isn't a picture model the engine knows (SD 1.x, SD 2.x or SDXL).")
    if not has("first_stage_model"):
        raise ModelFileError("This model file has no VAE inside it - pick a full checkpoint (most are).")
    if has("conditioner.embedders") or has("label_emb"):
        return "sdxl"
    if has("cond_stage_model.model.") or has("cond_stage_model.transformer.text_model.encoder.layers.22"):
        return "sd2"
    if has("cond_stage_model"):
        return "sd1"
    raise ModelFileError("This model has no text encoder inside it - pick a full checkpoint.")


def _custom_load() -> list:
    try:
        with open(CUSTOM_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [m for m in data.get("models", []) if isinstance(m, dict) and str(m.get("key", "")).startswith("my_")]
    except (OSError, ValueError, AttributeError):
        return []


def _custom_save(models: list) -> None:
    os.makedirs(os.path.dirname(CUSTOM_PATH), exist_ok=True)
    tmp = CUSTOM_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"models": models}, f, indent=1)
    os.replace(tmp, CUSTOM_PATH)


def _custom_register() -> None:
    """Your models into MODELS (and QUALITIES), the catalogue first."""
    global QUALITIES
    for k in [k for k in MODELS if k not in CATALOG]:
        MODELS.pop(k)
    for m in _custom_load():
        MODELS[m["key"]] = {k: v for k, v in m.items() if k != "key"}
        _typical.setdefault(m["key"], float(KINDS.get(m.get("kind"), {}).get("seconds", 20)))
    QUALITIES = tuple(MODELS)


def add_model(path: str, name: str = "") -> dict:
    """Add a checkpoint file to the list. Returns the new entry (with its key); ModelFileError says why not."""
    path = os.path.abspath(str(path or "").strip().strip('"'))
    if not os.path.isfile(path):
        raise ModelFileError("There's no file there - check the path.")
    if not path.lower().endswith(MODEL_EXT):
        raise ModelFileError("Pick a .safetensors or .gguf model file.")
    kind = model_kind(path)
    have = _custom_load()
    for m in have:
        if os.path.normcase(m.get("checkpoint", "")) == os.path.normcase(path):
            raise ModelFileError(f"It's in the list already, as {m['name']}.")
    stem = os.path.splitext(os.path.basename(path))[0]
    name = " ".join(str(name or "").split())[:60] or re.sub(r"[_-]+", " ", stem).strip()[:60]
    k = KINDS[kind]
    fast, anime = bool(_FAST.search(stem)), bool(_ANIME.search(stem))
    entry = {"key": "my_" + hashlib.sha256(os.path.normcase(path).encode()).hexdigest()[:10],
             "name": name, "by": "Added by you", "license": "its own licence - see where you got it",
             "checkpoint": path, "kind": kind, "custom": True,
             "clip_skip": 2 if anime else k["clip_skip"],
             "steps": 8 if fast else k["steps"], "cfg": 2.0 if fast else k["cfg"], "sampler": k["sampler"],
             "edit": False, "step_cache": False,
             "about": f"{k['about']}{' Few-step (turbo) settings.' if fast else ''}",
             "added": time.time()}
    _custom_save(have + [entry])
    _custom_register()
    return entry


def remove_model(key: str) -> bool:
    """Take one of your models off the list (its file stays where it is)."""
    have = _custom_load()
    left = [m for m in have if m["key"] != key]
    if len(left) == len(have):
        return False
    if _alive() and _proc_key == key:
        _stop()
    _custom_save(left)
    _custom_register()
    return True


def usable(key: str | None) -> str | None:
    """The model to draw with: the one asked for when it's on this PC, else the first that is (1.4.6 - the
    default, Balanced, was asked for on PCs that never downloaded it). None when none is."""
    if key in MODELS and installed(key):
        return key
    return next((k for k in MODELS if installed(k)), None)


SIZES ={"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832), "wide": (1344, 768), "tall": (768, 1344)}
MAX_COUNT = 4                         # pictures per request (the engine allows 8; 4 keeps a request under ~2 min)
DEFAULT_SETTINGS = {
    "image_root": DEFAULT_ROOT,       # where the engine and models are (image_setup.py)
    "image_extra_roots": [],          # other folders still holding a model after a partial move (move_models)
    "image_quality": "balanced",      # fast | balanced | best
    "image_size": "square",           # a SIZES key
    "image_keep_minutes": 5,          # sd-server stops after this long idle (frees RAM and VRAM)
    "image_free_gpu": True,           # unload the chat model while an image renders
    "image_filter": True,             # the optional part of check_prompt
    "image_speed_boost": True,        # the step cache, for the models it doesn't harm (MODELS step_cache)
    "image_vram_reserve": 1.5,        # GB of GPU memory kept free for the drawing itself (--max-vram -N)
    "image_warmup": True,             # read the model files into Windows' cache as soon as a picture is asked for
}

_lock = threading.RLock()             # one generation at a time
_proc: subprocess.Popen | None = None
_proc_key = ""
_proc_reserve = 0.0                   # the reserve the running engine was started with
_log_path: str | None = None          # the running engine's log (progress is read from it)
_last_used = 0.0
_start_free = 0                       # free VRAM (MB) when the running engine started - it planned for that much
_last_error = ""
_last_code = ""
_job: dict = {}                       # the job in progress (see _set_stage)
_waiting = 0                          # requests queued behind the one drawing
# Seconds a warm render takes here (measured 2026-10-01 on an RTX 3060 Ti at 1024x1024, desktop apps holding
# 2.3 GB of VRAM, step cache on for fast/balanced); replaced by each real one.
_typical = {"fast": 13.0, "balanced": 34.0, "best": 97.0, "animagine4": 25.0, "pony6": 30.0, "pony6_anime": 32.0, "pony6_warm": 32.0}
_custom_register()                    # your own models join the list (add_model)
_settings = lambda: {}                # noqa: E731
_save_settings = None                 # a function that saves settings changes (move_models switches the folder)
_idle_thread: threading.Thread | None = None
_stop_requested = threading.Event()   # Stop pressed in Images (or the chat card) - checked beside should_cancel


def request_stop() -> bool:
    """Stop the picture being drawn (and any waiting for it). True if one was running."""
    if not _job and not _waiting:
        return False
    _stop_requested.set()
    return True


class ImageError(Exception):
    """A reason the user can act on. code: filter, minors, empty, not_installed, oom, stopped, crashed, refused,
    no_engine_answer, busy, timeout."""

    def __init__(self, message: str, code: str = "failed"):
        super().__init__(message)
        self.code = code


def configure(settings_fn, save_fn=None) -> None:
    global _settings, _save_settings
    _settings = settings_fn
    _save_settings = save_fn


_PATH_RE = re.compile(r"[A-Za-z]:\\[^<>\"|?*]{0,200}")


def clean_settings(s: dict) -> dict:
    out = {}
    root = str(s.get("image_root") or DEFAULT_ROOT).strip()
    out["image_root"] = root if _PATH_RE.fullmatch(root) else DEFAULT_ROOT
    extra = s.get("image_extra_roots") if isinstance(s.get("image_extra_roots"), list) else []
    out["image_extra_roots"] = [r for r in dict.fromkeys(str(x).strip() for x in extra)
                                if _PATH_RE.fullmatch(r) and r != out["image_root"]][:4]
    out["image_quality"] = s.get("image_quality") if s.get("image_quality") in QUALITIES else "balanced"
    out["image_size"] = s.get("image_size") if s.get("image_size") in SIZES else "square"
    try:
        out["image_keep_minutes"] = max(0, min(120, int(s.get("image_keep_minutes", 5))))
    except (TypeError, ValueError):
        out["image_keep_minutes"] = 5
    for key in ("image_free_gpu", "image_filter", "image_speed_boost", "image_warmup"):
        out[key] = s.get(key) if isinstance(s.get(key), bool) else True
    try:
        out["image_vram_reserve"] = round(max(0.5, min(4.0, float(s.get("image_vram_reserve", 1.5)))), 2)
    except (TypeError, ValueError):
        out["image_vram_reserve"] = 1.5
    return out


def _cfg() -> dict:
    return clean_settings(_settings() or {})


# ---- the prompt filter ------------------------------------------------------------------------------
_SEXUAL = re.compile(
    r"\b(?:nude|nudes|naked|nsfw|porn\w*|sex|sexy|sexual\w*|erotic\w*|hentai|lingerie|topless|bottomless|undress\w*|"
    r"genital\w*|breasts?|nipples?|orgasm|explicit|xxx|lewd|seductive|stripp\w*)\b"
    r"|עירום|עירומה|ערום|ערומה|סקס|מיני|פורנו|ארוטי|הלבשה תחתונה|"
    r"голая|голый|обнаж\w*|порн\w*|секс\w*|эрот\w*|нижнее бельё", re.I)
_MINOR = re.compile(
    r"\b(?:child|children|kid|kids|minor|minors|underage|under-age|preteen|pre-teen|teen|teens|teenage\w*|schoolgirl|"
    r"schoolboy|loli|lolita|shota|toddler|infant|baby|babies|young (?:girl|boy)|little (?:girl|boy)|"
    r"(?:1[0-7]|[1-9])[- ]?(?:yo|y/o|years?[- ]old))\b"
    r"|ילד|ילדה|ילדים|קטין|קטינה|נער|נערה|תינוק|"
    r"ребён\w*|ребен\w*|девочк\w*|мальчик\w*|подрост\w*|несовершеннолет\w*", re.I)
_VIOLENT_SEX = re.compile(r"\b(?:rape|raped|raping|non-?consensual|sexual assault|molest\w*)\b|אונס|изнасил\w*", re.I)
_REAL_PERSON = re.compile(r"\b(?:celebrity|famous|actress|actor|singer|politician|president|prime minister|influencer|"
                          r"real person|my (?:ex|coworker|neighbou?r|classmate|teacher|boss))\b"
                          r"|\b[A-Z][a-z]+ [A-Z][a-z]{2,}\b")


def check_prompt(prompt: str, strict: bool | None = None) -> str | None:
    """Why a prompt is refused, or None. Minors + sexual content: always. With the filter on, also sexual
    content of real, named people and sexual violence."""
    return (_check(prompt, strict) or (None, None))[0]


def _check(prompt: str, strict: bool | None = None):
    """(reason, code) or None - check_prompt with the kind of refusal (the page shows each differently)."""
    p = str(prompt or "")
    sexual = bool(_SEXUAL.search(p))
    if sexual and _MINOR.search(p):
        return "It can't make sexual images involving minors - this can't be turned off.", "minors"
    if strict is None:
        strict = _cfg()["image_filter"]
    if strict and _VIOLENT_SEX.search(p):
        return "The image filter refuses depictions of sexual violence (Settings > Images).", "filter"
    if strict and sexual and _REAL_PERSON.search(p):
        return "The image filter refuses sexual images of real, named people (Settings > Images).", "filter"
    return None


# ---- files and the engine --------------------------------------------------------------------------
def _engine_dir(root: str) -> str:
    return os.path.join(root, "engine")


def server_exe(root: str | None = None) -> str | None:
    root = root or _cfg()["image_root"]
    for dirpath, _dirs, files in os.walk(_engine_dir(root)):
        if "sd-server.exe" in files:
            return os.path.join(dirpath, "sd-server.exe")
    return None


def model_files(key: str, root: str | None = None) -> dict:
    """{part: path} for a model - each file from image_root, or from an extra root that still holds it."""
    cfg = _cfg()
    roots = [root] if root else [cfg["image_root"], *cfg["image_extra_roots"]]
    m, out = MODELS[key], {}
    for part in (("checkpoint",) if "checkpoint" in m else ("diffusion", "llm", "vae")):
        paths = [os.path.join(r, "models", m[part]) for r in roots]
        out[part] = next((p for p in paths if os.path.exists(p)), paths[0])
    return out


def installed(key: str, root: str | None = None) -> bool:
    return all(os.path.exists(p) for p in model_files(key, root).values())


def _alive() -> bool:
    return _proc is not None and _proc.poll() is None


def free_vram_mb() -> int | None:
    """Free GPU memory now (nvidia-smi), or None when it can't be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def _wait_for_vram(before: int | None, want_more_mb: int = 1500, seconds: float = 8.0) -> None:
    """After unloading the chat model the driver frees its memory a moment later: wait for it (at most a few
    seconds), so the engine plans its memory for the GPU it will really have."""
    if before is None:
        return
    deadline, prev = time.time() + seconds, None
    while time.time() < deadline:
        now = free_vram_mb()
        if now is None or now - before >= want_more_mb:
            return
        # A small chat model frees less than want_more_mb: done once something was freed and it stopped changing.
        if prev is not None and now - before > 200 and abs(now - prev) < 50:
            return
        prev = now
        time.sleep(0.4)


def yield_gpu() -> bool:
    """The chat model is about to run (agent/models): with image_free_gpu on, the image engine lets go of the GPU
    first - its files stay in Windows' cache, so the next image starts again in seconds. True if it stopped."""
    if not _alive() or _job or not _cfg()["image_free_gpu"]:
        return False
    with _lock:
        if _alive() and not _job:
            _stop()
            return True
    return False


def _stop() -> None:
    global _proc, _proc_key
    if _proc is not None:
        try:
            _proc.terminate()
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
    _proc, _proc_key = None, ""


def stop() -> None:
    with _lock:
        _stop()


def _start(key: str, progress=None, reserve: float | None = None) -> None:
    """Start sd-server for this model (stopping another one first) and wait until it answers."""
    global _proc, _proc_key, _last_error, _start_free, _log_path, _proc_reserve
    cfg = _cfg()
    exe = server_exe(cfg["image_root"])
    if not exe:
        raise ImageError(f"The image engine isn't installed in {cfg['image_root']} - run image_setup.py (Settings > Images).",
                         "not_installed")
    if not installed(key):
        raise ImageError(f"{MODELS[key]['name']} isn't downloaded yet (Settings > Images).", "not_installed")
    _stop()
    _start_free = free_vram_mb() or 0
    reserve = cfg["image_vram_reserve"] if reserve is None else reserve
    f = model_files(key)
    model_args = (["--model", f["checkpoint"], "--clip-skip", str(MODELS[key]["clip_skip"])]
                  if "checkpoint" in f else ["--diffusion-model", f["diffusion"], "--llm", f["llm"], "--vae", f["vae"]])
    args = [exe, "--listen-ip", "127.0.0.1", "--listen-port", str(PORT), *model_args, "--diffusion-fa",
            # No --offload-to-cpu: auto-fit (on by default) keeps weights in RAM and caches what fits on the GPU.
            # --max-vram -N: that cache leaves N GB free for the drawing itself (see the module docstring).
            "--max-vram", f"-{reserve:g}",
            # mmap: Windows keeps the files cached (warm() fills that cache front to back); eager: the load is its
            # own step; tiling: the decode fits beside the desktop's own use of the GPU.
            "--mmap", "--eager-load", "--vae-tiling", "--log-level", "info"]
    _log_path = os.path.join(cfg["image_root"], "sd-server.log")
    log = open(_log_path, "ab")
    _proc = subprocess.Popen(args, cwd=os.path.dirname(exe), stdout=log, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _proc_key, _proc_reserve = key, reserve
    _set_stage("loading", progress, f"Loading {MODELS[key]['name']}")
    deadline = time.time() + 240
    while time.time() < deadline:
        if not _alive():
            _last_error = f"sd-server stopped while loading (see {cfg['image_root']}\\sd-server.log)"
            raise ImageError(_last_error, "crashed")
        try:
            if httpx.get(f"{BASE}/sdcpp/v1/capabilities", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    _stop()
    raise ImageError("The image engine didn't start within 4 minutes.", "timeout")


def _idle_watch() -> None:
    while True:
        time.sleep(20)
        with _lock:
            keep = _cfg()["image_keep_minutes"] * 60
            if _alive() and not _job and time.time() - _last_used > keep:
                _stop()


def _ensure_idle_watch() -> None:
    global _idle_thread
    if _idle_thread is None or not _idle_thread.is_alive():
        _idle_thread = threading.Thread(target=_idle_watch, name="image-idle", daemon=True)
        _idle_thread.start()


# ---- warming the file cache --------------------------------------------------------------------------
WARM_FRESH_S = 600                    # a file read this recently is taken to be in Windows' cache still
WARM_CHUNK = 16 * 1024 * 1024
_warm = {"key": "", "done": 0, "total": 0, "running": False, "error": ""}
_warmed_at: dict = {}
_warm_thread: threading.Thread | None = None
_warm_lock = threading.Lock()


def _warm_todo(key: str) -> list:
    if _alive() and _proc_key == key:
        return []                      # the running engine has them mapped already
    now = time.time()
    return [p for p in dict.fromkeys(model_files(key).values())
            if os.path.exists(p) and now - _warmed_at.get(p, 0) > WARM_FRESH_S]


def _warm_files(key: str, paths: list, should_cancel=None) -> None:
    with _warm_lock:
        try:
            _warm.update(key=key, done=0, total=sum(os.path.getsize(p) for p in paths), running=True, error="")
            for p in paths:
                with open(p, "rb", buffering=0) as f:
                    while True:
                        if should_cancel and should_cancel():
                            return
                        chunk = f.read(WARM_CHUNK)
                        if not chunk:
                            break
                        _warm["done"] += len(chunk)
                _warmed_at[p] = time.time()
        except OSError as e:
            _warm["error"] = str(e)[:200]
        finally:
            _warm["running"] = False


def warm(key: str | None = None) -> bool:
    """Start reading a model's files into Windows' cache in the background (a message asked for a picture, or
    Images was opened). True if it started; False when they are fresh, already being read, or warming is off."""
    global _warm_thread
    cfg = _cfg()
    key = key if key in MODELS else cfg["image_quality"]
    if not cfg["image_warmup"] or not installed(key) or (_warm_thread and _warm_thread.is_alive()):
        return False
    todo = _warm_todo(key)
    if not todo:
        return False
    _warm_thread = threading.Thread(target=_warm_files, args=(key, todo), name="image-warm", daemon=True)
    _warm_thread.start()
    return True


def _warm_wait(key: str, progress=None, should_cancel=None) -> None:
    """Before the engine starts: finish reading this model's files front to back (joining a warm-up already
    running), reporting "x of y GB" - so the first drawing step finds them in memory, not on the disk."""
    if not _cfg()["image_warmup"]:
        return
    if _warm_thread and _warm_thread.is_alive():
        while _warm_thread.is_alive():
            if should_cancel and should_cancel():
                raise ImageError("Stopped.", "stopped")
            _report_warm(progress)
            _warm_thread.join(0.7)
    todo = _warm_todo(key)
    if not todo:
        return
    t = threading.Thread(target=_warm_files, args=(key, todo, should_cancel), name="image-warm", daemon=True)
    t.start()
    while t.is_alive():
        _report_warm(progress)
        t.join(0.7)
    if should_cancel and should_cancel():
        raise ImageError("Stopped.", "stopped")


def _report_warm(progress) -> None:
    w = dict(_warm)
    if not w["total"]:
        return
    name = MODELS.get(w["key"], {}).get("name", "the model")
    gb = 1024 ** 3                     # GB as Windows and the rest of the app count them
    _set_stage("warming", progress, f"Reading {name} from disk, {w['done'] / gb:.1f} of {w['total'] / gb:.1f} GB",
               done=w["done"], total=w["total"])


# ---- progress, read from the engine's log --------------------------------------------------------------
_BAR = re.compile(r"\|\s*(\d+)/(\d+) - ([\d.]+)(s/it|it/s)")


def _log_offset() -> int:
    try:
        return os.path.getsize(_log_path) if _log_path else 0
    except OSError:
        return 0


def _log_since(offset: int) -> str:
    if not _log_path:
        return ""
    try:
        with open(_log_path, "rb") as f:
            f.seek(offset)
            return f.read(2_000_000).decode("utf-8", errors="replace")
    except OSError:
        return ""


def parse_progress(text: str) -> dict:
    """What the engine's log says about the job since it was sent: {"stage", "image", "images", "step", "steps",
    "sec_per_step"}. The sampling bars ("| 5/8 - 2.60s/it") come after "generating image: i/n"; the decoder's
    tile bars after "decoding"."""
    out = {"stage": "encoding", "image": 1, "images": 1, "step": 0, "steps": 0, "sec_per_step": 0.0}
    for m in re.finditer(r"generating image: (\d+)/(\d+)|decoding \d+ latents|(\|\s*\d+/\d+ - [\d.]+(?:s/it|it/s))", text):
        if m.group(1):
            out.update(stage="drawing", image=int(m.group(1)), images=int(m.group(2)), step=0, steps=0)
        elif m.group(0).startswith("decoding"):
            out["stage"] = "finishing"
        elif m.group(3) and out["stage"] == "drawing":
            b = _BAR.search(m.group(3))
            if b:
                rate = float(b.group(3))
                out.update(step=int(b.group(1)), steps=int(b.group(2)),
                           sec_per_step=rate if b.group(4) == "s/it" else (1 / rate if rate else 0.0))
    return out


def _set_stage(stage: str, progress=None, text: str = "", **extra) -> None:
    if _job is not None:
        _job.update(stage=stage, text=text, **extra)
    if progress and text:
        progress(text)


# ---- generating ------------------------------------------------------------------------------------
def parse_size(size) -> tuple:
    if isinstance(size, str) and size in SIZES:
        return SIZES[size]
    m = re.fullmatch(r"\s*(\d{3,4})\s*[x×*]\s*(\d{3,4})\s*", str(size or ""))
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        w, h = max(256, min(2048, w)) // 64 * 64, max(256, min(2048, h)) // 64 * 64
        return w, h
    return SIZES[_cfg()["image_size"]]


def _ran_out_of_memory(log_text: str) -> bool:
    return bool(re.search(r"cannot make enough memory available|workspace capacity check|out of memory|"
                          r"cudaMalloc failed|failed to allocate", log_text, re.I))


def generate(prompt: str, quality: str | None = None, size=None, seed: int | None = None, negative: str = "",
             ref_image: bytes | str | None = None, should_cancel=None, progress=None, count: int = 1,
             wait_while=None, init_image: bytes | None = None, strength: float = 0.6,
             extra_meta: dict | None = None, mask_image: bytes | None = None, after=None) -> dict:
    """Make one image (or `count`, 1-4, of the same prompt with consecutive seeds). Returns the first:
    {"id", "path", "url", "prompt", "model", "quality", "seed", "width", "height", "steps", "seconds", ...} with
    "images": every one made. Raises ImageError with a reason the user can act on (and a .code).
    wait_while: a function that is true while something else needs the GPU (a chat turn running) - the request
    waits its turn instead of unloading a model mid-answer. Never passed from inside a turn.
    init_image: a picture to start from (img2img, any model) - `strength` 0..1 is how much may change.
    extra_meta: saved into each picture's json (a plan, a sequence's place).
    mask_image: with init_image, only the mask's white part is drawn (inpainting, expand()); after(png) -> png runs on
    each result before it's saved (expand() puts the original's pixels back)."""
    global _last_used, _last_error, _last_code, _job, _waiting
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ImageError("Say what the image should show.", "empty")
    refused = _check(prompt)
    if refused:
        _last_error, _last_code = refused
        raise ImageError(*refused)
    cfg = _cfg()
    key = quality if quality in MODELS else cfg["image_quality"]
    if ref_image is not None and not MODELS[key]["edit"]:
        key = "fast"                                       # editing an attached image: FLUX.2 klein
    elif quality not in MODELS:
        key = usable(key) or key                           # the default isn't on this PC: one that is
    m = MODELS[key]
    w, h = parse_size(size)
    count = max(1, min(MAX_COUNT, int(count or 1)))
    seed = int(seed) if isinstance(seed, (int, float)) or str(seed or "").lstrip("-").isdigit() else secrets.randbelow(2**31)
    asked_to_stop = should_cancel
    should_cancel = lambda: _stop_requested.is_set() or bool(asked_to_stop and asked_to_stop())   # noqa: E731
    _waiting += 1
    try:
        while not _lock.acquire(timeout=0.5):
            if should_cancel():
                raise ImageError("Stopped before it started.", "stopped")
    finally:
        _waiting -= 1
    if not _waiting:
        _stop_requested.clear()           # a Stop pressed before this one started was for the one before it
    try:
        t0 = time.time()
        _job = {"quality": key, "model": m["name"], "progress": 0.0, "started": t0, "stage": "starting", "text": "",
                "count": count, "width": w, "height": h}
        try:
            while wait_while and wait_while():
                if should_cancel and should_cancel():
                    raise ImageError("Stopped.", "stopped")
                _set_stage("waiting", progress, "Waiting for the GPU - the chat model or a Labs run is using it")
                time.sleep(1)
            if not (_alive() and _proc_key == key):
                _warm_wait(key, progress, should_cancel)
            if cfg["image_free_gpu"]:
                import llama_local
                before = free_vram_mb()
                _set_stage("freeing", None, "Making room on the GPU")
                freed = llama_local.unload_all()
                if freed:
                    _set_stage("freeing", progress, f"Unloaded {', '.join(freed)} to free the GPU")
                    _wait_for_vram(before)
            # The engine plans where weights go (GPU or RAM) when it starts. One started while the GPU was taken
            # would keep everything in RAM and crawl (a 23 s render took 369 s): start it again with the room there is now.
            free_now = free_vram_mb()
            if _alive() and _proc_key == key and free_now is not None and free_now - _start_free > 1500:
                _stop()
            if not (_alive() and _proc_key == key):
                _start(key, progress)
            _ensure_idle_watch()
            images = None
            for attempt in (1, 2):
                try:
                    images = _render(key, prompt, negative, w, h, seed, count, ref_image, should_cancel, progress, cfg,
                                     init_image=init_image, strength=strength, mask_image=mask_image)
                    break
                except _OutOfMemory:
                    if attempt == 2 or _proc_reserve >= 4.0:
                        raise ImageError("It ran out of GPU memory - another program is using the GPU. Try a smaller "
                                         "size or Fast, or close a game or video.", "oom")
                    more = min(4.0, _proc_reserve + 1.0)
                    _set_stage("loading", progress, f"Out of GPU memory - starting again with {more:g} GB kept free")
                    _start(key, progress, reserve=more)
            _set_stage("saving", None, "Saving")
            made = []
            os.makedirs(IMAGES_DIR, exist_ok=True)
            total = round(time.time() - t0, 1)
            for i, data in enumerate(images):
                if after:
                    data = after(data)
                iid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
                path = os.path.join(IMAGES_DIR, f"{iid}.png")
                with open(path, "wb") as f:
                    f.write(data)
                meta = {"id": iid, "prompt": prompt, "negative": negative or "", "model": m["name"], "quality": key,
                        "seed": seed + i, "width": w, "height": h, "steps": m["steps"], "seconds": total,
                        "edit": ref_image is not None, "fav": False, "created": time.time(), **(extra_meta or {})}
                with open(os.path.join(IMAGES_DIR, f"{iid}.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=1)
                made.append({**meta, "path": path, "url": f"/api/images/{iid}.png"})
            _last_error = _last_code = ""
            return {**made[0], "images": made}
        except ImageError as e:
            _last_error, _last_code = str(e), e.code
            raise
        except httpx.HTTPError as e:
            _last_error, _last_code = f"The image engine didn't answer: {e}", "no_engine_answer"
            raise ImageError(_last_error, "no_engine_answer")
        finally:
            _last_used = time.time()
            _job = {}
    finally:
        _lock.release()


class _OutOfMemory(Exception):
    pass


def _render(key, prompt, negative, w, h, seed, count, ref_image, should_cancel, progress, cfg, init_image=None,
            strength=0.6, mask_image=None) -> list:
    """Send one job to the running engine and wait for it, reporting progress from its log. Returns the PNG bytes
    of each picture. Raises _OutOfMemory when the engine couldn't find room on the GPU."""
    m = MODELS[key]
    if m.get("prefix") and "score_9" not in prompt.lower():
        prompt = f"{m['prefix']}, {prompt.lstrip(' ,')}"
    if m.get("suffix") and m["suffix"].lower() not in prompt.lower():
        prompt = f"{prompt.rstrip(' ,')}, {m['suffix']}"
    body = {"prompt": prompt, "negative_prompt": negative or m.get("negative", ""), "width": w, "height": h, "seed": seed,
            "batch_count": count, "output_format": "png",
            "sample_params": {"sample_steps": m["steps"], "guidance": {"txt_cfg": m["cfg"]}}}
    if m["sampler"]:
        body["sample_params"]["sample_method"] = m["sampler"]
    if m.get("scheduler"):
        body["sample_params"]["scheduler"] = m["scheduler"]
    if cfg["image_speed_boost"] and m["step_cache"]:
        body["cache_mode"] = "easycache"
    if ref_image is not None:                     # a data URL (an attached file) or raw bytes
        body["ref_images"] = [ref_image if isinstance(ref_image, str) else
                              "data:image/png;base64," + base64.b64encode(ref_image).decode()]
    if init_image is not None:                    # img2img: start from this picture (sd-server's init_image)
        body["init_image"] = "data:image/png;base64," + base64.b64encode(init_image).decode()
        body["strength"] = max(0.05, min(1.0, float(strength)))
        if mask_image is not None:                # inpainting: only the mask's white part is drawn (sd-server mask_image)
            body["mask_image"] = "data:image/png;base64," + base64.b64encode(mask_image).decode()
    offset = _log_offset()
    r = httpx.post(f"{BASE}/sdcpp/v1/img_gen", json=body, timeout=30)
    if r.status_code not in (200, 202):
        raise ImageError(f"The image engine refused the request: HTTP {r.status_code} {r.text[:200]}", "refused")
    jid = r.json().get("id")
    t_job = time.time()
    _job.update(id=jid, t_job=t_job)
    _set_stage("encoding", progress, f"{m['name']}: reading the prompt")
    est = _typical.get(key, 30.0) * count
    while True:
        p = parse_progress(_log_since(offset))
        if should_cancel and should_cancel():
            # The engine can't stop a picture it is drawing (capabilities: cancel_generating false) - closing it
            # does; its files stay cached, so the next picture starts again in seconds.
            where = f" at step {p['step']} of {p['steps']}" if p["steps"] else ""
            try:
                httpx.post(f"{BASE}/sdcpp/v1/jobs/{jid}/cancel", timeout=5)
            except httpx.HTTPError:
                pass
            _stop()
            raise ImageError(f"Stopped{where}. Nothing was saved.", "stopped")
        time.sleep(0.7)
        if not _alive():
            raise ImageError("The image engine stopped while drawing - its log says why (Settings > Images).", "crashed")
        st = httpx.get(f"{BASE}/sdcpp/v1/jobs/{jid}", timeout=10).json()
        status = st.get("status")
        spent = time.time() - t_job
        p = parse_progress(_log_since(offset))
        _job.update(step=p["step"], steps=p["steps"], image=p["image"], images=p["images"])
        if p["steps"] and p["sec_per_step"]:
            left_steps = (p["steps"] - p["step"]) + (p["images"] - p["image"]) * p["steps"]
            left = left_steps * p["sec_per_step"] + 3.5 * (p["images"] - p["image"] + 1)
            est = spent + left
        _job["progress"] = round(min(0.97, spent / est), 3) if est else 0.0
        _job["left"] = round(max(0.0, est - spent), 1)
        if p["stage"] == "drawing" and p["steps"]:
            of = f"picture {p['image']} of {p['images']}, " if p["images"] > 1 else ""
            _set_stage("drawing", progress, f"{m['name']}: {of}step {p['step']} of {p['steps']}, about {_job['left']:.0f} s left")
        elif p["stage"] == "finishing":
            _set_stage("finishing", progress, f"{m['name']}: finishing")
        if status == "completed":
            break
        if status in ("failed", "cancelled"):
            if _ran_out_of_memory(_log_since(offset)):
                raise _OutOfMemory()
            err = st.get("error") or {}
            raise ImageError(f"The image engine failed: {err.get('message') or status}", "failed")
    _typical[key] = (time.time() - t_job) / max(1, count)          # the next estimate
    images = (st.get("result") or {}).get("images") or []
    if not images:
        raise ImageError("The image engine finished without an image.", "failed")
    out = []
    for item in images:
        raw = item.get("b64_json") if isinstance(item, dict) else item
        if isinstance(raw, dict):
            raw = raw.get("data") or raw.get("b64_json")
        out.append(base64.b64decode(str(raw).split(",", 1)[-1]))
    return out


def count() -> int:
    """Pictures in the gallery (not the trash) - the sidebar shows it beside Images."""
    try:
        return sum(1 for n in os.listdir(IMAGES_DIR) if n.endswith(".png"))
    except OSError:
        return 0


def status() -> dict:
    cfg = _cfg()
    w = dict(_warm)
    return {"root": cfg["image_root"], "extra_roots": cfg["image_extra_roots"], "engine": bool(server_exe(cfg["image_root"])),
            "count": count(),
            "running": _alive(), "loaded": MODELS[_proc_key]["name"] if _alive() and _proc_key else None,
            "job": {k: v for k, v in _job.items() if k != "t_job"} or (
                {"stage": "planning", "text": _planning["text"], "progress": 0.02} if _planning.get("text") else {}),
            "waiting": _waiting,
            "warm": w if w["running"] else None, "last_error": _last_error, "last_code": _last_code,
            "move": dict(_move) if _move.get("state") not in (None, "idle") else None,
            "models": [{"quality": k, **{x: m[x] for x in ("name", "by", "license", "steps", "edit", "about", "step_cache")},
                        "installed": installed(k), "seconds": round(_typical.get(k, 0)),
                        "custom": bool(m.get("custom")), "kind": m.get("kind", ""),
                        **({"file": m["checkpoint"]} if m.get("custom") else {})} for k, m in MODELS.items()],
            "default": usable(cfg["image_quality"]),        # what a picture uses when none is picked
            "sizes": {k: f"{w_}x{h_}" for k, (w_, h_) in SIZES.items()}}


# ---- the gallery -------------------------------------------------------------------------------------
_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")


def _trash_dir() -> str:
    return os.path.join(IMAGES_DIR, "trash")


def _read_meta(folder: str, iid: str) -> dict | None:
    try:
        with open(os.path.join(folder, f"{iid}.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find(iid: str) -> tuple[dict, str] | None:
    """(meta, folder) of a made image, in the gallery or the trash."""
    if not _ID.fullmatch(str(iid or "")):
        return None
    for folder in (IMAGES_DIR, _trash_dir()):
        meta = _read_meta(folder, iid)
        if meta and os.path.exists(os.path.join(folder, f"{iid}.png")):
            return meta, folder
    return None


# Presets of one model that the gallery and the model picker treat as one family (the base key first).
FAMILIES = {"pony6": ("pony6", "pony6_anime", "pony6_warm")}
THUMB_PX = 512


def _thumb_dir() -> str:
    return os.path.join(IMAGES_DIR, ".thumbs")


def _urls(meta: dict, trashed: bool) -> dict:
    return {"url": f"/api/images/{'trash/' if trashed else ''}{meta['id']}.png",
            "thumb": f"/api/images/thumb/{meta['id']}.webp"}


def thumbnail(iid: str) -> str | None:
    """A small WebP of a made image for the gallery grid (cached in data/images/.thumbs, rebuilt when the PNG is
    newer). The tiles are ~200 px, so loading every full PNG made the Images tab slow to open and scroll."""
    got = find(iid)
    if not got:
        return None
    src, dst = os.path.join(got[1], iid + ".png"), os.path.join(_thumb_dir(), iid + ".webp")
    try:
        if os.path.getmtime(dst) >= os.path.getmtime(src):
            return dst
    except OSError:
        pass
    from PIL import Image
    os.makedirs(_thumb_dir(), exist_ok=True)
    tmp = dst + ".tmp"
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
        im.save(tmp, "WEBP", quality=82, method=4)
    os.replace(tmp, dst)
    return dst


def recent(limit: int = 60, q: str = "", quality: str = "", fav: bool = False, trashed: bool = False,
           group: bool = False) -> list:
    """The gallery, newest first. quality may be a family (all its presets). With group, the pictures of one
    sequence (Angles/Frames) come back as one entry - its first picture, with group = {id, mode, count, ids}."""
    folder = _trash_dir() if trashed else IMAGES_DIR
    out, words = [], [w for w in str(q or "").lower().split() if w]
    quals = FAMILIES.get(quality, (quality,)) if quality else ()
    groups: dict = {}
    try:
        names = sorted((n for n in os.listdir(folder) if n.endswith(".json")), reverse=True)
    except OSError:
        return []
    for n in names:
        meta = _read_meta(folder, n[:-5])
        if not meta or "id" not in meta:
            continue
        if quals and meta.get("quality") not in quals or fav and not meta.get("fav"):
            continue
        hay = f"{meta.get('prompt', '')} {meta.get('model', '')}".lower()
        if words and not all(w in hay for w in words):
            continue
        item = {**meta, **_urls(meta, trashed), "trashed": trashed}
        seq = meta.get("seq") or {}
        if group and seq.get("id"):
            at = groups.get(seq["id"])
            if at is not None:
                g = out[at]["group"]
                g["count"] += 1
                g["ids"].append(meta["id"])
                if seq.get("index", 0) < (out[at].get("seq") or {}).get("index", 0):
                    out[at] = {**item, "group": g}           # the cover is the sequence's first picture
                continue
            groups[seq["id"]] = len(out)
            item["group"] = {"id": seq["id"], "mode": seq.get("mode") or "angles", "count": 1, "ids": [meta["id"]]}
        out.append(item)
        if len(out) >= limit:
            break
    return out


def set_favorite(iid: str, on: bool) -> dict:
    got = find(iid)
    if not got:
        raise ImageError("No such image.", "missing")
    meta, folder = got
    meta["fav"] = bool(on)
    with open(os.path.join(folder, f"{iid}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def _move_pair(iid: str, src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for ext in (".png", ".json"):
        os.replace(os.path.join(src, iid + ext), os.path.join(dst, iid + ext))


def trash(iid: str) -> None:
    """Into data/images/trash - never deleted for good here (restore() brings it back)."""
    got = find(iid)
    if not got or got[1] != IMAGES_DIR:
        raise ImageError("No such image.", "missing")
    _move_pair(iid, IMAGES_DIR, _trash_dir())


def restore(iid: str) -> None:
    got = find(iid)
    if not got or got[1] != _trash_dir():
        raise ImageError("That image isn't in the trash.", "missing")
    _move_pair(iid, _trash_dir(), IMAGES_DIR)


def purge(iid: str) -> int:
    """Delete one picture for good - only from the trash, and only on the user's click ("Delete forever").
    Returns the bytes freed."""
    got = find(iid)
    if not got or got[1] != _trash_dir():
        raise ImageError("Only a picture in the trash can be deleted for good.", "missing")
    freed = 0
    for path in (os.path.join(_trash_dir(), iid + ".png"), os.path.join(_trash_dir(), iid + ".json"),
                 os.path.join(_thumb_dir(), iid + ".webp")):
        try:
            freed += os.path.getsize(path)
            os.remove(path)
        except FileNotFoundError:
            pass
    return freed


def empty_trash() -> dict:
    """Delete every picture in the trash for good ("Empty trash", after the user confirms)."""
    count = freed = 0
    try:
        names = [n[:-5] for n in os.listdir(_trash_dir()) if n.endswith(".json")]
    except OSError:
        names = []
    for iid in names:
        if _ID.fullmatch(iid):
            try:
                freed += purge(iid)
                count += 1
            except (ImageError, OSError):
                pass
    return {"count": count, "freed": freed}


def image_bytes(iid: str) -> bytes:
    got = find(iid)
    if not got:
        raise ImageError("No such image.", "missing")
    with open(os.path.join(got[1], f"{iid}.png"), "rb") as f:
        return f.read()


def again(iid: str, mode: str = "again", **kw) -> dict:
    """The same prompt, quality and size again: "again" one new seed, "variations" two new seeds."""
    got = find(iid)
    if not got:
        raise ImageError("No such image.", "missing")
    meta = got[0]
    return generate(meta["prompt"], meta.get("quality"), f"{meta['width']}x{meta['height']}", None, meta.get("negative", ""),
                    count=2 if mode == "variations" else 1, **kw)


# ---- think first, and sequences (Pony V6 and its presets - image_plan.py) -------------------------------------
PONY_BASE_TAGS = "score_9, score_8_up, score_7_up, source_anime"
ANGLES = ["from front", "three-quarter view", "from side, profile", "from behind", "from above", "from below"]


def _pony_prompt(key: str, tags: str) -> str:
    """Planned tags as a prompt: base Pony gets its score tags here (the presets add theirs through `prefix`)."""
    return tags if MODELS[key].get("prefix") else f"{PONY_BASE_TAGS}, {tags}"


def _join(*parts) -> str:
    return ", ".join(p.strip(" ,") for p in parts if p and p.strip(" ,"))


_planning: dict = {}                  # a planned picture or a sequence in progress (status() shows it between drawings)


def _plan_progress(progress, text):
    _planning.update(text=text, at=time.time())
    if progress:
        progress(text)


def generate_planned(*args, **kw) -> dict:
    try:
        return _generate_planned(*args, **kw)
    finally:
        _planning.clear()


def generate_sequence(*args, **kw) -> dict:
    try:
        return _generate_sequence(*args, **kw)
    finally:
        _planning.clear()


def _generate_planned(request: str, quality: str, size=None, seed=None, negative: str = "", count: int = 1,
                      check_result: bool = True, progress=None, wait_while=None, should_cancel=None) -> dict:
    """Think first: the chat model plans the picture (image_plan.plan), Pony draws it, and - with check_result - a
    vision model checks it against the plan's must-be-true points; a clear mismatch gets one corrected attempt
    (the first picture stays in the gallery too). The plan and the check are saved with the picture."""
    import image_plan
    if quality not in image_plan.PONY_KEYS:
        raise ImageError("Think first is for Pony V6 and its presets.", "refused")
    refused = _check(request)
    if refused:
        raise ImageError(*refused)
    stop()                                              # the chat model needs the GPU to plan
    _plan_progress(progress, "Planning the picture")
    try:
        p = image_plan.plan(request)
    except Exception as e:
        raise ImageError(f"Planning didn't work: {e}", "failed")
    base_neg = negative or MODELS[quality].get("negative", "")
    plan_meta = {"request": request, "thinking": p["thinking"], "tags": p["tags"], "checks": p["checks"],
                 "planner": p.get("model")}
    img = generate(_pony_prompt(quality, p["tags"]), quality, size, seed, _join(base_neg, p["negative"]),
                   count=count, progress=progress, wait_while=wait_while, should_cancel=should_cancel,
                   extra_meta={"plan": plan_meta})
    if not check_result:
        return {**img, "plan": plan_meta}
    stop()                                              # the vision model needs the GPU to look
    _plan_progress(progress, "Checking the picture against the plan")
    try:
        from PIL import Image
        c = image_plan.check(Image.open(img["path"]).convert("RGB"), request, p["checks"])
    except Exception as e:
        return {**img, "plan": plan_meta, "check": {"ok": None, "problems": [f"Couldn't check: {str(e)[:160]}"]}}
    _update_meta(img["id"], {"check": c})
    if c["ok"] or not (c["add_tags"] or c["add_negative"]):
        return {**img, "plan": plan_meta, "check": c}
    _plan_progress(progress, "Fixing: " + "; ".join(c["problems"])[:120])
    fixed = generate(_pony_prompt(quality, _join(p["tags"], c["add_tags"])), quality, size, None,
                     _join(base_neg, p["negative"], c["add_negative"]), progress=progress, wait_while=wait_while,
                     should_cancel=should_cancel, extra_meta={"plan": {**plan_meta, "fixed_from": img["id"],
                                                                        "fix": c["problems"]}})
    return {**fixed, "plan": plan_meta, "check": c, "first": {k: img[k] for k in ("id", "url")}}


def _update_meta(iid: str, extra: dict) -> None:
    path = os.path.join(IMAGES_DIR, f"{iid}.json")
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        meta.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    except (OSError, ValueError):
        pass


def _generate_sequence(request: str, quality: str, mode: str = "angles", count: int = 4, size=None, seed=None,
                       negative: str = "", think: bool = True, strength: float = 0.7, progress=None, wait_while=None,
                       should_cancel=None) -> dict:
    """A set of pictures of the same character/scene. "angles": the same moment from different camera angles - the
    same identity tags and seed, a different camera each time. "frames": steps of one motion - each frame starts
    from the one before (img2img, `strength`), so the character, outfit and colours carry over.
    think: the chat model plans the identity and each picture's change (needed for "frames"). Returns
    {"sequence", "images": [...], "plan"}."""
    import image_plan
    if quality not in image_plan.PONY_KEYS:
        raise ImageError("Sequences are for Pony V6 and its presets.", "refused")
    if mode not in ("angles", "frames"):
        raise ImageError("Choose angles or frames.", "refused")
    refused = _check(request)
    if refused:
        raise ImageError(*refused)
    count = max(2, min(8, int(count or 4)))
    if think or mode == "frames":
        stop()
        _plan_progress(progress, f"Planning {count} {'angles' if mode == 'angles' else 'frames'}")
        try:
            plan = image_plan.plan_sequence(request, mode, count)
        except Exception as e:
            raise ImageError(f"Planning didn't work: {e}", "failed")
    else:
        plan = {"thinking": "", "identity": request, "negative": "", "model": None,
                "pictures": [{"label": a, "tags": a} for a in (ANGLES * 2)[:count]]}
    seed = int(seed) if str(seed or "").lstrip("-").isdigit() else secrets.randbelow(2**31)
    seq_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    neg = _join(negative or MODELS[quality].get("negative", ""), plan["negative"])
    made, prev = [], None
    for i, pic in enumerate(plan["pictures"]):
        _plan_progress(progress, f"Picture {i + 1} of {count}: {pic['label']}")
        # Angles: the identity tags weighted up a little - a side view lost the red hair of the other three
        # (2026-10-14); "(tags:1.15)" is sd.cpp's attention weighting.
        identity = f"({plan['identity']}:1.15)" if mode == "angles" else plan["identity"]
        prompt = _pony_prompt(quality, _join(identity, pic["tags"]))
        # Every frame starts from frame 1, not from the one before: chaining each from its neighbour compounded the
        # colour (each frame redder than the last) and, at 0.55, held the pose still (2026-10-14).
        init = prev if (mode == "frames" and prev is not None) else None
        img = generate(prompt, quality, size, seed, neg, progress=progress, wait_while=wait_while,
                       should_cancel=should_cancel, init_image=init, strength=strength,
                       extra_meta={"seq": {"id": seq_id, "index": i, "count": count, "label": pic["label"],
                                           "mode": mode, "request": request}})
        made.append(img)
        if prev is None:
            with open(img["path"], "rb") as f:
                prev = f.read()                         # frame 1: the start of every later frame
    return {"sequence": seq_id, "images": made, "plan": plan, **{k: made[0][k] for k in ("id", "url")}}


def sequence(seq_id: str) -> list:
    """The pictures of one sequence, in order (gallery and trash)."""
    out = []
    for folder, trashed in ((IMAGES_DIR, False), (_trash_dir(), True)):
        try:
            names = [n for n in os.listdir(folder) if n.endswith(".json")]
        except OSError:
            continue
        for n in names:
            meta = _read_meta(folder, n[:-5])
            if meta and (meta.get("seq") or {}).get("id") == seq_id:
                out.append({**meta, "trashed": trashed, **_urls(meta, trashed)})
    return sorted(out, key=lambda m: m["seq"]["index"])


# ---- Expand: draw past a picture's edges (outpainting, 1.4.5) ----------------------------------------------------
EXPAND_SIDES = ("left", "right", "top", "bottom")
EXPAND_MAX_SIDE = 1536            # px: what the models here draw well in 8 GB (measured at 1024-1216 for every model)
EXPAND_MAX_PIXELS = 1536 * 1024
# px the new area reaches into the old one. Measured live (29 Oct 2026): the engine keeps the unmasked picture almost
# exactly (mean change 1.8-2.5 of 255 on Balanced and Fast, 7.6 on DreamShaper), so the model continues from the real
# edge. A wide band (48-112 px) given back to the model got NEW things drawn in it - a second cabin, a ghost tree - and
# blending them with the original made the seam. So only a few pixels are redrawn, enough to hide the engine's edge.
FEATHER = 12


def expand_plan(w: int, h: int, sides, amount: float) -> dict:
    """The new canvas: {"w", "h", "left", "top", "scale"} - each chosen side grows by `amount` of the picture's size,
    the result rounded to 64 px and scaled down (picture included) to stay within what the models draw well."""
    sides = [s for s in EXPAND_SIDES if s in (sides or [])] or list(EXPAND_SIDES)
    amount = max(0.1, min(1.0, float(amount)))
    add_w, add_h = int(w * amount), int(h * amount)
    left = add_w if "left" in sides else 0
    top = add_h if "top" in sides else 0
    nw = w + left + (add_w if "right" in sides else 0)
    nh = h + top + (add_h if "bottom" in sides else 0)
    scale = min(1.0, EXPAND_MAX_SIDE / max(nw, nh), (EXPAND_MAX_PIXELS / (nw * nh)) ** .5)
    W, H = max(256, int(nw * scale) // 64 * 64), max(256, int(nh * scale) // 64 * 64)
    return {"w": W, "h": H, "left": round(left * W / nw), "top": round(top * H / nh), "scale": W / nw,
            "pw": round(w * W / nw), "ph": round(h * H / nh), "sides": sides}


def _expand_images(png: bytes, plan: dict) -> tuple:
    """(start picture, mask, the original placed on the canvas, its alpha) for expand()."""
    import io
    from PIL import Image, ImageFilter
    src = Image.open(io.BytesIO(png)).convert("RGB").resize((plan["pw"], plan["ph"]), Image.LANCZOS)
    W, H, x, y = plan["w"], plan["h"], plan["left"], plan["top"]
    # The start: the picture's own edges stretched outward and blurred - colours and light to continue from, not noise.
    canvas = src.resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(32))
    canvas.paste(src, (x, y))
    # The mask: white = draw. The new area, reaching FEATHER px into the picture, softened.
    f = min(FEATHER, plan["pw"] // 6, plan["ph"] // 6)
    mask = Image.new("L", (W, H), 255)
    keep = Image.new("L", (W, H), 0)
    inner = (x + (f if x else 0), y + (f if y else 0),
             x + plan["pw"] - (f if x + plan["pw"] < W else 0), y + plan["ph"] - (f if y + plan["ph"] < H else 0))
    mask.paste(0, inner)
    keep.paste(255, (x, y, x + plan["pw"], y + plan["ph"]))
    mask = mask.filter(ImageFilter.GaussianBlur(f / 4))
    # Where the original's pixels go back after drawing: all of it, faded out across the feather - a wide fade, so
    # the eye can't find the line.
    alpha = Image.new("L", (W, H), 0)
    alpha.paste(255, inner)
    alpha = alpha.filter(ImageFilter.GaussianBlur(f / 2))
    alpha = Image.composite(alpha, Image.new("L", (W, H), 0), keep)
    placed = Image.new("RGB", (W, H)); placed.paste(src, (x, y))

    def to_png(im):
        b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()
    return to_png(canvas), to_png(mask), placed, alpha


def match_colors(drawn, original, plan: dict):
    """The drawn canvas with its colours matched to the original's, per channel (mean and spread), measured on the
    band of the picture next to the new area - where both exist. Fixes the slight relighting that made seams."""
    import numpy as np
    from PIL import Image
    d = np.asarray(drawn, dtype=np.float32)
    o = np.asarray(original, dtype=np.float32)
    x, y, pw, ph, W, H = plan["left"], plan["top"], plan["pw"], plan["ph"], plan["w"], plan["h"]
    band = np.zeros(d.shape[:2], bool)
    b = min(2 * FEATHER, pw // 3, ph // 3)
    if x:
        band[y:y + ph, x:x + b] = True
    if x + pw < W:
        band[y:y + ph, x + pw - b:x + pw] = True
    if y:
        band[y:y + b, x:x + pw] = True
    if y + ph < H:
        band[y + ph - b:y + ph, x:x + pw] = True
    if band.sum() < 1000:
        return drawn
    dm, ds = d[band].mean(0), d[band].std(0) + 1e-3
    om, os_ = o[band].mean(0), o[band].std(0) + 1e-3
    gain = np.clip(os_ / ds, 0.7, 1.4)                       # never a wild change
    fixed = np.clip((d - dm) * gain + om, 0, 255).astype(np.uint8)
    return Image.fromarray(fixed)


EXPAND_STRENGTH = 0.75     # live test: 1.0 relit the new area (a lighter band at the join); 0.75 keeps the start's colours


def expand(iid: str, sides=None, amount: float = 0.5, prompt: str = "", quality: str | None = None,
           strength: float = EXPAND_STRENGTH, **kw) -> dict:
    """The picture with more of the scene around it, as a new image (the original stays). The new area is drawn by
    the picture's own model from its own prompt (plus `prompt`, if given); the original's pixels are put back
    exactly, blended across a narrow seam."""
    import io
    from PIL import Image
    got = find(iid)
    if not got:
        raise ImageError("No such image.", "missing")
    meta = got[0]
    plan = expand_plan(int(meta["width"]), int(meta["height"]), sides, amount)
    start, mask, placed, alpha = _expand_images(image_bytes(iid), plan)

    def put_back(png: bytes) -> bytes:
        out = Image.open(io.BytesIO(png)).convert("RGB")
        if out.size != placed.size:
            out = out.resize(placed.size, Image.LANCZOS)
        import numpy as np
        a = np.asarray(alpha) > 250                              # deep inside the original: the engine should keep it
        drift = float(np.abs(np.asarray(out, np.float32)[a] - np.asarray(placed, np.float32)[a]).mean()) if a.any() else 0.0
        expand.last_drift = round(drift, 1)                      # 0 = kept the picture; ~40+ = redrew it all
        out = match_colors(out, placed, plan)
        out = Image.composite(placed, out, alpha)
        b = io.BytesIO(); out.save(b, "PNG"); return b.getvalue()
    text = ", ".join(p for p in (str(prompt or "").strip(), meta.get("prompt") or "") if p) if str(prompt or "").strip() \
        else surroundings_prompt(meta.get("prompt") or "")
    quality = quality if quality in MODELS else meta.get("quality") if meta.get("quality") in MODELS else None
    extra = {"expanded_from": iid, "expand": {"sides": plan["sides"], "amount": amount}}
    return generate(text, quality, f"{plan['w']}x{plan['h']}", None, meta.get("negative", ""), init_image=start,
                    strength=strength, mask_image=mask, after=put_back, extra_meta=extra, **kw)


def surroundings_prompt(original: str) -> str:
    """What the new area of an expanded picture shows when you didn't say (1.4.6): the picture's setting and style
    without its subject. The whole original prompt drew the subject again in the new strip - a second cabin beside
    the cabin (1.4.5's known issue). Prompts name the subject first ("a red sports car, on a desert road, golden
    hour"), so the first part goes; the rest is the place, the light and the style. A one-part prompt ("a red
    wall") is the scene itself and stays."""
    parts = [p.strip() for p in re.split(r"[,;]", str(original or "")) if p.strip()]
    rest = ", ".join(parts[1:] if len(parts) > 1 else parts)
    return f"{rest}, more of the surroundings, background, seamless" if rest else "more of the same surroundings, background, seamless"


def edit(iid: str, prompt: str, **kw) -> dict:
    """A made image as the start of a new one (FLUX.2 klein edits it)."""
    got = find(iid)
    if not got:
        raise ImageError("No such image.", "missing")
    meta = got[0]
    return generate(prompt, "fast", f"{meta['width']}x{meta['height']}", None, "", ref_image=image_bytes(iid), **kw)


# ---- moving the models to another drive ----------------------------------------------------------------
MIN_FREE_AFTER = 5 * 1024 ** 3        # a move never leaves the destination drive with less than this free
_move: dict = {"state": "idle"}
_move_cancel = threading.Event()


def _manifest(root: str) -> dict:
    try:
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def plan_move(dest: str, qualities: list) -> dict:
    """What moving these models (and the engine) to `dest` would copy, and whether the drive has room."""
    cfg = _cfg()
    src = cfg["image_root"]
    dest = str(dest or "").strip().rstrip("\\")
    if not _PATH_RE.fullmatch(dest):
        return {"ok": False, "why": "Choose a folder like D:\\AgenticWorkspace\\Images."}
    if os.path.normcase(os.path.abspath(dest)) == os.path.normcase(os.path.abspath(src)):
        return {"ok": False, "why": "That is where the files are already."}
    # Your own models (add_model) stay where you keep them: only the app's own files move.
    qualities = [q for q in qualities if q in MODELS and installed(q) and not MODELS[q].get("custom")]
    if not qualities:
        return {"ok": False, "why": "Pick at least one downloaded model."}
    files, seen = [], set()
    for d, _sub, names in os.walk(_engine_dir(src)):
        for n in names:
            path = os.path.join(d, n)
            files.append({"src": path, "rel": os.path.relpath(path, src), "size": os.path.getsize(path), "sha256": None})
    manifest = _manifest(src)
    for q in qualities:
        for part, path in model_files(q).items():
            if path in seen:
                continue
            seen.add(path)
            rel = os.path.join("models", os.path.basename(path))
            files.append({"src": path, "rel": rel, "size": os.path.getsize(path),
                          "sha256": (manifest.get(rel.replace("\\", "/")) or {}).get("sha256")})
    total = sum(f["size"] for f in files)
    try:
        free = shutil.disk_usage(os.path.splitdrive(dest)[0] + "\\").free
    except OSError:
        return {"ok": False, "why": f"The drive {os.path.splitdrive(dest)[0]} can't be read."}
    ok = free - total >= MIN_FREE_AFTER
    return {"ok": ok, "dest": dest, "qualities": qualities, "files": files, "total": total, "free": free,
            "free_after": free - total,
            "why": "" if ok else f"That needs {total / 1e9:.1f} GB and would leave less than "
                                 f"{MIN_FREE_AFTER / 1024 ** 3:.0f} GB free on {os.path.splitdrive(dest)[0]}."}


def move_models(dest: str, qualities: list) -> dict:
    """Copy the engine and these models to `dest`, checking every file (models against the SHA-256 they were
    published with, the engine against its originals), then switch image_root there. Nothing is deleted from the
    old folder: it stays in image_extra_roots for the models left there. Runs in the background; status() shows it."""
    if _move.get("state") == "running":
        raise ImageError("A move is already running.", "busy")
    plan = plan_move(dest, qualities)
    if not plan["ok"]:
        raise ImageError(plan["why"], "refused")
    _move_cancel.clear()
    _move.clear()
    _move.update(state="running", dest=plan["dest"], qualities=plan["qualities"], total=plan["total"], done=0,
                 file="", stage="copying", error="", files_done=0, files=len(plan["files"]))
    threading.Thread(target=_move_run, args=(plan,), name="image-move", daemon=True).start()
    return {k: v for k, v in plan.items() if k != "files"}


def cancel_move() -> None:
    _move_cancel.set()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(WARM_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _move_run(plan: dict) -> None:
    cfg = _cfg()
    part = None
    try:
        for item in plan["files"]:
            target = os.path.join(plan["dest"], item["rel"])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            _move.update(file=os.path.basename(item["src"]), stage="copying")
            if os.path.exists(target) and item["sha256"] and _sha256(target) == item["sha256"]:
                _move["done"] += item["size"]                 # already there and intact (an earlier move)
                _move["files_done"] += 1
                continue
            part = target + ".part"
            src_hash = hashlib.sha256()
            with open(item["src"], "rb", buffering=0) as fin, open(part, "wb") as fout:
                for chunk in iter(lambda: fin.read(WARM_CHUNK), b""):
                    if _move_cancel.is_set():
                        raise ImageError("Cancelled - nothing was switched.", "stopped")
                    src_hash.update(chunk)
                    fout.write(chunk)
                    _move["done"] += len(chunk)
            _move["stage"] = "checking"
            expected = item["sha256"] or src_hash.hexdigest()
            if item["sha256"] and src_hash.hexdigest() != item["sha256"]:
                raise ImageError(f"{os.path.basename(item['src'])} on the old drive doesn't match its published SHA-256 - "
                                 "it was not copied. Run image_setup.py to repair it.", "failed")
            if _sha256(part) != expected:
                raise ImageError(f"The copy of {os.path.basename(item['src'])} doesn't match - nothing was switched; "
                                 "the old folder is still in use.", "failed")
            os.replace(part, target)
            part = None
            _move["files_done"] += 1
        manifest = _manifest(cfg["image_root"])
        if manifest:
            with open(os.path.join(plan["dest"], "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        with _lock:
            _stop()
            left = [q for q in MODELS if q not in plan["qualities"] and installed(q)]
            extra = ([cfg["image_root"]] if left else []) + [r for r in cfg["image_extra_roots"] if r != plan["dest"]]
            if _save_settings:
                _save_settings({"image_root": plan["dest"], "image_extra_roots": extra})
        _move.update(state="done", stage="done", file="")
    except ImageError as e:
        _move.update(state="cancelled" if e.code == "stopped" else "failed", error=str(e))
    except OSError as e:
        _move.update(state="failed", error=f"Copying failed: {e}")
    finally:
        if part and os.path.exists(part):
            try:
                os.remove(part)                               # our own unfinished copy, never a user file
            except OSError:
                pass


# ---- the assistant's tool --------------------------------------------------------------------------
_WANTS = re.compile(
    r"\b(?:draw|paint|sketch|illustrate|render|generate|create|make|design)\b[^.?!\n]{0,40}\b(?:image|picture|photo|"
    r"illustration|drawing|painting|poster|logo|wallpaper|portrait|icon|art|artwork|sketch)\b"
    r"|\b(?:image|picture|photo|drawing) of\b|\bdraw me\b|\bimagine\b"
    r"|^\W*(?:(?:please|can you|could you)\s+)?(?:draw|paint|sketch)\s+(?:a|an|the|some|me|my|us)\b"   # "draw a cat"
    r"|צייר|ציירי|תצייר|תמונה של|צור תמונה|תייצר תמונה|תעשה תמונה|"
    r"нарисуй|нарисуйте|сгенерируй (?:изображение|картинку)|картинк\w* с|изображение", re.I)


def wants_image(message: str) -> bool:
    return bool(_WANTS.search(str(message or "")))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def installed_models() -> list:
    """[{"key", "name", "about"}] of the picture models on this PC, yours included - what a chat can ask for."""
    return [{"key": k, "name": m["name"], "about": m.get("about", "")} for k, m in MODELS.items() if installed(k)]


def resolve_model(name: str) -> str | None:
    """A model named in a chat ("DreamShaper", "z image", "my juggernaut", "fast") -> its key, among the models on
    this PC. Exact key/name first, then the longest name contained in what was said (or the other way round); None
    when nothing fits, so the caller lists what there is instead of guessing (1.4.6)."""
    said = _norm(name)
    if not said:
        return None
    rows = installed_models()
    for r in rows:
        if said in (_norm(r["key"]), _norm(r["name"])):
            return r["key"]
    best, score = None, 0
    for r in rows:
        for cand in (r["name"], r["key"]):
            c = _norm(cand)
            if len(c) >= 3 and (c in said or said in c):
                s = min(len(c), len(said))
                if s > score:
                    best, score = r["key"], s
    if score >= 3:
        return best
    # Words said in another order or with others between ("pony anime clean" for "Pony V6 · Anime clean"): the
    # model whose name holds the most of the words said - at least two, or all of them.
    words = [w for w in re.findall(r"[a-z0-9]+", str(name).lower()) if len(w) >= 2]
    best, score = None, 0
    for r in rows:
        have = set(re.findall(r"[a-z0-9]+", f"{r['name']} {r['key']}".lower()))
        s = sum(w in have for w in words)
        if s > score:
            best, score = r["key"], s
    return best if score >= 2 or (words and score == len(words)) else None


def generate_image_tool(args: dict, ref_image: bytes | None = None, should_cancel=None, progress=None) -> dict:
    """The model's generate_image tool."""
    quality = args.get("quality")
    if args.get("model"):                     # a model the user named: matched to one on this PC, never guessed
        key = resolve_model(args["model"])
        if not key:
            have = "; ".join(f"{r['name']} ({r['key']})" for r in installed_models()) or "none"
            return {"ok": False, "status": "permanent_failure", "code": "unknown_model",
                    "message": f"No picture model called '{args['model']}' is on this PC. Installed: {have}. "
                               "Ask the user which of these to use, or call again with one of them."}
        quality = key
    try:
        img = generate(args.get("prompt") or "", quality, args.get("size"), args.get("seed"),
                       args.get("negative") or "", ref_image if args.get("edit") else None, should_cancel, progress,
                       count=args.get("count") or 1)
    except ImageError as e:
        return {"ok": False, "status": "permanent_failure", "message": str(e), "code": e.code}
    made = img.get("images") or [img]
    keep = ("id", "url", "model", "seed", "width", "height", "seconds")
    n = len(made)
    return {"ok": True, "image": {k: img[k] for k in keep}, "images": [{k: x[k] for k in keep} for x in made],
            "message": (f"Done - {'your image' if n == 1 else f'{n} images'} for this request "
                        f"{'is' if n == 1 else 'are'} ready ({img['model']}, {img['width']}x{img['height']}, "
                        f"{img['seconds']} s, seed {img['seed']}); the app puts {'it' if n == 1 else 'them'} under your "
                        "reply. Reply in one or two sentences: what the picture shows, and one variation you could make.")}
