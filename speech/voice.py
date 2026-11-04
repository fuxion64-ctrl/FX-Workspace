"""
voice.py
========
The voice assistant's ears and mouth, both on this PC - nothing is sent anywhere:

  hearing   whisper.cpp's server (whisper-server.exe) on 127.0.0.1:8092 with a
            ggml model; the page records the microphone, finds where you stop
            talking and sends one 16 kHz mono WAV per utterance
            (POST /api/voice/transcribe) - transcribe() returns the text.
  speaking  Piper (piper.exe) with an .onnx voice, kept running so each sentence
            is quick; speak() returns a WAV. The page asks sentence by sentence
            while the reply streams, so it starts talking early.

Nothing is bundled: Settings > Voice > "Set up voice" downloads the pieces you
choose (CATALOG - GitHub releases and Hugging Face, through the network layer's
"other" route) into data/voice/. Audio is never written to disk except as
Piper's short-lived output file (deleted at once) and is never logged; in a
temporary chat nothing of it is kept either.

Both run on the CPU (a BLAS build of whisper.cpp), so they don't take VRAM from
the local model. A CUDA build of whisper.cpp can be chosen instead.

Languages (LANGS): English uses the natural voice and the accurate listener
(voice_natural.py). Another language is heard by a multilingual Whisper model
and spoken by that language's own Piper voice; "auto" detects which of English
and the chosen languages you spoke, and the reply comes back in it.
"""
from __future__ import annotations

import paths
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
import wave
import zipfile

import httpx

import netlayer
import speech_text
import voice_natural as vn
import voice_hebrew as vh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(BASE_DIR, "data", "voice")
WHISPER_DIR = os.path.join(VOICE_DIR, "whisper")
PIPER_DIR = os.path.join(VOICE_DIR, "piper")
WHISPER_PORT = int(os.environ.get("AGENTIC_WHISPER_PORT") or 8092)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_WHISPER_REL = "https://github.com/ggml-org/whisper.cpp/releases/download/b5130"
_HF_WHISPER = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
_HF_PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# What "Set up voice" can download. mb: the size shown before downloading.
CATALOG = {
    "engines": {
        "whisper-cpu": {"label": "Speech recognition (whisper.cpp, CPU)", "url": f"{_WHISPER_REL}/whisper-blas-bin-x64.zip", "mb": 20},
        # CUDA 12.4: it brings its own cuBLAS 12 (the 11.8 zip has no cuBLAS, and a PC with only CUDA 12 then
        # falls back to the CPU - "no GPU found"). Installed beside the image/video engines on G: when it exists.
        "whisper-cuda": {"label": "Speech recognition (whisper.cpp, NVIDIA GPU - CUDA 12.4)", "url": f"{_WHISPER_REL}/whisper-cublas-12.4.0-bin-x64.zip", "mb": 643},
        "piper": {"label": "Speech (Piper)", "url": "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip", "mb": 21},
    },
    "models": {       # whisper.cpp ggml models
        "base.en": {"label": "Base (English) - fast, good", "mb": 148},
        "small.en": {"label": "Small (English) - better, ~3x slower", "mb": 488},
        "base": {"label": "Base (any language)", "mb": 148},
        "small": {"label": "Small (any language) - better, ~3x slower", "mb": 488},
    },
    "voices": {       # Piper voices
        "en_US-lessac-medium": {"label": "Lessac - US English, clear and neutral", "path": "en/en_US/lessac/medium", "mb": 63},
        "en_US-amy-medium": {"label": "Amy - US English, warm", "path": "en/en_US/amy/medium", "mb": 63},
        "en_US-ryan-medium": {"label": "Ryan - US English, deeper", "path": "en/en_US/ryan/medium", "mb": 63},
        "en_GB-alba-medium": {"label": "Alba - British English", "path": "en/en_GB/alba/medium", "mb": 63},
    },
    "natural": {k: {"label": v["label"], "mb": v["mb"]} for k, v in vn.PACKS.items()},     # voice_natural.py
    "natural_voices": {k: {"label": v[1], "grade": v[2]} for k, v in vn.VOICES.items()},
}
CATALOG["langs"] = {}                  # filled below, once LANGS exists
ENGINES = ("natural", "fast")          # natural: Kokoro · fast: Piper
LISTENERS = ("accurate", "whisper")    # accurate: Parakeet · whisper: whisper.cpp
# Languages voice mode speaks. English: the natural voice (Kokoro) or Piper, heard by Parakeet or Whisper
# English. Any other: heard by a multilingual Whisper model, spoken by a native Piper voice (one per
# language, downloaded when it's chosen). "auto": Whisper detects it and it answers in the same language.
# code -> (name, Piper voice, its path on Hugging Face, MB)
LANGS = {
    "en": ("English", None, None, 0),
    "he": ("Hebrew", "he_IL-saspeech-medium", "he/he_IL/saspeech/medium", 60),
    "ar": ("Arabic", "ar_JO-kareem-medium", "ar/ar_JO/kareem/medium", 60),
    "es": ("Spanish", "es_ES-davefx-medium", "es/es_ES/davefx/medium", 60),
    "fr": ("French", "fr_FR-siwis-medium", "fr/fr_FR/siwis/medium", 60),
    "de": ("German", "de_DE-thorsten-medium", "de/de_DE/thorsten/medium", 60),
    "it": ("Italian", "it_IT-paola-medium", "it/it_IT/paola/medium", 61),
    "pt": ("Portuguese", "pt_BR-faber-medium", "pt/pt_BR/faber/medium", 60),
    "ru": ("Russian", "ru_RU-irina-medium", "ru/ru_RU/irina/medium", 60),
    "uk": ("Ukrainian", "uk_UA-ukrainian_tts-medium", "uk/uk_UA/ukrainian_tts/medium", 73),
    "pl": ("Polish", "pl_PL-gosia-medium", "pl/pl_PL/gosia/medium", 60),
    "nl": ("Dutch", "nl_NL-pim-medium", "nl/nl_NL/pim/medium", 61),
    "tr": ("Turkish", "tr_TR-dfki-medium", "tr/tr_TR/dfki/medium", 60),
    "hi": ("Hindi", "hi_IN-priyamvada-medium", "hi/hi_IN/priyamvada/medium", 61),
    "zh": ("Chinese", "zh_CN-huayan-medium", "zh/zh_CN/huayan/medium", 60),
}
# Hebrew uses the separate CPU Phonikud/ONNX pipeline rather than piper.exe.
ADDON_LANGS = set()
CATALOG["langs"] = {k: {"label": v[0], "mb": v[3], "addon": k in ADDON_LANGS} for k, v in LANGS.items()}
MULTI_MODELS = ("small", "base")       # multilingual whisper models for other languages
PATIENCE = {"short": 450, "normal": 650, "long": 1200}          # ms of quiet that ends your turn
SENSITIVITY = ("low", "normal", "high")                          # how easily the mic hears you start
DEFAULT_SETTINGS = {
    "voice_engine": "natural",         # which voice speaks (falls back to the other one if it isn't set up)
    "voice_natural": "af_heart",       # Kokoro's voice
    "voice_natural_compact": False,    # Kokoro's int8 copy (126 MB; fast only on CPUs with AVX-512 VNNI)
    "voice_listener": "accurate",      # which recognizer makes the words that are sent (English)
    "voice_live": True,                # the secondary listener: notices the end of a sentence (and captions)
    "voice_captions": False,           # show what you say / what it says under the orb
    "voice_sounds": True,              # soft sounds when it starts and stops listening
    "voice_model": "base.en",          # Whisper for English (when the listener is Whisper)
    "voice_voice": "en_US-lessac-medium",
    "voice_lang": "en",                # LANGS or "auto"
    "voice_auto_langs": [],            # "auto": the languages besides English it listens for (and has voices for)
    "voice_multi_model": "small",      # Whisper for other languages
    "voice_rate": 1.0,                 # speaking speed (Piper's length scale is its inverse)
    "voice_breath": 0.14,              # the pause between pieces of speech, seconds
    "voice_volume": 1.0,
    "voice_patience": "normal",        # how long a pause ends your turn (PATIENCE)
    "voice_sensitivity": "normal",     # how easily it hears you start (SENSITIVITY)
    "voice_speak_replies": True,       # read replies aloud in voice mode
    "voice_conversation": True,        # listen again after speaking (hands-free)
    "voice_barge_in": True,            # talking over it stops it
    "voice_gpu": False,                # the CUDA build of whisper.cpp
    "voice_duplex": True,              # live conversation (voice_duplex.py); off = the turn-based mode
    "voice_duplex_model": "",          # its model ("" = the fastest good one configured)
}


def _num(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if v != v else round(min(hi, max(lo, v)), 2)


def clean_settings(s: dict) -> dict:
    out = {}
    out["voice_model"] = s.get("voice_model") if s.get("voice_model") in CATALOG["models"] else "base.en"
    out["voice_voice"] = s.get("voice_voice") if s.get("voice_voice") in CATALOG["voices"] else "en_US-lessac-medium"
    out["voice_lang"] = s.get("voice_lang") if s.get("voice_lang") in LANGS or s.get("voice_lang") == "auto" else "en"
    extra = s.get("voice_auto_langs") if isinstance(s.get("voice_auto_langs"), list) else []
    out["voice_auto_langs"] = [c for c in LANGS if c != "en" and c in extra]
    out["voice_multi_model"] = s.get("voice_multi_model") if s.get("voice_multi_model") in MULTI_MODELS else "small"
    out["voice_rate"] = _num(s.get("voice_rate", 1.0), 0.6, 1.6, 1.0)
    out["voice_breath"] = _num(s.get("voice_breath", 0.14), 0.0, 0.6, 0.14)
    out["voice_volume"] = _num(s.get("voice_volume", 1.0), 0.2, 1.5, 1.0)
    out["voice_patience"] = s.get("voice_patience") if s.get("voice_patience") in PATIENCE else "normal"
    out["voice_sensitivity"] = s.get("voice_sensitivity") if s.get("voice_sensitivity") in SENSITIVITY else "normal"
    for k in ("voice_speak_replies", "voice_conversation", "voice_barge_in", "voice_gpu", "voice_natural_compact", "voice_live",
              "voice_sounds", "voice_captions", "voice_duplex"):
        out[k] = s.get(k) if isinstance(s.get(k), bool) else DEFAULT_SETTINGS[k]
    out["voice_engine"] = s.get("voice_engine") if s.get("voice_engine") in ENGINES else "natural"
    out["voice_listener"] = s.get("voice_listener") if s.get("voice_listener") in LISTENERS else "accurate"
    out["voice_natural"] = s.get("voice_natural") if s.get("voice_natural") in vn.VOICES else "af_heart"
    m = s.get("voice_duplex_model")
    out["voice_duplex_model"] = m.strip()[:120] if isinstance(m, str) else ""
    return out


def langs_wanted(s: dict) -> list:
    """The languages other than English this setup listens for and speaks."""
    if s["voice_lang"] == "auto":
        return list(s["voice_auto_langs"])
    return [] if s["voice_lang"] == "en" else [s["voice_lang"]]


def lang_voice_ready(lang: str) -> bool:
    if lang == "he":
        return vh.ready()
    name = LANGS.get(lang, (None, None))[1]
    return bool(name and lang not in ADDON_LANGS and piper_exe() and piper_voice(name))


def multi_ready(s: dict) -> bool:
    return bool(whisper_exe() and whisper_model(s["voice_multi_model"]))


def _kokoro_pack(s: dict) -> str:
    return "kokoro-compact" if s["voice_natural_compact"] else "kokoro"


def natural_ready(s: dict) -> bool:
    return vn.available() and vn.installed(_kokoro_pack(s))


def accurate_ready() -> bool:
    return vn.available() and vn.installed("parakeet")


def live_ready(s: dict) -> bool:
    return s["voice_live"] and vn.available() and vn.installed("live")


def speaker_in_use(s: dict) -> str:
    """natural | fast | "" - what would speak now (the chosen one, else the other one if it's set up)."""
    fast = bool(piper_exe() and piper_voice(s["voice_voice"]))
    if s["voice_engine"] == "natural":
        return "natural" if natural_ready(s) else "fast" if fast else ""
    return "fast" if fast else "natural" if natural_ready(s) else ""


def listener_in_use(s: dict) -> str:
    whisper = bool(whisper_exe() and whisper_model(s["voice_model"]))
    if s["voice_listener"] == "accurate":
        return "accurate" if accurate_ready() else "whisper" if whisper else ""
    return "whisper" if whisper else "accurate" if accurate_ready() else ""


# ==============================================================================
# What's installed
# ==============================================================================
def _find(root: str, name: str) -> str | None:
    for folder, _, files in os.walk(root):
        if name in files:
            return os.path.join(folder, name)
    return None


def whisper_exe() -> str | None:
    return _find(WHISPER_DIR, "whisper-server.exe") if os.path.isdir(WHISPER_DIR) else None


WHISPER_GPU_DIRS = (os.path.join(paths.feature_root("Voice"), "whisper-cuda12"), os.path.join(VOICE_DIR, "whisper-cuda12"))


def whisper_gpu_exe() -> str | None:
    """The CUDA build of whisper-server (with its cuBLAS beside it), or None."""
    for d in WHISPER_GPU_DIRS:
        exe = _find(d, "whisper-server.exe") if os.path.isdir(d) else None
        if exe and os.path.isfile(os.path.join(os.path.dirname(exe), "ggml-cuda.dll")):
            return exe
    return None


def whisper_model(name: str) -> str | None:
    path = os.path.join(WHISPER_DIR, "models", f"ggml-{name}.bin")
    return path if os.path.isfile(path) and os.path.getsize(path) > 1_000_000 else None


def piper_exe() -> str | None:
    return _find(PIPER_DIR, "piper.exe") if os.path.isdir(PIPER_DIR) else None


def piper_voice(name: str) -> str | None:
    path = os.path.join(PIPER_DIR, "voices", f"{name}.onnx")
    return path if os.path.isfile(path) and os.path.isfile(path + ".json") else None


def status(settings: dict) -> dict:
    s = clean_settings(settings)
    speaker, listener = speaker_in_use(s), listener_in_use(s)
    # "Ready" means what you chose is set up; the other engine only fills in until it is.
    hearing = listener == ("accurate" if s["voice_listener"] == "accurate" else "whisper")
    speaking = speaker == s["voice_engine"]
    if s["voice_lang"] != "en":             # another language, or detecting: the multilingual model and its voices
        hearing = multi_ready(s)
        voices_ok = all(lang_voice_ready(c) for c in langs_wanted(s) if c not in ADDON_LANGS)
        speaking = voices_ok and (speaking if s["voice_lang"] == "auto" else True)
    return {"hearing": hearing, "speaking": speaking,
            "speaker": speaker, "listener": listener, "live": live_ready(s), "sherpa": vn.available(),
            "engine": bool(whisper_exe()), "piper": bool(piper_exe()), "gpu_listener": bool(s["voice_gpu"] and whisper_gpu_exe()),
            "models": {m: bool(whisper_model(m)) for m in CATALOG["models"]},
            "voices": {v: bool(piper_voice(v)) for v in CATALOG["voices"]},
            "packs": {p: vn.installed(p) for p in vn.PACKS},
            "langs": {c: lang_voice_ready(c) for c in LANGS if c != "en"}, "multi": multi_ready(s),
            "plan": [{"label": i[1], "mb": i[4]} for i in plan(s)],
            "catalog": CATALOG, "setup": dict(_setup), "running": bool(_whisper.proc and _whisper.proc.poll() is None)}


# ==============================================================================
# Setup: downloading the pieces (only when the user clicks)
# ==============================================================================
_setup = {"state": "idle", "part": "", "done_mb": 0.0, "total_mb": 0.0, "error": ""}
_setup_lock = threading.Lock()


def plan(settings: dict) -> list:
    """What "Set up voice" would download for these settings: [(key, label, url, dest, mb, kind)],
    kind: "zip" | "tar" (a .tar.bz2) | "file". Only what the chosen engines need."""
    s = clean_settings(settings)
    items = []
    if s["voice_engine"] == "natural" and vn.available() and not vn.installed(_kokoro_pack(s)):
        p = vn.PACKS[_kokoro_pack(s)]
        items.append((_kokoro_pack(s), p["label"], p["url"], vn.SHERPA_DIR, p["mb"], "tar"))
    if s["voice_listener"] == "accurate" and vn.available() and not vn.installed("parakeet"):
        p = vn.PACKS["parakeet"]
        items.append(("parakeet", p["label"], p["url"], vn.SHERPA_DIR, p["mb"], "tar"))
    if s["voice_live"] and vn.available() and not vn.installed("live"):
        p = vn.PACKS["live"]
        items.append(("live", p["label"], p["url"], vn.SHERPA_DIR, p["mb"], "tar"))
    other = s["voice_lang"] != "en"             # another language (or detecting one): a multilingual Whisper
    need_whisper = s["voice_listener"] == "whisper" or not vn.available() or other
    need_piper = s["voice_engine"] == "fast" or not vn.available() or any(c != "he" for c in langs_wanted(s))
    if need_whisper and s["voice_gpu"] and not whisper_gpu_exe():
        e = CATALOG["engines"]["whisper-cuda"]
        items.append(("whisper-gpu", e["label"], e["url"], os.path.join(VOICE_DIR, "whisper-cuda12"), e["mb"], "zip"))
    if need_whisper and not whisper_exe():
        e = CATALOG["engines"]["whisper-cpu"]
        items.append(("whisper", e["label"], e["url"], os.path.join(WHISPER_DIR, "bin"), e["mb"], "zip"))
    wanted = ([s["voice_model"]] if s["voice_listener"] == "whisper" or not vn.available() else []) + \
        ([s["voice_multi_model"]] if other else [])
    for name in dict.fromkeys(wanted):
        if not whisper_model(name):
            m = CATALOG["models"][name]
            items.append(("model" if name == s["voice_model"] else "model-multi", f"Recognition model: {m['label']}",
                          f"{_HF_WHISPER}/ggml-{name}.bin", os.path.join(WHISPER_DIR, "models", f"ggml-{name}.bin"), m["mb"], "file"))
    if "he" in langs_wanted(s):
        items.extend(vh.downloads())
    if not need_piper:
        return items
    if not piper_exe():
        e = CATALOG["engines"]["piper"]
        items.append(("piper", e["label"], e["url"], PIPER_DIR, e["mb"], "zip"))
    voices = [(s["voice_voice"], CATALOG["voices"][s["voice_voice"]]["label"], CATALOG["voices"][s["voice_voice"]]["path"],
               CATALOG["voices"][s["voice_voice"]]["mb"], "voice")] if s["voice_engine"] == "fast" or not vn.available() else []
    voices += [(LANGS[c][1], f"{LANGS[c][0]} voice", LANGS[c][2], LANGS[c][3], f"voice-{c}") for c in langs_wanted(s)
               if c != "he"]
    for name, label, path, mb, key in voices:
        if not piper_voice(name):
            base = f"{_HF_PIPER}/{path}/{name}"
            dest = os.path.join(PIPER_DIR, "voices", f"{name}.onnx")
            items.append((key, f"Voice: {label}", base + ".onnx", dest, mb, "file"))
            items.append((key + "-config", "Voice settings", base + ".onnx.json", dest + ".json", 0.01, "file"))
    return items


def _download(url: str, dest: str, on_bytes) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with netlayer.client("other", timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True, private=True) as c:
        with c.stream("GET", url) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"{url.split('/')[2]} answered {r.status_code}")
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    f.write(chunk)
                    on_bytes(len(chunk))
    os.replace(tmp, dest)


def setup(settings: dict) -> dict:
    """Starts downloading what's missing in the background; the page polls status()."""
    items = plan(settings)
    with _setup_lock:
        if _setup["state"] == "running":
            return dict(_setup)
        if not items:
            _setup.update(state="done", part="", error="", done_mb=0, total_mb=0)
            return dict(_setup)
        _setup.update(state="running", part="", error="", done_mb=0.0, total_mb=round(sum(i[4] for i in items), 1))
    threading.Thread(target=_run_setup, args=(items,), daemon=True, name="voice-setup").start()
    return dict(_setup)


def _run_setup(items: list):
    done = [0]                                  # bytes (rounding per chunk used to inflate the total)

    def count(n):
        done[0] += n
        _setup["done_mb"] = round(done[0] / 1048576, 1)
    try:
        for key, label, url, dest, _mb, kind in items:
            _setup["part"] = label
            kind = {True: "zip", False: "file"}.get(kind, kind)
            if kind == "zip":
                archive = os.path.join(VOICE_DIR, f"{key}.zip")
                _download(url, archive, count)
                with zipfile.ZipFile(archive) as z:
                    for member in z.namelist():                   # never outside the destination folder
                        target = os.path.realpath(os.path.join(dest, member))
                        if not target.startswith(os.path.realpath(dest) + os.sep) and target != os.path.realpath(dest):
                            raise RuntimeError(f"Unsafe path in {key}.zip: {member}")
                    z.extractall(dest)
                os.remove(archive)
            elif kind == "tar":
                archive = os.path.join(VOICE_DIR, f"{key}.tar.bz2")
                _download(url, archive, count)
                _setup["part"] = f"{label} - unpacking"
                os.makedirs(dest, exist_ok=True)
                with tarfile.open(archive, "r:bz2") as t:
                    # "data": no absolute paths, no "..", no links out, no device files.
                    t.extractall(dest, filter="data")
                os.remove(archive)
            else:
                _download(url, dest, count)
        _setup.update(state="done", part="")
    except Exception as e:
        _setup.update(state="error", error=str(e)[:300])


# ==============================================================================
# Hearing: whisper-server
# ==============================================================================
class _Whisper:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.model = ""
        self.gpu = False
        self.lock = threading.Lock()

    def ensure(self, model: str, gpu: bool = False) -> None:
        """whisper-server with this model; on the GPU (the CUDA build) when asked for and installed - small
        multilingual goes from ~6 s an utterance on the CPU to well under a second (measured: see HANDOFF)."""
        gpu = bool(gpu and whisper_gpu_exe())
        with self.lock:
            if self.proc and self.proc.poll() is None and self.model == model and self.gpu == gpu:
                return
            self.stop_locked()
            exe, path = (whisper_gpu_exe() if gpu else whisper_exe()), whisper_model(model)
            if not exe or not path:
                raise RuntimeError("Voice isn't set up yet - Settings > Voice > Set up voice.")
            self.gpu = gpu
            # Another program (or a second copy of the app) on the port answered "/" as if it were ours (1.4.7).
            global WHISPER_PORT
            import ports
            if not ports.is_free(WHISPER_PORT):
                WHISPER_PORT = ports.pick(WHISPER_PORT + 1)
                ports.note("whisper", WHISPER_PORT, "moved")
            threads = "4" if gpu else str(max(2, min(8, (os.cpu_count() or 4) - 2)))
            self.proc = subprocess.Popen([exe, "-m", path, "--host", "127.0.0.1", "--port", str(WHISPER_PORT), "-t", threads],
                                         cwd=os.path.dirname(exe), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
            self.model = model
            deadline = time.time() + 30
            with netlayer.local_client(timeout=2) as c:
                while time.time() < deadline:
                    if self.proc.poll() is not None:
                        raise RuntimeError("The speech recognizer stopped as it started (whisper-server).")
                    try:
                        c.get(f"http://127.0.0.1:{WHISPER_PORT}/")
                        return
                    except httpx.HTTPError:
                        time.sleep(0.25)
            raise RuntimeError("The speech recognizer didn't start within 30 seconds.")

    def stop_locked(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def stop(self):
        with self.lock:
            self.stop_locked()


_whisper = _Whisper()


_NOISE = ("blank_audio", "silence", "music", "inaudible", "noise", "")
_LANG_NAMES = {v[0].lower(): k for k, v in LANGS.items()}          # whisper names languages in full ("hebrew")


def transcript_review(text: str, lang: str, segments: list | None = None) -> dict:
    """Mark critical or uncertain words for confirmation without changing ASR output."""
    import re
    text = _clean_words(text)
    reasons = []
    if re.search(r"\d|\b[A-Z]{2,}\d+[A-Z0-9-]*\b", text):
        reasons.append("number or identifier")
    if lang == "en" and re.search(r"(?<![.!?]\s)(?<=\s)[A-Z][a-z]{2,}\b", text):
        reasons.append("possible name or place")
    if text and lang in ("en", "he", "ru"):
        from osint import geo
        if geo.places_in(text, most=1) and "possible name or place" not in reasons:
            reasons.append("possible place")
    if re.search(r"\b(?:send|delete|remove|install|run|buy|pay|transfer|approve|cancel|erase|move|publish)\b|(?:שלח|מחק|התקן|הרץ|אשר|העבר)|(?:отправ|удал|установ|запуст|оплат|подтверд)", text, re.I):
        reasons.append("possible action")
    scores = [s.get("avg_logprob") for s in (segments or []) if isinstance(s, dict) and isinstance(s.get("avg_logprob"), (int, float))]
    if scores and min(scores) < -0.8:
        reasons.append("low recognition confidence")
    return {"required": bool(reasons), "reasons": reasons}


def _clean_words(text: str) -> str:
    # whisper marks silence and noise like "[BLANK_AUDIO]" or "(music)" - not words.
    text = " ".join(str(text or "").replace("\n", " ").split())
    return "" if text.strip("[]() .").lower() in _NOISE else text


_whisper_http = None
_whisper_http_lock = threading.Lock()


def _whisper_once(wav: bytes, language: str) -> dict:
    # One kept-alive connection: opening a new one costs 300-450 ms on Windows here, per utterance.
    global _whisper_http
    with _whisper_http_lock:
        if _whisper_http is None:
            _whisper_http = netlayer.local_client(timeout=120)
        c = _whisper_http
    try:
        r = c.post(f"http://127.0.0.1:{WHISPER_PORT}/inference", files={"file": ("speech.wav", wav, "audio/wav")},
                   data={"response_format": "verbose_json", "temperature": "0.0", "language": language,
                         "translate": "false"})
    except httpx.HTTPError:
        with _whisper_http_lock:
            _whisper_http = None                        # a restarted server: the next call opens a new one
        raise
    if r.status_code >= 400:
        raise RuntimeError(f"The speech recognizer answered {r.status_code}.")
    return r.json() or {}


def _lang_code(name) -> str:
    name = str(name or "").lower()
    return name if name in LANGS else _LANG_NAMES.get(name, name)


def transcribe(wav: bytes, settings: dict) -> dict:
    """{"text", "ms", "by", "lang"} for one utterance (16 kHz mono WAV).
    English: the accurate listener (Parakeet) when it's chosen and set up, else whisper.cpp's English model.
    Another language: the multilingual Whisper model in that language. "auto": Whisper detects the language;
    if it guesses one that isn't English or one of the chosen languages (short phrases can fool it), the
    chosen language it rated highest is used instead."""
    s = clean_settings(settings)
    if len(wav) < 1000:
        return {"text": "", "ms": 0, "by": "", "lang": ""}
    started = time.time()
    ms = lambda: round((time.time() - started) * 1000)          # noqa: E731
    if s["voice_lang"] == "en":
        if listener_in_use(s) == "accurate":
            text = _clean_words(vn.accurate.transcribe(vn.wav_to_float16k(wav)))
            return {"text": text, "ms": ms(), "by": "accurate", "lang": "en", "review": transcript_review(text, "en")}
        _whisper.ensure(s["voice_model"], s["voice_gpu"])
        out = _whisper_once(wav, "en")
        text = _clean_words(out.get("text"))
        return {"text": text, "ms": ms(), "by": "whisper", "lang": "en", "review": transcript_review(text, "en", out.get("segments"))}
    _whisper.ensure(s["voice_multi_model"], s["voice_gpu"])
    if s["voice_lang"] != "auto":
        out = _whisper_once(wav, s["voice_lang"])
        text = _clean_words(out.get("text"))
        return {"text": text, "ms": ms(), "by": "whisper", "lang": s["voice_lang"], "review": transcript_review(text, s["voice_lang"], out.get("segments"))}
    allowed = ["en"] + s["voice_auto_langs"]
    out = _whisper_once(wav, "auto")
    # "language" is the one it decoded in. (This build's "detected_language" and its probabilities are
    # flat noise - about 1% each - so they're only trusted when one clearly stands out.)
    lang = _lang_code(out.get("language"))
    if lang not in allowed:
        probs = {_lang_code(k): v for k, v in (out.get("language_probabilities") or {}).items() if isinstance(v, (int, float))}
        best = max(allowed, key=lambda c: probs.get(c, 0.0))
        # Heard as some other language: with one language chosen besides English, it's most likely that one.
        lang = best if probs.get(best, 0.0) >= 0.2 else s["voice_auto_langs"][0] if len(allowed) == 2 else "en"
        out = _whisper_once(wav, lang)
    text = _clean_words(out.get("text"))
    return {"text": text, "ms": ms(), "by": "whisper", "lang": lang, "review": transcript_review(text, lang, out.get("segments"))}


def live_feed(sid: str, pcm16: bytes, settings: dict) -> dict:
    """Live captions: the next bit of the utterance (16 kHz mono 16-bit PCM) -> what it has heard so far."""
    s = clean_settings(settings)
    if not live_ready(s):
        return {"partial": "", "endpoint": False, "off": True}
    return vn.live.feed(str(sid)[:40], vn.pcm16_to_float(pcm16))


def live_end(sid: str):
    vn.live.end(str(sid)[:40])


# ==============================================================================
# Speaking: Piper, kept running
# ==============================================================================
class _Piper:
    def __init__(self, out: str = "out"):
        self.proc: subprocess.Popen | None = None
        self.voice, self.rate = "", 1.0
        self.out_dir = os.path.join(VOICE_DIR, out)
        self.lock = threading.Lock()

    def _start(self, voice: str, rate: float):
        self.stop_locked()
        exe, model = piper_exe(), piper_voice(voice)
        if not exe or not model:
            raise RuntimeError("Voice isn't set up yet - Settings > Voice > Set up voice.")
        shutil.rmtree(self.out_dir, ignore_errors=True)
        os.makedirs(self.out_dir, exist_ok=True)
        self.proc = subprocess.Popen([exe, "--model", model, "--output_dir", self.out_dir, "--length_scale", f"{1 / rate:.2f}",
                                      "--sentence_silence", "0.15", "--quiet"],
                                     cwd=os.path.dirname(exe), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, text=True, encoding="utf-8", bufsize=1)
        self.voice, self.rate = voice, rate

    def speak(self, text: str, voice: str, rate: float) -> bytes:
        with self.lock:
            if not (self.proc and self.proc.poll() is None and self.voice == voice and self.rate == rate):
                self._start(voice, rate)
            self.proc.stdin.write(text.replace("\n", " ").strip() + "\n")
            self.proc.stdin.flush()
            path = self.proc.stdout.readline().strip()          # Piper prints each file it writes
            if not path:
                self.stop_locked()
                raise RuntimeError("The voice stopped (Piper).")
            try:
                with open(path, "rb") as f:
                    return f.read()
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def stop_locked(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def stop(self):
        with self.lock:
            self.stop_locked()


_piper = _Piper()
_piper_lang = _Piper("out-lang")               # the other language's voice, kept loaded beside the English one
MAX_SPEAK_CHARS = 1200


def _piper_wav(engine: _Piper, text: str, name: str, rate: float) -> bytes:
    raw = engine.speak(text, name, rate)
    with wave.open(io.BytesIO(raw)) as w:
        sr, frames = w.getframerate(), w.readframes(w.getnframes())
    return vn.wav_bytes(vn.polish(vn.pcm16_to_float(frames), sr), sr)


def reply_lang(s: dict, heard: str = "") -> str:
    """The language the reply is spoken in: the one chosen, or with "auto" the one just heard."""
    if s["voice_lang"] != "auto":
        return s["voice_lang"]
    return heard if heard in ["en"] + s["voice_auto_langs"] else "en"


def speak(text: str, settings: dict, voice: str | None = None, lang: str | None = None) -> bytes:
    """One sentence or a few, as a WAV. English is said the way a person would (speech_text), by the natural
    voice (Kokoro) when it's chosen and set up, else Piper. Another language: its native Piper voice.
    Every voice is trimmed, faded and levelled the same way."""
    s = clean_settings(settings)
    lang = reply_lang(s, lang or "")
    text = " ".join(str(text or "").split())[:MAX_SPEAK_CHARS]
    text = speech_text.normalize(text) if lang == "en" else speech_text.plain(text)
    if not text or not any(ch.isalnum() for ch in text):
        return silence()
    if lang != "en":
        if lang == "he":
            pcm, rate = vh.speak(text, s["voice_rate"])
            return vn.wav_bytes(vn.polish(pcm, rate), rate)
        if lang in ADDON_LANGS:
            raise RuntimeError(f"The {LANGS[lang][0]} voice needs an add-on that isn't installed - the answer is in the chat.")
        if not lang_voice_ready(lang):
            raise RuntimeError(f"There's no {LANGS[lang][0]} voice yet - Settings > Voice > Set up voice.")
        return _piper_wav(_piper_lang, text, LANGS[lang][1], s["voice_rate"])
    engine = speaker_in_use(s)
    if engine == "natural":
        pcm, rate = vn.kokoro.speak(text, voice if voice in vn.VOICES else s["voice_natural"], s["voice_rate"], _kokoro_pack(s))
        return vn.wav_bytes(pcm, rate)
    if engine == "fast":
        return _piper_wav(_piper, text, s["voice_voice"], s["voice_rate"])
    raise RuntimeError("Voice isn't set up yet - Settings > Voice > Set up voice.")


def silence(ms: int = 50) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * (16 * ms))
    return buf.getvalue()


def stop_all():
    """The server is stopping (or "Free the memory"): every engine lets go of its model."""
    _whisper.stop()
    _piper.stop()
    _piper_lang.stop()
    vn.unload_all()
    vh.unload()


def voice_header(settings: dict, heard: str = "") -> str:
    lang = reply_lang(clean_settings(settings), heard)
    say = (
        f" Answer in {LANGS[lang][0]} (the user is speaking {LANGS[lang][0]} and hears a {LANGS[lang][0]} voice), "
        "written the way it's spoken, numbers included - unless they ask for another language.")
    return ("VOICE (the user is talking to you and hears your reply read aloud): talk the way a friendly, sharp person "
            "talks. Give the first useful answer in a short complete sentence as soon as you can, then expand only "
            "as needed. For live world facts, say the source and observation time or data age when it matters; do "
            "not treat a feed gap as proof that nothing happened. Use "
            "contractions and plain words, vary sentence length, and join ideas with 'so', 'but', 'and' instead of "
            "bullet points. No tables, code blocks or long lists unless asked (say you've put them in the chat); no "
            "URLs, symbols or Markdown; round numbers when precision doesn't matter ('about two hundred sixteen "
            "billion'). No 'Certainly!' or 'Great question' - just answer. If it helps, end with a short, natural "
            "follow-up question. These speech-style rules apply to your spoken answer; an action request still "
            "requires calling the available tools and waiting for their results." + say)
