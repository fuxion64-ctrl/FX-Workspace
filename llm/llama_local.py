"""
Local llama.cpp (llama-server router mode) support for Agentic Workspace.

Fixes in this revision:

1. resolve_candidates() used to append EVERY model the server reported to the
   end of the candidate list, for both text and vision requests. On a router
   running with --models-max 1 that meant a screenshot request could pick a
   text-only GGUF (which rejects image content outright), and the model picker
   in the UI listed the same superset under both tabs. Candidates are now
   filtered by kind, using name heuristics for server-reported models that
   aren't in the configured lists.

2. context_limit() exposes the real --ctx-size the launcher started the server
   with, so the caller can budget history in tokens instead of guessing. An
   8192-token context cannot hold what a 1M-token cloud context can, and the
   old char-based trimming had no way to know the difference.

3. Model swaps on a --models-max 1 router unload the previous model, so the
   cached model list is invalidated whenever a different model is used.
"""
import local_runtime
import configparser
import os
import shutil
import threading
import time
from functools import wraps

import requests
import atomic_io

# Scanning, downloads and Settings all edit the same models.ini. Lock the
# entire read/modify/write transaction, and let nested catalog reads re-enter.
_ini_lock = threading.RLock()


def serialized_ini(fn):
    @wraps(fn)
    def locked(*args, **kwargs):
        with _ini_lock:
            return fn(*args, **kwargs)
    return locked

LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8090"))
LLAMA_SERVER_BASE = os.environ.get("LLAMA_SERVER_BASE", f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1")

_no_proxy = os.environ.get("NO_PROXY", "")
for _host in ("127.0.0.1", "localhost", "::1"):
    if _host not in _no_proxy:
        _no_proxy = f"{_no_proxy},{_host}" if _no_proxy else _host
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

_MODELS_URL = f"{LLAMA_SERVER_BASE.rstrip('/')}/models"


def set_port(port: int) -> None:
    """Move the app to a router on another port (1.4.7: another program held 8090, llama_builds.ensure_running
    started ours on a free one). Every caller reads these names at call time - config.LLAMA_SERVER_BASE too."""
    global LLAMA_PORT, LLAMA_SERVER_BASE, _MODELS_URL
    LLAMA_PORT = int(port)
    LLAMA_SERVER_BASE = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1"
    _MODELS_URL = f"{LLAMA_SERVER_BASE}/models"
    os.environ["LLAMA_PORT"] = str(LLAMA_PORT)
    os.environ["LLAMA_SERVER_BASE"] = LLAMA_SERVER_BASE
    import sys
    cfg = sys.modules.get("core.config")
    if cfg is not None:
        cfg.LLAMA_SERVER_BASE = LLAMA_SERVER_BASE
    invalidate_cache()


_CACHE_TTL = 20.0
# A refused connection to a closed localhost port takes 1-2 s on Windows (TCP
# SYN retries), and /api/state asks "is the server up?" on every UI poll. A
# short negative cache keeps a stopped llama-server from stalling the UI.
_DOWN_TTL = 5.0
_cache = {"at": 0.0, "ids": None, "status": {}}

# Substrings that identify a multimodal GGUF by name. Server-reported models
# are classified with these when they aren't in the configured lists.
_VISION_HINTS = (
    "vision", "-vl", "vl-", "llava", "moondream", "minicpm-v", "pixtral",
    "bakllava", "internvl", "cogvlm", "idefics", "qwen2-vl", "qwen2.5-vl",
    "gemma-3", "phi-3-vision", "phi-4-multimodal", "smolvlm", "molmo",
)


def context_limit() -> int:
    """The server's --ctx-size, as set by run.bat. Callers budget against it."""
    try:
        return max(2048, int(os.environ.get("LLAMA_CTX", "65536")))
    except (TypeError, ValueError):
        return 65536


def _normalize(name: str) -> str:
    base = (name or "").strip().lower()
    if base.endswith(".gguf"):
        base = base[:-5]
    return base.replace(" ", "").replace("_", "-").replace(".", "-")


def normalize_name(name: str) -> str:
    """Public wrapper around the name-matching used throughout this module,
    for callers (main.py) that need to compare a favorite/config name against
    a server-reported model id the same way resolve_candidates() does."""
    return _normalize(name)


def looks_like_vision(name: str) -> bool:
    """Name heuristic only. Prefer is_vision_model(), which also knows which
    presets have an mmproj configured in models.ini."""
    key = _normalize(name)
    return any(hint.replace(".", "-") in key for hint in _VISION_HINTS)


def _shared_configs() -> list:
    """The parsed presets themselves - read-only; list_model_configs() hands out copies. The model catalogue asks
    for one model's preset several times per model, per tool list: copying all of them each time added up."""
    cached = _ini_cache["configs"]
    if cached is not None and _ini_cache["key"] == _ini_key(_ini_path()):
        return cached
    list_model_configs()                     # parses (under the models.ini lock) and fills the cache
    return _ini_cache["configs"] or []


def _config_for(name: str):
    if not name:
        return None
    key = _normalize(name)
    try:
        configs = _shared_configs()
    except Exception:
        return None
    found = next((c for c in configs if c["name"] == name or _normalize(c["name"]) == key), None)
    return dict(found) if found is not None else None


_sizes: dict = {}                            # path -> (when read, bytes or None)


def _file_size(path: str) -> int | None:
    hit = _sizes.get(path)
    now = time.time()
    if hit and now - hit[0] < 60:
        return hit[1]
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    _sizes[path] = (now, size)
    return size


def vision_known(name: str) -> bool | None:
    """Whether the model reads images, from what configures it - its
    models.ini preset's mmproj, or the projector on the command line the
    router reports - or None when nothing says (never guessed from the name)."""
    cfg = _config_for(name)
    if cfg is not None:
        return bool(cfg["mmproj_path"])
    status = router_status(name)
    return bool(status["mmproj"]) if "mmproj" in status else None


def is_vision_model(name: str) -> bool:
    """True when the model can read images: its models.ini preset has an
    mmproj or the router runs it with one (the authoritative signals). Only a
    model nothing describes falls back to a name that looks multimodal - a
    guess used to pick a model for a screenshot, never shown as a capability
    or used for routing (vision_known)."""
    known = vision_known(name)
    return known if known is not None else looks_like_vision(name)


def model_file_size(name: str) -> int | None:
    """Bytes of the model's GGUF on disk (plus mmproj), or None if unknown.
    Used to pick the cheapest model to evict through."""
    cfg = _config_for(name)
    if not cfg or not cfg["model_path"]:
        return None
    total = 0
    for path in (cfg["model_path"], cfg["mmproj_path"]):
        if path:
            size = _file_size(path)
            if size is None:
                if path == cfg["model_path"]:
                    return None
            else:
                total += size
    return total


def server_models(force: bool = False) -> list[str] | None:
    now = time.time()
    if not force and _cache["ids"] is not None and now - _cache["at"] < _CACHE_TTL:
        return _cache["ids"]
    if not force and _cache["ids"] is None and _cache["at"]:
        # Known down: answer "down" now; re-check in the background once the
        # negative cache expires. Only a forced call (an actual model request)
        # waits on the network.
        if now - _cache["at"] >= _DOWN_TTL:
            server_is_up_cached()
        return None
    try:
        resp = requests.get(_MODELS_URL, timeout=5, proxies={"http": None, "https": None})
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        _cache.update(at=now, ids=None)
        return None
    ids = [entry.get("id") for entry in payload.get("data") or []
           if isinstance(entry, dict) and entry.get("id")]
    _cache.update(at=now, ids=ids, status=_router_status(payload))
    return ids


def _router_status(payload: dict) -> dict:
    """Router mode reports, per model, whether it is loaded and the exact
    command line it runs with - including the REAL --ctx-size. A global
    --ctx-size on the router overrides a model's ctx-size in models.ini, so the
    ini value alone can be wrong (e.g. 250000 configured, 65536 actually used)."""
    out = {}
    for entry in payload.get("data") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("status"), dict):
            continue
        st, ctx = entry["status"], None
        args = st.get("args") if isinstance(st.get("args"), list) else []
        for flag in ("--ctx-size", "-c"):
            if flag in args and args.index(flag) + 1 < len(args):
                try:
                    ctx = int(args[args.index(flag) + 1])
                except ValueError:
                    pass
        # The projector on its command line is what makes a model read images.
        # Newer routers also say what a model reads (architecture.input_modalities, from its projector).
        arch = entry.get("architecture") if isinstance(entry.get("architecture"), dict) else {}
        inputs = arch.get("input_modalities") if isinstance(arch.get("input_modalities"), list) else []
        out[entry.get("id")] = {"loaded": st.get("value") == "loaded", "ctx": ctx,
                                "mmproj": any(a in args for a in ("--mmproj", "-mm", "--mmproj-url")) or "image" in inputs,
                                # The router starts each model with its own executable: which build this is.
                                "exe": str(args[0]) if args and not str(args[0]).startswith("-") else None}
    return out


# ---- PrismML's llama.cpp fork (Ternary Bonsai 2) ------------------------------
# Ternary Bonsai 2 ships in two packings, PTQ1_0 and PQ2_0: tensor types that
# upstream llama.cpp does not have, plus Hadamard-rotated weights. Only
# PrismML's fork (github.com/PrismML-Eng/llama.cpp, "prism" branch / its
# releases) loads them; a stock llama-server fails the load. The app moves the
# router to the fork for these models (llama_builds.py, ARCHITECTURE D29, D51).
PRISM_PACKS = ("ptq1_0", "pq2_0")


def needs_prism(model_name: str) -> bool:
    """True when this model's GGUF is a PrismML-only packing."""
    key = _normalize(model_name or "")
    paths = [model_name or ""]
    try:
        paths += [c["model_path"] for c in list_model_configs()
                  if c["name"] == model_name or _normalize(c["name"]) == key]
    except Exception:
        pass
    return any(p in os.path.basename(str(path)).lower() for path in paths for p in PRISM_PACKS)


def server_executable() -> str | None:
    """The llama-server executable the router starts models with (from the
    last /v1/models answer), or None when unknown."""
    for st in (_cache.get("status") or {}).values():
        if st.get("exe"):
            return st["exe"]
    return None


def is_prism_server() -> bool | None:
    """Whether the running router is the PrismML build (None: unknown)."""
    exe = server_executable()
    return None if exe is None else "prism" in exe.lower()


def load_hint(model_name: str) -> str:
    """A sentence to add when a model fails to load, if the reason is known:
    a PrismML-only model on a stock llama-server."""
    if not needs_prism(model_name) or is_prism_server():
        return ""
    import llama_builds
    exe = server_executable()
    if llama_builds.exe_of("prism") and llama_builds.mode() == "latest":
        fix = "Set Settings › Models › llama.cpp build to Auto or PrismML."
    else:
        fix = "Install the fork's Windows CUDA build in local\\llama.cpp\\prism - the app switches to it by itself."
    return (f" '{model_name}' is a Ternary Bonsai 2 (PTQ1_0 / PQ2_0) model: only PrismML's llama.cpp fork can "
            f"load it{f', and the running server is {exe}' if exe else ''}. {fix}")


def router_status(model_name: str) -> dict:
    """{'loaded': bool, 'ctx': int|None} from the last /v1/models answer, or {}."""
    return dict((_cache.get("status") or {}).get(model_name) or {})


def invalidate_cache():
    """Call after a model swap; a --models-max 1 router unloads the old one."""
    _cache.update(at=0.0, ids=None, status={})


def server_is_up() -> bool:
    return server_models() is not None


_refresh_lock = threading.Lock()
_refreshing = {"on": False}


def server_is_up_cached() -> bool:
    """Non-blocking reachability for the UI's status poll: answers from the
    cache immediately and, when that is stale, refreshes it on a background
    thread. A down server costs a 1-2 s refused connect on Windows, which used
    to stall every few polls."""
    now = time.time()
    ttl = _CACHE_TTL if _cache["ids"] is not None else _DOWN_TTL
    if now - _cache["at"] >= ttl:
        with _refresh_lock:
            if not _refreshing["on"]:
                _refreshing["on"] = True

                def _refresh():
                    try:
                        server_models(force=True)
                    finally:
                        _refreshing["on"] = False
                threading.Thread(target=_refresh, daemon=True, name="llama-probe").start()
    return _cache["ids"] is not None


def resolve_candidates(configured: list[str], kind: str = None) -> tuple[list[str], list[str]]:
    """Match the configured names against what the server actually reports.

    kind: 'vision', 'text', or None. When given, server-reported models that
    weren't configured are only appended if their name matches that kind - so a
    vision request can no longer fall through onto a text-only GGUF, which
    llama-server rejects the moment an image part is in the payload.
    """
    ids = server_models()
    if not ids:
        return list(configured), []

    by_key = {_normalize(i): i for i in ids}
    usable, skipped = [], []
    for name in configured:
        exact = name if name in ids else by_key.get(_normalize(name))
        if exact:
            if exact not in usable:
                usable.append(exact)
        else:
            skipped.append(name)

    want_vision = (kind == "vision")
    for model_id in ids:
        if model_id in usable:
            continue
        if kind is None or is_vision_model(model_id) == want_vision:
            usable.append(model_id)
    return usable, skipped


def _ini_path() -> str:
    """Where models.ini actually lives. run.bat sets MODELS_INI to the copy
    next to the launcher; fall back to one alongside this file so the
    settings endpoints still work if that env var is missing."""
    return os.environ.get("MODELS_INI") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models.ini")


_ini_cache = {"key": None, "configs": None, "defaults": None}


def _ini_key(path: str):
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return (path, None, None)


def _load_ini(path: str) -> configparser.ConfigParser:
    # default_section="*" makes [*] behave exactly like llama-server's own
    # preset defaults: every other section inherits its keys unless it sets
    # its own. interpolation=None so a literal "%" in a Windows path (or
    # anywhere else) is never mistaken for %-style interpolation syntax.
    cp = configparser.ConfigParser(default_section="*", interpolation=None)
    cp.optionxform = str  # keep "ctx-size" etc. exactly as written, not lowercased-mangled
    if os.path.exists(path):
        cp.read(path, encoding="utf-8")
    return cp


_RESOURCE_KEYS = {"ngl": "ngl", "ctx_size": "ctx-size", "batch_size": "batch-size", "threads": "threads"}


@serialized_ini
def list_model_configs() -> list[dict]:
    """Every model preset in models.ini, with its effective resource settings
    (own value if set, else the [*] default). This is what the Settings UI
    lists under Resources, and what force-load uses to know what else is
    configured to evict a stuck model with.

    Parsed once per change of the file (mtime/size), not on every call - the
    agent loop and every UI poll ask for context limits."""
    path = _ini_path()
    key = _ini_key(path)
    if _ini_cache["key"] == key and _ini_cache["configs"] is not None:
        return [dict(c) for c in _ini_cache["configs"]]
    cp = _load_ini(path)
    out = []
    for section in cp.sections():
        sec = cp[section]

        def _num(ini_key):
            raw = sec.get(ini_key)
            if raw is None or raw == "":
                return None
            try:
                return int(raw)
            except ValueError:
                return raw

        out.append({
            "name": section,
            "model_path": sec.get("model", ""),
            "mmproj_path": sec.get("mmproj", ""),
            "vision": bool(sec.get("mmproj")) or looks_like_vision(section),
            "ngl": _num("ngl"),
            "ctx_size": _num("ctx-size"),
            "batch_size": _num("batch-size"),
            "threads": _num("threads"),
        })
    _ini_cache.update(key=key, configs=[dict(c) for c in out])
    return out


@serialized_ini
def get_default_resources() -> dict:
    """The [*] section's resource settings - what a model uses if it has no
    override of its own."""
    path = _ini_path()
    cp = _load_ini(path)
    d = cp[cp.default_section]

    def _num(ini_key):
        raw = d.get(ini_key)
        try:
            return int(raw) if raw not in (None, "") else None
        except ValueError:
            return raw

    return {"ngl": _num("ngl"), "ctx_size": _num("ctx-size"),
            "batch_size": _num("batch-size"), "threads": _num("threads")}


@serialized_ini
def update_model_config(name: str, **fields) -> None:
    """Writes ngl / ctx_size / batch_size / threads overrides into model
    `name`'s own [section] in models.ini. Only keys actually passed in
    `fields` are touched; a value of None clears that override so the model
    falls back to the [*] default again. This is what lets the Settings UI
    hand out a VRAM/RAM budget (ngl, ctx-size, batch-size) per model without
    the user hand-editing the ini file.

    NOTE: llama-server only re-reads a model's preset when it (re)loads that
    model. If `name` is the model currently resident in the router, these
    changes take effect on its NEXT load - call force_load() (which evicts
    first) to apply them immediately rather than waiting for the next swap.
    """
    path = _ini_path()
    cp = _load_ini(path)
    if name not in cp.sections():
        raise KeyError(f"No model named '{name}' in {path}")

    for field, ini_key in _RESOURCE_KEYS.items():
        if field not in fields:
            continue
        value = fields[field]
        if value is None or value == "":
            try:
                cp.remove_option(name, ini_key)
            except configparser.NoOptionError:
                pass
        else:
            cp.set(name, ini_key, str(int(value)))

    # configparser.write() re-serializes the whole file and drops anything it
    # doesn't model - comments and commented-out lines included. That's a one
    # -way loss the user never asked for, so the pre-edit file is always
    # saved to <models.ini>.bak first (overwritten each edit, i.e. "undo my
    # last change" rather than a full history) before the normalized version
    # is written.
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as src, \
                 open(path + ".bak", "w", encoding="utf-8") as dst:
                dst.write(src.read())
        except OSError:
            pass  # best-effort safety net; never block the actual save on it

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        cp.write(f)
    atomic_io.replace(tmp, path)
    _ini_cache.update(key=None, configs=None)


# Preset keys diagnostics.py may set (llama-server option names without the dashes).
OPTION_KEYS = {"ctx-size", "ngl", "batch-size", "ubatch-size", "threads", "flash-attn", "cache-type-k", "cache-type-v",
               "n-cpu-moe", "fit", "fit-target", "load-mode", "moe-cache-mib"}   # moe-cache-mib: moe_test.py


@serialized_ini
def model_options(name: str) -> dict:
    """The model's effective preset options (its own, else the [*] defaults), as written."""
    cp = _load_ini(_ini_path())
    if name == "*":
        return dict(cp[cp.default_section])
    return dict(cp[name]) if name in cp.sections() else {}


@serialized_ini
def model_own_options(name: str) -> dict:
    """Only the options model `name`'s own section sets (not what it inherits from [*])."""
    cp = configparser.ConfigParser(default_section="\0none", interpolation=None)
    cp.optionxform = str
    path = _ini_path()
    if os.path.exists(path):
        cp.read(path, encoding="utf-8")
    return dict(cp[name]) if name in cp.sections() else {}


@serialized_ini
def set_model_options(changes: dict, backup: bool = True) -> str | None:
    """Write several models' options at once: {section ("*" for the defaults):
    {option: value, or None to remove it}}. Only OPTION_KEYS. The file as it
    was is kept as models.ini.bak-<time> (unless backup=False: a follow-up write
of one fix). Returns the backup's path. Applies
    to each model the next time it loads."""
    path = _ini_path()
    cp = _load_ini(path)
    for section, options in changes.items():
        if section != "*" and section not in cp.sections():
            raise KeyError(f"No model named '{section}' in {path}")
        target = cp.default_section if section == "*" else section
        for key, value in options.items():
            if key not in OPTION_KEYS:
                raise ValueError(f"'{key}' is not an option diagnostics may set.")
            if value is None:
                if section == "*":
                    cp.defaults().pop(key, None)
                else:
                    try:
                        cp.remove_option(target, key)
                    except configparser.NoOptionError:
                        pass
            elif section == "*":
                cp.defaults()[key] = str(value)
            else:
                cp.set(target, key, str(value))
    backup_path = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}" if backup else None
    if backup_path and os.path.exists(path) and not os.path.exists(backup_path):   # two in one second: keep the older
        shutil.copy2(path, backup_path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        cp.write(f)
    atomic_io.replace(tmp, path)
    _ini_cache.update(key=None, configs=None)
    return backup_path


def model_context_limit(model_name: str = None) -> int:
    """The real context window the given model (or the server's launch
    default, if no model or no override is found) will actually run with.

    Order of trust:
      1. what the running router reports for this model (/v1/models status
         args) - the command line it actually uses. Verified against build
         10621: a global --ctx-size on the router overrides a preset's
         ctx-size, so a model configured for 250000 runs with 65536;
      2. the model's ctx-size in models.ini;
      3. the launcher's global --ctx-size (context_limit()).
    """
    fallback = context_limit()
    if not model_name:
        return fallback
    reported = router_status(model_name).get("ctx")
    if reported:
        return reported
    try:
        configs = list_model_configs()
    except Exception:
        return fallback
    key = _normalize(model_name)
    for cfg in configs:
        if cfg["name"] == model_name or _normalize(cfg["name"]) == key:
            return cfg["ctx_size"] or fallback
    return fallback


@local_runtime.exclusive
def force_load(model_id: str, timeout: float = 180.0, back: bool = True) -> float:
    """Sends a throwaway 1-token completion naming model_id, so the router
    loads (or, if a DIFFERENT model is resident under --models-max 1, swaps
    to) it right now instead of paying that latency on the user's next real
    message. Returns elapsed seconds. Raises on failure (timeout, model not
    found, server down) - the caller decides how to report that.
    First the router moves to the build the model needs (llama_builds.py, D51);
    back=False never moves it back to the newest build (an eviction step)."""
    t0 = time.time()
    import llama_builds
    llama_builds.ensure_for(model_id, back=back)
    resp = requests.post(
        f"{LLAMA_SERVER_BASE.rstrip('/')}/chat/completions",
        json={"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        timeout=timeout, proxies={"http": None, "https": None},
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # The router's own words ("model name=X failed to load") are in the body, not in requests' "500 Server
        # Error" line - load_failure.explain needs them to find the real cause (1.4.7).
        body = (resp.text or "").strip()[:300]
        text, hint = f"{e}{f' - {body}' if body else ''}", load_hint(model_id)
        if hint:
            raise RuntimeError(f"{text}.{hint}") from e
        raise requests.HTTPError(text, response=resp) from e
    return time.time() - t0


@local_runtime.exclusive
def unload_all(timeout: float = 30.0) -> list[str]:
    """Unloads every model the router has loaded (POST /models/unload, one at a time - there is no
    "unload all"), so the GPU is free for something else (image_gen.py). The next chat message loads its
    model again. Returns the models unloaded; [] when none was loaded or the server is down."""
    server_models(force=True)
    loaded = [m for m, st in (_cache.get("status") or {}).items() if st.get("loaded")]
    root = LLAMA_SERVER_BASE.rstrip("/").removesuffix("/v1")
    done = []
    for model in loaded:
        try:
            requests.post(f"{root}/models/unload", json={"model": model}, timeout=timeout,
                          proxies={"http": None, "https": None}).raise_for_status()
            done.append(model)
        except requests.RequestException:
            continue
    if done:
        invalidate_cache()
    return done


def cheapest_other_model(exclude: str, candidates: list[str]) -> str | None:
    """The smallest model (by file size) other than `exclude` - the cheapest
    way to force a --models-max 1 router to unload `exclude`. Unknown sizes
    sort last, so a 20 GB vision model is never picked when a 1 GB one exists."""
    key = _normalize(exclude)
    others = [m for m in candidates if _normalize(m) != key]
    if not others:
        return None
    return min(others, key=lambda m: (model_file_size(m) is None, model_file_size(m) or 0))


def unreachable_message(restarted: str = "") -> str:
    """Kept parseable: the page matches "llama-server is not answering". No run.bat in it (1.4.7) - the installed
    app has none; the app restarts llama-server itself (llama_builds.ensure_running)."""
    tail = f" {restarted}" if restarted else (" The app tried to start it again and couldn't - see Settings › "
                                             "Diagnostics, or switch the mode selector to Cloud.")
    return f"llama-server is not answering on {LLAMA_SERVER_BASE}.{tail}"


def router_log_path() -> str:
    """The router's log when the app runs it windowless (launcher.pyw, llama_builds)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs", "llama-server.log")


def empty_registry_message() -> str:
    return (f"llama-server is running on {LLAMA_SERVER_BASE} but reports no usable model for this "
            "request. Check models.ini, or pick a different model in Local Models.")
