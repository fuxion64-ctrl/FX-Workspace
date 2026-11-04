"""
core/config.py
==============
Constants and persisted settings (settings.json): defaults, loading, cleaning, saving.
(Split out of main.py in Phase 2 - see ARCHITECTURE.md D14.)
"""
from __future__ import annotations

import json
import os
import re
import awareness
import desktop_input as di
import llama_builds
import llama_local
import effort
import netlayer
import permissions
import orchestrator
import prompt_prep
import voice
import personas
import image_gen
import video_gen
import music_gen
import careful
import response_modes
import auto_tools
import skills
import system_messages
import atomic_io
import tempfile




IS_WINDOWS = di.IS_WINDOWS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MAX_AGENT_ITERATIONS = 12


# Local mode takes one tool call per turn, so the same task needs more turns to
# reach the same place - with a lower ceiling it would hit the limit mid-plan.
MAX_LOCAL_ITERATIONS = 24
MAX_NUDGES = 2
MAX_BAD_CALL_ROUNDS = 3         # replies in a row with only unreadable/invalid tool calls before giving up
MAX_REPEAT_ROUNDS = 3           # replies in a row with only repeated calls before ending the run
MAX_VOLATILE_REPEATS = 3        # identical reads of changing state (screen, windows) allowed per machine state
MAX_HISTORY_CHARS = 60_000      # cloud budget


# A local reply may carry a whole script or file as a tool argument; 700
# tokens cut those off mid-JSON. The history budget reserves the same room.
LOCAL_MAX_TOKENS = 2048


# Extra reply room while the model may think (Thorough / Max): its
# reasoning is generated inside the same max_tokens. See _local_reply_tokens.
LOCAL_THINKING_TOKENS = 2048
LOCAL_RESPONSE_RESERVE = LOCAL_MAX_TOKENS   # tokens kept free for the local model's reply (without thinking)
MIN_KEPT_TURNS = 2
IMAGES_KEPT_CLOUD = 2
IMAGES_KEPT_LOCAL = 1
SHOT_SIZE_CLOUD = (1600, 1000)
SHOT_SIZE_LOCAL = (1024, 640)
SCREEN_UNCHANGED_THRESHOLD = 1.5
PRIMARY_MODELS = ["gemini/gemini-3.6-flash", "gemini/gemini-3.8-flash", "gemini/gemini-3.5-flash-lite", "gemini/gemini-3.1-pro", "groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b"]
VISION_MODELS = ["gemini/gemini-3.6-flash", "gemini/gemini-3.8-flash", "gemini/gemini-3.1-pro", "groq/meta-llama/llama-4-scout-17b-16e-instruct"]
LOCAL_TEXT_MODELS = ["Gemma-4-E4B-Tools", "DeepSeek-R1-1.5B", "Phi-4-mini-instruct-Q2_K_L.gguf", "Qwen-3.5-4B-Q4_K_M", "Mistral-7B", "DeepSeek-R1-7B", "Qwen2.5-Coder-7B-Instruct-abliterated-Q5_K_M", "DeepSeek-Coder-V2"]
LOCAL_VISION_MODELS = ["Gemma-4-Vision-Q4", "Gemma-4-Vision-Q8"]
LLAMA_SERVER_BASE = llama_local.LLAMA_SERVER_BASE
LOCAL_API_KEY = "sk-local-key"
MODE_CLOUD, MODE_LOCAL, MODE_AUTO = "cloud", "local", "auto"
VALID_MODES = {MODE_CLOUD, MODE_LOCAL, MODE_AUTO}


# ==============================================================================
# Persisted settings (favorites, theme/layout, etc.) - separate from
# app_state, which is in-memory-only conversation state that /api/reset
# clears. Settings survive a reset and a restart of this process.
# ==============================================================================
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
DEFAULT_SETTINGS = {
    # Empty list = unrestricted (existing behaviour: try every configured/
    # reported model of that kind). A non-empty list restricts auto-selection
    # to ONLY those models - it does not affect an explicit pin in the model
    # picker, which is already a stronger, explicit override.
    "favorite_local_text_models": [],
    "favorite_local_vision_models": [],
    "favorite_cloud_text_models": [],
    "favorite_cloud_vision_models": [],
    # Appearance: a family and a light/dark mode (the top bar's button flips the mode).
    "appearance": "classic",      # classic | sunrise | terminal | eclipse | aurora
    "color_mode": "system",       # system (follow Windows) | light | dark
    "terminal_phosphor": False,   # Terminal's dark side: the original phosphor green instead of Lite Dark
    "theme": "system",            # the concrete theme the three above make (clean_appearance) - kept for older files
    "look": "studio",             # studio | focus | operator | command (UI layout)
    "layout": "comfortable",
    "accent": "",                  # "" = the default gold (follows light/dark); a hex = a chosen accent
    "chat_font": "serif",         # serif | sans  (assistant replies)
    "think_mark": "auto",         # auto | orbs | legacy  (the thinking mark; auto = each look's own mark since 1.4.6)
    "marks_146": False,           # the one-time move from a saved "Orbs" to the 1.4.6 marks has run (load_settings)
    "user_name": "",
    # Privacy: cloud models never see screenshots unless this is on.
    "cloud_allow_screenshots": False,
    # Agent pipeline (orchestrator.py)
    "effort": "normal",                 # auto | low | normal | high | extra | max | ultra (effort.py)
    "effort_by_model": {},              # {model: level} - remembered per model
    "effort_overrides": {},             # {model: {level: {limit: number}}} - tuned limits only
    "cross_check": False,                # optional second-model review, independent of effort
    "effort_auto_escalate": True,       # Balanced -> Thorough for genuinely complex tasks
    "prompt_enhancement": "minimal",    # legacy (before Prompt Prep); read only to migrate
    # Prompt Prep (prompt_prep.py): off | clarify | expand | plan. None here means
    # "not chosen yet": clean_agent_settings derives it from prompt_enhancement.
    "prompt_prep": None,
    "prompt_prep_by_model": {},         # {model: mode} - remembered per model, like effort
    "auto_model_routing": False,        # let the agent request a model handoff
    "max_revisions": "auto",            # auto (effort decides) | 0 | 1 | 2
    "show_activity": False,             # planning/review rows under each reply (tool results always show) - D48: off
    "record_model_io": False,           # save local runs as replay recordings (data/recordings, recorder.py)
    "memory_enabled": True,             # put relevant saved memory (memory.py) in each run's brief
    "web_search_provider": "duckduckgo",  # duckduckgo (no account) | brave | tavily (a key, search_engine.KeyStore) | searxng (the user's own instance)
    "searxng_url": "",                  # e.g. http://127.0.0.1:8888 - only used with provider searxng
    # Web Search privacy (search_engine.MODES): standard | private | max | local | disabled
    "search_mode": "standard",
    "search_preview": False,            # show each search query to edit or cancel before it's sent
    "search_sources": True,             # also forums, Q&A sites, issues, papers (deep_sources.py, 1.4.4) in Standard mode
    "search_cache_minutes": 15,         # reuse a search's results for this long (0: never; never in Max Privacy)
    "search_history": False,            # keep searches and pages read on disk (search_history.py) - opt-in
    "search_history_days": 30,
    **netlayer.DEFAULT_SETTINGS,        # Settings > Network (netlayer.py)
    **voice.DEFAULT_SETTINGS,           # Settings > Voice (voice.py)
    **personas.DEFAULT_SETTINGS,        # the composer's persona (personas.py)
    # The Studio look's layout (index.html): classic | overlay - live panels over the workspace.
    "studio_layout": "classic",
    "overlay_panels": {},               # {panel: {shown, collapsed, x, y, dock}} - positions persist
    "overlay_clickthrough": False,
    "overlay_opacity": 0.88,
    "overlay_window": False,            # the mini window: always on top (desktop/main.js; app_window.py outside it)
    "mini_place": "studio",             # where it sits: over Roblox Studio (follows it), a screen corner, or where dragged
    "close_to_tray": True,              # closing the window keeps the app running in the tray (desktop/main.js)
    "start_with_windows": False,        # the desktop app starts when you sign in (Electron's login item)
    "start_hidden": True,               # ...in the tray, without a window
    "idle_unload": "15",                # minutes hidden and unused before the model leaves the GPU (idle_unload.py)
    "chat_warmup": True,                # typing a message loads the local chat model meanwhile (/api/local-models/warm)
    "update_check": True,               # look for a newer version by itself; Home offers it (app_version.start_auto)
    "claude_mark": "regular",           # the Claude look's thinking mark: regular (3D orbs) or liquid
    "ai_looks": "new",                  # Claude/ChatGPT/Grok/Gemini looks: new (from their design files, 1.4.6) or original
    "unlocked_themes": [],              # secret themes found (index.html SECRET_THEMES); never switched on by itself
    **image_gen.DEFAULT_SETTINGS,       # Settings > Images (image_gen.py)
    # MossadGPT's world monitor (osint/): feeds collected on this PC while it's on screen.
    "osint_background": False,          # also collect when MossadGPT isn't on screen
    "osint_layers": {},                 # {layer: false} switches a feed off (osint.collector.LAYERS)
    "osint_areas": [{"name": "Israel", "lat": 31.8, "lon": 35.0, "nm": 250}],   # flights are read around these
    "osint_days": 7,                    # history and events kept this long...
    "osint_max_mb": 2048,               # ...and at most this much (data/osint/osint.db)
    "research_depth": "auto",           # Research's default depth (agent/research.DEPTHS); the composer can override it
    "sec_contact": "",                  # the User-Agent contact SEC EDGAR asks for (finance.py); empty: the app's own
    "web_browser_fallback": True,       # read/search in the assistant's browser when a site refuses a direct request
    "browser_builtin": True,            # the assistant's browser runs hidden and shows in the app's Browser panel (D50)
    "llama_build": "auto",              # which llama.cpp runs local models: auto | latest | prism (llama_builds.py, D51)
    "web_auto_search": True,            # offer web_search / fetch_page to questions too, not only to read/action turns
    **system_messages.DEFAULT_SETTINGS, # Settings > Agent > System message
    **response_modes.DEFAULT_SETTINGS,  # Enhanced (the agent) or Raw (the model alone) - response_modes.py
    **careful.DEFAULT_SETTINGS,         # Careful and honest (careful.py): check answers after writing them
    **auto_tools.DEFAULT_SETTINGS,      # turn on safe tools by itself (auto_tools.py)
    **skills.DEFAULT_SETTINGS,          # action tools in skills, loaded as needed (skills.py)
    "presets": [],                      # the composer's saved presets: [{id, name, system, tools, model: {mode, id}, effort}]
    "ambient_idle": "10",               # minutes without input before the idle screen: off | 5 | 10 | 30
    "context_auto_compress": "90",      # % of the context limit that triggers compression: off | 80 | 90 | 95 | 97 (1.4.6: was 97)
    **awareness.DEFAULT_SETTINGS,
    # Permission mode/category-defaults/per-tool overrides - see
    # permissions.py, the single source of truth for their shape and for
    # how they combine into an auto/ask/deny decision per tool call.
    **permissions.DEFAULT_PERMISSION_SETTINGS,
}


UI_LOOKS = ("studio", "focus", "operator", "command")
LEGACY_LOOKS = {"companion": "focus"}      # Companion's narrow-window job moved into Focus
UI_THEMES = ("system", "light", "dark", "dawn", "dusk", "terminal", "termdark", "termlight",
             "eclipse", "eclipselight", "aurora", "auroralight",
             "claylight", "claydark", "monolight", "monodark", "voidlight", "voiddark", "prismlight", "prismdark")
# clay / mono / void / prism: the looks people know from Claude, ChatGPT and Codex, Grok and Gemini (2026-10-21).
UI_APPEARANCES = ("classic", "sunrise", "terminal", "eclipse", "aurora", "clay", "mono", "void", "prism")
COLOR_MODES = ("system", "light", "dark")
# Each family's (light, dark) theme. Terminal's dark one is Lite Dark unless terminal_phosphor.
FAMILY_THEMES = {"classic": ("light", "dark"), "sunrise": ("dawn", "dusk"), "terminal": ("termlight", "termdark"),
                 "eclipse": ("eclipselight", "eclipse"), "aurora": ("auroralight", "aurora"),
                 "clay": ("claylight", "claydark"), "mono": ("monolight", "monodark"),
                 "void": ("voidlight", "voiddark"), "prism": ("prismlight", "prismdark")}
# A theme from before families -> (appearance, color_mode, terminal_phosphor).
THEME_FAMILY = {"system": ("classic", "system", False), "light": ("classic", "light", False),
                "dark": ("classic", "dark", False), "dawn": ("sunrise", "light", False),
                "dusk": ("sunrise", "dark", False), "terminal": ("terminal", "dark", True),
                "termdark": ("terminal", "dark", False), "termlight": ("terminal", "light", False),
                "eclipse": ("eclipse", "dark", False), "eclipselight": ("eclipse", "light", False),
                "aurora": ("aurora", "dark", False), "auroralight": ("aurora", "light", False),
                **{f"{fam}{side}": (fam, side, False) for fam in ("clay", "mono", "void", "prism") for side in ("light", "dark")}}


def clean_look(look) -> str:
    look = LEGACY_LOOKS.get(look, look)
    return look if look in UI_LOOKS else "studio"


def clean_appearance(s: dict, from_theme: bool = False) -> dict:
    """The appearance family, light/dark mode and Terminal's phosphor option, plus "theme" - the
    concrete theme they make (older code and settings files know only that). from_theme: an older
    settings.json, or a client that sent only "theme" - the family comes from that theme."""
    if from_theme:
        theme = s.get("theme")
        # Themes from the UI before these ones map onto the light/dark pair.
        theme = theme if theme in UI_THEMES else "light" if theme == "light" else "dark" if theme else "system"
        appearance, mode, phosphor = THEME_FAMILY[theme]
    else:
        appearance = s.get("appearance") if s.get("appearance") in UI_APPEARANCES else "classic"
        mode = s.get("color_mode") if s.get("color_mode") in COLOR_MODES else "system"
        phosphor = s.get("terminal_phosphor") is True
    light, dark = FAMILY_THEMES[appearance]
    if appearance == "terminal" and phosphor:
        dark = "terminal"
    if mode == "system":
        theme = "system" if appearance == "classic" else dark
    else:
        theme = light if mode == "light" else dark
    return {"appearance": appearance, "color_mode": mode, "terminal_phosphor": phosphor, "theme": theme}
AMBIENT_IDLE = ("off", "5", "10", "30")    # idle-screen delay in minutes (index.html "Idle")
CONTEXT_COMPRESS_AT = ("off", "80", "90", "95", "97")   # agent/context_window.THRESHOLDS
WEB_SEARCH_PROVIDERS = ("duckduckgo", "brave", "tavily", "searxng")      # search_engine.chain


def clean_agent_settings(s: dict) -> dict:
    """Degrade bad agent-pipeline values to their defaults."""
    out = {}
    out["effort"] = s.get("effort") if s.get("effort") in effort.CHOICES else "normal"
    out["prompt_enhancement"] = (s.get("prompt_enhancement") if s.get("prompt_enhancement")
                                 in orchestrator.ENHANCEMENT_MODES else "minimal")
    out["prompt_prep"] = prompt_prep.migrated(s)
    pp_model = s.get("prompt_prep_by_model")
    out["prompt_prep_by_model"] = {m: v for m, v in (pp_model.items() if isinstance(pp_model, dict) else [])
                                   if isinstance(m, str) and v in prompt_prep.MODES}
    for key, default in (("cross_check", False), ("effort_auto_escalate", True), ("auto_model_routing", False), ("show_activity", False),
                         ("record_model_io", False), ("memory_enabled", True), ("web_browser_fallback", True),
                         ("web_auto_search", True), ("browser_builtin", True),
                         # memory.py sensitive topics; profile_store.py proposing a new profile by itself; agent_skills offers
                         ("memory_sensitive", False), ("profile_auto", True), ("skills_offer", True)):
        out[key] = bool(s.get(key)) if isinstance(s.get(key), bool) else default
    # The interface: "new" (the 2026-10 redesign) or "legacy" (everything as it was before it).
    out["ui_style"] = s.get("ui_style") if s.get("ui_style") in ("new", "legacy") else "new"
    kf = s.get("knowledge_folders")                      # knowledge.py: the folders the knowledge base indexes
    out["knowledge_folders"] = list(dict.fromkeys(f.strip() for f in kf if isinstance(f, str) and f.strip()))[:20] if isinstance(kf, list) else []
    mr = s.get("max_revisions")
    out["max_revisions"] = int(mr) if not isinstance(mr, bool) and str(mr) in ("0", "1", "2") else "auto"
    by_model = s.get("effort_by_model")
    out["effort_by_model"] = {m: l for m, l in (by_model.items() if isinstance(by_model, dict) else [])
                              if isinstance(m, str) and l in effort.CHOICES}
    out["effort_overrides"] = effort.clean_overrides(s.get("effort_overrides"))
    out["ambient_idle"] = str(s.get("ambient_idle")) if str(s.get("ambient_idle")) in AMBIENT_IDLE else "10"
    out["context_auto_compress"] = (str(s.get("context_auto_compress")) if str(s.get("context_auto_compress"))
                                    in CONTEXT_COMPRESS_AT else "90")
    out["web_search_provider"] = (s.get("web_search_provider") if s.get("web_search_provider") in WEB_SEARCH_PROVIDERS
                                 else "duckduckgo")
    url = str(s.get("searxng_url") or "").strip()
    out["searxng_url"] = url[:300] if url.lower().startswith(("http://", "https://")) else ""
    out["presets"] = clean_presets(s.get("presets"))
    out.update(system_messages.clean_settings(s))
    out.update(response_modes.clean_settings(s))
    out.update(careful.clean_settings(s))
    out.update(auto_tools.clean_settings(s))
    out.update(skills.clean_settings(s))
    out.update(clean_privacy_settings(s))
    return out


SEARCH_MODES = ("standard", "private", "max", "local", "disabled")        # search_engine.MODES
OVERLAY_PANELS = ("model", "studio", "network", "session")                # index.html Overlay
SECRET_THEMES = ("judaism",)
RESEARCH_DEPTHS = ("auto", "quick", "standard", "deep", "ultra")    # agent/research.DEPTHS + auto


def _num(value, lo, hi, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if v != v else max(lo, min(hi, v))


MINI_PLACES = ("studio", "tr", "tl", "br", "bl", "free")     # the mini window: over Studio, a corner, or dragged
IDLE_UNLOAD = ("off", "5", "15", "30", "60")   # minutes hidden in the tray and unused before the model leaves the GPU


def clean_privacy_settings(s: dict) -> dict:
    """Web Search, Network, the Studio Overlay and secret themes: bad values -> defaults.
    An older settings.json has none of these keys; it loads unchanged and gets the defaults."""
    out = {"search_mode": s.get("search_mode") if s.get("search_mode") in SEARCH_MODES else "standard"}
    for key, default in (("search_preview", False), ("search_history", False), ("overlay_clickthrough", False),
                         ("overlay_window", False), ("close_to_tray", True), ("start_with_windows", False),
                         ("start_hidden", True), ("search_sources", True), ("chat_warmup", True), ("update_check", True)):
        out[key] = s.get(key) if isinstance(s.get(key), bool) else default
    out["claude_mark"] = s.get("claude_mark") if s.get("claude_mark") in ("regular", "liquid") else "regular"
    out["ai_looks"] = s.get("ai_looks") if s.get("ai_looks") in ("new", "original") else "new"
    out["mini_place"] = s.get("mini_place") if s.get("mini_place") in MINI_PLACES else "studio"
    out["idle_unload"] = str(s.get("idle_unload")) if str(s.get("idle_unload")) in IDLE_UNLOAD else "15"
    out["search_cache_minutes"] = int(_num(s.get("search_cache_minutes", 15), 0, 1440, 15))
    out["search_history_days"] = int(_num(s.get("search_history_days", 30), 1, 365, 30))
    out.update(netlayer.clean_settings(s))
    out.update(voice.clean_settings(s))
    out.update(personas.clean_settings(s))
    out["studio_layout"] = s.get("studio_layout") if s.get("studio_layout") in ("classic", "overlay") else "classic"
    panels = s.get("overlay_panels") if isinstance(s.get("overlay_panels"), dict) else {}
    out["overlay_panels"] = {}
    for name in OVERLAY_PANELS:
        p = panels.get(name) if isinstance(panels.get(name), dict) else {}
        if not p:
            continue
        clean = {k: bool(p[k]) for k in ("shown", "collapsed") if isinstance(p.get(k), bool)}
        for k in ("x", "y"):                      # a fraction of the window (0..1), so it survives a resize
            if isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool):
                clean[k] = round(_num(p[k], 0, 1, 0), 4)
        if p.get("dock") in ("tl", "tr", "bl", "br", "free"):
            clean["dock"] = p["dock"]
        out["overlay_panels"][name] = clean
    out["overlay_opacity"] = round(_num(s.get("overlay_opacity", 0.88), 0.3, 1.0, 0.88), 2)
    out["unlocked_themes"] = [t for t in SECRET_THEMES if t in (s.get("unlocked_themes") or [])]
    out.update(clean_osint_settings(s))
    out.update(image_gen.clean_settings(s))
    out.update(video_gen.clean_settings(s))
    out.update(music_gen.clean_settings(s))
    out["research_depth"] = s.get("research_depth") if s.get("research_depth") in RESEARCH_DEPTHS else "auto"
    out["llama_build"] = s.get("llama_build") if s.get("llama_build") in llama_builds.CHOICES else "auto"
    contact = re.sub(r"[\r\n\t]", " ", str(s.get("sec_contact") or "")).strip()[:120]
    # SEC wants "Name email@domain"; anything else is dropped rather than sent.
    out["sec_contact"] = contact if re.fullmatch(r"[\w .,'&()-]{1,80}\s+[\w.+-]+@[\w-]+(\.[\w-]+)+", contact) else ""
    return out


OSINT_LAYERS = ("flights", "military", "flights_global", "satellites", "space", "quakes", "news",
                "jamming", "alerts")                                                                 # osint.collector.LAYERS


def clean_osint_settings(s: dict) -> dict:
    """The world monitor's settings: bad values -> defaults; at most 6 watched areas of 10-250 nm."""
    areas = []
    for a in s.get("osint_areas") if isinstance(s.get("osint_areas"), list) else DEFAULT_SETTINGS["osint_areas"]:
        if not isinstance(a, dict):
            continue
        try:
            lat, lon = float(a.get("lat")), float(a.get("lon"))
        except (TypeError, ValueError):
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            areas.append({"name": re.sub(r"\s+", " ", str(a.get("name") or "Area")).strip()[:40] or "Area",
                          "lat": round(lat, 4), "lon": round(lon, 4), "nm": int(_num(a.get("nm", 250), 10, 250, 250))})
    layers = s.get("osint_layers") if isinstance(s.get("osint_layers"), dict) else {}
    return {"osint_background": s.get("osint_background") is True,
            "osint_layers": {k: bool(v) for k, v in layers.items() if k in OSINT_LAYERS and isinstance(v, bool)},
            "osint_areas": areas[:6],
            "osint_days": int(_num(s.get("osint_days", 7), 1, 90, 7)),
            "osint_max_mb": int(_num(s.get("osint_max_mb", 2048), 100, 50_000, 2048))}


def clean_presets(value) -> list:
    """The composer's presets (index.html savePreset), at most 20, with only known fields."""
    import composer_tools
    out = []
    for p in value if isinstance(value, list) else []:
        if not isinstance(p, dict) or not str(p.get("name") or "").strip():
            continue
        model = p.get("model") if isinstance(p.get("model"), dict) else {}
        out.append({"id": str(p.get("id") or "")[:16] or str(len(out)), "name": str(p["name"]).strip()[:40],
                    "system": str(p.get("system") or "")[:4000],
                    "tools": [t for t in (p.get("tools") or []) if t in composer_tools.IDS and t != "files"],
                    "model": {"mode": model.get("mode") if model.get("mode") in VALID_MODES else "local",
                              "id": str(model.get("id") or "")[:200]},
                    "effort": p.get("effort") if p.get("effort") in effort.CHOICES else ""})
    return out[:20]


def load_settings() -> dict:
    data = dict(DEFAULT_SETTINGS)
    saved = {}
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k in DEFAULT_SETTINGS:
                if k in saved:
                    data[k] = saved[k]
    except Exception as e:
        print(f"[WARN] Could not read settings.json, using defaults: {e}")
    data.update(permissions.clean_settings(data))
    data.update(awareness.clean_settings(data))
    # A settings.json from before appearance families has only "theme": the family comes from it.
    data.update(clean_appearance(data, from_theme=isinstance(saved, dict) and "theme" in saved and "appearance" not in saved))
    # The old defaults (violet, then clay) become the new one - the logo's gold. Terracotta stays one click away.
    if data.get("accent") in ("#7c6cf0", "#c96442"):
        data["accent"] = ""
    if data.get("chat_font") not in ("serif", "sans"):
        data["chat_font"] = "serif"
    if data.get("think_mark") not in ("auto", "orbs", "legacy"):
        data["think_mark"] = "auto"
    # 1.4.6, once: "Orbs" chosen back when Auto meant the orbs becomes the new per-look marks (Auto). Orbs stays one
    # click away in Settings > Appearance > Extras, and a choice made after this is kept.
    if not data.get("marks_146"):
        if data.get("think_mark") == "orbs":
            data["think_mark"] = "auto"
        data["marks_146"] = True
    data["look"] = clean_look(data.get("look"))
    data.update(clean_agent_settings(data))
    return data


def save_settings(settings: dict) -> bool:
    pending = None
    try:
        # A failed save must not leave a partial settings file.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=os.path.dirname(SETTINGS_PATH),
                                         prefix=".settings-", suffix=".tmp", delete=False) as f:
            pending = f.name
            json.dump(settings, f, indent=2)
        atomic_io.replace(pending, SETTINGS_PATH)
        pending = None
        return True
    except Exception as e:
        print(f"[WARN] Could not write settings.json: {e}")
        return False
    finally:
        if pending and os.path.exists(pending):
            os.remove(pending)


def _dump_model(m) -> dict:
    """pydantic v1/v2 compatibility shim - the rest of the file only ever
    used direct attribute access, so this is new and kept local to the two
    settings endpoints that need dict-of-only-provided-fields semantics."""
    if hasattr(m, "model_dump"):
        return m.model_dump(exclude_unset=True)
    return m.dict(exclude_unset=True)
