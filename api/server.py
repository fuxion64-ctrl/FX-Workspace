"""
api/server.py
=============
The FastAPI app: middleware, every HTTP route, static UI.
(Split out of main.py in Phase 2 - see ARCHITECTURE.md D14.)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from types import SimpleNamespace
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import app_registry
import approval
import awareness
import app_version
import diagnostics
import browser
import context as ctx
import elements
import llama_local
import llama_builds
import effort
import memory
import prompt_prep
import system_messages
import model_caps
import routines
import app_window
import finance
import netlayer
import voice
import voice_duplex
import model_scan
import paths
from starlette.concurrency import iterate_in_threadpool
import permissions
import privacy
import search_engine
import search_history
import roblox_bridge
import stream_hub
from run_control import run_control
import tool_registry
import run_log
import undo_store
import usage
import system_metrics
import web_tools
from urllib.parse import urlparse
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

import agent.benchmark_runner as benchmark_runner
import agent.context_window as context_window
import agent.models as llm
import agent.parsing as parsing
import agent.prompts as prompts
import agent.routine_runner as routine_runner
import agent.sessions as sessions
import agent.tasks as tasks
import core.config as config
import core.observer as observer
import core.state as shared
import events as event_log
import routing.handoff as handoff
import tools.approvals as approvals
import tools.files as files
import tools.runtime as runtime
import tools.schemas as schemas
import image_gen
import video_gen
import video_audio
import music_gen
import labs
import httpx
import chat_guard
import osint.collector as osint_collector
import osint.geo as osint_geo
import osint.satellites as osint_satellites
import osint.store as osint_store


app = FastAPI()
# The network layer reads its settings (Settings > Network) from the live settings.
netlayer.configure(lambda: shared.app_state["settings"])
finance.configure(lambda: shared.app_state["settings"])
osint_collector.configure(lambda: shared.app_state["settings"])


def _save_image_settings(changes: dict) -> None:
    """image_gen.move_models switching the image folder once every copy checked out."""
    current = shared.app_state["settings"]
    current.update(changes)
    current.update(image_gen.clean_settings(current))
    config.save_settings(current)


image_gen.configure(lambda: shared.app_state["settings"], _save_image_settings)
video_gen.configure(lambda: shared.app_state["settings"])
music_gen.configure(lambda: shared.app_state["settings"])


ALLOWED_HOSTS = ["127.0.0.1", "localhost"] + [
    h.strip() for h in os.environ.get("AGENTIC_ALLOWED_HOSTS", "").split(",") if h.strip()]


def _refusal(request: Request) -> str | None:
    """Only this app's own page (and local non-browser clients such as the
    scheduled-routine trigger) may use the API; web pages may not - not even
    other pages on localhost (a dev server on another port is "same-site").
    Browsers mark every request with Sec-Fetch-Site and an Origin on writes;
    both must say "this exact origin". The Roblox bridge is for Studio's
    HttpService only, which sends neither header."""
    path = request.url.path
    site = request.headers.get("sec-fetch-site")
    origin = request.headers.get("origin")
    if path.startswith("/roblox/"):
        if site not in (None, "none") or origin is not None:
            return "The Studio bridge only answers Roblox Studio."
        return None
    if not path.startswith("/api/"):
        return None                         # the page itself may be opened from anywhere
    if site in ("cross-site", "same-site"):
        return "Cross-site request refused."
    if origin is not None:
        own = f"{request.url.scheme}://{request.headers.get('host', '')}".lower()
        if origin.lower().rstrip("/") != own:
            return "Cross-site request refused."
    return None


@app.middleware("http")
async def _same_origin_only(request: Request, call_next):
    why = _refusal(request)
    if why:
        return JSONResponse({"status": "error", "ok": False, "message": why}, status_code=403)
    return await call_next(request)


app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


def _busy(default: str) -> str:
    """The busy message, naming a running benchmark when that is the reason."""
    return benchmark_runner.busy_message() or default


def _gpu_wait() -> bool:
    """Image, video and music jobs wait while the chat model works or a Labs run trains on the GPU."""
    return shared.state_lock.locked() or labs.gpu_busy()


@app.post("/api/chat")
def start_chat(req: shared.ChatRequest):
    """Start a turn in the background and return its id at once. The UI then
    long-polls GET /api/turn/{id}; a reloaded page finds it in /api/state."""
    if not req.message.strip():
        return {"status": "error", "message": "Empty message."}
    open_temp = sessions.temporary_chat()
    if chat_guard.is_locked(shared.app_state.get("active_chat")):
        # Closed for pushing past the chat limits (chat_guard.py): readable, but nothing more is sent to it.
        return {"status": "error", "locked": True, "chat": sessions.public_chat(),
                "message": "This chat was closed for going past the chat limits. Start a new chat to continue."}
    if req.temporary_chat and (not open_temp or open_temp["id"] != req.temporary_chat):
        return {"status": "error", "temporary_ended": True, "message": (
            "That temporary chat has ended (the app restarted or it was closed) - nothing of it was kept. "
            "Start a new chat or a new temporary chat to send this.")}
    # "/plan ..." and "/design ..." (agent/tasks.slash_command): the command
    # travels with the request, the text after it is the request itself.
    command, text = tasks.slash_command(req.message)
    if command:
        if not text:
            return {"status": "error", "message": f"Say what to {command} after /{command}, for example "
                                                  f"\"/{command} {tasks.SLASH_COMMANDS[command]['example']}\"."}
        req.message, req.command = text, command
    else:
        req.command = None                    # only ever set from the message
    if _model_loading() or (shared.state_lock.locked() and not tasks.turns.running()):
        # A model force-load (its own lock), or something other than a turn holds state_lock.
        return {"status": "error", "message": _busy("Busy - a model is loading. Try again in a moment.")}
    if labs.gpu_busy() and getattr(req, "mode", config.MODE_LOCAL) != config.MODE_CLOUD:
        # A local model would load onto the GPU a Labs run is training on - both would run out of memory.
        return {"status": "error", "message": "A Labs run is training on the GPU, so a local model can't load now. "
                                              "Stop the run in Labs, wait for it to finish, or switch to a cloud model."}
    turn = tasks.turns.start(req, {"message": req.message[:500], "with_vision": bool(req.with_vision),
                             "command": req.command, "event_seq": shared.app_state.get("event_seq", 0),
                             "temp": open_temp["id"] if open_temp else None})
    if turn is None:
        return {"status": "error", "message": "Busy - a task is already running. Stop it or wait for it to finish."}
    return {"status": "accepted", "turn": turn}


@app.get("/api/turn/{turn_id}")
def get_turn(turn_id: str, wait: float = 0):
    """Long-poll: returns as soon as the turn finishes, or after `wait`
    seconds (max 25) with status 'running'."""
    turn = tasks.turns.wait(turn_id, max(0.0, min(float(wait or 0), 25.0)))
    if turn is None:
        return {"status": "unknown", "message": "The server has no record of that task (it may have restarted)."}
    return turn


@app.get("/api/tools")
def get_tools():
    """The typed tool catalogue: schema, tier, read-only, retry/timeout/cooldown policy."""
    return {"tools": runtime.TOOL_REGISTRY.describe(), "result_schema": tool_registry.RESULT_SCHEMA}


class BenchmarkBody(BaseModel):
    model: str


@app.get("/api/benchmarks")
def get_benchmarks():
    """Stored results per model, and the benchmark running now (if any)."""
    return benchmark_runner.public()


@app.post("/api/benchmarks")
def start_benchmark(body: BenchmarkBody):
    """Benchmark one local model (Settings > Models): loads it, runs the tool-call
    cases and context probes through the agent's own request path, stores the
    result and updates the model's capabilities. Never during a task."""
    if tasks.turns.running() or _model_loading():
        return {"ok": False, "message": "Busy - a task or model load is running. Run the benchmark when it has finished."}
    try:
        configured = {c["name"] for c in llama_local.list_model_configs()}
    except Exception:
        configured = set()
    names = set(llama_local.server_models() or []) | configured
    if body.model not in names:
        return {"ok": False, "message": "That isn't a local model llama-server knows."}
    res = benchmark_runner.start(body.model)
    res.pop("thread", None)
    return res


@app.post("/api/benchmarks/cancel")
def cancel_benchmark():
    return {"ok": benchmark_runner.cancel()}


@app.get("/api/models/registry")
def get_model_registry():
    return {"routing": bool(shared.app_state["settings"].get("auto_model_routing")),
            "active": llm._current_model(), "models": [m.public() for m in handoff.model_registry.models()]}


@app.get("/api/messaging/status")
def get_messaging_status():
    return shared.messaging_hub.status()


@app.post("/api/run/{action}")
def control_run(action: str):
    """Pause, resume or stop the running request. Takes effect at the agent's
    next checkpoint (before its next model call or tool action)."""
    fn = {"pause": run_control.pause, "resume": run_control.resume, "stop": run_control.stop}.get(action)
    if not fn:
        return {"status": "error", "message": f"Unknown action '{action}'."}
    ok = fn()
    if ok and action == "resume":
        event_log.log_action("Resumed by the user.")
        event_log.emit_event("note", "Resumed")
    return {"status": "success" if ok else "error", "run": run_control.state(),
            "message": "" if ok else "Nothing is running."}


@app.post("/api/reset")
def reset_conversation():
    # Refuse instead of waiting: blocking here made "New chat" hang for the
    # rest of a running task.
    if not shared.state_lock.acquire(timeout=2):
        return {"status": "error", "message": _busy("A task is still running - stop it first.")}
    try:
        _reset_conversation_locked()
        sessions.new_chat()              # the next message starts a new saved chat (in the open project); ends a temporary one
    finally:
        shared.state_lock.release()
    return {"status": "success", "message": "Reset.", "project": sessions.public_project()}


def _reset_conversation_locked():
    """Clear the conversation and task state. Caller holds state_lock."""
    shared.app_state["conversation"] = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
    shared.app_state["context_summary"] = ""               # compressed older turns belong to the old chat
    import skills
    skills.reset_chat()                                    # so do the skills it loaded (skills.py)
    shared.app_state["context_last_compression"] = None
    shared.app_state["last_shot"] = None
    shared.app_state["last_shot_meta"] = None
    shared.app_state["last_shot_at"] = 0.0
    shared.app_state["last_tool_signature_set"] = None
    shared.app_state["local_engaged"] = None
    shared.app_state["intent_guard"] = None
    shared.app_state["pending_clarification"] = None
    shared.app_state["resolved_clarification"] = None
    shared.app_state["current_user_request"] = ""
    shared.app_state["turn_policy"] = {"mode": "none", "allow_mutation": False, "reason": "no active request"}
    shared.app_state["last_desktop_intent"] = None
    shared.app_state["completed_app_tasks"] = []
    shared.app_state["task_queue"] = None
    shared.task_board.clear()
    shared.ledger.clear()


@app.post("/api/clarification-popup")
def clarification_popup(body: shared.ClarificationPopupRequest):
    """Show a native choice dialog when the web workspace is not visible."""
    question = re.sub(r"\s+", " ", (body.question or "")).strip()[:300]
    if not question:
        return {"status": "error", "message": "Missing clarification question."}
    options = parsing._normalize_clarification_options(question, body.options or [])
    raw_values = body.option_values if isinstance(body.option_values, list) else []
    values = raw_values if len(raw_values) == len(options) else options
    if not shared.clarification_popup_lock.acquire(blocking=False):
        return {"status": "busy", "message": "A clarification popup is already open."}
    try:
        answer = approval.ask_choice(question, options)
    finally:
        shared.clarification_popup_lock.release()
    if not answer:
        return {"status": "cancelled"}
    try:
        value = values[options.index(answer)]
    except (ValueError, IndexError):
        value = answer
    return {"status": "selected", "answer": answer, "value": value}


@app.get("/api/local-models")
def get_local_models():
    """One list, not two. The UI pins a single model; vision capability and
    favorite status are shown as properties on each entry rather than
    separate tabs, because the engine only ever escalates text -> vision and
    never the reverse."""
    # Cached (and invalidated on every swap): force=True here made the engine
    # menu hang ~2 s on "Loading models..." whenever llama-server was down.
    reported = llama_local.server_models() or []
    try:
        configured = [cfg["name"] for cfg in llama_local.list_model_configs()]
    except Exception:
        configured = []
    ids = list(reported)
    seen = {llama_local.normalize_name(name) for name in ids}
    for name in configured:
        if llama_local.normalize_name(name) not in seen:
            ids.append(name)
            seen.add(llama_local.normalize_name(name))
    fav_text = set(shared.app_state["settings"].get("favorite_local_text_models") or [])
    fav_vision = set(shared.app_state["settings"].get("favorite_local_vision_models") or [])
    models = []
    for m in ids:
        is_vision = llama_local.is_vision_model(m)
        favs = fav_vision if is_vision else fav_text
        is_fav = m in favs or llama_local.normalize_name(m) in {llama_local.normalize_name(f) for f in favs}
        models.append({"id": m, "vision": is_vision, "favorite": is_fav})
    models.sort(key=lambda m: (m["vision"], not m["favorite"], m["id"].lower()))
    return {
        "models": models,
        "pinned": shared.app_state.get("local_model") or "",
        "engaged": shared.app_state.get("local_engaged"),
        "engaged_is_vision": bool(shared.app_state.get("local_engaged")
                                  and llama_local.is_vision_model(shared.app_state["local_engaged"])),
    }


class SettingsUpdate(BaseModel):
    chat_font: str = None
    user_name: str = None
    cloud_allow_screenshots: bool = None
    proactive_enabled: bool = None
    proactive_level: str = None
    proactive_notify: bool = None
    proactive_clipboard: bool = None
    proactive_learn_habits: bool = None
    proactive_quiet_hours: str = None
    proactive_detectors: dict = None
    favorite_local_text_models: list = None
    favorite_local_vision_models: list = None
    favorite_cloud_text_models: list = None
    favorite_cloud_vision_models: list = None
    theme: str = None
    appearance: str = None
    color_mode: str = None
    terminal_phosphor: bool | None = None
    look: str = None
    layout: str = None
    accent: str = None
    permission_mode: str = None
    permission_categories: dict = None
    permission_overrides: dict = None
    messaging_permissions: dict = None
    effort: str = None
    effort_by_model: dict = None
    effort_overrides: dict = None
    cross_check: bool | None = None
    effort_auto_escalate: bool = None
    prompt_enhancement: str = None
    prompt_prep: str = None
    auto_model_routing: bool = None
    max_revisions: object = None
    show_activity: bool = None
    record_model_io: bool = None
    ambient_idle: str = None
    context_auto_compress: str = None
    memory_enabled: bool = None
    web_search_provider: str = None
    searxng_url: str = None
    web_browser_fallback: bool = None
    browser_builtin: bool | None = None
    llama_build: str | None = None
    web_auto_search: bool | None = None
    response_mode: str | None = None
    system_message_variant: str | None = None
    system_message_custom: str | None = None
    assistant_name: str | None = None
    careful_mode: bool | None = None
    auto_tools: bool | None = None
    presets: list | None = None
    # Web Search privacy, Network, the Studio Overlay, secret themes (core/config.clean_privacy_settings)
    search_mode: str | None = None
    search_preview: bool | None = None
    search_cache_minutes: int | None = None
    search_history: bool | None = None
    search_history_days: int | None = None
    net_route: str | None = None
    net_proxy: str | None = None
    net_tor_port: int | None = None
    net_components: dict | None = None
    net_require_vpn: bool | None = None
    net_log_save: bool | None = None
    net_indicator: bool | None = None
    net_indicator_scope: str | None = None
    studio_layout: str | None = None
    overlay_panels: dict | None = None
    overlay_clickthrough: bool | None = None
    overlay_opacity: float | None = None
    overlay_window: bool | None = None
    research_depth: str | None = None
    sec_contact: str | None = None
    voice_model: str | None = None
    voice_voice: str | None = None
    voice_lang: str | None = None
    voice_auto_langs: list | None = None
    voice_multi_model: str | None = None
    voice_captions: bool | None = None
    voice_breath: float | None = None
    voice_volume: float | None = None
    voice_patience: str | None = None
    voice_sensitivity: str | None = None
    voice_rate: float | None = None
    voice_speak_replies: bool | None = None
    voice_conversation: bool | None = None
    voice_barge_in: bool | None = None
    voice_gpu: bool | None = None
    voice_duplex: bool | None = None
    voice_duplex_model: str | None = None
    voice_engine: str | None = None
    voice_natural: str | None = None
    voice_natural_compact: bool | None = None
    voice_listener: str | None = None
    voice_live: bool | None = None
    voice_sounds: bool | None = None
    personas: list | None = None
    persona: str | None = None
    unlocked_themes: list | None = None
    image_root: str | None = None
    image_quality: str | None = None
    image_size: str | None = None
    image_keep_minutes: int | None = None
    image_free_gpu: bool | None = None
    image_filter: bool | None = None
    image_speed_boost: bool | None = None
    image_vram_reserve: float | None = None
    image_warmup: bool | None = None
    video_root: str | None = None
    video_size: str | None = None
    video_length: str | None = None
    video_best_quality: bool | None = None
    osint_background: bool | None = None
    osint_layers: dict | None = None
    osint_areas: list | None = None
    osint_days: int | None = None
    osint_max_mb: int | None = None
    # Not settings: which model a per-model effort/Prompt Prep choice is for -
    # the one the slider shows (the UI's mode and pinned model), see _next_model.
    for_mode: str = None
    for_model: str = None


@app.get("/api/settings")
def get_settings():
    return {
        "settings": shared.app_state["settings"],
        "available_cloud_text_models": config.PRIMARY_MODELS,
        "available_cloud_vision_models": config.VISION_MODELS,
    }


@app.get("/api/system-messages")
def get_system_messages():
    return system_messages.catalog()


@app.post("/api/settings")
def update_settings(body: SettingsUpdate):
    updates = config._dump_model(body)
    for_mode, for_model = updates.pop("for_mode", None), updates.pop("for_model", None)
    current = shared.app_state["settings"]
    system_change = bool(system_messages.SETTING_KEYS.intersection(updates))
    confirmed_change = system_change or bool({"effort", "prompt_prep", "context_auto_compress", "cross_check"}.intersection(updates))
    if "cross_check" in updates and not isinstance(updates["cross_check"], bool):
        raise HTTPException(status_code=422, detail="Cross-check must be On or Off.")
    if "context_auto_compress" in updates and updates["context_auto_compress"] not in config.CONTEXT_COMPRESS_AT:
        raise HTTPException(status_code=422, detail="Choose Off, 80%, 90%, 95% or 97% for automatic compression.")
    if system_change:
        try:
            system_messages.validate_updates(current, updates)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    if confirmed_change:
        current = dict(current)  # failed saves cannot change the in-memory selection
    # permission_categories/permission_overrides arrive as PARTIAL dicts from
    # the Settings UI (e.g. just {"destructive": "auto"} or just one tool's
    # override) - merge them into the existing dict instead of replacing it
    # wholesale, or every other category/tool the user had already set would
    # silently revert to its default the next time either one changes.
    for nested_key in ("permission_categories", "permission_overrides", "proactive_detectors", "messaging_permissions",
                       "net_components", "overlay_panels"):
        if nested_key in updates and updates[nested_key] is not None:
            merged = dict(current.get(nested_key) or {})
            merged.update(updates[nested_key])
            # A tool set back to its category default (None/"default") is
            # removed from the override dict rather than stored as a
            # meaningless value, so the table can tell "explicitly set" apart
            # from "using the category default".
            if nested_key == "proactive_detectors":
                updates[nested_key] = {k: v for k, v in merged.items() if v is not None}
            else:
                updates[nested_key] = {k: v for k, v in merged.items() if v}
    # Effort persists per model: choosing a level also remembers it for the
    # model in use (the global value stays the default for other models).
    # The model the slider showed: it reads with _next_model(mode, pinned), so it must be
    # written the same way. llm._current_model() is the last model that RAN - before a local
    # model has answered a message it is a different one, the choice went under its name, and
    # the slider kept showing (and snapping back to) the displayed model's old entry.
    # Only effort and Prompt Prep need the model - finding it can ask llama-server, which made every other save (a
    # colour, light/dark) wait for it.
    needs_model = updates.get("effort") in effort.CHOICES or updates.get("prompt_prep") in prompt_prep.MODES
    model = (_next_model(for_mode or None, for_model or None)[0] if for_mode else llm._current_model()) if needs_model else None
    if updates.get("effort") in effort.CHOICES and model and "effort_by_model" not in updates:
        updates["effort_by_model"] = {**(current.get("effort_by_model") or {}), model: updates["effort"]}
    if updates.get("prompt_prep") in prompt_prep.MODES and model:     # Prompt Prep: same rule
        updates["prompt_prep_by_model"] = {**(current.get("prompt_prep_by_model") or {}), model: updates["prompt_prep"]}
    current.update(updates)
    current.update(permissions.clean_settings(current))
    current.update(awareness.clean_settings(current))
    if isinstance(current.get("user_name"), str):
        current["user_name"] = current["user_name"].strip()[:40]
    current["look"] = config.clean_look(current.get("look"))
    # A client that sends only "theme" (the older UI) picks the family through it.
    from_theme = "theme" in updates and not {"appearance", "color_mode", "terminal_phosphor"} & set(updates)
    current.update(config.clean_appearance(current, from_theme=from_theme))
    current.update(config.clean_agent_settings(current))
    saved = config.save_settings(current)
    if confirmed_change and saved is False:
        detail = ("Could not save the system message. Your previous selection is still active." if system_change else
                  "Could not save these settings. Your previous choices are still active.")
        raise HTTPException(status_code=500, detail=detail)
    shared.app_state["settings"] = current
    if any(k.startswith("net_") for k in updates):
        netlayer.apply_env()                      # litellm's proxy variables follow the new route
        netlayer.clear_blocked()
        if {"net_route", "net_proxy", "net_tor_port", "net_components"} & set(updates):
            browser.close_all()                   # the assistant's browser restarts with the new route's flags
    if "overlay_window" in updates or "overlay_clickthrough" in updates or "overlay_opacity" in updates:
        app_window.overlay_sync(current)
    if updates:
        event_log.log_action(f"Settings updated: {', '.join(updates.keys())}.")
    if {"effort", "effort_by_model", "effort_overrides"} & set(updates):
        tasks._queue_effort_change()
    return {"status": "success", "settings": current}


class EffortSessionBody(BaseModel):
    level: str | None = None   # a level id, or null to clear the session override


@app.post("/api/effort/session")
def set_session_effort(body: EffortSessionBody):
    """A session-only effort level (not saved): overrides project, model and
    the setting until cleared or the app restarts."""
    level = body.level if body.level in effort.CHOICES else None
    shared.app_state["effort_session"] = level
    tasks._queue_effort_change()
    return {"status": "success", "session_effort": level}


class PrepSessionBody(BaseModel):
    mode: str | None = None    # a Prompt Prep mode, or null to clear the session override


@app.post("/api/prep/session")
def set_session_prep(body: PrepSessionBody):
    """A session-only Prompt Prep mode (not saved); applies from the next run."""
    shared.app_state["prep_session"] = body.mode if body.mode in prompt_prep.MODES else None
    return {"status": "success", "session_prep": shared.app_state["prep_session"]}


def _next_model(mode: str | None, model: str | None) -> tuple[str | None, bool]:
    """(model, local) the NEXT message will use: the UI's mode and pinned model
    when it says (they reach the server only with a message), else the last."""
    local = (mode or shared.app_state.get("mode")) == config.MODE_LOCAL
    if mode in (config.MODE_CLOUD, config.MODE_AUTO):          # Auto tries cloud models first
        return (shared.app_state.get("last_working_model") or config.PRIMARY_MODELS[0]), False
    if local and model:
        return model, True
    if local and mode == config.MODE_LOCAL and shared.app_state.get("mode") != config.MODE_LOCAL:
        return (shared.app_state.get("local_engaged") or shared.app_state.get("local_model") or None), True
    return llm._current_model(), local


@app.get("/api/prep")
def get_prep(mode: str | None = None, model: str | None = None):
    """Prompt Prep modes and which one the next run will use, and why."""
    next_model, _local = _next_model(mode, model)
    mode, source = prompt_prep.choose_mode(shared.app_state["settings"], next_model,
                                           shared.app_state.get("active_project"), shared.app_state.get("prep_session"))
    return {"modes": [{"id": m, "label": prompt_prep.LABELS[m]} for m in prompt_prep.MODES],
            "mode": mode, "source": source, "model": next_model, "session": shared.app_state.get("prep_session"),
            "pending": prompt_prep.gate.pending()}


class PrepAnswer(BaseModel):
    action: str                # send | raw | cancel
    text: str = None


@app.post("/api/prep/{card_id}")
def answer_prep(card_id: str, body: PrepAnswer):
    """The Expand / Plan first card: send (possibly edited), run as typed, or cancel."""
    ok = prompt_prep.gate.answer(card_id, body.action, body.text)
    return {"ok": ok, "message": None if ok else "That card is no longer waiting."}


# ---- Context window (agent/context_window.py) ------------------------------
def _context_status(mode: str | None = None, model: str | None = None) -> dict | None:
    try:
        next_model, local = _next_model(mode, model)
        if local and not model and not shared.app_state.get("local_engaged"):
            return context_window.status(None, True, known=False)   # nothing loaded or pinned: the model isn't known yet
        return context_window.status(next_model, local)
    except Exception as e:                 # the meter never breaks the page
        event_log.log_action(f"Context meter unavailable: {str(e)[:120]}")
        return None


@app.get("/api/context")
def get_context(mode: str | None = None, model: str | None = None):
    """How full the next request's context is, and the compression setting."""
    return {"status": _context_status(mode, model), "thresholds": list(context_window.THRESHOLDS)}


class ContextCompressBody(BaseModel):
    mode: str = None
    model: str = None


@app.post("/api/context/compress")
def compress_context(body: ContextCompressBody = None):
    """The "Compress now" button: summarize this chat's older turns. Not while a task runs."""
    if not shared.state_lock.acquire(blocking=False):
        return {"ok": False, "message": "A task is running - compress when it has finished."}
    try:
        next_model, local = _next_model(body.mode if body else None, body.model if body else None)
        result = context_window.compress(auto=False, model=next_model, local=local)
        chat = shared.app_state.get("active_chat")
        if result.get("ok") and chat:          # the saved chat reopens compressed too
            shared.chat_store.save({**chat, "conversation": shared.app_state["conversation"],
                                    "context_summary": shared.app_state.get("context_summary") or ""})
        return result
    except Exception as e:
        return {"ok": False, "message": f"Compression failed: {str(e)[:250]}"}
    finally:
        shared.state_lock.release()


@app.get("/api/effort")
def get_effort(mode: str | None = None, model: str | None = None):
    """Levels, their limits, what the model the next message will use supports
    (the UI passes its mode and pinned model) and what applies now."""
    model, local = _next_model(mode, model)
    caps = model_caps.get(model) if model and local else {}
    level, source = effort.choose_level(shared.app_state["settings"], model, shared.app_state.get("active_project"),
                                        shared.app_state.get("effort_session"))
    # A preview has no request yet; the turn event reports Auto's actual choice and reason.
    applied, fallback = effort.resolve("normal" if level == "auto" else level, caps)
    supported = effort.supported_levels(caps)
    return {"levels": [{"id": "auto", "label": "Auto", "limits": None}] + [{"id": l, "label": effort.LABELS[l],
                        "limits": effort.profile_for(l, model, shared.app_state["settings"].get("effort_overrides")).limits()}
                       for l in effort.LEVELS],
            "supported": ["auto"] + supported, "requested": level, "source": source,
            # why each level this model can't run is unavailable (shown on the slider)
            "unsupported": {l: effort.resolve(l, caps)[1] for l in effort.LEVELS if l not in supported},
            "applied": applied, "fallback": fallback, "model": model, "local": local, "capabilities": caps,
            "cross_check": {"enabled": shared.app_state["settings"].get("cross_check", False),
                            **llm.cross_check_status(mode or shared.app_state.get("mode"), model)},
            "session": shared.app_state.get("effort_session"),
            "project": (shared.app_state.get("active_project") or {}).get("name"),
            "run": _run_usage()}


@app.get("/api/permissions/tools")
def get_permission_tools():
    """Every known tool with its risk tier and description, for the
    Settings > Permissions > Advanced per-tool table. Sourced from the
    live TOOLS schema, so a tool added here always shows up there too."""
    mcp = schemas.mcp_schemas()          # tools of connected MCP servers can get their own rule too
    import approval_view
    descriptions = {t["function"]["name"]: t["function"].get("description", "") for t in schemas.TOOLS + mcp}
    # "label": the words the approval card uses ("Write a file") - the page leads with them, not the tool's id (D49).
    tools = [{"name": t["name"], "tier": t["tier"], "description": descriptions.get(t["name"], ""),
              "label": approval_view.friendly(t["name"])}
             for t in permissions.all_known_tools(schemas.TOOL_NAMES | {t["function"]["name"] for t in mcp})]
    return {"tools": tools, "modes": sorted(permissions.VALID_MODES),
            "decisions": sorted(permissions.VALID_DECISIONS)}


class FolderRuleBody(BaseModel):
    add: str = None
    remove: str = None


@app.post("/api/permissions/folders")
def update_folder_rules(body: FolderRuleBody):
    """Add or remove one per-folder rule (permissions.py "Per-folder rules").
    Adding says why a folder is refused instead of dropping it silently."""
    current = shared.app_state["settings"]
    rules = list(current.get("folder_rules") or [])
    if body.add is not None:
        problem = permissions.folder_rule_problem(body.add)
        if problem:
            return {"ok": False, "message": problem, "folder_rules": rules}
        if len(rules) >= permissions.MAX_FOLDER_RULES:
            return {"ok": False, "message": f"At most {permissions.MAX_FOLDER_RULES} folders.", "folder_rules": rules}
        rules.append({"path": body.add})
    elif body.remove is not None:
        target = os.path.normcase(body.remove)
        rules = [r for r in rules if os.path.normcase(r.get("path") or "") != target]
    current["folder_rules"] = permissions.clean_folder_rules(rules)
    shared.app_state["settings"] = current
    config.save_settings(current)
    event_log.log_action(f"Folder rules: {', '.join(r['path'] for r in current['folder_rules']) or 'none'}.")
    return {"ok": True, "folder_rules": current["folder_rules"], "settings": current}


# ---- The Browser panel (browser.Viewer, D50) ------------------------------------------------------------------
class BrowserInput(BaseModel):
    type: str
    x: float = 0
    y: float = 0
    dx: float = 0
    dy: float = 0
    clicks: int = 1
    button: str = "left"
    text: str = ""
    key: str = ""
    mods: dict = {}


class BrowserNav(BaseModel):
    action: str                 # open | go | back | forward | reload | close
    url: str = ""


@app.get("/api/browser/view/frame")
def browser_frame(h: str = ""):
    """The page as it looks now (JPEG), or 204 when it hasn't changed since frame `h` or nothing is open; the
    state rides in headers either way (URL and title percent-encoded)."""
    from urllib.parse import quote
    data, digest, st = browser.viewer.frame()
    head = {"Cache-Control": "no-store", "X-Running": "1" if st.get("running") else "0",
            "X-Builtin": "1" if st.get("builtin") else "0", "X-Frame-Hash": digest,
            "X-Url": quote(st.get("url") or "", safe=":/?&=#%"), "X-Title": quote(st.get("title") or ""),
            "X-Loading": "1" if st.get("loading") else "0", "X-Back": "1" if st.get("back") else "0",
            "X-Forward": "1" if st.get("forward") else "0", "X-W": str(st.get("w") or 0), "X-H": str(st.get("h") or 0)}
    if not data or digest == h:
        return Response(status_code=204, headers=head)
    return Response(content=data, media_type="image/jpeg", headers=head)


@app.post("/api/browser/view/input")
def browser_input(body: BrowserInput):
    return browser.viewer.input(body.model_dump())


@app.post("/api/browser/view/nav")
def browser_nav(body: BrowserNav):
    return browser.viewer.navigate(body.action, body.url)


@app.get("/api/models/assets")
def get_model_assets():
    """Vision files (mmproj) a model in models.ini could still get, and the download in progress (model_assets.py)."""
    import model_assets
    return {"offers": model_assets.offers(), "state": model_assets.status()}


@app.post("/api/models/assets/{model}")
def start_model_asset(model: str):
    import model_assets
    return model_assets.start(model)


@app.get("/api/local-models/config")
def get_local_model_configs():
    """Per-model VRAM/RAM knobs (ngl, ctx-size, batch-size, threads) as
    currently written in models.ini, for the Settings > Resources tab."""
    try:
        configs = llama_local.list_model_configs()
    except Exception as e:
        return {"status": "error", "message": f"Could not read models.ini: {e}", "models": [], "defaults": {}}
    for c in configs:      # checked on every call: a file can be added or removed without editing models.ini
        c["file_found"] = bool(c.get("model_path")) and os.path.isfile(c["model_path"])
    return {"status": "success", "models": configs, "defaults": llama_local.get_default_resources()}


@app.get("/api/local-models/ctx-info")
async def local_model_ctx_info():
    """Each model's context limits for Settings > Models' "Max context" buttons: the most it was trained for and
    the most that fits in VRAM. Separate from the list because reading every model's GGUF header the first time
    (multi-GB files, some on a hard disk) took ~16 s - the list shows at once and these fill in after."""
    def work():
        try:
            gpu = diagnostics.gpu_info()
        except Exception:
            gpu = {}
        out = {}
        for c in llama_local.list_model_configs():
            if c.get("model_path") and os.path.isfile(c["model_path"]):
                try:
                    out[c["name"]] = diagnostics.ctx_limits(c["name"], c, gpu)
                except Exception:
                    out[c["name"]] = {}
        return out
    return {"status": "success", "models": await asyncio.to_thread(work)}


class ModelAdd(BaseModel):
    path: str = ""
    restart: bool = True


def _models_added(added: list, restart: bool) -> dict:
    """After models.ini gained models: restart the router (when running) so they can be picked at once."""
    restarted, warning = False, ""
    if added and restart:
        try:
            restarted = model_scan.restart_router()
        except Exception as e:
            warning = f"Added, but llama-server didn't restart ({str(e)[:160]}) - start the app again to use them."
    if added:
        event_log.log_action("Models added: " + ", ".join(a["name"] for a in added))
    return {"restarted": restarted, "warning": warning}


@app.get("/api/local-models/folders")
def local_model_folders():
    """Where Scan looks for .gguf files, and where models.ini is."""
    return {"folders": paths.model_dirs(shared.app_state["settings"]), "ini": model_scan.ini_path(), "home": paths.HOME}


@app.post("/api/local-models/open-folder")
def local_models_open_folder():
    """Opens the models folder in Explorer (only that folder - this never opens a path the page sends)."""
    folder = paths.model_dirs(shared.app_state["settings"])[0]
    if not config.IS_WINDOWS:                              # os.startfile exists only on Windows
        return JSONResponse({"ok": False, "folder": folder,
                             "message": f"Opening the folder is only available on Windows. It is: {folder}"},
                            status_code=501)
    os.makedirs(folder, exist_ok=True)
    os.startfile(folder)                                   # noqa: S606 - a fixed, app-owned folder
    return {"ok": True, "folder": folder}


@app.post("/api/local-models/scan")
async def local_models_scan(body: ModelAdd):
    """Settings > Models > Scan for models: every new .gguf in the model folders gets a models.ini entry."""
    try:
        got = await asyncio.to_thread(model_scan.scan, shared.app_state["settings"])
    except Exception as e:
        return {"status": "error", "message": str(e)[:300]}
    return {"status": "success", **got, **await asyncio.to_thread(_models_added, got["added"], body.restart)}


@app.post("/api/local-models/add")
async def local_models_add(body: ModelAdd):
    """Settings > Models > Add a model file: one .gguf by its path."""
    try:
        got = await asyncio.to_thread(model_scan.add_file, body.path)
    except (ValueError, OSError) as e:
        return {"status": "error", "message": str(e)[:300]}
    return {"status": "success", "added": [got], **await asyncio.to_thread(_models_added, [got], body.restart)}


class ModelConfigUpdate(BaseModel):
    """null = clear the override (the model uses the [*] default again). The UI
    sends null for every blank box; with a plain `int = None` pydantic 2 refused
    that null (422), so every save - a context-size change included - failed."""
    name: str
    ngl: int | None = None
    ctx_size: int | None = None
    batch_size: int | None = None
    threads: int | None = None


_CONFIG_MIN = {"ngl": 0, "ctx_size": 512, "batch_size": 1, "threads": 1}


@app.post("/api/local-models/config")
def set_local_model_config(body: ModelConfigUpdate):
    fields = {k: v for k, v in config._dump_model(body).items() if k != "name"}
    low = [f"{k.replace('_', ' ')} must be at least {_CONFIG_MIN[k]} (got {v})"
           for k, v in fields.items() if v is not None and v < _CONFIG_MIN[k]]
    if low:
        return {"status": "error", "message": "; ".join(low) + "."}
    try:
        llama_local.update_model_config(body.name, **fields)
    except KeyError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Could not update models.ini: {e}"}
    event_log.log_action(f"Resource settings for '{body.name}' updated in models.ini "
               f"({', '.join(f'{k}={v}' for k, v in fields.items())}). Force-load it (or switch to a "
               "different model and back) for a currently-loaded model to pick up the change.")
    return {"status": "success", "models": llama_local.list_model_configs()}


@app.get("/api/llama/builds")
def llama_build_status():
    """Settings › Models: the newest llama.cpp and PrismML's fork - installed, version, which one runs (D51)."""
    return llama_builds.status()


@app.post("/api/llama/builds/apply")
def llama_build_apply():
    """Move llama-server to the build the setting asks for now, not on the next message."""
    if not shared.state_lock.acquire(blocking=False):
        return {"status": "error", "message": "Busy - it switches with your next message instead."}
    try:
        kind = llama_builds.apply(shared.app_state.get("local_engaged") or "")
        return {"status": "success", "switched": kind, **llama_builds.status()}
    except Exception as e:
        return {"status": "error", "message": str(e)[:400], **llama_builds.status()}
    finally:
        shared.state_lock.release()


class ForceLoadRequest(BaseModel):
    model: str
    evict_first: bool = True
    pin: bool = True


@app.post("/api/local-models/force-load")
def force_load_model(body: ForceLoadRequest):
    """Loads (or reloads) a specific model into the router right now, instead
    of paying that latency on the user's next real message. If the target is
    already the resident model, --models-max 1 means the router will NOT
    reload it on its own - so evict_first briefly swaps to any other
    configured model first, guaranteeing freshly-edited ngl/ctx-size/batch-
    size settings are actually re-read from models.ini.
    It holds its own lock, not state_lock: a load takes minutes, and switching or reading chats must keep working
    meanwhile. New messages wait for it (start_chat checks model_loading)."""
    if shared.state_lock.locked() or not _model_load_lock.acquire(blocking=False):
        return {"status": "error", "message": _busy("Busy - wait for the current turn or load to finish first.")}
    shared.app_state["model_loading"] = body.model
    try:
        if shared.state_lock.locked():          # a turn slipped in between the check and the flag
            return {"status": "error", "message": "Busy - wait for the current turn to finish first."}
        if not llama_local.server_is_up():
            return {"status": "error", "message": llama_local.unreachable_message()}
        reported = llama_local.server_models(force=True) or []
        try:
            configured = [cfg["name"] for cfg in llama_local.list_model_configs()]
        except Exception:
            configured = []
        available = reported + [name for name in configured if name not in reported]
        wanted_key = llama_local.normalize_name(body.model)
        resolved = next((name for name in available if llama_local.normalize_name(name) == wanted_key), None)
        if not resolved:
            return {"status": "error",
                    "message": f"'{body.model}' is not available. Check models.ini and that the name "
                               "matches a model preset exactly."}
        t_total = 0.0
        system_metrics.begin_model_load()
        if body.evict_first and llama_local.normalize_name(shared.app_state.get("local_engaged") or "") == wanted_key and len(available) > 1:
            # Swap through the smallest other model: evicting via a 20 GB
            # vision model costs a long load and can itself run out of VRAM.
            other = llama_local.cheapest_other_model(resolved, available)
            if other:
                event_log.log_action(f"Evicting '{resolved}' via '{other}' so its updated settings are re-read...")
                t_total += llama_local.force_load(other, timeout=180, back=False)
        t_total += llama_local.force_load(resolved, timeout=180)
        system_metrics.finish_model_load(resolved)
        if body.pin:
            shared.app_state["local_model"] = resolved
        llm.engage_local_model(resolved)
        return {"status": "success", "message": f"'{resolved}' is loaded and {'pinned' if body.pin else 'ready'} ({t_total:.1f}s).", "seconds": t_total,
                "model": resolved, "pinned": bool(body.pin)}
    except Exception as e:
        return {"status": "error", "message": f"Force-load failed: {str(e)[:600]}"}
    finally:
        shared.app_state["model_loading"] = None
        _model_load_lock.release()


_model_load_lock = threading.Lock()


def _model_loading() -> bool:
    return bool(shared.app_state.get("model_loading"))


def _connect_mcp_servers():
    shared.mcp_hub.load()
    shared.mcp_hub.connect_enabled()
    connected = [n for n, s in shared.mcp_hub.servers.items() if s.status == "connected"]
    if shared.mcp_hub.servers:
        event_log.log_action(f"MCP: {len(connected)} of {len(shared.mcp_hub.servers)} server(s) connected.")


@app.on_event("startup")
def _start_background_services():
    # A fresh install: models.ini from models.example.ini, and whatever is already in local\models added to it.
    try:
        model_scan.ensure_ini()
    except Exception as e:
        print(f"[WARNING] models.ini: {e}", flush=True)
    # Every outbound connection goes through the network layer; the guard catches what goes around it.
    if os.environ.get("AGENTIC_DISABLE_NETGUARD") != "1":
        netlayer.install()
    netlayer.apply_env()
    # A crash leaves temporary chats' folders behind: nothing of them survives a restart (privacy.py).
    swept = privacy.sweep()
    if swept:
        event_log.log_action(f"Removed {swept} temporary chat folder{'s' if swept != 1 else ''} left by the last run.")
    # MCP servers connect in the background (a slow one must not delay the UI).
    threading.Thread(target=_connect_mcp_servers, daemon=True, name="mcp-connect").start()
    # A hidden browser the last run left behind (closed without its shutdown) holds the browser's port and memory.
    threading.Thread(target=lambda: browser.agent_browser.close_leftover(), daemon=True, name="browser-leftover").start()
    # Reopen the chat and project that were open when the server stopped.
    try:
        sessions.restore()
    except Exception as e:
        event_log.log_action(f"Could not reopen the last chat: {e}")
    if os.environ.get("AGENTIC_DISABLE_AWARENESS") != "1":
        observer.engine.start()
    # The system check (GPU fit of every model, llama-server, disks) - Settings > Diagnostics shows it.
    if os.environ.get("AGENTIC_DISABLE_DIAGNOSTICS") != "1":
        diagnostics.run_in_background()
    # The Studio Overlay's always-on-top window, when it was on (app_window.py).
    if shared.app_state["settings"].get("overlay_window") and os.environ.get("AGENTIC_DISABLE_APP_WINDOW") != "1":
        app_window.overlay_sync(shared.app_state["settings"])
    # MossadGPT's world monitor collects in the background only when that setting is on (else while it's on screen).
    if shared.app_state["settings"].get("osint_background"):
        osint_collector.beat()


@app.on_event("shutdown")
def _stop_background_services():
    osint_collector.stop()
    image_gen.stop()                     # the image engine (sd-server) never outlives the app
    sessions.close_temporary()
    privacy.sweep()
    app_window.overlay_close()
    voice.stop_all()
    observer.engine.stop()
    browser.close_all()                  # only the windows this app started (browser.py)
    llama_builds.stop_started()          # a windowless llama-server the app restarted stops with it (D51)
    shared.mcp_hub.close()


class SuggestionAction(BaseModel):
    action: int = 0


class SuggestionDismiss(BaseModel):
    forever: bool = False


class SuggestionSnooze(BaseModel):
    minutes: float = 60


@app.get("/api/suggestions")
def get_suggestions():
    return {"suggestions": observer.engine.active(), "context": observer.engine.context()}


@app.post("/api/suggestions/{suggestion_id}/dismiss")
def dismiss_suggestion(suggestion_id: str, body: SuggestionDismiss = None):
    ok = observer.engine.dismiss(suggestion_id, forever=bool(body and body.forever))
    return {"status": "success" if ok else "error"}


@app.post("/api/suggestions/{suggestion_id}/snooze")
def snooze_suggestion(suggestion_id: str, body: SuggestionSnooze = None):
    ok = observer.engine.snooze(suggestion_id, minutes=(body.minutes if body else 60))
    return {"status": "success" if ok else "error"}


@app.post("/api/suggestions/{suggestion_id}/accept")
def accept_suggestion(suggestion_id: str, body: SuggestionAction = None):
    """The user's click is the authorization. Prompt actions are returned to
    the UI to send as a normal chat message; tool actions run here through
    the same permission gate and guarded executor as everything else."""
    suggestion = observer.engine.get(suggestion_id)
    if not suggestion:
        return {"status": "error", "message": "That suggestion is no longer available."}
    index = body.action if body else 0
    wanted = suggestion.actions[index] if 0 <= index < len(suggestion.actions) else {}
    if wanted.get("kind") == "tool" and shared.state_lock.locked():     # busy: keep the suggestion for later
        return {"status": "error", "message": _busy("Busy - try again when the current task finishes.")}
    action = observer.engine.accept(suggestion_id, index)
    if not action:
        return {"status": "error", "message": "That suggestion was already handled."}
    kind = action.get("kind")
    if kind != "tool":
        return {"status": "success", **action}
    tool_name, args = action.get("tool"), dict(action.get("args") or {})
    if tool_name not in schemas.TOOL_NAMES:
        return {"status": "error", "message": f"Unknown tool '{tool_name}'."}
    if not shared.state_lock.acquire(blocking=False):
        return {"status": "error", "message": "Busy - try again when the current task finishes."}
    try:
        shared.app_state["current_user_request"] = f"{action.get('label')} ({suggestion.title})"
        shared.app_state["turn_policy"] = {"mode": "action", "allow_mutation": True,
                                    "reason": "user accepted a proactive suggestion"}
        shared.app_state["intent_guard"] = None
        decision, approved = approvals.gate_batch([{"tool_name": tool_name, "args": args,
                                          "justification": args.get("justification", "")}])
        if not approved:
            message = ("That action is set to Deny in your permission settings." if decision == "deny"
                       else "Cancelled - nothing was changed.")
            return {"status": "error", "kind": "tool", "message": message}
        res = runtime.execute_tool_guarded(tool_name, args)
        message = res.get("message") or ("Done." if res.get("ok") else "It didn't work.")
        shared.app_state["conversation"].append({"role": "user", "content": f"[Accepted suggestion] {suggestion.title}: {action.get('label')}"})
        shared.app_state["conversation"].append({"role": "assistant", "content": message})
        event_log.log_action(f"Suggestion '{suggestion.title}' -> {tool_name}: {'ok' if res.get('ok') else 'failed'}")
        return {"status": "success" if res.get("ok") else "error", "kind": "tool",
                "message": message, "title": suggestion.title}
    finally:
        shared.app_state["turn_policy"] = {"mode": "none", "allow_mutation": False, "reason": "no active request"}
        shared.state_lock.release()


@app.get("/api/awareness/detectors")
def get_detectors():
    """Settings > Proactive help: every detector (with the signals it reads and
    whether feedback made it less frequent), every signal the observer can
    read, and the suggestions muted with "Don't suggest this again"."""
    stats = observer.engine.store.get("stats", {})
    return {"detectors": [{**d, "state": observer.engine.detector_state(d["id"]),
                           "muted": observer.engine.detector_state(d["id"]) == "muted", "stats": stats.get(d["id"], {})}
                          for d in awareness.catalogue()],
            "signals": awareness.signals_catalogue(), "muted": observer.engine.muted()}


class UnmuteBody(BaseModel):
    key: str


@app.post("/api/awareness/unmute")
def unmute_suggestion(body: UnmuteBody):
    ok = observer.engine.unmute(body.key)
    return {"ok": ok, "message": None if ok else "That suggestion wasn't muted."}


@app.post("/api/awareness/detectors/{detector_id}/reenable")
def reenable_detector(detector_id: str):
    observer.engine.reenable(detector_id)
    return {"status": "success"}


@app.post("/api/awareness/clear")
def clear_awareness():
    observer.engine.clear_data()
    event_log.log_action("Cleared locally stored activity data.")
    return {"status": "success", "message": "Activity history and learned habits were deleted."}


class UndoRequest(BaseModel):
    force: bool = False


@app.get("/api/stream")
def stream_events(request: Request, since: int = 0):
    """Live model output as server-sent events (stream_hub.py). The UI keeps
    this open while a turn runs; a comment line every 15 s keeps it alive.
    Every event carries its sequence number as the SSE id, so an automatic
    reconnect (Last-Event-ID) resumes exactly after the last event received."""
    from fastapi.responses import StreamingResponse
    try:
        since = max(since, int(request.headers.get("last-event-id") or 0))
    except ValueError:
        pass

    def gen():
        seq = since if since > 0 else stream_hub.hub.seq
        yield f"event: hello\ndata: {json.dumps({'seq': seq})}\n\n"
        idle = 0.0
        while True:
            events, seq = stream_hub.hub.since(seq, timeout=15.0)
            if not events:
                idle += 15
                yield ": keep-alive\n\n"
                if idle >= 600:                 # the page is gone or idle; it reconnects when needed
                    return
                continue
            idle = 0.0
            for e in events:
                yield f"id: {e['seq']}\ndata: {json.dumps(e, default=str)}\n\n"
            time.sleep(0.05)                    # coalesce: fragments arriving meanwhile go out merged

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/undo")
def list_undo(limit: int = 50):
    """The undo trail: recent file changes, newest first, and where backups live."""
    return {"changes": files.undo.list(max(1, min(int(limit), 300))), "backup_dir": files.UNDO_DIR,
            "retention": {"max_changes": undo_store.MAX_CHANGES, "max_days": undo_store.MAX_AGE_S // 86400,
                          "max_gb": undo_store.MAX_BYTES // 1024 ** 3}}


def _report_undo(res: dict) -> dict:
    event_log.log_action(res.get("message", ""))
    event_log.emit_event("undo", res.get("message", ""), ok=bool(res.get("ok")), change_id=res.get("id"),
               conflict=bool(res.get("conflict")))
    return res


@app.post("/api/undo/last")
def undo_last_change(body: UndoRequest = None):
    return _report_undo(files.undo.undo_last(force=bool(body and body.force)))


@app.post("/api/undo/{change_id}")
def undo_change(change_id: str, body: UndoRequest = None):
    if not re.fullmatch(r"[\w\-]{6,64}", change_id or ""):
        return {"ok": False, "message": "Unknown change."}
    return _report_undo(files.undo.undo(change_id, force=bool(body and body.force)))


# ---- Saved memory (memory.py; Settings > Memory) ---------------------------
class MemoryBody(BaseModel):
    text: str = None
    pinned: bool = None


class SearchKeyBody(BaseModel):
    provider: str
    key: str = ""               # empty: remove the saved key


@app.get("/api/search/keys")
def search_keys():
    """Which optional search keys are set (never the keys themselves)."""
    return {"keys": search_engine.KEYS.status(), "providers": list(config.WEB_SEARCH_PROVIDERS),
            "chain": [search_engine.NAMES[p] for p in search_engine.chain(shared.app_state["settings"],
                                                                           search_engine.KEYS)]}


@app.post("/api/search/keys")
def set_search_key(body: SearchKeyBody):
    try:
        search_engine.KEYS.set(body.provider, body.key)
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    except OSError as e:
        return {"ok": False, "message": f"Could not save the key: {e.strerror or e}"}
    search_engine.reset()                     # new chain: cached searches of the old one don't apply
    event_log.log_action(f"Search key for {search_engine.NAMES.get(body.provider, body.provider)} "
                         f"{'saved' if body.key.strip() else 'removed'}.")          # never the key itself
    return {"ok": True, **search_keys()}


# ---- Web Search privacy (search_engine.MODES) ----------------------------------------
@app.get("/api/search/modes")
def search_modes():
    """What each mode protects and what still leaves the machine (shown as is in Settings)."""
    s = shared.app_state["settings"]
    return {"modes": search_engine.MODE_INFO, "order": list(search_engine.MODES), "mode": s.get("search_mode"),
            "history": search_history.stats(), "route": netlayer.route_for("search").public(),
            "chains": {m: [search_engine.NAMES[p] for p in search_engine.chain(s, search_engine.KEYS, m)]
                       for m in search_engine.MODES}}


class SearchTestBody(BaseModel):
    mode: str | None = None


@app.post("/api/search/test")
def search_test(body: SearchTestBody):
    """"Test connection": one real query through the mode's backend and route, and which route it actually used."""
    s = shared.app_state["settings"]
    mode = body.mode if body.mode in search_engine.MODES else s.get("search_mode", "standard")
    route = netlayer.route_for("search")
    if mode in ("local", "disabled"):
        return {"ok": True, "mode": mode, "backend": "This PC" if mode == "local" else "none", "route": "no network",
                "detail": search_engine.MODE_INFO[mode]["protects"]}
    if mode == "max" and search_engine.max_privacy_route_problem():
        return {"ok": False, "mode": mode, "route": route.label, "message": search_engine.max_privacy_route_problem()}
    started = time.time()
    out = {"mode": mode, "route": route.label}
    try:
        with search_engine.scope(mode):
            if mode == "max" or (s.get("web_search_provider") == "searxng" and s.get("searxng_url")):
                if not s.get("searxng_url"):
                    return {**out, "ok": False, "message": "No SearXNG address is set."}
                results = search_engine.searxng_request(s["searxng_url"], "test")
                out["backend"] = f"SearXNG ({urlparse(s['searxng_url']).netloc})"
            else:
                # Never the cache: a test that answers from it proves nothing about the route.
                got = search_engine.search("test", settings={**s, "search_cache_minutes": 0}, mode=mode)
                results = got["results"]
                out["backend"] = ", ".join(got["providers"]) or "none answered"
                if not results and got["why"]:
                    raise RuntimeError("; ".join(got["why"][:2]))
        out.update(ok=True, results=len(results), ms=round((time.time() - started) * 1000))
    except Exception as e:
        out.update(ok=False, message=str(e)[:300])
    if route.kind == "tor" and out.get("ok"):
        ip = netlayer.public_ip(force=True)       # through the same route: does Tor see it as Tor?
        out["tor_confirmed"] = ip.get("is_tor")
        out["exit"] = {k: ip.get(k) for k in ("ip", "country")}
    return out


@app.post("/api/search/clear")
def search_clear():
    """"Clear search history & cache": the saved history, and the in-memory result and page caches."""
    removed = search_history.clear()
    search_engine.reset()
    web_tools.reset_search_state()
    event_log.log_action("Search history and cache cleared.")
    return {"ok": True, **removed}


# ---- Voice (voice.py): whisper.cpp and Piper on this PC ----------------------------------
@app.get("/api/voice")
def voice_status():
    return voice.status(shared.app_state["settings"])


@app.post("/api/voice/setup")
def voice_setup():
    """Downloads what's missing for the chosen model and voice (only on the user's click)."""
    got = voice.setup(shared.app_state["settings"])
    if got.get("state") == "running":
        event_log.log_action("Voice setup: downloading " + ", ".join(i[1] for i in voice.plan(shared.app_state["settings"])))
    return {"setup": got, "plan": [{"label": i[1], "mb": i[4]} for i in voice.plan(shared.app_state["settings"])]}


@app.post("/api/voice/transcribe")
async def voice_transcribe(request: Request):
    """One utterance, 16 kHz mono WAV in the body -> its text. The audio is never stored or logged."""
    body = await request.body()
    if len(body) > 12 * 1024 * 1024:
        return {"ok": False, "message": "That recording is too long (over about 6 minutes)."}
    try:
        got = await asyncio.to_thread(voice.transcribe, body, shared.app_state["settings"])
    except Exception as e:
        return {"ok": False, "message": str(e)[:300]}
    return {"ok": True, **got}


@app.post("/api/voice/live")
async def voice_live(request: Request, sid: str = "", end: int = 0):
    """Live captions: the next ~0.25 s of the utterance (16 kHz 16-bit PCM) -> what it has heard so far."""
    if end:
        voice.live_end(sid)
        return {"ok": True}
    body = await request.body()
    if len(body) > 2 * 16000 * 10:
        return {"ok": False, "message": "Too much audio at once."}
    try:
        got = await asyncio.to_thread(voice.live_feed, sid, body, shared.app_state["settings"])
    except Exception as e:
        return {"ok": False, "message": str(e)[:200]}
    return {"ok": True, **got}


class SpeakBody(BaseModel):
    text: str
    voice: str | None = None          # Settings > Voice's "Hear it": another natural voice than the chosen one
    lang: str | None = None           # the language this reply is in (with "auto", the one the user spoke)


@app.post("/api/voice/speak")
async def voice_speak(body: SpeakBody):
    from fastapi.responses import Response
    try:
        wav = await asyncio.to_thread(voice.speak, body.text, shared.app_state["settings"], body.voice, body.lang)
    except Exception as e:
        return Response(content=str(e)[:300], status_code=503, media_type="text/plain")
    return Response(content=wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})


# ---- Live conversation (voice_duplex.py): micro-turns while you talk ------------------------
@app.post("/api/voice/duplex")
async def voice_duplex_step(request: Request, sid: str = "", pause_ms: int = 0, final: int = 0):
    """One micro-turn: the utterance so far (16 kHz mono WAV) -> NDJSON lines: {"heard", "asr_ms"}, then the
    decision {"action": wait|mm|say|do, "llm_ms"}, and for "say" the reply as {"text"} pieces and {"done"}."""
    from fastapi.responses import StreamingResponse
    body = await request.body()
    if len(body) > 4 * 1024 * 1024:
        return {"ok": False, "message": "That utterance is too long for a live turn."}
    settings = shared.app_state["settings"]
    t0 = time.perf_counter()
    try:
        heard = await asyncio.to_thread(voice.transcribe, body, settings)
    except Exception as e:
        return {"ok": False, "message": str(e)[:300]}
    asr_ms = round((time.perf_counter() - t0) * 1000)

    def lines():
        yield json.dumps({"heard": heard.get("text", ""), "lang": heard.get("lang") or "en", "asr_ms": asr_ms}) + "\n"
        try:
            for item in voice_duplex.step(sid, heard.get("text", ""), pause_ms, bool(final), settings):
                yield json.dumps(item) + "\n"
        except Exception as e:
            yield json.dumps({"action": "error", "message": str(e)[:200]}) + "\n"
    return StreamingResponse(iterate_in_threadpool(lines()), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store"})


class DuplexBody(BaseModel):
    sid: str = ""
    spoken: str = ""


@app.post("/api/voice/duplex/warm")
async def voice_duplex_warm():
    """Voice mode opened in live mode: load its model now, not on the first answer."""
    return await asyncio.to_thread(voice_duplex.warm, shared.app_state["settings"])


@app.post("/api/voice/duplex/commit")
def voice_duplex_commit(body: DuplexBody):
    """Talked over: only what was heard of the last reply stays in the conversation."""
    voice_duplex.commit(body.sid, body.spoken)
    return {"ok": True}


@app.post("/api/voice/duplex/end")
def voice_duplex_end(body: DuplexBody):
    voice_duplex.end(body.sid)
    return {"ok": True}


@app.post("/api/voice/stop")
def voice_stop():
    """Voice mode closed: the recognizer and the voice stop, freeing their memory."""
    voice.stop_all()
    return {"ok": True}


# ---- The app window and the overlay window (app_window.py) -------------------------------
class ClientLog(BaseModel):
    kind: str = "error"
    message: str = ""
    detail: str = ""


_client_log_n = {"n": 0}


@app.post("/api/client-log")
def client_log(body: ClientLog):
    """The page's own errors (and a Settings page found blank) into data/logs/client.log - so a problem seen only
    in the desktop window can be read afterwards. Capped at 500 entries a run; nothing leaves the PC."""
    if _client_log_n["n"] >= 500:
        return {"ok": False}
    _client_log_n["n"] += 1
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs", "client.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{body.kind[:40]}] {body.message[:1000]}"
                + (f"\n    {body.detail[:3000]}" if body.detail else "") + "\n")
    return {"ok": True}


@app.get("/api/version")
def app_version_now():
    return {"version": app_version.current(), "releases": app_version.RELEASES_URL}


@app.post("/api/version/check")
def app_version_check():
    """Settings > About > Check for updates (only when pressed): the newest GitHub release and its installer."""
    return app_version.check(force=True)


@app.get("/api/app-window")
def app_window_status():
    return {"overlay": app_window.overlay_status(), "browser": bool(app_window.find_browser())}


@app.post("/api/app-window/open")
def app_window_open():
    """Open the workspace in a window of its own (from a browser tab: "Open as an app")."""
    how = app_window.open_app()
    return {"ok": True, "opened": how}


# ---- Network (netlayer.py) ---------------------------------------------------------------
@app.get("/api/network")
def network_status():
    s = shared.app_state["settings"]
    return {"settings": netlayer.clean_settings(s), "components": netlayer.COMPONENT_LABELS,
            "routes": {c: netlayer.route_for(c).public() for c in netlayer.COMPONENTS},
            "blocked": netlayer.blocked_state(), "session_direct": netlayer.session_direct(),
            "system_proxy": netlayer.system_proxy(), "guard": netlayer.installed(),
            "public": dict(netlayer._public) or None, "vpn": dict(netlayer._vpn["value"])}


@app.get("/api/network/stats")
def network_stats():
    """The speed indicator (polled about once a second while the window is visible)."""
    s = shared.app_state["settings"]
    return {**netlayer.stats(), "indicator": s.get("net_indicator", True), "scope": s.get("net_indicator_scope", "app"),
            "search_mode": search_engine.MODE_INFO.get(s.get("search_mode"), {}).get("short", "Standard"),
            "temporary": bool(sessions.temporary_chat())}


@app.get("/api/network/log")
def network_log(limit: int = 200):
    return {"entries": netlayer.log_entries(max(1, min(int(limit), netlayer.LOG_MAX))),
            "saving": shared.app_state["settings"].get("net_log_save", False)}


@app.delete("/api/network/log")
def network_log_clear():
    return {"ok": True, "cleared": netlayer.clear_log()}


class OnBody(BaseModel):
    on: bool


@app.post("/api/network/session-direct")
def network_session_direct(body: OnBody):
    """The kill switch's "Go direct for this session" (and back)."""
    netlayer.set_session_direct(body.on)
    netlayer.clear_blocked()
    event_log.log_action("Network: going direct for this session." if body.on else "Network: back to the configured route.")
    return {"ok": True, "session_direct": netlayer.session_direct()}


@app.get("/api/network/vpn")
def network_vpn(refresh: int = 0):
    return netlayer.vpn_status(force=bool(refresh))


@app.post("/api/network/ip")
def network_ip():
    """Public IP and approximate country - only on demand, through the active route."""
    return netlayer.public_ip(force=True)


class LeakBody(BaseModel):
    webrtc: dict | None = None


@app.post("/api/network/leak-test")
def network_leak_test(body: LeakBody):
    return netlayer.leak_test(body.webrtc)


@app.post("/api/network/speed-test")
def network_speed_test():
    """Only on the user's click: 10 MB from speed.cloudflare.com through the active route."""
    return netlayer.speed_test()


@app.get("/api/diagnostics")
def get_diagnostics(run: int = 0):
    """The last system check (diagnostics.py); run=1 runs it again (a few seconds)."""
    result = diagnostics.last()
    if run or result is None:
        try:
            result = diagnostics.run()
        except Exception as e:
            return {"ok": False, "message": f"The check failed: {e}"}
    return {"ok": True, **result}


class FixBody(BaseModel):
    id: str


@app.post("/api/diagnostics/fix")
def fix_diagnostic(body: FixBody):
    try:
        result = diagnostics.fix(body.id)
    except (KeyError, ValueError, OSError) as e:
        return {"ok": False, "message": str(e)}
    if result.get("ok"):
        event_log.log_action(f"Diagnostics fix '{body.id}': {result['message'][:160]}")
    return {**result, "diagnostics": diagnostics.last()}


_placement = {"at": 0.0, "model": None, "value": None}


def _model_placement(model):
    """Where the loaded model is (VRAM / spilled into RAM / layers on the CPU), for the model strip - every 3 s at most."""
    if not model:
        return None
    if _placement["model"] != model or time.time() - _placement["at"] > 3:
        try:
            _placement.update(at=time.time(), model=model, value=diagnostics.placement(model))
        except Exception:
            _placement.update(at=time.time(), model=model, value=None)
    return _placement["value"]


@app.get("/api/usage")
def get_usage(chat: str = "", days: int = 7):
    """Tokens, web searches and page reads (and cloud cost) for a chat and per day (usage.py)."""
    loaded = shared.chat_store.load(chat) if chat else None
    return {"chat": usage.chat_totals((loaded or {}).get("transcript")) if loaded else None,
            "days": shared.usage_store.days(max(1, min(int(days), 90)))}


class FileBody(BaseModel):
    name: str
    data: str                   # base64


@app.post("/api/files")
def upload_file(body: FileBody):
    """A file attached in the composer (composer_tools.FileStore); the message then carries its id.
    In a temporary chat it goes to that chat's own folder (sessions.file_store), deleted with it."""
    store = sessions.file_store()
    temporary = store is not shared.file_store
    try:
        meta = store.save(body.name, body.data)
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    except OSError as e:
        return {"ok": False, "message": f"Could not store the file: {e.strerror or e}"}
    event_log.log_action("File attached to the temporary chat." if temporary
                         else f"File attached: {meta['name']} ({meta['size']:,} bytes).")
    return {"ok": True, "file": meta}


@app.get("/api/files/{fid}")
def get_file(fid: str):
    meta = sessions.file_store().get(fid) or shared.file_store.get(fid)
    if not meta or not os.path.isfile(meta["path"]):
        return JSONResponse({"ok": False, "message": "That file is no longer here."}, status_code=404)
    inline = meta["kind"] in ("image", "pdf")
    return FileResponse(meta["path"], filename=meta["name"], content_disposition_type="inline" if inline else "attachment")


class CodeBody(BaseModel):
    code: str


@app.post("/api/code/run")
def run_code(body: CodeBody):
    """Run on a code block (Code runner): Python on this PC, in a temporary
    folder, isolated from user site-packages, 30 s at most. Not a sandbox - the
    click is the approval, and a run_python set to Deny in Permissions refuses."""
    import subprocess
    import sys
    import tempfile
    if permissions.decide("run_python", shared.app_state["settings"]) == "deny":
        return {"ok": False, "message": "run_python is set to Deny in Settings > Permissions."}
    code = body.code or ""
    if not code.strip() or len(code) > 200_000:
        return {"ok": False, "message": "No code to run." if not code.strip() else "That code is too long to run here."}
    started = time.time()
    temp_chat = sessions.temporary_chat()               # its scratch folder lives in the temporary chat's own
    with tempfile.TemporaryDirectory(prefix="aw-run-", dir=privacy.temp_dir(temp_chat["id"]) if temp_chat else None) as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp, capture_output=True, text=True,
                               errors="replace", timeout=30, stdin=subprocess.DEVNULL)
            out, err, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out, err, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", "Stopped after 30 s.", -1
    event_log.log_action(f"Ran a Python code block from the chat (exit {rc}).")
    return {"ok": rc == 0, "exit": rc, "stdout": out[-20000:], "stderr": err[-8000:],
            "seconds": round(time.time() - started, 2)}


_favicons: dict = {}          # domain -> (fetched at, bytes, media type) - None bytes: the site has none


@app.get("/api/favicon/{domain}")
def favicon(domain: str):
    """A source's icon for the citation cards - fetched once from the site
    itself (public addresses only) and kept for a day. 404: show a letter."""
    from fastapi.responses import Response
    domain = domain.strip().lower()
    if not re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", domain) or len(domain) > 120:
        return Response(status_code=404)
    hit = _favicons.get(domain)
    mode = shared.app_state["settings"].get("search_mode", "standard")
    if (not hit or time.time() - hit[0] > 86400) and mode in ("local", "disabled"):
        # Local Only / Disabled: nothing leaves this PC for search - not even a site's icon.
        return Response(status_code=404, headers={"Cache-Control": "no-store"})
    temp = sessions.temporary_chat()
    if not hit or time.time() - hit[0] > 86400:
        body, ctype = None, ""
        try:
            # Private / Max: their headers and route. In a temporary chat the request is that
            # chat's (tagged, never saved) and the icon is not cached - which sites it showed
            # stays with the chat.
            with privacy.scope(temp["id"] if temp else None), search_engine.scope(mode):
                resp, data, _final = web_tools.fetch(f"https://{domain}/favicon.ico")
            ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            if resp.status_code < 400 and data and len(data) < 200_000 and (
                    ctype.startswith("image/") or data[:4] in (b"\x00\x00\x01\x00", b"\x89PNG")):
                body = data
                ctype = ctype if ctype.startswith("image/") and "svg" not in ctype else "image/x-icon"
        except Exception:
            body = None
        if len(_favicons) > 500:
            _favicons.clear()
        hit = (time.time(), body, ctype)
        if not temp:
            _favicons[domain] = hit
    cache = "no-store" if temp else None
    if not hit[1]:
        return Response(status_code=404, headers={"Cache-Control": cache or "max-age=3600"})
    return Response(content=hit[1], media_type=hit[2], headers={"Cache-Control": cache or "max-age=86400"})


@app.get("/api/memory")
def list_memory():
    return {"entries": shared.memory_store.list(), "path": shared.memory_store.path,
            "enabled": shared.app_state["settings"].get("memory_enabled", True),
            "max_entries": memory.MAX_ENTRIES, "max_text": memory.MAX_TEXT}


@app.post("/api/memory")
def add_memory(body: MemoryBody):
    try:
        entry = shared.memory_store.add(body.text or "", source="user")
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    if body.pinned:
        entry = shared.memory_store.update(entry["id"], pinned=True)
    event_log.log_action("Saved memory: entry added in Settings.")
    return {"ok": True, "entry": entry}


@app.post("/api/memory/{entry_id}")
def edit_memory(entry_id: str, body: MemoryBody):
    try:
        entry = shared.memory_store.update(entry_id, text=body.text, pinned=body.pinned)
    except KeyError:
        return {"ok": False, "message": "That memory no longer exists."}
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    return {"ok": True, "entry": entry}


@app.delete("/api/memory/{entry_id}")
def delete_memory(entry_id: str):
    ok = shared.memory_store.delete(entry_id)
    if ok:
        event_log.log_action("Saved memory: entry deleted in Settings.")
    return {"ok": ok, "message": None if ok else "That memory no longer exists."}


# ---- Chats and projects (chat_store.py, projects.py, agent/sessions.py) ----
# Chats live in data/chats/<id>.json, projects in data/projects/<id>.json;
# deleting either moves the file to a trash/ folder next to it.
def _with_lock(fn):
    """Run fn() while no turn is running (switching chats replaces the
    conversation the running turn is using)."""
    if tasks.turns.running() or not shared.state_lock.acquire(timeout=2):
        return {"ok": False, "message": _busy("A task is still running - stop it first.")}
    try:
        return fn()
    finally:
        shared.state_lock.release()


class ChatUpdate(BaseModel):
    title: str = None
    project: str = None          # a project id, or "none"
    pinned: bool | None = None
    archived: bool | None = None


class ProjectBody(BaseModel):
    name: str = None
    folder: str = None
    files: list = None
    instructions: str = None
    notes: str = None
    roblox_place: str = None
    effort: str = None           # a level id, or "" for "use the model/global setting"
    prompt_prep: str = None      # a mode id, or ""


@app.get("/api/chats")
def list_chats(project: str = "*", q: str = "", limit: int = 200):
    """project: "*" (every chat), "none" (chats in no project) or a project id."""
    return {"chats": sessions.list_chats(project, q, max(1, min(int(limit), 1000))),
            "active": sessions.public_chat(), "dir": shared.chat_store.root}


@app.post("/api/chats/temporary")
def start_temporary_chat():
    """Open a temporary chat (privacy.py): kept in memory only - never in history,
    the sidebar, memory, a project, the search cache or the session file."""
    def go():
        _reset_conversation_locked()
        chat = sessions.new_temporary_chat()
        return {"ok": True, "chat": sessions.public_chat(), "transcript": chat["transcript"]}
    return _with_lock(go)


@app.post("/api/chats/temporary/save")
def save_temporary_chat():
    """"Save this chat": the open temporary chat becomes a normal saved chat. Only on the user's click."""
    def go():
        try:
            saved = sessions.save_temporary()
        except KeyError:
            return {"ok": False, "message": "No temporary chat is open."}
        except ValueError as e:                        # locked (chat_guard.py)
            return {"ok": False, "message": str(e)}
        return {"ok": True, "chat": {k: saved.get(k) for k in ("id", "title", "project", "created", "updated", "transcript")},
                "project": sessions.public_project()}
    return _with_lock(go)


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    if privacy.is_temp_id(chat_id):                    # the open temporary chat, from memory
        chat = sessions.temporary_chat()
        if not chat or chat["id"] != chat_id:
            return {"ok": False, "message": "That temporary chat has ended - nothing of it was kept."}
        return {"ok": True, "chat": {**{k: chat.get(k) for k in ("id", "title", "project", "created", "updated", "transcript")},
                                     "temporary": True}}
    chat = shared.chat_store.load(chat_id)
    if chat is None:
        return {"ok": False, "message": "That chat no longer exists."}
    return {"ok": True, "chat": _chat_view(chat)}


def _chat_view(chat: dict) -> dict:
    """A saved chat for the page, with `locked` when it was closed for going past the chat limits."""
    out = {k: chat.get(k) for k in ("id", "title", "project", "created", "updated", "transcript", "pending")}
    if chat_guard.is_locked(chat):
        out["locked"] = (chat.get("guard") or {}).get("reason") or True
    return out


@app.post("/api/chats/{chat_id}/open")
def open_chat(chat_id: str):
    def go():
        if shared.chat_store.load(chat_id) is None:
            return {"ok": False, "message": "That chat no longer exists."}
        _reset_conversation_locked()
        chat = sessions.open_chat(chat_id)
        return {"ok": True, "chat": _chat_view(chat), "project": sessions.public_project()}
    return _with_lock(go)


@app.post("/api/chats/{chat_id}")
def update_chat(chat_id: str, body: ChatUpdate):
    chat = None
    if body.title is not None:
        chat = shared.chat_store.rename(chat_id, body.title)
        active = shared.app_state.get("active_chat")
        if chat and active and active.get("id") == chat_id:
            active["title"] = chat["title"]
    if body.project is not None:
        try:
            chat = sessions.move_chat(chat_id, None if body.project in ("", "none") else body.project)
        except KeyError:
            return {"ok": False, "message": "That project no longer exists."}
    if body.pinned is not None or body.archived is not None:
        chat = shared.chat_store.set_flags(chat_id, body.pinned, body.archived)
    if chat is None:
        return {"ok": False, "message": "That chat no longer exists."}
    return {"ok": True, "chat": {k: chat.get(k) for k in ("id", "title", "project", "created", "updated", "pinned", "archived")}}


class MessageAction(BaseModel):
    op: str                      # pin | delete | truncate | branch
    index: int                   # the transcript entry
    pinned: bool = False


@app.post("/api/chats/{chat_id}/messages")
def message_action(chat_id: str, body: MessageAction):
    """Message actions (agent/sessions.edit_messages). Not while a task runs: the
    open chat's conversation is what that task is using."""
    def go():
        # A branch becomes the open chat: the task state of the old one is cleared first.
        chat = sessions.edit_messages(chat_id, body.op, body.index, body.pinned, before_open=_reset_conversation_locked)
        return {"ok": True, "chat": {k: chat.get(k) for k in ("id", "title", "project", "created", "updated", "transcript", "pending")},
                "project": sessions.public_project()}
    try:
        return _with_lock(go)
    except KeyError:
        return {"ok": False, "message": "That chat no longer exists."}
    except ValueError as e:
        return {"ok": False, "message": str(e)}


# ---- Chat limits (chat_guard.py): Settings changes the limits; a hidden gesture reopens a closed chat.
class GuardLimits(BaseModel):
    categories: list = None
    strike_limit: int = None
    limits: dict = None                  # per category: {"malware": 3, "abuse": 2, ...}
    decay_messages: int = None           # messages within the limits before each count drops by one (0 = never)


@app.get("/api/guard")
def guard_info():
    return chat_guard.public()


@app.post("/api/guard/limits")
def guard_limits(body: GuardLimits):
    info = chat_guard.update_limits(body.categories, body.strike_limit, body.limits, body.decay_messages)
    event_log.log_action(f"Chat limits changed by the owner: {', '.join(info['categories'])}; limits "
                         f"{', '.join(f'{k} {v}' for k, v in info['limits'].items())}; counts fade every {info['decay_messages']} messages.")
    return {"ok": True, **info}


@app.post("/api/chats/{chat_id}/unlock")
def unlock_chat(chat_id: str):
    """Reopens a chat closed by the chat limits (the banner's hidden gesture - a small model may refuse by mistake).
    A chat closed for minors stays closed."""
    def go():
        active = shared.app_state.get("active_chat")
        live = active if active and active.get("id") == chat_id else None
        chat = live if live is not None else shared.chat_store.load(chat_id)
        if chat is None:
            return {"ok": False, "message": "That chat no longer exists."}
        try:
            guard = chat_guard.unlock(chat)
        except chat_guard.GuardError as e:
            return {"ok": False, "code": e.code, "message": str(e)}
        if not privacy.is_temporary(chat):
            stored = shared.chat_store.load(chat_id) or chat
            shared.chat_store.save({**stored, "guard": guard}, touch=False)
        event_log.log_action(f"Chat limits: '{chat.get('title')}' was reopened.")
        return {"ok": True, "chat": sessions.public_chat() if live is not None else _chat_view({**chat, "guard": guard})}
    return _with_lock(go)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    def go():
        if privacy.is_temp_id(chat_id):                # "Discard": the temporary chat ends now
            was_open = sessions.close_temporary()
            _reset_conversation_locked()
            return {"ok": True, "was_open": was_open}
        if not shared.chat_store.delete(chat_id):
            return {"ok": False, "message": "That chat no longer exists."}
        shared.run_log.delete_where(chat=chat_id)          # its past runs go with it
        was_open = sessions.forget_active_if(chat_id=chat_id)
        if was_open:
            _reset_conversation_locked()
        event_log.log_action("Chat moved to data/chats/trash.")
        return {"ok": True, "was_open": was_open}
    return _with_lock(go)


def _project_row(p: dict, counts: dict) -> dict:
    return {**p, "chats": counts.get(p["id"], 0)}


@app.get("/api/projects")
def list_projects(q: str = ""):
    counts: dict = {}
    for c in shared.chat_store.list("*", "", 100000):
        if c.get("project"):
            counts[c["project"]] = counts.get(c["project"], 0) + 1
    return {"projects": [_project_row(p, counts) for p in shared.project_store.search(q)],
            "active": sessions.public_project(), "dir": shared.project_store.root}


def _project_fields(body: ProjectBody) -> dict:
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    for key in ("effort", "prompt_prep"):
        if fields.get(key) == "":
            fields[key] = None              # clean_fields: not a level -> "no project default"
    return fields


@app.post("/api/projects")
def create_project(body: ProjectBody):
    try:
        project = shared.project_store.create(_project_fields(body))
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    event_log.log_action(f"Project created: {project['name']}.")
    return {"ok": True, "project": project}


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    project = shared.project_store.get(pid)
    if project is None:
        return {"ok": False, "message": "That project no longer exists."}
    return {"ok": True, "project": project, "chats": sessions.list_chats(pid, "", 200)}


@app.post("/api/projects/{pid}")
def update_project(pid: str, body: ProjectBody):
    try:
        project = shared.project_store.update(pid, _project_fields(body))
    except KeyError:
        return {"ok": False, "message": "That project no longer exists."}
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    sessions.refresh_active_project()
    return {"ok": True, "project": project}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    def go():
        if not shared.project_store.delete(pid):
            return {"ok": False, "message": "That project no longer exists."}
        was_open = sessions.forget_active_if(pid=pid)
        if was_open:
            _reset_conversation_locked()
        event_log.log_action("Project moved to data/projects/trash (its chats and folder are untouched).")
        return {"ok": True, "was_open": was_open}
    return _with_lock(go)


@app.post("/api/projects/{pid}/activate")
def activate_project(pid: str):
    """Open a project ("none": leave every project). A new chat starts in it."""
    def go():
        try:
            project = sessions.activate_project(None if pid == "none" else pid)
        except KeyError:
            return {"ok": False, "message": "That project no longer exists."}
        _reset_conversation_locked()
        return {"ok": True, "project": sessions.public_project() if project else None}
    return _with_lock(go)


@app.get("/api/projects/{pid}/index")
def project_index(pid: str):
    """The file index the assistant gets for this project's folder (cached)."""
    project = shared.project_store.get(pid)
    if project is None:
        return {"ok": False, "message": "That project no longer exists."}
    if not project.get("folder"):
        return {"ok": True, "index": None}
    idx = shared.folder_index.get(project["folder"])
    if idx is None:
        return {"ok": False, "message": f"{project['folder']} is not reachable right now."}
    return {"ok": True, "index": {"folder": idx["folder"], "files": len(idx["files"]), "languages": idx["languages"],
                                  "truncated": idx["truncated"], "readme": bool(idx.get("readme")),
                                  "built": idx["built"], "sample": [f["path"] for f in idx["files"][:40]]}}


@app.get("/api/search")
def search_all(q: str = "", limit: int = 50):
    """Chats and projects matching every word of q."""
    if not (q or "").strip():
        return {"chats": [], "projects": []}
    limit = max(1, min(int(limit), 200))
    return {"chats": sessions.list_chats("*", q, limit), "projects": shared.project_store.search(q, limit)}


# ---- Past runs (run_log.py; the Activity panel's history) -------------------
@app.get("/api/runs")
def list_runs(q: str = "", limit: int = 50):
    """Saved runs, newest first; q matches every word against the request and
    the runs' step labels, results and tool names."""
    return {"runs": shared.run_log.list(q, max(1, min(int(limit), 200))), "max_runs": run_log.MAX_RUNS,
            "max_days": run_log.MAX_AGE_S // 86400}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = shared.run_log.get(run_id)
    if run is None:
        return {"ok": False, "run": None, "message": "That run is no longer saved."}
    # Each step's Undo as it stands now (the undo trail keeps 14 days, runs 30).
    ids = [e["undo_id"] for e in run.get("events") or [] if e.get("undo_id")]
    states = files.undo.states(ids) if ids else {}
    for e in run.get("events") or []:
        if e.get("undo_id"):
            e["undo_state"] = states.get(e["undo_id"]) or "gone"
    return {"ok": True, "run": run}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    return {"ok": shared.run_log.delete(run_id)}


@app.delete("/api/runs")
def clear_runs():
    """Settings > Privacy: forget every past run (not chats, not the undo trail)."""
    gone = shared.run_log.delete_where(None)
    event_log.log_action(f"Past runs cleared ({gone}).")
    return {"ok": True, "deleted": gone}


# ---- App adapters (app_registry.AppAdapter: Roblox Studio today) -----------
@app.get("/api/apps")
def list_app_adapters():
    return {"apps": [{"id": a.app_id, "name": a.display_name, **a.status()} for a in app_registry.adapters()]}


@app.post("/api/apps/{app_id}/check")
def check_app_adapter(app_id: str):
    """An active round trip to the app (Roblox: a ping through the bridge)."""
    adapter = app_registry.get_adapter(app_id)
    if adapter is None:
        return {"ok": False, "message": "Unknown app."}
    res = adapter.health_check()
    event_log.log_action(f"{adapter.display_name} check: {res.get('message')}")
    return {**res, "status": adapter.status()}


# ---- MCP servers (mcp_client.py; the MCP tab) ------------------------------
class MCPServerBody(BaseModel):
    name: str
    command: str = None
    args: list = None
    env: dict = None             # values "" = keep the saved value (the UI never sees secrets)
    url: str = None
    headers: dict = None
    enabled: bool = True


@app.get("/api/mcp")
def list_mcp_servers():
    return {"servers": [s.public() for s in shared.mcp_hub.servers.values()], "path": shared.mcp_hub.path}


def _mcp_run(name: str, action):
    try:
        server = action(name)
    except KeyError:
        return {"ok": False, "message": "No MCP server by that name."}
    return {"ok": server.status in ("connected", "disconnected"), "server": server.public(),
            "message": server.error or server.status}


@app.post("/api/mcp/servers")
def save_mcp_server(body: MCPServerBody):
    """Add or change a server. Adding one means this app will start that
    program (or talk to that URL) - its tools still ask before each use."""
    try:
        server = shared.mcp_hub.upsert(body.name, body.model_dump(exclude={"name"}))
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    event_log.log_action(f"MCP server saved: {body.name}.")
    if server.config.get("enabled", True):
        server.connect()
    return {"ok": True, "server": server.public()}


@app.delete("/api/mcp/servers/{name}")
def delete_mcp_server(name: str):
    ok = shared.mcp_hub.remove(name)
    if ok:
        event_log.log_action(f"MCP server removed: {name}.")
    return {"ok": ok, "message": None if ok else "No MCP server by that name."}


class ResourceBody(BaseModel):
    uri: str


@app.post("/api/mcp/servers/{name}/resource")
def preview_mcp_resource(name: str, body: ResourceBody):
    """The user opening a resource in the MCP tab. Reading one runs the
    server's code, so it is the mcp_read_resource tool through the one path
    (permission, approval card, executor, Activity) - not a side channel."""
    return _run_tool_for_user("mcp_read_resource", {"server": name, "uri": body.uri,
                                                    "justification": "You opened this resource in the MCP tab."},
                              f"open MCP resource {body.uri}")


@app.post("/api/mcp/servers/{name}/{action}")
def mcp_server_action(name: str, action: str):
    """connect / disconnect / retry (retry = connect again after an error)."""
    if action in ("connect", "retry"):
        res = _mcp_run(name, shared.mcp_hub.connect)
    elif action == "disconnect":
        res = _mcp_run(name, shared.mcp_hub.disconnect)
    else:
        return {"ok": False, "message": "Unknown action."}
    event_log.log_action(f"MCP {name}: {action} -> {res.get('server', {}).get('status')}")
    return res


# ---- Routines (routines.py, agent/routine_runner.py) -----------------------
APP_PORT = int(os.environ.get("AGENTIC_PORT") or 8000)      # run.bat serves the app on 8000


class RoutineBody(BaseModel):
    name: str = None
    task_id: str = None          # create: the finished run whose steps to keep
    keep: list = None            # update: indexes of the steps to keep
    prompt: str | None = None    # a prompt routine: what it asks on every run (routines.create_prompt)
    tools: list | None = None    # ... and the composer tools it uses (routines.PROMPT_TOOLS)


class ScheduleBody(BaseModel):
    repeat: str = "daily"
    time: str = ""
    date: str = ""


@app.get("/api/routines")
def list_routines():
    return {"routines": shared.routine_store.list(), "path": shared.routine_store.path, "port": APP_PORT}


@app.post("/api/routines")
def create_routine(body: RoutineBody):
    try:
        if body.prompt is not None:
            routine = shared.routine_store.create_prompt(body.name or "", body.prompt, body.tools,
                                                         sessions.active_project_id())
            event_log.log_action(f"Routine created: {routine['name']} (asks: {routine['prompt'][:80]}).")
            return {"ok": True, "routine": routine}
        routine = routine_runner.save_candidate(body.task_id or "", body.name or "")
    except KeyError:
        return {"ok": False, "message": "That run is no longer available to save - run it again first."}
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    return {"ok": True, "routine": routine}


@app.post("/api/routines/{rid}")
def update_routine(rid: str, body: RoutineBody):
    try:
        return {"ok": True, "routine": shared.routine_store.update(rid, name=body.name, keep=body.keep,
                                                                    prompt=body.prompt, tools=body.tools)}
    except KeyError:
        return {"ok": False, "message": "That routine no longer exists."}
    except ValueError as e:
        return {"ok": False, "message": str(e)}


@app.delete("/api/routines/{rid}")
def delete_routine(rid: str):
    routine = shared.routine_store.get(rid)
    if routine is None:
        return {"ok": False, "message": "That routine no longer exists."}
    if routine.get("schedule"):
        return {"ok": False, "message": "Remove its schedule first, so nothing in Task Scheduler points at it."}
    shared.routine_store.delete(rid)
    event_log.log_action(f"Routine deleted: {routine['name']}.")
    return {"ok": True}


@app.post("/api/routines/{rid}/run")
def run_routine(rid: str, trigger: str = "click"):
    """Start the routine as a turn (one at a time, Stop works, steps stream
    to the chat). A scheduled run posts here from Task Scheduler with
    trigger=schedule; it asks for the same approvals as a click."""
    routine = shared.routine_store.get(rid)
    if routine is None:
        return {"status": "error", "message": "That routine no longer exists."}
    trigger = "schedule" if trigger == "schedule" else "click"
    if _model_loading() or (shared.state_lock.locked() and not tasks.turns.running()):
        why = _busy("Busy - a model is loading. Try again in a moment.")
        if trigger == "schedule":                    # a scheduled run is never dropped silently
            routine_runner._record(routine, "skipped", why, trigger)
            event_log.emit_event("note", f"Routine '{routine['name']}' was due but the app was busy")
        return {"status": "error", "message": why}
    message = f"Run routine: {routine['name']}"
    turn = tasks.turns.start(SimpleNamespace(message=message, with_vision=False, routine_id=rid, trigger=trigger),
                             {"message": message, "with_vision": False, "routine": rid, "trigger": trigger,
                              "event_seq": shared.app_state.get("event_seq", 0)})
    if turn is None:
        if trigger == "schedule":
            routine_runner._record(routine, "skipped", "Another task was running at the scheduled time.", trigger)
            event_log.emit_event("note", f"Routine '{routine['name']}' was due but another task was running")
        return {"status": "error", "message": "Busy - a task is already running. Stop it or wait for it to finish."}
    return {"status": "accepted", "turn": turn}


def _run_tool_for_user(tool_name: str, args: dict, reason: str) -> dict:
    """A tool the user started from the UI (not a model): the same permission
    gate, approval card and guarded executor as every other call."""
    if tasks.turns.running() or not shared.state_lock.acquire(blocking=False):
        return {"ok": False, "message": _busy("Busy - try again when the current task finishes.")}
    try:
        shared.app_state["current_user_request"] = reason
        shared.app_state["turn_policy"] = {"mode": "action", "allow_mutation": True, "reason": reason}
        shared.app_state["intent_guard"] = None
        shared.app_state["untrusted_seen"] = []
        decision, approved = approvals.gate_batch([{"tool_name": tool_name, "args": args,
                                                    "justification": args.get("justification", "")}])
        if not approved:
            return {"ok": False, "message": "That action is set to Deny in your permission settings."
                    if decision == "deny" else "Cancelled - nothing was changed."}
        return runtime.execute_tool_guarded(tool_name, args)
    finally:
        shared.app_state["turn_policy"] = {"mode": "none", "allow_mutation": False, "reason": "no active request"}
        shared.state_lock.release()


@app.post("/api/routines/{rid}/schedule")
def schedule_routine(rid: str, body: ScheduleBody):
    """Schedule the routine with the existing schedule_task tool (Windows Task
    Scheduler), through its normal approval card."""
    routine = shared.routine_store.get(rid)
    if routine is None:
        return {"ok": False, "message": "That routine no longer exists."}
    try:
        when = routines.clean_schedule(body.model_dump())
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    run_at = f"{when['date'] or time.strftime('%Y-%m-%d')} {when['time']}"
    args = {"name": routines.task_name(routine), "run_at": run_at, "repeat": when["repeat"],
            "command": routines.trigger_command(routine, APP_PORT),
            "justification": f"Run your routine '{routine['name']}' on a schedule. Each run still asks for approvals."}
    res = _run_tool_for_user("schedule_task", args, f"schedule routine '{routine['name']}'")
    if not res.get("ok"):
        return {"ok": False, "message": res.get("message") or "Could not schedule it."}
    routine = shared.routine_store.update(rid, schedule={**when, "task": routines.task_name(routine)})
    return {"ok": True, "routine": routine, "message": res.get("message")}


@app.delete("/api/routines/{rid}/schedule")
def unschedule_routine(rid: str):
    routine = shared.routine_store.get(rid)
    if routine is None:
        return {"ok": False, "message": "That routine no longer exists."}
    if not routine.get("schedule"):
        return {"ok": True, "routine": routine}
    res = _run_tool_for_user("cancel_scheduled_task", {"name": routines.task_name(routine),
                                                       "justification": f"Stop running '{routine['name']}' on a schedule."},
                             f"unschedule routine '{routine['name']}'")
    gone = res.get("ok") or "not found" in str(res.get("message") or "").lower() or "cannot find" in str(res.get("message") or "").lower()
    if not gone:
        return {"ok": False, "message": res.get("message") or "Could not remove the schedule."}
    return {"ok": True, "routine": shared.routine_store.update(rid, schedule=None)}


class ApprovalAnswer(BaseModel):
    approved: bool
    via: str | None = None              # "voice": said in voice mode (approval_broker.voice_ok limits what it can approve)
    scope: str | None = None            # "request": allow these tools for the rest of this request (request_ok only)


@app.post("/api/approval/{request_id}")
def answer_approval(request_id: str, body: ApprovalAnswer):
    via = "voice" if body.via == "voice" else ""
    ok = approvals.broker.answer(request_id, body.approved, via, "request" if body.scope == "request" else "")
    if via:
        event_log.log_action(f"Approval {'allowed' if body.approved else 'denied'} by voice" + ("" if ok else " - refused"))
    return {"status": "success" if ok else "error",
            **({} if ok or not via else {"message": "This one can't be approved by voice - tap Allow."})}


def _run_usage() -> dict | None:
    budget = shared.app_state.get("run_budget")
    if budget is None:
        return None
    return {**budget.usage(), "tool_calls": runtime.tool_exec.calls}


@app.get("/api/state")
def get_state(focused: int = None, since: int = 0, log_since: int = 0):
    """Polled by the UI every ~1-2.5 s, so everything here must be cheap:
    llama-server reachability and models.ini are cached in llama_local, the
    audit log is sent incrementally, and nothing enumerates windows unused."""
    if focused is not None:
        approvals.broker.heartbeat(bool(focused))
        observer.engine.set_client_focus(bool(focused))
    active_model = shared.app_state.get("local_engaged") or shared.app_state.get("local_model") or None
    return {
        "events": event_log.events_since(since),
        "event_seq": shared.app_state.get("event_seq", 0),
        "server_started": event_log.SERVER_STARTED,
        "conversation_turns": sum(1 for m in shared.app_state["conversation"] if m.get("role") == "user"),
        "pending_approval": approvals.broker.pending(),
        "pending_prep": prompt_prep.gate.pending(),
        "suggestions": observer.engine.active(),
        "awareness": observer.engine.context(),
        "roblox": roblox_bridge.status_dict(),
        "log": event_log.audit_since(log_since),          # [{seq, time, msg}] newer than log_since
        "log_seq": shared.app_state.get("audit_seq", 0),
        "turn": tasks.turns.current(),                # running or last finished turn
        "chat": sessions.public_chat(),               # the open saved chat (None until its first message)
        "project": sessions.public_project(),         # the open project, or None
        "routed_model": shared.app_state.get("routed_model"),   # set by a model handoff; the UI adopts it
        "task_phase": tasks.agent.current.phase if tasks.agent.current else None,
        "busy": shared.state_lock.locked(),
        "run": run_control.state(),          # idle | running | paused | stopping
        "benchmark": ({"model": benchmark_runner.state["running"], "progress": benchmark_runner.state["progress"]}
                      if benchmark_runner.state["running"] else None),
        "waiting_for_os_approval": shared.app_state.get("waiting_for_os_approval", False),
        "model": shared.app_state.get("last_working_model", config.PRIMARY_MODELS[0]),
        "mode": shared.app_state.get("mode", config.MODE_CLOUD),
        "local_up": llama_local.server_is_up_cached(),   # never blocks the poll
        "local_ctx": llama_local.model_context_limit(active_model),
        "local_pinned": shared.app_state.get("local_model") or "",
        "local_engaged": shared.app_state.get("local_engaged"),
        "pywinauto_available": elements.PYWINAUTO_AVAILABLE,
        "plan": shared.task_board.as_dict(),
        "usage": _run_usage(),                   # the running run against its effort limits, or None
        "context": _context_status(),            # the context meter beside the message box
        "active_window": ctx.probe_active_window(),
        "metrics": system_metrics.snapshot_nonblocking(active_model=shared.app_state.get("local_engaged")),
        "placement": _model_placement(shared.app_state.get("local_engaged")),
        "diagnostics": _diagnostics_counts(),
        "net": {"blocked": netlayer.blocked_state(), "session_direct": netlayer.session_direct()},   # the kill-switch banner
    }


def _diagnostics_counts():
    d = diagnostics.last()
    return {"at": d["at"], **d["counts"]} if d else None


# ==============================================================================
# Images made on this PC (image_gen.py)
# ==============================================================================
@app.get("/api/images")
def images_list(q: str = "", quality: str = "", fav: bool = False, trash: bool = False, limit: int = 120,
                group: bool = False):
    """The engine's state (a picture being drawn, a warm-up, a move) and the gallery, filtered. With group, a
    sequence (Angles/Frames) is one entry."""
    ok = quality in image_gen.MODELS or quality in image_gen.FAMILIES
    return {"status": image_gen.status(),
            "images": image_gen.recent(max(1, min(500, limit)), q[:200], quality if ok else "", fav, trash, group)}


def _tree_size(path: str) -> tuple[int, int]:
    """(bytes, files) under path; unreadable entries are skipped."""
    if os.path.isfile(path):
        try:
            return os.path.getsize(path), 1
        except OSError:
            return 0, 0
    total = files = 0
    for folder, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(folder, name))
                files += 1
            except OSError:
                pass
    return total, files


@app.get("/api/storage")
def storage():
    """Settings › Diagnostics › Storage: what this app keeps on disk, and how full the drives are. Read-only."""
    data = os.path.join(image_gen.ROOT, "data")
    items = []
    try:
        names = sorted(os.listdir(data))
    except OSError:
        names = []
    for name in names:
        size, files = _tree_size(os.path.join(data, name))
        items.append({"name": name, "bytes": size, "files": files, "dir": os.path.isdir(os.path.join(data, name))})
    items.sort(key=lambda i: -i["bytes"])
    drives = []
    for label, path in (("App data", data), ("Image models", image_gen._cfg()["image_root"])):
        try:
            u = shutil.disk_usage(path)
            drives.append({"label": label, "path": path, "drive": os.path.splitdrive(os.path.abspath(path))[0] or path,
                           "total": u.total, "free": u.free})
        except OSError:
            pass
    return {"ok": True, "data": data, "items": items, "total": sum(i["bytes"] for i in items), "drives": drives}


@app.get("/api/images/status")
def images_status():
    """Polled by the page while a picture is drawn (Images view, the chat's drawing card)."""
    return image_gen.status()


_IMAGE_NAME = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}\.png")


def _image_file(folder: str, name: str):
    """A made image. Only names the app itself writes - never a path."""
    if not _IMAGE_NAME.fullmatch(name):
        return JSONResponse({"ok": False, "message": "No such image."}, status_code=404)
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "message": "No such image."}, status_code=404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=31536000, immutable"})


_THUMB_NAME = re.compile(r"(\d{8}-\d{6}-[0-9a-f]{6})\.webp")


@app.get("/api/images/thumb/{name}")
def images_thumb(name: str):
    """A gallery tile: a small cached WebP of a made image (gallery or trash)."""
    m = _THUMB_NAME.fullmatch(name)
    try:
        path = image_gen.thumbnail(m.group(1)) if m else None
    except Exception:
        path = None
    if not path:
        return JSONResponse({"ok": False, "message": "No such image."}, status_code=404)
    return FileResponse(path, media_type="image/webp", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/images/trash/{name}")
def images_trash_file(name: str):
    return _image_file(os.path.join(image_gen.IMAGES_DIR, "trash"), name)


@app.get("/api/images/{name}")
def images_file(name: str):
    return _image_file(image_gen.IMAGES_DIR, name)


class ImageRequest(BaseModel):
    prompt: str = ""
    quality: str | None = None
    size: str | None = None
    seed: int | None = None
    negative: str = ""
    count: int = 1
    ref_id: str | None = None        # a made image to start from (Edit) - FLUX.2 klein edits it
    file_id: str | None = None       # or an attached picture (composer_tools.FileStore)
    think: bool = False              # Pony: the chat model plans the picture first (image_plan.py)
    check: bool = False              # ... and a vision model checks it, with one corrected attempt
    mode: str = "single"             # single | angles | frames (a sequence of the same character/scene)
    seq_count: int = 4


def _public_image(img: dict) -> dict:
    out = {k: v for k, v in img.items() if k not in ("path", "images")}
    out["images"] = [{k: v for k, v in x.items() if k not in ("path", "images")} for x in img.get("images") or [img]]
    return out


def _image_call(fn):
    """Runs one image request from the page (not from a chat turn): it waits while a chat turn holds the GPU's
    model instead of unloading it mid-answer, and every refusal comes back with its code for the page."""
    try:
        return {"ok": True, "image": _public_image(fn(wait_while=_gpu_wait))}
    except image_gen.ImageError as e:
        return {"ok": False, "message": str(e), "code": e.code}


@app.post("/api/images/generate")
def images_generate(body: ImageRequest):
    """Images' Make it (and Settings > Images' Try it): straight to the engine, no model in between."""
    ref = None
    if body.ref_id:
        try:
            ref = image_gen.image_bytes(body.ref_id)
        except image_gen.ImageError as e:
            return {"ok": False, "message": str(e), "code": e.code}
    elif body.file_id:
        meta = sessions.file_store().get(body.file_id) or shared.file_store.get(body.file_id)
        if not meta or meta.get("kind") != "image":
            return {"ok": False, "message": "That picture is no longer attached.", "code": "missing"}
        with open(meta["path"], "rb") as f:
            ref = f.read()
    if body.mode in ("angles", "frames"):
        return _image_call(lambda **kw: image_gen.generate_sequence(body.prompt, body.quality, body.mode, body.seq_count,
                                                                    body.size, body.seed, body.negative, think=body.think,
                                                                    **kw))
    if body.think and ref is None:
        return _image_call(lambda **kw: image_gen.generate_planned(body.prompt, body.quality, body.size, body.seed,
                                                                   body.negative, count=body.count,
                                                                   check_result=body.check, **kw))
    return _image_call(lambda **kw: image_gen.generate(body.prompt, body.quality, body.size, body.seed, body.negative,
                                                       ref_image=ref, count=body.count, **kw))


@app.get("/api/images/sequence/{seq_id}")
def images_sequence(seq_id: str):
    """Every picture of one sequence (angles or frames), in order."""
    if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", seq_id):
        return {"ok": False, "message": "No such sequence."}
    return {"ok": True, "images": image_gen.sequence(seq_id)}


class ImageAction(BaseModel):
    on: bool = True
    mode: str = "again"              # again | variations
    prompt: str = ""


@app.post("/api/images/{iid}/again")
def images_again(iid: str, body: ImageAction):
    return _image_call(lambda **kw: image_gen.again(iid, "variations" if body.mode == "variations" else "again", **kw))


@app.post("/api/images/{iid}/edit")
def images_edit(iid: str, body: ImageAction):
    return _image_call(lambda **kw: image_gen.edit(iid, body.prompt, **kw))


@app.post("/api/images/{iid}/fav")
def images_fav(iid: str, body: ImageAction):
    try:
        return {"ok": True, "image": image_gen.set_favorite(iid, body.on)}
    except image_gen.ImageError as e:
        return {"ok": False, "message": str(e), "code": e.code}


@app.post("/api/images/{iid}/trash")
def images_trash(iid: str):
    """Into data/images/trash - restorable; nothing here deletes a picture for good."""
    try:
        image_gen.trash(iid)
    except image_gen.ImageError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    event_log.log_action(f"Moved the image {iid} to the image trash.")
    return {"ok": True}


@app.post("/api/images/trash/empty")
def images_empty_trash():
    """Empty the image trash for good - the page asks first."""
    done = image_gen.empty_trash()
    event_log.log_action(f"Emptied the image trash: {done['count']} pictures, {done['freed'] / 1e6:.1f} MB freed.")
    return {"ok": True, **done}


@app.post("/api/images/{iid}/purge")
def images_purge(iid: str):
    """Delete one picture in the trash for good - the page asks first."""
    try:
        freed = image_gen.purge(iid)
    except image_gen.ImageError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    event_log.log_action(f"Deleted the image {iid} for good from the trash.")
    return {"ok": True, "freed": freed}


@app.post("/api/images/{iid}/restore")
def images_restore(iid: str):
    try:
        image_gen.restore(iid)
    except image_gen.ImageError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    return {"ok": True}


IMPROVE_SYSTEM = (
    "You turn a short request for a picture into one prompt for a text-to-image model. Write it in English (translate "
    "if needed) as a vivid visual description: the subject, the setting, the style or medium, the light, the "
    "composition, and any text that must appear, in quotes. Keep everything the user specified and invent nothing "
    "that changes what they asked for. At most 70 words. Reply with the prompt only - no preamble, no quotes around it.")


class ImageImprove(BaseModel):
    prompt: str = ""


@app.post("/api/images/improve")
def images_improve(body: ImageImprove):
    """Images' "Improve my prompt": the chat model rewrites a short idea into a full prompt the user can edit
    before making it. Not while a chat answer is running (one model, one GPU)."""
    idea = " ".join((body.prompt or "").split())[:1000]
    if not idea:
        return {"ok": False, "message": "Write the idea first."}
    if shared.state_lock.locked():
        return {"ok": False, "message": "The chat model is busy with an answer - try again when it's done."}
    image_gen.yield_gpu()
    try:
        out = llm.model_text(IMPROVE_SYSTEM, idea, max_tokens=220, label="Improving the picture prompt", temperature=0.6)
    except Exception as e:
        return {"ok": False, "message": f"The model couldn't do it: {e}"}
    text = (out.get("text") or "").strip().strip('"').strip()
    if not text:
        return {"ok": False, "message": "The model gave nothing back - try again."}
    return {"ok": True, "prompt": text[:1200], "model": out.get("model")}


class ImageWarm(BaseModel):
    quality: str | None = None


@app.post("/api/images/warm")
def images_warm(body: ImageWarm):
    """Images was opened or its prompt focused: read the model's files into Windows' cache now."""
    return {"ok": True, "started": image_gen.warm(body.quality)}


@app.post("/api/images/stop")
def images_stop():
    return {"ok": True, "stopped": image_gen.request_stop()}


@app.post("/api/images/unload")
def images_unload():
    """Frees the RAM and VRAM the image engine holds (it starts again with the next image)."""
    image_gen.stop()
    return {"ok": True, "status": image_gen.status()}


class ImageMove(BaseModel):
    dest: str = ""
    qualities: list = []


@app.post("/api/images/move/plan")
def images_move_plan(body: ImageMove):
    plan = image_gen.plan_move(body.dest, body.qualities)
    return {k: v for k, v in plan.items() if k != "files"} | {"count": len(plan.get("files") or [])}


@app.post("/api/images/move")
def images_move(body: ImageMove):
    """Copies the engine and the chosen models, checks every file, then switches the folder. The old copies stay."""
    try:
        plan = image_gen.move_models(body.dest, body.qualities)
    except image_gen.ImageError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    event_log.log_action(f"Moving image models to {plan['dest']} ({plan['total'] / 1e9:.1f} GB): "
                         f"{', '.join(plan['qualities'])}.")
    return {"ok": True, "plan": plan}


@app.post("/api/images/move/cancel")
def images_move_cancel():
    image_gen.cancel_move()
    return {"ok": True}


# ==============================================================================
# Videos made on this PC (video_gen.py)
# ==============================================================================
_VIDEO_NAME = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}\.webm")


def _video_file(folder: str, name: str):
    """A made clip. Only names the app itself writes - never a path."""
    if not _VIDEO_NAME.fullmatch(name):
        return JSONResponse({"ok": False, "message": "No such video."}, status_code=404)
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "message": "No such video."}, status_code=404)
    return FileResponse(path, media_type="video/webm", headers={"Cache-Control": "max-age=31536000, immutable"})


def _videos_status() -> dict:
    """The video engine's state; while sound is being added to a clip (video_audio.py) that is the job shown."""
    s = video_gen.status()
    sound = video_audio.status()["job"]
    if not s.get("job") and sound:
        spent = time.time() - sound.get("started", time.time())
        s.update(running=True, job={"stage": sound.get("stage"), "text": sound.get("text"), "sound": True,
                                    "progress": round(min(0.95, spent / (spent + 90)), 3)})
    return s


@app.get("/api/videos")
def videos_list(q: str = "", fav: bool = False, trash: bool = False, limit: int = 120):
    return {"status": _videos_status(), "videos": video_gen.recent(max(1, min(500, limit)), q[:200], fav, trash)}


@app.get("/api/videos/status")
def videos_status():
    return _videos_status()


@app.get("/api/videos/trash/{name}")
def videos_trash_file(name: str):
    return _video_file(os.path.join(video_gen.VIDEOS_DIR, "trash"), name)


@app.get("/api/videos/{name}")
def videos_file(name: str):
    return _video_file(video_gen.VIDEOS_DIR, name)


class VideoRequest(BaseModel):
    prompt: str = ""
    size: str | None = None
    length: str | None = None
    seed: int | None = None
    negative: str = ""
    ref_id: str | None = None        # a picture from Images to start from (image -> video)
    file_id: str | None = None       # or an attached picture (composer_tools.FileStore)
    best: bool | None = None         # the full VAE (slower) instead of the fast decoder; None: the setting
    model: str | None = None         # a video_gen.MODELS key: "turbo" (general) or "anime"


@app.post("/api/videos/generate")
def videos_generate(body: VideoRequest):
    """Videos' Make it: straight to the engine. Waits while a chat turn holds the GPU's model."""
    ref = None
    if body.ref_id:
        got = image_gen.find(body.ref_id)
        if not got:
            return {"ok": False, "message": "That picture is no longer in Images.", "code": "missing"}
        ref = os.path.join(got[1], f"{body.ref_id}.png")
    elif body.file_id:
        meta = sessions.file_store().get(body.file_id) or shared.file_store.get(body.file_id)
        if not meta or meta.get("kind") != "image":
            return {"ok": False, "message": "That picture is no longer attached.", "code": "missing"}
        ref = meta["path"]
    try:
        clip = video_gen.generate(body.prompt, body.size, body.length, body.seed, body.negative, ref_image=ref,
                                  ref_id=body.ref_id, best=body.best, wait_while=_gpu_wait,
                                  key=body.model if body.model in video_gen.MODELS else "turbo")
    except video_gen.VideoError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    event_log.log_action(f"Made a video ({clip['width']}x{clip['height']}, {clip['frames']} frames) in {clip['seconds']} s.")
    return {"ok": True, "video": clip}


@app.post("/api/videos/stop")
def videos_stop():
    return {"ok": True, "stopped": video_gen.request_stop() or video_audio.request_stop()}


class SoundRequest(BaseModel):
    prompt: str = ""                 # what it should sound like (empty: the clip's own prompt)
    dialogue: str = ""               # a spoken line (optional)
    language: str = "en"
    voice_start: float = 0.3         # seconds into the clip
    emotion: float = 0.5
    music_id: str | None = None      # a song from Music under everything (optional)
    seed: int | None = None


@app.get("/api/videos/sound/status")
def videos_sound_status():
    return video_audio.status()


@app.post("/api/videos/{vid}/sound")
def videos_add_sound(vid: str, body: SoundRequest):
    """Add sound (effects + an optional spoken line + optional music) - a new copy of the clip; the original stays."""
    try:
        clip = video_audio.add_sound(vid, body.prompt, body.dialogue, body.language, body.voice_start, body.emotion,
                                     body.music_id, seed=body.seed, wait_while=_gpu_wait)
    except video_audio.SoundError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    event_log.log_action(f"Added sound to the video {vid} in {clip['sound']['seconds']} s.")
    return {"ok": True, "video": clip}


class VideoAction(BaseModel):
    on: bool = True


@app.post("/api/videos/trash/empty")
def videos_empty_trash():
    done = video_gen.empty_trash()
    event_log.log_action(f"Emptied the video trash: {done['count']} videos, {done['freed'] / 1e6:.1f} MB freed.")
    return {"ok": True, **done}


def _video_action(fn, log: str | None = None):
    try:
        out = fn()
    except video_gen.VideoError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    if log:
        event_log.log_action(log)
    return {"ok": True, **({"video": out} if isinstance(out, dict) else {"freed": out} if isinstance(out, int) else {})}


@app.post("/api/videos/{vid}/fav")
def videos_fav(vid: str, body: VideoAction):
    return _video_action(lambda: video_gen.set_favorite(vid, body.on))


@app.post("/api/videos/{vid}/trash")
def videos_trash(vid: str):
    return _video_action(lambda: video_gen.trash(vid), f"Moved the video {vid} to the video trash.")


@app.post("/api/videos/{vid}/restore")
def videos_restore(vid: str):
    return _video_action(lambda: video_gen.restore(vid))


@app.post("/api/videos/{vid}/purge")
def videos_purge(vid: str):
    return _video_action(lambda: video_gen.purge(vid), f"Deleted the video {vid} for good from the trash.")


# ==============================================================================
# Music made on this PC (music_gen.py - ACE-Step 1.5)
# ==============================================================================
_MUSIC_NAME = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}\.flac")


def _music_file(folder: str, name: str):
    if not _MUSIC_NAME.fullmatch(name) or not os.path.exists(os.path.join(folder, name)):
        return JSONResponse({"ok": False, "message": "No such song."}, status_code=404)
    return FileResponse(os.path.join(folder, name), media_type="audio/flac", headers={"Cache-Control": "max-age=31536000, immutable"})


@app.get("/api/music")
def music_list(q: str = "", fav: bool = False, trash: bool = False, limit: int = 200):
    return {"status": music_gen.status(), "songs": music_gen.recent(max(1, min(500, limit)), q[:200], fav, trash)}


@app.get("/api/music/status")
def music_status():
    return music_gen.status()


@app.get("/api/music/trash/{name}")
def music_trash_file(name: str):
    return _music_file(os.path.join(music_gen.MUSIC_DIR, "trash"), name)


@app.get("/api/music/{name}")
def music_file(name: str):
    return _music_file(music_gen.MUSIC_DIR, name)


class MusicRequest(BaseModel):
    prompt: str = ""
    lyrics: str = ""
    duration: float = 60
    instrumental: bool = False
    seed: int | None = None
    language: str = "en"
    bpm: int | None = None
    key: str = ""
    time_signature: str = ""
    title: str = ""
    edit: dict | None = None              # {"src", "start", "end", "change", "strength"}: remake that part as a new version


@app.post("/api/music/generate")
def music_generate(body: MusicRequest):
    try:
        song = music_gen.generate(body.prompt, body.lyrics, body.duration, body.instrumental, body.seed, body.language,
                                  body.bpm, wait_while=_gpu_wait, key=body.key,
                                  time_signature=body.time_signature, title=body.title, edit=body.edit)
    except music_gen.MusicError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    except (OSError, ValueError, TypeError) as e:
        return {"ok": False, "message": f"That didn't work: {str(e)[:200]}", "code": "failed"}
    event_log.log_action(f"{'Changed part of a song' if body.edit else 'Made a song'} ({song['duration']} s) in {song['seconds']} s.")
    return {"ok": True, "song": song}


class MusicTitle(BaseModel):
    title: str = ""


@app.post("/api/music/{mid}/title")
def music_title(mid: str, body: MusicTitle):
    return _music_action(lambda: music_gen.set_title(mid, body.title))


@app.post("/api/music/stop")
def music_stop():
    return {"ok": True, "stopped": music_gen.request_stop()}


@app.post("/api/music/unload")
def music_unload():
    music_gen.stop()
    return {"ok": True}


@app.post("/api/music/trash/empty")
def music_empty_trash():
    done = music_gen.empty_trash()
    event_log.log_action(f"Emptied the music trash: {done['count']} songs.")
    return {"ok": True, **done}


def _music_action(fn, log: str | None = None):
    try:
        out = fn()
    except music_gen.MusicError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    if log:
        event_log.log_action(log)
    return {"ok": True, **({"song": out} if isinstance(out, dict) else {})}


@app.post("/api/music/{mid}/fav")
def music_fav(mid: str, body: VideoAction):
    return _music_action(lambda: music_gen.set_favorite(mid, body.on))


@app.post("/api/music/{mid}/trash")
def music_trash(mid: str):
    return _music_action(lambda: music_gen.trash(mid), f"Moved the song {mid} to the music trash.")


@app.post("/api/music/{mid}/restore")
def music_restore(mid: str):
    return _music_action(lambda: music_gen.restore(mid))


@app.post("/api/music/{mid}/purge")
def music_purge(mid: str):
    return _music_action(lambda: music_gen.purge(mid), f"Deleted the song {mid} for good from the trash.")


# ==============================================================================
# Fx Labs (labs.py): train models on this PC - the engine, datasets, runs
# ==============================================================================
def _labs(fn, log: str | None = None):
    try:
        out = fn()
    except labs.LabsError as e:
        return {"ok": False, "message": str(e), "code": e.code}
    except OSError as e:
        return {"ok": False, "message": f"That didn't work: {e.strerror or e}", "code": "failed"}
    if log:
        event_log.log_action(log)
    return {"ok": True, **(out if isinstance(out, dict) else {"items": out} if isinstance(out, list) else {})}


class LabsDatasetBody(BaseModel):
    kind: str                         # text | folder | sample_text | sample_shapes
    name: str = ""
    text: str = ""
    path: str = ""


class LabsRunBody(BaseModel):
    template: str
    dataset: str
    size: str = "small"
    name: str = ""
    steps: int | None = None
    settings: dict | None = None


class LabsRunEdit(BaseModel):
    name: str | None = None
    notes: str | None = None
    more_steps: int = 0


class LabsTryBody(BaseModel):
    prompt: str = ""
    length: int = 400
    temperature: float = 0.8


@app.get("/api/labs")
def labs_status():
    return labs.status()


@app.post("/api/labs/setup")
def labs_setup():
    event_log.log_action("Labs: setting up its engine (PyTorch).")
    return {"ok": True, "setup": labs.setup()}


@app.get("/api/labs/datasets")
def labs_datasets():
    return {"datasets": labs.datasets()}


@app.post("/api/labs/datasets")
def labs_add_dataset(body: LabsDatasetBody):
    fn = {"text": lambda: labs.add_text(body.name, body.text), "folder": lambda: labs.add_folder(body.name, body.path),
          "chat": lambda: labs.add_chat(body.name, body.text),
          "sample_text": labs.add_sample_text, "sample_shapes": labs.add_sample_shapes,
          "sample_chat": labs.add_sample_chat}.get(body.kind)
    if not fn:
        return {"ok": False, "message": "Unknown kind of dataset."}

    def go():
        try:
            return {"dataset": fn()}
        except httpx.HTTPError as e:
            raise labs.LabsError(f"The download didn't work: {e}", "failed") from None
    return _labs(go, f"Labs: added a dataset ({body.kind}).")


@app.post("/api/labs/datasets/{did}/delete")
def labs_delete_dataset(did: str):
    return _labs(lambda: labs.delete_dataset(did), f"Labs: dataset {did} moved to the Labs trash.")


@app.get("/api/labs/runs")
def labs_runs():
    return {"runs": labs.runs(), "running": labs.status()["running"]}


@app.post("/api/labs/runs")
def labs_create_run(body: LabsRunBody):
    return _labs(lambda: {"run": labs.create(body.template, body.dataset, body.size, body.name, body.steps, body.settings)},
                 f"Labs: started a {body.template} run.")


@app.get("/api/labs/runs/{rid}")
def labs_run(rid: str, since: int = 0):
    return _labs(lambda: labs.metrics(rid, max(0, since)))


@app.post("/api/labs/runs/{rid}")
def labs_edit_run(rid: str, body: LabsRunEdit):
    return _labs(lambda: {"run": labs.update(rid, body.name, body.notes)})


@app.post("/api/labs/runs/{rid}/stop")
def labs_stop_run(rid: str):
    return _labs(lambda: {"run": labs.stop(rid)}, f"Labs: asked run {rid} to stop.")


@app.post("/api/labs/runs/{rid}/resume")
def labs_resume_run(rid: str, body: LabsRunEdit = None):
    return _labs(lambda: {"run": labs.start(rid, (body.more_steps if body else 0))}, f"Labs: resumed run {rid}.")


@app.post("/api/labs/runs/{rid}/delete")
def labs_delete_run(rid: str):
    return _labs(lambda: labs.delete_run(rid), f"Labs: run {rid} moved to the Labs trash.")


class LabsTraceBody(BaseModel):
    prompt: str = ""
    picture: str = ""


@app.post("/api/labs/runs/{rid}/trace")
def labs_trace(rid: str, body: LabsTraceBody):
    """One real pass through the checkpoint - what each layer did with this input."""
    return _labs(lambda: labs.trace(rid, body.prompt, body.picture))


@app.post("/api/labs/runs/{rid}/try")
def labs_try(rid: str, body: LabsTryBody):
    return _labs(lambda: labs.try_it(rid, body.prompt, body.length, body.temperature))


class LabsModelBody(BaseModel):
    template: str
    name: str = ""
    dataset: str | None = None
    size: str | None = None


class LabsDraftEdit(BaseModel):
    name: str | None = None
    dataset: str | None = None
    size: str | None = None
    steps: int | None = None
    settings: dict | None = None


@app.get("/api/labs/models")
def labs_models():
    """Every model with its versions (runs) - Labs' main list."""
    return {"models": labs.models(), "running": labs.status()["running"]}


@app.post("/api/labs/models")
def labs_new_model(body: LabsModelBody):
    return _labs(lambda: {"run": labs.new_model(body.template, body.name, body.dataset, body.size)}, "Labs: new model.")


@app.post("/api/labs/models/{mid}/delete")
def labs_delete_model(mid: str):
    return _labs(lambda: {"count": labs.delete_model(mid)}, f"Labs: model {mid} moved to the Labs trash.")


@app.post("/api/labs/runs/{rid}/draft")
def labs_edit_draft(rid: str, body: LabsDraftEdit):
    """The Data and Design steps: change a version that hasn't trained yet (checked by labs.validate_settings)."""
    return _labs(lambda: {"run": labs.update_draft(rid, body.model_dump(exclude_none=True))})


@app.post("/api/labs/runs/{rid}/start")
def labs_start(rid: str):
    return _labs(lambda: {"run": labs.start(rid)}, f"Labs: started run {rid}.")


@app.post("/api/labs/runs/{rid}/version")
def labs_new_version(rid: str):
    """The next version of a model, continuing from this one's weights."""
    return _labs(lambda: {"run": labs.new_version(rid)}, f"Labs: new version from {rid}.")


@app.post("/api/labs/build/cancel")
def labs_build_cancel():
    return {"ok": True, "build": labs.build_cancel()}


@app.post("/api/labs/runs/{rid}/port")
def labs_port(rid: str):
    """Finalize a fine-tune into a Workspace model (merge, GGUF, quantize, models.ini)."""
    return _labs(lambda: {"port": labs.port(rid)}, f"Labs: porting run {rid} to Workspace.")


class LabsBuildBody(BaseModel):
    name: str = ""
    docs: list = []                   # [{"name", "data": base64}]
    per_section: int = 4
    model: str | None = None


class LabsDraftBody(BaseModel):
    name: str = ""
    pairs: list = []


@app.post("/api/labs/build")
def labs_build(body: LabsBuildBody):
    return _labs(lambda: {"build": labs.build_start(body.name, body.docs, body.per_section, body.model)},
                 f"Labs: building training examples from {len(body.docs)} document(s).")


@app.get("/api/labs/drafts")
def labs_drafts():
    return {"drafts": labs.drafts(), "build": labs.build_status()}


@app.get("/api/labs/drafts/{bid}")
def labs_draft(bid: str):
    return _labs(lambda: {"draft": labs.draft(bid)})


@app.post("/api/labs/drafts/{bid}/save")
def labs_draft_save(bid: str, body: LabsDraftBody):
    return _labs(lambda: {"dataset": labs.draft_save(bid, body.name, body.pairs)}, "Labs: saved reviewed examples as a dataset.")


@app.post("/api/labs/drafts/{bid}/delete")
def labs_draft_delete(bid: str):
    return _labs(lambda: labs.draft_delete(bid))


@app.get("/api/labs/runs/{rid}/samples/{name}")
def labs_sample_file(rid: str, name: str):
    try:
        path = labs.sample_file(rid, name)
    except labs.LabsError:
        path = None
    if not path:
        return JSONResponse({"ok": False, "message": "No such picture."}, status_code=404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


# ==============================================================================
# MossadGPT's world monitor (osint/): the globe's data. Nothing here fetches from
# outside by itself - the collector does, while the page's heartbeat says MossadGPT is on.
# ==============================================================================
@app.post("/api/osint/beat")
def osint_beat():
    """MossadGPT is on screen: collect for three more minutes (osint.collector)."""
    return osint_collector.beat()


@app.get("/api/osint/status")
def osint_status():
    return osint_collector.status()


@app.get("/api/osint/basemap")
def osint_basemap():
    """The globe's land, lakes and borders (geo/world.json - Natural Earth, public domain)."""
    return FileResponse(osint_geo.WORLD_PATH, media_type="application/json", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/osint/place")
def osint_place(q: str = ""):
    p = osint_geo.place(q)
    return {"ok": bool(p), "place": p}


class OsintMentions(BaseModel):
    text: str = ""


@app.post("/api/osint/mentions")
def osint_mentions(body: OsintMentions):
    """What a sentence names that the globe can show (voice mode follows the reply as it is spoken)."""
    import osint.tool as osint_tool
    return {"items": osint_tool.mentions(body.text[:2000])}


_MOVERS = ("flights", "military", "flights_global")
_EVENT_LAYERS = ("quakes", "news", "space", "alerts")


def _mover_items(name: str, rows: list) -> list:
    return [[r["id"], r["lat"], r["lon"], r["alt"], r["hdg"], r["spd"], r["label"],
             ("m" if r["props"].get("mil") or name == "military" else "") + ("g" if r["props"].get("ground") else "")]
            for r in rows]


@app.get("/api/osint/replay")
def osint_replay(t: float, layers: str = "flights,military"):
    """Rewind: where the movers were at time t, from the one-a-minute history (store.at) - as the live layers."""
    names = [n for n in layers.split(",") if n in _MOVERS]
    oldest, newest = osint_store.history_range()
    by = {n: [] for n in names}
    for r in osint_store.at(names, t) if names else []:
        by[r["layer"]].append(r)
    return {"t": t, "oldest": oldest, "newest": newest, "layers": {n: {"t": t, "items": _mover_items(n, rows)} for n, rows in by.items()}}


@app.get("/api/osint/layer/{name}")
def osint_layer(name: str, hours: float = 24, dt: float = 60):
    """One layer, compact: movers as [id, lat, lon, alt m, heading, km/h, label, flags]; satellites as
    [id, name, group, lat, lon, alt km, lat+dt, lon+dt] (the page moves them between the two); events as
    [uid, lat, lon, t, kind, title, url, size]; the aurora oval as [lon, lat, %]."""
    now = time.time()
    if name in _MOVERS:
        return {"t": now, "items": _mover_items(name, osint_store.current(name))}
    if name == "jamming":                      # [lat, lon, share degraded, aircraft, level] per half-degree cell
        import osint.intel as osint_intel
        return {"t": now, "cell": osint_intel.CELL,
                "items": [[c["lat"], c["lon"], c["pct"], c["n"], c["level"]] for c in osint_intel.jamming()]}
    if name == "satellites":
        a = osint_satellites.positions(now)
        b = {s["id"]: s for s in osint_satellites.positions(now + dt)}
        return {"t": now, "dt": dt, "items": [[s["id"], s["name"], s["group"], s["lat"], s["lon"], s["alt_km"],
                                                b.get(s["id"], s)["lat"], b.get(s["id"], s)["lon"]] for s in a]}
    if name == "aurora":
        return {"t": now, "items": osint_collector.aurora()}
    if name in _EVENT_LAYERS:
        evs = osint_store.events(name, now - max(1.0, min(hours, 24 * 30)) * 3600, limit=5000)
        size = ((lambda e: e["props"].get("mag", 1)) if name == "quakes" else (lambda e: e["props"].get("sev", 1)) if name == "alerts"
                else (lambda e: e["props"].get("sources", 1)))
        return {"t": now, "items": [[e["uid"], e["lat"], e["lon"], e["t"], e["kind"], e["title"], e["url"], size(e)]
                                    for e in evs if e["lat"] is not None]}
    return JSONResponse({"ok": False, "message": f"No layer {name}."}, status_code=404)


# ==============================================================================
# Static frontend
# ==============================================================================
# StaticFiles refuses to mount a directory that does not exist, and the old
# build only ever created an empty one - index.html sat next to main.py and was
# never copied in, so the UI loaded as a 404. Handle it here rather than
# relying on the launcher script.
app.include_router(roblox_bridge.router)


@app.get("/", include_in_schema=False)
def index_page():
    """Serve the UI straight from index.html (no stale static/ copy, no
    restart needed after editing it)."""
    return FileResponse(os.path.join(config.BASE_DIR, "index.html"),
                        headers={"Cache-Control": "no-store"})


_static_dir = os.path.join(config.BASE_DIR, "static")
os.makedirs(_static_dir, exist_ok=True)
for _name in ("index.html",):
    _src, _dst = os.path.join(config.BASE_DIR, _name), os.path.join(_static_dir, _name)
    if os.path.exists(_src) and (not os.path.exists(_dst) or os.path.getmtime(_src) > os.path.getmtime(_dst)):
        try:
            shutil.copyfile(_src, _dst)
            print(f"[INIT] Published {_name} to static/")
        except Exception as _e:
            print(f"[WARN] Could not copy {_name} into static/: {_e}")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
