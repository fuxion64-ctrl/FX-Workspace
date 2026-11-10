"""
agent/models.py
===============
Model calls: history fitting, local (streamed, constrained) and cloud completions, retries.
(Split out of main.py in Phase 2 - see ARCHITECTURE.md D14.)
"""
from __future__ import annotations

import local_runtime
import base64
import composer_tools
import json
import os
import re
import netlayer
import threading
import time
from io import BytesIO
import context as ctx
import elements
import llama_builds
import llama_local
import load_failure
import history
import llm_compat
import llm_stream
import model_caps
import model_metrics
import model_monitor
import model_router
import model_output
import system_messages
import prompt_prep
import recorder
import stream_hub
from run_control import RunStopped, run_control
import system_metrics

import agent.parsing as parsing
import agent.prompts as prompts
import core.config as config
import core.observer as observer
import core.state as shared
import events as event_log
import tools.schemas as schemas


def is_local_engine() -> bool:
    return shared.app_state.get("mode") in (config.MODE_LOCAL, config.MODE_AUTO)


def encode_image(img, for_local: bool = False, quality: int = None) -> str:
    """JPEG-encode, clamping dimensions to what the target engine can actually
    afford. A local 8K-context model cannot hold a full-resolution capture."""
    max_w, max_h = (config.SHOT_SIZE_LOCAL if for_local else config.SHOT_SIZE_CLOUD)
    if img.width > max_w or img.height > max_h:
        img = img.copy()
        img.thumbnail((max_w, max_h))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality or (60 if for_local else 75))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def shot_limits() -> tuple:
    return config.SHOT_SIZE_LOCAL if shared.app_state.get("mode") == config.MODE_LOCAL else config.SHOT_SIZE_CLOUD


# ==============================================================================
# History management
# ==============================================================================
# The budgeting rules live in history.py (pure functions, tested alone); these
# wrappers apply them to the live conversation in app_state.
_approx_chars = history.approx_chars
_approx_tokens = history.approx_tokens
has_images = history.has_images
strip_images = history.strip_images


# Kept under its original name; the old call sites expected this behaviour.
sanitize_messages_for_text_model = strip_images


def demote_old_images(keep: int):
    history.demote_old_images(shared.app_state["conversation"], keep)


def trim_conversation(budget_tokens: int = None):
    """Fit the conversation into the budget: whole old turns first, then
    older tool output inside the current task (see history.fit)."""
    as_tokens = bool(budget_tokens)
    limit = budget_tokens if as_tokens else config.MAX_HISTORY_CHARS
    convo, notes = history.fit(shared.app_state["conversation"], limit, as_tokens, config.MIN_KEPT_TURNS)
    shared.app_state["conversation"] = convo
    for note in notes:
        event_log.log_action(note)


# What a turn without write tools tells the model, beside CURRENT TOOL MODE. Without it a model given
# no tools for "stack 10 parts" told the user it was "only a text assistant" (2026-09-29) - untrue, and
# it then kept saying so once the tools were there.
_MODE_NOTES = {
    # "No computer tools" alone read to small models as "no tools": they stopped calling web_search and
    # world_monitor on questions and made the answers up (2026-09-29).
    "none": ("no tools that control this computer for this message (it reads as conversation or a question) - any web "
             "or data tools you are given still work: use them for facts you don't know. If the user wants you to do "
             "something, say you can do it as soon as they ask for it directly - never say you are only a text assistant."),
    "read": ("only tools that look are offered - nothing that changes things. If the user wants a change, say you'll "
             "make it as soon as they ask for it directly (\"add ...\", \"stack ...\") - never say you can only read."),
}


def build_messages(base_prompt: str = None):
    """conversation[0] is a placeholder; the message list actually sent is
    assembled here with a freshly probed live-state block, so the agent's view
    of the machine, its plan and its own history can never be stale - and can
    never be trimmed away, because it is not part of history.

    WHERE the block goes differs by engine, and it matters:

      cloud - merged into the system message. Simple, and a cloud provider
              re-reads the whole prompt every call anyway.

      local - appended as the FINAL message. llama.cpp caches the KV state of
              the longest unchanged prefix of the prompt. The live block
              changes every single turn, so merging it into the system message
              puts volatile text at position zero and invalidates the entire
              cache on every call - the model reprocesses the whole
              conversation from scratch each step, which is a large part of why
              a local run stalls out before finishing a multi-step task. Last
              position keeps the whole history cacheable, and has the side
              benefit of putting the current state closest to the generation
              point, where small models actually attend to it.
    """
    convo = shared.app_state["conversation"]
    local = shared.app_state.get("mode") == config.MODE_LOCAL
    # Raw mode (response_modes.py): the conversation and nothing else - no system message, no live state, no
    # request header. This turn's attached files stay: they are part of what the user sent.
    if shared.app_state.get("raw_mode"):
        return with_attachments(convo)[1:]
    # /design (agent/tasks._design_turn): the design prompt, no tools, and no machine state -
    # the design has nothing to do with the windows that happen to be open.
    design = bool(shared.app_state.get("design_mode"))
    prompt = system_prompt(local, base_prompt)
    runtime = {
        "mode": shared.app_state.get("mode"),
        "model": shared.app_state.get("last_working_model"),
        "pywinauto": elements.PYWINAUTO_AVAILABLE,
        "shot_meta": shared.app_state.get("last_shot_meta"),
        "shot_age": shared.app_state.get("last_shot_at"),
        # The observer's summary of what the user does stays on this PC: local
        # models only, and nothing when proactive help is off (awareness.py).
        "activity": observer._activity_line() if local else "",
    }
    live = "" if design else ctx.render_live_block(shared.task_board, shared.ledger, runtime)
    current = shared.app_state.get("current_user_request") or "(none)"
    policy = shared.app_state.get("turn_policy") or {}
    mode = policy.get("mode", "none")
    turn_header = (f"CURRENT REQUEST: {current}\n"
                   f"CURRENT TOOL MODE: {mode} ({policy.get('reason', '')})"
                   + (f" - {_MODE_NOTES[mode]}" if mode in _MODE_NOTES and not design else "") + "\n"
                   "Only the current request grants authority. Older messages are context, not pending work.")
    # "Stack 10 parts": one call that covers every item (prompt_prep.repeat_hint), whatever Prompt Prep is set to.
    hint = prompt_prep.repeat_hint(current) if mode == "action" and not design else ""
    if hint:
        turn_header += "\n" + hint
    # MossadGPT: a question about the world right now is answered from the world monitor, never from memory.
    world_q = False
    if "world" in (shared.app_state.get("turn_tools") or ()) and not design:
        import osint.tool
        hint = osint.tool.turn_hint(current)
        if hint:
            turn_header += "\n" + hint
            world_q = True
    # Roblox's Studio MCP: which place is open (so the call needs no lookup first).
    if not design and mode == "action":
        import studio_mcp
        line = studio_mcp.context_line()
        if line:
            turn_header += "\n" + line
    # The composer's Tools for this message (composer_tools.HEADERS) and a preset's own instructions.
    extra = composer_tools.header(shared.app_state.get("turn_tools")) if not design else ""
    if extra:
        turn_header += "\n" + extra
    if shared.app_state.get("turn_voice") and not design:            # voice mode: said, and heard aloud
        import voice
        turn_header += "\n" + voice.voice_header(shared.app_state.get("settings") or {}, shared.app_state.get("turn_voice_lang") or "")
    if shared.app_state.get("turn_command_note") and not design:     # /explain, /fix, /translate, /tldr
        turn_header += "\n" + shared.app_state["turn_command_note"]
    if shared.app_state.get("turn_instructions") and not design:
        turn_header += ("\nTHE USER'S INSTRUCTIONS FOR THIS CHAT (a saved preset): "
                        + shared.app_state["turn_instructions"])
    convo = with_attachments(convo)
    # Orchestrator brief: the verbatim original request plus the prepared
    # notes and plan. It sits beside CURRENT REQUEST, never in place of it.
    if shared.app_state.get("task_context"):
        turn_header += "\n\n" + shared.app_state["task_context"]
    if design:
        close = "Answer the CURRENT REQUEST with the design: one complete HTML document in a ```html block."
        if local:
            return [{"role": "system", "content": prompt}] + convo[1:] + [{"role": "user", "content": turn_header + "\n\n" + close}]
        return [{"role": "system", "content": f"{prompt}\n\n{turn_header}\n\n{close}"}] + convo[1:]
    # Skills not loaded yet, one line each (skills.py) - what load_skill can still add.
    import skills
    more = skills.index(t["function"]["name"] for t in schemas.tools_for_current_turn(local))
    if local:
        offered = schemas.tools_for_current_turn(local=True)
        tool_note = ""
        if offered:
            tool_note = "\nAVAILABLE TOOLS THIS TURN: " + ", ".join(t["function"]["name"] for t in offered) + "."
            tool_note += (" These are callable tools, not examples. Use their exact names. "
                          "Connected MCP tools are available alongside built-in tools.")
        if more:
            tool_note += "\n" + more
        instruction = "Answer the CURRENT REQUEST. Use one offered tool only when it is necessary; otherwise answer directly (Markdown is rendered)."
        # The last line is the one a small model follows best (see world_q below): the ground rules ride along.
        speak_up = " If a ground rule stops you or something can't be done, say so first in a > **Warning:** line."
        if world_q and policy.get("mode") != "action" and any(t["function"]["name"] == "world_monitor" for t in offered):
            # The last line is the one a small model follows: "answer directly" had it make up passes and events.
            instruction = ("Call world_monitor now - one call for each part of the CURRENT REQUEST - and answer only from "
                           "what it returns (Markdown is rendered).")
            speak_up = ""                       # this line must stay last (test_osint.NudgeTests)
        elif policy.get("mode") == "action" and offered:
            instruction = ("Carry out the CURRENT REQUEST using the offered tools. Call the appropriate tool now "
                           "when the target is clear; ask a question only if required information is missing. "
                           "Tool calls go through the application's permission checks and approval UI before execution, "
                           "so do not add a separate 'shall I proceed?' question. Wait for tool results before claiming success.")
        return ([{"role": "system", "content": prompt}] + convo[1:]
                + [{"role": "user", "content": turn_header + "\n\n" + live +
                    tool_note + "\n\n" + instruction + speak_up}])
    return [{"role": "system", "content": f"{prompt}\n\n{turn_header}\n\n{live}" + (f"\n\n{more}" if more else "")}] + convo[1:]


def system_prompt(local: bool, base_prompt: str = None) -> str:
    """The system message as sent: the base prompt fitted to this turn's tools, the Studio rules, the
    persona and the compressed earlier turns. History fitting (_fit_for_local) measures this same text - it
    measured the bare base prompt, and the persona and careful rules on top of it pushed a 16k model 553 tokens
    over its window (2026-10-01). The honesty rules (careful.PROMPT) are part of the base prompt's Enhanced core
    since D47; Raw mode has no system message at all ("")."""
    if shared.app_state.get("raw_mode"):
        return ""
    if shared.app_state.get("design_mode"):
        prompt = prompts.DESIGN_PROMPT
    else:
        selected = shared.app_state.get("turn_system_message")
        if selected is None:
            selected = shared.app_state.get("settings")
        selected = system_messages.clean_settings(selected)
        prompt = system_messages.resolve(selected, local, base_prompt)
        offered = [t["function"]["name"] for t in schemas.tools_for_current_turn(local)]
        # Built-in instructions omit unavailable tools (D36); user-written restrictions stay intact.
        if selected["system_message_variant"] != "custom":
            prompt = prompts.for_tools(prompt, offered, schemas.TOOL_NAMES)
        # How to use the Studio tools that are there (Roblox's Studio MCP, the bridge's batch tool).
        studio = prompts.studio_rules(offered)
        if studio:
            prompt = f"{prompt}\n\n{studio}"
        # The composer's persona (personas.py) is part of the system prompt: it's the same all chat long,
        # so llama.cpp's prompt cache survives until it's changed. It comes after the Enhanced core, and its own
        # heading says it never overrides the ground rules.
        # The assistant's identity file (identity.py), with the header that says it defines the assistant itself.
        import identity
        soul = identity.block()
        if soul:
            prompt = f"{prompt}\n\n{soul}"
        # What it remembers about, one line per topic (memory_graph.py) - not the memories themselves.
        import memory_graph
        overview = memory_graph.overview_block()
        if overview:
            prompt = f"{prompt}\n\n{overview}"
        if shared.app_state.get("turn_persona"):
            import personas
            prompt = f"{prompt}\n\n{personas.block(shared.app_state['turn_persona'])}"
    # Compressed older turns (agent/context_window.py) sit beside the system
    # prompt: a message of their own would break templates that require
    # user/assistant alternation, and this prefix changes only when the chat
    # is compressed, so llama.cpp's prompt cache survives.
    if shared.app_state.get("context_summary"):
        import agent.context_window as context_window       # imports this module; resolved at call time
        prompt = f"{prompt}\n\n{context_window.summary_block(shared.app_state['context_summary'])}"
    return prompt


def _turn_tokens() -> int:
    """What the final user message adds besides the live block: the request, the orchestrator brief,
    the composer's tool notes, a preset's instructions and this turn's attached files."""
    s = shared.app_state
    parts = [s.get("current_user_request"), s.get("task_context"), s.get("turn_instructions"), s.get("turn_command_note"),
             composer_tools.header(s.get("turn_tools")), (s.get("turn_attachments") or {}).get("text")]
    return int(sum(len(p) for p in parts if isinstance(p, str)) / history.CHARS_PER_TOKEN) + 400   # + the fixed lines


def with_attachments(convo: list) -> list:
    """This turn's attached files (composer_tools.file_context), put in front of
    the request they came with - the first user message of the turn - for this
    call only; agent/tasks._run_turn keeps them there afterwards."""
    att = shared.app_state.get("turn_attachments")
    if not att:
        return convo
    start = att.get("start", 0)
    i = next((k for k in range(max(1, start), len(convo)) if convo[k].get("role") == "user"
              and isinstance(convo[k].get("content"), str)), None)
    if i is None:
        return convo
    convo = list(convo)
    convo[i] = attach_to(convo[i], att)
    return convo


def attach_to(message: dict, att: dict) -> dict:
    text = f"{att['text']}\n\n[THE USER'S MESSAGE]\n{message['content']}"
    if not att.get("images"):
        return {**message, "content": text}
    return {**message, "content": [{"type": "text", "text": text}]
            + [{"type": "image_url", "image_url": {"url": u}} for u in att["images"]]}


# ==============================================================================
# Model calling
# ==============================================================================
def get_keys_for_model(model_name: str) -> list:
    if model_name.startswith("gemini/"):
        env = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
    elif model_name.startswith("groq/"):
        env = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
    else:
        env = os.environ.get("OPENAI_API_KEY", "")
    return [k.strip() for k in env.split(",") if k.strip() and not k.strip().upper().startswith("YOUR_")]


def favorite_cloud_models(requires_vision: bool) -> list:
    key = "favorite_cloud_vision_models" if requires_vision else "favorite_cloud_text_models"
    return shared.app_state["settings"].get(key) or []


def candidate_pairs(requires_vision: bool):
    names = config.VISION_MODELS if requires_vision else config.PRIMARY_MODELS
    favorites = favorite_cloud_models(requires_vision)
    if favorites:
        names = parsing._restrict_to_favorites(names, favorites)
    return [(m, k) for m in names for k in get_keys_for_model(m)]


def call_cloud(messages, requires_vision: bool):
    if requires_vision and not shared.app_state["settings"].get("cloud_allow_screenshots"):
        # Privacy: screenshots stay on this machine unless the user opted in.
        event_log.log_action("Cloud call: screenshots withheld (Settings > Privacy > allow screenshots to cloud is off).")
        messages, requires_vision = strip_images(messages), False
    mode_key = "vision" if requires_vision else "text"
    all_pairs = candidate_pairs(requires_vision)
    if not all_pairs:
        kind = "vision" if requires_vision else "text"
        favorites = favorite_cloud_models(requires_vision)
        if favorites:
            raise RuntimeError(f"No API keys configured for any of your favorite {kind} models "
                               f"({', '.join(favorites)}). Add a key for one of them, or clear that "
                               f"favorites list in Settings to allow the full {kind} model list again.")
        raise RuntimeError(f"No API keys configured for any {kind} model.")
    now = time.time()
    sticky = shared.LAST_GOOD.get(mode_key)
    ordered = ([sticky] if sticky in all_pairs else []) + [p for p in all_pairs if p != sticky]
    preferred = shared.app_state.get("preferred_cloud_model")          # set by a model handoff
    if preferred:
        ordered = [p for p in ordered if p[0] == preferred] + [p for p in ordered if p[0] != preferred]
    fresh = [p for p in ordered if shared._cooldown_until.get(p, 0) <= now]
    attempt_order = fresh or ordered
    last_error = None
    for round_ in range(2):
        if round_:
            # Every model failed for a reason that passes (rate limit, timeout, overload):
            # one more round after a pause the user can see - and stop.
            if not _TRANSIENT.search(str(last_error or "")):
                break
            event_log.emit_event("note", "Cloud models didn't answer - retrying in 3 s…", status="done",
                                 detail=str(last_error)[:200])
            for _ in range(30):
                run_control.checkpoint()
                time.sleep(0.1)
            attempt_order = ordered
        result = _try_cloud_pairs(attempt_order, messages, mode_key, requires_vision)
        if result[0] is not None:
            return result[0]
        last_error = result[1]
    raise RuntimeError(f"All cloud models failed. Error: {str(last_error)[:200]}")


_TRANSIENT = re.compile(r"\b(429|500|502|503|504)\b|rate.?limit|timed? ?out|timeout|overloaded|unavailable|connection|temporar", re.I)


def _try_cloud_pairs(attempt_order, messages, mode_key: str, requires_vision: bool):
    """(response, None) from the first (model, key) that answers, or (None, the last error)."""
    last_error = None
    now = time.time()
    for model_name, key in attempt_order:
        try:
            event_log.log_action(f"Trying {model_name}...")
            supports_images = model_name.startswith("gemini/") or (requires_vision and model_name.startswith("groq/"))
            msgs = messages if supports_images else strip_images(messages)
            kwargs = {"model": model_name, "api_key": key, "messages": msgs, "timeout": 45}
            if shared.app_state.get("effort_native"):
                # The provider's own reasoning effort (LiteLLM maps it per
                # provider and drops it where unsupported - drop_params).
                kwargs["reasoning_effort"] = shared.app_state["effort_native"]
            turn_tools = schemas.tools_for_current_turn(local=False)
            if turn_tools:
                kwargs.update(tools=turn_tools, tool_choice="auto")
            res = llm_compat.completion(**kwargs)
            shared.LAST_GOOD[mode_key] = (model_name, key)
            shared._cooldown_until.pop((model_name, key), None)
            shared.app_state["last_working_model"] = model_name
            event_log.log_action(f"Success with {model_name}")
            return res, None
        except Exception as e:
            last_error = e
            event_log.log_action(f"{model_name} failed: {str(e)[:120]}")
            shared._cooldown_until[(model_name, key)] = now + shared.COOLDOWN_SECONDS
    return None, last_error


def resolve_local_model(requires_vision: bool) -> str:
    """Pick the ONE local model to use, honouring two rules.

    Rule 1 - one model at a time. The router runs with --models-max 1, so every
    change of model unloads the current weights and loads new ones from disk.
    The old code treated the model list as a fallback chain and would happily
    walk down it mid-task, paying a multi-second reload each step and losing
    llama.cpp's KV cache every time. A single choice is pinned instead.

    Rule 2 - escalation is one-way: text -> vision, never back. A vision GGUF
    answers text-only prompts perfectly well, so once a screenshot has forced
    the swap there is no reason to ever swap back, and doing so would mean two
    reloads per screenshot for the rest of the session.
    """
    engaged = shared.app_state.get("local_engaged")
    pinned = (shared.app_state.get("local_model") or "").strip()

    if engaged:
        if llama_local.is_vision_model(engaged):
            return engaged                      # vision handles everything
        if not requires_vision:
            return engaged                      # stay put
        # text engaged, vision needed -> fall through and escalate, once.

    if pinned:
        if not requires_vision or llama_local.is_vision_model(pinned):
            return pinned
        if not shared.app_state["settings"].get("auto_model_routing"):
            # Routing off: the user's chosen model is never swapped out. The
            # screenshot is withheld instead (call_local strips images for a
            # text-only model) and the activity timeline says so.
            if shared.app_state.get("_vision_withheld_note") != pinned:
                shared.app_state["_vision_withheld_note"] = pinned
                event_log.emit_activity("vision_withheld", f"{pinned} can't read images - screenshot not sent (automatic routing is off)",
                              level="warn")
            return pinned
        event_log.log_action(f"'{pinned}' is text-only and this turn has a screenshot - escalating to a vision model.")

    kind = "vision" if requires_vision else "text"
    configured = config.LOCAL_VISION_MODELS if requires_vision else config.LOCAL_TEXT_MODELS
    candidates, _ = llama_local.resolve_candidates(configured, kind=kind)

    fav_key = "favorite_local_vision_models" if requires_vision else "favorite_local_text_models"
    favorites = shared.app_state["settings"].get(fav_key) or []
    if favorites:
        restricted = parsing._restrict_to_favorites(candidates, favorites)
        if not restricted:
            event_log.log_action(f"None of your favorite {kind} models ({', '.join(favorites)}) are currently "
                       f"reported by llama-server; not falling back to the full list. Adjust your "
                       f"favorites in Settings, or check models.ini / that the model is loaded.")
        candidates = restricted

    # A model whose benchmark showed unreliable tool calls is passed over when another
    # candidate exists (measured, model_router.MIN_HANDOFF_VALIDITY) - never picked by name.
    def reliable(name):
        v = model_caps.get(name).get("tool_call_validity")
        return not (isinstance(v, (int, float)) and v < model_router.MIN_HANDOFF_VALIDITY)
    candidates = [c for c in candidates if reliable(c)] or candidates
    return candidates[0] if candidates else None


def engage_local_model(name: str):
    """Record the model actually in use and note a swap, so the reload cost is
    visible in the audit log rather than looking like a random stall."""
    previous = shared.app_state.get("local_engaged")
    if previous and previous != name:
        # Only a genuine swap unloads weights and changes what the router
        # reports. Invalidating on first engagement would throw away a
        # perfectly good model list for nothing.
        event_log.log_action(f"Local model swap: {previous} -> {name} (router reloads weights; expect a pause).")
        llama_local.invalidate_cache()
        if not shared.app_state.get("_handoff_in_progress"):
            event_log.emit_activity("handoff_completed", f"{previous} → {name}",
                          handoff={"from": previous, "to": name,
                                   "reason": "vision capability required" if llama_local.is_vision_model(name)
                                   and not llama_local.is_vision_model(previous) else "model selection"})
    shared.app_state["local_engaged"] = name
    shared.app_state["last_working_model"] = f"Local ({name})"


def _watch_load(model_name: str):
    """What the model is doing before its first word, on the running model event
    (the chat's Thinking row and model strip read it): phase "loading" while
    the router loads it into memory - a watcher asks the router every second -
    then "prompt" while it reads the conversation; the first streamed word makes
    it "output" (_local_complete). Returns the watcher's stop event, or None."""
    ev = shared.app_state.get("running_model_event")
    if not isinstance(ev, dict) or ev.get("status") != "running":
        return None
    if llama_local.router_status(model_name).get("loaded"):
        event_log.update_event(ev, phase="prompt", model=model_name)
        return None
    started = time.time()
    event_log.update_event(ev, phase="loading", loading=model_name, loading_since=started, model=model_name,
                           size_mb=round((llama_local.model_file_size(model_name) or 0) / 2 ** 20))
    event_log.log_action(f"Loading {model_name} into memory (the first message after a switch waits for it).")
    stop = threading.Event()

    def watch():
        while not stop.wait(1.0):
            if ev.get("status") != "running" or ev.get("phase") != "loading":
                return
            if llama_local.server_models(force=True) is not None and llama_local.router_status(model_name).get("loaded"):
                event_log.update_event(ev, phase="prompt", loading=None, loaded_in=round(time.time() - started, 1))
                return
    threading.Thread(target=watch, daemon=True, name="load-watch").start()
    return stop


def _ensure_build(model_name: str, back: bool = True):
    """Restart llama-server on the build this model needs (llama_builds.py, D51) - the chat shows "Starting
    PrismML's llama.cpp" meanwhile. back=False: only ever TO the build a model requires."""
    ev = shared.app_state.get("running_model_event")
    live = isinstance(ev, dict) and ev.get("status") == "running"

    def told(kind):
        event_log.log_action(f"{model_name} runs on {llama_builds.NAMES[kind]} - restarting llama-server on it "
                             "(a few seconds; models.ini is unchanged).")
        if live:
            event_log.update_event(ev, phase="loading", loading=model_name, loading_since=time.time(),
                                   engine=llama_builds.NAMES[kind], model=model_name)
    switched = llama_builds.ensure_for(model_name, back=back, on_switch=told)
    if switched and live:
        event_log.update_event(ev, engine=None)
    return switched


def _queued_wait():
    """While a reply waits for the local model: Stop can end the wait, and when the wait is a model loading (the
    load ahead while you type) the Thinking row says so - "Loading <model> into memory · 12s" - instead of a
    thinking phrase (1.4.4)."""
    run_control.checkpoint()
    name, ev = local_runtime.loading["model"], shared.app_state.get("running_model_event")
    if name and isinstance(ev, dict) and ev.get("status") == "running" and ev.get("phase") != "loading":
        event_log.update_event(ev, phase="loading", loading=name, loading_since=local_runtime.loading["since"], model=name)


@local_runtime.exclusive(stop=lambda: _queued_wait())
def call_local(messages, requires_vision: bool, model_name: str = None):
    reported = llama_local.server_models(force=True)
    if reported is None:
        # Down (crashed, closed, hung): start it again and carry on, instead of a dead end (1.4.7).
        ev = shared.app_state.get("running_model_event")
        if isinstance(ev, dict) and ev.get("status") == "running":
            event_log.update_event(ev, phase="loading", loading="llama-server", loading_since=time.time())
        back = llama_builds.ensure_running("a reply needs it")
        reported = llama_local.server_models(force=True) if back["ok"] else None
        if reported is None:
            raise RuntimeError(llama_local.unreachable_message(back.get("message") or ""))
    if not reported:
        raise RuntimeError(llama_local.empty_registry_message())

    if model_name and requires_vision and not llama_local.is_vision_model(model_name):
        model_name = None           # planned for text; this payload needs vision
    model_name = model_name or resolve_local_model(requires_vision)
    if model_name is None and requires_vision:
        # Sending image parts to a text-only GGUF makes llama-server reject the
        # whole request, so drop the images rather than fail the turn.
        event_log.log_action("No local vision model is available; continuing without the screenshot.")
        return call_local(strip_images(messages), requires_vision=False)
    if model_name is None:
        raise RuntimeError(llama_local.empty_registry_message())

    payload = messages if (requires_vision and llama_local.is_vision_model(model_name)) else strip_images(messages)
    event_log.log_action(f"Local model: {model_name}")
    _ensure_build(model_name)
    load_watch = _watch_load(model_name)        # "Loading <model>" in the chat while the router loads it
    retry_from_disk = False
    try:
        # nvidia-smi cannot reliably expose per-process VRAM on Windows/WDDM.
        # Capture the system-only floor before llama-server loads its first
        # model, then freeze the after-load delta once this request succeeds.
        system_metrics.begin_model_load()
        raw = bool(shared.app_state.get("raw_mode"))
        kwargs = dict(
            model=f"openai/{model_name}", api_base=config.LLAMA_SERVER_BASE, api_key=config.LOCAL_API_KEY,
            messages=payload,
            temperature=0.1, top_p=0.9, max_tokens=_local_reply_tokens(), timeout=300,
            # cache_prompt lets llama.cpp reuse the KV cache for the unchanged
            # prefix of the conversation. It only pays off because the volatile
            # live-state block is appended at the END of the message list for
            # local mode rather than merged into the system prompt - see
            # build_messages. Merged into the system message, the prefix would
            # differ on every single turn and the cache would never hit.
            extra_body={"cache_prompt": True},
        )
        if raw:
            # Raw mode: the model's own sampling (models.ini / the GGUF's defaults), not the agent's tool-steady 0.1.
            kwargs.pop("temperature")
            kwargs.pop("top_p")
        if shared.app_state.get("effort_thinking") is not None:
            # Native effort for reasoning GGUFs (Qwen3 & co.): thinking only at
            # high/max effort. Templates without the variable ignore it.
            kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": bool(shared.app_state["effort_thinking"])}
        turn_tools = schemas.tools_for_current_turn(local=True)
        _detect_local_caps(model_name)
        res, sent, protocol = _complete_with_protocol(kwargs, model_name, turn_tools)
        if not res.choices:
            raise RuntimeError("empty response")
        res = _retry_if_cut_off(res, sent)
        if protocol in ("schema", "text"):
            res = _unwrap_schema_reply(res, turn_tools)
        model_caps.observe_reply(model_name, res.choices[0].message)
        if turn_tools:
            _detect_local_caps(model_name, just_used=True)   # loaded now, so /props can be asked
            _announce_protocol(model_name)
        system_metrics.finish_model_load(model_name)
        engage_local_model(model_name)
        return res
    except llm_stream.StreamCancelled:
        raise RunStopped("user")          # Stop while the model was writing: nothing it wrote is used
    except Exception as e:
        if _connection_lost(e) and llama_local.server_models(force=True) is None:
            # It died under the reply (1.4.7): start it again now, so Try again works straight away.
            back = llama_builds.ensure_running("it stopped during a reply")
            if back["ok"]:
                raise RuntimeError("llama-server stopped in the middle of the reply and was started again - "
                                   "Try again to get the answer.")
            raise RuntimeError(llama_local.unreachable_message(back.get("message") or ""))
        over = context_overflow(e)
        if over is not None:
            # Kept parseable ("... exceeds the available context size ...") so call_model can trim and retry once.
            sent, window = over
            alone = _drop_oversized_request(window or llama_local.model_context_limit(model_name))
            if alone:
                raise RuntimeError(f"Your message is too long for {model_name}: about {alone:,} tokens, and it reads "
                                   f"{window or llama_local.model_context_limit(model_name):,} at once. Attach it as a file "
                                   "instead (the app reads files in parts), or pick a model with a bigger window. It was "
                                   "left out of the chat, so the chat goes on as before.")
            raise RuntimeError(f"This chat is too long for {model_name}: the request{f' ({sent} tokens)' if sent else ''} "
                               f"exceeds the available context size{f' ({window} tokens)' if window else ''} even after "
                               "older messages were dropped. Compress the chat "
                               "(the ring beside the message box), start a new one, or pick a model with a bigger window.")
        if load_failure.is_load_failure(str(e)) and _load_from_disk_instead(model_name, load_failure.diagnose(model_name, str(e))):
            retry_from_disk = True                         # once: the next try reads it memory-mapped (below)
        else:
            why = load_failure.explain(model_name, e)      # the real cause, from the router's log (1.4.7)
            if why:
                raise RuntimeError(why)
            raise RuntimeError(f"Local model '{model_name}' failed: {str(e)[:250]}" + llama_local.load_hint(model_name))
    finally:
        if load_watch:
            load_watch.set()
    if retry_from_disk:
        return call_local(messages, requires_vision, model_name)
    raise RuntimeError(f"Local model '{model_name}' failed." + llama_local.load_hint(model_name))   # not reached


def _load_from_disk_instead(model_name: str, diag: dict) -> bool:
    """Not enough free RAM to read a model into memory up front (load-mode = none - Ornith-1.5-35B needed 17.5 GB, 4 Nov):
    load it memory-mapped instead - its experts are read from disk as they are used, so the first reply is slower, but it
    loads. Once per model: True when the change was made and the load should be tried again (1.4.7)."""
    if diag.get("kind") != "ram":
        return False
    own = llama_local.model_own_options(model_name)
    if str(own.get("load-mode", "")).lower() != "none":
        return False
    try:
        llama_local.set_model_options({model_name: {"load-mode": None}})
    except (KeyError, ValueError, OSError):
        return False
    event_log.log_action(f"{model_name}: not enough free RAM to read it in up front - loading it from disk instead "
                         "(memory-mapped: the first reply is slower). Settings › Diagnostics can set it back.")
    event_log.emit_event("note", f"Not enough free RAM for {model_name} - loading it from disk instead (slower first reply)")
    try:
        llama_local.unload_all()
    except Exception:
        pass
    return True


def _drop_oversized_request(window: int) -> int:
    """The latest user message alone is bigger than the model's window: put a note in its place and return its
    estimated size (0 when it isn't). Kept, it failed every later message too - "Say hi" after a 166,000-character
    paste was still "too long" (live test, 4 Nov 2026), until the chat was compressed by hand."""
    conv = shared.app_state.get("conversation") or []
    for i in range(len(conv) - 1, -1, -1):
        m = conv[i]
        if m.get("role") != "user":
            continue
        text = m.get("content") if isinstance(m.get("content"), str) else ""
        size = _approx_tokens({"c": text})
        if not text or size < max(1000, int(window * 0.8)):
            return 0
        conv[i] = {**m, "content": f"[A message of {len(text):,} characters was here - too long for this model to read "
                                   "at once, so it was left out. Ask the user to attach it as a file or send a shorter part.]"}
        return size
    return 0


def _router_ready() -> bool:
    """A fresh look (never the 20 s cache) before a helper call, and a restart when it's down or hung (1.4.7).
    On the cached answer, a router that had stopped answering made the prompt-prep and planner calls each wait their
    full 120 s timeout before the reply itself began - 329 s for "hi" in the live test (4 Nov 2026)."""
    if llama_local.server_models(force=True) is not None:
        return True
    return bool(llama_builds.ensure_running("a helper call needs it")["ok"]) and \
        llama_local.server_models(force=True) is not None


def _connection_lost(e) -> bool:
    """The connection to llama-server broke or never opened (not an HTTP error it answered with)."""
    text = f"{type(e).__name__} {e}"
    return bool(re.search(r"Connection ?(?:error|refused|reset|aborted)|ConnectError|RemoteProtocolError|"
                          r"ReadError|Server disconnected|RemoteDisconnected|peer closed|incomplete chunked", text, re.I))


# ---- streaming (llm_stream.py -> stream_hub.py -> GET /api/stream) -----------------
STREAM_TRANSPORT = None          # None = real HTTP; the replay harness injects recorded chunks


@local_runtime.exclusive(stop=lambda: run_control.checkpoint())
def _local_complete(kwargs: dict, cancel=None, record=True, publish=True):
    """One local completion, streamed from llama-server. Every fragment is
    relayed to the UI as it arrives; Stop closes the stream at once; a runaway
    or looping tool call is cut off early (llm_stream). `cancel` replaces the
    run's Stop check for requests made outside a run (the benchmark).
    publish=False: a hidden helper - nothing reaches the chat's live draft."""
    payload = {k: v for k, v in kwargs.items() if k in (
        "messages", "temperature", "top_p", "max_tokens", "tools", "tool_choice", "parallel_tool_calls")}
    payload["model"] = str(kwargs.get("model") or "").removeprefix("openai/")
    payload.update(kwargs.get("extra_body") or {})
    source, _ = model_monitor.where()
    if model_monitor.blocked(source, shared.app_state.get("settings")):     # Settings > This PC: replies only
        raise RuntimeError(f"Model use outside your replies is off ({source}) - Settings > This PC.")
    ev = shared.app_state.get("running_model_event")
    label = ev.get("label", "") if isinstance(ev, dict) else ""
    call_id = shared.app_state.get("_stream_call") if publish else None
    got_output = {"any": False}
    # When the model stopped thinking: its first answer text or tool call.
    # Reasoning printed inline (<think>...</think>) is still thinking.
    timing = {"start": time.time(), "first": None, "content": "", "closed": None}

    def on_delta(channel, text, **extra):
        if not got_output["any"]:                  # the first word: it has loaded and read the conversation
            ev = shared.app_state.get("running_model_event")
            if isinstance(ev, dict) and ev.get("status") == "running" and ev.get("phase") != "output":
                event_log.update_event(ev, phase="output", loading=None, first_token_s=round(time.time() - timing["start"], 1))
        got_output["any"] = True
        if timing["first"] is None:
            if channel == "tool":
                timing["first"] = time.time()
            elif channel == "content" and text:
                timing["content"] += text
                _mark_answer_start(timing)
        if call_id is not None:
            # n: one streamed chunk is one token from llama-server - the page's live speed is tok/s, not char/s (1.4.7)
            stream_hub.hub.publish("delta", call=call_id, channel=channel, text=text, n=1, **extra)
    # A dropped connection BEFORE any output (router swapping models, a brief
    # restart) is retried twice with backoff. Once output has arrived nothing is
    # retried here - that would duplicate text the user already saw.
    for attempt in range(3):
        try:
            res = llm_stream.complete(f"{config.LLAMA_SERVER_BASE.rstrip('/')}/chat/completions", payload,
                                      api_key=config.LOCAL_API_KEY, timeout=kwargs.get("timeout", 300), on_delta=on_delta,
                                      should_cancel=cancel or run_control.is_stopped, transport=STREAM_TRANSPORT)
            break
        except llm_stream.StreamCancelled:
            if record and recorder.active():
                recorder.cancelled()
            raise
        except Exception as e:
            transient = type(e).__name__ in ("ConnectError", "RemoteProtocolError", "ReadError") \
                or bool(re.search(r"\b(502|503)\b|loading model", str(e)))
            if attempt == 2 or got_output["any"] or not transient or (cancel or run_control.is_stopped)():
                model_monitor.record(payload["model"], seconds=time.time() - timing["start"], label=label, ok=False)
                raise
            event_log.log_action(f"Local model connection failed ({type(e).__name__}); retrying ({attempt + 1}/2).")
            event_log.emit_event("note", f"Lost the connection to the local model - retrying ({attempt + 1} of 2)…", status="done")
            time.sleep(0.8 * (2 ** attempt))
    if record and recorder.active():         # raw reply, before any parsing (see recorder.py)
        recorder.response(res.choices[0].message, res.choices[0].finish_reason, res.usage)
    usage = getattr(res, "usage", None)
    model_monitor.record(payload["model"], getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0,
                         time.time() - timing["start"], label=label)
    try:                                     # the chat's "Thought for Xs" (agent/loop._emit_model_note)
        res.choices[0].message.thinking_s = round((timing["first"] or time.time()) - timing["start"], 1)
    except (AttributeError, IndexError, TypeError):
        pass
    return res


_THINK_OPEN = re.compile(r"^\s*<\s*(think|thinking|reasoning|thought)\s*>", re.I)
_THINK_CLOSE = re.compile(r"<\s*/\s*(think|thinking|reasoning|thought)\s*>", re.I)


def _mark_answer_start(timing: dict):
    """Set timing["first"] once streamed content is answer text rather than
    reasoning printed inline: text that doesn't open with <think>, or anything
    after its closing tag."""
    text = timing["content"]
    if timing["closed"] is None:
        if _THINK_OPEN.match(text) or not text.strip() or (text.lstrip().startswith("<") and len(text.strip()) < 12):
            close = _THINK_CLOSE.search(text)
            if not close:
                return
            timing["closed"] = close.end()
        else:
            timing["first"] = time.time()
            return
    if text[timing["closed"]:].strip():
        timing["first"] = time.time()


# ---- tool-call constraints per model (model_caps.py) ------------------------------
def _llama_root() -> str:
    return re.sub(r"/v1/?$", "", config.LLAMA_SERVER_BASE.rstrip("/"))


def _http_json(url: str, timeout: float):
    with netlayer.client("models", timeout=timeout) as c:       # llama-server: local, so direct
        r = c.get(url)
    r.raise_for_status()
    return r.json()


def _detect_local_caps(model_name: str, just_used: bool = False):
    """Ask llama-server what this model supports, once it is loaded (a /props
    query for an unloaded model could make the router load it). Re-checked at
    most every 10 minutes per model. `just_used`: a request to this model just
    succeeded, so it is loaded."""
    known = model_caps.get(model_name)
    if known.get("detected_at") and time.time() - known["detected_at"] < 600:
        return
    loaded = just_used or llama_local.router_status(model_name).get("loaded") \
        or shared.app_state.get("local_engaged") == model_name
    if loaded:
        model_caps.detect(model_name, _llama_root(), _http_json, loaded=True)


def _announce_protocol(model_name: str):
    """Once per run and model, and only once there is something to say."""
    announced = shared.app_state.setdefault("_protocol_announced", set())
    if model_name in announced or not model_caps.protocol(model_name):
        return
    announced.add(model_name)
    event_log.emit_activity("tool_protocol", model_caps.describe(model_name), model=model_name,
                  protocol=model_caps.protocol(model_name) or "unknown")


def _complete_with_protocol(kwargs: dict, model_name: str, turn_tools: list, cancel=None):
    """Send the request with the strongest tool-call constraint this model and
    server accept, stepping down only on an explicit refusal:
    native/generic (server grammar via `tools`) -> schema (our JSON schema on
    the whole reply) -> text (lenient parsing)."""
    protocol = model_caps.protocol(model_name)
    force = bool(shared.app_state.pop("force_tool_once", False))     # agent/loop: the retry after a described step
    if not turn_tools:
        return _local_complete(kwargs, cancel), kwargs, protocol
    while True:
        attempt = dict(kwargs)
        attempt["extra_body"] = dict(kwargs.get("extra_body") or {})
        if protocol in (None, "native", "generic"):
            attempt["tools"] = turn_tools
            attempt["tool_choice"] = "required" if force else "auto"
        else:
            attempt["messages"] = list(kwargs["messages"]) + [
                {"role": "system", "content": model_caps.catalogue_prompt(turn_tools)}]
            if protocol == "schema":
                attempt["extra_body"]["response_format"] = {
                    "type": "json_schema", "json_schema": {"name": "reply", "schema": model_caps.reply_schema(turn_tools)}}
        try:
            res = _local_complete(attempt, cancel)
        except llm_stream.StreamCancelled:
            raise
        except Exception as e:
            text = str(e)
            if protocol in (None, "native", "generic") and model_caps.tools_refused(text):
                event_log.log_action(f"{model_name}: llama-server refused native tools - constraining replies with a JSON schema.")
                protocol = "schema"
                model_caps.update(model_name, tool_protocol="schema")
                continue
            if protocol == "schema" and model_caps.schema_refused(text):
                event_log.log_action(f"{model_name}: JSON-schema replies refused - falling back to lenient parsing.")
                protocol = "text"
                model_caps.update(model_name, tool_protocol="text", json_schema=False)
                continue
            raise
        if protocol == "schema":
            model_caps.update(model_name, json_schema=True)
        return res, attempt, protocol


def _unwrap_schema_reply(res, turn_tools: list):
    """A {"tool", "arguments"} / {"answer"} reply as the native message shape,
    so everything downstream (parsing, approval, events) is unchanged."""
    try:
        msg = res.choices[0].message
        answer, _reasoning = model_output.split_reasoning(getattr(msg, "content", "") or "")
        data = json.loads(answer)
    except Exception:
        return res                      # not JSON: parse_response's lenient recovery handles it
    names = {t["function"]["name"] for t in turn_tools}
    if isinstance(data, dict) and data.get("tool") in names:
        msg.tool_calls = [parsing._ShimToolCall(data["tool"], json.dumps(data.get("arguments") or {}))]
        msg.content = ""
    elif isinstance(data, dict) and isinstance(data.get("answer"), str):
        msg.content = data["answer"]
    return res


def _local_needs_vision(messages) -> bool:
    """Whether this local call needs a vision model. An image from an earlier turn doesn't: a chat with Qwen chosen
    and one old picture in it swapped to the vision model on every message, and the typing warm-up swapped back -
    two reloads a message (1.4.4). Only an image in this turn escalates; when a vision model is already loaded,
    the old images go along at no cost."""
    if not has_images(messages):
        return False
    if history.images_this_turn(messages):
        return True
    return llama_local.is_vision_model(shared.app_state.get("local_engaged") or "")


def _local_payload(messages) -> tuple[list, bool]:
    """The messages for a local call and whether they need vision - earlier turns' images as a note when they don't."""
    if has_images(messages) and not _local_needs_vision(messages):
        return history.stub_earlier_images(messages), False
    return messages, has_images(messages)


def _fit_for_local() -> str | None:
    """Pick the local model that will run this call, then trim history to ITS
    real window. The order matters: budgeting against the currently loaded
    model and only then escalating to a vision model with a smaller window
    (a 250K text model -> a 32K vision model) sent requests the router
    rejected for exceeding its context. Returns the chosen model (or None)."""
    demote_old_images(config.IMAGES_KEPT_LOCAL)
    needs_vision = _local_needs_vision(shared.app_state["conversation"])
    try:
        planned = resolve_local_model(needs_vision)
    except Exception:
        planned = None
    target = planned or shared.app_state.get("local_engaged") or shared.app_state.get("local_model") or None
    # The tools are sized for this model's window (schemas._local_context_window). Under Local · Auto the model
    # that answers can differ from the pinned or last-loaded one: a 16k Gemma was sent a catalogue fitted for a
    # 64k window - 25k tokens - and refused the request (2026-10-01).
    shared.app_state["local_planned"] = target
    # The fixed cost of every call: the system prompt as sent (persona and careful
    # rules included), the tools actually offered this turn (Roblox tools add a
    # lot while Studio is connected), the request with its notes and files, the
    # live state block and the reply we reserve room for.
    overhead = (_approx_tokens({"t": schemas.tools_for_current_turn(local=True)})
                + _approx_tokens({"s": system_prompt(True)}) + _turn_tokens() + 900)
    # The model's own window is the limit - the context set for it in Settings › Models. The effort level's smaller
    # context size capped it too, so a model set to 80K still dropped history past ~30K (2026-10-21). Kept in 1.4.1
    # by the user's choice over the audit's faster cap (2026-10-27).
    budget = max(1500, llama_local.model_context_limit(target) - overhead - _local_reply_tokens())
    shared.app_state["_local_fit_budget"] = budget
    trim_conversation(budget_tokens=budget)
    return planned


_OVERFLOW = re.compile(r"n_prompt_tokens['\"]?\s*:\s*(\d+).*?n_ctx['\"]?\s*:\s*(\d+)|\((\d+) tokens\) exceeds the available context size \((\d+) tokens\)", re.S)


def context_overflow(error) -> tuple[int, int] | None:
    """(tokens sent, window) when llama-server refused a request as longer than its context, else None."""
    text = str(error)
    if "exceed_context_size" not in text and "exceeds the available context size" not in text:
        return None
    m = _OVERFLOW.search(text)
    if not m:
        return (0, 0)
    sent, window = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return int(sent), int(window)


def _refit_after_overflow(error) -> bool:
    """The token estimate is a character count (history.CHARS_PER_TOKEN); dense text - code, JSON, Hebrew -
    can run past it. llama-server says by how much: drop that much older history plus a margin, once."""
    over = context_overflow(error)
    if over is None:
        return False
    sent, window = over
    budget = shared.app_state.get("_local_fit_budget") or 4000
    cut = (sent - window if sent > window else 0) + max(512, window // 10)
    new_budget = max(1000, budget - cut)
    before = history.total(shared.app_state["conversation"][1:], True)
    trim_conversation(budget_tokens=new_budget)
    after = history.total(shared.app_state["conversation"][1:], True)
    if after >= before:
        return False                              # nothing older left to drop: the retry would fail the same way
    event_log.log_action(f"The request was {sent - window if sent > window else '?'} tokens over the model's "
                         f"{window or '?'}-token window - dropped older history ({before} -> {after} tokens) and retried.")
    event_log.emit_event("note", "The chat was longer than the model's window - trimmed older messages and retried",
                         detail="Older turns were dropped for this call; the last requests are kept word for word.")
    return True


DESIGN_REPLY_TOKENS = 8192          # room for a /design reply: a whole HTML document


def _local_reply_tokens() -> int:
    """Room for one local reply. With thinking on (Thorough and up), the
    model's reasoning is generated inside the same max_tokens as its answer,
    so the flat 2048 left a long tool call (a whole HTML file) only what the
    reasoning did not use - and it was cut off mid-arguments. The reasoning
    budget comes from the run's effort profile."""
    base = config.LOCAL_MAX_TOKENS
    if shared.app_state.get("design_mode"):
        base = max(base, DESIGN_REPLY_TOKENS)       # /design answers with a whole HTML document
    if not shared.app_state.get("effort_thinking"):
        return base
    budget = shared.app_state.get("run_budget")
    extra = budget.profile.reasoning_tokens if budget is not None else config.LOCAL_THINKING_TOKENS
    # A saved zero override is deliberate; only a missing run budget uses the legacy default.
    return base + extra


def _cut_off_kind(choice) -> str | None:
    """Why a reply that stopped at the length limit is unusable, or None.
      'tool_call'  - a tool call (native or printed as text) whose arguments
                     never finished;
      'reasoning'  - the whole budget went on thinking and nothing was said."""
    if getattr(choice, "finish_reason", None) != "length":
        return None
    msg = choice.message
    native = getattr(msg, "tool_calls", None) or []
    if any(isinstance(parsing._native_args(getattr(tc.function, "arguments", None)), parsing._MalformedArgs) for tc in native):
        return "tool_call"
    answer, reasoning = model_output.split_reasoning(getattr(msg, "content", "") or "")
    if not native and model_output.looks_truncated_json(answer):
        return "tool_call"
    if not native and not answer.strip() and (reasoning or getattr(msg, "reasoning_content", None)):
        return "reasoning"
    return None


def _retry_if_cut_off(res, kwargs, cancel=None):
    """A reply that hit max_tokens in the middle of a tool call (a long script
    for roblox_set_script_source, a file for write_file) arrives with broken
    arguments and would be lost - or, worse, turned into a call with no
    arguments. Retry once with double the room and a note asking for a
    shorter call; if that is also cut off, return it as is (the agent loop
    then answers the unreadable call with an error, never runs it)."""
    kind = _cut_off_kind(res.choices[0])
    if not kind:
        return res
    event_log.log_action(f"Local reply was cut off at {kwargs.get('max_tokens')} tokens "
               f"{'mid tool call' if kind == 'tool_call' else 'while still reasoning'} - retrying once with more room.")
    event_log.emit_event("note", "Reply was cut off by the length limit - retrying with more room",
               detail="The model's last reply ended mid tool call." if kind == "tool_call"
               else "The model used its whole reply budget on reasoning.")
    retry = dict(kwargs)
    retry["max_tokens"] = int(kwargs.get("max_tokens") or config.LOCAL_MAX_TOKENS) * 2
    note = ("[SYSTEM] Your last reply was cut off by the length limit in the middle of a tool call. "
            "Make the call again, keeping its arguments short - split long content into several smaller steps "
            "(for a long file: write_file the first part, then write_file with append=true for each next part).")
    if kind == "reasoning":
        # Thinking already ate one full budget; answer directly this time.
        extra = dict(retry.get("extra_body") or {})
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        retry["extra_body"] = extra
        note = "[SYSTEM] Your last reply ran out of room while reasoning. Act now: make the next tool call or answer directly."
    retry["messages"] = list(kwargs["messages"]) + [{"role": "user", "content": note}]
    if shared.app_state.get("_stream_call") is not None:
        stream_hub.hub.publish("restart", call=shared.app_state["_stream_call"], reason=kind)
    try:
        again = _local_complete(retry, cancel)
        return again if again.choices else res
    except llm_stream.StreamCancelled:
        raise
    except Exception as e:
        event_log.log_action(f"Retry after cut-off failed: {str(e)[:120]}")
        return res


def call_model():
    """Builds the outgoing message list fresh (live state + budgeted history)
    and dispatches it."""
    mode = shared.app_state.get("mode", config.MODE_CLOUD)

    # Near the limit, summarize older turns before anything has to be dropped.
    import agent.context_window as context_window
    context_window.maybe_compress()

    planned = None
    if mode == config.MODE_LOCAL:
        # The image engine (image_gen.py) gives the GPU back before the chat model loads onto it - or the model
        # would load squeezed beside it and answer slowly.
        import image_gen
        image_gen.yield_gpu()
        planned = _fit_for_local()
    else:
        demote_old_images(config.IMAGES_KEPT_CLOUD)
        run_budget = shared.app_state.get("run_budget")
        trim_conversation(budget_tokens=run_budget.profile.context_tokens if run_budget is not None else None)

    messages = build_messages()
    # Vision is decided by scanning the ENTIRE outgoing list. Looking only at
    # the last message routed a screenshot-bearing conversation to a text
    # model whenever any tool result landed after the image.
    requires_vision = has_images(messages)

    if mode == config.MODE_LOCAL:
        try:
            return call_local(*_local_payload(messages), model_name=planned)
        except RuntimeError as e:
            if not _refit_after_overflow(e):
                raise
        return call_local(*_local_payload(build_messages()), model_name=planned)
    if mode == config.MODE_CLOUD:
        return call_cloud(messages, requires_vision)
    try:
        return call_cloud(messages, requires_vision)
    except RuntimeError:
        event_log.log_action("Cloud unavailable, falling back to local...")
        # The cloud budget is far larger than most local windows; re-fit first.
        planned = _fit_for_local()
        messages = build_messages(prompts.SYSTEM_PROMPT_COMPACT)
        return call_local(*_local_payload(messages), model_name=planned)


# ==============================================================================
# Agent orchestration (orchestrator.py): prepare -> plan -> execute -> review
# -> bounded revision. The Executor stage IS chat_with_agent above, so every
# deterministic desktop route and the agent loop run exactly as before.
# ==============================================================================
def cross_check_status(mode: str | None = None, model: str | None = None) -> dict:
    """Availability only; never load a model or send a request to check this."""
    mode = mode or shared.app_state.get("mode")
    if mode == config.MODE_LOCAL:
        return {"available": False, "reason": "Unavailable locally: this backend runs one model at a time."}
    pairs = candidate_pairs(False)
    usable = [p for p in pairs if shared._cooldown_until.get(p, 0) <= time.time()]
    if not usable:
        return {"available": False, "reason": "No cloud review model is available."}
    primary = model or _current_model()
    if primary not in {p[0] for p in pairs}:
        primary = usable[0][0]
    second = next((p[0] for p in usable if p[0] != primary), None)
    if second is None:
        return {"available": False, "reason": "A different cloud model needs to be configured and available."}
    return {"available": True, "model": second, "reason": "Uses a different model to review the result. Applies to new requests."}


@local_runtime.exclusive(stop=lambda: run_control.checkpoint())
def model_json(system: str, user: str, max_tokens: int = 600, secondary: bool = False):
    """One tool-free, structured model call for the preparer / planner /
    reviewer. Returns a dict, or None when no model answers (every caller
    then falls back to its deterministic rules). Reasoning text is discarded."""
    if run_control.is_stopped():
        return None
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    mode = shared.app_state.get("mode")
    try:
        if mode == config.MODE_LOCAL or (mode == config.MODE_AUTO and not candidate_pairs(False)):
            if secondary:
                return None          # one local model at a time - no free second opinion
            if not _router_ready():
                return None
            model = shared.app_state.get("local_engaged") or resolve_local_model(False)
            if not model:
                return None
            res = llm_compat.completion(model=f"openai/{model}", api_base=config.LLAMA_SERVER_BASE, api_key=config.LOCAL_API_KEY,
                                        messages=messages, temperature=0, max_tokens=max_tokens, timeout=120,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            engage_local_model(model)
        else:
            pairs = candidate_pairs(False)
            now = time.time()
            usable = [p for p in pairs if shared._cooldown_until.get(p, 0) <= now] or pairs
            if not usable:
                return None
            if secondary:
                status = cross_check_status()
                if not status["available"]:
                    return None
                usable = [p for p in usable if p[0] == status["model"]]
                if not usable:
                    return None
            model_name, key = usable[0]
            res = llm_compat.completion(model=model_name, api_key=key, messages=messages, timeout=45,
                                        temperature=0, max_tokens=max_tokens)
        answer, _ = model_output.split_reasoning(res.choices[0].message.content or "")
        return parsing._coerce_args(answer) or None
    except Exception as e:
        event_log.log_action(f"Structured {'cross-check ' if secondary else ''}model call failed: {str(e)[:120]}")
        return None


@local_runtime.exclusive(stop=lambda: run_control.checkpoint())
def model_text(system: str, user: str, max_tokens: int = 1500, model: str | None = None, stream: bool = False,
               label: str = "Model is writing", temperature: float = 0.3, think: bool = False,
               images: list | None = None) -> dict:
    """One tool-free text answer - Research's notes and report, Critique, the
    second answer of Compare. model: None is the engine in use; "local:<name>"
    or a cloud model id picks one. stream: a local model's text streams into the
    chat's live draft, like a reply. It is a "model" event, so the reply's
    tokens and the usage meter count it. Stop ends it. images: PIL images sent
    with the user text (the model must see pictures - image_plan.check asks for
    a vision model by name).
    -> {"text", "model", "tokens_in", "tokens_out", "seconds"}. Raises RuntimeError."""
    run_control.checkpoint()
    content = user
    if images:
        local_target = bool(model and model.startswith("local:")) or shared.app_state.get("mode") == config.MODE_LOCAL
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(im, for_local=local_target)}"}}
            for im in images]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    mode = shared.app_state.get("mode")
    local = None
    if model and model.startswith("local:"):
        local = model[6:]
    elif not model and (mode == config.MODE_LOCAL or (mode == config.MODE_AUTO and not candidate_pairs(False))):
        if not _router_ready():
            raise RuntimeError(llama_local.unreachable_message(llama_builds.status_line()))
        local = shared.app_state.get("local_engaged") or resolve_local_model(False)
        if not local:
            raise RuntimeError(llama_local.empty_registry_message())
    ev = event_log.emit_event("model", label, status="running", step=0)
    started = time.time()
    prev_call = shared.app_state.get("_stream_call")
    prev_ev = shared.app_state.get("running_model_event")
    shared.app_state["running_model_event"] = ev
    load_watch = None
    stopped = False
    # Only a streamed helper takes over the live draft's call id. A hidden one must not touch it: restoring it
    # afterwards could wipe the id a reply set while this helper held the model (2026-10-27).
    if stream and local:
        shared.app_state["_stream_call"] = ev["id"]
        stream_hub.hub.publish("start", call=ev["id"], step=0, model=local)
    try:
        if local:
            _ensure_build(local, back=False)
            load_watch = _watch_load(local)
            kwargs = dict(model=f"openai/{local}", api_base=config.LLAMA_SERVER_BASE, api_key=config.LOCAL_API_KEY,
                          messages=messages, temperature=temperature, top_p=0.9, max_tokens=max_tokens, timeout=300,
                          extra_body={"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": bool(think)}})
            res = _local_complete(kwargs, record=False, publish=stream)
            engage_local_model(local)
            used = local
        else:
            pairs = [(model, k) for k in get_keys_for_model(model)] if model else candidate_pairs(False)
            now = time.time()
            usable = [p for p in pairs if shared._cooldown_until.get(p, 0) <= now] or pairs
            if not usable:
                raise RuntimeError(f"No API key is configured for {model or 'any cloud model'}.")
            last = None
            for used, key in usable[:3]:
                try:
                    res = llm_compat.completion(model=used, api_key=key, messages=messages, timeout=90,
                                                temperature=temperature, max_tokens=max_tokens)
                    break
                except Exception as e:
                    last = e
                    shared._cooldown_until[(used, key)] = now + shared.COOLDOWN_SECONDS
            else:
                raise RuntimeError(f"{used} failed: {str(last)[:200]}")
        answer, _ = model_output.split_reasoning(res.choices[0].message.content or "")
        usage = getattr(res, "usage", None)
        out = {"text": answer.strip(), "model": used, "seconds": round(time.time() - started, 1),
               "tokens_in": int(getattr(usage, "prompt_tokens", 0) or 0), "tokens_out": int(getattr(usage, "completion_tokens", 0) or 0)}
        event_log.finish_event(ev, ok=True, model=used, duration_ms=int(out["seconds"] * 1000),
                               tokens_in=out["tokens_in"], tokens_out=out["tokens_out"],
                               **model_metrics.generation_facts(res))
        return out
    except llm_stream.StreamCancelled:
        stopped = True
        event_log.finish_event(ev, ok=False, detail="Stopped")
        raise RunStopped("user")
    except RunStopped:
        stopped = True
        event_log.finish_event(ev, ok=False, detail="Stopped")
        raise
    except Exception as e:
        event_log.finish_event(ev, ok=False, detail=str(e)[:200])
        raise RuntimeError(str(e)[:300]) from e
    finally:
        if load_watch:
            load_watch.set()
        shared.app_state["running_model_event"] = prev_ev
        if stream and local:
            stream_hub.hub.publish("end", call=ev["id"], finish="stopped" if stopped else "stop", has_tools=False)
            shared.app_state["_stream_call"] = prev_call


def _current_model() -> str | None:
    if shared.app_state.get("mode") == config.MODE_LOCAL:
        return shared.app_state.get("local_engaged") or shared.app_state.get("local_model") or None
    return shared.app_state.get("last_working_model")
