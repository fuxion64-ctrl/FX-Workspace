"""
agent/sessions.py
=================
The active chat and project, and their persistence (chat_store.py,
projects.py). data/session.json remembers which chat and project were open,
so after a restart the same chat continues - the model sees its earlier
messages again, not just the page.

  record_turn(req, result, start_seq)  after every turn: append the user's
                                       message and the reply (with its compact
                                       steps and the model's thinking) and save
                                       the chat.
  new_chat()                           the next message starts a new chat.
  open_chat(id)                        load a saved chat (and its project).
  activate_project(id | None)          switch project; starts a new chat.
"""
from __future__ import annotations

import json
import os
import re
import time

import chat_guard
import chat_store
import model_metrics
import corroboration
import privacy
import roblox_places

import core.config as config
import core.state as shared
import events as event_log
import sources

SESSION_PATH = os.path.join(config.BASE_DIR, "data", "session.json")
# Kinds the chat's step list can show (model calls and notes are Activity-only).
_STEP_KINDS = {"tool", "approval", "activity", "note", "undo", "untrusted", "memory"}
THINKING_KEEP = 30_000        # characters of reasoning saved with one reply (the end is kept)


def _slim(e: dict) -> dict:
    return {k: v for k, v in {
        "id": e.get("id"), "label": e.get("label"), "kind": e.get("kind"), "status": e.get("status"),
        "ok": e.get("ok"), "detail": str(e.get("detail") or "")[:300], "verified": e.get("verified"),
        "phase": e.get("phase"), "level": e.get("level"), "verdict": e.get("verdict"), "steps": e.get("steps"),
        "findings": e.get("findings"), "handoff": e.get("handoff"), "items": e.get("items"),
        "snippets": e.get("snippets"), "tool": e.get("tool"),
        # Undo under a reopened chat's work lines, and what a declined approval asked for (D49).
        "undo_id": e.get("undo_id"), "asked": e.get("asked")}.items() if v not in (None, "", [])}


def web_of(events: list) -> dict | None:
    """The turn's web work for the reply's search row: the queries searched and
    the pages read (search_engine.py puts a "web" field on those tool events)."""
    queries, pages, searches, reads, last = [], [], 0, 0, None
    for e in events:
        web = e.get("web") if e.get("kind") == "tool" else None
        if not isinstance(web, dict):
            continue
        if web.get("op") == "search":
            searches += 1
            last = web
            queries += [q for q in web.get("queries") or [] if q not in queries]
        elif web.get("op") == "read":
            for p in web.get("pages") or []:
                reads += 1
                pages.append({k: p.get(k) for k in ("n", "url", "title", "domain", "ok") if p.get(k) not in (None, "")})
    if not searches and not reads:
        return None
    out = {"q": queries[:12], "pages": pages[:40], "searches": searches, "reads": reads}
    if last and last.get("mode"):             # the privacy mode, backend and route the search used (the row's meta line)
        out["via"] = {"mode": last.get("mode"), "providers": (last.get("providers") or [])[:3], "route": last.get("route")}
    return out


def metrics_of(events: list, started: float | None = None) -> dict:
    """Tokens, time and model of a reply (its footer; the usage meter adds them up)."""
    calls = [e for e in events if e.get("kind") == "model" and e.get("status") == "done"]
    began = started or next((e.get("t") for e in events if isinstance(e.get("t"), (int, float))), None)
    out = {"in": sum(int(e.get("tokens_in") or 0) for e in calls),
           "out": sum(int(e.get("tokens_out") or 0) for e in calls),
           "s": round(time.time() - began, 1) if began else 0,
           "model": next((e.get("model") for e in reversed(calls) if e.get("model")), "")}
    out.update(model_metrics.aggregate_speed(calls))
    first = next((e.get("first_token_s") for e in calls if isinstance(e.get("first_token_s"), (int, float))), None)
    if first is not None:
        out["ttft"] = round(float(first), 1)
    out["calls"] = len(calls)
    # (1.4.8) Prompt tokens llama-server read vs took from its cache, summed over the reply's calls - only when known.
    for key, name in (("tokens_read", "read"), ("tokens_reused", "reused")):
        known = [int(e[key]) for e in calls if isinstance(e.get(key), (int, float)) and not isinstance(e.get(key), bool)]
        if known:
            out[name] = sum(known)
    web = web_of(events) or {}
    out.update(searches=web.get("searches", 0), reads=web.get("reads", 0))
    return {k: v for k, v in out.items() if v not in (0, "", None) or k in ("read", "reused")}   # 0 reused is a fact


def sources_for(message: str) -> list:
    """The reply's sources, as saved with it (sources.Registry.for_reply) - each with its origin when that
    isn't the site itself (a Yahoo page carrying a Reuters story: "Reuters"; corroboration.py), and the
    origin group every source counts in ("group": Focus's reading rail counts independent origins by it)."""
    keep = ("n", "url", "title", "domain", "snippet", "date", "excerpt", "read", "cited")
    items = sources.current().for_reply(message)
    origin, group = corroboration.origin_labels(items), corroboration.origins(items) if items else {}
    return [{**{k: s[k] for k in keep if s.get(k) not in (None, "")}, **({"origin": origin[s["n"]]} if s["n"] in origin else {}),
             **({"group": group[s["n"]]} if group.get(s["n"]) else {})}
            for s in items]


def _save_session():
    chat = shared.app_state.get("active_chat") or {}
    project = shared.app_state.get("active_project") or {}
    # A temporary chat is never restored: the session file names no chat while one is open.
    chat_id = None if privacy.is_temporary(chat) else chat.get("id")
    try:
        os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
        tmp = SESSION_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"chat": chat_id, "project": project.get("id")}, f)
        os.replace(tmp, SESSION_PATH)
    except OSError as e:
        event_log.log_action(f"Could not remember the open chat: {e}")


def active_project_id() -> str | None:
    project = shared.app_state.get("active_project")
    return project.get("id") if isinstance(project, dict) else None


def thinking_of(events: list) -> dict | None:
    """What the model thought during a turn - the reasoning of its 'thought'
    events (agent/loop._emit_model_note), already masked - and for how long.
    Saved with the reply, so the chat's "Thought for Xs" row still opens after
    a reload. None when the model shared no reasoning."""
    parts, seconds = [], 0.0
    for e in events:
        text = str(e.get("reasoning_full") or e.get("reasoning") or "").strip() if e.get("kind") == "thought" else ""
        if text:
            parts.append(text)
            seconds += float(e.get("thinking_s") or 0)
    if not parts:
        return None
    text = "\n\n".join(parts)
    if len(text) > THINKING_KEEP:
        text = "…" + text[-THINKING_KEEP:].lstrip()
    return {"text": text, "seconds": round(seconds, 1)}


def ensure_chat(req) -> dict:
    """The open chat, started (in memory - record_turn saves it) when this is its first message."""
    chat = shared.app_state.get("active_chat")
    if chat is None:
        chat = {"id": shared.chat_store.new_id(), "title": chat_store.title_from(req.message),
                "project": active_project_id(), "created": time.time(), "transcript": []}
        shared.app_state["active_chat"] = chat
        _save_session()
    return chat


def mark_pending(req):
    """Crash safety: before a turn runs, its message is saved with the chat as
    "pending" (record_turn saves the chat without it). A chat that still has
    it was cut off - the app stopped mid-answer - and the page offers to send
    the message again. Never raises. A temporary chat is not saved - nothing
    of it comes back after a crash."""
    if privacy.is_temporary(shared.app_state.get("active_chat")):
        return
    try:
        chat = shared.app_state.get("active_chat")
        if chat is None:
            chat = {"id": shared.chat_store.new_id(), "title": chat_store.title_from(req.message),
                    "project": active_project_id(), "created": time.time(), "transcript": []}
            shared.app_state["active_chat"] = chat
            _save_session()
        pending = {"t": str(req.message or "")[:20000], "at": time.time()}
        if getattr(req, "command", None):
            pending["c"] = req.command
        saved = shared.chat_store.save({**chat, "conversation": shared.app_state.get("conversation") or [],
                                        "context_summary": shared.app_state.get("context_summary") or "",
                                        "pending": pending})
        chat["updated"] = saved["updated"]
    except (OSError, KeyError) as e:
        event_log.log_action(f"Could not save the message before running it: {e}")


def record_turn(req, result: dict, start_seq: int) -> dict | None:
    """Append this turn to the active chat (creating it on the first message)
    and save it. Returns the turn's thinking (thinking_of) for the reply."""
    result = result or {}
    chat = shared.app_state.get("active_chat")
    if chat is None:
        chat = {"id": shared.chat_store.new_id(), "title": chat_store.title_from(req.message),
                "project": active_project_id(), "created": time.time(), "transcript": []}
        shared.app_state["active_chat"] = chat
        _save_session()
    events = event_log.events_since(start_seq)
    steps = [_slim(e) for e in events if e.get("kind") in _STEP_KINDS]
    entries = [{"r": "u", "t": req.message, "v": bool(getattr(req, "with_vision", False))}]
    command = getattr(req, "command", None)
    if isinstance(command, str) and command:           # /plan, /design (shown as a pill on the message)
        entries[0]["c"] = command
    if getattr(req, "response_mode", None) == "raw":   # sent in Raw mode (response_modes.py) - a tag on the message
        entries[0]["rm"] = "raw"
    if shared.app_state.get("turn_files"):             # attached files (composer_tools), shown as chips
        entries[0]["f"] = shared.app_state["turn_files"]
    tools = sorted(t for t in (shared.app_state.get("turn_tools") or ()) if t != "files")
    if tools:                                          # the composer's Tools it was sent with
        entries[0]["tl"] = tools
    # What Prompt Prep / planning added to this message (shown under it in the chat).
    prepared = next((e.get("prepared") for e in events if e.get("phase") == "brief_sent" and e.get("prepared")), None)
    if prepared and prepared.get("additions"):
        entries[0]["p"] = prepared
    thinking = thinking_of(events)
    thought = {"k": thinking["text"], "kt": thinking["seconds"]} if thinking else {}
    status, message = result.get("status"), str(result.get("message") or "")
    # Sources (cited [n] links and the list under the reply), the search row, and the footer.
    web, src = web_of(events), sources_for(message)
    extras = {k: v for k, v in (("src", src), ("w", web), ("m", metrics_of(events, getattr(req, "started", None))),
                                ("cr", result.get("critique")), ("cmp", result.get("compare")),
                                ("rs", result.get("research")),
                                ("au", result.get("auto_tools")),        # tools it turned on by itself (auto_tools)
                                ("ck", result.get("careful"))) if v}     # Careful and honest's checks (careful.py)
    shared.app_state["last_reply_extras"] = extras        # agent/tasks._run_turn sends them with the reply
    temporary = privacy.is_temporary(chat)
    if not temporary:
        try:
            shared.usage_store.add(extras.get("m"))         # today's usage (usage.py)
        except OSError as e:
            event_log.log_action(f"Could not record usage: {e}")
    had_work = bool(steps or thought or src or web)
    thought.update(extras)
    if status in ("stopped", "error"):
        partial = str(result.get("partial") or "")        # what a stopped reply had written (agent/loop)
        if had_work or partial:
            entries.append({"r": "a", "t": partial, "s": steps, **thought})
        entries.append({"r": "s" if status == "stopped" else "e", "t": message or "Something went wrong."})
    elif status == "clarification":
        entries.append({"r": "a", "t": str(result.get("question") or message), "s": steps, **thought})
    else:
        entries.append({"r": "a", "t": message, "s": steps, **thought})
    chat["transcript"] = (chat.get("transcript") or []) + entries
    if temporary:                                        # kept in memory only (privacy.py)
        chat["updated"] = time.time()
        return thinking
    try:
        saved = shared.chat_store.save({**chat, "conversation": shared.app_state.get("conversation") or [],
                                        # compressed older turns (agent/context_window.py)
                                        "context_summary": shared.app_state.get("context_summary") or ""})
        chat["updated"] = saved["updated"]
    except OSError as e:
        event_log.log_action(f"Could not save the chat: {e}")
    return thinking


# ---- Message actions: pin, delete, edit & resend / retry (truncate), branch -------------
# The transcript (what the chat shows) and the conversation (what the model sees)
# are kept apart; a transcript entry's turn is found in the conversation by
# counting user turns from the end - host notes ("[SYSTEM] ...", observations,
# image notes) are not turns. The saved conversation may have lost its oldest
# messages (chat_store.MAX_CONVERSATION); cutting before them keeps nothing older.
# Old saved chats lack the explicit host name. Keep their synthetic notes
# inside the request that produced them, including revisions and task queues.
# [CLARIFICATION ANSWER] is deliberately absent: it represents a new user reply.
_NOT_A_TURN = ("[SYSTEM]", "[OBSERVATION from the host]", "[OBSERVATION]", "[Image from",
               "[REVIEWER FEEDBACK]", "[REMAINING TASK]", "[CURRENT TASK]", "[CURRENT TASK ", "[ANSWER]")
DELETED_REPLY = "(The user deleted this reply.)"


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return next((p.get("text") or "" for p in content if isinstance(p, dict) and p.get("type") == "text"), "")
    return str(content or "")


def _turn_starts(conversation: list) -> list:
    return [i for i, m in enumerate(conversation) if isinstance(m, dict) and m.get("role") == "user"
            and m.get("name") != "agent_host"
            and not _text_of(m).startswith(_NOT_A_TURN)]


def conv_index(conversation: list, transcript: list, index: int, after: bool = False) -> int:
    """Where the turn of transcript[index] starts in the conversation (after: where the next one starts)."""
    later = sum(1 for e in transcript[index + 1:] if e.get("r") == "u")
    starts = _turn_starts(conversation)
    k = len(starts) - 1 - later + (1 if after else 0)
    if k < 0:
        return 0
    return starts[k] if k < len(starts) else len(conversation)


def _turn_end(transcript: list, index: int) -> int:
    """The index after the last entry of transcript[index]'s turn (its reply, an error ...)."""
    j = index + 1
    while j < len(transcript) and transcript[j].get("r") != "u":
        j += 1
    return j


def edit_messages(chat_id: str, op: str, index: int, pinned: bool = False, before_open=None) -> dict:
    """pin / unpin, delete (a message of yours takes its reply with it), truncate
    (keep what came before a message of yours - Edit & resend and Retry send it
    again), branch (a new chat with everything up to and including this
    exchange, which becomes the open chat; before_open() runs just before it
    opens). Caller holds state_lock.
    Returns the chat (with transcript). Raises KeyError / ValueError."""
    temporary = privacy.is_temp_id(chat_id)
    if temporary:                                      # the open temporary chat, edited in memory
        live = shared.app_state.get("active_chat") or {}
        if live.get("id") != chat_id:
            raise KeyError(chat_id)
        if op == "branch":
            raise ValueError("A temporary chat can't be branched - save it first.")
        chat = dict(live)
    else:
        chat = shared.chat_store.load(chat_id)
    if chat is not None and op != "pin" and chat_guard.is_locked(chat):
        # Deleting, retrying or branching would get around the lock (chat_guard.py); pinning is fine.
        raise ValueError("This chat was closed for going past the chat limits - start a new chat to continue.")
    if chat is None:
        raise KeyError(chat_id)
    active = (shared.app_state.get("active_chat") or {}).get("id") == chat_id
    transcript = list(chat.get("transcript") or [])
    convo = list(shared.app_state["conversation"][1:]) if active else list(chat.get("conversation") or [])
    if not 0 <= index < len(transcript):
        raise ValueError("That message is no longer in this chat.")
    entry = transcript[index]
    if op == "pin":
        entry = dict(entry)
        if pinned:
            entry["pin"] = True
        else:
            entry.pop("pin", None)
        transcript[index] = entry
    elif op == "truncate":
        if entry.get("r") != "u":
            raise ValueError("Only a message of yours can be sent again.")
        convo = convo[:conv_index(convo, transcript, index)]
        transcript = transcript[:index]
    elif op == "delete":
        if entry.get("r") == "u":
            start, end = conv_index(convo, transcript, index), conv_index(convo, transcript, index, after=True)
            convo = convo[:start] + convo[end:]
            transcript = transcript[:index] + transcript[_turn_end(transcript, index):]
        else:
            if entry.get("r") == "a":
                start, end = conv_index(convo, transcript, index), conv_index(convo, transcript, index, after=True)
                last = next((k for k in range(end - 1, start - 1, -1) if convo[k].get("role") == "assistant"
                             and isinstance(convo[k].get("content"), str)), None)
                if last is not None:
                    convo[last] = {**convo[last], "content": DELETED_REPLY}
            transcript = transcript[:index] + transcript[index + 1:]
    elif op == "branch":
        end = _turn_end(transcript, index)
        new = {"id": shared.chat_store.new_id(), "title": chat_store.title_from(re.sub(r"\s*\(branch\)$", "", chat.get("title") or "Chat") + " (branch)"),
               "project": chat.get("project"), "created": time.time(), "branched_from": chat_id,
               "transcript": [dict(e) for e in transcript[:end]],
               "conversation": convo[:conv_index(convo, transcript, index, after=True)]}
        saved = shared.chat_store.save(new)
        if before_open:
            before_open()
        open_chat(saved["id"])
        event_log.log_action(f"Branched '{chat.get('title')}' into a new chat.")
        return saved
    else:
        raise ValueError("Unknown message action.")
    chat["transcript"] = transcript
    chat["conversation"] = convo
    if active:
        # (1.4.8) convo is the model's view (the open chat's), not the whole stored list: the store keeps what
        # came before it (chat_store._keep_older).
        chat.pop("conversation_start", None)
    if temporary:
        chat["updated"] = time.time()
        saved = {k: v for k, v in chat.items() if k != "conversation"}
    else:
        saved = shared.chat_store.save(chat)
    if active:
        shared.app_state["conversation"] = shared.app_state["conversation"][:1] + convo
        shared.app_state["active_chat"]["transcript"] = transcript
        shared.app_state["active_chat"]["updated"] = saved["updated"]
    return saved


def new_chat():
    """The next message starts a new chat (in the active project). The caller
    resets the conversation."""
    close_temporary()
    shared.app_state["active_chat"] = None
    _save_session()


# ---- Temporary chats (privacy.py) ----------------------------------------------------
def new_temporary_chat() -> dict:
    """Open a temporary chat: in memory only, in no project, never in the
    sidebar, the session file or history. The caller resets the conversation."""
    close_temporary()
    chat = {"id": privacy.new_temp_id(), "temporary": True, "title": "Temporary chat", "project": None,
            "created": time.time(), "updated": time.time(), "transcript": []}
    shared.app_state["active_chat"] = chat
    _save_session()
    return chat


def temporary_chat() -> dict | None:
    chat = shared.app_state.get("active_chat")
    return chat if privacy.is_temporary(chat) else None


def file_store():
    """Where the open chat's attachments go: the normal upload folder, or - in
    a temporary chat - the chat's own folder, deleted with it."""
    chat = temporary_chat()
    if chat is None:
        return shared.file_store
    root = os.path.join(privacy.temp_dir(chat["id"]), "uploads")
    store = shared.app_state.get("temp_file_store")
    if store is None or store.root != root:
        import composer_tools
        store = composer_tools.FileStore(root)
        shared.app_state["temp_file_store"] = store
    return store


def close_temporary() -> bool:
    """If a temporary chat is open: forget it - its folder, its events and
    audit entries, its search results. The caller resets the conversation
    (every path that closes one does). True when one was open."""
    chat = temporary_chat()
    if chat is None:
        return False
    shared.app_state["active_chat"] = None
    privacy.discard(chat["id"])                     # every module's on_discard cleanup, then its folder
    shared.app_state.pop("temp_file_store", None)
    event_log.log_action("Closed a temporary chat (nothing of it was kept).")
    return True


def save_temporary() -> dict:
    """The user asked to keep the open temporary chat: it becomes a normal saved
    chat (a new id; in the open project), and its attachments move into the
    normal upload folder. Only ever called from the user's "Save this chat"."""
    chat = temporary_chat()
    if chat is None:
        raise KeyError("No temporary chat is open.")
    if chat_guard.is_locked(chat):                         # saving it would make an unlocked copy
        raise ValueError("This chat was closed for going past the chat limits, so it can't be saved.")
    temp_store = shared.app_state.get("temp_file_store")
    moved = {}
    if temp_store is not None:
        moved = shared.file_store.adopt(temp_store)          # old id -> new id
    transcript = []
    for e in chat.get("transcript") or []:
        e = dict(e)
        if e.get("f"):
            e["f"] = [{**f, "id": moved.get(f.get("id"), f.get("id"))} for f in e["f"] if isinstance(f, dict)]
        transcript.append(e)
    first = next((e.get("t") for e in transcript if e.get("r") == "u"), "")
    saved = shared.chat_store.save({"id": shared.chat_store.new_id(), "title": chat_store.title_from(first),
                                    "project": active_project_id(), "created": chat.get("created") or time.time(),
                                    "transcript": transcript, "conversation": shared.app_state.get("conversation") or [],
                                    "context_summary": shared.app_state.get("context_summary") or ""})
    old_id = chat["id"]
    shared.app_state["active_chat"] = {k: saved.get(k) for k in ("id", "title", "project", "created", "updated", "transcript")}
    shared.app_state.pop("temp_file_store", None)
    privacy.discard(old_id, forget=False)            # saved on purpose: its Activity events may stay
    _save_session()
    event_log.log_action(f"Saved the temporary chat as '{saved.get('title')}'.")
    return saved


def open_chat(chat_id: str) -> dict | None:
    """Load a saved chat: its transcript for the page and its conversation for
    the model. Its project becomes the active one. Caller holds state_lock
    and has reset the conversation."""
    chat = shared.chat_store.load(chat_id)
    if chat is None:
        return None
    close_temporary()
    import agent.prompts as prompts
    # (1.4.8) The whole conversation is on disk; the model's view starts at conversation_start (what fitting and
    # compression had left of it), so a long chat's request is the same size as before.
    stored = [m for m in chat.get("conversation") or [] if isinstance(m, dict)]
    start = max(0, min(int(chat.get("conversation_start") or 0), len(stored)))
    shared.app_state["conversation"] = [{"role": "system", "content": prompts.SYSTEM_PROMPT}] + stored[start:]
    shared.app_state["context_summary"] = str(chat.get("context_summary") or "")
    import skills
    skills.reset_chat()                                    # skills loaded in another chat don't carry over
    shared.app_state["context_last_compression"] = None
    shared.app_state["active_chat"] = {k: chat.get(k) for k in ("id", "title", "project", "created", "updated", "transcript")}
    if chat.get("guard"):                                  # strikes / lock (chat_guard.py) stay with the chat
        shared.app_state["active_chat"]["guard"] = chat["guard"]
    if chat.get("tools_off"):                              # tool groups switched off for this chat (tools/schemas)
        shared.app_state["active_chat"]["tools_off"] = list(chat["tools_off"])
    if chat.get("sandboxed"):                              # a sandboxed chat (sandbox.py)
        shared.app_state["active_chat"]["sandboxed"] = True
    pid = chat.get("project")
    shared.app_state["active_project"] = shared.project_store.get(pid) if pid else None
    _save_session()
    event_log.log_action(f"Opened chat '{chat.get('title')}'.")
    return chat


def activate_project(pid: str | None) -> dict | None:
    """Make a project active (None: no project). A new chat starts in it."""
    project = shared.project_store.get(pid) if pid else None
    if pid and project is None:
        raise KeyError(pid)
    close_temporary()
    shared.app_state["active_project"] = project
    shared.app_state["active_chat"] = None
    _save_session()
    event_log.log_action(f"Project: {project['name'] if project else 'none'}.")
    return project


def restore():
    """At startup: reopen the chat and project that were open (if still there)."""
    try:
        with open(SESSION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    pid = data.get("project") if isinstance(data, dict) else None
    shared.app_state["active_project"] = shared.project_store.get(pid) if isinstance(pid, str) else None
    cid = data.get("chat") if isinstance(data, dict) else None
    if isinstance(cid, str) and shared.chat_store.load(cid):
        open_chat(cid)


def _project_folders() -> list:
    project = shared.app_state.get("active_project")
    return [project["folder"]] if isinstance(project, dict) and project.get("folder") else []


# "Open <place>" also finds place files in the open project's folder.
roblox_places.extra_dirs = _project_folders


def public_chat() -> dict | None:
    chat = shared.app_state.get("active_chat")
    if not chat:
        return None
    # n: the transcript's length - the page re-reads it when its own copy differs (index.html syncTranscript)
    out = {**{k: chat.get(k) for k in ("id", "title", "project", "created", "updated")}, "n": len(chat.get("transcript") or [])}
    if chat_guard.is_locked(chat):                         # closed for pushing past the chat limits
        out["locked"] = (chat.get("guard") or {}).get("reason") or True
    if privacy.is_temporary(chat):
        out.update(temporary=True, note=privacy.NOTE, remote_note=privacy.REMOTE_NOTE if remote_backend() else "")
    return out


def remote_backend() -> bool:
    """Is a remote model backend configured - could a message of this chat reach
    a cloud model (Cloud or Auto mode, or Compare with a cloud model, which any
    configured cloud API key allows)? The temporary chat's note then says so."""
    if shared.app_state.get("mode") in (config.MODE_CLOUD, config.MODE_AUTO):
        return True
    try:
        import agent.models as llm
        return bool(llm.candidate_pairs(False))
    except Exception:
        return False


def public_project() -> dict | None:
    project = shared.app_state.get("active_project")
    if not isinstance(project, dict):
        return None
    return {k: project.get(k) for k in ("id", "name", "folder", "roblox_place", "effort", "prompt_prep")}


def list_chats(project: str = "*", query: str = "", limit: int = 200) -> list:
    """project: "*" every chat, "none" chats in no project (including chats
    whose project was deleted - they are never hidden), or a project id."""
    if project == "none":
        known = {p["id"] for p in shared.project_store.list()}
        rows = shared.chat_store.list("*", query, 100000)
        return [r for r in rows if not r.get("project") or r["project"] not in known][:limit]
    return shared.chat_store.list(project, query, limit)


def recent_project_chats(pid: str, limit: int = 3) -> list:
    """The project's latest chats other than the open one (context assembly)."""
    active = (shared.app_state.get("active_chat") or {}).get("id")
    return [c for c in shared.chat_store.list(pid, "", limit + 1) if c["id"] != active][:limit]


def move_chat(chat_id: str, pid: str | None) -> dict | None:
    """Put a chat in a project (None: no project). The open chat keeps going
    in its new project."""
    if pid and shared.project_store.get(pid) is None:
        raise KeyError(pid)
    saved = shared.chat_store.set_project(chat_id, pid)
    if saved is None:
        return None
    chat = shared.app_state.get("active_chat")
    if chat and chat.get("id") == chat_id:
        chat["project"] = pid or None
        shared.app_state["active_project"] = shared.project_store.get(pid) if pid else None
        _save_session()
    return saved


def forget_active_if(chat_id: str | None = None, pid: str | None = None) -> bool:
    """After a chat or project is deleted: if it is the open one, the next
    message starts a new chat (the caller resets the conversation).
    Returns True when the open chat was affected."""
    chat = shared.app_state.get("active_chat") or {}
    project = shared.app_state.get("active_project") or {}
    hit = bool(chat_id and chat.get("id") == chat_id)
    if pid and project.get("id") == pid:
        shared.app_state["active_project"] = None
        hit = hit or chat.get("project") == pid
    if hit:
        shared.app_state["active_chat"] = None
    _save_session()
    return hit


def refresh_active_project():
    """Re-read the open project after it was edited (instructions, effort ...)."""
    project = shared.app_state.get("active_project")
    if isinstance(project, dict) and project.get("id"):
        shared.app_state["active_project"] = shared.project_store.get(project["id"])
