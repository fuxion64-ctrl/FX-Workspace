"""
core/state.py
=============
Shared in-memory runtime state: app_state, locks, the task board and action ledger.
(Split out of main.py in Phase 2 - see ARCHITECTURE.md D14.)
"""
from __future__ import annotations

import os
import threading
from pydantic import BaseModel
import chat_store as chat_store_mod
import composer_tools
import context as ctx
import mcp_client
import memory
import messaging
import projects
import routines
import run_log as run_log_mod
import usage

import agent.prompts as prompts
import core.config as config


# ==============================================================================
# State
# ==============================================================================
task_board = ctx.TaskBoard()
ledger = ctx.ActionLedger()
app_state = {
    "conversation": [{"role": "system", "content": prompts.SYSTEM_PROMPT}],
    "audit_log": [], "last_working_model": config.PRIMARY_MODELS[0], "mode": config.MODE_LOCAL,
    "events": [], "event_seq": 0,
    "waiting_for_os_approval": False, "last_shot": None, "last_shot_meta": None,
    "last_shot_at": 0.0,
    # local_model  = the one model the user pinned in the UI ("" means auto).
    # local_engaged = the one actually loaded in the router right now. They
    # differ only after a one-way text -> vision escalation.
    "local_model": "", "local_engaged": None,
    "last_tool_signature_set": None,
    "intent_guard": None,
    "pending_clarification": None,
    "resolved_clarification": None,
    "current_user_request": "",
    "turn_policy": {"mode": "none", "allow_mutation": False, "reason": "no active request"},
    "last_desktop_intent": None,
    "settings": config.load_settings(),
    # The most recent tool result that explicitly reported real, checked
    # machine state (has a 'verified' key at all). update_plan reads this
    # to refuse a silent 'done' when the last relevant action could not
    # confirm its own effect - see the comment on that handler.
    "last_verifiable_result": None,
    # Content this request read that tried to direct the assistant
    # (untrusted.py). Non-empty -> nothing side-effecting auto-runs.
    "untrusted_seen": [],
    # The open chat ({id, title, project, created, transcript}; None until the
    # first message of a new chat) and project dict (agent/sessions.py).
    "active_chat": None,
    "active_project": None,
    # Older turns of this chat, compressed into a summary, and the last
    # compression's report (agent/context_window.py).
    "context_summary": "",
    "context_last_compression": None,
}
state_lock = threading.Lock()
clarification_popup_lock = threading.Lock()


# Messaging providers (messaging.py). None ship configured, so the messaging
# tools stay unavailable (and are not offered to models) until one is.
messaging_hub = messaging.hub_from_env()
# Saved memory (memory.py): data/memory.json, edited in Settings > Memory.
memory_store = memory.MemoryStore(os.path.join(config.BASE_DIR, "data", "memory.json"))
file_store = composer_tools.FileStore(os.path.join(config.BASE_DIR, "data", "uploads"))
usage_store = usage.UsageStore(os.path.join(config.BASE_DIR, "data", "usage.json"))
# Chats (chat_store.py) and projects (projects.py); see agent/sessions.py.
chat_store = chat_store_mod.ChatStore(os.path.join(config.BASE_DIR, "data", "chats"))
project_store = projects.ProjectStore(os.path.join(config.BASE_DIR, "data", "projects"))
folder_index = projects.FolderIndex(os.path.join(config.BASE_DIR, "data", "index"))
# MCP servers (mcp_client.py): configured in the MCP tab, data/mcp_servers.json.
mcp_hub = mcp_client.Hub(os.path.join(config.BASE_DIR, "data", "mcp_servers.json"))
# Routines (routines.py): saved action steps re-run in one click or on a schedule.
routine_store = routines.RoutineStore(os.path.join(config.BASE_DIR, "data", "routines.json"))
# Past runs for the Activity panel (run_log.py): data/runs/<run id>.json.
run_log = run_log_mod.RunLog(os.path.join(config.BASE_DIR, "data", "runs"))
LAST_GOOD = {"text": None, "vision": None}   # cloud stickiness only; local pins a single model
COOLDOWN_SECONDS = 180
_cooldown_until: dict = {}


class ChatRequest(BaseModel):
    message: str
    with_vision: bool = False
    mode: str = config.MODE_LOCAL
    local_model: str = None          # the single pinned local model ("" = auto)
    local_text_model: str = None     # accepted for backwards compatibility
    local_vision_model: str = None
    command: str = None              # a slash command ("plan" | "design"), set from the message (agent/tasks.slash_command)
    # The composer's fields may arrive as null: a plain 'str = None' would refuse an explicit null (422).
    tools: list | None = None        # the composer's Tools for this message ("grounded", "research", ...)
    files: list | None = None        # attached file ids (POST /api/files, composer_tools.FileStore)
    compare_model: str | None = None  # Compare: the second model ("local:<name>" or a cloud model id)
    instructions: str | None = None  # a preset's own instructions for the model
    # The temporary chat the page is showing (privacy.py). If the server no longer has it (it restarted),
    # the message is refused instead of landing in a normal, saved chat.
    temporary_chat: str | None = None
    # The chat the page is showing when it sends (1.4.8): its id, or "" for a new chat. The server has one open chat
    # for every screen, so a message written in one chat could land in another (a second window, the phone, a
    # channel's turn); a message whose chat isn't the open one is refused and the page reopens its chat. None: not
    # checked (routines, channels, an older page).
    chat_id: str | None = None
    search_mode: str | None = None   # a per-message Web Search privacy mode (search_engine.MODES), or None
    research_depth: str | None = None   # Research's depth: auto | quick | standard | deep | ultra (agent/research.DEPTHS)
    voice: bool = False                 # said aloud in voice mode (voice.py): answer conversationally
    voice_lang: str | None = None       # the language it was said in (voice.transcribe), for the reply's language
    persona: str | None = None          # the composer's persona (personas.py): "custom", a saved one, or a built-in
    response_mode: str | None = None    # "enhanced" | "raw" for this message (response_modes.py); None: the setting


class ClarificationPopupRequest(BaseModel):
    question: str
    options: list = None
    option_values: list = None
