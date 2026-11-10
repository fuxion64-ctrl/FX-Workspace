"""
permissions.py
===============
Single source of truth for what needs the user's explicit approval before
it runs. Three layers, evaluated in this order (first match wins):

  1. LOCKDOWN takes absolute priority. It exists as a panic button: flip it
     on and EVERY tool call - even ones that are normally auto-run, like
     observe or list_windows - is held for approval, with no per-tool
     override able to relax it. That is the literal "requiring permissions
     always" contract.
  2. A per-tool override, if the user has set one for this exact tool. This
     lets someone leave everything else on its category default but pin
     ONE risky tool to always ask (or, going the other way, always
     auto-run a specific tool they trust even though its whole category
     normally asks). It also works as GOD MODE's safety valve: an override
     of 'ask' or 'deny' still holds even in god mode, which is what keeps
     god mode "almost all" rather than literally unconditional.
  3. GOD MODE, when nothing overrides it: skip approval for anything not
     already decided above.
  4. Normal mode with no override: fall back to the category default for
     this tool's risk tier (benign/input/destructive) - the exact same
     three tiers approval.py already used for the popup's colour coding.

A tool resolves to one of:
  "auto" - run immediately, no popup.
  "ask"  - goes into the batch approval popup, same as today.
  "deny" - never runs, not even a popup. The agent is told why in its own
           tool result, so it can tell the user or try something else,
           rather than the call silently vanishing.
"""
from __future__ import annotations

import os

# ==============================================================================
# Risk tiers - EVERY tool the agent can call is classified here. This is the
# one list approval.py's popup styling and this module's gating both read
# from, so the two can never quietly drift apart the way two independently
# hand-maintained sets eventually do.
# ==============================================================================
DESTRUCTIVE = {
    # pre-existing tools
    "run_python", "execute_command", "run_powershell", "write_file", "edit_file", "apply_patch",
    "close_window", "open_file_in_os",
    # filesystem & archives
    "copy_path", "move_path", "rename_path", "delete_path", "make_dir",
    "zip_paths", "unzip_file",
    # browser & web
    "download_file",
    # system / process / network
    "service_control", "kill_process", "set_env_var",
    # productivity
    "schedule_task", "cancel_scheduled_task",
    # Roblox Studio bridge
    "roblox_create_instance", "roblox_set_property", "roblox_create_script", "roblox_delete_instance", "roblox_batch",
    "roblox_set_script_source", "roblox_edit_script",
}

INPUT_LEVEL = {
    # pre-existing tools
    "click_at", "drag_at", "scroll_at", "keyboard_action", "type_text",
    "act_element", "click_by_name", "run_sequence", "set_clipboard",
    "media_control", "scan_drive",
    # browser & web - drives an already-open window with real keystrokes,
    # the same reason type_text/keyboard_action are 'input' rather than
    # 'benign'.
    "new_browser_tab",
    # window management - moves/resizes the user's windows
    "snap_windows",
    # model routing - loads a different model into the runtime
    "request_model_handoff",
    # saved memory (memory.py) - persists across chats, so the user sees it first
    "save_memory",
    # Roblox playtest: runs the place's scripts (which can reach the network or
    # DataStores) - nothing in the place changes, but it is not a read
    "roblox_run_playtest",
    # the assistant's browser (browser.py): clicks and form fills act on web pages
    "browser_click", "browser_fill",
    # reading an MCP resource runs the server's code
    "mcp_read_resource",
    # a skill shapes how future requests are done, so the user sees it before it's kept (agent_skills.py)
    "skill_save",
}

# Never automatic - not by override, not in god mode (the brief: every form
# fill, submit or click on an irreversible control needs the user's approval).
# sandbox_copy_out brings a result out of a sandboxed chat; windows_sandbox_run runs an unknown program (sandbox.py).
ALWAYS_ASK = {"browser_fill", "sandbox_copy_out", "windows_sandbox_run"}
# tool -> (args) -> a reason when THIS call must be approved (e.g. a browser
# click on a control that submits, buys or deletes). Registered by the tool's
# module; a check that fails counts as "ask".
ALWAYS_ASK_WHEN: dict = {}


def _note_replace(args: dict):
    return "it replaces a note's text" if (args or {}).get("note_id") else None


ALWAYS_ASK_WHEN["note_save"] = _note_replace

# Messaging has its own three permissions (read / draft / send), set in
# Settings > Permissions > Messaging, evaluated in decide() before the
# category defaults. Tiers here only drive the approval card's colour.
MESSAGE_KINDS = {
    "messages_list_conversations": "read", "messages_search": "read", "messages_read": "read",
    "messages_draft": "draft", "messages_send": "send",
}
DESTRUCTIVE |= {"messages_send"}
INPUT_LEVEL |= {"messages_draft"}

# Everything else - opening/focusing/reading something, or a pure query -
# is "benign" by default (classify() below falls back to it), so nothing
# needs to be listed here just to BE benign. This set exists only so
# all_known_tools() can show a complete, explicit catalogue in the
# Settings UI without silently omitting a tool that happens to need no
# special-casing.
BENIGN_EXTRA = {
    "search_files", "file_info", "list_drives", "disk_usage", "compare_files",
    "open_url", "get_url_status",
    "list_services", "ping_host", "network_info", "get_env_var",
    "get_system_info", "port_check",
    "send_notification", "list_scheduled_tasks", "clipboard_history",
    "screenshot_to_file", "top_processes",
    # Roblox Studio bridge - reads only
    "roblox_status", "roblox_get_selection", "roblox_list_children", "roblox_get_properties",
    "roblox_read_script", "roblox_find", "roblox_get_output", "roblox_get_tree",
    # web and the assistant's browser - reading and opening pages
    "web_search", "fetch_page", "web_read", "browser_open", "browser_read",
    "show_web_images",          # 1.4.7: pictures from pages, into data/web_images, shown in the chat

    # a local image into data/images (image_gen.py) - nothing leaves the PC, nothing is changed
    "generate_image",
    # statistics and charts computed by the app, and turning on more read-only tools (stats_tools, auto_tools)
    "compute_stats", "make_chart", "use_tools",
    # reading the user's notes, tasks and documents (notes.py, docs_store.py)
    "notes_search", "doc_list", "doc_read",
    # ...and adding to them: they stay in the app, and a doc keeps its old text as a version. Replacing a note's
    # text (no versions) asks - _note_replace below. After content that tried to give instructions, all of it asks.
    "note_save", "task_add", "task_done", "doc_write",
    # the knowledge base, earlier chats and saved skills - the user's own things, read only
    "kb_search", "chats_search", "skill_load",
}

RISK_TIERS = ("benign", "input", "destructive")


def classify(tool_name: str) -> str:
    # Tools of MCP servers (mcp_client.py): they can do anything their server
    # can, so they ask by default; a per-tool override is the user's explicit rule.
    if tool_name in DESTRUCTIVE or tool_name.startswith("mcp__"):
        return "destructive"
    if tool_name in INPUT_LEVEL:
        return "input"
    return "benign"


# ==============================================================================
# Settings shape + defaults. This dict lives inside settings.json (main.py's
# DEFAULT_SETTINGS merges DEFAULT_PERMISSION_SETTINGS in), so it persists
# across restarts the same way theme/layout do.
# ==============================================================================
DEFAULT_PERMISSION_SETTINGS = {
    # "normal" | "lockdown" | "god"
    "permission_mode": "normal",
    # Per-risk-tier default when there is no per-tool override, active only
    # in "normal" mode. These three defaults reproduce EXACTLY this app's
    # pre-existing, hardcoded behaviour (read-only tools auto-run,
    # everything else asks) - turning this system on changes nothing by
    # default, until someone actually opens Settings > Permissions.
    "permission_categories": {"benign": "auto", "input": "ask", "destructive": "ask"},
    # tool_name -> "auto" | "ask" | "deny". Empty by default; this is the
    # "advanced/detailed, per-tool" layer, and it always wins over the
    # category default (and over god mode) for any tool it names.
    "permission_overrides": {},
    # Personal messages: reading asks by default, drafting (stays on this
    # PC) runs, sending can only ever be "ask" or "deny".
    "messaging_permissions": {"read": "ask", "draft": "auto", "send": "ask"},
    # Folders where file writes run without asking: [{"path": "C:\\..."}].
    # See "Per-folder rules" below for what a rule can and cannot do.
    "folder_rules": [],
}

VALID_MODES = {"normal", "lockdown", "god"}
VALID_DECISIONS = {"auto", "ask", "deny"}


def clean_settings(settings: dict) -> dict:
    """Defensive normalisation: bad/missing/legacy data in settings.json
    (a hand-edited file, or an older version of this app) degrades to the
    safe default for that one field rather than raising."""
    settings = settings or {}
    mode = settings.get("permission_mode")
    if mode not in VALID_MODES:
        mode = "normal"
    categories = dict(DEFAULT_PERMISSION_SETTINGS["permission_categories"])
    given_categories = settings.get("permission_categories") or {}
    for tier in RISK_TIERS:
        value = given_categories.get(tier)
        if value in ("auto", "ask"):   # destructive/input/benign are never forced-deny by category
            categories[tier] = value
    overrides = {}
    for name, value in (settings.get("permission_overrides") or {}).items():
        if isinstance(name, str) and value in VALID_DECISIONS:
            overrides[name] = value
    messaging = dict(DEFAULT_PERMISSION_SETTINGS["messaging_permissions"])
    for kind, value in (settings.get("messaging_permissions") or {}).items():
        if kind in messaging and value in VALID_DECISIONS:
            messaging[kind] = "ask" if (kind == "send" and value == "auto") else value
    return {"permission_mode": mode, "permission_categories": categories, "permission_overrides": overrides,
            "messaging_permissions": messaging, "folder_rules": clean_folder_rules(settings.get("folder_rules"))}


# ==============================================================================
# Per-folder rules (Settings > Permissions > Folders)
# A rule can only RELAX the prompt for a file write whose every path is inside
# one of the named folders. It never reaches shell, browser, network, MCP,
# messaging, deleting, archives or anything else. Lockdown, a per-tool
# override for that tool, and a tainted request (untrusted.py) all still win.
# Writes a rule lets through are journaled and undoable like any other.
# ==============================================================================
FOLDER_WRITE_TOOLS = {"write_file", "edit_file", "apply_patch", "make_dir", "copy_path", "move_path", "rename_path"}
MAX_FOLDER_RULES = 50
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _inside(path: str, folder: str) -> bool:
    p, f = os.path.normcase(path), os.path.normcase(folder).rstrip("\\/")
    return p == f or p.startswith(f + os.sep)


def _system_folders() -> list:
    env = os.environ
    names = [env.get("SystemRoot") or r"C:\Windows", env.get("ProgramFiles"), env.get("ProgramFiles(x86)"),
             env.get("ProgramData"), env.get("ProgramW6432")]
    return [_real(n) for n in names if n]


def folder_rule_problem(path, must_exist: bool = True) -> str | None:
    """Why this folder cannot be a rule, or None if it can. A rule must name
    one specific folder the user works in - never a drive, the whole user
    profile, a system folder, or this app (the app's own code and settings).
    must_exist is checked when a rule is added, not when settings load (an
    unplugged drive must not silently delete the user's rule)."""
    if not isinstance(path, str) or not path.strip():
        return "Give a folder path."
    raw = os.path.expanduser(path.strip().strip('"'))
    if not os.path.isabs(raw) or raw.startswith(("\\\\", "//")):
        return "Use a full local path, for example C:\\Users\\you\\Projects\\game."
    real = _real(raw)
    if must_exist and not os.path.isdir(real):
        return f"{real} is not an existing folder."
    if not os.path.splitdrive(real)[1].strip("\\/"):
        return "A whole drive cannot be a rule - pick a folder inside it."
    home = _real("~")
    if _inside(home, real):
        return "Your whole user folder (or a folder above it) cannot be a rule - pick a project folder inside it."
    for sysdir in _system_folders():
        if _inside(real, sysdir) or _inside(sysdir, real):
            return f"System folders ({sysdir}) cannot be a rule."
    if _inside(real, _APP_DIR) or _inside(_APP_DIR, real):
        return "This app's own folder cannot be a rule - its code and settings always ask before changing."
    return None


def clean_folder_rules(rules) -> list:
    out, seen = [], set()
    for r in rules if isinstance(rules, list) else []:
        path = r.get("path") if isinstance(r, dict) else r
        if folder_rule_problem(path, must_exist=False) is None:
            real = _real(os.path.expanduser(path.strip().strip('"')))
            if os.path.normcase(real) not in seen:
                seen.add(os.path.normcase(real))
                out.append({"path": real})
        if len(out) >= MAX_FOLDER_RULES:
            break
    return out


def folder_write_targets(tool_name: str, args) -> list | None:
    """Every path a relaxable file write touches (source and destination for
    copy/move), resolved the way the tool will resolve it. None when the tool
    is not a relaxable write or its paths are missing/odd - that call asks."""
    a = args if isinstance(args, dict) else {}
    if tool_name in ("write_file", "edit_file", "apply_patch"):
        raw = [a.get("filename")]
    elif tool_name == "make_dir":
        raw = [a.get("path")]
    elif tool_name in ("copy_path", "move_path"):
        raw = [a.get("source"), a.get("destination")]
    elif tool_name == "rename_path":
        new = a.get("new_name")
        if not isinstance(new, str) or not new.strip() or any(c in new for c in "\\/:") or new.strip() in (".", ".."):
            return None
        raw = [a.get("path"), os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(a.get("path") or "."))), new)]
    else:
        return None
    if not all(isinstance(p, str) and p.strip() for p in raw):
        return None
    return [_real(p) for p in raw]


def _is_secret_path(path: str) -> bool:
    from tools.files import _is_secret_file        # the one secret-file rule (read_file uses it too)
    return _is_secret_file(path)


def folder_rule_for(tool_name: str, args, settings: dict) -> str | None:
    """The rule folder that lets this call run without asking, or None."""
    if tool_name not in FOLDER_WRITE_TOOLS:
        return None
    clean = clean_settings(settings)
    if clean["permission_mode"] == "lockdown" or clean["permission_overrides"].get(tool_name):
        return None
    folders = [r["path"] for r in clean["folder_rules"]]
    targets = folder_write_targets(tool_name, args) if folders else None
    if not targets:
        return None
    matched = None
    for t in targets:
        home = next((f for f in folders if _inside(t, _real(f))), None)
        if home is None or _inside(t, _APP_DIR) or _is_secret_path(t):
            return None
        matched = matched or home
    return matched


# Pure reads that stay local. After the assistant has read content that tried
# to instruct it (untrusted.py), only these may still run without asking:
# anything that changes the machine OR reaches the network (a URL can carry
# data out) goes to the approval card instead, whatever the settings say.
LOCAL_READS = {
    "observe", "list_elements", "inspect_screen", "find_apps", "list_windows", "list_processes",
    "get_active_window", "wait_for_window", "read_file", "list_dir", "search_files", "file_info",
    "list_drives", "disk_usage", "compare_files", "list_services", "network_info", "get_system_info",
    "list_scheduled_tasks", "top_processes",
    "roblox_status", "roblox_get_selection", "roblox_list_children", "roblox_get_properties",
    "roblox_read_script", "roblox_find", "roblox_get_output", "roblox_get_tree",
    # reads the page already open in the assistant's browser; sends nothing anywhere
    "browser_read",
}
HOST_BOOKKEEPING = {"set_plan", "update_plan", "ask_user"}


def decide(tool_name: str, settings: dict, tainted: bool = False, args=None) -> str:
    """Returns 'auto', 'ask', or 'deny' for one tool call under the given
    settings dict (same shape as DEFAULT_PERMISSION_SETTINGS)."""
    return explain(tool_name, settings, tainted=tainted, args=args)[0]


def explain(tool_name: str, settings: dict, tainted: bool = False, args=None) -> tuple:
    """(decision, reason) - the reason is what the Activity log says.

    args: the call's arguments; only per-folder rules read them.
    tainted: this request read content that tried to direct the assistant.
    It can only tighten the answer ('auto' -> 'ask'), never loosen it."""
    decision, reason = _decide(tool_name, settings), "permission settings"
    if decision != "deny":
        if tool_name in ALWAYS_ASK:
            return "ask", "this always needs your approval"
        check = ALWAYS_ASK_WHEN.get(tool_name)
        if check is not None:
            try:
                why = check(args or {})
            except Exception:
                why = "it could not be checked"
            if why:
                return "ask", why
    if decision == "ask" and args is not None:
        folder = folder_rule_for(tool_name, args, settings)
        if folder:
            decision, reason = "auto", f"folder rule: {folder}"
    if tainted and decision == "auto" and tool_name not in LOCAL_READS | HOST_BOOKKEEPING:
        return "ask", "content read earlier in this request tried to give instructions"
    return decision, reason


def request_grantable(tool_name: str, settings: dict, tainted: bool = False, args=None) -> bool:
    """Whether "Allow all like this for this request" (the approval card) may cover this call: only an
    ordinary ask from the permission settings. Never an always-ask tool or check (form fills, a click that
    submits), sending a message, lockdown mode, or a request that read content trying to give the
    assistant instructions - those ask every time."""
    if tainted or tool_name in ALWAYS_ASK or MESSAGE_KINDS.get(tool_name) == "send":
        return False
    if clean_settings(settings)["permission_mode"] == "lockdown":
        return False
    decision, reason = explain(tool_name, settings, tainted=False, args=args)
    return decision == "ask" and reason == "permission settings"


def _decide(tool_name: str, settings: dict) -> str:
    clean = clean_settings(settings)
    if clean["permission_mode"] == "lockdown":
        return "ask"
    kind = MESSAGE_KINDS.get(tool_name)
    if kind:
        chosen = clean["permission_overrides"].get(tool_name) or clean["messaging_permissions"][kind]
        if kind == "send":
            # Sending a message on the user's behalf is never automatic - not
            # by override, not in god mode.
            return "deny" if chosen == "deny" else "ask"
        if clean["permission_mode"] == "god" and chosen != "deny":
            return "auto"
        return chosen
    override = clean["permission_overrides"].get(tool_name)
    if override:
        return override
    if clean["permission_mode"] == "god":
        return "auto"
    return clean["permission_categories"].get(classify(tool_name), "ask")


def partition(calls: list, settings: dict):
    """Splits a batch of (tool_name, ...) tuples/lists - or plain tool-name
    strings - into (auto, ask, denied) using decide() on each item. Item
    order is preserved within each bucket."""
    auto, ask, denied = [], [], []
    for item in calls:
        name = item[0] if isinstance(item, (tuple, list)) else item
        bucket = decide(name, settings)
        if bucket == "auto":
            auto.append(item)
        elif bucket == "deny":
            denied.append(item)
        else:
            ask.append(item)
    return auto, ask, denied


def batch_decision(tool_names: list, settings: dict, tainted: bool = False) -> str:
    """For a FIXED, indivisible batch of actions (the desktop-router fast
    path builds one of these per step, e.g. 'open app + click Connect'):
    'deny' if anything in it is denied, 'auto' only if EVERY item is auto,
    else 'ask'. Never splits a single logical step into a partially
    approved batch."""
    decisions = {decide(name, settings, tainted=tainted) for name in tool_names}
    if "deny" in decisions:
        return "deny"
    if decisions == {"auto"}:
        return "auto"
    return "ask"


def all_known_tools(tool_names) -> list:
    """For the Settings > Permissions UI: every known tool with its risk
    tier, sourced from the CURRENT tool set rather than a second,
    separately hand-maintained list that could fall out of sync."""
    return [{"name": name, "tier": classify(name)} for name in sorted(tool_names)]