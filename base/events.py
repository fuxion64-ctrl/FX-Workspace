"""
events.py
=========
The audit log and the UI event stream (/api/state events) - one stream for every tool and phase.
(Split out of main.py in Phase 2 - see ARCHITECTURE.md D14.)
"""
from __future__ import annotations

import re
import threading
import time

import privacy

import core.state as shared


AUDIT_MAX = 500
_audit_lock = threading.Lock()


def log_action(msg: str):
    """Append to the audit log. Each entry carries a sequence number so the UI
    fetches only what it hasn't seen - it used to re-download all 500 entries
    every poll, and (comparing list lengths) froze once the cap was reached."""
    temp = privacy.current()
    with _audit_lock:
        shared.app_state["audit_seq"] = shared.app_state.get("audit_seq", 0) + 1
        entry = {"seq": shared.app_state["audit_seq"], "time": time.time(), "msg": msg}
        if temp:
            entry["temp"] = temp                    # purged with the temporary chat (forget_temporary)
        shared.app_state["audit_log"].append(entry)
        if len(shared.app_state["audit_log"]) > AUDIT_MAX:
            del shared.app_state["audit_log"][:-AUDIT_MAX]
    # The console can be redirected to a file: a temporary chat's content never goes there.
    print("[AUDIT] (temporary chat - details not logged)" if temp else f"[AUDIT] {msg}")


@privacy.on_discard
def forget_temporary(chat_id: str) -> int:
    """A temporary chat closed: drop its audit entries and UI events from memory. Returns how many."""
    with _audit_lock:
        before = len(shared.app_state["audit_log"])
        shared.app_state["audit_log"][:] = [e for e in shared.app_state["audit_log"] if e.get("temp") != chat_id]
        gone = before - len(shared.app_state["audit_log"])
    with _events_lock:
        events = shared.app_state.setdefault("events", [])
        before = len(events)
        events[:] = [e for e in events if e.get("temp") != chat_id]
        gone += before - len(events)
    return gone


def audit_since(seq: int, limit: int = 200) -> list:
    with _audit_lock:
        entries = [e for e in shared.app_state["audit_log"] if e["seq"] > seq]
    return entries[-limit:]


# ==============================================================================
# UI event stream: each executed tool becomes a step the chat can show live
# ("Opened Roblox Studio", "Read ServerScriptService/GameLoop" ...). The UI
# polls /api/state?since=<seq>.
# ==============================================================================
EVENTS_MAX = 400
SERVER_STARTED = time.time()
_events_lock = threading.Lock()
_TOOL_VERBS = {
    "open_application": "Open", "open_app_and_list": "Open", "click_by_name": "Click",
    "act_element": "Use control", "list_elements": "Inspect controls of", "inspect_screen": "Look at the screen",
    "click_at": "Click on screen", "type_text": "Type text", "keyboard_action": "Press",
    "execute_command": "Run command", "run_powershell": "Run PowerShell", "run_python": "Run Python", "write_file": "Write",
    "read_file": "Read", "edit_file": "Edit", "apply_patch": "Edit", "list_dir": "List folder", "search_files": "Search files in",
    "close_window": "Close", "observe": "Check the machine state", "set_plan": "Plan",
    "update_plan": "Update plan", "open_url": "Open", "download_file": "Download",
    "roblox_status": "Check Roblox Studio", "roblox_get_selection": "Read Studio selection",
    "roblox_list_children": "Browse", "roblox_get_properties": "Read properties of",
    "roblox_create_instance": "Create", "roblox_set_property": "Set property on",
    "roblox_create_script": "Create script", "roblox_delete_instance": "Delete",
    "roblox_read_script": "Read script", "roblox_set_script_source": "Edit script",
    "roblox_find": "Search the place for", "roblox_get_output": "Read Studio Output",
    "roblox_get_tree": "Read the place tree", "roblox_edit_script": "Edit script",
    "roblox_run_playtest": "Playtest the place", "roblox_batch": "Change in Studio:",
    "web_search": "Search the web for", "web_read": "Read", "fetch_page": "Read", "browser_open": "Open in the browser",
    "show_web_images": "Show pictures of",
    "browser_read": "Read the browser page", "browser_click": "Click in the browser", "browser_fill": "Fill in a field",
    "mcp_read_resource": "Read an MCP resource",
    "snap_windows": "Tile windows", "top_processes": "Check what's using the PC",
    "media_control": "Media", "schedule_task": "Schedule", "send_notification": "Notify",
}
_PRIMARY_ARG = ("app_name", "name", "window_title", "path", "filename", "directory", "command",
                "url", "action", "goal", "class_name", "parent_path", "left_title", "query")


def describe_tool(tool_name: str, args: dict) -> str:
    verb = _TOOL_VERBS.get(tool_name, tool_name.replace("_", " ").capitalize())
    args = args or {}
    if tool_name == "desktop_route":
        acts = [str(a) for a in args.get("actions") or [] if a and a != "open"]
        verb = "Close" if acts == ["close"] else "Open"
        target = str(args.get("app") or "") + (f", then {', '.join(acts)}" if acts and acts != ["close"] else "")
    elif tool_name == "save_memory":
        verb, target = "Remember", str(args.get("text") or "")
    elif tool_name.startswith("mcp__"):
        server, _, tool = tool_name[5:].partition("__")
        verb, target = f"MCP {server}:", tool
    elif tool_name == "roblox_create_instance":
        target = f"{args.get('class_name', '')} in {args.get('parent_path', '')}"
    elif tool_name == "roblox_set_property":
        target = f"{args.get('path', '')}.{args.get('property', '')}"
    elif tool_name == "roblox_batch":
        import roblox_adapter
        ops, _why = roblox_adapter.expand_batch(args.get("actions"))
        target = f"{len(ops)} change{'s' if len(ops) != 1 else ''}" if ops else "several changes"
    elif tool_name == "snap_windows":
        target = f"{args.get('left_title', '')} + {args.get('right_title', '')}"
    elif tool_name == "fetch_page":
        urls = [str(u) for u in (args.get("urls") or [])[:6]]
        target = urls[0] if len(urls) == 1 else f"{len(urls)} pages" if urls else ""
    else:
        target = next((str(args[k]) for k in _PRIMARY_ARG if args.get(k) not in (None, "", [])), "")
    target = re.sub(r"\s+", " ", target)[:90]
    return f"{verb} {target}".strip() if target and target.lower() not in verb.lower() else verb


def emit_event(kind: str, label: str, **fields) -> dict:
    temp = privacy.current()
    if temp:
        fields["temp"] = temp                       # purged with the temporary chat (forget_temporary)
    with _events_lock:
        shared.app_state["event_seq"] = shared.app_state.get("event_seq", 0) + 1
        event = {"id": shared.app_state["event_seq"], "seq": shared.app_state["event_seq"], "t": time.time(),
                 "kind": kind, "label": label, **fields}
        events = shared.app_state.setdefault("events", [])
        events.append(event)
        if len(events) > EVENTS_MAX:
            del events[:-EVENTS_MAX]
    return event


def emit_activity(phase: str, summary: str, **fields) -> dict:
    """A structured run-activity event for the UI timeline (planning, review,
    revision, handoff ...). Summaries are written for the user; model
    reasoning is never put here."""
    return emit_event("activity", summary, phase=phase, status="done", **fields)


def update_event(event: dict, **fields):
    """Change an event in place (its status stays) and bump its seq so pollers see it."""
    with _events_lock:
        shared.app_state["event_seq"] = shared.app_state.get("event_seq", 0) + 1
        event.update(fields, seq=shared.app_state["event_seq"])


def progress(**fields):
    """Live progress of the tool call running now (tools/runtime keeps it in
    app_state["running_tool_event"]): a web search's queries, the pages being
    read. The event stays 'running'."""
    event = shared.app_state.get("running_tool_event")
    if event and event.get("status") == "running":
        update_event(event, **fields)


def finish_event(event: dict, **fields):
    """Update a 'running' event in place and bump its seq so pollers see it."""
    with _events_lock:
        shared.app_state["event_seq"] = shared.app_state.get("event_seq", 0) + 1
        event.update(fields, status="done", seq=shared.app_state["event_seq"], ended=time.time())


def events_since(seq: int) -> list:
    with _events_lock:
        return [dict(e) for e in shared.app_state.get("events", []) if e["seq"] > seq]


def _announce_pause():
    emit_event("note", "Paused - press Resume to continue")
    log_action("Paused by the user.")
