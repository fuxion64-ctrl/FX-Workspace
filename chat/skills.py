"""
skills.py
=========
Skills: the action tools in groups, loaded when a request needs them instead of all at once.

An action turn used to send every tool it might need - 78 schemas, about 8,200 tokens, before the request was
even read ("open notepad and type hello" carried the Roblox, services, zip and scheduling tools too). Now it
sends a small core (ask_user, set_plan, update_plan, observe, load_skill) plus:

  - the skills the request picks (pick(): words in English, Hebrew and Russian - "open", "file", "powershell"),
  - the skills used earlier in this chat (kept, so the tool list - which llama.cpp caches at the start of the
    prompt - stays the same from turn to turn) until FADE_AFTER requests go by without them (D50: a long chat
    carried every skill it ever touched - 11,000 tokens before the request),
  - the skills the model loads itself with load_skill, which come with its next step,
  - the default when nothing matched: Roblox while Studio's bridge is connected, else desktop and files.

MCP servers' tools are skills too (D50): Roblox Studio's MCP is "studio" - it comes when the words are Roblox's
(not merely because Studio is open), or when loaded or used - and any other server's is "mcp". They used to ride
along on every action turn - with Roblox Studio's MCP, thousands of tokens on "open notepad".

The system prompt follows by itself: prompts.for_tools drops every sentence that names a tool not offered, so the
Roblox, browser and UI-control rules only arrive with their skills.

Skills only change what is SHOWN. Authority is still the turn's mode (schemas.tool_allowed_current_turn): a model
that calls a real tool of a skill it hasn't loaded - small models remember tool names - gets it run as before, and
that skill is loaded for the next step (note_used). Tools that belong to no skill (web and data tools offered to
questions, messaging, MCP servers, request_model_handoff) are offered exactly as before.
Settings > Agent "Load tools as needed" (skills_on) turns all of this off: every tool, every turn.
"""
from __future__ import annotations

import re

DEFAULT_SETTINGS = {"skills_on": True}

CORE = ("ask_user", "set_plan", "update_plan", "observe", "load_skill")
FADE_AFTER = 4             # requests without a skill (picked, loaded or a tool of it used) before it leaves the chat's list

# id: (one line for the index, tools)
SKILLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "desktop": ("open, focus, close and tile apps and windows; click named controls, read the screen, type and press keys",
                ("open_app_and_list", "open_application", "find_apps", "get_active_window", "wait_for_window", "list_windows",
                 "close_window", "snap_windows", "click_by_name", "list_elements", "act_element", "inspect_screen",
                 "click_at", "drag_at", "scroll_at", "keyboard_action", "type_text", "screenshot_to_file", "mini_window", "app_appearance")),
    "files": ("read, write, edit, copy, move, rename, delete, zip, find and open files and folders; disk space",
              ("write_file", "read_file", "apply_patch", "edit_file", "open_file_in_os", "list_dir", "copy_path", "move_path",
               "rename_path", "delete_path", "make_dir", "zip_paths", "unzip_file", "search_files", "file_info",
               "list_drives", "disk_usage", "compare_files", "sandbox_copy_out")),
    "shell": ("run PowerShell, cmd or Python; several steps after one approval; an unknown program in Windows Sandbox",
              ("run_powershell", "execute_command", "run_python", "run_sequence", "windows_sandbox_run")),
    "web": ("open links and tabs, download files, check URLs, search and read the web, use the assistant's own browser",
            ("open_url", "new_browser_tab", "download_file", "get_url_status", "web_search", "fetch_page", "web_read",
             "browser_open", "browser_read", "browser_click", "browser_fill")),
    "system": ("processes, services, network, ping and ports, environment variables, CPU/RAM/battery",
               ("list_processes", "top_processes", "kill_process", "list_services", "service_control", "ping_host",
                "network_info", "port_check", "get_env_var", "set_env_var", "get_system_info")),
    "clipboard": ("clipboard and media keys (play, pause, volume)",
                  ("get_clipboard", "set_clipboard", "clipboard_history", "media_control")),
    "reminders": ("notifications, reminders and scheduled tasks",
                  ("send_notification", "schedule_task", "cancel_scheduled_task", "list_scheduled_tasks")),
    "memory": ("remember a fact or preference the user asked you to keep; save a reusable skill they ask for",
               ("save_memory", "skill_save")),
    "notes": ("the user's notes, to-do tasks (with due dates) and documents in the app",
              ("notes_search", "note_save", "task_add", "task_done", "doc_list", "doc_read", "doc_write")),
    "roblox": ("Roblox Studio through its bridge: the place tree, objects, properties, scripts, playtests",
               ("roblox_status", "roblox_get_selection", "roblox_list_children", "roblox_get_properties",
                "roblox_create_instance", "roblox_set_property", "roblox_create_script", "roblox_delete_instance",
                "roblox_batch", "roblox_read_script", "roblox_set_script_source", "roblox_find", "roblox_get_output",
                "roblox_get_tree", "roblox_edit_script", "roblox_run_playtest")),
    "studio": ("Roblox Studio's own MCP tools: run Luau, search and edit the game tree, assets, playtests", ()),
    "mcp": ("the tools of your connected MCP servers", ()),         # filled live: mcp__<server>__<tool> (skill_of)
}
SKILL_OF = {tool: sid for sid, (_, tools) in SKILLS.items() for tool in tools}
# What the user sees when a skill comes in ("Added tools: Files, Commands") - not the internal ids.
NAMES = {"desktop": "Apps & windows", "files": "Files", "shell": "Commands", "web": "Web", "system": "System",
         "clipboard": "Clipboard & media", "reminders": "Reminders", "memory": "Memory", "notes": "Notes & docs", "roblox": "Roblox Studio",
         "studio": "Studio MCP", "mcp": "MCP servers"}


def names(ids) -> str:
    return ", ".join(NAMES.get(i, i) for i in ids)
DEFAULT = ("desktop", "files")

_W = r"(?<![\w-])"          # a word start that also works before Hebrew and Cyrillic
_RULES = {
    "desktop": _W + r"(?:open|launch|start|run|close|quit|exit|focus|switch to|minimi[sz]e|maximi[sz]e|window|app|click|press|"
               r"type|button|menu|tab|scroll|drag|select|screen|screenshot|notepad|chrome|edge|discord|spotify|steam|vs ?code|"
               r"settings|explorer|tile|side by side|snap|mini ?window|always on top|overlay|corner|see-?through|"
               r"transparent|dark mode|light mode|dark theme|light theme|theme|appearance|"
               r"(?:make|turn|switch|go) (?:it |the app |this app |everything )?(?:dark|light)|"
               r"פתח|תפתח|סגור|תסגור|הפעל|לחץ|תלחץ|הקלד|חלון|אפליקציה|מסך|"
               r"открой|открыть|закрой|запусти|нажми|напечатай|окно|приложение|экран)",
    "files": _W + r"(?:files?|folders?|director(?:y|ies)|documents?|downloads|desktop|txt|csv|json|pdf|docx?|xlsx?|zip|unzip|"
             r"extract|copy|move|rename|delete|remove|save|write|read|edit|create|make a|backup|disk|drive|space|path|"
             r"[a-z]:\\|\.\w{2,4}\b|"
             r"קובץ|קבצים|תיקייה|תיקיה|מסמך|העתק|העבר|מחק|שמור|שנה שם|"
             r"файл|папк|документ|скопируй|перемести|удали|сохрани|переименуй)",
    "shell": _W + r"(?:powershell|cmd|command|terminal|shell|script|python|pip|install|uninstall|winget|choco|npm|git|"
             r"registry|run (?:this|the|a) (?:command|code)|execute|"
             r"פקודה|סקריפט|התקן|команд|скрипт|установи|выполни)",
    "web": _W + r"(?:https?://|www\.|website|web ?page|site|url|link|browser|download|search|google|look up|online|log ?in|"
           r"sign ?in|form|fill|youtube|"
           r"אתר|קישור|דפדפן|הורד|חפש|сайт|ссылк|браузер|скачай|найди|поиск)",
    "system": _W + r"(?:process(?:es)?|task manager|kill|service|network|wi-?fi|internet|ping|port|ip address|dns|"
              r"environment variable|env var|cpu|ram|memory usage|battery|uptime|system info|"
              r"תהליך|שירות|רשת|סוללה|процесс|служб|сеть|батаре)",
    "clipboard": _W + r"(?:clipboard|copied|paste|volume|mute|unmute|play|pause|next (?:song|track)|previous track|music|"
                 r"לוח|הדבק|ווליום|השתק|буфер|вставь|громкост|музык)",
    "reminders": _W + r"(?:remind|reminder|notify|notification|schedule|timer|alarm|later|in \d+ (?:min|hour)|every (?:day|morning|hour)|"
                 r"tomorrow|תזכיר|תזכורת|התראה|напомни|напоминани|уведомлени)",
    "memory": _W + r"(?:remember|don't forget|keep in mind|from now on|my name is|i prefer|skills?\b|"
              r"תזכור|זכור|запомни)",
    "notes": _W + r"(?:(?:notes?|to-?dos?|tasks?|checklist|my list|docs?|documents?|draft (?:a|the|my) (?:doc|document|essay|letter)|"
             r"write (?:(?:it|this|that) )?down|jot(?: down)?)\b|"
             r"פתק|פתקים|משימה|משימות|מסמך|תרשום|заметк|задач|документ|запиши)",
    "roblox": _W + r"(?:(?:roblox|studio|luau|lua|workspace|baseplate|obby|parts?|bricks?|models?|npcs?|spawn(?:point)?|playtest|"
              r"serverscriptservice|replicatedstorage|startergui|localscript|modulescript)\b|"
              r"רובלוקס|роблокс)",
    "mcp": _W + r"(?:mcp\b|mcp server)",
}
_RULES["studio"] = _RULES["roblox"]                  # the same words; offered only while Studio's MCP is connected
_RX = {sid: re.compile(rx, re.I) for sid, rx in _RULES.items()}
# With Studio's bridge connected these mean Studio work too ("make the door open when touched" - "open" alone picks the
# desktop): the Roblox skill comes along.
_RX_STUDIO_ON = re.compile(_W + r"(?:doors?|touch(?:ed|es)?|players?|characters?|humanoids?|tweens?|leaderboards?|gui|"
                           r"guis|places?|games?|scripts?|tools?|swords?|maps?|levels?|terrain|lighting|sounds?|"
                           r"animations?|npc|spawn|kill ?brick|checkpoints?)\b", re.I)


def clean_settings(s: dict) -> dict:
    return {"skills_on": s.get("skills_on") if isinstance(s.get("skills_on"), bool) else True}


def on(settings: dict | None) -> bool:
    return clean_settings(settings or {})["skills_on"]


def _mcp_servers() -> list:
    """The connected MCP servers, by name (from the tools offered to models - schemas.mcp_schemas)."""
    try:
        from tools import schemas
        return sorted({t["function"]["name"].split("__")[1] for t in schemas.mcp_schemas()
                       if t["function"]["name"].count("__") >= 2})
    except Exception:
        return []


def _studio_server(server: str) -> bool:
    return "roblox" in server.lower() or "studio" in server.lower()


def skill_of(tool: str) -> str | None:
    """The skill a tool belongs to: its group, or for an MCP tool its server's (Roblox Studio's MCP is "studio")."""
    if isinstance(tool, str) and tool.startswith("mcp__") and tool.count("__") >= 2:
        return "studio" if _studio_server(tool.split("__")[1]) else "mcp"
    return SKILL_OF.get(tool)


def pick(message: str) -> list:
    """The skills a request's words point to, in SKILLS order. A connected MCP server is picked by its own name."""
    msg = str(message or "")
    servers = _mcp_servers()
    out = [sid for sid in SKILLS if _RX[sid].search(msg) and (sid != "studio" or any(_studio_server(s) for s in servers))]
    if "mcp" not in out and any(re.search(_W + re.escape(w), msg, re.I) for srv in servers if not _studio_server(srv)
                                for w in re.split(r"[_\-\s]+", srv) if len(w) > 3):
        out.append("mcp")
    return out


def _state() -> dict:
    import core.state as shared
    return shared.app_state


def _chat() -> list:
    """This chat's skills (a list, whatever was stored before)."""
    s = _state()
    if not isinstance(s.get("chat_skills"), list):
        s["chat_skills"] = []
    return s["chat_skills"]


def reset_chat() -> None:
    """A new or another chat: the skills used here don't carry over."""
    s = _state()
    s["chat_skills"] = []
    s["turn_skills"] = {"request": None, "ids": []}
    s["chat_skill_seen"] = {}
    s["chat_turn_no"] = 0


def _touch(ids) -> None:
    """These skills were wanted this request: they stay FADE_AFTER more requests."""
    s = _state()
    seen = s.get("chat_skill_seen") if isinstance(s.get("chat_skill_seen"), dict) else {}
    s["chat_skill_seen"] = seen
    for i in ids:
        seen[i] = int(s.get("chat_turn_no") or 0)


def active(request: str | None = None) -> list:
    """The skills shown this step: this chat's, the request's (or the default), and any loaded this turn."""
    s = _state()
    request = s.get("current_user_request") if request is None else request
    turn = s.get("turn_skills") or {}
    if turn.get("request") != request:                     # a new turn: pick for it
        n = s["chat_turn_no"] = int(s.get("chat_turn_no") or 0) + 1
        picked = pick(request)
        import roblox_bridge
        try:
            connected = roblox_bridge.is_connected()
        except Exception:
            connected = False
        chat = _chat()
        seen = s.get("chat_skill_seen") if isinstance(s.get("chat_skill_seen"), dict) else {}
        # Skills nothing has wanted for FADE_AFTER requests leave (a chat that once searched the web stopped carrying
        # the web tools forever); one stored before this existed counts from now.
        chat[:] = [i for i in chat if n - seen.setdefault(i, n) < FADE_AFTER]
        if connected and "roblox" not in picked and (not picked or _RX_STUDIO_ON.search(str(request or ""))):
            # Studio's bridge is connected and nothing else matched, or the words are Studio's: Studio work. (It used to
            # be added to EVERY request then - "what time is it" included.) Studio's MCP comes only with Studio's words.
            if _RX_STUDIO_ON.search(str(request or "")) and any(_studio_server(s) for s in _mcp_servers()):
                picked.append("studio")
            picked.append("roblox")
        if not picked:
            picked = [] if chat else list(DEFAULT)          # the chat's skills, or desktop and files in a new chat
        turn = s["turn_skills"] = {"request": request, "ids": picked}
        chat.extend(i for i in picked if i not in chat)
        _touch(picked)
    ids = set(_chat()) | set(turn.get("ids") or [])
    return [i for i in SKILLS if i in ids]


def load(ids) -> list:
    """Add skills to this chat (they fade like the rest); returns the ones that were new."""
    chat = _chat()
    ids = [i for i in ids if i in SKILLS]
    new = [i for i in ids if i not in chat]
    chat.extend(new)
    _touch(ids)
    return new


def note_used(tool: str) -> str | None:
    """A tool ran: its skill stays (or, not loaded yet, is loaded - it ran, authority is the mode's)."""
    sid = skill_of(tool)
    if sid and load([sid]):
        import events as event_log
        event_log.log_action(f"Loaded the {sid} skill: the model called {tool}.")
        event_log.emit_activity("skills_loaded", f"Added tools: {names([sid])}", detail=SKILLS[sid][0])
        return sid
    return None


def filter_tools(tools: list, settings: dict | None) -> list:
    """An action turn's tools: the core, the active skills' tools, and everything that belongs to no skill."""
    if not on(settings):
        return [t for t in tools if t["function"]["name"] != "load_skill"]
    shown, on_now = set(CORE), set(active())
    for sid in on_now:
        shown.update(SKILLS[sid][1])
    def keep(name):
        sid = skill_of(name)
        return name in shown or sid is None or sid in on_now
    out = [t for t in tools if keep(t["function"]["name"])]
    if all(t["function"]["name"].startswith("mcp__") for t in tools):
        return out                                          # an MCP list on its own (schemas.tools_for_mode): may be empty
    return out if any(t["function"]["name"] != "load_skill" for t in out) else tools


def index(offered_names) -> str:
    """The one line per skill not loaded yet that tells the model what it can still load."""
    offered = set(offered_names)
    servers = _mcp_servers()
    live = {"mcp": any(not _studio_server(s) for s in servers), "studio": any(_studio_server(s) for s in servers)}
    rest = [(sid, line) for sid, (line, tools) in SKILLS.items()
            if (not offered & set(tools) if sid not in live else
                live[sid] and not any(skill_of(n) == sid for n in offered))]
    rest = [(sid, f"{line} ({', '.join(s for s in servers if not _studio_server(s))})" if sid == "mcp" else line)
            for sid, line in rest]
    if not rest or "load_skill" not in offered:
        return ""
    return ("MORE SKILLS - tools you don't have yet. If the request needs one, call load_skill ONCE with every skill "
            "it needs (e.g. [\"files\", \"shell\"]); the tools arrive with your next step. Don't load skills \"just in "
            "case\", and never tell the user you can't do something a skill below covers: "
            + "; ".join(f"{sid} - {line}" for sid, line in rest) + ".")


def load_skill_tool(args: dict) -> dict:
    asked = args.get("skills") if isinstance(args.get("skills"), list) else [args.get("skills") or args.get("skill")]
    asked = [str(a or "").strip().lower() for a in asked if str(a or "").strip()]
    unknown = [a for a in asked if a not in SKILLS]
    new = load([a for a in asked if a in SKILLS])
    if unknown and not new:
        return {"ok": False, "status": "permanent_failure",
                "message": f"No skill called {', '.join(unknown)}. Skills: {', '.join(SKILLS)}."}
    if new:
        import events as event_log
        event_log.emit_activity("skills_loaded", f"Added tools: {names(new)}",
                                detail="; ".join(f"{NAMES.get(i, i)}: {SKILLS[i][0]}" for i in new))
    tools = sorted({t for sid in asked if sid in SKILLS for t in SKILLS[sid][1]})
    if {"mcp", "studio"} & set(asked):
        from tools import schemas
        tools += sorted(t["function"]["name"] for t in schemas.mcp_schemas() if skill_of(t["function"]["name"]) in asked)
    return {"ok": True, "loaded": new,
            "message": (f"Loaded {', '.join(new)}. " if new else "Already loaded - its tools are already in your list. ")
                       + f"These tools are yours from your next step: {', '.join(tools)}. Continue the task with them now - "
                         "don't load more unless a step needs a tool none of these has."}


SCHEMA = {"type": "function", "function": {
    "name": "load_skill",
    "description": "Loads more of your tools when the request needs a skill you don't have yet (see MORE SKILLS). "
                   "Their tools come with your next step. Never tell the user you can't do something before loading the "
                   "skill that does it.",
    "parameters": {"type": "object", "properties": {
        "skills": {"type": "array", "items": {"type": "string", "enum": list(SKILLS)}, "description": "Skill names."}},
        "required": ["skills"]}}}
