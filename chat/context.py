"""
context.py
==========
THE THIRD LEG: give the model a WORKING MEMORY and a LIVE VIEW.

The first two legs (elements.py = native control, vision.py = pixel
fallback) only answer "how do I act?". They do nothing about the three
failure modes that show up once a local model is driving:

  1. "It forgets the task."      -> nothing in the context restates the
                                    goal after a few tool round-trips, and
                                    history trimming eventually deletes the
                                    user's original request outright.
  2. "It repeats tasks."         -> nothing tells it what it already did.
                                    It re-opens an app that is already
                                    open, re-clicks a button it clicked,
                                    re-screenshots an unchanged screen.
  3. "It doesn't know the system"-> it guesses at what's running instead of
                                    looking, so it acts on a mental model
                                    that went stale several turns ago.

This module supplies all three as ONE regenerated-every-turn block that is
merged into the system message at call time and never stored in history:

  * TaskBoard    - an explicit goal + numbered steps with status. Survives
                   history trimming because it is not part of history.
  * probe_*()    - live OS truth: active window, open windows, cached
                   element handles, screenshot age.
  * ActionLedger - a signed record of every action actually executed, used
                   both to render "already done" and to hard-block literal
                   duplicate side-effecting calls.

Because the block is rebuilt from scratch on every model call, it can never
go stale and can never be trimmed away, which is exactly the property the
conversation history lacks.
"""
import hashlib
import json
import subprocess
import time

import desktop_input as di

# ==============================================================================
# Tunables
# ==============================================================================
LEDGER_MAX = 300           # entries retained in memory
LEDGER_RENDER = 10         # entries shown to the model each turn
WINDOW_RENDER = 22         # window titles shown to the model each turn
TITLE_CLIP = 64            # max chars per window title
DUPLICATE_WINDOW = 120.0   # seconds an identical side-effecting call is blocked

# Tools whose repetition is almost always an accident rather than intent.
# Clicking or pressing a key twice is frequently legitimate; launching the
# same app twice, or re-running the same shell command, essentially never is.
DEDUPE_TOOLS = {
    "open_application", "execute_command", "run_python", "write_file",
    "edit_file", "close_window", "open_file_in_os",
    # new destructive tools (generic_actions-style: repeating these is
    # almost always accidental, never intentional idempotent re-checking)
    "copy_path", "move_path", "rename_path", "delete_path",
    "zip_paths", "unzip_file", "download_file",
    "service_control", "kill_process",
    "schedule_task", "cancel_scheduled_task",
}

VALID_STATUSES = ("pending", "active", "done", "failed", "skipped")
_STATUS_TAG = {
    "pending": "[ ]", "active": "[>]", "done": "[x]",
    "failed": "[!]", "skipped": "[-]",
}


# ==============================================================================
# Task board
# ==============================================================================
class TaskBoard:
    """The agent's explicit plan. Lives outside the conversation so that
    trimming history can never delete the goal."""

    def __init__(self):
        self.goal = ""
        self.steps = []          # [{"n", "text", "status", "note"}]
        self.created = 0.0
        self.updated = 0.0

    # -- mutation ----------------------------------------------------------
    def set_plan(self, goal: str, steps) -> dict:
        clean = []
        for i, raw in enumerate(steps or [], 1):
            if isinstance(raw, dict):
                text = str(raw.get("text") or raw.get("step") or raw.get("name") or "").strip()
                status = str(raw.get("status", "pending")).lower()
            else:
                text, status = str(raw).strip(), "pending"
            if not text:
                continue
            clean.append({
                "n": i, "text": text[:220],
                "status": status if status in VALID_STATUSES else "pending",
                "note": "",
            })
        if not clean:
            return {"ok": False, "message": "A plan needs at least one non-empty step."}

        self.goal = (goal or "").strip()[:400]
        self.steps = clean
        self.created = self.updated = time.time()
        if not any(s["status"] == "active" for s in self.steps):
            # A continuation may include completed/skipped prefix steps.
            # Never reactivate finished work when rebuilding the plan.
            next_pending = next((s for s in self.steps if s["status"] == "pending"), None)
            if next_pending:
                next_pending["status"] = "active"
        return {"ok": True, "message": "Plan recorded:\n" + self.render(compact=False)}

    def update(self, step_n, status: str = None, note: str = None) -> dict:
        if not self.steps:
            return {"ok": False, "message": "No plan exists yet. Call set_plan first."}
        try:
            n = int(step_n)
        except (TypeError, ValueError):
            return {"ok": False, "message": f"step must be a step number, got {step_n!r}."}

        target = next((s for s in self.steps if s["n"] == n), None)
        if target is None:
            return {"ok": False, "message": f"No step {n}. Plan has steps 1..{len(self.steps)}."}

        if status:
            status = status.lower().strip()
            if status not in VALID_STATUSES:
                return {"ok": False, "message": f"status must be one of {', '.join(VALID_STATUSES)}."}
            target["status"] = status
        if note is not None:
            target["note"] = str(note)[:200]
        self.updated = time.time()

        # Auto-advance: finishing a step promotes the next unfinished one so
        # the model is never left with a plan that has no active step.
        if target["status"] in ("done", "skipped") and not any(s["status"] == "active" for s in self.steps):
            nxt = next((s for s in self.steps if s["status"] == "pending"), None)
            if nxt:
                nxt["status"] = "active"

        return {"ok": True, "message": "Plan updated:\n" + self.render(compact=False)}

    def clear(self):
        self.__init__()

    # -- reading -----------------------------------------------------------
    def active_step(self):
        return next((s for s in self.steps if s["status"] == "active"), None)

    def is_complete(self) -> bool:
        return bool(self.steps) and all(s["status"] in ("done", "skipped") for s in self.steps)

    def render(self, compact=True) -> str:
        if not self.steps:
            return "No plan set. If this request needs more than one action, call set_plan first."
        lines = [f"GOAL: {self.goal or '(unstated)'}"]
        for s in self.steps:
            tag = _STATUS_TAG.get(s["status"], "[ ]")
            line = f"  {tag} {s['n']}. {s['text']}"
            if s["note"]:
                line += f"   <- {s['note']}"
            lines.append(line)
        if not compact:
            return "\n".join(lines)

        active = self.active_step()
        if active:
            lines.append(f"YOUR NEXT TOOL CALL MUST ADVANCE STEP {active['n']}: {active['text']}")
        elif self.is_complete():
            lines.append("ALL STEPS ARE DONE. Stop calling tools and report the result to the user.")
        elif any(s["status"] == "failed" for s in self.steps):
            lines.append("A step just FAILED (see [!] and its note above, e.g. a user rejection) and "
                          "nothing is marked active. Do NOT guess a workaround with a different tool. Call "
                          "set_plan again with a corrected approach, or ask the user what they want instead.")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"goal": self.goal, "steps": list(self.steps),
                "complete": self.is_complete(), "updated": self.updated}


# ==============================================================================
# Action ledger
# ==============================================================================
def signature(tool_name: str, args: dict) -> str:
    """Stable identity for 'the same call again'. Bookkeeping-only fields are
    excluded so that rewording the justification cannot disguise a repeat."""
    payload = {k: v for k, v in sorted((args or {}).items())
               if k not in ("justification", "force", "reason")}
    blob = json.dumps({"t": tool_name, "a": payload}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:16]


class ActionLedger:
    def __init__(self):
        self.entries = []

    def record(self, tool_name: str, args: dict, ok: bool, message: str, verified=None) -> dict:
        # verified is None for tools that don't carry the concept at all
        # (informational reads); True/False for tools that explicitly
        # checked the machine's real state before answering. Kept SEPARATE
        # from 'ok' on purpose: 'ok' only means the call didn't error, which
        # is a much weaker claim than 'the effect was actually confirmed'.
        # Collapsing the two into one bit is exactly what let an unverified
        # click get narrated back to the user as a plain, unqualified
        # success.
        entry = {
            "sig": signature(tool_name, args), "tool": tool_name,
            "args": {k: v for k, v in (args or {}).items() if k != "justification"},
            "ok": bool(ok), "verified": verified, "message": (message or "")[:220], "at": time.time(),
        }
        self.entries.append(entry)
        if len(self.entries) > LEDGER_MAX:
            del self.entries[:-LEDGER_MAX]
        return entry

    def last_matching(self, tool_name: str, args: dict, within: float = None):
        sig = signature(tool_name, args)
        cutoff = time.time() - within if within else 0
        for entry in reversed(self.entries):
            if entry["sig"] == sig and entry["at"] >= cutoff:
                return entry
        return None

    def clear(self):
        self.entries = []

    def render(self, limit: int = LEDGER_RENDER) -> str:
        if not self.entries:
            return "Nothing executed yet this session."
        lines = []
        for entry in self.entries[-limit:]:
            stamp = time.strftime("%H:%M:%S", time.localtime(entry["at"]))
            args = json.dumps(entry["args"], ensure_ascii=False, default=str)
            if len(args) > 90:
                args = args[:87] + "..."
            if not entry["ok"]:
                verdict = "FAILED"
            elif entry.get("verified") is False:
                verdict = "OK/UNVERIFIED"   # ran without error, but did NOT confirm its effect
            elif entry.get("verified") is True:
                verdict = "OK/VERIFIED"
            else:
                verdict = "OK"              # verification concept doesn't apply to this tool
            detail = entry["message"].replace("\n", " ")[:90]
            lines.append(f"  {stamp} {entry['tool']}{args} -> {verdict}: {detail}")
        return "\n".join(lines)

    def as_list(self, limit: int = 30) -> list:
        return [{"tool": e["tool"], "ok": e["ok"], "at": e["at"],
                 "message": e["message"]} for e in self.entries[-limit:]]


def duplicate_block(ledger: ActionLedger, tool_name: str, args: dict):
    """Returns a ready-to-return tool result if this exact call is a recent
    repeat of a side-effecting call, else None. `force: true` overrides."""
    if tool_name not in DEDUPE_TOOLS or (args or {}).get("force"):
        return None
    prior = ledger.last_matching(tool_name, args, within=DUPLICATE_WINDOW)
    if not prior:
        return None
    age = int(time.time() - prior["at"])
    verdict = "succeeded" if prior["ok"] else "failed"
    return {
        "ok": False, "duplicate": True,
        "message": (
            f"SKIPPED - you already ran this exact {tool_name} call {age}s ago and it {verdict}: "
            f"{prior['message']}\n"
            "Do not repeat completed work. Check the live machine state in your system message, "
            "then either move to the next step of your plan or, if you are certain a genuine "
            "second execution is required, call it again with force=true."
        ),
    }


def soft_repeat_note(ledger: ActionLedger, tool_name: str, args: dict) -> str:
    """For tools that MAY legitimately repeat (clicks, keys), annotate rather
    than block, so the model notices a loop it did not intend."""
    if tool_name in DEDUPE_TOOLS:
        return ""
    prior = ledger.last_matching(tool_name, args, within=DUPLICATE_WINDOW)
    if not prior:
        return ""
    age = int(time.time() - prior["at"])
    return (f"\n[NOTE] This is an identical repeat of a call you made {age}s ago. "
            "If you expected a different outcome, re-inspect the screen instead of repeating again.")


# ==============================================================================
# Live environment probes
# ==============================================================================
_proc_cache = {"at": 0.0, "names": []}
_PROC_TTL = 20.0


def probe_active_window() -> str:
    if not di.IS_WINDOWS:
        return "(not Windows)"
    try:
        return di.get_active_window_title() or "(untitled)"
    except Exception:
        return "(unavailable)"


def probe_windows(limit: int = 200) -> list:
    if not di.IS_WINDOWS:
        return []
    try:
        seen, out = set(), []
        for _hwnd, title in di.enum_visible_windows():
            title = (title or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            out.append(title)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def window_is_open(name: str):
    """Live check used to stop the agent re-launching something already up."""
    if not name or not di.IS_WINDOWS:
        return None
    try:
        return di.find_window(name)
    except Exception:
        return None


def probe_processes(refresh: bool = False) -> list:
    """Process image names via tasklist, cached - useful for background apps
    that own no visible window."""
    now = time.time()
    if not refresh and _proc_cache["names"] and now - _proc_cache["at"] < _PROC_TTL:
        return _proc_cache["names"]
    if not di.IS_WINDOWS:
        return []
    try:
        res = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                             capture_output=True, text=True, timeout=15)
        names = set()
        for line in (res.stdout or "").splitlines():
            parts = line.split('","')
            if parts and parts[0].startswith('"'):
                names.add(parts[0][1:])
        ordered = sorted(names)
        _proc_cache.update(at=now, names=ordered)
        return ordered
    except Exception:
        return _proc_cache["names"]


def probe_element_handles() -> str:
    """Summarise elements.py's live handle cache without importing it at module
    load (keeps this module importable when pywinauto is missing)."""
    try:
        import elements
        registry = getattr(elements, "_REGISTRY", {}) or {}
        if not registry:
            return "none cached - call list_elements before act_element"
        now = time.time()
        fresh = [v for v in registry.values() if now - v["created"] <= getattr(elements, "_HANDLE_TTL", 300)]
        if not fresh:
            return "all cached handles have expired - call list_elements again"
        titles = {}
        for v in fresh:
            titles[v.get("name", "")] = titles.get(v.get("name", ""), 0) + 1
        newest = max(v["created"] for v in fresh)
        ids = sorted((k for k, v in registry.items() if now - v["created"] <= 300),
                     key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)
        span = f"{ids[0]}..{ids[-1]}" if len(ids) > 1 else (ids[0] if ids else "")
        return f"{len(fresh)} live handles ({span}), newest {int(now - newest)}s old"
    except Exception:
        return "unavailable"


def snapshot() -> dict:
    """Cheap structured picture of the machine right now."""
    return {"active": probe_active_window(), "windows": probe_windows(), "at": time.time()}


def snapshot_delta(before: dict, after: dict) -> str:
    """One-line summary of what actually changed on the machine, fed back to
    the model after a batch so success/failure is observed, not assumed."""
    if not before or not after:
        return ""
    prev, curr = set(before.get("windows", [])), set(after.get("windows", []))
    bits = []
    opened, closed = sorted(curr - prev), sorted(prev - curr)
    if opened:
        bits.append("new windows: " + ", ".join(t[:TITLE_CLIP] for t in opened[:5]))
    if closed:
        bits.append("closed windows: " + ", ".join(t[:TITLE_CLIP] for t in closed[:5]))
    if before.get("active") != after.get("active"):
        bits.append(f"focus moved to '{(after.get('active') or '')[:TITLE_CLIP]}'")
    return "; ".join(bits) if bits else "no window opened, closed, or took focus"


# ==============================================================================
# The block that gets merged into the system message every single turn
# ==============================================================================
# Windows' own invisible-in-practice windows: never what the user means, and they cost tokens every single turn.
# Left out of the block only - list_windows still lists everything.
_SHELL_WINDOWS = {"Program Manager", "Microsoft Text Input Application", "Windows Input Experience", "Cortana",
                  "NVIDIA GeForce Overlay", "NVIDIA GeForce Overlay DT", "Start", "Search", "Task Switching",
                  "Windows Shell Experience Host", "Default IME", "MSCTFIME UI"}


def render_live_block(board: TaskBoard, ledger: ActionLedger, runtime: dict) -> str:
    """runtime: mode, model, pywinauto, shot_meta, shot_age, budget_note.
    (1.4.8) runtime["offered"]: the tool names this turn offers (None = every hint, as before). runtime["machine"]:
    PC tools are offered. Without them only the time and any plan or done actions go - a greeting carried every open
    window's title and "call inspect_screen" for tools it didn't have (244 tokens, plan 1.4.8-model-efficiency)."""
    offered = runtime.get("offered")
    if offered is not None:
        return _render_live_trimmed(board, ledger, runtime, set(offered))
    now = time.time()
    windows = [w for w in probe_windows(WINDOW_RENDER + 16) if w not in _SHELL_WINDOWS]
    shown = [w[:TITLE_CLIP] for w in windows[:WINDOW_RENDER]]
    more = len(windows) - len(shown)

    shot_meta = runtime.get("shot_meta")
    if shot_meta:
        age = int(now - (runtime.get("shot_age") or now))
        shot_line = (f"{shot_meta.get('thumb_w')}x{shot_meta.get('thumb_h')} captured {age}s ago "
                     "- click_at/drag_at/scroll_at coordinates must be read off THAT image")
    else:
        shot_line = "none yet - call inspect_screen before any coordinate-based action"

    lines = [
        "=== LIVE MACHINE STATE (regenerated every turn - trust this over your own memory) ===",
        f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
        f"Engine: {runtime.get('mode', '?')} mode via {runtime.get('model', '?')}",
        f"Element control (UI Automation): {'available' if runtime.get('pywinauto') else 'UNAVAILABLE - use inspect_screen + click_at only'}",
        f"Focused window: '{probe_active_window()[:TITLE_CLIP]}'",
        f"Open windows ({len(windows)}): " + (" | ".join(shown) if shown else "none detected")
        + (f"  (+{more} more - call list_windows for all)" if more > 0 else ""),
        f"Cached element handles: {probe_element_handles()}",
        f"Last screenshot: {shot_line}",
    ]
    if runtime.get("activity"):
        lines.append(runtime["activity"])
    lines += [
        "",
        "=== CURRENT TASK ===",
        board.render(compact=True),
        "",
        "=== ALREADY EXECUTED THIS SESSION (do NOT redo any of these) ===",
        ledger.render(),
    ]
    if runtime.get("budget_note"):
        lines += ["", runtime["budget_note"]]
    return "\n".join(lines)


def _render_live_trimmed(board: TaskBoard, ledger: ActionLedger, runtime: dict, offered: set) -> str:
    """The live block for this turn's tools only (1.4.8)."""
    now = time.time()
    lines = ["=== LIVE STATE (regenerated every turn - trust this over your own memory) ===",
             f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"]
    if runtime.get("machine"):
        windows = [w for w in probe_windows(WINDOW_RENDER + 16) if w not in _SHELL_WINDOWS]
        shown = [w[:TITLE_CLIP] for w in windows[:WINDOW_RENDER]]
        more = len(windows) - len(shown)
        if not runtime.get("pywinauto"):
            lines.append("Element control (UI Automation): UNAVAILABLE"
                         + (" - use inspect_screen + click_at only" if {"inspect_screen", "click_at"} <= offered else ""))
        lines += [f"Focused window: '{probe_active_window()[:TITLE_CLIP]}'",
                  f"Open windows ({len(windows)}): " + (" | ".join(shown) if shown else "none detected")
                  + (f"  (+{more} more" + (" - call list_windows for all)" if "list_windows" in offered else ")")
                     if more > 0 else "")]
        if "act_element" in offered:
            handles = probe_element_handles()
            if "list_elements" not in offered:
                handles = handles.split(" - ")[0]
            lines.append(f"Cached element handles: {handles}")
        shot_meta = runtime.get("shot_meta")
        if shot_meta:
            age = int(now - (runtime.get("shot_age") or now))
            lines.append(f"Last screenshot: {shot_meta.get('thumb_w')}x{shot_meta.get('thumb_h')} captured {age}s ago "
                         "- click_at/drag_at/scroll_at coordinates must be read off THAT image")
        elif "inspect_screen" in offered and {"click_at", "drag_at", "scroll_at"} & offered:
            lines.append("Last screenshot: none yet - call inspect_screen before any coordinate-based action")
    if runtime.get("activity"):         # the app the user works in: "this" / "here" in a question means it
        lines.append(runtime["activity"])
    if board.steps:
        lines += ["", "=== CURRENT TASK ===", board.render(compact=True)]
    elif "set_plan" in offered:
        lines += ["", "=== CURRENT TASK ===", board.render(compact=True)]
    if ledger.entries:
        lines += ["", "=== ALREADY EXECUTED THIS SESSION (do NOT redo any of these) ===", ledger.render()]
    if runtime.get("budget_note"):
        lines += ["", runtime["budget_note"]]
    return "\n".join(lines)
