"""
agent/tasks.py
==============
Turn handling: deterministic desktop routes, task queues, the orchestrator instance.
(Split out of main.py in Phase 2 - see ARCHITECTURE.md D14.)
"""
from __future__ import annotations

import queue
import re
import time
import uuid
import auto_tools
import careful
import chat_guard
import desktop_router
import intent_router
import composer_tools
import stats_tools
import corroboration
import effort
import memory
import agent_skills
import profile_store
import model_caps
import personas
import privacy
import projects
import prompt_prep
import recorder
import response_modes
import system_messages
import roblox_bridge
import roblox_places
import search_engine
import sources
import untrusted
from run_control import RunStopped, run_control
from turns import TurnManager
import orchestrator
import vision

import agent.loop as agent_loop
import agent.models as llm
import agent.research as research
import agent.routine_runner as routine_runner
import agent.sessions as sessions
import core.config as config
import core.state as shared
import events as event_log
import tools.approvals as approvals
import tools.runtime as runtime


def continue_desktop_task(intent, message) -> dict:
    """Advance the saved remainder without asking the model to choose a task."""
    shared.app_state["conversation"].append({"role": "assistant", "content": message})
    shared.app_state.setdefault("completed_app_tasks", []).append(intent.app_name)
    remaining = intent.remaining_task
    shared.app_state["current_user_request"] = remaining
    shared.app_state["resolved_clarification"] = None
    shared.app_state["pending_clarification"] = None
    next_intent = intent_router.route(remaining)
    if isinstance(next_intent, intent_router.DesktopIntent) and next_intent.app_name.casefold() != intent.app_name.casefold():
        shared.app_state["intent_guard"] = {"expected_app": next_intent.app_name}
        response = run_desktop_intent(next_intent, shared.task_board.steps[:-1], shared.task_board.goal)
    elif isinstance(next_intent, intent_router.Clarification):
        response = next_intent.as_response()
        shared.app_state["intent_guard"] = None
        shared.app_state["pending_clarification"] = {
            "source": "desktop", "question": response["question"],
            "options": response["options"], "option_values": response["option_values"],
            "original_request": remaining, "created_at": time.time(),
        }
    else:
        shared.app_state["intent_guard"] = {"expected_app": intent.app_name}
        shared.app_state["conversation"].append({"role": "user", "name": "agent_host", "content": (
            f"[REMAINING TASK] {remaining}\n"
            f"{intent.app_name} is already open. Prior completed or skipped actions must not be repeated. "
            "Inspect the app controls, carry out the remaining task with approval, and verify the result. "
            "If focus is required, focus that existing window and inspect again. "
            "Do not ask which task to do next, or claim completion just because the app opened.")})
        response = agent_loop.run_agent_loop()
    if response.get("status") != "clarification":
        response["message"] = message + "\n\n" + response.get("message", "")
    return response


def _run_desktop_route(intent) -> dict:
    """Perform an approved deterministic desktop route through the one tool
    path (registry spec 'desktop_route', executor, event stream)."""
    return runtime.execute_internal("desktop_route", {"app": intent.app_name, "actions": list(intent.actions)},
                                    runner=lambda: desktop_router.execute(intent))


def run_desktop_intent(intent, completed_steps=None, goal="") -> dict:
    """Execute a narrow, pre-resolved desktop request without model drift."""
    shared.app_state["turn_engine"] = "deterministic"
    media_action = next((a for a in intent.actions if a in {"play", "stop", "pause", "previous", "next"}), None)
    if media_action:
        shared.task_board.set_plan(intent.original, [f"{media_action.title()} current playback in {intent.app_name} through its Windows media session"])
        actions = [{
            "tool_name": "media_control", "args": {"app": intent.app_name, "action": media_action},
            "justification": ("Read the Windows media-session state and send one explicit transport command only if needed. "
                              "Previous/Next never opens or focuses a player; Play may launch a named app only when no session exists."),
        }]
        decision, approved = approvals.gate_batch(actions)
        if decision == "deny":
            shared.task_board.update(1, "failed", "Blocked by permission settings (deny).")
            return {"status": "error", "message": "'media_control' is set to DENY in your permission settings."}
        if not approved:
            shared.task_board.update(1, "failed", "User rejected media control.")
            return {"status": "error", "rejected": True, "message": "Media control was rejected."}
        result = _run_desktop_route(intent)
        shared.task_board.update(1, "done" if result.get("verified") else "failed", result.get("message", ""))
        shared.ledger.record("media_control", {"app": intent.app_name, "action": media_action}, bool(result.get("ok")), result.get("message", ""))
        event_log.log_action(f"Media control for {intent.app_name}: {result.get('message', '')}")
        return {"status": "success" if result.get("ok") else "error", "verified": bool(result.get("verified")),
                "message": result.get("message", "Media control failed."), "plan": shared.task_board.as_dict()}
    if "close" in intent.actions:
        shared.task_board.set_plan(intent.original, [f"Close {intent.app_name} if it is open"])
        actions = [{
            "tool_name": "close_window", "args": {"window_title": intent.app_name},
            "justification": f"Close the existing {intent.app_name} window only; never launch it first.",
        }]
        decision, approved = approvals.gate_batch(actions)
        if decision == "deny":
            shared.task_board.update(1, "failed", "Blocked by permission settings (deny).")
            return {"status": "error", "message": "'close_window' is set to DENY in your permission settings."}
        if not approved:
            shared.task_board.update(1, "failed", "User rejected closing the app.")
            return {"status": "error", "rejected": True, "message": "Closing the app was rejected."}
        result = _run_desktop_route(intent)
        shared.task_board.update(1, "done" if result.get("verified") else "failed", result.get("message", ""))
        shared.ledger.record("close_window", {"app": intent.app_name}, bool(result.get("ok")), result.get("message", ""))
        return {"status": "success" if result.get("ok") else "error", "verified": bool(result.get("verified")),
                "message": result.get("message", "Could not close the app."), "plan": shared.task_board.as_dict()}
    shared.app_state["last_desktop_intent"] = {
        "app_name": intent.app_name,
        "actions": list(intent.actions),
    }
    prefix = list(completed_steps or [])
    open_step = len(prefix) + 1
    steps = prefix + [f"Open or reuse {intent.app_name}"]
    if "connect" in intent.actions:
        steps.append("Connect the VPN and verify its state")
    if "disconnect" in intent.actions:
        steps.append("Disconnect the VPN and verify its state")
    if "status" in intent.actions:
        steps.append("Read the VPN connection state")
    if "play" in intent.actions:
        steps.append(f"Resume current playback in {intent.app_name} and verify")
    if "scan" in intent.actions:
        steps.append(f"Scan {intent.arguments.get('drive', 'the requested drive')} and verify results")
    if intent.remaining_task:
        steps.append(intent.remaining_task)
    shared.task_board.set_plan(goal or intent.original or f"Open {intent.app_name}", steps)

    approval_actions = [{
        "tool_name": "open_application",
        "args": {"app_name": intent.app_name},
        "justification": f"Reuse {intent.app_name} if already open; otherwise launch the installed app.",
    }]
    if "connect" in intent.actions:
        approval_actions.append({
            "tool_name": "click_by_name",
            "args": {"window_title": intent.app_name, "name": "Quick Connect"},
            "justification": "Connect the VPN using its native app control.",
        })
    if "disconnect" in intent.actions:
        approval_actions.append({
            "tool_name": "click_by_name",
            "args": {"window_title": intent.app_name, "name": "Disconnect"},
            "justification": "Disconnect the VPN using its native app control, only after confirming it is connected.",
        })
    if "play" in intent.actions:
        approval_actions.append({
            "tool_name": "click_by_name",
            "args": {"window_title": intent.app_name, "name": "Play"},
            "justification": "Check playback first. If paused, resume the current queue once; if already playing, leave it unchanged.",
        })
    if "scan" in intent.actions:
        approval_actions.append({
            "tool_name": "scan_drive",
            "args": {"app_name": intent.app_name, "drive": intent.arguments.get("drive", "")},
            "justification": "Select the requested drive in WizTree, start one scan, and verify completion.",
        })

    decision, approved = approvals.gate_batch(approval_actions)
    if decision == "deny":
        shared.task_board.update(open_step, "failed", "Blocked by permission settings (deny).")
        denied_names = ", ".join(sorted({a["tool_name"] for a in approval_actions}))
        return {"status": "error", "message": f"One or more of these actions is set to DENY in your permission settings: {denied_names}."}
    if not approved:
        shared.task_board.update(open_step, "failed", "User rejected the desktop action.")
        return {"status": "success", "rejected": True, "message": "I stopped because you rejected the proposed desktop actions.",
                "plan": shared.task_board.as_dict()}

    result = _run_desktop_route(intent)
    if result.get("clarification"):
        action = "Play music in" if "play" in intent.actions else "Connect to" if "connect" in intent.actions else "Open"
        suffix = f" and {intent.remaining_task}" if intent.remaining_task else ""
        values = [f"{action} {value}{suffix}" for value in (result.get("options") or [])]
        shared.app_state["pending_clarification"] = {
            "source": "desktop", "question": result.get("question"),
            "options": result.get("options") or [], "option_values": values,
            "original_request": intent.original, "created_at": time.time(),
        }
        return {
            "status": "clarification",
            "message": result.get("question"),
            "question": result.get("question"),
            "options": result.get("options") or [],
            "option_values": values,
            "allow_other": True,
            "plan": shared.task_board.as_dict(),
        }

    opened = bool(result.get("title"))
    shared.task_board.update(open_step, "done" if opened else "failed", result.get("message", ""))
    if any(action in intent.actions for action in ("connect", "disconnect", "status", "play", "scan")):
        shared.task_board.update(open_step + 1, "done" if result.get("verified") else "failed", result.get("message", ""))
    shared.ledger.record("desktop_intent", {"app": intent.app_name, "actions": intent.actions},
                  bool(result.get("ok")), result.get("message", ""))
    event_log.log_action(f"Desktop intent for '{intent.app_name}': "
               f"{'ok' if result.get('ok') else 'incomplete'} - {result.get('message', '')[:180]}")
    # The executor's "[status: ...]" line is for the model; this message is the user's (D48).
    message = re.sub(r"\n?\[status: [^\]]*\]", "", str(result.get("message") or "")).strip()
    if not message:
        message = f"Opened {intent.app_name}." if opened else f"Couldn't open {intent.app_name}."
    elif not opened and not result.get("ok") and not message.lower().startswith(("couldn", "could not", "can't", "cannot")):
        message = f"Couldn't open {intent.app_name}: {message[0].lower() + message[1:]}"
    if "play" in intent.actions and not result.get("ok"):
        question = message + " Retry the current queue, or skip playback?"
        options = ["Retry current queue", "Skip playback"]
        shared.app_state["pending_clarification"] = {
            "source": "playback", "question": question, "options": options,
            "option_values": options, "original_request": intent.original,
            "intent": {"app_name": intent.app_name, "actions": intent.actions,
                       "original": intent.original, "remaining_task": intent.remaining_task},
            "completed_steps": prefix, "goal": shared.task_board.goal,
            "action_step": open_step + 1, "created_at": time.time(),
        }
        return {"status": "clarification", "message": question, "question": question,
                "options": options, "option_values": options, "allow_other": True,
                "plan": shared.task_board.as_dict()}
    if intent.remaining_task:
        if not result.get("ok") or result.get("needs_agent"):
            message += f" Remaining task not executed: {intent.remaining_task}"
        else:
            return continue_desktop_task(intent, message)
    if result.get("needs_agent"):
        message += (" I stopped instead of guessing or opening the web. If the control is graphical-only, "
                    "attach a screenshot and I can use the visual fallback.")
    return {"status": "error" if result.get("interrupted") else "success",
            "message": message, "plan": shared.task_board.as_dict(),
            "verified": bool(result.get("verified")),
            "interrupted": bool(result.get("interrupted"))}


# ==============================================================================
# API
# ==============================================================================
def finish_task_queue() -> dict:
    queue = shared.app_state["task_queue"]
    items = queue["items"]
    shared.task_board.set_plan(queue["goal"], [{"text": item["text"], "status": item["status"]} for item in items])
    for step, item in zip(shared.task_board.steps, items):
        step["note"] = item.get("message", "")[:200]
    summary = "\n".join(f"- {item['status'].title()}: {item['text']} — {item.get('message') or 'Not executed.'}" for item in items)
    return {"status": "success" if all(i["status"] in {"done", "skipped"} for i in items) else "partial",
            "message": summary, "plan": shared.task_board.as_dict()}


def run_task_queue(resumed=None) -> dict:
    """Run each clause separately; factual outcomes survive model confusion."""
    shared.app_state["turn_engine"] = "deterministic"
    queue = shared.app_state["task_queue"]
    while queue["index"] < len(queue["items"]):
        item = queue["items"][queue["index"]]
        shared.app_state["current_user_request"] = item["text"]
        shared.app_state["turn_policy"] = intent_router.classify_turn(item["text"])
        routed = intent_router.deserialize_route(item["compiled"])
        if resumed is not None:
            response, resumed = resumed, None
        elif isinstance(routed, intent_router.Clarification):
            response = routed.as_response()
            shared.app_state["pending_clarification"] = {
                "source": "desktop", "question": response["question"],
                "options": response["options"], "option_values": response["option_values"],
                "original_request": item["text"], "created_at": time.time(),
            }
        else:
            shared.app_state["resolved_clarification"] = None
            shared.app_state["conversation"].append({"role": "user", "name": "agent_host", "content": (
                f"[CURRENT TASK {queue['index'] + 1}/{len(queue['items'])}] {item['text']}\n"
                "Work only on this task. The host will handle the later tasks. Do not repeat previous tasks.")})
            if isinstance(routed, intent_router.DesktopIntent):
                shared.app_state["intent_guard"] = {"expected_app": routed.app_name}
                response = run_desktop_intent(routed)
            else:
                shared.app_state["intent_guard"] = None
                shared.task_board.set_plan(item["text"], [item["text"]])
                response = agent_loop.run_agent_loop()

        if response.get("status") == "clarification":
            item["status"] = "active"
            pending = shared.app_state.get("pending_clarification")
            if not pending:
                response = {"status": "error", "message": "Clarification had no resumable state."}
            else:
                pending["queue_index"] = queue["index"]
                pending["active_plan"] = shared.task_board.as_dict()
                response["plan"] = finish_task_queue()["plan"]
                return response

        verified = response.get("verified") is True or (response.get("status") == "success" and shared.task_board.is_complete())
        item["status"] = "skipped" if response.get("skipped") else "done" if verified else "failed"
        item["message"] = response.get("message", "No result was reported.")
        if not verified and response.get("status") == "success" and not response.get("skipped"):
            item["message"] = "Completion not verified. " + item["message"]
        effective_routed = intent_router.deserialize_route(item["compiled"])
        if isinstance(effective_routed, intent_router.DesktopIntent):
            shared.app_state.setdefault("completed_app_tasks", []).append(effective_routed.app_name)
        shared.app_state["conversation"].append({"role": "assistant", "content": item["message"]})
        event_log.log_action(f"Task {queue['index'] + 1}: {item['status']} — {item['text']}")
        queue["index"] += 1
        # A rejected batch always halts. A failed VPN prerequisite or explicit
        # 'then' chain also halts. Independent media/scan failures are recorded
        # while later requested tasks remain executable.
        vpn_failed = (isinstance(effective_routed, intent_router.DesktopIntent)
                      and "connect" in effective_routed.actions and item["status"] == "failed")
        if response.get("rejected") or vpn_failed or (queue["strict"] and item["status"] == "failed"):
            for rest in queue["items"][queue["index"]:]:
                rest.update(status="failed", message="Not executed because an earlier action was rejected or a prerequisite failed.")
            queue["index"] = len(queue["items"])
    return finish_task_queue()


def resume_task_queue(pending, answer) -> dict:
    queue = shared.app_state["task_queue"]
    if pending.get("queue_index") != queue["index"]:
        return {"status": "error", "message": "This question belongs to an earlier task."}
    saved = pending.get("active_plan") or {}
    shared.app_state["turn_policy"] = intent_router.classify_turn(
        queue["items"][queue["index"]]["text"])
    shared.task_board.set_plan(saved.get("goal", ""), saved.get("steps", []))
    shared.app_state["conversation"].append({"role": "user", "content": answer})
    if intent_router.skips_current_step(answer):
        return run_task_queue({"status": "success", "skipped": True, "message": "Skipped this task at your request."})
    source = pending.get("source")
    if source == "desktop":
        intent = intent_router.resolve_answer(pending["original_request"], answer)
        if not isinstance(intent, intent_router.DesktopIntent):
            shared.app_state["pending_clarification"] = pending
            return {"status": "clarification", "question": pending["question"], "message": pending["question"],
                    "options": pending["options"], "option_values": pending["option_values"], "allow_other": True}
        queue["items"][queue["index"]]["compiled"] = intent_router.serialize_route(intent)
        shared.app_state["intent_guard"] = {"expected_app": intent.app_name}
        response = run_desktop_intent(intent)
    elif source == "playback" and answer.strip().casefold() in {"retry", "retry current queue"}:
        response = run_desktop_intent(intent_router.DesktopIntent(**pending["intent"]))
    else:
        shared.app_state["resolved_clarification"] = {"question": pending.get("question", ""), "answer": answer}
        shared.app_state["conversation"].append({"role": "user", "name": "agent_host", "content": (
            f"[ANSWER] {answer}. Continue only {queue['items'][queue['index']]['text']}. Do not ask this question again.")})
        response = agent_loop.run_agent_loop()
    return run_task_queue(response)


def _previous_turn() -> dict:
    """The request before this message and the assistant's last reply - so a
    short "yes" / "up to you" answering the assistant's question continues
    that request (intent_router.classify_turn)."""
    last = next((m for m in reversed(shared.app_state.get("conversation") or [])
                 if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip()), None)
    return {"request": shared.app_state.get("current_user_request") or "", "assistant": (last or {}).get("content", "")}


# ---- Slash commands ------------------------------------------------------------
# The commands the server handles (index.html SLASH_COMMANDS lists these with the
# page's own - /new, /clear, /model ... - and test_commands checks the two agree):
#   /plan <request>      the model writes a numbered plan and stops; nothing runs
#                        until the user approves it (or edits it) on the plan card
#                        (orchestrator: Prompt Prep's plan gate, forced on).
#   /design <request>    a visual-design turn: a UI-design system prompt, no
#                        computer tools, and HTML/CSS/SVG the chat previews live.
#   /research <question> agent/research.py: a plan card, many sources, a report.
#   /search, /summarize, /biascheck   the composer's Grounded answers / Summarize / Bias check, for one message.
#   /explain, /fix, /translate, /tldr   an instruction added to the turn (note).
SLASH_COMMANDS = {
    "plan": {"example": "set up a new Roblox place"},
    "design": {"example": "a settings page for my app"},
    "research": {"example": "the best budget GPUs for running local models"},
    "invest": {"example": "Nvidia's valuation and risks after its latest results", "tools": ["invest"]},
    "market": {"example": "how has the S&P 500 done this year?", "tools": ["market"]},
    "search": {"example": "the latest Roblox Studio release notes", "tools": ["grounded"]},
    "biascheck": {"example": "whether remote work lowers productivity", "tools": ["biascheck"]},
    "image": {"example": "a lighthouse on a cliff at dusk, oil painting", "tools": ["image"],
              "note": "IMAGE: make the image described (generate_image) - it is what the user wants from this message."},
    "summarize": {"example": "https://example.com/article", "tools": ["summarize"]},
    "explain": {"example": "how async/await works in Luau",
                "note": "EXPLAIN: explain it clearly for someone new to it - the idea first, then how it works step "
                        "by step, with a small concrete example. Define jargon when you first use it."},
    "fix": {"example": "this script: ...",
            "note": "FIX: find what is wrong, then give the corrected version in full, followed by a short list of "
                    "each change and why. If nothing is wrong, say so."},
    "translate": {"example": "to Spanish: good morning",
                  "note": "TRANSLATE: translate the text into the language the user names (English if none is named, "
                          "or from English into the user's language when the text is already English and they name "
                          "none). Keep the meaning, tone and formatting; reply with the translation only."},
    "tldr": {"example": "https://example.com/long-post",
             "note": "TL;DR: give the gist in 1-3 sentences, then at most 5 short bullets of key points. Read a linked "
                     "page with fetch_page first.", "tools": ["summarize"]},
}
_SLASH_RE = re.compile(r"^\s*/(" + "|".join(SLASH_COMMANDS) + r")(?![\w-])[ \t]*", re.I)


def slash_command(message: str) -> tuple[str | None, str]:
    """'/plan Build X' -> ('plan', 'Build X'); anything else -> (None, message)."""
    m = _SLASH_RE.match(message or "")
    if not m:
        return None, message
    return m.group(1).lower(), message[m.end():].strip()


def _design_turn(req) -> dict:
    """/design: one answer with the design as HTML (inline CSS/SVG). The
    design system prompt replaces the computer-control one for this turn
    (agent/models.build_messages reads design_mode) and no tool is offered:
    designing never touches the machine."""
    shared.app_state["current_user_request"] = req.message
    shared.app_state["turn_policy"] = {"mode": "none", "allow_mutation": False,
                                       "reason": "/design: a visual design, no computer tools"}
    for key in ("pending_clarification", "resolved_clarification", "task_queue", "intent_guard"):
        shared.app_state[key] = None
    shared.app_state["completed_app_tasks"] = []
    shared.task_board.clear()
    shared.app_state["conversation"].append({"role": "user", "content": req.message})
    shared.app_state["design_mode"] = True
    try:
        return agent_loop.run_agent_loop()
    finally:
        shared.app_state.pop("design_mode", None)


def _raw_turn(req) -> dict:
    """Raw mode (response_modes.py): the model and the conversation, nothing else. No system message, no tools
    (tools/schemas offers none), no live machine state, brief, memory or persona (agent/models.build_messages
    and system_prompt read raw_mode), no greeting shortcut, no desktop routes and no clarification resume - the
    message goes to the model exactly as typed. The agent loop still streams it, honours Stop and fits the history
    to the model's window."""
    shared.app_state["current_user_request"] = req.message
    shared.app_state["turn_policy"] = {"mode": "none", "allow_mutation": False,
                                       "reason": "Raw mode: the model alone, no tools"}
    for key in ("pending_clarification", "resolved_clarification", "task_queue", "intent_guard"):
        shared.app_state[key] = None
    shared.app_state["completed_app_tasks"] = []
    shared.task_board.clear()
    shared.app_state["conversation"].append({"role": "user", "content": req.message})
    return agent_loop.run_agent_loop()


def _classify(message: str, with_vision: bool = False, previous: dict | None = None) -> dict:
    policy = intent_router.classify_turn(message, with_vision, previous if previous is not None else _previous_turn())
    if policy.get("follows"):
        event_log.log_action(f"'{message.strip()[:60]}' answers the assistant's question - continuing: {policy['follows'][:120]}")
    return policy


def _select_request_backend(req):
    """Apply the request before any model-backed stage resolves its settings.

    Routine triggers have no selection fields and retain the active backend;
    normal ChatRequests still use their explicit defaults (Local / Auto pin).
    This is also called by chat_with_agent for direct executor callers.
    """
    mode = getattr(req, "mode", shared.app_state.get("mode", config.MODE_LOCAL))
    shared.app_state["mode"] = mode if mode in config.VALID_MODES else config.MODE_LOCAL
    if not (hasattr(req, "local_model") or hasattr(req, "local_text_model")):
        return
    chosen = getattr(req, "local_model", None)
    if chosen is None:
        chosen = getattr(req, "local_text_model", None)
    chosen = str(chosen or "").strip()
    if chosen != shared.app_state.get("local_model"):
        shared.app_state["local_model"] = chosen
        # A new pin releases a previous text-to-vision escalation immediately,
        # including for Prompt Prep, planning and Research.
        shared.app_state["local_engaged"] = None
        event_log.log_action(f"Local model pinned to '{chosen or 'auto'}'.")


def chat_with_agent(req: shared.ChatRequest):
    """One whole turn, synchronously. Runs on a TurnManager thread."""
    if not req.message.strip():
        return {"status": "error", "message": "Empty message."}
    if not shared.state_lock.acquire(blocking=False):
        return {"status": "error", "message": "Busy."}
    run_control.begin()
    try:
        # Per request (agent_loop._report_embedded_instructions); project files
        # read into this run's brief count as content this request has read.
        shared.app_state["untrusted_seen"] = list(shared.app_state.get("brief_untrusted") or [])
        # "Allow all like this for this request" (tools/approvals) never outlives the request it was given in.
        shared.app_state["request_grants"] = set()
        _select_request_backend(req)
        if shared.app_state.get("raw_mode"):
            return _raw_turn(req)
        if getattr(req, "command", None) == "design":
            return _design_turn(req)

        previous = _previous_turn()             # before this message replaces them
        shared.app_state["current_user_request"] = req.message
        shared.app_state["turn_policy"] = _classify(req.message, req.with_vision, previous)
        if intent_router.is_greeting(req.message):
            # A greeting starts fresh: what an older request left open (a question, a queue, a plan) is dropped,
            # with no computer tools (turn_policy "none") - and the model answers it. A canned "Hi! What would you
            # like me to do?" from a regex read as fake (D48).
            shared.app_state["pending_clarification"] = None
            shared.app_state["resolved_clarification"] = None
            shared.app_state["task_queue"] = None
            shared.app_state["intent_guard"] = None
            shared.app_state["completed_app_tasks"] = []
            shared.task_board.clear()
            shared.app_state["conversation"].append({"role": "user", "content": req.message})
            event_log.log_action("Greeting: cleared what older requests left open; the model answers it.")
            return agent_loop.run_agent_loop()

        # A clarification answer belongs to the request that produced the
        # question. Consume it exactly once instead of treating labels such as
        # "Proton VPN" as brand-new, context-free requests.
        pending = shared.app_state.get("pending_clarification")
        shared.app_state["pending_clarification"] = None
        shared.app_state["resolved_clarification"] = None
        if pending and intent_router.is_new_request(req.message, pending):
            pending = None
            shared.app_state["task_queue"] = None
        if pending and "queue_index" in pending and shared.app_state.get("task_queue"):
            return resume_task_queue(pending, req.message)
        if pending:
            original = pending.get("original_request") or ""
            if pending.get("source") == "playback":
                intent = intent_router.DesktopIntent(**pending["intent"])
                shared.app_state["conversation"].append({"role": "user", "content": req.message})
                if intent_router.skips_current_step(req.message):
                    open_step = next((s for s in shared.task_board.steps if s["n"] == pending["action_step"] - 1), None)
                    if open_step and open_step["status"] not in {"done", "skipped"}:
                        shared.task_board.update(open_step["n"], "skipped", "User skipped this app task.")
                    shared.task_board.update(pending["action_step"], "skipped", "Playback skipped by user; not claimed as verified.")
                    message = "Skipped playback as requested."
                    if intent.remaining_task:
                        return continue_desktop_task(intent, message)
                    return {"status": "success", "message": message, "plan": shared.task_board.as_dict()}
                if req.message.strip().casefold() in {"retry", "retry current queue"}:
                    return run_desktop_intent(intent, pending.get("completed_steps"), pending.get("goal", ""))
            resolved_intent = (intent_router.resolve_answer(original, req.message)
                               if pending.get("source") == "desktop" else None)
            if isinstance(resolved_intent, intent_router.DesktopIntent):
                shared.app_state["current_user_request"] = original or req.message
                shared.app_state["intent_guard"] = {"expected_app": resolved_intent.app_name}
                shared.app_state["conversation"].append({"role": "user", "content": req.message})
                response = run_desktop_intent(resolved_intent)
                shared.app_state["conversation"].append({"role": "assistant", "content": response.get("message", "")})
                return response

            shared.app_state["current_user_request"] = original or req.message
            shared.app_state["turn_policy"] = dict(pending.get("turn_policy") or
                                              intent_router.classify_turn(original or req.message))
            shared.app_state["resolved_clarification"] = {
                "question": pending.get("question", ""),
                "answer": req.message.strip(),
            }
            shared.app_state["intent_guard"] = None
            shared.app_state["conversation"].append({"role": "user", "content": (
                f"[CLARIFICATION ANSWER] Original request: {original}\n"
                f"The user selected: {req.message.strip()}\n"
                "Continue the original request now. Do not ask the same question again.")})
            return agent_loop.run_agent_loop()

        policy = _classify(req.message, req.with_vision, previous)
        # A reply to the assistant's question carries the request it answers,
        # so CURRENT REQUEST says what to continue - not just "Up to you".
        shared.app_state["current_user_request"] = (f"{policy['follows']} (the user replied: {req.message.strip()})"
                                                   if policy.get("follows") else req.message)
        shared.app_state["turn_policy"] = policy
        shared.app_state["completed_app_tasks"] = []
        shared.task_board.clear()
        shared.app_state["task_queue"] = None
        compiled = intent_router.compile_request(req.message)
        if len(compiled) > 1:
            shared.app_state["conversation"].append({"role": "user", "content": req.message})
            shared.app_state["task_queue"] = {"goal": req.message, "index": 0,
                                       "strict": bool(re.search(r"\bthen\b", req.message, re.I)),
                                       "items": compiled}
            return run_task_queue()
        routed = intent_router.route(req.message)
        # Short corrections such as "Connect" / "it is not connected" refer
        # to the most recent deterministic VPN request. Keep them out of the
        # small-model ambiguity loop and retry the concrete action directly.
        correction = re.sub(r"[^a-z]+", " ", req.message.lower()).strip()
        last_intent = shared.app_state.get("last_desktop_intent") or {}
        if (routed is None and last_intent.get("app_name")
                and "connect" in (last_intent.get("actions") or [])
                and correction in {"connect", "quick connect", "not connected", "it is not connected",
                                   "it s not connected", "still not connected", "vpn is not connected"}):
            routed = intent_router.DesktopIntent(
                last_intent["app_name"], ["connect"], req.message)
        if isinstance(routed, intent_router.Clarification):
            response = routed.as_response()
            shared.app_state["intent_guard"] = None
            shared.app_state["pending_clarification"] = {
                "source": "desktop",
                "question": response["question"],
                "options": response.get("options") or [],
                "option_values": response.get("option_values") or response.get("options") or [],
                "original_request": req.message,
                "created_at": time.time(),
            }
            shared.app_state["conversation"].append({"role": "user", "content": req.message})
            shared.app_state["conversation"].append({"role": "assistant", "content": response["question"]})
            event_log.log_action(f"Clarification requested before execution: {response['question']}")
            return response
        if isinstance(routed, intent_router.ChatIntent):
            # The app closes the chat (back to Home; it stays in the chat list) - no tool, no approval, no model.
            shared.app_state["intent_guard"] = None
            shared.app_state["conversation"].append({"role": "user", "content": req.message})
            message = "Closing this chat - it stays in your chat list if you want to come back to it."
            shared.app_state["conversation"].append({"role": "assistant", "content": message})
            event_log.log_action("Closed the chat on request.")
            return {"status": "success", "verified": True, "message": message, "client": {"action": "close_chat"}}
        if isinstance(routed, intent_router.DesktopIntent):
            shared.app_state["intent_guard"] = {"expected_app": routed.app_name}
            shared.app_state["conversation"].append({"role": "user", "content": req.message})
            response = run_desktop_intent(routed)
            shared.app_state["conversation"].append({"role": "assistant", "content": response.get("message", "")})
            return response
        shared.app_state["intent_guard"] = None

        user_content = req.message
        if req.with_vision:
            max_w, max_h = llm.shot_limits()
            shot, meta, _info = vision.inspect(None, max_w, max_h)
            if shot is not None:
                shared.app_state["last_shot"], shared.app_state["last_shot_meta"] = shot, meta
                shared.app_state["last_shot_at"] = time.time()
                user_content = [
                    {"type": "text", "text": req.message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{llm.encode_image(shot, for_local=llm.is_local_engine())}"}},
                ]
        shared.app_state["conversation"].append({"role": "user", "content": user_content})
        return agent_loop.run_agent_loop()
    except RunStopped:
        return agent_loop._stopped_response()
    except Exception as e:
        event_log.log_action(f"Turn failed: {e}")
        return {"status": "error", "message": f"Turn failed: {str(e)[:300]}"}
    finally:
        run_control.end()
        shared.state_lock.release()


def _analyze_request(text: str, with_vision: bool) -> dict:
    # Runs before chat_with_agent, so the previous turn is still in state.
    policy = intent_router.classify_turn(text, with_vision, _previous_turn())
    try:
        parts = len(intent_router.compile_request(text)) or 1
    except Exception:
        parts = 1
    return orchestrator.analyze(text, policy.get("mode", "none"), parts)


def _set_task_context(state, profile):
    """Hand the task brief and effort limits to the execution engine."""
    if state is None:
        budget = shared.app_state.get("run_budget")
        if budget is not None:                  # what the run used, for its saved record (_save_run)
            budget.tool_calls = runtime.tool_exec.calls
            shared.app_state["last_run_usage"] = budget.usage()
        for key in ("task_context", "iteration_budget", "effort_native", "effort_thinking", "run_budget", "prep_mode",
                    "brief_untrusted"):
            shared.app_state.pop(key, None)
        # What ran is kept for "Save as routine" (routine_runner.offer_candidate)
        # before the executor's history is reset. The next tool call outside a
        # run (e.g. an accepted suggestion) must not inherit this run's tool
        # budget or failure memory.
        shared.app_state["last_run_history"] = runtime.tool_exec.history()
        runtime.tool_exec.begin_task(None, None)
        path = recorder.finish()
        if path:
            event_log.log_action(f"Run recorded for replay: {path}")
        return
    if shared.app_state["settings"].get("record_model_io") and shared.app_state.get("mode") == config.MODE_LOCAL:
        recorder.start(state.original_request, shared.app_state["settings"], llm._current_model(), "local", state.task_id)
    shared.app_state["_protocol_announced"] = set()     # say once per run how tool calls are constrained
    if shared.app_state.get("raw_mode"):
        # Raw: nothing is added to the request - no brief, project context or memory - and the model thinks (or
        # not) by its own defaults. The run's limits still apply (time, steps, Stop).
        shared.app_state["task_context"] = ""
        shared.app_state["prep_mode"] = "off"
        shared.app_state["iteration_budget"] = {"local": profile.local_iterations, "cloud": profile.cloud_iterations}
        shared.app_state["effort_native"] = None
        shared.app_state["effort_thinking"] = None
        shared.app_state["run_budget"] = effort.RunBudget(profile, local=shared.app_state.get("mode") == config.MODE_LOCAL)
        shared.app_state["turn_engine"] = None
        return
    project_part = _project_brief(state.original_request, profile)
    memory_part = _memory_brief(state.original_request, profile.name)
    shared.app_state["task_context"] = state.brief() + project_part + memory_part + _skills_brief()
    shared.app_state["prep_mode"] = state.prep_mode
    # Shown exactly as sent: the model gets this block beside CURRENT REQUEST.
    # "prepared" is the same additions, structured, for the view under the
    # user's message in the chat ("what Prompt Prep changed").
    prepared = {"mode": state.prep_mode, "command": getattr(state, "command", None), "original": state.original_request,
                "additions": [{k: a[k] for k in ("key", "label", "items", "text") if k in a} for a in state.additions()],
                "also": [x for x, part in (("project context", project_part), ("saved memory", memory_part)) if part]}
    event_log.emit_activity("brief_sent", "Brief sent with your request", text=shared.app_state["task_context"],
                            prep=state.prep_mode, prepared=prepared)
    shared.app_state["iteration_budget"] = {"local": profile.local_iterations, "cloud": profile.cloud_iterations}
    shared.app_state["effort_native"] = profile.native
    shared.app_state["effort_thinking"] = profile.thinking
    # Live usage against this run's limits (effort.RunBudget); a level change
    # during the run is applied between tool steps (_apply_effort_change).
    shared.app_state["run_budget"] = effort.RunBudget(profile, local=shared.app_state.get("mode") == config.MODE_LOCAL)
    shared.app_state["turn_engine"] = None


def _project_brief(request: str, profile) -> str:
    """Selective project context (projects.build_context): the open project's
    instructions and notes, the file index of its folder (and of any folder
    the request names), the lines of its files that match the request and a
    summary of its recent chats - sized by the effort profile, never
    everything. File text is framed as untrusted data; text in it that tries
    to direct the assistant is shown to the user and taints this request."""
    shared.app_state["brief_untrusted"] = []
    project = shared.app_state.get("active_project")
    project = project if isinstance(project, dict) and project.get("id") else None
    if project is None and not projects.mentioned_folders(request):
        return ""
    found_in: list = []

    def frame(source, text):
        found = untrusted.find_instructions(text)
        if found:
            found_in.append((source, found))
        return untrusted.frame(source, text, found)

    try:
        recent = sessions.recent_project_chats(project["id"]) if project else []
        text, report = projects.build_context(project, request, profile.name, shared.folder_index, recent, frame,
                                              budget=projects.budget_for(profile))
    except Exception as e:                       # context is a help, never a reason to fail the run
        event_log.log_action(f"Project context unavailable: {e}")
        return ""
    if not text:
        return ""
    wanted = str((project or {}).get("roblox_place") or "").strip()
    if wanted:
        st = roblox_bridge.status_dict()
        if st.get("connected") and st.get("place") and roblox_places.place_key(st["place"]) != roblox_places.place_key(wanted):
            text += (f"\nNOTE: Roblox Studio has '{st['place']}' open, not this project's place '{wanted}'. "
                     "Ask the user before changing anything in Studio.")
    for source, found in found_in:
        agent_loop._report_embedded_instructions(source, found)
    # chat_with_agent starts the request's taint list from this (it resets
    # whatever an earlier request left in untrusted_seen).
    shared.app_state["brief_untrusted"] = [{"source": source, "snippet": found[0][:220]} for source, found in found_in]
    parts = list(report["included"])
    folders = ", ".join(f"{f['folder']} ({f['files']} files)" for f in report["folders"])
    label = (f"Project context: {report['project']['name']}" if report.get("project") else "Folder context")
    event_log.emit_activity("context", label + (f" · {len(text):,} chars" if text else ""),
                            project=report.get("project"), folders=report["folders"], items=parts,
                            budget=projects.budget_for(profile),
                            detail="\n".join(([f"Folders: {folders}"] if folders else []) + [f"- {p}" for p in parts]))
    return "\n\n" + text


def _memory_brief(request: str, level: str) -> str:
    """Saved memory relevant to this request (memory.py), marked as used and
    shown in Activity. Background facts only - nothing reads it for decisions."""
    settings = shared.app_state["settings"]
    if not settings.get("memory_enabled", True):
        return ""
    # "About you" (profile_store.py) - the user's profile, background like memory.
    about = ""
    try:
        about = profile_store.brief()
    except OSError:
        pass
    about = ("\n\n" + about) if about else ""
    project = (shared.app_state.get("active_project") or {}).get("id") if isinstance(shared.app_state.get("active_project"), dict) else shared.app_state.get("active_project")
    try:
        entries = shared.memory_store.relevant(request, level, project, sensitive=bool(settings.get("memory_sensitive")))
        shared.memory_store.mark_used([e["id"] for e in entries])
    except OSError as e:
        event_log.log_action(f"Saved memory unavailable: {e}")
        return about
    if not entries:
        return about
    event_log.emit_event("memory", f"Used {len(entries)} saved memor{'y' if len(entries) == 1 else 'ies'}",
                         status="done", ok=True, items=[{"id": e["id"], "text": e["text"]} for e in entries],
                         detail="\n".join(f"- {e['text']}" for e in entries))
    return about + "\n\n" + memory.brief(entries)


def _skills_brief() -> str:
    """The names and one-line descriptions of the user's skills (agent_skills.py); the model loads one with
    skill_load when it fits."""
    try:
        block = agent_skills.index_block()
    except OSError:
        return ""
    return ("\n\n" + block) if block else ""


def _revise(state, feedback: str):
    """Re-enter the agent loop with the reviewer's findings. Only results the
    model loop produced can be revised - a deterministic desktop route has no
    model to revise, so None is returned and the orchestrator reports what
    remains instead."""
    if shared.app_state.get("turn_engine") != "agent":
        return None
    if not shared.state_lock.acquire(timeout=5):
        return None
    try:
        shared.app_state["conversation"].append({"role": "user", "name": "agent_host", "content": feedback})
        return agent_loop.run_agent_loop()
    except RunStopped as e:
        return agent_loop._stopped_response(e.reason)
    except Exception as e:
        event_log.log_action(f"Revision failed: {e}")
        return {"status": "error", "message": f"Revision failed: {str(e)[:300]}"}
    finally:
        shared.state_lock.release()


def _board() -> list:
    return [dict(s) for s in shared.task_board.steps]


# Every cross-module reference below is looked up at call time (lambdas), so
# this can be built while the modules it depends on are still importing
# (runtime -> dispatch -> handoff -> tasks is an import cycle by design).
agent = orchestrator.AgentOrchestrator(
    execute=chat_with_agent, revise=_revise, settings=lambda: shared.app_state["settings"], run_control=run_control,
    emit=lambda *a, **k: event_log.emit_activity(*a, **k), analyze_request=_analyze_request,
    preparer=orchestrator.PromptPreparer(lambda *a, **k: llm.model_json(*a, **k)),
    planner=orchestrator.Planner(lambda *a, **k: llm.model_json(*a, **k)),
    reviewer=orchestrator.Reviewer(lambda *a, **k: llm.model_json(*a, **k),
                                   lambda s, u, m: llm.model_json(s, u, m, secondary=True)),
    tool_scope=lambda task_id, budget: runtime.tool_exec.begin_task(task_id, budget),
    tool_history=lambda: runtime.tool_exec.history(), set_context=_set_task_context,
    stopped_response=lambda reason=None: agent_loop._stopped_response(reason),
    model_name=lambda: llm._current_model(),
    capabilities=lambda model: model_caps.get(model) if shared.app_state.get("mode") == config.MODE_LOCAL else {},
    cross_check_status=lambda: llm.cross_check_status(),
    session_effort=lambda: shared.app_state.get("effort_session"),
    project=lambda: shared.app_state.get("active_project"),
    prep_mode=lambda model: prompt_prep.choose_mode(shared.app_state["settings"], model,
                                                    shared.app_state.get("active_project"),
                                                    shared.app_state.get("prep_session")),
    prep_gate=prompt_prep.gate,
    pending_answer=lambda message: _pending_answer(message))


def _pending_answer(message: str):
    """The open question this message answers (chat_with_agent consumes it the
    same way), or None when it starts a new request."""
    pending = shared.app_state.get("pending_clarification")
    if not pending or intent_router.is_new_request(message, pending):
        return None
    return pending


def _with_timelines(result):
    """The composer's Claim timelines: the reply (or research report) with "Claim timelines" added -
    who reported each claim first and who confirmed it later, from this turn's sources (corroboration.py)."""
    if ("timeline" not in (shared.app_state.get("turn_tools") or ()) or not isinstance(result, dict)
            or result.get("status") != "success" or not result.get("message")):
        return result
    section = corroboration.timelines(result["message"], sources.current().items)
    if not section:
        return result
    message = f"{str(result['message']).rstrip()}\n\n{section}"
    convo = shared.app_state.get("conversation") or []
    if convo and convo[-1].get("role") == "assistant" and convo[-1].get("content") == result["message"]:
        convo[-1] = {**convo[-1], "content": message}          # what the model sees next turn matches the chat
    return {**result, "message": message}


def _with_images(result):
    """Images made this turn (generate_image, image_gen.py) go under the reply as local image lines - saved with
    the chat, drawn by the page (renderMarkdown), and seen by the model next turn as what was made."""
    made = shared.app_state.get("turn_images") or []
    if not made or not isinstance(result, dict) or not result.get("message"):
        return result
    if result.get("status") in ("error", "stopped"):
        # The picture was made, then the reply failed or was stopped (1.4.7: "This chat is too long" after a toaster
        # picture): the picture stays in the chat - it was lost with the failed reply - and the problem is said under it.
        note = str(result.get("partial") or "").strip()
        why = str(result["message"]).strip()
        result = {**result, "status": "success", "after_error": why,
                  "message": (note + "\n\n" if note else "") + agent_loop.picture_reply("", made) +
                             f"\n\n*{'Stopped' if result.get('status') == 'stopped' else 'Then this went wrong'}: {why}*"}
        shared.app_state.setdefault("conversation", []).append({"role": "assistant", "content": result["message"]})
    lines = "\n".join(f"![{img['model']} - seed {img['seed']}]({img['url']})" for img in made)
    message = f"{str(result['message']).rstrip()}\n\n{lines}"
    convo = shared.app_state.get("conversation") or []
    if convo and convo[-1].get("role") == "assistant" and convo[-1].get("content") == result["message"]:
        convo[-1] = {**convo[-1], "content": message}
    return {**result, "message": message}


def _with_charts(result):
    """Charts made this turn (make_chart, stats_tools.py) go under the reply as ```chart blocks the page draws -
    saved with the chat and seen by the model next turn as what it showed."""
    made = shared.app_state.get("turn_charts") or []
    if not made or not isinstance(result, dict) or not result.get("message"):
        return result
    blocks = "\n\n".join(stats_tools.chart_block(c) for c in made)
    message = f"{str(result['message']).rstrip()}\n\n{blocks}"
    convo = shared.app_state.get("conversation") or []
    if convo and convo[-1].get("role") == "assistant" and convo[-1].get("content") == result["message"]:
        convo[-1] = {**convo[-1], "content": message}
    return {**result, "message": message}


def _with_care(req, result):
    """Careful and honest (careful.py) after the answer: citations to no source are removed, links no tool gave
    are flagged, a factual answer given from memory is marked - and the tools it turned on by itself are listed.
    Both go under the reply (sessions extras "ck", "au")."""
    if not isinstance(result, dict) or shared.app_state.get("raw_mode"):
        return result
    auto = auto_tools.for_reply()
    if auto:
        result = {**result, "auto_tools": auto}
    if (result.get("status") != "success" or not result.get("message")
            or not careful.on(shared.app_state.get("settings"))):
        return result
    registry = sources.current()
    text, verdict = careful.review(result["message"], getattr(req, "message", ""), registry,
                                   tools_used=bool(registry.items), web_available=search_engine.turn_mode() != "disabled")
    # Ground rule 2 (agent/prompts.GROUND_RULES), checked: an action request answered "I've opened it" while no
    # tool changed anything this turn. A deterministic desktop route reports its own result, so it never counts.
    history = shared.app_state.get("turn_tool_history") or []
    changed = any(r.get("ok") and r.get("mutating") for r in history)
    action = ((shared.app_state.get("turn_policy") or {}).get("mode") == "action"
              and not any(r.get("internal") for r in history))
    claim = careful.claims_without_change(text, action, changed)
    if claim:
        verdict["unbacked"] = claim
        event_log.emit_activity("reviewer_findings", "The reply says it changed something, but no tool did",
                                level="warn", detail=claim)
    if text != result["message"]:
        convo = shared.app_state.get("conversation") or []
        if convo and convo[-1].get("role") == "assistant" and convo[-1].get("content") == result["message"]:
            convo[-1] = {**convo[-1], "content": text}
        event_log.log_action(f"Careful and honest: removed citation(s) {verdict['removed']} that pointed at no source.")
        result = {**result, "message": text}
    if careful.worth_showing(verdict):
        result = {**result, "careful": verdict}
    return result


def _run_turn(req) -> dict:
    """One turn through the orchestrator, then saved to the open chat
    (agent/sessions.py) so it survives a restart. Saving never loses the
    answer: a failure is logged and the result is returned as is.
    A temporary chat's turn runs inside privacy.scope: every store refuses it."""
    temp = sessions.temporary_chat()
    with privacy.scope(temp["id"] if temp else None):
        return _run_turn_in_chat(req)


def _run_turn_in_chat(req) -> dict:
    # Hold the selected base instructions steady through model calls and voice playback.
    shared.app_state["turn_system_message"] = system_messages.clean_settings(shared.app_state.get("settings"))
    # The run starts with the turn, not with the agent: Stop pressed during the message check (often while the model
    # is still loading) answered "Nothing is running" and the turn went on (2026-10-27). Runs nest; a Stop set here
    # ends the agent at its first checkpoint, with the usual "Stopped" reply.
    run_control.begin()
    try:
        start_seq = shared.app_state.get("event_seq", 0)
        _select_request_backend(req)                # before prep, effort/model resolution or Research
        shared.app_state.pop("last_run_usage", None)
        shared.app_state.pop("last_reply_extras", None)
        shared.app_state.pop("turn_tool_history", None)
        # Raw mode (response_modes.py) is decided first: it changes what every later step adds (nothing). A routine
        # always runs as the agent - its steps are tool calls.
        raw = (not getattr(req, "routine_id", None)
               and response_modes.for_request(shared.app_state.get("settings"), req) == response_modes.RAW)
        shared.app_state["raw_mode"] = raw
        if hasattr(req, "response_mode"):
            req.response_mode = response_modes.RAW if raw else response_modes.ENHANCED
        sources.start_turn()                        # this reply's [n] sources (search_engine.py)
        _begin_composer_tools(req)
        # The chat limits (chat_guard.py) - before anything runs or is saved as pending. A routine is the user's
        # own saved steps, not a new message, so it isn't screened.
        refused = None if getattr(req, "routine_id", None) else _screen(req, raw)
        if not getattr(req, "routine_id", None) and refused is None:
            sessions.mark_pending(req)              # crash safety: the message is on disk before anything runs
        routine = shared.routine_store.get(req.routine_id) if getattr(req, "routine_id", None) else None
        if refused is not None:
            result = refused                        # never reaches the model or its conversation
        elif routine and routine.get("kind") == "prompt":
            # A prompt routine: its request, answered afresh with its tools (agent/routine_runner.run_prompt).
            req = routine_runner.prompt_request(routine, getattr(req, "trigger", "click"))
            _begin_composer_tools(req)
            result = routine_runner.run_prompt(routine, req, lambda r: agent.run(r, board=_board))
        elif getattr(req, "routine_id", None):
            result = routine_runner.run(req.routine_id, getattr(req, "trigger", "click"))
            if isinstance(result, dict) and routine:
                result = {**result, "routine_name": routine["name"]}
        elif not raw and (getattr(req, "command", None) in ("research", "invest")
                          or {"research", "invest"} & set(shared.app_state["turn_tools"])):
            result = research.run(req)                 # agent/research.py: plan card, many sources, a report
        else:
            shared.app_state["last_run_history"] = []
            try:
                result = agent.run(req, board=_board)
            except RunStopped as e:                  # stopped in a model call before the orchestrator's run began
                result = agent_loop._stopped_response(e.reason)
            history = shared.app_state.pop("last_run_history", None) or []
            shared.app_state["turn_tool_history"] = history          # careful.claims_without_change (_with_care)
            if (not raw and isinstance(result, dict) and result.get("status") == "success" and not result.get("rejected")
                    and not result.get("prep_cancelled")):
                candidate = routine_runner.offer_candidate(result.get("task_id"), req.message, history)
                if candidate:
                    result = {**result, "routine_candidate": candidate}
            result = _after_answer(req, result)
        result = _with_timelines(result)
        result = _with_images(result)
        result = _with_charts(result)
        result = _with_care(req, result)
        _keep_attachments()
        try:
            thinking = sessions.record_turn(req, result, start_seq)
            if isinstance(result, dict):
                result = {**result, "chat": sessions.public_chat()}
                if thinking:                        # the reply's "Thought for Xs" row, as saved
                    result["thinking"] = thinking
                result.update(shared.app_state.pop("last_reply_extras", None) or {})   # src, w, m - as saved
        except Exception as e:
            event_log.log_action(f"Could not save the chat: {e}")
        run = _save_run(req, result, start_seq)
        if run and isinstance(result, dict):
            result = {**result, "run_id": run["id"]}
        for key in ("turn_tools", "turn_files", "turn_command_note", "turn_instructions", "turn_search_mode", "turn_voice",
                    "turn_voice_lang", "turn_persona", "request_grants", "turn_images", "turn_charts", "turn_auto",
                    "raw_mode", "turn_tool_history"):    # this turn's only
            shared.app_state.pop(key, None)
        return result
    finally:
        run_control.end()
        shared.app_state.pop("turn_system_message", None)


def _guard_model(system: str, user: str) -> str:
    """One short classification answer from the model in use (chat_guard's check)."""
    return llm.model_text(system, user, max_tokens=24, label="Checking the message", temperature=0.0)["text"]


def _screen(req, raw: bool) -> dict | None:
    """Screen this message against the chat limits. Returns the turn's result when it is refused (the chat gets the
    message and a short refusal; a strike - or the lock - is saved with it), or None to answer it normally. A
    message about self-harm is answered, with care (a note for the model; Raw mode gets a fixed supportive reply)."""
    chat = shared.app_state.get("active_chat")
    previous = next((e.get("t") or "" for e in reversed((chat or {}).get("transcript") or []) if e.get("r") == "u"), "")
    try:
        verdict = chat_guard.screen(req.message, previous, _guard_model)
    except RunStopped:                          # Stop during the check: the agent's first checkpoint ends the turn
        return None
    except Exception as e:                      # the limits never cost an answer by crashing
        event_log.log_action(f"Chat limits: the check failed ({e}); the message was not screened.")
        return None
    if verdict["action"] == "allow":
        chat_guard.note_clean(chat)                 # strikes fade after enough messages within the limits
        return None
    if verdict["action"] == "care":
        event_log.log_action("Chat limits: the message may be about self-harm - answering with care.")
        if raw:
            return {"status": "success", "message": chat_guard.CARE_REPLY, "guard": {"category": "self_harm"}}
        note = shared.app_state.get("turn_instructions") or ""
        shared.app_state["turn_instructions"] = (note + "\n\n" if note else "") + chat_guard.CARE_NOTE
        return None
    chat = sessions.ensure_chat(req)
    guard = chat_guard.record_refusal(chat, verdict["category"], verdict.get("via") or "model")
    category = verdict["category"]
    state = "the chat is now closed" if guard.get("locked") else f"strike {guard['strikes'].get(category, 0)} for {category}"
    event_log.log_action(f"Chat limits: refused a message ({category}, by {verdict['via']}); {state}.")
    return {"status": "success", "message": chat_guard.refusal_text(category, guard),
            "guard": {"category": category, "strikes": chat_guard.strike_total(guard), "locked": bool(guard.get("locked"))}}


def _begin_composer_tools(req):
    """The composer's Tools for this message (composer_tools.py): which are on, a
    preset's instructions, and the attached files, read for the model."""
    command = SLASH_COMMANDS.get(getattr(req, "command", None) or "", {})
    tools = {t for t in (getattr(req, "tools", None) or []) + command.get("tools", []) if t in composer_tools.IDS}
    if shared.app_state.get("raw_mode"):
        _begin_raw_turn(req)
        return
    shared.app_state["turn_command_note"] = command.get("note", "")
    shared.app_state["turn_instructions"] = str(getattr(req, "instructions", None) or "").strip()[:4000]
    # A per-message Web Search privacy mode (the composer's Search menu), else the setting (search_engine.turn_mode).
    mode = getattr(req, "search_mode", None)
    shared.app_state["turn_search_mode"] = mode if mode in search_engine.MODES else None
    shared.app_state["turn_voice"] = bool(getattr(req, "voice", False))
    shared.app_state["turn_voice_lang"] = str(getattr(req, "voice_lang", None) or "")[:8]
    # The persona chosen in the composer (personas.py): who it is and how it talks, for the whole reply.
    shared.app_state["turn_persona"] = personas.resolve(shared.app_state.get("settings") or {}, getattr(req, "persona", None))
    if getattr(req, "persona", None) == "mossadgpt":           # MossadGPT's turns get the world monitor (osint/)
        tools.add("world")
    import image_gen
    asked = set(tools)                                          # what the user chose (chips, a preset, a command)
    auto: list = []
    if image_gen.wants_image(getattr(req, "message", "")):      # "draw a ...", "תמונה של", "нарисуй" - images on request
        tools.add("image")
        if "image" not in asked:
            auto.append({"id": "image", "why": "you asked for a picture", "by": "app"})
    if "image" in tools:
        # The model files start coming off the disk now, while the chat model writes the prompt: the first
        # drawing step then reads memory, not a hard disk (it took 248 s cold - image_gen's docstring).
        image_gen.warm()
    shared.app_state["turn_images"] = []
    shared.app_state["turn_files"] = []
    shared.app_state["turn_attachments"] = None
    ids = [f for f in (getattr(req, "files", None) or []) if isinstance(f, str)]
    if ids:
        text, images, chips = composer_tools.file_context(sessions.file_store(), ids, req.message)
        if text:
            tools.add("files")
            shared.app_state["turn_files"] = chips
            shared.app_state["turn_attachments"] = {"text": text, "images": images,
                                                    "start": len(shared.app_state["conversation"])}
    # Tools it turns on by itself (auto_tools.py) - only ones that read or make things, never with a /command
    # (that already says what the message is for). Each one is listed with why under the reply.
    if not getattr(req, "command", None):
        settings = shared.app_state.get("settings") or {}
        has_table = any(c.get("kind") == "table" for c in shared.app_state["turn_files"])
        for a in auto_tools.pick(getattr(req, "message", ""), tools, settings, auto_tools.web_ok(settings), has_table):
            tools.add(a["id"])
            auto.append({**a, "by": "app"})
    shared.app_state["turn_auto"] = auto
    shared.app_state["turn_charts"] = []
    shared.app_state["turn_tools"] = tools


def _begin_raw_turn(req):
    """Raw mode: no tools, preset, persona, voice header or auto tools for this message - only the files the user
    attached, which are part of what they sent."""
    for key, value in (("turn_command_note", ""), ("turn_instructions", ""), ("turn_search_mode", None),
                       ("turn_voice", False), ("turn_voice_lang", ""), ("turn_persona", ""), ("turn_images", []),
                       ("turn_files", []), ("turn_attachments", None), ("turn_auto", []), ("turn_charts", [])):
        shared.app_state[key] = value
    tools: set = set()
    ids = [f for f in (getattr(req, "files", None) or []) if isinstance(f, str)]
    if ids:
        text, images, chips = composer_tools.file_context(sessions.file_store(), ids, req.message)
        if text:
            tools.add("files")
            shared.app_state["turn_files"] = chips
            shared.app_state["turn_attachments"] = {"text": text, "images": images,
                                                    "start": len(shared.app_state["conversation"])}
    shared.app_state["turn_tools"] = tools


def _keep_attachments():
    """The files stay with the message they came with, so later questions about them still see them."""
    att = shared.app_state.pop("turn_attachments", None)
    if not att:
        return
    convo = shared.app_state["conversation"]
    i = next((k for k in range(max(1, att["start"]), len(convo)) if convo[k].get("role") == "user"
              and isinstance(convo[k].get("content"), str)), None)
    if i is not None:
        convo[i] = llm.attach_to(convo[i], att)


def _after_answer(req, result) -> dict:
    """Critique (a self-review and a revised answer) and Compare (a second
    model's answer) - composer tools that work on the finished answer."""
    tools = shared.app_state.get("turn_tools") or set()
    if not isinstance(result, dict) or result.get("status") != "success" or result.get("prep_cancelled") \
            or not str(result.get("message") or "").strip() or not ({"critique", "compare"} & tools):
        return result
    result = dict(result)
    try:
        if "critique" in tools:
            event_log.emit_activity("reviewer_started", "Reviewing the answer (Critique)")
            got = composer_tools.critique(llm.model_text, req.message, result["message"])
            if got and got["changed"]:
                result["critique"] = {"notes": got["notes"], "draft": got["draft"]}
                result["message"] = got["revised"]
                _replace_last_answer(got["revised"])
                event_log.emit_activity("revision_started", "Revised after the review", detail=got["notes"][:300])
            elif got:
                result["critique"] = {"notes": got["notes"], "draft": None}
                event_log.emit_activity("verification_complete", "The review found nothing to change")
        if "compare" in tools and getattr(req, "compare_model", None):
            event_log.emit_activity("handoff_requested", f"Asking {req.compare_model} too (Compare)")
            result["compare"] = composer_tools.compare(llm.model_text, req.message, req.compare_model)
    except RunStopped:
        event_log.emit_activity("run_cancelled", "Stopped by you", level="warn")
    except Exception as e:
        event_log.log_action(f"Critique/Compare failed: {e}")
    return result


def _replace_last_answer(text: str):
    convo = shared.app_state["conversation"]
    for k in range(len(convo) - 1, 0, -1):
        if convo[k].get("role") == "assistant" and isinstance(convo[k].get("content"), str):
            convo[k] = {**convo[k], "content": text}
            return


def _save_run(req, result, start_seq: int) -> dict | None:
    """Keep this run for the Activity panel's past-run search (run_log.py):
    exactly the runtime events it produced, its request, effort, model,
    outcome and usage against its limits. Never raises."""
    try:
        events = event_log.events_since(start_seq)
        if not events:
            return None
        result = result if isinstance(result, dict) else {}
        started = next((e for e in events if e.get("kind") == "activity" and e.get("phase") == "run_started"), {})
        models = [e.get("model") for e in events if e.get("kind") == "model" and e.get("model")]
        chat, project = shared.app_state.get("active_chat") or {}, shared.app_state.get("active_project") or {}
        return shared.run_log.save({
            "id": started.get("task_id") or result.get("task_id") or uuid.uuid4().hex[:12],
            "request": str(getattr(req, "message", "") or "") or (started.get("label") or ""),
            "chat": chat.get("id"), "project": project.get("id"),
            "started": events[0].get("t"), "ended": time.time(),
            "status": "rejected" if result.get("rejected") else result.get("status") or "done",
            "effort": started.get("effort"), "model": models[-1] if models else started.get("model"),
            "routine": started.get("routine"), "usage": shared.app_state.pop("last_run_usage", None),
            "events": events})
    except Exception as e:                      # history only: never costs the answer
        event_log.log_action(f"Could not save the run for Activity history: {e}")
        return None


# Runs each turn through the orchestrator on a background thread (see turns.py).
turns = TurnManager(runner=_run_turn)
# A closed temporary chat's turn records (they hold its messages) go with it.
privacy.on_discard(lambda chat_id: turns.forget(lambda t: t.get("temp") == chat_id))


def _queue_effort_change():
    """An effort change while a run is going: never applied mid-call - the
    loop picks it up at the next boundary between tool steps."""
    budget = shared.app_state.get("run_budget")
    if budget is None:
        return
    model = llm._current_model()
    level, _src = effort.choose_level(shared.app_state["settings"], model, shared.app_state.get("active_project"),
                                      shared.app_state.get("effort_session"))
    if level == "auto":
        return  # The current run already made its per-turn choice; keep it at this boundary.
    level, _why = effort.resolve(level, model_caps.get(model) if shared.app_state.get("mode") == config.MODE_LOCAL and model else {})
    budget.request_change(effort.profile_for(level, model, shared.app_state["settings"].get("effort_overrides")))
