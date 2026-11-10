"""
tools/dispatch.py
=================
execute_tool: the implementation of every tool, plus shell helpers.
(Split out of main.py in Phase 2 - see ARCHITECTURE.md D14.)
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import app_registry
import browser
import context as ctx
import desktop_input as di
import elements
import fs_tools
import patch_edit
import permissions
import privacy
import productivity_tools
import roblox_adapter  # noqa: F401  (registers the Studio adapter)
import roblox_bridge
import roblox_places
from run_control import run_control
import messaging
import agent_skills
import docs_store
import knowledge
import notes
import profile_store
import search_engine
import sys_tools
import vision
import web_tools
import win_signals

import agent.models as llm
import core.config as config
import core.state as shared
import events as event_log
import routing.handoff as handoff
import tools.files as files


# ==============================================================================
# Tool execution
# ==============================================================================
def _screen_changed_since_last(shot) -> bool:
    """Cheap whole-screen diff against the previous capture."""
    prev = shared.app_state.get("last_shot")
    if prev is None or shot is None:
        return True
    try:
        a = prev.copy(); a.thumbnail((320, 200))
        b = shot.copy(); b.thumbnail((320, 200))
        score = vision._diff_score(a, b)
        return score is None or score >= config.SCREEN_UNCHANGED_THRESHOLD
    except Exception:
        return True


def _clip_output(stdout: str, stderr: str, limit: int = 3000) -> str:
    """Command output for the model. Keeps the start AND the end: the error
    that explains a failure is almost always printed last, and the old
    head-only cut removed exactly that part."""
    out = (stdout or "") + (("\n[stderr]\n" + stderr) if stderr else "")
    if len(out) <= limit:
        return out
    head, tail = out[:limit // 3], out[-(limit - limit // 3):]
    return f"{head}\n... [{len(out) - limit} chars omitted] ...\n{tail}"


def _run_powershell(command: str, timeout_seconds=None) -> dict:
    """Windows PowerShell 5.1, non-interactive, no profile. The command is
    passed as -EncodedCommand (UTF-16LE base64), so quotes, $ and newlines in
    it are never mangled by a second layer of quoting; output is UTF-8."""
    if not config.IS_WINDOWS:
        return {"ok": False, "message": "run_powershell is Windows only."}
    try:
        timeout = max(5, min(300, int(timeout_seconds or 60)))
    except (TypeError, ValueError):
        timeout = 60
    script = ("$ProgressPreference = 'SilentlyContinue'; "
              "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
              "$OutputEncoding = [Console]::OutputEncoding;\n" + command)
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return _run_cancellable(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                             "-EncodedCommand", encoded], shell=False, timeout=timeout, encoding="utf-8")


def _run_cancellable(cmd, shell: bool, timeout: float, encoding: str = None) -> dict:
    """Run a process the user can stop: Stop kills it instead of waiting up to
    `timeout` for it to finish. errors="replace": a program printing bytes
    outside the console code page used to fail the whole tool."""
    import sandbox
    try:
        # A sandboxed chat: Low integrity, in a job object, in its scratch folder (sandbox.py) - or not at all.
        proc = sandbox.boxed_popen(cmd, shell, encoding)
    except Exception as e:
        return {"ok": False, "status": "permanent_failure",
                "message": f"The sandbox couldn't restrict this command, so it didn't run ({e})."}
    if proc is None:
        proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, errors="replace", encoding=encoding)
    try:
        return _wait_cancellable(proc, timeout)
    finally:
        if isinstance(proc, sandbox.BoxedProcess):
            proc.close()


def _wait_cancellable(proc, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, err = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if run_control.is_stopped():
                _kill_tree(proc)
                out, err = proc.communicate()
                return {"ok": False, "status": "cancelled",
                        "message": f"Stopped by you; the process was terminated.\n{_clip_output(out, err)}"}
            if time.monotonic() >= deadline:
                _kill_tree(proc)
                out, err = proc.communicate()
                return {"ok": False, "status": "permanent_failure",
                        "message": f"Timed out after {timeout:.0f}s and was terminated.\n{_clip_output(out, err)}"}
    return {"ok": proc.returncode == 0,
            "message": f"Exit Code: {proc.returncode}\nOutput:\n{_clip_output(out, err)}"}


def _kill_tree(proc):
    """shell=True runs cmd.exe; killing only it would orphan the real process."""
    try:
        if config.IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _turn_tables() -> list:
    """This turn's attached files with their paths - stats_tools reads CSV tables by name."""
    import agent.sessions as sessions
    out = []
    for chip in shared.app_state.get("turn_files") or []:
        meta = sessions.file_store().get(chip.get("id")) or shared.file_store.get(chip.get("id"))
        if meta:
            out.append(meta)
    return out


def _browser_fallback(settings: dict) -> bool:
    """web_search/web_read may use the assistant's browser when a site refuses
    them: only with the setting on AND when browser_open itself would run
    without asking - the fallback never gets around the user's rule for it."""
    return (settings.get("web_browser_fallback", True) is not False
            and permissions.decide("browser_open", settings, tainted=bool(shared.app_state.get("untrusted_seen")),
                                   args={}) == "auto")


def execute_tool(tool_name: str, args: dict) -> dict:
    # A sandboxed chat (sandbox.py): files only in its scratch folder, no screen control.
    import sandbox
    args, refused = sandbox.confine(tool_name, args)
    if refused:
        return {"ok": False, "status": "permanent_failure", "message": refused}
    try:
        if tool_name == "sandbox_copy_out":
            return sandbox.copy_out(args.get("path", ""), args.get("destination", ""))
        if tool_name == "windows_sandbox_run":
            return sandbox.windows_sandbox_run(args.get("path", ""))
        # -------- messaging (messaging.py) and model routing (model_router.py)
        if tool_name in messaging.MESSAGING_TOOL_NAMES:
            return shared.messaging_hub.run_tool(tool_name, args)
        # -------- Notes & tasks, Docs (notes.py, docs_store.py)
        if tool_name in notes.TOOL_NAMES:
            return notes.run_tool(tool_name, args)
        if tool_name in docs_store.TOOL_NAMES:
            return docs_store.run_tool(tool_name, args)
        # -------- skills, knowledge base, earlier chats (agent_skills.py, knowledge.py, profile_store.py)
        if tool_name in agent_skills.TOOL_NAMES:
            return agent_skills.run_tool(tool_name, args)
        if tool_name in knowledge.TOOL_NAMES:
            return knowledge.run_tool(tool_name, args)
        if tool_name in profile_store.TOOL_NAMES:
            return profile_store.run_tool(tool_name, args, shared.chat_store)
        if tool_name == "request_model_handoff":
            return handoff._handoff_tool(args)

        # -------- working memory ------------------------------------------
        if tool_name == "ask_user":
            return {"ok": True, "message": args.get("question", "I need clarification."),
                    "clarification": True, "options": args.get("options") or []}

        if tool_name == "set_plan":
            res = shared.task_board.set_plan(args.get("goal", ""), args.get("steps") or [])
            if res["ok"]:
                event_log.log_action(f"Plan set: {shared.task_board.goal} ({len(shared.task_board.steps)} steps)")
            return res

        elif tool_name == "update_plan":
            status = (args.get("status") or "").lower().strip()
            note = args.get("note")
            if status == "done":
                # This is the fix for a real failure mode: a tool can run
                # without error (ok=True) yet be unable to confirm its
                # effect (verified=False - e.g. an Invoke-only button with
                # no native state, like a game's 'Play' control). Nothing
                # forced the model's own narration to respect that nuance,
                # so it could - and did - report plain success anyway. This
                # consumes the most recent such result exactly once: if it
                # says unverified, 'done' is refused and downgraded to
                # 'failed' with a concrete reason, so the task board (and
                # therefore next turn's LIVE MACHINE STATE block) reflects
                # reality instead of the model's optimism.
                evidence = shared.app_state.get("last_verifiable_result")
                shared.app_state["last_verifiable_result"] = None
                if evidence and evidence["verified"] is False:
                    res = shared.task_board.update(args.get("step"), "failed",
                                            f"AUTO-DOWNGRADED from 'done': {evidence['tool']} could not "
                                            f"confirm its effect ({evidence['message'][:140]}).")
                    if res["ok"]:
                        event_log.log_action(f"Plan: step {args.get('step')} -> failed (auto-downgraded, unverified).")
                    res["message"] = (
                        "NOT marked done: the most recent relevant action reported it could not verify its "
                        f"effect ({evidence['tool']}: {evidence['message'][:200]}). This step was marked "
                        "'failed' instead. Confirm what actually happened (list_elements/observe/"
                        "inspect_screen) before claiming success to the user, or tell them plainly that it "
                        "could not be confirmed.\n\n" + res["message"])
                    return res
            res = shared.task_board.update(args.get("step"), status or None, note)
            if res["ok"]:
                event_log.log_action(f"Plan: step {args.get('step')} -> {status or args.get('status')}")
                if shared.task_board.steps and shared.task_board.is_complete():
                    # Completion detection: stop the model from inventing
                    # more work once every step is verified done.
                    res = {**res, "message": res.get("message", "") + (
                        "\nAll plan steps are complete. Unless the ORIGINAL request asked for more, "
                        "give the user the final answer now without further tool calls.")}
            return res

        elif tool_name == "observe":
            snap = ctx.snapshot()
            parts = [
                f"Focused window: '{snap['active']}'",
                f"Open windows ({len(snap['windows'])}):\n" + "\n".join("  " + w for w in snap["windows"][:40]),
                f"Cached element handles: {ctx.probe_element_handles()}",
                "Already executed this session:\n" + shared.ledger.render(),
                "Task board:\n" + shared.task_board.render(compact=True),
            ]
            if args.get("include_processes"):
                procs = ctx.probe_processes()
                parts.append(f"Running processes ({len(procs)}): " + ", ".join(procs[:80]))
            return {"ok": True, "message": "\n\n".join(parts)}

        elif tool_name == "list_processes":
            procs = ctx.probe_processes()
            query = (args.get("query") or "").strip().lower()
            if query:
                procs = [p for p in procs if query in p.lower()]
            if not procs:
                return {"ok": False, "message": f"No running process matched '{query}'."}
            return {"ok": True, "message": f"{len(procs)} process(es): " + ", ".join(procs[:120])}

        # -------- apps and windows ----------------------------------------
        elif tool_name == "open_application":
            app_name = args["app_name"]
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "open_application is Windows only."}

            # Live check first: relaunching something already running is the
            # single most common way this agent duplicated work.
            if not args.get("force"):
                already = ctx.window_is_open(app_name)
                if already:
                    hwnd, real_title = already
                    di.focus_window(hwnd)
                    return {"ok": True, "message": (
                        f"'{real_title}' is ALREADY OPEN - focused the existing window instead of launching "
                        "a second copy. Continue with your plan from here; do not open it again.")}

            match, candidates = app_registry.resolve_app(app_name)
            if match:
                os.startfile(match["path"])
                event_log.log_action(f"Opened '{match['display']}' ({match['kind']}).")
                return {"ok": True, "message": f"Opened '{match['display']}'. Use wait_for_window to confirm it's up."}
            place_match, place_candidates = roblox_places.resolve_place(app_name)
            if place_match:
                os.startfile(place_match["path"])
                event_log.log_action(f"Opened Roblox place '{place_match['display']}'.")
                return {"ok": True, "message": (
                    f"Opened the Roblox place '{place_match['display']}'. If the Agentic Bridge plugin is "
                    "turned on in Studio it reconnects by itself once the place has loaded (a few seconds) - "
                    "check with roblox_status before using other roblox_* tools. Use wait_for_window to "
                    "confirm the window is up.")}
            all_candidates = candidates + place_candidates
            if all_candidates:
                names = ", ".join(c["display"] for c in all_candidates)
                return {"ok": False, "message": f"No exact match for '{app_name}'. Closest: {names}. Retry with one of these."}
            # Last resort: a bare program name on PATH (e.g. "notepad"), started
            # directly - never through a shell, so no other text can run as a
            # command under this benign (auto-approved) tool.
            exe = shutil.which(app_name) if re.fullmatch(r"[\w.\-]+", app_name) else None
            if exe:
                try:
                    subprocess.Popen([exe])
                    return {"ok": True, "message": f"No installed app matched '{app_name}'; started {exe} from PATH."}
                except OSError as e:
                    return {"ok": False, "message": f"Could not start '{exe}': {e}"}
            return {"ok": False, "message": f"Could not find an installed app called '{app_name}'. Try find_apps."}

        elif tool_name == "find_apps":
            apps = app_registry.refresh_app_registry(force=bool(args.get("refresh")))
            query = (args.get("query") or "").strip().lower()
            if not query:
                sample = sorted({v["display"] for v in apps.values()})[:60]
                return {"ok": True, "message": f"{len(apps)} apps indexed. Sample: " + ", ".join(sample)}
            hits = {v["display"] for k, v in apps.items() if query in k}
            if not hits:
                import difflib
                close_keys = difflib.get_close_matches(query, apps.keys(), n=8, cutoff=0.5)
                hits = {apps[k]["display"] for k in close_keys}
            if not hits:
                return {"ok": False, "message": f"No installed apps matched '{query}'."}
            return {"ok": True, "message": "Matches: " + ", ".join(sorted(hits))}

        # -------- composite one-call helpers ------------------------------
        elif tool_name == "open_app_and_list":
            # Collapses open_application -> wait_for_window -> list_elements
            # into a single call. Each of those was a separate turn, and for a
            # local model each turn is a fresh chance to lose the thread.
            app_name = args["app_name"]
            title = (args.get("window_title") or app_name).strip()
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}

            already = ctx.window_is_open(title) or ctx.window_is_open(app_name)
            if already:
                hwnd, real_title = already
                di.focus_window(hwnd)
                opened_note = f"'{real_title}' was already running; focused it."
            else:
                match, candidates = app_registry.resolve_app(app_name)
                if not match:
                    place_match, place_candidates = roblox_places.resolve_place(app_name)
                    if place_match:
                        os.startfile(place_match["path"])
                        opened_note = (f"Opened the Roblox place '{place_match['display']}'. The Agentic "
                                       "Bridge reconnects by itself once the place has loaded if it is turned "
                                       "on in Studio - check with roblox_status before any roblox_* tool. (list_elements "
                                       "below will likely find nothing useful, since Studio has no "
                                       "Win32 accessibility tree - use the roblox_* tools instead.)")
                        real_title = place_match["display"]
                        deadline = time.time() + 25
                        found = None
                        while time.time() < deadline:
                            found = di.find_window(place_match["display"]) or di.find_window(app_name)
                            if found:
                                break
                            time.sleep(0.4)
                        if found:
                            di.focus_window(found[0])
                            real_title = found[1]
                        listing = elements.list_elements(real_title)
                        return {"ok": True, "fallback": listing.get("fallback"),
                                "message": f"{opened_note}\n{listing.get('message', '')}"}
                    all_candidates = candidates + place_candidates
                    if all_candidates:
                        names = ", ".join(c["display"] for c in all_candidates)
                        return {"ok": False, "message": f"No exact match for '{app_name}'. Closest: {names}. Retry with one of these."}
                    return {"ok": False, "message": f"Could not find '{app_name}'. Try find_apps."}
                os.startfile(match["path"])
                opened_note = f"Launched '{match['display']}'."
                real_title = title
                deadline = time.time() + 15      # apps like VM managers are slow
                found = None
                while time.time() < deadline:
                    found = di.find_window(title) or di.find_window(app_name)
                    if found:
                        break
                    time.sleep(0.4)
                if not found:
                    return {"ok": True, "message": (
                        f"{opened_note} No window matching '{title}' appeared within 15s - it may still be "
                        "starting. Call observe to check, then click_by_name once it is up.")}
                di.focus_window(found[0])
                real_title = found[1]
                time.sleep(0.4)

            listing = elements.list_elements(real_title)
            return {"ok": True, "fallback": listing.get("fallback"),
                    "message": f"{opened_note}\n{listing.get('message', '')}"}

        elif tool_name == "click_by_name":
            return elements.act_by_name(args["window_title"], args["name"],
                                        args.get("action", "click"), args.get("text"))


        elif tool_name == "list_elements":
            return elements.list_elements(args["window_title"])

        elif tool_name == "act_element":
            return elements.act_element(args["element_id"], args.get("action", "click"), args.get("text"))
        # -------- visual fallback -----------------------------------------
        elif tool_name == "inspect_screen":
            max_w, max_h = llm.shot_limits()
            shot, meta, info = vision.inspect(args.get("window_title"), max_w, max_h)
            if shot is None:
                return info
            age = time.time() - (shared.app_state.get("last_shot_at") or 0)
            if (not args.get("force") and shared.app_state.get("last_shot") is not None
                    and age < 30 and not _screen_changed_since_last(shot)):
                shared.app_state["last_shot_at"] = time.time()
                return {"ok": True, "message": (
                    f"The screen is UNCHANGED since your last capture {int(age)}s ago, so no new image was "
                    "sent. Keep using the coordinates from that image. If you expected a change, the last "
                    "action did not take effect - investigate rather than repeating it. Use force=true to "
                    "capture anyway.")}
            shared.app_state["last_shot"], shared.app_state["last_shot_meta"] = shot, meta
            shared.app_state["last_shot_at"] = time.time()
            return {"ok": True,
                    "message": f"Screenshot captured ({meta['thumb_w']}x{meta['thumb_h']}). Read coordinates off this image.",
                    "image_b64": llm.encode_image(shot, for_local=llm.is_local_engine())}

        elif tool_name == "click_at":
            meta = shared.app_state.get("last_shot_meta")
            res = vision.click_and_verify(meta, args["x"], args["y"], args.get("button", "left"), bool(args.get("double")))
            out = {"ok": res.get("ok", False), "message": res.get("message", "")}
            # NOTE: last_shot_meta is deliberately NOT replaced here. The
            # verification capture is full-resolution, while the image the
            # model is reading coordinates from may be a thumbnail; swapping
            # the meta made every subsequent click scale incorrectly.
            if res.get("after_shot") is not None:
                shared.app_state["last_shot"] = res["after_shot"]
            img = res.get("composite_image")
            if img is not None:
                out["image_b64"] = llm.encode_image(img, for_local=llm.is_local_engine())
            return out

        elif tool_name == "drag_at":
            meta = shared.app_state.get("last_shot_meta")
            if meta is None:
                return {"ok": False, "message": "No reference screenshot. Call inspect_screen first."}
            vision.move_and_drag(meta, args["x1"], args["y1"], args["x2"], args["y2"], args.get("button", "left"))
            return {"ok": True, "message": "Drag performed. Call inspect_screen to see the result."}

        elif tool_name == "scroll_at":
            meta = shared.app_state.get("last_shot_meta")
            if meta is None:
                return {"ok": False, "message": "No reference screenshot. Call inspect_screen first."}
            vision.scroll_at(meta, args["x"], args["y"], args.get("amount", 3))
            return {"ok": True, "message": "Scrolled. Call inspect_screen to see the result."}

        elif tool_name == "get_active_window":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            return {"ok": True, "message": f"Active window: '{di.get_active_window_title()}'"}

        elif tool_name == "wait_for_window":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            timeout = min(float(args.get("timeout", 5) or 5), 20)
            deadline = time.time() + timeout
            while time.time() < deadline:
                match = di.find_window(args["window_title"])
                if match:
                    return {"ok": True, "message": f"Window '{match[1]}' is now available."}
                time.sleep(0.25)
            return {"ok": False, "message": f"No window matching '{args['window_title']}' appeared within {timeout}s."}

        elif tool_name == "list_windows":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            titles = ctx.probe_windows(80)
            return {"ok": True, "message": "Visible windows:\n" + "\n".join(titles)}

        elif tool_name == "close_window":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            match = di.find_window(args["window_title"])
            if not match:
                return {"ok": False, "message": f"No window matching '{args['window_title']}' is open (it may already be closed)."}
            hwnd, real_title = match
            ok = di.close_window(hwnd, force=bool(args.get("force")))
            return {"ok": bool(ok), "message": f"{'Force-terminated' if args.get('force') else 'Sent close to'} '{real_title}'."}

        elif tool_name == "keyboard_action":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            action = args["action"].lower()
            if action in di.MEDIA_KEYS:
                di.press_media_key(action)
            else:
                di.send_key_combo(action)
            return {"ok": True, "message": f"Sent key(s): {args['action']} to '{di.get_active_window_title()}'"}

        elif tool_name == "type_text":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            text = args["text"]
            target = di.get_active_window_title()
            if args.get("use_paste", True):
                app_registry.set_clipboard(text)
                di.send_key_combo("ctrl+v")
            else:
                di.send_unicode_text(text)
            return {"ok": True, "message": f"Typed {len(text)} characters into '{target}'."}

        elif tool_name == "get_clipboard":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            text = app_registry.get_clipboard()
            productivity_tools.record_clipboard_sample(text)
            return {"ok": True, "message": f"Clipboard: {text[:2000]}"}

        elif tool_name == "set_clipboard":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            app_registry.set_clipboard(args.get("text", ""))
            productivity_tools.record_clipboard_sample(args.get("text", ""))
            return {"ok": True, "message": "Clipboard updated."}

        elif tool_name == "media_control":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            di.press_media_key(args["action"])
            return {"ok": True, "message": f"Media action: {args['action']}"}

        elif tool_name == "execute_command":
            return _run_cancellable(args["command"], shell=True, timeout=60)

        elif tool_name == "run_powershell":
            return _run_powershell(args["command"], args.get("timeout_seconds"))

        elif tool_name == "run_python":
            return _run_cancellable([sys.executable, "-c", args["code"]], shell=False, timeout=60)

        elif tool_name == "write_file":
            path = files.safe_path(args["filename"])
            journal = files._journal()
            fs_tools._makedirs(os.path.dirname(path) or ".", journal)
            append = bool(args.get("append"))
            journal.backup_file(path, replacing=not append)
            old = ""
            if append and os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    old = f.read()
            files._atomic_write(path, old + args["content"])
            if append:
                return {"ok": True, "message": f"Appended {len(args['content'])} chars to {path} (now {os.path.getsize(path)} bytes)"}
            return {"ok": True, "message": f"Wrote {len(args['content'])} chars to {path}"}

        elif tool_name == "read_file":
            path = files.safe_path(args["filename"])
            if files._is_secret_file(path):
                return {"ok": False, "status": "permanent_failure", "message": (
                    f"Refused: '{os.path.basename(path)}' holds secrets (API keys, passwords or private keys). "
                    "read_file runs without approval, so it never reads files like this. Ask the user for "
                    "what you need instead.")}
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return {"ok": True, "message": content[:3000] + ("... [truncated]" if len(content) > 3000 else "")}

        elif tool_name == "edit_file":
            path = files.safe_path(args["filename"])
            with open(path, "r", encoding="utf-8", newline="") as f:
                content = f.read()
            old, new = files._match_line_endings(content, args["old_text"], args["new_text"])
            if old not in content:
                return {"ok": False, "message": "old_text not found in file."}
            content = content.replace(old, new, 1)
            files._journal().backup_file(path)
            files._atomic_write(path, content)
            return {"ok": True, "message": f"Edited {path}"}

        elif tool_name == "apply_patch":
            path = files.safe_path(args["filename"])
            if not os.path.isfile(path):
                return {"ok": False, "message": f"'{path}' does not exist. Create files with write_file."}
            with open(path, "r", encoding="utf-8", errors="strict", newline="") as f:
                text = f.read()
            result = patch_edit.dry_run(text, args.get("edits"), os.path.basename(path))
            if not result["ok"]:
                return {"ok": False, "message": f"Patch not applied - nothing was changed. {result['error']}"}
            files._journal().backup_file(path)
            files._atomic_write(path, result["text"])
            return {"ok": True, "message": f"Patched {path}: {patch_edit.describe_matches(result['matches'])}."}

        elif tool_name == "open_file_in_os":
            path = files.safe_path(args["filename"])
            if config.IS_WINDOWS:
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])
            return {"ok": True, "message": f"Opened {path}"}

        elif tool_name == "list_dir":
            path = files.safe_path(args["directory"])
            entries = os.listdir(path)
            return {"ok": True, "message": "\n".join(entries[:200])}

        elif tool_name == "run_sequence":
            return _run_sequence(args.get("steps", []))

        # -------- filesystem & archives ------------------------------------
        elif tool_name == "copy_path":
            return fs_tools.copy_path(args["source"], args["destination"], bool(args.get("overwrite")), journal=files._journal())

        elif tool_name == "move_path":
            return fs_tools.move_path(args["source"], args["destination"], bool(args.get("overwrite")), journal=files._journal())

        elif tool_name == "rename_path":
            return fs_tools.rename_path(args["path"], args["new_name"], journal=files._journal())

        elif tool_name == "delete_path":
            return fs_tools.delete_path(args["path"], bool(args.get("force")), journal=files._journal())

        elif tool_name == "make_dir":
            return fs_tools.make_dir(args["path"], journal=files._journal())

        elif tool_name == "zip_paths":
            return fs_tools.zip_paths(args.get("paths") or [], args["output_zip"], journal=files._journal())

        elif tool_name == "unzip_file":
            return fs_tools.unzip_file(args["zip_path"], args["destination"], journal=files._journal())

        elif tool_name == "search_files":
            return fs_tools.search_files(args["directory"], args.get("pattern", "*"),
                                         args.get("content_query"), args.get("max_results", 50),
                                         private=files._is_secret_file)

        elif tool_name == "file_info":
            return fs_tools.file_info(args["path"])

        elif tool_name == "list_drives":
            return fs_tools.list_drives()

        elif tool_name == "disk_usage":
            return fs_tools.disk_usage(args.get("path", "."))

        elif tool_name == "compare_files":
            return fs_tools.compare_files(args["path_a"], args["path_b"])

        # -------- browser & web ---------------------------------------------
        elif tool_name == "open_url":
            return web_tools.open_url(args["url"])

        elif tool_name == "web_search":
            s = shared.app_state["settings"]
            return search_engine.web_search_tool(args, s, browser_fallback=_browser_fallback(s),
                                                 should_cancel=run_control.is_stopped, progress=event_log.progress)

        elif tool_name == "fetch_page":
            return search_engine.fetch_page_tool(args, browser_fallback=_browser_fallback(shared.app_state["settings"]),
                                                 should_cancel=run_control.is_stopped, progress=event_log.progress)

        elif tool_name == "market_data":
            import finance
            return finance.market_data_tool(args)

        elif tool_name == "generate_image":           # local image generation (image_gen.py)
            import image_gen
            att = shared.app_state.get("turn_attachments") or {}
            ref = (att.get("images") or [None])[0]
            res = image_gen.generate_image_tool(args, ref_image=ref, should_cancel=run_control.is_stopped,
                                                progress=lambda msg: event_log.progress(label=f"Making an image - {msg}"))
            if res.get("ok"):                          # shown under the reply (agent/tasks._with_images)
                made = res.get("images") or [res["image"]]
                shared.app_state.setdefault("turn_images", []).extend(made)
                # ...and in the chat at once (1.4.7): the page draws them from this event, without waiting for the
                # chat model to load back onto the GPU and reply.
                event_log.emit_event("image", "Picture ready" if len(made) == 1 else f"{len(made)} pictures ready",
                                     status="done", images=[{k: m.get(k) for k in ("url", "model", "seed", "width", "height")}
                                                            for m in made])
            return res

        elif tool_name in ("compute_stats", "make_chart"):      # exact statistics and checked charts (stats_tools.py)
            import stats_tools
            tables = _turn_tables()
            if tool_name == "compute_stats":
                return stats_tools.compute_stats(args, tables)
            res = stats_tools.make_chart(args, tables)
            if res.get("ok"):                          # drawn under the reply (agent/tasks._with_charts)
                shared.app_state.setdefault("turn_charts", []).append(res["chart"])
            return res

        elif tool_name == "use_tools":                # turn on safe tools mid-answer (auto_tools.py)
            import auto_tools
            return auto_tools.use_tools_tool(args)

        elif tool_name == "load_skill":               # more action tools from the next step (skills.py)
            import skills
            return skills.load_skill_tool(args)

        elif tool_name == "world_monitor":            # MossadGPT's world monitor (osint/)
            import osint.tool
            return osint.tool.world_monitor_tool(args)

        elif tool_name == "show_web_images":           # pictures from the web, in the chat (web_images.py, 1.4.7)
            import web_images
            settings = shared.app_state["settings"]
            mode = search_engine.turn_mode(settings)
            if mode == "max" and search_engine.max_privacy_route_problem():
                return {"ok": False, "status": "permanent_failure", "message": search_engine.max_privacy_route_problem()}
            with search_engine.scope(mode):
                res = web_images.show_web_images_tool(args, settings, should_cancel=run_control.is_stopped,
                                                      progress=lambda msg: event_log.progress(label=f"Pictures from the web - {msg}"))
            if res.get("ok"):                          # shown at once, and under the reply (agent/tasks._with_images)
                shared.app_state.setdefault("turn_images", []).extend(res["images"])
                event_log.emit_event("image", f"{len(res['images'])} picture{'s' if len(res['images']) != 1 else ''} from the web",
                                     status="done", images=[{k: m.get(k) for k in ("url", "alt", "domain", "width", "height", "source")}
                                                            for m in res["images"]])
            return res

        elif tool_name == "web_read":
            mode = search_engine.turn_mode()
            if mode == "disabled":
                return {"ok": False, "status": "permanent_failure", "message": "Reading web pages is turned off (Settings > Web Search)."}
            if mode == "max" and search_engine.max_privacy_route_problem():
                return {"ok": False, "status": "permanent_failure", "message": search_engine.max_privacy_route_problem()}
            with search_engine.scope(mode):             # Private/Max headers; Local Only: offline
                return web_tools.web_read(args["url"], args.get("start") or 0,
                                          browser_fallback=_browser_fallback(shared.app_state["settings"]) and mode == "standard")

        elif tool_name in ("browser_open", "browser_read", "browser_click", "browser_fill"):
            try:
                b = browser.current()               # a temporary chat's private browser (browser.py)
                if tool_name == "browser_open":
                    return b.open(args["url"])
                if tool_name == "browser_read":
                    return b.read(args.get("start") or 0)
                if tool_name == "browser_click":
                    return b.click(args["element"])
                return b.fill(args["element"], args.get("value", ""))
            except browser.BrowserError as e:
                return {"ok": False, "message": str(e)}

        elif tool_name == "new_browser_tab":
            return web_tools.new_browser_tab(args["window_title"], args.get("url", ""))

        elif tool_name == "download_file":
            files._journal().backup_file(files.safe_path(args["filename"]))
            return web_tools.download_file(args["url"], args["filename"], should_cancel=run_control.is_stopped)

        elif tool_name == "get_url_status":
            return web_tools.get_url_status(args["url"])

        # -------- system / process / network --------------------------------
        elif tool_name == "list_services":
            return sys_tools.list_services(args.get("query", ""))

        elif tool_name == "service_control":
            return sys_tools.service_control(args["name"], args["action"])

        elif tool_name == "kill_process":
            return sys_tools.kill_process(args["name_or_pid"], bool(args.get("force")))

        elif tool_name == "ping_host":
            return sys_tools.ping_host(args["host"], args.get("count", 4))

        elif tool_name == "network_info":
            return sys_tools.network_info()

        elif tool_name == "get_env_var":
            return sys_tools.get_env_var(args["name"])

        elif tool_name == "set_env_var":
            return sys_tools.set_env_var(args["name"], args.get("value", ""), bool(args.get("persist")))

        elif tool_name == "get_system_info":
            return sys_tools.get_system_info()

        elif tool_name == "port_check":
            return sys_tools.port_check(args["host"], args["port"])

        # -------- productivity ------------------------------------------------
        elif tool_name == "send_notification":
            return productivity_tools.send_notification(args["title"], args["message"],
                                                         args.get("duration_seconds", 8))

        elif tool_name == "app_appearance":
            import app_look                             # this app's own look only: nothing outside the app changes
            return app_look.apply(args.get("mode"), args.get("appearance"))

        elif tool_name == "mini_window":
            import mini_window                          # this app's own window only: nothing outside the app changes
            return mini_window.apply(args.get("action", "toggle"), args.get("opacity"), args.get("click_through"),
                                     args.get("place"))

        elif tool_name == "save_memory":
            if privacy.active():
                return {"ok": False, "status": "permanent_failure", "message": (
                    "Not saved: this is a temporary chat, and nothing from it goes into memory. Tell the user they can "
                    "add it themselves in Settings > Memory, or save this chat first.")}
            store = shared.memory_store
            files._journal().backup_file(store.path)
            try:
                entry = store.add(args["text"], source="assistant")
            except ValueError as e:
                return {"ok": False, "message": str(e)}
            return {"ok": True, "message": f"Saved to memory (the user can edit it in Settings > Memory): {entry['text']}"}

        elif tool_name == "schedule_task":
            return productivity_tools.schedule_task(args["name"], args.get("delay_seconds"),
                                                     args.get("run_at"), args.get("command"),
                                                     args.get("message"), args.get("repeat") or "once")

        elif tool_name == "cancel_scheduled_task":
            return productivity_tools.cancel_scheduled_task(args["name"])

        elif tool_name == "list_scheduled_tasks":
            return productivity_tools.list_scheduled_tasks(args.get("query", ""))

        elif tool_name == "clipboard_history":
            return productivity_tools.clipboard_history(args.get("limit", 20))

        elif tool_name == "screenshot_to_file":
            # Always name the file here, so the journal records it (a screenshot
            # without a filename used to be "undoable" with nothing to undo).
            filename = files.safe_path(args.get("filename") or f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png")
            files._journal().backup_file(filename)
            return productivity_tools.screenshot_to_file(filename, args.get("window_title"))

        elif tool_name == "top_processes":
            return sys_tools.top_processes(args.get("limit", 10), args.get("sort", "memory"))

        elif tool_name == "snap_windows":
            if not config.IS_WINDOWS:
                return {"ok": False, "message": "Windows only."}
            left = di.find_window(args["left_title"])
            right = di.find_window(args["right_title"])
            if not left or not right:
                missing = args["left_title"] if not left else args["right_title"]
                return {"ok": False, "message": f"No open window matches '{missing}'."}
            ok = win_signals.snap_side_by_side(left[0], right[0])
            return {"ok": ok, "verified": ok, "message": (f"Placed '{left[1]}' on the left and '{right[1]}' on the right."
                                                        if ok else "Windows refused to move one of the windows.")}

        # -------- MCP servers (mcp_client.py) -----------------------------------
        elif tool_name.startswith("mcp__"):
            return shared.mcp_hub.call(tool_name, args, cancelled=run_control.is_stopped)

        elif tool_name == "mcp_read_resource":
            return shared.mcp_hub.read_resource(args["server"], args["uri"])

        # -------- App adapters (app_registry.AppAdapter): Roblox Studio ------
        # The adapter supplies app knowledge; permissions, approval and the
        # undo journal were already handled by the one tool path around this.
        elif app_registry.adapter_for_tool(tool_name) is not None:
            return app_registry.adapter_for_tool(tool_name).call(tool_name, args, journal=files._journal(),
                                                                 cancelled=run_control.is_stopped)

        else:
            return {"ok": False, "message": f"Unknown tool: {tool_name}"}

    except Exception as e:
        return {"ok": False, "message": f"{tool_name} failed: {e}"}


# Each run_sequence step does what one of these single-action tools does.
_SEQUENCE_STEP_TOOL = {"element": "act_element", "click": "click_at", "drag": "drag_at", "scroll": "scroll_at",
                       "key": "keyboard_action", "type": "type_text"}


def _run_sequence(steps: list) -> dict:
    # A sequence must not be a way around a per-tool Deny: if the user denied
    # any tool a step stands for, nothing in the sequence runs.
    denied = sorted({_SEQUENCE_STEP_TOOL[s.get("kind")] for s in steps if isinstance(s, dict)
                     and s.get("kind") in _SEQUENCE_STEP_TOOL
                     and permissions.decide(_SEQUENCE_STEP_TOOL[s["kind"]], shared.app_state["settings"]) == "deny"})
    if denied:
        return {"ok": False, "status": "permanent_failure",
                "message": (f"NOT EXECUTED - this sequence uses {', '.join(denied)}, which the user set to DENY in "
                            "Settings > Permissions. No step was run.")}
    log_lines = []
    for i, step in enumerate(steps, 1):
        kind = step.get("kind")
        try:
            if kind == "element":
                res = elements.act_element(step["element_id"], step.get("action", "click"), step.get("text"))
            elif kind == "click":
                meta = shared.app_state.get("last_shot_meta")
                res = vision.click_and_verify(meta, step["x"], step["y"], step.get("button", "left"), bool(step.get("double")))
                if res.get("after_shot") is not None:
                    shared.app_state["last_shot"] = res["after_shot"]  # meta intentionally preserved
            elif kind == "drag":
                vision.move_and_drag(shared.app_state.get("last_shot_meta"), step["x1"], step["y1"], step["x2"], step["y2"], step.get("button", "left"))
                res = {"ok": True, "message": "dragged"}
            elif kind == "scroll":
                vision.scroll_at(shared.app_state.get("last_shot_meta"), step["x"], step["y"], step.get("amount", 3))
                res = {"ok": True, "message": "scrolled"}
            elif kind == "key":
                di.send_key_combo(step["combo"])
                res = {"ok": True, "message": f"key {step['combo']}"}
            elif kind == "type":
                app_registry.set_clipboard(step["text"])
                di.send_key_combo("ctrl+v")
                res = {"ok": True, "message": "typed"}
            elif kind == "wait":
                time.sleep(min(float(step.get("seconds", 1)), 15))
                res = {"ok": True, "message": "waited"}
            else:
                res = {"ok": False, "message": f"unknown step kind '{kind}'"}
        except Exception as e:
            res = {"ok": False, "message": str(e)}

        log_lines.append(f"step {i} ({kind}): {'OK' if res.get('ok') else 'FAILED'} - {res.get('message', '')[:150]}")
        if not res.get("ok"):
            log_lines.append(f"Sequence stopped at step {i} of {len(steps)}. Steps 1-{i - 1} DID run - do not rerun them.")
            return {"ok": False, "message": "\n".join(log_lines)}
    return {"ok": True, "message": "\n".join(log_lines)}
