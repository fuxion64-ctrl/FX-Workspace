"""
agent/loop.py
=============
The agent loop: model step -> validation -> permission/approval -> execution -> result.
(Split out of main.py in Phase 2 - see ARCHITECTURE.md D14.)
"""
from __future__ import annotations

import difflib
import json
import re
import time
import context as ctx
import llm_compat
import llm_stream
import model_caps
import model_metrics
import model_output
import permissions
import secret_mask
import shell_lint
import stream_hub
from run_control import RunStopped, run_control
import studio_mcp
import tool_registry
import untrusted
import url_guard

import agent.models as llm
import agent.parsing as parsing
import core.config as config
import core.state as shared
import events as event_log
import tools.approvals as approvals
import tools.runtime as runtime
import tools.schemas as schemas


def _stopped_response(reason: str = None) -> dict:
    """Leave the conversation consistent after a stop: every tool call the
    model asked for gets a result, so the next request isn't rejected by the
    model API for a dangling tool call. What the reply had written when it was stopped stays: in the chat
    ("partial") and in the conversation, so the model knows what the user already read (1.4.2)."""
    partial = model_output.split_reasoning(stream_hub.hub.take_stopped())[0].strip()
    conv = shared.app_state["conversation"]
    for i in range(len(conv) - 1, -1, -1):
        m = conv[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            answered = {x.get("tool_call_id") for x in conv[i + 1:] if x.get("role") == "tool"}
            for call in m["tool_calls"]:
                if call.get("id") not in answered:
                    conv.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "name": call.get("function", {}).get("name", ""),
                                 "content": "NOT EXECUTED - the user stopped the task before this ran."})
            break
        if m.get("role") == "user":
            break
    conv.append({"role": "assistant", "content": f"{partial}\n\n(The user stopped this reply here. Nothing further was run.)"
                 if partial else "(The user stopped this task. Nothing further was run.)"})
    active = shared.task_board.active_step()
    if active:
        shared.task_board.update(active["n"], "failed", "Stopped by you")
    shared.app_state["pending_clarification"] = None
    shared.app_state["task_queue"] = None
    event_log.log_action("Stopped by the user.")
    event_log.emit_event("note", "Stopped by you")
    timed_out = (reason or run_control.stop_reason()) == "time"
    return {"status": "stopped",
            "message": ("Stopped: this task reached the time limit for its effort level. Nothing else will run - "
                        "raise the effort in Settings > Agent for longer tasks." if timed_out else
                        "Stopped. Nothing else will run for this request."),
            **({"partial": partial} if partial else {}),
            "plan": shared.task_board.as_dict()}


# ==============================================================================
# Agent loop
# ==============================================================================
# Host bookkeeping whose result text the host writes itself; every other tool
# result can carry text from outside (files, pages, window titles, command
# output, MCP, Studio) and is framed as data (untrusted.py).
# generate_image: the app's own line about the image it made (image_gen.py) - framed, a small model copied the
# frame into its reply.
_HOST_RESULT_TOOLS = {"set_plan", "update_plan", "ask_user", "generate_image", "make_chart", "use_tools", "load_skill"}


def _held_note() -> str:
    """The "not run yet" note for calls held back this step, said once - on whichever tool message answers the
    kept call (1.4.8, audit 2.11: an invalid, repeated, denied or rejected kept call used to drop it)."""
    held = shared.app_state.pop("held_back_calls", None)
    if not held:
        return ""
    return (f"\n\n[NOT RUN YET - you asked for these in the same step, before this result was known: "
            f"{'; '.join(held)}. Call again any you still need now that you have this result.]")


def _append_tool_result(tc, tool_name, res):
    message = res.get("message", "") or ("ok" if res.get("ok") else "failed")
    if tool_name not in _HOST_RESULT_TOOLS:
        found = untrusted.find_instructions(message)
        if found:
            _report_embedded_instructions(tool_name, found)
        url_guard.note_read(message)            # from here on, only addresses seen so far open by themselves
        message = untrusted.frame(tool_name, message, found)
    shared.app_state["conversation"].append({
        "role": "tool", "tool_call_id": tc.id, "name": tool_name,
        "content": message + _held_note(),
    })
    if res.get("image_b64"):
        shared.app_state["conversation"].append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"[Image from {tool_name}. Use this exact image for any coordinates.]"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res['image_b64']}"}},
            ],
        })


def _report_embedded_instructions(source: str, found: list):
    """Show the user text that tried to direct the agent - it is never followed.
    From here on this request's side-effecting calls all go to the approval
    card (permissions.decide tainted=True), with a warning on the card."""
    seen = shared.app_state.setdefault("untrusted_seen", [])
    seen.append({"source": source, "snippet": found[0][:220]})
    event_log.log_action(f"Content from {source} contains instructions aimed at the assistant; not followed: "
                         f"{found[0][:160]}")
    event_log.emit_event("untrusted", f"Ignored instructions inside {source} output", status="done", ok=False,
                         level="warn", source=source, snippets=found,
                         detail="This text was in content the assistant read. It was shown to you and not followed:\n- "
                                + "\n- ".join(found))


def _append_observation(text: str):
    # The host writes the observation; the window titles inside it are data.
    found = untrusted.find_instructions(text)
    if found:
        _report_embedded_instructions("window titles", found)
    shared.app_state["conversation"].append({"role": "user", "content": (
        "[OBSERVATION from the host] " + untrusted.frame("machine state (window titles)", text, found))})


# ---- a picture is the answer (1.4.7) -------------------------------------------------------------------------
# "Make an image of a toaster": after generate_image the loop used to ask the text model for a reply. That meant
# waiting for it to load back onto the GPU the image engine had just used (Ornith 9B timed out), and a small model
# that couldn't see the picture went looking for it - eight file searches in a fresh chat, then "This chat is too
# long for the model" (4 Nov 2026). The picture is already in the chat (the page shows it from the tool's event),
# so a request that only asks for pictures ends right there, with the app's own line.
_MORE_THAN_A_PICTURE = re.compile(
    r"\?|\b(?:and|then|also|plus)\b[^.!\n]{0,60}\b(?:write|tell|explain|describe|give|list|save|send|post|email|search|"
    r"find|look up|compare|summari[sz]e|translate|code|open|upload|copy|move|rename|print|read)\b", re.I)


def _wants_more_than_pictures(request: str) -> bool:
    return bool(_MORE_THAN_A_PICTURE.search(request or ""))


def picture_reply(prose: str, made: list) -> str:
    """The reply under pictures that are the whole answer: what the model wrote with the call (when it is short and
    not a description of calling the tool), else a plain line from the app."""
    text = model_output.split_reasoning(prose or "")[0].strip()
    if text and len(text) <= 500 and not re.search(r"generate_image|tool|I(?:'ll| will) (?:now )?(?:make|create|generate)", text, re.I):
        return text
    n = len(made)
    web = [m for m in made if isinstance(m, dict) and m.get("web")]
    if web and len(web) == n:                    # pictures from the web: where they came from, not "your pictures"
        sites = list(dict.fromkeys(m.get("domain") or "the web" for m in web))
        where = sites[0] if len(sites) == 1 else ", ".join(sites[:-1]) + " and " + sites[-1]
        return f"Here {'is a picture' if n == 1 else f'are {n} pictures'} from {where}."
    return ("Here's your picture." if n <= 1 else f"Here are your {n} pictures.") + \
        " Tell me what to change and I'll make another."


def _signature_set(parsed) -> frozenset:
    return frozenset(ctx.signature(name, args) for _tc, name, args in parsed)


def final_summary() -> str:
    """One tool-free call so a long task still ends with a real answer rather
    than a bare 'Max steps reached'."""
    try:
        messages = llm.build_messages() + [{
            "role": "user",
            "content": ("[SYSTEM] You have reached the step limit for this turn. Do not call any more tools. "
                        "Tell the user plainly what you completed, what is still outstanding according to "
                        "your task board, and what you need from them to continue."),
        }]
        if shared.app_state.get("mode") == config.MODE_LOCAL:
            # (1.4.8, audit 2.5) streamed and hidden, so Stop ends it at once and model_monitor counts it.
            res = llm._local_complete(dict(
                model=f"openai/{shared.app_state.get('local_engaged') or llm.resolve_local_model(False)}",
                messages=llm.strip_images(messages), temperature=0.1, max_tokens=500, timeout=180),
                record=False, publish=False)
        else:
            model_name, key = shared.LAST_GOOD.get("text") or llm.candidate_pairs(False)[0]
            res = llm_compat.completion(model=model_name, api_key=key, messages=llm.strip_images(messages), timeout=45)
        return res.choices[0].message.content or ""
    except llm_stream.StreamCancelled:
        raise RunStopped("user")
    except Exception as e:
        event_log.log_action(f"Final summary failed: {str(e)[:120]}")
        return ""


_FABRICATED_CLAIM_RE = re.compile(r"\b(?:successfully|done|added|created|completed|connected|opened)\b", re.I)
_FABRICATED_CODE_SPAN_RE = re.compile(r"`([a-zA-Z][^`\n]{2,})`")
# (1.4.8, audit 1.4) Only a span that reads as an invocation counts: a run/call verb just before it, or arguments
# inside it. "Done - I created `notes.txt`" names a file, not a tool, and was flagged as made up.
_INVOKE_VERB_RE = re.compile(r"\b(?:run|runs|ran|running|call|calls|called|calling|execute|executes|executed|"
                             r"executing|invoke|invokes|invoked|invoking)\b[^`\n]{0,20}$", re.I)
_FILE_LIKE_RE = re.compile(r"[\\/:]|\.[A-Za-z0-9]{1,6}$")


def _fabricated_tool_name(prose: str):
    """Catches the exact failure mode in the bug report: the model writes
    prose like 'Running `roblox-studio-tool add_part Test1 1 1 1`...
    Part added successfully.' instead of making a real tool call. Native
    tool_calls and _extract_tool_calls_from_text (main.py's text-recovery
    fallback) both require an actual call SHAPE - JSON, or a recognized
    function name - so free-form narration like this produces neither and
    was previously returned to the user as a genuine, unquestioned success.
    This is a last-resort net: if the prose both names something in a
    code span that is NOT a real tool, and separately claims completion,
    it is fabricated - nothing was actually executed.
    The caller only asks when no tool call succeeded in this run (1.4.8)."""
    if not prose or not _FABRICATED_CLAIM_RE.search(prose):
        return None
    for m in _FABRICATED_CODE_SPAN_RE.finditer(prose):
        span = m.group(1).strip()
        first = span.split()[0]
        if _FILE_LIKE_RE.search(first):
            continue                            # a file, folder or path: something made, not something run
        has_args = len(span.split()) > 1
        if not has_args and not _INVOKE_VERB_RE.search(prose[max(0, m.start() - 40):m.start()]):
            continue                            # a name mentioned, not a call described
        if not schemas.known_tool(first.replace("-", "_")):
            return first
    return None


# ---- Runtime activity for the Workspace panel ---------------------------------
# Real facts about each model call (timing, tokens, how it ended) and what the
# model itself wrote alongside its actions. All text passes through _user_safe:
# secrets are masked, control characters dropped, and length capped.
_SECRET_RE = secret_mask.SECRET_RE             # the one masking rule (secret_mask.py)


def _user_safe(text, limit: int) -> str:
    text = _SECRET_RE.sub(lambda m: (m.group(1) or "") + "[hidden]", str(text or ""))
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _model_call_facts(response, started: float) -> dict:
    """What the runtime reports about one finished model call."""
    choice = (getattr(response, "choices", None) or [None])[0]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)

    def num(obj, key):
        value = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    finish = getattr(choice, "finish_reason", None)
    finish = finish if isinstance(finish, str) else None
    model = llm._current_model()
    facts = {"duration_ms": int((time.time() - started) * 1000), "model": model if isinstance(model, str) else None,
             "finish": finish, "tokens_out": num(usage, "completion_tokens"), "tokens_in": num(usage, "prompt_tokens"),
             "reasoning_tokens": num(details, "reasoning_tokens"),
             "protocol": model_caps.protocol(model) if shared.app_state.get("mode") == config.MODE_LOCAL and isinstance(model, str) else None,
             "label": {"length": "Reply cut off by the length limit",
                       "repetition": "Reply stopped early: the model was repeating itself",
                       "runaway": "Reply stopped early: a tool call grew far too long"}.get(finish, "Model replied"),
             "streamed": True if getattr(response, "streamed", False) else None}
    facts.update(model_metrics.generation_facts(response))
    if finish in ("length", "repetition", "runaway"):
        facts["level"] = "warn"
    return {k: v for k, v in facts.items() if v not in (None, "")}


REASONING_KEEP = 20_000


def _emit_model_note(msg, note: str, tools: list, seconds: float | None = None) -> int | None:
    """The model's own words next to its actions: the short message it wrote
    before calling tools, and its reasoning when a reasoning model produced
    some (an excerpt, plus the full text - capped - for the Activity panel's
    "Raw reasoning" section). Nothing here is acted on - it is shown, never
    executed. Returns the event id, so the tool calls of this step can point
    at it (their "Why?"). `seconds`: how long it thought before answering
    (agent/sessions.py saves it with the reply as "Thought for Xs")."""
    _answer, reasoning = model_output.split_reasoning(getattr(msg, "content", "") or "")
    extra = getattr(msg, "reasoning_content", None)
    if isinstance(extra, str) and extra.strip():
        reasoning = f"{reasoning}\n\n{extra}".strip() if reasoning else extra.strip()
    note = _user_safe(note, 400)
    excerpt = full = ""
    if reasoning:
        tail = _user_safe(reasoning[-100_000:], 100_000)       # the END of long reasoning, not a middle slice
        excerpt = tail if len(tail) <= 600 else "…" + tail[-600:].lstrip()
        full = tail if len(tail) <= REASONING_KEEP else "…" + tail[-REASONING_KEEP:].lstrip()
    if not note and not excerpt:
        return None
    label = note or ("Reasoned before acting" if tools else "Reasoned before answering")
    ev = event_log.emit_event("thought", label, status="done", note=note, reasoning=excerpt,
                              reasoning_full=full if full != excerpt else "",
                              reasoning_words=len(reasoning.split()) if reasoning else 0, next_tools=list(tools)[:4],
                              thinking_s=round(seconds, 1) if isinstance(seconds, (int, float)) else None)
    return ev.get("id") if isinstance(ev, dict) else None


def _step_links(thought_id, args) -> dict:
    """What a tool event carries so the Activity panel can answer "Why?": the
    plan step it served, the model's stated reason (its justification
    argument) and the model note/reasoning it came from - only what exists."""
    out = {}
    step = shared.task_board.active_step()
    if step:
        out["plan_step"] = {"n": step["n"], "text": str(step["text"])[:200]}
    why = args.get("justification") if isinstance(args, dict) else None
    if isinstance(why, str) and why.strip():
        out["why"] = _user_safe(why, 400)
    if thought_id:
        out["thought_id"] = thought_id
    return out


def _invalid_call_reason(name: str, args) -> str | None:
    """Why this call must not be approved or run, or None. Checked BEFORE the
    approval card: asking the user to approve a call the executor would then
    reject (or one with arguments that never arrived) is meaningless."""
    if isinstance(args, parsing._MalformedArgs):
        if args.truncated:
            return (f"NOT EXECUTED - your {name} call was cut off by the output length limit: its arguments "
                    f"({args.raw_len} characters so far) never finished, so nothing ran. Make the call again with "
                    "much shorter arguments. For a long file, write_file the first part, then write_file with "
                    "append=true for each further part (each part under about 4000 characters). Never put a "
                    "whole file inside a shell command.")
        return (f"NOT EXECUTED - the arguments of your {name} call were not a readable JSON object, so nothing "
                "ran. Call it again with a JSON object of arguments.")
    spec = runtime.TOOL_REGISTRY.get(name) if name not in schemas.INTERNAL_TOOL_NAMES else None
    if spec is None:
        if name == llm_stream.UNNAMED_TOOL:
            return "NOT EXECUTED - your tool call arrived without a tool name, so nothing ran. Make the call again."
        # Small models invent plausible names (roblox_list_tree, roblox_inspect):
        # name the real ones closest to what it meant so the retry can succeed.
        offered = [t["function"]["name"] for t in schemas.tools_for_current_turn(shared.app_state.get("mode") == config.MODE_LOCAL)]
        names = list(offered)
        if "load_skill" in offered:                   # skills not loaded yet still hold the real names (skills.py)
            import skills
            names += [n for n in skills.SKILL_OF if n not in offered]
        close = difflib.get_close_matches(name, names, n=4, cutoff=0.45)
        if not close:
            prefix = name.split("_")[0] + "_"
            close = [n for n in names if n.startswith(prefix)][:6]
        if not offered:
            return (f"NOT EXECUTED - '{name}' is not a tool, and you have no tools for this message, so nothing ran. "
                    "Answer the user from what you know, or ask them to say what they want done.")
        hint = f" Closest: {', '.join(close)}." if close else ""
        return (f"NOT EXECUTED - '{name}' is not one of your tools, so nothing ran.{hint} Your tools are exactly: "
                f"{', '.join(sorted(offered))}. Call one of them as a tool call (not as text), or answer the user.")
    _args, errors = tool_registry.validate_args(spec.input_schema, args)
    if errors:
        return (f"NOT EXECUTED - invalid arguments for {name}: {'; '.join(errors)}. Nothing ran. "
                "Fix the arguments and call it again.")
    # POSIX-only commands would fail on Windows - after the user approved them.
    if name in _SHELL_OF and isinstance(args.get("command"), str):
        findings = shell_lint.lint(args["command"], _SHELL_OF[name])
        if findings:
            return shell_lint.hint_message(name, findings, _SHELL_OF[name])
    return None


_SHELL_OF = {"run_powershell": shell_lint.POWERSHELL, "execute_command": shell_lint.CMD}


def _answer_invalid_calls(invalid: list):
    """Give every rejected call a tool result (the API requires one per call)
    and show it in the activity timeline as a failed, not-run step."""
    for tc, name, args, reason in invalid:
        cut = isinstance(args, parsing._MalformedArgs) and args.truncated
        event_log.log_action(f"Did not run {name}: {'arguments were cut off' if cut else reason[:160]}")
        ev = event_log.emit_event("tool", event_log.describe_tool(name, args), tool=name, status="running", tool_call_id=tc.id)
        event_log.finish_event(ev, ok=False, result="permanent_failure",
                     detail="Not run: the model's arguments were cut off by the length limit." if cut
                     else "Not run: " + reason.replace("NOT EXECUTED - ", "")[:300])
        shared.app_state["conversation"].append({"role": "tool", "tool_call_id": tc.id, "name": name,
                                                 "content": reason + _held_note()})


# Tools whose result can change without anything this agent did (the screen,
# windows, processes): an identical call may be repeated a few times, e.g. to
# wait for an app to finish loading. Plan bookkeeping is never "a repeat".
_PLAN_TOOLS = {"set_plan", "update_plan", "ask_user"}
_VOLATILE_TOOLS = {"observe", "inspect_screen", "wait_for_window", "list_windows", "get_active_window",
                   "list_processes", "list_elements", "top_processes", "get_system_info", "roblox_get_output",
                   "roblox_status", "roblox_get_selection", "roblox_get_tree"}


MAX_SAME_TOOL_BATCH = 30


def _same_tool_batch(parsed) -> bool:
    """Several calls of ONE tool at once (a model placing ten parts in one reply): independent items of one
    batch, not a chain of guesses - kept together so they are one step and one approval card."""
    names = {n for _tc, n, _a in parsed}
    return len(names) == 1 and len(parsed) <= MAX_SAME_TOOL_BATCH and not names & _PLAN_TOOLS


def _decide(name: str, args, tainted: bool) -> tuple:
    """permissions.explain, plus "Allow all like this for this request" from an approval card: a tool the user
    allowed for the rest of this request runs without asking again - only where the permission settings would
    ask (never an always-ask tool, a message send, lockdown, or a request that read content trying to direct
    the assistant; permissions.request_grantable). The grants end with the request (agent/tasks)."""
    settings = shared.app_state["settings"]
    bucket, why = permissions.explain(name, settings, tainted=tainted, args=args)
    guard = url_guard.check(name, args) if bucket != "deny" else None
    if guard and (guard[0] == "deny" or bucket == "auto"):
        return guard                            # an address a page could have built, or one carrying a key
    if (bucket == "ask" and name in (shared.app_state.get("request_grants") or ())
            and permissions.request_grantable(name, settings, tainted=tainted, args=args)):
        return "auto", "you allowed it for the rest of this request"
    return bucket, why


def _is_repeat(run_sigs: dict, epoch: int, name: str, args) -> bool:
    if name in _PLAN_TOOLS:
        return False
    prev = run_sigs.get(ctx.signature(name, args))
    if not prev or prev[0] != epoch:
        return False
    return prev[1] >= (config.MAX_VOLATILE_REPEATS if name in _VOLATILE_TOOLS else 1)


def _apply_effort_change(budget) -> None:
    """A level change requested during the run takes effect here - between
    tool steps, never mid-call. Usage so far is kept; only the limits change."""
    new = budget.apply_pending()
    if new is None:
        return
    shared.app_state["effort_thinking"] = new.thinking
    shared.app_state["effort_native"] = new.native
    shared.app_state["iteration_budget"] = {"local": new.local_iterations, "cloud": new.cloud_iterations}
    runtime.tool_exec.set_budget(new.max_tool_calls)
    run_control.set_deadline(new.time_limit_s - (time.time() - budget.started))
    event_log.log_action(f"Effort changed to {new.label} mid-run; applied at the next step.")
    # Only the run's budget changes here; its plan and reviewer were set up at the start (said in the label).
    event_log.emit_activity("effort_changed", f"Effort changed to {new.label} - its limits apply from this step "
                            "(planning and review keep the starting level)",
                  effort=new.name, limits=new.limits(), usage=budget.usage())


def _budget_stop(budget, spent: str) -> dict:
    """The run used up its effort budget: end cleanly with what was done."""
    what = {"step": f"the step limit ({budget.step_limit} model steps)",
            "token": f"the token budget ({budget.profile.token_budget:,} tokens)"}[spent]
    event_log.log_action(f"Run reached {what} for {budget.profile.label} effort; closing with a summary.")
    event_log.emit_activity("budget_reached", f"Reached {what} for {budget.profile.label} effort", level="warn",
                  usage=budget.usage())
    summary = final_summary()
    outstanding = shared.task_board.render(compact=False)
    return {"status": "success", "budget_reached": spent,
            "message": summary or (f"I stopped at {what} for {budget.profile.label} effort. "
                                   f"Where the task stands:\n\n{outstanding}"),
            "plan": shared.task_board.as_dict()}


def run_agent_loop():
    shared.app_state["last_tool_signature_set"] = None
    shared.app_state.pop("force_tool_once", None)        # a stopped run's unused one never reaches the next request
    local = shared.app_state.get("mode") == config.MODE_LOCAL
    # The orchestrator sets a budget from the effort profile; otherwise the defaults.
    budget = shared.app_state.get("iteration_budget") or {}
    iterations = (budget.get("local") or config.MAX_LOCAL_ITERATIONS) if local else (budget.get("cloud") or config.MAX_AGENT_ITERATIONS)
    shared.app_state["turn_engine"] = "agent"          # this result can be revised by re-entering the loop
    nudges = 0
    blocked_questions = 0
    bad_rounds = 0                # consecutive replies whose every tool call was unusable
    repeat_rounds = 0             # consecutive replies whose every tool call was a repeat
    run_sigs = {}                 # call signature -> (mutation epoch, times run in that epoch)
    mutation_epoch = 0            # bumps whenever a tool changes the machine
    ran_ok = 0                    # tool calls that succeeded in this run (1.4.8: the fabrication net needs none)
    run_budget = shared.app_state.get("run_budget")    # effort.RunBudget for orchestrated runs

    iteration = -1
    while True:
        iteration += 1
        # ---- safe boundary: between steps, never inside a model or tool call ----
        if run_budget is not None:
            _apply_effort_change(run_budget)
            spent = run_budget.exhausted()
            if spent:
                return _budget_stop(run_budget, spent)
        elif iteration >= iterations:
            break
        run_control.checkpoint(on_pause=event_log._announce_pause)
        asked_at = time.time()
        model_ev = event_log.emit_event("model", "Model is working", status="running", step=iteration + 1)
        call_id = model_ev.get("id") if isinstance(model_ev, dict) else None
        shared.app_state["_stream_call"] = call_id
        shared.app_state["running_model_event"] = model_ev      # agent/models: "loading the model" / "reading" on it
        if call_id is not None:
            stream_hub.hub.publish("start", call=call_id, step=iteration + 1, model=llm._current_model() or "")

        def end_stream(**fields):
            if call_id is not None:
                stream_hub.hub.publish("end", call=call_id, **fields)
            shared.app_state["_stream_call"] = None
        try:
            response = llm.call_model()
        except RuntimeError as e:
            event_log.finish_event(model_ev, ok=False, detail=_user_safe(str(e), 200))
            end_stream(finish="error", has_tools=False)
            return {"status": "error", "message": str(e)}
        except Exception as e:
            event_log.finish_event(model_ev, ok=False, detail=_user_safe(str(e), 200))
            end_stream(finish="error", has_tools=False)
            event_log.log_action(f"Unexpected model error: {e}")
            return {"status": "error", "message": f"Model call failed: {str(e)[:250]}"}
        except BaseException:
            event_log.finish_event(model_ev, ok=False, detail="Stopped")
            end_stream(finish="stopped", has_tools=False)
            raise
        shared.app_state["running_model_event"] = None
        facts = _model_call_facts(response, asked_at)
        if run_budget is not None:
            run_budget.record_call(facts.get("tokens_in"), facts.get("tokens_out"))
            run_budget.tool_calls = runtime.tool_exec.calls
            facts["run_usage"] = run_budget.usage()
        event_log.finish_event(model_ev, ok=True, **facts)
        # Stopped while the model was thinking: discard its reply, run nothing.
        try:
            run_control.checkpoint(on_pause=event_log._announce_pause)
        except BaseException:
            end_stream(finish="stopped", has_tools=False)
            raise

        msg = response.choices[0].message
        parsed, prose = parsing.parse_response(msg)
        if shared.app_state.get("raw_mode"):
            # Raw mode offers no tools, so nothing in the reply is a call: what the model wrote is the answer, as is.
            raw_text = model_output.split_reasoning(getattr(msg, "content", "") or "")[0]
            parsed, prose = [], (raw_text or prose or "").strip()
        end_stream(finish=getattr(response.choices[0], "finish_reason", None) or "stop", has_tools=bool(parsed),
                   aborted=getattr(response, "aborted", None))
        # Streamed local replies know when the thinking ended; otherwise the whole call counts.
        think_s = getattr(msg, "thinking_s", None)
        if not isinstance(think_s, (int, float)):
            think_s = facts.get("duration_ms", 0) / 1000
        thought_id = _emit_model_note(msg, prose if parsed else "", [n for _tc, n, _a in parsed], think_s)

        if not parsed:
            policy = shared.app_state.get("turn_policy") or {}
            # (1.4.8) After a real tool call succeeded, "Done - I created `notes.txt`" is a report, not a fabrication.
            fabricated = _fabricated_tool_name(prose) if policy.get("allow_mutation") and not ran_ok else None
            if fabricated and nudges < config.MAX_NUDGES:
                # This check is NOT gated to local mode - the bug report showed
                # a cloud model doing the exact same thing, so the net has to
                # catch both engines.
                nudges += 1
                event_log.log_action(f"Model narrated running '{fabricated}' as if it were a tool, but that is not "
                           f"a real tool and nothing was actually called - nudging ({nudges}/{config.MAX_NUDGES}).")
                shared.app_state["conversation"].append({"role": "assistant", "content": prose})
                shared.app_state["conversation"].append({"role": "user", "content": (
                    f"[SYSTEM] You described running '{fabricated}', but that is not one of your real "
                    "tools and no tool call was actually made - nothing happened on the machine, no "
                    "matter what your last message said. Either call one of your ACTUAL available tools "
                    "now, or tell the user plainly that you are not able to do this.")})
                continue
            # Small local models very often announce the action instead of
            # calling the tool ("I will now open VirtualBox.") and then stop,
            # which reads to the user as the agent giving up halfway. If a plan
            # is still open, nudge once or twice rather than ending the turn.
            if local and nudges < config.MAX_NUDGES and shared.task_board.steps and not shared.task_board.is_complete():
                nudges += 1
                active = shared.task_board.active_step()
                target = f"step {active['n']}: {active['text']}" if active else "the next step"
                event_log.log_action(f"Local model replied without calling a tool - nudging ({nudges}/{config.MAX_NUDGES}).")
                shared.app_state["conversation"].append({"role": "assistant", "content": prose})
                shared.app_state["conversation"].append({"role": "user", "content": (
                    f"[SYSTEM] You replied with text but called no tool, and {target} is not done. "
                    "Do not describe the action - make the tool call now. If you are actually finished, "
                    "mark the remaining steps with update_plan first.")})
                # The retry may only be a tool call (llama.cpp's tool_choice "required", 1.4.6): asked in words alone, a
                # small model often described the step a second time and used up both retries. update_plan is
                # offered with the plan, so "I'm finished" stays possible; every call still goes through approval.
                shared.app_state["force_tool_once"] = True
                continue
            shared.app_state["conversation"].append({"role": "assistant", "content": prose})
            # Say what really happened: an empty reply is not a confirmation.
            message = prose or "The model finished without writing a reply. Check the steps above for what was done."
            # Grounding check: the model's own prose is not the source of
            # truth about what happened - the task board is (and, per
            # update_plan above, the task board itself can no longer be
            # talked into 'done' by an unverified action). If steps remain
            # failed/unfinished while the model believes it's done talking,
            # say so plainly instead of only showing whatever the model
            # chose to narrate.
            if shared.task_board.steps:
                unfinished = [s for s in shared.task_board.steps if s["status"] not in ("done", "skipped")]
                if unfinished:
                    summary = "; ".join(
                        f"step {s['n']} [{s['status']}]: {s['text']}" + (f" - {s['note']}" if s.get("note") else "")
                        for s in unfinished)
                    message += f"\n\n(Task board shows this was not fully completed: {summary})"
            if fabricated:
                # Nudges exhausted and it's STILL fabricating - do not let the
                # false claim reach the user unlabeled.
                message += (f"\n\n(Note: this response describes running '{fabricated}', but that is not "
                            "a real tool - no action was actually taken on your machine.)")
            return {"status": "success", "message": message,
                    "plan": shared.task_board.as_dict()}

        # Roblox's Studio MCP needs studio_id and datamodel_type on every call; with one Studio open
        # there is one right answer, filled in before validation and the approval card (studio_mcp.py).
        parsed = [(tc, n, studio_mcp.fill_args(n, a)) for tc, n, a in parsed]

        # A local model asked for several calls at once is guessing at steps it
        # has not seen the result of yet - the second call is built on an
        # assumption about the first. Keep the first, drop the rest; the loop
        # comes straight back round with the real result in hand. The exception
        # is one tool called for many items at once (ten roblox_create_instance
        # calls): those don't build on each other, and kept together they are one
        # step and one approval card instead of ten.
        # The model is told which ones were held back (1.4.6): they vanished without a word, and it carried on as
        # if they had run (audit, 31 Oct 2026). The note rides on the kept call's result (_append_tool_result).
        shared.app_state["held_back_calls"] = None
        if local and len(parsed) > 1 and not _same_tool_batch(parsed):
            held = [f"{n} ({event_log.describe_tool(n, a)})" for _tc, n, a in parsed[1:]]
            parsed = parsed[:1]
            shared.app_state["held_back_calls"] = held
            event_log.log_action(f"Local mode: kept the first tool call, deferred {len(held)} more until its result is known.")

        # Real tools this message may not use end the run. A name that isn't a
        # tool at all is answered below with the real list instead: ending the
        # run on it left a model that guessed "roblox_list" on a read turn with
        # no way to find roblox_get_tree (D36).
        disallowed = [(tc, name, args) for tc, name, args in parsed
                      if schemas.known_tool(name) and not schemas.tool_allowed_current_turn(name)]
        if disallowed:
            names = ", ".join(sorted({name for _tc, name, _args in disallowed}))
            policy = shared.app_state.get("turn_policy") or {}
            event_log.log_action(f"Blocked stale/unauthorized tool proposal before approval: {names}.")
            shared.task_board.clear()
            if policy.get("mode") == "read":
                message = ("I stopped before changing anything because your current request is read-only. "
                           "Please rephrase it as an explicit action if you want me to change the computer.")
            else:
                message = (f"I did not run {names}: your message reads as a question or a comment, not a request to "
                           "change something. If you want me to do it, say so - for example \"make ...\", \"add ...\" "
                           "or \"open ...\".")
            shared.app_state["conversation"].append({"role": "assistant", "content": message})
            return {"status": "success", "message": message, "plan": shared.task_board.as_dict()}

        # Unreadable or invalid calls are answered with an error below and never
        # reach the approval card or the executor.
        invalid = [(tc, n, a, why) for tc, n, a in parsed for why in [_invalid_call_reason(n, a)] if why]
        if invalid:
            bad = {id(tc) for tc, *_ in invalid}
            parsed = [(tc, n, a) for tc, n, a in parsed if id(tc) not in bad]

        # Clarification is a first-class UI response, not a fake tool result
        # that the model must remember to repeat in prose on its next turn.
        clarification = next((((a.get("question") or "").strip(), a.get("options") or [])
                              for _tc, n, a in parsed if n == "ask_user"), None)
        if clarification:
            question, raw_options = clarification
            question = question or "Could you clarify what you want me to do?"
            options = parsing._normalize_clarification_options(question, raw_options)

            expected = (shared.app_state.get("intent_guard") or {}).get("expected_app", "")
            # (1.4.8, audit 2.11) a leftover hardcoded "proton" blocked any question that mentioned it.
            previous_apps = list(shared.app_state.get("completed_app_tasks") or [])
            unrelated = next((name for name in previous_apps
                              if name.casefold() not in expected.casefold()
                              and name.casefold() in (question + " " + " ".join(options)).casefold()), None)
            if expected and unrelated:
                blocked_questions += 1
                if blocked_questions >= 2:
                    return {"status": "error", "message": "Stopped this task after repeated unrelated clarification attempts; no further action was taken."}
                shared.app_state["conversation"].append({"role": "user", "content": (
                    f"[CURRENT TASK] Work only on {shared.app_state.get('current_user_request')}. "
                    f"The application is {expected}, not {unrelated}. Inspect its controls; "
                    "do not repeat an unrelated question from previous tasks.")})
                event_log.log_action(f"Blocked unrelated {unrelated} clarification for the current app.")
                continue

            resolved = shared.app_state.get("resolved_clarification") or {}
            if resolved and parsing._same_clarification(question, resolved.get("question", "")):
                blocked_questions += 1
                if blocked_questions >= 2:
                    return {"status": "error", "message": "The model repeated an answered question twice. This task was stopped without repeating completed actions."}
                answer = resolved.get("answer", "")
                event_log.log_action(f"Blocked repeated clarification after the user already answered: {question}")
                shared.app_state["conversation"].append({"role": "assistant", "content": prose or ""})
                shared.app_state["conversation"].append({"role": "user", "content": (
                    f"[SYSTEM] Do not ask that question again. The user already answered '{answer}'. "
                    "Continue the original request now, using that answer.")})
                continue
            if resolved and shared.app_state.get("prep_mode") == "clarify":
                # Prompt Prep "Clarify" allows one question per request, and it was asked.
                blocked_questions += 1
                if blocked_questions >= 2:
                    return {"status": "error", "message": "The model kept asking questions after its one question "
                                                          "was answered. This task was stopped; nothing else ran."}
                event_log.log_action(f"Blocked a second question (Prompt Prep: Clarify allows one): {question}")
                shared.app_state["conversation"].append({"role": "assistant", "content": prose or ""})
                shared.app_state["conversation"].append({"role": "user", "content": (
                    f"[SYSTEM] Prompt Prep allows one question per request and it was already answered "
                    f"('{resolved.get('answer', '')}'). Do not ask another. Continue with the most reasonable reading "
                    "and say in your reply what you assumed.")})
                continue

            pending = {
                "question": question,
                "options": options,
                "option_values": options,
                "original_request": shared.app_state.get("current_user_request") or "",
                "created_at": time.time(),
                "turn_policy": dict(shared.app_state.get("turn_policy") or {}),
            }
            shared.app_state["pending_clarification"] = pending
            shared.app_state["conversation"].append({"role": "assistant", "content": question})
            return {"status": "clarification", "message": question, "question": question,
                    "options": options, "option_values": options, "allow_other": True,
                    "plan": shared.task_board.as_dict()}

        shared.app_state["conversation"].append({
            "role": "assistant", "content": prose,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args, default=str)}}
                           for tc, name, args in parsed + [(tc, n, a) for tc, n, a, _why in invalid]],
        })
        if invalid:
            _answer_invalid_calls(invalid)
            if not parsed:
                bad_rounds += 1
                if bad_rounds >= config.MAX_BAD_CALL_ROUNDS:
                    message = ("I stopped because my tool calls kept arriving incomplete or invalid, so nothing was run. "
                               "The content was probably too long for one step - try asking for it in smaller parts.")
                    shared.app_state["conversation"].append({"role": "assistant", "content": message})
                    return {"status": "error", "message": message, "plan": shared.task_board.as_dict()}
                continue          # the model sees the error and makes the call again, smaller
        bad_rounds = 0

        # ---- loop detection -----------------------------------------------
        # The same call with the same arguments, already run in this turn with
        # nothing changed on the machine since, is not run again. (The old
        # breaker only compared against the previous batch, so A, A, A ran
        # every other time.)
        repeats = [(tc, n, a) for tc, n, a in parsed if _is_repeat(run_sigs, mutation_epoch, n, a)]
        if repeats:
            rep_ids = {id(tc) for tc, _n, _a in repeats}
            for tc, n, a in repeats:
                event_log.log_action(f"Did not repeat {n}: identical call already ran and nothing has changed since.")
                event_log.emit_event("note", f"Skipped a repeated call: {event_log.describe_tool(n, a)}",
                           detail="The model asked for the exact same call again; its result is already in the conversation.")
                shared.app_state["conversation"].append({"role": "tool", "tool_call_id": tc.id, "name": n, "content": (
                    "NOT EXECUTED - you already made this exact call in this task and nothing on the machine has "
                    "changed since; its result is above in this conversation. Use that result: take a DIFFERENT "
                    "next action, mark the step with update_plan, or give the user your final answer.") + _held_note()})
            parsed = [x for x in parsed if id(x[0]) not in rep_ids]
            if not parsed:
                repeat_rounds += 1
                if repeat_rounds >= config.MAX_REPEAT_ROUNDS:
                    event_log.log_action("Model kept repeating the same call - ending the run with a summary.")
                    event_log.emit_event("note", "Stopped: the model kept repeating the same action")
                    summary = final_summary()
                    return {"status": "success",
                            "message": summary or ("I stopped because the same action kept being repeated without "
                                                   "anything changing. Here is where the task stands:\n\n"
                                                   + shared.task_board.render(compact=False)),
                            "plan": shared.task_board.as_dict()}
                continue
        repeat_rounds = 0

        step_ran = []                    # (tool, ok) of this step's calls - a picture may end the turn (below)

        def run_and_record(tc, n, a):
            nonlocal mutation_epoch, ran_ok
            res = runtime.execute_tool_guarded(n, a, tool_call_id=tc.id, links=_step_links(thought_id, a))
            step_ran.append((n, bool(res.get("ok"))))
            ran_ok += 1 if res.get("ok") else 0
            if res.get("ok") and n not in schemas.READ_ONLY_TOOLS and n not in _PLAN_TOOLS:
                mutation_epoch += 1             # the machine changed: earlier reads may be stale now
            sig = ctx.signature(n, a)
            prev = run_sigs.get(sig)
            run_sigs[sig] = (mutation_epoch, (prev[1] + 1) if prev and prev[0] == mutation_epoch else 1)
            return res

        auto_run, need_approval, denied = [], [], []
        tainted = bool(shared.app_state.get("untrusted_seen"))
        for tc, n, a in parsed:
            bucket, why = _decide(n, a, tainted)
            (auto_run if bucket == "auto" else need_approval if bucket == "ask" else denied).append((tc, n, a, why))

        for tc, n, a, why in auto_run:
            # Decided before any call of this batch ran - but an earlier call can
            # change what this one does (browser_open + browser_read, then a click
            # on the NEW page's element 7) or taint the request. Decide again now.
            bucket, why = _decide(n, a, bool(shared.app_state.get("untrusted_seen")))
            if bucket != "auto":
                event_log.log_action(f"{n} now needs {'approval' if bucket == 'ask' else 'to be refused'}: {why}.")
                (need_approval if bucket == "ask" else denied).append((tc, n, a, why))
                continue
            event_log.log_action(f"Auto-running {n} ({why}).")
            _append_tool_result(tc, n, run_and_record(tc, n, a))

        denied = [(tc, n, a) for tc, n, a, _ in denied]
        need_approval = [(tc, n, a) for tc, n, a, _ in need_approval]
        for tc, n, _a in denied:
            event_log.log_action(f"BLOCKED {n}: set to 'deny' in your permission settings.")
            shared.app_state["conversation"].append({
                "role": "tool", "tool_call_id": tc.id, "name": n,
                "content": (f"NOT EXECUTED - '{n}' is set to DENY in the user's permission settings "
                            "(Settings > Permissions). It will never run. Tell the user, or try a "
                            "different approach that does not need this tool.") + _held_note(),
            })

        if need_approval:
            event_log.log_action(f"Agent wants to run {len(need_approval)} action(s). Waiting for approval...")
            shared.app_state["waiting_for_os_approval"] = True
            actions = [{"tool_name": n, "args": a, "justification": a.get("justification", "")} for _, n, a in need_approval]
            try:
                approved = approvals.ask_batch_approval_os(actions)
            finally:
                shared.app_state["waiting_for_os_approval"] = False

            if approved:
                event_log.log_action(f"User approved {len(need_approval)} action(s).")
                before = ctx.snapshot()
                for (tc, n, a), shown in zip(need_approval, actions):
                    changed = approvals._changed_since_approval(shown.get("view"))
                    if changed:
                        _append_tool_result(tc, n, {"ok": False, "message": changed})
                        continue
                    res = run_and_record(tc, n, a)
                    event_log.log_action(f"{n}: {'ok' if res.get('ok') else 'FAILED'} - {res.get('message', '')[:150]}")
                    _append_tool_result(tc, n, res)
                # Observed, not assumed: tell the model what actually moved on
                # the machine so it verifies instead of re-firing the action.
                delta = ctx.snapshot_delta(before, ctx.snapshot())
                _append_observation(f"After those actions, the machine shows: {delta}. "
                                    "Mark the finished step with update_plan before continuing.")
            else:
                event_log.log_action("User rejected the batch.")
                rejected_names = ", ".join(sorted({n for _, n, _ in need_approval}))
                for tc, n, _a in need_approval:
                    shared.app_state["conversation"].append({
                        "role": "tool", "tool_call_id": tc.id, "name": n,
                        "content": ("The user REJECTED this action. Do not retry it. Ask the user what they "
                                    "would prefer, or propose a different approach.") + _held_note(),
                    })
                # Without this, the active step stays [>] and the LIVE MACHINE
                # STATE block re-asserts "YOUR NEXT TOOL CALL MUST ADVANCE STEP
                # N" every subsequent turn - the exact wording that pressured
                # the model into inventing a DIFFERENT tool call for the same
                # rejected idea instead of stopping to re-plan or ask. Marking
                # the step failed removes that pointer, so the prompt's
                # "rejected -> call set_plan again or ask" rule actually has
                # room to take effect.
                active = shared.task_board.active_step()
                if active:
                    shared.task_board.update(active["n"], "failed",
                                       f"User rejected: {rejected_names}")
                return {"status": "success", "rejected": True,
                        "message": "I stopped because you rejected the proposed actions. Tell me how you'd like to proceed.",
                        "plan": shared.task_board.as_dict()}

        # Pictures that are the whole answer: done now, without another model call (see _MORE_THAN_A_PICTURE).
        made = shared.app_state.get("turn_images") or []
        # Pictures from the web too (1.4.7): "show me pictures of the Nothing Phone 4a" - once they are shown, a small model
        # kept searching for more (an 81 % full context, a minute more) instead of ending.
        if (step_ran and made and all(n in ("generate_image", "show_web_images") and ok for n, ok in step_ran)
                and not (shared.task_board.steps and not shared.task_board.is_complete())
                and not _wants_more_than_pictures(shared.app_state.get("current_user_request") or "")):
            message = picture_reply(prose, made)
            shared.app_state["conversation"].append({"role": "assistant", "content": message})
            event_log.log_action("The picture is the answer - finished without asking the model again.")
            return {"status": "success", "message": message, "plan": shared.task_board.as_dict()}

    event_log.log_action("Step limit reached; asking for a closing summary.")
    summary = final_summary()
    outstanding = shared.task_board.render(compact=False)
    return {"status": "success",
            "message": summary or f"I hit the step limit for this turn. Where the task stands:\n\n{outstanding}",
            "plan": shared.task_board.as_dict()}
