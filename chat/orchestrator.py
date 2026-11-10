"""
orchestrator.py
===============
The run pipeline around the existing execution engine:

    request -> PromptPreparer -> Planner -> Executor -> Reviewer
            -> (bounded Revision -> Reviewer)* -> final response

  * TaskState      one run: immutable original request, enhanced prompt,
                   plan, criteria, tool results, verification, events.
  * EffortProfile  what each effort level actually changes: which stages
                   run, revision/handoff/tool/time limits, the runtime's
                   native reasoning effort.
  * PromptPreparer extracts intent / constraints / criteria. Never rewrites
                   or replaces the request - both are passed downstream.
  * Planner        structured plan; does not execute anything.
  * Reviewer       evidence-based findings (deterministic checks always,
                   a model review at higher effort). Stores concise findings
                   only - never model reasoning.
  * AgentOrchestrator  the controller. The Executor stage is the existing
                   chat_with_agent (deterministic desktop routes + agent
                   loop), injected as a callable, so nothing it did before is
                   bypassed.

Every stage works without a model: model-backed steps fall back to
deterministic rules when no model answers, so the pipeline never blocks on
an unavailable provider.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass

from run_control import RunStopped

import effort as effort_mod
import prompt_prep

# The effort system lives in effort.py; these names are kept for callers.
EFFORTS = effort_mod.LEVELS
# What the user sees. Stored values stay low/normal/high/max(/extra/ultra) so
# saved settings keep working; the labels describe the job each level suits,
# since no level is "normal" for everyone.
EFFORT_LABELS = effort_mod.LABELS
ENHANCEMENT_MODES = ("off", "minimal", "standard", "detailed")
FINDING_TYPES = ("complete", "missing_requirement", "incorrect_result", "questionable_assumption",
                 "tool_failure", "factual_inconsistency", "improvement_suggested", "revision_required", "blocked")
SEVERITIES = ("info", "low", "medium", "high")


# ==============================================================================
# Effort
# ==============================================================================
EffortProfile = effort_mod.EffortProfile
EFFORT_PROFILES = effort_mod.PROFILES

_GREETING = re.compile(r"^\s*(hi|hello|hey|yo|thanks?|thank you|thx|ok(ay)?|cool|nice|good (morning|night|evening))\W*\s*$", re.I)
_COMPLEX = re.compile(r"\b(refactor|migrate|rewrite|implement|build|set ?up|audit|every|all (of )?the|multiple|step[- ]by[- ]step|"
                      r"compare|research|investigate|debug|fix (all|every))\b", re.I)


def analyze(request: str, mode: str, parts: int = 1) -> dict:
    """Cheap request analysis. `mode` is the turn policy (none/read/action)."""
    words = len(re.findall(r"\w+", request or ""))
    trivial = bool(_GREETING.match(request or "")) or (mode == "none" and words <= 30)
    complex_ = (not trivial) and (parts >= 3 or words >= 60 or bool(_COMPLEX.search(request or "")))
    return {"mode": mode, "words": words, "parts": parts, "trivial": trivial, "complex": complex_}


def choose_effort(setting: str, analysis: dict, auto_escalate: bool = True) -> tuple[str, bool]:
    """Returns (effort, escalated). A genuinely complex task at Balanced
    ("normal") is raised to Thorough ("high"); nothing is ever raised higher
    automatically, and Quick ("low") is respected as an explicit request for speed."""
    effort = setting if setting in EFFORTS else "normal"
    if auto_escalate and effort == "normal" and analysis.get("complex") and analysis.get("mode") == "action":
        return "high", True
    return effort, False


def planner_mode(profile: EffortProfile, analysis: dict) -> str:
    if analysis["trivial"] or profile.planner == "off":
        return "off"
    if profile.planner == "heuristic":
        return "model" if analysis["complex"] else ("heuristic" if analysis["mode"] == "action" else "off")
    return profile.planner


def reviewer_mode(profile: EffortProfile, analysis: dict, mutations: int) -> str:
    if analysis["trivial"]:
        return "off"                   # "hi" / a short question: nothing to verify
    if profile.name == "normal" and mutations and analysis["mode"] == "action":
        return "model"
    return profile.reviewer


# ==============================================================================
# Task state
# ==============================================================================
class TaskState:
    """One run. `original_request` is read-only for the life of the task:
    enhancement, planning, handoff and revision add alongside it, never over it."""

    def __init__(self, request: str, effort: str = "normal", task_id: str | None = None, emit=None, clock=time.time):
        self.__dict__["_original"] = request
        self.task_id = task_id or uuid.uuid4().hex[:12]
        self.effort = effort
        self.enhanced: dict | None = None
        self.plan: dict | None = None
        self.requirements: list = []
        self.constraints: list = []
        self.success_criteria: list = []
        self.phase = "created"
        self.completed_steps: list = []
        self.remaining_steps: list = []
        self.context_refs: list = []
        self.tool_results: list = []
        self.approval_state = "none"
        self.active_model: str | None = None
        self.model_history: list = []
        self.revision_count = 0
        self.handoff_count = 0
        self.cancelled = False
        self.events: list = []
        self.verification: dict = {"status": "not_run", "findings": []}
        self.prep_mode = "off"                 # prompt_prep.MODES, fixed for the run
        self.command: str | None = None        # the slash command it was sent with ("plan" | "design")
        self.spec: str | None = None           # Expand: the spec text the user sent
        self.plan_accepted = False             # Plan first: the user accepted state.plan
        self.started_at = clock()
        self._emit = emit
        self._clock = clock

    @property
    def original_request(self) -> str:
        return self.__dict__["_original"]

    def __setattr__(self, name, value):
        if name in ("original_request", "_original"):
            raise AttributeError("The original request is immutable.")
        super().__setattr__(name, value)

    def event(self, kind: str, summary: str, **data):
        entry = {"kind": kind, "summary": summary, "at": self._clock(), **data}
        self.events.append(entry)
        if self._emit:
            self._emit(kind, summary, **data)
        return entry

    def record_handoff(self, previous: str | None, new: str, reason: str):
        if previous and previous not in self.model_history:
            self.model_history.append(previous)
        self.handoff_count += 1
        self.active_model = new
        self.event("handoff_recorded", f"{previous or 'auto'} → {new}", level="info",
                   handoff={"from": previous, "to": new, "reason": reason})

    def additions(self) -> list:
        """What preparation and planning add to the request, in brief order:
        [{"key", "label" (for people), "items" | "text", "line" (as sent)}].
        brief() is built from this, so the chat can show exactly what was added."""
        out = []
        if self.spec:
            out.append({"key": "spec", "label": "Spec you sent", "text": self.spec,
                        "line": "PREPARED SPEC (the user reviewed and sent this before the run):\n" + self.spec})
        elif self.enhanced:
            if self.enhanced.get("intent"):
                out.append({"key": "intent", "label": "Understood as", "text": self.enhanced["intent"],
                            "line": f"Interpreted intent: {self.enhanced['intent']}"})
            for label, key in (("Constraints", "constraints"), ("Open questions", "ambiguities")):
                if self.enhanced.get(key):
                    out.append({"key": key, "label": label, "items": list(self.enhanced[key]),
                                "line": f"{label}: " + "; ".join(self.enhanced[key])})
        if self.prep_mode == "clarify":
            out.append({"key": "clarify", "label": "Clarify", "text": "It may ask you one question before acting.",
                        "line": "Prompt Prep is Clarify: if the request is ambiguous in a way that changes the result, ask ONE "
                                "question with ask_user before acting. Only one question is allowed for this request."})
        if self.plan:
            steps = list(self.plan.get("steps") or [])
            if steps and self.plan_accepted:
                out.append({"key": "plan", "label": "Plan you accepted", "items": steps,
                            "line": "ACCEPTED PLAN (the user accepted these steps before any action; every action still needs "
                                    "its normal approval): " + " | ".join(f"{i}. {s}" for i, s in enumerate(steps, 1))})
            elif steps:
                out.append({"key": "steps", "label": "Suggested steps", "items": steps,
                            "line": "Suggested steps: " + " | ".join(f"{i}. {s}" for i, s in enumerate(steps, 1))})
            if self.plan.get("risks"):
                out.append({"key": "risks", "label": "Needs approval", "items": list(self.plan["risks"]),
                            "line": "Risks needing approval: " + "; ".join(self.plan["risks"])})
        if self.success_criteria:
            out.append({"key": "done", "label": "Done when", "items": list(self.success_criteria),
                        "line": "Done when: " + "; ".join(self.success_criteria)})
        return out

    def brief(self) -> str:
        """What the executor sees in addition to the request itself."""
        lines = [f"ORIGINAL REQUEST (verbatim, authoritative): {self.original_request}"]
        lines += [a["line"] for a in self.additions()]
        lines.append("If the enhanced notes disagree with the original request, the original request wins.")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {"task_id": self.task_id, "original_request": self.original_request, "enhanced": self.enhanced,
                "plan": self.plan, "requirements": self.requirements, "constraints": self.constraints,
                "success_criteria": self.success_criteria, "phase": self.phase, "effort": self.effort,
                "completed_steps": self.completed_steps, "remaining_steps": self.remaining_steps,
                "tool_results": self.tool_results[-20:], "approval_state": self.approval_state,
                "active_model": self.active_model, "model_history": list(self.model_history),
                "revision_count": self.revision_count, "handoff_count": self.handoff_count,
                "cancelled": self.cancelled, "verification": self.verification}


# ==============================================================================
# Stages
# ==============================================================================
_CONSTRAINT_RE = re.compile(r"[^.;\n]*\b(only|don'?t|do not|never|without|must|keep|avoid|no more than|at most|except)\b[^.;\n]*", re.I)
_SPLIT_RE = re.compile(r"\s*(?:;|\bthen\b|,\s*and\b|\band then\b|\n)\s*", re.I)
_TOOL_HINTS = (
    (r"\b(open|launch|start)\b", "open_application"), (r"\bclick|press|button\b", "click_by_name"),
    (r"\b(file|folder|directory)\b", "read_file / list_dir"), (r"\b(delete|remove)\b", "delete_path"),
    (r"\b(script|studio|roblox|part)\b", "roblox_*"), (r"\b(screen|see|look)\b", "inspect_screen"),
    (r"\b(message|reply|text|dm)\b", "messages_*"), (r"\b(run|command|terminal)\b", "execute_command"),
)
_RISK_HINTS = ((r"\b(delete|remove|wipe|erase)\b", "Deleting data"), (r"\b(send|post|reply)\b", "Sending a message"),
               (r"\b(overwrite|replace|rewrite)\b", "Overwriting content"), (r"\b(install|uninstall)\b", "Installing software"),
               (r"\b(kill|terminate|close)\b", "Closing apps or processes"), (r"\b(run|execute)\b", "Running commands"))


def _clauses(text: str) -> list:
    parts = [p.strip(" .") for p in _SPLIT_RE.split(text or "") if p and p.strip(" .")]
    return parts[:8] or ([text.strip()] if text and text.strip() else [])


def _strs(value, limit=8, width=200) -> list:
    if isinstance(value, str):
        value = [value]
    return [str(v).strip()[:width] for v in (value or []) if str(v).strip()][:limit]


def _with_repeat(request: str, constraints: list) -> list:
    """Prompt Prep's rewrite of repeat-style requests ("stack 10 parts"): one script, one loop - first,
    so the Expand card and the brief lead with it (prompt_prep.repeat_constraint)."""
    rule = prompt_prep.repeat_constraint(request)
    return ([rule] + [c for c in constraints if c != rule])[:8] if rule else constraints


class PromptPreparer:
    def __init__(self, model_json=None):
        self.model_json = model_json

    def heuristic(self, request: str) -> dict:
        clauses = _clauses(request)
        return {"intent": clauses[0][:200] if clauses else request[:200],
                "constraints": _with_repeat(request, _strs([m.group(0) for m in _CONSTRAINT_RE.finditer(request or "")])),
                "ambiguities": [], "success_criteria": [f"Done: {c}" for c in clauses][:6],
                "metadata": {"parts": len(clauses)}, "source": "rules"}

    def prepare(self, state: TaskState, mode: str) -> dict | None:
        """Prompt Prep modes: clarify = quick rules (no model call; the one
        question, if any, is asked by the executor through ask_user); expand
        and plan = the model writes the spec. The older enhancement levels
        (minimal/standard/detailed) still work for direct callers."""
        mode = {"clarify": "minimal", "expand": "detailed", "plan": "detailed"}.get(mode, mode)
        if mode not in ENHANCEMENT_MODES or mode == "off":
            return None
        base = self.heuristic(state.original_request)
        if mode == "minimal" or not self.model_json:
            return base
        detail = "Also list genuine ambiguities and useful metadata." if mode == "detailed" else ""
        got = self.model_json(
            "You prepare a user's request for an assistant that controls their Windows PC. Do NOT rewrite the request. "
            "Return only JSON: {\"intent\": str, \"constraints\": [str], \"ambiguities\": [str], "
            "\"success_criteria\": [str], \"metadata\": {}}. " + detail,
            f"Request: {state.original_request}", 450)
        if not isinstance(got, dict):
            return base
        return {"intent": str(got.get("intent") or base["intent"])[:300],
                "constraints": _with_repeat(state.original_request, _strs(got.get("constraints"))) or base["constraints"],
                "ambiguities": _strs(got.get("ambiguities")) if mode == "detailed" else [],
                "success_criteria": _strs(got.get("success_criteria")) or base["success_criteria"],
                "metadata": got.get("metadata") if isinstance(got.get("metadata"), dict) else {}, "source": "model"}


class Planner:
    def __init__(self, model_json=None):
        self.model_json = model_json

    def heuristic(self, state: TaskState) -> dict:
        req = state.original_request
        steps = _clauses(req)
        tools = sorted({tool for pat, tool in _TOOL_HINTS if re.search(pat, req, re.I)})
        risks = [label for pat, label in _RISK_HINTS if re.search(pat, req, re.I)]
        criteria = (state.enhanced or {}).get("success_criteria") or [f"Done: {s}" for s in steps]
        return {"interpretation": steps[0] if steps else req[:200], "requirements": steps,
                "constraints": (state.enhanced or {}).get("constraints", []), "proposed_actions": steps,
                "tools_likely": tools, "risks": risks, "success_criteria": criteria, "steps": steps, "source": "rules"}

    def plan(self, state: TaskState, mode: str) -> dict | None:
        if mode == "off":
            return None
        base = self.heuristic(state)
        if mode == "heuristic" or not self.model_json:
            return base
        depth = "Be thorough: include every requirement and risk." if mode == "detailed" else "Keep it short."
        got = self.model_json(
            "You plan tasks for an assistant that controls the user's Windows PC with tools. Do not execute anything. "
            "Repeated work (the same change to many items, e.g. 10 parts) is ONE step done by one script with a loop "
            "or one batch call - never one step per item. "
            "Return only JSON: {\"interpretation\": str, \"requirements\": [str], \"constraints\": [str], "
            "\"proposed_actions\": [str], \"tools_likely\": [str], \"risks\": [str], \"success_criteria\": [str], "
            "\"steps\": [str]}. " + depth,
            f"Request: {state.original_request}\nNotes: {json.dumps(state.enhanced or {})[:1200]}", 700)
        if not isinstance(got, dict) or not _strs(got.get("steps")):
            return base
        plan = {k: _strs(got.get(k), limit=10) or base[k] for k in
                ("requirements", "constraints", "proposed_actions", "tools_likely", "risks", "success_criteria", "steps")}
        plan["interpretation"] = str(got.get("interpretation") or base["interpretation"])[:300]
        plan["source"] = "model"
        return plan


_CLAIMS_DONE = re.compile(r"\b(done|completed?|finished|succeeded|successfully|all set|i('ve| have) (opened|created|fixed|sent|moved|deleted|updated))\b", re.I)


def _finding(type_, severity, explanation, requirement="", action=""):
    return {"type": type_ if type_ in FINDING_TYPES else "improvement_suggested",
            "severity": severity if severity in SEVERITIES else "medium",
            "requirement": str(requirement or "")[:160], "explanation": str(explanation or "")[:300],
            "action": str(action or "")[:200]}


class Reviewer:
    def __init__(self, model_json=None, secondary_json=None):
        self.model_json = model_json
        self.secondary_json = secondary_json

    def deterministic(self, state: TaskState, result: dict, evidence: dict) -> list:
        findings = []
        status = result.get("status")
        reply = (result.get("message") or "").strip()
        if status == "error":
            findings.append(_finding("blocked", "high", reply[:200] or "The run failed.", action="Report the failure."))
            return findings
        if result.get("rejected"):
            findings.append(_finding("blocked", "info", "You declined an action, so the task stopped there.",
                                     action="Wait for the user's direction."))
            return findings
        for step in evidence.get("board") or []:
            if step.get("status") not in ("done", "skipped"):
                findings.append(_finding("missing_requirement", "high" if step.get("status") == "failed" else "medium",
                                         f"Step {step.get('n')} is {step.get('status')}: {step.get('text')}"
                                         + (f" ({step.get('note')})" if step.get("note") else ""),
                                         requirement=step.get("text", ""),
                                         action="Finish this step or tell the user why it cannot be done."))
        failed_last = {}
        for r in evidence.get("tool_results") or []:
            failed_last[r.get("tool")] = r
        failures = [r for r in failed_last.values() if r.get("status") in ("permanent_failure", "retryable_failure", "unavailable")]
        for r in failures:
            findings.append(_finding("tool_failure", "medium", f"{r.get('tool')} failed: {str(r.get('message', ''))[:160]}",
                                     action="Use a different approach or report it."))
        if failures and _CLAIMS_DONE.search(reply) and not re.search(r"\b(fail|couldn'?t|could not|unable|error|not)\b", reply, re.I):
            findings.append(_finding("factual_inconsistency", "high",
                                     "The reply says the task is done, but tool results show failures it doesn't mention.",
                                     action="Correct the reply or finish the failed part."))
        unverified = [r for r in evidence.get("tool_results") or [] if r.get("verified") is False and r.get("ok")]
        if unverified and _CLAIMS_DONE.search(reply):
            findings.append(_finding("questionable_assumption", "medium",
                                     f"{unverified[-1].get('tool')} ran but could not confirm its effect.",
                                     action="Check the real state before claiming success."))
        if not reply and status == "success":
            findings.append(_finding("incorrect_result", "high", "The run produced no reply.", action="Answer the user."))
        return findings

    def _model(self, fn, state, result, evidence, label, report=None) -> list:
        if report is not None:
            report.update(status="unavailable")
        if not fn:
            return []
        got = fn(
            "You verify whether an assistant completed a user's request. Compare the result with the ORIGINAL request, "
            "requirements and success criteria, using only the evidence given. Do not include reasoning. Return only JSON: "
            "{\"verdict\": \"complete\"|\"revision_required\"|\"blocked\", \"findings\": [{\"type\": one of "
            + json.dumps(FINDING_TYPES) + ", \"severity\": \"info\"|\"low\"|\"medium\"|\"high\", \"requirement\": str, "
            "\"explanation\": str (one sentence), \"action\": str}]}",
            json.dumps({"original_request": state.original_request, "enhanced": state.enhanced,
                        "plan_steps": (state.plan or {}).get("steps"), "success_criteria": state.success_criteria,
                        "board": evidence.get("board"), "tool_results": [
                            {k: r.get(k) for k in ("tool", "status", "verified", "message")} for r in (evidence.get("tool_results") or [])[-12:]],
                        "reply": (result.get("message") or "")[:2500]}, default=str)[:7000], 600)
        if report is not None:
            valid = isinstance(got, dict) and got.get("verdict") in ("complete", "revision_required", "blocked") and isinstance(got.get("findings"), list)
            report["status"] = "checked" if valid else "failed"
            if not valid:
                return []
        if not isinstance(got, dict):
            return []
        out = []
        for f in (got.get("findings") or [])[:8]:
            if isinstance(f, dict) and f.get("type") != "complete":
                item = _finding(f.get("type"), f.get("severity"), f.get("explanation"), f.get("requirement"), f.get("action"))
                item["source"] = label
                out.append(item)
        if got.get("verdict") == "blocked" and not any(f["type"] == "blocked" for f in out):
            out.append({**_finding("blocked", "high", "Reviewer reports that verification is blocked."), "source": label})
        if got.get("verdict") == "revision_required" and not out:
            out.append({**_finding("revision_required", "medium", "Reviewer asked for a revision."), "source": label})
        return out

    def review(self, state: TaskState, result: dict, evidence: dict, mode: str) -> dict:
        findings = self.deterministic(state, result, evidence)
        blocked = any(f["type"] == "blocked" for f in findings)
        if not blocked and mode in ("model", "cross"):
            findings += self._model(self.model_json, state, result, evidence, "model")
        cross = {"status": "skipped" if blocked else "off"}
        if not blocked and mode == "cross":
            findings += self._model(self.secondary_json, state, result, evidence, "cross-check", cross)
        needs = [f for f in findings if f["severity"] in ("medium", "high") and f["type"] in
                 ("missing_requirement", "incorrect_result", "factual_inconsistency", "revision_required")]
        blocked = any(f["type"] == "blocked" for f in findings)
        verdict = "blocked" if blocked else "revision_required" if needs else "complete"
        summary = ("Verification passed" if verdict == "complete" else
                   f"Reviewer found {len(needs)} issue{'s' if len(needs) != 1 else ''} to fix" if verdict == "revision_required"
                   else "Blocked - nothing to revise")
        return {"verdict": verdict, "findings": findings, "summary": summary, "cross_check": cross}


def findings_signature(findings: list) -> str:
    key = sorted((f["type"], f["requirement"], f["explanation"][:80]) for f in findings)
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]


def feedback_text(review: dict) -> str:
    lines = ["[REVIEWER FEEDBACK] The result does not yet satisfy the ORIGINAL request. Fix only these points, "
             "then give the final answer:"]
    for f in review["findings"]:
        if f["severity"] in ("medium", "high"):
            lines.append(f"- {f['type'].replace('_', ' ')}: {f['explanation']}" + (f" -> {f['action']}" if f["action"] else ""))
    return "\n".join(lines)


# ==============================================================================
# Controller
# ==============================================================================
class AgentOrchestrator:
    def __init__(self, *, execute, revise, settings, run_control, emit, analyze_request,
                 preparer: PromptPreparer, planner: Planner, reviewer: Reviewer,
                 tool_scope, tool_history, set_context, stopped_response, model_name=lambda: None,
                 capabilities=lambda model: {}, session_effort=lambda: None, project=lambda: None,
                 prep_mode=None, prep_gate=None, pending_answer=lambda message: None, cross_check_status=None):
        self.execute = execute                  # (request) -> result dict  [existing engine]
        self.revise = revise                    # (state, feedback) -> result dict | None
        self.settings = settings                # () -> dict
        self.run_control = run_control
        self.emit = emit                        # (kind, summary, **data)
        self.analyze_request = analyze_request  # (text, with_vision) -> analysis dict
        self.preparer, self.planner, self.reviewer = preparer, planner, reviewer
        self.tool_scope = tool_scope            # (task_id, budget) -> None
        self.tool_history = tool_history        # () -> list of tool results this task
        self.set_context = set_context          # (state|None, profile|None) -> None
        self.stopped_response = stopped_response  # (reason) -> result dict
        self.model_name = model_name
        self.capabilities = capabilities        # (model) -> detected caps (model_caps.py)
        self.session_effort = session_effort    # () -> session-only level override, or None
        self.project = project                  # () -> active project dict, or None
        # Prompt Prep (prompt_prep.py): (model) -> (mode, source); the card gate
        # (None: Expand/Plan cards are shown in Activity and sent unchanged);
        # (message) -> the pending question this message answers, or None.
        self.prep_mode = prep_mode or (lambda model: prompt_prep.choose_mode(
            self.settings() or {}, model, self.project(), None))
        self.prep_gate = prep_gate
        self.pending_answer = pending_answer
        self.cross_check_status = cross_check_status or (lambda: {"available": bool(self.reviewer.secondary_json)})
        self.current: TaskState | None = None

    def _evidence(self, board) -> dict:
        return {"board": board() if callable(board) else board, "tool_results": self.tool_history()}

    def run(self, request, board=lambda: []) -> dict:
        s = self.settings() or {}
        cross_requested = s.get("cross_check") is True
        cross_result = {"requested": cross_requested, "status": "skipped" if cross_requested else "off"}
        analysis = self.analyze_request(request.message, getattr(request, "with_vision", False))
        model = self.model_name()
        requested, source = effort_mod.choose_level(s, model, self.project(), self.session_effort())
        auto_reason = None
        if requested == "auto":
            world_question = "world" in (getattr(request, "tools", None) or ())
            if getattr(request, "persona", None) == "mossadgpt":
                from osint import tool as world_tool
                world_question = world_question or bool(world_tool.turn_hint(request.message))
            if world_question:
                analysis = {**analysis, "mode": "read", "trivial": False}
            effort, auto_reason = effort_mod.auto_level(analysis, bool(getattr(request, "voice", False)))
            escalated = False
        else:
            # Voice keeps an explicit fixed level; only Auto chooses per turn.
            effort, escalated = choose_effort(requested, analysis,
                                              s.get("effort_auto_escalate", True) and not getattr(request, "voice", False))
        effort, fallback = effort_mod.resolve(effort, self.capabilities(model) if model else {})
        profile = effort_mod.profile_for(effort, model, s.get("effort_overrides"))
        state = TaskState(request.message, effort=effort, emit=self.emit)
        state.active_model = model
        prep, prep_source = self.prep_mode(model)
        # Slash commands (agent/tasks.slash_command): /plan always stops at a plan
        # the model writes, for any request; /design is one visual answer, so it
        # is not prepared, planned or reviewed.
        command = state.command = getattr(request, "command", None)
        # Raw mode (response_modes.py) and /design are one answer each: not prepared, planned or reviewed.
        bare = command == "design" or getattr(request, "response_mode", None) == "raw"
        if command == "plan" and not bare:
            prep, prep_source = "plan", "command"
        elif bare:
            prep, prep_source = "off", "raw" if command != "design" else "command"
        state.prep_mode = prep if prep in prompt_prep.MODES else "off"
        # Answering the one question from the previous turn: that request was
        # already prepared, so no second prep (and no second card).
        answering = self.pending_answer(request.message)
        self.current = state
        cap = s.get("max_revisions")
        try:
            max_revisions = profile.max_revisions if cap in (None, "auto") else max(0, min(int(cap), profile.max_revisions))
        except (TypeError, ValueError):          # a bad stored value must not break (or wedge) the run
            max_revisions = profile.max_revisions
        self.run_control.begin(profile.time_limit_s)
        self.tool_scope(state.task_id, profile.max_tool_calls)
        try:
            state.event("run_started", f"Working · {EFFORT_LABELS[effort]} effort", effort=effort, task_id=state.task_id,
                        requested=requested, source=source, auto_reason=auto_reason, model=model, limits=profile.limits(),
                        prep=state.prep_mode, prep_label=prompt_prep.LABELS[state.prep_mode], prep_source=prep_source,
                        command=command, cross_check=cross_requested)
            if fallback:
                state.event("effort_fallback", f"{EFFORT_LABELS[requested]} isn't available for this model - "
                            f"using {EFFORT_LABELS[effort]}", level="warn", detail=fallback)
            if escalated:
                state.event("effort_escalated", f"Stepped up to {EFFORT_LABELS['high']} for a complex task", level="info")

            mode = state.prep_mode
            if mode != "off" and not bare and (command == "plan" or (not analysis["trivial"] and not answering)):
                state.phase = "preparing"
                self.run_control.checkpoint()
                state.enhanced = self.preparer.prepare(state, mode)
                if state.enhanced:
                    state.constraints = list(state.enhanced.get("constraints") or [])
                    state.success_criteria = list(state.enhanced.get("success_criteria") or [])
                    state.event("prompt_enhanced", "Understood the request" + (
                        f" · {len(state.constraints)} constraint{'s' if len(state.constraints) != 1 else ''}" if state.constraints else ""),
                        detail=state.enhanced.get("intent", ""))
                if mode == "expand":
                    answer = self._prep_card(state, profile, "spec", prompt_prep.spec_text(state.enhanced))
                    if answer["action"] == "cancel":
                        return self._prep_cancelled(state)
                    if answer["action"] == "send" and answer["text"].strip():
                        state.spec = answer["text"].strip()
                    else:
                        state.enhanced = None                 # "Run as typed"
                elif mode == "plan":
                    state.phase = "planning"
                    state.event("planning_started", "Planning the steps")
                    # /plan: the model writes the whole plan (rules only if no model answers).
                    pm = "detailed" if command == "plan" else planner_mode(profile, analysis)
                    state.plan = self.planner.plan(state, "heuristic" if pm == "off" else pm) or {"steps": []}
                    answer = self._prep_card(state, profile, "plan", "\n".join(state.plan.get("steps") or []),
                                             risks=list(state.plan.get("risks") or []), command=command)
                    if answer["action"] == "cancel":
                        return self._prep_cancelled(state, "Plan rejected - nothing was run." if command == "plan" else None)
                    steps = prompt_prep.steps_from_text(answer["text"]) if answer["action"] == "send" else []
                    if steps:
                        state.plan = {**state.plan, "steps": steps}
                        state.plan_accepted = True
                        state.remaining_steps = list(steps)
                        state.event("plan_ready", f"Plan accepted · {len(steps)} step{'s' if len(steps) != 1 else ''}",
                                    steps=steps[:8], risks=state.plan.get("risks", []), accepted=True)
                    else:
                        state.plan = None                     # "Run as typed" (or every step removed)

            pmode = ("off" if bare or (state.prep_mode == "plan" and not answering)
                     else planner_mode(profile, analysis))
            if pmode != "off":
                state.phase = "planning"
                self.run_control.checkpoint()
                state.event("planning_started", "Planning the steps")
                state.plan = self.planner.plan(state, pmode)
                if state.plan:
                    state.requirements = list(state.plan.get("requirements") or [])
                    state.success_criteria = state.success_criteria or list(state.plan.get("success_criteria") or [])
                    state.remaining_steps = list(state.plan.get("steps") or [])
                    n = len(state.remaining_steps)
                    state.event("plan_ready", f"Plan ready · {n} step{'s' if n != 1 else ''}",
                                steps=state.remaining_steps[:8], risks=state.plan.get("risks", []))

            self.run_control.checkpoint()
            self.set_context(state, profile)
            state.phase = "executing"
            result = self.execute(request)
            state.tool_results = self.tool_history()
            if result.get("status") == "stopped":
                return self._cancelled(state, result)

            # Model-made changes only: a deterministic desktop route (internal
            # tool) is checked by the route itself, as before the Phase 2 split.
            mutations = sum(1 for r in state.tool_results if r.get("ok") and r.get("mutating") and not r.get("internal"))
            # A design is judged by looking at it (the chat previews it), not by a text review. A turn where only a
            # deterministic desktop route ran was checked by the route itself: a model review of it only repeated the
            # route's own result as "issues to fix" (D48).
            routed_only = bool(state.tool_results) and all(r.get("internal") for r in state.tool_results)
            # Pictures that are the answer are judged by looking at them too: a text review only reloaded the chat
            # model after the image engine and kept the user waiting for nothing (1.4.7).
            pictured = bool(state.tool_results) and all(r.get("tool") in ("generate_image", "show_web_images") and r.get("ok")
                                                        for r in state.tool_results)
            rmode = "off" if bare or routed_only or pictured else reviewer_mode(profile, analysis, mutations)
            if cross_requested and rmode != "off" and result.get("status") == "success" and not result.get("rejected"):
                status = self.cross_check_status()
                if status.get("available"):
                    rmode = "cross"
                    cross_result.update(status="pending", model=status.get("model"))
                else:
                    cross_result.update(status="unavailable", reason=status.get("reason", "No second model is available."))
                    state.event("cross_check_unavailable", "Cross-check unavailable", level="warn", detail=cross_result["reason"])
            seen = set()
            while rmode != "off" and result.get("status") == "success" and not result.get("rejected"):
                state.phase = "reviewing"
                self.run_control.checkpoint()
                state.event("reviewer_started", "Checking the result against your request")
                review = self.reviewer.review(state, result, self._evidence(board), rmode)
                if rmode == "cross":
                    cross_result.update(review.get("cross_check") or {"status": "failed"})
                    checked = cross_result.get("status") == "checked"
                    state.event("cross_check_complete" if checked else "cross_check_failed",
                                "Second-model review completed" if checked else "Second-model review did not return a valid result",
                                level="info" if checked else "warn", cross_check=dict(cross_result))
                state.verification = {"status": review["verdict"], "findings": review["findings"], "revisions": state.revision_count}
                level = "info" if review["verdict"] == "complete" else "warn"
                state.event("reviewer_findings", review["summary"], level=level, findings=review["findings"][:8],
                            verdict=review["verdict"])
                if review["verdict"] != "revision_required":
                    break
                sig = findings_signature(review["findings"])
                if state.revision_count >= max_revisions or sig in seen:
                    why = "revision limit reached" if state.revision_count >= max_revisions else "the same issues came back"
                    state.event("revision_limit", f"Stopped revising ({why})", level="warn")
                    result = self._attach_remaining(result, review)
                    break
                seen.add(sig)
                self.run_control.checkpoint()
                state.revision_count += 1
                state.phase = "revising"
                state.event("revision_started", f"Revising · attempt {state.revision_count} of {max_revisions}")
                revised = self.revise(state, feedback_text(review))
                if revised is None:
                    state.event("revision_limit", "This result can't be revised automatically", level="warn")
                    result = self._attach_remaining(result, review)
                    break
                result = revised
                state.tool_results = self.tool_history()
                if result.get("status") == "stopped":
                    return self._cancelled(state, result)

            state.phase = "done"
            verdict = state.verification.get("status")
            if result.get("status") == "success" and verdict == "complete":
                state.event("verification_complete", "Verification passed")
            result = dict(result)
            result["verification"] = {"status": verdict, "revisions": state.revision_count, "effort": effort,
                                      "findings": state.verification.get("findings", [])[:8], "cross_check": cross_result}
            result["task_id"] = state.task_id
            if cross_result.get("status") in ("unavailable", "failed"):
                note = cross_result.get("reason") or "The second model did not return a valid review."
                result["message"] = (result.get("message") or "") + "\n\n> **Note:** Cross-check was not completed. " + note
            return result
        except RunStopped as e:
            return self._cancelled(state, self.stopped_response(e.reason))
        except Exception as e:
            state.phase = "failed"
            state.event("run_failed", "The run failed", level="error", detail=str(e)[:200])
            return {"status": "error", "message": f"Turn failed: {str(e)[:300]}", "task_id": state.task_id}
        finally:
            self.set_context(None, None)
            self.run_control.end()
            self.current = None

    def _prep_card(self, state: TaskState, profile, kind: str, text: str, **extra) -> dict:
        """Show the Expand spec / the plan and wait for the user. Nothing has
        run yet, so no tool can run before this returns. The wait does not
        use up the run's time limit. Without a gate (tests, embedding), the
        card is only reported and the text is sent unchanged."""
        what = {"spec": "the request", "plan": "the plan"}[kind]
        if self.prep_gate is None:
            state.event("prep_shown", f"Prepared {what}", card=kind, text=text)
            return {"action": "send", "text": text}
        state.phase = "waiting"
        state.event("prep_waiting", f"Waiting for you to check {what}", card=kind, text=text)
        self.run_control.set_deadline(24 * 3600)     # reading the card is not run time
        answer = self.prep_gate.request(kind, {"task_id": state.task_id, "text": text,
                                               "request": state.original_request, **extra},
                                        cancelled=self.run_control.is_stopped)
        self.run_control.checkpoint()               # Stop while the card was open ends the run here
        answer = answer or {"action": "cancel", "text": ""}
        self.run_control.set_deadline(profile.time_limit_s)
        edited = answer["action"] == "send" and answer["text"].strip() != text.strip()
        state.event("prep_answered", {"send": f"Sent {what}" + (" · edited" if edited else ""),
                                      "raw": "Running as typed", "cancel": "Cancelled before anything ran"}[answer["action"]],
                    card=kind, action=answer["action"], edited=edited)
        return answer

    def _prep_cancelled(self, state: TaskState, message: str | None = None) -> dict:
        state.phase = "done"
        return {"status": "success", "message": message or "Cancelled - nothing was run.", "prep_cancelled": True,
                "task_id": state.task_id, "verification": {"status": "not_run", "revisions": 0,
                                                           "effort": state.effort, "findings": []}}

    def _cancelled(self, state: TaskState, result: dict) -> dict:
        state.cancelled = True
        state.phase = "cancelled"
        reason = "Time limit reached" if "time limit" in (result.get("message") or "").lower() else "Stopped by you"
        state.event("run_cancelled", reason, level="warn")
        return {**result, "task_id": state.task_id}

    @staticmethod
    def _attach_remaining(result: dict, review: dict) -> dict:
        open_items = [f for f in review["findings"] if f["severity"] in ("medium", "high")]
        if not open_items:
            return result
        note = "\n\n> **Note:** not everything is finished:\n" + "\n".join(
            f"> - {f['explanation']}" for f in open_items[:5])
        return {**result, "message": (result.get("message") or "") + note}
