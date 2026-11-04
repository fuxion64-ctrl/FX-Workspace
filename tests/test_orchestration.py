"""Deterministic tests for the production pass: typed tools + executor,
messaging providers and permissions, non-disruptive inspection, the
orchestration pipeline (effort, prompt preparation, planner, reviewer,
revision, cancellation), model routing/handoff, and the theme system.

Everything external is faked: no real windows are touched, no messages are
sent, no model is called."""
import json
import re
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import messaging
import testing_fakes
import model_router
import orchestrator
import permissions
import window_state
from run_control import RunControl, RunStopped
from tool_executor import ToolExecutor, classify
from tool_registry import ToolRegistry, ToolSpec, validate_args
import agent.loop as agent_loop
import agent.models as llm
import agent.tasks as tasks
import core.config as config
import core.state as shared
import events as event_log
import routing.handoff as handoff
import tools.approvals as approvals
import tools.runtime as runtime
import tools.schemas as schemas
from run_control import run_control as run_control
import llama_local
import system_metrics


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _registry(*specs):
    reg = ToolRegistry()
    for s in specs:
        reg.register(s)
    return reg


def _spec(name, **kw):
    schema = kw.pop("schema", {"type": "object", "properties": {"x": {"type": "integer"}}, "required": []})
    return ToolSpec(name=name, description=name, input_schema=schema, **kw)


# ==============================================================================
# Tools
# ==============================================================================
class ToolRegistryTests(unittest.TestCase):
    def test_app_registry_has_typed_metadata_for_every_tool(self):
        import main
        names = {t["function"]["name"] for t in schemas.TOOLS}
        self.assertEqual(runtime.TOOL_REGISTRY.names(), names | schemas.INTERNAL_TOOL_NAMES)
        # Internal tools are registered (one executor/event path) but never offered to a model.
        self.assertFalse(schemas.INTERNAL_TOOL_NAMES & names)
        saved = shared.app_state.get("turn_policy")
        try:
            shared.app_state["turn_policy"] = {"mode": "action"}
            for local in (True, False):
                offered = {t["function"]["name"] for t in schemas.tools_for_current_turn(local=local)}
                self.assertFalse(schemas.INTERNAL_TOOL_NAMES & offered)
        finally:
            shared.app_state["turn_policy"] = saved
        read = runtime.TOOL_REGISTRY.get("read_file")
        self.assertTrue(read.read_only and read.retryable and read.idempotent and not read.mutating)
        write = runtime.TOOL_REGISTRY.get("write_file")
        self.assertTrue(write.mutating and not write.retryable and write.approval_by_default)
        self.assertEqual(runtime.TOOL_REGISTRY.get("messages_send").tier, "destructive")
        self.assertEqual(runtime.TOOL_REGISTRY.get("request_model_handoff").tier, "input")
        self.assertIn("required", runtime.TOOL_REGISTRY.get("inspect_screen").output_schema)

    def test_validation_and_coercion(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}, "b": {"type": "boolean"},
                                                   "tags": {"type": "array"}}, "required": ["n"]}
        args, errors = validate_args(schema, {"n": "12", "b": "true", "tags": "one"})
        self.assertEqual((args["n"], args["b"], args["tags"], errors), (12, True, ["one"], []))
        _, errors = validate_args(schema, {"b": True})
        self.assertIn("missing required argument 'n'", errors)
        _, errors = validate_args(schema, {"n": "twelve"})
        self.assertTrue(errors)

    def test_classification(self):
        self.assertEqual(classify({"ok": True}), "success")
        self.assertEqual(classify({"ok": False, "message": "connection reset by peer"}), "retryable_failure")
        self.assertEqual(classify({"ok": False, "message": "No such file"}), "permanent_failure")
        self.assertEqual(classify({"ok": False, "message": "pywinauto is not installed"}), "unavailable")
        self.assertEqual(classify({"ok": False, "message": "The user REJECTED this action."}), "approval_required")
        self.assertEqual(classify({"ok": False, "message": "CURRENT-TURN GATE: blocked"}), "permanent_failure")


class ToolExecutorTests(unittest.TestCase):
    def make(self, run, *specs, **kw):
        clock = kw.pop("clock", FakeClock())
        return ToolExecutor(_registry(*specs), run, clock=clock, sleep=kw.pop("sleep", clock.sleep), **kw), clock

    def test_simple_call(self):
        ex, _ = self.make(lambda n, a: {"ok": True, "message": "done"}, _spec("t", read_only=True))
        res = ex.execute("t", {"x": 1}, tool_call_id="call_1")
        self.assertEqual((res["status"], res["ok"], res["attempts"], res["tool_call_id"]), ("success", True, 1, "call_1"))

    def test_multi_tool_task_history(self):
        ex, _ = self.make(lambda n, a: {"ok": True, "message": n}, _spec("a", read_only=True), _spec("b"))
        ex.begin_task("task1", tool_budget=10)
        ex.execute("a", {})
        ex.execute("b", {})
        self.assertEqual([h["tool"] for h in ex.history()], ["a", "b"])
        self.assertEqual([h["mutating"] for h in ex.history()], [False, True])

    def test_invalid_arguments_never_run(self):
        run = Mock()
        spec = _spec("t", schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
        ex, _ = self.make(run, spec)
        res = ex.execute("t", {})
        self.assertEqual(res["status"], "permanent_failure")
        run.assert_not_called()

    def test_retryable_failure_is_retried_with_backoff(self):
        calls = []

        def run(n, a):
            calls.append(1)
            return {"ok": False, "message": "connection reset"} if len(calls) < 2 else {"ok": True, "message": "ok"}
        clock = FakeClock()
        sleeps = []
        ex, _ = self.make(run, _spec("t", read_only=True, retryable=True), clock=clock, sleep=lambda s: sleeps.append(s))
        res = ex.execute("t", {})
        self.assertEqual((res["status"], res["attempts"]), ("success", 2))
        self.assertEqual(sleeps, [0.5])

    def test_retries_are_bounded(self):
        run = Mock(return_value={"ok": False, "message": "service busy, try again"})
        ex, _ = self.make(run, _spec("t", read_only=True, retryable=True), max_retries=2)
        res = ex.execute("t", {})
        self.assertEqual((res["status"], res["attempts"], run.call_count), ("retryable_failure", 3, 3))

    def test_permanent_failure_and_non_retryable_tools_run_once(self):
        run = Mock(return_value={"ok": False, "message": "Access is denied"})
        ex, _ = self.make(run, _spec("t", read_only=True, retryable=True))
        self.assertEqual(ex.execute("t", {"x": 1})["attempts"], 1)
        run2 = Mock(return_value={"ok": False, "message": "timed out"})
        ex2, _ = self.make(run2, _spec("w", retryable=False))
        self.assertEqual((ex2.execute("w", {})["attempts"], run2.call_count), (1, 1))

    def test_identical_failed_call_is_not_repeated_until_state_changes(self):
        outcomes = iter([{"ok": False, "message": "No window matching 'X'"}, {"ok": True, "message": "opened"},
                         {"ok": True, "message": "found it"}])
        run = Mock(side_effect=lambda n, a: next(outcomes))
        ex, _ = self.make(run, _spec("list_elements", read_only=True), _spec("open_app"))
        ex.begin_task("t")
        ex.execute("list_elements", {"x": 1})
        again = ex.execute("list_elements", {"x": 1})
        self.assertIn("Not repeated", again["message"])
        self.assertEqual(run.call_count, 1)
        ex.execute("open_app", {})                          # a mutation succeeded: the world changed
        self.assertEqual(ex.execute("list_elements", {"x": 1})["status"], "success")

    def test_timeout_enforced_for_isolated_tools(self):
        def slow(n, a):
            time.sleep(1.0)
            return {"ok": True, "message": "late"}
        ex = ToolExecutor(_registry(_spec("r", read_only=True, isolate=True, timeout_s=0.15),
                                    _spec("w", isolate=True, timeout_s=0.15)), slow, max_retries=0)
        started = time.monotonic()
        read = ex.execute("r", {})
        write = ex.execute("w", {})
        self.assertLess(time.monotonic() - started, 0.9)
        self.assertEqual(read["status"], "permanent_failure")
        self.assertEqual(write["status"], "permanent_failure")
        self.assertIn("outcome is unknown", write["message"])

    def test_timeout_does_not_retry_a_running_read_worker(self):
        release = threading.Event()
        entered = threading.Event()
        def slow(n, a):
            entered.set()
            release.wait(3)
            return {"ok": True}
        run = Mock(side_effect=slow)
        ex = ToolExecutor(_registry(_spec("r", read_only=True, retryable=True,
                                         isolate=True, timeout_s=0.05)), run, max_retries=3)
        try:
            result = ex.execute("r", {})
            self.assertTrue(entered.is_set())
            self.assertEqual(result["status"], "permanent_failure")
            self.assertEqual(run.call_count, 1)
        finally:
            release.set()

    def test_cancellation_during_tool_execution(self):
        flag = threading.Event()

        def slow(n, a):
            time.sleep(2)
            return {"ok": True, "message": "late"}
        ex = ToolExecutor(_registry(_spec("r", read_only=True, isolate=True, timeout_s=10)), slow,
                          is_cancelled=flag.is_set)
        threading.Timer(0.1, flag.set).start()
        started = time.monotonic()
        res = ex.execute("r", {})
        self.assertEqual(res["status"], "cancelled")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_cooldown(self):
        clock = FakeClock()
        waits = []
        ex = ToolExecutor(_registry(_spec("fast", read_only=True, cooldown_s=1.0), _spec("slow", cooldown_s=30.0)),
                          lambda n, a: {"ok": True, "message": "ok"}, clock=clock, sleep=lambda s: waits.append(s))
        ex.execute("fast", {"x": 1})
        ex.execute("fast", {"x": 2})
        self.assertAlmostEqual(waits[0], 1.0)             # short cooldown is waited out
        ex.execute("slow", {})
        clock.t += 5
        res = ex.execute("slow", {"x": 3})
        self.assertEqual(res["status"], "retryable_failure")
        self.assertIn("cooling down", res["message"])

    def test_tool_budget(self):
        ex, _ = self.make(lambda n, a: {"ok": True, "message": "ok"}, _spec("t", read_only=True))
        ex.begin_task("t", tool_budget=2)
        ex.execute("t", {"x": 1})
        ex.execute("t", {"x": 2})
        self.assertIn("budget", ex.execute("t", {"x": 3})["message"])


# ==============================================================================
# Messaging
# ==============================================================================
class MessagingTests(unittest.TestCase):
    def setUp(self):
        self.provider = testing_fakes.MockMessagingProvider()
        self.hub = messaging.MessagingHub()
        self.hub.register(self.provider)

    def test_read_paths(self):
        self.assertIn("Alex", self.hub.run_tool("messages_list_conversations", {})["message"])
        self.assertIn("KillBrick", self.hub.run_tool("messages_search", {"query": "killbrick"})["message"])
        self.assertIn("playtest", self.hub.run_tool("messages_read", {"conversation_id": "c1"})["message"])

    def test_draft_never_sends(self):
        res = self.hub.run_tool("messages_draft", {"conversation_id": "c1", "text": "Yes, 6pm works"})
        self.assertTrue(res["ok"])
        self.assertEqual(self.provider.sent, [])
        self.assertEqual(len(self.provider.conversations["c1"]["messages"]), 1)

    def test_send_only_a_saved_draft(self):
        bad = self.hub.run_tool("messages_send", {"draft_id": "nope"})
        self.assertFalse(bad["ok"])
        draft = self.hub.run_tool("messages_draft", {"conversation_id": "c1", "text": "On my way"})["data"]
        sent = self.hub.run_tool("messages_send", {"draft_id": draft["id"]})
        self.assertTrue(sent["ok"])
        self.assertEqual(self.provider.sent[0]["text"], "On my way")

    def test_unavailable_without_provider(self):
        res = messaging.MessagingHub().run_tool("messages_read", {"conversation_id": "c1"})
        self.assertEqual(res["status"], "unavailable")

    def test_read_draft_send_permissions_are_separate(self):
        base = dict(permissions.DEFAULT_PERMISSION_SETTINGS)
        self.assertEqual(permissions.decide("messages_read", base), "ask")
        self.assertEqual(permissions.decide("messages_draft", base), "auto")
        self.assertEqual(permissions.decide("messages_send", base), "ask")
        custom = {**base, "messaging_permissions": {"read": "auto", "draft": "deny", "send": "deny"}}
        self.assertEqual([permissions.decide(t, custom) for t in ("messages_search", "messages_draft", "messages_send")],
                         ["auto", "deny", "deny"])

    def test_send_is_never_automatic(self):
        god = {**permissions.DEFAULT_PERMISSION_SETTINGS, "permission_mode": "god",
               "permission_overrides": {"messages_send": "auto"}, "messaging_permissions": {"send": "auto"}}
        self.assertEqual(permissions.decide("messages_send", god), "ask")
        self.assertEqual(permissions.clean_settings(god)["messaging_permissions"]["send"], "ask")

    def test_sending_goes_through_the_approval_flow(self):
        """End to end through the real agent loop: the model asks to send a
        draft; the approval is declined, then granted."""
        import main
        hub = messaging.MessagingHub()
        provider = testing_fakes.MockMessagingProvider()
        hub.register(provider)
        draft = provider.create_draft("c1", "See you at 6")

        def scripted(answers):
            calls = iter([
                SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
                    content="", tool_calls=[SimpleNamespace(id="call_s", function=SimpleNamespace(
                        name="messages_send", arguments=json.dumps({"draft_id": draft["id"]})))]))]),
                SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                    content="Done.", tool_calls=None))]),
            ])
            return Mock(side_effect=lambda: next(calls))

        for approved in (False, True):
            shared.app_state["conversation"] = [{"role": "system", "content": "s"}, {"role": "user", "content": "send it"}]
            shared.app_state["turn_policy"] = {"mode": "action", "allow_mutation": True, "reason": "test"}
            shared.app_state["intent_guard"] = None
            shared.app_state["last_tool_signature_set"] = None
            shared.task_board.clear()
            shared.ledger.clear()
            runtime.tool_exec.begin_task(f"t{approved}")
            ask = Mock(return_value=approved)
            with patch.object(shared, "messaging_hub", hub), patch.object(llm, "call_model", scripted(None)), \
                 patch.object(approvals, "ask_batch_approval_os", ask):
                result = agent_loop.run_agent_loop()
            ask.assert_called_once()
            self.assertEqual(ask.call_args[0][0][0]["tool_name"], "messages_send")
            self.assertEqual(len(provider.sent), 1 if approved else 0)
            if not approved:
                self.assertTrue(result.get("rejected"))


# ==============================================================================
# Inspection
# ==============================================================================
class FakeWindows:
    """A tiny window manager with the OLD bug built in: SW_RESTORE on a
    maximized window un-maximizes it."""
    NORMAL, MINIMIZED, MAXIMIZED = 1, 2, 3

    def __init__(self):
        self.wins = {1: {"show": self.MAXIMIZED, "rect": (0, 0, 800, 600)},     # the user's fullscreen app
                     2: {"show": self.MAXIMIZED, "rect": (10, 10, 900, 700)},   # the app being inspected
                     3: {"show": self.MINIMIZED, "rect": (50, 50, 400, 300)}}
        self.foreground = 1
        self.calls = []

    def is_minimized(self, h):
        return self.wins[h]["show"] == self.MINIMIZED

    def get_placement(self, h):
        return dict(self.wins[h]) if h in self.wins else None

    def set_placement(self, h, p):
        self.calls.append(("set_placement", h))
        self.wins[h].update(show=p["show"], rect=p["rect"])
        return True

    def get_foreground(self):
        return self.foreground

    def show(self, h, cmd):
        self.calls.append(("show", h, cmd))
        if cmd == window_state.SW_RESTORE:
            self.wins[h]["show"] = self.NORMAL
        elif cmd == window_state.SW_SHOWNOACTIVATE:
            self.wins[h]["show"] = self.NORMAL
        elif cmd == window_state.SW_SHOWMINNOACTIVE:
            self.wins[h]["show"] = self.MINIMIZED
        return True

    def focus(self, h):
        self.calls.append(("focus", h))
        self.foreground = h
        if self.wins[1]["show"] == self.MAXIMIZED and h != 1:
            self.wins[1]["show"] = self.MINIMIZED      # a fullscreen app drops out when covered
        return True


class InspectionTests(unittest.TestCase):
    def test_inspecting_a_maximized_app_changes_nothing(self):
        import vision
        fw = FakeWindows()
        shot, meta, info = vision.inspect("Studio", be=fw, find=lambda t: (2, "Studio"),
                                          capture=lambda w, h: ("IMG", {"vx": 0, "vy": 0, "orig_w": 1, "orig_h": 1}))
        self.assertTrue(info["ok"])
        self.assertEqual(fw.wins[2]["show"], FakeWindows.MAXIMIZED)          # never un-maximized
        self.assertNotIn(("show", 2, window_state.SW_RESTORE), fw.calls)
        self.assertEqual(fw.foreground, 1)                                   # user's app is in front again
        self.assertEqual(fw.wins[1]["show"], FakeWindows.MAXIMIZED)          # and back to fullscreen
        self.assertEqual(meta["target_hwnd"], 2)                             # clicks re-front the target

    def test_failed_inspection_still_restores_everything(self):
        import vision
        fw = FakeWindows()

        def boom(w, h):
            raise RuntimeError("capture failed")
        with self.assertRaises(RuntimeError):
            vision.inspect("Studio", be=fw, find=lambda t: (2, "Studio"), capture=boom)
        self.assertEqual((fw.foreground, fw.wins[1]["show"], fw.wins[2]["show"]),
                         (1, FakeWindows.MAXIMIZED, FakeWindows.MAXIMIZED))
        fw2 = FakeWindows()
        _, _, info = vision.inspect("Studio", be=fw2, find=lambda t: (2, "Studio"), capture=lambda w, h: (None, None))
        self.assertFalse(info["ok"])
        self.assertEqual(fw2.foreground, 1)

    def test_minimized_window_is_shown_without_focus_then_reminimized(self):
        fw = FakeWindows()
        with window_state.temporarily_visible(3, be=fw) as was_min:
            self.assertTrue(was_min)
            self.assertEqual(fw.wins[3]["show"], FakeWindows.NORMAL)
        self.assertEqual(fw.wins[3]["show"], FakeWindows.MINIMIZED)
        self.assertEqual(fw.foreground, 1)
        with window_state.temporarily_visible(2, be=fw):
            pass
        self.assertEqual([c for c in fw.calls if c[1] == 2], [])             # visible window untouched

    @unittest.skipUnless(__import__("desktop_input").IS_WINDOWS, "Win32 focus path")
    def test_focus_window_no_longer_restores_a_maximized_window(self):
        import desktop_input as di
        user32 = Mock()
        user32.IsIconic.return_value = 0
        with patch.object(di, "user32", user32), patch.object(di, "kernel32", Mock()):
            di.focus_window(1234)
        self.assertNotIn(9, [c.args[1] for c in user32.ShowWindow.call_args_list])
        user32.IsIconic.return_value = 1
        with patch.object(di, "user32", user32), patch.object(di, "kernel32", Mock()):
            di.focus_window(1234)
        self.assertIn(9, [c.args[1] for c in user32.ShowWindow.call_args_list])   # minimized: restore is right


# ==============================================================================
# Orchestration
# ==============================================================================
class FakeModels:
    """model_json stand-in: canned structured answers, counts calls per purpose."""

    def __init__(self, review_verdicts=None, on_review=None):
        self.calls = []
        self.review_verdicts = list(review_verdicts or [])
        self.on_review = on_review

    def __call__(self, system, user, max_tokens=600):
        purpose = "review" if "verify" in system else "plan" if "plan tasks" in system else "prepare"
        self.calls.append(purpose)
        if purpose == "prepare":
            return {"intent": "open the app", "constraints": ["only Notepad"], "ambiguities": ["which file?"],
                    "success_criteria": ["Notepad is open"]}
        if purpose == "plan":
            return {"interpretation": "open", "steps": ["Open Notepad", "Type text"], "requirements": ["Notepad open"],
                    "risks": [], "success_criteria": ["Notepad shows the text"]}
        if self.on_review:
            self.on_review()
        verdict = self.review_verdicts.pop(0) if self.review_verdicts else "complete"
        findings = [] if verdict == "complete" else [
            {"type": "missing_requirement", "severity": "high", "requirement": "Type text", "explanation": "Text was not typed.",
             "action": "Type it"}]
        return {"verdict": verdict, "findings": findings}


class OrchestratorHarness:
    def __init__(self, settings=None, execute=None, revise=None, models=None, secondary=None, board=None):
        self.rc = RunControl()
        self.events = []
        self.settings = {"effort": "normal", "prompt_enhancement": "minimal", "effort_auto_escalate": True, **(settings or {})}
        self.models = models or FakeModels()
        self.secondary = secondary
        self.board = board if board is not None else []
        self.history = []
        self.executed = []
        self.revisions = []
        self.contexts = []
        self._execute = execute or (lambda req: {"status": "success", "message": "Opened Notepad and typed the text."})
        self._revise = revise
        self.agent = orchestrator.AgentOrchestrator(
            execute=self.execute, revise=self.revise, settings=lambda: self.settings, run_control=self.rc,
            emit=lambda kind, summary, **d: self.events.append((kind, summary, d)),
            analyze_request=lambda text, v: orchestrator.analyze(
                text, "none" if re.match(r"^(hi|thanks|what is)", text, re.I) else "action",
                3 if " then " in text else 1),
            preparer=orchestrator.PromptPreparer(self.models), planner=orchestrator.Planner(self.models),
            reviewer=orchestrator.Reviewer(self.models, self.secondary),
            tool_scope=lambda tid, budget: None, tool_history=lambda: list(self.history),
            set_context=lambda state, profile: self.contexts.append(state.brief() if state else None),
            stopped_response=lambda reason: {"status": "stopped", "message": "Stopped."})

    def execute(self, req):
        self.executed.append(req.message)
        return self._execute(req)

    def revise(self, state, feedback):
        self.revisions.append((state.original_request, feedback))
        return self._revise(state, feedback) if self._revise else None

    def run(self, text):
        return self.agent.run(SimpleNamespace(message=text, with_vision=False), board=lambda: list(self.board))

    def kinds(self):
        return [e[0] for e in self.events]


class OrchestratorTests(unittest.TestCase):
    def test_low_effort_is_direct(self):
        h = OrchestratorHarness({"effort": "low"})
        res = h.run("open notepad")
        self.assertEqual(res["status"], "success")
        self.assertNotIn("planning_started", h.kinds())
        self.assertNotIn("plan", h.models.calls)
        self.assertNotIn("review", h.models.calls)
        self.assertEqual(res["verification"]["effort"], "low")

    def test_normal_effort_plans_with_rules_and_verifies(self):
        h = OrchestratorHarness()
        res = h.run("open notepad")
        self.assertIn("plan_ready", h.kinds())
        self.assertNotIn("plan", h.models.calls)                 # heuristic plan: no model call
        self.assertIn("verification_complete", h.kinds())
        self.assertEqual(res["verification"]["status"], "complete")

    def test_normal_escalates_complex_tasks_to_high(self):
        h = OrchestratorHarness()
        res = h.run("open notepad then type hello then save it")
        self.assertIn("effort_escalated", h.kinds())
        self.assertEqual(res["verification"]["effort"], "high")
        self.assertIn("plan", h.models.calls)
        self.assertIn("review", h.models.calls)

    def test_fixed_voice_effort_does_not_escalate_but_auto_still_selects(self):
        for requested, expected in (("normal", "normal"), ("auto", "extra")):
            with self.subTest(effort=requested):
                h = OrchestratorHarness({"effort": requested})
                req = SimpleNamespace(message="open notepad then type hello then save it", with_vision=False, voice=True)
                res = h.agent.run(req)
                self.assertEqual(res["verification"]["effort"], expected)
                self.assertNotIn("effort_escalated", h.kinds())

    def test_high_effort_uses_model_review(self):
        h = OrchestratorHarness({"effort": "high"})
        h.run("open notepad")
        self.assertEqual(h.models.calls.count("review"), 1)

    def test_cross_check_toggle_uses_a_second_model_independently_of_effort(self):
        second = FakeModels()
        h = OrchestratorHarness({"effort": "low", "cross_check": True}, secondary=second)
        res = h.run("open notepad")
        self.assertIn("review", second.calls)
        self.assertEqual(res["verification"]["effort"], "low")
        self.assertEqual(res["verification"]["cross_check"]["status"], "checked")

    def test_max_has_normal_model_review_without_cross_check(self):
        second = FakeModels()
        h = OrchestratorHarness({"effort": "max", "cross_check": False}, secondary=second)
        res = h.run("open notepad")
        self.assertEqual(h.models.calls.count("review"), 1)
        self.assertEqual(second.calls, [])
        self.assertEqual(res["verification"]["cross_check"]["status"], "off")

    def test_unavailable_cross_check_is_reported_without_claiming_it_ran(self):
        h = OrchestratorHarness({"effort": "max", "cross_check": True})
        res = h.run("open notepad")
        self.assertIn("cross_check_unavailable", h.kinds())
        self.assertEqual(res["verification"]["cross_check"]["status"], "unavailable")
        self.assertIn('Cross-check was not completed', res['message'])

    def test_failed_second_review_is_distinct_from_a_completed_cross_check(self):
        h = OrchestratorHarness({"effort": "max", "cross_check": True}, secondary=lambda *a: None)
        res = h.run("open notepad")
        self.assertIn("cross_check_failed", h.kinds())
        self.assertNotIn("cross_check_complete", h.kinds())
        self.assertEqual(res["verification"]["cross_check"]["status"], "failed")
        self.assertIn('Cross-check was not completed', res['message'])

    def test_malformed_second_review_cannot_claim_a_completed_cross_check(self):
        h = OrchestratorHarness({"effort": "max", "cross_check": True}, secondary=lambda *a: {'verdict':'complete'})
        res = h.run("open notepad")
        self.assertEqual(res['verification']['cross_check']['status'], 'failed')
        self.assertNotIn('cross_check_complete', h.kinds())

    def test_blocked_second_review_is_not_a_passed_verification(self):
        h = OrchestratorHarness({"effort": "max", "cross_check": True}, secondary=lambda *a: {'verdict':'blocked','findings':[]})
        res = h.run("open notepad")
        self.assertEqual(res['verification']['status'], 'blocked')
        self.assertNotIn('verification_complete', h.kinds())

    def test_cross_check_change_during_execution_applies_to_the_next_request(self):
        second = FakeModels()
        h = OrchestratorHarness({"effort": "max", "cross_check": False}, secondary=second)
        def execute(req):
            h.settings['cross_check'] = True
            return {"status": "success", "message": "Result is ready."}
        h._execute = execute
        first = h.run("open notepad")
        self.assertEqual(second.calls, [])
        self.assertEqual(first["verification"]["cross_check"]["status"], "off")
        second_run = h.run("open notepad")
        self.assertIn("review", second.calls)
        self.assertEqual(second_run["verification"]["cross_check"]["status"], "checked")

    def test_trivial_request_skips_the_pipeline(self):
        h = OrchestratorHarness({"effort": "max", "prompt_enhancement": "detailed"})
        res = h.run("hi")
        self.assertEqual(h.models.calls, [])
        self.assertEqual(h.kinds(), ["run_started"])            # no prep, plan or review for a greeting
        self.assertEqual(res["verification"]["status"], "not_run")

    def test_prompt_prep_modes(self):
        off = OrchestratorHarness({"prompt_prep": "off"})
        off.run("open notepad, only notepad")
        self.assertNotIn("prompt_enhanced", off.kinds())
        self.assertNotIn("prepare", off.models.calls)
        clarify = OrchestratorHarness({"prompt_prep": "clarify"})
        clarify.run("open notepad, only notepad")
        self.assertIn("prompt_enhanced", clarify.kinds())
        self.assertNotIn("prepare", clarify.models.calls)          # rules, no model call
        self.assertIn("Only one question is allowed", clarify.contexts[0])
        expand = OrchestratorHarness({"prompt_prep": "expand"})     # no gate here: shown, sent unchanged
        expand.run("open notepad, only notepad")
        self.assertIn("prepare", expand.models.calls)
        self.assertIn("prep_shown", expand.kinds())
        self.assertIn("PREPARED SPEC", expand.contexts[0])

    def test_old_prompt_preparation_settings_migrate_without_blocking(self):
        # None of the old levels ever stopped a run, so none maps to Expand / Plan first.
        for old in ("minimal", "standard", "detailed"):
            h = OrchestratorHarness({"prompt_enhancement": old})
            h.run("open notepad, only notepad")
            self.assertNotIn("prep_shown", h.kinds(), old)
            self.assertNotIn("prepare", h.models.calls, old)
        h = OrchestratorHarness({"prompt_enhancement": "off"})
        h.run("open notepad, only notepad")
        self.assertNotIn("prompt_enhanced", h.kinds())

    def test_original_request_preserved_through_every_stage(self):
        original = "Open Notepad, type 'hello' - do not save anything"
        seen = []

        def revise(state, feedback):
            seen.append(state.original_request)
            h.board[:] = [{"n": 1, "text": "Type text", "status": "done"}]
            return {"status": "success", "message": "Typed hello."}
        h = OrchestratorHarness({"effort": "high", "prompt_enhancement": "detailed"}, revise=revise,
                                models=FakeModels(review_verdicts=["revision_required", "complete"]))
        h.board = [{"n": 1, "text": "Type text", "status": "pending"}]
        h.run(original)
        self.assertEqual(h.executed, [original])                 # the executor got the exact words
        self.assertEqual(seen, [original])                       # so did the revision
        self.assertIn(f"ORIGINAL REQUEST (verbatim, authoritative): {original}", h.contexts[0])
        state = orchestrator.TaskState(original)
        with self.assertRaises(AttributeError):
            state.original_request = "something else"

    def test_reviewer_catches_incomplete_result_and_revision_fixes_it(self):
        def revise(state, feedback):
            self.assertIn("Step 2 is pending", feedback)
            h.board[1]["status"] = "done"
            return {"status": "success", "message": "Typed the text too."}
        h = OrchestratorHarness(revise=revise)
        h.board = [{"n": 1, "text": "Open Notepad", "status": "done"}, {"n": 2, "text": "Type text", "status": "pending"}]
        res = h.run("open notepad and type hello")
        self.assertEqual(len(h.revisions), 1)
        self.assertEqual(res["verification"]["status"], "complete")
        self.assertIn("revision_started", h.kinds())
        findings = [e for e in h.events if e[0] == "reviewer_findings"][0][2]["findings"]
        self.assertEqual(findings[0]["type"], "missing_requirement")

    def test_revisions_are_bounded_and_identical_issues_stop_early(self):
        h = OrchestratorHarness({"effort": "high"}, revise=lambda s, f: {"status": "success", "message": "Tried again."})
        h.board = [{"n": 1, "text": "Type text", "status": "failed", "note": "window gone"}]
        res = h.run("type hello in notepad")
        self.assertEqual(len(h.revisions), 1)                    # same findings came back -> stop
        self.assertIn("revision_limit", h.kinds())
        self.assertIn("not everything is finished", res["message"])
        h0 = OrchestratorHarness({"max_revisions": 0}, revise=lambda s, f: {"status": "success", "message": "x"})
        h0.board = [{"n": 1, "text": "Type text", "status": "pending"}]
        h0.run("type hello")
        self.assertEqual(h0.revisions, [])

    def test_cancellation_during_execution(self):
        h = OrchestratorHarness(execute=lambda req: {"status": "stopped", "message": "Stopped."})
        res = h.run("open notepad")
        self.assertEqual(res["status"], "stopped")
        self.assertIn("run_cancelled", h.kinds())
        self.assertNotIn("reviewer_started", h.kinds())
        self.assertEqual(h.rc.state(), "idle")

    def test_cancellation_during_review(self):
        box = {}
        models = FakeModels(on_review=lambda: box["rc"].stop())
        h = OrchestratorHarness({"effort": "high"}, models=models,
                                revise=lambda s, f: self.fail("a cancelled run must not revise"))
        box["rc"] = h.rc
        h.board = [{"n": 1, "text": "Type text", "status": "pending"}]
        res = h.run("open notepad")
        self.assertEqual(res["status"], "stopped")
        self.assertIn("run_cancelled", h.kinds())
        self.assertEqual(h.rc.state(), "idle")

    def test_full_pipeline_through_the_real_engine(self):
        """tasks.agent (orchestrator) -> chat_with_agent -> run_agent_loop -> tool
        executor -> reviewer -> revision -> re-review, with only the model
        scripted. The model 'forgets' step 2; the reviewer catches it from the
        task board and the revision finishes it."""
        import main

        def call(name, args, cid):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
                content="", tool_calls=[SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))]))])

        def say(text):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text, tool_calls=None))])
        script = iter([
            call("set_plan", {"goal": "Organize notes", "steps": ["Collect the notes", "Write the plan"]}, "c1"),
            call("update_plan", {"step": 1, "status": "done", "note": "collected"}, "c2"),
            say("Done - your notes are organized."),                 # step 2 was never done
            call("update_plan", {"step": 2, "status": "done", "note": "plan written"}, "c3"),
            say("Both steps are finished."),
        ])
        # Cloud mode (still scripted - nothing leaves the machine): local mode's
        # own "unfinished plan" nudge would finish step 2 inside the executor
        # and the reviewer-driven revision would never be needed.
        saved = {k: shared.app_state.get(k) for k in ("conversation", "settings", "local_model", "local_engaged", "mode")}
        events = []
        try:
            shared.app_state["settings"] = {**shared.app_state["settings"], "effort": "normal", "prompt_enhancement": "minimal",
                                            "prompt_prep": "clarify", "prompt_prep_by_model": {}, "effort_by_model": {}}
            shared.task_board.clear()
            shared.ledger.clear()
            real_emit = event_log.emit_event
            import memory, tempfile
            with tempfile.TemporaryDirectory() as tmp, \
                 patch.object(shared, "memory_store", memory.MemoryStore(tmp + "\\memory.json")), \
                 patch.object(llm, "call_model", side_effect=lambda: next(script)), \
                 patch.object(llm, "model_json", return_value=None), \
                 patch.object(event_log, "emit_event", side_effect=lambda k, l, **f: (events.append((k, l, f)), real_emit(k, l, **f))[1]):
                result = tasks.agent.run(SimpleNamespace(message="Organize my project notes into a plan", with_vision=False,
                                                        mode="cloud", local_model="", local_text_model=None),
                                        board=tasks._board)
        finally:
            shared.app_state.update(saved)
        phases = [f.get("phase") for k, l, f in events if k == "activity"]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Both steps are finished.")
        self.assertEqual(result["verification"]["status"], "complete")
        self.assertEqual(result["verification"]["revisions"], 1)
        for phase in ("run_started", "prompt_enhanced", "plan_ready", "reviewer_findings", "revision_started", "verification_complete"):
            self.assertIn(phase, phases)
        self.assertEqual(sum(1 for k, *_ in events if k == "tool"), 3)          # set_plan + 2x update_plan
        self.assertEqual(run_control.state(), "idle")

    def test_run_control_nests_so_inner_end_keeps_a_stop(self):
        rc = RunControl()
        rc.begin()
        rc.begin()
        rc.stop()
        rc.end()                                                 # inner stage finished
        with self.assertRaises(RunStopped):
            rc.checkpoint()                                      # outer run still sees the stop
        rc.end()
        self.assertEqual(rc.state(), "idle")

    def test_time_limit_stops_the_run(self):
        clock = FakeClock()
        rc = RunControl(clock=clock)
        rc.begin(deadline_s=10)
        clock.t += 11
        with self.assertRaises(RunStopped) as cm:
            rc.checkpoint()
        self.assertEqual(cm.exception.reason, "time")
        rc.end()


# ==============================================================================
# Model routing
# ==============================================================================
class FakeTask:
    def __init__(self):
        self.handoff_count = 0
        self.model_history = []
        self.active_model = None

    def record_handoff(self, previous, new, reason):
        if previous:
            self.model_history.append(previous)
        self.handoff_count += 1
        self.active_model = new

    def snapshot(self):
        return {"task_id": "t", "plan": None}


def _models():
    return [model_router.ModelInfo("Text-4B", "Text 4B", "local", vision=False, available=True, memory_gb=3, context_limit=32000),
            model_router.ModelInfo("Vision-11B", "Vision 11B", "local", vision=True, available=True, memory_gb=8, context_limit=32000),
            model_router.ModelInfo("Coder-7B", "Coder 7B", "local", vision=False, available=True, memory_gb=5, context_limit=16000),
            model_router.ModelInfo("Huge-70B", "Huge", "local", vision=True, available=True, memory_gb=48)]


class RoutingTests(unittest.TestCase):
    def make(self, enabled=True, fail_on=(), clock=None):
        self.loaded = []
        self.events = []

        def loader(model_id, provider):
            self.loaded.append(model_id)
            if model_id in fail_on:
                raise RuntimeError("out of memory")
        return model_router.ModelRouter(model_router.ModelRegistry([_models]), loader, enabled=lambda: enabled,
                                        vram_gb=lambda: 12, emit=lambda k, s, **d: self.events.append(k),
                                        clock=clock or FakeClock(), cooldown_s=60)

    def test_disabled_routing_never_switches(self):
        router = self.make(enabled=False)
        res = router.request_handoff(FakeTask(), "Vision-11B", "vision capability required", "Text-4B", 1)
        self.assertEqual(res["status"], "unavailable")
        self.assertEqual(self.loaded, [])

    def test_successful_handoff(self):
        router, task = self.make(), FakeTask()
        res = router.request_handoff(task, "Vision-11B", "vision capability required", "Text-4B", 1)
        self.assertTrue(res["ok"])
        self.assertEqual((self.loaded, task.handoff_count, task.model_history), (["Vision-11B"], 1, ["Text-4B"]))
        self.assertEqual(self.events, ["handoff_requested", "handoff_completed"])

    def test_failed_handoff_falls_back_to_previous_model(self):
        router, task = self.make(fail_on={"Vision-11B"}), FakeTask()
        res = router.request_handoff(task, "Vision-11B", "vision capability required", "Text-4B", 1)
        self.assertFalse(res["ok"])
        self.assertEqual(self.loaded, ["Vision-11B", "Text-4B"])  # tried, then restored
        self.assertEqual(task.handoff_count, 0)
        self.assertIn("handoff_failed", self.events)

    def test_loop_limit_capability_resource_and_cooldown_checks(self):
        clock = FakeClock()
        router, task = self.make(clock=clock), FakeTask()
        self.assertFalse(router.request_handoff(task, "Coder-7B", "needs vision for a screenshot", "Text-4B", 2)["ok"])
        self.assertFalse(router.request_handoff(task, "Huge-70B", "bigger", "Text-4B", 2)["ok"])      # 48 GB > 12 GB
        self.assertTrue(router.request_handoff(task, "Vision-11B", "vision", "Text-4B", 2)["ok"])
        clock.t += 120
        loop = router.request_handoff(task, "Text-4B", "back to text", "Vision-11B", 2)
        self.assertFalse(loop["ok"])
        self.assertIn("loop", loop["message"])
        self.assertFalse(router.request_handoff(task, "Coder-7B", "coding", "Vision-11B", 1)["ok"])   # limit
        # Cooldown: two handoffs back to back (different tasks, so no loop/limit involved).
        router2 = self.make(clock=FakeClock())
        self.assertTrue(router2.request_handoff(FakeTask(), "Vision-11B", "vision", "Text-4B", 2)["ok"])
        self.assertIn("cooldown", router2.request_handoff(FakeTask(), "Coder-7B", "coding", "Vision-11B", 2)["message"])

    def test_handoff_persists_as_the_selected_model(self):
        import main
        saved = {k: shared.app_state.get(k) for k in ("local_model", "local_engaged", "routed_model", "settings")}
        try:
            shared.app_state["settings"] = {**shared.app_state["settings"], "auto_model_routing": True}
            with patch.object(llama_local, "server_models", return_value=["Vision-11B"]), \
                 patch.object(llama_local, "force_load", return_value=1.0), \
                 patch.object(system_metrics, "begin_model_load"), patch.object(system_metrics, "finish_model_load"):
                handoff._load_model("Vision-11B", "local")
            self.assertEqual(shared.app_state["local_model"], "Vision-11B")
            self.assertEqual(shared.app_state["routed_model"]["model"], "Vision-11B")
        finally:
            shared.app_state.update(saved)

    def test_routing_off_keeps_a_pinned_text_model_even_for_screenshots(self):
        import main
        saved = {k: shared.app_state.get(k) for k in ("local_model", "local_engaged", "settings")}
        try:
            shared.app_state.update(local_model="Mistral-7B", local_engaged=None)
            shared.app_state["settings"] = {**shared.app_state["settings"], "auto_model_routing": False}
            with patch.object(llama_local, "is_vision_model", return_value=False):
                self.assertEqual(llm.resolve_local_model(True), "Mistral-7B")
        finally:
            shared.app_state.update(saved)

    def test_local_mode_refuses_a_cloud_handoff(self):
        import main
        saved = shared.app_state.get("mode")
        try:
            shared.app_state["mode"] = config.MODE_LOCAL
            cloud = model_router.ModelInfo("gemini/x", "x", "cloud", available=True)
            with patch.object(handoff.model_registry, "get", return_value=cloud):
                res = handoff._handoff_tool({"target_model": "gemini/x", "reason": "faster"})
            self.assertFalse(res["ok"])
            self.assertIn("Local mode", res["message"])
        finally:
            shared.app_state["mode"] = saved


# ==============================================================================
# Themes (static checks of index.html; runtime switching was checked in a browser)
# ==============================================================================
def _hex_lum(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(a, b):
    la, lb = sorted((_hex_lum(a), _hex_lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


class ThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = Path(__file__).resolve().parent.parent.joinpath("index.html").read_text(encoding="utf-8")
        cls.css = cls.html[cls.html.index("<style>"):cls.html.index("</style>")]

    def blocks(self, skin):
        """Every top-level token block for a skin, merged like the cascade does
        (a skin can have several: palette, reveal timing, ...)."""
        return "\n".join(m.group(1) for m in re.finditer(r'^:root\[data-skin="%s"\] \{(.*?)\}' % skin, self.css, re.S | re.M))

    def palette(self, skin):
        return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", self.blocks(skin)))

    def test_theme_lists_match_between_ui_and_server(self):
        import main
        ui = re.search(r"const THEMES = \[(.*?)\]", self.html).group(1)
        self.assertEqual(tuple(re.findall(r"'(\w+)'", ui)), config.UI_THEMES)

    def test_every_skin_defines_a_full_palette(self):
        for skin in ("dawn", "dusk", "terminal", "eclipse", "aurora"):
            names = set(re.findall(r"--([\w-]+):", self.blocks(skin)))
            for token in ("bg", "surface", "text", "muted", "accent", "accent-ink", "border", "ok", "warn", "danger"):
                self.assertIn(token, names, f"{skin} is missing --{token}")

    def test_dawn_is_warm_light_and_readable(self):
        p = self.palette("dawn")
        self.assertGreater(_hex_lum(p["bg"]), 0.85)                          # light, off-white ground
        self.assertIn("color-scheme: light", self.blocks("dawn"))
        for fg in ("text", "text-2", "muted"):
            self.assertGreaterEqual(_contrast(p[fg], p["bg"]), 4.5, fg)
            self.assertGreaterEqual(_contrast(p[fg], p["surface"]), 4.5, fg)
        self.assertGreaterEqual(_contrast(p["accent-ink"], p["accent"]), 4.5)
        self.assertLess(_contrast(p["bg"], p["surface-2"]), 1.2)            # soft, tinted surfaces

    def test_dusk_is_warm_dark_and_readable(self):
        # Dawn after sunset: the same family on a dark ground.
        p = self.palette("dusk")
        self.assertLess(_hex_lum(p["bg"]), 0.05)
        self.assertIn("color-scheme: dark", self.blocks("dusk"))
        for fg in ("text", "text-2", "muted"):
            self.assertGreaterEqual(_contrast(p[fg], p["bg"]), 4.5, fg)
            self.assertGreaterEqual(_contrast(p[fg], p["surface"]), 4.5, fg)
        self.assertGreaterEqual(_contrast(p["text"], p["bg"]), 12)          # body text well above AA
        self.assertGreaterEqual(_contrast(p["accent-ink"], p["accent"]), 4.5)

    def test_dawn_and_dusk_reuse_eclipse_rules_instead_of_copying_them(self):
        shared = self.css.count(':root:is([data-skin="eclipse"],[data-skin="dawn"],[data-skin="dusk"])')
        self.assertGreater(shared, 20)
        # Only the palette and the starfield are Eclipse's own (its by-day tone adds [data-tone="light"] rules).
        self.assertEqual(len(re.findall(r':root\[data-skin="eclipse"\](?![\[:])', self.css)), 2)

    def test_dawn_is_calmer_than_eclipse(self):
        dawn_idle = float(re.search(r':root\[data-skin="dawn"\] \.halo \{ opacity: ([\d.]+)', self.css).group(1))
        self.assertLess(dawn_idle, 0.45)                                     # eclipse idle corona is .45
        self.assertTrue(re.search(r':root\[data-skin="dawn"\]\[data-state="working"\] \.halo', self.css))

    def test_reduced_motion_disables_animation(self):
        block = "\n".join(re.findall(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n\}", self.css, re.S))
        self.assertIn("animation-duration: 0s", block)
        self.assertIn("transition-duration: 0s", block)
        self.assertIn("reducedMotion.matches", self.html)                   # view transitions skipped too

    def test_every_theme_has_a_reveal_style_and_no_keyframed_drift(self):
        self.assertIn("--reveal-anim: reveal-soft", self.css)                # light/warm: soft
        self.assertIn('[data-skin="classic"][data-scheme="dark"] { --reveal-anim: reveal-mist', self.css)
        self.assertIn('[data-skin="aurora"]) { --reveal-anim: reveal-glow', self.css)
        self.assertIn('[data-skin="terminal"] { --reveal-anim: reveal-type', self.css)
        # The old target-point keyframes (aurora drift, 100vh scan) are gone;
        # motion now comes from the reflecting Ambient engine.
        self.assertNotIn("@keyframes aurora", self.css)
        self.assertNotIn("@keyframes scan", self.css)
        self.assertIn("animation: corona 9s linear infinite paused", self.css)   # paused, never removed

    def test_no_universal_normal_labels(self):
        for label in ("'Normal'", ">Normal<", "Normal mode", "default: 'Default'"):
            self.assertNotIn(label, self.html)
        import orchestrator
        self.assertEqual(orchestrator.EFFORT_LABELS["normal"], "Balanced")

    def test_light_skins_resolve_to_a_light_scheme(self):
        self.assertIn("const LIGHT_SKINS = ['dawn', 'termlight', 'eclipselight', 'auroralight', 'claylight', 'monolight', 'voidlight', 'prismlight']", self.html)
        self.assertIn("const themeScheme = t => t === 'light' || LIGHT_SKINS.includes(t) ? 'light' : 'dark';", self.html)
        self.assertIn("root.dataset.scheme = scheme;", self.html)

    def tone(self, tone):
        block = re.search(r'^:root\[data-skin="terminal"\]\[data-tone="%s"\] \{(.*?)\}' % tone, self.css, re.S | re.M).group(1)
        return block, dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", block))

    def test_terminal_lite_is_terminal_with_a_calmer_palette(self):
        # Both ride on the Terminal skin (so every terminal rule and motion
        # applies) and only swap the palette and drop the heavy effects.
        self.assertIn("const TONES = { termdark: 'lite-dark', termlight: 'lite-light', eclipselight: 'light', auroralight: 'light',", self.html)
        self.assertIn("const SKIN_OF = { termdark: 'terminal', termlight: 'terminal',", self.html)
        self.assertIn("root.dataset.skin = classic ? 'classic' : SKIN_OF[theme] || theme;", self.html)
        self.assertIn(':root[data-skin="terminal"][data-tone] .app::before { content: none; }', self.css)   # no scanlines
        self.assertIn(':root[data-skin="terminal"][data-tone] .greeting { text-shadow: none; }', self.css)  # no glow
        full = self.palette("terminal")
        dark_block, dark = self.tone("lite-dark")
        light_block, light = self.tone("lite-light")
        self.assertLess(_hex_lum(dark["bg"]), 0.05)
        self.assertGreater(_hex_lum(light["bg"]), 0.85)
        self.assertIn("color-scheme: light", light_block)
        for p in (dark, light):
            for token in ("bg", "surface", "text", "muted", "accent", "accent-ink", "border", "ok", "warn", "danger"):
                self.assertIn(token, p)
            for fg in ("text", "text-2", "muted"):
                self.assertGreaterEqual(_contrast(p[fg], p["bg"]), 4.5, fg)
                self.assertGreaterEqual(_contrast(p[fg], p["surface"]), 4.5, fg)
            self.assertGreaterEqual(_contrast(p["accent"], p["bg"]), 4.5)
            self.assertGreaterEqual(_contrast(p["accent-ink"], p["accent"]), 4.5)
        # Less glare than full Terminal: body text is softer against the ground.
        self.assertLess(_contrast(dark["text"], dark["bg"]), _contrast(full["text"], full["bg"]))
        self.assertLess(_contrast(dark["accent"], dark["bg"]), _contrast(full["accent"], full["bg"]))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main
        cls.main = main
        cls.client = TestClient(main.app, base_url="http://127.0.0.1:8000")

    def test_tool_catalogue_endpoint(self):
        tools = {t["name"]: t for t in self.client.get("/api/tools").json()["tools"]}
        self.assertTrue(tools["read_file"]["read_only"])
        self.assertEqual(tools["messages_send"]["permissions"], ["messages.send"])

    def test_model_registry_endpoint(self):
        body = self.client.get("/api/models/registry").json()
        self.assertIn("models", body)
        self.assertIn("routing", body)

    def test_agent_settings_are_validated(self):
        before = dict(shared.app_state["settings"])
        try:
            # save_settings is patched where it lives, so the real settings.json is never written.
            with patch.object(config, "save_settings") as save:
                s = self.client.post("/api/settings", json={"effort": "ludicrous", "prompt_enhancement": "detailed",
                                                            "max_revisions": 7, "theme": "dawn"}).json()["settings"]
            self.assertEqual((s["effort"], s["prompt_enhancement"], s["max_revisions"], s["theme"]),
                             ("normal", "detailed", "auto", "dawn"))
            save.assert_called_once()
        finally:
            shared.app_state["settings"] = before


if __name__ == "__main__":
    unittest.main()
