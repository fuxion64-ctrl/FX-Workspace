"""Offline orchestration tests: production functions, mocked OS/model boundaries.

AST loading keeps FastAPI, Windows GUI threads, and model servers out of these
tests; the function bodies being tested are taken unchanged from the modules
they live in since the Phase 2 split (agent/tasks.py, agent/loop.py, ...).

Those bodies reach other modules through aliases (`shared.app_state`,
`agent_loop.run_agent_loop`, ...). Every alias here is a view onto the one
namespace `self.g`, so the stand-ins below (Mocks, a plain dict for
app_state) are what the production code sees, exactly as before the split.
"""
import ast
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import context
import intent_router
import model_output
import permissions
import stream_hub
from run_control import RunControl, RunStopped


SOURCES = ("agent/tasks.py", "agent/loop.py", "agent/parsing.py", "tools/schemas.py", "tools/approvals.py")
ALIASES = ("shared", "config", "event_log", "agent_loop", "tasks", "parsing", "schemas", "approvals",
           "runtime", "llm", "files", "dispatch", "prompts", "handoff", "observer")


class _View:
    """A module alias that reads and writes the test's flat namespace."""

    def __init__(self, g):
        object.__setattr__(self, "_g", g)

    def __getattr__(self, name):
        try:
            return self._g[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self._g[name] = value


class TaskFlowTests(unittest.TestCase):
    def setUp(self):
        names = {"chat_with_agent", "_select_request_backend", "run_desktop_intent", "_run_desktop_route", "continue_desktop_task", "run_agent_loop",
                 "run_task_queue", "resume_task_queue", "finish_task_queue",
                 "_normalize_clarification_options", "_same_clarification", "_clarification_key",
                 "tool_allowed_current_turn", "gate_batch", "_stopped_response", "_previous_turn", "_classify",
                 "_decide", "_same_tool_batch", "_held_note"}
        root = Path(__file__).resolve().parent.parent
        nodes = [node for src in SOURCES for node in ast.parse((root / src).read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({n.name for n in nodes}, names)
        for node in nodes:
            node.decorator_list = []
        self.state = {"conversation": [], "local_model": "", "pending_clarification": None,
                      "turn_policy": {"mode": "action", "allow_mutation": True, "reason": "test action"},
                      "settings": dict(permissions.DEFAULT_PERMISSION_SETTINGS),
                      "waiting_for_os_approval": False}
        self.g = {
            "ChatRequest": SimpleNamespace, "re": re, "time": time,
            "intent_router": intent_router, "app_state": self.state,
            "permissions": permissions,
            "state_lock": threading.Lock(), "MODE_CLOUD": "cloud", "MODE_LOCAL": "local",
            "VALID_MODES": ["cloud", "local", "auto"], "log_action": Mock(),
            "task_board": context.TaskBoard(), "ledger": context.ActionLedger(),
            "ask_batch_approval_os": Mock(return_value=True),
            "MAX_LOCAL_ITERATIONS": 3, "MAX_AGENT_ITERATIONS": 3, "MAX_NUDGES": 1,
            # run_agent_loop's fabricated-tool-call net; not under test here.
            "_fabricated_tool_name": lambda prose: None,
            # Pause/stop: a real controller, so checkpoints behave as in production.
            "run_control": RunControl(), "RunStopped": RunStopped,
            "_announce_pause": lambda: None, "emit_event": Mock(),
            # Activity reporting and malformed-call handling (tested in test_tool_calls).
            "finish_event": Mock(), "_model_call_facts": lambda response, started: {},
            "_emit_model_note": Mock(), "_user_safe": lambda text, limit: str(text)[:limit],
            "_invalid_call_reason": lambda name, args: None, "_answer_invalid_calls": Mock(),
            # every name these flows use is a real tool (unknown names: test_bugfixes_d36)
            "known_tool": lambda name: True,
            "MAX_BAD_CALL_ROUNDS": 3,
            # The one tool path for approved desktop routes: here it just runs the route.
            "execute_internal": lambda name, args, runner: runner(),
            # The Web Search privacy mode takes no tools away here (test_search_privacy covers it).
            "search_mode_hidden": lambda: set(), "world_hidden": lambda: {"world_monitor"},
            # Roblox's Studio MCP arguments (test_studio_tools covers studio_mcp); one-tool batches of a local model.
            "studio_mcp": SimpleNamespace(fill_args=lambda name, args: args),
            "_PLAN_TOOLS": {"set_plan", "update_plan", "ask_user"}, "MAX_SAME_TOOL_BATCH": 30,
            # A stopped reply keeps what it had streamed (stream_hub.take_stopped; test_streaming covers it).
            "stream_hub": stream_hub, "model_output": model_output,
        }
        for alias in ALIASES:
            self.g[alias] = _View(self.g)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "agent/tasks.py", "exec"), self.g)
        self.real_loop = self.g["run_agent_loop"]
        self.g["run_agent_loop"] = Mock(side_effect=self.complete_model_task)
        self.g["desktop_router"] = SimpleNamespace(execute=Mock(side_effect=lambda intent: {
            "ok": True, "verified": True, "title": intent.app_name, "message": f"Opened {intent.app_name}"}))

    def chat(self, message):
        return self.g["chat_with_agent"](SimpleNamespace(
            message=message, mode="local", local_model="", local_text_model=None, with_vision=False))

    def complete_model_task(self):
        for step in list(self.g["task_board"].steps):
            self.g["task_board"].update(step["n"], "done", "Mock observation confirmed completion")
        return {"status": "success", "message": "Mock scan complete"}

    def test_stop_mid_tool_batch_leaves_conversation_consistent(self):
        # The model asked for a tool, then the user pressed Stop before it ran.
        def stopped_loop():
            self.state["conversation"].append({"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "delete_path", "arguments": "{}"}}]})
            raise RunStopped()
        self.g["run_agent_loop"] = Mock(side_effect=stopped_loop)
        response = self.chat("tidy my downloads folder")
        self.assertEqual(response["status"], "stopped")
        tool_msgs = [m for m in self.state["conversation"] if m.get("role") == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["call_1"])
        self.assertIn("stopped", tool_msgs[0]["content"])
        self.assertEqual(self.g["run_control"].state(), "idle")
        self.assertFalse(self.g["state_lock"].locked())

    def test_original_screenshot_sequence_continues_to_wiztree(self):
        response = self.chat("Connect to proton and scan C with wiztree")
        self.assertEqual(response["status"], "clarification")
        response = self.chat(response["option_values"][0])
        self.assertEqual(response["status"], "success")
        calls = self.g["desktop_router"].execute.call_args_list
        self.assertEqual([c.args[0].app_name.casefold() for c in calls], ["proton vpn", "wiztree"])
        self.g["run_agent_loop"].assert_not_called()
        self.assertEqual(calls[1].args[0].arguments, {"drive": "C:"})
        self.assertIsNone(self.state["pending_clarification"])
        self.assertEqual(self.state["intent_guard"], {"expected_app": "wiztree"})

    def test_music_then_scan_preserves_completed_playback(self):
        response = self.chat("Play music in tidal and scan C with wiztree")
        self.assertEqual(response["status"], "success")
        calls = self.g["desktop_router"].execute.call_args_list
        self.assertEqual([c.args[0].app_name for c in calls], ["tidal", "wiztree"])
        self.assertEqual(calls[0].args[0].actions, ["play"])
        self.g["run_agent_loop"].assert_not_called()
        steps = self.g["task_board"].steps
        self.assertEqual([s["status"] for s in steps], ["done", "done"])
        self.assertIn("Play music", steps[0]["text"])
        self.assertIsNone(self.state["pending_clarification"])

    def test_failed_playback_advances_without_replaying(self):
        execute = self.g["desktop_router"].execute
        execute.side_effect = lambda intent: (
            {"ok": False, "verified": False, "title": "TIDAL", "message": "Playback not verified"}
            if intent.app_name == "tidal" else
            {"ok": True, "verified": True, "title": "WizTree", "message": "Opened WizTree"})
        response = self.chat("Play music in tidal and scan C with wiztree")
        self.assertEqual(response["status"], "partial")
        self.assertEqual([c.args[0].app_name for c in execute.call_args_list], ["tidal", "wiztree"])
        self.assertEqual([s["status"] for s in self.g["task_board"].steps], ["failed", "done"])
        self.g["run_agent_loop"].assert_not_called()

    def test_playback_failure_is_not_retried_automatically(self):
        execute = self.g["desktop_router"].execute
        execute.side_effect = [
            {"ok": False, "title": "TIDAL", "message": "No visible Play control"},
            {"ok": True, "verified": True, "title": "WizTree", "message": "Opened WizTree"},
        ]
        response = self.chat("Play music in tidal and scan C with wiztree")
        self.assertEqual(response["status"], "partial")
        self.assertEqual([c.args[0].app_name for c in execute.call_args_list], ["tidal", "wiztree"])
        self.g["run_agent_loop"].assert_not_called()

    def test_unrelated_new_request_discards_pending_queue(self):
        self.chat("Connect to proton and scan C with wiztree")
        self.chat("Open Notepad")
        execute = self.g["desktop_router"].execute
        self.assertEqual([c.args[0].app_name for c in execute.call_args_list], ["Notepad"])
        self.g["run_agent_loop"].assert_not_called()

    def test_new_scan_discards_pending_proton_question(self):
        self.chat("Connect to proton")
        self.chat("Scan drive C with wiztree")
        self.assertEqual(self.g["desktop_router"].execute.call_args.args[0].app_name, "wiztree")
        self.assertIsNone(self.state["pending_clarification"])

    def test_rejected_connect_does_not_run_scan(self):
        self.g["ask_batch_approval_os"].return_value = False
        self.chat("Connect to Proton VPN and scan C with wiztree")
        self.g["desktop_router"].execute.assert_not_called()
        self.g["run_agent_loop"].assert_not_called()

    def test_failed_connection_does_not_run_scan(self):
        self.g["desktop_router"].execute.side_effect = None
        self.g["desktop_router"].execute.return_value = {
            "ok": False, "title": "Proton VPN", "message": "Connection not verified"}
        response = self.chat("Connect to Proton VPN and scan C with wiztree")
        self.assertIn("Not executed because", response["message"])
        self.g["desktop_router"].execute.assert_called_once()
        self.g["run_agent_loop"].assert_not_called()

    def test_installed_app_question_preserves_scan(self):
        self.g["desktop_router"].execute.side_effect = None
        self.g["desktop_router"].execute.return_value = {
            "ok": False, "clarification": True, "question": "Which installed app?",
            "options": ["WizTree", "WizTree Portable"]}
        response = self.chat("open wiztree and scan C")
        self.assertEqual(response["status"], "clarification")
        self.assertEqual(response["option_values"][0], "Open WizTree")
        self.assertEqual(self.state["task_queue"]["items"][1]["text"], "scan C")
        self.assertEqual(self.state["pending_clarification"]["source"], "desktop")

    def test_failed_deterministic_scan_reports_partial(self):
        execute = self.g["desktop_router"].execute
        execute.side_effect = [
            {"ok": True, "verified": True, "title": "WizTree", "message": "Opened WizTree"},
            {"ok": False, "verified": False, "title": "WizTree", "message": "Scan not verified"},
        ]
        response = self.chat("open wiztree and scan C")
        self.assertEqual(response["status"], "partial")
        self.assertIn("opened wiztree", response["message"].casefold())
        self.assertIn("Scan not verified", response["message"])
        self.assertFalse(self.g["task_board"].is_complete())

    def test_model_cannot_reopen_unrelated_proton_clarification(self):
        # 1.4.8 (audit 2.11): blocked because Proton was a finished task, not because of a hardcoded "proton".
        self.state.update(mode="local", intent_guard={"expected_app": "wiztree"}, completed_app_tasks=["proton"],
                          current_user_request="Scan drive C with wiztree")
        self.g["call_model"] = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=object())]))
        self.g["parse_response"] = Mock(side_effect=[
            ([(None, "ask_user", {"question": "Which Proton product?", "options": ["Proton VPN", "Proton Mail"]})], ""),
            ([], "Could not inspect the scan controls"),
        ])
        response = self.real_loop()
        self.assertNotEqual(response["status"], "clarification")
        self.assertIsNone(self.state["pending_clarification"])
        self.assertEqual(self.g["call_model"].call_count, 2)

    def test_scan_cannot_ask_again_about_completed_tidal(self):
        self.state.update(mode="local", intent_guard={"expected_app": "wiztree"},
                          completed_app_tasks=["tidal"], current_user_request="Scan drive C with wiztree")
        self.g["call_model"] = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=object())]))
        self.g["parse_response"] = Mock(side_effect=[
            ([(None, "ask_user", {"question": "Which Tidal playlist?", "options": ["My Favorites", "Recently Played"]})], ""),
            ([], "Could not inspect scan controls"),
        ])
        response = self.real_loop()
        self.assertNotEqual(response["status"], "clarification")
        self.assertIsNone(self.state["pending_clarification"])
        self.assertEqual(self.g["call_model"].call_count, 2)

    def test_plan_does_not_reactivate_completed_or_skipped_prefix(self):
        board = self.g["task_board"]
        board.set_plan("Continue", [{"text": "Open TIDAL", "status": "done"},
                                    {"text": "Playback", "status": "skipped"}, "Scan C"])
        self.assertEqual([s["status"] for s in board.steps], ["done", "skipped", "active"])

    def test_fully_completed_plan_stays_completed(self):
        board = self.g["task_board"]
        board.set_plan("Done", [{"text": "Playback", "status": "done"}])
        self.assertTrue(board.is_complete())

    def test_screenshot_partial_failure_keeps_scan_and_resumes_only_proton(self):
        execute = self.g["desktop_router"].execute
        execute.side_effect = [
            {"ok": False, "verified": False, "message": "TIDAL rejected pause"},
            {"ok": True, "verified": True, "title": "WizTree", "message": "Opened WizTree"},
            {"ok": True, "verified": True, "title": "Proton VPN", "message": "VPN connected"},
        ]
        response = self.chat("stop music in tidal, scan drive C in wiztree and connect to proton")
        self.assertEqual(response["status"], "clarification")
        self.assertEqual([i["status"] for i in self.state["task_queue"]["items"]], ["failed", "done", "active"])
        response = self.chat("Proton VPN")
        self.assertEqual(response["status"], "partial")
        self.assertEqual([s["status"] for s in response["plan"]["steps"]], ["failed", "done", "done"])
        self.assertIn("Done: scan drive C in wiztree", response["message"])
        self.assertIn("TIDAL rejected pause", response["message"])
        self.assertIn("VPN connected", response["message"])
        self.assertEqual([c.args[0].app_name for c in execute.call_args_list], ["tidal", "wiztree", "Proton VPN"])
        self.assertEqual(execute.call_args_list[0].args[0].actions, ["stop"])
        self.g["run_agent_loop"].assert_not_called()
        self.assertIsNone(self.state["pending_clarification"])

    def test_all_three_tasks_complete(self):
        self.chat("stop music in tidal, scan drive C in wiztree and connect to proton")
        response = self.chat("Proton VPN")
        self.assertEqual(response["status"], "success")
        self.assertEqual([s["status"] for s in response["plan"]["steps"]], ["done"] * 3)
        self.assertEqual(self.g["desktop_router"].execute.call_count, 3)

    def test_exact_reported_sentence_needs_no_model_or_clarification(self):
        response = self.chat(
            "play music on tidal, scan drive G in wiztree and establish vpn connection in proton")
        self.assertEqual(response["status"], "success")
        calls = self.g["desktop_router"].execute.call_args_list
        self.assertEqual([call.args[0].app_name for call in calls], ["tidal", "wiztree", "Proton VPN"])
        self.assertEqual([call.args[0].actions for call in calls], [["play"], ["scan"], ["connect"]])
        self.assertEqual(calls[1].args[0].arguments, {"drive": "G:"})
        self.g["run_agent_loop"].assert_not_called()
        self.assertIsNone(self.state["pending_clarification"])

    def test_previous_bare_scan_and_proton_resume_without_model(self):
        response = self.chat("Prev song, scan drive G and connect to proton")
        self.assertEqual(response["status"], "clarification")
        calls = self.g["desktop_router"].execute.call_args_list
        self.assertEqual([call.args[0].app_name for call in calls], ["active media", "WizTree"])
        self.assertEqual([call.args[0].actions for call in calls], [["previous"], ["scan"]])
        self.assertEqual(calls[1].args[0].arguments, {"drive": "G:"})
        response = self.chat("Proton VPN")
        self.assertEqual(response["status"], "success")
        calls = self.g["desktop_router"].execute.call_args_list
        self.assertEqual([call.args[0].app_name for call in calls],
                         ["active media", "WizTree", "Proton VPN"])
        self.g["run_agent_loop"].assert_not_called()

    def test_skip_disconnect_and_scan_never_enter_the_model_loop(self):
        response = self.chat("Skip song, disconnect from vpn, scan drive g")
        self.assertEqual(response["status"], "success")
        calls = self.g["desktop_router"].execute.call_args_list
        self.assertEqual([call.args[0].app_name for call in calls],
                         ["active media", "Proton VPN", "WizTree"])
        self.assertEqual([call.args[0].actions for call in calls],
                         [["next"], ["disconnect"], ["scan"]])
        self.assertEqual(calls[2].args[0].arguments, {"drive": "G:"})
        self.g["run_agent_loop"].assert_not_called()

    def test_generic_pause_and_close_wiztree_never_ask_a_model_question(self):
        response = self.chat("pause music and close wiztree")
        self.assertEqual(response["status"], "success")
        calls = self.g["desktop_router"].execute.call_args_list
        self.assertEqual([call.args[0].actions for call in calls], [["pause"], ["close"]])
        self.g["run_agent_loop"].assert_not_called()

    def test_closed_proton_is_reported_as_interrupted_not_visual_unknown(self):
        self.g["desktop_router"].execute.side_effect = None
        self.g["desktop_router"].execute.return_value = {
            "ok": False, "verified": False, "title": "Proton VPN", "interrupted": True,
            "message": "The Proton VPN window was closed before verification."}
        response = self.chat("Connect to Proton VPN")
        self.assertEqual(response["status"], "error")
        self.assertIn("closed before verification", response["message"])
        self.assertNotIn("screenshot", response["message"].casefold())

    def test_unverified_scan_does_not_block_connection_or_claim_completion(self):
        execute = self.g["desktop_router"].execute
        execute.side_effect = [
            {"ok": True, "verified": False, "title": "WizTree", "message": "Scan click sent"},
            {"ok": True, "verified": True, "title": "Proton VPN", "message": "VPN connected"},
        ]
        response = self.chat("scan C with wiztree and connect to Proton VPN")
        self.assertEqual([s["status"] for s in response["plan"]["steps"]], ["failed", "done"])
        self.assertIn("Completion not verified", response["message"])

    def test_rejected_media_stops_remaining_tasks(self):
        self.g["ask_batch_approval_os"].return_value = False
        response = self.chat("stop music in tidal, scan C with wiztree and connect to Proton VPN")
        self.g["desktop_router"].execute.assert_not_called()
        self.g["run_agent_loop"].assert_not_called()
        self.assertIn("Not executed because", response["message"])

    def test_explicit_then_preserves_dependency(self):
        self.g["desktop_router"].execute.side_effect = None
        self.g["desktop_router"].execute.return_value = {"ok": False, "message": "Pause failed"}
        response = self.chat("stop music in tidal then scan C with wiztree")
        self.g["desktop_router"].execute.assert_called_once()
        self.g["run_agent_loop"].assert_not_called()
        self.assertIn("Not executed because", response["message"])

    def test_answered_model_question_is_bounded(self):
        self.state.update(mode="local", current_user_request="Connect to Proton VPN",
                          resolved_clarification={"question": "Which Proton product?", "answer": "Proton VPN"})
        self.g["call_model"] = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=object())]))
        self.g["parse_response"] = Mock(return_value=(
            [(None, "ask_user", {"question": "Which Proton product?", "options": ["Proton VPN", "Proton Mail"]})], ""))
        response = self.real_loop()
        self.assertEqual(response["status"], "error")
        self.assertEqual(self.g["call_model"].call_count, 2)
        self.assertIsNone(self.state["pending_clarification"])

    def test_model_question_resumes_saved_plan_before_connecting(self):
        def clarify_scan():
            self.state["pending_clarification"] = {"question": "Which drive?", "options": ["C:", "D:"],
                                                   "option_values": ["C:", "D:"], "original_request": "scan C with wiztree"}
            return {"status": "clarification", "question": "Which drive?"}
        self.g["run_agent_loop"].side_effect = clarify_scan
        response = self.chat("stop music in tidal, search for large files and connect to Proton VPN")
        self.assertEqual(response["status"], "clarification")
        self.assertEqual(self.state["task_queue"]["index"], 1)
        self.g["run_agent_loop"].side_effect = self.complete_model_task
        response = self.chat("C:")
        self.assertEqual(response["status"], "success")
        self.assertEqual([s["status"] for s in response["plan"]["steps"]], ["done"] * 3)
        self.assertEqual([c.args[0].app_name for c in self.g["desktop_router"].execute.call_args_list],
                         ["tidal", "Proton VPN"])

    def test_connect_pronoun_stays_with_original_app(self):
        response = self.chat("open Proton VPN and connect it")
        self.assertEqual(response["status"], "success")
        execute = self.g["desktop_router"].execute
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0].app_name, "Proton VPN")
        self.assertEqual(execute.call_args.args[0].actions, ["connect"])

    def test_greeting_clears_stale_vm_task_and_the_model_answers(self):
        # D48: the model answers a greeting (a canned reply read as fake); what older requests left open is
        # still dropped first, and the turn has no computer tools.
        self.g["task_board"].set_plan("Start old VM", ["open VirtualBox", "start old VM"])
        self.state["pending_clarification"] = {"question": "Which VM?", "options": ["A", "B"]}
        seen = {}

        def answer():
            seen["steps"] = list(self.g["task_board"].steps)
            seen["policy"] = dict(self.state["turn_policy"])
            return {"status": "success", "message": "Hi Alex! What are we working on?"}
        self.g["run_agent_loop"] = Mock(side_effect=answer)
        response = self.chat("hi")
        self.assertEqual(response["message"], "Hi Alex! What are we working on?")
        self.g["run_agent_loop"].assert_called_once()
        self.assertEqual(seen["steps"], [])
        self.assertEqual(seen["policy"]["mode"], "none")
        self.assertIsNone(self.state["pending_clarification"])
        self.assertEqual(self.state["conversation"][-1], {"role": "user", "content": "hi"})
        self.g["ask_batch_approval_os"].assert_not_called()
        self.g["desktop_router"].execute.assert_not_called()

    def test_stale_mutating_tool_is_blocked_before_approval(self):
        self.state.update(mode="local", current_user_request="Tell me a joke",
                          turn_policy={"mode": "none", "allow_mutation": False,
                                       "reason": "no explicit computer action"})
        self.g["call_model"] = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=object())]))
        self.g["parse_response"] = Mock(return_value=(
            [(None, "open_app_and_list", {"app_name": "VMware Workstation"})], "Opening it."))
        response = self.real_loop()
        self.assertIn("did not run", response["message"])
        self.g["ask_batch_approval_os"].assert_not_called()
        self.g["desktop_router"].execute.assert_not_called()
        self.assertFalse(self.g["task_board"].steps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
