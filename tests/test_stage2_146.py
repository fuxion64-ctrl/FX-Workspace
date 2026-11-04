"""1.4.6 stage 2 - tools faster and on by themselves: what each change does for someone using the app.

Measured live first (4 Nov 2026, Qwen 3.5 4B and Ornith 9B on the dev server, through a logging proxy in front of
llama.cpp): docs/plans/1.4.6-stage2.md has the numbers. These tests hold the behaviour the numbers came from.
"""
import time
import unittest
from unittest.mock import patch

import core.state as shared
from chat import effort, intent_router
from llm import speed_tune
from tools import schemas

PAGE = open("index.html", encoding="utf-8").read()
SERVER = open("api/server.py", encoding="utf-8").read()


class WriteItInTheReplyTests(unittest.TestCase):
    """"Write a Python function ..." got run_python, ran it "to test it", and the user got an approval card for code
    they had only asked to read (measured: the 4B model did this every time)."""

    def mode(self, text):
        return intent_router.classify_turn(text)["mode"]

    def test_text_the_reply_is_needs_no_tools(self):
        for text in ("Write a Python function that reverses the words in a sentence.", "write me a poem about the sea",
                     "can you draft an email to my landlord about the heater", "write a short story",
                     "give me a regex for emails", "compose a polite reply to Dana"):
            self.assertEqual(self.mode(text), "none", text)

    def test_somewhere_to_put_it_is_still_an_action(self):
        for text in ("write a function and save it to reverse.py", "write a lua script that makes the part spin",
                     "write a script for a kill brick", "Write the file", "write this down", "write my plan down as a doc",
                     "write an email and send it to Dana", "write a poem in notes", "write a function and test it",
                     "Write a report about the supplied private notes"):
            self.assertEqual(self.mode(text), "action", text)


class QuickLookEffortTests(unittest.TestCase):
    """Auto effort: a one-step look is Balanced - no model-written plan, no model review (they cost 1.2-2.2 s and
    ~1.1 s on the 4B model before the answer); a change stays High."""

    def test_levels(self):
        self.assertEqual(effort.auto_level({"mode": "read", "words": 8})[0], "normal")
        self.assertEqual(effort.auto_level({"mode": "read", "words": 60})[0], "high")
        self.assertEqual(effort.auto_level({"mode": "action", "words": 5})[0], "high")
        self.assertEqual(effort.auto_level({"mode": "read", "words": 8, "complex": True})[0], "high")

    def test_balanced_reads_get_no_model_plan_or_review(self):
        from chat import orchestrator
        profile = effort.PROFILES["normal"]
        analysis = {"mode": "read", "trivial": False, "complex": False}
        self.assertEqual(orchestrator.planner_mode(profile, analysis), "off")
        self.assertEqual(orchestrator.reviewer_mode(profile, analysis, 0), "deterministic")


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


class StableToolOrderTests(unittest.TestCase):
    """Local chat templates put the tool list first: llama.cpp reuses what it read up to the first tool that differs."""

    def test_shared_tools_first_use_tools_last(self):
        got = [t["function"]["name"] for t in schemas.stable_order(
            [_tool(n) for n in ("ask_user", "web_search", "use_tools", "observe", "web_read", "fetch_page", "list_dir")])]
        self.assertEqual(got, ["web_search", "fetch_page", "web_read", "ask_user", "observe", "list_dir", "use_tools"])

    def test_two_kinds_of_turn_share_their_start(self):
        chat = schemas.stable_order([_tool(n) for n in ("web_search", "fetch_page", "use_tools", "web_read")])
        numbers = schemas.stable_order([_tool(n) for n in ("web_search", "fetch_page", "compute_stats", "make_chart",
                                                           "use_tools", "web_read")])
        self.assertEqual(chat[:3], numbers[:3])

    def test_local_only(self):
        self.assertIn("return stable_order(tools) if local else tools", open("tools/schemas.py", encoding="utf-8").read())


class ForcedToolRetryTests(unittest.TestCase):
    """A local model that describes an open step instead of doing it: the retry may only be a tool call."""

    def test_the_flag_makes_the_next_call_required_once(self):
        from agent import models
        sent = []
        with patch.object(models, "_local_complete", side_effect=lambda a, c=None: sent.append(a) or "res"), \
                patch("model_caps.protocol", return_value="native"):
            shared.app_state["force_tool_once"] = True
            models._complete_with_protocol({"messages": []}, "m", [_tool("write_file")])
            models._complete_with_protocol({"messages": []}, "m", [_tool("write_file")])
        self.assertEqual([s["tool_choice"] for s in sent], ["required", "auto"])
        self.assertNotIn("force_tool_once", shared.app_state)

    def test_set_by_the_nudge_and_cleared_at_the_start(self):
        loop = open("agent/loop.py", encoding="utf-8").read()
        self.assertIn('shared.app_state["force_tool_once"] = True', loop)
        self.assertIn('shared.app_state.pop("force_tool_once", None)', loop)

class TuneSpeedLetsChatsInTests(unittest.TestCase):
    def test_pauses_while_a_chat_runs_and_restores_the_models_settings(self):
        states = iter([True, True, False])
        with patch.object(speed_tune, "_chat_active", side_effect=lambda: next(states)), \
                patch("llama_local.set_model_options") as setopts, patch.object(speed_tune.time, "sleep"):
            self.assertTrue(speed_tune._yield_to_chats("m", {"batch-size": None}))
        setopts.assert_called_once_with({"m": {"batch-size": None}}, backup=False)

    def test_no_pause_without_a_chat(self):
        with patch.object(speed_tune, "_chat_active", return_value=False), patch("llama_local.set_model_options") as s:
            self.assertFalse(speed_tune._yield_to_chats("m", {}))
        s.assert_not_called()

    def test_a_reply_that_just_ended_counts(self):
        class T:
            def __init__(self, cur): self.cur = cur
            def current(self): return self.cur
        with patch("agent.tasks.turns", T({"status": "done", "ended": time.time() - 5})):
            self.assertTrue(speed_tune._chat_active())
        with patch("agent.tasks.turns", T({"status": "done", "ended": time.time() - 60})):
            self.assertFalse(speed_tune._chat_active())

    def test_the_page_says_so(self):
        self.assertIn("10-20 minutes per model; you can keep chatting - it pauses for you.", PAGE)


if __name__ == "__main__":
    unittest.main()
