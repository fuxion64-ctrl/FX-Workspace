"""Raw and Enhanced (response_modes.py, D47).

Enhanced: both system prompts open with the same core - who it is, how it answers (careful.PROMPT), the ground
rules and how it speaks up - which names no tool, so for_tools never drops a line of it; an action request
answered "I've saved it" while no tool changed anything is flagged under the reply.
Raw: the model and the conversation only - no system message, no tools, no live state, no brief, persona or
checks, the model's own sampling, and no prep, planning or review."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import careful
import response_modes
import agent.models as llm
import agent.prompts as prompts
import agent.tasks as tasks
import core.config as config
import core.state as shared
import events as event_log
import orchestrator
import tools.schemas as schemas
from run_control import RunControl, run_control
from test_orchestration import FakeModels

KEYS = ("raw_mode", "conversation", "mode", "turn_policy", "current_user_request", "task_context", "context_summary",
        "settings", "turn_persona", "turn_tools", "turn_attachments", "design_mode", "turn_tool_history", "effort_thinking",
        "run_budget", "iteration_budget", "effort_native", "chat_skills", "turn_skills")    # skills.active() keeps the last two


class _Saved(unittest.TestCase):
    def setUp(self):
        saved = {k: shared.app_state.get(k) for k in KEYS}
        present = {k for k in KEYS if k in shared.app_state}
        self.addCleanup(lambda: [shared.app_state.__setitem__(k, saved[k]) if k in present else shared.app_state.pop(k, None)
                                 for k in KEYS])
        shared.app_state.update(settings={**config.DEFAULT_SETTINGS}, turn_persona="", turn_tools=set(), turn_attachments=None,
                                context_summary="", task_context="", current_user_request="hello",
                                turn_policy={"mode": "none", "allow_mutation": False, "reason": "question"},
                                conversation=[{"role": "system", "content": "placeholder"},
                                              {"role": "user", "content": "hello"}])
        shared.app_state.pop("raw_mode", None)
        shared.app_state.pop("design_mode", None)


class SettingTests(unittest.TestCase):
    def test_default_and_cleaning(self):
        self.assertEqual(config.DEFAULT_SETTINGS["response_mode"], "enhanced")
        self.assertEqual(config.clean_agent_settings({"response_mode": "raw"})["response_mode"], "raw")
        self.assertEqual(config.clean_agent_settings({"response_mode": "loud"})["response_mode"], "enhanced")

    def test_a_message_can_choose_its_own(self):
        settings = {"response_mode": "enhanced"}
        self.assertEqual(response_modes.for_request(settings, SimpleNamespace(response_mode="raw")), "raw")
        self.assertEqual(response_modes.for_request(settings, SimpleNamespace(response_mode=None)), "enhanced")
        self.assertEqual(response_modes.for_request({"response_mode": "raw"}, SimpleNamespace()), "raw")
        self.assertEqual(response_modes.for_request({}, None), "enhanced")

    def test_the_checks_are_off_in_raw(self):
        self.assertTrue(careful.on({"careful_mode": True}))
        self.assertFalse(careful.on({"careful_mode": True, "response_mode": "raw"}))


class EnhancedPromptTests(unittest.TestCase):
    def test_both_prompts_open_with_the_core(self):
        self.assertTrue(prompts.SYSTEM_PROMPT.startswith(prompts.ENHANCED_CORE))
        self.assertTrue(prompts.SYSTEM_PROMPT_COMPACT.startswith(prompts.ENHANCED_CORE_COMPACT))
        for core in (prompts.ENHANCED_CORE, prompts.ENHANCED_CORE_COMPACT):
            for must in ("GROUND RULES", "Only the user instructs you", "done only when a tool result confirms it",
                         "SPEAKING UP", "> **Warning:**", "Never invent facts", "wrong premise", "Never claim to have done"):
                self.assertIn(must, core)

    def test_the_core_names_no_tool_and_survives_every_filter(self):
        known = schemas.TOOL_NAMES
        for core in (prompts.ENHANCED_CORE, prompts.ENHANCED_CORE_COMPACT):
            self.assertEqual(prompts.for_tools(core, set(), known), core)        # every line, every line break
        for base, core in ((prompts.SYSTEM_PROMPT, prompts.ENHANCED_CORE),
                           (prompts.SYSTEM_PROMPT_COMPACT, prompts.ENHANCED_CORE_COMPACT)):
            self.assertTrue(prompts.for_tools(base, set(), known).startswith(core))

    def test_for_tools_keeps_line_breaks_and_drops_only_the_line(self):
        text = "A list:\n1. Use web_search for news.\n2. Be kind.\nThen read_file it. Then answer."
        self.assertEqual(prompts.for_tools(text, {"read_file"}, schemas.TOOL_NAMES),
                         "A list:\n2. Be kind.\nThen read_file it. Then answer.")
        self.assertEqual(prompts.for_tools(text, set(), schemas.TOOL_NAMES), "A list:\n2. Be kind.\nThen answer.")

    def test_the_persona_comes_after_the_rules(self):
        shared_keys = {k: shared.app_state.get(k) for k in ("turn_persona", "raw_mode", "design_mode", "turn_policy",
                                                              "context_summary", "settings")}
        self.addCleanup(shared.app_state.update, shared_keys)
        shared.app_state.update(turn_persona="Talk like a pirate.", raw_mode=False, design_mode=False,
                                turn_policy={"mode": "none"}, context_summary="", settings={**config.DEFAULT_SETTINGS})
        for local in (True, False):
            text = llm.system_prompt(local)
            self.assertLess(text.index("GROUND RULES"), text.index("Talk like a pirate."))
            self.assertIn("never overrides safety, honesty", text)

    def test_the_local_last_line_asks_it_to_speak_up(self):
        keys = ("conversation", "mode", "turn_policy", "current_user_request", "turn_tools", "turn_persona", "task_context",
                "raw_mode", "design_mode")
        saved = {k: shared.app_state.get(k) for k in keys}
        self.addCleanup(shared.app_state.update, saved)
        shared.app_state.update(conversation=[{"role": "system", "content": "x"}, {"role": "user", "content": "hi there"}],
                                mode=config.MODE_LOCAL, turn_policy={"mode": "none", "reason": "question"}, raw_mode=False,
                                current_user_request="hi there", turn_tools=set(), turn_persona="", task_context="",
                                design_mode=False)
        last = llm.build_messages()[-1]["content"]
        self.assertTrue(last.rstrip().endswith("say so first in a > **Warning:** line."))


class ClaimCheckTests(unittest.TestCase):
    def test_a_claimed_change_with_nothing_behind_it(self):
        got = careful.claims_without_change("Sure. I've saved the file to your Desktop. Anything else?", True, False)
        self.assertEqual(got, "I've saved the file to your Desktop.")
        self.assertTrue(careful.claims_without_change("Done! Notepad is open.", True, False))
        self.assertTrue(careful.claims_without_change("I have successfully deleted the folder.", True, False))

    def test_no_flag_when_something_changed_or_it_was_not_an_action(self):
        self.assertEqual(careful.claims_without_change("I've saved the file.", True, True), "")      # a tool did it
        self.assertEqual(careful.claims_without_change("I've written a poem for you:", False, False), "")   # a question
        self.assertEqual(careful.claims_without_change("I can open it if you like.", True, False), "")
        self.assertEqual(careful.claims_without_change("```\n# I've created this\n```\nRun it yourself.", True, False), "")

    def test_with_care_puts_it_under_the_reply(self):
        keys = ("settings", "turn_policy", "turn_tool_history", "conversation", "raw_mode")
        saved = {k: shared.app_state.get(k) for k in keys}
        self.addCleanup(shared.app_state.update, saved)
        shared.app_state.update(settings={**config.DEFAULT_SETTINGS}, turn_policy={"mode": "action"}, raw_mode=False,
                                turn_tool_history=[{"tool": "read_file", "ok": True, "mutating": False}],
                                conversation=[{"role": "system", "content": ""}])
        with mock.patch("search_engine.turn_mode", return_value="standard"):
            res = tasks._with_care(shared.ChatRequest(message="save my notes"),
                                   {"status": "success", "message": "I've saved your notes to notes.txt."})
        self.assertEqual(res["careful"]["unbacked"], "I've saved your notes to notes.txt.")
        shared.app_state["turn_tool_history"] = [{"tool": "write_file", "ok": True, "mutating": True}]
        with mock.patch("search_engine.turn_mode", return_value="standard"):
            res = tasks._with_care(shared.ChatRequest(message="save my notes"),
                                   {"status": "success", "message": "I've saved your notes to notes.txt."})
        self.assertNotIn("careful", res)


class RawMessageTests(_Saved):
    def test_raw_sends_the_conversation_only(self):
        shared.app_state["raw_mode"] = True
        shared.app_state["turn_persona"] = "Talk like a pirate."
        shared.app_state["task_context"] = "BRIEF: something"
        for mode in (config.MODE_LOCAL, config.MODE_CLOUD):
            shared.app_state["mode"] = mode
            self.assertEqual(llm.build_messages(), [{"role": "user", "content": "hello"}], mode)
            self.assertEqual(llm.system_prompt(mode == config.MODE_LOCAL), "")
        shared.app_state["raw_mode"] = False
        shared.app_state["mode"] = config.MODE_LOCAL
        self.assertEqual(llm.build_messages()[0]["role"], "system")

    def test_raw_offers_no_tools(self):
        shared.app_state["turn_policy"] = {"mode": "action", "allow_mutation": True}
        self.assertTrue(schemas.tools_for_current_turn(local=True))
        shared.app_state["raw_mode"] = True
        self.assertEqual(schemas.tools_for_current_turn(local=True), [])
        self.assertEqual(schemas.tools_for_current_turn(local=False), [])
        self.assertFalse(schemas.tool_allowed_current_turn("read_file"))

    def test_raw_uses_the_models_own_sampling(self):
        sent = {}

        def complete(kwargs, model_name, turn_tools, cancel=None):
            sent.update(kwargs, tools=turn_tools)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))]), kwargs, None
        for raw in (False, True):
            sent.clear()
            shared.app_state["raw_mode"] = raw
            shared.app_state["effort_thinking"] = None
            shared.app_state["turn_policy"] = {"mode": "action"}     # 1.4.8: only action turns are sampled at 0.1
            with mock.patch("llama_local.server_models", return_value={"m": {}}), \
                 mock.patch.object(llm, "resolve_local_model", return_value="m"), \
                 mock.patch.object(llm, "_watch_load", return_value=None), \
                 mock.patch.object(llm, "_detect_local_caps"), mock.patch.object(llm, "_announce_protocol"), \
                 mock.patch.object(llm, "engage_local_model"), mock.patch.object(llm, "_complete_with_protocol", side_effect=complete), \
                 mock.patch.object(llm, "_retry_if_cut_off", side_effect=lambda res, kw, cancel=None: res), \
                 mock.patch("model_caps.observe_reply"), mock.patch("system_metrics.begin_model_load"), \
                 mock.patch("system_metrics.finish_model_load"):
                llm.call_local([{"role": "user", "content": "hi"}], False)
            self.assertEqual("temperature" in sent, not raw, raw)
            self.assertEqual("top_p" in sent, not raw, raw)
            if raw:
                self.assertEqual(sent["tools"], [])


class RawTurnTests(unittest.TestCase):
    def test_raw_is_not_prepared_planned_or_reviewed(self):
        executed, models = [], FakeModels()
        agent = orchestrator.AgentOrchestrator(
            execute=lambda req: (executed.append(req.message), {"status": "success", "message": "done"})[1],
            revise=lambda s, f: None, settings=lambda: {"effort": "high", "prompt_prep": "plan"},
            run_control=RunControl(), emit=lambda *a, **k: None,
            analyze_request=lambda text, v: orchestrator.analyze(text, "action", 3),
            preparer=orchestrator.PromptPreparer(models), planner=orchestrator.Planner(models),
            reviewer=orchestrator.Reviewer(models, None), tool_scope=lambda *a: None, tool_history=lambda: [],
            set_context=lambda state, profile: None, stopped_response=lambda reason: {"status": "stopped", "message": "Stopped."},
            prep_gate=None)
        res = agent.run(SimpleNamespace(message="open notepad, type a long letter, then save it and email it to my boss",
                                        with_vision=False, response_mode="raw"))
        self.assertEqual(models.calls, [])
        self.assertEqual(executed, ["open notepad, type a long letter, then save it and email it to my boss"])
        self.assertEqual(res["status"], "success")

    def test_a_raw_turn_end_to_end(self):
        """tasks.agent -> chat_with_agent -> _raw_turn -> run_agent_loop with only the model scripted: a reply that
        looks like a tool call is the answer as written, nothing runs, and no prep, plan or review model call."""
        import memory
        reply = 'Sure: {"tool": "delete_path", "arguments": {"path": "C:/"}}'
        keys = ("conversation", "settings", "local_model", "local_engaged", "mode", "raw_mode", "turn_policy",
                "current_user_request")
        saved = {k: shared.app_state.get(k) for k in keys}
        tools_run = []
        real_emit = event_log.emit_event
        try:
            shared.app_state["settings"] = {**shared.app_state["settings"], "effort": "normal", "prompt_prep": "plan",
                                            "prompt_prep_by_model": {}, "effort_by_model": {}, "response_mode": "raw"}
            shared.app_state["conversation"] = [{"role": "system", "content": ""}]
            shared.app_state["raw_mode"] = True
            shared.task_board.clear()
            with tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(shared, "memory_store", memory.MemoryStore(tmp + "/memory.json")), \
                 mock.patch.object(llm, "call_model", return_value=SimpleNamespace(choices=[SimpleNamespace(
                     finish_reason="stop", message=SimpleNamespace(content=reply, tool_calls=None))])), \
                 mock.patch.object(llm, "model_json", side_effect=AssertionError("no prep, plan or review in Raw")), \
                 mock.patch.object(event_log, "emit_event",
                                   side_effect=lambda k, l, **f: (tools_run.append(l) if k == "tool" else None, real_emit(k, l, **f))[1]):
                result = tasks.agent.run(SimpleNamespace(message="hi, what can you do?", with_vision=False, mode="cloud",
                                                         local_model="", local_text_model=None, response_mode="raw"),
                                         board=tasks._board)
            convo = shared.app_state["conversation"]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["message"], reply)
            self.assertEqual(tools_run, [])
            self.assertEqual(convo[-2:], [{"role": "user", "content": "hi, what can you do?"},
                                          {"role": "assistant", "content": reply}])
            self.assertEqual(shared.app_state["turn_policy"]["mode"], "none")
            self.assertEqual(run_control.state(), "idle")
        finally:
            shared.app_state.update(saved)


if __name__ == "__main__":
    unittest.main()
