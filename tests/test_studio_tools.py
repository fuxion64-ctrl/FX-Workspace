"""Studio work in one call (2026-09-29): requests like "stack 10 parts" get the tools that change things
(intent_router), every effort level sends the same tools and the same write-access line (effort changes
how much it thinks, not what it can do), repeated work is one call - a Luau loop through Roblox's Studio MCP
(studio_mcp fills in its routing arguments) or one roblox_batch through the bridge - and "Allow all like
this" makes the rest of a request one approval."""
import json
import threading
import unittest
from unittest.mock import patch

import main  # noqa: F401  (wires the app together)
import agent.loop as loop
import agent.models as models
import agent.prompts as prompts
import approval_broker
import core.config as config
import core.state as shared
import effort
import intent_router
import orchestrator
import permissions
import prompt_prep
import roblox_adapter
import studio_mcp
import tools.schemas as schemas

ASKED = {"request": "stack 10 parts on each other", "assistant": "Sure - what should I stack them on?"}


class ClassifierTests(unittest.TestCase):
    """Messages from the user's own chats that got no tools - and the model said it was 'a text assistant'."""

    def mode(self, message, previous=None):
        return intent_router.classify_turn(message, previous=previous)["mode"]

    def test_requests_to_act_get_the_tools(self):
        for message in ("stack 10 parts on each other", "Okay, so I want you to stack ten of these on top of each other.",
                        "Yes, I want you to do it.", "No you do run the script", "Um so yeah open it.", "Do it",
                        "noo it says you  have access to it with mcp so do it pls", "go build a house in studio",
                        "place 10 parts on top of each other"):
            self.assertEqual(self.mode(message), "action", message)

    def test_an_answer_to_the_assistants_question_continues_the_request(self):
        got = intent_router.classify_turn("regular part instance in roblox studio", previous=ASKED)
        self.assertEqual((got["mode"], got["follows"]), ("action", ASKED["request"]))
        # ... but not a refusal, thanks, a question, or a new request of its own.
        for message in ("no.", "no thanks", "thanks", "tell me a joke", "why?", "what do you mean"):
            self.assertNotEqual(self.mode(message, ASKED), "action", message)
        # ... and only when the assistant asked something about an action.
        self.assertEqual(self.mode("regular parts", {"request": "what is a part?", "assistant": "Which one?"}), "none")

    def test_questions_stay_questions(self):
        for message, want in (("do you know how to stack parts?", "none"), ("how do I stack parts in studio?", "read"),
                              ("why can't you just do it?", "none"), ("don't delete anything", "none"),
                              ("no need to do that", "none"), ("I don't know how they make it", "none")):
            self.assertEqual(self.mode(message), want, message)


class PromptTests(unittest.TestCase):
    def setUp(self):
        keys = ("conversation", "mode", "design_mode", "turn_policy", "current_user_request", "run_budget",
                "effort_thinking", "effort_native", "turn_persona", "turn_voice", "turn_instructions", "task_context")
        saved = {k: shared.app_state.get(k) for k in keys}
        self.addCleanup(lambda: [shared.app_state.__setitem__(k, v) if v is not None else shared.app_state.pop(k, None)
                                 for k, v in saved.items()])
        shared.app_state.update(conversation=[{"role": "system", "content": "x"}, {"role": "user", "content": "stack 10 parts"}],
                                mode=config.MODE_LOCAL, design_mode=False, current_user_request="stack 10 parts",
                                turn_policy={"mode": "action", "allow_mutation": True, "reason": "test"},
                                turn_persona="", turn_voice=False, turn_instructions="", task_context=None)

    def test_every_effort_level_sends_the_same_tools_and_system_prompt(self):
        seen = set()
        for level in effort.LEVELS:
            profile = effort.PROFILES[level]
            shared.app_state.update(run_budget=effort.RunBudget(profile, local=True), effort_thinking=profile.thinking,
                                    effort_native=profile.native)
            system = models.build_messages()[0]["content"]
            tools = tuple(t["function"]["name"] for t in schemas.tools_for_current_turn(True))
            seen.add((system, tools))
            self.assertIn(prompts.STUDIO_WRITE_ACCESS, system, level)
            self.assertIn("never one call per item", system, level)
        self.assertEqual(len(seen), 1, "effort changed the tools or the system prompt")

    def test_the_write_access_line_survives_the_tool_filter(self):
        for base in (prompts.SYSTEM_PROMPT, prompts.SYSTEM_PROMPT_COMPACT):
            for offered in ([], ["roblox_status"], ["mcp__roblox_studio__execute_luau"]):
                self.assertIn(prompts.STUDIO_WRITE_ACCESS, prompts.for_tools(base, offered, schemas.TOOL_NAMES))
        # The compact prompt no longer tells a model to make one call per step for everything.
        self.assertNotIn("call one appropriate tool, wait for its observed result", prompts.SYSTEM_PROMPT_COMPACT)

    def test_studio_rules_follow_the_tools_that_are_there(self):
        self.assertEqual(prompts.studio_rules(["roblox_status", "web_search"]), "")
        mcp = prompts.studio_rules(["mcp__roblox_studio__execute_luau", "mcp__roblox_studio__insert_asset"])
        self.assertIn("mcp__roblox_studio__execute_luau", mcp)
        self.assertIn("for i = 1, 10 do", mcp)                       # the Luau loop example
        self.assertIn("mcp__roblox_studio__insert_asset", mcp)
        self.assertNotIn("roblox_batch", mcp)
        bridge = prompts.studio_rules(["roblox_batch"])
        self.assertIn("ONE roblox_batch call", bridge)
        self.assertNotIn("execute_luau", bridge)

    def test_turn_header_notes(self):
        header = models.build_messages()[-1]["content"]
        self.assertIn("REPEATED WORK", header)                        # "stack 10 parts" is one call
        shared.app_state["turn_policy"] = {"mode": "none", "allow_mutation": False, "reason": "conversation"}
        self.assertIn("never say you are only a text assistant", models.build_messages()[-1]["content"])
        shared.app_state["turn_policy"] = {"mode": "read", "allow_mutation": False, "reason": "read-only"}
        self.assertIn("never say you can only read", models.build_messages()[-1]["content"])


class PromptPrepTests(unittest.TestCase):
    def test_repeat_style_requests(self):
        for text in ("stack 10 parts", "make ten more parts", "5 rows of trees", "color all the parts red", "rename every script",
                     "place 10 of these", "duplicate it 3 times"):
            self.assertTrue(prompt_prep.repeat_hint(text), text)
        for text in ("wait 10 minutes and open notepad", "give me 3 ideas", "write 50 lines", "all right, open notepad",
                     "open notepad"):
            self.assertEqual(prompt_prep.repeat_hint(text), "", text)

    def test_prompt_prep_rewrites_it_as_one_script_one_loop(self):
        prep = orchestrator.PromptPreparer()
        rule = prompt_prep.repeat_constraint("stack 10 parts")
        self.assertIn("one script with a loop", rule)
        self.assertEqual(prep.heuristic("stack 10 parts, keep them anchored")["constraints"][0], rule)
        self.assertNotIn(rule, prep.heuristic("open notepad")["constraints"])
        state = orchestrator.TaskState("stack 10 parts")
        model = orchestrator.PromptPreparer(lambda *a, **k: {"intent": "stack", "constraints": ["anchored"]})
        self.assertEqual(model.prepare(state, "expand")["constraints"][:2], [rule, "anchored"])
        self.assertIn(rule, prompt_prep.spec_text(model.prepare(state, "expand")))


class BatchTests(unittest.TestCase):
    STACK = [{"action": "create_instance", "class_name": "Folder", "parent_path": "Workspace", "name": "Stack"},
             {"action": "create_instance", "class_name": "Part", "parent_path": "Workspace/Stack", "name": "Part",
              "properties": {"Anchored": True, "Size": [4, 1, 4], "Position": [0, 0.5, 0]}, "repeat": 10,
              "step": {"Position": [0, 1, 0]}}]

    def test_repeat_and_step(self):
        ops, why = roblox_adapter.expand_batch(self.STACK)
        self.assertIsNone(why)
        self.assertEqual(len(ops), 11)
        self.assertEqual(ops[0]["name"], "Stack")
        self.assertEqual([o["name"] for o in ops[1:3]], ["Part1", "Part2"])
        self.assertEqual([o["properties"]["Position"][1] for o in ops[1:]], [0.5 + i for i in range(10)])
        self.assertEqual(ops[10]["properties"]["Size"], [4, 1, 4])

    def test_bad_batches_run_nothing(self):
        for actions, why in (([], "non-empty"), ([{"action": "run_luau"}], "needs action"),
                             ([{"action": "create_instance"}], "class_name"),
                             ([{"action": "create_instance", "class_name": "Part", "repeat": 500}], "repeat"),
                             ([{"action": "set_property", "path": "Workspace/A"}], "path, property and value"),
                             ([{"action": "create_instance", "class_name": "Part", "repeat": 100}] * 3, "more than 200")):
            ops, got = roblox_adapter.expand_batch(actions)
            self.assertEqual(ops, [])
            self.assertIn(why, got)

    def test_runs_in_order_and_stops_at_the_first_failure(self):
        adapter = roblox_adapter.adapter
        sent = []

        def fake_send(command, payload, cancelled=None, timeout=20.0):
            sent.append((command, payload.get("name") or payload.get("path")))
            return {"ok": payload.get("name") != "Part3", "message": "no" if payload.get("name") == "Part3" else "ok"}
        with patch.object(adapter, "_send", fake_send):
            res = adapter.call("roblox_batch", {"actions": self.STACK})
        self.assertFalse(res["ok"])
        self.assertEqual(sent, [("create_instance", "Stack"), ("create_instance", "Part1"), ("create_instance", "Part2"),
                                ("create_instance", "Part3")])
        self.assertIn("Made 3 of 11 changes", res["message"])
        with patch.object(adapter, "_send", lambda *a, **k: {"ok": True, "message": "ok"}):
            self.assertTrue(adapter.call("roblox_batch", {"actions": self.STACK})["ok"])
        # Scripts still only through the checked script tools.
        with patch.object(adapter, "_send", lambda *a, **k: {"ok": True}):
            res = adapter.call("roblox_batch", {"actions": [{"action": "create_instance", "class_name": "Script", "parent_path": "Workspace"}]})
        self.assertFalse(res["ok"])

    def test_one_approval_card_lists_every_change(self):
        preview = roblox_adapter.adapter.preview("roblox_batch", {"actions": self.STACK})
        self.assertIn("11 changes", preview["note"])
        self.assertIn("create Part 'Part10' in Workspace/Stack", preview["text"])
        self.assertEqual(permissions.classify("roblox_batch"), "destructive")
        self.assertIn("roblox_batch", schemas.ROBLOX_TOOL_NAMES)


class _FakeServer:
    def __init__(self, tools):
        self.status, self.tools = "connected", tools


class _FakeHub:
    def __init__(self, studios):
        props = {"code": {"type": "string"}, "studio_id": {"type": "string"},
                 "datamodel_type": {"type": "string", "enum": ["Edit", "Client", "Server"]}}
        self.servers = {"roblox_studio": _FakeServer([{"name": "list_roblox_studios", "inputSchema": {}},
                                                      {"name": "execute_luau", "inputSchema": {"properties": props}}]),
                        "other": _FakeServer([{"name": "execute_luau", "inputSchema": {"properties": props}}])}
        self.studios, self.calls = studios, 0

    def call(self, name, args, cancelled=None):
        self.calls += 1
        return {"ok": True, "message": json.dumps({"studios": self.studios})}


class StudioMcpTests(unittest.TestCase):
    def setUp(self):
        studio_mcp._cache.clear()
        self.addCleanup(studio_mcp._cache.clear)

    def test_routing_arguments_are_filled_in_with_one_studio(self):
        hub = _FakeHub([{"id": "abc", "name": "Scratch Test (placeId: 1)"}])
        self.assertEqual(studio_mcp.studio_servers(hub), ["roblox_studio"])            # "other" lacks list_roblox_studios
        got = studio_mcp.fill_args("mcp__roblox_studio__execute_luau", {"code": "return 1"}, hub)
        self.assertEqual(got, {"code": "return 1", "studio_id": "abc", "datamodel_type": "Edit"})
        # A made-up id is corrected; a valid datamodel is kept; other servers and tools are left alone.
        got = studio_mcp.fill_args("mcp__roblox_studio__execute_luau", {"code": "x", "studio_id": "guess", "datamodel_type": "Server"}, hub)
        self.assertEqual((got["studio_id"], got["datamodel_type"]), ("abc", "Server"))
        self.assertEqual(studio_mcp.fill_args("mcp__other__execute_luau", {"code": "x"}, hub), {"code": "x"})
        self.assertEqual(studio_mcp.fill_args("roblox_batch", {"actions": []}, hub), {"actions": []})
        self.assertIn("'Scratch Test (placeId: 1)' is open", studio_mcp.context_line(hub))
        self.assertEqual(hub.calls, 1)                                                  # cached

    def test_several_studios_are_the_models_choice(self):
        hub = _FakeHub([{"id": "a", "name": "One"}, {"id": "b", "name": "Two"}])
        got = studio_mcp.fill_args("mcp__roblox_studio__execute_luau", {"code": "x"}, hub)
        self.assertNotIn("studio_id", got)
        self.assertIn("One = a", studio_mcp.context_line(hub))
        self.assertEqual(studio_mcp.context_line(_FakeHub([])).split(":")[0], "ROBLOX STUDIO (MCP)")


class FitTests(unittest.TestCase):
    """A local model's catalogue fits its window (2026-09-29: 21.7k tokens of prompt for a 16k window)."""

    def fake_mcp(self):
        # Big signatures: even with their prose shortened, all eight don't fit a 16k window.
        big = {"type": "object", "properties": {f"parameter_{i}": {"type": "string", "description": "x" * 300} for i in range(30)}}
        names = ["user_keyboard_input", "generate_mesh", "subagent", "execute_luau", "multi_edit", "insert_asset",
                 "list_roblox_studios", "search_game_tree"]
        return [{"type": "function", "function": {"name": f"mcp__roblox_studio__{n}", "description": "d" * 600,
                                                  "parameters": big}} for n in names]

    def offered(self, window):
        mcp = self.fake_mcp()
        hub = _FakeHub([{"id": "a", "name": "Scratch Test"}])
        hub.servers["roblox_studio"].tools = [{"name": f["function"]["name"].split("__")[-1]} for f in mcp]
        # The fitting itself, with every built-in tool competing for room (skills off). With skills on the same
        # 16k window has room for all eight - test_skills_leave_room_for_studio_mcp.
        with patch.object(schemas, "_local_context_window", lambda: window), \
                patch.object(schemas.roblox_bridge, "is_connected", lambda: True), \
                patch.object(schemas, "mcp_schemas", lambda: mcp), patch.object(shared, "mcp_hub", hub), \
                patch.dict(shared.app_state["settings"], {"skills_on": self.skills_on, "auto_model_routing": False}), \
                patch.dict(shared.app_state, {"chat_skills": [], "turn_skills": None,       # a fresh chat's skills
                                              # nothing left from earlier tests: no open chat or project, no composer
                                              # tools, a turn from this PC, and (skills off) no request words
                                              "active_chat": None, "active_project": None, "turn_tools": (),
                                              "turn_origin": "pc", "current_user_request":
                                              shared.app_state.get("current_user_request") if self.skills_on else ""}):
            return schemas.tools_for_mode(True, "action")

    skills_on = False

    def test_skills_leave_room_for_studio_mcp(self):
        self.skills_on = True
        saved = shared.app_state.get("current_user_request")
        self.addCleanup(shared.app_state.__setitem__, "current_user_request", saved)
        shared.app_state["current_user_request"] = "stack 10 parts in Studio"
        names = [t["function"]["name"] for t in self.offered(16_384)]
        self.assertEqual(sum(n.startswith("mcp__") for n in names), 8)            # all eight fit now

    def test_small_windows_get_the_core_first(self):
        tools = self.offered(16_384)
        names = [t["function"]["name"] for t in tools]
        mcp = [n.split("__")[-1] for n in names if n.startswith("mcp__")]
        self.assertEqual(mcp[:3], ["execute_luau", "multi_edit", "insert_asset"])
        self.assertLess(len(mcp), 8)                                    # some left out - never the core ones
        self.assertTrue({"execute_luau", "multi_edit", "insert_asset", "search_game_tree"} <= set(mcp),
                        f"MCP fitted: {mcp}; built-in ({schemas._cost([t for t in tools if not t['function']['name'].startswith('mcp__')])} "
                        f"tokens): {sorted(n for n in names if not n.startswith('mcp__'))}")
        self.assertLessEqual(schemas._cost(tools) + schemas._cost(prompts.SYSTEM_PROMPT_COMPACT),
                             int(16_384 * schemas.TOOLS_SHARE))
        # Roblox's Studio MCP replaced the bridge tools it overlaps; roblox_status stays.
        self.assertEqual([n for n in names if n.startswith("roblox_")], ["roblox_status"])
        luau = next(t for t in tools if t["function"]["name"].endswith("__execute_luau"))
        self.assertEqual(len(luau["function"]["parameters"]["properties"]), 30)      # every parameter kept
        self.assertNotIn("description", luau["function"]["parameters"]["properties"]["parameter_0"])

    def test_big_windows_keep_everything(self):
        names = [t["function"]["name"] for t in self.offered(1_000_000)]
        self.assertEqual(sum(n.startswith("mcp__") for n in names), 8)
        self.assertIn("roblox_batch", names)

    def test_a_parameter_called_description_survives(self):
        tool = {"type": "function", "function": {"name": "mcp__x__subagent", "description": "d", "parameters": {
            "type": "object", "properties": {"description": {"type": "string", "description": "what to do"},
                                             "key": {"type": "string", "enum": [str(i) for i in range(40)]}}}}}
        slim = schemas._slim(tool)["function"]["parameters"]["properties"]
        self.assertEqual(slim, {"description": {"type": "string"}, "key": {"type": "string"}})


class ApprovalTests(unittest.TestCase):
    def test_same_tool_calls_stay_one_step(self):
        calls = [(object(), "roblox_create_instance", {"name": f"Part{i}"}) for i in range(10)]
        self.assertTrue(loop._same_tool_batch(calls))
        self.assertFalse(loop._same_tool_batch(calls[:2] + [(object(), "roblox_set_property", {})]))
        self.assertFalse(loop._same_tool_batch([(object(), "update_plan", {})] * 2))
        self.assertFalse(loop._same_tool_batch(calls * 4))                              # over MAX_SAME_TOOL_BATCH

    def test_held_back_calls_are_named_in_the_kept_calls_result(self):
        # 1.4.6 (audit 2.1): a local model's extra calls vanished without a word; it then acted as if they had run.
        import core.state as shared
        from types import SimpleNamespace as NS
        with patch.dict(shared.app_state, conversation=[], held_back_calls=["write_file (Write notes.txt)"]):
            loop._append_tool_result(NS(id="1"), "update_plan", {"ok": True, "message": "Plan saved."})
            loop._append_tool_result(NS(id="2"), "update_plan", {"ok": True, "message": "Again."})
            first, second = shared.app_state["conversation"]
        self.assertIn("NOT RUN YET", first["content"])
        self.assertIn("write_file", first["content"])
        self.assertNotIn("NOT RUN YET", second["content"])                             # said once, on that result

    def test_what_allow_all_may_cover(self):
        s = dict(permissions.DEFAULT_PERMISSION_SETTINGS)
        self.assertTrue(permissions.request_grantable("roblox_create_instance", s))
        self.assertTrue(permissions.request_grantable("mcp__roblox_studio__execute_luau", s))
        self.assertFalse(permissions.request_grantable("roblox_create_instance", s, tainted=True))
        self.assertFalse(permissions.request_grantable("browser_fill", s))              # always asks
        self.assertFalse(permissions.request_grantable("messages_send", s))
        self.assertFalse(permissions.request_grantable("roblox_create_instance", {**s, "permission_mode": "lockdown"}))

    def test_a_grant_lasts_for_the_request(self):
        keys = ("request_grants", "settings")
        saved = {k: shared.app_state.get(k) for k in keys}
        self.addCleanup(lambda: [shared.app_state.__setitem__(k, v) for k, v in saved.items()])
        shared.app_state["settings"] = dict(permissions.DEFAULT_PERMISSION_SETTINGS)
        shared.app_state["request_grants"] = set()
        self.assertEqual(loop._decide("roblox_create_instance", {}, False)[0], "ask")
        shared.app_state["request_grants"] = {"roblox_create_instance"}
        self.assertEqual(loop._decide("roblox_create_instance", {}, False),
                         ("auto", "you allowed it for the rest of this request"))
        self.assertEqual(loop._decide("roblox_create_instance", {}, True)[0], "ask")    # the request read hostile content
        self.assertEqual(loop._decide("roblox_delete_instance", {}, False)[0], "ask")   # another tool asks again

    def test_the_broker_records_the_scope(self):
        broker = approval_broker.ApprovalBroker(native_ask=lambda a, c: True)

        def pend(request_ok=True):
            broker._pending = {"id": "x", "actions": [{"tool": "roblox_create_instance", "request_ok": request_ok}],
                               "context": "", "created_at": 0, "answer": None, "event": threading.Event()}
        pend()
        self.assertTrue(broker.pending()["request_ok"])
        self.assertTrue(broker.answer("x", True, "", "request"))
        self.assertEqual(broker._pending["scope"], "request")
        pend(request_ok=False)
        self.assertFalse(broker.pending()["request_ok"])
        self.assertFalse(broker.answer("x", True, "", "request"))                      # refused, still open
        self.assertTrue(broker.answer("x", True))
        self.assertEqual(broker._pending["scope"], "once")

    def test_grants_are_reset_per_request(self):
        import inspect
        import agent.tasks as tasks
        self.assertIn('shared.app_state["request_grants"] = set()', inspect.getsource(tasks.chat_with_agent))
        self.assertIn('"request_grants"', inspect.getsource(tasks._run_turn_in_chat))


class PlaceNameTests(unittest.TestCase):
    def test_a_quoted_place_is_the_same_place(self):
        # A project saved with "Scratch Test" in quotes made every request say another place was open - and
        # "ask the user before changing anything in Studio".
        import roblox_places
        key = roblox_places.place_key
        self.assertEqual(key('"Scratch Test"'), key("Scratch Test"))
        self.assertEqual(key("Scratch Test (placeId: 102944812720114)"), key("scratch  test"))
        self.assertNotEqual(key("Scratch Test"), key("Obby World"))
        import projects
        self.assertEqual(projects.clean_fields({"roblox_place": ' "Scratch Test" '})["roblox_place"], "Scratch Test")


class PageTests(unittest.TestCase):
    def test_allow_all_button(self):
        import os
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html"), encoding="utf-8") as f:
            html = f.read()
        self.assertIn("p.request_ok ?", html)
        self.assertIn("answer(true, '', 'request')", html)
        self.assertIn("ask.request_ok ?", html)                        # the voice-mode card too
        self.assertIn("roblox_batch: 'Make several changes in Studio'", html)


if __name__ == "__main__":
    unittest.main()
