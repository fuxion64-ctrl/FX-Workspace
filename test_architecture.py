"""Tests for the architecture pieces split out of main.py and the bugs they fix:
history fitting, reasoning-model output, background turns, the local model
catalog, and local-only API access."""
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import history
import llama_local
import model_output
from turns import TurnManager


def _turn(user, tool_outputs=(), tool_size=0):
    msgs = [{"role": "user", "content": user}]
    for i, out in enumerate(tool_outputs):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read_file", "content": out * (tool_size or 1)})
    return msgs


class HistoryTests(unittest.TestCase):
    def test_long_single_task_is_compacted_not_left_over_budget(self):
        # One user turn, many large tool results: no turn boundary to cut at.
        convo = [{"role": "system", "content": "sys"}] + _turn("do the task", ["x"] * 10, tool_size=4000)
        limit = 6000
        self.assertGreater(history.total(convo, True), limit)
        fitted, notes = history.fit(convo, limit, as_tokens=True)
        self.assertLessEqual(history.total(fitted, True), limit)
        self.assertTrue(any("Shortened" in n for n in notes))
        # Structure intact: every tool result still follows its call.
        self.assertEqual([m["role"] for m in fitted], [m["role"] for m in convo])
        # The newest results are untouched - the model is acting on them.
        self.assertEqual(fitted[-1]["content"], convo[-1]["content"])

    def test_whole_turns_are_dropped_before_compacting(self):
        convo = ([{"role": "system", "content": "sys"}] + _turn("old", ["y"], 20000)
                 + _turn("mid", ["z"], 100) + _turn("new", ["w"], 100))
        fitted, notes = history.fit(convo, 2000, as_tokens=True, min_kept_turns=2)
        self.assertEqual(fitted[1]["content"], "mid")
        self.assertTrue(any("Trimmed" in n for n in notes))

    def test_trim_never_cuts_between_tool_call_and_result(self):
        convo = [{"role": "system", "content": "s"}] + _turn("a", ["1", "2"], 3000) + _turn("b", ["3"], 10) + _turn("c", ["4"], 10)
        fitted, _, _ = history.trim_turns(convo, 500, True, min_kept_turns=2)
        self.assertEqual(fitted[1]["role"], "user")

    def test_image_priced_as_constant(self):
        img = {"role": "user", "content": [{"type": "text", "text": "hi"},
                                           {"type": "image_url", "image_url": {"url": "data:" + "A" * 400000}}]}
        self.assertLess(history.approx_tokens(img), 2000)


class ReasoningTests(unittest.TestCase):
    def test_tool_call_inside_think_is_not_an_answer(self):
        answer, reasoning = model_output.split_reasoning(
            '<think>Maybe {"name": "delete_path", "arguments": {"path": "C:/"}}? No.</think>Here is the summary.')
        self.assertEqual(answer, "Here is the summary.")
        self.assertIn("delete_path", reasoning)

    def test_unclosed_think_is_all_reasoning(self):
        answer, reasoning = model_output.split_reasoning("Sure. <think>still pondering {")
        self.assertEqual(answer, "Sure.")
        self.assertTrue(reasoning.startswith("still pondering"))

    def test_close_only_template(self):
        answer, reasoning = model_output.split_reasoning("pondering...</think>The answer is 4.")
        self.assertEqual((answer, reasoning), ("The answer is 4.", "pondering..."))

    def test_plain_text_untouched(self):
        self.assertEqual(model_output.split_reasoning("Just text {a:1}"), ("Just text {a:1}", ""))

    def test_truncated_json_detection(self):
        self.assertTrue(model_output.looks_truncated_json('{"name": "write_file", "arguments": {"content": "abc'))
        self.assertFalse(model_output.looks_truncated_json('{"name": "x", "arguments": {"a": "}"}}'))


class TurnManagerTests(unittest.TestCase):
    def test_second_start_is_refused_while_running(self):
        gate = threading.Event()
        tm = TurnManager(runner=lambda req: (gate.wait(2), {"status": "success", "message": req})[1])
        first = tm.start("one")
        self.assertIsNotNone(first)
        self.assertIsNone(tm.start("two"))
        gate.set()
        done = tm.wait(first["id"], 2)
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["result"]["message"], "one")
        self.assertIsNotNone(tm.start("three"))          # free again

    def test_runner_exception_becomes_error_result(self):
        tm = TurnManager(runner=lambda req: 1 / 0)
        t = tm.start("x")
        self.assertEqual(tm.wait(t["id"], 2)["result"]["status"], "error")
        self.assertFalse(tm.running())

    def test_wait_returns_running_after_timeout_and_current_survives(self):
        gate = threading.Event()
        tm = TurnManager(runner=lambda req: (gate.wait(2), {"status": "success"})[1])
        t = tm.start("x", {"event_seq": 7})
        self.assertEqual(tm.wait(t["id"], 0.05)["status"], "running")
        self.assertEqual(tm.current()["event_seq"], 7)   # a reloaded page can reattach
        gate.set()
        tm.wait(t["id"], 2)
        self.assertIsNone(tm.wait("nope", 0))


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        for name, size in (("big.gguf", 5000), ("small.gguf", 100), ("mid.gguf", 1000), ("proj.gguf", 50)):
            with open(os.path.join(d, name), "wb") as f:
                f.write(b"0" * size)
        self.ini = os.path.join(d, "models.ini")
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write(f"[*]\nctx-size = 8192\n\n[Huihui-Qwen3.5-9B]\nmodel = {d}\\big.gguf\nmmproj = {d}\\proj.gguf\nctx-size = 65536\n\n"
                    f"[Tiny-Text]\nmodel = {d}\\small.gguf\n\n[Mid-Text]\nmodel = {d}\\mid.gguf\n#mmproj = {d}\\proj.gguf\n")
        self.env = patch.dict(os.environ, {"MODELS_INI": self.ini})
        self.env.start()
        llama_local._ini_cache.update(key=None, configs=None)

    def tearDown(self):
        self.env.stop()
        llama_local._ini_cache.update(key=None, configs=None)
        self.dir.cleanup()

    def test_mmproj_makes_a_model_vision_capable(self):
        self.assertFalse(llama_local.looks_like_vision("Huihui-Qwen3.5-9B"))   # the old, name-only check
        self.assertTrue(llama_local.is_vision_model("Huihui-Qwen3.5-9B"))
        self.assertFalse(llama_local.is_vision_model("Mid-Text"))             # commented-out mmproj
        self.assertTrue(llama_local.is_vision_model("Some-LLaVA-Unconfigured"))

    def test_context_limit_per_model_and_cache_refresh(self):
        self.assertEqual(llama_local.model_context_limit("Huihui-Qwen3.5-9B"), 65536)
        llama_local.update_model_config("Huihui-Qwen3.5-9B", ctx_size=4096)
        self.assertEqual(llama_local.model_context_limit("Huihui-Qwen3.5-9B"), 4096)

    @unittest.skipUnless(sys.platform == "win32", "models.ini uses Windows paths")
    def test_eviction_goes_through_the_smallest_model(self):
        names = ["Huihui-Qwen3.5-9B", "Mid-Text", "Tiny-Text", "Unknown-Model"]
        self.assertEqual(llama_local.cheapest_other_model("Mid-Text", names), "Tiny-Text")
        self.assertEqual(llama_local.cheapest_other_model("Tiny-Text", names), "Mid-Text")

    def test_down_server_is_negatively_cached(self):
        calls = []

        def refuse(*a, **k):
            calls.append(1)
            raise ConnectionError("refused")
        llama_local.invalidate_cache()
        with patch.object(llama_local.requests, "get", side_effect=refuse):
            self.assertIsNone(llama_local.server_models())
            self.assertFalse(llama_local.server_is_up())
            self.assertFalse(llama_local.server_is_up())
        self.assertEqual(len(calls), 1)                  # not once per UI poll
        llama_local.invalidate_cache()


class LocalOnlyAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main
        cls.client = TestClient(main.app, base_url="http://127.0.0.1:8000")

    def test_cross_site_post_is_refused(self):
        r = self.client.post("/api/run/stop", headers={"Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 403)

    def test_same_origin_and_no_origin_posts_pass(self):
        self.assertEqual(self.client.post("/api/run/stop", headers={"Origin": "http://127.0.0.1:8000"}).status_code, 200)
        self.assertEqual(self.client.post("/api/run/stop").status_code, 200)   # Roblox plugin: no Origin

    def test_other_local_pages_are_refused(self):
        """A page on another localhost port is "same-site", not this app: refused (Phase 3 review)."""
        c = self.client
        self.assertEqual(c.post("/api/run/stop", headers={"Origin": "http://localhost:3000"}).status_code, 403)
        self.assertEqual(c.post("/api/run/stop", headers={"Origin": "http://127.0.0.1:8001"}).status_code, 403)
        self.assertEqual(c.post("/api/undo/last", headers={"Sec-Fetch-Site": "same-site"}).status_code, 403)
        self.assertEqual(c.get("/api/state", headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)
        self.assertEqual(c.get("/api/settings", headers={"Sec-Fetch-Site": "same-origin"}).status_code, 200)
        # The page itself may still be opened from a link.
        self.assertEqual(c.get("/", headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}).status_code, 200)

    def test_the_studio_bridge_answers_studio_only(self):
        """Studio's HttpService sends no Origin / Sec-Fetch-Site; a web page always does."""
        c = self.client
        self.assertEqual(c.get("/roblox/poll", headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)
        self.assertEqual(c.post("/roblox/context", json={}, headers={"Origin": "http://127.0.0.1:8000"}).status_code, 403)

    def test_dns_rebinding_host_is_refused(self):
        r = self.client.get("/api/settings", headers={"Host": "attacker.example"})
        self.assertEqual(r.status_code, 400)

    def test_unknown_turn(self):
        self.assertEqual(self.client.get("/api/turn/doesnotexist").json()["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
