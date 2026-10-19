"""Streaming (llm_stream.py, stream_hub.py): fragment reassembly, early
cut-off of runaway/looping output, Stop mid-stream, and the relay the UI reads."""
import json
import sys
import unittest

import llm_stream as ls
import stream_hub


def _chunks(*deltas, finish="stop", usage=None):
    for d in deltas:
        yield {"choices": [{"index": 0, "delta": d}]}
    yield {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    if usage:
        yield {"choices": [], "usage": usage}


class _Recorder:
    def __init__(self, gen):
        self.gen, self.closed, self.consumed = gen, False, 0

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.gen)
        self.consumed += 1
        return item

    def close(self):
        self.closed = True


class ReassemblyTests(unittest.TestCase):
    def test_content_reasoning_and_usage(self):
        deltas = []
        res = ls.complete("u", {}, transport=lambda *a: _chunks({"reasoning_content": "think "}, {"reasoning_content": "more"},
                                                                  {"content": "Hel"}, {"content": "lo"},
                                                                  usage={"prompt_tokens": 10, "completion_tokens": 4}),
                          on_delta=lambda ch, text, **k: deltas.append((ch, text)))
        msg = res.choices[0].message
        self.assertEqual(msg.content, "Hello")
        self.assertEqual(msg.reasoning_content, "think more")
        self.assertIsNone(msg.tool_calls)
        self.assertEqual(res.usage.completion_tokens, 4)
        self.assertEqual(res.choices[0].finish_reason, "stop")
        self.assertIn(("content", "Hel"), deltas)
        self.assertIn(("reasoning", "think "), deltas)

    def test_interleaved_tool_call_fragments(self):
        res = ls.complete("u", {}, transport=lambda *a: _chunks(
            {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "write_", "arguments": ""}}]},
            {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "list_dir", "arguments": "{\"d"}}]},
            {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": "{\"x\": \"a\\"}}]},
            {"tool_calls": [{"index": 1, "function": {"arguments": "\": 1}"}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": "nb\"}"}}]}, finish="tool_calls"))
        calls = res.choices[0].message.tool_calls
        self.assertEqual([(c.id, c.function.name) for c in calls], [("a", "write_file"), ("b", "list_dir")])
        self.assertEqual(json.loads(calls[0].function.arguments), {"x": "a\nb"})
        self.assertEqual(json.loads(calls[1].function.arguments), {"d": 1})

    def test_request_asks_for_a_stream_with_usage(self):
        seen = {}

        def transport(url, payload, headers, timeout):
            seen.update(payload=payload, headers=headers)
            return _chunks({"content": "ok"})
        ls.complete("http://x/v1/chat/completions", {"model": "m", "messages": []}, api_key="k", transport=transport)
        self.assertTrue(seen["payload"]["stream"])
        self.assertEqual(seen["payload"]["stream_options"], {"include_usage": True})
        self.assertEqual(seen["headers"]["Authorization"], "Bearer k")


class EarlyStopTests(unittest.TestCase):
    def test_stop_closes_the_stream_immediately(self):
        rec = _Recorder(_chunks(*[{"content": "word "} for _ in range(500)]))
        calls = {"n": 0}

        def should_cancel():
            calls["n"] += 1
            return calls["n"] > 5
        with self.assertRaises(ls.StreamCancelled):
            ls.complete("u", {}, transport=lambda *a: rec, should_cancel=should_cancel)
        self.assertTrue(rec.closed)                                         # connection closed -> server stops
        self.assertLess(rec.consumed, 10)

    def test_runaway_arguments_are_cut_off_early(self):
        rec = _Recorder(_chunks({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "write_file", "arguments": "{\"content\": \""}}]},
                                *[{"tool_calls": [{"index": 0, "function": {"arguments": f"row {i:05d} value {i * 7919 % 10007} ok. " * 3}}]}
                                  for i in range(1000)]))
        res = ls.complete("u", {}, transport=lambda *a: rec, max_arg_chars=5000)
        self.assertEqual(res.choices[0].finish_reason, "runaway")
        self.assertTrue(rec.closed)
        self.assertLess(rec.consumed, 100)                                  # stopped long before the 1000 fragments

    def test_repetition_loop_is_detected(self):
        rec = _Recorder(_chunks(*[{"content": "<div class=x></div>"} for _ in range(1000)]))
        res = ls.complete("u", {}, transport=lambda *a: rec)
        self.assertEqual(res.choices[0].finish_reason, "repetition")
        self.assertLess(rec.consumed, 260)                      # ~4.1k chars in, long before 19k

    def test_legitimate_repetitive_output_is_not_a_loop(self):
        samples = {
            "rule": "-" * 6000,
            "zeros": "[" + "0, " * 3000 + "]",
            "tile map": "[\n" + "  [0,0,0,0,0,0,0,0,0,0],\n" * 300 + "]",
            "table": "| a | b |\n" + "| --- | --- |\n" * 400,
            "prose": " ".join(f"word{i}" for i in range(1500)),
        }
        for label, text in samples.items():
            res = ls.complete("u", {}, transport=lambda *a, t=text: _chunks(*[{"content": t[i:i + 9]} for i in range(0, len(t), 9)]))
            self.assertEqual(res.choices[0].finish_reason, "stop", label)
            self.assertEqual(res.choices[0].message.content, text, label)

    def test_unnamed_calls_and_string_errors_are_not_dropped(self):
        res = ls.complete("u", {}, transport=lambda *a: _chunks(
            {"tool_calls": [{"index": 0, "function": {"arguments": "{\"x\": 1}"}}]}, finish="tool_calls"))
        self.assertEqual(res.choices[0].message.tool_calls[0].function.name, ls.UNNAMED_TOOL)
        with self.assertRaises(ls.StreamHTTPError) as cm:
            ls.complete("u", {}, transport=lambda *a: iter([{"error": "the model crashed"}]))
        self.assertIn("crashed", str(cm.exception))

    @unittest.skipUnless(sys.platform == "win32", "closing a socket from another thread only interrupts a blocked read on Windows")
    def test_stop_before_the_first_token_closes_the_connection(self):
        """The server is still reading the prompt (nothing arrives); Stop must
        still end the request promptly - a watcher closes the connection."""
        import http.server
        import threading
        import time

        class Stall(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(20)                                   # "prompt processing": no bytes for 20 s

            def log_message(self, *a):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stall)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        stop_at = time.monotonic() + 0.6
        started = time.monotonic()
        try:
            with self.assertRaises(ls.StreamCancelled):
                ls.complete(f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions", {"model": "m"},
                            should_cancel=lambda: time.monotonic() > stop_at, timeout=30)
        finally:
            server.shutdown()
        self.assertLess(time.monotonic() - started, 5)

    def test_http_error_body_is_kept_for_capability_detection(self):
        def transport(*a):
            raise ls.StreamHTTPError("Error code: 500 - tools param requires --jinja flag")
        with self.assertRaises(ls.StreamHTTPError) as cm:
            ls.complete("u", {}, transport=transport)
        self.assertIn("--jinja", str(cm.exception))


class HubTests(unittest.TestCase):
    def test_since_merges_fragments_and_times_out(self):
        hub = stream_hub.StreamHub()
        hub.publish("start", call=1)
        for piece in ("Hel", "lo ", "there"):
            hub.publish("delta", call=1, channel="content", text=piece)
        hub.publish("delta", call=1, channel="reasoning", text="hm")
        hub.publish("end", call=1, has_tools=False)
        events, newest = hub.since(0, timeout=0.1)
        self.assertEqual([e["kind"] for e in events], ["start", "delta", "delta", "end"])
        self.assertEqual(events[1]["text"], "Hello there")
        self.assertEqual(newest, hub.seq)
        self.assertEqual(hub.since(newest, timeout=0.05), ([], newest))

    def test_chunk_message_round_trip(self):
        msg = {"content": "Hi", "reasoning_content": "r", "finish_reason": "tool_calls",
               "tool_calls": [{"id": "x", "name": "list_dir", "arguments": "{\"directory\": \"C:\\\\\"}"}]}
        res = ls.complete("u", {}, transport=lambda *a: ls.chunk_message(msg, size=3))
        self.assertEqual(res.choices[0].message.tool_calls[0].function.arguments, msg["tool_calls"][0]["arguments"])
        self.assertEqual(res.choices[0].message.content, "Hi")


if __name__ == "__main__":
    unittest.main()
