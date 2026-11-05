"""1.4.7 reliability - the core runtime's failure paths, each with its fallback (docs/plans/1.4.7.md):
a failed model load says its real cause (from the router's own log), Run on RAM too really works, llama-server is
started again when it dies or hangs, the parts move off ports another program holds, and a picture is the answer
on its own - shown at once, never searched for. Temporary folders and dummy listeners only; nothing real is touched."""
import json
import os
import re
import shutil
import socket
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import llama_builds
import llama_local
import load_failure
import ports

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = open(os.path.join(HERE, "index.html"), encoding="utf-8").read()

# What the router wrote on this PC on 4 Nov 2026 (data/logs/llama-server.log of the installed app), trimmed.
LOG_MISSING = """\
3.48.669.248 I srv          load: spawning server instance with name=Gemma-4-E4B-Tools on port 59056
[59056] 0.01.147.936 I srv    load_model: loading model 'E:\\x\\gemma-4-E4B_q4_0-it.gguf'
[59056] 0.01.148.466 E gguf_init_from_file: failed to open GGUF file 'E:\\x\\gemma-4-E4B_q4_0-it.gguf' (No such file or directory)
[59056] 0.01.216.574 E common_fit_params: encountered an error while trying to fit params to free device memory: failed to load model
[59056] 0.01.258.174 E srv  llama_server: exiting due to model loading error
3.50.274.217 I srv           run: instance name=Gemma-4-E4B-Tools exited with status 1
"""
LOG_RAM = """\
0.38.000.000 I srv          load: spawning server instance with name=Big-35B on port 55036
[55036] 0.00.870.521 W model has unused tensor blk.40.attn_norm.weight (size = 8192 bytes) -- ignoring
[55036] 0.02.899.956 E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 18745196544
[55036] 0.02.899.967 E ggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA_Host buffer of size 18745196544
[55036] 0.03.053.880 E llama_model_load: error loading model: unable to allocate CUDA_Host buffer
[55036] 0.03.054.987 E srv  llama_server: exiting due to model loading error
0.39.129.133 I srv           run: instance name=Big-35B exited with status 1
"""
LOG_VRAM = """\
1.00.000.000 I srv          load: spawning server instance with name=M on port 60001
[60001] 0.00.500.000 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000.00 MiB on device 0: cudaMalloc failed: out of memory
[60001] 0.00.600.000 E llama_model_load: error loading model: unable to allocate CUDA0 buffer
1.00.900.000 I srv           run: instance name=M exited with status 1
"""
ROUTER_500 = 'Error code: 500 - {"error":{"code":500,"message":"model name=M failed to load","type":"server_error"}}'


def page_kind(text: str) -> str:
    """The title index.html's errorInfo gives this text: the first ERROR_KINDS pattern that matches."""
    block = PAGE[PAGE.index("const ERROR_KINDS = ["):PAGE.index("function errorInfo(text)")]
    for m in re.finditer(r"\[/(.+?)/(i?), '([^']*)'", block):
        if re.search(m.group(1).replace("\\/", "/"), text, re.I if m.group(2) else 0):
            return m.group(3)
    return ""


class TempIni(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.gguf = os.path.join(self.tmp, "m.gguf")
        with open(self.gguf, "wb") as f:
            f.write(b"GGUF")
        self.ini = os.path.join(self.tmp, "models.ini")
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write(f"[*]\nngl = auto\nctx-size = 65536\n\n[M]\nmodel = {self.gguf}\nngl = 99\nctx-size = 32768\n\n"
                    f"[Gone]\nmodel = {os.path.join(self.tmp, 'moved-away.gguf')}\n\n[Big-35B]\nmodel = {self.gguf}\n")
        p = patch.dict(os.environ, {"MODELS_INI": self.ini})
        p.start()
        self.addCleanup(p.stop)
        llama_local._ini_cache.update(key=None, configs=None)
        self.addCleanup(llama_local._ini_cache.update, key=None, configs=None)
        self.log = os.path.join(self.tmp, "llama-server.log")

    def write_log(self, text):
        with open(self.log, "w", encoding="utf-8") as f:
            f.write("0.00.000.000 I srv  main: starting router\n" + text)


class LoadFailureTests(TempIni):
    """Report 1: a generic 500 is never "out of GPU memory" without evidence."""

    def test_a_moved_model_file_is_said_so_not_vram(self):
        self.write_log(LOG_MISSING)
        d = load_failure.diagnose("Gone", ROUTER_500, self.log)
        self.assertEqual(d["kind"], "missing_file")
        msg = load_failure.message("Gone", d)
        self.assertIn("its model file isn't there", msg)
        self.assertEqual(page_kind(msg), "The model’s file is missing")
        self.assertNotIn("ramsplit", PAGE[PAGE.index("'The model’s file is missing'"):][:200])

    def test_the_missing_file_line_in_the_log_alone_is_enough(self):
        # The preset's file exists now (it came back), but the log of the failed load says it wasn't there.
        self.write_log(LOG_MISSING.replace("Gemma-4-E4B-Tools", "M"))
        self.assertEqual(load_failure.diagnose("M", ROUTER_500, self.log)["kind"], "missing_file")

    def test_system_ram_is_not_gpu_memory(self):
        self.write_log(LOG_RAM)
        d = load_failure.diagnose("Big-35B", ROUTER_500, self.log)
        self.assertEqual(d["kind"], "ram")
        self.assertEqual(d["bytes"], 18745196544)
        msg = load_failure.message("Big-35B", d)
        self.assertIn("17.5 GB", msg)
        self.assertEqual(page_kind(msg), "Not enough free RAM for this model")

    def test_real_gpu_memory_evidence_offers_run_on_ram(self):
        self.write_log(LOG_VRAM)
        d = load_failure.diagnose("M", ROUTER_500, self.log)
        self.assertEqual(d["kind"], "vram")
        self.assertEqual(page_kind(load_failure.message("M", d)), "Not enough GPU memory for this model")

    def test_no_evidence_is_no_guess(self):
        self.write_log("")
        d = load_failure.diagnose("M", ROUTER_500, self.log)
        self.assertEqual(d["kind"], "unknown")
        msg = load_failure.message("M", d)
        self.assertEqual(page_kind(msg), "The model couldn’t load")
        acts = PAGE[PAGE.index("A load that failed for a reason nobody could read"):][:400]
        self.assertNotIn("ramsplit", acts)                         # Run on RAM too only with evidence
        self.assertNotIn("doesn’t fit in the GPU", acts)

    def test_only_the_last_load_of_that_model_counts(self):
        self.write_log(LOG_VRAM + LOG_RAM.replace("Big-35B", "M").replace("55036", "60002"))
        self.assertEqual(load_failure.diagnose("M", ROUTER_500, self.log)["kind"], "ram")

    def test_explain_ignores_errors_that_are_not_load_failures(self):
        self.assertIsNone(load_failure.explain("M", "Error code: 400 - bad request"))
        self.assertTrue(load_failure.is_load_failure(ROUTER_500))

    def test_force_load_keeps_the_routers_words(self):
        resp = Mock(status_code=500, text='{"error":{"message":"model name=M failed to load"}}')
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        with patch("requests.post", return_value=resp), patch.object(llama_builds, "ensure_for"):
            with self.assertRaises((RuntimeError, requests.HTTPError)) as cm:
                llama_local.force_load("M")
        self.assertTrue(load_failure.is_load_failure(str(cm.exception)))


class RamSplitTests(TempIni):
    """Report 2: Run on RAM too said "That didn't work" every time - set_model_options returns a backup path."""

    def call(self, model, meta=None):
        from api import server
        with patch.object(server.tasks.turns, "running", return_value=False), patch("llama_local.unload_all"), \
                patch("diagnostics.meta_for", return_value=meta or {"arch": "llama"}):
            return server.ram_split(server.RamSplitRequest(model=model))

    def test_it_writes_the_split_and_says_ok(self):
        r = self.call("M")
        self.assertTrue(r["ok"], r)
        self.assertEqual(llama_local.model_own_options("M")["ngl"], "auto")
        self.assertEqual(llama_local.model_own_options("M")["fit-target"], "256")
        self.assertTrue(any(n.startswith("models.ini.bak-") for n in os.listdir(self.tmp)))   # the old file is kept

    def test_a_missing_file_is_not_offered_ram(self):
        r = self.call("Gone")
        self.assertFalse(r["ok"])
        self.assertIn("model file isn't there", r["message"])

    def test_a_hybrid_model_stays_whole_with_a_smaller_window(self):
        r = self.call("M", meta={"arch": "qwen35", "full_attention_interval": 4})
        self.assertTrue(r["ok"], r)
        own = llama_local.model_own_options("M")
        self.assertEqual((own["ngl"], own["ctx-size"], own["cache-type-k"]), ("99", "16384", "q8_0"))


class RouterRecoveryTests(unittest.TestCase):
    """Report 3: a router that died or hung is started again, and the message never mentions run.bat."""

    def setUp(self):
        llama_builds._watch.update(restarts=[], last="", seen_up=False, fails=0)
        self.addCleanup(llama_builds._watch.update, restarts=[], last="", seen_up=False, fails=0)
        self.port = llama_local.LLAMA_PORT
        self.addCleanup(llama_local.set_port, self.port)
        p = patch.dict(os.environ, {"AUTOSTART_LLAMA": "1"})            # tests/__init__ turns it off for the rest
        p.start()
        self.addCleanup(p.stop)

    def test_up_means_nothing_to_do(self):
        with patch.object(llama_builds, "_answers", return_value=True), patch.object(llama_builds, "_start") as start:
            r = llama_builds.ensure_running()
        self.assertEqual((r["ok"], r["restarted"]), (True, False))
        start.assert_not_called()

    def test_down_is_started_again_with_the_same_kind_of_command(self):
        answers = iter([False, True])
        with patch.object(llama_builds, "_answers", side_effect=lambda: next(answers)), \
                patch.object(llama_builds, "_restart_exe", return_value="C:\\llama\\llama-server.exe"), \
                patch("ports.holder", return_value=None), patch("ports.is_free", return_value=True), \
                patch.object(llama_builds, "_start") as start, patch.object(llama_builds, "_wait_ready", return_value=True):
            r = llama_builds.ensure_running("test")
        self.assertTrue(r["ok"] and r["restarted"], r)
        args = start.call_args[0][1]
        self.assertIn("--models-preset", args)
        self.assertEqual(args[args.index("--port") + 1], str(self.port))
        self.assertIn("started it again", llama_builds.status_line())

    def test_a_hung_router_of_ours_is_stopped_first(self):
        mine = {"pid": 4242, "name": "llama-server.exe", "exe": "", "cmdline": [], "cwd": ""}
        with patch.object(llama_builds, "_answers", side_effect=[False, True]), \
                patch.object(llama_builds, "_restart_exe", return_value="x.exe"), \
                patch("ports.holder", return_value=mine), patch.object(llama_builds, "_router_at", return_value=mine), \
                patch("ports.stop_tree", return_value=True) as stop, patch.object(llama_builds, "_start"), \
                patch.object(llama_builds, "_wait_ready", return_value=True):
            self.assertTrue(llama_builds.ensure_running()["ok"])
        stop.assert_called_once_with(4242, wait_port=self.port)

    def test_a_foreign_program_on_the_port_moves_the_router(self):
        other = {"pid": 77, "name": "other.exe", "exe": "C:\\Other\\other.exe", "cmdline": [], "cwd": "C:\\Other"}
        with patch.object(llama_builds, "_answers", side_effect=[False, True]), \
                patch.object(llama_builds, "_restart_exe", return_value="x.exe"), \
                patch("ports.holder", return_value=other), patch.object(llama_builds, "_router_at", return_value=None), \
                patch("ports.pick", return_value=self.port + 7), patch("ports.note"), \
                patch.object(llama_builds, "_start") as start, patch.object(llama_builds, "_wait_ready", return_value=True):
            r = llama_builds.ensure_running()
        self.assertTrue(r["ok"])
        self.assertEqual(llama_local.LLAMA_PORT, self.port + 7)
        self.assertIn(f":{self.port + 7}/v1", llama_local.LLAMA_SERVER_BASE)
        import core.config as config
        self.assertEqual(config.LLAMA_SERVER_BASE, llama_local.LLAMA_SERVER_BASE)   # the chat calls follow it
        args = start.call_args[0][1]
        self.assertEqual(args[args.index("--port") + 1], str(self.port + 7))
        self.assertIn("other.exe", r["message"])

    def test_a_crash_loop_is_left_alone(self):
        import time
        llama_builds._watch["restarts"] = [time.time()] * llama_builds.RESTART_LIMIT
        with patch.object(llama_builds, "_answers", return_value=False), \
                patch.object(llama_builds, "_restart_exe", return_value="x.exe"), patch.object(llama_builds, "_start") as start:
            r = llama_builds.ensure_running()
        self.assertFalse(r["ok"])
        start.assert_not_called()
        self.assertIn("llama-server.log", r["message"])

    def test_messages_say_what_happened_without_run_bat(self):
        self.assertNotIn("run.bat", llama_local.unreachable_message())
        self.assertNotIn("Is run.bat still running", PAGE)
        self.assertEqual(page_kind(llama_local.unreachable_message("x")), "The local model server isn’t answering")
        self.assertEqual(page_kind("llama-server stopped in the middle of the reply and was started again - Try again"),
                         "The local model server restarted")

    def test_a_reply_that_finds_it_down_starts_it_and_carries_on(self):
        import agent.models as models
        seen = iter([None, ["M"], ["M"]])
        with patch.object(llama_local, "server_models", side_effect=lambda force=False: next(seen)), \
                patch.object(llama_builds, "ensure_running", return_value={"ok": True, "restarted": True, "message": ""}) as er, \
                patch.object(models, "resolve_local_model", side_effect=RuntimeError("got past the restart")):
            with self.assertRaisesRegex(RuntimeError, "got past the restart"):
                models.call_local.__wrapped__([{"role": "user", "content": "hi"}], False)
        er.assert_called_once()

    def test_helper_calls_look_fresh_and_recover_instead_of_waiting_out_a_hung_router(self):
        # Live, 4 Nov: on the 20 s cache the prompt-prep and planner calls each waited 120 s on a hung router.
        import agent.models as models
        with patch.object(llama_local, "server_models", side_effect=[None, ["M"]]) as sm, \
                patch.object(llama_builds, "ensure_running", return_value={"ok": True}) as er:
            self.assertTrue(models._router_ready())
        self.assertEqual(sm.call_args_list[0].kwargs, {"force": True})
        er.assert_called_once()
        with patch.object(llama_local, "server_models", return_value=None), \
                patch.object(llama_builds, "ensure_running", return_value={"ok": False}):
            self.assertFalse(models._router_ready())
        src = open(os.path.join(HERE, "agent", "models.py"), encoding="utf-8").read()
        self.assertEqual(src.count("if not _router_ready():"), 2)            # model_json and model_text

    def test_a_message_bigger_than_the_window_is_left_out_so_the_chat_goes_on(self):
        import agent.models as models
        import core.state as shared
        saved = shared.app_state.get("conversation")
        self.addCleanup(shared.app_state.__setitem__, "conversation", saved)
        shared.app_state["conversation"] = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"},
                                            {"role": "assistant", "content": "hello"},
                                            {"role": "user", "content": "log line " * 30000}]
        self.assertGreater(models._drop_oversized_request(16384), 16384)
        self.assertIn("too long for this model", shared.app_state["conversation"][-1]["content"])
        self.assertEqual(shared.app_state["conversation"][1]["content"], "hi")
        self.assertEqual(models._drop_oversized_request(16384), 0)           # a normal message is kept as it is

    def test_a_broken_connection_is_told_apart_from_an_http_error(self):
        import agent.models as models
        self.assertTrue(models._connection_lost(ConnectionResetError("Connection reset by peer")))
        self.assertTrue(models._connection_lost(Exception("RemoteProtocolError: peer closed connection")))
        self.assertFalse(models._connection_lost(Exception("Error code: 500 - failed to load")))

    def test_the_watchdog_restarts_a_dead_router_but_not_a_busy_one(self):
        dead = Mock()
        dead.poll.return_value = 1
        llama_builds._watch["seen_up"] = True
        with patch.dict(llama_builds._started, {"proc": dead}), patch.object(llama_builds, "ensure_running") as er, \
                patch.object(llama_builds._watch["stop"], "wait", side_effect=[False, True]):
            llama_builds._watchdog()
        er.assert_called_once_with("watchdog")
        alive = Mock()
        alive.poll.return_value = None
        import local_runtime
        with patch.dict(llama_builds._started, {"proc": alive}), patch.object(llama_builds, "ensure_running") as er, \
                patch.object(local_runtime, "busy", return_value=True), patch.object(llama_builds, "_answers", return_value=False), \
                patch.object(llama_builds._watch["stop"], "wait", side_effect=[False, False, False, False, True]):
            llama_builds._watchdog()
        er.assert_not_called()                                   # a reply is running on it: slow, not hung


class DummyListener:
    """A program that isn't ours, listening on a port."""

    def __init__(self, port=0):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]

    def close(self):
        self.sock.close()


class PortTests(unittest.TestCase):
    """Report 6: a port another program holds is told apart from our own leftover, and every part follows."""

    def setUp(self):
        self.dummy = DummyListener()
        self.addCleanup(self.dummy.close)

    def test_busy_and_free(self):
        self.assertFalse(ports.is_free(self.dummy.port))
        self.assertNotEqual(ports.pick(self.dummy.port), self.dummy.port)
        self.assertGreater(ports.pick(self.dummy.port), self.dummy.port)

    def test_a_foreign_holder_moves_us_up(self):
        with patch("ports.holder", return_value={"pid": 1, "name": "other.exe", "exe": "C:\\Elsewhere\\other.exe"}):
            c = ports.claim(self.dummy.port)
        self.assertEqual(c["how"], "moved")
        self.assertNotEqual(c["port"], self.dummy.port)
        self.assertIn("other.exe", c["held_by"])

    def test_our_own_answering_server_is_reused_and_a_dead_one_reclaimed(self):
        mine = {"pid": 2, "name": "pythonw.exe", "exe": os.path.join(ports.APP_DIR, ".venv", "Scripts", "pythonw.exe")}
        with patch("ports.holder", return_value=mine):
            self.assertEqual(ports.claim(self.dummy.port, alive=lambda p: True)["how"], "reuse")
            with patch("ports.stop_tree", side_effect=lambda pid, wait_port=None: self.dummy.close() or True) as stop:
                self.assertEqual(ports.claim(self.dummy.port, alive=lambda p: False)["how"], "reclaimed")
            stop.assert_called_once()

    def test_what_counts_as_ours(self):
        self.assertTrue(ports.is_ours({"exe": "x", "cwd": ports.APP_DIR}))
        self.assertTrue(ports.is_ours({"cmdline": ["llama-server.exe", "--models-preset", os.path.join(ports.APP_DIR, "models.ini")]}))
        self.assertFalse(ports.is_ours({"exe": "C:\\Program Files\\Other\\x.exe", "cwd": "C:\\Other", "cmdline": ["x", "-p", "8000"]}))
        self.assertFalse(ports.is_ours({"exe": ports.APP_DIR + "-copy\\x.exe"}))        # another install beside it

    def test_the_run_file(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "run", "ports.json")
        ports.write_run({"app": 8001}, path)
        ports.write_run({"llama": 8091}, path)
        self.assertEqual((ports.read_run(path)["app"], ports.read_run(path)["llama"]), (8001, 8091))

    def test_the_image_engine_moves_off_a_taken_port(self):
        import image_gen
        old = (image_gen.PORT, image_gen.BASE)
        self.addCleanup(lambda: (setattr(image_gen, "PORT", old[0]), setattr(image_gen, "BASE", old[1])))
        image_gen.PORT = self.dummy.port
        with patch("ports.note"):
            image_gen._claim_port()
        self.assertNotEqual(image_gen.PORT, self.dummy.port)
        self.assertEqual(image_gen.BASE, f"http://127.0.0.1:{image_gen.PORT}")

    def test_phone_access_moves_off_a_taken_port(self):
        from connect import remote_access
        started = []

        class FakeServer:
            def __init__(self, config):
                started.append(config.port)
                self.should_exit = False

            def run(self):
                pass
        with patch.object(remote_access, "has_password", return_value=True), \
                patch.object(remote_access, "bind_address", return_value="127.0.0.1"), \
                patch.object(remote_access, "allowed_address", return_value=True), \
                patch("uvicorn.Server", FakeServer), patch.object(remote_access, "status", return_value={}):
            remote_access.sync({"remote_access": "lan", "remote_port": self.dummy.port})
        self.addCleanup(remote_access.stop)
        self.assertEqual(len(started), 1)
        self.assertNotEqual(started[0], self.dummy.port)
        self.assertIn(f"Port {self.dummy.port} is used by", remote_access._server["moved"])

    def test_launcher_and_desktop_follow_the_chosen_port(self):
        with open(os.path.join(HERE, "launcher.pyw"), encoding="utf-8") as f:
            launcher = f.read()
        self.assertIn("choose_ports()", launcher)
        self.assertIn('port=APP_PORT', launcher)
        self.assertIn("llama_builds.adopt(llama)", launcher)
        self.assertNotIn('message("FX Workspace could not start (is port 8000 taken?)', launcher)
        js = open(os.path.join(HERE, "desktop", "main.js"), encoding="utf-8").read()
        self.assertIn("'/api/whoami'", js)
        self.assertIn("path.join(root, 'data', 'run', 'ports.json')", js)
        self.assertIn("setInterval(watchServer, 3000);", js)
        bat = open(os.path.join(HERE, "run.bat"), encoding="utf-8").read()
        self.assertIn("ports.pick(8000)", bat)
        self.assertIn("--port %APP_PORT%", bat)

    def test_whoami_names_this_install(self):
        from api import server
        w = server.whoami()
        self.assertEqual(w["app"], "fx-workspace")
        self.assertEqual(os.path.normcase(w["root"]), os.path.normcase(HERE))
        from connect import remote_access
        self.assertIn(("GET", "/api/whoami"), remote_access.PHONE_BLOCKED)


class PictureTests(unittest.TestCase):
    """Reports 5 and 7: a picture is shown at once and is the answer - the model never goes looking for it."""

    def setUp(self):
        import core.state as shared
        self.shared = shared
        keys = ("conversation", "turn_images", "current_user_request", "turn_policy", "mode", "intent_guard")
        saved = {k: shared.app_state.get(k) for k in keys}
        self.addCleanup(lambda: shared.app_state.update(saved))
        shared.task_board.clear()
        self.addCleanup(shared.task_board.clear)

    def run_loop(self, request, after=None):
        import agent.loop as agent_loop
        import agent.models as llm
        from tools import runtime
        shared = self.shared
        shared.app_state.update(conversation=[{"role": "system", "content": "s"}, {"role": "user", "content": request}],
                                turn_images=[], current_user_request=request, turn_policy={"mode": "action", "allow_mutation": True},
                                intent_guard=None, mode="local")
        call = SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
            content="", tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(
                name="generate_image", arguments=json.dumps({"prompt": "a toaster"})))]))])
        answer = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=after or "A toaster, and here's the poem.", tool_calls=None))])
        script = iter([call, answer])
        model = Mock(side_effect=lambda: next(script))

        def made(name, args, **kw):
            shared.app_state["turn_images"].append({"model": "Z", "seed": 1, "url": "/api/images/x.png"})
            return {"ok": True, "message": "Done"}
        with patch.object(llm, "call_model", model), patch.object(agent_loop, "_decide", return_value=("auto", "test")), \
                patch.object(agent_loop.schemas, "tool_allowed_current_turn", return_value=True), \
                patch.object(agent_loop, "_invalid_call_reason", return_value=None), \
                patch.object(runtime, "execute_tool_guarded", side_effect=made):
            result = agent_loop.run_agent_loop()
        return result, model.call_count

    def test_a_picture_request_ends_with_the_picture(self):
        result, calls = self.run_loop("make an image of a toaster")
        self.assertEqual(calls, 1)                          # no second call: no reload of the text model, no search
        self.assertEqual(result["status"], "success")
        self.assertIn("Here's your picture", result["message"])

    def test_a_request_for_more_still_gets_its_reply(self):
        result, calls = self.run_loop("make an image of a toaster and write a poem about it")
        self.assertEqual(calls, 2)
        self.assertIn("poem", result["message"])

    def test_the_tool_result_says_it_is_already_shown(self):
        dispatch = open(os.path.join(HERE, "tools", "dispatch.py"), encoding="utf-8").read()
        self.assertIn('event_log.emit_event("image"', dispatch)
        gen = open(os.path.join(HERE, "media", "image_gen.py"), encoding="utf-8").read()
        self.assertIn("ALREADY SHOWN to the user", gen)

    def test_the_page_draws_it_from_the_event(self):
        self.assertIn("e.kind === 'image' && Array.isArray(e.images)", PAGE)
        self.assertIn("turn.picsEl.className = 'assistant-body live-pictures'", PAGE)

    def test_a_reply_that_fails_after_the_picture_keeps_it(self):
        import agent.tasks as tasks
        self.shared.app_state["turn_images"] = [{"model": "Z", "seed": 5, "url": "/api/images/t.png"}]
        self.shared.app_state["conversation"] = [{"role": "system", "content": "s"}]
        out = tasks._with_images({"status": "error", "message": "This chat is too long for M"})
        self.assertEqual(out["status"], "success")
        self.assertIn("![Z - seed 5](/api/images/t.png)", out["message"])
        self.assertIn("This chat is too long for M", out["message"])

    def test_no_text_review_of_a_picture(self):
        orch = open(os.path.join(HERE, "chat", "orchestrator.py"), encoding="utf-8").read()
        self.assertIn('rmode = "off" if bare or routed_only or pictured else', orch)

    def test_picture_reply_uses_short_prose_only(self):
        import agent.loop as agent_loop
        self.assertEqual(agent_loop.picture_reply("A red toaster on a counter.", [1]), "A red toaster on a counter.")
        self.assertIn("Here's your picture", agent_loop.picture_reply("I'll now generate the image with generate_image", [1]))
        self.assertIn("2 pictures", agent_loop.picture_reply("", [1, 2]))


class MissingFileTests(TempIni):
    """Suggestion 1 (approved 5 Nov): a model whose file is gone is marked, never picked by Auto, and can be found or
    removed - the entry only, never a file."""

    def setUp(self):
        super().setUp()
        llama_local._sizes.clear()
        self.addCleanup(llama_local._sizes.clear)

    def test_it_is_told_apart(self):
        self.assertEqual(llama_local.file_missing("Gone")["what"], "model")
        self.assertIsNone(llama_local.file_missing("M"))
        with open(self.ini, "a", encoding="utf-8") as f:
            f.write(f"\n[Eyes]\nmodel = {self.gguf}\nmmproj = {os.path.join(self.tmp, 'no-mmproj.gguf')}\n")
        llama_local._ini_cache.update(key=None, configs=None)
        self.assertEqual(llama_local.file_missing("Eyes")["what"], "mmproj")

    def test_auto_passes_over_it(self):
        with patch.object(llama_local, "server_models", return_value=["M", "Gone"]):
            usable, skipped = llama_local.resolve_candidates(["Gone", "M"])
        self.assertEqual(usable, ["M"])
        self.assertIn("Gone", skipped)

    def test_the_list_marks_it(self):
        from api import server
        with patch.object(llama_local, "server_models", return_value=[]):
            models = {m["id"]: m for m in server.get_local_models()["models"]}
        self.assertEqual(models["Gone"]["missing"], "model")
        self.assertNotIn("missing", models["M"])

    def test_find_points_it_at_the_moved_file(self):
        from api import server
        moved = os.path.join(self.tmp, "elsewhere")
        os.makedirs(moved)
        target = os.path.join(moved, "moved-away.gguf")
        with open(target, "wb") as f:
            f.write(b"GGUF")
        with patch("paths.model_dirs", return_value=[self.tmp]), patch.object(server.tasks.turns, "running", return_value=False), \
                patch.object(server, "_model_entry_changed"):
            r = server.find_model_file(server.ModelEntryRequest(model="Gone"))
        self.assertTrue(r["ok"], r)
        self.assertEqual(os.path.normcase(llama_local.model_own_options("Gone")["model"]), os.path.normcase(target))

    def test_remove_takes_only_the_entry(self):
        from api import server
        with patch.object(server.tasks.turns, "running", return_value=False), patch.object(server, "_model_entry_changed"):
            r = server.remove_model_entry(server.ModelEntryRequest(model="M"))
        self.assertTrue(r["ok"], r)
        self.assertNotIn("M", [c["name"] for c in llama_local.list_model_configs()])
        self.assertTrue(os.path.exists(self.gguf))                       # the file stays
        self.assertTrue(any(n.startswith("models.ini.bak-") for n in os.listdir(self.tmp)))

    def test_the_picker_shows_it_with_find_and_remove(self):
        self.assertIn("if (m.missing) {", PAGE)
        self.assertIn("fixModelEntry('find', ", PAGE)
        self.assertIn("fixModelEntry('remove', ", PAGE)
        from connect import remote_access
        self.assertIn(("POST", "/api/local-models/remove"), remote_access.PHONE_BLOCKED)
        self.assertIn(("POST", "/api/local-models/find-file"), remote_access.PHONE_BLOCKED)


class ApprovedSuggestionTests(TempIni):
    """Suggestions 2-7 (approved 5 Nov)."""

    def test_short_of_ram_it_loads_from_disk_instead_once(self):
        import agent.models as models
        llama_local.set_model_options({"Big-35B": {"load-mode": "none"}}, backup=False)
        with patch("llama_local.unload_all"):
            self.assertTrue(models._load_from_disk_instead("Big-35B", {"kind": "ram"}))
            self.assertNotIn("load-mode", llama_local.model_own_options("Big-35B"))
            self.assertFalse(models._load_from_disk_instead("Big-35B", {"kind": "ram"}))     # once: then it says why
        self.assertFalse(models._load_from_disk_instead("M", {"kind": "vram"}))

    def test_a_hard_crash_leaves_a_crash_log(self):
        from api import server
        self.assertIn("faulthandler.enable(file=f, all_threads=True)", open(os.path.join(HERE, "api", "server.py"), encoding="utf-8").read())
        self.assertIn("    _crash_log()\n", open(os.path.join(HERE, "api", "server.py"), encoding="utf-8").read())
        self.assertTrue(callable(server._crash_log))

    def test_text_output_of_windows_commands_never_crashes_its_reader(self):
        import importlib.machinery
        import importlib.util
        import subprocess
        loader = importlib.machinery.SourceFileLoader("launcher_rel", os.path.join(HERE, "launcher.pyw"))
        launcher = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
        loader.exec_module(launcher)
        seen = []
        original = subprocess.Popen.__init__
        self.addCleanup(setattr, subprocess.Popen, "__init__", original)
        subprocess.Popen.__init__ = lambda self, *a, **k: seen.append(k)
        with patch.object(launcher, "IS_WIN", True):
            launcher.quiet_children()
        subprocess.Popen(["x"], text=True)
        subprocess.Popen(["x"], text=True, errors="strict")
        subprocess.Popen(["x"])
        self.assertEqual([k.get("errors") for k in seen], ["replace", "strict", None])
        self.assertIn("sherpa_onnx", launcher.REPAIR_IMPORTS)
        self.assertEqual(launcher.missing_packages(), [])            # this environment has them all

    def test_voice_setup_says_what_to_do_without_run_bat(self):
        self.assertNotIn("run run.bat once", PAGE)
        self.assertIn("it installs missing packages as it starts", PAGE)

    def test_a_spilled_model_is_offered_the_split_once(self):
        self.assertIn("offerSpillFix(S.placement);", PAGE)
        self.assertIn("action: 'Run on RAM too'", PAGE)
        self.assertIn("spillOffered.add(p.model);", PAGE)

    def test_test_bonsai_never_switches_a_real_router(self):
        src = open(os.path.join(HERE, "tests", "test_bonsai.py"), encoding="utf-8").read()
        self.assertIn('patch.object(llama_builds, "ensure_for", return_value=None)', src)


if __name__ == "__main__":
    unittest.main()
