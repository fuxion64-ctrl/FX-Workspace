"""1.4.8 server fixes from the 10 Nov 2026 audit: New chat during a turn (2.2), the phone/PC origin (2.10), the
llama-server watchdog (3.5), the phone allow-list (3.6), the builds answer (1.3), 4xx/5xx in server.log (1.2) and
the endpoints that must answer without a request thread (3.3). No model, no network, nothing in data/."""
import asyncio
import inspect
import os
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import llama_builds
import local_runtime
import remote_access

import api.server as server
import agent.tasks as tasks
import core.state as shared


def _client():
    from fastapi.testclient import TestClient
    return TestClient(server.app, base_url="http://127.0.0.1:8000")


class ResetDuringATurnTests(unittest.TestCase):
    """Audit 2.2: "New chat" wiped a turn still in its message check, Prompt Prep or planning."""

    def test_refused_while_a_turn_runs_even_with_the_lock_free(self):
        convo = [{"role": "user", "content": "the running turn's message"}]
        with patch.object(tasks.turns, "running", return_value=True), \
                patch.dict(shared.app_state, conversation=convo), \
                patch.object(server, "_reset_conversation_locked") as wipe:
            self.assertFalse(shared.state_lock.locked())
            r = _client().post("/api/reset").json()
        self.assertEqual(r["status"], "error")
        wipe.assert_not_called()

    def test_allowed_when_nothing_runs(self):
        with patch.object(tasks.turns, "running", return_value=False), \
                patch.object(server, "_reset_conversation_locked") as wipe, \
                patch.object(server.sessions, "new_chat"), patch.object(server.sessions, "public_project", return_value=None):
            r = _client().post("/api/reset").json()
        self.assertEqual(r["status"], "success")
        wipe.assert_called_once()


class TurnOriginTests(unittest.TestCase):
    """Audit 2.10: the origin is set on every turn start and back to "pc" when a turn ends."""

    def test_reset_to_pc_when_a_turn_ends_even_when_it_fails(self):
        with patch.dict(shared.app_state, turn_origin="phone"), \
                patch.object(tasks.sessions, "temporary_chat", return_value=None), \
                patch.object(tasks, "_run_turn_in_chat", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                tasks._run_turn(NS(message="x"))
            self.assertEqual(shared.app_state["turn_origin"], "pc")
        with patch.dict(shared.app_state, turn_origin="phone"), \
                patch.object(tasks.sessions, "temporary_chat", return_value=None), \
                patch.object(tasks, "_run_turn_in_chat", return_value={"status": "success"}):
            tasks._run_turn(NS(message="x"))
            self.assertEqual(shared.app_state["turn_origin"], "pc")

    def test_a_routine_turn_starts_as_the_pc(self):
        seen = {}

        def start(req, meta=None, on_start=None):
            seen["on_start"] = on_start
            return {"id": "t1"}
        with patch.object(shared.routine_store, "get", return_value={"name": "R", "kind": "steps"}), \
                patch.object(server, "_model_loading", return_value=False), \
                patch.object(tasks.turns, "start", side_effect=start), \
                patch.dict(shared.app_state, turn_origin="phone"):
            r = _client().post("/api/routines/r1/run").json()
            self.assertEqual(r["status"], "accepted")
            self.assertIsNotNone(seen["on_start"])
            seen["on_start"]()
            self.assertEqual(shared.app_state["turn_origin"], "pc")


class WatchdogTests(unittest.TestCase):
    """Audit 3.5: crashes counted per model; the left-off message says when it tries again."""

    def setUp(self):
        llama_builds._watch.update(restarts=[], last="", seen_up=False, fails=0, in_use=None, crashes={})
        self.addCleanup(llama_builds._watch.update, restarts=[], last="", seen_up=False, fails=0, in_use=None, crashes={})
        p = patch.dict(os.environ, {"AUTOSTART_LLAMA": "1"})
        p.start()
        self.addCleanup(p.stop)

    def _restart(self):
        with patch.object(llama_builds, "_answers", return_value=False), \
                patch.object(llama_builds, "_restart_exe", return_value="x.exe"), \
                patch("ports.holder", return_value=None), patch("ports.is_free", return_value=True), \
                patch.object(llama_builds, "_start"), patch.object(llama_builds, "_wait_ready", return_value=True):
            return llama_builds.ensure_running.__wrapped__("test")

    def test_one_crashing_model_is_set_aside_and_the_router_still_restarts(self):
        for _ in range(llama_builds.RESTART_LIMIT + 1):
            llama_builds.note_in_use("Crashy-27B")
            self.assertTrue(self._restart()["ok"])                    # the router itself keeps coming back
        self.assertEqual(llama_builds._watch["restarts"], [])          # no blame on the router
        self.assertIn("Crashy-27B", llama_builds.set_aside("Crashy-27B"))
        self.assertIn("Other models still work", llama_builds.set_aside("Crashy-27B"))
        self.assertEqual(llama_builds.set_aside("Fine-9B"), "")

    def test_a_set_aside_model_is_refused_before_it_loads(self):
        llama_builds._watch["crashes"]["Crashy-27B"] = [time.time()] * llama_builds.RESTART_LIMIT
        import agent.models as llm
        with patch("llama_local.server_models", return_value={"Crashy-27B": {}}), \
                patch.object(llm, "resolve_local_model", return_value="Crashy-27B"), \
                patch.object(llm, "_complete_with_protocol", side_effect=AssertionError("must not load")):
            with self.assertRaises(RuntimeError) as cm:
                llm.call_local.__wrapped__([{"role": "user", "content": "hi"}], False)
        self.assertIn("set it aside", str(cm.exception))

    def test_the_router_limit_says_when_it_tries_again(self):
        llama_builds._watch["restarts"] = [time.time()] * llama_builds.RESTART_LIMIT
        r = self._restart()
        self.assertFalse(r["ok"])
        self.assertIn("tries again by itself from", r["message"])
        self.assertNotIn("Restart the app to try again", r["message"])

    def test_the_window_passing_lets_it_restart(self):
        llama_builds._watch["restarts"] = [time.time() - llama_builds.RESTART_WINDOW - 1] * llama_builds.RESTART_LIMIT
        self.assertTrue(self._restart()["ok"])


class PhoneAllowListTests(unittest.TestCase):
    """Audit 3.6: an allow-list, and every /api/ route is in one list or the other."""

    def _api_routes(self):
        for r in server.app.routes:
            path = getattr(r, "path", "")
            if path.startswith("/api/"):
                for m in sorted(getattr(r, "methods", None) or ()):
                    if m != "HEAD":
                        yield m, path

    def test_every_api_route_is_listed(self):
        def covered_by_blocked(m, p):
            return any((bm == "*" or bm == m) and (p == bp or p.startswith(bp + "/")) for bm, bp in remote_access.PHONE_BLOCKED)
        missing = [(m, p) for m, p in self._api_routes()
                   if not covered_by_blocked(m, p) and m not in set(remote_access.PHONE_ALLOWED.get(p, "").split())]
        self.assertEqual(missing, [], "New endpoints: add each to remote_access.PHONE_ALLOWED (the phone may call it) "
                                      "or PHONE_BLOCKED (PC only).")

    def test_an_unlisted_endpoint_is_refused_from_a_phone(self):
        self.assertTrue(remote_access.blocked("GET", "/api/something-new"))
        self.assertTrue(remote_access.blocked("POST", "/api/chat/new-thing"))
        self.assertFalse(remote_access.blocked("GET", "/"))                          # the page itself, as before

    def test_todays_behaviour_is_kept(self):
        self.assertFalse(remote_access.blocked("POST", "/api/chat"))
        self.assertFalse(remote_access.blocked("GET", "/api/turn/abc123"))
        self.assertFalse(remote_access.blocked("GET", "/api/images/thumb/x.png"))
        self.assertFalse(remote_access.blocked("HEAD", "/api/state"))
        self.assertTrue(remote_access.blocked("POST", "/api/settings"))
        self.assertTrue(remote_access.blocked("GET", "/api/whoami"))
        self.assertTrue(remote_access.blocked("DELETE", "/api/chat"))                # wrong method


class BuildsCacheTests(unittest.TestCase):
    """Audit 1.3: the builds answer is asked again only when an executable changes."""

    def setUp(self):
        llama_builds._versions.clear()
        llama_builds._builds_cache.update(key=None, builds=None)
        self.addCleanup(llama_builds._versions.clear)
        self.addCleanup(llama_builds._builds_cache.update, key=None, builds=None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.exe = os.path.join(self.tmp.name, "llama-server.exe")
        with open(self.exe, "w") as f:
            f.write("v1")

    def test_versions_are_read_once_until_the_exe_changes(self):
        import subprocess
        done = subprocess.CompletedProcess([], 0, stdout="", stderr="version: 1 (build 100, commit x)")
        with patch.object(llama_builds, "exe_of", side_effect=lambda k: self.exe if k == "latest" else None), \
                patch.object(llama_builds, "running", return_value={"kind": "latest", "exe": self.exe}), \
                patch.object(llama_builds, "mode", return_value="auto"), \
                patch.object(llama_builds.subprocess, "run", return_value=done) as run:
            for _ in range(3):
                self.assertEqual(llama_builds.status()["builds"]["latest"]["version"], "b100")
            self.assertEqual(run.call_count, 1)
            with open(self.exe, "w") as f:                  # an update in place
                f.write("v2 - a bigger file")
            os.utime(self.exe, (time.time() + 5, time.time() + 5))
            done.stderr = "version: 1 (build 200, commit y)"
            self.assertEqual(llama_builds.status()["builds"]["latest"]["version"], "b200")
            self.assertEqual(run.call_count, 2)

    def test_warmed_at_startup(self):
        src = inspect.getsource(server._start_background_services)
        self.assertIn("llama_builds.warm", src)


class HttpErrorLogTests(unittest.TestCase):
    """Audit 1.2: every 4xx/5xx reaches server.log with its method and path; 422 says which field."""

    def test_errors_are_logged_with_method_and_path(self):
        with patch("builtins.print") as out:
            r = _client().post("/api/effort/session", content="null", headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 422)
        lines = " ".join(str(c.args[0]) for c in out.call_args_list if c.args)
        self.assertIn("[HTTP 422] POST /api/effort/session (pc)", lines)
        self.assertIn("[HTTP 422 detail] POST /api/effort/session - body", lines)

    def test_ok_answers_are_not_logged(self):
        with patch("builtins.print") as out:
            _client().get("/api/whoami")
        self.assertFalse([c for c in out.call_args_list if c.args and "[HTTP" in str(c.args[0])])

    def test_previous_runs_are_kept(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "launcher.pyw")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        ns = {"os": os, "LOGS": self._logs()}
        exec(src[src.index("KEEP_RUNS = "):src.index("def answers(")], ns)       # log_file, as launcher.pyw has it
        for n in range(5):
            f = ns["log_file"]("server")
            f.write(f"run {n}")
            f.close()
        names = sorted(os.listdir(self.logs))
        self.assertEqual(names, ["server.log", "server.prev.log", "server.prev2.log", "server.prev3.log"])
        with open(os.path.join(self.logs, "server.prev3.log")) as f:
            self.assertEqual(f.read(), "run 1")

    def _logs(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.logs = tmp.name
        return tmp.name


class NoWorkerThreadTests(unittest.TestCase):
    """Audit 3.3: /api/whoami and /api/shell answer on the event loop."""

    def test_whoami_and_shell_are_async(self):
        self.assertTrue(asyncio.iscoroutinefunction(server.whoami))
        self.assertTrue(asyncio.iscoroutinefunction(server.get_shell))

    def test_they_answer_while_every_request_thread_is_taken(self):
        import anyio.to_thread

        async def ask():
            from httpx import ASGITransport, AsyncClient
            limiter = anyio.to_thread.current_default_thread_limiter()
            old = limiter.total_tokens
            limiter.total_tokens = 1
            try:
                await limiter.acquire()                     # the only request thread is busy
                try:
                    async with AsyncClient(transport=ASGITransport(app=server.app), base_url="http://127.0.0.1:8000") as c:
                        who = await asyncio.wait_for(c.get("/api/whoami"), 5)
                        shell = await asyncio.wait_for(c.get("/api/shell"), 5)
                finally:
                    limiter.release()
            finally:
                limiter.total_tokens = old
            return who, shell
        with patch.dict(shared.app_state["settings"], overlay_window=False):
            who, shell = asyncio.run(ask())
        self.assertEqual(who.json()["app"], "fx-workspace")
        self.assertIn("close_to_tray", shell.json())


if __name__ == "__main__":
    unittest.main()
