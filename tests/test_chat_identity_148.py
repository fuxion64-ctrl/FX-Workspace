"""1.4.8, audit 2.1: a message says which chat it was written in, and never lands in another one. The check runs
under the turn manager's lock; chat switches look for a running turn again once they hold state_lock; a channel's
turn holds the chat while its own is open and always puts the PC's chat back. No model, no network, nothing in data/."""
import threading
import unittest
from unittest.mock import MagicMock, patch

import api.server as server
import agent.tasks as tasks
import core.state as shared
from turns import TurnManager


def _req(chat_id=None, message="hello there"):
    return shared.ChatRequest(message=message, chat_id=chat_id)


class TurnCheckTests(unittest.TestCase):
    def test_a_refusing_check_starts_nothing_and_changes_nothing(self):
        ran, started = [], []
        tm = TurnManager(runner=lambda r: ran.append(r) or {"status": "success"})
        got = tm.start("x", on_start=lambda: started.append(1), check=lambda: {"status": "error", "why": "other chat"})
        self.assertEqual(got, {"refused": {"status": "error", "why": "other chat"}})
        self.assertFalse(tm.running())
        self.assertEqual((ran, started), ([], []))

    def test_a_passing_check_starts_the_turn(self):
        done = threading.Event()
        tm = TurnManager(runner=lambda r: done.set() or {"status": "success"})
        got = tm.start("x", check=lambda: None)
        self.assertIn("id", got)
        self.assertTrue(done.wait(5))

    def test_the_check_is_not_asked_while_a_turn_runs(self):
        gate, asked = threading.Event(), []
        tm = TurnManager(runner=lambda r: gate.wait(5))
        tm.start("x")
        try:
            self.assertIsNone(tm.start("y", check=lambda: asked.append(1)))
            self.assertEqual(asked, [])
        finally:
            gate.set()


class WrongChatTests(unittest.TestCase):
    def test_a_message_for_another_chat_is_refused(self):
        with patch.dict(shared.app_state, active_chat={"id": "bbbbbbbbbbbb", "title": "B"}, chat_hold=None), \
                patch.object(server.sessions, "public_chat", return_value={"id": "bbbbbbbbbbbb"}):
            r = server._wrong_chat(_req("aaaaaaaaaaaa"))
        self.assertTrue(r["chat_changed"])
        self.assertEqual(r["chat"], {"id": "bbbbbbbbbbbb"})

    def test_the_open_chat_and_a_new_chat_pass(self):
        with patch.dict(shared.app_state, active_chat={"id": "aaaaaaaaaaaa"}, chat_hold=None):
            self.assertIsNone(server._wrong_chat(_req("aaaaaaaaaaaa")))
        with patch.dict(shared.app_state, active_chat=None, chat_hold=None):
            self.assertIsNone(server._wrong_chat(_req("")))

    def test_a_new_chat_message_is_refused_while_a_chat_is_open(self):
        with patch.dict(shared.app_state, active_chat={"id": "aaaaaaaaaaaa"}, chat_hold=None), \
                patch.object(server.sessions, "public_chat", return_value={"id": "aaaaaaaaaaaa"}):
            self.assertTrue(server._wrong_chat(_req(""))["chat_changed"])

    def test_no_chat_id_is_not_checked(self):
        # Routines, channels and a page from before 1.4.8 send none: they run in the open chat, as before.
        with patch.dict(shared.app_state, active_chat={"id": "aaaaaaaaaaaa"}, chat_hold=None):
            self.assertIsNone(server._wrong_chat(_req(None)))

    def test_busy_while_a_channel_holds_the_chat_or_a_switch_holds_the_lock(self):
        with patch.dict(shared.app_state, active_chat={"id": "aaaaaaaaaaaa"}, chat_hold="Discord"):
            r = server._wrong_chat(_req("aaaaaaaaaaaa"))
            self.assertIn("Discord", r["message"])
            self.assertNotIn("chat_changed", r)
        with patch.dict(shared.app_state, active_chat={"id": "aaaaaaaaaaaa"}, chat_hold=None):
            self.assertTrue(shared.state_lock.acquire(timeout=1))
            try:
                self.assertEqual(server._wrong_chat(_req("aaaaaaaaaaaa"))["status"], "error")
            finally:
                shared.state_lock.release()

    def test_start_chat_sends_nothing_to_the_wrong_chat(self):
        with patch.dict(shared.app_state, active_chat={"id": "bbbbbbbbbbbb"}, chat_hold=None, turn_origin="pc"), \
                patch.object(server.sessions, "public_chat", return_value={"id": "bbbbbbbbbbbb"}), \
                patch.object(server.sessions, "temporary_chat", return_value=None), \
                patch.object(server.chat_guard, "is_locked", return_value=False), \
                patch.object(server.labs, "gpu_busy", return_value=False), \
                patch.object(tasks.turns, "_runner") as runner:
            r = server.start_chat(_req("aaaaaaaaaaaa"))
        self.assertTrue(r["chat_changed"])
        self.assertEqual(r["status"], "error")
        runner.assert_not_called()
        self.assertFalse(tasks.turns.running())


class SwitchDuringATurnTests(unittest.TestCase):
    def test_with_lock_looks_again_once_it_holds_the_lock(self):
        # A turn that started between the first look and the lock: the switch must not run.
        looks = iter([False, True])
        ran = []
        with patch.object(tasks.turns, "running", side_effect=lambda: next(looks)):
            r = server._with_lock(lambda: ran.append(1) or {"ok": True})
        self.assertFalse(r["ok"])
        self.assertEqual(ran, [])
        self.assertFalse(shared.state_lock.locked())


class ChannelTurnTests(unittest.TestCase):
    def _patches(self):
        return (patch.object(server, "_reset_conversation_locked"),
                patch.object(server.sessions, "open_chat", return_value={"id": "x"}),
                patch.object(server.sessions, "new_chat"),
                patch.object(server.shared.chat_store, "load", return_value={"id": "x"}))

    def test_the_chat_is_held_during_the_turn_and_put_back_after(self):
        seen = {}

        def start(req, meta, on_start=None, check=None):
            seen["hold"] = shared.app_state.get("chat_hold")
            return {"id": "t1"}
        p1, p2, p3, p4 = self._patches()
        with p1, p2 as opened, p3, p4, patch.dict(shared.app_state, active_chat={"id": "pcpcpcpcpcpc"}, chat_hold=None), \
                patch.object(tasks.turns, "running", return_value=False), \
                patch.object(tasks.turns, "start", side_effect=start), \
                patch.object(tasks.turns, "wait", return_value={"status": "done", "result": {"status": "success", "message": "hi"}}):
            used, result = server._channel_turn("chchchchchch", "hello", "Discord", lambda p: None)
            self.assertEqual(seen["hold"], "Discord")
            self.assertIsNone(shared.app_state.get("chat_hold"))
            self.assertEqual(shared.app_state.get("turn_origin"), "pc")
        self.assertEqual(result["message"], "hi")
        self.assertEqual([c.args[0] for c in opened.call_args_list], ["chchchchchch", "pcpcpcpcpcpc"])

    def test_the_pc_chat_comes_back_when_the_turn_could_not_start(self):
        p1, p2, p3, p4 = self._patches()
        with p1, p2 as opened, p3, p4, patch.dict(shared.app_state, active_chat={"id": "pcpcpcpcpcpc"}, chat_hold=None), \
                patch.object(tasks.turns, "running", return_value=False), \
                patch.object(tasks.turns, "start", return_value=None):
            _used, result = server._channel_turn("chchchchchch", "hello", "Discord", lambda p: None)
            self.assertIsNone(shared.app_state.get("chat_hold"))
        self.assertEqual(result["status"], "error")
        self.assertEqual(opened.call_args_list[-1].args[0], "pcpcpcpcpcpc")

    def test_not_being_able_to_put_it_back_is_logged(self):
        with patch.object(tasks.turns, "running", return_value=False), \
                patch.object(server.shared, "state_lock", MagicMock(acquire=MagicMock(return_value=False))), \
                patch.object(server.event_log, "log_action") as log:
            self.assertFalse(server._restore_pc_chat("pcpcpcpcpcpc", None))
        self.assertIn("Could not reopen", log.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
