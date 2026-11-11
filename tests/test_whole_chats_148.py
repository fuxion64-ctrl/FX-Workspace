"""1.4.8 whole chats on disk (audit 2.8, docs/plans/1.4.8-chat-identity.md stage 3): every transcript entry and
conversation message is kept, the page gets the last 200 with their absolute offset and asks for older ones, and the
model's view of a long chat stays the size it was. Temp folders only."""
import tempfile
import unittest
from unittest import mock

import chat_store

import agent.sessions as sessions
import api.server as server
import core.state as shared


def _transcript(n_turns):
    out = []
    for i in range(n_turns):
        out += [{"r": "u", "t": f"question {i}"}, {"r": "a", "t": f"answer {i}"}]
    return out


def _conversation(n_turns, start=0):
    out = []
    for i in range(start, n_turns):
        out += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    return out


class _Store(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = chat_store.ChatStore(tmp.name)
        before = dict(shared.app_state)
        self.addCleanup(lambda: (shared.app_state.clear(), shared.app_state.update(before)))
        for p in (mock.patch.object(shared, "chat_store", self.store), mock.patch.object(sessions, "_save_session"),
                  mock.patch("events.log_action")):
            p.start()
            self.addCleanup(p.stop)

    def client(self):
        from fastapi.testclient import TestClient
        return TestClient(server.app, base_url="http://127.0.0.1:8000")


class KeptWholeTests(_Store):
    def test_a_long_chat_is_saved_and_reopened_whole(self):
        chat = {"id": "a" * 12, "title": "long", "transcript": _transcript(600), "conversation": _conversation(600)}
        self.store.save(chat)
        loaded = self.store.load("a" * 12)
        self.assertEqual(len(loaded["transcript"]), 1200)                 # the cap was 400
        self.assertEqual(len(loaded["conversation"]), 1200)               # the cap was 300
        self.assertEqual(loaded["transcript"][0]["t"], "question 0")

    def test_a_trimmed_view_keeps_the_older_messages_on_disk(self):
        cid = "b" * 12
        self.store.save({"id": cid, "title": "t", "transcript": _transcript(50), "conversation": _conversation(50)})
        # The open chat's view after fitting dropped turns 0-29 and one more turn came in.
        view = _conversation(51, start=30)
        self.store.save({"id": cid, "title": "t", "transcript": _transcript(51), "conversation": view})
        stored = self.store.load(cid)
        self.assertEqual(len(stored["conversation"]), 102)
        self.assertEqual(stored["conversation"][0]["content"], "question 0")
        self.assertEqual(stored["conversation_start"], 60)

    def test_reopening_gives_the_model_only_its_view(self):
        cid = "c" * 12
        self.store.save({"id": cid, "title": "t", "transcript": _transcript(50), "conversation": _conversation(50)})
        self.store.save({"id": cid, "title": "t", "transcript": _transcript(50), "conversation": _conversation(50, 40)})
        with mock.patch("skills.reset_chat"):
            sessions.open_chat(cid)
        convo = shared.app_state["conversation"]
        self.assertEqual(len(convo), 1 + 20)                               # the system message + the view, as before
        self.assertEqual(convo[1]["content"], "question 40")

    def test_an_old_file_without_a_view_start_opens_as_before(self):
        cid = "d" * 12
        self.store.save({"id": cid, "title": "t", "transcript": _transcript(5), "conversation": _conversation(5),
                         "conversation_start": 0})
        with mock.patch("skills.reset_chat"):
            sessions.open_chat(cid)
        self.assertEqual(len(shared.app_state["conversation"]), 11)


class PagingTests(_Store):
    def setUp(self):
        super().setUp()
        self.cid = "e" * 12
        self.store.save({"id": self.cid, "title": "t", "transcript": _transcript(300), "conversation": _conversation(300)})

    def test_getting_a_chat_gives_the_last_200_with_offset_and_total(self):
        chat = self.client().get(f"/api/chats/{self.cid}").json()["chat"]
        self.assertEqual((chat["offset"], chat["total"], len(chat["transcript"])), (400, 600, 200))
        self.assertEqual(chat["transcript"][0]["t"], "question 200")

    def test_older_entries_page_by_absolute_index(self):
        got = self.client().get(f"/api/chats/{self.cid}/transcript", params={"before": 400, "limit": 200}).json()
        self.assertEqual((got["ok"], got["offset"], got["total"], len(got["entries"])), (True, 200, 600, 200))
        self.assertEqual(got["entries"][0]["t"], "question 100")
        first = self.client().get(f"/api/chats/{self.cid}/transcript", params={"before": 150}).json()
        self.assertEqual((first["offset"], len(first["entries"])), (0, 150))
        self.assertFalse(self.client().get("/api/chats/ffffffffffff/transcript").json()["ok"])

    def test_a_message_action_takes_the_absolute_index(self):
        with mock.patch.object(server.tasks.turns, "running", return_value=False):
            r = self.client().post(f"/api/chats/{self.cid}/messages", json={"op": "pin", "index": 10, "pinned": True}).json()
        self.assertTrue(r["ok"], r)
        self.assertTrue(self.store.load(self.cid)["transcript"][10].get("pin"))      # an entry from an older page
        self.assertEqual((r["chat"]["offset"], r["chat"]["total"]), (400, 600))

    def test_listed_for_the_phone(self):
        import remote_access
        self.assertIn("/api/chats/{chat_id}/transcript", remote_access.PHONE_ALLOWED)


if __name__ == "__main__":
    unittest.main()
