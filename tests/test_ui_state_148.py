"""1.4.8: the page's look-and-feel state kept on the server (chat/ui_state.py, /api/ui-state). Temp folder only."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import remote_access
import ui_state

import api.server as server


def _client():
    from fastapi.testclient import TestClient
    return TestClient(server.app, base_url="http://127.0.0.1:8000")


class UiStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "ui_state.json")
        p = patch.object(ui_state, "PATH", self.path)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        self.c = _client()

    def _phone(self):
        """As a signed-in phone: the phone middleware also asks is_remote, so give it a session to pass."""
        return patch.multiple(remote_access, is_remote=lambda scope: True, session=lambda token: {"device": "t"})

    def test_empty_by_default(self):
        r = self.c.get("/api/ui-state").json()
        self.assertEqual(r, {"ok": True, "profile": "pc", "state": {}})

    def test_profiles_kept_apart(self):
        self.c.post("/api/ui-state", json={"state": {"mode": "dark"}})
        with self._phone():
            r = self.c.post("/api/ui-state", json={"state": {"mode": "light", "zoom": 2}}).json()
            self.assertEqual(r["profile"], "phone")
            self.assertEqual(r["state"], {"mode": "light", "zoom": 2})
        self.assertEqual(self.c.get("/api/ui-state").json()["state"], {"mode": "dark"})

    def test_install_keys_shared(self):
        self.c.post("/api/ui-state", json={"state": {"welcomed": True, "mode": "dark"}})
        with self._phone():
            self.assertEqual(self.c.get("/api/ui-state").json()["state"], {"welcomed": True})

    def test_unknown_keys_dropped(self):
        r = self.c.post("/api/ui-state", json={"state": {"evil": 1, "mode": "dark"}}).json()
        self.assertEqual(r["state"], {"mode": "dark"})
        with open(self.path, encoding="utf-8") as f:
            self.assertNotIn("evil", json.dumps(json.load(f)))

    def test_too_big_value_dropped(self):
        r = self.c.post("/api/ui-state", json={"state": {"pinned": "x" * 5000, "mode": "dark"}}).json()
        self.assertEqual(r["state"], {"mode": "dark"})

    def test_null_removes(self):
        self.c.post("/api/ui-state", json={"state": {"mode": "dark", "zoom": 1}})
        r = self.c.post("/api/ui-state", json={"state": {"mode": None}}).json()
        self.assertEqual(r["state"], {"zoom": 1})

    def test_bad_file_falls_back_to_empty(self):
        for text in ("{not json", "[1, 2]", json.dumps({"install": [], "pc": "x", "phone": 3})):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(text)
            self.assertEqual(self.c.get("/api/ui-state").json()["state"], {}, text)

    def test_bad_body_refused_and_nothing_written(self):
        for body in ({}, {"state": "x"}, {"state": [1]}, [1, 2]):
            r = self.c.post("/api/ui-state", json=body)
            self.assertEqual(r.status_code, 200)
            self.assertFalse(r.json()["ok"])
            self.assertIn("message", r.json())
        self.assertFalse(os.path.exists(self.path))

    def test_file_is_valid_json_after_write(self):
        self.c.post("/api/ui-state", json={"state": {"mode": "dark", "welcomed": True}})
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["install"], {"welcomed": True})
        self.assertEqual(data["pc"], {"mode": "dark"})
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_route_in_phone_allowed(self):
        self.assertEqual(set(remote_access.PHONE_ALLOWED["/api/ui-state"].split()), {"GET", "POST"})


if __name__ == "__main__":
    unittest.main()
