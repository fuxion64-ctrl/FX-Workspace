"""1.4.2 Phone access (remote_access.py): private addresses only, a password, sessions, and what a phone may do."""
import os
import tempfile
import unittest
from unittest.mock import patch

import core.state as shared
import remote_access


class RemoteAccessTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        p = patch.object(remote_access, "DATA", os.path.join(self.dir, "remote.json"))
        p.start()
        self.addCleanup(p.stop)
        remote_access._fails.clear()

    def test_only_private_and_tailscale_addresses(self):
        for ip in ("100.101.102.103", "192.168.1.20", "10.0.0.5", "172.16.4.4"):
            self.assertTrue(remote_access.allowed_address(ip), ip)
        for ip in ("8.8.8.8", "0.0.0.0", "127.0.0.1", "169.254.1.1", "::1", "not an ip", "100.128.0.1"):
            self.assertFalse(remote_access.allowed_address(ip), ip)

    def test_the_bind_follows_the_mode(self):
        found = {"tailscale": "100.64.1.2", "lan": ["192.168.0.9"]}
        self.assertEqual(remote_access.bind_address("tailscale", found), "100.64.1.2")
        self.assertEqual(remote_access.bind_address("lan", found), "192.168.0.9")
        self.assertIsNone(remote_access.bind_address("off", found))
        self.assertIsNone(remote_access.bind_address("tailscale", {"tailscale": None, "lan": []}))

    def test_password_hashed_and_login(self):
        self.assertFalse(remote_access.set_password("short")["ok"])
        self.assertTrue(remote_access.set_password("correct horse")["ok"])
        with open(remote_access.DATA, encoding="utf-8") as f:
            self.assertNotIn("correct horse", f.read())              # only the scrypt hash is stored
        self.assertFalse(remote_access.login("1.2.3.4", "wrong", "x")["ok"])
        r = remote_access.login("1.2.3.4", "correct horse", "Pixel 9")
        self.assertTrue(r["ok"])
        s = remote_access.session(r["token"])
        self.assertEqual(s["device"], "Pixel 9")
        self.assertIsNone(remote_access.session("made-up"))
        self.assertEqual([d["device"] for d in remote_access.devices()], ["Pixel 9"])
        self.assertEqual(remote_access.sign_out(), 1)
        self.assertIsNone(remote_access.session(r["token"]))

    def test_lock_out_after_five_wrong_tries(self):
        remote_access.set_password("correct horse")
        for _ in range(5):
            remote_access.login("5.5.5.5", "nope")
        r = remote_access.login("5.5.5.5", "correct horse")
        self.assertFalse(r["ok"])
        self.assertIn("Too many", r["message"])
        self.assertTrue(remote_access.login("6.6.6.6", "correct horse")["ok"])   # another address isn't locked

    def test_a_new_password_signs_everyone_out(self):
        remote_access.set_password("first password")
        token = remote_access.login("1.1.1.1", "first password")["token"]
        remote_access.set_password("second password")
        self.assertIsNone(remote_access.session(token))

    def test_what_a_phone_may_not_do(self):
        for method, path in (("POST", "/api/settings"), ("POST", "/api/remote"), ("GET", "/api/remote"),
                             ("POST", "/api/remote/password"), ("POST", "/api/channels"), ("POST", "/api/keys/openai")):
            self.assertTrue(remote_access.blocked(method, path), path)
        # 1.4.8: an allow-list - a real route ("/api/approve" never existed; an unlisted path is refused now)
        for method, path in (("POST", "/api/chat"), ("GET", "/api/state"), ("GET", "/api/settings"), ("POST", "/api/approval/ab12")):
            self.assertFalse(remote_access.blocked(method, path), path)

    def test_never_starts_without_a_password(self):
        with patch.object(remote_access, "addresses", return_value={"tailscale": "100.64.1.2", "lan": []}), \
                patch.dict(shared.app_state, settings={"remote_access": "tailscale"}):
            st = remote_access.sync({"remote_access": "tailscale"})
        self.assertFalse(st["running"])
        self.assertIn("password", st["problem"])

    def test_phone_turns_get_no_screen_control(self):
        import tools.schemas as schemas
        with patch.dict(shared.app_state, turn_origin="phone", settings={"remote_allow_control": False}):
            self.assertEqual(schemas.origin_off(), {"desktop"})
        with patch.dict(shared.app_state, turn_origin="phone", settings={"remote_allow_control": True}):
            self.assertEqual(schemas.origin_off(), set())
        with patch.dict(shared.app_state, turn_origin="pc", settings={}):
            self.assertEqual(schemas.origin_off(), set())

    def test_qr(self):
        self.assertIn("<svg", remote_access.qr_svg("http://100.64.1.2:8765/"))


class RemoteGateTests(unittest.TestCase):
    """The phone listener through the real app: sign-in required, PC-only paths refused."""
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api.server import app
        cls.client = TestClient(app, base_url="http://100.64.1.2:8765")

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for target in (patch.object(remote_access, "DATA", os.path.join(self.dir, "remote.json")),
                       patch.dict(remote_access._server, bind=("100.64.1.2", 8765)),
                       patch.object(remote_access, "is_remote", return_value=True)):
            target.start()
            self.addCleanup(target.stop)
        remote_access._fails.clear()
        remote_access.set_password("correct horse")

    def test_signed_out_gets_the_sign_in_page(self):
        r = self.client.get("/", follow_redirects=False)
        self.assertEqual((r.status_code, r.headers["location"]), (303, "/remote/login"))
        self.assertEqual(self.client.get("/api/state").status_code, 401)
        self.assertEqual(self.client.get("/remote/login").status_code, 200)

    def test_sign_in_then_chat_allowed_settings_refused(self):
        r = self.client.post("/remote/login", json={"password": "correct horse", "device": "phone"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/api/settings").status_code, 200)
        self.assertEqual(self.client.post("/api/settings", json={"user_name": "x"}).status_code, 403)
        self.assertEqual(self.client.get("/api/remote").status_code, 403)

    def test_wrong_password(self):
        r = self.client.post("/remote/login", json={"password": "nope"})
        self.assertEqual(r.status_code, 401)

    def test_other_hosts_refused(self):
        from fastapi.testclient import TestClient
        from api.server import app
        self.assertEqual(TestClient(app, base_url="http://evil.example").get("/").status_code, 400)


if __name__ == "__main__":
    unittest.main()
