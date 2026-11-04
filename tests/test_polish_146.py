"""1.4.6, asked 4 Nov: the update check by itself, the app_appearance tool, and the underglow in Aurora and Eclipse."""
import unittest
from unittest.mock import patch

import core.state as shared
from base import app_version
from chat import intent_router, skills
from core import config
from pc import app_look, permissions
from tools import schemas

PAGE = open("index.html", encoding="utf-8").read()
SERVER = open("api/server.py", encoding="utf-8").read()


class UpdateCheckTests(unittest.TestCase):
    def test_known_never_asks_github(self):
        with patch.dict(app_version._cache, {"at": 0.0, "result": None}), patch("netlayer.client") as client:
            d = app_version.known()
        client.assert_not_called()
        self.assertEqual(d["current"], app_version.current())
        self.assertNotIn("newer", d)

    def test_known_gives_the_last_answer(self):
        with patch.dict(app_version._cache, {"at": 123.0, "result": {"latest": "9.9.9", "newer": True}}):
            d = app_version.known()
        self.assertEqual((d["latest"], d["newer"], d["checked"]), ("9.9.9", True, 123.0))

    def test_a_round_checks_only_while_on(self):
        with patch.object(app_version, "check") as check:
            self.assertTrue(app_version.auto_round(lambda: True))
            self.assertFalse(app_version.auto_round(lambda: False))   # switched off in Settings > About
        check.assert_called_once_with(force=True)
        with patch.object(app_version, "check", side_effect=OSError("offline")):
            self.assertFalse(app_version.auto_round(lambda: True))    # offline: no crash, the next round tries

    def test_on_by_default_and_wired(self):
        self.assertTrue(config.DEFAULT_SETTINGS["update_check"])
        self.assertFalse(config.clean_privacy_settings({"update_check": False})["update_check"])
        self.assertIn("app_version.start_auto(", SERVER)
        self.assertIn('@app.get("/api/version/latest")', SERVER)
        self.assertIn('id="update-banner"', PAGE)
        self.assertIn("store.set('updateSkip'", PAGE)                    # Not now: until a newer one comes


class AppAppearanceToolTests(unittest.TestCase):
    def setUp(self):
        self.saved = patch.object(config, "save_settings", return_value=True)
        self.saved.start()
        self.state = patch.dict(shared.app_state, {"settings": dict(config.DEFAULT_SETTINGS, appearance="aurora", color_mode="light")})
        self.state.start()

    def tearDown(self):
        self.state.stop()
        self.saved.stop()

    def test_dark_and_light(self):
        r = app_look.apply("dark")
        s = shared.app_state["settings"]
        self.assertTrue(r["ok"])
        self.assertEqual((s["color_mode"], s["theme"]), ("dark", "aurora"))
        self.assertEqual(app_look.apply("light")["mode"], "light")
        self.assertEqual(shared.app_state["settings"]["theme"], "auroralight")
        self.assertTrue(shared.app_state["settings_at"])                # the page reloads its settings

    def test_appearance_by_its_everyday_name(self):
        r = app_look.apply(appearance="ChatGPT")
        self.assertEqual((r["appearance"], shared.app_state["settings"]["appearance"]), ("ChatGPT", "mono"))

    def test_bad_values_change_nothing(self):
        before = dict(shared.app_state["settings"])
        self.assertFalse(app_look.apply("purple")["ok"])
        self.assertFalse(app_look.apply(appearance="mossad")["ok"])     # a secret theme stays the logo's secret
        self.assertFalse(app_look.apply()["ok"])
        self.assertEqual(shared.app_state["settings"], before)

    def test_registered_and_runs_without_asking(self):
        self.assertIn("app_appearance", schemas.TOOL_NAMES)
        self.assertEqual(skills.SKILL_OF["app_appearance"], "desktop")
        self.assertEqual(permissions.decide("app_appearance", config.DEFAULT_SETTINGS), "auto")
        for text in ("switch to dark mode", "make it light", "change the theme to aurora"):
            self.assertIn("desktop", skills.pick(text), text)
            self.assertEqual(intent_router.classify_turn(text)["mode"], "action", text)


class UnderglowTests(unittest.TestCase):
    def test_aurora_and_eclipse_have_mossads_underglow(self):
        self.assertIn(':root:is([data-skin="aurora"], [data-skin="eclipse"]) :is(.icon-btn, .tool-btn):hover', PAGE)
        self.assertIn('.sb-temp[aria-pressed="true"] { box-shadow: 0 0 0 1px var(--border-strong), 0 0 14px -5px var(--glow); }', PAGE)


if __name__ == "__main__":
    unittest.main()
