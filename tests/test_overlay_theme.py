"""The Studio Overlay, the speed indicator, Judaism Mode (the secret theme), the app window
and settings/chat migrations: settings degrade safely, an old settings.json and an old chat
load unchanged, the secret theme can never switch itself on, its music only plays on a
click, everything that moves stops for reduced motion, and the app window only ever opens
the local page."""
import json
import os
import re
import tempfile
import unittest
from unittest import mock

import app_window
import chat_store
import core.config as config

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def page() -> str:
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()


def block(html: str, start: str, end: str) -> str:
    return html[html.index(start):html.index(end, html.index(start))]


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        d = config.DEFAULT_SETTINGS
        self.assertEqual((d["studio_layout"], d["overlay_panels"], d["overlay_clickthrough"], d["overlay_window"]),
                         ("classic", {}, False, False))
        self.assertEqual(d["unlocked_themes"], [])
        self.assertFalse(d["net_indicator"])           # D48: the top bar's speed meter is off unless chosen

    def test_bad_values_degrade(self):
        out = config.clean_privacy_settings({
            "studio_layout": "floating", "overlay_opacity": 7, "overlay_clickthrough": "yes", "overlay_window": 1,
            "unlocked_themes": ["judaism", "hacker", 3],
            "overlay_panels": {"model": {"shown": True, "x": 1.7, "y": -2, "dock": "middle", "evil": "<script>"},
                               "studio": "not a dict", "unknown": {"shown": True}, "network": {"collapsed": 1}}})
        self.assertEqual(out["studio_layout"], "classic")
        self.assertEqual(out["overlay_opacity"], 1.0)
        self.assertIs(out["overlay_clickthrough"], False)
        self.assertIs(out["overlay_window"], False)
        self.assertEqual(out["unlocked_themes"], ["judaism"])
        self.assertEqual(out["overlay_panels"], {"model": {"shown": True, "x": 1, "y": 0}, "network": {}})

    def test_good_values_kept(self):
        panels = {"model": {"shown": False, "collapsed": True, "x": .25, "y": .5, "dock": "free"}, "session": {"shown": True}}
        out = config.clean_privacy_settings({"studio_layout": "overlay", "overlay_opacity": .5, "overlay_panels": panels,
                                             "overlay_window": True, "unlocked_themes": ["judaism"]})
        self.assertEqual((out["studio_layout"], out["overlay_opacity"], out["overlay_window"]), ("overlay", .5, True))
        self.assertEqual(out["overlay_panels"], panels)

    def test_the_secret_theme_is_not_a_saveable_theme(self):
        # "theme" can never be set to it (so a saved settings.json can't start the app in it).
        self.assertNotIn("judaism", config.UI_THEMES)
        self.assertEqual(config.SECRET_THEMES, ("judaism",))
        html = page()
        self.assertNotIn("'judaism'", re.search(r"const THEMES = \[(.*?)\];", html).group(1))


class MigrationTests(unittest.TestCase):
    """What an older install has on disk loads unchanged, with the new keys at their defaults."""

    def test_old_settings_json_loads_unchanged(self):
        old = {"theme": "aurora", "look": "operator", "user_name": "Sam", "effort": "high", "accent": "#4a7fa8",
               "favorite_local_text_models": ["qwen"], "web_search_provider": "duckduckgo", "gone_key": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(old, f)
            with mock.patch.object(config, "SETTINGS_PATH", path):
                s = config.load_settings()
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), old)          # loading never rewrites the file
        for key in ("theme", "look", "user_name", "effort", "accent", "favorite_local_text_models"):
            self.assertEqual(s[key], old[key], key)
        self.assertNotIn("gone_key", s)
        for key in ("studio_layout", "overlay_panels", "overlay_window", "unlocked_themes", "search_mode", "net_route",
                    "net_indicator", "search_history"):
            self.assertEqual(s[key], config.DEFAULT_SETTINGS[key], key)
        self.assertEqual(s["search_mode"], "standard")
        self.assertEqual(s["net_route"], "direct")                  # no proxy and no kill switch by surprise

    def test_old_chat_file_loads_and_saves(self):
        chat = {"id": "0123456789ab", "title": "Old chat", "version": 1, "created": 1700000000.0, "updated": 1700000000.0,
                "transcript": [{"r": "u", "t": "hello"}, {"r": "a", "t": "hi"}],
                "conversation": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "0123456789ab.json"), "w", encoding="utf-8") as f:
                json.dump(chat, f)
            store = chat_store.ChatStore(tmp)
            self.assertEqual(store.load("0123456789ab"), chat)
            saved = store.save(store.load("0123456789ab"), touch=False)       # not mistaken for a temporary chat
            self.assertEqual(saved["transcript"], chat["transcript"])
            self.assertEqual(saved["updated"], chat["updated"])


class SecretThemeTests(unittest.TestCase):
    def setUp(self):
        self.html = page()
        self.js = block(self.html, "const SECRET_THEME_INFO", "/* ---- Settings > Diagnostics")

    def test_never_switched_on_at_startup(self):
        # Only in memory: nothing reads it back from storage, and the boot never turns it on.
        self.assertIn("let secretTheme = null;", self.js)
        self.assertNotRegex(self.js, r"secretTheme\s*=\s*(store|tabStore|localStorage|sessionStorage)")
        boot = block(self.html, "(function boot() {", "</script>")
        self.assertNotIn("setSecretTheme", boot)
        self.assertNotIn("unlockSecretTheme", boot)
        # It also needs the unlock in settings.json.
        active = block(self.js, "function activeSecretTheme() {", "function setSecretTheme")
        self.assertIn("unlocked_themes", active)
        self.assertIn(".includes(secretTheme)", active)

    def test_unlock_is_a_hidden_logo_pattern(self):
        self.assertIn("n >= 6", self.js)
        self.assertIn(".brand-mark", self.js)
        self.assertIn("unlocked_themes: [...have, id]", self.js)

    def test_music_is_opt_in_local_and_click_only(self):
        self.assertIn("store.get('secret.music', false) === true", self.js)          # off by default
        self.assertNotRegex(self.html, r"<audio[^>]*autoplay")
        self.assertNotRegex(self.js, r"https?://[^'\"`]*\.(mp3|ogg|wav|m4a)")      # nothing bundled or downloaded
        self.assertIn("URL.createObjectURL(blob)", self.js)                        # a file the user picked
        plays = [m.start() for m in re.finditer(r"audio\.play\(", self.js)]
        self.assertEqual(len(plays), 1)
        play_fn = block(self.js, "function play() {", "function setVolume")
        self.assertIn("audio.play(", play_fn)                                      # only from the Play button

    def test_kill_switch_is_always_reachable(self):
        self.assertIn('id="sfx-btn"', self.html)                                  # in the top bar while it is on
        self.assertIn("(e.key || '').toLowerCase() === 'k'", self.js)            # Ctrl+Alt+K
        kill = block(self.js, "function kill(quiet) {", "document.addEventListener('keydown'")
        self.assertIn("audio.pause()", kill)
        self.assertIn("sfx-still", kill)

    def test_reduced_motion_and_still(self):
        css = block(self.html, "/* ---- Judaism Mode: the secret theme", "/* ---- The Studio Overlay")
        rm = re.findall(r"@media \(prefers-reduced-motion: reduce\) \{\n(.*?)\n\}", self.html, re.S)[-1]   # the global block
        self.assertIn("animation-duration: 0s", rm)
        for sel in (".sfx, .sfx-top", ".brand-mark", ".composer", ".sfx-btn svg", ".ov-dot.busy", ".sfx-motes"):
            self.assertIn(sel, rm)
        self.assertIn("html.sfx-still", css)
        self.assertIn("still()", block(self.js, "function pulse(kind) {", "function state(a)"))
        self.assertIn("motionStill()", self.js)

    def test_moving_parts_animate_transform_or_opacity(self):
        css = block(self.html, "/* ---- Judaism Mode: the secret theme", "/* ---- The Studio Overlay")
        for name, frames in re.findall(r"@keyframes (sfx-[\w-]+) \{(.*?)\}\s*\}", css, re.S):
            props = set(re.findall(r"([\w-]+)\s*:", frames))
            self.assertLessEqual(props - {"transform", "opacity", "--sfx-a", "filter"}, set(), name)

    def test_reacts_to_activity(self):
        for hook in ("SecretFx.pulse('open')", "SecretFx.pulse('tool')", "SecretFx.state(a)", "SecretFx.pulse(data.status === 'success' ? 'done'"):
            self.assertIn(hook, self.html)
        self.assertIn("SecretFx.sync()", block(self.html, "function applyAppearance() {", "function fitTextColors()"))


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.html = page()
        self.js = block(self.html, "const OVERLAY_PANELS = {", "/* ---- Voice mode (voice.py")

    def test_panels_and_defaults_match_the_server(self):
        names = re.findall(r"^\s+(\w+): \{ title:", self.js, re.M)
        self.assertEqual(tuple(names), config.OVERLAY_PANELS)
        self.assertIn("model: { title: 'Model', icon: 'i-cpu', shown: true", self.js)
        self.assertIn("network: { title: 'Network', icon: 'i-globe', shown: false", self.js)

    def test_hotkey_and_paint_budget(self):
        self.assertIn("e.ctrlKey && e.altKey && !e.shiftKey && (e.key || '').toLowerCase() === 'o'", self.js)
        self.assertIn("250 - (performance.now() - lastPaint)", self.js)             # at most 4 paints a second
        self.assertNotIn("requestAnimationFrame", self.js)                        # no animation loop

    def test_follows_every_theme(self):
        css = block(self.html, "/* ---- The Studio Overlay (JS \"Overlay\")", "/* ---- Theme-switch transitions")
        rules = [r for r in re.findall(r"([^{}]+)\{([^{}]*)\}", css) if ".ov-" in r[0] and "data-skin" not in r[0]]
        self.assertTrue(rules)
        for sel, body in rules:
            self.assertNotRegex(body, r"#[0-9a-fA-F]{3,8}\b|rgba?\(", sel.strip())    # theme tokens only
        self.assertNotIn("backdrop-filter", css)                                 # no re-blur over streaming text

    def test_the_hud_page_is_the_overlay_window(self):
        self.assertIn(f"document.title = '{app_window.HUD_TITLE}'", self.js)
        self.assertIn(f"document.title = Overlay.hud ? '{app_window.HUD_TITLE}'", self.html)   # chat titles never rename it


class IndicatorTests(unittest.TestCase):
    def test_polls_about_once_a_second_only_while_visible(self):
        js = block(page(), "const NetIndicator = (() => {", "/* ---- The Studio Overlay")
        self.assertIn("setTimeout(poll, 1000)", js)
        self.assertIn("visible() && wanted()", js)
        self.assertIn("visibilitychange", js)
        self.assertNotIn("speed-test", js)                                        # the speed test is manual only


class AppWindowTests(unittest.TestCase):
    def test_app_args_open_only_the_local_page(self):
        args = app_window.app_args("edge.exe", "http://127.0.0.1:8000", size=(1440, 920))
        self.assertEqual(args[1], "--app=http://127.0.0.1:8000")
        self.assertIn(f"--user-data-dir={app_window.PROFILE}", args)
        self.assertTrue(app_window.PROFILE.startswith(os.path.join(HERE, "data")))
        self.assertIn("--force-webrtc-ip-handling-policy=disable_non_proxied_udp", args)
        self.assertTrue(app_window.APP_URL.startswith("http://127.0.0.1:"))

    def test_falls_back_to_the_browser(self):
        with mock.patch.object(app_window, "find_browser", return_value=None), \
             mock.patch("webbrowser.open") as opened:
            self.assertEqual(app_window.open_app("http://127.0.0.1:8000"), "browser")
        opened.assert_called_once_with("http://127.0.0.1:8000")

    def test_overlay_sync_off_starts_nothing(self):
        with mock.patch.object(app_window._overlay, "start") as start, \
             mock.patch.object(app_window, "find_hud", return_value=None):
            app_window.overlay_sync({"overlay_window": False})
        start.assert_not_called()

    def test_overlay_sync_on_starts_the_follower(self):
        if not app_window.WINDOWS:
            self.skipTest("Windows only")
        with mock.patch.object(app_window._overlay, "start") as start:
            app_window.overlay_sync({"overlay_window": True, "overlay_opacity": .5, "overlay_clickthrough": True})
        start.assert_called_once()
        self.assertEqual(app_window._overlay.settings, {"opacity": .5, "click_through": True})

    def test_run_bat_opens_the_app_window_and_keeps_the_tab(self):
        with open(os.path.join(HERE, "run.bat"), encoding="utf-8") as f:
            bat = f.read()
        self.assertIn("app_window.py --wait-and-open", bat)
        self.assertIn("app_window.py --open", bat)
        self.assertIn("Start-Process 'http://127.0.0.1:%APP_PORT%'", bat)             # APP_WINDOW=0: a browser tab
        self.assertIn('--host 127.0.0.1 --port %APP_PORT%', bat)                       # the server is still local-only


if __name__ == "__main__":
    unittest.main()
