"""1.4.4 Stage 1 fixes."""
import unittest
from unittest.mock import patch

import history

import core.state as shared
from agent import models

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def convo(image_turn_is_last: bool):
    old = [{"role": "system", "content": "sys"},
           {"role": "user", "content": [{"type": "text", "text": "what is this?"}, IMG]},
           {"role": "assistant", "content": "A cat."}]
    if image_turn_is_last:
        return old
    return old + [{"role": "user", "content": "write me a poem"}]


class EarlierImagesTests(unittest.TestCase):
    """An image from an earlier turn no longer swaps a text model for a vision one on every message."""

    def test_this_turn(self):
        self.assertTrue(history.images_this_turn(convo(True)))
        self.assertFalse(history.images_this_turn(convo(False)))

    def test_stub_keeps_text_and_drops_only_earlier_images(self):
        out = history.stub_earlier_images(convo(False))
        self.assertFalse(history.has_images(out))
        self.assertIn("what is this?", out[1]["content"])
        self.assertIn("not sent", out[1]["content"])
        self.assertTrue(history.has_images(history.stub_earlier_images(convo(True))))

    def test_text_model_stays_for_an_old_image(self):
        with patch.dict(shared.app_state, {"local_engaged": "Qwen-3.5-4B-Q4_K_M"}), \
                patch.object(models.llama_local, "is_vision_model", side_effect=lambda n: "Vision" in n):
            msgs, vision = models._local_payload(convo(False))
            self.assertFalse(vision)
            self.assertFalse(history.has_images(msgs))
            self.assertTrue(models._local_payload(convo(True))[1])         # a new image still escalates

    def test_loaded_vision_model_keeps_old_images(self):
        with patch.dict(shared.app_state, {"local_engaged": "Gemma-4-Vision-Q8"}), \
                patch.object(models.llama_local, "is_vision_model", side_effect=lambda n: "Vision" in n):
            msgs, vision = models._local_payload(convo(False))
            self.assertTrue(vision)
            self.assertTrue(history.has_images(msgs))


class ResearchRetryTests(unittest.TestCase):
    def test_retry_words_pick_up_the_last_research(self):
        from agent import research
        for text in ("retry", "Retry.", "try again", "resume", "continue the research", "redo it", "keep going"):
            self.assertTrue(research.wants_resume(text), text)
        for text in ("retry logic in python", "continue reading about Rome", "again and again by Bowie"):
            self.assertFalse(research.wants_resume(text), text)


class LoadingNoticeTests(unittest.TestCase):
    def test_a_reply_queued_behind_a_load_says_loading(self):
        import local_runtime
        ev = {"status": "running", "phase": None}
        with patch.dict(shared.app_state, {"running_model_event": ev}), \
                patch.dict(local_runtime.loading, {"model": "Qwen-3.5-4B-Q4_K_M", "since": 100.0}), \
                patch.object(models.run_control, "checkpoint"), \
                patch.object(models.event_log, "update_event", side_effect=lambda e, **k: e.update(k)):
            models._queued_wait()
        self.assertEqual((ev["phase"], ev["loading"], ev["loading_since"]), ("loading", "Qwen-3.5-4B-Q4_K_M", 100.0))


class CalmMotionTests(unittest.TestCase):
    page = open("index.html", encoding="utf-8").read()

    def test_phrases_and_marks_change_only_for_a_reason(self):
        self.assertNotIn("claySay(el) || pick(el._phrase)", self.page)          # no new phrase every 2.5 s
        self.assertIn("el._want = act; clearTimeout(el._wantT);", self.page)      # a change must last 0.6 s
        self.assertNotIn("(i + t / 1.2) % 3", self.page)                          # no colour cycling

    def test_one_moving_signal_at_a_time(self):
        # Measured live in every appearance (harness anim.js, 29 Oct 2026): during a reply only the thinking mark moves
        # (plus the Claude look's shimmer); these used to loop alongside it.
        p = self.page
        self.assertNotIn("animation: if-float 3.6s", p)                       # the incognito ghost's loops
        self.assertNotIn("animation: ml-pulse", p)                            # the model line's dot
        self.assertNotIn("'<span class=\"spinner\"></span>' : icon({ waiting", p)  # the top bar chip's spinner
        self.assertNotIn('[data-state="working"] .halo { opacity: .8; animation: halo-breathe', p)
        self.assertNotIn("animation: corona 9s linear infinite running", p)
        self.assertIn("SPEED = { idle: 1, working: 1, waiting: 1", p)          # Aurora's sky isn't a status light
        self.assertNotIn("skin === 'terminal' && state === 'working' ? 'scan'", p)
        self.assertIn("animation: clay-shimmer 3s linear infinite", p)         # the Claude look keeps its shimmer

    def test_icon_less_sites_get_a_coloured_tile(self):
        self.assertIn('style="--fh:${hue}"', self.page)
        self.assertIn("background: hsl(var(--fh, 220) 38% 44%)", self.page)


class PageScriptTests(unittest.TestCase):
    def test_the_page_script_parses(self):
        # A second top-level `const fmtBytes` stopped the whole page from loading (1.4.4, caught live): every inline
        # script must compile, which also catches a name declared twice.
        import shutil
        import subprocess
        if not shutil.which("node"):
            self.skipTest("node isn't installed")
        js = ("const s=require('fs').readFileSync('index.html','utf8');let bad=[];"
              "for(const m of s.matchAll(/<script(?![^>]*src)[^>]*>([\\s\\S]*?)<\\/script>/g)){try{new Function(m[1])}catch(e){bad.push(e.message)}}"
              "console.log(JSON.stringify(bad))")
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)

    def test_settings_pages_for_tor_and_data(self):
        page = CalmMotionTests.page
        self.assertIn("['data', 'Data and files', 'i-folder']", page)
        self.assertIn("data: renderDataFiles", page)
        self.assertIn("function hideIpGroup(d)", page)       # 1.4.6: Tor's rows became one "Hide my IP" choice

    def test_the_tor_tag_says_what_goes_through_tor(self):
        from web import netlayer
        base = {"net_route": "direct", "net_components": {"search": "app", "models": "app", "other": "app"}}
        cases = [({}, None), ({"net_route": "tor"}, "all"), ({"net_components": {"search": "tor", "models": "app", "other": "app"}}, "search"),
                 ({"net_route": "tor", "net_components": {"search": "app", "models": "direct", "other": "app"}}, "search")]
        for change, want in cases:
            with patch.object(netlayer, "settings", return_value={**base, **change}), patch.dict(netlayer._state, {"session_direct": False}):
                self.assertEqual(netlayer.tor_use(), want, change)
        with patch.object(netlayer, "settings", return_value={**base, "net_route": "tor"}), patch.dict(netlayer._state, {"session_direct": True}):
            self.assertIsNone(netlayer.tor_use())                  # "Go direct for this session" - no tag
        self.assertIn('id="tor-tag"', CalmMotionTests.page)

    def test_one_switch_stops_loading_while_typing(self):
        # 1.4.6 (a user's request): "Get models ready while I type" off - typing loads nothing, chat or pictures.
        from core import config
        self.assertTrue(config.clean_privacy_settings({})["chat_warmup"])
        self.assertFalse(config.clean_privacy_settings({"chat_warmup": False})["chat_warmup"])
        self.assertIn("S.settings?.chat_warmup === false", CalmMotionTests.page)
        self.assertIn("saveSettings({chat_warmup:${s.chat_warmup === false}, image_warmup:", CalmMotionTests.page)


if __name__ == "__main__":
    unittest.main()
