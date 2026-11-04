"""1.4.6 stage 5 - the approved designs built: the unslop fixes, the AI looks from their design files (with the
originals one switch away in Extras) and each look's own thinking mark."""
import unittest

from core import config

PAGE = open("index.html", encoding="utf-8").read()
SERVER = open("api/server.py", encoding="utf-8").read()


class AiLooksTests(unittest.TestCase):
    def test_new_by_default_original_in_extras(self):
        self.assertEqual(config.DEFAULT_SETTINGS["ai_looks"], "new")
        self.assertEqual(config.clean_privacy_settings({"ai_looks": "original"})["ai_looks"], "original")
        self.assertEqual(config.clean_privacy_settings({"ai_looks": "nonsense"})["ai_looks"], "new")
        self.assertIn("ai_looks: str | None = None", SERVER)
        self.assertIn("if (s.ai_looks === 'original') root.dataset.aiClassic = '1'; else delete root.dataset.aiClassic;", PAGE)
        extras = PAGE[PAGE.index("<summary>Extras <small>"):]
        extras = extras[:extras.index("</details>`;")]
        self.assertIn("rowHtml('Looks redrawn in 1.4.6',", extras)
        self.assertIn(':root[data-skin="eclipse"]:not([data-tone="light"]):not([data-ai-classic]) {', PAGE)                    # Eclipse way 1
        self.assertIn("seg([['auto', 'New'], ['orbs', 'Orbs'], ['legacy', 'Legacy']]", extras)   # the marks from before

    def test_each_new_look_steps_aside_for_the_original(self):
        for sel in (':root[data-skin="clay"][data-tone="light"]:not([data-ai-classic])',
                    ':root[data-skin="mono"][data-tone="light"]:not([data-ai-classic])',
                    ':root[data-skin="void"]:not([data-tone="light"]):not([data-ai-classic])',
                    ':root[data-skin="prism"][data-tone="light"]:not([data-ai-classic])'):
            self.assertIn(sel + " {", PAGE)
        block = PAGE[PAGE.index("/* ---- The AI looks from their public design files"):PAGE.index(':root[data-skin="mono"][data-tone="light"] .lv {')]
        self.assertNotIn("url(", block)                                  # no logos, no web fonts
        self.assertNotIn("@font-face", block)


class MarksTests(unittest.TestCase):
    def test_the_claude_mark_switch_shows_in_the_claude_look(self):
        self.assertIn("${fam === 'clay' ? `<div class=\"row tm-row\"><div class=\"row-text\"><b>Claude's thinking mark</b>", PAGE)
        self.assertIn("seg([['regular', 'Regular'], ['liquid', 'Liquid']]", PAGE)

    def test_marks_keep_still_when_motion_is_off(self):
        self.assertIn("if (md === 'marks') paintMark(el, o, hold ? 1.1 : t, now, hold);", PAGE)
        self.assertIn("if (still) { const sig = act + key + root.dataset.scheme + w + h; if (o.drawn === sig) return; o.drawn = sig; }", PAGE)


class UnslopBuiltTests(unittest.TestCase):
    def test_places(self):
        self.assertIn(".sb-group { padding: 12px 8px 3px; font-size: 12px; font-weight: 500; color: var(--muted); }", PAGE)
        self.assertIn(".place-wrap:has(.place-empty .btn.primary) .place-head .place-acts .btn.primary { display: none; }", PAGE)
        self.assertIn(":hover > .iv-tag, :focus-visible > .iv-tag { opacity: 1; }", PAGE)
        self.assertIn("#iv-prompt, #vv-prompt {", PAGE)
        self.assertIn("$('chat-title').textContent = '';", PAGE)

    def test_settings(self):
        self.assertIn(':root[data-ui="new"] .set-group { border-radius: 0; }', PAGE)
        self.assertIn(".sm-cards { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }", PAGE)
        self.assertIn(".theme-cards.families { grid-template-columns: repeat(5, minmax(0, 1fr));", PAGE)
        self.assertIn("Not on this PC <small>· ${gone.length} listed in models.ini whose file is gone</small>", PAGE)
        self.assertIn("const dataPath = p =>", PAGE)
        self.assertIn("rowHtml('Check now', 'Asks GitHub right away.'", PAGE)
        self.assertIn(".pill.vision { color: var(--info-text); }", PAGE)
        from connect import connectors
        self.assertFalse([c for c in connectors.CATALOG if "official MCP" in c.get("about", "")] if hasattr(connectors, "CATALOG") else [])


class HomeChatAndLooksTests(unittest.TestCase):
    """The first /design rows: Home, a chat, Aurora, Sunrise, Classic dark, ChatGPT's chips (built 4 Nov)."""

    def test_home_and_chat(self):
        self.assertIn("const html = false ? ", PAGE)                                      # no Continue list
        self.assertIn("promptEl.placeholder = show && S.transcript.length ? 'Reply…' : askText();", PAGE)
        self.assertIn("eff === 'Balanced' || (eff === 'Auto' && model === 'Auto') ? '' : eff", PAGE)
        self.assertIn("(S.mode === 'local' ? 'off' : 'Local models off · cloud first')", PAGE)
        self.assertIn('.home-credit { display: none !important; }', PAGE)
        self.assertIn(':root[data-ui="new"] :is(#pins-btn, #mini-btn, #stats-btn) { display: none !important; }', PAGE)
        self.assertIn("stats: () => SessionStats.open(), mini: () => $('mini-btn').click(),", PAGE)

    def test_looks_under_the_original_switch(self):
        for sel in (':root[data-skin="aurora"]:not([data-ai-classic])', ':root:is([data-skin="dawn"], [data-skin="dusk"]):not([data-ai-classic]) .halo { opacity: 0; }',
                    ':root[data-skin="classic"][data-scheme="dark"]:not([data-ai-classic]) .chip .chip-ic', ':root[data-skin="mono"]:not([data-ai-classic]) #composer-chips'):
            self.assertIn(sel, PAGE)

    def test_creating_image_and_building_cards(self):
        self.assertIn("const DRAW_FX = (() => {", PAGE)
        for key in ("classic", "sunrise", "terminal", "eclipse", "aurora", "clay", "mono", "void", "prism"):
            self.assertIn(f"  {key}(c, p", PAGE)
        self.assertIn("DRAW_FX.play(turn.drawEl);", PAGE)
        self.assertIn("if (el.querySelector('canvas.dc-fx')) DRAW_FX.play(el);", PAGE)
        self.assertIn("const what = L === 'chart' ? 'Building a chart'", PAGE)

    def test_globe_opens_ops_and_relocks(self):
        self.assertIn("if (mode !== 'ops' && root.dataset.secret === 'mossad' && runId !== autoOpsRun && runId !== opsClosedRun)", PAGE)
        self.assertIn("function relock(layer) {", PAGE)
        self.assertIn("data[id] = d; relock(id); schedule();", PAGE)

    def test_saved_orbs_moves_to_the_new_marks_once(self):
        import json, os, tempfile
        from unittest.mock import patch
        path = os.path.join(tempfile.mkdtemp(), "settings.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"think_mark": "orbs"}, fh)
        with patch.object(config, "SETTINGS_PATH", path):
            s = config.load_settings()
        self.assertEqual((s["think_mark"], s["marks_146"]), ("auto", True))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"think_mark": "orbs", "marks_146": True}, fh)            # chosen again after 1.4.6: kept
        with patch.object(config, "SETTINGS_PATH", path):
            self.assertEqual(config.load_settings()["think_mark"], "orbs")

class GpuMemoryOptionsTests(unittest.TestCase):
    """Out of GPU memory isn't a dead end (asked 4 Nov): run on RAM too, or see what holds the GPU."""

    def test_the_error_card_offers_both(self):
        self.assertIn("'Not enough GPU memory for this model',", PAGE)
        self.assertIn("['ramsplit', 'vram', 'models']],", PAGE)
        self.assertIn("ramsplit: ['i-layers2', 'Run on RAM too'], vram: ['i-activity', 'What’s using the GPU'],", PAGE)
        self.assertIn("async function showVram(card, btn) {", PAGE)
        self.assertIn("<summary>More detail</summary>", PAGE)

    def test_ram_split_endpoint_sets_fit_and_is_phone_blocked(self):
        from unittest.mock import patch
        from api import server
        from connect import remote_access
        self.assertIn(("POST", "/api/local-models/ram-split"), remote_access.PHONE_BLOCKED)
        # set_model_options returns its backup's path on success - the mock used to return None, which hid that the
        # endpoint read the path as an error (1.4.6 reliability; tests/test_reliability_146.py runs it for real).
        with patch("llama_local.list_model_configs", return_value=[{"name": "M", "model_path": __file__}]), \
                patch("llama_local.set_model_options", return_value="models.ini.bak-1") as setopts, \
                patch("llama_local.unload_all"), patch("diagnostics.meta_for", return_value={"arch": "llama"}), \
                patch.object(server.tasks.turns, "running", return_value=False):
            r = server.ram_split(server.RamSplitRequest(model="M"))
        self.assertTrue(r["ok"], r)
        self.assertEqual(setopts.call_args[0][0]["M"]["ngl"], "auto")

    def test_apps_grouped_by_name_biggest_first(self):
        from unittest.mock import patch
        from pc import gpu_apps
        with patch.object(gpu_apps, "_counters", return_value=[(1, 100), (2, 300), (3, 50)]), \
                patch("psutil.Process", side_effect=lambda pid: type("P", (), {"name": lambda self: {1: "chrome.exe", 2: "dwm.exe", 3: "chrome.exe"}[pid]})()):
            r = gpu_apps.snapshot()
        self.assertEqual([(a["name"], a["bytes"]) for a in r["apps"]], [("dwm", 300), ("chrome", 150)])
        self.assertIn("graphics acceleration", r["apps"][1]["tip"])


class PromptBoxTests(unittest.TestCase):
    def test_no_scrollbar_until_full_and_refit_on_resize(self):
        self.assertIn("promptEl.style.overflowY = h > PROMPT_MAX ? 'auto' : 'hidden';", PAGE)
        self.assertIn("box-sizing: border-box; overflow-y: hidden;", PAGE)
        self.assertIn("window.addEventListener('resize', () => { if (promptEl.value) autoGrow(); });", PAGE)
        self.assertIn('.welcome::after { content: ""; margin-top: auto; flex: none; }', PAGE)


if __name__ == "__main__":
    unittest.main()
