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


if __name__ == "__main__":
    unittest.main()
