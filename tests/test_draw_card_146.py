"""The picture being drawn in a chat (1.4.6 feedback): the animation floats in its own frame shaped like the picture,
with one tiny line (step count, model picker, Stop) instead of the old card; the live steps row stays closed."""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as f:
    HTML = f.read()


class DrawCard(unittest.TestCase):
    def body(self):
        i = HTML.index("function paintChatDraw(s)")
        return HTML[i:HTML.index("async function load()", i)]

    def test_floating_frame_not_the_old_card(self):
        b = self.body()
        self.assertIn('class="dc-meta"', b)
        self.assertIn("dc-prog", b)
        self.assertNotIn("dc-top", b)                       # the old "Drawing / bar / About N s left" body is gone
        self.assertNotIn('class="bar"', b)
        self.assertIn("pic.style.width", b)                 # framed like the picture (its aspect ratio)
        css = HTML[HTML.index(".draw-card {"):][:400]
        self.assertNotIn("background: var(--surface); overflow: hidden; margin-top", css)

    def test_model_picker_for_the_next_pictures(self):
        b = self.body()
        self.assertIn("ImagesView.pickModel(this)", b)
        self.assertIn("saveSettings({ image_quality: b.dataset.q })", b)
        self.assertRegex(HTML, r"stop, unload, set, pickModel,")

    def test_live_steps_row_does_not_open_by_itself(self):
        i = HTML.index("function buildSteps(events, live)")
        self.assertIsNone(re.search(r"live \? ' open'", HTML[i:i + 400]))


class Feedback4Nov(unittest.TestCase):
    def test_crowded_one_line_box_takes_two_rows(self):
        self.assertIn("function composerCrowded(box)", HTML)
        self.assertIn("|| composerCrowded(box);", HTML)

    def test_no_char_rate_while_thinking(self):
        i = HTML.index("const writing = !!splitThink(c.content).answer.trim();")
        self.assertIn(": 'thinking';", HTML[i:i + 300])

    def test_one_mark_per_reply(self):
        self.assertIn(".msg-assistant .think[data-still] { display: none !important; }", HTML)
        i = HTML.index("function oneMover(")
        self.assertIn("closest('.research-progress')", HTML[i:i + 700])

    def test_dot_field_fills_the_frame(self):
        i = HTML.index("  mono(c, p, e, t) {")
        body = HTML[i:i + 2600]
        self.assertIn("x = q * sx + sx / 2, y = r * sy + sy / 2", body)
        self.assertNotIn("20 + q * sx", body)

    def test_claude_shimmer_is_a_mask_not_clipped_text(self):
        self.assertIn("content: attr(data-t)", HTML)
        self.assertIn("function setTp(el, text)", HTML)


class Dictation(unittest.TestCase):
    def test_empty_send_dictates_in_the_chatgpt_look(self):
        i = HTML.index("const Dictate = (() => {")
        body = HTML[i:i + 9000]
        self.assertIn("['mono', 'void'].includes(document.documentElement.dataset.skin)", body)   # ChatGPT and Grok only
        self.assertIn("'/api/voice/transcribe'", body)                      # transcribed on this PC
        self.assertIn("promptEl.value = ", body)                            # into the box, not sent
        self.assertIn("else if (Dictate.wanted()) Dictate.toggle();", HTML)
        self.assertIn('<symbol id="i-dictate"', HTML)
        self.assertIn("redock: applyDock, wav16k, pcm16k,", HTML)

    def test_live_words_and_a_level_meter(self):
        i = HTML.index("const Dictate = (() => {")
        body = HTML[i:i + 9000]
        self.assertIn("/api/voice/live?sid=", body)          # the words as you speak
        self.assertIn("function fullPreview()", body)         # ...or, without the live recognizer, every ~1.5 s
        self.assertIn("dict-meter", body)                     # the level meter and timer
        self.assertIn("['en', 'auto'].includes(lang)", body)
        self.assertIn(".dict-meter i {", HTML)


class Feedback4NovLate(unittest.TestCase):
    def test_box_does_not_flip_back_and_forth(self):
        i = HTML.index("function autoGrow()")
        body = HTML[i:i + 1800]
        self.assertIn("box._openLen * 0.9", body)             # closes only when clearly shorter
        self.assertIn("if (el !== promptEl) ro.observe(el)", HTML)   # the field's own fitting doesn't re-trigger it

    def test_finish_plays_out_before_the_card_goes(self):
        self.assertIn("el._target = 1.15", HTML)
        self.assertIn("Math.min(1.15, o.shown)", HTML)
        self.assertIn(".draw-card.dc-out", HTML)

    def test_tour_after_the_welcome(self):
        self.assertIn("const Tour = (() => {", HTML)
        for at in ("'.sb-user'", "'.sb-quit'", "'.brand'"):
            self.assertIn(at, HTML[HTML.index("const Tour = (() => {"):][:900])
        self.assertIn("Tour.start(again)", HTML)


if __name__ == "__main__":
    unittest.main()
