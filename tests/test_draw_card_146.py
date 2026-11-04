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


if __name__ == "__main__":
    unittest.main()
