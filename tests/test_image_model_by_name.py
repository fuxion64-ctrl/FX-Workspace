"""A picture model named in the chat is matched to one on this PC (1.4.6): generate_image's `model`."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "media"))
import image_gen  # noqa: E402

HAVE = {"fast", "balanced", "dreamshaper8", "pony6", "pony6_anime"}


class ByName(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(image_gen, "installed", side_effect=lambda k, root=None: k in HAVE)
        p.start(); self.addCleanup(p.stop)

    def test_names_as_people_say_them(self):
        r = image_gen.resolve_model
        self.assertEqual(r("DreamShaper"), "dreamshaper8")
        self.assertEqual(r("dreamshaper 8"), "dreamshaper8")
        self.assertEqual(r("z image"), "balanced")          # Z-Image-Turbo is Balanced's model
        self.assertEqual(r("fast"), "fast")
        self.assertEqual(r("Pony"), "pony6")
        self.assertEqual(r("pony anime clean"), "pony6_anime")

    def test_not_installed_or_unknown_is_none(self):
        self.assertIsNone(image_gen.resolve_model("Animagine"))   # in the catalogue, not on this PC
        self.assertIsNone(image_gen.resolve_model("midjourney"))
        self.assertIsNone(image_gen.resolve_model(""))

    def test_tool_lists_the_models_instead_of_guessing(self):
        with mock.patch.object(image_gen, "generate") as gen:
            out = image_gen.generate_image_tool({"prompt": "a cat", "model": "midjourney"})
        gen.assert_not_called()
        self.assertFalse(out["ok"])
        self.assertEqual(out["code"], "unknown_model")
        self.assertIn("DreamShaper 8", out["message"])

    def test_tool_draws_with_the_named_model(self):
        fake = {"id": "i", "url": "u", "model": "DreamShaper 8", "seed": 1, "width": 512, "height": 512, "seconds": 3}
        with mock.patch.object(image_gen, "generate", return_value=fake) as gen:
            out = image_gen.generate_image_tool({"prompt": "a cat", "model": "DreamShaper", "quality": "best"})
        self.assertTrue(out["ok"])
        self.assertEqual(gen.call_args[0][1], "dreamshaper8")    # the named model wins over a guessed quality

    def test_schema_offers_it(self):
        import tools.schemas as s
        fn = next(t["function"] for t in s.TOOLS if t["function"]["name"] == "generate_image")
        self.assertIn("model", fn["parameters"]["properties"])


if __name__ == "__main__":
    unittest.main()
