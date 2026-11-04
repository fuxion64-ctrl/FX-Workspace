"""Model downloads can be paused and resumed (1.4.6, stage 3.3): a pause ends as "paused", keeps what's on disk,
and remembers the file so Resume can start it again; a cancel still ends as "cancelled"."""
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "llm"))
import model_finder  # noqa: E402


class PauseResume(unittest.TestCase):
    def setUp(self):
        model_finder._state.update(state="idle", error="")
        model_finder._cancel.clear()
        model_finder._paused.clear()

    def _run_interrupted(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(model_finder, "_fetch", side_effect=InterruptedError("stopped")), \
                mock.patch.object(model_finder.netlayer, "client", mock.MagicMock()):
            model_finder._run("o/r", d, [{"path": "m.gguf", "size": 1}], None)

    def test_pause_needs_a_running_download(self):
        self.assertFalse(model_finder.pause()["ok"])
        self.assertFalse(model_finder._cancel.is_set())

    def test_pause_ends_as_paused(self):
        model_finder._state["state"] = "running"
        self.assertTrue(model_finder.pause()["ok"])
        self.assertTrue(model_finder._cancel.is_set())
        self._run_interrupted()
        self.assertEqual(model_finder.status()["state"], "paused")

    def test_cancel_ends_as_cancelled(self):
        model_finder._state["state"] = "running"
        model_finder.cancel()
        self._run_interrupted()
        self.assertEqual(model_finder.status()["state"], "cancelled")


class Wiring(unittest.TestCase):
    def test_endpoint_and_buttons(self):
        with open(os.path.join(ROOT, "api", "extras.py"), encoding="utf-8") as f:
            self.assertIn('"/api/model-finder/pause"', f.read())
        with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as f:
            html = f.read()
        for bit in ("function finderPause()", "function finderResume()", "MB/s · about", "'paused'"):
            self.assertIn(bit, html)
        with open(os.path.join(ROOT, "llm", "model_finder.py"), encoding="utf-8") as f:
            self.assertTrue(re.search(r"path=path, mmproj=bool\(with_mmproj\)", f.read()))

    def test_added_image_model_gets_a_test_picture(self):
        with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as f:
            html = f.read()
        body = html[html.index("async function testImageModel("):][:1600]
        self.assertIn("testImageModel(r.model);", html)
        self.assertIn("quality: m.key", body)              # drawn with the new model itself
        self.assertIn("/trash`", body)                     # kept out of the gallery, restorable
        self.assertIn("removeImageModel(", body)           # a failure offers taking it off the list


if __name__ == "__main__":
    unittest.main()
