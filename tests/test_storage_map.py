"""Settings > Storage (1.4.6): each drive and what of the app's is on it - every file counted once."""
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from base import storage_map

PAGE = open("index.html", encoding="utf-8").read()


def _file(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


class StorageMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        t = self.tmp
        self.app = os.path.join(t, "app")
        self.models = os.path.join(t, "llama", "models")
        _file(os.path.join(self.models, "used.gguf"), 4000)
        _file(os.path.join(self.models, "used.mmproj.gguf"), 500)
        _file(os.path.join(self.models, "old.gguf"), 3000)           # in the folder, not in models.ini
        _file(os.path.join(t, "llama", "llama-server.exe"), 700)
        _file(os.path.join(t, "llama", "ggml.dll"), 300)
        _file(os.path.join(t, "Images", "flux.gguf"), 2000)
        _file(os.path.join(self.app, "main.py"), 100)
        _file(os.path.join(self.app, "data", "images", "a.png"), 50)
        _file(os.path.join(self.app, "data", "chats.json"), 20)
        _file(os.path.join(self.app, ".venv", "lib.py"), 200)
        cfg = [{"name": "Used", "model_path": os.path.join(self.models, "used.gguf"),
                "mmproj_path": os.path.join(self.models, "used.mmproj.gguf")}]
        self.patches = [patch.object(storage_map.paths, "APP_DIR", self.app),
                        patch.object(storage_map.paths, "llama_dirs", return_value=(os.path.join(t, "llama"),)),
                        patch.object(storage_map.paths, "prism_dirs", return_value=()),
                        patch.object(storage_map.paths, "model_dirs", return_value=[self.models]),
                        patch.object(storage_map, "_feature_roots", return_value=[("pictures", "Picture models", "", os.path.join(t, "Images"))]),
                        patch("llama_local.list_model_configs", return_value=cfg)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_every_part_and_each_file_once(self):
        r = storage_map.build()
        parts = {p["name"]: p for d in r["drives"] for p in d["parts"]}
        self.assertEqual(parts["Chat models"]["bytes"], 4500)                    # the model and its vision file
        self.assertEqual(parts["Chat models"]["items"], [{"name": "Used", "bytes": 4500}])
        self.assertEqual(parts["Model files not in models.ini"]["bytes"], 3000)  # the old download, once
        self.assertEqual(parts["llama.cpp engine"]["bytes"], 1000)               # not the models next to it
        self.assertEqual(parts["Picture models"]["bytes"], 2000)
        self.assertEqual(parts["Pictures you made"]["bytes"], 50)
        self.assertEqual(parts["Chats, memory and the rest"]["bytes"], 20)
        self.assertEqual(parts["The app's Python"]["bytes"], 200)
        self.assertEqual(parts["The app"]["bytes"], 100)
        d = r["drives"][0]
        self.assertEqual(d["app"], sum(p["bytes"] for p in d["parts"]))
        self.assertEqual(d["other"], max(0, d["used"] - d["app"]))

    def test_the_page_never_waits_for_a_measure(self):
        storage_map._cache.update(at=0.0, result=None)
        storage_map._state["measuring"] = False
        with patch.object(storage_map, "build", side_effect=lambda: (time.sleep(.3), {"ok": True, "drives": [], "at": 1})[1]):
            first = storage_map.snapshot()
            self.assertTrue(first["measuring"])
            for _ in range(40):
                if not storage_map._state["measuring"]:
                    break
                time.sleep(.05)
            self.assertFalse(storage_map.snapshot()["measuring"])

    def test_on_its_own_settings_page(self):
        self.assertIn("['storage', [['storage', 'Storage']]],", PAGE)
        self.assertIn("['storage', 'Storage', 'i-disk']", PAGE)
        self.assertIn("storage: renderStorage,", PAGE)
        self.assertIn("async function renderStorage(force = false) {", PAGE)
        self.assertIn('@app.get("/api/storage/map")', open("api/server.py", encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
