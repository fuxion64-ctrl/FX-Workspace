"""Ternary Bonsai 2 on PrismML's llama.cpp fork (ARCHITECTURE D29): the
PrismML-only packings are recognised, the running build is read from the
router, a load on a stock server says why it failed, and the launcher lets a
model keep its own context window without changing any other model's."""
import os
import unittest
from unittest.mock import patch

import requests

import llama_local

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _has_bonsai() -> bool:
    """These tests read this PC's own models.ini (git-ignored): a fresh checkout has none, or one without Bonsai
    (pointed out in PR #1 by 123timsolotech) - they're skipped there instead of failing."""
    try:
        with open(os.path.join(HERE, "models.ini"), encoding="utf-8", errors="replace") as f:
            return "[Ternary-Bonsai-2-27B]" in f.read()
    except OSError:
        return False


NEEDS_BONSAI = unittest.skipUnless(_has_bonsai(), "needs a models.ini with Ternary-Bonsai-2-27B (this PC's own)")


def _status(exe):
    return llama_local._router_status({"data": [{"id": "Ternary-Bonsai-2-27B", "status": {
        "value": "unloaded", "args": [exe, "--ctx-size", "16384", "--model", "x-PTQ1_0.gguf"]}}]})


class PrismDetectionTests(unittest.TestCase):
    def setUp(self):
        saved = dict(llama_local._cache)
        self.addCleanup(llama_local._cache.update, saved)

    def test_prism_only_packings_are_recognised(self):
        self.assertTrue(llama_local.needs_prism("Ternary-Bonsai-2-27B-PTQ1_0.gguf"))
        self.assertTrue(llama_local.needs_prism("Ternary-Bonsai-2-27B-PQ2_0"))
        if _has_bonsai():
            self.assertTrue(llama_local.needs_prism("Ternary-Bonsai-2-27B"))      # via its models.ini path
        self.assertFalse(llama_local.needs_prism("Qwen-3-8B-Q5_K_M"))
        self.assertFalse(llama_local.needs_prism("Mistral-7B"))

    @NEEDS_BONSAI
    def test_the_running_build_comes_from_the_router(self):
        llama_local._cache["status"] = _status(r"E:\ClankerLocalCpp\prism\llama-server.exe")
        self.assertEqual(llama_local.router_status("Ternary-Bonsai-2-27B")["ctx"], 16384)
        self.assertTrue(llama_local.is_prism_server())
        self.assertEqual(llama_local.load_hint("Ternary-Bonsai-2-27B"), "")
        llama_local._cache["status"] = _status(r"E:\ClankerLocalCpp\llama-server.exe")
        self.assertFalse(llama_local.is_prism_server())
        hint = llama_local.load_hint("Ternary-Bonsai-2-27B")
        self.assertIn("PrismML", hint)
        self.assertIn(r"E:\ClankerLocalCpp\llama-server.exe", hint)
        self.assertEqual(llama_local.load_hint("Mistral-7B"), "")               # nothing to add for stock models
        llama_local._cache["status"] = {}
        self.assertIsNone(llama_local.is_prism_server())

    @NEEDS_BONSAI
    def test_a_failed_load_on_a_stock_server_says_why(self):
        llama_local._cache["status"] = _status(r"E:\ClankerLocalCpp\llama-server.exe")
        resp = requests.Response()
        resp.status_code = 500
        resp.url = "http://127.0.0.1:8090/v1/chat/completions"
        # force_load first moves the router to the build the model needs (llama_builds.ensure_for): unpatched, this test
        # really restarted a running llama-server on PrismML's build, left one on 8090 and kept the run from exiting
        # (1.4.7). The switch has its own tests (test_llama_builds); here only the message is checked.
        import llama_builds
        stub = patch.object(llama_builds, "ensure_for", return_value=None)
        stub.start()
        self.addCleanup(stub.stop)
        with patch.object(llama_local.requests, "post", return_value=resp):
            with self.assertRaises(RuntimeError) as cm:
                llama_local.force_load("Ternary-Bonsai-2-27B", timeout=1)
        self.assertIn("PrismML", str(cm.exception))
        with patch.object(llama_local.requests, "post", return_value=resp):
            with self.assertRaises(requests.HTTPError):                          # other models: unchanged
                llama_local.force_load("Mistral-7B", timeout=1)


class LauncherConfigTests(unittest.TestCase):
    def test_router_has_no_global_context_and_starts_the_newest_build(self):
        # D51: the newest stock llama.cpp starts; the app moves the router to PrismML's fork for Bonsai (llama_builds).
        with open(os.path.join(HERE, "run.bat"), encoding="utf-8", errors="replace") as f:
            bat = f.read()
        start = next(l for l in bat.splitlines() if "--models-preset" in l)
        self.assertNotIn("--ctx-size", start)      # it would override every model's own ctx-size
        self.assertLess(bat.index(r'if exist "E:\ClankerLocalCpp\llama-server.exe"'),
                        bat.index(r"E:\ClankerLocalCpp\prism\llama-server.exe"))

    @NEEDS_BONSAI
    def test_every_model_has_its_own_window_and_bonsai_fits_8gb(self):
        # The windows come from Settings > Diagnostics (diagnostics.py): each model's largest context that
        # fits an 8 GB card, never past what it was trained on, never below 16k; [*] lets llama.cpp's --fit
        # place the layers at load time instead of pinning ngl = 99 (which spills into shared memory).
        import diagnostics
        with patch.dict(os.environ, {"MODELS_INI": os.path.join(HERE, "models.ini")}):
            llama_local._ini_cache.update(key=None, configs=None)
            configs = {c["name"]: c for c in llama_local.list_model_configs()}
            defaults = llama_local.model_options("*")
            self.assertEqual((defaults["ngl"], defaults["fit-target"]), ("auto", str(diagnostics.MARGIN_MB)))
            bonsai = configs["Ternary-Bonsai-2-27B"]
            self.assertTrue(bonsai["model_path"].lower().endswith("ptq1_0.gguf"))
            self.assertEqual(bonsai["ctx_size"], 16384)
            opts = llama_local.model_options("Ternary-Bonsai-2-27B")
            self.assertEqual((opts["flash-attn"], opts["cache-type-k"], opts["cache-type-v"]), ("on", "q8_0", "q8_0"))
            for name, cfg in configs.items():
                self.assertGreaterEqual(cfg["ctx_size"], 16384, name)
                meta = diagnostics.meta_for(cfg["model_path"]) if os.path.exists(cfg["model_path"]) else None
                # Only models that must stay whole on the GPU (hybrid Qwen 3.5) pin their layer count.
                own_ngl = llama_local.model_own_options(name).get("ngl")
                self.assertEqual(own_ngl, "99" if diagnostics.keeps_whole(meta) else None, name)
                if llama_local.model_options(name).get("n-cpu-moe"):          # experts in RAM: read in at load
                    self.assertEqual(llama_local.model_options(name).get("load-mode"), "none", name)
                if meta and meta.get("trained_ctx"):
                    self.assertLessEqual(cfg["ctx_size"], meta["trained_ctx"], name)
        llama_local._ini_cache.update(key=None, configs=None)


if __name__ == "__main__":
    unittest.main()
