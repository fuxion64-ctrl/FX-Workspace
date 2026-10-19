"""Local images (image_gen.py): the prompt filter, sizes, settings, the engine's job flow (a fake sd-server),
giving the GPU back to the chat model, when a turn is offered generate_image, the reply's image lines, the
file endpoint, and the page. No engine, GPU or network: made images go to a temporary folder, never data/."""
import base64
import os
import sys
import shutil
import tempfile
import unittest
from unittest import mock

import core.config as config
import core.state as shared
import image_gen
import llama_local

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class FilterTests(unittest.TestCase):
    def test_minors_are_always_refused(self):
        for p in ("a nude teen on a beach", "sexy schoolgirl", "ילדה עירומה", "голая девочка", "naked 12 year old"):
            for strict in (True, False):
                self.assertIn("minors", image_gen.check_prompt(p, strict) or "", (p, strict))

    def test_the_rest_follows_the_switch(self):
        for p in ("nude portrait of Taylor Swift", "a sexy photo of a famous actress", "a rape scene", "изнасилование"):
            self.assertIsNotNone(image_gen.check_prompt(p, True), p)
            self.assertIsNone(image_gen.check_prompt(p, False), p)

    def test_ordinary_prompts_pass(self):
        for p in ("a lighthouse on a cliff at dusk, oil painting", "a kid flying a kite", "classical nude marble statue",
                  "Tel Aviv skyline at night", "מגדלור על צוק", "кот в космосе"):
            self.assertIsNone(image_gen.check_prompt(p, True), p)

    def test_default_follows_the_setting(self):
        self.addCleanup(image_gen.configure, lambda: {})
        image_gen.configure(lambda: {"image_filter": False})
        self.assertIsNone(image_gen.check_prompt("a rape scene"))
        image_gen.configure(lambda: {})
        self.assertIsNotNone(image_gen.check_prompt("a rape scene"))           # on unless turned off


class SettingsTests(unittest.TestCase):
    def test_clean(self):
        out = image_gen.clean_settings({"image_root": "..\\evil", "image_quality": "ultra", "image_size": "huge",
                                        "image_keep_minutes": "999", "image_free_gpu": "yes", "image_filter": False,
                                        "image_extra_roots": ["D:\\Old", "..\\x", "D:\\Old", 5], "image_vram_reserve": "9",
                                        "image_speed_boost": False, "image_warmup": "no"})
        self.assertEqual(out, {"image_root": image_gen.DEFAULT_ROOT, "image_extra_roots": ["D:\\Old"],
                               "image_quality": "balanced", "image_size": "square", "image_keep_minutes": 120,
                               "image_free_gpu": True, "image_filter": False, "image_speed_boost": False,
                               "image_warmup": True, "image_vram_reserve": 4.0})
        self.assertEqual(image_gen.clean_settings({"image_root": "D:\\AI\\Images"})["image_root"], "D:\\AI\\Images")
        self.assertEqual(image_gen.clean_settings({"image_vram_reserve": "x"})["image_vram_reserve"], 1.5)
        self.assertEqual(image_gen.clean_settings({"image_vram_reserve": 0.1})["image_vram_reserve"], 0.5)
        same = image_gen.clean_settings({"image_root": "D:\\A", "image_extra_roots": ["D:\\A", "E:\\B"]})
        self.assertEqual(same["image_extra_roots"], ["E:\\B"])                  # never the main folder twice

    def test_defaults_and_config(self):
        d = config.DEFAULT_SETTINGS
        self.assertEqual((d["image_root"], d["image_quality"], d["image_free_gpu"], d["image_filter"]),
                         (r"E:\ClankerLocalCpp\Images", "balanced", True, True))
        self.assertEqual(config.clean_privacy_settings({"image_quality": "best"})["image_quality"], "best")

    def test_sizes(self):
        self.addCleanup(image_gen.configure, lambda: {})
        image_gen.configure(lambda: {"image_size": "wide"})
        self.assertEqual(image_gen.parse_size("portrait"), (832, 1216))
        self.assertEqual(image_gen.parse_size("1000x700"), (960, 640))       # multiples of 64
        self.assertEqual(image_gen.parse_size("5000 x 100"), (2048, 256))    # clamped
        self.assertEqual(image_gen.parse_size("big please"), (1344, 768))    # the setting

    def test_registered_models(self):
        self.assertEqual(image_gen.QUALITIES, ("fast", "balanced", "best", "dreamshaper8", "anythingv5", "animagine4", "pony6", "pony6_anime", "pony6_warm"))
        self.assertTrue(image_gen.MODELS["fast"]["edit"])
        self.assertFalse(any(image_gen.MODELS[k]["edit"] for k in ("balanced", "best")))
        self.assertEqual(image_gen.MODELS["best"]["license"], "Krea 2 Community License")
        # The step cache turned Krea 2's sky to grain (2026-10-01): Best never uses it.
        self.assertEqual([k for k, m in image_gen.MODELS.items() if m["step_cache"]], ["fast", "balanced"])

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_single_file_checkpoints_are_detected_and_load_with_model_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "models"))
            self.addCleanup(image_gen.configure, lambda: {})
            image_gen.configure(lambda: {"image_root": tmp})
            for key, skip in (("dreamshaper8", "1"), ("anythingv5", "2"), ("animagine4", "2"), ("pony6", "2")):
                self.assertFalse(image_gen.installed(key))
                name = image_gen.MODELS[key]["checkpoint"]
                path = os.path.join(tmp, "models", name)
                with open(path, "wb") as f:
                    f.write(b"test checkpoint")
                self.assertEqual(image_gen.model_files(key), {"checkpoint": path})
                self.assertTrue(image_gen.installed(key))
                self.assertEqual(image_gen.clean_settings({"image_quality": key})["image_quality"], key)
                with mock.patch.object(image_gen, "server_exe", return_value="sd-server.exe"), \
                     mock.patch.object(image_gen, "_stop"), \
                     mock.patch.object(image_gen, "free_vram_mb", return_value=6000), \
                     mock.patch.object(image_gen, "_alive", return_value=True), \
                     mock.patch.object(image_gen, "_set_stage"), \
                     mock.patch.object(image_gen.httpx, "get", return_value=mock.Mock(status_code=200)), \
                     mock.patch.object(image_gen.subprocess, "Popen") as spawn, \
                     mock.patch.object(image_gen, "_proc"), mock.patch.object(image_gen, "_proc_key"), \
                     mock.patch.object(image_gen, "_log_path"), mock.patch.object(image_gen, "_start_free"), \
                     mock.patch.object(image_gen, "_proc_reserve"):
                    image_gen._start(key)
                    args = spawn.call_args.args[0]
                    spawn.call_args.kwargs["stdout"].close()
                    self.assertEqual(args[args.index("--model") + 1], path)
                    self.assertEqual(args[args.index("--clip-skip") + 1], skip)
                    self.assertNotIn("--llm", args)
                    self.assertNotIn("--vae", args)
                    self.assertNotIn("--diffusion-model", args)

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_extra_roots_find_a_model_left_behind(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        self.addCleanup(image_gen.configure, lambda: {})
        new, old = os.path.join(tmp, "new"), os.path.join(tmp, "old")
        m = image_gen.MODELS["best"]
        os.makedirs(os.path.join(old, "models"))
        for part in ("diffusion", "llm", "vae"):
            open(os.path.join(old, "models", m[part]), "wb").close()
        image_gen.configure(lambda: {"image_root": new, "image_extra_roots": [old]})
        self.assertTrue(image_gen.installed("best"))
        self.assertTrue(image_gen.model_files("best")["diffusion"].startswith(old))
        image_gen.configure(lambda: {"image_root": new})
        self.assertFalse(image_gen.installed("best"))


class FakeProc:
    def __init__(self):
        self.stopped = False

    def poll(self):
        return 0 if self.stopped else None

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0


class EngineTests(unittest.TestCase):
    """generate() against a fake sd-server: the job is submitted, polled, and the picture saved."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = (image_gen.IMAGES_DIR, image_gen._proc, image_gen._proc_key, image_gen._start_free, dict(image_gen._typical),
                      image_gen._log_path, image_gen._proc_reserve)
        image_gen.IMAGES_DIR = os.path.join(self.tmp, "images")
        image_gen._log_path = os.path.join(self.tmp, "sd-server.log")
        open(image_gen._log_path, "wb").close()
        image_gen._stop_requested.clear()
        self.settings = {"image_free_gpu": True, "image_warmup": False}           # never reads the real model files
        image_gen.configure(lambda: self.settings)
        self.posts, self.starts = [], []
        self.status = {"status": "completed", "result": {"images": [{"b64_json": base64.b64encode(PNG).decode()}]}}
        self.log_lines = []                                                     # written to the log on each poll

        def start(key, progress=None, reserve=None):
            self.starts.append((key, reserve))
            image_gen._proc, image_gen._proc_key, image_gen._start_free = FakeProc(), key, 6000
            image_gen._proc_reserve = reserve or 1.5

        def post(url, json=None, timeout=None):
            self.posts.append((url, json))
            return mock.Mock(status_code=202, json=lambda: {"id": "job1"}, text="")

        def get(*a, **k):
            if self.log_lines:
                with open(image_gen._log_path, "a", encoding="utf-8") as f:
                    f.write(self.log_lines.pop(0) + "\n")
            st = self.status(len(self.posts)) if callable(self.status) else self.status
            return mock.Mock(json=lambda: st)

        for p in (mock.patch.object(image_gen, "_start", side_effect=start),
                  mock.patch.object(image_gen, "_ensure_idle_watch"),
                  mock.patch.object(image_gen, "free_vram_mb", return_value=6000),
                  mock.patch.object(image_gen, "_wait_for_vram"),
                  mock.patch.object(image_gen.time, "sleep"),
                  mock.patch.object(image_gen.httpx, "post", side_effect=post),
                  mock.patch.object(image_gen.httpx, "get", side_effect=get),
                  mock.patch.object(llama_local, "unload_all", return_value=["gemma"])):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        (image_gen.IMAGES_DIR, image_gen._proc, image_gen._proc_key, image_gen._start_free, typical,
         image_gen._log_path, image_gen._proc_reserve) = self.saved
        image_gen._typical.clear()
        image_gen._typical.update(typical)
        image_gen._stop_requested.clear()
        image_gen.configure(lambda: {})
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_step_cache_only_where_it_does_no_harm(self):
        image_gen.generate("a lighthouse", "balanced")
        image_gen.generate("a lighthouse", "best")
        self.assertEqual(self.posts[0][1].get("cache_mode"), "easycache")
        self.assertNotIn("cache_mode", self.posts[1][1])
        self.settings["image_speed_boost"] = False
        image_gen.generate("a lighthouse", "fast")
        self.assertNotIn("cache_mode", self.posts[2][1])

    def test_several_pictures_with_consecutive_seeds(self):
        two = base64.b64encode(PNG).decode()
        self.status = {"status": "completed", "result": {"images": [{"b64_json": two}, {"b64_json": two}]}}
        img = image_gen.generate("a boat", "fast", count=2, seed=100)
        self.assertEqual(self.posts[0][1]["batch_count"], 2)
        self.assertEqual([x["seed"] for x in img["images"]], [100, 101])
        self.assertEqual(len({x["id"] for x in img["images"]}), 2)
        self.assertEqual(len(image_gen.recent()), 2)
        image_gen.generate("a boat", count=99)
        self.assertEqual(self.posts[-1][1]["batch_count"], image_gen.MAX_COUNT)    # at most 4 per request

    def test_progress_comes_from_the_engine_log(self):
        said = []
        self.log_lines = ["[INFO] image.cpp:866  - generating image: 1/1 - seed 7",
                          "  |======>    | 3/8 - 2.50s/it\x1b[K", "  |======>    | 5/8 - 2.50s/it\x1b[K"]
        self.status = lambda n: ({"status": "generating"} if self.log_lines else
                                 {"status": "completed", "result": {"images": [{"b64_json": base64.b64encode(PNG).decode()}]}})
        image_gen.generate("a cat", "balanced", progress=said.append)
        self.assertTrue(any(s.startswith("Z-Image-Turbo: step 3 of 8, about") for s in said), said)
        self.assertTrue(any("step 5 of 8" in s for s in said), said)

    def test_out_of_gpu_memory_is_retried_with_more_room(self):
        fail = {"status": "failed", "error": {"message": "generate_image returned no results"}}
        ok = {"status": "completed", "result": {"images": [{"b64_json": base64.b64encode(PNG).decode()}]}}
        self.log_lines = ["[WARN] model manager cannot make enough memory available on CUDA0: need 1202 MB"]
        self.status = lambda n: fail if n == 1 else ok
        img = image_gen.generate("a cat", "balanced")
        self.assertEqual(img["model"], "Z-Image-Turbo")
        self.assertEqual(self.starts, [("balanced", None), ("balanced", 2.5)])  # 1.5 GB kept free, then 2.5
        self.log_lines = ["[WARN] model manager cannot make enough memory available on CUDA0: need 1202 MB",
                          "[WARN] model manager cannot make enough memory available on CUDA0: need 1202 MB"]
        self.status = fail
        with self.assertRaises(image_gen.ImageError) as e:
            image_gen.generate("a cat", "balanced")
        self.assertEqual(e.exception.code, "oom")
        self.assertIn("another program is using the GPU", str(e.exception))

    def test_other_failures_are_not_retried(self):
        self.status = {"status": "failed", "error": {"message": "bad prompt"}}
        with self.assertRaises(image_gen.ImageError) as e:
            image_gen.generate("a cat", "balanced")
        self.assertEqual((e.exception.code, len(self.starts)), ("failed", 1))

    def test_stop_from_images_view(self):
        self.status = lambda n: (image_gen.request_stop() and {"status": "generating"}) or {"status": "generating"}
        with self.assertRaises(image_gen.ImageError) as e:
            image_gen.generate("a cat")
        self.assertEqual(e.exception.code, "stopped")
        self.assertIsNone(image_gen._proc)                  # the engine can't stop a picture mid-draw: it is closed
        self.assertFalse(image_gen.request_stop())          # nothing running now

    def test_waits_while_a_chat_turn_holds_the_gpu(self):
        busy = iter([True, True, False])
        said = []
        image_gen.generate("a cat", wait_while=lambda: next(busy), progress=said.append)
        self.assertIn("Waiting for the GPU - the chat model or a Labs run is using it", said)
        with self.assertRaises(image_gen.ImageError) as e:
            image_gen.generate("a cat", wait_while=lambda: True, should_cancel=lambda: True)
        self.assertEqual(e.exception.code, "stopped")

    def test_refusals_have_codes(self):
        for prompt, code in (("nude child", "minors"), ("a rape scene", "filter"), ("", "empty")):
            with self.assertRaises(image_gen.ImageError) as e:
                image_gen.generate(prompt)
            self.assertEqual(e.exception.code, code, prompt)
        self.assertEqual(self.posts, [])

    def test_makes_and_saves_an_image(self):
        said = []
        img = image_gen.generate("a lighthouse", "balanced", "landscape", 7, progress=said.append)
        self.assertEqual((img["model"], img["width"], img["height"], img["seed"]), ("Z-Image-Turbo", 1216, 832, 7))
        self.assertRegex(img["url"], r"^/api/images/\d{8}-\d{6}-[0-9a-f]{6}\.png$")
        with open(img["path"], "rb") as f:
            self.assertEqual(f.read(), PNG)
        self.assertTrue(img["path"].startswith(self.tmp))
        url, body = self.posts[0]
        self.assertTrue(url.endswith("/sdcpp/v1/img_gen"))
        self.assertEqual(body["sample_params"], {"sample_steps": 8, "guidance": {"txt_cfg": 1.0}})
        self.assertNotIn("ref_images", body)
        self.assertIn("Unloaded gemma to free the GPU", said)
        self.assertEqual([m["id"] for m in image_gen.recent()], [img["id"]])
        self.assertEqual(image_gen._job, {})

    def test_an_attached_image_is_edited_with_flux2_klein(self):
        img = image_gen.generate("make the sky red", "best", ref_image=PNG)
        self.assertEqual(img["quality"], "fast")
        body = self.posts[0][1]
        self.assertTrue(body["ref_images"][0].startswith("data:image/png;base64,"))
        self.assertEqual(body["sample_params"]["sample_method"], "euler")

    def test_stop_cancels_the_job(self):
        self.status = {"status": "generating"}
        with self.assertRaisesRegex(image_gen.ImageError, "Stopped"):
            image_gen.generate("a cat", should_cancel=lambda: True)
        self.assertTrue(self.posts[-1][0].endswith("/sdcpp/v1/jobs/job1/cancel"))

    def test_refused_prompts_never_reach_the_engine(self):
        with self.assertRaisesRegex(image_gen.ImageError, "minors"):
            image_gen.generate("nude child")
        self.assertEqual(self.posts, [])

    def test_an_engine_started_on_a_crowded_gpu_is_restarted(self):
        image_gen._proc, image_gen._proc_key, image_gen._start_free = FakeProc(), "balanced", 1200
        old = image_gen._proc
        image_gen.generate("a cat", "balanced")                                 # 6000 MB free now: plan again
        self.assertTrue(old.stopped)
        image_gen.generate("a dog", "balanced")                                 # same room: kept
        self.assertEqual(image_gen._start.call_count, 1)
        self.assertFalse(image_gen._proc.stopped)

    def test_the_tool(self):
        ok = image_gen.generate_image_tool({"prompt": "a boat", "quality": "fast"}, ref_image=PNG)
        self.assertTrue(ok["ok"])
        self.assertNotIn("ref_images", self.posts[0][1])                         # only with edit: true
        self.assertEqual(set(ok["image"]), {"id", "url", "model", "seed", "width", "height", "seconds"})
        self.assertEqual(len(ok["images"]), 1)
        self.assertIn("the app puts it under your reply", ok["message"])
        bad = image_gen.generate_image_tool({"prompt": ""})
        self.assertEqual((bad["ok"], bad["status"], bad["code"]), (False, "permanent_failure", "empty"))

    def test_gallery_actions(self):
        first = image_gen.generate("a lighthouse at dusk", "balanced", "portrait", 5)
        second = image_gen.generate("a boat at sunrise", "fast")
        iid = first["id"]
        self.assertEqual([m["id"] for m in image_gen.recent(q="lighthouse")], [iid])
        self.assertEqual([m["id"] for m in image_gen.recent(quality="fast")], [second["id"]])
        image_gen.set_favorite(iid, True)
        self.assertEqual([m["id"] for m in image_gen.recent(fav=True)], [iid])
        image_gen.trash(iid)                                                     # into the trash, never deleted
        self.assertNotIn(iid, [m["id"] for m in image_gen.recent()])
        self.assertEqual([m["url"] for m in image_gen.recent(trashed=True)], [f"/api/images/trash/{iid}.png"])
        self.assertTrue(os.path.exists(os.path.join(image_gen.IMAGES_DIR, "trash", f"{iid}.png")))
        image_gen.restore(iid)
        self.assertIn(iid, [m["id"] for m in image_gen.recent()])
        self.assertEqual(image_gen.image_bytes(iid), PNG)
        with self.assertRaises(image_gen.ImageError):
            image_gen.trash("../../settings")
        self.posts.clear()
        again = image_gen.again(iid, "variations")
        body = self.posts[0][1]
        self.assertEqual((body["prompt"], body["width"], body["height"], body["batch_count"]), ("a lighthouse at dusk", 832, 1216, 2))
        self.assertNotEqual(again["seed"], 5)
        self.posts.clear()
        edited = image_gen.edit(iid, "the same at night")
        self.assertEqual(edited["quality"], "fast")
        self.assertTrue(self.posts[0][1]["ref_images"][0].startswith("data:image/png;base64,"))


class GpuTests(unittest.TestCase):
    def setUp(self):
        self.saved = (image_gen._proc, image_gen._proc_key, dict(image_gen._job))
        self.addCleanup(image_gen.configure, lambda: {})

    def tearDown(self):
        image_gen._proc, image_gen._proc_key, job = self.saved
        image_gen._job = job

    def test_yield_gpu(self):
        image_gen.configure(lambda: {})
        image_gen._proc, image_gen._job = FakeProc(), {}
        self.assertTrue(image_gen.yield_gpu())
        self.assertIsNone(image_gen._proc)
        self.assertFalse(image_gen.yield_gpu())                                   # nothing running
        image_gen._proc, image_gen._job = FakeProc(), {"id": "j"}
        self.assertFalse(image_gen.yield_gpu())                                   # never mid-image
        image_gen._job = {}
        image_gen.configure(lambda: {"image_free_gpu": False})
        self.assertFalse(image_gen.yield_gpu())                                   # the user keeps both

    def test_chat_calls_yield_first(self):
        import inspect
        import agent.models as models
        src = inspect.getsource(models.call_model)
        self.assertLess(src.index("image_gen.yield_gpu()"), src.index("_fit_for_local()"))

    def test_unload_all(self):
        calls = []
        with mock.patch.object(llama_local, "server_models"), \
                mock.patch.dict(llama_local._cache, {"status": {"a": {"loaded": True}, "b": {"loaded": False}, "c": {"loaded": True}}}), \
                mock.patch.object(llama_local, "invalidate_cache") as inv, \
                mock.patch.object(llama_local.requests, "post", side_effect=lambda url, json=None, **k: calls.append((url, json)) or mock.Mock()):
            self.assertEqual(llama_local.unload_all(), ["a", "c"])
        self.assertTrue(all(u.endswith("/models/unload") for u, _ in calls))
        self.assertEqual([j["model"] for _, j in calls], ["a", "c"])
        inv.assert_called_once()

    def test_wait_for_vram_without_a_reading(self):
        with mock.patch.object(image_gen, "free_vram_mb") as f:
            image_gen._wait_for_vram(None)
        f.assert_not_called()

    def test_wait_for_vram_stops_when_a_small_model_is_out(self):
        readings = iter([1800, 2300, 2310, 2310, 2310])          # +500 MB freed, then steady
        with mock.patch.object(image_gen, "free_vram_mb", side_effect=lambda: next(readings)), \
                mock.patch.object(image_gen.time, "sleep") as sleep:
            image_gen._wait_for_vram(1800)
        self.assertLessEqual(sleep.call_count, 3)                 # not the full 8 s
        with mock.patch.object(image_gen, "free_vram_mb", return_value=5000), mock.patch.object(image_gen.time, "sleep") as sleep:
            image_gen._wait_for_vram(1800)                       # a big model freed: straight on
        sleep.assert_not_called()


REAL_LOG = """[INFO   ] image.cpp:814  - generate_image 832x1216
[INFO   ] image.cpp:529  - get_learned_condition completed, taking 1.26s
[INFO   ] image.cpp:866  - generating image: 1/2 - seed 1584296985
  |======>                                           | 1/8 - 5.03s/it\x1b[K
  |============>                                     | 2/8 - 2.56s/it\x1b[K
[INFO   ] image.cpp:866  - generating image: 2/2 - seed 1584296986
  |======>                                           | 1/8 - 2.61s/it\x1b[K
  |=========================>                        | 4/8 - 2.57s/it\x1b[K
"""


class ProgressTests(unittest.TestCase):
    def test_reads_the_real_log_format(self):
        p = image_gen.parse_progress(REAL_LOG)
        self.assertEqual((p["stage"], p["image"], p["images"], p["step"], p["steps"]), ("drawing", 2, 2, 4, 8))
        self.assertAlmostEqual(p["sec_per_step"], 2.57)
        done = image_gen.parse_progress(REAL_LOG + "[INFO   ] image.cpp:554  - decoding 2 latents\n  |=> | 1/40 - 5.68it/s\n")
        self.assertEqual((done["stage"], done["step"]), ("finishing", 4))       # decoder tiles aren't steps
        loading = image_gen.parse_progress("  |#####   | 453/1095 - 0.00MB/s\n")
        self.assertEqual((loading["stage"], loading["steps"]), ("encoding", 0))  # the file loader's bar isn't either
        fast = image_gen.parse_progress("generating image: 1/1 - seed 1\n  |==| 2/4 - 3.20it/s\n")
        self.assertAlmostEqual(fast["sec_per_step"], 1 / 3.2)

    def test_out_of_memory_is_recognised(self):
        self.assertTrue(image_gen._ran_out_of_memory("model manager cannot make enough memory available on CUDA0"))
        self.assertTrue(image_gen._ran_out_of_memory("z_image segment 1/1 (graph) failed during workspace capacity check"))
        self.assertFalse(image_gen._ran_out_of_memory("generate_image completed in 25.13s"))


class WarmTests(unittest.TestCase):
    """Reading the model files into Windows' cache - here, three small temp files."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        m = image_gen.MODELS["balanced"]
        os.makedirs(os.path.join(self.tmp, "models"))
        for i, part in enumerate(("diffusion", "llm", "vae")):
            with open(os.path.join(self.tmp, "models", m[part]), "wb") as f:
                f.write(b"x" * (1000 * (i + 1)))
        self.settings = {"image_root": self.tmp}
        image_gen.configure(lambda: self.settings)
        self.addCleanup(image_gen.configure, lambda: {})
        self.saved = (dict(image_gen._warmed_at), image_gen._proc)
        image_gen._warmed_at.clear()
        image_gen._proc = None

    def tearDown(self):
        image_gen._warmed_at.clear()
        image_gen._warmed_at.update(self.saved[0])
        image_gen._proc = self.saved[1]

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_reads_every_file_then_skips_fresh_ones(self):
        said = []
        image_gen._warm_wait("balanced", progress=said.append)
        self.assertEqual(image_gen._warm["total"], 6000)
        self.assertEqual(image_gen._warm["done"], 6000)
        self.assertEqual(len(image_gen._warmed_at), 3)
        self.assertEqual(image_gen._warm_todo("balanced"), [])                  # read moments ago: still cached
        image_gen._warmed_at.update({p: 0 for p in image_gen._warmed_at})
        self.assertEqual(len(image_gen._warm_todo("balanced")), 3)               # long ago: read again

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_background_warm_and_the_switch(self):
        self.assertTrue(image_gen.warm("balanced"))
        image_gen._warm_thread.join(5)
        self.assertEqual(len(image_gen._warmed_at), 3)
        self.assertFalse(image_gen.warm("balanced"))                             # nothing left to read
        image_gen._warmed_at.clear()
        self.settings["image_warmup"] = False
        self.assertFalse(image_gen.warm("balanced"))
        self.assertFalse(image_gen.warm("best"))                                 # not downloaded here

    def test_a_running_engine_needs_no_warming(self):
        image_gen._proc, saved_key = FakeProc(), image_gen._proc_key
        image_gen._proc_key = "balanced"
        self.addCleanup(setattr, image_gen, "_proc_key", saved_key)
        self.assertEqual(image_gen._warm_todo("balanced"), [])

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_stop_while_reading(self):
        with self.assertRaises(image_gen.ImageError) as e:
            image_gen._warm_wait("balanced", should_cancel=lambda: True)
        self.assertEqual(e.exception.code, "stopped")


class MoveTests(unittest.TestCase):
    """Moving the models: every copy checked, the folder switched only at the end, the old copies kept."""

    def setUp(self):
        import hashlib
        import json
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src, self.dest = os.path.join(self.tmp, "E"), os.path.join(self.tmp, "C", "Images")
        os.makedirs(os.path.join(self.src, "engine", "bin"))
        os.makedirs(os.path.join(self.src, "models"))
        with open(os.path.join(self.src, "engine", "bin", "sd-server.exe"), "wb") as f:
            f.write(b"engine" * 100)
        manifest = {}
        for key in ("fast", "balanced", "best"):
            for part in ("diffusion", "llm", "vae"):
                name = image_gen.MODELS[key][part]
                data = name.encode() * 50
                with open(os.path.join(self.src, "models", name), "wb") as f:
                    f.write(data)
                manifest[f"models/{name}"] = {"sha256": hashlib.sha256(data).hexdigest()}
        with open(os.path.join(self.src, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        self.manifest = manifest
        self.settings = {"image_root": self.src}
        self.saved = []
        image_gen.configure(lambda: self.settings, lambda changes: (self.saved.append(changes), self.settings.update(changes)))
        self.addCleanup(image_gen.configure, lambda: {})
        self.addCleanup(image_gen._move.clear)
        for p in (mock.patch.object(image_gen, "_PATH_RE", __import__("re").compile(r".+")),   # temp paths may not be X:\...
                  mock.patch.object(image_gen.shutil, "disk_usage", return_value=mock.Mock(free=50 * 1024 ** 3))):
            p.start()
            self.addCleanup(p.stop)

    def _run(self, qualities):
        plan = image_gen.plan_move(self.dest, qualities)
        self.assertTrue(plan["ok"], plan.get("why"))
        image_gen._move.update(state="running", done=0, files_done=0)
        image_gen._move_run(plan)
        return plan

    def test_moves_checks_and_switches(self):
        plan = self._run(["fast", "balanced"])
        names = {os.path.basename(f["src"]) for f in plan["files"]}
        self.assertIn("sd-server.exe", names)
        self.assertNotIn(image_gen.MODELS["best"]["diffusion"], names)
        self.assertEqual(sum(1 for f in plan["files"] if f["src"].endswith("Qwen3-4B-Q4_K_M.gguf")), 1)  # shared, copied once
        self.assertEqual(image_gen._move["state"], "done")
        self.assertEqual(self.saved, [{"image_root": self.dest, "image_extra_roots": [self.src]}])
        self.assertTrue(os.path.exists(os.path.join(self.dest, "engine", "bin", "sd-server.exe")))
        self.assertTrue(os.path.exists(os.path.join(self.src, "models", image_gen.MODELS["fast"]["diffusion"])))  # kept
        self.assertTrue(image_gen.installed("best"))                              # found in the old folder
        self.assertTrue(image_gen.model_files("fast")["diffusion"].startswith(self.dest))
        self.assertFalse(any(n.endswith(".part") for _d, _s, fs in os.walk(self.dest) for n in fs))

    def test_moving_everything_drops_the_old_folder(self):
        self._run(["fast", "balanced", "best"])
        self.assertEqual(self.saved[-1]["image_extra_roots"], [])

    def test_a_bad_source_file_stops_it_and_nothing_switches(self):
        name = image_gen.MODELS["balanced"]["diffusion"]
        with open(os.path.join(self.src, "models", name), "ab") as f:
            f.write(b"corrupt")
        plan = image_gen.plan_move(self.dest, ["balanced"])
        image_gen._move.update(state="running", done=0, files_done=0)
        image_gen._move_run(plan)
        self.assertEqual(image_gen._move["state"], "failed")
        self.assertIn("doesn't match its published SHA-256", image_gen._move["error"])
        self.assertEqual(self.saved, [])
        self.assertFalse(os.path.exists(os.path.join(self.dest, "models", name)))
        self.assertFalse(os.path.exists(os.path.join(self.dest, "models", name + ".part")))

    def test_no_room_is_refused_up_front(self):
        with mock.patch.object(image_gen.shutil, "disk_usage", return_value=mock.Mock(free=1000)):
            plan = image_gen.plan_move(self.dest, ["fast"])
        self.assertFalse(plan["ok"])
        self.assertIn("would leave less than 5 GB free", plan["why"])
        with mock.patch.object(image_gen.shutil, "disk_usage", return_value=mock.Mock(free=1000)):
            with self.assertRaises(image_gen.ImageError):
                image_gen.move_models(self.dest, ["fast"])
        self.assertFalse(image_gen.plan_move(self.src, ["fast"])["ok"])          # already there
        self.assertFalse(image_gen.plan_move(self.dest, [])["ok"])

    def test_cancel_removes_only_its_own_partial_copy(self):
        image_gen._move_cancel.set()
        self.addCleanup(image_gen._move_cancel.clear)
        plan = image_gen.plan_move(self.dest, ["fast"])
        image_gen._move_run(plan)
        self.assertEqual(image_gen._move["state"], "cancelled")
        self.assertEqual(self.saved, [])
        self.assertFalse(any(n.endswith(".part") for _d, _s, fs in os.walk(self.dest) for n in fs))
        self.assertTrue(os.path.exists(os.path.join(self.src, "engine", "bin", "sd-server.exe")))


class TurnTests(unittest.TestCase):
    def setUp(self):
        keys = ("turn_tools", "turn_persona", "turn_files", "turn_attachments", "turn_images", "conversation",
                "local_planned", "local_model", "local_engaged")
        saved = {k: shared.app_state.get(k) for k in keys}
        self.addCleanup(lambda: shared.app_state.update(saved))

    def test_wants_image(self):
        for m in ("draw a cat", "Please paint the sea at night", "make me a picture of a dog", "can you create an image of Haifa", "imagine a castle",
                  "צייר חתול", "תמונה של ים", "нарисуй кота"):
            self.assertTrue(image_gen.wants_image(m), m)
        for m in ("what's the weather", "generate a report", "create a new file", "render the page", "hello",
                  "what can we draw a conclusion from", "draw conclusions from this data"):
            self.assertFalse(image_gen.wants_image(m), m)

    def test_only_turns_that_ask_get_the_tool(self):
        import agent.tasks as tasks
        import tools.schemas as schemas
        for message, command, has in (("draw me a fox", None, True), ("hello there", None, False), ("a fox", "image", True)):
            req = shared.ChatRequest(message=message)
            if command:
                req.command = command
                req.tools = list(tasks.SLASH_COMMANDS[command]["tools"])
            with mock.patch.object(image_gen, "warm") as warm:
                tasks._begin_composer_tools(req)
            self.assertEqual("generate_image" in schemas.world_hidden(), not has, message)
            self.assertEqual(warm.called, has, message)                     # the files start coming off the disk
        import composer_tools
        self.assertIn("image", composer_tools.IDS)
        self.assertEqual(composer_tools.OFFERS["image"], {"generate_image"})
        self.assertIn("IMAGE", composer_tools.HEADERS["image"].upper())

    def test_policy(self):
        import agent.loop as loop
        import permissions
        import tool_registry
        import tools.schemas as schemas
        self.assertIn("generate_image", loop._HOST_RESULT_TOOLS)                  # its result isn't framed as untrusted
        self.assertIn("generate_image", permissions.BENIGN_EXTRA)
        self.assertIn("generate_image", schemas.LOCAL_TOOL_NAMES)
        self.assertEqual(tool_registry._POLICY["generate_image"]["timeout_s"], 900)

    def test_images_go_under_the_reply(self):
        import agent.tasks as tasks
        shared.app_state["turn_images"] = [{"model": "Z-Image-Turbo", "seed": 5, "url": "/api/images/20261001-000000-abcdef.png"}]
        shared.app_state["conversation"] = [{"role": "user", "content": "draw"}, {"role": "assistant", "content": "A lighthouse."}]
        out = tasks._with_images({"message": "A lighthouse."})
        line = "![Z-Image-Turbo - seed 5](/api/images/20261001-000000-abcdef.png)"
        self.assertEqual(out["message"], "A lighthouse.\n\n" + line)
        self.assertEqual(shared.app_state["conversation"][-1]["content"], out["message"])
        shared.app_state["turn_images"] = []
        self.assertEqual(tasks._with_images({"message": "x"}), {"message": "x"})

    def test_tools_are_sized_for_the_model_that_answers(self):
        import tools.schemas as schemas
        shared.app_state.update(local_planned="small", local_model="big", local_engaged="big")
        with mock.patch.object(llama_local, "model_context_limit", side_effect=lambda t: {"small": 16384, "big": 131072}[t]):
            self.assertEqual(schemas._local_context_window(), 16384)
            shared.app_state["local_planned"] = None
            self.assertEqual(schemas._local_context_window(), 131072)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main
        cls.client = TestClient(main.app, base_url="http://127.0.0.1:8000")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = image_gen.IMAGES_DIR
        image_gen.IMAGES_DIR = self.tmp

    def tearDown(self):
        image_gen.IMAGES_DIR = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_only_the_apps_own_names(self):
        with open(os.path.join(self.tmp, "20261001-000000-abcdef.png"), "wb") as f:
            f.write(PNG)
        r = self.client.get("/api/images/20261001-000000-abcdef.png")
        self.assertEqual((r.status_code, r.content), (200, PNG))
        for bad in ("..%2F..%2Fsettings.json", "20261001-000000-abcdef.json", "x.png", "20261001-000000-ABCDEF.png"):
            self.assertEqual(self.client.get(f"/api/images/{bad}").status_code, 404, bad)

    def test_refusal_is_a_message(self):
        r = self.client.post("/api/images/generate", json={"prompt": "nude child"}).json()
        self.assertFalse(r["ok"])
        self.assertIn("minors", r["message"])
        self.assertEqual(r["code"], "minors")

    def test_gallery_endpoints(self):
        import json
        iid = "20261001-000000-abcdef"
        with open(os.path.join(self.tmp, f"{iid}.png"), "wb") as f:
            f.write(PNG)
        with open(os.path.join(self.tmp, f"{iid}.json"), "w", encoding="utf-8") as f:
            json.dump({"id": iid, "prompt": "a boat", "model": "Z-Image-Turbo", "quality": "balanced", "seed": 1,
                       "width": 1024, "height": 1024}, f)
        self.assertTrue(self.client.post(f"/api/images/{iid}/fav", json={"on": True}).json()["ok"])
        listed = self.client.get("/api/images?fav=true&q=boat").json()
        self.assertEqual([m["id"] for m in listed["images"]], [iid])
        self.assertIn("job", listed["status"])
        self.assertTrue(self.client.post(f"/api/images/{iid}/trash").json()["ok"])
        self.assertEqual(self.client.get(f"/api/images/trash/{iid}.png").status_code, 200)
        self.assertEqual(self.client.get(f"/api/images/{iid}.png").status_code, 404)
        self.assertTrue(self.client.post(f"/api/images/{iid}/restore").json()["ok"])
        self.assertEqual(self.client.get(f"/api/images/{iid}.png").status_code, 200)
        self.assertFalse(self.client.post("/api/images/..%2Fx/trash").json().get("ok", False))
        self.assertFalse(self.client.post("/api/images/stop").json()["stopped"])          # nothing drawing
        self.assertIn("job", self.client.get("/api/images/status").json())

    def _made(self, iid, quality="pony6", seq=None):
        import json
        with open(os.path.join(self.tmp, f"{iid}.png"), "wb") as f:
            f.write(PNG)
        meta = {"id": iid, "prompt": "a knight", "model": "Pony", "quality": quality, "seed": 1, "width": 8, "height": 8}
        if seq:
            meta["seq"] = seq
        with open(os.path.join(self.tmp, f"{iid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def test_a_sequence_is_one_gallery_entry_and_a_family_filters_together(self):
        seq = "20261001-000000-aaaaaa"
        for i in range(3):        # newest first in the listing, so the cover must be picked by index
            self._made(f"20261001-00000{i}-aaaaa{i}", "pony6_anime",
                       {"id": seq, "index": i, "count": 3, "mode": "frames", "label": f"frame {i}"})
        self._made("20261001-000009-bbbbbb", "pony6_warm")
        self._made("20261001-000008-cccccc", "balanced")
        grouped = self.client.get("/api/images?group=true").json()["images"]
        self.assertEqual(len(grouped), 3)
        g = next(m for m in grouped if m.get("group"))
        self.assertEqual((g["id"], g["group"]["count"], g["group"]["mode"]), ("20261001-000000-aaaaa0", 3, "frames"))
        self.assertEqual(len(self.client.get("/api/images").json()["images"]), 5)              # ungrouped by default
        fam = self.client.get("/api/images?quality=pony6").json()["images"]
        self.assertEqual({m["quality"] for m in fam}, {"pony6_anime", "pony6_warm"})

    def test_thumbnail_is_a_cached_webp_and_goes_with_delete_forever(self):
        from PIL import Image
        iid = "20261001-000000-dddddd"
        self._made(iid)
        Image.new("RGB", (1024, 768), "red").save(os.path.join(self.tmp, f"{iid}.png"))
        r = self.client.get(f"/api/images/thumb/{iid}.webp")
        self.assertEqual((r.status_code, r.headers["content-type"]), (200, "image/webp"))
        thumb = os.path.join(self.tmp, ".thumbs", f"{iid}.webp")
        with Image.open(thumb) as im:
            self.assertEqual(max(im.size), image_gen.THUMB_PX)
        self.assertEqual(self.client.get("/api/images/thumb/..%2Fsettings.webp").status_code, 404)
        self.assertEqual(self.client.get("/api/images/thumb/20261001-000000-eeeeee.webp").status_code, 404)
        image_gen.trash(iid)
        image_gen.purge(iid)
        self.assertFalse(os.path.exists(thumb))

    def test_move_plan_refuses_a_bad_folder(self):
        r = self.client.post("/api/images/move/plan", json={"dest": "not a path", "qualities": ["fast"]}).json()
        self.assertFalse(r["ok"])
        r = self.client.post("/api/images/move", json={"dest": "relative\\path", "qualities": ["fast"]}).json()
        self.assertFalse(r["ok"])


class PageTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            self.html = f.read()

    def test_images_render_from_this_pc_only(self):
        self.assertIn(r"line.match(/^!\[([^\]]*)\]\((\/api\/images\/[\w.-]+\.png)\)\s*$/)", self.html)
        self.assertIn("image: { icon: 'i-image', desc:", self.html)             # /image
        self.assertIn("image: { icon: 'i-image', label: 'Image'", self.html)     # the chip
        self.assertIn("images: renderImages", self.html)

    def test_the_bar_fits_its_box(self):
        css = self.html[self.html.index("<style>"):self.html.index("</style>")]
        self.assertIn(".composer { container: composer / inline-size; }", css)
        self.assertIn(".composer-row { flex-wrap: nowrap;", css)
        for w in (360, 420, 540, 700):
            self.assertIn(f"@container composer (max-width: {w}px)", css)

    def test_static_copy_matches(self):
        with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
            self.assertEqual(f.read(), self.html)


if __name__ == "__main__":
    unittest.main()
