"""Videos (video_gen.py) and deleting for good from the image and video trash. Nothing touches data/ or runs the
engine: the folders are temporary and sd-cli is mocked."""
import json
import sys
import os
import tempfile
import unittest
from unittest import mock

import image_gen
import video_gen


def _make(folder, name, ext, data=b"x" * 2000, **meta):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name + ext), "wb") as f:
        f.write(data)
    with open(os.path.join(folder, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"id": name, "prompt": "a cat", **meta}, f)


class _Dirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.vdir = os.path.join(self.tmp.name, "videos")
        self.idir = os.path.join(self.tmp.name, "images")
        for p in (mock.patch.object(video_gen, "VIDEOS_DIR", self.vdir), mock.patch.object(image_gen, "IMAGES_DIR", self.idir)):
            p.start()
            self.addCleanup(p.stop)


class ImageTrashTests(_Dirs):
    def test_purge_only_from_the_trash(self):
        _make(self.idir, "20261013-000000-aaaaaa", ".png")
        with self.assertRaises(image_gen.ImageError):
            image_gen.purge("20261013-000000-aaaaaa")             # still in the gallery
        image_gen.trash("20261013-000000-aaaaaa")
        self.assertGreater(image_gen.purge("20261013-000000-aaaaaa"), 0)
        self.assertIsNone(image_gen.find("20261013-000000-aaaaaa"))

    def test_empty_trash_leaves_the_gallery(self):
        for n in ("20261013-000000-aaaaaa", "20261013-000000-bbbbbb", "20261013-000000-cccccc"):
            _make(self.idir, n, ".png")
        image_gen.trash("20261013-000000-aaaaaa")
        image_gen.trash("20261013-000000-bbbbbb")
        self.assertEqual(image_gen.empty_trash()["count"], 2)
        self.assertEqual([i["id"] for i in image_gen.recent()], ["20261013-000000-cccccc"])


class VideoGalleryTests(_Dirs):
    def test_trash_restore_purge_and_empty(self):
        for n in ("20261013-000000-aaaaaa", "20261013-000000-bbbbbb"):
            _make(self.vdir, n, ".webm")
        self.assertEqual(video_gen.count(), 2)
        video_gen.trash("20261013-000000-aaaaaa")
        self.assertEqual([v["id"] for v in video_gen.recent(trashed=True)], ["20261013-000000-aaaaaa"])
        self.assertTrue(video_gen.recent(trashed=True)[0]["url"].startswith("/api/videos/trash/"))
        video_gen.restore("20261013-000000-aaaaaa")
        with self.assertRaises(video_gen.VideoError):
            video_gen.purge("20261013-000000-aaaaaa")             # not in the trash
        video_gen.trash("20261013-000000-aaaaaa")
        self.assertGreater(video_gen.purge("20261013-000000-aaaaaa"), 0)
        video_gen.trash("20261013-000000-bbbbbb")
        self.assertEqual(video_gen.empty_trash()["count"], 1)
        self.assertEqual(video_gen.count(), 0)

    def test_bad_ids_are_refused(self):
        self.assertIsNone(video_gen.find("..\\..\\settings"))
        with self.assertRaises(video_gen.VideoError):
            video_gen.trash("../x")

    def test_sizes_and_lengths(self):
        self.assertEqual(video_gen.parse_size("portrait"), (480, 832))
        self.assertEqual(video_gen.parse_size("1000x500"), (992, 480))          # 32-pixel blocks
        self.assertEqual(video_gen.parse_frames("medium"), 49)
        self.assertEqual(video_gen.parse_frames("50"), 49)                     # 4n+1
        self.assertEqual(video_gen.parse_frames(500), 121)


class VideoGenerateTests(_Dirs):
    """The command line it builds and what it saves - with sd-cli mocked."""

    def _run(self, **kw):
        root = os.path.join(self.tmp.name, "root")
        os.makedirs(os.path.join(root, "models", "loras"))
        for part in ("Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf", "umt5-xxl-encoder-Q5_K_M.gguf", "wan2.2_vae.safetensors", "taew2_2.pth",
                     os.path.join("loras", "wan22_5B_anime_v1.0.safetensors")):
            open(os.path.join(root, "models", part), "wb").close()
        video_gen.configure(lambda: {"video_root": root, **kw.pop("settings", {})})
        self.addCleanup(video_gen.configure, lambda: {})
        calls = []

        def popen(args, **_kw):
            calls.append(args)
            out = args[args.index("-o") + 1]
            with open(out, "wb") as f:
                f.write(b"w" * 5000)
            proc = mock.Mock(returncode=0)
            proc.poll.return_value = 0
            return proc
        with mock.patch.object(video_gen, "cli_exe", return_value=os.path.join(root, "sd-cli.exe")), \
             mock.patch.object(video_gen, "_free_gpu"), \
             mock.patch.object(video_gen.subprocess, "Popen", side_effect=popen):
            clip = video_gen.generate("a cat walking", **kw)
        return clip, calls[0]

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_full_vae_by_default_and_never_offload(self):
        # The tiny decoder flickers (video_gen.clean_settings), so the full VAE is the default.
        clip, args = self._run(size="landscape", length="short")
        self.assertIn("vid_gen", args)
        self.assertNotIn("--tae", args)
        self.assertNotIn("--offload-to-cpu", args)
        self.assertEqual(args[args.index("--steps") + 1], "4")
        self.assertEqual((clip["width"], clip["height"], clip["frames"]), (832, 480, 33))
        self.assertTrue(os.path.exists(os.path.join(self.vdir, clip["id"] + ".json")))
        self.assertTrue(clip["best"])

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_fast_preview_uses_the_tiny_decoder(self):
        clip, args = self._run(best=False)
        self.assertIn("--tae", args)
        self.assertFalse(clip["best"])

    def test_progress_follows_the_decoder_tiles(self):
        log = ("get_learned_condition completed, taking 2.3s\n  |====| 2/4 - 3.10s/it\n"
               "generating latent video completed, taking 12s\nVAE decode ran out of memory; retrying with temporal tiling\n"
               "Reducing VAE decode tiles\nVAE decode ran out of memory; retrying with spatial+temporal tiling\n"
               "  |==| 3/10 - 10.00s/it")
        p = video_gen.parse_progress(log)
        self.assertEqual((p["stage"], p["tile"], p["tiles"], p["retries"]), ("decoding", 3, 10, 2))
        self.assertAlmostEqual(video_gen.estimate_left(p, 832, 480, 49, 4, False, 30), 71, delta=1)
        self.assertEqual(video_gen.parse_progress(log[:60])["stage"], "encoding")

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_anime_adds_its_lora_and_prefix(self):
        clip, args = self._run(key="anime")
        prompt = args[args.index("-p") + 1]
        self.assertTrue(prompt.startswith("anime style, a cat walking"))
        self.assertTrue(prompt.endswith("<lora:wan22_5B_anime_v1.0:0.8>"))
        self.assertIn("--lora-model-dir", args)
        self.assertEqual(clip["prompt"], "a cat walking")          # what the user wrote is what is shown and saved

    @unittest.skipUnless(sys.platform == "win32", "needs Windows drive-letter paths")
    def test_image_to_video_passes_the_picture(self):
        pic = os.path.join(self.tmp.name, "p.png")
        open(pic, "wb").close()
        clip, args = self._run(ref_image=pic, ref_id="20261013-000000-aaaaaa")
        self.assertEqual(args[args.index("-i") + 1], pic)
        self.assertEqual(clip["ref_id"], "20261013-000000-aaaaaa")

    def test_the_image_filter_applies(self):
        with mock.patch.object(image_gen, "_check", return_value=("refused", "filter")):
            with self.assertRaises(video_gen.VideoError) as e:
                video_gen.generate("anything")
        self.assertEqual(e.exception.code, "filter")


if __name__ == "__main__":
    unittest.main()
