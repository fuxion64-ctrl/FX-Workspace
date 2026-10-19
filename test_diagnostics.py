"""Diagnostics (Settings > Diagnostics): the GGUF header reader, the VRAM
estimate per architecture, the fit plans it writes to models.ini (and that it
recognises them afterwards), where a loaded model sits, and the in-process
Windows GPU counters. Everything runs on synthetic headers and a temporary
models.ini - the real one is never touched."""
import os
import sys
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

import diagnostics
import gguf_meta
import llama_local
import system_metrics

MB = 1024 * 1024
_REAL_GETSIZE = os.path.getsize


def _s(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, kv):
    """A GGUF v3 file with only a metadata header. kv: {key: (type, value)};
    arrays are (9, (item_type, [items]))."""
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, (vtype, value) in kv.items():
        out += _s(key) + struct.pack("<I", vtype)
        if vtype == 8:
            out += _s(value)
        elif vtype == 9:
            itype, items = value
            out += struct.pack("<IQ", itype, len(items))
            out += b"".join(_s(i) if itype == 8 else struct.pack(gguf_meta._SCALARS[itype], i) for i in items)
        else:
            out += struct.pack(gguf_meta._SCALARS[vtype], value)
    with open(path, "wb") as f:
        f.write(out)


LLAMA = {"general.architecture": (8, "llama"), "llama.block_count": (4, 32), "llama.attention.head_count": (4, 32),
         "llama.attention.head_count_kv": (4, 8), "llama.embedding_length": (4, 4096), "llama.context_length": (4, 32768),
         "tokenizer.ggml.tokens": (9, (8, [f"t{i}" for i in range(5000)])), "llama.rope.freqs": (9, (6, [1.0, 2.0]))}
DEEPSEEK = {"general.architecture": (8, "deepseek2"), "deepseek2.block_count": (4, 27), "deepseek2.attention.head_count": (4, 16),
            "deepseek2.attention.head_count_kv": (4, 16), "deepseek2.embedding_length": (4, 2048),
            "deepseek2.context_length": (4, 163840), "deepseek2.expert_count": (4, 64),
            "deepseek2.attention.kv_lora_rank": (4, 512), "deepseek2.rope.dimension_count": (4, 64),
            "deepseek2.vocab_size": (4, 102400)}


class Workspace(unittest.TestCase):
    """A temp folder with model files and its own models.ini (MODELS_INI)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="diag-test-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.ini = os.path.join(self.dir, "models.ini")
        env = patch.dict(os.environ, {"MODELS_INI": self.ini})
        env.start()
        self.addCleanup(env.stop)
        llama_local._ini_cache.update(key=None, configs=None)
        self.addCleanup(llama_local._ini_cache.update, key=None, configs=None)

    def model(self, name, kv):
        path = os.path.join(self.dir, name + ".gguf")
        write_gguf(path, kv)
        return path

    def write_ini(self, text):
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write(text)
        llama_local._ini_cache.update(key=None, configs=None)

    def cfg(self, name):
        return next(c for c in llama_local.list_model_configs() if c["name"] == name)


class GgufHeaderTests(Workspace):
    def test_reads_the_shape_and_skips_the_vocabulary(self):
        meta = gguf_meta.read(self.model("m", LLAMA))
        self.assertEqual((meta["arch"], meta["layers"], meta["heads"], meta["kv_heads"]), ("llama", 32, 32, 8))
        self.assertEqual((meta["key_len"], meta["value_len"], meta["trained_ctx"]), (128, 128, 32768))   # 4096 / 32
        self.assertIsNone(meta["raw"]["tokenizer.ggml.tokens"])          # 5000 items: skipped, not kept
        self.assertEqual(meta["raw"]["llama.rope.freqs"], [1.0, 2.0])

    def test_rejects_what_is_not_gguf(self):
        path = os.path.join(self.dir, "x.gguf")
        with open(path, "wb") as f:
            f.write(b"NOPE" + b"\0" * 32)
        with self.assertRaises(gguf_meta.GGUFError):
            gguf_meta.read(path)
        self.assertIsNone(diagnostics.meta_for(path))
        self.assertIsNone(diagnostics.meta_for(os.path.join(self.dir, "missing.gguf")))


class EstimateTests(Workspace):
    def test_kv_cache_per_architecture(self):
        llama = gguf_meta.read(self.model("m", LLAMA))
        # 32 layers x 8 KV heads x (128 + 128) dims x 2 bytes = 128 KiB a token
        self.assertEqual(diagnostics.kv_bytes(llama, 32768), 4096 * MB)
        self.assertEqual(diagnostics.kv_bytes(llama, 32768, "q8_0", "q8_0"), 4096 * MB * 34 / 64)
        hybrid = dict(llama, full_attention_interval=4)                  # Qwen 3.5: 1 layer in 4 attends
        self.assertEqual(diagnostics.kv_bytes(hybrid, 32768), 1024 * MB + 48 * MB)
        mla = gguf_meta.read(self.model("d", DEEPSEEK))                  # one latent (512 + 64 rope) per layer
        self.assertEqual(diagnostics.kv_bytes(mla, 65536), 27 * 65536 * (576 * 2 + 512 * 2))

    def test_cpu_moe_moves_expert_weights_off_the_gpu(self):
        path = self.model("d", DEEPSEEK)
        meta, cfg = gguf_meta.read(path), {"model_path": path}
        with patch.object(diagnostics.os.path, "getsize", side_effect=lambda p: 10000 * MB if p == path else _REAL_GETSIZE(p)):
            full = diagnostics.estimate(meta, cfg, {}, 4096)
            half = diagnostics.estimate(meta, cfg, {}, 4096, cpu_moe=13)
            none = diagnostics.estimate(meta, cfg, {}, 4096, cpu_moe=27)
        self.assertEqual(full["weights_mb"], 10000)
        self.assertEqual(none["weights_mb"], 2000)                       # attention, shared experts, embeddings
        self.assertTrue(none["weights_mb"] < half["weights_mb"] < full["weights_mb"])


class PlanTests(Workspace):
    def setUp(self):
        super().setUp()
        self.path = self.model("llama", LLAMA)
        self.write_ini(f"[*]\nngl = 99\nctx-size = 65536\n\n[Llama]\nmodel = {self.path}\n")

    def assess(self, usable):
        return diagnostics.assess("Llama", self.cfg("Llama"), {"usable_mb": usable})

    def test_the_largest_context_that_fits_with_a_q8_cache(self):
        row = self.assess(3000)
        self.assertEqual(row["status"], "spills")
        self.assertTrue(any("trained on" in i for i in row["issues"]))    # 65536 > 32768
        self.assertEqual(row["change"], {"ctx-size": 32768, "cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on"})
        self.assertLessEqual(row["after"]["total_mb"], 3000)
        llama_local.set_model_options({"Llama": row["change"]})
        again = self.assess(3000)
        self.assertEqual(again["status"], "fits")
        self.assertFalse(again["change"])

    def test_never_below_16k_puts_layers_on_the_cpu_instead(self):
        row = self.assess(1500)                                          # 16k with q8 needs ~1.6 GB here
        self.assertEqual(row["change"], {"ctx-size": 16384, "cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on",
                                         "ngl": "auto", "fit-target": diagnostics.MARGIN_MB})   # [*] still pins ngl = 99
        llama_local.set_model_options({"Llama": row["change"]})
        again = self.assess(1500)                                        # recognised: planned, not spilling
        self.assertEqual(again["status"], "partial")
        self.assertFalse(again["change"])
        self.assertGreater(again["cpu_share"], 0)
        # More room later (a bigger context was set, other apps closed): all on the GPU again.
        llama_local.set_model_options({"Llama": {"ctx-size": 32768}})
        roomy = self.assess(1700)
        self.assertEqual(roomy["change"]["ctx-size"], 16384)
        self.assertEqual((roomy["change"]["ngl"], roomy["change"]["fit-target"]), (None, None))
        llama_local.set_model_options({"Llama": roomy["change"]})
        self.assertNotIn("ngl", llama_local.model_options("Llama").keys() - llama_local.model_options("*").keys())
        self.assertEqual(llama_local.model_options("Llama")["ngl"], "99")    # back to the [*] default

    def test_a_moe_model_keeps_as_many_experts_on_the_gpu_as_fit(self):
        path = self.model("coder", DEEPSEEK)
        self.write_ini(f"[*]\nngl = 99\nctx-size = 65536\n\n[Coder]\nmodel = {path}\n")
        big = lambda p: 9884 * MB if p == path else _REAL_GETSIZE(p)
        with patch.object(diagnostics.os.path, "getsize", side_effect=big):
            row = diagnostics.assess("Coder", self.cfg("Coder"), {"usable_mb": 6720})
            self.assertEqual(row["status"], "spills")
            n = row["change"]["n-cpu-moe"]
            self.assertTrue(0 < n < 27, n)
            self.assertLessEqual(row["after"]["total_mb"], 6720 - diagnostics.MARGIN_MB)
            self.assertIn(f"experts of {n} of its 27 layers", row["plan"])
            self.assertEqual(row["change"]["load-mode"], "none")          # read into RAM at load, not from disk on first use
            llama_local.set_model_options({"Coder": row["change"]})
            again = diagnostics.assess("Coder", self.cfg("Coder"), {"usable_mb": 6720})
        self.assertEqual(again["status"], "fits")
        self.assertEqual(again["cpu_moe"], n)
        self.assertFalse(again["change"])
        llama_local.set_model_options({"Coder": {"load-mode": None}})    # set by hand without it: offered again
        with patch.object(diagnostics.os.path, "getsize", side_effect=big):
            self.assertEqual(diagnostics.assess("Coder", self.cfg("Coder"), {"usable_mb": 6720})["change"], {"load-mode": "none"})

    def test_missing_files_are_reported_not_planned(self):
        self.write_ini(f"[*]\nngl = 99\n\n[Gone]\nmodel = {os.path.join(self.dir, 'gone.gguf')}\n")
        row = diagnostics.assess("Gone", self.cfg("Gone"), {"usable_mb": 6000})
        self.assertEqual(row["status"], "missing")
        self.assertFalse(row["change"])


class FitPolicyTests(Workspace):
    """[*] ngl = auto with a small --fit margin: llama.cpp places each model at
    load time against the VRAM that is free right then, instead of a fixed
    ngl = 99 that spills when another app grows."""

    def setUp(self):
        super().setUp()
        self.path = self.model("llama", LLAMA)
        self.write_ini(f"[*]\nngl = 99\nctx-size = 65536\n\n[Llama]\nmodel = {self.path}\nngl = 99\n\n"
                       f"[Small]\nmodel = {self.path}\nctx-size = 8192\n")

    def test_the_policy_unpins_every_model(self):
        self.assertEqual(diagnostics.fit_policy(), {"*": {"ngl": "auto", "fit-target": diagnostics.MARGIN_MB}, "Llama": {"ngl": None}})
        llama_local.set_model_options(diagnostics.fit_policy())
        self.assertEqual(diagnostics.fit_policy(), {})
        self.assertEqual(llama_local.model_own_options("Llama"), {"model": self.path})
        # Now a model that doesn't fit is placed by --fit (not spilling), and its plan leaves ngl to [*].
        row = diagnostics.assess("Llama", self.cfg("Llama"), {"usable_mb": 1500})
        self.assertEqual(row["status"], "partial")                       # at 65536 most of it would run on the CPU:
        self.assertEqual(row["change"], {"ctx-size": 16384, "cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on"})
        llama_local.set_model_options({"Llama": row["change"]})
        settled = diagnostics.assess("Llama", self.cfg("Llama"), {"usable_mb": 1500})
        self.assertEqual((settled["status"], settled["change"]), ("partial", {}))   # the best this card can do
        self.assertNotIn("plan", settled)

    def test_hybrid_models_stay_whole_on_the_gpu(self):
        # Qwen 3.5's Gated Delta Net crawls when split between GPU and CPU (measured: no 120-token reply in 10 min),
        # so these keep ngl = 99 - even under the [*] auto policy - and are never planned as a split.
        hybrid = self.model("hybrid", {**LLAMA, "llama.full_attention_interval": (4, 4)})
        self.write_ini(f"[*]\nngl = auto\nfit-target = 256\nctx-size = 32768\n\n[Hybrid]\nmodel = {hybrid}\n")
        roomy = diagnostics.assess("Hybrid", self.cfg("Hybrid"), {"usable_mb": 6000})
        self.assertEqual((roomy["status"], roomy["change"]), ("fits", {"ngl": 99}))
        tight = diagnostics.assess("Hybrid", self.cfg("Hybrid"), {"usable_mb": 500})
        self.assertEqual(tight["status"], "spills")                     # not "partial": --fit may not split it
        self.assertEqual(tight["change"], {"ctx-size": 16384, "cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on", "ngl": 99})
        self.assertIn("stays whole on the GPU", tight["plan"])
        llama_local.set_model_options({"Hybrid": tight["change"]})
        self.assertEqual(diagnostics.fit_policy(), {})                   # the pin is kept
        settled = diagnostics.assess("Hybrid", self.cfg("Hybrid"), {"usable_mb": 500})
        self.assertEqual((settled["status"], settled["change"]), ("spills", {}))   # nothing better on this card

    def test_plans_use_the_everyday_baseline_not_a_spike(self):
        snap = {"gpu": {"vram_total_mb": 8192, "vram_used_mb": 1900, "name": "Test GPU"}}
        with patch.object(system_metrics, "snapshot", return_value=snap), \
             patch.object(system_metrics, "gpu_process_memory", return_value=None):
            g = diagnostics.gpu_info()
        self.assertEqual((g["other_mb"], g["usable_mb"]), (1900, 8192 - 1900 - diagnostics.MARGIN_MB))
        self.assertEqual(g["plan_other_mb"], diagnostics.DEFAULT_OTHER_MB)
        self.assertEqual(g["plan_usable_mb"], 8192 - diagnostics.DEFAULT_OTHER_MB - diagnostics.MARGIN_MB)
        quiet = dict(snap, gpu=dict(snap["gpu"], vram_used_mb=700))           # a quieter PC than the baseline: its own number
        with patch.object(system_metrics, "snapshot", return_value=quiet), \
             patch.object(system_metrics, "gpu_process_memory", return_value=None):
            self.assertEqual(diagnostics.gpu_info()["plan_other_mb"], 700)
        row = diagnostics.assess("Llama", self.cfg("Llama"), {"usable_mb": 1000, "plan_usable_mb": 3000})
        self.assertEqual(row["usable_mb"], 3000)

    @unittest.skipUnless(sys.platform == "win32", "looks up drive letters (Windows)")
    def test_apply_all_writes_the_policy_then_the_plans_with_one_backup(self):
        gpu = {"name": "Test GPU", "total_mb": 4000, "used_mb": 900, "other_mb": 900, "usable_mb": 2844,
               "plan_other_mb": 900, "plan_usable_mb": 2844, "model_mb": 0, "spilled_mb": 0}
        with open(self.ini, encoding="utf-8") as f:
            before = f.read()
        with patch.object(diagnostics, "gpu_info", return_value=gpu), patch.object(diagnostics, "_server_exe", return_value=None), \
             patch.object(llama_local, "server_models", return_value=None), \
             patch.object(diagnostics, "drive_media", return_value={"media": "HDD", "name": "Test HDD"}):
            result = diagnostics.run(self.dir)
            self.assertEqual(result["checks"][0]["fix"]["id"], "all")
            self.assertTrue(any(c.get("fix", {}).get("id") == "fit-policy" for c in result["checks"]))
            self.assertTrue(any("on a hard disk" in c["title"] for c in result["checks"]))
            out = diagnostics.fix("all", self.dir)
            after = diagnostics.last()
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["changed"], ["*", "Llama"])
        with open(out["backup"], encoding="utf-8") as f:
            self.assertEqual(f.read(), before)                                  # the file as it was before Apply all
        self.assertEqual([f for f in os.listdir(self.dir) if ".bak-" in f], [os.path.basename(out["backup"])])
        self.assertEqual(llama_local.model_options("*")["ngl"], "auto")
        self.assertEqual(llama_local.model_options("Llama")["ctx-size"], "32768")
        self.assertFalse(any(r.get("change") for r in after["models"]))
        self.assertFalse(any(c.get("fix", {}).get("id") in ("all", "fit-policy") for c in after["checks"]))


class ModelsIniTests(Workspace):
    def test_options_are_written_with_a_backup(self):
        self.write_ini("; my notes\n[*]\nngl = 99\n\n[A]\nmodel = a.gguf\nbatch-size = 512\n")
        backup = llama_local.set_model_options({"A": {"ctx-size": 16384, "batch-size": None}, "*": {"fit-target": 512}})
        with open(backup, encoding="utf-8") as f:
            self.assertIn("; my notes", f.read())                         # the old file, comments and all
        opts = llama_local.model_options("A")
        self.assertEqual(opts["ctx-size"], "16384")
        self.assertNotIn("batch-size", opts)
        self.assertEqual(llama_local.model_options("*")["fit-target"], "512")
        with self.assertRaises(ValueError):
            llama_local.set_model_options({"A": {"model": "other.gguf"}})   # only the options diagnostics owns
        with self.assertRaises(KeyError):
            llama_local.set_model_options({"Nope": {"ctx-size": 1}})


class PlacementTests(Workspace):
    def setUp(self):
        super().setUp()
        path = self.model("llama", LLAMA)
        self.write_ini(f"[*]\nngl = 99\n\n[Llama]\nmodel = {path}\nctx-size = 32768\n")   # needs ~4.6 GB (f16)
        ctx = patch.object(llama_local, "model_context_limit", return_value=32768)
        ctx.start()
        self.addCleanup(ctx.stop)

    def where(self, dedicated, shared):
        with patch.object(system_metrics, "gpu_process_memory", return_value={"dedicated_mb": dedicated, "shared_mb": shared}):
            return diagnostics.placement("Llama")

    def test_on_the_gpu_partly_on_the_cpu_or_spilled(self):
        self.assertEqual(self.where(4800, 0)["state"], "gpu")
        self.assertEqual(self.where(2000, 0)["state"], "partial")       # 2.6 GB is neither in VRAM nor lent RAM
        spilled = self.where(3500, 1200)                                 # needs ~4.6 GB: ~1.2 GB came from RAM
        self.assertEqual(spilled["state"], "spilled")
        self.assertAlmostEqual(spilled["spilled_mb"], 1169, delta=5)
        # RAM llama.cpp pins on purpose (weights run on the CPU, load-mode = none) is not a spill.
        self.assertEqual(self.where(4800, 8000)["state"], "gpu")
        self.assertIsNone(diagnostics.placement(None))


@unittest.skipUnless(os.name == "nt", "Windows GPU counters")
class GpuCounterTests(unittest.TestCase):
    def test_counters_read_in_process_without_powershell(self):
        counters = system_metrics._GpuCounters()
        if not counters.ok:
            self.skipTest("GPU Process Memory counters unavailable on this PC")
        with patch("subprocess.run", side_effect=AssertionError("no subprocess")):
            got = counters.read({os.getpid()})
        self.assertEqual(set(got), {"dedicated_mb", "shared_mb"})
        self.assertIsNone(counters.read(set()))


if __name__ == "__main__":
    unittest.main()
