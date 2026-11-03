"""1.4.3's polish of what 1.4.2 added: notifications held back by Windows (focus_assist.py), and the rest of part 4
of docs/plans/1.4.3.md."""
import unittest
from pathlib import Path
from unittest.mock import patch

import focus_assist
import remote_access

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
SERVER = (ROOT / "api" / "server.py").read_text(encoding="utf-8")


class FocusAssistTests(unittest.TestCase):
    def test_states(self):
        cases = [((0, 5), {"quiet": False, "why": "", "known": True}),
                 ((1, 5), {"quiet": True, "why": "priority", "known": True}),
                 ((2, 5), {"quiet": True, "why": "alarms", "known": True}),
                 ((0, 3), {"quiet": True, "why": "fullscreen", "known": True}),        # a game full screen
                 ((0, 4), {"quiet": True, "why": "presentation", "known": True}),
                 ((None, None), {"quiet": False, "why": "", "known": False})]          # unreadable: say nothing
        for (profile, quns), want in cases:
            with patch.object(focus_assist, "_profile", return_value=profile), patch.object(focus_assist, "_quns", return_value=quns):
                self.assertEqual(focus_assist.state(), want, (profile, quns))

    def test_reads_this_pc(self):
        s = focus_assist.state()                                             # never raises, whatever Windows says
        self.assertEqual(set(s), {"quiet", "why", "known"})

    def test_opens_the_windows_page(self):
        with patch.object(focus_assist.os, "startfile", create=True) as start, patch.object(focus_assist.sys, "platform", "win32"):
            self.assertTrue(focus_assist.open_settings())
        start.assert_called_once_with("ms-settings:quiethours")

    def test_the_settings_hint_and_routes(self):
        self.assertIn('@app.get("/api/notify/quiet")', SERVER)
        self.assertIn('@app.post("/api/notify/quiet/open")', SERVER)
        self.assertIn('<div id="notify-quiet"></div>', HTML)
        self.assertIn("notifyQuietHint();", HTML)
        # From a phone, the PC's own Windows settings never open.
        self.assertTrue(remote_access.blocked("POST", "/api/notify/quiet/open"))
        self.assertFalse(remote_access.blocked("GET", "/api/notify/quiet"))


class StarterModelTests(unittest.TestCase):
    """The first-run welcome's one-click starter (model_finder.starter): the biggest that fits, no network."""

    def pick(self, vram, ram=16.0):
        import model_finder
        with patch.object(model_finder, "_on_disk_names", return_value=set()):
            r = model_finder.starter({"gpu": "GPU", "vram_gb": vram, "free_vram_gb": vram, "ram_gb": ram, "free_ram_gb": ram})
        return (r["pick"] or {}).get("name"), (r["pick"] or {}).get("fit")

    def test_the_biggest_that_fits_on_the_gpu(self):
        self.assertEqual(self.pick(10), ("Qwen3 8B", "gpu"))
        self.assertEqual(self.pick(6.5), ("Qwen3 8B", "gpu"))
        self.assertEqual(self.pick(5), ("Qwen3 4B", "gpu"))
        self.assertEqual(self.pick(2.6), ("Qwen3 1.7B", "gpu"))

    def test_no_gpu_room_runs_the_smallest_from_ram(self):
        self.assertEqual(self.pick(0, ram=8), ("Qwen3 1.7B", "partial"))
        self.assertEqual(self.pick(0, ram=1), (None, None))

    def test_the_starters_are_real_files(self):
        # Sizes checked against Hugging Face on 2026-10-28 (the fit is computed from them without a request).
        import model_finder
        self.assertEqual([(s["repo"], s["path"], s["size"]) for s in model_finder.STARTERS],
                         [("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", 5027783488),
                          ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", 2497280256),
                          ("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf", 1834426016)])

    def test_the_welcome_has_three_steps(self):
        self.assertIn("const STEPS = ['A chat model', 'A look', 'What stays here'];", HTML)
        self.assertIn("api('/api/model-finder/starter')", HTML)
        self.assertIn('@router.get("/api/model-finder/starter")', (ROOT / "api" / "extras.py").read_text(encoding="utf-8"))
        self.assertIn("if (e.key === 'Escape') { e.preventDefault(); done(); }", HTML)    # skippable at every step


class WhatsNewTests(unittest.TestCase):
    def test_notes_of_a_version(self):
        import app_version
        n = app_version.notes("1.4.2")
        self.assertEqual(n["version"], "1.4.2")
        self.assertFalse(n["markdown"].startswith("# "))                    # the title line is the dialog's own
        self.assertIn("Stop keeps what was written", n["markdown"])
        self.assertEqual(app_version.notes("9.9.9")["markdown"], "")
        self.assertEqual(app_version.notes("../../VERSION")["markdown"], "")   # only a version number reads a file

    def test_shown_once_after_an_update(self):
        self.assertIn("WhatsNew.check();", HTML)
        self.assertIn("if (seen === d.version || !welcomed) return;", HTML)   # same version, or a first install: nothing
        self.assertIn('@app.get("/api/version/notes")', SERVER)


class SettingsTests(unittest.TestCase):
    def test_search_goes_to_the_setting(self):
        self.assertIn("S.setQuery = q; S.setJumped = false;", HTML)
        self.assertIn("if (S.setQuery && S.settingsTab === tab) searchHighlight(S.setQuery);", HTML)
        self.assertIn("@keyframes set-hit", HTML)

    def test_connections_and_web_search_and_network_are_one_part_each(self):
        self.assertIn("const SETTINGS_STACKS = { connections: ['connectors', 'mcp', 'channels'], webnet: ['websearch', 'network'] };", HTML)
        # 1.4.6: nine pages by goal - web search and the network have their own page, connectors theirs.
        self.assertIn("['webnet', [['webnet', 'Web search and network'], ['privacy', 'Privacy']]]", HTML)
        self.assertIn("['connections', [['connections', 'Connectors and channels'], ['remote', 'Phone access']]]", HTML)
        # Every renderer of a section draws into its own section, and checks its section is showing.
        for name in ("connectors", "mcp", "channels", "websearch", "network"):
            self.assertIn(f"setBody('{name}')", HTML)
            self.assertNotIn(f"S.settingsTab !== '{name}'", HTML)
        # The old names still work: they open the part at their section.
        self.assertIn("const section = STACK_OF[tab] ? tab : null;", HTML)

    def test_the_ai_appearances_are_out_of_beta(self):
        self.assertNotIn("AI_APPEARANCES.includes(a) ? ' (Beta)'", HTML)
        self.assertNotIn('AI-inspired <span class="beta-tag">Beta</span>', HTML)


class AiLooksTests(unittest.TestCase):
    """The Claude, ChatGPT, Grok and Gemini appearances follow values measured on the apps' own pages (1.4.3)."""

    def test_each_has_its_own_type(self):
        self.assertIn('--sans: "Inter", ui-sans-serif', HTML)                              # Grok: its own fallback
        self.assertIn('--sans: ui-sans-serif, -apple-system, system-ui, "Segoe UI"', HTML)    # ChatGPT: the system font
        self.assertIn('--sans: "Google Sans Flex", "Google Sans"', HTML)                       # Gemini
        self.assertIn('--serif: Georgia, "Source Serif 4"', HTML)                               # Claude's serif fallback
        css = (ROOT / "static" / "fonts" / "fonts.css").read_text(encoding="utf-8")
        self.assertIn("font-family: 'Google Sans Flex';", css)
        for f in ("googlesansflex-normal-latin.woff2", "googlesansflex-normal-latin-ext.woff2", "OFL-GoogleSansFlex.txt"):
            self.assertTrue((ROOT / "static" / "fonts" / f).is_file(), f)
        self.assertIn("Google Sans Flex", (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8"))

    def test_measured_shapes_and_colours(self):
        self.assertIn('--bg: #131211;', HTML)                                                  # Grok's warm near-black
        self.assertIn(':root[data-ui][data-skin="void"] .composer { border-radius: 24px;', HTML)   # not a pill
        self.assertIn(':root[data-ui][data-skin="mono"] .composer { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; border-radius: 28px;', HTML)
        self.assertIn('--bg: #0e0f12; --bg-sidebar: #1f1f1f;', HTML)                           # Gemini
        self.assertIn("box.classList.toggle('multiline', multi)", HTML)                      # one line opens to two rows


class ImmersiveTests(unittest.TestCase):
    """The AI-inspired appearances' Immersive switch: each app's voice from the local model, honest about what it is."""

    def test_four_styles_each_honest(self):
        import personas
        for skin, pid in personas.AI_STYLES.items():
            text = personas.resolve({}, pid)
            self.assertIn("never claim to be", text, pid)
            self.assertIn("running on this PC", text, pid)
        self.assertIn("made by Anthropic", personas.resolve({}, "ai-clay"))
        self.assertIn("witty", personas.resolve({}, "ai-void"))

    def test_the_switch_is_a_setting_and_ids_stay_builtin(self):
        import personas
        self.assertFalse(personas.clean_settings({})["ai_immersive"])
        self.assertTrue(personas.clean_settings({"ai_immersive": True})["ai_immersive"])
        self.assertFalse(personas.clean_settings({"ai_immersive": "yes"})["ai_immersive"])
        saved = personas.clean_settings({"personas": [{"id": "ai-clay", "name": "Fake", "text": "x"}]})["personas"]
        self.assertNotIn("ai-clay", [p["id"] for p in saved])            # a style can't be saved over
        self.assertIn("ai_immersive: bool | None = None", SERVER)

    def test_the_page_sends_it_in_customs_place(self):
        self.assertIn("return p.id === 'custom' && aiStyle() ? aiStyle() : p.id;", HTML)
        self.assertIn('id="immersive-btn" hidden onclick="toggleImmersive()"', HTML)
        self.assertIn("renderImmersiveBtn();", HTML)


class NcmWorkerTests(unittest.TestCase):
    def test_the_first_level_comes_from_the_checkpoint_when_the_model_cant_write_it(self):
        # Found in 1.4.3's live check: HazyResearch's current TinyStories config has neither predict_level0 nor
        # query_level0, so generate(None, ...) asserted. Like the repo's own eval script, the first code is drawn
        # from generation.level0_unigram.
        src = (ROOT / "workers" / "labs_ncm_worker.py").read_text(encoding="utf-8")
        self.assertIn('getattr(net, "predict_level0", False) or getattr(net, "query_level0", False)', src)
        self.assertIn("torch.multinomial(p, tk.vqvae.vq.levels[0], replacement=True)", src)
        self.assertIn("codes = net.generate(l0, x,", src)


class ShellWatchTests(unittest.TestCase):
    """launcher.pyw stops the server when the desktop app has closed - and only then. Found in 1.4.3's live checks:
    one failed process check, while the PC loaded a 9.7 GB model for the MoE test, stopped the server under a
    running app."""

    def setUp(self):
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader("fx_launcher", str(ROOT / "launcher.pyw"))
        spec = importlib.util.spec_from_loader("fx_launcher", loader)
        self.launcher = importlib.util.module_from_spec(spec)
        loader.exec_module(self.launcher)

    def test_an_unclear_check_is_not_gone(self):
        import psutil
        L = self.launcher
        with patch.object(psutil, "pid_exists", return_value=True), patch.object(psutil, "Process", side_effect=psutil.AccessDenied(1)):
            self.assertTrue(L._alive(1))
        with patch.object(psutil, "pid_exists", return_value=True), patch.object(psutil, "Process", side_effect=OSError("busy")):
            self.assertTrue(L._alive(1))
        with patch.object(psutil, "pid_exists", return_value=True), patch.object(psutil, "Process", side_effect=psutil.NoSuchProcess(1)):
            self.assertFalse(L._alive(1))
        with patch.object(psutil, "pid_exists", return_value=False):
            self.assertFalse(L._alive(1))

    def test_it_takes_three_seconds_gone_in_a_row(self):
        L, server = self.launcher, type("S", (), {"should_exit": False})()
        answers = iter([True, False, True, False, False, True, False, False, False, True])
        seen = []

        def alive(pid):
            v = next(answers)
            seen.append(v)
            return v
        with patch.dict(L.os.environ, {"FX_PARENT_PID": "1234"}), patch.object(L, "_alive", alive), patch.object(L.time, "sleep"):
            L.watch_shell(server)
        self.assertTrue(server.should_exit)
        self.assertEqual(len(seen), 9)              # the odd False answers before never stopped it


if __name__ == "__main__":
    unittest.main()
