"""launcher.pyw - the windowless start: the same llama-server and loopback rules as run.bat, keys.local.bat read
the way cmd sets them, every console child without a window, and which windows count as the app's."""
import importlib.machinery
import importlib.util
import os
import sys
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    loader = importlib.machinery.SourceFileLoader("launcher_under_test", os.path.join(HERE, "launcher.pyw"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.launcher = load()
        with open(os.path.join(HERE, "launcher.pyw"), encoding="utf-8") as f:
            self.src = f.read()

    def tearDown(self):
        os.chdir(self.cwd)

    def test_importing_it_changes_nothing(self):
        self.assertEqual(os.getcwd(), self.cwd)

    def test_same_llama_and_loopback_rules_as_run_bat(self):
        start = self.src[self.src.index("def start_llama"):self.src.index("def stop_tree")]
        self.assertIn('"--models-preset"', start)
        self.assertNotIn("--ctx-size", start)                 # it would override every model's own ctx-size
        # stock llama.cpp first, PrismML's fork last (both lists come from paths.py, local\llama.cpp first)
        self.assertIn("for d in (*paths.llama_dirs(), *paths.prism_dirs()):", self.src)
        import paths
        self.assertEqual(paths.llama_dirs()[0], paths.home("llama.cpp"))
        self.assertEqual(paths.prism_dirs()[0], paths.home("llama.cpp", "prism"))
        self.assertIn('env["AGENTIC_WINDOWLESS"] = "1"', self.src)    # a router the app restarts stays windowless
        self.assertIn('host="127.0.0.1", port=8000', self.src)
        self.assertIn('env.setdefault("LLAMA_HOST", "127.0.0.1")', self.src)
        self.assertNotIn("0.0.0.0", self.src)

    def test_keys_are_read_like_cmd_sets_them(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        with open(os.path.join(tmp, "keys.local.bat"), "w", encoding="utf-8") as f:
            f.write('@echo off\nset AW_TEST_ONE=abc=def\nset "AW_TEST_TWO=two words"\nREM set AW_TEST_NO=x\n')
        with mock.patch.object(self.launcher, "BASE", tmp), mock.patch.dict(os.environ, {}):
            self.launcher.load_keys()
            self.assertEqual(os.environ.get("AW_TEST_ONE"), "abc=def")
            self.assertEqual(os.environ.get("AW_TEST_TWO"), "two words")
            self.assertNotIn("AW_TEST_NO", os.environ)

    @unittest.skipUnless(sys.platform == "win32", "subprocess.DETACHED_PROCESS exists only on Windows")
    def test_console_children_get_no_window_but_own_console_launches_keep_theirs(self):
        seen = []
        original = subprocess.Popen.__init__
        self.addCleanup(setattr, subprocess.Popen, "__init__", original)
        subprocess.Popen.__init__ = lambda self, *a, **k: seen.append(k.get("creationflags", 0))
        self.launcher.quiet_children()
        subprocess.Popen(["powershell", "-Command", "Get-Clipboard"])
        subprocess.Popen(["msedge.exe"], creationflags=subprocess.DETACHED_PROCESS)
        subprocess.Popen(["cmd.exe"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.assertEqual(seen, [subprocess.CREATE_NO_WINDOW, subprocess.DETACHED_PROCESS, subprocess.CREATE_NEW_CONSOLE])

    def test_which_windows_are_the_app(self):
        import app_window
        titles = [(1, "FX Workspace"), (2, "Plan the trip - FX Workspace"), (3, "FX Workspace - HUD"),
                  (4, "Notepad"), (5, "FX Workspace - Personal - Microsoft Edge")]
        with mock.patch.object(app_window, "_windows", return_value=titles, create=True):
            self.assertEqual(self.launcher.app_windows(""), 3)

    def test_run_bat_hands_over_unless_console_and_the_vbs_hides_it(self):
        with open(os.path.join(HERE, "run.bat"), encoding="utf-8") as f:
            bat = f.read()
        self.assertIn('if /i not "%~1"=="console"', bat)
        self.assertLess(bat.index("launcher.pyw"), bat.index("pip install"))        # before any setup step
        with open(os.path.join(HERE, "Agentic Workspace.vbs"), encoding="utf-8") as f:
            vbs = f.read()
        self.assertIn('launcher.pyw""", 0, False', vbs)                                # window style 0: hidden
        self.assertIn("run.bat\"\" console", vbs)                                       # first run: visible setup


if __name__ == "__main__":
    unittest.main()
