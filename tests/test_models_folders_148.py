"""1.4.8: choose where chat models live (not only under the install folder or on C:), pick a model file or folder with
Windows' own picker, and a setup that repairs itself. Temp folders only; no picker is really opened."""
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import paths

import api.server as server
import core.config as config
import core.state as shared


def _client():
    from fastapi.testclient import TestClient
    return TestClient(server.app, base_url="http://127.0.0.1:8000")


class FolderSettingTests(unittest.TestCase):
    def test_only_full_paths_are_kept(self):
        self.assertEqual(config.clean_folder(r"D:\Models"), r"D:\Models")
        self.assertEqual(config.clean_folder(r"\\nas\share\models"), r"\\nas\share\models")
        for bad in ("models", r"..\models", "", None, 5, "D:\\a|b", "C:\\x\ny"):
            self.assertEqual(config.clean_folder(bad), "", bad)

    def test_the_chosen_folder_comes_first_and_the_default_stays(self):
        dirs = paths.model_dirs({"models_folder": r"D:\Models", "model_dirs": [r"E:\More"]})
        self.assertEqual(dirs, [r"D:\Models", paths.home("models"), r"E:\More"])
        self.assertEqual(paths.model_dirs({})[0], paths.home("models"))

    def test_downloads_follow_the_choice(self):
        with mock.patch.dict(shared.app_state, settings={**shared.app_state["settings"], "models_folder": r"D:\Models"}):
            self.assertEqual(paths.model_dirs()[0], r"D:\Models")        # model_finder calls it with no settings


class FolderEndpointTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        p = mock.patch.object(config, "SETTINGS_PATH", os.path.join(tmp.name, "settings.json"))
        p.start()
        self.addCleanup(p.stop)
        before = dict(shared.app_state["settings"])
        self.addCleanup(lambda: shared.app_state.update(settings=before))

    def test_choose_a_main_folder_and_more_folders(self):
        more = os.path.join(self.dir, "more")
        os.makedirs(more)
        with mock.patch("events.log_action"):
            r = _client().post("/api/local-models/folders", json={"main": self.dir, "extra": [more]}).json()
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["main"], os.path.normpath(self.dir))
        self.assertEqual(r["extra"], [os.path.normpath(more)])
        self.assertEqual(config.load_settings()["models_folder"], os.path.normpath(self.dir))   # saved
        with mock.patch("events.log_action"):
            r = _client().post("/api/local-models/folders", json={"main": ""}).json()
        self.assertEqual(r["main"], paths.home("models"))                                       # back to the default

    def test_a_folder_that_doesnt_exist_is_refused(self):
        r = _client().post("/api/local-models/folders", json={"main": os.path.join(self.dir, "nope")}).json()
        self.assertFalse(r["ok"])
        r = _client().post("/api/local-models/folders", json={"extra": ["relative\\path"]}).json()
        self.assertFalse(r["ok"])

    def test_the_picker_returns_a_path_and_opens_nothing_else(self):
        done = subprocess.CompletedProcess([], 0, stdout="D:/Models/qwen.gguf\n", stderr="")
        with mock.patch.object(server.subprocess, "run", return_value=done) as run:
            r = _client().post("/api/local-models/browse", json={"kind": "file"}).json()
        self.assertEqual((r["ok"], r["kind"], r["path"]), (True, "file", os.path.normpath("D:/Models/qwen.gguf")))
        self.assertIn("askopenfilename", run.call_args.args[0][2])
        with mock.patch.object(server.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")) as run:
            r = _client().post("/api/local-models/browse", json={"kind": "folder"}).json()
        self.assertEqual((r["ok"], r["kind"], r["path"]), (True, "folder", ""))                # cancelled
        self.assertIn("askdirectory", run.call_args.args[0][2])

    def test_phone_lists(self):
        import remote_access
        self.assertTrue(remote_access.blocked("POST", "/api/local-models/folders"))
        self.assertTrue(remote_access.blocked("POST", "/api/local-models/browse"))
        self.assertFalse(remote_access.blocked("GET", "/api/local-models/folders"))


class VisionFileTests(unittest.TestCase):
    """A vision file (mmproj) for a model already in the list - one the user downloaded themselves."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.proj = os.path.join(tmp.name, "mmproj-test-f16.gguf")
        with open(self.proj, "wb") as f:
            f.write(b"GGUF" + b"\0" * 16)
        self.notgguf = os.path.join(tmp.name, "readme.txt")
        with open(self.notgguf, "w") as f:
            f.write("hello")

    def _post(self, body, loaded=False):
        with mock.patch("llama_local.list_model_configs", return_value=[{"name": "My-9B"}]), \
                mock.patch("llama_local.set_model_files") as setf, \
                mock.patch("llama_local.router_status", return_value={"loaded": loaded}), \
                mock.patch.object(server, "_models_added", return_value={"restarted": True, "warning": ""}) as restart, \
                mock.patch("events.log_action"):
            r = _client().post("/api/local-models/mmproj", json=body).json()
        return r, setf, restart

    def test_attach_and_remove(self):
        r, setf, restart = self._post({"model": "My-9B", "path": self.proj})
        self.assertTrue(r["ok"], r)
        setf.assert_called_once_with("My-9B", mmproj=os.path.normpath(self.proj))
        restart.assert_not_called()                                   # not loaded: applies at the next load
        r, setf, _ = self._post({"model": "My-9B", "path": ""})
        self.assertTrue(r["ok"])
        setf.assert_called_once_with("My-9B", mmproj=None)

    def test_the_loaded_model_reloads_with_it(self):
        r, _setf, restart = self._post({"model": "My-9B", "path": self.proj}, loaded=True)
        self.assertTrue(r["restarted"])
        restart.assert_called_once()

    def test_refused_for_unknown_models_missing_files_and_non_gguf(self):
        for body in ({"model": "Nope", "path": self.proj}, {"model": "My-9B", "path": self.proj + ".gone"},
                     {"model": "My-9B", "path": self.notgguf}):
            r, setf, _ = self._post(body)
            self.assertFalse(r["ok"], body)
            setf.assert_not_called()

    def test_phone_blocked(self):
        import remote_access
        self.assertTrue(remote_access.blocked("POST", "/api/local-models/mmproj"))


class EngineGpuTests(unittest.TestCase):
    """10 Nov: the engine's NVIDIA runtime files were cut short; it listed no GPU and ran everything on the CPU."""

    def _devices(self, out):
        with mock.patch("diagnostics.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=out, stderr="")):
            import diagnostics
            return diagnostics.gpu_devices("llama-server.exe")

    def test_devices_are_read_from_list_devices(self):
        self.assertEqual(self._devices("Available devices:\n  CUDA0: NVIDIA GeForce RTX 3060 Ti (8191 MiB, 7140 MiB free)\n"),
                         ["CUDA0: NVIDIA GeForce RTX 3060 Ti (8191 MiB, 7140 MiB free)"])
        self.assertEqual(self._devices("Available devices:\n  (none)\n"), [])
        self.assertIsNone(self._devices("error: unknown argument: --list-devices"))   # an older build: unknown

    def test_setup_checks_the_gpu_and_room(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "install.ps1")
        with open(path, encoding="utf-8") as f:
            ps = f.read()
        self.assertIn("function Get-LlamaDevices", ps)
        self.assertIn("the CUDA build cannot see the GPU", ps)        # a new build that can't is not installed
        self.assertIn("function Assert-Room", ps)                      # downloads only with room to unpack them
        self.assertIn("cudart-llama-bin-win-cuda-12.4-x64.zip", ps)    # the repair of an existing install
        self.assertIn("import core.modpath, model_scan", ps)           # the model list step works again


class SelfRepairTests(unittest.TestCase):
    """The desktop app sets itself up and ends a stuck earlier copy; setup finds and installs Python by itself."""

    def _src(self, rel):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel), encoding="utf-8") as f:
            return f.read()

    def test_the_launcher_says_when_an_earlier_copy_is_stuck(self):
        src = self._src("launcher.pyw")
        self.assertIn("STUCK_EXIT = 3", src)
        self.assertIn("return 0 if running else STUCK_EXIT", src)

    def test_the_desktop_app_runs_setup_and_ends_a_stuck_copy(self):
        js = self._src(os.path.join("desktop", "main.js"))
        self.assertIn("runSetup(root)", js)
        self.assertIn("'-Yes', '-SkipShortcut', '-SkipModel', '-Quiet'", js)
        self.assertIn("exited === STUCK_EXIT && !retried", js)
        self.assertNotIn("Run the installer (or install.bat) first", js)

    def test_setup_installs_python_itself_and_never_asks_for_a_rerun(self):
        ps = self._src("install.ps1")
        self.assertIn("function Find-Python", ps)
        self.assertIn("Python Software Foundation", ps)                    # the downloaded installer's signature
        self.assertIn("InstallAllUsers=0", ps)                             # no admin needed
        self.assertNotIn("Close this window and run install.bat again", ps)
        self.assertIn("WindowsApps", ps)                                   # the Store stub isn't taken for Python


if __name__ == "__main__":
    unittest.main()
