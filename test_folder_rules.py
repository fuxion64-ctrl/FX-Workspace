"""Per-folder permission rules: they only relax file-write prompts inside the
named folders, and nothing else. Lockdown, per-tool overrides, tainted
requests, secret files, this app's own folder and path tricks all still ask."""
import json
import sys
import os
import shutil
import subprocess
import tempfile
import unittest

import permissions
import replay_harness as rh


def settings(folders, **extra):
    return {**permissions.DEFAULT_PERMISSION_SETTINGS, "folder_rules": [{"path": f} for f in folders], **extra}


class RuleValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rules_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_project_folder_is_accepted(self):
        self.assertIsNone(permissions.folder_rule_problem(self.tmp))

    def test_broad_or_protected_locations_are_refused(self):
        drive = os.path.splitdrive(self.tmp)[0] + "\\"
        refused = [drive, os.path.expanduser("~"), os.path.dirname(os.path.expanduser("~")),
                   os.environ.get("SystemRoot", r"C:\Windows"), os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"),
                   os.path.dirname(os.path.abspath(permissions.__file__)),
                   os.path.join(os.path.dirname(os.path.abspath(permissions.__file__)), "data"),
                   "relative\\folder", r"\\server\share\x", os.path.join(self.tmp, "missing"), "", None]
        for path in refused:
            self.assertIsNotNone(permissions.folder_rule_problem(path), path)

    def test_existing_settings_load_unchanged(self):
        self.assertEqual(permissions.clean_settings({})["folder_rules"], [])
        # A rule on a drive that is unplugged right now is kept, not silently dropped.
        gone = os.path.join(self.tmp, "on-a-drive-that-is-unplugged")
        self.assertEqual(permissions.clean_settings(settings([gone]))["folder_rules"], [{"path": gone}])
        # Bad entries from a hand-edited file degrade to nothing.
        self.assertEqual(permissions.clean_settings({"folder_rules": "C:\\"})["folder_rules"], [])
        self.assertEqual(permissions.clean_settings({"folder_rules": [{"path": "C:\\"}, 5]})["folder_rules"], [])


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="rules_")
        self.rule = os.path.join(self.root, "game")
        self.outside = os.path.join(self.root, "other")
        os.makedirs(self.rule)
        os.makedirs(self.outside)
        self.s = settings([self.rule])

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def d(self, tool, args, s=None, tainted=False):
        return permissions.decide(tool, s or self.s, tainted=tainted, args=args)

    def test_file_writes_inside_run_without_asking(self):
        inside = os.path.join(self.rule, "src", "main.lua")
        self.assertEqual(self.d("write_file", {"filename": inside, "content": "x"}), "auto")
        self.assertEqual(self.d("edit_file", {"filename": inside, "old_text": "a", "new_text": "b"}), "auto")
        self.assertEqual(self.d("apply_patch", {"filename": inside, "edits": []}), "auto")
        self.assertEqual(self.d("make_dir", {"path": os.path.join(self.rule, "assets")}), "auto")
        self.assertEqual(self.d("move_path", {"source": inside, "destination": os.path.join(self.rule, "b.lua")}), "auto")
        self.assertEqual(self.d("rename_path", {"path": inside, "new_name": "init.lua"}), "auto")
        self.assertEqual(permissions.explain("write_file", self.s, args={"filename": inside})[1],
                         f"folder rule: {os.path.realpath(self.rule)}")

    def test_everything_else_still_asks(self):
        inside = os.path.join(self.rule, "a.txt")
        self.assertEqual(self.d("write_file", {"filename": os.path.join(self.outside, "a.txt")}), "ask")
        self.assertEqual(self.d("delete_path", {"path": inside}), "ask")
        self.assertEqual(self.d("run_powershell", {"command": "Set-Content a.txt x", "cwd": self.rule}), "ask")
        self.assertEqual(self.d("unzip_file", {"zip_path": inside, "destination": self.rule}), "ask")
        self.assertEqual(self.d("download_file", {"url": "https://example.com", "filename": inside}), "ask")
        self.assertEqual(self.d("write_file", {"filename": inside}, tainted=True), "ask")
        self.assertEqual(self.d("write_file", {}), "ask")
        self.assertEqual(self.d("write_file", None), "ask")

    def test_copy_and_move_need_both_ends_inside(self):
        inside = os.path.join(self.rule, "a.txt")
        out = os.path.join(self.outside, "a.txt")
        self.assertEqual(self.d("copy_path", {"source": out, "destination": inside}), "ask")
        self.assertEqual(self.d("move_path", {"source": inside, "destination": out}), "ask")

    @unittest.skipUnless(sys.platform == "win32", "needs Windows paths (backslashes, case-insensitive NTFS)")
    def test_path_tricks_do_not_escape(self):
        self.assertEqual(self.d("write_file", {"filename": os.path.join(self.rule, "..", "other", "x.txt")}), "ask")
        self.assertEqual(self.d("write_file", {"filename": self.rule + "-evil\\x.txt"}), "ask")   # prefix, not inside
        self.assertEqual(self.d("rename_path", {"path": os.path.join(self.rule, "a"), "new_name": "..\\..\\x"}), "ask")
        self.assertEqual(self.d("rename_path", {"path": os.path.join(self.rule, "a"), "new_name": ".."}), "ask")
        self.assertEqual(self.d("write_file", {"filename": os.path.join(self.rule.upper(), "CASE.txt")}), "auto")

    @unittest.skipUnless(sys.platform == "win32", "needs NTFS junctions (cmd /c mklink /J)")
    def test_a_junction_inside_the_folder_does_not_widen_it(self):
        link = os.path.join(self.rule, "link")
        made = subprocess.run(["cmd", "/c", "mklink", "/J", link, self.outside], capture_output=True).returncode == 0
        if not made:
            self.skipTest("could not create a junction here")
        try:
            self.assertEqual(self.d("write_file", {"filename": os.path.join(link, "x.txt")}), "ask")
        finally:
            os.rmdir(link)

    def test_secret_files_and_this_app_always_ask(self):
        self.assertEqual(self.d("write_file", {"filename": os.path.join(self.rule, ".env")}), "ask")
        self.assertEqual(self.d("write_file", {"filename": os.path.join(self.rule, "keys.local.bat")}), "ask")

    def test_lockdown_overrides_and_deny_win(self):
        inside = {"filename": os.path.join(self.rule, "a.txt")}
        self.assertEqual(self.d("write_file", inside, settings([self.rule], permission_mode="lockdown")), "ask")
        self.assertEqual(self.d("write_file", inside, settings([self.rule], permission_overrides={"write_file": "ask"})), "ask")
        self.assertEqual(self.d("write_file", inside, settings([self.rule], permission_overrides={"write_file": "deny"})), "deny")

    def test_without_args_nothing_is_relaxed(self):
        # The desktop-router batch path has no per-path arguments: rules never apply there.
        self.assertEqual(permissions.decide("write_file", self.s), "ask")
        self.assertEqual(permissions.batch_decision(["write_file"], self.s), "ask")


class FolderRuleReplayTests(unittest.TestCase):
    """Through the real loop: a write inside a rule folder runs without a card
    and is still journaled (undoable); a write outside still gets a card."""

    def _fixture(self, target, approvals):
        return {
            "format": "agentic-replay/1",
            "meta": {"name": "folder rule", "provenance": "reconstructed", "model": "unrecorded", "evidence": "test"},
            "setup": {"mode": "local", "entry": "loop", "turn_policy": "action",
                      "settings": {"folder_rules": [{"path": "{sandbox}\\game"}]}},
            "request": "Write the file",
            "responses": [
                {"finish_reason": "tool_calls", "tool_calls": [{"id": "w1", "name": "write_file", "arguments": json.dumps(
                    {"filename": target, "content": "print('hi')\n", "justification": "Write it"})}]},
                {"finish_reason": "stop", "content": "Done."},
            ],
            "approvals": approvals,
            "tools": {"mode": "real"},
        }

    @unittest.skipUnless(sys.platform == "win32", "needs Windows paths (backslashes)")
    def test_inside_runs_without_a_card_and_stays_undoable(self):
        with tempfile.TemporaryDirectory() as sandbox:
            os.makedirs(os.path.join(sandbox, "game"))
            run = rh.run_fixture(self._fixture("{sandbox_json}\\game\\main.lua", []), sandbox=sandbox)
            self.assertEqual(run.approvals, [])
            self.assertEqual([n for n, _ in run.executed], ["write_file"])
            self.assertTrue(os.path.exists(os.path.join(sandbox, "game", "main.lua")))
            tool_ev = [e for e in run.events if e.get("kind") == "tool" and e.get("status") != "running"]
            self.assertTrue(tool_ev and tool_ev[-1].get("undo_id"), "a rule-approved write must still be journaled")

    def test_outside_still_asks(self):
        with tempfile.TemporaryDirectory() as sandbox:
            os.makedirs(os.path.join(sandbox, "game"))
            run = rh.run_fixture(self._fixture("{sandbox_json}\\elsewhere.txt", [False]), sandbox=sandbox)
            self.assertEqual(len(run.approvals), 1)
            self.assertEqual(run.executed, [])
            self.assertFalse(os.path.exists(os.path.join(sandbox, "elsewhere.txt")))


if __name__ == "__main__":
    unittest.main()
