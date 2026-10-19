"""Tests for productivity_tools.py. subprocess/desktop_input are mocked -
the same OS boundaries sys_tools.py mocks (schtasks.exe, powershell.exe)
plus desktop_input's real capture_screenshot()/find_window()."""
import tempfile
import unittest
import sys
from unittest.mock import MagicMock, patch

import productivity_tools as pt


def fake_run(returncode=0, stdout="", stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class SendNotificationTests(unittest.TestCase):
    def test_non_windows_rejected(self):
        with patch.object(pt, "IS_WINDOWS", False):
            result = pt.send_notification("Title", "Message")
        self.assertFalse(result["ok"])

    def test_spawns_powershell_without_blocking(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt.subprocess, "Popen") as popen:
            result = pt.send_notification("Reminder", "Time's up", duration_seconds=5)
        self.assertTrue(result["ok"])
        popen.assert_called_once()
        cmd = popen.call_args[0][0]
        self.assertEqual(cmd[0], "powershell")

    def test_duration_is_clamped(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt.subprocess, "Popen") as popen:
            pt.send_notification("T", "M", duration_seconds=9999)
        script = popen.call_args[0][0][-1]
        self.assertIn("30000", script)   # clamped to 30s -> 30000ms

    def test_single_quotes_are_escaped(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt.subprocess, "Popen") as popen:
            pt.send_notification("It's here", "Don't miss it")
        script = popen.call_args[0][0][-1]
        self.assertIn("It''s here", script)
        self.assertIn("Don''t miss it", script)


class ScheduleTaskTests(unittest.TestCase):
    def setUp(self):
        # Reminder scripts go to a temp folder, never the real data/scheduled.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(pt, "SCRIPT_DIR", tmp.name)
        p.start()
        self.addCleanup(p.stop)

    def test_invalid_name_rejected(self):
        with patch.object(pt, "IS_WINDOWS", True):
            result = pt.schedule_task("bad/name", delay_seconds=60, message="hi")
        self.assertFalse(result["ok"])
        self.assertIn("plain characters", result["message"])

    def test_requires_command_or_message(self):
        with patch.object(pt, "IS_WINDOWS", True):
            result = pt.schedule_task("MyTask", delay_seconds=60)
        self.assertFalse(result["ok"])
        self.assertIn("Provide either", result["message"])

    def test_delay_too_short_rejected(self):
        with patch.object(pt, "IS_WINDOWS", True):
            result = pt.schedule_task("MyTask", delay_seconds=5, message="hi")
        self.assertFalse(result["ok"])
        self.assertIn("at least 30", result["message"])

    def test_bad_run_at_format_rejected(self):
        with patch.object(pt, "IS_WINDOWS", True):
            result = pt.schedule_task("MyTask", run_at="not-a-date", message="hi")
        self.assertFalse(result["ok"])
        self.assertIn("YYYY-MM-DD HH:MM", result["message"])

    def test_delay_based_schedule_calls_schtasks(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt, "_run", return_value=fake_run(0, "SUCCESS")) as run:
            result = pt.schedule_task("MyReminder", delay_seconds=120, message="Take a break")
        self.assertTrue(result["ok"])
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[0], "schtasks")
        self.assertIn("AgenticWorkspace_MyReminder", cmd)

    def test_command_based_schedule_uses_command_as_action(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt, "_run", return_value=fake_run(0, "SUCCESS")) as run:
            result = pt.schedule_task("Backup", delay_seconds=120, command="backup.exe --run")
        self.assertTrue(result["ok"])
        cmd = run.call_args[0][0]
        tr_index = cmd.index("/tr")
        self.assertEqual(cmd[tr_index + 1], "backup.exe --run")

    def test_schtasks_failure_is_reported(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt, "_run", return_value=fake_run(1, "", "Access denied")):
            result = pt.schedule_task("MyTask", delay_seconds=60, message="hi")
        self.assertFalse(result["ok"])
        self.assertIn("Access denied", result["message"])

    def test_start_date_follows_the_windows_short_date_order(self):
        from datetime import datetime
        when = datetime(2026, 3, 4, 9, 30)
        self.assertEqual(pt.schtasks_date(when, "M/d/yyyy"), "03/04/2026")        # en-US
        self.assertEqual(pt.schtasks_date(when, "dd/MM/yyyy"), "04/03/2026")      # en-GB, fr-FR
        self.assertEqual(pt.schtasks_date(when, "yyyy-MM-dd"), "2026/03/04")      # sv-SE, ISO
        self.assertEqual(pt.schtasks_date(when, "dd.MM.yyyy"), "04/03/2026")      # de-DE
        self.assertEqual(pt.schtasks_date(when, "nonsense"), "03/04/2026")        # unknown -> month first
        with patch.object(pt, "IS_WINDOWS", True), patch.object(pt, "_short_date_pattern", return_value="dd/MM/yyyy"), \
             patch.object(pt, "_run", return_value=fake_run(0, "SUCCESS", "")) as run:
            pt.schedule_task("Dated", run_at="2027-03-04 09:30", message="hi")
        args = run.call_args[0][0]
        self.assertEqual(args[args.index("/sd") + 1], "04/03/2027")


class CancelAndListScheduledTaskTests(unittest.TestCase):
    def setUp(self):
        # Reminder scripts go to a temp folder, never the real data/scheduled.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(pt, "SCRIPT_DIR", tmp.name)
        p.start()
        self.addCleanup(p.stop)

    def test_cancel_success(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt, "_run", return_value=fake_run(0, "SUCCESS")) as run:
            result = pt.cancel_scheduled_task("MyTask")
        self.assertTrue(result["ok"])
        cmd = run.call_args[0][0]
        self.assertIn("AgenticWorkspace_MyTask", cmd)

    def test_cancel_not_found(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt, "_run", return_value=fake_run(1, "", "")):
            result = pt.cancel_scheduled_task("Nonexistent")
        self.assertFalse(result["ok"])

    def test_list_only_shows_agent_scheduled_tasks(self):
        stdout = (
            "TaskName: \\AgenticWorkspace_Reminder1\nStatus: Ready\n\n"
            "TaskName: \\Microsoft\\Windows\\SomeOtherTask\nStatus: Ready\n\n"
            "TaskName: \\AgenticWorkspace_Backup\nStatus: Ready\n"
        )
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt, "_run", return_value=fake_run(0, stdout)):
            result = pt.list_scheduled_tasks()
        self.assertTrue(result["ok"])
        self.assertIn("Reminder1", result["message"])
        self.assertIn("Backup", result["message"])
        self.assertNotIn("SomeOtherTask", result["message"])

    def test_list_with_no_agent_tasks(self):
        with patch.object(pt, "IS_WINDOWS", True), \
             patch.object(pt, "_run", return_value=fake_run(0, "TaskName: \\Microsoft\\Windows\\Other\n")):
            result = pt.list_scheduled_tasks()
        self.assertFalse(result["ok"])


class ClipboardHistoryTests(unittest.TestCase):
    def setUp(self):
        pt._CLIPBOARD_HISTORY.clear()

    def test_empty_history_reports_no_data(self):
        result = pt.clipboard_history()
        self.assertFalse(result["ok"])

    def test_records_and_lists_samples(self):
        pt.record_clipboard_sample("first")
        pt.record_clipboard_sample("second")
        result = pt.clipboard_history()
        self.assertTrue(result["ok"])
        self.assertIn("first", result["message"])
        self.assertIn("second", result["message"])

    def test_consecutive_duplicates_are_not_recorded_twice(self):
        pt.record_clipboard_sample("same")
        pt.record_clipboard_sample("same")
        self.assertEqual(len(pt._CLIPBOARD_HISTORY), 1)

    def test_empty_text_is_not_recorded(self):
        pt.record_clipboard_sample("")
        self.assertEqual(len(pt._CLIPBOARD_HISTORY), 0)

    def test_history_is_capped(self):
        for i in range(pt._CLIPBOARD_HISTORY_MAX + 10):
            pt.record_clipboard_sample(f"entry-{i}")
        self.assertEqual(len(pt._CLIPBOARD_HISTORY), pt._CLIPBOARD_HISTORY_MAX)

    def test_limit_is_respected(self):
        for i in range(10):
            pt.record_clipboard_sample(f"entry-{i}")
        result = pt.clipboard_history(limit=3)
        # most recent 3 entries only
        self.assertIn("entry-9", result["message"])
        self.assertNotIn("entry-6", result["message"])


class ScreenshotToFileTests(unittest.TestCase):
    def test_capture_failure_is_reported(self):
        with patch.object(pt.di, "capture_screenshot", return_value=(None, {})):
            result = pt.screenshot_to_file()
        self.assertFalse(result["ok"])

    def test_saves_with_generated_filename(self):
        fake_img = MagicMock()
        with patch.object(pt.di, "capture_screenshot", return_value=(fake_img, {"orig_w": 1920, "orig_h": 1080})), \
             patch.object(pt.os, "makedirs"):
            result = pt.screenshot_to_file()
        self.assertTrue(result["ok"])
        fake_img.save.assert_called_once()
        self.assertIn("1920x1080", result["message"])

    @unittest.skipUnless(sys.platform == "win32", "window focusing is Windows-only")
    def test_focuses_window_first_when_given(self):
        fake_img = MagicMock()
        with patch.object(pt.di, "find_window", return_value=(42, "Notepad")) as find_window, \
             patch.object(pt.di, "focus_window") as focus, \
             patch.object(pt.di, "capture_screenshot", return_value=(fake_img, {"orig_w": 800, "orig_h": 600})), \
             patch.object(pt.os, "makedirs"), \
             patch.object(pt.time, "sleep"):
            result = pt.screenshot_to_file(window_title="Notepad")
        self.assertTrue(result["ok"])
        find_window.assert_called_once_with("Notepad")
        focus.assert_called_once_with(42)

    def test_save_failure_is_reported(self):
        fake_img = MagicMock()
        fake_img.save.side_effect = OSError("disk full")
        with patch.object(pt.di, "capture_screenshot", return_value=(fake_img, {"orig_w": 800, "orig_h": 600})), \
             patch.object(pt.os, "makedirs"):
            result = pt.screenshot_to_file()
        self.assertFalse(result["ok"])
        self.assertIn("disk full", result["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
