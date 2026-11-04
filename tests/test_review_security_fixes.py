"""Security review regressions. No real keys, browser process, or external request."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main  # noqa: F401 -- production wiring, in the isolated test checkout
import agent.research as research
import browser
import core.config as config
import core.state as shared
import netlayer
import permissions
import search_engine
from tools import approvals, dispatch, files


class PrivateStorageTests(unittest.TestCase):
    def test_saved_search_and_finance_keys_never_reach_file_tools(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(config, "BASE_DIR", folder):
            path = Path(folder, "data", "search_keys.json")
            keys = search_engine.KeyStore(str(path))
            keys.set("brave", "synthetic-brave-canary")
            keys.set("fred", "synthetic-fred-canary")
            self.assertEqual(permissions.decide("read_file", {}), "auto")
            read = dispatch.execute_tool("read_file", {"filename": str(path)})
            self.assertFalse(read["ok"])
            self.assertNotIn("canary", read["message"])
            found = dispatch.execute_tool("search_files", {"directory": folder, "content_query": "canary"})
            self.assertNotIn("search_keys.json", found["message"])

    @unittest.skipUnless(os.name == "nt", "NTFS path alias")
    def test_unnamed_stream_cannot_bypass_secret_protection(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(config, "BASE_DIR", folder):
            for rel in (".env", "settings.json", "data/search_keys.json"):
                path = Path(folder, rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic-canary", encoding="utf-8")
                alias = str(path) + "::$DATA"
                self.assertTrue(files._is_secret_file(alias), alias)
                self.assertFalse(dispatch.execute_tool("read_file", {"filename": alias})["ok"])


class ResearchPermissionsTests(unittest.TestCase):
    def setUp(self):
        for p in (patch.dict(shared.app_state, {"untrusted_seen": [], "waiting_for_os_approval": False}),
                  patch.object(research.event_log, "emit_event", return_value={}),
                  patch.object(research.event_log, "finish_event"),
                  patch.object(research.run_control, "checkpoint")):
            p.start()
            self.addCleanup(p.stop)

    def test_denied_tools_are_not_called_in_new_or_resumed_research(self):
        settings = {"permission_overrides": {"web_search": "deny", "fetch_page": "deny"}}
        with tempfile.TemporaryDirectory() as folder, patch.object(research, "LEDGER_DIR", folder), \
                patch.object(search_engine, "web_search_tool") as search, \
                patch.object(search_engine, "fetch_page_tool") as fetch, \
                patch.object(approvals, "ask_batch_approval_os") as ask:
            r = research.Research("Synthetic question", "normal", settings)
            r.save("stopped")
            resumed = research.Research.load_unfinished(settings)
            self.assertIsNotNone(resumed)
            for current in (r, resumed):
                self.assertEqual(current.search(["Synthetic query"]), [])
                self.assertEqual(current.read_pages(["https://example.invalid/a"], "question"), ("", []))
            search.assert_not_called()
            fetch.assert_not_called()
            ask.assert_not_called()

    def test_lockdown_and_ask_override_use_the_normal_approval_card(self):
        for settings in ({"permission_mode": "lockdown"},
                         {"permission_overrides": {"web_search": "ask"}}):
            with self.subTest(settings=settings), \
                    patch.object(search_engine, "web_search_tool", return_value={"ok": True, "data": {"results": []}}) as search, \
                    patch.object(approvals, "ask_batch_approval_os", return_value=False) as ask:
                r = research.Research("Synthetic question", "normal", settings)
                r.search(["Synthetic query"])
                search.assert_not_called()
                action = ask.call_args.args[0][0]
                self.assertEqual(action["tool_name"], "web_search")
                self.assertEqual(action["args"]["query"], "Synthetic query")
                ask.return_value = True
                r.search(["Approved query"])
                search.assert_called_once()
                self.assertFalse(shared.app_state["waiting_for_os_approval"])

    def test_deny_changed_while_card_open_wins(self):
        settings = {"permission_overrides": {"web_search": "ask"}}
        def approve(*args, **kwargs):
            settings["permission_overrides"]["web_search"] = "deny"
            return True
        with patch.object(approvals, "ask_batch_approval_os", side_effect=approve), \
                patch.object(search_engine, "web_search_tool") as search:
            research.Research("q", "normal", settings).search(["query"])
            search.assert_not_called()

    def test_replaced_live_settings_are_checked_after_approval(self):
        settings = {"permission_overrides": {"web_search": "ask"}}
        def approve(*args, **kwargs):
            shared.app_state["settings"] = {"permission_overrides": {"web_search": "deny"}}
            return True
        with patch.dict(shared.app_state, {"settings": settings}), \
                patch.object(approvals, "ask_batch_approval_os", side_effect=approve), \
                patch.object(search_engine, "web_search_tool") as search:
            research.Research("q", "normal", settings).search(["query"])
            search.assert_not_called()


class BrowserPolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"net_route": "direct", "net_require_vpn": True}
        self.vpn = {"active": False}
        for p in (patch.object(netlayer, "_settings_fn", lambda: self.settings),
                  patch.object(netlayer, "vpn_status", side_effect=lambda **kw: self.vpn),
                  patch.object(netlayer, "_log", side_effect=lambda entry: entry)):
            p.start()
            self.addCleanup(p.stop)
        self.b = browser.AgentBrowser()
        self.addCleanup(self.b.close)
        p = patch.object(browser.subprocess, "Popen", side_effect=AssertionError("must not launch a real browser"))
        self.spawn = p.start()
        self.addCleanup(p.stop)

    def _existing(self):
        cdp = Mock()
        cdp.send.return_value = {}
        proc = Mock()
        proc.poll.return_value = None
        self.b._cdp, self.b._proc, self.b._route = cdp, proc, netlayer.DIRECT
        return cdp, proc

    def test_direct_startup_requires_vpn(self):
        with self.assertRaises(netlayer.RouteBlocked):
            browser.route_args()
        with patch.object(self.b, "_alive", side_effect=AssertionError("must gate before probing browser")):
            with self.assertRaises(netlayer.RouteBlocked):
                self.b.ensure()
        self.spawn.assert_not_called()

    def test_reuse_closes_browser_before_navigation_when_vpn_is_lost(self):
        cdp, proc = self._existing()
        with self.assertRaises(netlayer.RouteBlocked):
            self.b.open("https://example.invalid/")
        self.assertNotIn("Page.navigate", [call.args[0] for call in cdp.send.call_args_list])
        self.assertIn("Browser.close", [call.args[0] for call in cdp.send.call_args_list])
        proc.terminate.assert_called_once()
        self.assertIsNone(self.b._cdp)

    def test_offline_scope_blocks_new_and_existing_browsers(self):
        self.vpn["active"] = True
        with netlayer.offline("Synthetic offline scope"):
            with self.assertRaises(netlayer.RouteBlocked):
                browser.route_args()
            cdp, proc = self._existing()
            with self.assertRaises(netlayer.RouteBlocked):
                self.b.ensure()
            proc.terminate.assert_called_once()

    def test_live_vpn_allows_existing_browser(self):
        self.vpn["active"] = True
        cdp, proc = self._existing()
        self.assertIs(self.b.ensure(), cdp)
        proc.terminate.assert_not_called()
        self.assertIn("--no-proxy-server", browser.route_args())

    def test_changed_route_closes_existing_process(self):
        self.vpn["active"] = True
        cdp, proc = self._existing()
        self.settings.update(net_route="proxy", net_proxy="http://127.0.0.1:3128")
        with patch.object(netlayer, "probe", return_value=True):
            route = self.b._check_network()
        self.assertTrue(route.routed)
        proc.terminate.assert_called_once()
        self.assertIsNone(self.b._cdp)

    def test_watchdog_stops_background_traffic_after_failed_check(self):
        cdp, proc = self._existing()
        # Run exactly one watch iteration synchronously, with no sleeping thread.
        stop = Mock()
        stop.wait.return_value = False
        stop.is_set.return_value = False
        threads = []
        def thread(**kwargs):
            threads.append(kwargs["target"])
            return SimpleNamespace(start=lambda: None)
        with patch.object(browser.threading, "Event", return_value=stop), \
                patch.object(browser.threading, "Thread", side_effect=thread):
            self.b._start_network_watch()
        threads[0]()
        stop.set.assert_called_once()
        proc.terminate.assert_called_once()
        self.assertIsNone(self.b._watch_stop)

    def test_unknown_browser_is_neither_reused_nor_closed(self):
        # 1.4.7: an unknown browser on the control port is still never reused or closed - ours now starts on a free
        # port instead of refusing to start at all.
        self.vpn["active"] = True
        taken = self.b.port
        with patch.object(self.b, "_alive", side_effect=lambda: self.b.port == taken), \
                patch.object(self.b, "close_leftover", return_value=False) as leftover, \
                patch("ports.pick", return_value=taken + 5), patch("ports.note"), \
                patch.object(browser, "CDP") as connect:
            with self.assertRaisesRegex(AssertionError, "must not launch a real browser"):   # the launch, faked
                self.b.ensure()
            connect.assert_not_called()
            leftover.assert_called_once()                    # only a browser of our own profile may be closed
            # The patch is process-wide, so a GPU poll from another test's background thread can land here too;
            # only a browser launch (it always carries the debugging port) counts.
            launches = [c for c in self.spawn.call_args_list
                        if any(str(a).startswith("--remote-debugging-port") for a in c.args[0])]
            self.assertEqual(len(launches), 1)
            self.assertIn(f"--remote-debugging-port={taken + 5}", launches[0].args[0])
        self.b.port = taken


if __name__ == "__main__":
    unittest.main()
