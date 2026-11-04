"""The network layer (netlayer.py): every outbound client goes through it, the
kill switch blocks instead of going direct, names are resolved by the proxy
(socks5h / HTTP CONNECT) and never looked up here in a routed mode, Require
VPN blocks without a tunnel, offline work can't open a socket, and the
connection log never saves a temporary chat's entries.

The proxies here are real servers on 127.0.0.1 (an HTTP proxy; a SOCKS5 one
when socksio is installed) - the requests really go through them."""
from __future__ import annotations

import http.server
import json
import os
import re
import socket
import socketserver
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import netlayer
import privacy

HERE = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class RecordingHTTPProxy:
    """A forward proxy on 127.0.0.1 that answers every request itself and records
    the target it was asked for (absolute-URI GET, or CONNECT host:port)."""

    def __init__(self):
        self.seen = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append(("GET", self.path))
                body = b"proxied"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_CONNECT(self):
                outer.seen.append(("CONNECT", self.path))
                self.send_response(502)
                self.end_headers()

            def log_message(self, *a):
                pass

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class RecordingSocks5:
    """A SOCKS5 server on 127.0.0.1: records the address type and name of each
    CONNECT (3 = a domain name: resolved by the proxy), then answers HTTP itself."""

    def __init__(self):
        self.seen = []
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                c = self.request
                greeting = c.recv(2)                             # ver, nmethods
                if len(greeting) < 2:
                    return                                       # the kill switch's probe: connect and close
                c.recv(greeting[1])                              # the methods
                c.sendall(b"\x05\x00")                           # no auth
                head = c.recv(4)
                atyp = head[3]
                if atyp == 3:
                    n = c.recv(1)[0]
                    host = c.recv(n).decode()
                elif atyp == 1:
                    host = socket.inet_ntoa(c.recv(4))
                else:
                    host = socket.inet_ntop(socket.AF_INET6, c.recv(16))
                port = struct.unpack(">H", c.recv(2))[0]
                outer.seen.append((atyp, host, port))
                c.sendall(b"\x05\x00\x00\x01" + socket.inet_aton("127.0.0.1") + struct.pack(">H", port))
                c.recv(65536)
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nsocks")

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class LayerTest(unittest.TestCase):
    def setUp(self):
        self.settings = {}
        saved = netlayer._settings_fn
        netlayer.configure(lambda: self.settings)
        self.addCleanup(netlayer.configure, saved)
        netlayer._probe_cache.clear()
        netlayer.clear_blocked()
        netlayer.set_session_direct(False)
        self.addCleanup(netlayer._probe_cache.clear)
        self.addCleanup(netlayer.clear_blocked)
        # These tests cover a Tor already running on this PC; the built-in one (1.4.4) has its own tests
        # (test_tor_service) - none of them may start the real one.
        no_builtin = patch.object(netlayer.tor_service, "available", return_value=False)
        no_builtin.start()
        self.addCleanup(no_builtin.stop)

    def guard(self):
        """Install the socket/DNS guard for this test only."""
        was = netlayer.installed()
        netlayer.install()
        if not was:
            self.addCleanup(netlayer.uninstall)


# ==============================================================================
class EveryClientGoesThroughTheLayerTests(unittest.TestCase):
    """No module opens its own outbound connection. The few that talk only to this
    PC (llama-server, the browser's DevTools) are named here, with why."""
    PATTERNS = re.compile(r"httpx\.(?:Client|AsyncClient)\(|requests\.(?:get|post|put|delete|head|request|Session)\(|"
                          r"urlopen\(|aiohttp\.ClientSession\(|socket\.create_connection\(|http\.client\.HTTPS?Connection\(|"
                          r"\bOpenAI\(")
    LOOPBACK_ONLY = {
        "llm/llama_local.py": "requests to LLAMA_SERVER_BASE - llama-server on 127.0.0.1 (checked below)",
        "llm/llm_compat.py": "OpenAI() only in the _is_local branch - llama-server on this PC",
        "pc/sys_tools.py": "port_check: gated by _not_through_route and the socket guard",
        "web/netlayer.py": "the layer itself",
        "web/tor_service.py": "socket.create_connection only to the built-in Tor's control port on 127.0.0.1",
    }

    def test_no_stray_direct_clients(self):
        offenders = []
        for path in HERE.rglob("*.py"):
            rel = path.relative_to(HERE).as_posix()
            if rel.startswith((".venv/", "_backup", "static/", "data/", "fixtures/", ".checkpoints/", "benchmarks/", "local/", "tests/")) \
                    or path.name == "replay_harness.py":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for m in self.PATTERNS.finditer(text):
                if rel not in self.LOOPBACK_ONLY:
                    line = text[:m.start()].count("\n") + 1
                    offenders.append(f"{rel}:{line}: {m.group(0)}")
        self.assertEqual(offenders, [], "outbound clients outside netlayer.py")

    def test_the_loopback_only_ones_really_are(self):
        import llama_local
        from urllib.parse import urlparse
        self.assertTrue(netlayer.is_local(urlparse(llama_local.LLAMA_SERVER_BASE).hostname))
        bat = (HERE / "run.bat").read_text(encoding="utf-8", errors="replace")
        self.assertIn("--host 127.0.0.1 --port %APP_PORT%", bat)                   # the app
        self.assertIn('set "LLAMA_HOST=127.0.0.1"', bat)                    # llama-server
        self.assertNotIn("0.0.0.0", bat)


class RouteTests(LayerTest):
    def test_settings_are_cleaned(self):
        c = netlayer.clean_settings({"net_route": "tor", "net_proxy": "socks5://10.0.0.2:1080", "net_tor_port": "abc",
                                     "net_components": {"search": "direct", "models": "bogus"}})
        self.assertEqual(c["net_proxy"], "socks5h://10.0.0.2:1080")        # names resolved by the proxy
        self.assertEqual((c["net_route"], c["net_tor_port"]), ("tor", 0))
        self.assertEqual(c["net_components"], {"search": "direct", "models": "app", "other": "app"})
        self.assertEqual(netlayer.clean_settings({"net_route": "vpn", "net_proxy": "ftp://x:1"})["net_route"], "direct")
        self.assertEqual(netlayer.clean_settings({"net_proxy": "127.0.0.1:9050"})["net_proxy"], "socks5h://127.0.0.1:9050")

    def test_components_follow_or_override_the_app_route(self):
        self.settings = {"net_route": "proxy", "net_proxy": "http://127.0.0.1:3128",
                         "net_components": {"models": "direct"}}
        self.assertTrue(netlayer.route_for("search").routed)
        self.assertFalse(netlayer.route_for("models").routed)
        with patch.object(netlayer, "probe", return_value=False):
            self.settings = {"net_route": "tor"}
            self.assertEqual(netlayer.route_for("other").proxy, "socks5h://127.0.0.1:9050")   # not running: 9050

    def test_tor_is_found_on_9150(self):
        self.settings = {"net_route": "tor"}
        with patch.object(netlayer, "probe", side_effect=lambda h, p, **k: p == 9150):
            self.assertEqual(netlayer.route_for("search").proxy, "socks5h://127.0.0.1:9150")


class KillSwitchTests(LayerTest):
    def test_a_dead_route_blocks_instead_of_going_direct(self):
        self.settings = {"net_route": "proxy", "net_proxy": f"http://127.0.0.1:{_free_port()}"}   # nothing listens
        direct = httpx.MockTransport(lambda r: self.fail("went direct"))
        with netlayer.client("search", direct=direct) as c:
            with self.assertRaises(netlayer.RouteBlocked) as cm:
                c.get("http://example.com/page?q=secret")
        self.assertEqual(cm.exception.kind, "kill")
        self.assertIn("kill switch", str(cm.exception))
        self.assertEqual(netlayer.blocked_state()["component"], "search")
        entry = netlayer.log_entries(1)[0]
        self.assertTrue(entry["blocked"])
        self.assertNotIn("secret", json.dumps(entry))                       # never the path or the query

    def test_the_route_dropping_mid_session_blocks_the_next_request(self):
        proxy = RecordingHTTPProxy()
        self.settings = {"net_route": "proxy", "net_proxy": f"http://127.0.0.1:{proxy.port}"}
        with netlayer.client("other") as c:
            self.assertEqual(c.get("http://example.test/a").text, "proxied")
        proxy.stop()
        netlayer._probe_cache.clear()
        with netlayer.client("other") as c:
            with self.assertRaises(netlayer.RouteBlocked):
                c.get("http://example.test/b")
        self.assertEqual([p for _m, p in proxy.seen], ["http://example.test/a"])

    def test_going_direct_is_the_users_explicit_choice(self):
        self.settings = {"net_route": "proxy", "net_proxy": f"http://127.0.0.1:{_free_port()}"}
        direct = httpx.MockTransport(lambda r: httpx.Response(200, text="direct"))
        with netlayer.allow_direct(), netlayer.client("search", direct=direct) as c:
            self.assertEqual(c.get("http://example.com/").text, "direct")      # this one action
        netlayer.set_session_direct(True)
        self.addCleanup(netlayer.set_session_direct, False)
        with netlayer.client("search", direct=direct) as c:
            self.assertEqual(c.get("http://example.com/").text, "direct")      # this session
        self.assertIn("by your choice", netlayer.route_for("search").label)


class ProxyAndDnsTests(LayerTest):
    def test_http_proxy_gets_the_name_and_nothing_is_resolved_here(self):
        proxy = RecordingHTTPProxy()
        self.addCleanup(proxy.stop)
        self.settings = {"net_route": "proxy", "net_proxy": f"http://127.0.0.1:{proxy.port}"}
        self.guard()
        lookups = []
        real = netlayer._orig["getaddrinfo"]
        with patch.dict(netlayer._orig, getaddrinfo=lambda h, *a, **k: lookups.append(h) or real(h, *a, **k)):
            with netlayer.client("search") as c:
                self.assertEqual(c.get("http://leak-check.example.test/x").text, "proxied")
        self.assertEqual(proxy.seen, [("GET", "http://leak-check.example.test/x")])
        self.assertNotIn("leak-check.example.test", lookups)

    @unittest.skipUnless(__import__("importlib").util.find_spec("socksio"), "socksio is not installed")
    def test_socks5h_sends_the_name_to_the_proxy(self):
        socks = RecordingSocks5()
        self.addCleanup(socks.stop)
        self.settings = {"net_route": "proxy", "net_proxy": f"socks5://127.0.0.1:{socks.port}"}
        self.guard()
        with netlayer.client("search") as c:
            self.assertEqual(c.get("http://dns-through-proxy.example.test/").text, "socks")
        self.assertEqual(socks.seen, [(3, "dns-through-proxy.example.test", 80)])     # ATYP 3: a domain name

    def test_the_guard_blocks_direct_sockets_and_local_dns_in_a_routed_mode(self):
        proxy = RecordingHTTPProxy()
        self.addCleanup(proxy.stop)
        self.settings = {"net_route": "proxy", "net_proxy": f"http://127.0.0.1:{proxy.port}"}
        self.guard()
        with self.assertRaises(OSError):
            socket.create_connection(("93.184.216.34", 80), timeout=2)       # a library going its own way
        with self.assertRaises(socket.gaierror):
            socket.getaddrinfo("example.com", 443)                           # DNS would leak around the proxy
        s = socket.create_connection(("127.0.0.1", proxy.port), timeout=2)  # loopback is fine
        s.close()
        self.assertTrue(any(e["blocked"] and e["kind"] == "socket" for e in netlayer.log_entries(20)))


class VpnAndOfflineTests(LayerTest):
    def test_require_vpn_blocks_without_a_tunnel(self):
        self.settings = {"net_require_vpn": True}
        ok = httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))
        with patch.object(netlayer, "vpn_status", return_value={"active": False}):
            with netlayer.client("other", direct=ok) as c, self.assertRaises(netlayer.RouteBlocked) as cm:
                c.get("https://example.com/")
        self.assertEqual(cm.exception.kind, "vpn")
        with patch.object(netlayer, "vpn_status", return_value={"active": True}):
            with netlayer.client("other", direct=ok) as c:
                self.assertEqual(c.get("https://example.com/").text, "ok")

    def test_vpn_detection_reads_the_adapters_and_routes(self):
        wg = {"a": [{"Name": "Ethernet", "InterfaceDescription": "Intel(R) Ethernet", "InterfaceIndex": 4},
                    {"Name": "wg0", "InterfaceDescription": "WireGuard Tunnel", "InterfaceIndex": 17}],
              "r": [{"DestinationPrefix": "0.0.0.0/1", "InterfaceIndex": 17, "RouteMetric": 0},
                    {"DestinationPrefix": "0.0.0.0/0", "InterfaceIndex": 4, "RouteMetric": 25}]}
        v = netlayer.vpn_status(force=True, query=lambda: wg)
        self.assertEqual((v["active"], v["all_traffic"], v["adapters"][0]["name"]), (True, True, "wg0"))
        plain = {"a": [{"Name": "Wi-Fi", "InterfaceDescription": "Intel Wi-Fi 6", "InterfaceIndex": 9}],
                 "r": [{"DestinationPrefix": "0.0.0.0/0", "InterfaceIndex": 9, "RouteMetric": 35}]}
        self.assertFalse(netlayer.vpn_status(force=True, query=lambda: plain)["active"])

    def test_offline_work_cannot_open_a_socket_or_resolve_a_name(self):
        self.guard()
        with netlayer.offline("Local Only test"):
            with self.assertRaises(OSError):
                socket.create_connection(("93.184.216.34", 443), timeout=2)
            with self.assertRaises(socket.gaierror):
                socket.getaddrinfo("example.com", 443)
            with netlayer.client("search", direct=httpx.MockTransport(lambda r: self.fail("sent"))) as c:
                with self.assertRaises(netlayer.RouteBlocked):
                    c.get("https://example.com/")


class LogTests(LayerTest):
    def test_the_saved_log_never_holds_a_temporary_chats_entries(self):
        path = os.path.join(tempfile.mkdtemp(), "netlog.jsonl")
        self.settings = {"net_log_save": True}
        ok = httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))
        with patch.object(netlayer, "LOG_PATH", path):
            with netlayer.client("search", direct=ok) as c:
                c.get("https://normal.example/")
            with privacy.scope("tmp-abcdefabcdef"), netlayer.client("search", direct=ok) as c:
                c.get("https://temporary-canary.example/")
            with open(path, encoding="utf-8") as f:
                saved = f.read()
        self.assertIn("normal.example", saved)
        self.assertNotIn("temporary-canary", saved)
        self.assertTrue(any(e["host"] == "temporary-canary.example" for e in netlayer.log_entries(10)))   # in memory only

    def test_nothing_saved_unless_turned_on(self):
        path = os.path.join(tempfile.mkdtemp(), "netlog.jsonl")
        with patch.object(netlayer, "LOG_PATH", path), \
                netlayer.client("other", direct=httpx.MockTransport(lambda r: httpx.Response(200))) as c:
            c.get("https://example.org/")
        self.assertFalse(os.path.exists(path))

    def test_traffic_is_counted_for_the_indicator(self):
        before = netlayer.traffic()["in"]
        with netlayer.client("other", direct=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 5000))) as c:
            c.get("https://example.org/")
        self.assertGreaterEqual(netlayer.traffic()["in"] - before, 5000)
        stats = netlayer.stats()
        self.assertIn("down", stats["app"])
        self.assertIn("route", stats)


class EnvTests(LayerTest):
    def test_litellm_gets_the_models_route_as_proxy_variables(self):
        proxy = RecordingHTTPProxy()
        self.addCleanup(proxy.stop)
        saved = {k: os.environ.get(k) for k in ("HTTPS_PROXY", "NO_PROXY")}
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v) for k, v in saved.items()])
        self.settings = {"net_route": "proxy", "net_proxy": f"http://127.0.0.1:{proxy.port}"}
        netlayer.apply_env()
        self.assertEqual(os.environ["HTTPS_PROXY"], f"http://127.0.0.1:{proxy.port}")
        self.assertIn("127.0.0.1", os.environ["NO_PROXY"])
        self.settings = {}
        netlayer.apply_env()
        self.assertEqual(os.environ.get("HTTPS_PROXY"), saved["HTTPS_PROXY"])


if __name__ == "__main__":
    unittest.main()
