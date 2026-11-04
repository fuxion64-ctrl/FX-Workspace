"""
netlayer.py
===========
The one way this app reaches the network (Settings > Network). Web search,
page reads, favicons, downloads, cloud models, MCP servers - every outbound
request goes through here, and a guard at the socket level catches anything
that tries to go around it.

Routes (app-wide, with simple per-component overrides)
  direct   no proxy
  system   Windows' proxy settings (Internet Options); direct when none is set
  proxy    a SOCKS5 or HTTP proxy the user gives; SOCKS always as socks5h - the
           proxy resolves names, so DNS doesn't leak around it
  tor      Tor's SOCKS port on this PC: 9050 (Tor Expert Bundle) or 9150 (Tor
           Browser), found automatically unless a port is set
  Components: search (web search, page reads, favicons), models (cloud model
  APIs), other (downloads, update checks, MCP servers, integrations). Each
  follows the app route ("app") or overrides it: direct / proxy / tor.
  Loopback (127.0.0.1, ::1, localhost - llama-server, this app, Roblox Studio's
  plugin, the assistant's browser's DevTools) never leaves the machine: it is
  never routed, blocked or logged.

Kill switch. In a routed mode, when the route is down, a request is blocked
(RouteBlocked) - never sent direct. The user can go direct for one action
(allow_direct()) or for the session (set_session_direct(True)).
Require VPN. With it on and no tunnel adapter up, outbound traffic is blocked.
Offline. offline(reason) blocks every request and socket of that work (Local
Only search: its tools cannot reach the network at all).

The guard (install()). socket.connect and getaddrinfo are wrapped: loopback
passes; a connection made by a request of this layer passes (to the proxy
when routed); anything else - a library opening its own connection - is
blocked in a routed mode, a VPN-required mode or offline work, and logged
otherwise. In a routed mode a local DNS lookup of a public name is refused:
names are resolved by the proxy. (Connections made by asyncio's proactor on
Windows bypass socket.connect; this app's outbound requests are synchronous.)

Connection log. The last LOG_MAX connections in memory - component, host,
route, status, bytes, time, blocked - never the path or the query. Saved to
data/netlog.jsonl only when the user turns that on, and never a temporary
chat's entries (privacy.py).
Traffic. Bytes in/out of this app's requests (the speed indicator), and the
latency of requests already made (no extra pings).
"""
from __future__ import annotations

import collections
import contextlib
import contextvars
import http.cookiejar
import ipaddress
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request
from urllib.parse import urlparse

import httpx

import privacy
import tor_service

COMPONENTS = ("search", "models", "other")
COMPONENT_LABELS = {"search": "Web search & page reads", "models": "Cloud models", "other": "Downloads, MCP & integrations"}
ROUTES = ("direct", "system", "proxy", "tor")
ROUTE_LABELS = {"direct": "Direct", "system": "System proxy", "proxy": "Custom proxy", "tor": "Tor"}
OVERRIDES = ("app", "direct", "proxy", "tor")
TOR_PORTS = (9050, 9150)                  # Tor Expert Bundle, Tor Browser
TOR_WAIT_S = 60                           # how long a request waits for the built-in Tor to connect
TOR_REFUSED = (403, 429, 503)             # answers a site often gives one Tor exit and not another
LOG_MAX = 500
LOG_SAVE_MAX = 5000                       # lines kept in data/netlog.jsonl
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "netlog.jsonl")
GENERIC_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"

DEFAULT_SETTINGS = {
    "net_route": "direct",                # direct | system | proxy | tor
    "net_proxy": "",                      # socks5h://host:port or http://host:port
    "net_tor_port": 0,                    # 0: find 9050 / 9150
    "net_tor_source": "builtin",          # builtin: the app's own Tor (tor_service, 1.4.4) | external: one on this PC
    "net_tor_bridges": "off",             # off | obfs4 | snowflake (the built-in Tor only)
    "net_tor_bridge_lines": "",           # obfs4 bridge lines the user pasted
    "net_components": {c: "app" for c in COMPONENTS},
    "net_require_vpn": False,
    "net_log_save": False,                # connection log on disk (never a temporary chat's)
    "net_indicator": False,               # the speed indicator in the top bar (D48: off unless you want it)
    "net_indicator_scope": "app",         # app | system
}


class RouteBlocked(httpx.ConnectError, OSError):
    """A request or connection this layer refused (kill switch, no VPN, offline).
    An httpx error and an OSError, so every caller's existing error handling
    reports it instead of crashing."""

    def __init__(self, message: str, kind: str = "kill", component: str = "other"):
        Exception.__init__(self, message)
        self._request = None
        self.kind, self.component = kind, component

    def __str__(self):
        return self.args[0] if self.args else "Blocked."


# ==============================================================================
# Settings
# ==============================================================================
_settings_fn = lambda: {}                  # noqa: E731 - api/server configures the real one


def configure(settings_fn):
    global _settings_fn
    _settings_fn = settings_fn


def _proxy_url(value) -> str:
    """A proxy address as the layer uses it: socks5h:// (names resolved by the
    proxy) or http(s)://. "" when it isn't one."""
    v = str(value or "").strip()
    if not v:
        return ""
    if "://" not in v:
        v = "socks5h://" + v
    scheme = v.split("://", 1)[0].lower()
    if scheme in ("socks5", "socks", "socks5h"):
        v = "socks5h://" + v.split("://", 1)[1]
    elif scheme not in ("http", "https"):
        return ""
    p = urlparse(v)
    try:
        port = p.port
    except ValueError:
        return ""
    return v if p.hostname and port else ""


def clean_settings(s: dict) -> dict:
    s = s or {}
    comps = s.get("net_components") if isinstance(s.get("net_components"), dict) else {}
    try:
        tor_port = int(s.get("net_tor_port") or 0)
    except (TypeError, ValueError):
        tor_port = 0
    return {
        "net_route": s.get("net_route") if s.get("net_route") in ROUTES else "direct",
        "net_proxy": _proxy_url(s.get("net_proxy"))[:300],
        "net_tor_port": tor_port if 0 < tor_port < 65536 else 0,
        "net_tor_source": s.get("net_tor_source") if s.get("net_tor_source") in ("builtin", "external") else "builtin",
        "net_tor_bridges": s.get("net_tor_bridges") if s.get("net_tor_bridges") in tor_service.BRIDGE_MODES else "off",
        "net_tor_bridge_lines": "\n".join(tor_service.clean_bridge_lines(s.get("net_tor_bridge_lines"))),
        "net_components": {c: comps.get(c) if comps.get(c) in OVERRIDES else "app" for c in COMPONENTS},
        "net_require_vpn": bool(s.get("net_require_vpn")) if isinstance(s.get("net_require_vpn"), bool) else False,
        "net_log_save": bool(s.get("net_log_save")) if isinstance(s.get("net_log_save"), bool) else False,
        "net_indicator": bool(s.get("net_indicator")) if isinstance(s.get("net_indicator"), bool) else False,
        "net_indicator_scope": s.get("net_indicator_scope") if s.get("net_indicator_scope") in ("app", "system") else "app",
    }


def settings() -> dict:
    try:
        return clean_settings(_settings_fn() or {})
    except Exception:
        return clean_settings({})


# ==============================================================================
# Scopes (context variables: they follow the work's thread, and privacy.wrap
# carries them into thread pools)
# ==============================================================================
_COMPONENT: contextvars.ContextVar = contextvars.ContextVar("net_component", default=None)
_OFFLINE: contextvars.ContextVar = contextvars.ContextVar("net_offline", default=None)
_DIRECT_OK: contextvars.ContextVar = contextvars.ContextVar("net_direct_ok", default=False)
_INSIDE: contextvars.ContextVar = contextvars.ContextVar("net_inside", default=None)
_PRIVATE: contextvars.ContextVar = contextvars.ContextVar("net_private", default=False)


@contextlib.contextmanager
def _var(var, value):
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def component(name: str):
    """Work of one component (litellm's cloud calls, a download): the guard routes and logs it as that."""
    return _var(_COMPONENT, name if name in COMPONENTS else "other")


def offline(reason: str):
    """Nothing of this work reaches the network: requests and sockets are refused."""
    return _var(_OFFLINE, reason)


def allow_direct():
    """The user chose to go direct for this one action (the kill switch asked)."""
    return _var(_DIRECT_OK, True)


def private_requests(on: bool = True):
    """Requests of this work carry no cookies, no Referer and a generic User-Agent (search_engine Private mode)."""
    return _var(_PRIVATE, bool(on))


def is_private() -> bool:
    return bool(_PRIVATE.get())


def offline_reason() -> str | None:
    return _OFFLINE.get()


# ==============================================================================
# Routes
# ==============================================================================
class Route:
    __slots__ = ("kind", "proxy", "label", "reason")

    def __init__(self, kind: str, proxy: str = "", label: str = "", reason: str = ""):
        self.kind, self.proxy, self.label, self.reason = kind, proxy, label or ROUTE_LABELS.get(kind, kind), reason

    @property
    def routed(self) -> bool:
        return self.kind in ("proxy", "tor") or bool(self.proxy)

    def endpoint(self) -> tuple[str, int] | None:
        if not self.proxy:
            return None
        p = urlparse(self.proxy)
        return (p.hostname or ""), int(p.port or 0)

    def public(self) -> dict:
        ep = self.endpoint()
        return {"kind": self.kind, "label": self.label, "routed": self.routed,
                "endpoint": f"{ep[0]}:{ep[1]}" if ep else "", "scheme": self.proxy.split("://")[0] if self.proxy else ""}

    def __eq__(self, other):
        return isinstance(other, Route) and (self.kind, self.proxy) == (other.kind, other.proxy)

    def __hash__(self):
        return hash((self.kind, self.proxy))


DIRECT = Route("direct")
_state = {"session_direct": False, "blocked": None}
_probe_cache: dict = {}                    # (host, port) -> (at, ok, ms)
_lock = threading.Lock()


def set_session_direct(on: bool):
    """Go direct for the rest of this session (the kill switch's "Go direct for this session")."""
    _state["session_direct"] = bool(on)
    apply_env()


def session_direct() -> bool:
    return bool(_state["session_direct"])


def system_proxy() -> str:
    """Windows' proxy (Internet Options), as a proxy address, or "" when none is set."""
    try:
        found = urllib.request.getproxies()
    except Exception:
        return ""
    for key in ("https", "http", "socks"):
        url = found.get(key)
        if url:
            return _proxy_url(url.replace("socks://", "socks5h://"))
    return ""


def _open(host: str, port: int, timeout: float = 0.8) -> tuple[bool, float]:
    """Can a TCP connection be made? (ok, ms) - with the unguarded socket functions."""
    t0 = time.monotonic()
    connect = _orig.get("create_connection") or socket.create_connection
    try:
        with _var(_INSIDE, DIRECT):
            s = connect((host, port), timeout=timeout)
        s.close()
        return True, (time.monotonic() - t0) * 1000
    except OSError:
        return False, (time.monotonic() - t0) * 1000


def probe(host: str, port: int, max_age: float = 3.0) -> bool:
    now = time.monotonic()
    hit = _probe_cache.get((host, port))
    if hit and now - hit[0] < (max_age if hit[1] else 1.0):
        return hit[1]
    ok, ms = _open(host, port)
    _probe_cache[(host, port)] = (now, ok, ms)
    if ok:
        _latency["proxy"] = {"host": f"{host}:{port}", "ms": round(ms), "at": time.time()}
    return ok


def _tor_proxy(port_setting: int) -> str:
    ports = (port_setting,) if port_setting else TOR_PORTS
    for p in ports:
        if probe("127.0.0.1", p):
            return f"socks5h://127.0.0.1:{p}"
    return f"socks5h://127.0.0.1:{ports[0]}"      # not running: the kill switch will say so


_isolation_fn = lambda: ""                 # noqa: E731 - api/server sets it: the current chat's id


def set_isolation(fn):
    """fn() -> a key per chat; through the built-in Tor each key gets its own circuits (IsolateSOCKSAuth)."""
    global _isolation_fn
    _isolation_fn = fn


def _builtin_tor(s: dict) -> bool:
    return s["net_tor_source"] == "builtin" and tor_service.available()


def uses_builtin_tor() -> bool:
    """Does any component's route go through the app's own Tor? (tor_service stops it when none does.)"""
    s = settings()
    if not _builtin_tor(s) or _state["session_direct"]:
        return False
    return any((s["net_route"] if s["net_components"].get(c, "app") == "app" else s["net_components"][c]) == "tor"
               for c in COMPONENTS)


tor_service.set_wanted(uses_builtin_tor)


def start_builtin_tor():
    """Start (or restart, for new bridge settings) the built-in Tor when a route uses it - before the first request."""
    s = settings()
    if uses_builtin_tor():
        tor_service.ensure(s["net_tor_bridges"], s["net_tor_bridge_lines"])


def route_for(comp: str = "other") -> Route:
    """The route a component's traffic takes right now."""
    s = settings()
    over = s["net_components"].get(comp if comp in COMPONENTS else "other", "app")
    kind = s["net_route"] if over == "app" else over
    if kind != "direct" and (_state["session_direct"] or _DIRECT_OK.get()):
        why = "for this session" if _state["session_direct"] else "for this action"
        return Route("direct", "", f"Direct ({why}, by your choice)")
    if kind == "tor" and _builtin_tor(s):
        # The app's own Tor, started on first use. Each chat's requests carry their own SOCKS username, so they get
        # their own circuits (and exit).
        proxy = tor_service.ensure(s["net_tor_bridges"], s["net_tor_bridge_lines"])
        if not proxy:
            return Route("tor", "", "Tor (built in)", reason=tor_service.status().get("error") or "The built-in Tor couldn't start.")
        key = re.sub(r"[^\w-]", "", str(_isolation_fn() or ""))[:40] or "app"
        return Route("tor", proxy.replace("://", f"://fx-{key}:x@", 1), "Tor (built in)")
    if kind == "tor":
        proxy = _tor_proxy(s["net_tor_port"])
        return Route("tor", proxy, f"Tor ({proxy.split('://')[1]})")
    if kind == "proxy":
        proxy = s["net_proxy"]
        if not proxy:
            return Route("proxy", "", "Custom proxy (no address set)", reason="No proxy address is set in Settings > Network.")
        return Route("proxy", proxy, f"Proxy {proxy.split('://')[0]}://{proxy.split('://')[1]}")
    if kind == "system":
        proxy = system_proxy()
        return Route("system", proxy, f"System proxy ({proxy.split('://')[1]})" if proxy else "System (no proxy set - direct)")
    return DIRECT


def tor_use() -> str | None:
    """What goes through Tor now - "all", "search" or None - for the main screen's Tor tag. Reads the settings
    only: unlike route_for it never starts the built-in Tor."""
    if _state["session_direct"]:
        return None
    s = settings()
    comps = s["net_components"]
    kind = lambda comp: s["net_route"] if comps.get(comp, "app") == "app" else comps.get(comp)
    if all(kind(c) == "tor" for c in COMPONENTS):
        return "all"
    return "search" if kind("search") == "tor" else None


def is_local(host: str) -> bool:
    h = str(host or "").strip("[]").lower()
    if h in ("localhost", "") or h.endswith(".localhost"):
        return h != ""
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(str(host).strip("[]"))
        return True
    except ValueError:
        return False


# ==============================================================================
# The gate: kill switch, VPN, offline
# ==============================================================================
def _block(reason: str, kind: str, comp: str, host: str = ""):
    if kind in ("kill", "vpn"):
        _state["blocked"] = {"reason": reason, "kind": kind, "component": comp, "at": time.time()}
    _log({"component": comp, "host": host, "route": route_for(comp).label, "blocked": reason, "kind": "blocked"})
    raise RouteBlocked(reason, kind=kind, component=comp)


def check(route: Route, comp: str, host: str = ""):
    """Raise RouteBlocked when this request must not go out."""
    reason = _OFFLINE.get()
    if reason:
        _block(reason, "offline", comp, host)
    if str(host).lower().endswith(".onion") and route.kind != "tor":
        _block(f"{host} is a Tor onion address: it opens only through Tor, and looking it up any other way would "
               "leak it.", "offline", comp, host)
    s = settings()
    if s["net_require_vpn"] and not vpn_status().get("active"):
        _block("Require VPN is on and no VPN is connected, so nothing was sent. Connect your VPN, or turn "
               "Require VPN off in Settings > Network.", "vpn", comp, host)
    if route.routed:
        if not route.proxy:
            _block(route.reason or "The route has no address.", "kill", comp, host)
        if route.label == "Tor (built in)":
            # While the app's own Tor connects (about 10 s, longer the first time), a request waits for it instead of
            # failing - and never goes direct.
            if not tor_service.wait_ready(TOR_WAIT_S):
                st = tor_service.status()
                why = (f"Tor couldn't connect: {st['error']}" if st["state"] == "failed" else
                       f"Tor is still connecting ({st['progress']}%: {st['summary'] or 'starting'})")
                _block(f"{why}, so the request was blocked instead of going direct (kill switch). Wait a moment and try "
                       "again, use bridges if your network blocks Tor, or go direct for this action.", "kill", comp, host)
            return
        ep = route.endpoint()
        if not probe(*ep):
            _block(f"{route.label} isn't reachable, so the request was blocked instead of going direct (kill switch). "
                   "Start it, or go direct for this action or this session.", "kill", comp, host)


def blocked_state() -> dict | None:
    return dict(_state["blocked"]) if _state["blocked"] else None


def clear_blocked():
    _state["blocked"] = None


# ==============================================================================
# Connection log and traffic
# ==============================================================================
_log_lock = threading.Lock()
_entries: collections.deque = collections.deque(maxlen=LOG_MAX)
_seq = [0]
_traffic = {"in": 0, "out": 0}
_latency: dict = {}                        # "search" / "proxy" -> {host, ms, at}


def _log(entry: dict) -> dict:
    with _log_lock:
        _seq[0] += 1
        entry = {"id": _seq[0], "t": time.time(), "status": None, "bytes_in": 0, "bytes_out": 0, "ms": None,
                 "blocked": None, **entry}
        if privacy.active():
            entry["temp"] = True             # never saved (privacy.py)
        _entries.append(entry)
    if entry.get("blocked") or entry.get("kind") == "socket":
        _save(entry)
    return entry


def _finish(entry: dict):
    _save(entry)


def _save(entry: dict):
    if entry.get("temp") or not settings()["net_log_save"]:
        return
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: v for k, v in entry.items() if k != "temp"}) + "\n")
        if os.path.getsize(LOG_PATH) > LOG_SAVE_MAX * 400:
            with _log_lock:
                with open(LOG_PATH, encoding="utf-8") as f:
                    lines = f.readlines()[-LOG_SAVE_MAX:]
                with open(LOG_PATH, "w", encoding="utf-8") as f:
                    f.writelines(lines)
    except OSError:
        pass


def log_entries(limit: int = 200) -> list:
    with _log_lock:
        return [dict(e) for e in list(_entries)[-limit:]][::-1]


def clear_log(saved: bool = True) -> int:
    with _log_lock:
        n = len(_entries)
        _entries.clear()
    if saved:
        try:
            os.remove(LOG_PATH)
        except OSError:
            pass
    return n


def traffic() -> dict:
    return dict(_traffic)


def latency() -> dict:
    return {k: dict(v) for k, v in _latency.items()}


class _Counted(httpx.SyncByteStream):
    """A response body that counts its bytes into the entry and the traffic totals."""

    def __init__(self, stream, entry: dict, started: float):
        self._stream, self._entry, self._started = stream, entry, started

    def __iter__(self):
        for chunk in self._stream:
            n = len(chunk)
            self._entry["bytes_in"] += n
            _traffic["in"] += n
            yield chunk

    def close(self):
        try:
            if hasattr(self._stream, "close"):
                self._stream.close()
        finally:
            if self._entry.get("ms") is None:
                self._entry["ms"] = round((time.monotonic() - self._started) * 1000)
                _finish(self._entry)


# ==============================================================================
# Clients
# ==============================================================================
class _NoCookies(http.cookiejar.CookieJar):
    """A cookie jar that never keeps one (Private search)."""

    def set_cookie(self, cookie):
        return

    def extract_cookies(self, response, request):
        return


class GuardedTransport(httpx.BaseTransport):
    """Every request: the gate (offline, VPN, kill switch), the current route
    (a proxy transport per proxy address), the log and the byte counts.
    direct: the transport to use when the route is direct (web_tools' SSRF-safe one)."""

    def __init__(self, comp: str = "other", direct=None, **transport_kwargs):
        self.comp = comp if comp in COMPONENTS else "other"
        self._direct = direct
        self._kw = transport_kwargs
        self._pools: dict = {}

    def _inner(self, route: Route):
        key = route.proxy or ""
        t = self._pools.get(key)
        if t is None:
            if key:
                try:
                    t = httpx.HTTPTransport(proxy=key, trust_env=False, **self._kw)
                except ImportError:
                    raise RouteBlocked("SOCKS proxies need the 'socksio' package, which isn't installed - run run.bat once "
                                       "to install it. Nothing was sent.", kind="kill", component=self.comp)
            else:
                t = self._direct() if callable(self._direct) else (self._direct or httpx.HTTPTransport(trust_env=False, **self._kw))
            self._pools[key] = t
        return t

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if is_local(host):                               # never leaves the machine
            return self._inner(DIRECT).handle_request(request)
        route = route_for(self.comp)
        check(route, self.comp, host)
        entry = _log({"component": self.comp, "host": host, "route": route.label, "method": request.method, "kind": "http"})
        body = request.headers.get("content-length")
        if body and body.isdigit():
            entry["bytes_out"] = int(body)
            _traffic["out"] += int(body)
        started = time.monotonic()
        try:
            with _var(_INSIDE, route):
                resp = self._inner(route).handle_request(request)
                if route.label == "Tor (built in)" and request.method in ("GET", "HEAD") and resp.status_code in TOR_REFUSED:
                    # Many sites refuse some Tor exits, not all: once, the same request on a new circuit (another
                    # SOCKS username - another exit) before giving up (1.4.4).
                    resp.close()
                    again = Route("tor", route.proxy.replace(":x@", "-r:x@", 1), route.label)
                    resp = self._inner(again).handle_request(request)
                    entry["retried"] = "new Tor circuit"
        except RouteBlocked:
            raise
        except Exception as e:
            entry.update(status="error", ms=round((time.monotonic() - started) * 1000), error=type(e).__name__)
            _finish(entry)
            raise
        ttfb = round((time.monotonic() - started) * 1000)
        entry["status"] = resp.status_code
        if self.comp == "search":
            _latency["search"] = {"host": host, "ms": ttfb, "at": time.time()}
        if route.routed and _state["blocked"] and _state["blocked"].get("component") == self.comp:
            clear_blocked()                              # the route works again
        if hasattr(resp, "_content"):                    # a body already read (an in-memory transport)
            entry["bytes_in"] = len(resp._content)
            _traffic["in"] += len(resp._content)
            entry["ms"] = ttfb
            _finish(entry)
            return resp
        resp.stream = _Counted(resp.stream, entry, started)
        return resp

    def close(self):
        for t in list(self._pools.values()):             # a copy: Stop closes it from another thread mid-request
            try:
                t.close()
            except Exception:
                pass
        self._pools.clear()


def client(comp: str = "other", *, direct=None, timeout=20.0, private: bool | None = None, headers: dict | None = None,
           follow_redirects: bool = False, **transport_kwargs) -> httpx.Client:
    """An httpx client for one component: routed, gated, logged. private (default:
    the work's private_requests scope): no cookies kept, generic User-Agent."""
    private = is_private() if private is None else private
    kwargs = {"timeout": timeout, "follow_redirects": follow_redirects, "trust_env": False,
              "transport": GuardedTransport(comp, direct=direct, **transport_kwargs)}
    h = dict(headers or {})
    if private:
        kwargs["cookies"] = _NoCookies()          # the jar itself: httpx.Cookies(Cookies) would copy it into a normal one
        h = {k: v for k, v in h.items() if k.lower() not in ("referer", "origin", "cookie")}
        h["User-Agent"] = GENERIC_UA
    if h:
        kwargs["headers"] = h
    return httpx.Client(**kwargs)


def ws_session(comp: str = "other", **kwargs):
    """An aiohttp session for a long-lived websocket (a chat channel's listener, channels.py). Direct mode only:
    aiohttp can't follow a proxy or Tor route, and a channel must never go around one - in a routed mode, offline
    work or Require VPN without a VPN it's refused (RouteBlocked), like any other connection the kill switch stops."""
    import aiohttp
    reason = _OFFLINE.get()
    if reason:
        _block(f"Blocked: {reason}", "offline", comp)
    route = route_for(comp)
    if route.routed:
        _block(f"Channels connect directly, and the network route is {route.label} - turn it to Direct in Settings > "
               "Network to use them.", "kill", comp)
    if settings()["net_require_vpn"] and not vpn_status().get("active"):
        _block("Require VPN is on and no VPN is connected.", "kill", comp)
    _log({"component": comp, "host": "websocket", "route": route.label, "kind": "socket", "status": "channel"})
    return aiohttp.ClientSession(trust_env=False, **kwargs)


def local_client(timeout=10.0) -> httpx.Client:
    """For this PC's own services (llama-server, the browser's DevTools): refuses any other host."""
    return httpx.Client(timeout=timeout, trust_env=False, transport=_LocalOnly())


class _LocalOnly(httpx.HTTPTransport):
    def handle_request(self, request):
        if not is_local(request.url.host):
            raise RouteBlocked(f"{request.url.host} isn't on this PC; this client only talks to local services.",
                               kind="offline")
        return super().handle_request(request)


def private_headers(headers: dict) -> dict:
    """headers as Private search sends them: generic User-Agent, no Referer, Origin or cookies."""
    out = {k: v for k, v in (headers or {}).items() if k.lower() not in ("referer", "origin", "cookie", "user-agent")}
    out["User-Agent"] = GENERIC_UA
    out["Accept-Language"] = "en-US,en;q=0.5"
    return out


# ==============================================================================
# Third-party libraries (litellm's cloud calls) get the route as proxy variables
# ==============================================================================
_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_env_saved: dict = {}


def apply_env():
    """Set (or clear) the proxy environment variables for libraries that make
    their own connections (litellm -> httpx trust_env) to the "models" route.
    Local addresses stay direct (NO_PROXY). The guard still blocks anything
    that tries to go around the route."""
    if not _env_saved:
        _env_saved.update({k: os.environ.get(k) for k in _ENV_KEYS + ("NO_PROXY", "no_proxy")})
    # litellm downloads its model price list from raw.githubusercontent.com when it is first
    # imported (usage.py prices, the first cloud call) - a request no one asked for, mid-chat.
    # Its bundled copy is used instead; usage prices may be a little older.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    route = route_for("models")
    if route.routed and route.proxy:
        for k in _ENV_KEYS:
            os.environ[k] = route.proxy
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    else:
        for k, v in _env_saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


MODEL_HOSTS = {"gemini": "generativelanguage.googleapis.com", "groq": "api.groq.com", "openai": "api.openai.com",
               "anthropic": "api.anthropic.com", "mistral": "api.mistral.ai", "openrouter": "openrouter.ai"}


@contextlib.contextmanager
def model_call(model: str):
    """A cloud model call made by litellm: gated, logged (by the provider's host)
    and attributed to "models" for the guard."""
    provider = str(model or "").split("/", 1)[0].lower()
    host = MODEL_HOSTS.get(provider, provider or "cloud model")
    route = route_for("models")
    apply_env()
    check(route, "models", host)
    entry = _log({"component": "models", "host": host, "route": route.label, "method": "POST", "kind": "http"})
    started = time.monotonic()
    try:
        with component("models"), _var(_INSIDE, route):
            yield
        entry["status"] = "ok"
    except Exception as e:
        entry["status"] = "error"
        entry["error"] = type(e).__name__
        raise
    finally:
        entry["ms"] = round((time.monotonic() - started) * 1000)
        _finish(entry)


# ==============================================================================
# The guard: socket.connect and getaddrinfo
# ==============================================================================
_orig: dict = {}
_names: dict = {}                          # ip -> the name it was resolved from (for the log)


def _endpoint_ips(route: Route) -> set:
    ep = route.endpoint()
    if not ep:
        return set()
    host = ep[0]
    if _is_ip(host):
        return {host.strip("[]")}
    try:
        return {i[4][0] for i in _orig["getaddrinfo"](host, ep[1])}
    except (OSError, KeyError):
        return set()


def _decide(ip: str, port: int) -> str | None:
    """Why a connection to ip:port must not be made - or None."""
    if is_local(ip):
        return None
    reason = _OFFLINE.get()
    if reason:
        return reason
    inside = _INSIDE.get()
    if inside is not None:
        if inside.routed and ip not in _endpoint_ips(inside):
            return f"A connection to {ip} would bypass {inside.label} - blocked (kill switch)."
        return None
    comp = _COMPONENT.get() or "other"
    route = route_for(comp)
    s = settings()
    if s["net_require_vpn"] and not vpn_status().get("active"):
        return "Require VPN is on and no VPN is connected."
    if route.routed and ip not in _endpoint_ips(route):
        return f"A direct connection to {_names.get(ip, ip)} outside {route.label} was blocked (kill switch)."
    return None


def _guarded_connect(self, address, _fn="connect"):
    if self.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
        ip, port = str(address[0]), int(address[1])
        why = _decide(ip, port)
        comp = _COMPONENT.get() or "other"
        if why:
            _log({"component": comp, "host": _names.get(ip, ip), "route": route_for(comp).label, "blocked": why,
                  "kind": "socket", "port": port})
            raise RouteBlocked(why, kind="offline" if _OFFLINE.get() else "kill", component=comp)
        if _INSIDE.get() is None and not is_local(ip):
            # A connection this layer didn't make (a library's own): allowed in this mode, and logged.
            _log({"component": comp, "host": _names.get(ip, ip), "route": route_for(comp).label, "kind": "socket",
                  "port": port, "status": "unattributed"})
    return _orig[_fn](self, address)


def _guarded_connect_ex(self, address):
    try:
        return _guarded_connect(self, address, "connect_ex")
    except RouteBlocked:
        return 10013 if os.name == "nt" else 13         # permission denied


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    h = host.decode() if isinstance(host, bytes) else host
    if h is None or _is_ip(h) or is_local(h):
        return _orig["getaddrinfo"](host, port, *args, **kwargs)
    reason = _OFFLINE.get()
    if reason:
        raise socket.gaierror(socket.EAI_NONAME, f"Blocked: {reason}")
    route = _INSIDE.get() or route_for(_COMPONENT.get() or "other")
    ep = route.endpoint()
    if route.routed and (not ep or h.lower() != ep[0].lower()):
        raise socket.gaierror(socket.EAI_NONAME, f"Blocked: resolving {h} here would leak the lookup around "
                                                 f"{route.label} (names are resolved by the proxy in this mode).")
    res = _orig["getaddrinfo"](host, port, *args, **kwargs)
    if len(_names) > 4000:
        _names.clear()
    for info in res[:8]:
        _names[info[4][0]] = h
    return res


def install():
    """Wrap socket.connect/connect_ex, getaddrinfo and create_connection (idempotent)."""
    with _lock:
        if _orig:
            return
        _orig.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex,
                     getaddrinfo=socket.getaddrinfo, create_connection=socket.create_connection)
        socket.socket.connect = _guarded_connect
        socket.socket.connect_ex = _guarded_connect_ex
        socket.getaddrinfo = _guarded_getaddrinfo


def uninstall():
    with _lock:
        if not _orig:
            return
        socket.socket.connect = _orig["connect"]
        socket.socket.connect_ex = _orig["connect_ex"]
        socket.getaddrinfo = _orig["getaddrinfo"]
        _orig.clear()


def installed() -> bool:
    return bool(_orig)


# ==============================================================================
# VPN detection (Windows)
# ==============================================================================
_VPN_RE = re.compile(r"wireguard|wintun|openvpn|tap-windows|\btap\b|\btun\b|nordlynx|nordvpn|proton ?vpn|mullvad|"
                     r"expressvpn|surfshark|windscribe|private internet access|\bpia\b|anyconnect|globalprotect|pangp|"
                     r"fortinet|forticlient|juniper|pulse secure|zscaler|cloudflare warp|\bwarp\b|tailscale|zerotier|"
                     r"softether|ivpn|airvpn|hotspot shield|tunnelbear|\bvpn\b|ppp adapter|wan miniport \((?:ikev2|l2tp|pptp|sstp)\)",
                     re.I)
_vpn = {"at": 0.0, "value": {"active": False, "adapters": [], "checked": False}}


def _query_adapters() -> dict:
    """Up adapters and the default routes (0.0.0.0/0 and the /1 halves VPNs add), from PowerShell."""
    script = ("$a = @(Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object Status -eq 'Up' | "
              "Select-Object Name, InterfaceDescription, InterfaceIndex); "
              "$r = @(Get-NetRoute -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { "
              "$_.DestinationPrefix -in '0.0.0.0/0','0.0.0.0/1','128.0.0.0/1' } | "
              "Select-Object DestinationPrefix, InterfaceIndex, RouteMetric); "
              "@{a=$a; r=$r} | ConvertTo-Json -Depth 3 -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                         text=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    return json.loads(out or "{}")


def vpn_status(force: bool = False, query=None) -> dict:
    """{"active", "adapters": [{name, description}], "all_traffic": bool, "checked"} - cached 15 s.
    active: a VPN/tunnel adapter is up. all_traffic: the default route goes through it."""
    now = time.monotonic()
    if not force and _vpn["value"]["checked"] and now - _vpn["at"] < 15:
        return dict(_vpn["value"])
    if os.name != "nt" and query is None:
        value = {"active": False, "adapters": [], "all_traffic": False, "checked": False,
                 "note": "VPN detection works on Windows only."}
    else:
        try:
            data = (query or _query_adapters)()
        except Exception as e:
            data, err = {}, str(e)[:120]
        else:
            err = ""
        adapters = data.get("a") or []
        adapters = adapters if isinstance(adapters, list) else [adapters]
        routes = data.get("r") or []
        routes = routes if isinstance(routes, list) else [routes]
        tunnels = [a for a in adapters if isinstance(a, dict)
                   and _VPN_RE.search(f"{a.get('Name', '')} {a.get('InterfaceDescription', '')}")]
        idx = {t.get("InterfaceIndex") for t in tunnels}
        best = sorted((r for r in routes if isinstance(r, dict)), key=lambda r: (r.get("DestinationPrefix") == "0.0.0.0/0",
                                                                                 r.get("RouteMetric") or 0))
        all_traffic = any(r.get("InterfaceIndex") in idx for r in routes if isinstance(r, dict)
                          and r.get("DestinationPrefix") in ("0.0.0.0/1", "128.0.0.0/1")) or \
            bool(best and best[0].get("InterfaceIndex") in idx)
        value = {"active": bool(tunnels), "all_traffic": bool(tunnels) and all_traffic, "checked": not err,
                 "adapters": [{"name": t.get("Name"), "description": t.get("InterfaceDescription")} for t in tunnels]}
        if err:
            value["note"] = f"Couldn't read the network adapters: {err}"
    _vpn.update(at=now, value=value)
    return dict(value)


# ==============================================================================
# On demand: public IP, leak test, speed test, local services
# ==============================================================================
_public: dict = {}


def public_ip(force: bool = False, comp: str = "other") -> dict:
    """This app's public address and approximate country, fetched through the
    active route - only on demand or when the route changed. comp: another part's
    route (Search through Tor's test asks through "search"; not kept as the app's)."""
    route = route_for(comp)
    if comp == "other" and not force and _public.get("route") == route.label and time.time() - _public.get("at", 0) < 600:
        return dict(_public)
    out = {"route": route.label, "at": time.time()}
    try:
        with client(comp, timeout=15, private=True) as c:
            data = c.get("https://ipinfo.io/json").json()
        out.update(ip=data.get("ip"), country=data.get("country"), city=data.get("city"), org=data.get("org"),
                   service="ipinfo.io")
    except Exception as e:
        out["error"] = str(e)[:200]
    if route.kind == "tor":
        # On its own: ipinfo.io often answers Tor exits with a page instead of its JSON (1.4.4).
        try:
            with client(comp, timeout=20, private=True) as c:
                tor = c.get("https://check.torproject.org/api/ip").json()
            out["is_tor"] = bool(tor.get("IsTor"))
            if not out.get("ip") and tor.get("IP"):
                out.update(ip=tor["IP"], service="check.torproject.org")
                out.pop("error", None)
        except Exception as e:
            out.setdefault("error", str(e)[:200])
    if comp != "other":
        return out
    _public.clear()
    _public.update(out)
    return dict(out)


def _dns_leak(route: Route) -> dict:
    """bash.ws: names only its server can answer are looked up through the route;
    it then lists the DNS resolvers that asked. Through a proxy or Tor those are
    the exit's resolvers, not your ISP's."""
    out = {"name": "DNS", "service": "bash.ws"}
    try:
        with client("other", timeout=20, private=True) as c:
            test_id = c.get("https://bash.ws/id").text.strip()
            if not re.fullmatch(r"[a-z0-9]{3,40}", test_id):        # digits once, letters and digits now
                raise ValueError("bash.ws didn't give a test id")
            for i in range(1, 4):
                try:
                    c.get(f"http://{i}.{test_id}.bash.ws", timeout=5)
                except Exception:
                    pass                         # the lookup is the point, not the page
            rows = c.get(f"https://bash.ws/dnsleak/test/{test_id}?json").json()
        dns = [r for r in rows if isinstance(r, dict) and r.get("type") == "dns"]
        ip = next((r for r in rows if isinstance(r, dict) and r.get("type") == "ip"), {})
        out.update(resolvers=[{"ip": r.get("ip"), "country": r.get("country_name"), "asn": r.get("asn")} for r in dns],
                   seen_ip=ip.get("ip"), ok=True,
                   detail=(f"{len(dns)} resolver{'s' if len(dns) != 1 else ''} looked up the test names: "
                           + "; ".join(f"{r.get('ip')} ({r.get('country_name') or '?'}, {r.get('asn') or '?'})" for r in dns[:4])))
    except Exception as e:
        out.update(ok=None, detail=f"The DNS test service couldn't be reached: {str(e)[:160]}")
    if route.routed:
        try:
            _guarded_getaddrinfo("dns-leak-check.example.com", 443) if installed() else None
            out["local_lookup"] = "allowed" if installed() else "guard not installed"
        except socket.gaierror:
            out["local_lookup"] = "blocked"          # names are resolved by the proxy only
    return out


def local_services() -> list:
    """What this app and llama-server listen on: 127.0.0.1 only is right; 0.0.0.0 / :: is reachable from the network."""
    try:
        import psutil
        import system_metrics
        pids = {os.getpid()} | set(system_metrics._llama_pids())
        rows = []
        for c in psutil.net_connections(kind="inet"):
            if c.status != psutil.CONN_LISTEN or c.pid not in pids or not c.laddr:
                continue
            try:
                name = psutil.Process(c.pid).name()
            except Exception:
                name = str(c.pid)
            ip = c.laddr.ip
            rows.append({"process": name, "address": f"{ip}:{c.laddr.port}",
                         "exposed": ip in ("0.0.0.0", "::", "")})
        return sorted(rows, key=lambda r: r["address"])
    except Exception as e:
        return [{"process": "?", "address": "", "exposed": None, "error": str(e)[:160]}]


def leak_test(webrtc: dict | None = None) -> dict:
    """The Leak test button: public IP through the route, DNS, WebRTC (the page's
    own check, passed in) and what local services listen on."""
    route = route_for("other")
    checks = []
    ip = public_ip(force=True)
    if ip.get("error"):
        checks.append({"name": "Public IP", "ok": None, "detail": f"Couldn't check through {route.label}: {ip['error']}"})
    else:
        where = ", ".join(x for x in (ip.get("city"), ip.get("country")) if x)
        tor = "" if route.kind != "tor" else (" · Tor confirmed by check.torproject.org" if ip.get("is_tor")
                                               else " · NOT seen as Tor by check.torproject.org")
        checks.append({"name": "Public IP", "ok": (ip.get("is_tor") is True) if route.kind == "tor" else True,
                       "detail": f"Sites see {ip.get('ip')} ({where or 'unknown place'}; {ip.get('org') or '?'}) through "
                                 f"{route.label}{tor}."})
    dns = _dns_leak(route)
    if route.routed and dns.get("local_lookup") == "blocked":
        dns["detail"] = (dns.get("detail") or "") + " Local lookups of web names are blocked in this mode."
    checks.append({"name": "DNS", "ok": dns.get("ok"), "detail": dns.get("detail"), "resolvers": dns.get("resolvers")})
    if webrtc is not None:
        leaked = [c for c in webrtc.get("candidates") or [] if c]
        checks.append({"name": "WebRTC", "ok": not webrtc.get("available") or not leaked,
                       "detail": "WebRTC is turned off in this window." if not webrtc.get("available")
                       else f"WebRTC could reveal: {', '.join(leaked[:4])}" if leaked else "WebRTC is available but revealed no address."})
    svc = local_services()
    exposed = [s for s in svc if s.get("exposed")]
    checks.append({"name": "Local services", "ok": not exposed if svc and not svc[0].get("error") else None,
                   "detail": ("Reachable from your network: " + ", ".join(f"{s['process']} on {s['address']}" for s in exposed))
                   if exposed else "Only on 127.0.0.1: " + ", ".join(f"{s['process']} {s['address']}" for s in svc[:6])
                   if svc and not svc[0].get("error") else "Couldn't list them."})
    return {"route": route.public(), "checks": checks, "at": time.time()}


SPEED_URL = "https://speed.cloudflare.com/__down?bytes={n}"


def speed_test(nbytes: int = 10_000_000) -> dict:
    """Only when the user clicks: download nbytes from speed.cloudflare.com through the active route."""
    route = route_for("other")
    started = time.monotonic()
    got = 0
    try:
        with client("other", timeout=60, private=True) as c, c.stream("GET", SPEED_URL.format(n=int(nbytes))) as r:
            for chunk in r.iter_bytes():
                got += len(chunk)
        secs = max(0.001, time.monotonic() - started)
        return {"ok": True, "mbps": round(got * 8 / secs / 1e6, 1), "bytes": got, "seconds": round(secs, 2),
                "route": route.label, "host": "speed.cloudflare.com"}
    except Exception as e:
        return {"ok": False, "message": str(e)[:200], "route": route.label, "host": "speed.cloudflare.com"}


# ==============================================================================
# Speed indicator samples
# ==============================================================================
_samples: collections.deque = collections.deque(maxlen=12)


def stats() -> dict:
    """Rates for the indicator: this app's traffic and the whole system's (psutil),
    from the samples of recent calls (the page polls about once a second while visible)."""
    now = time.monotonic()
    try:
        import psutil
        io = psutil.net_io_counters()
        sys_in, sys_out = io.bytes_recv, io.bytes_sent
    except Exception:
        sys_in = sys_out = None
    _samples.append((now, _traffic["in"], _traffic["out"], sys_in, sys_out))
    base = next((s for s in _samples if now - s[0] <= 3.0 and now - s[0] >= 0.5), None)
    rate = lambda i: None if base is None or _samples[-1][i] is None or base[i] is None else \
        max(0, round((_samples[-1][i] - base[i]) / (now - base[0])))                              # noqa: E731
    route = route_for("search")
    app_route = route_for("other")
    return {"app": {"down": rate(1), "up": rate(2)}, "system": {"down": rate(3), "up": rate(4)},
            "route": app_route.public(), "search_route": route.public(), "blocked": blocked_state(),
            "latency": latency(), "session_direct": session_direct(), "vpn": dict(_vpn["value"])}
