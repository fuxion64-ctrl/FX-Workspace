"""
browser.py
==========
Browser control for the assistant: read the page, navigate, click and fill
forms - in the assistant's OWN browser window, never the user's.

How: Microsoft Edge (or Chrome) is started with its own profile in
data/browser-profile and a DevTools port on 127.0.0.1 (9333), and driven
over the Chrome DevTools Protocol with the `websockets` package the app
already ships with. The separate profile means the assistant's browser has
none of the user's saved passwords, cookies or payment methods; the user
signs in there themselves when a site needs it.

Safety (the rules are enforced here and in permissions.py, not by a prompt):
  * browser_read returns the page as untrusted content (agent/loop.py frames
    every tool result); text on a page never directs the assistant.
  * browser_fill always needs the user's approval, and never types into a
    password, payment or ID field, or a value that looks like a card or ID
    number - it asks the user to do that step themselves.
  * browser_click asks by default (input tier); a click on a control that
    submits, buys, sends, deletes or publishes is flagged irreversible, and
    then it asks every time - no override or mode can make it automatic.
  * Every click and fill is "Not undoable".

Elements are numbered by browser_read ([7] Button "Place order"); click and
fill name that number. The numbers belong to the page as it was read: right
before a click or fill the live control is read again, and the call is refused
when it is no longer the control that was read (and approved) - its label,
kind or form changed, or something now covers it.

The assistant's browser never opens or drives this app's own pages (or the
local model server): it could otherwise click its own Settings or cards.

Built in (D50, Settings > Web search "browser_builtin", on by default): the browser runs hidden (headless) - no
Edge window - and the app's Browser panel shows its page (Viewer: a picture of the page, refreshed while it is
open) and passes YOUR clicks, scrolling and typing through, so you can sign in or finish a step yourself. The
assistant's own clicks and fills still go through its tools, their approvals and refusals; the panel is you.
Headless Chromium names itself "HeadlessChrome", which bot checks refuse - each page session sends the
browser's normal name instead.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import threading
import time

import httpx
from urllib.parse import urlparse

import netlayer
import privacy

PORT =int(os.environ.get("AGENTIC_BROWSER_PORT") or 9333)
READ_WINDOW = 8_000
MAX_ELEMENTS = 120
PROFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "browser-profile")
APP_PORT = int(os.environ.get("AGENTIC_PORT") or 8000)


def _protected_ports() -> set:
    ports = {APP_PORT, PORT}
    try:
        import llama_local
        ports.add(urlparse(llama_local.LLAMA_SERVER_BASE).port or 8090)
    except Exception:
        ports.add(8090)
    return ports


def is_app_url(url: str) -> bool:
    """This app's own UI/API, its DevTools port or the local model server."""
    try:
        u = urlparse(url or "")
        port = u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        return False
    host = (u.hostname or "").lower().strip("[]")
    if host == "localhost" or host.endswith(".localhost"):
        local = True
    else:
        try:
            ip = ipaddress.ip_address(host)
            local = ip.is_loopback or ip.is_unspecified
        except ValueError:
            local = False
    return local and port in _protected_ports()


APP_URL_REFUSAL = ("Refused: the assistant's browser doesn't open or use this app's own pages or the local model "
                   "server.")

# ---- what must never be typed, and what can't be taken back ---------------------------
# Names and labels mix spaces, dashes, underscores and camelCase ("card-number",
# "national_id", "securityCode"): the gap between words is any of them.
_G = r"[\s_-]*"
_SENSITIVE_FIELD = re.compile(
    rf"pass{_G}(word|wd|code)|\bpin\b|one{_G}time|\botp\b|2fa|verification{_G}code|security{_G}code|"
    rf"card|\bcvv|\bcvc|\bcsc\b|expir|\bexp(iry)?\b|iban|swift|routing|account{_G}(number|no)|sort{_G}code|"
    rf"\bssn\b|social{_G}security|passport|national{_G}(id|insurance)|driver'?s?{_G}licen[cs]e|tax{_G}(id|number)|"
    rf"\btin\b|\bnin\b|personal{_G}(id|number)|\bid{_G}number|identity{_G}(number|card)", re.I)
_SENSITIVE_AUTOCOMPLETE = re.compile(r"^(cc-|current-password|new-password|one-time-code)")
_IRREVERSIBLE = re.compile(
    r"\b(buy|purchase|pay|checkout|check out|place (the )?order|order now|confirm( (order|payment|purchase))?|"
    r"submit|send|post|publish|delete|remove|erase|transfer|donate|subscribe|unsubscribe|sign up|register|"
    r"book( now)?|reserve|cancel (my )?(subscription|order|account|membership)|close (my )?account|deactivate|"
    r"accept|agree)\b", re.I)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def looks_sensitive_value(value: str) -> str | None:
    """A reason if the text looks like a card number or an ID number."""
    digits = re.sub(r"[\s-]", "", str(value or ""))
    if digits.isdigit() and 13 <= len(digits) <= 19 and _luhn(digits):
        return "it looks like a payment card number"
    if re.fullmatch(r"\d{3}-\d{2}-\d{4}", str(value or "").strip()):
        return "it looks like a social security number"
    return None


def sensitive_field(el: dict) -> str | None:
    """A reason if this form field is for a password, payment or ID detail."""
    if (el.get("type") or "") == "password":
        return "it is a password field"
    if _SENSITIVE_AUTOCOMPLETE.match(el.get("autocomplete") or ""):
        return "it is marked as a password or payment field"
    words = " ".join(str(el.get(k) or "") for k in ("label", "name", "elid", "placeholder"))
    if _SENSITIVE_FIELD.search(words):
        return f"it asks for sensitive details ('{(el.get('label') or el.get('name') or '').strip()[:60]}')"
    return None


def irreversible(el: dict) -> str | None:
    """A reason if clicking this control likely can't be taken back."""
    if el.get("tag") in ("input", "button") and (el.get("type") in ("submit", "image")
                                                 or (el.get("tag") == "button" and el.get("in_form") and not el.get("type"))):
        return "it submits a form"
    words = " ".join(str(el.get(k) or "") for k in ("label", "name", "elid"))
    m = _IRREVERSIBLE.search(words)
    if m and (el.get("tag") in ("button", "input", "a", "summary") or el.get("role") in ("button", "link", "menuitem")):
        return f"it says '{m.group(0)}'"
    return None


def describe(el: dict) -> str:
    kind = el.get("role") or el.get("tag") or "element"
    if el.get("tag") == "input":
        kind = f"{el.get('type') or 'text'} field"
    elif el.get("tag") in ("textarea", "select"):
        kind = "text box" if el["tag"] == "textarea" else "drop-down"
    elif el.get("tag") == "a":
        kind = "link"
    label = (el.get("label") or el.get("placeholder") or el.get("name") or "").strip()
    return f'{kind} "{label[:80]}"' if label else kind


# ---- the DevTools connection -------------------------------------------------------------
class BrowserError(Exception):
    pass


class CDP:
    """A minimal synchronous DevTools session on one page target."""

    def __init__(self, ws_url: str, connect=None):
        if connect is None:
            from websockets.sync.client import connect as ws_connect
            try:                                  # websockets >= 15 reads proxy settings unless told not to
                opened = ws_connect(ws_url, max_size=32 * 1024 * 1024, open_timeout=10, proxy=None)
            except TypeError:
                opened = ws_connect(ws_url, max_size=32 * 1024 * 1024, open_timeout=10)
            # Newer websockets hand back a context manager, older ones the
            # connection itself; entering works for both.
            self.ws = opened.__enter__() if hasattr(opened, "__enter__") else opened
        else:
            self.ws = connect(ws_url)
        self._id = 0
        self._lock = threading.Lock()

    def send(self, method: str, params: dict | None = None, timeout: float = 20.0) -> dict:
        with self._lock:
            self._id += 1
            my = self._id
            self.ws.send(json.dumps({"id": my, "method": method, "params": params or {}}))
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise BrowserError(f"The browser did not answer {method} in time.")
                msg = json.loads(self.ws.recv(timeout=remaining))
                if msg.get("id") != my:
                    continue                      # an event, or a late answer
                if "error" in msg:
                    raise BrowserError(msg["error"].get("message") or str(msg["error"]))
                return msg.get("result") or {}

    def evaluate(self, expression: str, timeout: float = 20.0):
        res = self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True},
                        timeout)
        if res.get("exceptionDetails"):
            raise BrowserError("Page script failed: " + str(res["exceptionDetails"].get("text") or "")[:200])
        return (res.get("result") or {}).get("value")

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


# How a control is described - the same for the page read and the live
# re-check right before a click or fill.
_INFO_JS = r"""
  const clean = t => String(t || '').replace(/\s+/g, ' ').trim().slice(0, 120);
  const labelOf = el => {
    let t = el.getAttribute('aria-label') || '';
    if (!t && el.id) { const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]'); if (l) t = l.innerText; }
    if (!t && el.closest('label')) t = el.closest('label').innerText;
    if (!t && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) t = el.getAttribute('placeholder') || el.getAttribute('title') || '';
    if (!t) t = el.innerText || el.value || el.getAttribute('title') || el.getAttribute('alt') || '';
    return clean(t);
  };
  const info = (el, id) => {
    const form = el.form || el.closest('form');
    return { id, tag: el.tagName.toLowerCase(), type: (el.getAttribute('type') || '').toLowerCase(),
      role: el.getAttribute('role') || '', label: labelOf(el), name: el.getAttribute('name') || '', elid: el.id || '',
      placeholder: el.getAttribute('placeholder') || '', autocomplete: (el.getAttribute('autocomplete') || '').toLowerCase(),
      href: el.href || '', disabled: !!el.disabled, in_form: !!form, form_action: form ? String(form.action || '') : '',
      checked: el.checked === true, has_value: !!(el.value && el.value.length) };
  };
"""
_SNAPSHOT_JS = r"""
(() => {
  const visible = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  INFO
  document.querySelectorAll('[data-agentic-id]').forEach(e => e.removeAttribute('data-agentic-id'));
  const all = [...document.querySelectorAll('a[href], button, input, select, textarea, summary, [role=button], [role=link], [role=checkbox], [role=tab], [role=menuitem], [contenteditable=true]')]
    .filter(el => visible(el) && (el.getAttribute('type') || '').toLowerCase() !== 'hidden');
  const elements = all.slice(0, MAX).map((el, i) => {
    el.setAttribute('data-agentic-id', String(i + 1));
    return info(el, i + 1);
  });
  return { url: location.href, title: document.title, ready: document.readyState,
           text: (document.body ? document.body.innerText : '').replace(/\n{3,}/g, '\n\n'),
           elements, total: all.length };
})()
""".replace("MAX", str(MAX_ELEMENTS)).replace("INFO", _INFO_JS)


def _live_js(el_id: int) -> str:
    """The control as it is now: its description, where its centre is, and
    whether a click there really lands on it (not on something covering it)."""
    return ("(() => {" + _INFO_JS + f"const el = document.querySelector('[data-agentic-id=\"{int(el_id)}\"]'); "
            "if (!el) return {url: location.href, gone: true}; "
            "el.scrollIntoView({block: 'center', inline: 'center'}); const r = el.getBoundingClientRect(); "
            "const x = r.left + r.width / 2, y = r.top + r.height / 2; const top = document.elementFromPoint(x, y); "
            "const lbl = el.closest('label') || (el.id ? document.querySelector('label[for=\"' + CSS.escape(el.id) + '\"]') : null); "
            "const hit = !!top && (top === el || el.contains(top) || (!!lbl && lbl.contains(top))); "
            f"return Object.assign(info(el, {int(el_id)}), {{x, y, hit, url: location.href}}); }})()")


# The title of a bot check that clears by itself in a real browser.
_INTERSTITIAL_RE = re.compile(r"just a moment|checking your browser|one more step|please wait|verifying", re.I)

# What must not have changed between the page read (and the approval card,
# which shows it) and the click or fill.
_SAME_FIELDS = ("tag", "type", "role", "label", "name", "href", "in_form", "form_action")


# The browser's own background traffic, off: Edge and Chrome otherwise call their makers while
# the assistant works (updates, sync, sign-in, metrics uploads, link-audit pings, error reports).
QUIET_ARGS = ("--disable-background-networking", "--disable-component-update", "--disable-sync", "--no-pings",
              "--disable-domain-reliability", "--metrics-recording-only", "--disable-default-apps",
              "--disable-breakpad", "--disable-client-side-phishing-detection")


def route_args(route=None) -> list:
    """The network route for the assistant's browser (a separate process: the network layer
    can't see its connections, so it gets the route as its own proxy flags). Routed: the proxy,
    and no DNS lookups of its own - the proxy resolves names; down: blocked (kill switch)."""
    args = ["--force-webrtc-ip-handling-policy=disable_non_proxied_udp"]
    route = route or netlayer.route_for("search")
    netlayer.check(route, "search", "the assistant's browser")
    if route.routed:
        host, port = route.endpoint()
        scheme = "socks5" if route.proxy.startswith("socks5") else route.proxy.split("://")[0]
        args += [f"--proxy-server={scheme}://{host}:{port}",
                 f"--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE {host}"]
    else:
        args.append("--no-proxy-server")
    return args


def builtin() -> bool:
    """The assistant's browser runs hidden and shows in the app's Browser panel (settings browser_builtin)."""
    try:
        import core.state as shared
        return (shared.app_state.get("settings") or {}).get("browser_builtin", True) is not False
    except Exception:
        return True


def where_to_type() -> str:
    """Where the user finishes a step themselves - the panel, or the window."""
    return "the Browser panel beside the chat" if builtin() else "the assistant's browser window"


class AgentBrowser:
    """The assistant's browser (one per app run): hidden, shown in the Browser panel - or its own window."""

    def __init__(self, port: int = PORT, profile_dir: str = PROFILE_DIR, executable: str | None = None,
                 headless: bool | None = None, http=None, connect=None, extra_args: tuple = ()):
        """headless: None follows the setting (builtin()); tests pass True. Chromium's sandbox stays on."""
        self.port, self.profile_dir, self.executable, self.headless = port, profile_dir, executable, headless
        self._headless_now = None                 # how the running process was started
        self._ua_name = None                      # the browser's own name without "Headless"
        self.page_ws = None                       # the page the assistant works in (the Viewer shows it)
        self.extra_args = tuple(extra_args)
        self._http = http                         # tests: an httpx.Client with a mock transport
        self._connect = connect
        self._cdp: CDP | None = None
        self._proc = None
        self._lock = threading.RLock()
        self.snapshot: dict | None = None         # the last browser_read (element numbers are its)
        self._route = None                       # the flags of this browser process
        self._watch_stop = None

    def _check_network(self, refresh_vpn: bool = False):
        """Check even a reused browser: Chromium is outside Python's socket guard."""
        try:
            if refresh_vpn and netlayer.settings()["net_require_vpn"]:
                netlayer.vpn_status(force=True)
            route = netlayer.route_for("search")
            netlayer.check(route, "search", "the assistant's browser")
        except netlayer.RouteBlocked:
            self.close()                         # stop existing tabs' background traffic too
            raise
        if self._route is not None and route != self._route:
            self.close()                         # flags only change when Chromium restarts
        return route

    def _start_network_watch(self):
        """Close idle/background tabs when policy fails, rather than waiting for a tool.

        VPN loss is detected by adapter polling, not a packet-level firewall.
        Poll fresh adapter state every two seconds while this browser exists.
        """
        if self._watch_stop is not None:
            return
        stop = self._watch_stop = threading.Event()
        chat_id = privacy.current()

        def watch():
            with privacy.scope(chat_id):
                while not stop.wait(2.0):
                    with self._lock:
                        if stop.is_set():
                            return
                        try:
                            self._check_network(refresh_vpn=True)
                        except netlayer.RouteBlocked:
                            return
                        except Exception:
                            # A failed policy check must not leave an unguarded process.
                            self.close()
                            return
        threading.Thread(target=watch, daemon=True, name="browser-network-watch").start()

    def _leftover_pids(self) -> list:
        """The app's own browser left by an earlier run (the app was closed without its shutdown - run.bat's window
        closed, a crash): a browser process on this control port with this profile folder. Its proxy flags can't be
        read back, so it is never reused - only closed. Any other browser on the port is not ours to touch."""
        try:
            import psutil
        except ImportError:
            return []
        profile = os.path.normcase(os.path.abspath(self.profile_dir))
        pids = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = p.info["cmdline"] or []
                if any(a.startswith("--type=") for a in cmd) or f"--remote-debugging-port={self.port}" not in cmd:
                    continue                                  # a helper process, or another port
                dirs = [a.split("=", 1)[1].strip('"') for a in cmd if a.startswith("--user-data-dir=")]
                if dirs and os.path.normcase(os.path.abspath(dirs[0])) == profile:
                    pids.append(p.pid)
            except (psutil.Error, OSError):
                continue
        return pids

    def close_leftover(self) -> bool:
        """Close the app's own browser from an earlier run (see _leftover_pids). True when the port is free after."""
        pids = self._leftover_pids()
        if not pids:
            return False
        import psutil
        for pid in pids:
            try:
                main = psutil.Process(pid)
                for proc in main.children(recursive=True) + [main]:
                    proc.kill()
            except psutil.Error:
                pass
        deadline = time.time() + 5
        while self._alive() and time.time() < deadline:
            time.sleep(0.2)
        return not self._alive()

    def _refuse_unowned_browser(self):
        """Do not reuse or close a browser whose process/profile we do not own."""
        raise BrowserError("A browser already uses the assistant's control port. Close that browser and try again; "
                           "its network settings could not be verified.")

    # ---- process + target ----------------------------------------------------
    def _get(self, path: str, method: str = "GET"):
        client = self._http or netlayer.local_client(timeout=3.0)      # the DevTools port on 127.0.0.1 only
        try:
            return client.request(method, f"http://127.0.0.1:{self.port}{path}")
        finally:
            if client is not self._http:
                client.close()

    def _alive(self) -> bool:
        try:
            return self._get("/json/version").status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    def find_executable() -> str | None:
        explicit = os.environ.get("AGENTIC_BROWSER")
        if explicit and os.path.isfile(explicit):
            return explicit
        candidates = []
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            if base:
                candidates += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                               os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
        for path in candidates:
            if os.path.isfile(path):
                return path
        for name in ("msedge", "chrome", "chromium", "chromium-browser"):
            found = shutil.which(name)
            if found:
                return found
        return None

    def _headless(self) -> bool:
        return self.headless if self.headless is not None else builtin()

    def _ua(self, cdp: CDP) -> None:
        """A hidden browser calls itself HeadlessChrome - sites' bot checks refuse that; send its normal name."""
        if not self._headless_now:
            return
        try:
            if self._ua_name is None:
                self._ua_name = str(self._get("/json/version").json().get("User-Agent") or "").replace("HeadlessChrome", "Chrome")
            if self._ua_name:
                cdp.send("Network.setUserAgentOverride", {"userAgent": self._ua_name}, timeout=5)
        except Exception:
            pass

    def ensure(self):
        """Start the assistant's browser if it isn't running; returns a page session."""
        with self._lock:
            if self._headless_now is not None and self._headless_now != self._headless():
                self.close()                      # built in <-> own window: only a restart changes it
            route = self._check_network(refresh_vpn=True)
            if self._cdp is not None:
                try:
                    self._cdp.send("Runtime.evaluate", {"expression": "1", "returnByValue": True}, timeout=3)
                    return self._cdp
                except Exception:
                    self._cdp.close()
                    self._cdp = None
            if self._route is None and self._alive() and not self.close_leftover():
                # Someone else's browser (or another program) has the control port: never touch it - ours moves to
                # a free port instead of refusing to start (1.4.7).
                import ports
                self.port = ports.pick(self.port + 1, avoid=(TEMP_PORT,) if self.port != TEMP_PORT else ())
                ports.note("browser" if self.port != TEMP_PORT else "browser_temp", self.port, "moved")
            elif self._route is None and not self._alive():
                import ports
                if not ports.is_free(self.port):          # a program that isn't a browser holds it
                    self.port = ports.pick(self.port + 1, avoid=(TEMP_PORT,))
            if not self._alive():
                exe = self.executable or self.find_executable()
                if not exe:
                    raise BrowserError("Neither Microsoft Edge nor Google Chrome was found, so the assistant can't open "
                                       "its browser. Install one, or set AGENTIC_BROWSER to its .exe.")
                os.makedirs(self.profile_dir, exist_ok=True)
                args = [exe, f"--remote-debugging-port={self.port}", f"--user-data-dir={self.profile_dir}",
                        "--no-first-run", "--no-default-browser-check", *QUIET_ARGS,
                        "--remote-allow-origins=http://127.0.0.1", "about:blank"]
                headless = self._headless()
                if headless:
                    args[1:1] = ["--headless=new", "--window-size=1280,860", "--hide-scrollbars"]
                args[1:1] = list(self.extra_args) + route_args(route)
                self._proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if headless else 0)
                self._route, self._headless_now, self._ua_name = route, headless, None
                deadline = time.time() + 20
                while not self._alive():
                    if time.time() > deadline:
                        raise BrowserError("The assistant's browser did not start within 20 seconds.")
                    time.sleep(0.25)
            pages = [t for t in self._get("/json/list").json() if t.get("type") == "page"]
            if not pages:
                resp = self._get("/json/new?about:blank", "PUT")
                pages = [resp.json()]
            if self._headless_now is None:
                self._headless_now = self._headless()     # a browser this process didn't start (tests)
            self.page_ws = pages[0]["webSocketDebuggerUrl"]
            self._cdp = CDP(self.page_ws, connect=self._connect)
            self._cdp.send("Page.enable")
            self._ua(self._cdp)
            self._start_network_watch()
            return self._cdp

    def _wait_ready(self, cdp: CDP, timeout: float = 15.0):
        deadline = time.time() + timeout
        time.sleep(0.2)
        while time.time() < deadline:
            try:
                if cdp.evaluate("document.readyState", timeout=5) == "complete":
                    return
            except BrowserError:
                pass
            time.sleep(0.25)

    # ---- tools ---------------------------------------------------------------------
    def open(self, url: str) -> dict:
        from web_tools import _normalize_url, is_web_url
        target = _normalize_url(url)
        if not is_web_url(target):
            return {"ok": False, "message": f"Refused '{target}': the assistant's browser only opens http(s) pages."}
        if is_app_url(target):
            return {"ok": False, "message": APP_URL_REFUSAL}
        cdp = self.ensure()
        res = cdp.send("Page.navigate", {"url": target}, timeout=30)
        if res.get("errorText"):
            return {"ok": False, "message": f"Could not open {target}: {res['errorText']}"}
        self._wait_ready(cdp)
        self.snapshot = None
        title = cdp.evaluate("document.title") or ""
        return {"ok": True, "message": f"Opened {target} in the assistant's browser ('{title}'). Use browser_read to see it.",
                "data": {"url": target, "title": title}}

    def read(self, start: int = 0) -> dict:
        cdp = self.ensure()
        if is_app_url(cdp.evaluate("location.href") or ""):
            self.snapshot = None
            return {"ok": False, "message": APP_URL_REFUSAL}
        snap = cdp.evaluate(_SNAPSHOT_JS)
        if not isinstance(snap, dict):
            return {"ok": False, "message": "Could not read the page."}
        self.snapshot = snap
        text = snap.get("text") or ""
        start = max(0, int(start or 0))
        window = text[start:start + READ_WINDOW]
        end = start + len(window)
        lines = [f"Page: {snap.get('title') or snap.get('url')}", f"Source: {snap.get('url')}", "", window]
        if end < len(text):
            lines.append(f"\n[NOT SHOWN: characters {end:,}-{len(text):,}. Call browser_read with start={end} to read on.]")
        els = snap.get("elements") or []
        if els:
            lines.append("\nControls (use the number with browser_click / browser_fill):")
            for el in els:
                flags = []
                if el.get("disabled"):
                    flags.append("disabled")
                if el.get("has_value") and el.get("tag") in ("input", "textarea"):
                    flags.append("filled")
                if sensitive_field(el):
                    flags.append("sensitive - the user fills this in")
                if irreversible(el):
                    flags.append("can't be undone")
                lines.append(f"[{el['id']}] {describe(el)}" + (f" -> {el['href']}" if el.get("href") else "")
                             + (f" ({', '.join(flags)})" if flags else ""))
            if (snap.get("total") or 0) > len(els):
                lines.append(f"(first {len(els)} of {snap['total']} controls)")
        return {"ok": True, "message": "\n".join(lines),
                "data": {"url": snap.get("url"), "title": snap.get("title"), "elements": len(els)}}

    def load(self, url: str, extract: str | None = None, wait_for: str | None = None, settle: float = 12.0) -> dict:
        """web_read's and web_search's fallback when a site refuses direct
        requests: a real browser passes the checks sites put in front of
        scripts. Loads url in a temporary tab and returns {url, title, html},
        or {url, title, value} where value is what the `extract` script
        returns. The tab closes afterwards, so the page the assistant has
        open (and its element numbers) stays as it was. An interstitial bot
        check ("Just a moment...") gets up to `settle` seconds to clear."""
        from web_tools import _normalize_url, is_web_url
        target = _normalize_url(url)
        if not is_web_url(target):
            raise BrowserError(f"Refused '{target}': the assistant's browser only opens http(s) pages.")
        if is_app_url(target):
            raise BrowserError(APP_URL_REFUSAL)
        with self._lock:
            self.ensure()
            tab = self._get("/json/new?about:blank", "PUT").json()
        cdp = CDP(tab["webSocketDebuggerUrl"], connect=self._connect)
        try:
            cdp.send("Page.enable")
            self._ua(cdp)
            res = cdp.send("Page.navigate", {"url": target}, timeout=30)
            if res.get("errorText"):
                raise BrowserError(f"Could not open {target}: {res['errorText']}")
            self._wait_ready(cdp)
            deadline = time.time() + settle
            while time.time() < deadline and _INTERSTITIAL_RE.search(cdp.evaluate("document.title") or ""):
                time.sleep(0.5)
            # Results and script-built pages arrive after the load event.
            probe = (f"!!document.querySelector({json.dumps(wait_for)})" if wait_for
                     else "(document.body ? document.body.innerText.length : 0) >= 200")
            until = time.time() + (8 if wait_for else 4)
            while time.time() < until and not cdp.evaluate(probe):
                time.sleep(0.3)
            where = cdp.evaluate("location.href") or target
            if is_app_url(where):
                raise BrowserError(APP_URL_REFUSAL)
            out = {"url": where, "title": cdp.evaluate("document.title") or ""}
            if extract:
                out["value"] = cdp.evaluate(extract)
            else:
                out["html"] = cdp.evaluate("document.documentElement.outerHTML") or ""
            return out
        finally:
            try:
                cdp.send("Page.close", timeout=3)
            except Exception:
                pass
            cdp.close()
            try:
                self._get(f"/json/close/{tab['id']}")      # already gone after Page.close: harmless
            except Exception:
                pass

    def element(self, el_id) -> dict | None:
        try:
            wanted = int(el_id)
        except (TypeError, ValueError):
            return None
        return next((e for e in (self.snapshot or {}).get("elements") or [] if e.get("id") == wanted), None)

    def _live(self, cdp: CDP, el: dict, action: str) -> dict:
        """Re-read the control right before acting and refuse unless it is
        still the control that was read (and shown on the approval card)."""
        live = cdp.evaluate(_live_js(el["id"]))
        if not isinstance(live, dict) or is_app_url(live.get("url") or ""):
            raise BrowserError(APP_URL_REFUSAL if isinstance(live, dict) else "Could not read the page.")
        if live.get("gone"):
            raise BrowserError(f"Nothing was {action}: that control is no longer on the page (it changed). "
                               "Call browser_read again.")
        changed = [k for k in _SAME_FIELDS if (live.get(k) or "") != (el.get(k) or "")]
        if changed:
            self.snapshot = None
            raise BrowserError(f"Nothing was {action}: the page changed since it was read - [{el['id']}] was "
                               f"{describe(el)}, now it is {describe(live)}. Call browser_read again.")
        if not live.get("hit"):
            raise BrowserError(f"Nothing was {action}: something on the page covers [{el['id']}] {describe(el)}, so "
                               "the click would land on something else. Call browser_read again, or ask the user.")
        return live

    def click(self, el_id) -> dict:
        el = self.element(el_id)
        if el is None:
            return {"ok": False, "message": f"There is no control [{el_id}] - call browser_read first and use its numbers."}
        if el.get("disabled"):
            return {"ok": False, "message": f"[{el_id}] {describe(el)} is disabled."}
        cdp = self.ensure()
        before = cdp.evaluate("location.href")
        where = self._live(cdp, el, "clicked")
        for kind in ("mousePressed", "mouseReleased"):
            cdp.send("Input.dispatchMouseEvent", {"type": kind, "x": where["x"], "y": where["y"], "button": "left",
                                                  "clickCount": 1})
        self._wait_ready(cdp, 10)
        self.snapshot = None
        after = cdp.evaluate("location.href")
        moved = f" The page is now {after}." if after != before else ""
        return {"ok": True, "message": f"Clicked [{el['id']}] {describe(el)}.{moved} Use browser_read to see the result.",
                "data": {"url": after}}

    def fill(self, el_id, value: str) -> dict:
        el = self.element(el_id)
        if el is None:
            return {"ok": False, "message": f"There is no control [{el_id}] - call browser_read first and use its numbers."}
        why = sensitive_field(el) or looks_sensitive_value(value)
        if why:
            return {"ok": False, "sensitive": True, "message": (
                f"I won't type into [{el['id']}] {describe(el)} because {why}. Passwords, payment details and ID "
                f"numbers are for the user to enter: ask them to fill it in themselves in {where_to_type()}, then "
                "continue.")}
        if el.get("tag") not in ("input", "textarea", "select") and not el.get("role"):
            return {"ok": False, "message": f"[{el['id']}] {describe(el)} is not a field you can type into."}
        cdp = self.ensure()
        live = self._live(cdp, el, "typed")
        why = sensitive_field(live)
        if why:
            return {"ok": False, "sensitive": True, "message": (
                f"I won't type into [{el['id']}] {describe(live)} because {why}. Ask the user to fill it in "
                f"themselves in {where_to_type()}, then continue.")}
        sel = f'[data-agentic-id="{int(el["id"])}"]'
        if el.get("tag") == "select":
            ok = cdp.evaluate(
                f"(() => {{ const el = document.querySelector('{sel}'); const want = {json.dumps(str(value))}.toLowerCase(); "
                "const opt = [...el.options].find(o => o.value.toLowerCase() === want || o.text.trim().toLowerCase() === want); "
                "if (!opt) return false; el.value = opt.value; el.dispatchEvent(new Event('input', {bubbles: true})); "
                "el.dispatchEvent(new Event('change', {bubbles: true})); return true; })()")
            if not ok:
                return {"ok": False, "message": f"[{el['id']}] has no option '{value}'."}
        else:
            cdp.evaluate(f"(() => {{ const el = document.querySelector('{sel}'); el.focus(); "
                         "if (el.select) el.select(); return true; })()")
            cdp.send("Input.insertText", {"text": str(value)})
            cdp.evaluate(f"(() => {{ const el = document.querySelector('{sel}'); "
                         "el.dispatchEvent(new Event('change', {bubbles: true})); return true; })()")
        now = cdp.evaluate(f"(() => {{ const el = document.querySelector('{sel}'); return el ? String(el.value) : null; }})()")
        verified = now == str(value) if el.get("tag") != "select" else bool(now)
        return {"ok": True, "verified": verified,
                "message": f"Filled [{el['id']}] {describe(el)}." + ("" if verified else " It shows a different value now - check it.")}

    def close(self):
        with self._lock:
            if self._watch_stop is not None:
                self._watch_stop.set()
                self._watch_stop = None
            if self._cdp:
                try:
                    self._cdp.send("Browser.close", timeout=3)
                except Exception:
                    pass
                self._cdp.close()
                self._cdp = None
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
            self._route = None
            self._headless_now = None
            self.page_ws = None
            self.snapshot = None


# The assistant's browser (started on first use).
agent_browser = AgentBrowser()
# A temporary chat's own browser (privacy.py): a separate window whose profile - history,
# cookies, cache - lives in that chat's temporary folder and goes with it.
TEMP_PORT = PORT + 1
_temp_browsers: dict = {}
_temp_lock = threading.Lock()


def current() -> AgentBrowser:
    """The browser this work uses: the assistant's own, or - inside a temporary
    chat - that chat's private one (nothing it visits reaches data/browser-profile)."""
    chat_id = privacy.current()
    if not chat_id:
        return agent_browser
    with _temp_lock:
        b = _temp_browsers.get(chat_id)
        if b is None:
            b = AgentBrowser(port=TEMP_PORT, profile_dir=os.path.join(privacy.temp_dir(chat_id), "browser-profile"),
                             extra_args=("--incognito" if "chrome" in (AgentBrowser.find_executable() or "").lower()
                                         else "--inprivate",))
            _temp_browsers[chat_id] = b
        return b


@privacy.on_discard
def _close_temp_browser(chat_id: str):
    with _temp_lock:
        b = _temp_browsers.pop(chat_id, None)
    if b is not None:
        b.close()


def close_all():
    agent_browser.close()
    for chat_id in list(_temp_browsers):
        _close_temp_browser(chat_id)


# ---- The Browser panel (D50) --------------------------------------------------------------------------------
# Keys the panel passes on as keys (the rest of what you type arrives as text): Windows virtual-key codes.
_KEYS = {"Enter": (13, "\r"), "Backspace": (8, None), "Tab": (9, None), "Escape": (27, None), "Delete": (46, None),
         "ArrowLeft": (37, None), "ArrowUp": (38, None), "ArrowRight": (39, None), "ArrowDown": (40, None),
         "Home": (36, None), "End": (35, None), "PageUp": (33, None), "PageDown": (34, None)}
_MODS = {"alt": 1, "ctrl": 2, "meta": 4, "shift": 8}


class Viewer:
    """What the Browser panel sees and sends: a picture of the page the assistant works in, its address and
    history, and YOUR input - clicks, scrolling, typing, the address bar. Its own DevTools session on the same
    page, so the panel never waits behind a tool (or a tool behind the panel)."""

    def __init__(self):
        self._lock = threading.RLock()
        self._cdp = None
        self._ws = None
        self._last = b""

    def _session(self, start: bool = False):
        b = current()
        with self._lock:
            if start:
                b.ensure()
            if b._route is None or not b.page_ws:            # not running (or not ours): nothing to show
                self._drop()
                return None, b
            if self._cdp is None or self._ws != b.page_ws:
                self._drop()
                self._cdp, self._ws = CDP(b.page_ws, connect=b._connect), b.page_ws
                b._ua(self._cdp)
            return self._cdp, b

    def _drop(self):
        if self._cdp is not None:
            self._cdp.close()
        self._cdp, self._ws = None, None

    def _guard(self, fn):
        try:
            return fn()
        except (BrowserError, OSError, RuntimeError, ValueError, httpx.HTTPError, Exception):
            with self._lock:
                self._drop()
            return None

    def state(self, cdp=None) -> dict:
        cdp = cdp or self._session()[0]
        if cdp is None:
            return {"running": False, "builtin": builtin()}
        def read():
            info = cdp.evaluate("({url: location.href, title: document.title, ready: document.readyState})", timeout=5) or {}
            hist = cdp.send("Page.getNavigationHistory", timeout=5)
            i, n = hist.get("currentIndex", 0), len(hist.get("entries") or [])
            vp = cdp.send("Page.getLayoutMetrics", timeout=5).get("cssVisualViewport") or {}
            return {"running": True, "builtin": builtin(), "url": info.get("url") or "", "title": info.get("title") or "",
                    "loading": info.get("ready") != "complete", "back": i > 0, "forward": i < n - 1,
                    "w": int(vp.get("clientWidth") or 0), "h": int(vp.get("clientHeight") or 0)}
        return self._guard(read) or {"running": False, "builtin": builtin()}

    def frame(self) -> tuple:
        """(jpeg bytes or None, its hash, the state) - the page as it looks now."""
        cdp = self._session()[0]
        if cdp is None:
            return None, "", {"running": False, "builtin": builtin()}
        shot = self._guard(lambda: cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 62}, timeout=8))
        st = self.state(cdp)
        if not shot:
            return None, "", st
        data = base64.b64decode(shot.get("data") or "")
        return data, hashlib.md5(data).hexdigest()[:16], st

    def input(self, ev: dict) -> dict:
        cdp = self._session()[0]
        if cdp is None:
            return {"ok": False, "message": "The assistant's browser isn't open."}
        kind = str(ev.get("type") or "")
        x, y = float(ev.get("x") or 0), float(ev.get("y") or 0)
        mods = sum(v for k, v in _MODS.items() if (ev.get("mods") or {}).get(k))
        def send():
            if kind == "click":
                n = max(1, min(3, int(ev.get("clicks") or 1)))
                button = "right" if ev.get("button") == "right" else "left"
                cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y, "modifiers": mods})
                for t in ("mousePressed", "mouseReleased"):
                    cdp.send("Input.dispatchMouseEvent", {"type": t, "x": x, "y": y, "button": button, "clickCount": n, "modifiers": mods})
            elif kind == "move":
                cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y, "modifiers": mods})
            elif kind == "wheel":
                cdp.send("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": x, "y": y, "modifiers": mods,
                                                      "deltaX": float(ev.get("dx") or 0), "deltaY": float(ev.get("dy") or 0)})
            elif kind == "resize":
                # The page lays itself out at the panel's own size, so the picture is 1:1 - sharp, and a click lands
                # where you clicked. The assistant's reads see the same page.
                w, h = int(max(320, min(2560, x))), int(max(240, min(1600, y)))
                dpr = max(1.0, min(2.0, float(ev.get("dx") or 1)))
                cdp.send("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": dpr, "mobile": False})
            elif kind == "text":
                cdp.send("Input.insertText", {"text": str(ev.get("text") or "")[:2000]})
            elif kind == "key":
                key = str(ev.get("key") or "")
                code, text = _KEYS.get(key, (ord(key.upper()) if len(key) == 1 else 0, None))
                down = {"type": "keyDown" if text else "rawKeyDown", "key": key, "windowsVirtualKeyCode": code,
                        "nativeVirtualKeyCode": code, "modifiers": mods, **({"text": text} if text else {})}
                cdp.send("Input.dispatchKeyEvent", down)
                cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": key, "windowsVirtualKeyCode": code,
                                                    "nativeVirtualKeyCode": code, "modifiers": mods})
            else:
                raise ValueError(kind)
            return True
        return {"ok": bool(self._guard(send))}

    def navigate(self, action: str, url: str | None = None) -> dict:
        if action == "close":
            current().close()
            with self._lock:
                self._drop()
            return {"ok": True}
        try:
            cdp, b = self._session(start=action in ("open", "go"))
        except (BrowserError, netlayer.RouteBlocked) as e:
            return {"ok": False, "message": str(e)}
        if cdp is None:
            return {"ok": False, "message": "The assistant's browser isn't open."}
        if action == "go":
            from web_tools import _normalize_url, is_web_url
            raw = str(url or "").strip()
            if raw and not re.search(r"^[a-z]+://|^[\w-]+(\.[\w-]+)+(/|$)", raw, re.I):
                from urllib.parse import quote_plus
                raw = "https://duckduckgo.com/?q=" + quote_plus(raw)      # words, not an address: search for them
            target = _normalize_url(raw)
            if not is_web_url(target):
                return {"ok": False, "message": f"Only http(s) pages open here, not '{target}'."}
            if is_app_url(target):
                return {"ok": False, "message": APP_URL_REFUSAL}
            res = self._guard(lambda: cdp.send("Page.navigate", {"url": target}, timeout=30)) or {}
            b.snapshot = None                                   # the assistant reads the page again before acting
            return {"ok": not res.get("errorText"), "message": res.get("errorText") or ""}
        def hist():
            if action == "reload":
                cdp.send("Page.reload", {})
            elif action in ("back", "forward"):
                h = cdp.send("Page.getNavigationHistory")
                i = h.get("currentIndex", 0) + (-1 if action == "back" else 1)
                entries = h.get("entries") or []
                if 0 <= i < len(entries):
                    cdp.send("Page.navigateToHistoryEntry", {"entryId": entries[i]["id"]})
            return True
        ok = action == "open" or bool(self._guard(hist))
        b.snapshot = None
        return {"ok": ok}


viewer = Viewer()


def click_needs_approval(args: dict) -> str | None:
    """permissions.ALWAYS_ASK_WHEN hook: a click on a control that can't be
    taken back must be approved every time."""
    el = current().element((args or {}).get("element"))
    if el is None:
        # Not in the last page read (or nothing read yet): what it does can't be checked.
        return "the control isn't in the last page read, so what it does can't be checked"
    why = irreversible(el)
    return f"this click can't be undone ({why})" if why else None


def approval_preview(tool: str, args: dict) -> dict | None:
    """What the approval card shows for a click or fill: the control, the page,
    and whether it can be taken back."""
    b = current()
    el = b.element((args or {}).get("element"))
    page = (b.snapshot or {}).get("url") or "the current page"
    if el is None:
        return {"kind": "text", "text": f"Control [{(args or {}).get('element')}] on {page} (not in the last page read)."}
    lines = [f"Page: {page}", f"Control: [{el['id']}] {describe(el)}" + (f" -> {el['href']}" if el.get("href") else "")]
    if tool == "browser_fill":
        lines.append(f"Types: {args.get('value')}")
    why = irreversible(el) if tool == "browser_click" else None
    note = f"This click probably can't be taken back: {why}." if why else None
    return {"kind": "text", "text": "\n".join(lines), **({"note": note} if note else {})}


# Registered here so the rules live with the browser code but are enforced by
# the one permission path and the one approval view.
def _register():
    import approval_view
    import permissions
    permissions.ALWAYS_ASK_WHEN["browser_click"] = click_needs_approval
    approval_view.EXTRA_PREVIEWS["browser_click"] = approval_preview
    approval_view.EXTRA_PREVIEWS["browser_fill"] = approval_preview


_register()
