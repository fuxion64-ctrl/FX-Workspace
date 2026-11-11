"""
remote_access.py
================
Use the app from your phone (1.4.2), off by default - Settings > Connectors > Phone access.

  Where it listens   a second listener of the same app, on your Tailscale address (100.64.0.0/10, encrypted by
                     Tailscale) or your home network's private address - never a public one, and never 0.0.0.0.
                     The app on 127.0.0.1:8000 is unchanged.
  Who gets in        a password you set here, stored as a scrypt hash (data/remote.json); a sign-in page; a session
                     cookie (HttpOnly, SameSite=Strict, 30 days); five wrong tries from an address lock it out for
                     15 minutes. Signed-in devices are listed and can be signed out.
  What a phone may do  chat, approve or refuse, notes and tasks, read results. Not unless you allow it on the PC:
                     screen, mouse and keyboard control (the Apps & windows tools). Never from a phone: Settings
                     changes, API keys, this page's own switches (PHONE_BLOCKED).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import threading
import time

import core.state as shared

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "remote.json")
MODES = ("off", "tailscale", "lan")
DEFAULT_PORT = 8765
COOKIE = "fxr"
SESSION_DAYS = 30
MAX_TRIES, LOCK_SECONDS = 5, 15 * 60
MIN_PASSWORD = 8
TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
# Reachable without signing in: the sign-in page itself and what an installed home-screen app needs.
OPEN_PATHS = ("/remote/login", "/remote/manifest.webmanifest", "/remote/icon.svg")
# Never from a phone: settings, keys, permissions, network, this feature's own controls, installing an update and
# quitting the desktop app, opening Windows' own settings.
PHONE_BLOCKED = (("POST", "/api/settings"), ("POST", "/api/keys"), ("*", "/api/remote"), ("POST", "/api/permissions"),
                 ("POST", "/api/network"), ("POST", "/api/channels"), ("POST", "/api/mcp"), ("POST", "/api/connectors"),
                 ("POST", "/api/version/install"), ("POST", "/api/shell"), ("POST", "/api/notify/quiet/open"),
                 ("*", "/api/data-files"), ("POST", "/api/images/models"),
                 ("POST", "/api/local-models/ram-split"),   # 1.4.6: it rewrites models.ini
                 ("GET", "/api/whoami"),
                 ("POST", "/api/local-models/find-file"), ("POST", "/api/local-models/remove"),
                 ("POST", "/api/local-models/folders"), ("POST", "/api/local-models/browse"),   # 1.4.8: settings, a PC dialog
                 ("POST", "/api/local-models/mmproj"))                                          # 1.4.8: models.ini   # 1.4.7: models.ini                    # 1.4.7: the install folder and process - for this PC only
# (1.4.8, audit 3.6) What a phone MAY call under /api/: path template -> methods. An endpoint in neither list is
# refused from a phone, so a new endpoint is PC-only until it is added here on purpose (tests/test_server_148.py
# walks app.routes and fails on one that is in neither). Taken from every route PHONE_BLOCKED let through on
# 10 Nov 2026, so behaviour is unchanged. PHONE_BLOCKED still wins over this list.
PHONE_ALLOWED = {
    "/api/app-window": "GET",
    "/api/app-window/open": "POST",
    "/api/approval/{request_id}": "POST",
    "/api/apps": "GET",
    "/api/apps/{app_id}/check": "POST",
    "/api/awareness/clear": "POST",
    "/api/awareness/detectors": "GET",
    "/api/awareness/detectors/{detector_id}/reenable": "POST",
    "/api/awareness/unmute": "POST",
    "/api/benchmarks": "GET POST",
    "/api/benchmarks/cancel": "POST",
    "/api/browser/view/frame": "GET",
    "/api/browser/view/input": "POST",
    "/api/browser/view/nav": "POST",
    "/api/channels": "GET",
    "/api/chat": "POST",
    "/api/chat-sandbox": "POST",
    "/api/chat-tools": "POST",
    "/api/chats": "GET",
    "/api/chats/temporary": "POST",
    "/api/chats/temporary/save": "POST",
    "/api/chats/{chat_id}": "DELETE GET POST",
    "/api/chats/{chat_id}/messages": "POST",
    "/api/chats/{chat_id}/open": "POST",
    "/api/chats/{chat_id}/transcript": "GET",      # (1.4.8) older messages of a long chat, as the page scrolls up
    "/api/chats/{chat_id}/unlock": "POST",
    "/api/clarification-popup": "POST",
    "/api/client-log": "POST",
    "/api/code/run": "POST",
    "/api/compare": "GET POST",
    "/api/compare/{rid}": "DELETE GET",
    "/api/compare/{rid}/combine": "POST",
    "/api/compare/{rid}/vote": "POST",
    "/api/connectors": "GET",
    "/api/connectors/{cid}": "DELETE",
    "/api/context": "GET",
    "/api/context/compress": "POST",
    "/api/diagnostics": "GET",
    "/api/diagnostics/fix": "POST",
    "/api/docs": "GET POST",
    "/api/docs/ai/edit": "POST",
    "/api/docs/ai/proofread": "POST",
    "/api/docs/{did}": "DELETE GET POST",
    "/api/docs/{did}/restore/{n}": "POST",
    "/api/docs/{did}/versions": "GET",
    "/api/effort": "GET",
    "/api/effort/session": "POST",
    "/api/favicon/{domain}": "GET",
    "/api/files": "POST",
    "/api/files/{fid}": "GET",
    "/api/gpu/apps": "GET",
    "/api/guard": "GET",
    "/api/guard/limits": "POST",
    "/api/identity": "GET POST",
    "/api/images": "GET",
    "/api/images/generate": "POST",
    "/api/images/improve": "POST",
    "/api/images/move": "POST",
    "/api/images/move/cancel": "POST",
    "/api/images/move/plan": "POST",
    "/api/images/sequence/{seq_id}": "GET",
    "/api/images/status": "GET",
    "/api/images/stop": "POST",
    "/api/images/thumb/{name}": "GET",
    "/api/images/trash/empty": "POST",
    "/api/images/trash/{name}": "GET",
    "/api/images/unload": "POST",
    "/api/images/warm": "POST",
    "/api/images/{iid}/again": "POST",
    "/api/images/{iid}/edit": "POST",
    "/api/images/{iid}/expand": "POST",
    "/api/images/{iid}/fav": "POST",
    "/api/images/{iid}/purge": "POST",
    "/api/images/{iid}/restore": "POST",
    "/api/images/{iid}/trash": "POST",
    "/api/images/{name}": "GET",
    "/api/knowledge": "GET",
    "/api/knowledge/folders": "POST",
    "/api/knowledge/reindex": "POST",
    "/api/knowledge/remove": "POST",
    "/api/knowledge/search": "GET",
    "/api/labs": "GET",
    "/api/labs/build": "POST",
    "/api/labs/build/cancel": "POST",
    "/api/labs/datasets": "GET POST",
    "/api/labs/datasets/{did}/delete": "POST",
    "/api/labs/drafts": "GET",
    "/api/labs/drafts/{bid}": "GET",
    "/api/labs/drafts/{bid}/delete": "POST",
    "/api/labs/drafts/{bid}/save": "POST",
    "/api/labs/models": "GET POST",
    "/api/labs/models/{mid}/delete": "POST",
    "/api/labs/ncm": "GET",
    "/api/labs/ncm/compare": "POST",
    "/api/labs/ncm/setup": "POST",
    "/api/labs/ncm/write": "POST",
    "/api/labs/runs": "GET POST",
    "/api/labs/runs/{rid}": "GET POST",
    "/api/labs/runs/{rid}/delete": "POST",
    "/api/labs/runs/{rid}/draft": "POST",
    "/api/labs/runs/{rid}/port": "POST",
    "/api/labs/runs/{rid}/resume": "POST",
    "/api/labs/runs/{rid}/samples/{name}": "GET",
    "/api/labs/runs/{rid}/start": "POST",
    "/api/labs/runs/{rid}/stop": "POST",
    "/api/labs/runs/{rid}/trace": "POST",
    "/api/labs/runs/{rid}/try": "POST",
    "/api/labs/runs/{rid}/version": "POST",
    "/api/labs/setup": "POST",
    "/api/llama/builds": "GET",
    "/api/llama/builds/apply": "POST",
    "/api/local-models": "GET",
    "/api/local-models/add": "POST",
    "/api/local-models/config": "GET POST",
    "/api/local-models/ctx-info": "GET",
    "/api/local-models/folders": "GET",
    "/api/local-models/force-load": "POST",
    "/api/local-models/open-folder": "POST",
    "/api/local-models/scan": "POST",
    "/api/local-models/warm": "POST",
    "/api/mail/accounts": "GET POST",
    "/api/mail/accounts/{aid}": "DELETE POST",
    "/api/mail/act": "POST",
    "/api/mail/ai/reply": "POST",
    "/api/mail/ai/summary": "POST",
    "/api/mail/ai/triage": "POST",
    "/api/mail/attachment": "GET",
    "/api/mail/drafts": "POST",
    "/api/mail/drafts/{did}": "DELETE",
    "/api/mail/folders": "GET",
    "/api/mail/list": "GET",
    "/api/mail/message": "GET",
    "/api/mail/preset": "GET",
    "/api/mail/send": "POST",
    "/api/mcp": "GET",
    "/api/mcp/servers/{name}": "DELETE",
    "/api/memory": "GET POST",
    "/api/memory-export": "GET",
    "/api/memory-import": "POST",
    "/api/memory/graph": "GET",
    "/api/memory/maintain": "POST",
    "/api/memory/{entry_id}": "DELETE POST",
    "/api/messaging/status": "GET",
    "/api/mini-window": "POST",
    "/api/model-finder/cancel": "POST",
    "/api/model-finder/download": "POST",
    "/api/model-finder/files": "GET",
    "/api/model-finder/pause": "POST",
    "/api/model-finder/search": "GET",
    "/api/model-finder/starter": "GET",
    "/api/model-finder/status": "GET",
    "/api/model-monitor": "GET",
    "/api/models/assets": "GET",
    "/api/models/assets/{model}": "POST",
    "/api/models/registry": "GET",
    "/api/moe-test": "GET POST",
    "/api/music": "GET",
    "/api/music/generate": "POST",
    "/api/music/status": "GET",
    "/api/music/stop": "POST",
    "/api/music/trash/empty": "POST",
    "/api/music/trash/{name}": "GET",
    "/api/music/unload": "POST",
    "/api/music/{mid}/fav": "POST",
    "/api/music/{mid}/purge": "POST",
    "/api/music/{mid}/restore": "POST",
    "/api/music/{mid}/title": "POST",
    "/api/music/{mid}/trash": "POST",
    "/api/music/{name}": "GET",
    "/api/network": "GET",
    "/api/network/log": "DELETE GET",
    "/api/network/stats": "GET",
    "/api/network/tor": "GET",
    "/api/network/vpn": "GET",
    "/api/notes": "GET POST",
    "/api/notes/{nid}": "DELETE POST",
    "/api/notify/quiet": "GET",
    "/api/osint/basemap": "GET",
    "/api/osint/beat": "POST",
    "/api/osint/layer/{name}": "GET",
    "/api/osint/mentions": "POST",
    "/api/osint/place": "GET",
    "/api/osint/replay": "GET",
    "/api/osint/status": "GET",
    "/api/permissions/tools": "GET",
    "/api/prep": "GET",
    "/api/prep/session": "POST",
    "/api/prep/{card_id}": "POST",
    "/api/profile": "GET POST",
    "/api/profile/accept": "POST",
    "/api/profile/discard": "POST",
    "/api/profile/propose": "POST",
    "/api/projects": "GET POST",
    "/api/projects/{pid}": "DELETE GET POST",
    "/api/projects/{pid}/activate": "POST",
    "/api/projects/{pid}/index": "GET",
    "/api/reset": "POST",
    "/api/roblox/context": "POST",
    "/api/roblox/poll": "GET",
    "/api/roblox/result": "POST",
    "/api/roblox/status": "GET",
    "/api/routines": "GET POST",
    "/api/routines/{rid}": "DELETE POST",
    "/api/routines/{rid}/run": "POST",
    "/api/routines/{rid}/schedule": "DELETE POST",
    "/api/run/{action}": "POST",
    "/api/runs": "DELETE GET",
    "/api/runs/{run_id}": "DELETE GET",
    "/api/search": "GET",
    "/api/search/clear": "POST",
    "/api/search/keys": "GET POST",
    "/api/search/modes": "GET",
    "/api/search/test": "POST",
    "/api/settings": "GET",
    "/api/shell": "GET",
    "/api/skills": "GET POST",
    "/api/skills-draft": "POST",
    "/api/skills-import": "POST",
    "/api/skills/{slug}": "DELETE GET POST",
    "/api/skills/{slug}/export": "GET",
    "/api/speed-tune": "GET POST",
    "/api/state": "GET",
    "/api/storage": "GET",
    "/api/storage/map": "GET",
    "/api/stream": "GET",
    "/api/suggestions": "GET",
    "/api/suggestions/{suggestion_id}/accept": "POST",
    "/api/suggestions/{suggestion_id}/dismiss": "POST",
    "/api/suggestions/{suggestion_id}/snooze": "POST",
    "/api/system-messages": "GET",
    "/api/tasks": "POST",
    "/api/tasks/clear-done": "POST",
    "/api/tasks/{tid}": "DELETE POST",
    "/api/tools": "GET",
    "/api/turn/{turn_id}": "GET",
    "/api/ui-state": "GET POST",   # (1.4.8) the page's own look-and-feel choices, kept per device profile
    "/api/undo": "GET",
    "/api/undo/last": "POST",
    "/api/undo/{change_id}": "POST",
    "/api/usage": "GET",
    "/api/version": "GET",
    "/api/version/check": "POST",
    "/api/version/install": "GET",
    "/api/version/latest": "GET",
    "/api/version/notes": "GET",
    "/api/videos": "GET",
    "/api/videos/generate": "POST",
    "/api/videos/sound/status": "GET",
    "/api/videos/status": "GET",
    "/api/videos/stop": "POST",
    "/api/videos/trash/empty": "POST",
    "/api/videos/trash/{name}": "GET",
    "/api/videos/{name}": "GET",
    "/api/videos/{vid}/fav": "POST",
    "/api/videos/{vid}/purge": "POST",
    "/api/videos/{vid}/restore": "POST",
    "/api/videos/{vid}/sound": "POST",
    "/api/videos/{vid}/trash": "POST",
    "/api/voice": "GET",
    "/api/voice/duplex": "POST",
    "/api/voice/duplex/commit": "POST",
    "/api/voice/duplex/end": "POST",
    "/api/voice/duplex/warm": "POST",
    "/api/voice/live": "POST",
    "/api/voice/setup": "POST",
    "/api/voice/speak": "POST",
    "/api/voice/stop": "POST",
    "/api/voice/transcribe": "POST",
    "/api/web-images/{name}": "GET",
}

_lock = threading.RLock()
_fails: dict = {}                     # ip -> [failure times]
_server = {"uvicorn": None, "thread": None, "bind": None}


# ---- settings ------------------------------------------------------------------------------------------------
def clean_settings(s: dict) -> dict:
    port = s.get("remote_port", DEFAULT_PORT)
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    return {"remote_access": s.get("remote_access") if s.get("remote_access") in MODES else "off",
            "remote_port": port if 1024 <= port <= 65535 and port != 8000 else DEFAULT_PORT,
            "remote_allow_control": s.get("remote_allow_control") is True}


# ---- addresses -----------------------------------------------------------------------------------------------
def allowed_address(ip: str) -> bool:
    """A Tailscale or private home-network IPv4 address - the only kinds this ever listens on."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if a.version != 4 or a.is_loopback or a.is_unspecified or a.is_link_local or a.is_multicast:
        return False
    return a in TAILSCALE_NET or a.is_private


def addresses() -> dict:
    """This PC's Tailscale address and home-network addresses."""
    out = {"tailscale": None, "lan": []}
    try:
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if getattr(a.family, "name", "") != "AF_INET" or not allowed_address(a.address):
                    continue
                ip = ipaddress.ip_address(a.address)
                if ip in TAILSCALE_NET:
                    out["tailscale"] = out["tailscale"] or a.address
                elif not name.lower().startswith(("vethernet", "vmware", "virtualbox")):
                    out["lan"].append(a.address)
    except Exception:
        pass
    return out


def bind_address(mode: str, found: dict | None = None) -> str | None:
    found = addresses() if found is None else found
    if mode == "tailscale":
        return found["tailscale"]
    if mode == "lan":
        return (found["lan"] or [None])[0]
    return None


# ---- the password and sessions (data/remote.json) ------------------------------------------------------------
def _load() -> dict:
    try:
        with open(DATA, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, DATA)


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


def has_password() -> bool:
    return bool(_load().get("password"))


def set_password(password: str) -> dict:
    password = str(password or "")
    if len(password) < MIN_PASSWORD:
        return {"ok": False, "message": f"Use at least {MIN_PASSWORD} characters."}
    salt = secrets.token_bytes(16)
    with _lock:
        d = _load()
        d["password"] = {"salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(_hash(password, salt)).decode()}
        d["sessions"] = {}                         # a new password signs every device out
        _save(d)
    return {"ok": True, "message": "Password set. Devices that were signed in have to sign in again."}


def _check(password: str) -> bool:
    p = _load().get("password") or {}
    try:
        salt, want = base64.b64decode(p["salt"]), base64.b64decode(p["hash"])
    except (KeyError, ValueError, TypeError):
        return False
    return hmac.compare_digest(_hash(str(password or ""), salt), want)


def locked_for(ip: str, now: float | None = None) -> int:
    now = time.time() if now is None else now
    with _lock:
        recent = [t for t in _fails.get(ip, []) if now - t < LOCK_SECONDS]
        _fails[ip] = recent
        return int(LOCK_SECONDS - (now - recent[-MAX_TRIES])) if len(recent) >= MAX_TRIES else 0


def login(ip: str, password: str, device: str = "") -> dict:
    wait = locked_for(ip)
    if wait:
        return {"ok": False, "message": f"Too many wrong passwords - try again in {max(1, wait // 60)} min."}
    if not has_password() or not _check(password):
        with _lock:
            _fails.setdefault(ip, []).append(time.time())
        import events as event_log
        event_log.log_action(f"Phone access: a wrong password from {ip}.")
        return {"ok": False, "message": "Wrong password."}
    token = secrets.token_urlsafe(32)
    with _lock:
        _fails.pop(ip, None)
        d = _load()
        d.setdefault("sessions", {})[hashlib.sha256(token.encode()).hexdigest()] = {
            "device": (device or "A device")[:80], "ip": ip, "created": time.time(), "seen": time.time()}
        _save(d)
    import events as event_log
    event_log.log_action(f"Phone access: {(device or 'a device')[:40]} signed in from {ip}.")
    return {"ok": True, "token": token}


def session(token: str | None) -> dict | None:
    if not token:
        return None
    key = hashlib.sha256(token.encode()).hexdigest()
    with _lock:
        d = _load()
        s = (d.get("sessions") or {}).get(key)
        if not s or time.time() - s.get("created", 0) > SESSION_DAYS * 86400:
            return None
        if time.time() - s.get("seen", 0) > 60:          # written at most once a minute
            s["seen"] = time.time()
            _save(d)
        return {**s, "id": key[:12]}


def devices() -> list:
    rows = [{"id": k[:12], **{f: v.get(f) for f in ("device", "ip", "created", "seen")}}
            for k, v in (_load().get("sessions") or {}).items()]
    return sorted(rows, key=lambda r: -(r["seen"] or 0))


def sign_out(device_id: str | None = None) -> int:
    with _lock:
        d = _load()
        sessions = d.get("sessions") or {}
        gone = [k for k in sessions if device_id is None or k.startswith(device_id)]
        for k in gone:
            sessions.pop(k)
        d["sessions"] = sessions
        _save(d)
    return len(gone)


# ---- requests --------------------------------------------------------------------------------------------------
def is_remote(scope: dict) -> bool:
    """The request came in on the phone listener (not the PC's 127.0.0.1:8000)."""
    server = scope.get("server") or ("", 0)
    bind = _server["bind"]
    return bool(bind) and tuple(server[:2]) == tuple(bind)


def _template_re(template: str):
    import re
    parts = re.split(r"(\{[^}]+\})", template)
    return re.compile("^" + "".join((".+" if p.endswith(":path}") else "[^/]+") if p.startswith("{") else re.escape(p)
                                    for p in parts) + "$")


_ALLOWED_RES = [(_template_re(p), set(m.split())) for p, m in PHONE_ALLOWED.items()]


def phone_allowed(method: str, path: str) -> bool:
    """An /api/ call the phone may make (PHONE_ALLOWED). HEAD counts as GET."""
    method = "GET" if method == "HEAD" else method
    return any(method in methods and rx.match(path) for rx, methods in _ALLOWED_RES)


def blocked(method: str, path: str) -> bool:
    if any((m == "*" or m == method) and (path == p or path.startswith(p + "/")) for m, p in PHONE_BLOCKED):
        return True
    # (1.4.8) Under /api/, only what PHONE_ALLOWED lists; pages and files outside /api/ are as before.
    return path.startswith("/api/") and not phone_allowed(method, path)


def allowed_hosts() -> list:
    return [_server["bind"][0]] if _server["bind"] else []


def control_allowed() -> bool:
    return (shared.app_state.get("settings") or {}).get("remote_allow_control") is True


# ---- the listener ------------------------------------------------------------------------------------------------
def running() -> bool:
    return bool(_server["thread"] and _server["thread"].is_alive())


def stop() -> None:
    with _lock:
        srv, th = _server["uvicorn"], _server["thread"]
        _server.update(uvicorn=None, thread=None, bind=None)
    if srv is not None:
        srv.should_exit = True
    if th is not None:
        th.join(timeout=8)


def sync(settings: dict | None = None) -> dict:
    """Start, move or stop the phone listener to match the settings. It never starts without a password."""
    s = clean_settings(settings or shared.app_state.get("settings") or {})
    want = None
    if s["remote_access"] != "off" and has_password():
        ip = bind_address(s["remote_access"])
        want = (ip, s["remote_port"]) if ip and allowed_address(ip) else None
    if want and _server.get("asked") == want and running():
        return status()                  # already up on the port it moved to
    if want == _server["bind"] and (want is None or running()):
        return status()
    stop()
    _server["moved"] = ""
    if want:
        import ports
        import uvicorn
        from api.server import app
        asked = want
        # Another program on the phone port (1.4.7): the listener died silently and the page still showed its
        # address. Now it takes the next free port and says so; the address shown is the one it really listens on.
        if not ports.is_free(want[1], want[0]):
            held = ports.describe(ports.holder(want[1]))
            want = (want[0], ports.pick(want[1] + 1, want[0]))
            _server["moved"] = f"Port {asked[1]} is used by {held}, so phone access uses {want[1]}."
        _server["asked"] = asked
        # lifespan off: the app's startup (background services) already ran on 127.0.0.1:8000.
        srv = uvicorn.Server(uvicorn.Config(app, host=want[0], port=want[1], log_level="warning", lifespan="off",
                                            timeout_graceful_shutdown=3))
        srv.install_signal_handlers = lambda: None     # not the main thread
        th = threading.Thread(target=srv.run, daemon=True, name="remote-access")
        with _lock:
            _server.update(uvicorn=srv, thread=th, bind=want)
        th.start()
    return status()


def status() -> dict:
    s = clean_settings(shared.app_state.get("settings") or {})
    found = addresses()
    bind = _server["bind"]
    why = None
    if s["remote_access"] != "off":
        if not has_password():
            why = "Set a password first - phone access never starts without one."
        elif not bind_address(s["remote_access"], found):
            why = ("Tailscale isn't running on this PC (no 100.x address). Install it from tailscale.com, sign in on "
                   "the PC and the phone, then come back." if s["remote_access"] == "tailscale"
                   else "No home-network address found - is the PC connected to your router?")
        elif bind and not running():
            why = f"Phone access couldn't listen on port {bind[1]} - another program may hold it. Turn it off and on again."
    return {**s, "running": running(), "url": f"http://{bind[0]}:{bind[1]}/" if bind and running() else None,
            "addresses": found, "has_password": has_password(), "devices": devices(), "problem": why,
            **({"note": _server["moved"]} if _server.get("moved") else {})}


def qr_svg(text: str) -> str:
    import segno
    import io
    buf = io.BytesIO()
    segno.make(text, error="m").save(buf, kind="svg", scale=6, border=2, dark="#111", light="#fff")
    return buf.getvalue().decode()


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>FX Workspace - sign in</title>
<link rel="manifest" href="/remote/manifest.webmanifest"><meta name="theme-color" content="#1f1e1c">
<meta name="apple-mobile-web-app-capable" content="yes"><link rel="apple-touch-icon" href="/remote/icon.svg">
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#1f1e1c;color:#ece9e2;font:16px system-ui,sans-serif}
form{width:min(340px,90vw);display:grid;gap:12px}h1{font-size:22px;margin:0 0 4px}p{color:#a8a397;margin:0;font-size:14px}
input,button{height:46px;border-radius:12px;border:1px solid #3a3833;background:#2a2926;color:inherit;font:inherit;padding:0 14px}
button{background:#ffc24a;color:#1d1606;border:0;font-weight:600}.err{color:#e7786e;min-height:20px}</style></head>
<body><form id="f"><h1>FX Workspace</h1><p>Your PC's assistant. Enter the password set in Settings &rsaquo; Phone access.</p>
<input id="pw" type="password" autocomplete="current-password" placeholder="Password" required autofocus>
<button>Sign in</button><div class="err" id="e"></div></form>
<script>document.getElementById('f').onsubmit=async e=>{e.preventDefault();const r=await fetch('/remote/login',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({password:document.getElementById('pw').value,
device:navigator.userAgent.match(/\\(([^)]+)\\)/)?.[1]||'A phone'})});const d=await r.json();
if(d.ok)location.href='/';else document.getElementById('e').textContent=d.message||'Not signed in.';};</script></body></html>"""

MANIFEST = {"name": "FX Workspace", "short_name": "FX", "start_url": "/", "display": "standalone",
            "background_color": "#1f1e1c", "theme_color": "#1f1e1c",
            "icons": [{"src": "/remote/icon.svg", "sizes": "any", "type": "image/svg+xml"}]}
ICON = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='7' fill='#1f1e1c'/>"
        "<circle cx='16' cy='16' r='6' fill='#ffc24a'/><ellipse cx='16' cy='16' rx='13' ry='6.5' fill='none' stroke='#ffc24a' "
        "stroke-width='2.2' transform='rotate(-28 16 16)'/></svg>")
