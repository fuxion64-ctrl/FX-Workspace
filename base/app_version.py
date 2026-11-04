"""
app_version.py
==============
The app's version (the VERSION file - the one number the installer, the desktop app and Settings > About use)
and the update check: Settings > About > Check for updates asks GitHub for the newest release and, when it is
newer, links its installer. Updating = run the new installer over this one: it keeps chats, settings, keys,
models and the Python environment, and replaces only the app's own files (installer/FXWorkspace.iss).

Checked by itself (1.4.6, "Check for updates by itself", on by default): 90 s after start and every 12 hours, so Home
can offer the update; or when the user presses the button. It goes through netlayer like every other request.

Update now (1.4.2): the app downloads that installer itself into data\\updates - only from this repo's own
release downloads, and only when it matches the SHA-256 GitHub publishes for it - then starts it silently
(/FXRESTART=1). The installer closes the app, updates it, tidies up (install.ps1: packages and engine builds the
new version no longer uses) and opens it again. The downloaded installer is deleted on the next start.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "fuxion64-ctrl/FX-Workspace"
RELEASES_URL = f"https://github.com/{REPO}/releases"
DOWNLOADS = f"https://github.com/{REPO}/releases/download/"
UPDATES_DIR = os.path.join(ROOT, "data", "updates")
_cache: dict = {"at": 0.0, "result": None}
_lock = threading.Lock()
_job: dict = {"state": "idle"}      # idle | downloading | verifying | starting | started | error


def current() -> str:
    try:
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as f:
            return f.read().strip() or "0.0.0"
    except OSError:
        return "0.0.0"


def notes(version: str | None = None) -> dict:
    """The release notes shipped with a version (docs/releases/<version>.md) - shown once after an update (1.4.3).
    {"version", "markdown"} with the title line dropped, or "" when this version has none."""
    v = version or current()
    if not re.fullmatch(r"\d+(\.\d+){1,3}", v or ""):
        return {"version": v, "markdown": ""}
    try:
        with open(os.path.join(ROOT, "docs", "releases", f"{v}.md"), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return {"version": v, "markdown": ""}
    return {"version": v, "markdown": re.sub(r"\A#\s[^\n]*\n+", "", text).strip()}


def _parts(v: str) -> tuple:
    """'v1.4.0-beta' -> (1, 4, 0). Missing parts count as 0."""
    nums = [int(n) for n in re.findall(r"\d+", str(v or ""))[:3]]
    return tuple(nums + [0] * (3 - len(nums)))


def newer(latest: str, mine: str) -> bool:
    return _parts(latest) > _parts(mine)


def check(force: bool = False) -> dict:
    """{"current", "latest", "newer", "url", "installer", "notes", "published"} - or {"current", "error"}."""
    mine = current()
    if not force and _cache["result"] and time.time() - _cache["at"] < 600:
        return {**_cache["result"], "current": mine}
    import netlayer
    try:
        with netlayer.client("other", timeout=15.0, headers={"Accept": "application/vnd.github+json",
                                                              "User-Agent": "fx-workspace-update-check"}) as c:
            r = c.get(f"https://api.github.com/repos/{REPO}/releases/latest")
        if r.status_code == 404:
            return {"current": mine, "error": "No release has been published yet."}
        r.raise_for_status()
        rel = r.json()
    except Exception as e:                                   # offline, blocked by the network settings, rate limit
        return {"current": mine, "error": f"Couldn't reach GitHub ({str(e)[:160]})."}
    latest = str(rel.get("tag_name") or rel.get("name") or "").lstrip("vV")
    asset = next((a for a in rel.get("assets") or []
                  if str(a.get("name", "")).lower().endswith(".exe") and "setup" in str(a.get("name", "")).lower()), {})
    digest = str(asset.get("digest") or "")
    result = {"latest": latest, "newer": newer(latest, mine), "url": rel.get("html_url") or RELEASES_URL,
              "installer": asset.get("browser_download_url"), "notes": str(rel.get("body") or "")[:4000],
              "published": rel.get("published_at"), "size": int(asset.get("size") or 0),
              "sha256": digest[7:].lower() if digest.lower().startswith("sha256:") else None}
    result["can_install"] = bool(result["newer"] and result["sha256"] and _ours(result["installer"]) and os.name == "nt")
    _cache.update(at=time.time(), result=result)
    return {**result, "current": mine}


def known() -> dict:
    """The last check's answer without asking GitHub again - what Home's update banner reads."""
    return {**(_cache["result"] or {}), "current": current(), "checked": _cache["at"] or None}


AUTO_FIRST_S, AUTO_EVERY_S = 90, 12 * 3600


def auto_round(enabled) -> bool:
    """One round of the check by itself: asks GitHub only while enabled() is true. True when it asked."""
    try:
        if enabled():
            check(force=True)
            return True
    except Exception:
        pass                                            # offline or blocked: the next round tries again
    return False


def start_auto(enabled) -> threading.Thread:
    """Checks by itself while enabled() is true: first AUTO_FIRST_S after start, then every AUTO_EVERY_S."""
    def loop():
        time.sleep(AUTO_FIRST_S)
        while True:
            auto_round(enabled)
            time.sleep(AUTO_EVERY_S)
    t = threading.Thread(target=loop, name="update-check", daemon=True)
    t.start()
    return t


def _ours(url) -> bool:
    """Only this repo's own release downloads are ever run (no redirect target, mirror or other repo)."""
    return isinstance(url, str) and url.startswith(DOWNLOADS) and url.lower().endswith(".exe") and ".." not in url


# ---- Update now ------------------------------------------------------------------------------------------
def progress() -> dict:
    with _lock:
        return dict(_job)


def _set(**fields):
    with _lock:
        _job.update(fields)


def install(force_check: bool = True) -> dict:
    """Start the download in the background. Returns the progress right away."""
    with _lock:
        if _job.get("state") in ("downloading", "verifying", "starting", "started"):
            return dict(_job)
        _job.clear()
        _job.update(state="downloading", done_mb=0.0, total_mb=0.0)
    info = check(force=force_check)
    if not info.get("can_install"):
        why = (info.get("error") or ("You're up to date." if not info.get("newer") else
               "This release has no installer with a published checksum - download it from the releases page."))
        _set(state="error", error=why)
        return progress()
    threading.Thread(target=_run, args=(info,), name="app-update", daemon=True).start()
    return progress()


def _run(info: dict):
    import netlayer
    try:
        os.makedirs(UPDATES_DIR, exist_ok=True)
        name = os.path.basename(info["installer"])
        if not re.fullmatch(r"[\w.\-]+\.exe", name):
            raise RuntimeError("The installer's name looks wrong - not running it.")
        dest, part = os.path.join(UPDATES_DIR, name), os.path.join(UPDATES_DIR, name + ".part")
        h = hashlib.sha256()
        total = info.get("size") or 0
        _set(total_mb=round(total / 1048576, 1))
        with netlayer.client("other", timeout=60.0, follow_redirects=True,
                             headers={"User-Agent": "fx-workspace-updater"}) as c, c.stream("GET", info["installer"]) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"GitHub answered {r.status_code}.")
            got = 0
            with open(part, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    got += len(chunk)
                    _set(done_mb=round(got / 1048576, 1))
        _set(state="verifying")
        if h.hexdigest() != info["sha256"] or (total and got != total):
            os.remove(part)
            raise RuntimeError("The download didn't match the checksum GitHub published for it, so it was deleted.")
        os.replace(part, dest)
        _set(state="starting")
        # Started through "cmd /c start", so it isn't this app's child: the installer closes FX Workspace before
        # it replaces anything, and a forced close (taskkill /T on the app's processes) mustn't take it along.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(["cmd.exe", "/c", "start", "", dest, "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-",
                          "/FXRESTART=1"], cwd=UPDATES_DIR, creationflags=flags, close_fds=True)
        _set(state="started", version=info.get("latest"))
    except Exception as e:
        _set(state="error", error=str(e)[:300])


def clean_downloads() -> int:
    """At start: installers downloaded by Update now are spent once this version runs. Returns how many went."""
    gone = 0
    for p in glob.glob(os.path.join(UPDATES_DIR, "*.exe")) + glob.glob(os.path.join(UPDATES_DIR, "*.part")):
        try:
            os.remove(p)
            gone += 1
        except OSError:
            pass                                   # still running (the installer itself) - next time
    return gone
