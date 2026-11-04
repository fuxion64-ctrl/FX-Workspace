"""
llama_builds.py - which llama.cpp build runs the local models (ARCHITECTURE D51).

Two builds sit side by side:
  latest  stock llama.cpp in local\\llama.cpp (paths.py; older installs: E:\\ClankerLocalCpp, C:\\llama.cpp);
  prism   PrismML's fork in local\\llama.cpp\\prism - the only build that loads Ternary Bonsai 2
          (PTQ1_0 / PQ2_0), but it trails upstream: its October 2026 release still sits on upstream's
          code from August 25, about 900 commits behind.

The router starts every model with its own executable, so the build is chosen per router. run.bat and
launcher.pyw start the latest build. With the "llama_build" setting on "auto" (the default), a model that
only PrismML loads makes the app restart the router on PrismML's build - the same command line, so
models.ini, host, port and flags stay as they were - and picking a model the latest build runs moves it
back. "latest" and "prism" keep one build. Only the model the user picks moves the router back: the app's
own helper calls (titles, plans) use the model already loaded, and they never start a switch.

A switch kills the router and the model servers it started (taskkill /T), like closing its window. The new
router starts the way the old one did: in its own minimized console window under run.bat, or windowless
with data/logs/llama-server.log under launcher.pyw (AGENTIC_WINDOWLESS=1), and then the app stops it on
exit just as the launcher would have stopped its own.
"""
from __future__ import annotations

import local_runtime
import ntpath
import os
import re
import subprocess
import threading
import time

import llama_local
import paths

CHOICES = ("auto", "latest", "prism")
PRISM_DIRS = paths.prism_dirs()
PRISM_DIR = PRISM_DIRS[0]
LATEST_DIRS = paths.llama_dirs()
NAMES = {"latest": "the newest llama.cpp", "prism": "PrismML's llama.cpp"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
READY_TIMEOUT = 60.0
_RUN_TTL = 20.0

_lock = threading.RLock()
_run_cache = {"at": 0.0, "run": None}
_started = {"proc": None}            # the windowless router this app started (launcher.pyw), stopped on exit
_versions: dict = {}


def _exists(path: str) -> bool:
    return os.path.exists(path)


def exe_of(kind: str) -> str | None:
    """The llama-server.exe of a build, or None when it isn't installed."""
    dirs = PRISM_DIRS if kind == "prism" else LATEST_DIRS if kind == "latest" else ()
    for d in dirs:
        exe = ntpath.join(d, "llama-server.exe")
        if _exists(exe):
            return exe
    return None


def kind_of(exe: str | None) -> str | None:
    """Which build an executable belongs to: "prism" when it sits in a folder named for PrismML's fork."""
    if not exe:
        return None
    folder = ntpath.basename(ntpath.dirname(str(exe).replace("/", "\\"))).lower()
    return "prism" if "prism" in folder else "latest"


def mode() -> str:
    try:
        import core.state as shared
        value = (shared.app_state.get("settings") or {}).get("llama_build")
    except Exception:
        value = None
    return value if value in CHOICES else "auto"


def wanted(model_name: str, how: str | None = None) -> str | None:
    """The build this model should run on, or None when that build isn't installed (nothing to switch to)."""
    how = how if how in CHOICES else mode()
    if how == "auto":
        kind = "prism" if llama_local.needs_prism(model_name) else "latest"
    else:
        kind = how
    return kind if exe_of(kind) else None


# ---- the running router ------------------------------------------------------------------------------
def _port_of(cmdline: list) -> int:
    for flag in ("--port",):
        if flag in cmdline and cmdline.index(flag) + 1 < len(cmdline):
            try:
                return int(cmdline[cmdline.index(flag) + 1])
            except ValueError:
                pass
    return 8080                       # llama-server's own default


def _router_at(psutil, pid) -> dict | None:
    """{"pid", "exe", "cmdline"} when process `pid` is the router on our port, else None."""
    try:
        p = psutil.Process(pid)
        if not p.name().lower().startswith("llama-server"):
            return None
        cmd = list(p.cmdline() or [])
        if "--models-preset" not in cmd or _port_of(cmd) != llama_local.LLAMA_PORT:
            return None
        try:
            exe = p.exe()
        except Exception:
            exe = None
        return {"pid": pid, "exe": exe or (cmd[0] if cmd else None), "cmdline": cmd}
    except Exception:
        return None


def _find_router() -> dict | None:
    """The router process on our port: {"pid", "exe", "cmdline"} (psutil), or None.
    The process we found last, else the one listening on the port - a few ms. Reading every process's command line
    took 3-8 s on a PC with ~450 processes, and the Labs page and every reply asked for it (2026-10-27)."""
    try:
        import psutil
    except ImportError:
        return None
    last = (_run_cache.get("run") or {}).get("pid")
    if last:
        found = _router_at(psutil, last)
        if found:
            return found
    try:
        for c in psutil.net_connections("tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == llama_local.LLAMA_PORT and c.pid:
                found = _router_at(psutil, c.pid)
                if found:
                    return found
    except Exception:
        pass
    for p in psutil.process_iter(["pid", "name"]):           # last resort: by name, command lines of those only
        if str(p.info.get("name") or "").lower().startswith("llama-server"):
            found = _router_at(psutil, p.info["pid"])
            if found:
                return found
    return None


_find_lock = threading.Lock()


def running(force: bool = False) -> dict:
    """{"kind", "exe", "pid", "cmdline"} of the router now serving; kind None when unknown.
    One lookup at a time: callers that arrive while one runs take its answer."""
    if not force and _run_cache["run"] is not None and time.time() - _run_cache["at"] < _RUN_TTL:
        return dict(_run_cache["run"])
    arrived = _run_cache.get("starts", 0)           # lookups started so far (one at a time, under _find_lock)
    with _find_lock:
        if _run_cache["run"] is not None and _run_cache.get("done_start", 0) > arrived:
            return dict(_run_cache["run"])          # one that started after this call arrived finished meanwhile
        start = _run_cache["starts"] = _run_cache.get("starts", 0) + 1
        found = _find_router() or {}
        exe = found.get("exe") or llama_local.server_executable()
        run = {"kind": kind_of(exe), "exe": exe, "pid": found.get("pid"), "cmdline": found.get("cmdline") or []}
        _run_cache.update(at=time.time(), run=run, done_start=start)
        return dict(run)


def forget():
    _run_cache.update(at=0.0, run=None)


# ---- switching ---------------------------------------------------------------------------------------
@local_runtime.exclusive
def ensure_for(model_name: str, back: bool = True, on_switch=None) -> str | None:
    """Make the router run on the build model_name needs. back=False only ever moves TO the build a model
    requires (PrismML for Ternary Bonsai), never back - for calls that aren't the user's own pick.
    on_switch(kind) is told before a switch starts. Returns the build switched to, or None.
    Raises RuntimeError when the new build doesn't come up (the old one is started again first)."""
    if not model_name:
        return None
    with _lock:
        want = wanted(model_name)
        if want is None:
            return None
        run = running()
        if run["kind"] is None or run["kind"] == want:
            return None
        if want == "latest" and not back:
            return None               # PrismML runs this model too; only the user's pick moves the router back
        if on_switch:
            try:
                on_switch(want)
            except Exception:
                pass
        switch(want, run)
        return want


def _args_like(run: dict) -> list:
    """The old router's own arguments when they are known, else run.bat's."""
    cmd = list(run.get("cmdline") or [])
    if len(cmd) > 1 and "--models-preset" in cmd:
        return cmd[1:]
    return ["--host", llama_local.LLAMA_HOST, "--port", str(llama_local.LLAMA_PORT),
            "--models-preset", os.environ.get("MODELS_INI") or llama_local._ini_path(),
            "--models-max", "1", "--jinja", "--parallel", "1"]


def _stop(pid: int | None) -> None:
    if pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=20,
                       creationflags=NO_WINDOW)
    proc = _started["proc"]
    if proc is not None and proc.poll() is None and (pid is None or proc.pid == pid):
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=20,
                       creationflags=NO_WINDOW)
    deadline = time.time() + 15
    while time.time() < deadline and _answers():
        time.sleep(0.3)


def _answers() -> bool:
    """Whether llama-server answers right now (never the cached answer)."""
    llama_local.invalidate_cache()
    return llama_local.server_models(force=True) is not None


def _log(exe: str):
    logs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs")
    os.makedirs(logs, exist_ok=True)
    f = open(os.path.join(logs, "llama-server.log"), "a", encoding="utf-8", errors="replace", buffering=1)
    f.write(f"\n---- {time.strftime('%Y-%m-%d %H:%M:%S')} - the app restarted llama-server on {exe} ----\n")
    return f


def _start(exe: str, args: list) -> None:
    cwd = ntpath.dirname(exe) if os.name == "nt" else None
    if os.environ.get("AGENTIC_WINDOWLESS") == "1":
        log = _log(exe)
        try:
            _started["proc"] = subprocess.Popen([exe, *args], cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
                                                stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        finally:
            log.close()                # the router has its own handle now
    else:
        # As run.bat does: its own minimized console window, which outlives the app like run.bat's did.
        subprocess.Popen(["cmd", "/c", "start", "llama-server (router)", "/min", exe, *args], cwd=cwd,
                         stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)


def _wait_ready(timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _answers():
            return True
        proc = _started["proc"]
        if proc is not None and proc.poll() is not None and os.environ.get("AGENTIC_WINDOWLESS") == "1":
            return False
        time.sleep(0.5)
    return False


@local_runtime.exclusive
def switch(kind: str, run: dict | None = None) -> None:
    """Restart the router on `kind`'s build. Raises RuntimeError when it doesn't come up."""
    import events as event_log
    with _lock:
        exe = exe_of(kind)
        if not exe:
            raise RuntimeError(f"{NAMES.get(kind, kind)} isn't installed.")
        run = run or running(force=True)
        ours = _started["proc"] is not None and _started["proc"].poll() is None
        if not run.get("pid") and not ours and _answers():
            raise RuntimeError(f"Couldn't find llama-server's process to restart it on {NAMES.get(kind, kind)}. "
                               "Close llama-server's window and start the app again.")
        args = _args_like(run)
        started = time.time()
        event_log.log_action(f"Restarting llama-server on {NAMES[kind]} ({exe}).")
        _stop(run.get("pid"))
        _start(exe, args)
        ok = _wait_ready(READY_TIMEOUT)
        forget()
        llama_local.invalidate_cache()
        if ok:
            event_log.log_action(f"llama-server is running on {NAMES[kind]} ({time.time() - started:.1f}s).")
            return
        old = run.get("exe")
        if old and old != exe and _exists(old):
            event_log.log_action(f"{NAMES[kind]} didn't start - starting {old} again.")
            stuck = _find_router()
            _stop(stuck.get("pid") if stuck else None)
            _start(old, args)
            _wait_ready(READY_TIMEOUT)
            forget()
        raise RuntimeError(f"llama-server didn't start on {NAMES[kind]} ({exe}). "
                           "See data/logs/llama-server.log, or set Settings › Models › llama.cpp build to the other build.")


# ---- keeping the router up (1.4.7) ---------------------------------------------------------------------------
# When llama-server died (a crash in a model load, closed by hand, killed by Windows for memory) or stopped answering,
# nothing started it again: every later message said "Couldn't reach the local server. Is run.bat still running?"
# until the whole app was restarted. Now a reply that finds it down starts it first (models.call_local), and a
# watchdog notices on its own - a dead process at once, a hung one after HUNG_CHECKS probes in a row.
RESTART_LIMIT = 3                     # restarts per RESTART_WINDOW before it is left down (a crash loop)
RESTART_WINDOW = 600.0
WATCH_EVERY = 10.0
HUNG_CHECKS = 3
_watch = {"thread": None, "restarts": [], "last": "", "seen_up": False, "fails": 0, "stop": threading.Event()}


def adopt(proc) -> None:
    """launcher.pyw's own router: stopped with the app and watched like one this module started."""
    if proc is not None:
        _started["proc"] = proc


def _autostart() -> bool:
    return os.environ.get("AUTOSTART_LLAMA", "1") == "1"


def _restart_exe() -> str | None:
    """The build to start: the one that ran last, else the newest llama.cpp, else PrismML's."""
    last = (_run_cache.get("run") or {}).get("exe") or llama_local.server_executable()
    if last and _exists(last):
        return last
    return exe_of("latest") or exe_of("prism")


def _too_many() -> bool:
    now = time.time()
    _watch["restarts"] = [t for t in _watch["restarts"] if now - t < RESTART_WINDOW]
    return len(_watch["restarts"]) >= RESTART_LIMIT


def status_line() -> str:
    """What the last recovery did, for Diagnostics and the error text."""
    return _watch["last"]


@local_runtime.exclusive
def ensure_running(reason: str = "", wait: float = READY_TIMEOUT) -> dict:
    """Make sure llama-server answers; start it again when it doesn't. {"ok", "restarted", "message"}.
    A hung router of ours is stopped first; a port another program holds moves the router to a free one."""
    import ports
    import events as event_log
    with _lock:
        if _answers():
            _watch["seen_up"] = True
            return {"ok": True, "restarted": False, "message": ""}
        if not _autostart():
            return {"ok": False, "restarted": False, "message": "llama-server isn't started by the app here (AUTOSTART_LLAMA=0)."}
        exe = _restart_exe()
        if not exe:
            return {"ok": False, "restarted": False,
                    "message": "llama-server isn't installed (local\\llama.cpp) - install it from Settings › Models, or use Cloud."}
        if _too_many():
            msg = (f"llama-server stopped {RESTART_LIMIT} times in {int(RESTART_WINDOW / 60)} minutes, so the app left it "
                   "off. The log says why: data\\logs\\llama-server.log. Restart the app to try again.")
            _watch["last"] = msg
            return {"ok": False, "restarted": False, "message": msg}
        _watch["restarts"].append(time.time())
        port = llama_local.LLAMA_PORT
        info = ports.holder(port)
        moved = ""
        if info and (_router_at(_psutil(), info["pid"]) or ports.is_ours(info)):
            event_log.log_action(f"llama-server (process {info['pid']}) holds port {port} but doesn't answer - stopping it.")
            ports.stop_tree(info["pid"], wait_port=port)
        elif info or not ports.is_free(port):
            new = ports.pick(port + 1)
            moved = f" Port {port} is used by {ports.describe(info)}, so it now runs on {new}."
            llama_local.set_port(new)
            ports.note("llama", new, "moved", ports.describe(info))
        proc = _started["proc"]
        if proc is not None and proc.poll() is None:
            _stop(proc.pid)                  # ours, but not answering
        event_log.log_action(f"llama-server isn't answering{f' ({reason})' if reason else ''} - starting it again.{moved}")
        forget()
        _start(exe, _args_like({}))
        ok = _wait_ready(wait)
        forget()
        llama_local.invalidate_cache()
        if ok:
            _watch["seen_up"] = True
            _watch["last"] = f"llama-server had stopped; the app started it again at {time.strftime('%H:%M')}.{moved}"
            event_log.log_action(_watch["last"])
            return {"ok": True, "restarted": True, "message": _watch["last"]}
        _watch["last"] = f"llama-server didn't start again ({exe}). The log says why: data\\logs\\llama-server.log."
        event_log.log_action(_watch["last"])
        return {"ok": False, "restarted": True, "message": _watch["last"]}


def _psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def _hung_or_dead() -> bool:
    """Down right now: our process exited, or nothing answers on the port (probed fresh)."""
    proc = _started["proc"]
    if proc is not None and proc.poll() is not None:
        return True
    return not _answers()


def _watchdog() -> None:
    while not _watch["stop"].wait(WATCH_EVERY):
        try:
            if not _autostart() or not _watch["seen_up"] or switching():
                if not _watch["seen_up"] and _answers():
                    _watch["seen_up"] = True
                continue
            if local_runtime.busy():
                # A reply or a load runs on the router: a slow answer then is the GPU at work, not a hang - a dead
                # process still counts.
                proc = _started["proc"]
                if not (proc is not None and proc.poll() is not None):
                    _watch["fails"] = 0
                    continue
            if _hung_or_dead():
                _watch["fails"] += 1
                proc = _started["proc"]
                dead = proc is not None and proc.poll() is not None
                if dead or _watch["fails"] >= HUNG_CHECKS:
                    _watch["fails"] = 0
                    ensure_running("watchdog")
            else:
                _watch["fails"] = 0
        except Exception:
            pass


def switching() -> bool:
    """A build switch holds the router lock (it stops and starts the router itself)."""
    got = _lock.acquire(blocking=False)
    if got:
        _lock.release()
    return not got


def start_watchdog() -> None:
    if os.environ.get("AGENTIC_DISABLE_LLAMA_WATCHDOG") == "1":
        return
    t = _watch["thread"]
    if t is None or not t.is_alive():
        _watch["stop"].clear()
        _watch["thread"] = threading.Thread(target=_watchdog, daemon=True, name="llama-watchdog")
        _watch["thread"].start()


def stop_started() -> None:
    """On app exit: the windowless router this app started stops with it (as launcher.pyw's own would)."""
    _watch["stop"].set()
    proc = _started["proc"]
    if proc is not None and proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=15,
                           creationflags=NO_WINDOW)
        except Exception:
            pass
    _started["proc"] = None


# ---- what Settings shows -----------------------------------------------------------------------------
def version(exe: str | None) -> str:
    """"b11541" from `llama-server --version`, cached per executable; "" when unknown."""
    if not exe:
        return ""
    if exe in _versions:
        return _versions[exe]
    out = ""
    try:
        res = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
        text = (res.stdout or "") + (res.stderr or "")
        m = re.search(r"build (\d+)", text) or re.search(r"version: (\d+)", text)
        out = f"b{m.group(1)}" if m else ""
    except Exception:
        out = ""
    _versions[exe] = out
    return out


def status() -> dict:
    """For Settings › Models: the setting, each build (installed, path, version) and the one running."""
    run = running()
    builds = {}
    for kind in ("latest", "prism"):
        exe = exe_of(kind)
        builds[kind] = {"installed": bool(exe), "exe": exe, "version": version(exe), "name": NAMES[kind]}
    return {"mode": mode(), "running": run["kind"], "running_exe": run["exe"], "builds": builds}


@local_runtime.exclusive
def apply(model_name: str = "") -> str | None:
    """Right after the setting changes: move the router now (under "auto" the model in use decides).
    Returns the build switched to, or None when it already runs the right one (or can't be told)."""
    with _lock:
        want = wanted(model_name or "")
        run = running(force=True)
        if want is None or run["kind"] is None or run["kind"] == want:
            return None
        switch(want, run)
        return want
