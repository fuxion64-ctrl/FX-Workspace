"""
launcher.pyw
============
Agentic Workspace without console windows. `Agentic Workspace.vbs` (or run.bat, which hands over to this
file) runs it with the project's pythonw.exe, so no console opens at any point. It does what run.bat does -
dependency sync, keys.local.bat, llama-server, the page copy - serves the app in this process and opens the
app window. Closing the app window stops the server, and llama-server too when this launcher started it, so
nothing keeps running unseen (or holding the GPU).

Under the desktop app (desktop/, Electron) the launcher runs with FX_SHELL=electron: Electron shows the window,
so this opens none - it watches Electron's process instead (FX_PARENT_PID) and stops the server, and llama-server,
once that is gone.

Logs, since there is no console: data/logs/server.log and data/logs/llama-server.log (the run before is kept
as *.prev.log). Problems show as a message box. `run.bat console` still runs everything in a console window.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import core.modpath  # noqa: F401  (the module folders on sys.path)

BASE =os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
LOGS = os.path.join(BASE, "data", "logs")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
IS_WIN = os.name == "nt"
VENV_PY = os.path.join(BASE, ".venv", "Scripts", "python.exe") if IS_WIN else os.path.join(BASE, ".venv", "bin", "python")
LLAMA_EXE = "llama-server.exe" if IS_WIN else "llama-server"
TITLE = "FX Workspace"
APP_PORT = 8000                       # preferred; another program there moves the app up (choose_ports, 1.4.7)
APP_URL = f"http://127.0.0.1:{APP_PORT}"
CLOSE_GRACE = 4.0                     # seconds without the app window before the server stops (a reload is quicker)


def _set_app_port(port: int) -> None:
    global APP_PORT, APP_URL
    APP_PORT, APP_URL = int(port), f"http://127.0.0.1:{int(port)}"
    os.environ["AGENTIC_PORT"] = str(APP_PORT)
    os.environ["AGENTIC_APP_URL"] = APP_URL


def message(text: str, error: bool = True) -> None:
    if not IS_WIN:                    # Linux: the desktop app shows its own error; the log has the details
        print(("[ERROR] " if error else "") + text, flush=True)
        return
    import ctypes
    ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x10 if error else 0x40)


def log_file(name: str):
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, f"{name}.log")
    if os.path.exists(path):
        os.replace(path, os.path.join(LOGS, f"{name}.prev.log"))
    return open(path, "w", encoding="utf-8", errors="replace", buffering=1)


def answers(url: str, timeout: float = 2.0) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))     # loopback: never a proxy
    try:
        with opener.open(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def is_this_app(port: int, timeout: float = 2.0) -> bool:
    """This install's server answers on the port (its /api/whoami names this folder). An older copy without whoami
    counts when /api/settings answers and its process runs from this folder."""
    import json
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{int(port)}/api/whoami", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        return data.get("app") == "fx-workspace" and \
            os.path.normcase(os.path.abspath(data.get("root") or "")) == os.path.normcase(BASE)
    except Exception:
        pass
    if answers(f"http://127.0.0.1:{int(port)}/api/settings", timeout):
        import ports
        return ports.is_ours(ports.holder(int(port)), BASE)
    return False


def find_running() -> int | None:
    """The port this app already serves on (started earlier, or by run.bat console), or None."""
    import ports
    run = ports.read_run()
    for port in dict.fromkeys(p for p in (run.get("app"), APP_PORT) if p):
        if is_this_app(port):
            return int(port)
    return None


def choose_ports() -> None:
    """The app's port and llama-server's (1.4.7). The preferred one when it's free; a crashed run's leftover of this
    app is stopped and its port taken back; another program keeps its port and ours moves to a free one - the desktop
    app, a second start and the page follow it through data/run/ports.json. In 1.4.5 another program on 8000 meant
    "FX Workspace could not start (is port 8000 taken?)"."""
    import ports
    app = ports.claim(int(os.environ.get("AGENTIC_PORT") or APP_PORT))
    _set_app_port(app["port"])
    if app["how"] != "free":
        print(f"Port: the app uses {app['port']} ({app['how']}{' - ' + app['held_by'] if app.get('held_by') else ''}).", flush=True)

    def router_alive(port: int) -> bool:
        return answers(f"http://127.0.0.1:{port}/v1/models", 3)
    llama = ports.claim(int(os.environ.get("LLAMA_PORT") or 8090), alive=router_alive, avoid=(app["port"],))
    os.environ["LLAMA_PORT"] = str(llama["port"])
    if llama["how"] != "free":
        print(f"Port: llama-server uses {llama['port']} ({llama['how']}"
              f"{' - ' + llama['held_by'] if llama.get('held_by') else ''}).", flush=True)
    ports.write_run({"app": app["port"], "llama": llama["port"], "pid": os.getpid(), "root": BASE,
                     "app_note": app.get("held_by") or "", "llama_note": llama.get("held_by") or ""})


def sync_dependencies(log) -> bool:
    """pip install -r requirements.txt, as run.bat does - but only when requirements.txt changed since the last sync."""
    req = os.path.join(BASE, "requirements.txt")
    stamp = os.path.join(BASE, ".venv", "requirements.sha256")
    with open(req, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    try:
        with open(stamp, encoding="ascii") as f:
            if f.read().strip() == digest:
                return True
    except OSError:
        pass
    print("Syncing pinned dependencies...", flush=True)
    python = VENV_PY
    done = subprocess.run([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", req],
                          stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    if done.returncode:
        return False
    with open(stamp, "w", encoding="ascii") as f:
        f.write(digest)
    return True


def load_keys() -> None:
    """keys.local.bat's `set NAME=value` lines (made with placeholders when missing, like run.bat)."""
    path = os.path.join(BASE, "keys.local.bat")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("@echo off\nset GEMINI_API_KEYS=YOUR_GEMINI_KEY_HERE\nset GROQ_API_KEY=YOUR_GROQ_KEY_HERE\n"
                    "set OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE\n")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.lower().startswith("set "):
                continue
            body = line[4:].strip()
            if len(body) > 1 and body[0] == body[-1] == '"':
                body = body[1:-1]
            name, eq, value = body.partition("=")
            if eq and name.strip():
                os.environ[name.strip()] = value.strip()


def llama_dir() -> str:
    # The newest stock llama.cpp first, PrismML's fork last - as in run.bat. The app moves the router to the
    # fork when a model needs it (Ternary Bonsai 2) and back (llama_builds.py).
    import paths
    for d in (*paths.llama_dirs(), *paths.prism_dirs()):
        if os.path.exists(os.path.join(d, LLAMA_EXE)):
            return d
    return paths.llama_dirs()[0]


def environment() -> None:
    env = os.environ
    env.setdefault("LLAMA_HOST", "127.0.0.1")
    env.setdefault("LLAMA_PORT", "8090")
    env.setdefault("MODELS_INI", os.path.join(BASE, "models.ini"))
    import model_scan
    model_scan.ensure_ini()                       # a fresh install: models.ini from models.example.ini, plus local\models
    env.setdefault("LLAMA_CTX", "65536")          # a fallback only; the real window is models.ini's ([*] ctx-size)
    env.setdefault("AUTOSTART_LLAMA", "1")
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost,::1"
    env["LLAMA_SERVER_BASE"] = f"http://{env['LLAMA_HOST']}:{env['LLAMA_PORT']}/v1"
    # No console windows: a llama-server the app restarts on the other build is windowless too, and stops with
    # the app as this launcher's own would (llama_builds.py).
    env["AGENTIC_WINDOWLESS"] = "1"


def start_llama() -> subprocess.Popen | None:
    """llama-server's router, windowless, when it isn't already answering; waits up to 45 s for it."""
    env = os.environ
    models = f"{env['LLAMA_SERVER_BASE']}/models"
    if answers(models, 3) or env.get("AUTOSTART_LLAMA") != "1":
        return None
    exe = os.path.join(llama_dir(), LLAMA_EXE)
    if not os.path.exists(exe):
        print(f"[WARNING] llama-server.exe was not found at {exe}. Cloud mode can still be used.", flush=True)
        return None
    print("Starting llama-server with a strict one-model VRAM limit...", flush=True)
    proc = subprocess.Popen([exe, "--host", env["LLAMA_HOST"], "--port", env["LLAMA_PORT"], "--models-preset", env["MODELS_INI"],
                             "--models-max", "1", "--jinja", "--parallel", "1"],
                            stdin=subprocess.DEVNULL, stdout=log_file("llama-server"), stderr=subprocess.STDOUT,
                            creationflags=NO_WINDOW)
    for _ in range(45):
        if answers(models, 2):
            break
        if proc.poll() is not None:
            print("[WARNING] llama-server stopped right away - see data/logs/llama-server.log.", flush=True)
            return None
        time.sleep(1)
    else:
        print("[WARNING] llama-server did not become ready. Cloud mode can still be used.", flush=True)
    return proc


def stop_tree(proc: subprocess.Popen | None) -> None:
    """llama-server's router and the model servers it started."""
    if proc and proc.poll() is None:
        if not IS_WIN:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=15, creationflags=NO_WINDOW)


def app_windows(browser_name: str) -> int:
    """How many app windows are open: the page's title ends in "FX Workspace" (a chat's title comes first),
    in the browser that shows it. The Studio overlay ("FX Workspace - HUD") doesn't count."""
    import app_window
    try:
        import psutil
    except ImportError:
        psutil = None
    n = 0
    for hwnd, title in app_window._windows():
        if "HUD" in title or not (title == TITLE or title.startswith(TITLE + " ") or f" - {TITLE}" in title):
            continue
        if psutil and browser_name:
            try:
                if psutil.Process(app_window._pid(hwnd)).name().lower() != browser_name:
                    continue
            except Exception:
                continue
        n += 1
    return n


def shell_mode() -> bool:
    """Started by the desktop app (Electron), which owns the window."""
    return os.environ.get("FX_SHELL") == "electron"


def _alive(pid: int) -> bool:
    """False only when the process is really gone. A check that fails for another reason (access, a PC busy loading
    a big model) says nothing about it - it used to read as "gone" and stopped the server under a running app."""
    try:
        import psutil
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return True


def watch_shell(server) -> None:
    """Desktop app: stop once its process (FX_PARENT_PID) has exited - closing the app closes everything."""
    try:
        pid = int(os.environ.get("FX_PARENT_PID") or 0)
    except ValueError:
        pid = 0
    if not pid:
        return
    gone = 0
    while not server.should_exit:
        gone = 0 if _alive(pid) else gone + 1
        if gone >= 3:                      # three seconds in a row: one odd answer never stops the server
            print("The desktop app closed - stopping.", flush=True)
            server.should_exit = True
            return
        time.sleep(1)


def watch(server) -> None:
    """Open the app window once the server answers; stop the server once the window has been closed."""
    if shell_mode():
        watch_shell(server)
        return
    import app_window
    if not app_window.wait_for_server(APP_URL, timeout=120):
        return
    if os.environ.get("APP_WINDOW", "1") == "0" or app_window.open_app(APP_URL) != "app":
        import webbrowser
        if os.environ.get("APP_WINDOW") == "0":
            webbrowser.open(APP_URL)
        return                        # a browser tab: there's no telling when it closes, so the server stays up
    exe = app_window.find_browser() or ""
    name = os.path.basename(exe).lower()
    started, seen, gone = time.time(), False, None
    while not server.should_exit:
        if app_windows(name):
            seen, gone = True, None
        elif seen:
            gone = gone or time.time()
            if time.time() - gone >= CLOSE_GRACE:
                print("The app window was closed - stopping.", flush=True)
                server.should_exit = True
                return
        elif time.time() - started > 120:
            print("[WARNING] The app window never appeared; the server keeps running at " + APP_URL, flush=True)
            return
        time.sleep(1)


def quiet_children() -> None:
    """Under run.bat every console program the app runs (PowerShell for the clipboard and the app list,
    tasklist, the assistant's shell commands...) shared run.bat's console. Under pythonw there is no console to
    share, so each would open a window of its own: here they get none instead - as before, minus the window.
    (Windowed programs ignore the flag; launches that ask for their own console or none at all keep theirs.)"""
    if not IS_WIN:
        return
    own = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10) | getattr(subprocess, "DETACHED_PROCESS", 0x8)
    original = subprocess.Popen.__init__

    def init(self, *args, **kwargs):
        if len(args) <= 13 and not kwargs.get("creationflags", 0) & own:          # creationflags is the 14th
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | NO_WINDOW
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = init


_lock = None


def single_instance() -> bool:
    if not IS_WIN:                    # Linux: an exclusive lock on a file in data/ (released when the process ends)
        global _lock
        import fcntl
        os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
        _lock = open(os.path.join(BASE, "data", ".launcher.lock"), "w")
        try:
            fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.CreateMutexW.restype = ctypes.c_void_p
    handle = k32.CreateMutexW(None, False, "Local\\AgenticWorkspaceLauncher")
    if k32.GetLastError() != 183:                                  # ERROR_ALREADY_EXISTS
        return True
    # Not ours: let go of this handle, or the mutex would outlive the other launcher and a later try could never
    # take it (wait_for_previous asks again).
    k32.CloseHandle(ctypes.c_void_p(handle))
    return False


def wait_for_previous(limit: float = 30.0) -> bool:
    """Reopened right after quitting (1.4.6): the old launcher is still stopping llama-server and letting go of port
    8000 for a few seconds. This copy used to give up at once - the desktop app then said "the server stopped while
    starting" - so now it waits for the old one to finish, then starts. False when the old one is still running
    for real (it answers) or doesn't finish in time."""
    end = time.time() + limit
    while time.time() < end:
        if find_running():
            return False
        time.sleep(0.5)
        if single_instance():
            return True
    return False


def main() -> int:
    os.chdir(BASE)
    if not single_instance() and not wait_for_previous():
        # Already starting or running: show its window (when it's still starting, its own window is coming).
        # No logging here - the running launcher has the log open. Under the desktop app, Electron shows it.
        sys.stdout = sys.stderr = open(os.devnull, "w")
        running = find_running()
        if not shell_mode() and running:
            import app_window
            _set_app_port(running)
            app_window.open_app(APP_URL)
        return 0
    out = log_file("server")
    sys.stdout = sys.stderr = out
    print(f"FX Workspace - {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    quiet_children()
    # Tray icons left by copies that were closed by force (an installer, Task Manager) go now, not when the mouse
    # happens to pass over them - they read as dozens of servers (1.4.4, tray_sweep.py).
    try:
        import threading
        import tray_sweep
        threading.Thread(target=tray_sweep.sweep, daemon=True, name="tray-sweep").start()
    except Exception:
        pass
    running = find_running()
    if running:                                        # a server started some other way (run.bat console)
        _set_app_port(running)
        import ports
        ports.write_run({"app": running})
        if not shell_mode():
            import app_window
            app_window.open_app(APP_URL)
        return 0
    if not sync_dependencies(out):
        message("Installing the Python dependencies failed.\n\nDetails: data\\logs\\server.log\n"
                "Run run.bat console to see it happen.")
        return 1
    choose_ports()
    environment()
    load_keys()
    import shutil
    os.makedirs(os.path.join(BASE, "static"), exist_ok=True)
    shutil.copyfile(os.path.join(BASE, "index.html"), os.path.join(BASE, "static", "index.html"))
    llama = start_llama()
    try:
        import llama_builds
        llama_builds.adopt(llama)          # watched and restarted if it stops (1.4.7), stopped with the app
    except Exception:
        pass
    try:
        import uvicorn
        # A stream still open (the Studio overlay's, say) must not hold the shutdown up.
        server = uvicorn.Server(uvicorn.Config("main:app", host="127.0.0.1", port=APP_PORT, timeout_graceful_shutdown=5))
        threading.Thread(target=watch, args=(server,), daemon=True, name="app-window-watch").start()
        print(f"Starting the API at {APP_URL} ...", flush=True)
        try:
            server.run()
        except SystemExit:
            pass
        if not server.started:
            import ports
            held = ports.describe(ports.holder(APP_PORT)) if not ports.is_free(APP_PORT) else ""
            message(f"FX Workspace could not listen on port {APP_PORT}{f' - {held} holds it' if held else ''}.\n\n"
                    "Start it again: it picks a free port by itself. Details: data\\logs\\server.log")
            return 1
        print("FX Workspace stopped.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        message("FX Workspace stopped with an error.\n\nDetails: data\\logs\\server.log")
        return 1
    finally:
        stop_tree(llama)


if __name__ == "__main__":
    code = main()
    os._exit(code)                    # leftover worker threads must not keep a windowless process alive
