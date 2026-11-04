"""
ports.py - which local ports this app's parts listen on (1.4.6 reliability).

In 1.4.5 every part had a fixed port - the app 8000, llama-server 8090, sd-server 8095, whisper 8092, the agent's
browser 9333, the music engine 8011 - and another program on one of them broke that part: the server didn't start
("is port 8000 taken?"), or worse, the app talked to the other program as if it were its own engine.

Now each part asks here before it starts:
  * the port it prefers is free             -> it gets that one;
  * a process of THIS app holds it          -> the caller decides: reuse it when it answers, else stop it
                                               (a crashed run's leftover) and take the port back;
  * another program holds it                -> the next free port above it, and the part follows that one.
What each part runs on is written to data/run/ports.json, so the desktop app, run.bat and a second start find the
running app wherever it ended up.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_FILE = os.path.join(APP_DIR, "data", "run", "ports.json")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SPAN = 40                       # how far above the preferred port a fallback is looked for


def is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Whether a server could listen on host:port now. Exclusive on Windows: a plain bind succeeds beside a listener
    that didn't ask for exclusivity, which said "free" for a port uvicorn then couldn't use."""
    if not 0 < int(port) < 65536:
        return False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        s.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        s.close()


def holder(port: int) -> dict | None:
    """The process listening on the port: {"pid", "name", "exe", "cmdline", "cwd"}, or None (free, or unknown)."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        conns = psutil.net_connections("tcp")
    except Exception:
        return None
    for c in conns:
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == int(port) and c.pid:
            info = {"pid": c.pid, "name": "", "exe": "", "cmdline": [], "cwd": ""}
            try:
                p = psutil.Process(c.pid)
                info["name"] = p.name()
                for key, get in (("exe", p.exe), ("cmdline", p.cmdline), ("cwd", p.cwd)):
                    try:
                        info[key] = get() or info[key]
                    except Exception:
                        pass
            except Exception:
                pass
            return info
    return None


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)) if path else ""


def is_ours(info: dict | None, root: str = APP_DIR) -> bool:
    """A process this app (this install - not another copy elsewhere) started: its program, working folder or a
    path on its command line is inside the app's folder, or it runs this app's models.ini."""
    if not info:
        return False
    base = _norm(root)
    paths = [info.get("exe") or "", info.get("cwd") or ""] + [str(a) for a in (info.get("cmdline") or [])]
    ini = _norm(os.environ.get("MODELS_INI") or os.path.join(root, "models.ini"))
    for p in paths:
        if not p or len(p) < 3:
            continue
        n = _norm(p) if (":" in p or p.startswith("\\")) else ""
        if n and (n == ini or n == base or n.startswith(base + os.sep)):
            return True
    return False


def pick(preferred: int, host: str = "127.0.0.1", avoid=()) -> int:
    """The preferred port when it's free, else the first free one above it (within SPAN), else one the system picks."""
    avoid = {int(a) for a in avoid if a}
    for port in [int(preferred)] + list(range(int(preferred) + 1, int(preferred) + SPAN)):
        if port not in avoid and port < 65536 and is_free(port, host):
            return port
    s = socket.socket()
    try:
        s.bind((host, 0))
        return s.getsockname()[1]
    finally:
        s.close()


def stop_tree(pid: int, wait_port: int | None = None, timeout: float = 15.0) -> bool:
    """Stop a process and what it started (a leftover of this app). True once the port (when given) is free."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=20,
                           creationflags=NO_WINDOW)
        else:
            os.kill(pid, 9)
    except Exception:
        pass
    if wait_port is None:
        return True
    end = time.time() + timeout
    while time.time() < end:
        if is_free(wait_port):
            return True
        time.sleep(0.3)
    return is_free(wait_port)


def claim(preferred: int, alive=None, host: str = "127.0.0.1", avoid=()) -> dict:
    """The port a part should use: {"port", "how"} with how one of
       "free"     the preferred port was free;
       "reuse"    a process of this app holds it and `alive()` says it answers - use it as it is;
       "reclaimed" a process of this app held it without answering (a crashed run's): stopped, port taken back;
       "moved"    another program holds it - `port` is a free one above it ("held_by" names that program).
    """
    if is_free(preferred, host) and int(preferred) not in {int(a) for a in avoid if a}:
        return {"port": int(preferred), "how": "free"}
    info = holder(preferred)
    if is_ours(info):
        if alive is not None:
            try:
                if alive(int(preferred)):
                    return {"port": int(preferred), "how": "reuse", "pid": info["pid"]}
            except Exception:
                pass
        if stop_tree(info["pid"], wait_port=int(preferred)):
            return {"port": int(preferred), "how": "reclaimed", "pid": info["pid"]}
    port = pick(int(preferred) + 1, host, avoid=avoid)
    return {"port": port, "how": "moved", "held_by": describe(info) if info else "another program"}


def describe(info: dict | None) -> str:
    if not info:
        return "another program"
    name = info.get("name") or os.path.basename(info.get("exe") or "") or "a program"
    return f"{name} (process {info.get('pid')})"


# ---- what runs where (data/run/ports.json) -------------------------------------------------------------
def read_run(path: str = RUN_FILE) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_run(update: dict, path: str = RUN_FILE) -> dict:
    """Merge `update` into the run file (atomically) and return what it holds now."""
    data = read_run(path)
    data.update({k: v for k, v in update.items() if v is not None})
    data["updated"] = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    return data


def note(name: str, port: int, how: str = "", held_by: str = "") -> None:
    """Record a part's port in the app's run file. Never from the test suite (a fake listener's port there would
    send the desktop app to the wrong place)."""
    import sys
    if "unittest" in sys.modules and os.environ.get("FX_PORTS_NOTE_IN_TESTS") != "1":
        return
    try:
        write_run({name: int(port), **({f"{name}_note": f"{how}: {held_by}"} if how == "moved" else {})})
    except Exception:
        pass
