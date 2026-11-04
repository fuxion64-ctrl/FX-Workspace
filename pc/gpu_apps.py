"""
gpu_apps.py
===========
Which apps hold the GPU's memory right now (1.4.6): shown when a model couldn't load for lack of VRAM, with what to do.

nvidia-smi can't tell per-process memory on Windows (WDDM), but Windows itself counts it: the "GPU Process Memory"
performance counters give each process's dedicated (on-card) memory. Read once, on demand - it takes a second or two.
Read-only: nothing here closes or changes another app.
"""
from __future__ import annotations

import re
import subprocess

# What to do about the usual suspects, by process name (lower case, without .exe).
TIPS = {
    "chrome": "Chrome: close heavy tabs, or turn off \"Use graphics acceleration when available\" (Settings › System).",
    "msedge": "Edge: close heavy tabs, or turn off hardware acceleration (Settings › System and performance).",
    "firefox": "Firefox: close heavy tabs, or turn off hardware acceleration (Settings › Performance).",
    "brave": "Brave: close heavy tabs, or turn off hardware acceleration (Settings › System).",
    "discord": "Discord: turn off hardware acceleration (Settings › Advanced).",
    "discordptb": "Discord: turn off hardware acceleration (Settings › Advanced).",
    "steamwebhelper": "Steam: close it while you use a local model, or turn off GPU acceleration in its web views.",
    "nvidia overlay": "NVIDIA overlay: turn off the in-game overlay in the NVIDIA app (Settings › Features).",
    "obs64": "OBS: stop the preview or close it while a model runs.",
    "llama-server": "Another llama-server: a second model is loaded - unload it (Settings › Models) or close the app that started it.",
    "sd-cli": "The picture engine: it gives the GPU back by itself; Settings › Pictures › Free the GPU does it now.",
    "python": "A Python program (a picture, music or Labs job): let it finish, or stop it in its place.",
    "pythonw": "A Python program (a picture, music or Labs job): let it finish, or stop it in its place.",
    "dwm": "Windows' own desktop drawing - it always needs some. More monitors or a higher resolution use more.",
    "explorer": "Windows' Explorer - small and always there.",
}
_INSTANCE = re.compile(r"pid_(\d+)_")


def _counters() -> list:
    """[(pid, bytes)] from \\GPU Process Memory(*)\\Dedicated Usage - in-process (pdh.dll), else through PowerShell."""
    try:
        import system_metrics
        got = system_metrics.gpu_by_process()
        if got:
            return [(pid, n) for pid, n in got.items() if n > 0]
    except Exception:
        pass
    ps = ("(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | "
          "Where-Object CookedValue -gt 0 | ForEach-Object { $_.InstanceName + '|' + [int64]$_.CookedValue }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                             text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        name, _, val = line.strip().rpartition("|")
        m = _INSTANCE.search(name)
        if m and val.isdigit():
            rows.append((int(m.group(1)), int(val)))
    return rows


def snapshot() -> dict:
    """{"ok", "apps": [{"name", "bytes", "pids", "tip"}], "total", "used", "free"} - the biggest first."""
    import psutil
    by_name: dict = {}
    for pid, n in _counters():
        try:
            name = psutil.Process(pid).name()
        except Exception:
            name = f"process {pid}"
        key = re.sub(r"\.exe$", "", name, flags=re.I)
        row = by_name.setdefault(key.lower(), {"name": key, "bytes": 0, "pids": []})
        row["bytes"] += n
        row["pids"].append(pid)
    apps = sorted(by_name.values(), key=lambda r: -r["bytes"])
    for r in apps:
        r["tip"] = TIPS.get(r["name"].lower(), "")
    gpu = {}
    try:
        import system_metrics
        g = (system_metrics.snapshot(allow_stale=True) or {}).get("gpu") or {}
        gpu = {"total": int((g.get("vram_total_mb") or 0) * 1048576), "used": int((g.get("vram_used_mb") or 0) * 1048576)}
    except Exception:
        pass
    total, used = gpu.get("total") or 0, gpu.get("used") or sum(r["bytes"] for r in apps)
    return {"ok": True, "apps": apps, "total": total, "used": used, "free": max(0, total - used) if total else None,
            "note": "Windows counts some shared desktop memory under dwm, so the apps can add up to more than is in use." if apps else "Windows didn't report per-app GPU memory on this PC."}
