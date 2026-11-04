"""Live CPU/RAM/NVIDIA telemetry with Windows-safe model VRAM tracking.

On Windows/WDDM, ``nvidia-smi --query-compute-apps`` commonly reports N/A or
zero per-process memory even while llama-server has a model resident.  Direct
process accounting is still preferred when NVIDIA supplies it; otherwise the
module measures the stable before/after VRAM delta around a model load.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

try:
    import psutil
except ImportError:  # The UI degrades cleanly if requirements are incomplete.
    psutil = None


_CACHE = {"at": 0.0, "value": {}}
_PROCESS_CACHE = {"at": 0.0, "value": None}
_MODEL = {
    "baseline_used_mb": None,
    "baseline_total_mb": None,
    "estimated_mb": None,
    "name": None,
}
_LOCK = threading.Lock()
_TTL = 1.5


def _number(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


_SMI = {"path": None, "at": 0.0}


def _nvidia_smi_path():
    # Searching PATH is slow with a long PATH, and every sample asked again - remembered for 5 minutes.
    now = time.time()
    if _SMI["at"] and now - _SMI["at"] < 300:
        return _SMI["path"]
    found = shutil.which("nvidia-smi")
    if not found:
        candidate = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe",
        )
        found = candidate if os.path.exists(candidate) else None
    _SMI.update(path=found, at=now)
    return found


def _run_smi(arguments: list[str]) -> str:
    binary = _nvidia_smi_path()
    if not binary:
        return ""
    try:
        result = subprocess.run(
            [binary, *arguments], capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


def _parse_gpu_row(text: str) -> dict:
    line = next((row.strip() for row in text.splitlines() if row.strip()), "")
    parts = [part.strip() for part in line.split(",", 3)]
    if len(parts) != 4:
        return {}
    used, total, load, name = parts
    used_mb, total_mb, load_pct = _number(used), _number(total), _number(load)
    if used_mb is None or total_mb is None:
        return {}
    # Under WDDM nvidia-smi now and then reports memory.used as 2^64-1 bytes (a wrapped negative), which the System
    # panel showed as 17179865097.2 / 8.0 GB (2026-10-27). A reading past the card's size is no reading.
    if used_mb < 0 or (total_mb > 0 and used_mb > total_mb * 1.02):
        return {}
    if load_pct is not None and not 0 <= load_pct <= 100:
        load_pct = None
    return {
        "name": name,
        "vram_used_mb": used_mb,
        "vram_total_mb": total_mb,
        "vram_percent": round((used_mb / total_mb) * 100, 1) if total_mb else 0.0,
        "gpu_percent": load_pct,
    }


def _parse_model_vram(text: str) -> float | None:
    total = 0.0
    found = False
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        process_name, used = parts
        value = _number(used)
        if value is None:
            continue
        if any(token in process_name.lower() for token in ("llama-server", "llama_server", "llama.cpp")):
            total += value
            found = True
    return round(total, 1) if found else None


class _GpuCounters:
    """Windows' "GPU Process Memory" performance counters, read in-process
    through pdh.dll: each process's Dedicated Usage (its VRAM) and Shared Usage
    (system RAM the GPU driver lent it - where a model that doesn't fit in VRAM
    spills). It replaced a PowerShell Get-Counter call that took ~1.6 s and ran
    every 1.5 s while a model was loaded; this takes a few milliseconds."""
    PDH_FMT_LARGE = 0x00000400
    PDH_MORE_DATA = 0x800007D2

    def __init__(self):
        self.ok = False
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes
            self.ctypes, self.wintypes = ctypes, wintypes
            self.pdh = ctypes.WinDLL("pdh")

            class Value(ctypes.Structure):
                _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]

            class Item(ctypes.Structure):
                _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", Value)]
            self.Item = Item
            self.query = wintypes.HANDLE()
            if self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query)) != 0:
                return
            self.counters = {}
            for key, path in (("dedicated", "\\GPU Process Memory(*)\\Dedicated Usage"),
                              ("shared", "\\GPU Process Memory(*)\\Shared Usage")):
                handle = wintypes.HANDLE()
                if self.pdh.PdhAddEnglishCounterW(self.query, path, None, ctypes.byref(handle)) != 0:
                    return
                self.counters[key] = handle
            self.ok = True
        except Exception:
            self.ok = False

    def read(self, pids: set) -> dict | None:
        """{"dedicated_mb", "shared_mb"} summed over the given processes, or None."""
        if not self.ok or not pids:
            return None
        ctypes, wintypes = self.ctypes, self.wintypes
        try:
            if self.pdh.PdhCollectQueryData(self.query) != 0:
                return None
            out = {}
            for key, handle in self.counters.items():
                size, count = wintypes.DWORD(0), wintypes.DWORD(0)
                status = self.pdh.PdhGetFormattedCounterArrayW(handle, self.PDH_FMT_LARGE, ctypes.byref(size),
                                                              ctypes.byref(count), None) & 0xFFFFFFFF
                if status != self.PDH_MORE_DATA or not size.value:
                    out[key] = 0.0
                    continue
                buf = (ctypes.c_byte * size.value)()
                if self.pdh.PdhGetFormattedCounterArrayW(handle, self.PDH_FMT_LARGE, ctypes.byref(size),
                                                         ctypes.byref(count), buf) != 0:
                    return None
                items = ctypes.cast(buf, ctypes.POINTER(self.Item))
                total = 0
                for i in range(count.value):
                    name = items[i].szName or ""
                    if name.startswith("pid_"):
                        try:
                            pid = int(name.split("_")[1])
                        except (IndexError, ValueError):
                            continue
                        if pid in pids:
                            total += items[i].FmtValue.largeValue
                out[key] = round(total / (1024 * 1024), 1)
            return {"dedicated_mb": out.get("dedicated", 0.0), "shared_mb": out.get("shared", 0.0)}
        except Exception:
            return None

    def per_pid(self) -> dict:
        """{pid: dedicated bytes} for every process using the GPU's memory (pc/gpu_apps.py, 1.4.6)."""
        if not self.ok:
            return {}
        ctypes, wintypes = self.ctypes, self.wintypes
        try:
            if self.pdh.PdhCollectQueryData(self.query) != 0:
                return {}
            handle = self.counters["dedicated"]
            size, count = wintypes.DWORD(0), wintypes.DWORD(0)
            status = self.pdh.PdhGetFormattedCounterArrayW(handle, self.PDH_FMT_LARGE, ctypes.byref(size),
                                                          ctypes.byref(count), None) & 0xFFFFFFFF
            if status != self.PDH_MORE_DATA or not size.value:
                return {}
            buf = (ctypes.c_byte * size.value)()
            if self.pdh.PdhGetFormattedCounterArrayW(handle, self.PDH_FMT_LARGE, ctypes.byref(size),
                                                     ctypes.byref(count), buf) != 0:
                return {}
            items, out = ctypes.cast(buf, ctypes.POINTER(self.Item)), {}
            for i in range(count.value):
                name = items[i].szName or ""
                if name.startswith("pid_"):
                    try:
                        pid = int(name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    out[pid] = out.get(pid, 0) + max(0, items[i].FmtValue.largeValue)
            return out
        except Exception:
            return {}


def gpu_by_process() -> dict:
    """{pid: dedicated VRAM bytes} for every process (Windows), {} elsewhere."""
    global _COUNTERS
    if os.name != "nt":
        return {}
    if _COUNTERS is None:
        _COUNTERS = _GpuCounters()
    return _COUNTERS.per_pid()


_COUNTERS = None
_LAST_PROCESS = {"shared_mb": None}


def _llama_pids() -> set:
    if psutil is None:
        return set()
    pids = set()
    for proc in psutil.process_iter(["pid", "name"]):
        name = str(proc.info.get("name") or "").lower()
        if any(token in name for token in ("llama-server", "llama_server", "llama.cpp")):
            pids.add(int(proc.info["pid"]))
    return pids


def gpu_process_memory() -> dict | None:
    """llama-server's VRAM and the system RAM the GPU driver lent it (Windows),
    or None when unknown."""
    global _COUNTERS
    if os.name != "nt":
        return None
    if _COUNTERS is None:
        _COUNTERS = _GpuCounters()
    pids = _llama_pids()
    return _COUNTERS.read(pids) if pids else None


def _windows_model_vram() -> float | None:
    """llama-server's dedicated VRAM from Windows GPU counters - the reliable
    WDDM path when nvidia-smi reports per-process memory as N/A. Also keeps
    its shared (spilled) memory for placement()."""
    try:
        got = gpu_process_memory()
    except Exception:
        got = None
    if not got:
        return None
    _LAST_PROCESS["shared_mb"] = got["shared_mb"]
    value = got["dedicated_mb"]
    return round(value, 1) if value is not None and value >= 64 else None


def spilled_mb() -> float | None:
    """System RAM the GPU driver lent llama-server at the last reading (Windows sysmem fallback)."""
    return _LAST_PROCESS["shared_mb"]


class _Nvml:
    """The NVIDIA driver's own library (nvml.dll - what nvidia-smi itself reads), called in-process (1.4.6). A
    reading takes ~0.01 ms instead of starting nvidia-smi (~60 ms idle, longer under load) every 1.5 s, and it
    can't time out. memory_info v2 reports "used" without the driver's reserve, as nvidia-smi does."""
    NOT_AVAILABLE = 2 ** 64 - 1

    def __init__(self):
        self.ok = False
        if os.name != "nt":
            return
        try:
            import ctypes
            self.ctypes = ctypes
            self.lib = ctypes.WinDLL("nvml.dll")
            if self.lib.nvmlInit_v2() != 0:
                return
            self.dev = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.dev)) != 0:
                return
            name = ctypes.create_string_buffer(96)
            self.name = name.value.decode(errors="replace") if self.lib.nvmlDeviceGetName(self.dev, name, 96) == 0 else "NVIDIA GPU"

            class Util(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

            class MemV2(ctypes.Structure):
                _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong), ("reserved", ctypes.c_ulonglong),
                            ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

            class MemV1(ctypes.Structure):
                _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
            self.Util, self.MemV2, self.MemV1 = Util, MemV2, MemV1
            self.v2 = hasattr(self.lib, "nvmlDeviceGetMemoryInfo_v2")
            self.ok = True
        except Exception:
            self.ok = False

    def _memory(self):
        ctypes = self.ctypes
        if self.v2:
            m = self.MemV2(version=ctypes.sizeof(self.MemV2) | (2 << 24))
            if self.lib.nvmlDeviceGetMemoryInfo_v2(self.dev, ctypes.byref(m)) == 0:
                return m.used, m.total
            self.v2 = False                                 # an older driver: v1 from now on
        m = self.MemV1()
        return (m.used, m.total) if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0 else (None, None)

    def read(self) -> dict:
        """The same shape as _parse_gpu_row, {} when nothing could be read. A part that fails is left out alone:
        the load still shows when memory can't be read, and the other way round."""
        if not self.ok:
            return {}
        try:
            used, total = self._memory()
            u = self.Util()
            load = float(u.gpu) if self.lib.nvmlDeviceGetUtilizationRates(self.dev, self.ctypes.byref(u)) == 0 else None
        except Exception:
            return {}
        mb = 1024 * 1024
        used_mb = round(used / mb, 1) if used is not None and used != self.NOT_AVAILABLE else None
        total_mb = round(total / mb, 1) if total else None
        if used_mb is not None and total_mb and used_mb > total_mb * 1.02:
            used_mb = None                                  # a wrapped reading is no reading (as in _parse_gpu_row)
        if used_mb is None and load is None:
            return {}
        return {"name": self.name, "vram_used_mb": used_mb, "vram_total_mb": total_mb,
                "vram_percent": round(used_mb / total_mb * 100, 1) if used_mb is not None and total_mb else None,
                "gpu_percent": load if load is None or 0 <= load <= 100 else None}


_NVML = None


def _gpu_sample() -> dict:
    global _NVML
    if _NVML is None:
        _NVML = _Nvml()
    if _NVML.ok:
        got = _NVML.read()
        if got:
            return got
    return _parse_gpu_row(_run_smi([
        "--query-gpu=memory.used,memory.total,utilization.gpu,name",
        "--format=csv,noheader,nounits",
    ]))


_CPU = {"times": None, "value": None}


def _cpu_percent() -> float | None:
    """CPU use since the previous reading, whichever thread asks. psutil 7 keeps cpu_percent(interval=None)'s
    "since the last call" per thread, and every background refresh here runs on a new thread - so each one was a
    first call and read 0.0 (the System panel's "CPU 0.0%", 3 Nov 2026)."""
    if psutil is None:
        return None
    now = psutil.cpu_times()
    with _LOCK:
        last = _CPU["times"]
        total = sum(now) - sum(last) if last is not None else 0.0
        # Windows counts CPU time in ~16 ms ticks: too soon after the last reading there is nothing to measure, so
        # the older baseline stays and the last value is given again.
        if last is None or total < 0.05:
            if last is None:
                _CPU["times"] = now
            return _CPU["value"]
        _CPU["times"] = now
        busy = total - (now.idle - last.idle)
        _CPU["value"] = round(max(0.0, min(100.0, busy / total * 100)), 1)
        return _CPU["value"]


if psutil is not None:
    try:
        _cpu_percent()                                    # the first reading needs one before it
    except Exception:
        pass


def _copy_result(value: dict) -> dict:
    copied = dict(value)
    if isinstance(value.get("gpu"), dict):
        copied["gpu"] = dict(value["gpu"])
    return copied


def begin_model_load() -> None:
    """Capture the no-model VRAM floor immediately before a first load.

    The first valid floor is retained across model swaps; a swap begins while
    the old model is still resident and therefore cannot establish a new
    system-only baseline.
    """
    with _LOCK:
        if _MODEL["baseline_used_mb"] is not None:
            return
    sample = _gpu_sample()
    if not sample or sample.get("vram_used_mb") is None:
        return
    with _LOCK:
        if _MODEL["baseline_used_mb"] is None:
            _MODEL["baseline_used_mb"] = sample["vram_used_mb"]
            _MODEL["baseline_total_mb"] = sample["vram_total_mb"]
        _CACHE.update(at=0.0, value={})
        _PROCESS_CACHE.update(at=0.0, value=None)


def finish_model_load(model_name: str | None = None) -> None:
    """Freeze the loaded model's VRAM delta after llama-server finishes."""
    with _LOCK:
        if model_name and _MODEL["name"] == model_name and _MODEL["estimated_mb"] is not None:
            return
    sample = _gpu_sample()
    if not sample or sample.get("vram_used_mb") is None:
        return
    with _LOCK:
        baseline = _MODEL["baseline_used_mb"]
        same_gpu = _MODEL["baseline_total_mb"] == sample["vram_total_mb"]
        if baseline is not None and same_gpu:
            delta = max(0.0, sample["vram_used_mb"] - baseline)
            # Tiny deltas are normal desktop noise, not a resident model.
            _MODEL["estimated_mb"] = round(delta, 1) if delta >= 64 else None
        _MODEL["name"] = model_name
        _CACHE.update(at=0.0, value={})
        _PROCESS_CACHE.update(at=0.0, value=None)


def _add_vram_breakdown(gpu: dict, active_model: str | None) -> dict:
    gpu = dict(gpu)
    used = gpu.get("vram_used_mb")
    total = gpu.get("vram_total_mb")
    direct = gpu.pop("process_vram_mb", None)

    if not active_model:
        model_mb = 0.0
        source = "none"
        if _MODEL["baseline_used_mb"] is None and used is not None:
            _MODEL["baseline_used_mb"] = used
            _MODEL["baseline_total_mb"] = total
    elif used is None:                                    # memory unreadable this time (NVML gives parts alone)
        model_mb = None
        source = "unavailable"
    elif direct is not None and direct >= 64:
        model_mb = min(float(direct), float(used))
        source = "process"
    elif _MODEL["estimated_mb"] is not None:
        model_mb = min(float(_MODEL["estimated_mb"]), float(used))
        source = "load_delta"
    else:
        model_mb = None
        source = "unavailable"

    system_mb = max(0.0, float(used) - model_mb) if model_mb is not None and used is not None else None
    gpu.update({
        "model_vram_mb": round(model_mb, 1) if model_mb is not None else None,
        "model_vram_percent": round(model_mb / total * 100, 1) if model_mb is not None and total else None,
        "model_vram_source": source,
        "system_vram_mb": round(system_mb, 1) if system_mb is not None else None,
        "system_vram_percent": round(system_mb / total * 100, 1) if system_mb is not None and total else None,
    })
    return gpu


def _collect() -> dict:
    result = {
        "cpu_percent": None,
        "ram_percent": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "gpu": None,
    }
    if psutil is not None:
        try:
            memory = psutil.virtual_memory()
            result.update({
                "cpu_percent": _cpu_percent(),
                "ram_percent": round(float(memory.percent), 1),
                "ram_used_gb": round((memory.total - memory.available) / (1024 ** 3), 1),
                "ram_total_gb": round(memory.total / (1024 ** 3), 1),
            })
        except Exception:
            pass

    gpu = _gpu_sample()
    if gpu:
        # Per-process memory from nvidia-smi is N/A under Windows' WDDM; with NVML in use, snapshot() reads
        # llama-server's from the Windows GPU counters instead of starting nvidia-smi for it.
        gpu["process_vram_mb"] = None if _NVML is not None and _NVML.ok else _parse_model_vram(_run_smi([
            "--query-compute-apps=process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]))
        last = _LAST_GPU["value"] or {}
        for key in ("vram_used_mb", "vram_percent", "gpu_percent"):     # one part missing keeps its last value
            if gpu.get(key) is None and last.get(key) is not None and time.monotonic() - _LAST_GPU["at"] < _KEEP_GPU_S:
                gpu[key] = last[key]
        if gpu.get("vram_used_mb") is not None and gpu.get("gpu_percent") is not None:
            _LAST_GPU.update(at=time.monotonic(), value=dict(gpu))
        result["gpu"], result["gpu_state"] = gpu, "ok"
    elif _LAST_GPU["value"] and time.monotonic() - _LAST_GPU["at"] < _KEEP_GPU_S:
        # A bad or missing reading keeps the last good one on screen, with its age. It was dropped after 15 s,
        # and every GPU row then read "Unavailable" while the GPU worked (3 Nov 2026).
        result["gpu"] = dict(_LAST_GPU["value"], stale_s=round(time.monotonic() - _LAST_GPU["at"]))
        result["gpu_state"] = "stale"
    else:
        found = (_NVML is not None and _NVML.ok) or bool(_nvidia_smi_path())
        result["gpu_state"] = "unreadable" if found else "none"
    return result


_LAST_GPU = {"at": 0.0, "value": None}
_KEEP_GPU_S = 60


_BG = {"on": False}


def snapshot_nonblocking(active_model: str | None = None) -> dict:
    """For the UI's status poll: return the latest reading at once and, if it
    is stale, refresh it on a background thread. Collecting (nvidia-smi,
    Windows GPU counters) takes ~150-200 ms, which snapshot() paid inside the
    request on roughly every other poll."""
    if not _CACHE["value"]:
        return snapshot(active_model)
    stale = time.monotonic() - _CACHE["at"] >= _TTL
    if stale and not _BG["on"]:
        _BG["on"] = True

        def _refresh():
            try:
                snapshot(active_model, force=True)
            finally:
                _BG["on"] = False
        threading.Thread(target=_refresh, daemon=True, name="metrics-refresh").start()
    return snapshot(active_model, allow_stale=True)


def snapshot(active_model: str | None = None, force: bool = False, allow_stale: bool = False) -> dict:
    """The slow parts (_collect, _windows_model_vram) run OUTSIDE _LOCK, which
    only guards reading and swapping the caches. Collecting under the lock made
    every cached read wait ~170 ms behind a background refresh."""
    now = time.monotonic()
    with _LOCK:
        cached_ok = not force and _CACHE["value"] and (allow_stale or now - _CACHE["at"] < _TTL)
        value = _copy_result(_CACHE["value"]) if cached_ok else None
    if value is None:
        value = _collect()
        with _LOCK:
            _CACHE.update(at=now, value=_copy_result(value))
    if isinstance(value.get("gpu"), dict):
        # Keep reporting the physically resident model after "New chat"
        # or while a different pin is waiting for its first request.
        effective_model = active_model or _MODEL["name"]
        direct = value["gpu"].get("process_vram_mb")
        if effective_model and (direct is None or direct < 64):
            with _LOCK:
                refresh = now - _PROCESS_CACHE["at"] >= _TTL and not allow_stale
            if refresh:
                measured = _windows_model_vram()
                with _LOCK:
                    _PROCESS_CACHE.update(at=now, value=measured)
            with _LOCK:
                process_mb = _PROCESS_CACHE["value"]
            if process_mb is not None:
                value["gpu"]["process_vram_mb"] = process_mb
        with _LOCK:
            value["gpu"] = _add_vram_breakdown(value["gpu"], effective_model)
    return value
