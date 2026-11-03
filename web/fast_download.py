"""
fast_download.py
================
Big downloads in pieces at once (1.4.6), the way download managers such as FDM do it: one connection to a CDN is
often held to a fraction of the line, several share it out.

  fetch(url, tmp, size, cancel=None, progress=None)
      Downloads `size` bytes of `url` into `tmp` (the caller checks the SHA-256 and renames it). The file is cut
      into pieces of ~32 MB that CONNECTIONS workers pull one after another, each with an HTTP Range request
      written at its own place in a file of the full size. tmp + ".json" records how far each piece got, so a pause,
      a crash or a lost connection resumes where every piece stopped. A .part left by the old one-stream download
      counts as its first bytes. A server that ignores Range gets one plain stream.
      progress(n) is called with each new byte count (the bytes already there first); cancel.is_set() stops it.

Every request goes through netlayer (the route, the kill switch, the connection log). Polite: at most MAX_CONNECTIONS
to a server (Hugging Face serves ranges and allows this), 2 through Tor (slower circuits, shared exits).
"""
from __future__ import annotations

import json
import os
import threading

import httpx

import netlayer

CONNECTIONS = 4
MAX_CONNECTIONS = 8
TOR_CONNECTIONS = 2
PIECE = 32 * 1024 * 1024
MIN_SPLIT = 96 * 1024 * 1024            # smaller files: one stream is as fast
SAVE_EVERY = 8 * 1024 * 1024


class Cancelled(InterruptedError):
    pass


def connections() -> int:
    try:
        return TOR_CONNECTIONS if netlayer.route_for("other").kind == "tor" else CONNECTIONS
    except Exception:
        return CONNECTIONS


def _client():
    return netlayer.client("other", timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True, private=True)


def _pieces(size: int) -> list:
    return [[s, min(size, s + PIECE)] for s in range(0, size, PIECE)]


def _load_state(tmp: str, size: int) -> list:
    """How many bytes of each piece are there. An old one-stream .part (no .json) is its first `len` bytes."""
    pieces = _pieces(size)
    try:
        with open(tmp + ".json", encoding="utf-8") as f:
            st = json.load(f)
        if st.get("size") == size and len(st.get("have", [])) == len(pieces):
            return [max(0, min(int(h), e - s)) for h, (s, e) in zip(st["have"], pieces)]
    except (OSError, ValueError, TypeError):
        pass
    old = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if old > size:
        old = 0
    return [max(0, min(e, old) - s) for s, e in pieces]


def _save_state(tmp: str, size: int, have: list) -> None:
    part = tmp + ".json.tmp"
    with open(part, "w", encoding="utf-8") as f:
        json.dump({"size": size, "have": have}, f)
    os.replace(part, tmp + ".json")


def _ranges_work(url: str, size: int) -> bool:
    with _client() as c, c.stream("GET", url, headers={"Range": "bytes=0-0"}) as r:
        total = (r.headers.get("Content-Range") or "").rpartition("/")[2]
        return r.status_code == 206 and total.isdigit() and int(total) == size


def _one_stream(url: str, tmp: str, size: int, cancel, progress) -> None:
    with _client() as c, c.stream("GET", url) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"The server answered {r.status_code}.")
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                if cancel is not None and cancel.is_set():
                    raise Cancelled("Cancelled.")
                f.write(chunk)
                progress(len(chunk))


def fetch(url: str, tmp: str, size: int, cancel=None, progress=None, conns: int | None = None) -> None:
    progress = progress or (lambda n: None)
    if size < MIN_SPLIT or not _ranges_work(url, size):
        _one_stream(url, tmp, size, cancel, progress)
        return
    pieces = _pieces(size)
    have = _load_state(tmp, size)
    with open(tmp, "ab") as f:                          # the file at full size, so every piece writes in place
        if f.tell() < size:
            f.truncate(size)
    progress(sum(have))
    lock, todo, errors = threading.Lock(), [i for i, (s, e) in enumerate(pieces) if have[i] < e - s], []
    unsaved = [0]

    def worker():
        try:
            with _client() as c, open(tmp, "r+b") as f:
                while True:
                    with lock:
                        if errors or not todo:
                            return
                        i = todo.pop(0)
                    s, e = pieces[i]
                    start = s + have[i]
                    with c.stream("GET", url, headers={"Range": f"bytes={start}-{e - 1}"}) as r:
                        if r.status_code != 206:
                            raise RuntimeError(f"The server answered {r.status_code} for a piece of the file.")
                        f.seek(start)
                        for chunk in r.iter_bytes(1 << 16):
                            if cancel is not None and cancel.is_set():
                                raise Cancelled("Cancelled - what came down is kept and resumes next time.")
                            chunk = chunk[:e - (s + have[i])]
                            f.write(chunk)
                            with lock:
                                have[i] += len(chunk)
                                unsaved[0] += len(chunk)
                                if unsaved[0] >= SAVE_EVERY:
                                    f.flush()
                                    _save_state(tmp, size, have)
                                    unsaved[0] = 0
                            progress(len(chunk))
                    if have[i] < e - s:
                        raise RuntimeError("A piece of the file came down short - try again to resume.")
        except BaseException as ex:                     # one worker's failure stops the others at their next piece
            with lock:
                errors.append(ex)

    threads = [threading.Thread(target=worker, daemon=True, name=f"download-{n}")
               for n in range(max(1, min(MAX_CONNECTIONS, conns or connections(), len(todo) or 1)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with lock:
        _save_state(tmp, size, have)
    if errors:
        cancelled = next((x for x in errors if isinstance(x, Cancelled)), None)
        raise cancelled or errors[0]
    if sum(have) != size:
        raise RuntimeError("The download is incomplete - try again to resume.")
    try:
        os.remove(tmp + ".json")
    except OSError:
        pass
