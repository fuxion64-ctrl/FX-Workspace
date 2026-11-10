"""
turns.py
========
Runs each chat request ("turn") on its own background thread.

Before this, POST /api/chat held one HTTP request open for the whole task -
often minutes. Refreshing the page mid-task lost the answer, the new page
didn't know the server was busy (so it showed no Stop button and the next
message failed with "Busy."), and every running task pinned a web worker.

Now:
  POST /api/chat           -> starts the turn, returns its id immediately
  GET  /api/turn/{id}?wait -> long-polls until it finishes (or `wait` s pass)
  GET  /api/state          -> includes the current/last turn, so a reloaded
                              page can reattach to a running task or pick up
                              an answer it missed.

Only one turn runs at a time; the manager refuses a second start atomically,
so two quick sends can never both slip past a check-then-act race.
"""
from __future__ import annotations

import threading
import time
import uuid


class TurnManager:
    def __init__(self, runner, keep: int = 8):
        """runner(request) -> response dict. It may raise; the error becomes
        the turn's result instead of killing the thread."""
        self._runner = runner
        self._lock = threading.Lock()
        self._turns: dict[str, dict] = {}
        self._order: list[str] = []
        self._events: dict[str, threading.Event] = {}
        self._keep = keep
        self._current: str | None = None

    def start(self, request, meta: dict | None = None, on_start=None, check=None) -> dict | None:
        """Start a turn. Returns its public record, or None if one is running. on_start() runs under the lock once
        the turn is accepted - for state that belongs to this turn only, so a refused second start can't change it
        (the phone/PC origin was set before the check: two sends at once could swap it - 1.4.6).
        check() also runs under the lock, before the turn is accepted (1.4.8): what it returns (a reply for the
        sender - the message was written for another chat, something else holds the chat) refuses the start, as
        {"refused": reply}. Under the lock, nothing can start or switch chats between the check and the turn."""
        with self._lock:
            if self._current and self._turns[self._current]["status"] == "running":
                return None
            if check is not None:
                refused = check()
                if refused:
                    return {"refused": refused}
            if on_start is not None:
                on_start()
            turn_id = uuid.uuid4().hex[:12]
            turn = {"id": turn_id, "status": "running", "started": time.time(), "ended": None,
                    "result": None, **(meta or {})}
            self._turns[turn_id] = turn
            self._events[turn_id] = threading.Event()
            self._order.append(turn_id)
            self._current = turn_id
            while len(self._order) > self._keep:
                old = self._order.pop(0)
                self._turns.pop(old, None)
                self._events.pop(old, None)
            public = dict(turn)
        threading.Thread(target=self._run, args=(turn_id, request), daemon=True,
                         name=f"turn-{turn_id}").start()
        return public

    def _run(self, turn_id: str, request):
        try:
            result = self._runner(request)
        except BaseException as e:          # never lose a turn to an exception
            result = {"status": "error", "message": f"Turn failed: {str(e)[:300]}"}
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is not None:
                turn.update(status="done", result=result, ended=time.time())
            done = self._events.get(turn_id)
        if done:
            done.set()

    def get(self, turn_id: str) -> dict | None:
        with self._lock:
            turn = self._turns.get(turn_id)
            return dict(turn) if turn else None

    def wait(self, turn_id: str, timeout: float) -> dict | None:
        """Block until the turn finishes or `timeout` seconds pass."""
        with self._lock:
            done = self._events.get(turn_id)
        if done is None:
            return None
        if timeout > 0:
            done.wait(timeout)
        return self.get(turn_id)

    def current(self) -> dict | None:
        """The running turn, or the most recent finished one."""
        with self._lock:
            if not self._current:
                return None
            return dict(self._turns.get(self._current) or {}) or None

    def running(self) -> bool:
        with self._lock:
            return bool(self._current and self._turns.get(self._current, {}).get("status") == "running")

    def forget(self, where) -> int:
        """Drop finished turns whose record matches where(turn) (a closed temporary
        chat's: they hold its messages). A running one is kept. Returns how many."""
        with self._lock:
            gone = [t for t, turn in self._turns.items() if turn.get("status") != "running" and where(turn)]
            for t in gone:
                self._turns.pop(t, None)
                self._events.pop(t, None)
                if t in self._order:
                    self._order.remove(t)
                if self._current == t:
                    self._current = None
            return len(gone)
