"""
stream_hub.py
=============
Relays live model output (llm_stream fragments) to the browser.

publish() appends an event to a bounded in-memory buffer; subscribers (the
GET /api/stream server-sent-events endpoint) wait on a condition and receive
everything newer than the last sequence number they saw, with consecutive
text fragments of the same call and channel merged into one message.

Events
  start    {call, step, model}                      a model call began
  delta    {call, channel: content|reasoning|tool, text, name?}
  restart  {call, reason}                           the same call is being re-asked (e.g. cut off)
  end      {call, finish, has_tools, aborted?}      the reply is complete
Nothing here is persisted. The running call's answer text is also kept on its own (take_stopped()), so a reply
the user stops halfway keeps what it had written (agent/loop._stopped_response, 1.4.2).
"""
from __future__ import annotations

import threading
import time
from collections import deque

import privacy

MAX_EVENTS = 6000


class StreamHub:
    def __init__(self):
        self._cond = threading.Condition()
        self._events = deque(maxlen=MAX_EVENTS)
        self._seq = 0
        self._live = {"call": None, "text": []}      # the running call's answer text so far
        self._stopped = ""                           # what a stopped call had written (take_stopped)

    @property
    def seq(self) -> int:
        return self._seq

    def publish(self, kind: str, **data):
        temp = privacy.current()
        with self._cond:
            self._seq += 1
            self._events.append({"seq": self._seq, "kind": kind, "t": time.time(), **data, **({"temp": temp} if temp else {})})
            if kind in ("start", "restart"):
                self._live, self._stopped = {"call": data.get("call"), "text": []}, ""
            elif kind == "delta" and data.get("channel") == "content" and data.get("call") == self._live["call"]:
                self._live["text"].append(data.get("text") or "")
            elif kind == "end" and data.get("call") == self._live["call"]:
                if data.get("finish") == "stopped":
                    self._stopped = "".join(self._live["text"])
                self._live = {"call": None, "text": []}
            self._cond.notify_all()

    def take_stopped(self) -> str:
        """The answer text a call had streamed when the user stopped it ("" if none), once."""
        with self._cond:
            text, self._stopped = self._stopped, ""
            return text

    def since(self, seq: int, timeout: float = 15.0) -> tuple[list, int]:
        """Events newer than `seq` (waits up to `timeout` for some), merged.
        Returns (events, newest_seq)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= seq:
                left = deadline - time.monotonic()
                if left <= 0:
                    return [], self._seq
                self._cond.wait(left)
            fresh = []
            for e in reversed(self._events):
                if e["seq"] <= seq:
                    break
                fresh.append(e)
            fresh.reverse()
            newest = self._seq
        return merge(fresh), newest

    def forget(self, chat_id: str) -> int:
        """A temporary chat closed: drop what it streamed. Returns how many events."""
        with self._cond:
            before = len(self._events)
            self._events = deque((e for e in self._events if e.get("temp") != chat_id), maxlen=MAX_EVENTS)
            return before - len(self._events)


def merge(events: list) -> list:
    """Collapse runs of text deltas for the same call/channel/tool index."""
    out = []
    texts = []
    for e in events:
        prev = out[-1] if out else None
        if (prev and e["kind"] == "delta" and prev["kind"] == "delta" and prev.get("call") == e.get("call")
                and prev.get("channel") == e.get("channel") and prev.get("index") == e.get("index")
                and prev.get("temp") == e.get("temp")):
            texts.append(e.get("text") or "")
            prev["seq"] = e["seq"]
            if "n" in e or "n" in prev:              # tokens in the merged run, for the live tok/s (1.4.7)
                prev["n"] = int(prev.get("n") or 0) + int(e.get("n") or 0)
            if e.get("name"):
                prev["name"] = e["name"]
        else:
            if texts:
                prev["text"] = "".join(texts)
            out.append(dict(e))
            texts = [e.get("text") or ""] if e["kind"] == "delta" else []
    if texts:
        out[-1]["text"] = "".join(texts)
    return out


hub = StreamHub()
privacy.on_discard(hub.forget)
