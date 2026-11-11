"""
chat_store.py
=============
Persistent chats: one JSON file per chat in data/chats/<id>.json.

    {"version": 1, "id": "...", "title": "...", "project": "<project id>" | null,
     "created": <epoch>, "updated": <epoch>,
     "transcript":   [{"r": "u"|"a"|"e"|"s", "t": text, "v": bool?, "s": [steps]?,
                       "c": "plan"|"design"?, "k": thinking?, "kt": seconds?}, ...],
     "conversation": [model messages, system prompt excluded, images removed]}

  * transcript   - what the chat shows (user / assistant / error / stopped
                   entries, with each reply's compact step list; a message's
                   slash command "c"; a reply's reasoning "k" and how long the
                   model thought "kt" - the "Thought for Xs" row).
  * conversation - what the model sees when the chat is reopened, so a chat
                   continues where it left off after a restart. Screenshots
                   are dropped (they are large, and stale once reopened).

Deleting a chat moves its file to data/chats/trash/ - nothing is destroyed.
Search matches every word of the query against the title and the transcript.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
import atomic_io
import privacy
import history

MAX_TRANSCRIPT = 400
# Kept from the saved file when a save doesn't mention them (the open chat is
# saved from its in-memory copy, which doesn't carry them).
KEEP_KEYS = ("pinned", "archived", "branched_from", "guard",     # guard: strikes / lock (chat_guard.py)
             "tools_off",                                       # tool groups switched off for this chat (tools/schemas)
             "sandboxed")                                       # files only in a scratch folder, restricted commands (sandbox.py)
MAX_CONVERSATION = 300
_ID = re.compile(r"^[a-f0-9]{12}$")


def _strip_images(message: dict) -> dict:
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                parts.append({"type": "text", "text": "[screenshot not kept]"})
            else:
                parts.append(part)
        message = {**message, "content": parts}
    return message


def _key(message: dict) -> str:
    return json.dumps(message, ensure_ascii=False, sort_keys=True, default=str)


def _keep_older(old: dict | None, view: list, start=None) -> tuple:
    """(1.4.8) The conversation to store, and where the model's view starts in it (conversation_start).
    The open chat's conversation in memory is only what the model sees: fitting (history.fit) and compression drop
    its oldest messages, and saving it used to drop them from the file too. The file keeps its older messages in
    front: the view is matched against the stored list from the old view's start, and whatever came before the
    match stays. A caller that already gives the whole list passes conversation_start itself (start)."""
    if start is not None:                                 # the whole conversation was given, with its view start
        return view, max(0, min(int(start), len(view)))
    if not old:
        return view, 0
    stored = [m for m in (old.get("conversation") or []) if isinstance(m, dict)]
    from_ = max(0, min(int(old.get("conversation_start") or 0), len(stored)))
    if not view:
        return stored[:from_], from_
    first, keys = _key(view[0]), None
    for p in range(from_, len(stored)):
        if _key(stored[p]) != first:
            continue
        keys = keys or [_key(m) for m in view]
        tail = [_key(m) for m in stored[p:p + len(view)]]
        if tail == keys[:len(tail)]:
            return stored[:p] + view, p
    return stored[:from_] + view, from_                   # the view no longer matches: keep what was before it


def title_from(text: str) -> str:
    line = next((l.strip() for l in str(text or "").splitlines() if l.strip()), "")
    return (line[:70] + "…") if len(line) > 70 else (line or "New chat")


class ChatStore:
    def __init__(self, root: str):
        self.root = root
        self.trash = os.path.join(root, "trash")
        self._lock = threading.RLock()
        self._summary: dict = {}          # id -> (mtime, summary, folded text, text)

    # ---------------------------------------------------------------- files
    def _path(self, chat_id: str) -> str:
        if not _ID.match(chat_id or ""):
            raise KeyError(chat_id)
        return os.path.join(self.root, f"{chat_id}.json")

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def load(self, chat_id: str) -> dict | None:
        try:
            with open(self._path(chat_id), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) and data.get("id") == chat_id else None
        except (OSError, ValueError, KeyError):
            return None

    def save(self, chat: dict, touch: bool = True) -> dict:
        """touch=False keeps "updated" (pinning or archiving is not activity).
        A temporary chat (privacy.py) is refused, and so is any save while a
        temporary chat's work runs: nothing of one can land in a saved chat."""
        with self._lock:
            if privacy.is_temporary(chat):
                raise privacy.TemporaryChatError("A temporary chat is never saved to history.")
            privacy.refuse("A chat")
            chat = dict(chat)
            if any(k not in chat for k in KEEP_KEYS):
                old = self.load(chat.get("id") or "") or {}
                chat.update({k: old[k] for k in KEEP_KEYS if k not in chat and k in old})
            chat["version"] = 1
            if touch or not chat.get("updated"):
                chat["updated"] = time.time()
            chat.setdefault("created", chat["updated"])
            # (1.4.8, audit 2.8) Every transcript entry and conversation message is kept: the last 400 / 300 were, and
            # older ones were deleted from the file for good. The model's view is fitted when a request is built.
            chat["transcript"] = list(chat.get("transcript") or [])
            view = [_strip_images(m) for m in (chat.get("conversation") or [])
                    if isinstance(m, dict) and m.get("role") != "system"]
            old = self.load(chat.get("id") or "") if "conversation_start" not in chat else None
            chat["conversation"], chat["conversation_start"] = _keep_older(old, view, chat.get("conversation_start"))
            path = self._path(chat["id"])
            os.makedirs(self.root, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(chat, f, ensure_ascii=False, default=str)
            atomic_io.replace(tmp, path)
            return chat

    def delete(self, chat_id: str) -> bool:
        """Moves the chat to data/chats/trash/ (recoverable by hand)."""
        try:
            path = self._path(chat_id)
        except KeyError:
            return False
        with self._lock:
            if not os.path.exists(path):
                return False
            os.makedirs(self.trash, exist_ok=True)
            os.replace(path, os.path.join(self.trash, f"{chat_id}-{int(time.time())}.json"))
            self._summary.pop(chat_id, None)
        return True

    def rename(self, chat_id: str, title: str) -> dict | None:
        with self._lock:
            chat = self.load(chat_id)
            if chat is None:
                return None
            chat["title"] = title_from(title)
            return self.save(chat)

    def set_flags(self, chat_id: str, pinned: bool | None = None, archived: bool | None = None) -> dict | None:
        """Pinned chats list first; archived ones are listed apart (the sidebar's Archived)."""
        with self._lock:
            chat = self.load(chat_id)
            if chat is None:
                return None
            for key, value in (("pinned", pinned), ("archived", archived)):
                if value is not None:
                    chat[key] = bool(value)           # False, not removed: save() would bring the old value back
            return self.save(chat, touch=False)

    def set_tools_off(self, chat_id: str, groups: list) -> dict | None:
        """The tool groups switched off for this chat (tools/schemas.tool_group); an empty list is kept, not
        removed - save() would bring the old value back."""
        with self._lock:
            chat = self.load(chat_id)
            if chat is None:
                return None
            chat["tools_off"] = sorted({g for g in groups if isinstance(g, str) and g})
            return self.save(chat, touch=False)

    def set_project(self, chat_id: str, project: str | None) -> dict | None:
        """Move a chat into a project (None: out of every project)."""
        with self._lock:
            chat = self.load(chat_id)
            if chat is None:
                return None
            chat["project"] = project or None
            return self.save(chat)

    # --------------------------------------------------------------- listing

    def _summaries(self) -> list:
        try:
            names = [n for n in os.listdir(self.root) if n.endswith(".json")]
        except OSError:
            return []
        out = []
        for name in names:
            chat_id = name[:-5]
            if not _ID.match(chat_id):
                continue
            path = os.path.join(self.root, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            cached = self._summary.get(chat_id)
            if not cached or cached[0] != mtime:
                chat = self.load(chat_id)
                if chat is None:
                    continue
                transcript = chat.get("transcript") or []
                text = " ".join(str(e.get("t") or "") for e in transcript)
                last = next((str(e.get("t") or "") for e in reversed(transcript) if e.get("r") == "a"), "")
                summary = {"id": chat_id, "title": chat.get("title") or "New chat", "project": chat.get("project"),
                           "created": chat.get("created"), "updated": chat.get("updated"),
                           "turns": sum(1 for e in transcript if e.get("r") == "u"), "last_reply": last[:300],
                           "pinned": bool(chat.get("pinned")), "archived": bool(chat.get("archived")),
                           # The sidebar's status dot: a message saved as pending was never answered (the app stopped
                           # mid-reply), or the last thing in the chat is an error.
                           "pending": bool(chat.get("pending")),
                           "last_error": bool(transcript) and isinstance(transcript[-1], dict) and transcript[-1].get("r") == "e"}
                full = summary["title"] + " " + text
                cached = (mtime, summary, full.casefold(), full)
                self._summary[chat_id] = cached
            out.append(cached)
        return out

    def list(self, project: str | None = "*", query: str = "", limit: int = 200) -> list:
        """Newest first. project="*" -> all chats, None -> chats in no project,
        an id -> that project's chats. query -> every word must appear."""
        words = [w for w in re.findall(r"\w+", (query or "").casefold()) if w]
        rows = []
        for _mtime, summary, text, full in self._summaries():
            if project != "*" and summary.get("project") != project:
                continue
            if words and not all(w in text for w in words):
                continue
            row = dict(summary)
            if words:
                i = text.find(words[0])
                row["snippet"] = full[max(0, i - 60): i + 120].strip()
            rows.append(row)
        rows.sort(key=lambda r: (bool(r.get("pinned")), r.get("updated") or 0), reverse=True)     # pinned first
        return rows[:limit]
