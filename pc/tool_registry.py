"""
tool_registry.py
================
Typed, discoverable metadata for every tool the agent can call.

The OpenAI-style schemas in tools/schemas.py TOOLS stay the single source of each tool's
name, description and input schema (they are what the model sees). This
module layers the execution policy on top of them:

  read_only / mutating, retryable, idempotent, timeout, cooldown,
  risk tier and whether that tier needs approval by default, required
  permissions, and the normalized output schema every tool result follows.

ToolExecutor (tool_executor.py) enforces these policies; the Settings UI and
/api/tools read them.
"""
from __future__ import annotations

import threading

from dataclasses import dataclass, field, asdict

# Every tool returns this shape after normalization by ToolExecutor.
RESULT_STATUSES = ("success", "retryable_failure", "permanent_failure",
                   "approval_required", "cancelled", "unavailable")
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(RESULT_STATUSES)},
        "ok": {"type": "boolean"},
        "message": {"type": "string"},
        "data": {},
        "verified": {"type": "boolean"},
        "attempts": {"type": "integer"},
        "duration_ms": {"type": "integer"},
        "tool_call_id": {"type": "string"},
        "task_id": {"type": "string"},
    },
    "required": ["status", "ok", "message"],
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    tier: str = "benign"                 # benign | input | destructive (permissions.py)
    read_only: bool = False
    retryable: bool = False              # safe to retry automatically after a transient failure
    idempotent: bool = False             # running it twice has the same effect as once
    timeout_s: float = 60.0              # hard limit when run isolated; advisory otherwise
    cooldown_s: float = 0.0              # minimum spacing between two calls of this tool
    isolate: bool = False                # run on a worker thread so the timeout is enforced
    permissions: tuple = ()              # e.g. ("messages.send",)
    category: str = "general"
    undoable: bool = False               # changes are journaled (undo_store) and can be undone
    output_schema: dict = field(default_factory=lambda: RESULT_SCHEMA)

    @property
    def mutating(self) -> bool:
        return not self.read_only

    @property
    def approval_by_default(self) -> bool:
        """Whether this tool's tier asks for approval under default settings.
        The live decision is permissions.decide(), which honours the user's
        mode, category defaults and per-tool overrides."""
        return self.tier != "benign"

    def public(self) -> dict:
        d = asdict(self)
        d.pop("output_schema", None)
        d.update(mutating=self.mutating, approval_by_default=self.approval_by_default)
        return d


# ------------------------------------------------------------------------------
# Execution policy per tool. Anything not listed gets a default derived from
# whether it is read-only: reads are retryable + idempotent, writes are
# neither (a side effect is never repeated blindly).
# ------------------------------------------------------------------------------
_POLICY = {
    # Reads that talk to flaky things - worth an automatic retry
    "get_url_status": dict(isolate=True, timeout_s=20, category="web"),
    "ping_host": dict(isolate=True, timeout_s=60, category="network"),
    "port_check": dict(isolate=True, timeout_s=20, category="network"),
    "network_info": dict(isolate=True, timeout_s=20, category="network"),
    "list_services": dict(isolate=True, timeout_s=30, category="system"),
    "list_scheduled_tasks": dict(isolate=True, timeout_s=30, category="system"),
    "search_files": dict(isolate=True, timeout_s=90, category="files"),
    "compare_files": dict(isolate=True, timeout_s=90, category="files"),
    "top_processes": dict(isolate=True, timeout_s=20, cooldown_s=1.0, category="system"),
    # Screen / UI inspection: cheap to repeat but pointless in a tight loop
    "inspect_screen": dict(cooldown_s=1.0, timeout_s=20, category="inspection"),
    "list_elements": dict(cooldown_s=0.5, timeout_s=30, category="inspection"),
    "observe": dict(cooldown_s=0.5, category="inspection"),
    "wait_for_window": dict(timeout_s=65, category="inspection"),
    # Mutations
    "execute_command": dict(timeout_s=75, category="system"),
    "run_powershell": dict(timeout_s=310, category="system"),
    "run_python": dict(timeout_s=75, category="system"),
    # File changes: journaled by undo_store before they happen (see ToolExecutor).
    # Inline (not isolated): the download checks Stop and its own 300 s limit
    # between chunks and closes the file before returning, so the undo
    # rollback never races a still-writing worker thread.
    "download_file": dict(timeout_s=300, category="web", undoable=True),
    "write_file": dict(category="files", undoable=True),
    "edit_file": dict(category="files", undoable=True),
    "apply_patch": dict(category="files", undoable=True),
    "copy_path": dict(category="files", undoable=True),
    "move_path": dict(category="files", undoable=True),
    "rename_path": dict(category="files", undoable=True),
    "delete_path": dict(category="files", undoable=True),
    "zip_paths": dict(category="files", undoable=True),
    "unzip_file": dict(category="files", undoable=True),
    "screenshot_to_file": dict(category="files", undoable=True),
    "open_url": dict(idempotent=True, cooldown_s=1.0, category="web"),
    "make_dir": dict(idempotent=True, category="files", undoable=True),
    "set_clipboard": dict(idempotent=True, category="clipboard"),
    "media_control": dict(cooldown_s=0.5, category="media"),
    "snap_windows": dict(idempotent=True, category="windows"),
    "send_notification": dict(cooldown_s=2.0, category="productivity"),
    # data/memory.json is journaled before the entry is added, so Undo removes it
    "save_memory": dict(category="memory", undoable=True),
    "roblox_set_property": dict(idempotent=True, category="roblox"),
    # Script edits are journaled (undo_store "external" ops: the old source is
    # kept here) and written compare-and-set, so they are never retried: a
    # retry after a write that did land would be refused as "changed since read".
    "roblox_set_script_source": dict(category="roblox", undoable=True),
    "roblox_edit_script": dict(category="roblox", undoable=True),
    "roblox_create_script": dict(category="roblox", undoable=True),
    "roblox_create_instance": dict(category="roblox"),
    "roblox_delete_instance": dict(category="roblox"),
    # Up to 200 changes, each a round trip to the plugin.
    "roblox_batch": dict(category="roblox", timeout_s=240),
    # Waits for the playtest itself (up to 120 s) - the bridge enforces its own deadline.
    "roblox_run_playtest": dict(category="roblox", timeout_s=200),
    "roblox_get_tree": dict(category="roblox", timeout_s=30),
    # Web (web_tools.py) and the assistant's browser (browser.py)
    # Room for the fallbacks: pacing, then the assistant's browser (and, for
    # web_read, the Wayback Machine) when a site refuses a direct request.
    "web_search": dict(isolate=True, timeout_s=90, cooldown_s=1.0, category="web"),      # up to 4 queries
    "web_read": dict(isolate=True, timeout_s=75, category="web"),
    # A search, then up to 5 pages and their pictures (web_images.py stops itself after 90 s).
    "show_web_images": dict(isolate=True, timeout_s=120, category="web"),
    # Up to 6 pages in parallel (web_tools.read_pages gives up on them after 45 s).
    "fetch_page": dict(isolate=True, timeout_s=90, category="web"),
    # Reads the local store; a stale feed is refreshed first (CelesTrak's orbit files can take ~100 s cold).
    "world_monitor": dict(isolate=True, timeout_s=150, category="web"),
    # Loading an image model the first time takes a while; Krea 2 at 2K is slow on an 8 GB card. Not retried.
    "generate_image": dict(isolate=True, timeout_s=900, category="media", retryable=False, idempotent=False),
    # stats_tools.py and auto_tools.py: computed here, nothing outside touched.
    "compute_stats": dict(timeout_s=30, category="data"),
    "make_chart": dict(timeout_s=30, category="data"),
    "use_tools": dict(timeout_s=10, category="models"),
    "load_skill": dict(timeout_s=10, category="models"),
    "browser_open": dict(timeout_s=60, category="browser"),
    "browser_read": dict(timeout_s=30, category="browser"),
    "browser_click": dict(timeout_s=30, category="browser"),
    "browser_fill": dict(timeout_s=30, category="browser"),
    # Messaging (messaging.py) - send is deliberately rate limited
    "messages_list_conversations": dict(category="messaging", permissions=("messages.read",)),
    "messages_search": dict(category="messaging", permissions=("messages.read",)),
    "messages_read": dict(category="messaging", permissions=("messages.read",)),
    "messages_draft": dict(category="messaging", permissions=("messages.draft",)),
    "messages_send": dict(category="messaging", cooldown_s=5.0, permissions=("messages.send",)),
    # Notes & tasks and Docs (notes.py, docs_store.py): the app's own files under data/
    "notes_search": dict(category="notes"),
    "note_save": dict(category="notes"),
    "task_add": dict(category="notes"),
    "task_done": dict(idempotent=True, category="notes"),
    "doc_list": dict(category="docs"),
    "doc_read": dict(category="docs"),
    "doc_write": dict(category="docs"),
    # Skills, knowledge base, earlier chats (agent_skills.py, knowledge.py, profile_store.py)
    "skill_load": dict(category="skills"),
    "skill_save": dict(category="skills"),
    "kb_search": dict(category="knowledge", timeout_s=30),
    "chats_search": dict(category="memory", timeout_s=30),
    # Model routing (model_router.py)
    "request_model_handoff": dict(category="models", cooldown_s=30.0, permissions=("models.switch",)),
    # MCP (mcp_client.py). Every mcp__<server>__<tool> gets MCP_TOOL_POLICY.
    "mcp_read_resource": dict(isolate=True, timeout_s=60, category="mcp"),
}
# Tools of connected MCP servers: run on a worker thread with a hard timeout
# (the client also cancels on Stop); never retried (a call may have acted).
MCP_TOOL_POLICY = dict(isolate=True, timeout_s=130, category="mcp", retryable=False, idempotent=False)


class ToolRegistry:
    """Read on every call and changed from MCP threads (tools appearing and
    disappearing as servers connect), so changes and snapshots are locked."""

    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._schemas: dict[str, dict] = {}
        self._lock = threading.RLock()

    def register(self, spec: ToolSpec, openai_schema: dict | None = None):
        with self._lock:
            self._specs[spec.name] = spec
            if openai_schema is not None:
                self._schemas[spec.name] = openai_schema

    def unregister(self, name: str):
        with self._lock:
            self._specs.pop(name, None)
            self._schemas.pop(name, None)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> set:
        with self._lock:
            return set(self._specs)

    def specs(self) -> list:
        with self._lock:
            return [self._specs[n] for n in sorted(self._specs)]

    def openai_schema(self, name: str) -> dict | None:
        return self._schemas.get(name)

    def describe(self) -> list:
        return [s.public() for s in self.specs()]


def build_registry(tools: list, read_only: set, classify) -> ToolRegistry:
    """Build the registry from OpenAI tool schemas. `read_only` is the set of
    tools with no side effects; `classify(name)` returns the risk tier."""
    reg = ToolRegistry()
    for tool in tools:
        fn = tool["function"]
        name = fn["name"]
        ro = name in read_only
        policy = {"retryable": ro, "idempotent": ro, **_POLICY.get(name, {})}
        reg.register(ToolSpec(
            name=name, description=fn.get("description", ""),
            input_schema=fn.get("parameters") or {"type": "object", "properties": {}},
            tier=classify(name), read_only=ro, **policy), tool)
    return reg


# ------------------------------------------------------------------------------
# Argument validation - a small JSON-schema subset (the one these schemas
# use), no dependency. Coerces the mistakes small models make constantly:
# "12" for an integer, "true" for a boolean, a lone string for an array.
# ------------------------------------------------------------------------------
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict}


def _coerce(value, typ):
    if typ == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    if typ == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if typ == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if typ == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    if typ == "array" and isinstance(value, str):
        return [value]
    if typ == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


def validate_args(schema: dict, args) -> tuple[dict, list]:
    """Returns (coerced_args, errors). Unknown extra keys are kept (models add
    'justification' and similar) but never required."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {}, [f"arguments must be an object, got {type(args).__name__}"]
    props = (schema or {}).get("properties") or {}
    out, errors = dict(args), []
    for name in (schema or {}).get("required") or []:
        value = out.get(name)
        # An empty string counts as missing, except for free text a user may
        # legitimately want empty (type_text's text).
        if value is None or (isinstance(value, str) and not value.strip() and name not in ("text", "value")):
            errors.append(f"missing required argument '{name}'")
    for name, rule in props.items():
        if name not in out or out[name] is None:
            continue
        typ = rule.get("type")
        if typ in _TYPES:
            out[name] = _coerce(out[name], typ)
            ok = isinstance(out[name], _TYPES[typ]) and not (typ in ("integer", "number") and isinstance(out[name], bool))
            if not ok:
                errors.append(f"'{name}' must be {typ}, got {type(out[name]).__name__}")
                continue
        if "enum" in rule and out[name] not in rule["enum"]:
            errors.append(f"'{name}' must be one of {rule['enum']}")
        if typ == "array":
            if "minItems" in rule and len(out[name]) < rule["minItems"]:
                errors.append(f"'{name}' needs at least {rule['minItems']} items")
            if "maxItems" in rule and len(out[name]) > rule["maxItems"]:
                out[name] = out[name][:rule["maxItems"]]
    return out, errors
