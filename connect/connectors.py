"""
connectors.py
=============
One-click connectors (Settings > Connections): well-known MCP servers, set up with a click and, where needed, one
key - the way Claude's Customize > Connectors works. Each becomes an ordinary MCP server in mcp_client's hub, so
everything already true of MCP stays true: its tools ask before each use (permissions.classify: every mcp__ tool
is "destructive" until you set a rule), and the MCP tab shows and edits it.

  kind "npx"   a Node package run with npx (needs Node.js - the card says so when it's missing)
  kind "pip"   a Python package installed into the app's own environment, run with `python -m`
  kind "http"  a remote MCP endpoint, with its key sent as a header

Only official or first-party servers are listed; the catalog is code, not a download, so no one can add to it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading

CATALOG = [
    {"id": "filesystem", "name": "Folders", "about": "Lets the assistant read and edit files in folders you choose.",
     "kind": "npx", "package": "@modelcontextprotocol/server-filesystem", "icon": "i-folder",
     "fields": [{"key": "folders", "label": "Folders it may use (one per line)", "type": "folders", "to": "args", "required": True}]},
    {"id": "fetch", "name": "Web pages (Fetch)", "about": "Reads web pages as Markdown.",
     "kind": "pip", "package": "mcp-server-fetch", "module": "mcp_server_fetch", "icon": "i-globe", "fields": []},
    {"id": "time", "name": "Time and time zones", "about": "Current time and conversions between time zones.",
     "kind": "pip", "package": "mcp-server-time", "module": "mcp_server_time", "icon": "i-clock", "fields": []},
    {"id": "git", "name": "Git", "about": "Reads and works with a Git repository: status, diffs, log, commits.",
     "kind": "pip", "package": "mcp-server-git", "module": "mcp_server_git", "icon": "i-branch",
     "fields": [{"key": "repository", "label": "Repository folder", "type": "folder", "to": "arg:--repository", "required": True}]},
    {"id": "github", "name": "GitHub", "about": "Issues, pull requests, code and repositories on GitHub.",
     "kind": "http", "url": "https://api.githubcopilot.com/mcp/", "icon": "i-code",
     "fields": [{"key": "token", "label": "GitHub personal access token", "type": "secret", "to": "header:Authorization:Bearer ",
                 "required": True, "help": "https://github.com/settings/personal-access-tokens"}]},
    {"id": "notion", "name": "Notion", "about": "Search, read and update your Notion pages and databases.",
     "kind": "npx", "package": "@notionhq/notion-mcp-server", "icon": "i-book",
     "fields": [{"key": "NOTION_TOKEN", "label": "Notion integration token", "type": "secret", "to": "env", "required": True,
                 "help": "https://www.notion.so/profile/integrations"}]},
    {"id": "brave", "name": "Brave Search", "about": "Web, news and image search through Brave.",
     "kind": "npx", "package": "@brave/brave-search-mcp-server", "icon": "i-search",
     "fields": [{"key": "BRAVE_API_KEY", "label": "Brave Search API key", "type": "secret", "to": "env", "required": True,
                 "help": "https://api-dashboard.search.brave.com/"}]},
    {"id": "playwright", "name": "Browser (Playwright)", "about": "Drives a browser - pages read as an accessibility tree.",
     "kind": "npx", "package": "@playwright/mcp", "icon": "i-screen", "fields": []},
    {"id": "context7", "name": "Library docs (Context7)", "about": "Up-to-date documentation and code examples for programming libraries.",
     "kind": "npx", "package": "@upstash/context7-mcp", "icon": "i-file", "fields": []},
    {"id": "homeassistant", "name": "Home Assistant", "about": "Your smart home, through Home Assistant's MCP Server integration (turn it on there first).",
     "kind": "http", "url": "http://homeassistant.local:8123/api/mcp", "icon": "i-home",
     "fields": [{"key": "url", "label": "Home Assistant MCP address", "type": "text", "to": "url", "required": True},
                {"key": "token", "label": "Long-lived access token", "type": "secret", "to": "header:Authorization:Bearer ", "required": True,
                 "help": "https://www.home-assistant.io/integrations/mcp_server/"}]},
]
BY_ID = {c["id"]: c for c in CATALOG}
_lock = threading.Lock()
_jobs: dict[str, dict] = {}            # id -> {"state": running|done|failed, "message"}


def server_name(cid: str) -> str:
    return f"fx-{cid}"


def node_ok() -> bool:
    return bool(shutil.which("npx"))


def _pip_installed(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def catalog(hub) -> list[dict]:
    out = []
    for c in CATALOG:
        srv = hub.servers.get(server_name(c["id"]))
        job = _jobs.get(c["id"]) or {}
        out.append({**{k: v for k, v in c.items() if k not in ("package", "module")},
                    "installed": srv is not None, "status": srv.status if srv else None, "error": srv.error if srv else None,
                    "needs": "Node.js" if c["kind"] == "npx" and not node_ok() else None,
                    "job": job.get("state"), "job_message": job.get("message")})
    return out


def build_config(c: dict, values: dict) -> dict:
    """The MCP server entry for this connector with the user's values (raises ValueError for a missing one)."""
    cfg: dict = {"enabled": True, "env": {}, "headers": {}}
    args: list = []
    for f in c["fields"]:
        v = (values.get(f["key"]) or "").strip()
        if not v and f.get("default"):
            v = f["default"]
        if not v:
            if f.get("required"):
                raise ValueError(f"Fill in: {f['label']}.")
            continue
        to = f["to"]
        if to == "env":
            cfg["env"][f["key"]] = v
        elif to == "url":
            cfg["url"] = v
        elif to.startswith("header:"):
            _, name, prefix = to.split(":", 2)
            cfg["headers"][name] = prefix + v
        elif to == "args":
            for line in v.splitlines():
                if line.strip():
                    if not os.path.isdir(line.strip()):
                        raise ValueError(f"There's no folder at {line.strip()}.")
                    args.append(os.path.abspath(line.strip()))
        elif to.startswith("arg:"):
            if f["type"] == "folder" and not os.path.isdir(v):
                raise ValueError(f"There's no folder at {v}.")
            args += [to[4:], v]
    if c["kind"] == "npx":
        cfg["command"], cfg["args"] = "npx", ["-y", c["package"]] + args
    elif c["kind"] == "pip":
        cfg["command"], cfg["args"] = sys.executable, ["-m", c["module"]] + args
    else:
        cfg.setdefault("url", c["url"])
    if not cfg["env"]:
        cfg.pop("env")
    if not cfg["headers"]:
        cfg.pop("headers")
    return cfg


def install(cid: str, values: dict, hub, log=None) -> dict:
    """Validates now; installs (pip) and connects on a thread. job() follows it."""
    c = BY_ID.get(cid)
    if not c:
        raise KeyError("No such connector.")
    if c["kind"] == "npx" and not node_ok():
        raise ValueError("This connector needs Node.js - install it from nodejs.org, then try again.")
    cfg = build_config(c, values)
    with _lock:
        if (_jobs.get(cid) or {}).get("state") == "running":
            raise ValueError("Already setting this one up.")
        _jobs[cid] = {"state": "running", "message": "Installing…" if c["kind"] == "pip" else "Connecting…"}

    def run():
        try:
            if c["kind"] == "pip" and not _pip_installed(c["module"]):
                r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", c["package"]],
                                   capture_output=True, text=True, timeout=600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if r.returncode != 0:
                    raise RuntimeError("pip couldn't install it: " + (r.stderr or r.stdout).strip().splitlines()[-1][:200])
            server = hub.upsert(server_name(cid), cfg)
            server = hub.connect(server_name(cid))
            ok = server.status == "connected"
            _jobs[cid] = {"state": "done" if ok else "failed",
                          "message": f"Connected - {len(getattr(server, 'tools', []) or [])} tools." if ok else (server.error or server.status)}
            if log:
                log(f"Connector {c['name']}: {_jobs[cid]['message']}")
        except Exception as e:
            _jobs[cid] = {"state": "failed", "message": str(e)[:300]}
    threading.Thread(target=run, name=f"connector-{cid}", daemon=True).start()
    return {"ok": True, "job": "running"}


def remove(cid: str, hub) -> bool:
    _jobs.pop(cid, None)
    return hub.remove(server_name(cid))
