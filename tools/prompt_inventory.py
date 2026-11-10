"""
tools/prompt_inventory.py
=========================
What one request to a local model is made of, per kind of turn (1.4.8 model-efficiency plan).

No model, no network, nothing written: the default settings (a temporary settings path), this folder's own data
(identity, memory, skills), and the app's own code path - intent_router.classify_turn, agent/tasks._begin_composer_tools,
agent/models.build_messages and tools/schemas.tools_for_current_turn. Tokens are the app's own estimate
(history.CHARS_PER_TOKEN = 3.6 characters a token), not a tokenizer's count.

    .venv\\Scripts\\python.exe tools\\prompt_inventory.py [--ctx 32768] [--tools]

--ctx: the local model's window (the tool list is smaller under 32k - schemas.LOCAL_FULL_TOOLS_MIN_CTX).
--tools: also list every tool schema's cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("AGENTIC_DISABLE_LLAMA_WATCHDOG", "1")
os.environ.setdefault("AUTOSTART_LLAMA", "0")
import core.modpath  # noqa: E402,F401

KINDS = [
    ("greeting", "hi there", {}),
    ("plain question", "why is the sky blue?", {}),
    ("web question", "what is the latest news about the James Webb telescope?", {}),
    ("picture", "make a picture of a red fox in the snow", {}),
    ("PC action", "open notepad and type hello", {}),
    ("Roblox action", "add a red part called Floor to the workspace in Roblox Studio", {"roblox": True}),
    ("voice", "what is the weather like on Mars", {"voice": True}),
]


def tok(text) -> int:
    import history
    if not isinstance(text, str):
        text = json.dumps(text, default=str)
    return round(len(text) / history.CHARS_PER_TOKEN)


def paragraphs(prompt: str) -> list:
    """The system prompt in its blank-line paragraphs, each named by its first words."""
    out = []
    for p in prompt.split("\n\n"):
        if p.strip():
            name = p.strip().split("\n")[0][:46].replace("|", "/")
            out.append((name, tok(p)))
    return out


def measure(message: str, opts: dict, ctx: int) -> dict:
    import core.config as config
    import core.state as shared
    import agent.models as llm
    import agent.prompts as prompts
    import agent.tasks as tasks
    import chat_guard
    import identity
    import intent_router
    import memory_graph
    import skills
    import system_messages
    import tools.schemas as schemas

    shared.app_state.update(
        conversation=[{"role": "system", "content": prompts.SYSTEM_PROMPT}], mode=config.MODE_LOCAL,
        current_user_request=message, raw_mode=False, design_mode=False, task_context="", context_summary="",
        turn_origin="pc", active_project=None, active_chat=None, local_model="", local_engaged=None)
    shared.app_state["settings"] = config.load_settings()
    shared.app_state["turn_policy"] = intent_router.classify_turn(message, False, None)
    shared.task_board.clear()
    req = SimpleNamespace(message=message, command=None, tools=[], files=[], voice=bool(opts.get("voice")),
                          voice_lang="en", persona=None, instructions="", search_mode=None)
    with mock.patch("image_gen.warm"):
        tasks._begin_composer_tools(req)
    shared.app_state["turn_system_message"] = system_messages.clean_settings(shared.app_state["settings"])

    offered = schemas.tools_for_current_turn(local=True)
    names = [t["function"]["name"] for t in offered]
    msgs = llm.build_messages()
    system, last = msgs[0]["content"], msgs[-1]["content"]

    selected = system_messages.clean_settings(shared.app_state["settings"])
    base = prompts.for_tools(system_messages.resolve(selected, True, None), names, schemas.TOOL_NAMES)
    studio = prompts.studio_rules(names)
    soul, overview = identity.block(), memory_graph.overview_block()

    from context import render_live_block
    live = render_live_block(shared.task_board, shared.ledger, {
        "mode": config.MODE_LOCAL, "model": None, "pywinauto": False, "shot_meta": None, "shot_age": 0, "activity": ""})
    tool_line = next((l for l in last.split("\n") if l.startswith("AVAILABLE TOOLS THIS TURN")), "")
    more = skills.index(names) or ""
    head = last.split("\n\n" + live)[0] if live and live in last else last.split("\n\n")[0]
    tail = last[last.rfind("\n\n") + 2:] if "\n\n" in last else ""

    groups: dict = {}
    for t in offered:
        g = schemas.tool_group(t["function"]["name"])
        groups[g] = groups.get(g, 0) + schemas._cost([t])
    text = message.strip()
    guard = not (len(text) < chat_guard.MIN_CHECK_CHARS) and not intent_router.is_greeting(text)
    return {
        "policy": shared.app_state["turn_policy"].get("mode"), "turn_tools": sorted(shared.app_state.get("turn_tools") or []),
        "names": names, "groups": groups, "tool_costs": {t["function"]["name"]: schemas._cost([t]) for t in offered},
        "tools_total": schemas._cost(offered), "system": tok(system), "base": tok(base), "base_parts": paragraphs(base),
        "studio": tok(studio) if studio else 0, "identity": tok(soul) if soul else 0, "memory": tok(overview) if overview else 0,
        "last": tok(last), "header": tok(head), "live": tok(live), "tool_line": tok(tool_line), "skills_index": tok(more),
        "instruction": tok(tail), "guard_call": guard,
        "prefix": json.dumps(offered, default=str) + "\n" + system,
        "sys_hash": hashlib.sha1(system.encode()).hexdigest()[:8],
        "tools_hash": hashlib.sha1(json.dumps(names).encode()).hexdigest()[:8],
        "roblox_mentions": system.lower().count("roblox") + last.lower().count("roblox"),
    }


def shared_prefix_tokens(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return tok(a[:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tools", action="store_true")
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="prompt-inventory-")
    import core.config as config
    import llama_local
    import core.observer as observer
    import roblox_bridge
    rows = []
    with mock.patch.object(config, "SETTINGS_PATH", os.path.join(tmp, "settings.json")), \
            mock.patch.object(llama_local, "model_context_limit", return_value=args.ctx), \
            mock.patch.object(observer, "_activity_line", return_value=""), \
            mock.patch("events.log_action"), mock.patch("events.emit_event", return_value={}), \
            mock.patch("events.emit_activity"):
        for kind, message, opts in KINDS:
            with mock.patch.object(roblox_bridge, "is_connected", return_value=bool(opts.get("roblox"))):
                rows.append((kind, message, measure(message, opts, args.ctx)))

    print(f"# Prompt inventory - local model, {args.ctx:,}-token window, default settings\n")
    print("Tokens are the app's estimate (3.6 characters a token). One request = tool schemas + system message + the "
          "chat + the last user message.\n")
    print("| Turn | Mode | Tools on | Tool schemas | System msg | of which: base / Studio rules / identity / memory | "
          "Last msg | of which: header / live state / tool-name line / skills index / closing line | Total (no chat) | "
          "Message check call |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for kind, _msg, r in rows:
        total = r["tools_total"] + r["system"] + r["last"]
        print(f"| {kind} | {r['policy']} | {len(r['names'])} | {r['tools_total']:,} | {r['system']:,} | "
              f"{r['base']:,} / {r['studio']:,} / {r['identity']:,} / {r['memory']:,} | {r['last']:,} | "
              f"{r['header']:,} / {r['live']:,} / {r['tool_line']:,} / {r['skills_index']:,} / {r['instruction']:,} | "
              f"{total:,} | {'yes' if r['guard_call'] else 'no'} |")

    print("\n## Tool groups per turn (tokens)\n")
    groups = sorted({g for _k, _m, r in rows for g in r["groups"]})
    print("| Group | " + " | ".join(k for k, _m, _r in rows) + " |")
    print("|---|" + "---|" * len(rows))
    for g in groups:
        print(f"| {g} | " + " | ".join(str(r["groups"].get(g, "")) for _k, _m, r in rows) + " |")

    print("\n## Base system prompt, paragraph by paragraph (the plain question's)\n")
    print("| Paragraph | Tokens |")
    print("|---|---|")
    q = dict((k, r) for k, _m, r in rows)["plain question"]
    for name, n in q["base_parts"]:
        print(f"| {name} | {n} |")

    print("\n## What stays the same from one kind of turn to the next (llama.cpp's prompt cache)\n")
    print("Order as a Qwen-style template sends it: tool schemas, then the system message. 'Shared start' is how much "
          "of that is identical to the turn before, so it can be reused.\n")
    print("| From -> to | Same system msg | Same tool list | Shared start (tokens) | of the new turn's (tokens) |")
    print("|---|---|---|---|---|")
    for (k1, _m1, a), (k2, _m2, b) in zip(rows, rows[1:]):
        print(f"| {k1} -> {k2} | {'yes' if a['sys_hash'] == b['sys_hash'] else 'no'} | "
              f"{'yes' if a['tools_hash'] == b['tools_hash'] else 'no'} | {shared_prefix_tokens(a['prefix'], b['prefix']):,} | "
              f"{tok(b['prefix']):,} |")

    print("\n## Roblox mentions in turns that are not about Roblox\n")
    print("| Turn | 'roblox' in system + last message |")
    print("|---|---|")
    for kind, _m, r in rows:
        print(f"| {kind} | {r['roblox_mentions']} |")

    if args.tools:
        print("\n## Every tool schema (tokens), per turn\n")
        for kind, _m, r in rows:
            print(f"- **{kind}**: " + ", ".join(f"{n} {c}" for n, c in sorted(r["tool_costs"].items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
