"""
speed_tune.py
=============
"Tune speed" (1.4.5, Settings > Models): find the llama.cpp settings that make one model fastest on THIS PC, by
measuring - never assumed (moe_test.py's way: llama-server's own timings, the model reloaded for each setting).

  1. Prompt reading (every model): the batch / ubatch sizes llama-server reads a prompt in. Bigger batches read long
     prompts much faster on a GPU (measured before on Qwen3.5-35B-A3B: 2048/2048 read 966 tokens/s against 393),
     but take VRAM for their compute buffer. Tried: what it has now, 1024/512, 2048/512, 2048/2048.
  2. GPU split (mixture-of-experts models only): n-cpu-moe = how many layers keep their experts in system RAM. Fewer
     means more on the GPU and faster writing - until it no longer fits. Tried: what it has now and 4 / 8 / 12 fewer,
     stopping at the first that doesn't load or that spills into shared GPU memory (Windows lending RAM to the GPU:
     several times slower, so never kept), and 4 more as a check.

A setting is kept only when it is clearly better: prompt reading +10% (writing no worse than -3%) for step 1,
writing +5% (prompt reading no worse than -2%) for step 2. Otherwise the model keeps what it had. models.ini is
backed up first, and anything that fails puts the old values back. Chats can run while it tunes (1.4.6): between
measurements it pauses for them and gives the model its own settings back; a chat waits at most for the step in
hand. Results: data/speed_tune.json, shown in Settings.
"""
from __future__ import annotations

import json
import os
import threading
import time

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "speed_tune.json")
RUNS = 2
PREFILL_GAIN, PREFILL_GEN_FLOOR = 1.10, 0.97
SPLIT_GAIN, SPLIT_PROMPT_FLOOR = 1.05, 0.98
BATCHES = ((1024, 512), (2048, 512), (2048, 2048))
SPLIT_STEPS = (-4, -8, -12)
# A long prompt (~1,800 tokens): prompt reading is what the batch size changes.
PROMPT = ("Read these design notes for a Roblox obby and list every obstacle with its stage number, then suggest "
          "three new ones.\n\n" + "Stage {n}: lava floors with moving platforms, a checkpoint, a timer, a spinning "
          "beam and a leaderboard; players respawn at the last checkpoint and the tower at the end has a jump pad. ")
PROMPT = PROMPT.split("\n\n")[0] + "\n\n" + "".join(PROMPT.split("\n\n")[1].format(n=i) for i in range(1, 46))
state = {"running": None, "progress": "", "error": None}
_lock = threading.Lock()


def _load() -> dict:
    try:
        with open(DATA, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, DATA)


def candidates() -> list:
    """Every configured model; `moe` marks the ones with experts (step 2 applies)."""
    import diagnostics
    import llama_local
    out = []
    for cfg in llama_local.list_model_configs():
        opts = llama_local.model_options(cfg["name"])
        meta = diagnostics.meta_for(cfg["model_path"]) if cfg.get("model_path") and os.path.exists(cfg["model_path"]) else None
        out.append({"name": cfg["name"], "moe": bool((meta or {}).get("expert_count") or opts.get("n-cpu-moe")),
                    "layers": (meta or {}).get("block_count"), "batch": opts.get("batch-size"),
                    "ubatch": opts.get("ubatch-size"), "n_cpu_moe": opts.get("n-cpu-moe")})
    return out


def results() -> dict:
    return {"state": dict(state), "models": candidates(), "results": _load()}


# ---- the decisions (plain functions: tested without a GPU) -----------------------------------------------------
def decide_prefill(base: dict, tried: list) -> tuple:
    """(row to keep or None, why). Rows: {"batch", "ubatch", "prompt_tps", "gen_tps"}."""
    if not base.get("prompt_tps"):
        return None, "Nothing to compare - the setting stays as it was."
    good = [t for t in tried if t.get("prompt_tps") and t.get("gen_tps") and t["prompt_tps"] >= base["prompt_tps"] * PREFILL_GAIN
            and t["gen_tps"] >= (base.get("gen_tps") or 0) * PREFILL_GEN_FLOOR]
    if not good:
        best = max((t for t in tried if t.get("prompt_tps")), key=lambda t: t["prompt_tps"], default=None)
        if best:
            return None, (f"No batch size helped enough: the best ({best['batch']}/{best['ubatch']}) read prompts "
                          f"{(best['prompt_tps'] / base['prompt_tps'] - 1) * 100:+.0f}% - kept as it was.")
        return None, "No other batch size loaded - kept as it was."
    best = max(good, key=lambda t: t["prompt_tps"])
    return best, (f"Prompt reading {best['batch']}/{best['ubatch']}: {(best['prompt_tps'] / base['prompt_tps'] - 1) * 100:+.0f}% "
                  f"({base['prompt_tps']:.0f} -> {best['prompt_tps']:.0f} tokens/s), writing "
                  f"{(best['gen_tps'] / base['gen_tps'] - 1) * 100 if base.get('gen_tps') else 0:+.0f}%.")


def decide_split(base: dict, tried: list) -> tuple:
    """(row to keep or None, why). Rows: {"n_cpu_moe", "prompt_tps", "gen_tps", "spilled"}."""
    if not base.get("gen_tps"):
        return None, "Nothing to compare - the split stays as it was."
    good = [t for t in tried if not t.get("spilled") and t.get("gen_tps") and t["gen_tps"] >= base["gen_tps"] * SPLIT_GAIN
            and (t.get("prompt_tps") or 0) >= (base.get("prompt_tps") or 0) * SPLIT_PROMPT_FLOOR]
    if not good:
        return None, "No other split was faster without overflowing the GPU - kept as it was."
    best = max(good, key=lambda t: t["gen_tps"])
    return best, (f"GPU split: {best['n_cpu_moe']} layers' experts in RAM (was {base['n_cpu_moe']}) - writing "
                  f"{(best['gen_tps'] / base['gen_tps'] - 1) * 100:+.0f}% ({base['gen_tps']:.1f} -> {best['gen_tps']:.1f} tokens/s).")


def split_values(current: int, layers: int | None) -> list:
    """The n-cpu-moe values to try below the current one (fewer layers on the CPU), in order, then one above."""
    top = int(layers or current + 12)
    down = [current + s for s in SPLIT_STEPS if current + s >= 0]
    up = [current + 4] if current + 4 <= top else []
    return down + up


# ---- measuring ---------------------------------------------------------------------------------------------------
def _measure(name: str) -> dict:
    """RUNS identical completions of the long prompt; the averages of llama-server's own speeds."""
    import llama_local
    import local_runtime
    import netlayer
    p, g = [], []
    with local_runtime.reserve(), netlayer.local_client(timeout=600) as http:      # llama-server on this PC only
        for _ in range(RUNS):
            r = http.post(f"{llama_local.LLAMA_SERVER_BASE.rstrip('/')}/chat/completions",
                          json={"model": name, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 120,
                                "temperature": 0, "cache_prompt": False, "stream": False})
            r.raise_for_status()
            t = r.json().get("timings") or {}
            if t.get("prompt_per_second"):
                p.append(float(t["prompt_per_second"]))
            if t.get("predicted_per_second"):
                g.append(float(t["predicted_per_second"]))
    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else None      # noqa: E731
    return {"prompt_tps": avg(p), "gen_tps": avg(g)}


QUIET_S = 20          # after a reply the next message often follows: the tune waits this long before it goes on


def _chat_active() -> bool:
    """A reply is running, or one ended less than QUIET_S ago."""
    try:
        import agent.tasks as tasks
        cur = tasks.turns.current()
    except Exception:
        return False
    if not cur:
        return False
    return cur.get("status") == "running" or time.time() - (cur.get("ended") or 0) < QUIET_S


def _yield_to_chats(name: str, before: dict | None) -> bool:
    """Between measurements (1.4.6): while you chat, the tune waits - and the model gets its own settings back, so
    the chat loads it as it normally is, not with a setting being tried. The next measurement reloads it anyway."""
    if before is None or not _chat_active():
        return False
    import llama_local
    llama_local.set_model_options({name: before}, backup=False)
    saved = state["progress"]
    state["progress"] = "Paused while you chat - it goes on by itself"
    while _chat_active():
        time.sleep(2)
    state["progress"] = saved
    return True


def _with(name: str, opts: dict, before: dict | None = None) -> dict:
    """Load the model with these options and measure it; notes when it spilled into shared GPU memory. One step holds
    the model from the load to the last measurement, so a chat that starts meanwhile waits for this step only (about
    a minute) instead of finding the model unloaded under it; between steps the tune pauses for chats."""
    import diagnostics
    import llama_local
    import local_runtime
    _yield_to_chats(name, before)
    with local_runtime.reserve():
        llama_local.set_model_options({name: opts}, backup=False)
        llama_local.unload_all()
        llama_local.force_load(name, timeout=600)
        place = diagnostics.placement(name) or {}
        spilled = place.get("state") == "spilled"
        row = {**opts, "spilled": spilled}
        if spilled:                                               # several times slower: not worth measuring
            return {**row, "prompt_tps": None, "gen_tps": None}
        return {**row, **_measure(name)}


def run(name: str) -> dict:
    """The whole tune for one model (on the caller's thread)."""
    import llama_local
    model = next((c for c in candidates() if c["name"] == name), None)
    if not model:
        return {"ok": False, "message": f"{name} isn't in models.ini."}
    keys = ("batch-size", "ubatch-size", "n-cpu-moe")
    # What the model's own section sets (None = it inherits [*]): putting these back leaves models.ini as it was.
    before = {k: llama_local.model_own_options(name).get(k) for k in keys}
    effective = llama_local.model_options(name)
    llama_local.set_model_options({name: {}})                         # a models.ini backup before anything changes
    row = {"at": time.time(), "before": before}
    started = time.time()
    try:
        # 1. prompt reading
        state["progress"] = "Measuring as it is"
        cur_b, cur_ub = int(effective.get("batch-size") or 2048), int(effective.get("ubatch-size") or 512)
        base = {"batch": cur_b, "ubatch": cur_ub, **_with(name, {}, before)}
        tried = []
        for i, (b, ub) in enumerate([x for x in BATCHES if x != (cur_b, cur_ub)], 1):
            state["progress"] = f"Prompt reading {b}/{ub}"
            try:
                tried.append({"batch": b, "ubatch": ub, **_with(name, {"batch-size": b, "ubatch-size": ub}, before)})
            except Exception as e:                                    # didn't load: noted, next one
                tried.append({"batch": b, "ubatch": ub, "error": str(e)[:160], "prompt_tps": None, "gen_tps": None})
        keep, why = decide_prefill(base, [t for t in tried if not t.get("spilled")])
        chosen = {"batch-size": keep["batch"], "ubatch-size": keep["ubatch"]} if keep else \
                 {"batch-size": before["batch-size"], "ubatch-size": before["ubatch-size"]}
        llama_local.set_model_options({name: chosen}, backup=False)
        row["prefill"] = {"base": base, "tried": tried, "kept": bool(keep), "why": why}
        # 2. the GPU split (mixture-of-experts models)
        if model["moe"]:
            cur = int(effective.get("n-cpu-moe") or 0)
            state["progress"] = "GPU split as it is"
            base2 = {"n_cpu_moe": cur, **_with(name, {"n-cpu-moe": cur or None}, normal)}
            tried2 = []
            for n in split_values(cur, model.get("layers")):
                state["progress"] = f"GPU split: {n} layers' experts in RAM"
                try:
                    t = {"n_cpu_moe": n, **_with(name, {"n-cpu-moe": n}, normal)}
                except Exception as e:
                    t = {"n_cpu_moe": n, "error": str(e)[:160], "prompt_tps": None, "gen_tps": None}
                tried2.append(t)
                if n < cur and (t.get("error") or t.get("spilled")):
                    break                                             # fewer won't fit either
            keep2, why2 = decide_split(base2, tried2)
            llama_local.set_model_options({name: {"n-cpu-moe": keep2["n_cpu_moe"] if keep2 else before["n-cpu-moe"]}}, backup=False)
            row["split"] = {"base": base2, "tried": tried2, "kept": bool(keep2), "why": why2}
        llama_local.unload_all()
        row["why"] = " ".join(x["why"] for x in (row.get("prefill"), row.get("split")) if x)
        row["minutes"] = round((time.time() - started) / 60, 1)
    except Exception as e:
        llama_local.set_model_options({name: before}, backup=False)
        llama_local.unload_all()
        row.update(error=str(e)[:300], why="The tune stopped - the settings are back as they were.")
    d = _load()
    d[name] = row
    _save(d)
    import events as event_log
    event_log.log_action(f"Speed tune for {name}: {row.get('why')}")
    return {"ok": "error" not in row, **row}


def start(name: str) -> dict:
    with _lock:
        if state["running"]:
            return {"ok": False, "message": f"Already tuning {state['running']}."}
        state.update(running=name, progress="Starting", error=None)

    def work():
        try:
            r = run(name)
            state["error"] = None if r.get("ok") else r.get("message") or r.get("error")
        except Exception as e:
            state["error"] = str(e)[:300]
        finally:
            state.update(running=None, progress="")
    threading.Thread(target=work, daemon=True, name="speed-tune").start()
    return {"ok": True, "message": f"Tuning {name} - about 10-20 minutes. You can keep chatting: it pauses for you."}
