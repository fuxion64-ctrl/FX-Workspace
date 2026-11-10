"""
llm_stream.py
=============
Streaming chat completions from llama-server (OpenAI-compatible SSE).

complete(url, payload, ...) sends the request with "stream": true and builds
the same response shape the non-streaming client returned (choices[0].message
with content / reasoning_content / tool_calls, finish_reason, usage), while:

  * publishing every fragment as it arrives (on_delta) - the chat shows the
    answer as it is written and the Activity feed shows reasoning and tool
    calls taking shape;
  * reassembling tool calls from fragments (index / id / name / argument
    pieces), exactly as the server split them;
  * checking each fragment early:
      - should_cancel()  -> the connection is closed (llama-server stops
                            generating) and StreamCancelled is raised;
      - a tool call's arguments grow past `max_arg_chars`, or the output starts
        repeating itself -> the stream is aborted and the reply is returned with
        finish_reason "runaway" / "repetition" and the partial call, which the
        agent loop answers as unreadable instead of running it.

The HTTP transport is injectable: the replay harness feeds recorded chunks.
"""
from __future__ import annotations

import json
import re
import socket
import threading
import time
from types import SimpleNamespace

DEFAULT_MAX_ARG_CHARS = 60_000         # a single tool call larger than this is a runaway
# A loop = the last REPEAT_SPAN chars are one unit of 8-128 chars repeated
# exactly, and the unit has at least REPEAT_DISTINCT different characters.
# Deliberately conservative: long arrays of zeros, table rules and banners are
# legitimate output; runaways that are not exact loops are caught by the size cap.
REPEAT_SPAN = 4096
REPEAT_DISTINCT = 6
CHECK_EVERY = 256                       # re-run the loop check after this many new chars


UNNAMED_TOOL = "(unnamed tool call)"


class StreamCancelled(Exception):
    """The user stopped the run while the model was generating."""


class StreamHTTPError(RuntimeError):
    """The server answered with an error status; the message carries its body."""


class FirstTokenTimeout(TimeoutError):
    """(1.4.8) The model never started answering within the first-token allowance."""


class StreamStalled(TimeoutError):
    """(1.4.8) The model started answering, then went quiet for longer than the gap allowance."""


# (1.4.8, audit 2.11) The wait before the first token is the model reading the whole prompt - minutes for a big
# model partly on CPU with a long chat - while a gap between tokens is a few seconds at most. One 300 s read
# timeout covered both, so a slow prompt read ended as a generic failure. The first wait gets its own allowance,
# grown with the prompt's size; gaps keep the plain timeout.
FIRST_TOKEN_BASE_S = 300
FIRST_TOKEN_TOKENS_PER_S = 40            # a slow, partly-on-CPU prompt read - the allowance, not a prediction
FIRST_TOKEN_MAX_S = 1800


def first_token_allowance(payload: dict, timeout: float) -> float:
    """Seconds to wait for the first output: at least `timeout`, more for a long prompt (~4 chars a token)."""
    try:
        chars = len(json.dumps(payload.get("messages") or [], default=str))
    except (TypeError, ValueError):
        chars = 0
    return max(float(timeout), min(FIRST_TOKEN_MAX_S, FIRST_TOKEN_BASE_S + chars / 4 / FIRST_TOKEN_TOKENS_PER_S))


def http_transport(url: str, payload: dict, headers: dict, timeout: float, should_cancel=None,
                   first_token_timeout: float | None = None):
    """Yield parsed SSE `data:` objects from llama-server. Closing the
    generator closes the connection, which makes llama-server stop.

    A watcher thread closes the connection as soon as should_cancel() is true,
    so Stop works even before the first token (while the server is still
    processing the prompt nothing arrives to check between).
    first_token_timeout (1.4.8): the read timeout until the first line arrives; after it, the same watcher ends a
    stream that goes quiet for longer than `timeout`."""
    import httpx
    import netlayer
    should_cancel = should_cancel or (lambda: False)
    first_wait = max(float(timeout), float(first_token_timeout or timeout))
    done = threading.Event()
    active_socket = None
    seen = {"last": None, "stalled": False}         # time of the last line; set once output has started

    def cut():
        if active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass                 # already closed by the reader
    # Through the network layer: llama-server on this PC stays direct and unlogged; a model server
    # elsewhere would take the "models" route (proxy / Tor, kill switch, log).
    with netlayer.client("models", timeout=httpx.Timeout(first_wait, connect=10.0)) as client:
        def watcher():
            while not done.wait(0.2):
                if should_cancel():
                    # close() on another thread need not interrupt recv() on
                    # Unix. Shutdown the established stream first so Stop
                    # does not wait for the model's entire read timeout.
                    cut()
                    client.close()
                    return
                if seen["last"] is not None and time.time() - seen["last"] > timeout:
                    seen["stalled"] = True
                    cut()
                    client.close()
                    return
        threading.Thread(target=watcher, daemon=True, name="stream-stop-watcher").start()
        try:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                stream = resp.extensions.get("network_stream")
                if stream is not None:
                    active_socket = stream.get_extra_info("socket")
                if should_cancel():
                    raise StreamCancelled("stopped while the model was generating")
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", "replace")
                    raise StreamHTTPError(f"Error code: {resp.status_code} - {body[:600]}")
                for line in resp.iter_lines():
                    seen["last"] = time.time()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except ValueError:
                        continue
                if should_cancel():
                    raise StreamCancelled("stopped while the model was generating")
        except (httpx.HTTPError, RuntimeError) as e:
            if should_cancel():
                raise StreamCancelled("stopped while the model was generating") from e
            if seen["stalled"]:
                raise StreamStalled(f"The model stopped sending its answer for over {round(timeout)} s.") from e
            if seen["last"] is None and isinstance(e, httpx.ReadTimeout):
                raise FirstTokenTimeout(
                    f"The model was still reading the conversation after {round(first_wait)} s and had not started "
                    "its answer. A big model partly on the CPU reads long chats slowly - try a smaller model, "
                    "compress the chat, or start a new one.") from e
            raise
        finally:
            done.set()


def _repeating(text: str) -> bool:
    """True when the tail of text is one short unit repeated many times."""
    if len(text) < REPEAT_SPAN:
        return False
    tail = text[-REPEAT_SPAN:]
    for period in range(8, 129):
        # tail is periodic with this period: every char equals the one `period` later.
        if tail[period:] == tail[:-period]:
            unit = tail[-period:]
            # Degenerate loops repeat words or markup; legitimate repetitive
            # data (tile maps, zero arrays, table rules) is digits and punctuation.
            return len(set(unit)) >= REPEAT_DISTINCT and len({c for c in unit if c.isalpha()}) >= 3
    return False


def complete(url: str, payload: dict, *, api_key: str = "", timeout: float = 300,
             on_delta=None, should_cancel=None, transport=None,
             max_arg_chars: int = DEFAULT_MAX_ARG_CHARS):
    transport = transport or http_transport
    should_cancel = should_cancel or (lambda: False)
    on_delta = on_delta or (lambda *a, **k: None)
    body = dict(payload)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    content, reasoning = [], []
    calls: dict[int, dict] = {}
    finish, usage, aborted, timings = None, None, None, None
    first_output = last_output = None
    content_watch, arg_watch = {}, {}

    def watch(state: dict, parts: list):
        """Early-stop verdict for one growing output ('runaway' / 'repetition' / None)."""
        seen = state.get("seen", 0)
        for piece in parts[seen:]:
            state["size"] = state.get("size", 0) + len(piece)
            state["tail"] = (state.get("tail", "") + piece)[-REPEAT_SPAN:]
        state["seen"] = len(parts)
        if parts is not content and state.get("size", 0) > max_arg_chars:
            return "runaway"
        if state.get("size", 0) - state.get("checked", 0) >= CHECK_EVERY:
            state["checked"] = state["size"]
            if _repeating(state["tail"]):
                return "repetition"
        return None
    chunks = transport(url, body, headers, timeout, should_cancel=should_cancel,
                       first_token_timeout=first_token_allowance(body, timeout)) if transport is http_transport \
        else transport(url, body, headers, timeout)
    try:
        for chunk in chunks:
            if should_cancel():
                raise StreamCancelled("stopped while the model was generating")
            if chunk.get("usage"):
                usage = chunk["usage"]
            if isinstance(chunk.get("timings"), dict):      # llama-server: generation speed, prompt speed
                timings = chunk["timings"]
            err = chunk.get("error")
            if err:                          # llama-server sends errors as an object or a plain string
                raise StreamHTTPError(str(err.get("message") or err) if isinstance(err, dict) else str(err))
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content") or delta.get("content") or delta.get("tool_calls"):
                    last_output = time.monotonic()
                    if first_output is None:
                        first_output = last_output
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                    on_delta("reasoning", delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                    on_delta("content", delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", len(calls))
                    rec = calls.setdefault(idx, {"id": None, "name": "", "arguments": []})
                    if tc.get("id"):
                        rec["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        rec["name"] += fn["name"]
                        on_delta("tool", "", name=rec["name"], index=idx)
                    if fn.get("arguments"):
                        rec["arguments"].append(fn["arguments"])
                        on_delta("tool", fn["arguments"], name=rec["name"], index=idx)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
            # ---- early checks on what has been generated so far ----
            # Running sizes and bounded tails: no re-joining of the whole output,
            # and the loop check runs only after every CHECK_EVERY new chars.
            for idx, rec in calls.items():
                aborted = aborted or watch(arg_watch.setdefault(idx, {}), rec["arguments"])
            aborted = aborted or watch(content_watch, content)
            if aborted:
                on_delta("abort", aborted)
                break
    finally:
        close = getattr(chunks, "close", None)
        if close:
            close()                          # closes the HTTP stream -> the server stops generating

    # A call whose name never arrived is kept (named UNNAMED_TOOL) so the agent
    # loop answers it as unusable instead of it silently vanishing.
    tool_calls = [SimpleNamespace(id=rec["id"] or f"call_{i}", type="function",
                                  function=SimpleNamespace(name=rec["name"] or UNNAMED_TOOL, arguments="".join(rec["arguments"])))
                  for i, rec in sorted(calls.items()) if rec["name"] or rec["arguments"]]
    message = SimpleNamespace(content="".join(content), reasoning_content="".join(reasoning) or None,
                              tool_calls=tool_calls or None)
    u = usage or {}
    details = u.get("completion_tokens_details") or {}
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=aborted or finish or "stop", message=message)],
        usage=SimpleNamespace(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=details.get("reasoning_tokens"))
                              if details else None) if usage else None,
        streamed=True, aborted=aborted, timings=timings,
        stream_timing={"generation_s": last_output - first_output}
        if first_output is not None and last_output is not None else {})


def chunk_message(msg: dict, size: int = 7):
    """Split a recorded complete message into SSE-style chunks (used by the
    replay harness to exercise fragment reassembly). Explicit msg["chunks"]
    are used as-is."""
    if msg.get("chunks"):
        yield from msg["chunks"]
        return
    if msg.get("reasoning_content"):
        r = msg["reasoning_content"]
        for i in range(0, len(r), size):
            yield {"choices": [{"index": 0, "delta": {"reasoning_content": r[i:i + size]}}]}
    text = msg.get("content") or ""
    for i in range(0, len(text), size):
        yield {"choices": [{"index": 0, "delta": {"content": text[i:i + size]}}]}
    for idx, call in enumerate(msg.get("tool_calls") or []):
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": idx, "id": call.get("id") or f"call_{idx}", "type": "function",
             "function": {"name": call["name"], "arguments": ""}}]}}]}
        args = call.get("arguments") or ""
        for i in range(0, len(args), size):
            yield {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": idx, "function": {"arguments": args[i:i + size]}}]}}]}
    yield {"choices": [{"index": 0, "delta": {}, "finish_reason": msg.get("finish_reason", "stop")}]}
    if msg.get("usage"):
        yield {"choices": [], "usage": msg["usage"]}
