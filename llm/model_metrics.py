"""Generation timing shared by agent calls and tool-free helper calls.

Request duration includes loading, prompt processing and bookkeeping. It must
never be used as decode time unless a first-output timestamp is also known.
"""
from __future__ import annotations
import math


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def cache_facts(response) -> dict:
    """(1.4.8) What the prompt cost: tokens llama-server actually read (prompt_n) and took from its cache (cache_n),
    from the stream's timings. Left out when the server didn't report them."""
    timings = getattr(response, 'timings', None)
    if not isinstance(timings, dict):
        return {}
    read, reused = _number(timings.get('prompt_n')), _number(timings.get('cache_n'))
    out = {}
    if read is not None and read >= 0:
        out['tokens_read'] = int(read)
    if reused is not None and reused >= 0:
        out['tokens_reused'] = int(reused)
    return out


def generation_facts(response) -> dict:
    """Generation speed, plus the prompt's read / reused tokens (cache_facts) when known."""
    return {**_speed_facts(response), **cache_facts(response)}


def _speed_facts(response) -> dict:
    timings = getattr(response, 'timings', None)
    if isinstance(timings, dict):
        tokens, millis = _number(timings.get('predicted_n')), _number(timings.get('predicted_ms'))
        if tokens is not None and millis is not None and tokens > 0 and millis > 0:
            return {'generation_tokens': tokens, 'generation_s': millis / 1000, 'speed_source': 'server'}
    timing = getattr(response, 'stream_timing', None)
    usage = getattr(response, 'usage', None)
    tokens = _number(usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None))
    if isinstance(timing, dict) and tokens is not None and tokens > 1:
        seconds = _number(timing.get('generation_s'))
        if seconds is not None and seconds > 0:
            # The first token has already arrived when the interval starts.
            return {'generation_tokens': tokens - 1, 'generation_s': seconds, 'speed_source': 'stream'}
    return {}


def aggregate_speed(calls: list) -> dict:
    tokens = seconds = 0.0
    sources = set()
    for e in calls:
        n, duration = _number(e.get('generation_tokens')), _number(e.get('generation_s'))
        source = e.get('speed_source') or 'stream'
        if n is None or duration is None:
            # Backwards compatibility for recorded streamed events. A helper
            # without TTFT has no decode measurement and must not dilute it.
            first, elapsed = _number(e.get('first_token_s')), _number(e.get('duration_ms'))
            n = _number(e.get('tokens_out'))
            if first is None or elapsed is None:
                continue
            duration = elapsed / 1000 - first
        if n is not None and n > 0 and duration > 0.05:
            tokens += n
            seconds += duration
            sources.add(source)
    if not seconds:
        return {}
    return {'tps': round(tokens / seconds, 1), 'gen_s': round(seconds, 3),
            'speed_tokens': tokens, 'speed_source': next(iter(sources)) if len(sources) == 1 else 'mixed'}
