"""
effort.py
=========
The effort system as runtime data: what each level concretely allows, which
levels the active model supports, how a level is chosen, and a per-run budget
that tracks live usage against those limits.

Levels (ids are stored; labels are what the UI shows):
    low     Quick         short reasoning, few steps, fast - simple questions, small edits
    normal  Balanced      the everyday default
    high    Thorough      deeper planning, larger context, a reviewer pass
    extra   Rigorous      multi-pass reasoning plus a reviewer check
    max     Max           largest conventional effort budget and model review
    ultra   Ultra         sustained multi-step work: largest run budget, detailed planning and review

A profile controls ONLY effort: reasoning budget, step/tool/time/token limits,
context-selection size, verification depth and orchestration. It has no
permission, approval, undo, folder-access or external-action fields, and
nothing here can change those (non-negotiable 7; see test_effort.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, replace

LEVELS = ("low", "normal", "high", "extra", "max", "ultra")
CHOICES = ("auto",) + LEVELS
LABELS = {"low": "Quick", "normal": "Balanced", "high": "Thorough", "extra": "Rigorous",
          "max": "Max", "ultra": "Ultra", "auto": "Auto"}
VERIFICATION = ("none", "self", "reviewer", "multi")


@dataclass(frozen=True)
class EffortProfile:
    name: str
    planner: str              # off | heuristic | model | detailed
    reviewer: str             # deterministic | model | cross   (how verification is carried out)
    max_revisions: int
    max_handoffs: int
    max_tool_calls: int       # tool calls per run
    time_limit_s: int
    local_iterations: int     # max model steps per turn (local)
    cloud_iterations: int     # max model steps per turn (cloud)
    native: str               # reasoning effort hint for runtimes that take one: low | medium | high
    thinking: bool            # let reasoning models think
    reasoning_tokens: int = 0         # extra reply room reserved for reasoning (0 = none)
    token_budget: int = 60_000        # prompt+completion tokens per run
    context_tokens: int = 16_000      # at most this much history/context is selected per call
    verification: str = "self"        # none | self | reviewer | multi
    # Reserved metadata: no active profile advertises unenforced stage caps or delegation.
    orchestration: str = "none"
    stage_caps: tuple = ()
    delegation: bool = False
    min_context: int = 0              # the model's window must be at least this for the level

    @property
    def label(self) -> str:
        return LABELS.get(self.name, self.name)

    def limits(self) -> dict:
        """The numbers the Activity panel shows as 'applied limits'."""
        return {"steps_local": self.local_iterations, "steps_cloud": self.cloud_iterations,
                "tool_calls": self.max_tool_calls, "time_s": self.time_limit_s, "tokens": self.token_budget,
                "reasoning_tokens": self.reasoning_tokens, "context_tokens": self.context_tokens,
                "revisions": self.max_revisions, "verification": self.verification,
                "orchestration": self.orchestration, "delegation": self.delegation}


PROFILES = {
    "low":    EffortProfile("low", "off", "deterministic", 0, 0, 12, 300, 12, 8, "low", False,
                            reasoning_tokens=0, token_budget=30_000, context_tokens=6_000, verification="none"),
    "normal": EffortProfile("normal", "heuristic", "deterministic", 1, 1, 24, 600, 24, 12, "medium", False,
                            reasoning_tokens=0, token_budget=80_000, context_tokens=16_000, verification="self"),
    "high":   EffortProfile("high", "model", "model", 2, 1, 36, 900, 30, 16, "high", True,
                            reasoning_tokens=2048, token_budget=150_000, context_tokens=32_000, verification="reviewer"),
    "extra":  EffortProfile("extra", "detailed", "model", 2, 1, 42, 1050, 34, 18, "high", True,
                            reasoning_tokens=4096, token_budget=200_000, context_tokens=48_000, verification="reviewer",
                            min_context=16_384),
    "max":    EffortProfile("max", "detailed", "model", 3, 2, 48, 1200, 40, 20, "high", True,
                            reasoning_tokens=6144, token_budget=260_000, context_tokens=64_000, verification="reviewer",
                            min_context=16_384),
    "ultra":  EffortProfile("ultra", "detailed", "model", 3, 2, 64, 1800, 48, 24, "high", True,
                            reasoning_tokens=4096, token_budget=320_000, context_tokens=64_000, verification="reviewer",
                            min_context=32_768),
}

# Per-model overrides may only tune these numeric limits, within these bounds.
TUNABLE = {"max_tool_calls": (1, 200), "time_limit_s": (30, 7200), "local_iterations": (1, 120),
           "cloud_iterations": (1, 120), "reasoning_tokens": (0, 32_768), "token_budget": (1_000, 2_000_000),
           "context_tokens": (1_000, 1_000_000), "max_revisions": (0, 5)}


def clean_overrides(raw) -> dict:
    """{model: {level: {field: number}}} with anything else dropped."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for model, per_level in raw.items():
        if not isinstance(model, str) or not isinstance(per_level, dict):
            continue
        for level, values in per_level.items():
            if level not in LEVELS or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if key in TUNABLE and isinstance(value, (int, float)) and not isinstance(value, bool):
                    lo, hi = TUNABLE[key]
                    out.setdefault(model, {}).setdefault(level, {})[key] = int(max(lo, min(hi, value)))
    return out


def profile_for(level: str, model: str | None = None, overrides: dict | None = None) -> EffortProfile:
    base = PROFILES.get(level) or PROFILES["normal"]
    tweaks = ((overrides or {}).get(model or "") or {}).get(base.name) or {}
    return replace(base, **{k: v for k, v in tweaks.items() if k in TUNABLE}) if tweaks else base


# ---- capability detection ---------------------------------------------------------
def supported_levels(caps: dict | None) -> list:
    """Levels the model can run, from DETECTED capabilities only (model_caps):
    context window and tool-call protocol (and, later, measured reliability).
    Unknown facts do not rule a level out - they cannot be held against it."""
    caps = caps or {}
    ctx = caps.get("context")
    out = []
    for level in LEVELS:
        p = PROFILES[level]
        if ctx and p.min_context and ctx < p.min_context:
            continue
        if level == "ultra" and caps.get("tool_protocol") == "text":
            continue                        # sustained multi-step work needs structured tool calls
        rel = caps.get("tool_call_validity")
        if level == "ultra" and isinstance(rel, (int, float)) and rel < 0.8:
            continue                        # measured (Phase 5 benchmark) reliability too low
        out.append(level)
    return out or ["low"]


def resolve(requested: str, caps: dict | None) -> tuple[str, str | None]:
    """(level to run, fallback reason or None). Falls back to the nearest
    supported level below the request (never above it)."""
    requested = requested if requested in LEVELS else "normal"
    ok = supported_levels(caps)
    if requested in ok:
        return requested, None
    idx = LEVELS.index(requested)
    lower = [l for l in ok if LEVELS.index(l) < idx]
    chosen = lower[-1] if lower else ok[0]
    caps = caps or {}
    p = PROFILES[requested]
    why = []
    if caps.get("context") and p.min_context and caps["context"] < p.min_context:
        why.append(f"the model's context window ({caps['context']:,} tokens) is below {p.min_context:,}")
    if requested == "ultra" and caps.get("tool_protocol") == "text":
        why.append("the model has no structured tool calls")
    if requested == "ultra" and isinstance(caps.get("tool_call_validity"), (int, float)) and caps["tool_call_validity"] < 0.8:
        why.append(f"measured tool-call validity is {round(caps['tool_call_validity'] * 100)}%")
    return chosen, "; ".join(why) or "not supported by this model"


def choose_level(settings: dict, model: str | None = None, project: dict | None = None,
                 session: str | None = None) -> tuple[str, str]:
    """Which level the user asked for, and where that choice came from:
    session override > project > this model > the global setting."""
    if session in CHOICES:
        return session, "session"
    if project and project.get("effort") in CHOICES:
        return project["effort"], "project"
    by_model = settings.get("effort_by_model") or {}
    if model and by_model.get(model) in CHOICES:
        return by_model[model], "model"
    level = settings.get("effort")
    return (level if level in CHOICES else "normal"), "setting"


def auto_level(analysis: dict, voice: bool = False) -> tuple[str, str]:
    """A cheap per-turn choice; never invokes a second model just to choose effort."""
    mode = analysis.get("mode", "none")
    words = int(analysis.get("words") or 0)
    parts = int(analysis.get("parts") or 1)
    if analysis.get("trivial") and mode == "none":
        return "low", "brief conversation"
    if mode == "action" and (parts >= 5 or words >= 120):
        return "max", "large multi-step action"
    if mode == "action" and analysis.get("complex"):
        return "extra", "complex action"
    if analysis.get("complex") or parts >= 3:
        return "high", "multi-part request"
    # A one-step look ("what's using my PC", "list the files in ...") changes nothing: a model-written plan and a
    # model review only made the user wait - 1.2-2.2 s for the plan and ~1.1 s for the review on Qwen 3.5 4B, measured
    # 4 Nov 2026 (stage 2). Changes keep High: they are planned and checked.
    if mode == "read" and words < 40:
        return "normal", "a quick look - nothing changes"
    if mode in ("read", "action"):
        return "high", "tool-based request"
    return "normal", "conversation" if voice else "standard request"


# ---- per-run budget: live usage against the applied limits ------------------------
@dataclass
class RunBudget:
    profile: EffortProfile
    local: bool
    started: float = field(default_factory=time.time)
    steps: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    pending: EffortProfile | None = None      # a level change waiting for the next safe boundary
    _last_prompt: int = 0

    # The token budget counts work, not repetition: every step re-sends the
    # same system prompt and tool list (~9k tokens with the full catalogue),
    # which llama.cpp serves from its prompt cache. So a step's prompt counts
    # only what is new since the previous step (its first step counts fully),
    # plus everything the model generated.

    @property
    def step_limit(self) -> int:
        return self.profile.local_iterations if self.local else self.profile.cloud_iterations

    def record_call(self, tokens_in=None, tokens_out=None):
        self.steps += 1
        prompt = int(tokens_in or 0)
        new = prompt - self._last_prompt if prompt >= self._last_prompt else prompt   # history trimmed: count it fresh
        self._last_prompt = prompt
        self.tokens_in += max(0, new)
        self.tokens_out += int(tokens_out or 0)

    def exhausted(self) -> str | None:
        """Why the run must stop now (checked between steps), or None."""
        if self.steps >= self.step_limit:
            return "step"
        if self.tokens_in + self.tokens_out >= self.profile.token_budget:
            return "token"
        return None

    def request_change(self, profile: EffortProfile):
        self.pending = profile

    def apply_pending(self) -> EffortProfile | None:
        """At a safe boundary (between tool steps, never mid-call): switch to the
        pending profile. Usage so far is kept; only the limits change."""
        if self.pending is None or self.pending == self.profile:
            self.pending = None
            return None
        self.profile, self.pending = self.pending, None
        return self.profile

    def usage(self) -> dict:
        return {"steps": self.steps, "step_limit": self.step_limit, "tool_calls": self.tool_calls,
                "tool_limit": self.profile.max_tool_calls, "tokens": self.tokens_in + self.tokens_out,
                "token_budget": self.profile.token_budget, "elapsed_s": round(time.time() - self.started, 1),
                "time_limit_s": self.profile.time_limit_s, "level": self.profile.name}


def permission_fields() -> set:
    """Names on a profile that could touch permissions - must be empty."""
    words = ("permission", "approv", "undo", "folder", "allow", "deny", "external", "auto_run", "god")
    return {f.name for f in fields(EffortProfile) if any(w in f.name for w in words)}
