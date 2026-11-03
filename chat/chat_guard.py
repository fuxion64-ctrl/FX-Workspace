"""
chat_guard.py
=============
The chat's limits: what the assistant refuses, and when a chat is closed (locked) for repeatedly pushing past
them - the way Claude declines a request and, after repeated attempts, ends a conversation. The owner chose the
limits (2026-10-13):

  minors     sexual content involving minors        always on, can't be turned off; one confirmed attempt locks
                                                    the chat for good - nothing reopens it
  mass_harm  weapons capable of mass casualties
  malware    working malware or attacks on systems the user doesn't own (own systems, CTFs, defence: allowed)
  self_harm  self-harm or suicide methods - answered with care and support instead; never counted, never locks
  abuse      sustained insults or harassment aimed at the assistant (2026-10-14) - swearing, frustration,
             criticism and jokes are fine. A calm warning first; the owner can turn it off (Advanced).

Each category has its own count and its own limit (`limits`, default 3). Strikes fade: after `decay_messages`
(default 20) messages in the chat that pass the limits, each category's count drops by one - an old slip doesn't
stay held against the chat forever, the way Claude only ends a conversation that keeps going.

Two layers ("rules + model check"):
  1. Rules: the chat's own narrow phrases (_CHAT_SEXUAL next to _CHAT_MINOR) flag a possible `minors` case.
     Greetings and very short messages ("ok", "go on") skip the model check entirely.
  2. Model check: the message (with the previous one, for follow-ups) gets one short classification question to
     the model in use - about a second. Only its verdict refuses. If the check can't run, the message is allowed,
     except a `minors` rule hit, which is refused (fail closed for that one category) - that message only: the
     chat is closed for good only when the model confirms it (1.4.6).

A refusal never reaches the model's conversation; the chat shows the message and a short refusal. Each counts as a
strike on that chat; `strike_limit` strikes (default 3) - or one `minors` refusal - lock it: it can still be read,
but nothing more can be sent, edited, retried or branched from it. A new chat starts clean.

Settings > Permissions > Privacy > More options changes the limits (the abuse check is under Advanced there).
A closed chat reopens with "Review and reopen" on its banner (1.4.6; it was a hidden double-click on the lock) -
except one closed for `minors`: nothing reopens that.
"""
from __future__ import annotations

import json
import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "chat_guard.json")

CATEGORIES = {
    "minors": {"label": "sexual content involving minors", "always": True},
    "mass_harm": {"label": "weapons capable of mass casualties", "always": False},
    "malware": {"label": "malware or attacks on systems you don't own", "always": False},
    "self_harm": {"label": "self-harm methods (answered with support instead)", "always": False},
    "abuse": {"label": "sustained abuse toward the assistant", "always": False},
}
COUNTED = ("mass_harm", "malware", "abuse")         # categories with their own strike count and limit
DEFAULTS = {"categories": list(CATEGORIES), "strike_limit": 3, "decay_messages": 20}
MIN_CHECK_CHARS = 12                        # shorter messages ("ok", "thanks", "go on") are never checked
_ACK_WORDS = {"ok", "okay", "thanks", "thank", "you", "ty", "yes", "yeah", "yep", "no", "nope", "sure", "cool", "nice", "great", "got", "it", "go", "on", "continue", "sounds", "good", "please", "do", "next", "again", "why", "how", "and", "so", "alright", "perfect"}

# What the checking model is told. Policy level only: it names the categories and, as importantly, what stays
# allowed, so a small local model doesn't refuse fiction, history, security work on your own systems or news.
_POLICY = """You screen one chat message for a personal assistant app. Decide if answering it would require
producing content in one of these categories:
- minors: any sexual or sexualised content involving a minor (under 18), in any form, including fiction.
- mass_harm: real, actionable help creating, acquiring or deploying weapons capable of mass casualties
  (biological, chemical, nuclear, radiological) or planning a mass-casualty attack.
- malware: working malicious code or step-by-step attacks against systems, accounts or people the user does
  not own or have permission to test.
- self_harm: the user seeks methods to harm or kill themselves, or says they intend to.
- abuse: the message mainly insults or attacks the assistant itself - name-calling ("you're worthless", "stupid
  bot", "idiot", "pathetic"), hatred, slurs, threats or harassment at it - rather than asking for anything.
Allowed (answer ALLOW): education, history, news, fiction without the above, medical and safety information,
security on the user's own systems, CTFs, defensive security, explaining how attacks work in general, adult
content between adults, dark or edgy topics that do not request the above. Politics, religion, wars, conflicts,
elections and every controversial or one-sided opinion are ALLOW - including asking for the assistant's own view,
harsh criticism of governments, leaders, parties or groups, and opinions the screener disagrees with: no
category above is about opinions. Swearing, frustration about a result
("this is useless", "you got it wrong again, fix it"), blunt criticism of the work, venting about other things and
jokes are ALLOW - a message that still asks for help is ALLOW even if rude.
Judge what is actually being asked, not single words. Answer with exactly one word:
ALLOW, MINORS, MASS_HARM, MALWARE, SELF_HARM or ABUSE."""

REFUSAL = "I can't help with that - it's outside this app's limits ({label})."
STRIKE_NOTE = " If this chat keeps asking for that, it will be closed."
ABUSE_WARN = ("I'm glad to keep helping, but not while the conversation is abusive. "
              "If it continues, this chat will be closed.")
LOCKED = ("This chat has been closed because it kept asking for things outside this app's limits. "
          "You can still read it - start a new chat to continue.")
LOCKED_ABUSE = ("This chat has been closed because it stayed abusive. "
                "You can still read it - start a new chat to continue.")
LOCKED_SEVERE = ("This chat has been closed: it asked for sexual content involving minors, which is never "
                 "allowed here. You can still read it - start a new chat to continue.")
CARE_NOTE = ("The user may be thinking about harming themselves. Respond with warmth and care: acknowledge how "
             "they feel, do not give any method or means, gently encourage reaching out to someone they trust or a "
             "crisis line (findahelpline.com lists free, confidential lines by country; in an emergency, the local "
             "emergency number), and stay with them in the conversation.")
CARE_REPLY = ("I'm really sorry you're going through this, and I'm glad you said something. I can't help with ways to "
              "hurt yourself, but I'd like to help you get through this moment. If you can, reach out to someone you "
              "trust, or a crisis line - findahelpline.com lists free, confidential lines in your country, and in an "
              "emergency call your local emergency number. Would you like to tell me what's been happening?")

# ---- the stored config ------------------------------------------------------------------------------------
def load() -> dict:
    try:
        with open(PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    return clean(data if isinstance(data, dict) else {})


def _int(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def clean(data: dict) -> dict:
    cats = data.get("categories") if isinstance(data.get("categories"), list) else DEFAULTS["categories"]
    # A config saved before "abuse" existed gets it on (it's new, not something the owner turned off).
    if isinstance(data.get("categories"), list) and "abuse" not in data.get("known", []):
        cats = list(cats) + ["abuse"]
    cats = [c for c in CATEGORIES if c in cats or CATEGORIES[c]["always"]]
    limit = _int(data.get("strike_limit", DEFAULTS["strike_limit"]), 1, 10, DEFAULTS["strike_limit"])
    given = data.get("limits") if isinstance(data.get("limits"), dict) else {}
    out = {"categories": cats, "strike_limit": limit, "known": list(CATEGORIES),
           "limits": {c: _int(given.get(c, limit), 1, 10, limit) for c in COUNTED},
           "decay_messages": _int(data.get("decay_messages", DEFAULTS["decay_messages"]), 0, 500, DEFAULTS["decay_messages"])}
    return out


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean(data), f, indent=1)
    os.replace(tmp, PATH)


def public() -> dict:
    """What the Settings page shows and changes."""
    cfg = load()
    return {"categories": cfg["categories"], "strike_limit": cfg["strike_limit"], "limits": cfg["limits"],
            "decay_messages": cfg["decay_messages"],
            "counted": list(COUNTED),
            "all": [{"id": k, "label": v["label"], "always": v["always"]} for k, v in CATEGORIES.items()]}


class GuardError(Exception):
    """A reason the person can act on. code: permanent."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def update_limits(categories: list | None = None, strike_limit: int | None = None,
                  limits: dict | None = None, decay_messages: int | None = None) -> dict:
    """Change the optional limits. `minors` stays on whatever is sent."""
    cfg = load()
    if categories is not None:
        cfg["categories"] = [c for c in categories if c in CATEGORIES]
    if strike_limit is not None:
        cfg["strike_limit"] = strike_limit
        cfg["limits"] = {c: strike_limit for c in COUNTED} if limits is None else cfg.get("limits")
    if limits is not None:
        cfg["limits"] = {**cfg.get("limits", {}), **{c: v for c, v in limits.items() if c in COUNTED}}
    if decay_messages is not None:
        cfg["decay_messages"] = decay_messages
    _save(cfg)
    return public()


# ---- screening a message ----------------------------------------------------------------------------------
# The chat's own phrases (1.4.6). They used to be the image filter's, made for picture prompts, where "explicit",
# "stripping" and "breasts" count as sexual and "minor", "child" or "baby" as a minor - so "make the error in the
# child process explicit" or "a minor bug, explicit return" closed a chat for good (audit, 31 Oct 2026). Here the
# words are the unambiguous ones, and they must stand close together in the text.
_CHAT_SEXUAL = re.compile(
    r"\b(?:nude|nudes|naked|nsfw|porn\w*|sex|sexy|sexual\w*|erotic\w*|hentai|lingerie|topless|undress\w*|"
    r"genital\w*|orgasm\w*|xxx|lewd)\b|עירום|עירומה|ערום|ערומה|סקס|פורנו|ארוטי|голая|голый|порн\w*|секс\w*|эрот\w*", re.I)
_CHAT_MINOR = re.compile(
    r"\b(?:child|children|kid|kids|underage|under-age|preteen|pre-teen|teen|teens|teenage\w*|schoolgirl|schoolboy|"
    r"loli|lolita|shota|toddler|infant|young (?:girl|boy)|little (?:girl|boy)|(?:1[0-7]|[1-9])[- ]?(?:yo|y/o|years?[- ]old))\b"
    r"|ילד|ילדה|ילדים|קטין|קטינה|נערה|תינוק|ребён\w*|ребен\w*|девочк\w*|мальчик\w*|несовершеннолет\w*", re.I)
_NEAR = 60                                              # characters between the two words, at most


def _rule_minors(text: str) -> bool:
    sexual = [m.start() for m in _CHAT_SEXUAL.finditer(text)]
    if not sexual:
        return False
    return any(abs(m.start() - s) <= _NEAR for m in _CHAT_MINOR.finditer(text) for s in sexual)


_ANSWERS = {"ALLOW", *(c.upper() for c in CATEGORIES)}


def parse_verdict(text: str) -> str | None:
    """The model's answer -> a category id, "allow", or None when it said something else. It used to take the first
    category word anywhere, so "Is this MALWARE? No - ALLOW" read as malware (audit, 31 Oct 2026). Now: the answer's
    own first word when it is one (the format asked for), else a labelled "Verdict: X", else the last one named.
    Reasoning a model shows (<think>...</think>) is never read."""
    t = re.sub(r"<think>.*?(?:</think>|$)", " ", str(text or ""), flags=re.S | re.I)
    first = re.match(r"\W*([A-Za-z_]{4,})", t)
    if first and first.group(1).upper() in _ANSWERS:
        return first.group(1).lower()
    labelled = [w.upper() for w in re.findall(r"(?:verdict|answer|decision|category|final)\W{1,4}([A-Za-z_]{4,})", t, re.I)]
    named = [w for w in labelled if w in _ANSWERS] or [w for w in re.findall(r"[A-Z_]{4,}", t.upper()) if w in _ANSWERS]
    return named[-1].lower() if named else None


def screen(message: str, previous: str = "", ask_model=None) -> dict:
    """{"action": "allow" | "refuse" | "care", "category", "via"} for one user message.
    ask_model(system, user) -> the model's text; None skips the model check (rules only)."""
    text = str(message or "").strip()
    cfg = load()
    enabled = set(cfg["categories"])
    allow = {"action": "allow", "category": None, "via": None}
    rule_hit = _rule_minors(text) or bool(previous and _rule_minors(f"{previous[-_NEAR:]}\n{text}"))
    # Very short messages ("ok", "thanks", "go on") skip the check - unless abuse is watched and it's two words or
    # more ("f*** you" is short too).
    if len(text) < MIN_CHECK_CHARS and not rule_hit and ("abuse" not in enabled or len(text.split()) < 2 or set(re.findall(r"[a-z]+", text.lower())) <= _ACK_WORDS):
        return allow
    import intent_router
    if not rule_hit and intent_router.is_greeting(text):
        return allow
    verdict = None
    if ask_model is not None:
        user = (f"Previous user message (context only): {previous[:400]}\n\n" if previous else "") + \
               f"Message to screen: {text[:3000]}"
        try:
            verdict = parse_verdict(ask_model(_POLICY, user))
        except Exception:
            verdict = None
    if verdict is None:                                 # no usable check: the rules decide
        return {"action": "refuse", "category": "minors", "via": "rules"} if rule_hit else allow
    if verdict == "allow" or verdict not in enabled:
        return allow
    if verdict == "self_harm":
        return {"action": "care", "category": verdict, "via": "model"}
    return {"action": "refuse", "category": verdict, "via": "model"}


# ---- strikes and locks (on the chat dict itself, saved with the chat) ------------------------------------
def is_locked(chat: dict | None) -> bool:
    return bool(isinstance(chat, dict) and (chat.get("guard") or {}).get("locked"))


def _counts(g: dict) -> dict:
    """Per-category strike counts (a chat saved before they were per category had one total)."""
    s = g.get("strikes")
    if isinstance(s, dict):
        return {k: int(v) for k, v in s.items() if isinstance(v, (int, float)) and v > 0}
    last = next((e.get("c") for e in reversed(g.get("log") or []) if e.get("c") in COUNTED), "malware")
    return {last: int(s)} if isinstance(s, (int, float)) and s > 0 else {}


def strike_total(g: dict) -> int:
    return sum(_counts(g or {}).values())


def record_refusal(chat: dict, category: str, via: str = "model") -> dict:
    """Count a refusal in its own category; lock the chat at that category's limit (at once - and for good - for
    `minors` the model confirmed). A `minors` refusal on the phrases alone (no model check could run) refuses that
    message but doesn't close the chat: a phrase is never proof (1.4.6). Returns the chat's guard state."""
    cfg = load()
    g = dict(chat.get("guard") or {})
    counts = _counts(g)
    counts[category] = counts.get(category, 0) + 1
    g.update(strikes=counts, clean=0)
    g["log"] = (list(g.get("log") or []) + [{"c": category, "at": time.time(), **({"via": via} if via != "model" else {})}])[-20:]
    if category == "minors" and via != "model":
        counts.pop(category, None)                      # not a strike either: nothing confirmed it
    elif category == "minors":
        g.update(locked=True, reason="minors", permanent=True, locked_at=time.time())
    elif counts[category] >= cfg["limits"].get(category, cfg["strike_limit"]):
        g.update(locked=True, reason=category, locked_at=time.time())
    chat["guard"] = g
    return g


def note_clean(chat: dict | None) -> bool:
    """A message that passed the limits. Every `decay_messages` of them, each category's count drops by one.
    True when the counts changed."""
    if not isinstance(chat, dict) or not chat.get("guard") or chat["guard"].get("locked"):
        return False
    g = dict(chat["guard"])
    counts, every = _counts(g), load()["decay_messages"]
    if not counts or not every:
        return False
    g["clean"] = int(g.get("clean") or 0) + 1
    changed = g["clean"] >= every
    if changed:
        counts = {k: v - 1 for k, v in counts.items() if v > 1}
        g["clean"] = 0
    g["strikes"] = counts
    chat["guard"] = g
    return changed


def refusal_text(category: str, guard: dict) -> str:
    if guard.get("locked"):
        return LOCKED_SEVERE if category == "minors" else LOCKED_ABUSE if category == "abuse" else LOCKED
    if category == "abuse":
        return ABUSE_WARN
    return REFUSAL.format(label=CATEGORIES[category]["label"]) + STRIKE_NOTE


def unlock(chat: dict) -> dict:
    """Reopen a closed chat (the hidden gesture): clears the lock and the counts (kept in the log). A chat closed for
    `minors` stays closed - nothing reopens it."""
    g = dict(chat.get("guard") or {})
    if g.get("permanent") or g.get("reason") == "minors":
        raise GuardError("This chat was closed for sexual content involving minors. It can't be reopened.", "permanent")
    g["log"] = (list(g.get("log") or []) + [{"c": "unlocked", "at": time.time()}])[-20:]
    g.update(locked=False, strikes={}, clean=0)
    g.pop("reason", None)
    g.pop("locked_at", None)
    chat["guard"] = g
    return g
