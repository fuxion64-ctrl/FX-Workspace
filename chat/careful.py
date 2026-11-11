"""
careful.py
==========
Careful and honest: the assistant says when it doesn't know, checks facts instead of making them up, and keeps
what it found apart from what it guesses. Inspired by the honesty principles Anthropic publishes for Claude -
truthful, calibrated, transparent, non-deceptive - written here for a small local model. It works with every
persona (MossadGPT too).

Since D47 the rules (PROMPT) are part of Enhanced mode's core (agent/prompts.ENHANCED_CORE) and always sent in
Enhanced; the careful_mode setting ("Check answers", on by default) turns parts 2-4 on and off. Raw mode
(response_modes.py) has none of it.

Four parts, from cheapest to most involved:
  1. PROMPT - the rules, in the system prompt (stable across a chat, so llama.cpp's prompt cache survives).
  2. checkable(message) - a question about specific facts (who, when, how many, prices, records, "is it true")
     gets Grounded answers turned on by itself (agent/auto_tools.py): it searches and cites before answering.
  3. review(...) after the answer, deterministic, no extra model call:
       - a citation [n] that points at no source this turn is removed (an invented reference),
       - a link the tools never returned is flagged (an invented URL),
       - an answer to a factual question that states specific figures with no source is marked
         "from memory" - the page shows it under the reply with a "Check it" button.
  4. The reply keeps the verdict (sessions extras "ck"), so it survives a reload.
  Also after the answer: an action request answered with "I've opened / saved / deleted ..." while no tool changed
  anything this turn is flagged (claims_without_change) - the reply says it did something nothing did.
"""
from __future__ import annotations

import re

DEFAULT_SETTINGS = {"careful_mode": True}

PROMPT = (
    "BE TRUTHFUL:\n"
    "- Say plainly when you don't know or aren't sure, and how sure you are when it matters. An honest \"I don't "
    "know\" beats a confident guess.\n"
    "- Never invent facts, numbers, names, dates, quotes, sources or links. Check them with the tools you are given; "
    "if you can't check something, say it is unverified.\n"
    "- Keep what a source or tool showed apart from what you infer or estimate, and label estimates as estimates.\n"
    "- If a question rests on a wrong premise, say so politely before answering. Disagree when you have good reason - "
    "agreeing just to please is not help.\n"
    "- Never claim to have done, seen, run or checked anything you haven't. Admit a mistake directly.\n"
    "- When sources disagree or the evidence is thin, say so instead of quietly picking a side."
)


def clean_settings(s: dict) -> dict:
    return {"careful_mode": s.get("careful_mode") if isinstance(s.get("careful_mode"), bool) else True}


def on(settings: dict | None) -> bool:
    """The checks after the answer (and auto grounding) - never in Raw mode, which is the model alone."""
    settings = settings or {}
    return clean_settings(settings)["careful_mode"] and settings.get("response_mode") != "raw"


def block(settings: dict | None) -> str:
    """Kept for callers outside the agent prompt; the agent prompt carries PROMPT in its core (D47)."""
    return PROMPT if on(settings) else ""


# ---- which questions need checking ---------------------------------------------------------------------
# A question about specific, checkable facts - the kind a model fills in from memory and gets wrong.
_FACT_Q = re.compile(
    r"\b(?:who (?:is|was|are|were|won|wrote|invented|founded|owns|runs|said)|when (?:is|was|did|does|will)|"
    r"what (?:year|date|day|time)|how (?:many|much|old|tall|far|long ago)|which (?:year|country|company|team)|"
    r"is it true|did .{1,40} (?:really|actually)|fact[- ]?check|according to|statistics? (?:on|for|about)|"
    r"population of|capital of|price of|net worth|record for|latest|current(?:ly)?|right now|today|this (?:week|month|year)|"
    r"released|announced|elected|died|born)\b"
    r"|מי (?:זה|היה|הוא|היא|ניצח)|מתי|כמה|האם זה נכון|"
    r"кто (?:такой|был|выиграл)|когда|сколько|правда ли", re.I)
# Questions that don't depend on outside facts: maths, code, writing, opinions, the conversation itself.
_NOT_FACT = re.compile(r"\b(?:write|rewrite|draft|translate|code|script|function|regex|poem|story|joke|"
                       r"calculate|solve|simplify|derive|explain how|how do i|how to|should i|what do you think|"
                       r"summari[sz]e (?:this|the (?:above|text))|this file|the attached)\b", re.I)


def checkable(message: str) -> bool:
    m = str(message or "")
    return bool(_FACT_Q.search(m)) and not _NOT_FACT.search(m) and len(m) < 600


# ---- after the answer ------------------------------------------------------------------------------------
_CITE = re.compile(r"\s?\[(\d{1,3})\](?!\()")
_URL = re.compile(r"https?://[^\s)\]>\"']+")
# Specific figures a reader might take as checked: years, percentages, money, big or precise numbers.
_FIGURE = re.compile(r"\b(?:1[5-9]\d\d|20\d\d)\b|\d+(?:\.\d+)?\s?%|[$€£₪]\s?\d|\b\d{1,3}(?:,\d{3})+\b|\b\d+\.\d+\b|"
                     r"\b\d+\s?(?:million|billion|trillion|km|kg|mph|kph|metres|meters|feet|miles)\b", re.I)


def _strip_code(text: str) -> str:
    return re.sub(r"```.*?```", " ", str(text or ""), flags=re.S)


def review(answer: str, question: str, registry, tools_used: bool, web_available: bool) -> tuple[str, dict]:
    """(the answer with invented citations removed, the verdict for the page). The verdict:
    {"removed": [n...], "links": [url...], "from_memory": bool, "checked": int (sources cited)}."""
    text = str(answer or "")
    known = len(getattr(registry, "items", []) or [])
    removed: list = []

    def drop(m):
        n = int(m.group(1))
        if 1 <= n <= known:
            return m.group(0)
        removed.append(n)
        return ""

    # Only outside code blocks: "[0]" in code is an index, not a citation.
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    text = "".join(p if p.startswith("```") else _CITE.sub(drop, p) for p in parts)
    prose = _strip_code(text)
    seen_urls = {str(i.get("url") or "") for i in getattr(registry, "items", []) or []}
    seen_keys = set()
    try:
        import sources as _sources
        seen_keys = {_sources.canonical(u) for u in seen_urls if u}
        canon = _sources.canonical
    except Exception:                     # never costs the answer
        canon = lambda u: u               # noqa: E731
    given = {canon(u.rstrip(".,;:")) for u in _URL.findall(str(question or ""))}     # links the user gave are fine
    links = []
    # (1.4.8) Not inside code: an HTML page's image and script addresses are part of the code it was asked to write,
    # and listing them under every generated site read as a warning about the answer (11 Nov, owner).
    outside_code = re.sub(r"```.*?(?:```|$)", " ", prose, flags=re.S)
    for u in _URL.findall(outside_code):
        u = u.rstrip(".,;:")
        key = canon(u)
        if key not in seen_keys and key not in given and u not in links and not re.match(r"https?://(?:127\.0\.0\.1|localhost)", u):
            links.append(u)
    cited = len({int(n) for n in _CITE.findall(prose)})
    from_memory = bool(checkable(question) and _FIGURE.search(prose) and not cited and not tools_used)
    verdict = {"removed": sorted(set(removed))[:20], "links": links[:10], "from_memory": from_memory,
               "checked": cited, "could_check": bool(web_available)}
    return text, verdict


def worth_showing(verdict: dict) -> bool:
    return bool(verdict and (verdict.get("removed") or verdict.get("links") or verdict.get("from_memory")
                             or verdict.get("unbacked")))


# ---- "I've opened it" when nothing ran -------------------------------------------------------------------
# A reply to an action request that reports a finished change in the first person. Only checked when no tool
# changed anything this turn: then the claim has nothing behind it (ground rule 2, agent/prompts.GROUND_RULES).
_DONE_VERBS = (r"opened|created|saved|written|wrote|deleted|removed|moved|renamed|installed|uninstalled|launched|started|"
               r"closed|sent|posted|ran|executed|changed|updated|edited|added|inserted|built|made|set|turned (?:on|off)|"
               r"enabled|disabled|connected|disconnected|downloaded|copied|replaced|fixed|applied|scheduled|stacked|placed|"
               r"generated")
_DONE_CLAIM = re.compile(
    r"\b(?:I(?:'ve| have)?(?: just| now| already| successfully)?|I've now)\s+(?:" + _DONE_VERBS + r")\b"
    r"(?!\s+(?:is|are)\s+(?:below|above|here))"           # "the script I wrote is below" is text, not a change
    # "Saved your notes to notes.txt." - the same claim without the "I", at the start of a line or a sentence
    r"|(?:^|(?<=[.!?] ))[ \t]*(?:\*\*)?(?:" + _DONE_VERBS + r")\b(?= \w)"
    r"|^\s*(?:\*\*)?(?:done|all done|completed|finished)\b",
    re.I | re.M)


def claims_without_change(answer: str, action_turn: bool, changed: bool) -> str:
    """The first sentence that reports a change no tool made ("" when there is none, or when something did change)."""
    if not action_turn or changed:
        return ""
    prose = _strip_code(answer)
    m = _DONE_CLAIM.search(prose)
    if not m:
        return ""
    # The sentence around it: a sentence ends at . ! or ? followed by a space or the end ("notes.txt" doesn't).
    before = [b.end() for b in _SENT_END.finditer(prose, 0, m.start())]
    start = before[-1] if before else 0
    after = _SENT_END.search(prose, m.end())
    end = after.start() + 1 if after else len(prose)
    return " ".join(prose[start:end].split())[:200]


_SENT_END = re.compile(r"[.!?](?=\s|$)|\n")
