"""
model_output.py
===============
Cleans up what a model returns before the agent acts on it.

Reasoning models (DeepSeek-R1 distills, Qwen3, QwQ ...) often write their chain
of thought into `content` as <think> ... </think>, depending on the chat
template and llama-server's --reasoning-format. Left in place that text:

  * reaches the user as if it were the answer, and
  * gets scanned by the text tool-call recovery, which can then "recover" a
    tool call the model merely considered while thinking - and run it.

split_reasoning() removes it and returns it separately, so it can be logged
without ever being treated as an answer or an action.
"""
from __future__ import annotations

import re

# Tag names used by common reasoning templates.
_TAGS = ("think", "thinking", "reasoning", "thought")
_BLOCK_RE = re.compile(r"<\s*(%s)\s*>(.*?)<\s*/\s*\1\s*>" % "|".join(_TAGS), re.I | re.S)
_OPEN_RE = re.compile(r"<\s*(%s)\s*>" % "|".join(_TAGS), re.I)
_CLOSE_RE = re.compile(r"<\s*/\s*(%s)\s*>" % "|".join(_TAGS), re.I)


# 1.4.4: small models sometimes open a reply with their own planning, outside any <think> tag - "The user said
# something vague, I should clarify. ..." - and only then talk to the user. Those opening sentences go; the
# reply's own words stay. Only at the very start, only while every sentence is about the user's message or the
# plan for answering, and never when nothing would be left.
_SELF_TALK_RE = re.compile(
    r"^(?:okay,?\s+|ok,?\s+|alright,?\s+|so,?\s+|hmm,?\s+)?(?:"
    r"(?:the|this) user (?:just |has |simply |only )?(?:said|says|sent|wrote|typed|asked|asks|wants|wanted|greeted|"
    r"started|opened|replied|responded|seems|appears|is (?:just |simply )?(?:asking|saying|greeting|testing|sending|"
    r"starting|being|probably|likely|not))\b|"                                    # "The user said / wants / is asking"
    r"(?:the|this) user's (?:message|input|request|reply|question|prompt)\b|"
    r"(?:since|as|because) the user (?:said|wrote|sent|asked|only|just|hasn't|didn't)\b|"
    r"(?:the|this|their) (?:message|input|request|prompt|query) (?:is|was|seems|appears|looks|reads|just)\b|"
    r"(?:it|this) (?:seems|appears|looks) (?:like )?(?:the user|they)\b|"
    r"(?:i'll|we'll|i'm going to|(?:i|we) (?:should|need to|must|will|am going to|want to)) (?:ask|clarify|respond|reply|answer|greet|keep|"
    r"acknowledge|prompt|invite|check what|find out|figure out|be (?:brief|concise|friendly|helpful))\b|"
    r"let me (?:ask|clarify|respond|reply|greet|acknowledge|keep it)\b|"
    r"my (?:response|reply|answer) (?:should|will)\b|"
    # Talking about the message to the user's face (Ornith 9B, 29 Oct 2026): "Nothing to do yet - your message was
    # just "okay"", "\"hm\" on its own isn't a task I can act on".
    r"nothing to do (?:yet|here)\b|"
    r"your (?:message|input|request) (?:was|is) (?:just|only|simply)\b|"
    r"(?:\"[^\"\n]{1,24}\"|'[^'\n]{1,24}'|that|this) (?:on its own |alone |by itself )?(?:isn't|is not|doesn't|does not|"
    r"isn't really) (?:a |an |really a )?(?:task|request|question|instruction|clear request|something i can)\b)", re.I)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+[\"'”’)\]]*|\n+|$)\s*")      # a closing quote stays with it


def strip_self_talk(text: str) -> str:
    """The reply without opening sentences in which the model talks to itself about the user's message."""
    if not text or not _SELF_TALK_RE.match(text.lstrip()):
        return text
    rest = text.lstrip()
    while rest and _SELF_TALK_RE.match(rest):
        m = _SENTENCE_RE.match(rest)
        if not m or not m.group(0).strip():
            break
        rest = rest[m.end():].lstrip()
    return rest if rest.strip() else text


class SelfTalkGate:
    """(1.4.8) The live draft without the opening self-talk. strip_self_talk cleans the finished reply, but while it
    streamed the chat showed "The user said..., I should..." as the answer and then deleted it - the owner saw every
    reply write and erase its first sentences. The opening is held until its first sentence is whole: self-talk goes
    to the thinking channel (it is thinking), anything else is released as the answer. Held at most HOLD characters.
    feed(text) / finish() -> [(channel, text), ...] to publish."""
    HOLD = 240

    def __init__(self):
        self.buf, self.mode = "", "probe"        # probe -> talk (sentences go to thinking) -> pass (all is answer)

    def feed(self, text: str) -> list:
        if self.mode == "pass":
            return [("content", text)] if text else []
        self.buf += text or ""
        return self._drain(final=False)

    def finish(self) -> list:
        if self.mode == "pass" or not self.buf:
            return []
        return self._drain(final=True)

    def _drain(self, final: bool) -> list:
        out = []
        while self.buf and self.mode != "pass":
            head = self.buf.lstrip()
            if not head:
                break                             # only whitespace so far
            if not _SELF_TALK_RE.match(head):
                if len(head) < 24 and not final and not re.search(r"[.!?\n]", head):
                    break                         # too little to tell yet ("The user" could still be "The users table")
                self.mode = "pass"
                out.append(("content", self.buf))
                self.buf = ""
                break
            m = _SENTENCE_RE.match(head)
            whole = m and m.group(0).strip() and (m.end() < len(head) or re.search(r"[.!?\n]\s*$", m.group(0)))
            if not whole:
                if final or len(head) > self.HOLD:     # no sentence end for a long while: it is the answer after all
                    self.mode = "pass"
                    out.append(("content", self.buf))
                    self.buf = ""
                break
            self.mode = "talk"
            out.append(("reasoning", m.group(0)))
            self.buf = head[m.end():]
        if final and self.buf:
            out.append(("content", self.buf))
            self.buf = ""
        return out


def split_reasoning(content: str) -> tuple[str, str]:
    """Return (answer, reasoning). Handles closed blocks, a reply cut off
    inside an unclosed <think> (everything after it is reasoning), and a
    template that opened <think> in the prompt so only </think> appears."""
    if not content:
        return content or "", ""
    reasoning = []

    def _take(match):
        reasoning.append(match.group(2).strip())
        return " "

    text = _BLOCK_RE.sub(_take, content)
    # Only a closing tag: the template put <think> in the prompt itself.
    close = _CLOSE_RE.search(text)
    if close and not _OPEN_RE.search(text[:close.start()]):
        reasoning.append(text[:close.start()].strip())
        text = text[close.end():]
    # An opening tag with no close: the reply ran out mid-thought.
    open_ = _OPEN_RE.search(text)
    if open_:
        reasoning.append(text[open_.end():].strip())
        text = text[:open_.start()]
    return text.strip(), "\n\n".join(r for r in reasoning if r)


def looks_truncated_json(text: str) -> bool:
    """A reply that opens more braces than it closes was cut off by the
    token limit - usually mid tool call with a long argument."""
    depth = in_str = esc = 0
    for ch in text or "":
        if in_str:
            if esc:
                esc = 0
            elif ch == "\\":
                esc = 1
            elif ch == '"':
                in_str = 0
            continue
        if ch == '"':
            in_str = 1
        elif ch in "{[":
            depth += 1
        elif ch in "}]" and depth:
            depth -= 1
    return depth > 0
