"""Deterministic preflight routing for short desktop commands.

The language model is deliberately *not* asked to decide what an explicit app
name means.  This module runs before inference, resolves product families such
as "Proton", and emits either a clarification or a constrained desktop intent.
That prevents a small model from turning "open Proton VPN" into a browser
search or an unrelated application.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Clarification:
    question: str
    options: list[str]
    context: str = ""
    action: str = ""
    remaining_task: str = ""

    def as_response(self) -> dict:
        if self.action == "connect":
            option_values = [f"Connect to {value}" for value in self.options[:6]]
        elif self.action == "open":
            option_values = [f"Open {value}" for value in self.options[:6]]
        else:
            option_values = self.options[:6]
        if self.remaining_task:
            option_values = [f"{value} and {self.remaining_task}" for value in option_values]
        return {
            "status": "clarification",
            "message": self.question,
            "question": self.question,
            "options": self.options[:6],
            "option_values": option_values,
            "allow_other": True,
            "context": self.context,
        }


@dataclass
class DesktopIntent:
    app_name: str
    actions: list[str] = field(default_factory=list)
    original: str = ""
    remaining_task: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass
class ChatIntent:
    """A request about this chat itself ("close this chat") - done by the app, never by a desktop tool."""
    action: str
    original: str = ""


# "close this chat", "end the conversation", "exit chat": the user means this conversation in the app, not a
# window called "this chat" - routed to the desktop it closed nothing and claimed "already closed".
_CHAT_CLOSE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:can|could)\s+you\s+)?(?:close|end|exit|quit|leave|finish|stop)\s+"
    r"(?:(?:this|the|our|current|my)\s+)?(?:chat|conversation|convo|thread|session)(?:\s+(?:now|here|for\s+me))?"
    r"(?:\s+please)?[.!?]*\s*$", re.IGNORECASE)


# A product family is intentionally explicit.  Suggestions are useful even
# when only one of these products is installed: the user's phrase may refer to
# a web product, while desktop execution must never silently assume which one.
PRODUCT_FAMILIES = {
    "proton": {
        "vpn": "Proton VPN",
        "mail": "Proton Mail",
        "drive": "Proton Drive",
        "pass": "Proton Pass",
        "calendar": "Proton Calendar",
    },
}

ACTIVE_MEDIA = "active media"

_OPEN = r"(?:open|launch|start|run)"
_SIMPLE_OPEN_RE = re.compile(
    rf"^\s*(?:please\s+)?{_OPEN}\s+(?P<target>.+?)(?:\s+please)?[.!?]*\s*$",
    re.IGNORECASE,
)
_SIMPLE_CONNECT_RE = re.compile(
    r"^\s*(?:please\s+)?connect(?:\s+me)?(?:\s+to)?\s+(?P<target>.+?)(?:\s+please)?[.!?]*\s*$",
    re.IGNORECASE,
)
_SIMPLE_CLOSE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:close|exit|quit)\s+(?P<target>.+?)(?:\s+please)?[.!?]*\s*$",
    re.IGNORECASE,
)
_CONNECT_TAIL_RE = re.compile(
    r"[\s,;]+(?:and\s+)?(?:then\s+)?(?:quick\s+)?connect(?:\s+(?:it|me))?\s*$",
    re.IGNORECASE,
)


def _clean_target(value: str) -> str:
    value = value.strip(" \t\r\n\"'.,!?")
    value = re.sub(r"\b(?:the|application|app)\b", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def split_tasks(message: str) -> list[str]:
    """Split imperative task boundaries before any app lookup or model call.

    The boundary vocabulary is deliberately larger than the router vocabulary:
    an unknown clause must remain isolated instead of leaking into the previous
    app name and being fuzzy-matched to an unrelated installed program.
    """
    parts = re.split(
        r"\s*(?:,\s*(?:(?:and|then)\s+)?|\band\s+(?:then\s+)?|\bthen\s+|;\s*)"
        r"(?=(?:(?:stop|pause|play|resume|skip|prev|previous|next|scan|check|status|open|launch|start|run|close|exit|quit|search|write|create|delete|send|establish|enable|disconnect)\b|connect\s+(?!(?:it|me)\b)(?:to\s+)?\w))",
        message.strip(), flags=re.I)
    return [part.strip() for part in parts if part.strip()]


def _contains_task_boundary(value: str) -> bool:
    """Reject a target that still contains another imperative task. (1.4.8) Not only the listed verbs: "open paint
    and draw a horse" looked for an app called "paint and draw a horse" (11 Nov). Any "then", and an "and" followed by
    two or more words, is a second task for the agent; a name like "Tom and Jerry" (one word after it) stays a name."""
    return bool(re.search(
        r"(?:,|;|\band\b|\bthen\b)\s+(?:stop|pause|play|resume|skip|prev|previous|next|scan|check|status|open|launch|start|run|close|exit|quit|"
        r"search|write|create|delete|send|establish|enable|disconnect|connect)\b",
        value, re.I) or re.search(r"\bthen\b|\band\s+\S+\s+\S+", value, re.I))


def route(message: str):
    """Return DesktopIntent, Clarification, or None.

    Only short, imperative app-opening requests are intercepted.  Richer tasks
    still go to the agent, but can use ``expected_app`` as a safety constraint.
    """
    message = message or ""
    # Split only at an imperative clause, not at 'and' inside an app name.
    clauses = split_tasks(message)
    if len(clauses) > 1:
        first = route(clauses[0].strip())
        if isinstance(first, (DesktopIntent, Clarification)):
            first.remaining_task = " and ".join(clauses[1:])
            if isinstance(first, DesktopIntent):
                first.original = message
            else:
                first.context = message
            return first
        return None
    # Generic playback means resume the existing queue, not invent a playlist.
    # Requests naming a song/playlist are deliberately left to the model.
    playback = re.fullmatch(
        r"\s*(?:please\s+)?(?:play|resume)\s+(?:music|playback|the\s+(?:current\s+)?(?:music|playlist|queue))"
        r"\s+(?:in|on|with|using)\s+([\w .-]+?)[.!?]*\s*", message, re.I)
    if playback:
        return DesktopIntent(_clean_target(playback.group(1)), ["play"], message)
    stop = re.fullmatch(r"\s*(?:please\s+)?(stop|pause)\s+(?:music|playback|the music)\s+(?:in|on|with)\s+([\w .-]+?)[.!?]*\s*", message, re.I)
    if stop:
        return DesktopIntent(_clean_target(stop.group(2)), [stop.group(1).lower()], message)
    generic_stop = re.fullmatch(
        r"\s*(?:please\s+)?(stop|pause)\s+(?:music|playback|the music)[.!?]*\s*", message, re.I)
    if generic_stop:
        return DesktopIntent(ACTIVE_MEDIA, [generic_stop.group(1).lower()], message)
    transport = re.fullmatch(
        r"\s*(?:please\s+)?(?P<action>skip|prev(?:ious)?|next)(?:\s+(?:song|track))?"
        r"(?:\s+(?:in|on|with)\s+(?P<app>[\w .-]+?))?[.!?]*\s*", message, re.I)
    if transport:
        word = transport.group("action").casefold()
        action = "previous" if word.startswith("prev") else "next"
        app = _clean_target(transport.group("app") or "") or ACTIVE_MEDIA
        return DesktopIntent(app, [action], message)

    # VPN state is a real, read-only app state. Compile it rather than making
    # the model inspect pixels or ask which product "VPN" means.
    vpn_status = re.fullmatch(
        r"\s*(?:please\s+)?(?:check|show|read|what(?:'s| is)|is)\s+(?:the\s+)?(?:proton\s+)?vpn"
        r"(?:\s+connection)?(?:\s+status|\s+connected)?[?!.]*\s*", message, re.I)
    if vpn_status:
        return DesktopIntent("Proton VPN", ["status"], message)
    disconnect = re.fullmatch(
        r"\s*(?:please\s+)?disconnect(?:\s+(?:from\s+)?)?(?:the\s+)?(?:proton\s+)?vpn"
        r"(?:\s+connection)?[.!?]*\s*", message, re.I)
    if disconnect:
        return DesktopIntent("Proton VPN", ["disconnect"], message)
    # An explicit 'scan ... with APP' is not a product-family question.
    scan = re.fullmatch(r"\s*(?:please\s+)?scan\s+(?:drive\s+)?([a-z])(?::)?\s+(?:with|using|in)\s+(.+?)\s*[.!?]?", message, re.I)
    if scan:
        target = _clean_target(scan.group(2))
        if _contains_task_boundary(target):
            return None
        return DesktopIntent(target, ["scan"], message, "",
                             {"drive": f"{scan.group(1).upper()}:"})
    # A bare drive scan has one deterministic local meaning in this app:
    # inspect the requested volume with WizTree. It must never fall through to
    # a small model, which can only narrate a fake list_elements call.
    bare_scan = re.fullmatch(
        r"\s*(?:please\s+)?scan\s+(?:drive\s+)?([a-z])(?::)?[.!?]*\s*", message, re.I)
    if bare_scan:
        return DesktopIntent("WizTree", ["scan"], message, "",
                             {"drive": f"{bare_scan.group(1).upper()}:"})

    # Natural VPN wording is compiled to the same deterministic connect
    # action. Mentioning VPN disambiguates the Proton product by itself.
    establish = re.fullmatch(
        r"\s*(?:please\s+)?(?:establish|enable|start)\s+(?:a\s+|the\s+)?vpn(?:\s+connection)?"
        r"\s+(?:in|on|with|using|through|via)\s+(.+?)[.!?]*\s*", message, re.I)
    if establish:
        target = _clean_target(establish.group(1))
        if _contains_task_boundary(target):
            return None
        if target.casefold() == "proton":
            target = "Proton VPN"
        return DesktopIntent(target, ["connect"], message)
    if _CHAT_CLOSE_RE.match(message):
        return ChatIntent("close", message)
    close = _SIMPLE_CLOSE_RE.match(message)
    if close:
        target = _clean_target(close.group("target"))
        if not target or _contains_task_boundary(target):
            return None
        return DesktopIntent(target, ["close"], message)
    match = _SIMPLE_OPEN_RE.match(message)
    action = "open"
    actions: list[str] = []
    if match:
        raw_target = match.group("target")
        # "open Test1 in studio" / "open Test1 in Roblox Studio": the place
        # Test1, not an app called "Test1 in studio".
        in_studio = re.search(r"\s+(?:in|with|using|on)\s+(?:roblox\s+)?studio\s*$", raw_target, re.I)
        if in_studio and raw_target[:in_studio.start()].strip():
            raw_target = raw_target[:in_studio.start()]
            return DesktopIntent(_clean_target(raw_target), [], message, "", {"place_only": True})
        connect = _CONNECT_TAIL_RE.search(raw_target)
        if connect:
            actions.append("connect")
            raw_target = raw_target[:connect.start()]
    else:
        match = _SIMPLE_CONNECT_RE.match(message)
        if not match:
            return None
        action = "connect"
        actions = ["connect"]
        raw_target = match.group("target")
    target = _clean_target(raw_target)
    if _contains_task_boundary(target):
        return None
    # A file path or URL is not an installed app - let the agent open it
    # with open_file_in_os / open_url instead of fuzzy-matching app names.
    if re.search(r"^[a-z]:[\\/]|[\\/]|^~|^https?:|^www\.", target, re.I):
        return None
    if not target:
        verb = "connect with" if action == "connect" else "open"
        return Clarification(f"Which application should I {verb}?", [], message, action)

    words = set(re.findall(r"[a-z0-9]+", target.lower()))
    for family_name, products in PRODUCT_FAMILIES.items():
        if family_name not in words:
            continue
        selected = [(key, label) for key, label in products.items() if key in words]
        if len(selected) == 1:
            return DesktopIntent(selected[0][1], actions, message)
        if not selected:
            verb = "do you want to connect with" if action == "connect" else "do you mean"
            return Clarification(
                f"Which {family_name.title()} product {verb}?",
                list(products.values())[:3],
                message,
                "connect" if actions else action,
            )

    return DesktopIntent(target, actions, message)


def serialize_route(routed) -> dict:
    """Store a compiled task without retaining live Python objects in state."""
    if isinstance(routed, DesktopIntent):
        return {"kind": "desktop", "app_name": routed.app_name,
                "actions": list(routed.actions), "original": routed.original,
                "remaining_task": routed.remaining_task,
                "arguments": dict(routed.arguments)}
    if isinstance(routed, Clarification):
        return {"kind": "clarification", "question": routed.question,
                "options": list(routed.options), "context": routed.context,
                "action": routed.action, "remaining_task": routed.remaining_task}
    return {"kind": "model"}


def deserialize_route(compiled: dict):
    if compiled.get("kind") == "desktop":
        return DesktopIntent(
            compiled.get("app_name", ""), list(compiled.get("actions") or []),
            compiled.get("original", ""), compiled.get("remaining_task", ""),
            dict(compiled.get("arguments") or {}))
    if compiled.get("kind") == "clarification":
        return Clarification(
            compiled.get("question", ""), list(compiled.get("options") or []),
            compiled.get("context", ""), compiled.get("action", ""),
            compiled.get("remaining_task", ""))
    return None


def compile_request(message: str) -> list[dict]:
    """Compile each task once, before execution mutates any desktop state."""
    clauses = split_tasks(message)
    compiled = []
    last_app = ""
    for clause in clauses:
        routed = route(clause)
        if routed is None and last_app:
            short_scan = re.fullmatch(
                r"\s*(?:please\s+)?scan\s+(?:drive\s+)?([a-z])(?::)?[.!?]*\s*", clause, re.I)
            if short_scan:
                routed = DesktopIntent(last_app, ["scan"], clause, "",
                                       {"drive": f"{short_scan.group(1).upper()}:"})
        if isinstance(routed, DesktopIntent):
            last_app = routed.app_name
        compiled.append({"text": clause, "status": "pending",
                         "compiled": serialize_route(routed)})
    return compiled


def resolve_answer(original_request: str, answer: str):
    """Resolve one clarification answer while preserving the original verb.

    This also accepts answers produced by an older frontend (for example the
    raw label ``Proton VPN`` rather than ``Connect to Proton VPN``).
    """
    original_route = route(original_request)
    remaining = getattr(original_route, "remaining_task", "")
    direct = route(answer)
    if isinstance(direct, DesktopIntent):
        if remaining and not direct.remaining_task:
            direct.remaining_task = remaining
        direct.original = original_request
        return direct

    original = original_request or ""
    answer = _clean_target(answer or "")
    if not answer:
        return None

    original_words = set(re.findall(r"[a-z0-9]+", original.lower()))
    answer_words = set(re.findall(r"[a-z0-9]+", answer.lower()))
    for family_name, products in PRODUCT_FAMILIES.items():
        if (family_name in original_words and family_name not in answer_words
                and answer_words.intersection(products.keys())):
            answer = f"{family_name.title()} {answer}"
            break

    if _SIMPLE_CONNECT_RE.match(original) or "connect" in original_words:
        resolved = route(f"Connect to {answer}")
    elif _SIMPLE_OPEN_RE.match(original) or original_words.intersection({"open", "launch", "start", "run"}):
        resolved = route(f"Open {answer}")
    else:
        return None
    if isinstance(resolved, DesktopIntent):
        resolved.remaining_task = remaining
        resolved.original = original_request
    return resolved


def is_new_request(message: str, pending: dict) -> bool:
    """Named choices are answers; an independent imperative starts a new task."""
    value = message.strip().casefold()
    choices = (pending.get("options") or []) + (pending.get("option_values") or [])
    if value in {str(choice).strip().casefold() for choice in choices}:
        return False
    selected = re.sub(r"^(?:open|connect(?:\s+to)?)\s+", "", value)
    if selected in {str(choice).strip().casefold() for choice in pending.get("options") or []}:
        return False
    return bool(re.match(r"^(?:please\s+)?(?:stop|pause|play|resume|skip|prev|previous|next|scan|check|status|open|launch|start|run|close|exit|quit|connect|search|write|create|delete|send|disconnect)\b", value))


def skips_current_step(message: str) -> bool:
    value = re.sub(r"[^a-z]+", " ", message.casefold()).strip()
    return (value.startswith("skip") or value in {
        "it s already playing", "it is already playing", "already playing", "continue to the next task"})


def expected_app(message: str) -> str | None:
    """Best explicit app constraint for the current turn, if one exists."""
    routed = route(message)
    return routed.app_name if isinstance(routed, DesktopIntent) else None


def is_greeting(message: str) -> bool:
    value = re.sub(r"\s+", " ", (message or "").strip().casefold())
    return bool(re.fullmatch(
        r"(?:hi|hello|hey|yo|sup|good (?:morning|afternoon|evening)|"
        r"thanks|thank you|thx)[!.? ]*", value))


# How people actually phrase a request, besides "<verb> ...":
#   lead-ins      "ok, make ...", "let's add ...", "I'd like to fix ...", "go ahead and ..."
#   a scope first "in Test1 make ...", "for my obby, add ...", "on the desktop create ..."
_LEAD_RE = re.compile(r"^(?:(?:ok(?:ay)?|so|now|then|also|and|alright|right|hey|hi|hello|yo|pls|plz|just|next|"
                      r"first|finally|great|cool|thanks|thank you|"
                      # spoken fillers and a leading yes/no ("No, you do it", "Um so yeah, open it")
                      r"no+|nope|nah|yes|yeah|yep|yup|u+m+|u+h+|uhm|erm|hm+|well|actually)[,!.]?\s+)+")
_ASK = (r"(?:(?:please|pls|plz)\s+|(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+)?|"
        r"i\s+(?:want|need|would like)\s+(?:you\s+)?to\s+|i'?d\s+like\s+(?:you\s+)?to\s+|i'?m\s+(?:going|trying)\s+to\s+|"
        r"let'?s\s+|let\s+us\s+|go\s+ahead\s+and\s+|try\s+(?:and|to)\s+|help\s+me(?:\s+to)?\s+|now\s+|then\s+|also\s+|just\s+|"
        # "you do run the script", "you just make it", "you should add ...", "go build ..."
        r"(?:you|u)\s+(?:(?:do|just|can|could|should|must|will|need\s+to|have\s+to|gotta)\s+)*|go\s+)")
_SCOPE = (r"(?:(?:in|on|inside|within|for|at|into|with|using|under|from|over)\s+"
          r"(?!(?:i|you|it|we|they|he|she|me|which|what|how)\b)(?:[\w'.\-]+\s+){0,3}?[\w'.\-]+,?\s+)")
# "do" opens a question ("do you know ...") - but "do it", "do that", "do the stacking" are requests.
_QUESTION_RE = re.compile(r"^(?:how|what|why|when|where|who|which|whose|is|are|was|were|does|"
                          r"do(?!\s+(?:it|that|this|so|them|those|these|all|everything|my|our)\b)|did|has|have|had|"
                          r"should|shall|may|might|can\s+i|could\s+i|would\s+i|am\s+i)\b")
# "... so do it pls", "it's open, run it": a request to act that isn't at the start of the message.
_DO_IT_RE = re.compile(r"(?:^|[,;.!]|\b(?:so|then|now|just|please|pls|plz|and|but))\s*(?:just\s+|please\s+)?"
                       r"(?:do|run|execute|make|build|create|finish|try)\s+(?:it|that|this|them|those|these|the\s+(?:script|code))\b")
# Not an answer to the assistant's question: a refusal ("no.", "nope, stop"), thanks, or a short reply
# that starts with its own verb - a new request ("tell me a joke"), not a detail of the old one.
_NOT_AN_ANSWER_RE = re.compile(r"^(?:no|nope|nah|not|don'?t|do not|never|nothing|stop|cancel|forget|nvm|never\s*mind|"
                               r"thanks|thank you|thx|ok|okay|remember|tell|explain|show|give|write|say|list|describe|"
                               r"find|search|help|read|check|look|summari[sz]e|translate|compare|define|teach)\b")
# A short reply that hands the decision back or says yes - it continues the
# request the assistant just asked about or offered to do.
_FOLLOW_UP_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|yes|yeah|yep|yup|sure|alright|all right|fine|please|pls|go|go ahead|do it|do that|proceed|"
    r"continue|carry on|keep going|sounds good|that works|perfect|great|up to you|it'?s up to you|your (?:call|choice)|"
    r"you (?:decide|choose|pick)|whatever you (?:think|want|prefer)(?: is best)?|(?:do|use) (?:whatever|what) you think"
    r"(?: is best)?|as you (?:like|see fit|think)|try (?:it|again)|retry|again|let'?s do (?:it|that)|make it so|"
    r"yes please|please do|go for it|any|either|the first one|the second one)[\s,.!]*)+$")
_OFFER_RE = re.compile(r"\?\s*$|\b(?:should i|shall i|do you want|would you like|want me to|if you(?:'d)? like|"
                       r"if you want|let me know|which (?:one|option)|i can\b)", re.I)


# Text the reply itself is (classify_turn): "write / draft / compose (me) a <poem | email | function ...>".
_TEXT_ANSWER_RE = re.compile(
    rf"^(?:{_ASK})*(?:write|draft|compose|give)\s+(?:me\s+|us\s+)?(?:(?:a|an|the|some|\d+|one|two|three|few|short|quick|"
    r"simple|small|nice|funny|polite|formal|python|javascript|typescript|js|c\+\+|c#|java|rust|go|sql|bash|regex|html|css)\s+)*"
    r"(?:function|method|class|regex|regular expression|one-?liner|snippet|query|poem|haiku|limerick|song|lyrics|joke|"
    r"story|essay|paragraph|email|e-mail|letter|cover letter|message|reply|tweet|post|caption|bio|slogan|speech|toast|"
    r"outline|summary|description|definition|example|algorithm)s?\b")
# ...unless it says where the text goes: then it's something to make on the PC.
_HAS_TARGET_RE = re.compile(
    r"\b(?:file|folder|directory|desktop|documents|downloads|disk|drive|save|saved|saving|store|into|doc|document|note|"
    r"notes|studio|roblox|lua|luau|place|run|execute|test it|and test|send|post it|clipboard)\b|[a-z]:\\|\.\w{2,4}\b",
    re.I)


def _follows_offer(message: str, previous: dict | None) -> bool:
    """A short yes / "up to you" right after the assistant asked or offered."""
    if not previous or not _FOLLOW_UP_RE.match(message):
        return False
    return bool(_OFFER_RE.search((previous.get("assistant") or "").strip()[-400:]))


def _answers_question(message: str, previous: dict | None) -> bool:
    """A short answer to the question the assistant just asked about an ACTION request
    ("stack 10 parts" -> "what should I stack?" -> "regular parts in Roblox Studio") continues
    that request - it isn't a new, context-free message. Only when the assistant's reply ends
    in a question, the request it asked about was an action, and the answer is short, not a
    question, not a refusal and not a new request of its own ("tell me ...")."""
    if not previous or not previous.get("request") or not message:
        return False
    if not (previous.get("assistant") or "").rstrip().endswith("?"):
        return False
    answer = _LEAD_RE.sub("", message)
    if (not answer or len(message.split()) > 16 or message.endswith("?") or _QUESTION_RE.match(answer)
            or _NOT_AN_ANSWER_RE.match(message) or _NOT_AN_ANSWER_RE.match(answer)):
        return False
    return classify_turn(previous["request"]).get("mode") == "action"


def classify_turn(message: str, with_vision: bool = False, previous: dict | None = None) -> dict:
    """Set the maximum tool authority granted by the current user message.

    This is a safety/relevance boundary, not semantic task planning. The model
    may choose fewer tools, but it cannot inherit mutation authority from an
    older request merely because that request remains in conversation history.
    The one exception is a short yes / "up to you" that answers the
    assistant's own question or offer: that reply continues the request it was
    about (previous = {"request", "assistant"}); every change still asks first.
    """
    value = re.sub(r"\s+", " ", (message or "").strip().casefold())
    multilingual = _multilingual_authority(value)
    if multilingual is not None:
        return multilingual
    if is_greeting(value):
        return {"mode": "none", "allow_mutation": False, "reason": "greeting"}
    if _follows_offer(value, previous) or _answers_question(value, previous):
        return {"mode": "action", "allow_mutation": True, "reason": "reply to the assistant's question or offer",
                "follows": previous.get("request") or ""}
    work = re.sub(r"^(?:hi|hello|hey|yo)[,!]\s*", "", value)

    # "Remember that ...", "keep in mind ...", "from now on ...": the only
    # authority granted is save_memory (which still goes to the approval card).
    if re.match(r"^(?:please\s+)?(?:remember|memori[sz]e|keep in mind|note that|don'?t forget|do not forget|"
                r"from now on|going forward)\b", work):
        return {"mode": "memory", "allow_mutation": False, "reason": "asked to remember something"}

    action = (
        r"open|launch|start|run|connect|disconnect|play|resume|pause|stop|skip|prev|previous|next|scan|check|status|close|click|press|"
        r"select|choose|type|write|create|edit|delete|remove|send|set|change|move|rename|"
        r"download|upload|install|uninstall|mute|enable|disable|restart|shutdown|power|"
        r"copy|paste|drag|scroll|establish|"
        # Building/fixing verbs - "Add a kill brick", "Fix the error in my
        # script", "Snap these side by side" are explicit requests to act.
        r"add|make|build|generate|insert|spawn|put|place|fix|debug|diagnose|repair|implement|refactor|"
        r"resume|continue|remind|schedule|save|undo|duplicate|clone|resize|anchor|arrange|organi[sz]e|"
        r"snap|tile|tidy|clean\s+up|color|colour|paint|script|code|update|replace|rewrite|"
        r"design|configure|set\s+up|setup|animate|test|playtest|publish|tweak|improve|polish|redo|finish|complete|"
        r"convert|restyle|style|rotate|scale|position|align|cent(?:er|re)|group|ungroup|weld|attach|draw|compile|"
        r"format|translate|fill|submit|record|capture|take|turn\s+(?:on|off)|switch|toggle|use|"
        # Arranging many things, and "do it" / "execute the script" (2026-09-29: "stack 10 parts" got no tools)
        r"stack|pile|line\s+up|lay\s+out|spread|repeat|loop|array|multiply|do|execute|apply|rebuild"
    )
    work = _LEAD_RE.sub("", work)
    # A question ("how do I open ...?", "in which file does it run?") asks for
    # an answer, not an action - unless it asks the assistant to do it.
    asks_assistant = bool(re.search(r"\b(?:can|could|would|will)\s+(?:you|u)\b", work))
    question = bool(_QUESTION_RE.match(work)) or (work.endswith("?") and not asks_assistant)
    explicit_action = not question and bool(re.match(
        rf"^(?:{_ASK})*(?:{_SCOPE})?(?:{_ASK})*(?:{action})\b", work, re.I))
    if not explicit_action and asks_assistant:
        explicit_action = bool(re.match(rf"^(?:{_SCOPE})?(?:{_ASK})+(?:{_SCOPE})?(?:{action})\b", work, re.I))
    # "New experience", "a new script for the door": a request to create one.
    if not explicit_action and not question:
        explicit_action = bool(re.match(r"^(?:(?:a|an|another)\s+)?new\s+[\w'\-]+", work))
    # "it says you have access to it, so do it pls": the request comes at the end.
    if not explicit_action and not question:
        explicit_action = bool(_DO_IT_RE.search(work))
    # "Show / hide the mini window" changes the app's own window (the mini_window tool) - not a read, not chat.
    if not explicit_action and not question:
        explicit_action = bool(re.match(r"^(?:please\s+)?(?:show|hide|bring\s+(?:up|back)|pop\s+out)\b.*\bmini[\s-]?window\b", work))
    # "Switch to dark mode", "make it light", "change the theme to Aurora": the app's own look (app_appearance).
    if not explicit_action and not question:
        explicit_action = bool(re.match(r"^(?:please\s+)?(?:switch|change|make|turn|set|put|go|use)\b.*\b(?:dark|light)(?:\s+(?:mode|theme))?\b", work)
                               or re.match(r"^(?:please\s+)?(?:switch|change|set)\b.*\b(?:theme|appearance)\b", work))
    # "Write a Python function that ...", "draft an email to my landlord": the answer IS the text, written in the
    # reply - nothing on the PC changes (1.4.6, stage 2: on the "write" verb alone the 4B model got run_python, ran the
    # function "to test it" and the user got an approval card for code they had only asked to see). Anything with
    # somewhere to put it - a file, a folder, a doc or note, Studio / Lua - stays an action.
    if explicit_action and _TEXT_ANSWER_RE.match(work) and not _HAS_TARGET_RE.search(work):
        return {"mode": "none", "allow_mutation": False, "reason": "asked to write something in the reply"}
    if explicit_action:
        return {"mode": "action", "allow_mutation": True, "reason": "explicit current-turn action"}

    machine_read = bool(re.match(
        r"^(?:please\s+)?(?:show|list|check|inspect|look at|find|read|get|review|analy[sz]e|summari[sz]e|what(?:'s| is) (?:using|on|in)|"
        r"search|look (?:it |this |that )?up|google|research|browse)\b", work))
    # Web lookups are reads too ("what does the Roblox docs page say about ...").
    web_question = bool(re.search(r"\b(?:on the (?:web|internet)|online|web search|search the web|look (?:it |this |that )?up|"
                                  r"with sources|cite (?:your )?sources|this (?:page|url|link|website|site)|https?://|"
                                  # current events can only be answered from the web
                                  r"latest|news|headlines?|breaking|this week|today'?s|right now|currently|"
                                  r"weather|forecast|stock price|exchange rate|release date|patch notes|changelog)\b", work))
    machine_question = bool(re.search(
        r"\b(?:open windows?|windows? (?:is|are) open|running (?:apps?|applications?|processes)|"
        r"(?:apps?|applications?|processes) (?:is|are) running|clipboard|files? on (?:my|the) (?:pc|computer)|"
        r"screen|active window|computer state|system usage|"
        # Questions about the user's own Roblox project are read-only lookups
        # through the Studio bridge ("why does my script error?").
        r"(?:my|this|the) (?:script|scripts|place|game|selection|selected \w+|output|explorer|model|part)|"
        # ... also when the place is named ("what do you see in Test1 roblox place?")
        r"roblox|studio|[\w'\-]+\s+place)\b", work))
    if machine_read or machine_question or web_question:
        return {"mode": "read", "allow_mutation": False, "reason": "read-only current-turn request"}
    return {"mode": "none", "allow_mutation": False,
            "reason": "no explicit computer action in the current message"}


def _multilingual_authority(value: str) -> dict | None:
    """Recognize explicit Russian/Hebrew requests without giving questions,
    negations or arbitrary non-English text permission to modify the machine.
    Tool execution still goes through the normal permission/approval layer.
    """
    action = (r"переименуй(?:те)?|скопируй(?:те)?|создай(?:те)?|сделай(?:те)?|открой(?:те)?|"
              r"закрой(?:те)?|исправь(?:те)?|добавь(?:те)?|перемести(?:те)?|удали(?:те)?|"
              r"установи(?:те)?|скачай(?:те)?|запусти(?:те)?|измени(?:те)?|организуй(?:те)?|"
              r"упорядочи(?:те)?|дублируй(?:те)?|продолжи(?:те)?|"
              r"שנה|תשנה|שני|שנו|העתק|תעתיק|העתיקי|צור|תיצור|צרי|פתח|תפתח|פתחי|"
              r"סגור|תסגור|סגרי|תקן|תתקן|תקני|הוסף|תוסיף|הוסיפי|העבר|תעביר|"
              r"מחק|תמחק|מחקי|התקן|תתקין|הורד|תוריד|הפעל|תפעיל|סדר|תסדר|סדרי|"
              r"ארגן|תארגן|שכפל|תשכפל|המשך|תמשיך")
    polite = (r"пожалуйста[, ]+|(?:можешь|можете|мог бы|могли бы)\s+(?:ты\s+|вы\s+)?|"
              r"בבקשה[, ]+|(?:אתה יכול|את יכולה|תוכל|תוכלי)\s+")
    # Infinitives only grant authority following an explicit 'can you' request.
    infinitive = (r"переименовать|скопировать|создать|открыть|закрыть|исправить|добавить|"
                  r"переместить|удалить|установить|скачать|запустить|изменить|организовать|"
                  r"לשנות|להעתיק|ליצור|לפתוח|לסגור|לתקן|להוסיף|להעביר|למחוק|להתקין|"
                  r"להוריד|להפעיל|לסדר|לארגן|לשכפל")
    is_action = bool(re.match(rf"^(?:{polite})*(?:{action})(?=\s|$|[!?.,])", value))
    is_action |= bool(re.match(rf"^(?:{polite})+(?:{infinitive})(?=\s|$|[!?.,])", value))
    if is_action:
        return {"mode": "action", "allow_mutation": True, "reason": "explicit Russian/Hebrew current-turn action"}
    if re.match(r"^(?:покажи|покажите|найди|найдите|проверь|проверьте|прочитай|прочитайте|"
                r"הצג|תציג|בדוק|תבדוק|חפש|תחפש|קרא|תקרא)(?=\s|$|[!?.,])", value):
        return {"mode": "read", "allow_mutation": False, "reason": "Russian/Hebrew read-only request"}
    return None
