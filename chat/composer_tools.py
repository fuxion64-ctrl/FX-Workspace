"""
composer_tools.py
=================
The composer's Tools (index.html TOOLS) on the server side: what each one
adds to a message's turn. The page sends the ids that are on with the message
(ChatRequest.tools); agent/tasks._run_turn keeps them in app_state["turn_tools"].

  grounded   web search is offered to questions too, and the model must search
             and cite before answering (tools/schemas.web_offered, HEADERS)
  research   agent/research.py instead of the normal turn
  summarize  read the URL(s) with fetch_page and summarize with citations
  code       run_python is offered to questions too; show code and real output
  canvas     one self-contained html/svg block; the page opens it beside the chat
  files      attached files (attach / file_context below) - on whenever a file is attached
  critique   after the answer, a self-review and a revised answer (critique())
  compare    the same question to a second model, side by side (compare())
  biascheck  Bias check: search both sides of every claim, rate each source's trust,
             the raw findings in full and a verdict (BIAS_RULES; with Research, its
             report follows the same rules - agent/research.py)

Attachments: POST /api/files stores a file (base64 JSON, at most MAX_FILE_BYTES)
under data/uploads/<id>/; the message carries the ids. file_context() turns
them into text for the model - a CSV's columns and statistics, a PDF's text
(the passages about the question when it is long), a text file - framed as
untrusted data, and images as image parts for a vision model.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import statistics
import time
import uuid

import pdf_text
import untrusted
import web_tools

IDS = ("grounded", "research", "invest", "market", "biascheck", "timeline", "summarize", "code", "canvas", "files",
       "critique", "compare", "world", "image", "charts", "visuals")
# "world" is never a chip: MossadGPT's turns carry it (agent/tasks._begin_composer_tools) - the world monitor.

# Bias check, as the user wrote it (a composer tool; Research follows it too).
BIAS_RULES = (
    "BIAS CHECK (the user turned it on): analyze the request using web search - do not rely on what you remember from "
    "training; base the analysis on sources you find now. Aim for the most unbiased, true analysis: for every source "
    "that supports one side, find another source that disputes it, so both sides are weighed equally. Present the raw "
    "findings in full - do not leave out important details; being short or convenient is not a priority. For every "
    "source you list, state how much you trust it and why, e.g. a social-media post: low trust; a news site: higher "
    "trust; a news site with a record of tampering with data, lying or spreading misinformation: lower trust. When "
    "needed, research the source itself to decide. If a source reports something out of the ordinary, extreme or "
    "unusual, double-check it with other sources; with no confirmation, lower its trust. If a conflict looks one-sided, "
    "dig for deeper clues and look beyond mainstream outlets - include independent journalists and vary through many "
    "different sources. Cite every claim as [n]. End with a final verdict or summary that says how confident it is.")

# Added to the turn header (agent/models.build_messages) while the tool is on.
HEADERS = {
    "grounded": ("GROUNDED ANSWER (the user turned it on for this message): search the web and read the best sources "
                 "before answering, even for things you think you know, and cite every factual claim as [n]. If the "
                 "sources don't settle it, say so."),
    "summarize": ("SUMMARIZE (the user turned it on): read the page(s) the request links to with fetch_page - or use the "
                  "text given - and reply with a one-line gist, then the key points as bullets citing [n], then anything "
                  "notable (dates, numbers, caveats, who wrote it). Say plainly if a page could not be read."),
    "code": ("CODE RUNNER (the user turned it on): when code would answer or check the question, write it and run it "
             "with run_python, then show the code and its real output - never an output you did not see. Keep it "
             "harmless: no deleting, installing or network access unless the user asked for it."),
    "canvas": ("CANVAS (the user turned it on): put the visual or document result in ONE ```html block (or ```svg) - a "
               "complete, self-contained page with inline CSS and JS and no external files or network requests. It opens "
               "in the canvas beside the chat. Around it, say briefly what you made."),
    "biascheck": BIAS_RULES,
    # The timeline itself is built by the app from the sources (corroboration.timelines, agent/tasks).
    "timeline": ("CLAIM TIMELINES (the user turned it on): search and read several independent sources, cite every "
                 "claim as [n], and keep the dates the sources give - the app adds a timeline of who reported each "
                 "claim first and who confirmed it later."),
    "market": ("MARKET DATA (the user turned it on): for prices, returns, company financials, filings, the economy or "
               "exchange rates, call market_data first - it reads official and market sources (SEC filings, the US "
               "Treasury, BLS, Yahoo Finance) and labels each figure FACT / MARKET / COMPUTED. Use web_search for news "
               "and views, and keep the labels: say what is a reported fact, what is a forecast (ESTIMATE) and what is "
               "a view (OPINION, SPECULATION). Cite figures as [n] with their dates. Don't tell the user to buy, sell or "
               "hold - describe the numbers, the evidence and the trade-offs."),
    "image": ("IMAGES (made on this PC): to make the user a picture, call generate_image - one call per image asked "
              "for. To show what something REAL looks like (a product, a place, a person, its colours), call "
              "show_web_images instead - it shows real pictures from the web. For a new picture: "
              "turn the request into a vivid English prompt (subject, setting, style, light, composition; text "
              "to appear in quotes); keep what the user specified. Pick quality: 'fast' for quick drafts or to edit "
              "an attached image (edit true), 'balanced' by default, 'best' when they ask for the highest quality. "
              "The app shows the image under your reply: don't describe it as if you saw it and don't paste links."),
    "world": ("WORLD MONITOR (MossadGPT): for anything happening in the world now - aircraft, military flights, "
              "earthquakes, conflict and protest events, space weather, satellites and their passes - call world_monitor "
              "first (near / events / track / passes; 'show' turns the globe). Report positions and events as what the "
              "sources broadcast or reported, with their times, and cite [n]. GDELT events are machine-coded from single "
              "articles: say 'reported' until other sources confirm. Say what the feeds can't see (aircraft with "
              "transponders off, ships are not covered yet). Use web_search for context and confirmation."),
    "files": ("ATTACHED FILES: the user attached the files shown under [ATTACHED FILES]. Answer from them. For "
              "statistics or a chart of a table, call compute_stats or make_chart with file set to the file's name when "
              "they are offered; otherwise write a ```chart block with JSON: {\"type\": \"bar\" | \"line\" | \"pie\", "
              "\"title\": str, \"labels\": [..], \"series\": [{\"name\": str, \"values\": [numbers]}]} - the chat draws it."),
    # stats_tools.py - turned on by the chip, or by itself for numbers to compute or plot (auto_tools.py).
    # Interactive replies (D50): blocks the chat draws itself - no tool call, nothing runs on the PC.
    # (1.4.8, item 6) The same four blocks said shorter; the JSON keys and the formula syntax are unchanged.
    "visuals": ("VISUALS (on because a picture may help): when one does, add ONE block and still explain in words; skip "
                "it for short answers.\n"
                "```chart {\"type\":\"bar|line|area|pie|scatter\",\"title\":\"..\",\"labels\":[..],\"series\":[{\"name\":\"..\","
                "\"values\":[..]}],\"unit\":\"..\",\"source\":\"..\"} - only numbers you have, never invented.\n"
                "```chart {\"type\":\"function\",\"title\":\"..\",\"functions\":[{\"name\":\"y\",\"expr\":\"a*x^2+b\"}],"
                "\"params\":{\"a\":[1,-5,5]},\"x\":[-10,10]} - a formula, a slider per parameter [value, min, max]; "
                "math: + - * / ^, sin cos tan sqrt abs log exp, pi, e.\n"
                "```widget - a self-contained HTML page (inline CSS/JS; CSS variables --bg --surface --text --muted --accent "
                "--border) for a diagram or explainer.\n"
                "```choices - one option per line, each becomes a button."),
    "charts": ("CHARTS AND STATISTICS: for any statistic - mean, median, spread, percentages, growth, correlation - call "
               "compute_stats and quote its numbers; never work them out in your head. To show numbers as a chart, call "
               "make_chart with the exact numbers you have (from sources, an attached file, or compute_stats): type bar, "
               "hbar, stacked, line, area, pie, donut or scatter; label the axes, give the unit and the source. Never "
               "invent, pad or round numbers you weren't given."),
}
# Tools offered to a question ("none" mode) while the composer tool is on.
OFFERS = {"grounded": {"web_search", "fetch_page", "web_read"}, "summarize": {"web_search", "fetch_page", "web_read"},
          "biascheck": {"web_search", "fetch_page", "web_read"}, "timeline": {"web_search", "fetch_page", "web_read"},
          "market": {"market_data", "web_search", "fetch_page"},
          "world": {"world_monitor", "web_search", "fetch_page"},
          "image": {"generate_image"},
          "charts": {"compute_stats", "make_chart"},
          "code": {"run_python"}}


def header(turn_tools) -> str:
    return "\n".join(HEADERS[t] for t in IDS if t in (turn_tools or ()) and t in HEADERS)


def offered(turn_tools) -> set:
    out = set()
    for t in turn_tools or ():
        out |= OFFERS.get(t, set())
    return out


# ==============================================================================
# Attachments
# ==============================================================================
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_FILES = 5
TEXT_EXT = {".txt", ".md", ".markdown", ".json", ".py", ".js", ".ts", ".html", ".htm", ".css", ".xml", ".yaml", ".yml",
            ".ini", ".cfg", ".toml", ".log", ".lua", ".luau", ".sql", ".sh", ".ps1", ".bat", ".c", ".cpp", ".h", ".cs",
            ".java", ".go", ".rs", ".rb", ".php", ".tex", ".rst"}
TABLE_EXT = {".csv", ".tsv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
CONTEXT_CHARS = 24_000                 # of one file's text given to the model


def kind_of(name: str) -> str:
    ext = os.path.splitext(name.lower())[1]
    return ("table" if ext in TABLE_EXT else "pdf" if ext == ".pdf" else "image" if ext in IMAGE_EXT
            else "text" if ext in TEXT_EXT else "")


class FileStore:
    """data/uploads/<id>/attachment.<ext> plus meta.json; reads legacy filenames too.
    Files older than KEEP_DAYS go."""
    KEEP_DAYS = 30

    def __init__(self, root: str):
        self.root = root

    def _dir(self, fid: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{12}", fid or ""):
            raise KeyError(fid)
        return os.path.join(self.root, fid)

    def save(self, name: str, data_b64: str) -> dict:
        name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", os.path.basename(str(name or "")).strip())[:120] or "file"
        kind = kind_of(name)
        if not kind:
            raise ValueError(f"{name}: this file type can't be read here (CSV, PDF, images and text files can).")
        try:
            data = base64.b64decode(data_b64 or "", validate=True)
        except (ValueError, TypeError):
            raise ValueError("The file arrived damaged - try attaching it again.")
        if not data:
            raise ValueError(f"{name} is empty.")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"{name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")
        fid = uuid.uuid4().hex[:12]
        folder = self._dir(fid)
        os.makedirs(folder, exist_ok=True)
        # Keep the payload separate from our metadata, including on Windows
        # where META.JSON and meta.json name the same file.
        stored_name = "attachment" + os.path.splitext(name)[1].lower()
        with open(os.path.join(folder, stored_name), "wb") as f:
            f.write(data)
        meta = {"id": fid, "name": name, "stored_name": stored_name, "kind": kind, "size": len(data), "created": time.time()}
        meta.update(self._describe(os.path.join(folder, stored_name), kind))
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self.prune()
        return meta

    def _describe(self, path: str, kind: str) -> dict:
        """A line for the chip under the message: rows x columns, pages, pixels."""
        try:
            if kind == "table":
                t = read_table(path)
                return {"summary": f"{len(t['rows']):,} rows × {len(t['columns'])} columns"}
            if kind == "pdf":
                with open(path, "rb") as f:
                    got = pdf_text.extract(f.read(), max_chars=2000)
                return {"summary": f"{got['pages']} page{'s' if got['pages'] != 1 else ''}"}
            if kind == "image":
                from PIL import Image
                with Image.open(path) as im:
                    return {"summary": f"{im.width}×{im.height}"}
        except Exception as e:
            return {"summary": "", "problem": str(e)[:160]}
        return {"summary": ""}

    def get(self, fid: str) -> dict | None:
        try:
            with open(os.path.join(self._dir(fid), "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except (KeyError, OSError, ValueError):
            return None
        meta["path"] = os.path.join(self._dir(fid), meta.get("stored_name") or meta["name"])
        return meta

    def adopt(self, other: "FileStore") -> dict:
        """Move every file of another store (a temporary chat's) into this one,
        each under a new id. Returns {old id: new id}."""
        import shutil
        moved = {}
        try:
            names = os.listdir(other.root)
        except OSError:
            return moved
        for fid in names:
            meta = other.get(fid)
            if not meta:
                continue
            new = uuid.uuid4().hex[:12]
            os.makedirs(self.root, exist_ok=True)
            shutil.move(other._dir(fid), self._dir(new))
            meta = {k: v for k, v in meta.items() if k != "path"}
            meta.update(id=new, created=time.time())
            with open(os.path.join(self._dir(new), "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            moved[fid] = new
        return moved

    def prune(self):
        cutoff = time.time() - self.KEEP_DAYS * 86400
        try:
            for fid in os.listdir(self.root):
                meta = self.get(fid)
                if meta and meta.get("created", 0) < cutoff:
                    import shutil
                    shutil.rmtree(self._dir(fid), ignore_errors=True)
        except OSError:
            pass


def read_table(path: str, limit: int = 200_000) -> dict:
    with open(path, "rb") as f:
        raw = f.read(20 * 1024 * 1024)
    text = raw.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if path.lower().endswith(".tsv") else csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return {"columns": [], "rows": []}
    head = [c.strip() or f"column {i + 1}" for i, c in enumerate(rows[0])]
    return {"columns": head, "rows": rows[1:limit + 1]}


def _num(v: str):
    v = str(v).strip().replace(",", "")
    v = re.sub(r"^[$€£₪]|%$", "", v)
    try:
        return float(v)
    except ValueError:
        return None


def table_summary(path: str) -> str:
    """Columns, their types and statistics, and the first rows - what a model
    needs to answer questions about a CSV it can't hold whole."""
    t = read_table(path)
    cols, rows = t["columns"], t["rows"]
    lines = [f"{len(rows):,} rows, {len(cols)} columns."]
    for i, name in enumerate(cols[:40]):
        values = [r[i] for r in rows if i < len(r) and r[i].strip()]
        nums = [n for n in (_num(v) for v in values) if n is not None]
        if values and len(nums) >= 0.8 * len(values):
            lines.append(f"- {name}: numeric, {len(nums):,} values, min {min(nums):g}, max {max(nums):g}, "
                         f"mean {statistics.fmean(nums):.4g}, median {statistics.median(nums):g}, sum {sum(nums):.6g}")
        else:
            distinct = {}
            for v in values:
                distinct[v] = distinct.get(v, 0) + 1
            top = sorted(distinct.items(), key=lambda kv: -kv[1])[:6]
            lines.append(f"- {name}: text, {len(values):,} values, {len(distinct):,} distinct"
                         + (f"; most common: {', '.join(f'{k[:40]} ({n})' for k, n in top)}" if top else ""))
    shown = rows[:30]
    lines.append("\nFirst rows:\n| " + " | ".join(cols[:12]) + " |\n|" + "---|" * min(12, len(cols)))
    lines += ["| " + " | ".join((r[i] if i < len(r) else "")[:40] for i in range(min(12, len(cols)))) + " |" for r in shown]
    if len(rows) <= 300 and len(cols) <= 12:            # small enough to give whole
        lines.append("\nAll rows (CSV):\n" + "\n".join(",".join(r[:12]) for r in [cols] + rows))
    return "\n".join(lines)


def file_context(store: FileStore, ids: list, question: str) -> tuple[str, list, list]:
    """(text for the model, image data URLs, chips for the transcript)."""
    blocks, images, chips = [], [], []
    for fid in list(dict.fromkeys(ids or []))[:MAX_FILES]:
        meta = store.get(fid)
        if not meta:
            continue
        chips.append({k: meta.get(k) for k in ("id", "name", "kind", "size", "summary")})
        try:
            if meta["kind"] == "table":
                body = table_summary(meta["path"])
            elif meta["kind"] == "pdf":
                with open(meta["path"], "rb") as f:
                    got = pdf_text.extract(f.read())
                body = web_tools.select_passages(got["text"], question, CONTEXT_CHARS)["text"] or "(no text - a scanned PDF needs OCR)"
                body = f"{got['pages']} pages.\n{body}"
            elif meta["kind"] == "image":
                with open(meta["path"], "rb") as f:
                    data = f.read()
                ext = os.path.splitext(meta["name"].lower())[1].lstrip(".").replace("jpg", "jpeg")
                images.append(f"data:image/{ext};base64,{base64.b64encode(data).decode()}")
                body = f"An image ({meta.get('summary') or 'size unknown'}) - attached below for a model that can see images."
            else:
                with open(meta["path"], "rb") as f:
                    body = f.read(CONTEXT_CHARS * 2).decode("utf-8", errors="replace")
                body = web_tools.select_passages(body, question, CONTEXT_CHARS)["text"]
        except Exception as e:
            body = f"(could not be read: {str(e)[:200]})"
        found = untrusted.find_instructions(body)
        blocks.append(untrusted.frame(f"the attached file {meta['name']}", f"File: {meta['name']}\n{body}", found))
    text = ("[ATTACHED FILES]\n" + "\n\n".join(blocks)) if blocks else ""
    return text, images, chips


# ==============================================================================
# Critique and Compare - after the answer (agent/tasks._run_turn)
# ==============================================================================
CRITIQUE_SYSTEM = (
    "You review an assistant's answer to a user's request, then improve it. First list the problems you find - "
    "wrong or unsupported claims, missing parts of the request, unclear or bloated wording, errors in code - as short "
    "bullets (or 'No significant problems.'). Then write the complete revised answer in Markdown, keeping every [n] "
    "citation that is still used and adding none. Format exactly:\nREVIEW:\n- ...\nREVISED:\n<the whole revised answer>")


def critique(model_text, request: str, answer: str) -> dict | None:
    """{"notes", "draft", "revised", "changed"} or None when the model gave nothing usable."""
    out = model_text(CRITIQUE_SYSTEM, f"Request:\n{request}\n\nAnswer to review:\n{answer}", max_tokens=3000,
                     label="Reviewing the answer", temperature=0.2)
    text = out["text"]
    m = re.search(r"^\s*REVISED:\s*$", text, re.M) or re.search(r"REVISED:", text)
    if not m:
        return None
    notes = re.sub(r"^\s*REVIEW:\s*", "", text[:m.start()]).strip()
    revised = text[m.end():].strip()
    if len(revised) < min(40, len(answer) // 3):
        return None
    return {"notes": notes[:3000], "draft": answer, "revised": revised,
            "changed": re.sub(r"\s+", " ", revised) != re.sub(r"\s+", " ", answer)}


COMPARE_SYSTEM = ("You are a helpful assistant. Answer the user's request directly and accurately. Replies are rendered "
                  "as Markdown.")


def compare(model_text, request: str, model: str) -> dict:
    """The second answer: {"model", "text", "seconds", "tokens_out"} or {"model", "error"}."""
    try:
        out = model_text(COMPARE_SYSTEM, request, max_tokens=2048, model=model, label=f"Answering with {model}")
        return {"model": out["model"], "text": out["text"], "seconds": out["seconds"], "tokens_out": out["tokens_out"]}
    except Exception as e:
        return {"model": model, "error": str(e)[:300]}
