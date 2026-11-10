"""
web_images.py - pictures from the web, shown in the chat (1.4.7).

"Show some pictures of the Nothing Phone 4a and its colours": the assistant searched and read four pages, but had no
way to put a picture in the chat - it wrote that the pages were "text-heavy" and stopped. The show_web_images tool
takes pages (or a search) and shows their main pictures:

  * from each page: its social-card picture (og:image / twitter:image) first, then the large pictures in its body
    whose alt text matches the request - logos, icons, avatars, tracking pixels and tiny images are left out;
  * each picture is downloaded through the network layer like every page (web_tools.fetch: its route, Tor, the kill
    switch, public addresses only) - the page in the app never loads anything from the web itself;
  * saved in data/web_images/ and served from /api/web-images/<name>, then shown in the chat at once (the same
    "image" event a made picture uses) and kept under the reply with the page it came from.
"""
from __future__ import annotations

import hashlib
import html
import io
import os
import re
import time
from urllib.parse import urljoin, urlparse

import web_tools

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(APP_DIR, "data", "web_images")
MAX_BYTES = 8 * 1024 * 1024
MIN_SIDE = 200                         # smaller is an icon, a thumbnail strip or a pixel
MAX_SHOWN = 8
NAME_RE = re.compile(r"[0-9a-f]{20}\.(jpg|png|webp|gif)")
HEADERS = {"User-Agent": "FXWorkspace/1.4.7 (a local assistant showing a page's pictures to its user)",
           "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8"}

_META = re.compile(r"<meta\b[^>]*>", re.I)
_IMG = re.compile(r"<img\b[^>]*>", re.I)
_ATTR = re.compile(r"""([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")
_SKIP = re.compile(r"logo|icon|sprite|avatar|favicon|badge|banner-ad|pixel|spacer|blank|placeholder|emoji|gravatar|"
                   r"tracking|1x1|\.svg(\?|$)|data:", re.I)


def _attrs(tag: str) -> dict:
    out = {}
    for m in _ATTR.finditer(tag):
        out[m.group(1).lower()] = html.unescape(m.group(2) if m.group(2) is not None else
                                                m.group(3) if m.group(3) is not None else m.group(4) or "")
    return out


def _best_src(a: dict) -> str:
    """The largest candidate of srcset / data-srcset, else src / data-src (lazy-loaded pages)."""
    for key in ("srcset", "data-srcset"):
        best, width = "", -1
        for part in (a.get(key) or "").split(","):
            bits = part.strip().split()
            if not bits:
                continue
            w = int(bits[1][:-1]) if len(bits) > 1 and bits[1].endswith("w") and bits[1][:-1].isdigit() else 0
            if w > width:
                best, width = bits[0], w
        if best:
            return best
    return a.get("data-src") or a.get("data-lazy-src") or a.get("data-original") or a.get("src") or ""


def _num(v) -> int:
    m = re.match(r"\s*(\d+)", str(v or ""))
    return int(m.group(1)) if m else 0


def candidates(page_html: str, base: str, query: str = "") -> list[dict]:
    """[{"src", "alt", "score"}] - a page's pictures worth showing, best first."""
    words = {w for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
    out, seen = [], set()

    def add(src: str, alt: str, score: float):
        if not src:
            return
        src = urljoin(base, src.strip())
        if not src.startswith(("http://", "https://")) or _SKIP.search(src):
            return
        key = re.sub(r"[?#].*$", "", src)
        if key in seen:
            return
        seen.add(key)
        out.append({"src": src, "alt": re.sub(r"\s+", " ", alt or "").strip()[:160], "score": score})

    for tag in _META.findall(page_html):
        a = _attrs(tag)
        prop = (a.get("property") or a.get("name") or "").lower()
        if prop in ("og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src"):
            add(a.get("content") or "", "", 2.0)
    title = re.search(r"<title[^>]*>(.*?)</title>", page_html, re.I | re.S)
    if out and title:
        for c in out:
            c["alt"] = html.unescape(re.sub(r"\s+", " ", title.group(1))).strip()[:160]
    for tag in _IMG.findall(page_html):
        a = _attrs(tag)
        w, h = _num(a.get("width")), _num(a.get("height"))
        if (w and w < MIN_SIDE) or (h and h < MIN_SIDE):
            continue
        alt = a.get("alt") or a.get("title") or ""
        if _SKIP.search(alt) or _SKIP.search(" ".join(a.get("class", "").split())):
            continue
        hits = len(words & set(re.findall(r"[a-z0-9]{3,}", alt.lower()))) if words else 0
        size = 0.5 if (w >= 600 or h >= 600) else 0
        add(_best_src(a), alt, 1.0 + hits * 0.6 + size)
    out.sort(key=lambda c: -c["score"])
    return out[:20]


def _ext(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def save(src: str, page: str = "", alt: str = "") -> dict | None:
    """Download one picture (through the network layer) into data/web_images; None when it isn't a picture worth
    showing (not an image, too small, too big, or the site refused)."""
    try:
        resp, body, final = web_tools.fetch(src, headers={**HEADERS, **({"Referer": page} if page else {})},
                                            component="search")
    except Exception:
        return None
    if getattr(resp, "status_code", 200) >= 400 or len(body) < 2000 or len(body) > MAX_BYTES:
        return None
    ext = _ext(body)
    if not ext:
        return None
    w = h = 0
    try:
        from PIL import Image
        with Image.open(io.BytesIO(body)) as im:
            w, h = im.size
    except Exception:
        return None
    if min(w, h) < MIN_SIDE:
        return None
    name = hashlib.sha1(final.encode("utf-8", "replace") + body[:4096]).hexdigest()[:20] + "." + ext
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, path)
    return {"url": f"/api/web-images/{name}", "width": w, "height": h, "alt": alt, "source": page or final,
            "domain": urlparse(page or final).netloc.removeprefix("www."), "web": True}


def link_safe(url: str) -> str:
    """An address safe inside a Markdown link: brackets and spaces percent-encoded."""
    return str(url or "").replace("(", "%28").replace(")", "%29").replace(" ", "%20")


def path_of(name: str) -> str | None:
    """The cached picture's file for /api/web-images/<name> - only names this module writes, never a path."""
    if not NAME_RE.fullmatch(name or ""):
        return None
    p = os.path.join(DIR, name)
    return p if os.path.isfile(p) else None


def show_web_images_tool(args: dict, settings: dict | None = None, should_cancel=None, progress=None) -> dict:
    """The model's show_web_images: pictures from the pages given (or found by a search for `query`), shown in the chat."""
    import search_engine
    settings = settings or {}
    should_cancel = should_cancel or (lambda: False)
    progress = progress or (lambda msg: None)
    query = str(args.get("query") or "").strip()
    count = max(1, min(int(args.get("count") or 4), MAX_SHOWN))
    pages = [web_tools._normalize_url(str(u)) for u in ([args.get("url")] if args.get("url") else []) + list(args.get("urls") or [])]
    pages = [u for u in pages if u][:6]
    mode = search_engine.turn_mode(settings)
    if mode in ("disabled", "local"):
        return {"ok": False, "status": "permanent_failure",
                "message": "Web pictures need web search, which is off or set to this PC only (Settings › Web and privacy)."}
    if not pages:
        if not query:
            return {"ok": False, "message": "Give the pages to take pictures from (urls), or what to look for (query)."}
        progress(f"Looking for pictures of {query}")
        found = search_engine.search(query, num_results=6, settings=settings, should_cancel=should_cancel, mode=mode)
        if found.get("blocked"):
            return {"ok": False, "status": "permanent_failure", "message": str(found.get("why") or "The search was blocked.")}
        pages = [r.get("url") for r in (found.get("results") or []) if r.get("url")][:5]
    if not pages:
        return {"ok": False, "message": f"No pages found for {query!r}."}
    shown, tried, started = [], 0, time.time()
    per_page = max(1, -(-count // max(1, len(pages))))          # spread over the pages: different shots, not one gallery
    for page in pages:
        if should_cancel() or len(shown) >= count or time.time() - started > 90:
            break
        progress(f"Reading {urlparse(page).netloc}")
        try:
            resp, body, final = web_tools.fetch(page)
        except Exception:
            continue
        ctype = (resp.headers.get("content-type") or "").lower() if hasattr(resp, "headers") else ""
        if "html" not in ctype and not body.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
            continue
        text = body.decode("utf-8", "replace")
        took = 0
        for c in candidates(text, final, query):
            if should_cancel() or len(shown) >= count or took >= per_page + (1 if page == pages[-1] else 0):
                break
            tried += 1
            img = save(c["src"], final, c["alt"])
            if img and not any(s["url"] == img["url"] for s in shown):
                shown.append(img)
                took += 1
                progress(f"{len(shown)} of {count} pictures")
    if not shown:
        return {"ok": False, "message": f"None of those pages had a picture that could be shown ({tried} tried) - "
                                        "describe them in words instead, or try other pages."}
    # The pages are this turn's sources: the reply's "Pictures from" links are then known, not flagged as invented
    # (careful.py). A ")" in an address would end a Markdown link early, so it is linked as %29 - that form too.
    try:
        import sources
        reg = sources.current()
        for s in shown:
            reg.add(s["source"])
            if "(" in s["source"] or ")" in s["source"]:
                reg.add(link_safe(s["source"]))
    except Exception:
        pass
    where = ", ".join(dict.fromkeys(s["domain"] for s in shown))
    return {"ok": True, "images": shown,
            "message": (f"Done - {len(shown)} picture{'s are' if len(shown) != 1 else ' is'} ALREADY SHOWN to the user in "
                        f"the chat (from {where}). Do NOT call show_web_images again for them, and don't search for the "
                        "files. Say in a sentence or two what they show, naming the sites they came from.")}
