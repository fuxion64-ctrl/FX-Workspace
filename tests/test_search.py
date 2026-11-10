"""
test_search.py
==============
Web search and reading (search_engine.py, sources.py, pdf_text.py and the
web_tools additions they use): numbering and citing sources, ranking and
merging mirrors, the provider chain and its fallbacks, the optional keys,
fetch_page, when questions get the web tools, and what a reply saves.
No network: providers and page loads are faked.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import zlib
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import main  # noqa: F401  (wires the app together)
import pdf_text
import search_engine as se
import sources
import web_tools

import agent.sessions as sessions
import core.state as shared
import tools.schemas as schemas


# ==============================================================================
# sources.py
# ==============================================================================
class SourcesTests(unittest.TestCase):
    def test_canonical_merges_mirrors_and_drops_tracking(self):
        a = sources.canonical("https://www.example.com/news/story/?utm_source=x&id=7&fbclid=abc#top")
        self.assertEqual(a, sources.canonical("http://m.example.com/news/story?id=7"))
        self.assertEqual(a, "example.com/news/story?id=7")
        self.assertEqual(sources.canonical("https://example.com/a/amp"), sources.canonical("https://example.com/a"))
        self.assertNotEqual(sources.canonical("https://example.com/a?page=2"), sources.canonical("https://example.com/a"))

    def test_registry_numbers_are_stable_and_updated(self):
        reg = sources.Registry()
        self.assertEqual(reg.add("https://a.org/x", "A"), 1)
        self.assertEqual(reg.add("https://b.org/y", "B"), 2)
        self.assertEqual(reg.add("https://www.a.org/x/?utm_medium=feed", "", "snippet", "2026-01-02", read=True), 1)
        item = reg.get(1)
        self.assertEqual((item["title"], item["snippet"], item["date"], item["read"]), ("A", "snippet", "2026-01-02", True))
        reg.alias("https://short.link/q", 2)
        self.assertEqual(reg.add("https://short.link/q"), 2)
        self.assertEqual(reg.find("https://b.org/y"), 2)

    def test_reply_sources_are_the_cited_ones_else_the_pages_read(self):
        reg = sources.Registry()
        for i in range(4):
            reg.add(f"https://s{i}.org/", f"S{i}", read=(i == 3))
        cited = reg.for_reply("It shipped in May [2] and was delayed [4]. See [docs](https://x.org) [9].")
        self.assertEqual([s["n"] for s in cited], [2, 4])
        self.assertTrue(all(s["cited"] for s in cited))
        self.assertEqual([s["n"] for s in reg.for_reply("No citations here.")], [4])
        self.assertEqual(sources.cited_numbers("[1] [2](http://x) [30]"), {1, 30})

    def test_registry_is_capped(self):
        reg = sources.Registry()
        for i in range(sources.MAX_SOURCES):
            reg.add(f"https://s{i}.org/")
        self.assertEqual(reg.add("https://one-too-many.org/"), 0)


# ==============================================================================
# Query analysis and ranking
# ==============================================================================
class QueryTests(unittest.TestCase):
    def test_recency_is_automatic_for_news(self):
        self.assertEqual(se.auto_recency("latest news on the rover landing"), "week")
        self.assertEqual(se.auto_recency("election results today"), "day")
        self.assertEqual(se.auto_recency("how does TCP slow start work"), "any")
        self.assertEqual(se.auto_recency("latest python version"), "any")       # freshness ranks it instead
        self.assertTrue(se.time_sensitive("latest python version"))
        self.assertFalse(se.time_sensitive("what is a monad"))

    def test_site_and_region_are_cleaned(self):
        self.assertEqual(se.clean_site("https://Docs.Python.org/3/library"), "docs.python.org")
        self.assertEqual(se.clean_site("site:roblox.com"), "roblox.com")
        self.assertEqual(se.clean_site("not a site"), "")
        self.assertEqual(se.clean_region("US"), "us")
        self.assertEqual(se.clean_region("usa!"), "")

    def test_dates(self):
        today = date(2026, 9, 26)
        self.assertEqual(se.parse_date("2026-08-30T10:00:00Z"), date(2026, 8, 30))
        self.assertEqual(se.parse_date("Aug 30, 2026 - the release"), date(2026, 8, 30))
        self.assertEqual(se.parse_date("30 August 2026"), date(2026, 8, 30))
        self.assertEqual(se.parse_date("3 days ago", today), date(2026, 9, 23))
        self.assertIsNone(se.parse_date("no date here"))
        self.assertIsNone(se.parse_date("2026-13-45"))


def _r(url, title="t", snippet="", d=""):
    return {"url": url, "title": title, "snippet": snippet, "date": d}


class RankTests(unittest.TestCase):
    def test_mirrors_merge_and_count_as_repeated_hits(self):
        ranked = se.rank([_r("https://www.site.org/a?utm_source=x", "A"), _r("https://other.org/b", "B"),
                          _r("https://m.site.org/a", "A", "a longer snippet than before")], ["query"], False)
        self.assertEqual(len(ranked), 2)
        first = next(r for r in ranked if r["domain"] == "site.org")
        self.assertEqual(first["hits"], 2)
        self.assertEqual(first["snippet"], "a longer snippet than before")

    def test_same_title_on_one_site_is_one_result(self):
        ranked = se.rank([_r("https://site.org/a", "Guide to X"), _r("https://site.org/print/a", "Guide to X")],
                         ["x"], False)
        self.assertEqual(len(ranked), 1)

    def test_authoritative_sources_rise_and_content_farms_sink(self):
        ranked = se.rank([_r("https://www.pinterest.com/pin/1", "Python tips"),
                          _r("https://www.quora.com/What-is", "What is"),
                          _r("https://someblog.net/post", "Python 3.14 is out"),
                          _r("https://docs.python.org/3/whatsnew/3.14.html", "What's new")], ["python 3.14"], False)
        self.assertEqual(ranked[0]["domain"], "docs.python.org")
        self.assertEqual(ranked[-1]["domain"], "pinterest.com")
        self.assertTrue(se.low_quality(ranked[-1]))

    def test_the_subject_site_is_a_primary_source(self):
        ranked = se.rank([_r("https://randomnews.com/roblox-update", "Roblox update"),
                          _r("https://create.roblox.com/docs/reference", "Reference")], ["roblox studio api"], False)
        self.assertEqual(ranked[0]["domain"], "create.roblox.com")

    def test_fresh_pages_win_on_time_sensitive_questions(self):
        today = date(2026, 9, 26)
        results = [_r("https://a.org/old", "Old", d="2024-01-01"), _r("https://b.org/new", "New", d="2026-09-20")]
        self.assertEqual(se.rank(results, ["latest release"], True, today=today)[0]["domain"], "b.org")
        self.assertEqual(se.rank(results, ["latest release"], False, today=today)[0]["domain"], "a.org")

    def test_other_sites_sink_when_a_site_is_named(self):
        ranked = se.rank([_r("https://elsewhere.com/x"), _r("https://docs.python.org/x")], ["x"], False,
                         site="python.org")
        self.assertEqual(ranked[0]["domain"], "docs.python.org")

    def test_non_web_urls_are_dropped(self):
        self.assertEqual(se.rank([_r("javascript:alert(1)"), _r("file:///c:/x")], ["q"], False), [])


# ==============================================================================
# Keys and the provider chain
# ==============================================================================
class KeyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.keys = se.KeyStore(os.path.join(self.dir.name, "data", "search_keys.json"))
        self.env = patch.dict(os.environ, {"BRAVE_API_KEY": "", "TAVILY_API_KEY": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.dir.cleanup()

    def test_status_never_contains_the_key(self):
        self.keys.set("brave", "BSA-secret-key-1234")
        status = self.keys.status()
        self.assertEqual(status["brave"], {"set": True, "source": "settings", "hint": "…1234"})
        self.assertFalse(status["tavily"]["set"])
        self.assertNotIn("secret", repr(status))
        self.assertEqual(self.keys.get("brave"), "BSA-secret-key-1234")
        self.keys.set("brave", "")
        self.assertFalse(self.keys.status()["brave"]["set"])

    def test_bad_keys_and_providers_are_refused(self):
        with self.assertRaises(ValueError):
            self.keys.set("brave", "has spaces in it")
        with self.assertRaises(ValueError):
            self.keys.set("google", "abcdefgh1234")

    def test_environment_keys_count(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-from-env-9999"}):
            self.assertEqual(self.keys.status()["tavily"]["source"], "environment")
            self.assertEqual(se.chain({}, self.keys), ["tavily", "duckduckgo", "ddg_instant"])

    def test_chain_order(self):
        self.assertEqual(se.chain({}, self.keys), ["duckduckgo", "ddg_instant"])
        self.keys.set("brave", "brave-key-0001")
        self.keys.set("tavily", "tavily-key-0002")
        self.assertEqual(se.chain({"web_search_provider": "tavily"}, self.keys),
                         ["tavily", "brave", "duckduckgo", "ddg_instant"])
        self.assertEqual(se.chain({"web_search_provider": "searxng", "searxng_url": "http://127.0.0.1:8888"}, self.keys),
                         ["searxng", "brave", "tavily", "duckduckgo", "ddg_instant"])

    def test_keys_are_masked_in_messages(self):
        self.assertEqual(se.mask("401 for key abc123secret", ["abc123secret"]), "401 for key [key hidden]")

    def test_error_classes(self):
        self.assertEqual(se._classify_error(SimpleNamespace(status_code=429)), "rate")
        err = type("AuthenticationError", (Exception,), {})("bad")
        self.assertEqual(se._classify_error(err), "refused")
        self.assertEqual(se._classify_error(TimeoutError("read timed out")), "transient")


class SearchTests(unittest.TestCase):
    def setUp(self):
        se.reset()
        sources.start_turn()
        self.calls = []

    def fake(self, answers):
        """_run_provider replaced: answers[provider] -> (results, why)."""
        def run(provider, query, n, recency, site, region, settings, keys, browser_fallback, should_cancel):
            self.calls.append((provider, query, recency))
            return answers.get(provider, ([], f"{provider} down"))
        return patch.object(se, "_run_provider", side_effect=run)

    def test_falls_back_along_the_chain(self):
        with self.fake({"duckduckgo": ([], "DuckDuckGo refused"),
                        "ddg_instant": ([_r("https://en.wikipedia.org/wiki/X", "X")], "")}):
            got = se.search("what is x", settings={}, keys=se.KeyStore(os.devnull + ".missing"))
        self.assertEqual([c[0] for c in self.calls], ["duckduckgo", "ddg_instant"])
        self.assertEqual(got["providers"], ["DuckDuckGo Instant Answers"])
        self.assertEqual(got["why"], ["DuckDuckGo refused"])

    def test_several_queries_are_merged_and_cached(self):
        answers = {"duckduckgo": ([_r("https://a.org/1", "A"), _r("https://b.org/2", "B")], "")}
        with self.fake(answers):
            got = se.search("q one", queries=["q two", "Q ONE", "q three", "q four", "q five"], settings={})
            self.assertEqual(got["queries"], ["q one", "q two", "q three", "q four"])     # deduplicated, at most 4
            self.assertEqual(len(got["results"]), 2)                                     # mirrors merged
            self.assertEqual(got["results"][0]["hits"], 4)
            again = se.search("q one", settings={})
        self.assertEqual(again["cached"], 1)
        self.assertEqual(len(self.calls), 4)

    def test_recency_is_passed_through_and_chosen_for_news(self):
        with self.fake({"duckduckgo": ([_r("https://a.org/")], "")}):
            se.search("breaking news storm", settings={})
            se.search("tcp handshake", recency="month", settings={})
        self.assertEqual([c[2] for c in self.calls], ["day", "month"])

    def test_stop_ends_the_search(self):
        with self.fake({"duckduckgo": ([_r("https://a.org/")], "")}):
            got = se.search("q", queries=["r"], settings={}, should_cancel=lambda: True)
        self.assertEqual(self.calls, [])
        self.assertIn("stopped by you", got["why"])

    def test_provider_errors_never_raise(self):
        keys = se.KeyStore(os.devnull + ".missing")
        with patch.object(se, "_api_search", side_effect=RuntimeError("boom")), \
             patch.object(se.time, "sleep"):
            results, why = se._run_provider("searxng", "q", 5, "any", "", "", {"searxng_url": "http://x"}, keys,
                                            False, lambda: False)
        self.assertEqual(results, [])
        self.assertIn("SearXNG failed", why)

    def test_rate_limit_pauses_the_provider(self):
        keys = se.KeyStore(os.devnull + ".missing")
        with patch.object(se, "_api_search", side_effect=Exception("429 Too Many Requests")):
            _r1, why = se._run_provider("tavily", "q", 5, "any", "", "", {}, keys, False, lambda: False)
            _r2, why2 = se._run_provider("tavily", "q", 5, "any", "", "", {}, keys, False, lambda: False)
        self.assertIn("rate-limited", why)
        self.assertIn("paused", why2)

    def test_the_tool_numbers_results_and_describes_the_search(self):
        answers = {"duckduckgo": ([_r("https://docs.python.org/3/", "Docs", "The docs", "2026-09-01"),
                                   _r("https://blog.example.com/p", "Post")], "")}
        with self.fake(answers):
            res = se.web_search_tool({"query": "python docs", "queries": ["python documentation"]}, {})
        self.assertTrue(res["ok"])
        self.assertIn("[1] Docs - docs.python.org · 2026-09-01", res["message"])
        self.assertIn("cite a claim as [n]", res["message"])
        web = res["event"]["web"]
        self.assertEqual((web["op"], web["queries"]), ("search", ["python docs", "python documentation"]))
        self.assertEqual([r["n"] for r in web["results"]], [1, 2])
        self.assertEqual(sources.current().get(1)["url"], "https://docs.python.org/3/")

    def test_nothing_found_says_so_honestly(self):
        with self.fake({}):
            res = se.web_search_tool({"query": "zxqv"}, {})
        self.assertFalse(res["ok"])
        self.assertEqual(res["status"], "permanent_failure")         # not retried by the executor
        self.assertIn("could not find a reliable source", res["message"])

    def test_progress_hears_each_query(self):
        heard = []
        with self.fake({"duckduckgo": ([_r("https://a.org/")], "")}):
            se.web_search_tool({"query": "a", "queries": ["b"]}, {}, progress=lambda **k: heard.append(k["web"]))
        self.assertEqual([h["current"] for h in heard], [0, 1])


# ==============================================================================
# fetch_page
# ==============================================================================
PAGE_TEXT = ("# Release notes\nThe tool is a thing [1] that does stuff.\n\n"
             + "Unrelated filler paragraph about other topics entirely. " * 40
             + "\n\n## Pricing\nThe price is 20 dollars per month as of September 2026 [2].\n\n"
             + "More filler text that says nothing much at all. " * 40)


class FetchPageTests(unittest.TestCase):
    def setUp(self):
        sources.start_turn()

    def pages(self, entries):
        return patch.object(web_tools, "read_pages", side_effect=lambda urls, **kw: (
            [kw["on_done"](e) or e for e in entries] if kw.get("on_done") else entries))

    def test_reads_pages_keeps_search_numbers_and_strips_link_markers(self):
        reg = sources.current()
        reg.add("https://short.example.com/r", "From search")                       # [1], a search result
        ok = {"url": "https://short.example.com/r", "final": "https://example.com/pricing", "note": "",
              "page": {"title": "Pricing", "text": PAGE_TEXT, "links": [("a", "x"), ("b", "y")], "date": "2026-09-02"}}
        bad = {"url": "https://paywalled.com/a", "error": "paywalled.com refused automated reading (403)."}
        heard = []
        with self.pages([ok, bad]):
            res = se.fetch_page_tool({"urls": [ok["url"], bad["url"]], "query": "price per month", "max_chars": 1500},
                                     progress=lambda **k: heard.append(k["web"]))
        self.assertTrue(res["ok"])
        msg = res["message"]
        self.assertIn("Read 1 of 2 pages", msg)
        self.assertIn("[1] Pricing - example.com · published 2026-09-02", msg)
        self.assertIn("20 dollars per month", msg)
        self.assertNotIn("thing [1]", msg)                                           # page link markers gone
        self.assertIn("[2] https://paywalled.com/a\n(Could not read it: paywalled.com refused", msg)
        self.assertEqual(reg.find("https://example.com/pricing"), 1)
        self.assertTrue(reg.get(1)["read"])
        self.assertIn("20 dollars", reg.get(1)["excerpt"])
        pages = res["event"]["web"]["pages"]
        self.assertEqual([(p["n"], p["ok"]) for p in pages], [(1, True), (2, False)])
        self.assertEqual(heard[-1]["pages"][1]["state"], "failed")

    def test_every_page_failing_is_a_failure(self):
        with self.pages([{"url": "https://a.org/", "error": "Not read: stopped by you.", "cancelled": True}]):
            res = se.fetch_page_tool({"urls": ["https://a.org/"]})
        self.assertFalse(res["ok"])

    def test_at_most_six_pages(self):
        urls = [f"https://s{i}.org/" for i in range(8)]
        seen = {}
        with patch.object(web_tools, "read_pages", side_effect=lambda u, **kw: seen.setdefault("urls", u) and [
                {"url": x, "error": "x"} for x in u]):
            res = se.fetch_page_tool({"urls": urls})
        self.assertEqual(len(seen["urls"]), 6)
        self.assertIn("Not read (at most 6 per call)", res["message"])

    def test_no_url(self):
        self.assertFalse(se.fetch_page_tool({"urls": []})["ok"])


class PassageTests(unittest.TestCase):
    def test_long_pages_are_cut_to_the_relevant_passages(self):
        sel = web_tools.select_passages(PAGE_TEXT, "price per month", 900)
        self.assertLessEqual(sel["shown"], 1100)
        self.assertIn("20 dollars per month", sel["text"])
        self.assertTrue(sel["text"].startswith("# Release notes"))                  # the opening is kept
        self.assertIn("[…]", sel["text"])
        self.assertIn("20 dollars", sel["excerpt"])

    def test_short_pages_are_whole(self):
        sel = web_tools.select_passages("Short page.", "anything", 6000)
        self.assertEqual((sel["text"], sel["shown"], sel["total"]), ("Short page.", 11, 11))

    def test_published_date(self):
        html = '<html><head><meta property="article:published_time" content="2026-03-04T08:00:00Z"></head></html>'
        self.assertEqual(web_tools.published_date(html), "2026-03-04")
        self.assertEqual(web_tools.published_date('<time datetime="2025-12-01">Dec 1</time>'), "2025-12-01")
        self.assertEqual(web_tools.published_date("<p>no date</p>"), "")


def _pdf(text_ops: bytes, compress: bool = True) -> bytes:
    stream = zlib.compress(text_ops) if compress else text_ops
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length %d%s >>\nstream\n" % (len(stream), b" /Filter /FlateDecode" if compress else b"") + stream
            + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = b"%PDF-1.4\n"
    for i, o in enumerate(objs, 1):
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    return out + b"trailer\n<< /Root 1 0 R >>\n%%EOF"


class PdfTests(unittest.TestCase):
    def test_extracts_text_from_a_compressed_page(self):
        got = pdf_text.extract(_pdf(b"BT /F1 12 Tf 72 700 Td (Hello PDF world) Tj 0 -14 Td [(Second) -250 (line)] TJ ET"))
        self.assertEqual(got["pages"], 1)
        self.assertIn("Hello PDF world", got["text"])
        self.assertIn("Second line", got["text"])

    def test_not_a_pdf(self):
        with self.assertRaises(pdf_text.PDFError):
            pdf_text.extract(b"<html>nope</html>")


# ==============================================================================
# When the web tools are offered, and what a reply saves
# ==============================================================================
class OfferTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: shared.app_state.get(k) for k in ("turn_policy", "turn_tools", "design_mode")}
        self.settings = dict(shared.app_state["settings"])
        shared.app_state["turn_policy"] = {"mode": "none"}

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                shared.app_state.pop(k, None)
            else:
                shared.app_state[k] = v
        shared.app_state["settings"] = self.settings

    def names(self):
        return {t["function"]["name"] for t in schemas.tools_for_current_turn(local=True)}

    def test_questions_get_only_the_web_tools(self):
        shared.app_state["settings"] = {**self.settings, "web_auto_search": True, "auto_tools": False}
        self.assertEqual(self.names(), {"web_search", "fetch_page", "web_read", "show_web_images"})   # 1.4.7: pictures from the web
        self.assertTrue(schemas.tool_allowed_current_turn("fetch_page"))
        self.assertFalse(schemas.tool_allowed_current_turn("write_file"))
        self.assertFalse(schemas.tool_allowed_current_turn("browser_open"))
        # With "Turn on helpful tools by itself" on, the one addition is use_tools (auto_tools.py).
        shared.app_state["settings"] = {**self.settings, "web_auto_search": True, "auto_tools": True}
        self.assertEqual(self.names(), {"web_search", "fetch_page", "web_read", "show_web_images", "use_tools"})

    def test_the_setting_turns_it_off_and_grounded_turns_it_back_on(self):
        shared.app_state["settings"] = {**self.settings, "web_auto_search": False, "auto_tools": False}
        self.assertEqual(self.names(), set())
        self.assertFalse(schemas.tool_allowed_current_turn("web_search"))
        shared.app_state["turn_tools"] = {"grounded"}
        self.assertEqual(self.names(), {"web_search", "fetch_page", "web_read", "show_web_images"})   # 1.4.7: pictures from the web

    def test_design_never_searches(self):
        shared.app_state["settings"] = {**self.settings, "web_auto_search": True}
        shared.app_state["design_mode"] = True
        self.assertEqual(self.names(), set())

    def test_fetch_page_is_read_only_and_benign(self):
        import permissions
        import tools.runtime as runtime
        spec = runtime.TOOL_REGISTRY.get("fetch_page")
        self.assertTrue(spec.read_only and spec.isolate)
        self.assertEqual(permissions.decide("fetch_page", {}), "auto")
        self.assertEqual(permissions.decide("fetch_page", {}, tainted=True), "ask")   # network egress once tainted


class SavedReplyTests(unittest.TestCase):
    def test_web_work_metrics_and_sources(self):
        events = [
            {"kind": "activity", "t": 100.0},
            {"kind": "model", "status": "done", "tokens_in": 1200, "tokens_out": 300, "model": "m1", "t": 101.0},
            {"kind": "tool", "tool": "web_search", "t": 102.0,
             "web": {"op": "search", "queries": ["a", "b"], "results": []}},
            {"kind": "tool", "tool": "fetch_page", "t": 103.0,
             "web": {"op": "read", "pages": [{"n": 1, "url": "https://a.org/", "domain": "a.org", "ok": True},
                                             {"n": 2, "url": "https://b.org/", "domain": "b.org", "ok": False}]}},
            {"kind": "model", "status": "done", "tokens_in": 2000, "tokens_out": 150, "model": "m2", "t": 104.0},
        ]
        web = sessions.web_of(events)
        self.assertEqual((web["q"], web["searches"], web["reads"]), (["a", "b"], 1, 2))
        m = sessions.metrics_of(events, started=None)
        self.assertEqual((m["in"], m["out"], m["model"], m["searches"], m["reads"]), (3200, 450, "m2", 1, 2))
        self.assertIsNone(sessions.web_of([{"kind": "tool", "tool": "read_file"}]))
        sources.start_turn()
        sources.current().add("https://a.org/", "A", read=True)
        sources.current().add("https://b.org/", "B")
        self.assertEqual([s["n"] for s in sessions.sources_for("As [2] says.")], [2])

    def test_steps_keep_the_tool_name(self):
        self.assertEqual(sessions._slim({"kind": "tool", "tool": "web_search", "label": "Search"})["tool"], "web_search")


if __name__ == "__main__":
    unittest.main()
