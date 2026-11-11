"""MossadGPT's world monitor (osint/): places, the local store, the feeds' parsing, satellite maths, the
collector's on/off rule, the world_monitor tool, and that only MossadGPT's turns are offered it.
No network: feeds are exercised on recorded samples; the store lives in a temporary folder."""
import calendar
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import core.config as config
import core.state as shared
from osint import collector, geo, satellites, sources, store, tool
import sources as citations

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISS = ("ISS (ZARYA)", "1 25544U 98067A   26272.11005302  .00004557  00000+0  91790-4 0  9991",
       "2 25544  51.6312 145.7721 0007123 200.7262 159.3438 15.48685648587861")


class TempStore(unittest.TestCase):
    """Every test gets its own store and orbit file - never data/osint."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        store.close()
        self._paths = (store.DB_PATH, satellites.TLE_PATH)
        store.DB_PATH = os.path.join(self.tmp, "osint.db")
        satellites.TLE_PATH = os.path.join(self.tmp, "tle.txt")
        collector.configure(lambda: {})

    def tearDown(self):
        store.close()
        store.DB_PATH, satellites.TLE_PATH = self._paths
        shutil.rmtree(self.tmp, ignore_errors=True)


class GeoTests(unittest.TestCase):
    def test_places_in_any_language(self):
        self.assertEqual(geo.place("Haifa")["name"], "Haifa")
        self.assertEqual(geo.place("חיפה")["name"], "Haifa")
        self.assertEqual(geo.place("בחיפה")["name"], "Haifa")                # a Hebrew prefix
        self.assertEqual(geo.place("Хайфа")["name"], "Haifa")
        self.assertEqual(geo.place("the Red Sea")["kind"], "region")
        self.assertEqual(geo.place("הים האדום")["name"].lower(), "red sea")
        p = geo.place("32.1, 34.8")
        self.assertEqual((p["kind"], p["lat"], p["lon"]), ("point", 32.1, 34.8))
        self.assertIsNone(geo.place("Xyzzyqwerty"))
        self.assertIsNone(geo.place("95, 10"))                                 # not a latitude

    def test_distance_and_where(self):
        self.assertAlmostEqual(geo.km(32.0853, 34.7818, 31.7683, 35.2137), 54, delta=3)   # Tel Aviv - Jerusalem
        self.assertIn("Haifa", geo.where(32.80, 35.02))
        self.assertIn("no city within 400 km", geo.where(-50.0, -140.0))


class StoreTests(TempStore):
    def test_positions_history_and_expiry(self):
        now = time.time()
        rows = [{"id": "abc123", "t": now, "lat": 32.0, "lon": 35.0, "alt": 1000, "hdg": 90, "spd": 500, "label": "ELY1",
                 "props": {"reg": "4X-EKA"}}]
        self.assertEqual(store.put_positions("flights", rows, now=now), 1)
        store.put_positions("flights", [{**rows[0], "t": now + 30, "lon": 35.1}], now=now + 30)      # < 1 min: no history point
        store.put_positions("flights", [{**rows[0], "t": now + 70, "lon": 35.2}], now=now + 70)
        self.assertEqual(len(store.track("abc123", now - 1)), 2)
        self.assertEqual(store.find("ely1")[0]["id"], "abc123")
        self.assertEqual(store.find("4X-EKA")[0]["id"], "abc123")
        store.put_positions("flights", [], now=now + 1000, max_age_s=120)   # not seen for long: gone
        self.assertEqual(store.current("flights"), [])

    def test_events_dedupe_and_prune(self):
        now = time.time()
        ev = {"uid": "usgs:1", "layer": "quakes", "t": now - 10 * 86400, "lat": 1, "lon": 2, "kind": "earthquake", "title": "M5"}
        self.assertEqual(store.put_events([ev, ev]), 1)
        store.put_events([{**ev, "uid": "usgs:2", "t": now}])
        self.assertEqual(store.count_since("quakes", 0), 2)
        store.prune(days=7, max_mb=1000, now=now)
        self.assertEqual([e["uid"] for e in store.events("quakes")], ["usgs:2"])


class FeedParsingTests(unittest.TestCase):
    def test_aircraft(self):
        a = sources._aircraft({"hex": "AE03F6", "flight": "PAT171  ", "r": "84-00171", "t": "BE20", "dbFlags": 1,
                               "alt_baro": 26000, "gs": 251.8, "track": 236.5, "lat": 60.28, "lon": -152.6, "seen_pos": 1.0}, 1000.0, "adsb.lol")
        self.assertEqual((a["id"], a["label"], a["alt"], a["spd"]), ("ae03f6", "PAT171", 7925, 466))   # feet -> m, knots -> km/h
        self.assertEqual(a["props"]["mil"], 1)
        self.assertEqual(a["t"], 999.0)
        g = sources._aircraft({"hex": "1", "alt_baro": "ground", "lat": 1, "lon": 2}, 5.0, "x")
        self.assertEqual((g["alt"], g["props"]["ground"]), (0, 1))
        self.assertIsNone(sources._aircraft({"hex": "1"}, 5.0, "x"))

    def test_gdelt_names(self):
        self.assertEqual(sources._actor("UNITED STATES"), "United States")
        self.assertEqual(sources._actor("THE US"), "The US")

    def test_gdelt_rows(self):
        row = [""] * 61
        row[0], row[6], row[16], row[26], row[28], row[29] = "123", "ISRAEL", "HAMAS", "190", "19", "4"
        row[30], row[31], row[32], row[52], row[53], row[56], row[57], row[59], row[60] = "-10", "5", "2", "Gaza, Israel", "IS", "31.5", "34.46", "20260929120000", "https://example.org/a"
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("x.CSV", "\t".join(row) + "\n" + "\t".join(["1"] * 61) + "\n")
        calls = iter([b"1 abc http://data.gdeltproject.org/gdeltv2/x.export.CSV.zip\n", buf.getvalue()])
        with mock.patch.object(sources, "_get", side_effect=lambda *a, **k: next(calls)):
            out = sources.gdelt_events()
        self.assertEqual(len(out), 1)                                          # the unrelated row is dropped
        self.assertEqual((out[0]["kind"], out[0]["title"]), ("fighting", "Fighting: Israel -> Hamas - Gaza, Israel"))
        self.assertEqual(out[0]["t"], calendar.timegm((2026, 9, 29, 12, 0, 0)))    # 2026-09-29 12:00 UTC


class SatelliteTests(TempStore):
    def setUp(self):
        super().setUp()
        sources.save_tles(satellites.TLE_PATH, {"stations": [ISS]})

    def test_positions_and_passes(self):
        pos = satellites.positions(1790698000)
        self.assertEqual(len(pos), 1)
        self.assertTrue(400 < pos[0]["alt_km"] < 440)
        self.assertTrue(-52 <= pos[0]["lat"] <= 52)                             # the ISS's inclination
        passes = satellites.passes("iss", 32.08, 34.78, hours=24, start=1790698000)
        self.assertTrue(passes)
        for p in passes:
            self.assertTrue(p["rise"] <= p["peak"] <= p["set"])
            self.assertGreaterEqual(p["max_elevation"], 10)

    def test_missing_groups_keep_old_orbits(self):
        def fetch(group):
            if group == "starlink":
                raise sources.NotUpdated("nothing new")
            return [ISS] if group == "stations" else []
        with mock.patch.object(sources, "tle_group", side_effect=fetch):
            n = collector._job_satellites()
        self.assertEqual(n, 1)
        self.assertEqual(collector._extra["sat_missing"], ["starlink"])        # never had it: said, and retried later
        self.assertEqual(collector._extra["sat_retry"], 7200)


class CollectorTests(TempStore):
    def test_only_while_mossadgpt_is_on_screen(self):
        collector._last_beat = 0
        self.assertFalse(collector.active())
        collector.configure(lambda: {"osint_background": True})
        self.assertTrue(collector.active())
        collector.configure(lambda: {})
        with mock.patch.object(collector, "_ensure_thread"):
            collector.beat()
        self.assertTrue(collector.active())
        collector._last_beat = time.time() - collector.BEAT_TIMEOUT_S - 1
        self.assertFalse(collector.active())

    def test_a_failing_feed_backs_off_and_says_why(self):
        with mock.patch.object(collector, "_job", side_effect=sources.FeedError("HTTP 503")):
            st = collector.run_layer("quakes")
        self.assertEqual(st["state"], "error")
        self.assertIn("HTTP 503", st["error"])
        self.assertGreater(st["next"], time.time() + 100)


class ToolTests(TempStore):
    def setUp(self):
        super().setUp()
        now = time.time()
        for name in collector.LAYERS:                                         # everything fresh: no network
            collector._status[name].update(last_ok=now, state="live")
        store.put_positions("flights", [{"id": "738053", "t": now, "lat": 32.9, "lon": 34.6, "alt": 3000, "hdg": 137, "spd": 552,
                                         "label": "ELY397", "props": {"type": "B739", "reg": "4X-EHD"}}])
        store.put_positions("military", [{"id": "ae1234", "t": now, "lat": 32.6, "lon": 35.3, "alt": 7000, "hdg": 10, "spd": 700,
                                          "label": "RCH123", "props": {"mil": 1}}])
        store.put_events([{"uid": "usgs:x", "layer": "quakes", "t": now - 600, "lat": 32.7, "lon": 35.5, "kind": "earthquake",
                           "title": "M3.1 - 10 km E of Tiberias", "url": "https://earthquake.usgs.gov/x", "props": {"mag": 3.1, "depth_km": 8}},
                          {"uid": "gdelt:1", "layer": "news", "t": now - 900, "lat": 33.2, "lon": 35.5, "kind": "fighting",
                           "title": "Fighting: A -> B - South Lebanon", "url": "https://example.org/story", "props": {"sources": 1}}])
        sources.save_tles(satellites.TLE_PATH, {"stations": [ISS]})
        self.cites = []
        patcher = mock.patch.object(tool, "_cite", side_effect=lambda url, title, **k: self.cites.append(url) or len(self.cites))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_near(self):
        r = tool.world_monitor_tool({"action": "near", "place": "Haifa", "radius_km": 150})
        self.assertTrue(r["ok"])
        m = r["message"]
        self.assertIn("AIRCRAFT: 2 broadcasting positions (1 flagged military)", m)
        self.assertIn("RCH123 MILITARY", m)
        self.assertLess(m.index("RCH123"), m.index("ELY397"))                  # military first
        self.assertIn("3,000 m", m)
        self.assertIn("M3.1 - 10 km E of Tiberias", m)
        self.assertIn("treat each as one outlet's report", m)
        self.assertIn("https://example.org/story", self.cites)
        self.assertIn("[1]", m)
        self.assertEqual(collector.status()["focus"]["label"], "Haifa")        # the globe goes there

    def test_track_passes_status_and_errors(self):
        self.assertIn("ELY397 (B739, 4X-EHD)", tool.world_monitor_tool({"action": "track", "query": "ely397"})["message"])
        self.assertIn("satellite", tool.world_monitor_tool({"action": "track", "query": "ISS"})["message"])
        self.assertIn("No aircraft or satellite matching", tool.world_monitor_tool({"action": "track", "query": "zzz999"})["message"])
        self.assertIn("passes over Tel Aviv", tool.world_monitor_tool({"action": "passes", "place": "Tel Aviv"})["message"])
        self.assertIn("WORLD MONITOR status", tool.world_monitor_tool({"action": "status"})["message"])
        bad = tool.world_monitor_tool({"action": "near", "place": "Xyzzyqwerty"})
        self.assertFalse(bad["ok"])
        self.assertIn("Couldn't place", bad["message"])
        self.assertFalse(tool.world_monitor_tool({"action": "launch"})["ok"])


class MentionTests(TempStore):
    """Voice mode's globe follows what each spoken sentence names."""

    def test_places_in_order_any_language(self):
        name = lambda s: [p["name"] for p in tool.mentions(s)]          # noqa: E731
        self.assertEqual(name("Fighting near Sanaa in Yemen and threats in the Red Sea."), ["Sanaa", "Yemen", "Red Sea"])
        self.assertEqual(name("Then over Tel Aviv, not Tel Mond."), ["Tel Aviv"])       # the longest name wins
        self.assertEqual(name("בחיפה ובירושלים שקט."), ["Haifa", "Jerusalem"])         # Hebrew prefixes
        self.assertEqual(name("Над Москвой и Хайфой тихо."), ["Moscow", "Haifa"])      # Russian endings
        self.assertEqual(name("Nice weather for reading today."), [])                  # a sentence's first capital proves nothing
        self.assertEqual(name("no places here at all"), [])

    def test_aircraft_and_the_iss(self):
        store.put_positions("military", [{"id": "ae1234", "t": time.time(), "lat": 32.6, "lon": 35.3, "label": "RCH123", "props": {}}])
        sources.save_tles(satellites.TLE_PATH, {"stations": [ISS]})
        got = tool.mentions("RCH123 is heading north while the ISS passes over Haifa.")
        self.assertEqual([(p["name"], p["kind"]) for p in got], [("RCH123", "aircraft"), ("ISS", "satellite"), ("Haifa", "city")])
        self.assertEqual((got[0]["layer"], got[0]["id"]), ("military", "ae1234"))


class DockPageTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            self.html = f.read()

    def test_voice_docks_beside_the_globe_in_mossadgpt_only(self):
        voice = self.html[self.html.index("const Voice = (() => {"):self.html.index("function close() {", self.html.index("function applyDock() {"))]
        self.assertIn("let d = ops && !stage.hidden ? dock : 'full';", voice)       # only MossadGPT docks
        self.assertIn("if (d === 'mini' && ask) d = 'right';", voice)               # an OK to give needs room
        self.assertIn("Globe.voiceFocus(d !== 'full');", voice)
        self.assertIn("Globe.follow(q.text);", self.html)                          # the globe goes to what each sentence names
        self.assertIn("Globe.highlightEvidence(q.refs)", self.html)               # spoken citations highlight evidence
        self.assertIn("Globe.cancelFollow(true)", self.html)                       # interrupted: pending follow is invalidated
        self.assertIn("Voice.redock();", self.html)
        css = self.html[self.html.index("<style>"):self.html.index("</style>")]
        for d in ("left", "right", "bottom", "mini"):
            self.assertIn(f'data-dock="{d}"', css)
        self.assertIn('html.voice-docked[data-secret="mossad"] :is(#welcome, #scroll, #composer-dock) { display: none !important; }', css)


class NudgeTests(unittest.TestCase):
    """Small local models answered world questions from memory (made-up ISS times) until the turn said plainly
    to call the tool first - in the header and in the final instruction line."""

    def test_world_questions_get_the_nudge(self):
        for q in ("When does the ISS next pass over Tel Aviv?", "what's happening in the Red Sea", "any military aircraft near Haifa",
                  "מה קורה בעזה", "רעידת אדמה היום?", "Какие самолёты над Хайфой?"):
            self.assertIn("Call world_monitor before answering", tool.turn_hint(q), q)
            self.assertIn("earlier chats", tool.turn_hint(q))
        for q in ("write me a poem", "stack 10 parts right now", "fix this bug currently in main.py"):
            self.assertEqual(tool.turn_hint(q), "", q)

    def test_the_last_line_says_call_it(self):
        import agent.models as models
        keys = ("turn_tools", "turn_policy", "current_user_request", "conversation", "mode", "turn_persona", "task_context")
        saved = {k: shared.app_state.get(k) for k in keys}
        self.addCleanup(lambda: shared.app_state.update(saved))
        shared.app_state.update(turn_tools={"world"}, turn_policy={"mode": "none", "reason": "question"}, mode=config.MODE_LOCAL,
                                current_user_request="When does the ISS pass over Haifa?", turn_persona="", task_context="",
                                conversation=[{"role": "system", "content": "x"}, {"role": "user", "content": "When does the ISS pass over Haifa?"}])
        last = models.build_messages()[-1]["content"]
        self.assertTrue(last.rstrip().endswith("answer only from what it returns (Markdown is rendered)."))
        self.assertIn("WORLD MONITOR NEEDED", last)
        self.assertIn("web or data tools you're given still work", last)       # not "no tools" (said shorter in 1.4.8)


class OfferTests(unittest.TestCase):
    def test_only_mossadgpt_turns_get_it(self):
        import agent.tasks as tasks
        import tools.schemas as schemas
        saved = {k: shared.app_state.get(k) for k in ("turn_tools", "turn_persona", "turn_files", "turn_attachments")}
        self.addCleanup(lambda: shared.app_state.update(saved))
        for persona, has in (("mossadgpt", True), ("israelgpt", False), ("custom", False), (None, False)):
            tasks._begin_composer_tools(shared.ChatRequest(message="what's over Haifa", persona=persona))
            self.assertEqual("world" in shared.app_state["turn_tools"], has, persona)
            self.assertEqual("world_monitor" in schemas.world_hidden(), not has, persona)
        self.assertIn("world_monitor", schemas.READ_ONLY_TOOLS)
        import composer_tools
        self.assertIn("world_monitor", composer_tools.OFFERS["world"])
        self.assertIn("WORLD MONITOR", composer_tools.HEADERS["world"])

    def test_settings(self):
        out = config.clean_osint_settings({"osint_areas": [{"name": "Home", "lat": 32.8, "lon": 35.0, "nm": 900}, {"lat": "x"}, "junk"],
                                           "osint_layers": {"flights": False, "evil": True, "news": "no"}, "osint_days": 0,
                                           "osint_max_mb": 10 ** 9, "osint_background": "yes"})
        self.assertEqual(out["osint_areas"], [{"name": "Home", "lat": 32.8, "lon": 35.0, "nm": 250}])
        self.assertEqual(out["osint_layers"], {"flights": False})
        self.assertEqual((out["osint_days"], out["osint_max_mb"], out["osint_background"]), (1, 50000, False))
        d = config.DEFAULT_SETTINGS
        self.assertEqual((d["osint_background"], d["osint_days"], d["osint_max_mb"]), (False, 7, 2048))


class PageTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            self.html = f.read()

    def test_globe_is_mossadgpt_only_and_local(self):
        css = self.html[self.html.index("<style>"):self.html.index("</style>")]
        self.assertIn(".ops-globe, .ops-hud, .ops-panel, .ops-card, .ops-tip, .ops-btn { display: none; }", css)
        # The HUD (focus label, evidence, controls) shares the globe's box, so it never covers the chat or voice panel.
        self.assertIn("html.ops-view[data-secret=\"mossad\"] .ops-hud { display: block; position: absolute; z-index: 3; top: var(--topbar-h); left: var(--globe-l); right: var(--globe-r);", css)
        hud = self.html[self.html.index('<div class="ops-hud" id="ops-hud">'):self.html.index('<div class="ops-card"')]
        for part in ('id="ops-evidence"', 'id="ops-focus"', 'id="ops-panel"', 'role="toolbar"'):
            self.assertIn(part, hud)
        self.assertIn(':root[data-secret="mossad"] .ops-btn { display: inline-flex; }', css)
        js = self.html[self.html.index("const Globe = (() => {"):self.html.index("/* ---- Settings > Diagnostics", self.html.index("const Globe = (() => {"))]
        self.assertIn("root.dataset.secret === 'mossad'", js)
        self.assertIn("fetch('/api/osint/basemap')", js)
        self.assertNotRegex(js, r"https?://(?!globe\.adsb\.lol)")               # the page fetches nothing from outside
        self.assertIn("motionStill()", js)
        self.assertIn("Globe.sync();", self.html)
        self.assertIn("Date.now() / 1000", js)                                 # data times are unix seconds



class EvidenceTests(unittest.TestCase):
    def test_tool_result_carries_cited_sources_and_feed_age(self):
        citations.start_turn()
        n = citations.current().add("https://example.org/report", title="Observed report", date="today")
        status = {"focus": {}, "layers": [{"id": "news", "state": "live", "age_s": 12, "source": "GDELT"}]}
        with mock.patch.object(tool, "_status", return_value=f"Checked [{n}]"), mock.patch.object(collector, "status", return_value=status):
            result = tool.world_monitor_tool({"action": "status"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["world"]["evidence"][0]["url"], "https://example.org/report")
        self.assertEqual(result["world"]["layers"][0]["age_s"], 12)


class IntelTests(TempStore):
    """osint/intel.py - Velocity's ideas over this app's own data: GPS jamming, the watch officer, what changed."""

    def setUp(self):
        super().setUp()
        from osint import intel
        self.intel = intel
        intel._seen.clear()
        self.addCleanup(intel._seen.clear)

    def aircraft(self, n_bad, n_good, lat=32.6, lon=35.1, **extra):
        now = time.time()
        rows = [{"id": f"b{i}", "t": now, "lat": lat + i * .01, "lon": lon, "label": f"BAD{i}", "props": {"nac_p": 2, "nic": 8}}
                for i in range(n_bad)]
        rows += [{"id": f"g{i}", "t": now, "lat": lat + i * .01, "lon": lon, "label": f"OK{i}", "props": {"nac_p": 10, "nic": 8}}
                 for i in range(n_good)]
        rows += [{"id": "gr", "t": now, "lat": lat, "lon": lon, "label": "GROUND", "props": {"nac_p": 0, "ground": 1}}]   # ignored
        store.put_positions("flights", rows)

    def test_degraded_means_poor_accuracy(self):
        self.assertTrue(self.intel.degraded({"nac_p": 7}))
        self.assertTrue(self.intel.degraded({"nic": 6, "nac_p": 9}))
        self.assertFalse(self.intel.degraded({"nac_p": 9, "nic": 8}))
        self.assertIsNone(self.intel.degraded({}))

    def test_jamming_cells_need_enough_aircraft_and_rank_by_share(self):
        self.aircraft(2, 3)
        self.intel.sample()
        cells = self.intel.jamming()
        self.assertEqual(len(cells), 1)
        self.assertEqual((cells[0]["bad"], cells[0]["n"], cells[0]["level"]), (2, 5, "high"))     # the ground vehicle isn't counted
        self.intel._seen.clear()
        self.aircraft(1, 1, lat=10, lon=10)                                                      # 2 aircraft: too few to say
        self.intel.sample()
        self.assertEqual([c for c in self.intel.jamming() if c["lat"] < 20], [])

    def test_the_watch_officer_raises_each_alert_once(self):
        now = time.time()
        store.put_positions("military", [{"id": "e1", "t": now, "lat": 31.0, "lon": 34.5, "label": "ELY7", "props": {"squawk": "7700"}},
                                         {"id": "h1", "t": now, "lat": 31.1, "lon": 34.6, "label": "HJK1", "props": {"squawk": "7500"}}])
        new = self.intel.scan([{"name": "Israel", "lat": 31.8, "lon": 35.0}])
        self.assertEqual(sorted(a["props"]["severity"] for a in new), ["high", "medium"])
        self.assertIn("7700 (emergency)", next(a["title"] for a in new if "ELY7" in a["title"]))
        self.assertEqual(self.intel.scan([]), [])                                                  # not again this hour

    def test_strong_quakes_near_the_areas_alert(self):
        now = time.time()
        store.put_events([{"uid": "q1", "layer": "quakes", "t": now, "lat": 33.0, "lon": 35.5, "kind": "earthquake",
                           "title": "M 4.8 - Lebanon", "url": "https://earthquake.usgs.gov/q1", "props": {"mag": 4.8}},
                          {"uid": "q2", "layer": "quakes", "t": now, "lat": -40, "lon": 170, "kind": "earthquake",
                           "title": "M 4.9 - New Zealand", "url": "", "props": {"mag": 4.9}}])
        titles = [a["title"] for a in self.intel.scan([{"name": "Israel", "lat": 31.8, "lon": 35.0}])]
        self.assertEqual(titles, ["M 4.8 - Lebanon"])                                              # far and small: no alert

    def test_rewind_and_changes_read_the_history(self):
        now = time.time()
        store.put_positions("military", [{"id": "m1", "t": now - 3600, "lat": 32.0, "lon": 35.0, "label": "RCH1", "props": {}}], now=now - 3600)
        store.put_positions("military", [{"id": "m1", "t": now, "lat": 32.5, "lon": 35.2, "label": "RCH1", "props": {}},
                                         {"id": "m2", "t": now, "lat": 32.1, "lon": 34.9, "label": "RCH2", "props": {}}])
        then = store.at(["military"], now - 3600)
        self.assertEqual([(r["id"], r["lat"], r["label"]) for r in then], [("m1", 32.0, "RCH1")])
        c = self.intel.changes({"name": "Israel", "lat": 31.8, "lon": 35.0, "radius_km": 300})
        self.assertEqual(c["aircraft"]["military"], {"now": 2, "before": 1})

    def test_the_tool_answers_jamming_alerts_and_changes(self):
        self.aircraft(2, 3)
        with mock.patch.object(tool, "_ensure", return_value=[]):
            jam = tool.world_monitor_tool({"action": "jamming", "place": "Haifa"})
            self.assertIn("2 of 5 aircraft degraded", jam["message"])
            self.assertIn("doesn't prove it", jam["message"])                                     # said for what it is
            self.assertTrue(tool.world_monitor_tool({"action": "alerts"})["ok"])
            self.assertIn("Military aircraft seen", tool.world_monitor_tool({"action": "changes", "place": "Israel"})["message"])
        self.assertIn("Call world_monitor", tool.turn_hint("is there gps jamming over haifa?"))


class FocusTests(TempStore):
    def test_track_names_the_thing_so_the_globe_can_lock_on(self):
        store.put_positions("military", [{"id": "ae1234", "t": time.time(), "lat": 32.6, "lon": 35.3, "label": "RCH123", "props": {}}])
        # No network: a live refresh once replaced this row with the real ae1234 (a US military hex).
        with mock.patch.object(tool, "_ensure", return_value=[]):
            result = tool.world_monitor_tool({"action": "track", "query": "RCH123"})
        focus = result["world"]["focus"]
        self.assertEqual((focus["kind"], focus["layer"], focus["id"], focus["label"]), ("aircraft", "military", "ae1234", "RCH123"))

    def test_a_new_focus_drops_the_old_ones_object(self):
        collector.set_focus(32.6, 35.3, "RCH123", 150, kind="aircraft", layer="military", id="ae1234")
        collector.set_focus(32.8, 35.0, "Haifa", 60, kind="city")
        focus = collector.status()["focus"]
        self.assertEqual((focus["label"], focus["kind"]), ("Haifa", "city"))
        self.assertNotIn("layer", focus)
        self.assertNotIn("id", focus)


if __name__ == "__main__":
    unittest.main()
