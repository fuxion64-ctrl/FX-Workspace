"""The Orbs thinking mark (index.html, 1.4.3): formations, blends between kinds of work, the setting, and the
loop's sleeping rules. The maths runs under node; the rest is checked in the page source."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
MATH = HTML[HTML.index("// orbs-math:start"):HTML.index("// orbs-math:end")]
ORBS = HTML[HTML.index("var Orbs = (() => {"):HTML.index("function paintAct(turn, thinkLive)")]


def run_js(body):
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("node is not installed")
    out = subprocess.run([node, "-e", MATH + "\n" + body], capture_output=True, text=True, encoding="utf-8", timeout=60)
    if out.returncode:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


class FormationTests(unittest.TestCase):
    def test_every_formation_stays_in_its_box(self):
        # Seven kinds of work, three orbs each, sampled over 20 s: finite numbers, inside the mark (with its radius).
        r = run_js("""
          const bad = [];
          for (const act of ORB_ACTS) for (let t = 0; t < 20; t += 1 / 30) for (let i = 0; i < 3; i++) {
            const q = ORB_FORMS[act](t, i);
            if (q.length !== 6 || !q.every(Number.isFinite)) bad.push([act, t, i, 'nan']);
            else if (Math.abs(q[0]) + q[2] > 1.02 || Math.abs(q[1]) + q[2] > 1.02) bad.push([act, +t.toFixed(2), i, 'out']);
            else if (q[3] < 0 || q[3] > 1 || q[5] < 0 || q[5] >= 3) bad.push([act, t, i, 'range']);
          }
          console.log(JSON.stringify({ acts: ORB_ACTS, bad: bad.slice(0, 5) }));""")
        self.assertEqual(r["acts"], ["think", "read", "search", "tool", "write", "draw", "wait"])
        self.assertEqual(r["bad"], [])

    def test_each_kind_of_work_has_its_own_shape(self):
        # Averaged over a cycle, no two formations put the orbs in the same places.
        r = run_js("""
          const sig = act => [0, 1, 2].map(i => { let x = 0, y = 0, n = 0; for (let t = 0; t < 6; t += .05) { const q = ORB_FORMS[act](t, i); x += Math.abs(q[0]); y += Math.abs(q[1]); n++; } return [x / n, y / n]; }).flat();
          const s = Object.fromEntries(ORB_ACTS.map(a => [a, sig(a)]));
          let min = 9, pair = '';
          for (const a of ORB_ACTS) for (const b of ORB_ACTS) if (a < b) { const d = Math.hypot(...s[a].map((v, j) => v - s[b][j])); if (d < min) { min = d; pair = a + '/' + b; } }
          console.log(JSON.stringify({ min, pair }));""")
        self.assertGreater(r["min"], .1, r["pair"])


class BlendTests(unittest.TestCase):
    def test_changes_never_make_the_orbs_jump(self):
        # 60 frames a second for 60 s, the work changing every 50-500 ms at random (and every one of the 42 pairs
        # in turn): per frame, no orb moves much more than the fastest formation moves on its own.
        r = run_js("""
          const form = orbForm;
          const step = 1000 / 60, dist = (p, q) => Math.max(...ORB_IDX.map(i => Math.hypot(p[i][0] - q[i][0], p[i][1] - q[i][1])));
          let natural = 0;
          for (const act of ORB_ACTS) { let prev = null; for (let f = 0; f < 600; f++) { const t = f * step / 1000, p = ORB_IDX.map(i => form(act, t, i)); if (prev) natural = Math.max(natural, dist(p, prev)); prev = p; } }
          let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
          const run = changes => { const o = { act: 'think', from: null, t0: 0 }; let prev = null, worst = 0, n = 0, at = 0;
            for (let f = 0; f < 3600; f++) { const now = f * step, t = now / 1000;
              if (now >= at && n < changes.length) { orbRetarget(o, changes[n++], t, now, form); at = now + (changes.delay ? changes.delay : 50 + rnd() * 450); }
              const p = orbState(o, t, now, form); if (!p.every(q => q.every(Number.isFinite))) return 99; if (prev) worst = Math.max(worst, dist(p, prev)); prev = p; }
            return worst; };
          const random = Array.from({ length: 400 }, () => ORB_ACTS[Math.floor(rnd() * 7)]);
          const pairs = []; for (const a of ORB_ACTS) for (const b of ORB_ACTS) if (a !== b) pairs.push(a, b);
          const slow = pairs.slice(); slow.delay = 700;
          const fast = pairs.slice(); fast.delay = 30;
          console.log(JSON.stringify({ natural, random: run(random), pairs: run(slow), fast: run(fast) }));""")
        limit = r["natural"] * 1.5 + .03
        for name in ("random", "pairs", "fast"):
            self.assertLess(r[name], limit, f"{name}: {r}")

    def test_a_blend_lands_on_the_new_formation(self):
        r = run_js("""
          const form = orbForm, o = { act: 'think', from: null, t0: 0 };
          orbRetarget(o, 'tool', 1, 1000, form);
          const e = 1000 + ORB_BLEND + 50;
          const mid = orbState(o, 1.2, 1200, form), end = orbState(o, e / 1000, e, form), want = ORB_IDX.map(i => form('tool', e / 1000, i));
          console.log(JSON.stringify({ blending: mid !== null, done: o.from === null, same: JSON.stringify(end) === JSON.stringify(want) }));""")
        self.assertEqual(r, {"blending": True, "done": True, "same": True})

    def test_the_same_work_again_is_not_a_change(self):
        r = run_js("""
          const form = orbForm, o = { act: 'read', from: null, t0: 0 };
          orbRetarget(o, 'read', 1, 1000, form); console.log(JSON.stringify(o.from === null && o.t0 === 0));""")
        self.assertTrue(r)


class EngineTests(unittest.TestCase):
    def test_one_loop_that_sleeps(self):
        self.assertEqual(ORBS.count("requestAnimationFrame(frame)"), 3)                  # wake, the next frame, a skipped one (30 fps)
        self.assertIn("if (document.hidden) return;", ORBS)                                # hidden in the tray: no frames
        self.assertIn("if (live) raf = requestAnimationFrame(frame); else last = 0;", ORBS)   # nothing moving: it stops
        self.assertIn("const shown = el => el.getClientRects().length > 0;", ORBS)         # marks not on screen cost nothing
        self.assertIn("root.dataset.state !== 'paused'", ORBS)                             # a paused task freezes them

    def test_reduced_motion_draws_still_shapes(self):
        self.assertIn("let t = still ? .4 : clock", ORBS)
        self.assertIn("if (o.drawn === sig && !o.from) return;", ORBS)                     # drawn once per change
        self.assertIn(':root:is([data-motion="reduced"], [data-motion="off"]) .act-line .think.orb-mark { display: inline-block; }', HTML)

    def test_the_setting(self):
        from core import config
        self.assertEqual(config.DEFAULT_SETTINGS["think_mark"], "auto")
        self.assertIn("think_mark: str | None = None", (ROOT / "api" / "server.py").read_text(encoding="utf-8"))
        # 1.4.6: Auto ("New" in Extras) is each look's own mark from the approved board; Orbs and Legacy stay in Extras.
        self.assertIn("seg([['auto', 'New'], ['orbs', 'Orbs'], ['legacy', 'Legacy']]", HTML)
        self.assertIn("if (m === 'auto') return markKey() ? 'marks' : 'orbs';", ORBS)
        self.assertIn("const MARK_LIB = (() => {", HTML)
        for key in ("classic", "sunrise", "terminal", "eclipse", "aurora", "claude", "clay2", "mono", "void", "prism"):
            self.assertIn(f"LOOKS.{key} = {{", HTML)
        self.assertIn("(S.settings && S.settings.claude_mark) === 'liquid' ? 'clay2' : 'claude'", ORBS)
        self.assertIn("if (d.secret || d.tone === 'mossad') return 'legacy';", ORBS)

    def test_legacy_keeps_the_old_marks(self):
        # Legacy is the CSS from before, untouched, and the orbs hide it only on a mark they draw.
        self.assertIn('offset-path: path("M 22.2 7.3 A 11 4.5 -28 1 1 3.8 16.7 A 11 4.5 -28 1 1 22.2 7.3")', HTML)   # the Saturn rings
        self.assertIn(".think.orb-mark > i { display: none !important; }", HTML)
        self.assertIn("if (window.Orbs && Orbs.on()) {", HTML)                            # setThinkAct: no fade-out/in for orbs
        self.assertIn("function release(el) {", ORBS)

    def test_every_appearance_dresses_them(self):
        for skin in ("classic", "dawn", "dusk", "terminal", "eclipse", "aurora", "clay", "mono", "void", "prism"):
            self.assertRegex(ORBS, rf"\b{skin}: ", skin)


class ClaudeNodesTests(unittest.TestCase):
    """The Claude look's mark: a turning graph of three nodes - four while a tool runs, three in a row for a search."""

    def test_nodes_stay_in_the_box_and_count_right(self):
        r = run_js("""
          const bad = [], count = (act, t) => ORB_IDX.map(i => orbClayForm(act, t, i)).filter(q => q[2] > .02).length;
          for (const act of ORB_ACTS) for (let t = 0; t < 12; t += 1 / 30) for (const i of ORB_IDX) {
            const q = orbClayForm(act, t, i);
            if (!q.every(Number.isFinite) || Math.abs(q[0]) + q[2] > 1.02 || Math.abs(q[1]) + q[2] > 1.02) bad.push([act, i]);
          }
          console.log(JSON.stringify({ bad: bad.slice(0, 3), tool: count('tool', 1), search: count('search', 1), wait: count('wait', 1), think0: count('think', 0), think5: count('think', 5) }));""")
        self.assertEqual(r, {"bad": [], "tool": 4, "search": 3, "wait": 1, "think0": 3, "think5": 4})

    def test_claude_blends_without_jumps(self):
        r = run_js("""
          const step = 1000 / 60, o = { act: 'think', from: null, t0: 0 }; let prev = null, worst = 0, n = 0;
          for (let f = 0; f < 2400; f++) { const now = f * step, t = now / 1000;
            if (f % 37 === 0) orbRetarget(o, ORB_ACTS[n++ % ORB_ACTS.length], t, now, orbClayForm);
            const p = orbState(o, t, now, orbClayForm);
            if (prev) worst = Math.max(worst, ...ORB_IDX.map(i => Math.hypot(p[i][0] - prev[i][0], p[i][1] - prev[i][1]))); prev = p; }
          console.log(JSON.stringify(worst));""")
        self.assertLess(r, .12)

    def test_the_page_draws_it(self):
        self.assertIn("clay: { ...base, nodes: true,", ORBS)
        self.assertIn("const form = (act, t, i) => dress && dress.nodes ? orbClayForm(act, t, i)", ORBS)
        self.assertIn("function claySay(el)", HTML)                      # the label: what it's thinking about
        self.assertIn(".think-secs", HTML)                               # and the seconds so far


if __name__ == "__main__":
    unittest.main()
