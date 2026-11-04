"""The effort system (effort.py) as runtime data: concrete limits per level,
capability-based fallback, per-model tuning, run budgets, safe-boundary
changes - and proof that no level touches permissions, approvals or undo."""
import copy
import unittest

import effort
import replay_harness as rh
import core.config as config


class ProfileTests(unittest.TestCase):
    def test_six_levels_with_growing_limits(self):
        self.assertEqual(effort.LEVELS, ("low", "normal", "high", "extra", "max", "ultra"))
        ordered = [effort.PROFILES[l] for l in ("low", "normal", "high", "extra", "max")]
        for a, b in zip(ordered, ordered[1:]):
            for attr in ("token_budget", "context_tokens", "max_tool_calls", "time_limit_s", "local_iterations"):
                self.assertLessEqual(getattr(a, attr), getattr(b, attr), f"{a.name}->{b.name} {attr}")
        self.assertEqual([p.verification for p in ordered], ["none", "self", "reviewer", "reviewer", "reviewer"])
        ultra = effort.PROFILES["ultra"]
        self.assertEqual(ultra.orchestration, "none")
        self.assertEqual(ultra.stage_caps, ())
        self.assertFalse(ultra.delegation)
        self.assertGreater(ultra.local_iterations, effort.PROFILES["max"].local_iterations)
        self.assertGreater(ultra.max_tool_calls, effort.PROFILES["max"].max_tool_calls)
        self.assertFalse(effort.PROFILES["low"].thinking)
        self.assertGreater(effort.PROFILES["max"].reasoning_tokens, effort.PROFILES["high"].reasoning_tokens)

    def test_overrides_only_tune_limits(self):
        raw = {"m": {"high": {"max_tool_calls": 9999, "token_budget": 5, "permission_mode": "god",
                              "verification": "none", "undo": False},
                     "bogus-level": {"max_tool_calls": 3}}, 5: {}}
        clean = effort.clean_overrides(raw)
        self.assertEqual(clean, {"m": {"high": {"max_tool_calls": 200, "token_budget": 1000}}})
        p = effort.profile_for("high", "m", clean)
        self.assertEqual((p.max_tool_calls, p.token_budget, p.verification), (200, 1000, "reviewer"))
        self.assertEqual(effort.profile_for("high", "other-model", clean), effort.PROFILES["high"])


class CapabilityTests(unittest.TestCase):
    def test_small_context_falls_back_to_the_nearest_lower_level(self):
        caps = {"context": 8192, "tool_protocol": "native"}
        self.assertEqual(effort.supported_levels(caps), ["low", "normal", "high"])
        level, why = effort.resolve("max", caps)
        self.assertEqual(level, "high")
        self.assertIn("8,192", why)

    def test_workflow_needs_structured_calls_and_reliability(self):
        self.assertEqual(effort.resolve("ultra", {"context": 65536, "tool_protocol": "text"})[0], "max")
        self.assertEqual(effort.resolve("ultra", {"context": 65536, "tool_call_validity": 0.5})[0], "max")
        self.assertEqual(effort.resolve("ultra", {"context": 65536, "tool_protocol": "native"}), ("ultra", None))

    def test_unknown_capabilities_do_not_block_a_level(self):
        self.assertEqual(effort.resolve("max", {}), ("max", None))
        self.assertEqual(effort.resolve("nonsense", {}), ("normal", None))

    def test_choice_precedence(self):
        s = {"effort": "normal", "effort_by_model": {"m": "high"}}
        self.assertEqual(effort.choose_level(s, "m"), ("high", "model"))
        self.assertEqual(effort.choose_level(s, "other"), ("normal", "setting"))
        self.assertEqual(effort.choose_level(s, "m", {"effort": "extra"}), ("extra", "project"))
        self.assertEqual(effort.choose_level(s, "m", {"effort": "extra"}, session="low"), ("low", "session"))

    def test_auto_is_selected_with_existing_precedence_and_resolved_per_turn(self):
        s = {"effort": "auto", "effort_by_model": {"m": "high"}}
        self.assertEqual(effort.choose_level(s, "other"), ("auto", "setting"))
        self.assertEqual(effort.choose_level(s, "m"), ("high", "model"))
        self.assertEqual(effort.choose_level(s, "m", session="auto"), ("auto", "session"))
        self.assertEqual(effort.auto_level({"mode": "none", "words": 2, "trivial": True})[0], "low")
        # A quick look is Balanced (no model plan or review); a long read and any change are High (1.4.6).
        self.assertEqual(effort.auto_level({"mode": "read", "words": 7, "trivial": False})[0], "normal")
        self.assertEqual(effort.auto_level({"mode": "read", "words": 45, "trivial": False})[0], "high")
        self.assertEqual(effort.auto_level({"mode": "action", "words": 7, "trivial": False})[0], "high")
        self.assertEqual(effort.auto_level({"mode": "action", "words": 90, "parts": 4, "complex": True})[0], "extra")
        self.assertEqual(effort.auto_level({"mode": "action", "words": 140, "parts": 6, "complex": True})[0], "max")


class BudgetTests(unittest.TestCase):
    def test_steps_and_tokens(self):
        b = effort.RunBudget(effort.profile_for("low", "m", {"m": {"low": {"local_iterations": 2, "token_budget": 1000}}}), local=True)
        self.assertIsNone(b.exhausted())
        b.record_call(400, 100)
        self.assertIsNone(b.exhausted())
        b.record_call(400, 200)
        self.assertEqual(b.exhausted(), "step")
        b2 = effort.RunBudget(effort.profile_for("low", "m", {"m": {"low": {"token_budget": 1000}}}), local=True)
        b2.record_call(900, 200)
        self.assertEqual(b2.exhausted(), "token")

    def test_repeated_prompt_is_not_counted_again(self):
        """Every step re-sends the same system prompt and tools (served from the
        prompt cache); only new prompt tokens and generated tokens count."""
        b = effort.RunBudget(effort.PROFILES["normal"], local=True)
        b.record_call(9000, 50)            # first step: the whole prompt
        b.record_call(9200, 40)            # +200 new (the tool result), not 9,200 again
        b.record_call(9300, 30)
        self.assertEqual(b.tokens_in, 9300)
        self.assertEqual(b.tokens_out, 120)
        b.record_call(4000, 10)            # history trimmed: the prompt is reprocessed
        self.assertEqual(b.tokens_in, 13300)

    def test_pending_change_waits_for_a_boundary_and_keeps_usage(self):
        b = effort.RunBudget(effort.PROFILES["normal"], local=True)
        b.record_call(10, 10)
        b.request_change(effort.PROFILES["max"])
        self.assertEqual(b.profile.name, "normal")                  # nothing changes until apply_pending
        self.assertEqual(b.apply_pending().name, "max")
        self.assertEqual((b.profile.name, b.steps), ("max", 1))
        self.assertIsNone(b.apply_pending())


class ReplayTests(unittest.TestCase):
    def run_fx(self, name, fx=None):
        fx = fx or rh.load_fixture(name)
        run = rh.run_fixture(fx)
        self.assertEqual(rh.check_expectations(fx, run), [], name)
        return run

    def test_run_stops_cleanly_at_its_step_budget(self):
        run = self.run_fx("effort_step_budget")
        self.assertEqual(run.result.get("budget_reached"), "step")
        started = next(e for e in run.events if e.get("phase") == "run_started")
        self.assertEqual(started["limits"]["steps_local"], 3)          # the tuned limit was applied

    def test_change_mid_run_applies_between_steps(self):
        run = self.run_fx("effort_change_midrun")
        ids = {e.get("phase") or e["kind"]: e["id"] for e in run.events}
        change = next(e for e in run.events if e.get("phase") == "effort_changed")
        model_calls = [e for e in run.events if e["kind"] == "model"]
        # The change was requested while reply #2 streamed; that call finished
        # untouched and the new level applied before step 3 started.
        self.assertTrue(model_calls[1]["id"] < change["id"] < model_calls[2]["id"])
        self.assertEqual(model_calls[1]["status"], "done")
        self.assertEqual(change["effort"], "high")
        self.assertTrue(ids)

    def test_no_level_changes_permissions_approvals_or_undo(self):
        """Non-negotiable 7: the same run at every level asks for exactly the
        same approvals, with the same undo markers, and runs the same tools."""
        import permissions
        base = rh.load_fixture("snake_game_cutoff")
        seen = {}
        for level in effort.LEVELS:
            fx = copy.deepcopy(base)
            fx["setup"]["settings"] = {**fx["setup"]["settings"], "effort": level}
            run = rh.run_fixture(fx)
            cards = [[(a["tool_name"], a["view"]["undoable"], a["view"]["undo_text"]) for a in card] for card in run.approvals]
            seen[level] = (cards, [n for n, _ in run.executed])
            self.assertEqual(effort.permission_fields(), set())
        self.assertEqual(len({repr(v) for v in seen.values()}), 1, seen)
        settings = dict(config.DEFAULT_SETTINGS)
        for level in effort.LEVELS:
            s = {**settings, "effort": level}
            for tool in ("write_file", "delete_path", "run_powershell", "list_dir", "open_url"):
                self.assertEqual(permissions.decide(tool, s), permissions.decide(tool, settings), (level, tool))


if __name__ == "__main__":
    unittest.main()
