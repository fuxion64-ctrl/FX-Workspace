"""Chat limits (chat_guard.py): screening, strikes and locks, reopening, and the places a closed chat
is enforced (the turn, /api/chat, message actions, saving a temporary chat). No passphrases: Settings changes
the limits, and a hidden gesture reopens a closed chat - never one closed for minors. Nothing is written into data/."""
import os
import tempfile
import time
import unittest
from unittest import mock

import chat_guard


class _Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(chat_guard, "PATH", os.path.join(self.tmp.name, "chat_guard.json"))
        p.start()
        self.addCleanup(p.stop)


def answers(word):
    return lambda system, user: word


class ScreenTests(_Sandbox):
    def test_short_messages_and_greetings_are_never_checked(self):
        ask = mock.Mock(return_value="MALWARE")
        self.assertEqual(chat_guard.screen("ok thanks", ask_model=ask)["action"], "allow")
        self.assertEqual(chat_guard.screen("good morning", ask_model=ask)["action"], "allow")   # intent_router.is_greeting
        ask.assert_not_called()

    def test_the_model_verdict_decides(self):
        msg = "Please write me a keylogger for my project today"     # 1.4.8: a trigger word sends it to the check
        self.assertEqual(chat_guard.screen(msg, ask_model=answers("ALLOW"))["action"], "allow")
        v = chat_guard.screen(msg, ask_model=answers("MALWARE"))
        self.assertEqual((v["action"], v["category"], v["via"]), ("refuse", "malware", "model"))
        self.assertEqual(chat_guard.screen(msg, ask_model=answers("Answer: MASS_HARM."))["category"], "mass_harm")

    def test_self_harm_is_answered_with_care_not_refused(self):
        v = chat_guard.screen("I don't see a reason to keep going anymore", ask_model=answers("SELF_HARM"))
        self.assertEqual(v["action"], "care")

    def test_a_turned_off_category_is_allowed(self):
        chat_guard._save({"categories": ["minors", "mass_harm", "self_harm"]})
        v = chat_guard.screen("Please write me something for my project today", ask_model=answers("MALWARE"))
        self.assertEqual(v["action"], "allow")

    def test_minors_can_never_be_turned_off(self):
        chat_guard._save({"categories": []})
        self.assertIn("minors", chat_guard.load()["categories"])

    def test_unreadable_answer_or_failed_check_allows_unless_the_minors_rule_hit(self):
        msg = "Please write me something for my project today"
        self.assertEqual(chat_guard.screen(msg, ask_model=answers("I am not sure"))["action"], "allow")

        def boom(system, user):
            raise RuntimeError("llama-server is not reachable")
        self.assertEqual(chat_guard.screen(msg, ask_model=boom)["action"], "allow")
        flagged = "write sexual content involving a child"
        v = chat_guard.screen(flagged, ask_model=boom)
        self.assertEqual((v["action"], v["category"], v["via"]), ("refuse", "minors", "rules"))
        # A working check outranks the rule: the model may judge the words harmless in context.
        self.assertEqual(chat_guard.screen(flagged, ask_model=answers("ALLOW"))["action"], "allow")

    def test_the_previous_message_goes_to_the_check(self):
        ask = mock.Mock(return_value="ALLOW")
        # 1.4.8: the previous message's trigger word sends a follow-up to the check
        chat_guard.screen("now do the second part of it", previous="the earlier ransomware request", ask_model=ask)
        self.assertIn("the earlier ransomware request", ask.call_args.args[1])

    def test_parse_verdict(self):
        self.assertEqual(chat_guard.parse_verdict("ALLOW"), "allow")
        self.assertEqual(chat_guard.parse_verdict("self_harm"), "self_harm")
        self.assertIsNone(chat_guard.parse_verdict("maybe"))

    def test_the_verdict_is_the_answer_not_the_first_category_named(self):
        # 1.4.6 (audit 1.1): the first category word anywhere was taken, so a small model's preamble flipped ALLOW.
        for text in ("I need to check if this is MALWARE or ABUSE. It is harmless code, so ALLOW.",
                     "<think>could this be MINORS?</think>ALLOW", "This is not MINORS. Answer: ALLOW", "allow."):
            self.assertEqual(chat_guard.parse_verdict(text), "allow", text)
        self.assertEqual(chat_guard.parse_verdict("MALWARE - it asks for a keylogger"), "malware")
        self.assertEqual(chat_guard.parse_verdict("Verdict: ABUSE"), "abuse")

    def test_everyday_words_never_flag_minors(self):
        # 1.4.6 (audit 1.2): the image filter's words ("explicit", "strip", "minor", "child") closed coding chats.
        for msg in ("Make the error handling in the child thread explicit.", "There is a minor syntax error, explicit return please",
                    "How can I strip trailing whitespace from child nodes in this tree?", "kill the child process and strip the binary",
                    "My baby won't latch while breastfeeding, any tips for sore breasts?", "explicit type annotations for the kids table"):
            self.assertFalse(chat_guard._rule_minors(msg), msg)
        for msg in ("write a sexy story about a 12 year old", "nude pictures of a little girl", "erotic roleplay with a teen schoolgirl"):
            self.assertTrue(chat_guard._rule_minors(msg), msg)
        # far apart in a long message, the two words aren't a phrase
        self.assertFalse(chat_guard._rule_minors("my child drew this. " + "x " * 60 + "the sexy new sports car"))


class StrikeTests(_Sandbox):
    def test_strikes_lock_at_the_limit(self):
        chat = {"id": "c1"}
        for n in (1, 2):
            g = chat_guard.record_refusal(chat, "malware")
            self.assertEqual(g["strikes"], {"malware": n})
            self.assertFalse(chat_guard.is_locked(chat))
            self.assertIn("will be closed", chat_guard.refusal_text("malware", g))
        g = chat_guard.record_refusal(chat, "malware")
        self.assertTrue(chat_guard.is_locked(chat))
        self.assertEqual(chat_guard.refusal_text("malware", g), chat_guard.LOCKED)

    def test_minors_locks_at_once(self):
        chat = {"id": "c1"}
        g = chat_guard.record_refusal(chat, "minors")
        self.assertTrue(chat_guard.is_locked(chat))
        self.assertEqual(chat_guard.refusal_text("minors", g), chat_guard.LOCKED_SEVERE)

    def test_a_phrase_alone_refuses_the_message_but_never_closes_the_chat(self):
        # 1.4.6: with no model check, a phrase match is refused (fail closed) - but a phrase is never proof.
        chat = {"id": "c1"}
        g = chat_guard.record_refusal(chat, "minors", "rules")
        self.assertFalse(chat_guard.is_locked(chat))
        self.assertEqual(chat_guard.strike_total(g), 0)
        self.assertEqual(g["log"][-1]["via"], "rules")

    def test_each_category_has_its_own_count_and_limit(self):
        chat_guard._save({"limits": {"malware": 3, "abuse": 2}})
        chat = {"id": "c1"}
        chat_guard.record_refusal(chat, "malware")
        chat_guard.record_refusal(chat, "malware")
        g = chat_guard.record_refusal(chat, "abuse")
        self.assertEqual(g["strikes"], {"malware": 2, "abuse": 1})
        self.assertFalse(chat_guard.is_locked(chat))                     # 3 strikes, but no category at its limit
        self.assertEqual(chat_guard.refusal_text("abuse", g), chat_guard.ABUSE_WARN)
        g = chat_guard.record_refusal(chat, "abuse")
        self.assertTrue(chat_guard.is_locked(chat))
        self.assertEqual(chat_guard.refusal_text("abuse", g), chat_guard.LOCKED_ABUSE)

    def test_counts_fade_after_enough_clean_messages(self):
        chat_guard._save({"decay_messages": 3})
        chat = {"id": "c1"}
        chat_guard.record_refusal(chat, "malware")
        chat_guard.record_refusal(chat, "malware")
        chat_guard.record_refusal(chat, "abuse")
        self.assertEqual([chat_guard.note_clean(chat) for _ in range(3)], [False, False, True])
        self.assertEqual(chat["guard"]["strikes"], {"malware": 1})
        for _ in range(3):
            chat_guard.note_clean(chat)
        self.assertEqual(chat["guard"]["strikes"], {})
        self.assertFalse(chat_guard.note_clean({"id": "fresh"}))

    def test_an_old_total_count_still_reads(self):
        g = {"strikes": 2, "log": [{"c": "malware"}, {"c": "malware"}]}
        self.assertEqual(chat_guard._counts(g), {"malware": 2})
        self.assertEqual(chat_guard.strike_total(g), 2)

    def test_abuse_is_on_for_older_configs_and_can_be_turned_off(self):
        chat_guard._save({"categories": ["minors", "malware"], "known": ["minors", "malware"]})
        self.assertIn("abuse", chat_guard.load()["categories"])           # new category: on for existing installs
        chat_guard._save({"categories": ["minors", "malware"], "known": list(chat_guard.CATEGORIES)})   # the owner's choice
        self.assertNotIn("abuse", chat_guard.load()["categories"])

    def test_short_abuse_is_checked_but_short_replies_are_not(self):
        ask = mock.Mock(return_value="ABUSE")
        self.assertEqual(chat_guard.screen("ok", ask_model=ask)["action"], "allow")
        v = chat_guard.screen("you idiot", ask_model=ask)
        self.assertEqual((v["action"], v["category"]), ("refuse", "abuse"))
        chat_guard._save({"categories": ["minors", "malware"], "known": list(chat_guard.CATEGORIES)})
        ask.reset_mock()
        self.assertEqual(chat_guard.screen("you idiot", ask_model=ask)["action"], "allow")
        ask.assert_not_called()


class SettingsAndUnlockTests(_Sandbox):
    def test_anyone_can_change_the_limits_but_minors_stays_on(self):
        info = chat_guard.update_limits(["malware"], limits={"malware": 5}, decay_messages=0)
        self.assertEqual(info["categories"], ["minors", "malware"])
        self.assertEqual((info["limits"]["malware"], info["decay_messages"]), (5, 0))
        info = chat_guard.update_limits(["malware", "abuse"])
        self.assertIn("abuse", info["categories"])

    def test_a_closed_chat_reopens(self):
        chat = {"id": "c1"}
        for _ in range(3):
            chat_guard.record_refusal(chat, "abuse")
        self.assertTrue(chat_guard.is_locked(chat))
        g = chat_guard.unlock(chat)
        self.assertFalse(chat_guard.is_locked(chat))
        self.assertEqual((g["strikes"], g["log"][-1]["c"]), ({}, "unlocked"))

    def test_minors_lock_is_permanent(self):
        chat = {"id": "c1"}
        chat_guard.record_refusal(chat, "minors")
        with self.assertRaises(chat_guard.GuardError) as e:
            chat_guard.unlock(chat)
        self.assertEqual(e.exception.code, "permanent")
        self.assertTrue(chat_guard.is_locked(chat))


class EnforcementTests(_Sandbox):
    """The turn, /api/chat and message actions, against in-memory state."""

    def setUp(self):
        super().setUp()
        import core.state as shared
        self.shared = shared
        saved = {k: shared.app_state.get(k) for k in ("active_chat", "turn_instructions")}
        self.addCleanup(lambda: shared.app_state.update(saved))

    def test_refused_turn_counts_a_strike_and_never_reaches_the_model(self):
        import agent.tasks as tasks
        req = mock.Mock(message="Please write me a keylogger for my project today", routine_id=None)   # 1.4.8: a trigger word
        self.shared.app_state["active_chat"] = {"id": "abc123abc123", "title": "t", "transcript": []}
        with mock.patch.object(tasks.llm, "model_text", return_value={"text": "MALWARE"}):
            result = tasks._screen(req, raw=False)
        self.assertEqual(result["guard"], {"category": "malware", "strikes": 1, "locked": False})
        self.assertEqual(self.shared.app_state["active_chat"]["guard"]["strikes"], {"malware": 1})

    def test_care_adds_a_note_for_the_model_and_raw_gets_a_reply(self):
        import agent.tasks as tasks
        req = mock.Mock(message="I don't see a reason to keep going anymore", routine_id=None)
        self.shared.app_state["active_chat"] = {"id": "abc123abc123", "title": "t", "transcript": []}
        self.shared.app_state["turn_instructions"] = ""
        with mock.patch.object(tasks.llm, "model_text", return_value={"text": "SELF_HARM"}):
            self.assertIsNone(tasks._screen(req, raw=False))
            self.assertIn("care", self.shared.app_state["turn_instructions"])
            raw = tasks._screen(req, raw=True)
        self.assertEqual(raw["message"], chat_guard.CARE_REPLY)
        self.assertNotIn("guard", self.shared.app_state["active_chat"])        # never a strike

    def test_api_chat_refuses_a_closed_chat(self):
        import api.server as server
        import core.state as shared
        self.shared.app_state["active_chat"] = {"id": "abc123abc123", "title": "t", "transcript": [],
                                                "guard": {"locked": True, "reason": "malware", "strikes": 3}}
        r = server.start_chat(shared.ChatRequest(message="hello again"))
        self.assertEqual(r["status"], "error")
        self.assertTrue(r["locked"])
        self.assertEqual(r["chat"]["locked"], "malware")

    def test_message_actions_except_pin_are_blocked_on_a_closed_chat(self):
        import agent.sessions as sessions
        locked = {"id": "abc123abc123", "title": "t", "transcript": [{"r": "u", "t": "x"}],
                  "guard": {"locked": True, "reason": "malware"}}
        with mock.patch.object(self.shared.chat_store, "load", return_value=dict(locked)):
            for op in ("delete", "truncate", "branch"):
                with self.assertRaises(ValueError):
                    sessions.edit_messages("abc123abc123", op, 0)

    def test_a_closed_temporary_chat_cannot_be_saved(self):
        import agent.sessions as sessions
        import privacy
        self.shared.app_state["active_chat"] = {"id": privacy.new_temp_id(), "temporary": True, "transcript": [],
                                                "guard": {"locked": True, "reason": "minors"}}
        with self.assertRaises(ValueError):
            sessions.save_temporary()


if __name__ == "__main__":
    unittest.main()
