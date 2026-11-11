"""1.4.4: a reply never opens with the model talking to itself about the user's message."""
import unittest

import model_output
from agent import prompts


class SelfTalkTests(unittest.TestCase):
    def test_the_planning_opening_goes(self):
        cases = {
            "The user said something vague, I should clarify. What would you like to work on?": "What would you like to work on?",
            "The user just said \"okay\". I'll ask what they need.\n\nHi! What can I help with?": "Hi! What can I help with?",
            "The user's message is just a greeting. Hey! What's on your mind?": "Hey! What's on your mind?",
            "Okay, the user wants help but didn't say with what. Sure - what should we tackle?": "Sure - what should we tackle?",
            "I should ask a clarifying question. Could you tell me a bit more?": "Could you tell me a bit more?",
            "The message is short and unclear. Let me ask. What do you have in mind?": "What do you have in mind?",
            # Ornith 9B's own words (29 Oct 2026):
            "Nothing to do yet — your message was just \"okay.\"\n\nI'm ready to help. What would you like me to do?":
                "I'm ready to help. What would you like me to do?",
            "\"hm\" on its own isn't a task I can act on. Want me to do something specific, or was that just a test?":
                "Want me to do something specific, or was that just a test?",
        }
        for raw, want in cases.items():
            self.assertEqual(model_output.strip_self_talk(raw), want, raw)

    def test_real_answers_stay_as_they_are(self):
        for text in ("The user account control setting is under Security.", "The user interface has two panes.",
                     "Hi! What would you like to do?", "I should mention: the backup failed at 3 am.",
                     "The users table needs an index on email.", "Since the update, Windows uses a new driver.",
                     ""):
            self.assertEqual(model_output.strip_self_talk(text), text, text)

    def test_never_leaves_an_empty_reply(self):
        only = "The user said okay. I should clarify."
        self.assertEqual(model_output.strip_self_talk(only), only)

    @staticmethod
    def _stream(text, step=3):
        """What the chat's live draft gets: (channel, text) parts, fed a few characters at a time like a stream."""
        gate, out = model_output.SelfTalkGate(), []
        for i in range(0, len(text), step):
            out += gate.feed(text[i:i + step])
        out += gate.finish()
        joined = {}
        for ch, t in out:
            joined[ch] = joined.get(ch, "") + t
        return joined

    def test_the_live_draft_never_writes_and_erases_self_talk(self):
        # 1.4.8: the owner watched every reply write its first sentences and delete them.
        got = self._stream("The user said something vague, I should clarify. What would you like to work on?")
        self.assertEqual(got.get("content", "").strip(), "What would you like to work on?")
        self.assertIn("The user said something vague", got.get("reasoning", ""))
        got = self._stream("The user just said \"okay\". I'll ask what they need.\n\nHi! What can I help with?", step=1)
        self.assertEqual(got.get("content", "").strip(), "Hi! What can I help with?")

    def test_real_answers_stream_unchanged(self):
        for text in ("The user account control setting is under Security.", "The users table needs an index on email.",
                     "Hi! What would you like to do?", "Sure", "Here is the plan:\n1. Open it\n2. Save it"):
            for step in (1, 4, 50):
                self.assertEqual(self._stream(text, step), {"content": text}, (text, step))

    def test_a_reply_that_is_only_self_talk_still_shows(self):
        got = self._stream("The user said okay. I should clarify.")
        self.assertEqual(got.get("reasoning", "") + got.get("content", ""), "The user said okay. I should clarify.")

    def test_the_prompt_says_it_too(self):
        self.assertIn("Talk to the user, not about them", prompts.IDENTITY)
        self.assertIn("Talk to the user, not about them", prompts.IDENTITY_COMPACT)


if __name__ == "__main__":
    unittest.main()
