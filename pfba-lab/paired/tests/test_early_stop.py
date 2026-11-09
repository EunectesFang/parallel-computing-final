"""CPU tests for vote early-stop decisions. No torch or model weights."""
from __future__ import annotations

import unittest

import run_pfba as pfba


class CompletedWordTests(unittest.TestCase):
    def test_trailing_space_finishes_the_last_word_before_strip(self):
        self.assertEqual(pfba.completed_word_text("the cat ", False), "the cat")
        self.assertEqual(pfba.completed_word_text("the cat", False), "the")

    def test_eos_finishes_a_word_without_a_trailing_space(self):
        self.assertEqual(pfba.completed_word_text("the cat", True), "the cat")

    def test_partial_token_is_not_a_new_vote(self):
        self.assertEqual(pfba.completed_word_text("the cat s", False), "the cat")
        self.assertEqual(pfba.completed_word_text("hello", False), "")


class VoteDecisionTests(unittest.TestCase):
    def test_unfinished_row_blocks_a_false_disagreement(self):
        rows = [("the cat sat", False), ("the cat", False), ("the cat sat", False), ("the cat", False)]
        decision = pfba.vote_early_stop_decision(rows, 0.6, 4)
        self.assertEqual(decision.action, "continue")
        self.assertEqual(decision.resolved_words, 2)

    def test_resolved_disagreement_stops_and_preserves_full_ralcp(self):
        full = ["the cat sat down", "the cat sat up", "the cat ran", "the dog"]
        decision = pfba.vote_early_stop_decision([(text, True) for text in full], 0.6, 4)
        self.assertEqual(decision.action, "stop")
        self.assertEqual(pfba.upstream_ralcp(decision.texts, 0.6), pfba.upstream_ralcp(full, 0.6))
        self.assertEqual(pfba.upstream_ralcp(decision.texts, 0.6), "the cat")

    def test_ended_short_row_still_votes(self):
        rows = [("the", True), ("the cat sat", False), ("the cat ran", False), ("the cat sat", True)]
        decision = pfba.vote_early_stop_decision(rows, 0.6, 4)
        self.assertEqual(decision.action, "stop")
        self.assertEqual(pfba.upstream_ralcp(decision.texts, 0.6), "the cat")

    def test_fourth_agreed_word_finishes_the_original_budget(self):
        rows = [("a b c d", False), ("a b c d", False), ("a b c d", False), ("a b c d", False)]
        decision = pfba.vote_early_stop_decision(rows, 0.6, 4)
        self.assertEqual(decision.action, "finish")
        self.assertEqual(decision.selected_words, 4)

    def test_agreement_shorter_than_the_checkpoint_keeps_generating(self):
        rows = [("a b", True), ("a b", True), ("a b", True), ("a b", True)]
        decision = pfba.vote_early_stop_decision(rows, 0.6, 4)
        self.assertEqual(decision.action, "continue")


class CallSignatureTests(unittest.TestCase):
    def test_unknown_arguments_are_not_forwarded(self):
        seen = {}

        class Translator:
            def translate_batch(self, sources, prefix):
                seen["sources"] = list(sources)
                seen["prefix"] = prefix
                return ["ok"]

        result = pfba.call_translate_batch(
            Translator(), ["de"], "en", final=True, early_stop_k=4, threshold=0.6)
        self.assertEqual(result, ["ok"])
        self.assertEqual(seen, {"sources": ["de"], "prefix": "en"})

    def test_accepted_arguments_are_forwarded(self):
        seen = {}

        class Translator:
            def translate_batch(self, sources, prefix, final=False, early_stop_k=0, threshold=0.6):
                seen.update(final=final, early_stop_k=early_stop_k, threshold=threshold)
                return ["ok"]

        pfba.call_translate_batch(Translator(), ["de"], "en", final=False, early_stop_k=4, threshold=0.6)
        self.assertEqual(seen, {"final": False, "early_stop_k": 4, "threshold": 0.6})


if __name__ == "__main__":
    unittest.main()
