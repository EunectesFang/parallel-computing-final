import unittest

import week_run as week


class WeekSplitTests(unittest.TestCase):
    def test_holdout_is_stable_and_about_one_tenth(self):
        ids = [f"dev:{index:04d}" for index in range(3003)]
        holdout = [sentence_id for sentence_id in ids if week.is_holdout(sentence_id)]
        train = [sentence_id for sentence_id in ids if week.is_train(sentence_id)]
        burned = [sentence_id for sentence_id in ids if week.hash_bucket(sentence_id) == 0]
        self.assertEqual(holdout, [sentence_id for sentence_id in ids if week.is_holdout(sentence_id)])
        self.assertGreater(len(holdout), 250)
        self.assertLess(len(holdout), 360)
        self.assertGreater(len(train), 2200)
        self.assertEqual(set(holdout) & set(train), set())
        self.assertEqual(set(burned) & set(train), set())
        self.assertEqual(set(burned) & set(holdout), set())

    def test_action_index_matches_the_vote_loop(self):
        seen = []
        for horizon in range(1, 11):
            for branches in range(1, 11):
                seen.append(week.action_index(branches, horizon))
        self.assertEqual(seen, list(range(1, 101)))
        for index in range(1, 101):
            branches, horizon = week.decode_action(index)
            self.assertEqual(week.action_index(branches, horizon), index)

    def test_every_fourth_decision_is_kept(self):
        positions = week.decision_positions(11)
        self.assertEqual(positions, [3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(week.labeled_positions(positions, 4), [3, 7])

    def test_truncation_stops_at_eos_and_horizon(self):
        self.assertEqual(week.truncate_token_ids([1, 2, 99, 4], 10, 99), [1, 2])
        self.assertEqual(week.truncate_token_ids([1, 2, 3, 4], 2, 99), [1, 2])

    def test_translation_confidence_enters_the_policy_features(self):
        features = week.step_features("Hallo Welt.", ["Hello"], {
            "mt_next_max_prob": 0.7,
            "mt_next_entropy": 1.2,
            "mt_prefix_logprob": -0.4,
        })
        self.assertEqual([features[name] for name in week.CONFIDENCE_FEATURES], [0.7, 1.2, -0.4])
        self.assertEqual(len(features), len(week.FEATURE_NAMES))
        max_prob, entropy = week.softmax_confidence([0.0, 20.0, 0.0])
        self.assertGreater(max_prob, 0.99)
        self.assertLess(entropy, 0.02)
        self.assertEqual(week.mean_or_zero([]), 0.0)
        self.assertAlmostEqual(week.mean_or_zero([-1.0, -3.0]), -2.0)

    def test_read_reward_is_minus_the_correct_words_not_emitted(self):
        waiting = [{"b": 1, "h": 1, "matched": 0, "incorrect": 1}]
        rewards = week.state_rewards(waiting, pred_s=1.0, mt_s=[1.0], median_46=1.0)
        self.assertEqual(rewards[0], 0.0)
        self.assertAlmostEqual(rewards[1], -1.0 - week.BETA * (0.01 + 0.1))

        writing = [
            {"b": 1, "h": 1, "matched": 2, "incorrect": 0},
            {"b": 2, "h": 1, "matched": 0, "incorrect": 1},
        ]
        rewards = week.state_rewards(writing, pred_s=0.0, mt_s=[0.0], median_46=1.0)
        self.assertEqual(rewards[0], -2.0)
        self.assertEqual(week.reward_winner(rewards), 1)
        self.assertGreater(rewards[1], rewards[0])


if __name__ == "__main__":
    unittest.main()
