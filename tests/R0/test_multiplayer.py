from itertools import permutations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brass_ai.multiplayer import backup_path, chance_expectation, choose_for_actor, to_stable_seats


class MultiplayerTests(unittest.TestCase):
    def test_each_seat_chooses_its_own_utility(self):
        outcomes = [(0.8, 0.1, 0.05, 0.05), (0.1, 0.6, 0.2, 0.1),
                    (0.1, 0.2, 0.6, 0.1), (0.1, 0.1, 0.1, 0.7)]
        for actor in range(4):
            self.assertEqual(choose_for_actor(actor, outcomes), (actor, outcomes[actor]))

    def test_repeated_actor_and_chance_edges_never_flip_sign(self):
        self.assertEqual(backup_path([0, 0, None, 2, None, 1], [0.4, 0.35, 0.2, 0.05]),
                         (0.4, 0.4, 0.4, 0.2, 0.2, 0.35))

    def test_self_interested_opponent_is_not_a_coalition(self):
        # Seat 1 prefers its 0.4 even though the other branch gives the three
        # opponents a larger combined share and hurts root seat 0 more.
        outcomes = [(0.1, 0.2, 0.6, 0.1), (0.5, 0.4, 0.05, 0.05)]
        self.assertEqual(choose_for_actor(1, outcomes)[0], 1)

    def test_small_tree_with_consecutive_decisions_and_chance(self):
        chance = chance_expectation([(0.25, [1, 0, 0, 0]), (0.75, [0, 0, 1, 0])])
        second_actor_zero = choose_for_actor(0, [chance, [0.4, 0.3, 0.2, 0.1]])[1]
        self.assertEqual(choose_for_actor(0, [[0.3, 0.5, 0.1, 0.1], second_actor_zero])[0], 1)

    def test_stable_seats_survive_all_twenty_four_orders(self):
        expected = (0.4, 0.35, 0.2, 0.05)
        for order in permutations(range(4)):
            self.assertEqual(to_stable_seats([expected[s] for s in order], order), expected)

    def test_invalid_chance_and_seat_inputs_fail(self):
        for chance in ([], [(0.3, [1, 0, 0, 0])], [(-1, [1, 0, 0, 0]), (2, [0, 1, 0, 0])]):
            with self.subTest(chance=chance), self.assertRaises(ValueError):
                chance_expectation(chance)
        with self.assertRaises(ValueError):
            backup_path([None, 0], [0.25] * 4)
        with self.assertRaises(ValueError):
            choose_for_actor(4, [[0.25] * 4])
        with self.assertRaises(ValueError):
            choose_for_actor(0, [[0.4, 0.6]])
        with self.assertRaises(ValueError):
            to_stable_seats([0.25] * 4, [0, 0, 2, 3])


if __name__ == "__main__":
    unittest.main()
