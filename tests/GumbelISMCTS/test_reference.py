"""Independent equation checks and executable trace contract for the Rust port."""

import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gumbel_reference", ROOT / "scripts/reproduce-gumbel-is-mcts.py")
ref = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ref
SPEC.loader.exec_module(ref)


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


class OneStep:
    players = 4

    def __init__(self, rewards=((1., 2., 3., 4.),), logits=None):
        self.rewards = rewards
        self.logits = logits or (0.,) * len(rewards)

    def root_info(self):
        return ref.Info(0, (), (), tuple(range(len(self.rewards))))

    def sample_root(self, draws):
        return None

    def information(self, world):
        return self.root_info()

    def evaluate(self, info):
        return self.logits, (0.,) * self.players

    def chance_outcomes(self, world):
        return None

    def step(self, world, action):
        return action

    def terminal_returns(self, world):
        return None if world is None else self.rewards[world]


class EquationTests(unittest.TestCase):
    def test_completed_q_is_prior_weighted_and_actor_relative(self):
        node = ref.Node(ref.Info(2, (), (), (0, 1, 2)),
                        tuple(math.log(p) for p in (.2, .3, .5)), (9., 8., .6, 7.), 4)
        self.assertEqual(node.completed_q(), ([.6] * 3, .6))
        node.count[:] = [2, 1, 0]
        node.returns[:] = [[100., -100., .4, 50.], [100., -100., .8, 50.], [0.] * 4]
        # Visited prior mean = (.2*.2 + .3*.8)/(.2+.3) = .56.
        # v_mix = (.6 + 3*.56)/4 = .57, NOT visit-weighted mean .4.
        q, mixed = node.completed_q()
        for actual, expected in zip(q, (.2, .8, .57)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(mixed, .57)
        self.assertAlmostEqual(node.scores(50, .1)[0], math.log(.2) + .1 * 52 * .2)

    def test_equation_three_matches_literal_squared_discrepancy(self):
        rng = random.Random(61)
        for _ in range(150):
            width = rng.randint(2, 12)
            node = ref.Node(ref.Info(1, (), (), tuple(range(width))),
                            [rng.uniform(-2, 2) for _ in range(width)], (0., .2), 2)
            node.count[:] = [rng.randrange(10) for _ in range(width)]
            node.returns[:] = [[0., n * rng.uniform(-1, 1)] for n in node.count]
            improved = node.policy(50, .1)
            denom = 1 + sum(node.count)
            losses = [sum((p - (n + int(a == b)) / denom) ** 2
                          for b, (p, n) in enumerate(zip(improved, node.count)))
                      for a in range(width)]
            self.assertEqual(node.select(50, .1), min(range(width), key=losses.__getitem__))

    def test_softmax_underflow_does_not_destroy_visited_prior_normalization(self):
        node = ref.Node(ref.Info(0, (), (), (0, 1, 2)), (0., -1000., -1001.), (.25,), 1)
        node.count[:] = [0, 1, 3]
        node.returns[:] = [[0.], [.2], [2.4]]
        mean = (.2 + math.exp(-1) * .8) / (1 + math.exp(-1))
        self.assertAlmostEqual(node.completed_q()[1], (.25 + 4 * mean) / 5)
        self.assertTrue(all(math.isfinite(p) for p in node.policy(50, .1)))

    def test_canonical_ties(self):
        node = ref.Node(ref.Info(0, (), (), (3, 7)), (0., 0.), (0.,), 1)
        self.assertEqual(node.select(50, .1), 0)


class SearchTests(unittest.TestCase):
    def test_paper_schedule_floor_remainder_and_exact_budget(self):
        result = ref.search(ref.HiddenOffer())
        self.assertEqual([r["quota_per_action"] for r in result["rounds"]], [3, 6, 12, 25])
        self.assertEqual([len(r["candidates"]) for r in result["rounds"]], [16, 8, 4, 2])
        self.assertEqual(result["remainder_simulations"], 6)
        self.assertEqual(len(result["trajectories"]), 200)
        self.assertEqual(sum(result["nodes"][0]["count"]), 200)
        self.assertEqual(len(result["gumbels"]), 16)
        self.assertEqual(result["gumbels"], [-math.log(-math.log(u)) for u in result["draws"][:16]])

    def test_halving_observes_last_backup(self):
        game = OneStep(((0., 8., -4., 9.), (10., -5., 2., 1.)))
        result = ref.search(game, simulations=2, k=2, tape=[.5, .5])
        self.assertEqual(result["selected_action"], 1)
        self.assertEqual(result["rounds"][0]["retained"], [1])
        self.assertEqual(result["nodes"][0]["returns"], [list(r) for r in game.rewards])

    def test_tiny_budgets_one_action_and_non_power_of_two_legal_sets(self):
        for width in (1, 3, 5, 16):
            game = OneStep(((1., 2., 3., 4.),) * width)
            for budget in (0, 1, 2, 3, 7, 8, 31, 64):
                result = ref.search(game, simulations=budget, k=32)
                self.assertEqual(sum(result["nodes"][0]["count"]), budget)
                self.assertEqual(len(result["trajectories"]), budget)
                self.assertIn(result["selected_action"], range(width))
                self.assertTrue(all(r["quota_per_action"] > 0 for r in result["rounds"]))
                self.assertEqual(result["status"], "prior_fallback" if budget == 0 else "completed")
        self.assertEqual(ref.effective_k(16, 5, 200), 4)
        self.assertEqual(ref.effective_k(16, 16, 23), 4)
        self.assertEqual(ref.effective_k(16, 16, 24), 8)
        self.assertEqual(ref.effective_k(16, 16, 31), 8)

    def test_first_new_opponent_information_set_is_the_leaf(self):
        result = ref.search(ref.HiddenOffer(), simulations=1)
        trajectory = result["trajectories"][0]
        self.assertEqual(len(trajectory["path"]), 1)
        leaf = result["nodes"][trajectory["leaf"]]
        self.assertEqual(leaf["info"]["player"], 1)
        self.assertEqual(sum(leaf["count"]), 0)

    def test_bootstrap_uses_new_actors_full_value_vector(self):
        class VectorLeaves(ref.HiddenOffer):
            def evaluate(self, info):
                logits, _ = super().evaluate(info)
                return logits, (.1, .2, .3, .4) if info.player else (0.,) * 4

        result = ref.search(VectorLeaves(), simulations=1)
        trajectory = result["trajectories"][0]
        self.assertEqual(trajectory["value"], (.1, .2, .3, .4))
        root = result["nodes"][0]
        self.assertEqual(root["returns"][trajectory["path"][0][1]], [.1, .2, .3, .4])

    def test_terminal_chance_vector_backup_and_information_set_merging(self):
        game = ref.HiddenOffer()
        result = ref.search(game)
        # One root for every resampled world; actor 1 distinguishes its own bit.
        roots = [n for n in result["nodes"] if n["info"]["player"] == 0]
        self.assertEqual(len(roots), 1)
        self.assertEqual({n["info"]["private"] for n in result["nodes"]
                          if n["info"]["player"] == 1}, {(0,), (1,)})
        terminal = [t for t in result["trajectories"] if t["leaf"] == "terminal"]
        self.assertTrue(terminal)
        self.assertTrue(all(len(t["chance"]) == 1 for t in terminal))
        self.assertEqual({t["chance"][0] for t in terminal}, {-1, 1})
        # Independently sum every leaf's four-seat vector along each trace path.
        for node_id, node in enumerate(result["nodes"]):
            for edge in range(len(node["count"])):
                visits = [t for t in result["trajectories"] if (node_id, edge) in t["path"]]
                self.assertEqual(node["count"][edge], len(visits))
                for player in range(game.players):
                    self.assertAlmostEqual(node["returns"][edge][player], sum(t["value"][player] for t in visits))

    def test_toy_belief_uses_conditional_probability_not_prior(self):
        game = ref.HiddenOffer()
        self.assertEqual(game.sample_root(ref.Draws(0, [.79, .49, .51])).types, (0, 1, 1, 0))
        self.assertEqual(game.sample_root(ref.Draws(0, [.81, .49, .51])).types, (0, 0, 1, 0))
        # Other hidden bits cannot change a player's permitted input/evaluation.
        a = ref.World((0, 1, 0, 0), (7,))
        b = ref.World((0, 1, 1, 1), (7,))
        self.assertEqual(game.information(a), game.information(b))
        self.assertEqual(game.evaluate(game.information(a)), game.evaluate(game.information(b)))

    def test_recorded_draws_override_seed_and_replay_every_statistic(self):
        first = ref.search(ref.HiddenOffer(), seed=13)
        second = ref.search(ref.HiddenOffer(), seed=999, tape=first["draws"])
        second["config"]["seed"] = 13
        self.assertEqual(canonical(first), canonical(second))

    def test_checked_in_golden_trace(self):
        path = ROOT / "tests/GumbelISMCTS/fixtures/reference-v1.json"
        golden = json.loads(path.read_text(encoding="utf-8"))
        actual = ref.search(ref.HiddenOffer(golden["players"]), **golden["config"], tape=golden["draws"])
        actual["players"] = golden["players"]
        self.assertEqual(canonical(actual), golden)

    def test_reject_invalid_inputs_and_tapes(self):
        for k, budget in ((0, 200), (16, -1)):
            with self.assertRaises(ValueError):
                ref.search(OneStep(), k=k, simulations=budget)
        for tape in ([], [0.], [1.], [float("nan")], [.5, .5], [.5, None]):
            with self.assertRaises(ValueError):
                ref.search(OneStep(), simulations=1, tape=tape)
        for logits, value in (([float("nan")], [0.]), ([0.], [float("inf")]), ([], [0.])):
            with self.assertRaises(ValueError):
                ref.Node(ref.Info(0, (), (), (0,)), logits, value, 1)
        with self.assertRaises(ValueError):
            ref.Draws(1).categorical(((0, 0.),))


if __name__ == "__main__":
    unittest.main()
