import copy
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from brass_ai.beliefs import sample_world_setup
from brass_ai.host import RulesHost, HostError

COMPOSITION = json.loads((ROOT / "configs/contracts/v0.2.0/card-composition.json").read_text())


class HostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b = RulesHost(), RulesHost()

    @classmethod
    def tearDownClass(cls):
        cls.a.close()
        cls.b.close()

    def setUp(self):
        self.context = self.a.reset(1201)

    def view(self):
        return self.a.observe(self.context["actor"])

    def choose(self, kind, **fields):
        choice = next(c for c in self.a.choices() if c["kind"] == kind and all(c.get(k) == v for k, v in fields.items()))
        self.context = self.a.apply(self.context, choice)
        return choice

    def load_view(self, view, seed=99):
        self.context = self.a.reset(seed, sample_world_setup(view, COMPOSITION, seed))

    def put(self, view, slot, kind, level=1, seat=None, resource=None, flipped=False):
        seat = view["seat"] if seat is None else seat
        mat = next(m for m in view["players"][seat]["mat"] if m["type"] == kind)
        index = next(i for i, t in enumerate(mat["tiles"]) if t["level"] == level)
        tile = mat["tiles"][index]
        del mat["tiles"][:index + 1]
        tile["flipped"] = flipped
        if resource is not None: tile["resource"] = resource
        next(s for s in view["slots"] if s["id"] == slot)["tile"] = tile
        return tile

    def connect(self, view, link, seat=None):
        next(l for l in view["links"] if l["id"] == link)["owner"] = view["seat"] if seat is None else seat

    def hand(self, view, cards):
        view["deck_count"] += len(view["own_hand"]) - len(cards)
        view["players"][view["seat"]]["hand_count"] = len(cards)
        view["own_hand"] = sorted(cards)

    def rail(self, view):
        view["context"].update(era="RailEra", round=2, actions_remaining=2)
        # Return the four unknown setup burns to the now-publicly reshuffled pack.
        for p in view["players"]:
            view["deck_count"] += p["hidden_discards"]
            p["hidden_discards"] = 0

    def test_develop_all_types_two_tiles_and_board_iron_priority(self):
        view = self.view()
        other = (view["seat"] + 1) % 4
        self.put(view, "Coalbrookdale:1", "IronWorks", seat=other, resource=2)
        self.put(view, "Dudley:1", "IronWorks", seat=other, level=2, resource=2)
        self.load_view(view)
        types = {c["industry"] for c in self.a.choices() if c["kind"] == "develop"}
        self.assertEqual({"CoalMine", "IronWorks", "CottonMill", "Manufacturer", "Brewery"}, types)
        self.choose("develop", industry="CottonMill")
        self.assertEqual({"Coalbrookdale:1", "Dudley:1"}, {c["target"] for c in self.a.choices()})
        self.choose("iron", target="Dudley:1")
        self.assertIn("finish", {c["kind"] for c in self.a.choices()})
        self.choose("develop", industry="Brewery")
        self.choose("iron", target="Coalbrookdale:1")
        self.assertEqual(["finish"], [c["kind"] for c in self.a.choices()])

    def test_loan_boundary_and_no_double_confirmation(self):
        view = self.view()
        view["players"][view["seat"]]["income_marker"] = 3
        self.load_view(view)
        self.assertEqual(-7, self.view()["players"][view["seat"]]["income"])
        self.choose("loan")
        self.assertEqual(-10, self.view()["players"][view["seat"]]["income"])
        self.choose("finish")
        view = self.view()
        view["players"][view["seat"]]["income_marker"] = 2
        self.load_view(view)
        self.assertNotIn("loan", {c["kind"] for c in self.a.choices()})

    def test_build_overbuild_single_slot_and_multi_coal_payment(self):
        view = self.view()
        self.rail(view)
        self.hand(view, ["location:Coventry"])
        view["players"][view["seat"]]["cash"] = 40
        # Own pottery II can be replaced by III; needs two coal, including a
        # market purchase once the connected one-cube mine is exhausted.
        self.put(view, "Coventry:0", "Pottery", level=2, flipped=True)
        self.put(view, "Coventry:1", "CoalMine", level=2, seat=(view["seat"] + 1) % 4, resource=1)
        self.connect(view, "RailBirminghamCoventry")
        self.connect(view, "RailBirminghamOxford")
        self.load_view(view)
        self.choose("build", target="Coventry:0", industry="Pottery")
        self.assertEqual(["Coventry:1"], [c["target"] for c in self.a.choices()])
        self.choose("coal", target="Coventry:1")
        self.assertEqual(["market"], [c["target"] for c in self.a.choices()])
        self.choose("coal", target="market")
        self.assertEqual("continuation", self.context["phase"])
        # A vacant single-type iron slot must win over a vacant dual-type slot.
        view = self.view()
        view["pending"].update(kind="", target="", merchant="", parts=0, coal=0, iron=0, beer=0)
        view["context"]["phase"] = "action"
        self.hand(view, ["location:Coalbrookdale"])
        mat = next(m for m in view["players"][view["seat"]]["mat"] if m["type"] == "IronWorks")
        mat["tiles"] = [t for t in mat["tiles"] if t["level"] >= 2]
        self.put(view, "Coalbrookdale:2", "CoalMine", level=2, seat=(view["seat"] + 2) % 4, resource=3)
        self.load_view(view)
        iron_slots = {c["target"] for c in self.a.choices() if c["kind"] == "build" and c["industry"] == "IronWorks"}
        self.assertEqual({"Coalbrookdale:1"}, iron_slots)

    def test_network_second_rail_stop_and_nearest_coal(self):
        view = self.view()
        self.rail(view)
        view["players"][view["seat"]]["cash"] = 40
        self.put(view, "Birmingham:1", "Manufacturer", level=2, flipped=True)
        other = (view["seat"] + 1) % 4
        self.put(view, "Coventry:1", "CoalMine", level=2, seat=other, resource=1)
        self.put(view, "Dudley:0", "CoalMine", level=3, seat=other, resource=3)
        self.put(view, "Central:0", "Brewery", level=2, resource=2)  # own beer anywhere
        self.load_view(view)
        self.choose("network", target="RailBirminghamCoventry")
        self.assertEqual({"Coventry:1"}, {c["target"] for c in self.a.choices()})
        self.choose("coal", target="Coventry:1")
        self.assertIn("finish", {c["kind"] for c in self.a.choices()})
        self.choose("network", target="RailBirminghamDudley")
        self.assertEqual({"coal", "beer"}, {c["kind"] for c in self.a.choices()})
        self.choose("beer", target="Central:0")
        self.choose("coal", target="Dudley:0")
        self.assertEqual(["finish"], [c["kind"] for c in self.a.choices()])

    def test_merchant_pairing_bonus_restore_multisell_and_no_two_merchants(self):
        view = self.view()
        self.put(view, "Worcester:0", "CottonMill")
        self.put(view, "Worcester:1", "CottonMill")
        self.connect(view, "CanalWorcesterGloucester")
        for m in view["merchants"]:
            if m["id"].startswith("Gloucester"):
                m.update(types=["CottonMill"], beer=1)
        self.load_view(view)
        sales = [c for c in self.a.choices() if c["kind"] == "sell"]
        self.assertEqual({"Gloucester:0", "Gloucester:1"}, {c["merchant"] for c in sales})
        self.choose("sell", target="Worcester:0", merchant="Gloucester:0")
        self.assertEqual(["merchant:Gloucester:0"], [c["target"] for c in self.a.choices()])
        self.choose("beer")
        self.assertEqual("bonus", self.context["phase"])
        cp = self.a.call("checkpoint")
        self.b.call("restore", checkpoint=cp)
        self.assertEqual(self.a.choices(), self.b.choices())
        bonus = self.view()
        self.b.reset(90, sample_world_setup(bonus, COMPOSITION, 90))
        self.assertEqual(bonus, self.b.observe(bonus["seat"]))
        self.choose("bonus_develop", industry="Brewery")
        self.assertIn("finish", {c["kind"] for c in self.a.choices()})
        self.choose("sell", target="Worcester:1", merchant="Gloucester:1")
        self.choose("beer")
        self.choose("bonus_skip")
        self.choose("finish")

    def test_rival_resource_overbuild_requires_empty_board_and_market(self):
        view = self.view()
        self.rail(view)
        self.hand(view, ["location:Coventry"])
        view["players"][view["seat"]]["cash"] = 30
        other = (view["seat"] + 1) % 4
        self.put(view, "Coventry:1", "CoalMine", seat=other, resource=0, flipped=True)
        mat = next(m for m in view["players"][view["seat"]]["mat"] if m["type"] == "CoalMine")
        mat["tiles"] = [t for t in mat["tiles"] if t["level"] >= 2]
        view["coal_market"] = 1
        self.load_view(view)
        def overbuilds():
            return [c for c in self.a.choices() if c["kind"] == "build" and c["target"] == "Coventry:1" and c["industry"] == "CoalMine"]
        self.assertEqual([], overbuilds())
        view["coal_market"] = 0
        self.load_view(view)
        self.assertTrue(overbuilds())
        self.choose("build", target="Coventry:1", industry="CoalMine")
        self.choose("finish")

    def test_debt_owner_restore_and_public_world_constructor(self):
        view = self.view()
        debtor = (view["seat"] + 1) % 4
        view["context"].update(turn=4, actions_remaining=1)
        # Keep the last actor solvent; another seat has a genuine liquidation choice.
        view["players"][debtor].update(cash=0, income_marker=0, vp=12)
        self.put(view, "Dudley:0", "CoalMine", seat=debtor, resource=2)
        self.put(view, "Dudley:1", "IronWorks", seat=debtor, resource=4)
        self.load_view(view)
        self.choose("pass")
        self.choose("finish")
        self.assertEqual("debt", self.context["phase"])
        self.assertEqual(debtor, self.context["actor"])
        self.assertEqual(view["seat"], self.context["turn_owner"])
        self.assertEqual({"Dudley:0", "Dudley:1"}, {c["target"] for c in self.a.choices()})
        cp = self.a.call("checkpoint")
        self.b.call("restore", checkpoint=cp)
        self.assertEqual(self.a.choices(), self.b.choices())
        debt_view = self.a.observe(debtor)
        self.b.reset(45, sample_world_setup(debt_view, COMPOSITION, 45))
        self.assertEqual(debt_view, self.b.observe(debtor))
        self.choose("liquidate", target="Dudley:0")
        self.choose("liquidate", target="Dudley:1")
        self.assertEqual("action", self.context["phase"])
        self.assertEqual(2, self.context["round"])
        self.assertLess(self.a.observe(debtor)["players"][debtor]["vp"], 12)

    def test_empty_develop_bonus_and_all_opening_links(self):
        view = self.view()
        self.assertEqual({l["id"] for l in view["links"] if l["era"] == "Canal"},
                         {c["target"] for c in self.a.choices() if c["kind"] == "network"})
        self.put(view, "Worcester:0", "CottonMill")
        self.connect(view, "CanalWorcesterGloucester")
        self.load_view(view)
        self.choose("sell", target="Worcester:0", merchant="Gloucester:1")
        self.choose("beer")
        view = self.view()
        for mat in view["players"][view["seat"]]["mat"]: mat["tiles"] = []
        self.load_view(view)
        self.assertEqual(["bonus_skip"], [c["kind"] for c in self.a.choices()])
        self.choose("bonus_skip")
        self.choose("finish")

    def test_exhaustive_small_decision_tree_five_complete_plans(self):
        view = self.view()
        actor = view["seat"]
        self.hand(view, ["industry:IronWorks"])
        view["players"][actor].update(cash=0, income_marker=0)
        for mat in view["players"][actor]["mat"]:
            mat["tiles"] = mat["tiles"][:2] if mat["type"] == "CottonMill" else []
        self.put(view, "Coalbrookdale:1", "IronWorks", seat=(actor + 1) % 4, resource=1)
        self.put(view, "Dudley:1", "IronWorks", seat=(actor + 2) % 4, resource=1)
        self.load_view(view)
        root = self.a.call("checkpoint")
        self.assertEqual({"develop", "pass"}, {c["kind"] for c in self.a.choices()})
        complete = []
        def walk(checkpoint, path):
            ctx = self.a.call("restore", checkpoint=checkpoint)
            for choice in self.a.choices():
                ctx = self.a.call("restore", checkpoint=checkpoint)
                ctx = self.a.apply(ctx, choice)
                extended = path + [choice["kind"] + ":" + choice.get("target", "")]
                if choice["kind"] == "finish":
                    complete.append(extended)
                else:
                    self.assertLess(len(extended), 7)
                    walk(self.a.call("checkpoint"), extended)
        walk(root, [])
        # Pass; stop after either iron; or take both iron cubes in either order.
        self.assertEqual(5, len(complete))
        self.assertEqual(2, sum(p.count("develop:") == 2 for p in complete))

    def test_wild_cards_and_draws_only_after_turn(self):
        actor = self.context["actor"]
        before = self.view()
        self.choose("scout")
        self.assertEqual(7, len(self.view()["own_hand"]))
        self.assertEqual(before["deck_count"], self.view()["deck_count"])
        self.choose("finish")
        self.assertEqual(8, len(self.a.observe(actor)["own_hand"]))
        while self.context["actor"] != actor:
            self.choose("pass")
            self.choose("finish")
        choices = self.a.choices()
        self.assertNotIn("scout", {c["kind"] for c in choices})
        wild_location = [c for c in choices if c["kind"] == "build" and c["card"] == "wild:location"]
        self.assertTrue(wild_location)
        self.choose("develop", industry="CottonMill", card="wild:industry")
        self.choose("iron")
        self.choose("finish")
        # Round two has two actions; there is no intermediate replacement draw.
        self.assertEqual(actor, self.context["actor"])
        self.assertEqual(7, len(self.view()["own_hand"]))
        self.assertEqual(["wild:location"], self.view()["players"][actor]["known_wild"])

    def test_equally_near_coal_sources_and_market_exclusion(self):
        view = self.view()
        self.rail(view)
        self.hand(view, ["location:Derby"])
        view["players"][view["seat"]]["cash"] = 40
        mat = next(m for m in view["players"][view["seat"]]["mat"] if m["type"] == "CottonMill")
        mat["tiles"] = [t for t in mat["tiles"] if t["level"] >= 2]
        other = (view["seat"] + 1) % 4
        self.put(view, "Belper:1", "CoalMine", level=2, seat=other, resource=3)
        self.put(view, "BurtonOnTrent:0", "CoalMine", level=2, seat=other, resource=3)
        self.connect(view, "RailBelperDerby")
        self.connect(view, "RailDerbyBurtonOnTrent")
        self.connect(view, "RailDerbyNottingham")
        self.load_view(view)
        self.choose("build", target="Derby:0", industry="CottonMill")
        self.assertEqual({"Belper:1", "BurtonOnTrent:0"}, {c["target"] for c in self.a.choices()})
        # Pottery III requires two beers: two merchants cannot substitute for a brewery.
        view = self.view()
        self.rail(view)
        self.put(view, "Coventry:0", "Pottery", level=3)
        self.connect(view, "RailBirminghamCoventry")
        self.connect(view, "RailBirminghamOxford")
        for m in view["merchants"]:
            if m["id"].startswith("Oxford"): m.update(types=["Pottery"], beer=1)
        self.load_view(view)
        self.assertFalse(any(c["kind"] == "sell" and c["target"] == "Coventry:0" for c in self.a.choices()))

    def test_build_coal_and_iron_order_and_iron_source_alternatives(self):
        view = self.view()
        self.rail(view)
        self.hand(view, ["location:Derby"])
        view["players"][view["seat"]]["cash"] = 40
        mat = next(m for m in view["players"][view["seat"]]["mat"] if m["type"] == "CottonMill")
        mat["tiles"] = [t for t in mat["tiles"] if t["level"] >= 3]
        other = (view["seat"] + 1) % 4
        self.put(view, "Belper:1", "CoalMine", level=2, seat=other, resource=3)
        self.put(view, "Dudley:1", "IronWorks", level=2, seat=other, resource=3)
        self.connect(view, "RailBelperDerby")
        self.load_view(view)
        self.choose("build", target="Derby:0", industry="CottonMill")
        self.assertEqual({"coal", "iron"}, {c["kind"] for c in self.a.choices()})
        self.choose("iron", target="Dudley:1")
        self.choose("coal", target="Belper:1")
        self.choose("finish")

    def test_seed_replay_partial_action_and_branch_isolation(self):
        original = self.a.call("checkpoint")
        self.choose("develop")
        partial = self.a.call("checkpoint")
        self.assertEqual(self.context["phase"], "payment")
        restored = self.b.call("restore", checkpoint=partial)
        self.assertEqual(self.a.choices(), self.b.choices())
        before = self.a.call("canonical_hash")
        self.b.apply(restored, self.b.choices()[0])
        self.assertEqual(before, self.a.call("canonical_hash"))
        self.a.call("restore", checkpoint=original)
        self.assertEqual(original["hash"], self.a.call("canonical_hash"))

    def test_invalid_stale_duplicate_and_atomic_action(self):
        before = self.a.call("canonical_hash")
        choice = next(c for c in self.a.choices() if c["kind"] == "develop")
        with self.assertRaisesRegex(HostError, "incomplete_action"):
            self.a.call("apply_action", game_id=self.context["game_id"], revision=0, choices=[choice["id"]])
        self.assertEqual(before, self.a.call("canonical_hash"))
        with self.assertRaisesRegex(HostError, "illegal_choice"):
            self.a.call("apply", game_id=self.context["game_id"], revision=0, choice="forged")
        checkpoint = self.a.call("checkpoint")
        pass_choice = self.choose("pass")
        finish = self.choose("finish")
        with self.assertRaisesRegex(HostError, "stale_revision"):
            self.a.call("apply", game_id=self.context["game_id"], revision=1, choice=finish["id"])
        restored = self.a.call("restore", checkpoint=checkpoint)
        with self.assertRaisesRegex(HostError, "stale_game"):
            self.a.call("apply", game_id=checkpoint["game_id"], revision=0, choice=pass_choice["id"])
        self.context = self.a.call("apply_action", game_id=restored["game_id"], revision=0, choices=[pass_choice["id"], finish["id"]])
        self.assertEqual(2, self.context["revision"])
        old = self.context
        self.a.reset(1201)
        with self.assertRaisesRegex(HostError, "stale_game"):
            self.a.apply(old, pass_choice)

    def test_complete_action_cannot_contain_two_actions(self):
        first = next(c for c in self.a.choices() if c["kind"] == "pass")
        before = self.a.call("canonical_hash")
        with self.assertRaisesRegex(HostError, "more_than_one_action"):
            self.a.call("apply_action", game_id=self.context["game_id"], revision=0,
                        choices=[first["id"], "finish|||||", first["id"]])
        self.assertEqual(before, self.a.call("canonical_hash"))

    def test_four_seat_observation_and_choices_counterfactual(self):
        for _ in range(4):
            actor = self.context["actor"]
            views = [self.a.observe(seat) for seat in range(4)]
            legal = self.a.choices()
            for seat, view in enumerate(views):
                cp_a = sample_world_setup(view, COMPOSITION, 321)
                cp_b = sample_world_setup(view, COMPOSITION, 654)
                self.assertNotEqual(cp_a["cards"]["deck"], cp_b["cards"]["deck"])
                for setup in (cp_a, cp_b):
                    self.b.reset(77, setup)
                    self.assertEqual(view, self.b.observe(seat))
                    if seat == actor:
                        self.assertEqual(legal, self.b.choices())
            self.choose("pass")
            self.choose("finish")

    def test_scout_multisets_and_public_wild_tracking(self):
        before = self.view()
        choice = self.choose("scout")
        after = self.view()
        self.assertEqual(len(before["own_hand"]) - 1, len(after["own_hand"]))
        self.assertEqual(["wild:industry", "wild:location"], after["players"][after["seat"]]["known_wild"])
        self.assertEqual(3, len(choice["cards"]) + 1)
        self.assertEqual(before["deck_count"], after["deck_count"])
        setup = sample_world_setup(after, COMPOSITION, 9)
        self.b.reset(9, setup)
        self.assertEqual(after, self.b.observe(after["seat"]))
        self.choose("finish")

    def test_no_private_fields_and_setup_burns_are_hidden(self):
        view = self.view()
        all_keys = set()
        def walk(value):
            if isinstance(value, dict):
                all_keys.update(value)
                for v in value.values(): walk(v)
            elif isinstance(value, list):
                for v in value: walk(v)
        walk(view)
        self.assertTrue({"seed", "hands", "deck", "UniqueId", "Randomer", "$type", "game_id"}.isdisjoint(all_keys))
        self.assertTrue(all(p["hidden_discards"] == 1 and p["discards"] == [] for p in view["players"]))
        samples = [sample_world_setup(view, COMPOSITION, i) for i in range(5)]
        self.assertGreater(len({s["cards"]["discards"][0][0] for s in samples}), 1)
        for s in samples:
            self.assertEqual(sum(COMPOSITION.values()), len(s["cards"]["deck"]) + sum(len(h) for h in s["cards"]["hands"]) + sum(len(d) for d in s["cards"]["discards"]))

    def test_inconsistent_belief_fails_closed(self):
        view = self.view()
        view["deck_count"] += 1
        with self.assertRaises(ValueError): sample_world_setup(view, COMPOSITION, 0)

    def test_partial_payment_world_reconstruction(self):
        self.choose("develop")
        view = self.view()
        self.b.reset(44, sample_world_setup(view, COMPOSITION, 44))
        self.assertEqual(view, self.b.observe(view["seat"]))
        self.assertEqual(self.a.choices(), self.b.choices())
        ctx = self.b.call("context")
        ctx = self.b.apply(ctx, self.b.choices()[0])
        self.assertEqual("continuation", ctx["phase"])
        cp = self.b.call("checkpoint")
        self.b.call("restore", checkpoint=cp)
        self.assertEqual(cp["hash"], self.b.call("canonical_hash"))


if __name__ == "__main__":
    unittest.main()
