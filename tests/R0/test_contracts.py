from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from brass_ai.contracts import ContractError, Contracts, ROOT, canonical_digest, read_json
from fixture_factory import examples


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = Contracts()
        cls.messages = {m["kind"]: m for m in read_json(ROOT / "tests/R0/fixtures/messages.json")}

    def sample(self, kind):
        return deepcopy(self.messages[kind])

    def test_all_twelve_examples_validate(self):
        self.assertEqual(len(self.messages), 12)
        for message in self.messages.values():
            self.contracts.validate(message)

    def test_checked_in_examples_match_factory(self):
        self.assertEqual(list(self.messages.values()), examples())

    def test_normal_request_response_pair(self):
        self.contracts.validate_response(self.sample("decision_request"), self.sample("decision_response"))

    def test_reject_three_players_and_incompatible_versions(self):
        for field, value in [("player_count", 3), ("contract_version", "0.2.0"),
                             ("rules_profile_id", "stock"), ("contract_sha256", "f" * 64),
                             ("rules_profile_sha256", "f" * 64), ("objective", "team-reward")]:
            message = self.sample("value_prediction")
            message["compatibility"][field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.contracts.validate(message)

    def test_scalar_or_short_values_are_rejected(self):
        for value in (0.4, [0.4, 0.6], [0.4, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1, 0]):
            message = self.sample("value_prediction")
            message["win_share"] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                self.contracts.validate(message)

    def test_nonfinite_values_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            message = self.sample("value_prediction")
            message["expected_final_vp"][0] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                self.contracts.validate(message)

    def test_value_scales_are_checked(self):
        for field, value in [("win_share", [0.4, 0.4, 0.4, 0.4]),
                             ("expected_rank", [1, 1, 1, 1])]:
            message = self.sample("value_prediction")
            message[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.contracts.validate(message)

    def test_stable_seat_order_required(self):
        message = self.sample("value_prediction")
        message["seat_ids"] = [3, 2, 1, 0]
        with self.assertRaises(ContractError):
            self.contracts.validate(message)
        observation = self.sample("player_observation")
        observation["public_state"]["players"].reverse()
        with self.assertRaises(ContractError):
            self.contracts.validate(observation)

    def test_explicit_hidden_fields_cannot_enter_observation(self):
        for location, field in [("public_state", "opponent_hands"),
                                ("public_state", "deck_order"),
                                ("private_state", "engine_checkpoint"),
                                ("private_state", "environment_seed")]:
            message = self.sample("player_observation")
            message[location][field] = []
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.contracts.validate(message)

    def test_trusted_checkpoint_is_not_a_player_view(self):
        request = self.sample("decision_request")
        request["observation"] = self.sample("trusted_checkpoint")
        with self.assertRaises(ContractError):
            self.contracts.validate(request)

    def test_hash_must_describe_permitted_view(self):
        message = self.sample("player_observation")
        message["public_state"]["players"][0]["cash"] += 1
        with self.assertRaises(ContractError):
            self.contracts.validate(message)

    def test_duplicate_cards_and_choices_rejected(self):
        message = self.sample("legal_choices")
        message["choices"].append(deepcopy(message["choices"][0]))
        with self.assertRaises(ContractError):
            self.contracts.validate(message)
        message = self.sample("player_observation")
        message["private_state"]["own_hand"] *= 2
        message["public_state"]["players"][0]["hand_size"] = 2
        with self.assertRaises(ContractError):
            self.contracts.validate(message)

    def test_actor_must_receive_own_view(self):
        request = self.sample("decision_request")
        request["observation"]["observer_seat"] = 1
        with self.assertRaises(ContractError):
            self.contracts.validate(request)

    def test_legal_choices_bound_to_the_observation(self):
        request = self.sample("decision_request")
        request["legal_choices"]["context"]["revision"] += 1
        with self.assertRaises(ContractError):
            self.contracts.validate(request)

    def test_stale_responses_and_unknown_first_choices_rejected(self):
        request = self.sample("decision_request")
        for field in ("request_id", "policy_id", "turn_budget_id", "observation_id"):
            response = self.sample("decision_response")
            response[field] = "f" * 64
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.contracts.validate_response(request, response)
        response = self.sample("decision_response")
        response["context"]["revision"] -= 1
        with self.assertRaises(ContractError):
            self.contracts.validate_response(request, response)
        response = self.sample("decision_response")
        response["plan"]["actions"][0]["choice_ids"][0] = "not-offered"
        with self.assertRaises(ContractError):
            self.contracts.validate_response(request, response)

    def test_one_action_boundary_and_liquidation(self):
        response = self.sample("decision_response")
        response["context"]["actions_remaining"] = 1
        response["plan"]["actions"] *= 2
        with self.assertRaises(ContractError):
            self.contracts.validate(response)
        response = self.sample("decision_response")
        response["context"].update(phase="forced_liquidation", actions_remaining=0, actor_seat=2)
        response["plan"]["actions"][0].update(action_type="liquidate", choice_ids=["liquidate:industry-2"])
        self.contracts.validate(response)

    def test_cancelled_response_has_no_plan(self):
        response = self.sample("decision_response")
        response["status"] = "cancelled"
        with self.assertRaises(ContractError):
            self.contracts.validate(response)
        response["plan"] = None
        self.contracts.validate(response)

    def test_turn_budget_cannot_be_renewed(self):
        earlier = self.sample("decision_request")
        later = deepcopy(earlier)
        later["budget"]["remaining_ms"] = 1000
        self.contracts.validate_continuation(earlier, later)
        later["budget"]["remaining_ms"] = 2500
        with self.assertRaises(ContractError):
            self.contracts.validate_continuation(earlier, later)
        later = deepcopy(earlier)
        later["budget"]["turn_budget_id"] = "renewed"
        with self.assertRaises(ContractError):
            self.contracts.validate_continuation(earlier, later)

    def test_fixed_work_has_one_remaining_limit(self):
        earlier = self.sample("decision_request")
        earlier["budget"].update(mode="fixed_work", work_limit=100)
        later = deepcopy(earlier)
        later["budget"]["work_limit"] = 50
        self.contracts.validate_continuation(earlier, later)
        later["budget"]["work_limit"] = 101
        with self.assertRaises(ContractError):
            self.contracts.validate_continuation(earlier, later)

    def test_terminal_ties_split_wins_and_preserve_competition_ranks(self):
        result = self.sample("terminal_result")
        self.contracts.validate(result)
        result["win_share"] = [1, 0, 0, 0]
        with self.assertRaises(ContractError):
            self.contracts.validate(result)
        result = self.sample("terminal_result")
        for player in result["players"]:
            player.update(final_vp=100, rank=1)
        result["win_share"] = [0.25] * 4
        self.contracts.validate(result)

    def test_income_then_cash_tiebreaks(self):
        result = self.sample("terminal_result")
        result["players"][1]["income_level"] = 11
        result["players"][0]["rank"] = 2
        result["win_share"] = [0, 1, 0, 0]
        self.contracts.validate(result)
        result["players"][1]["income_level"] = 10
        result["players"][1]["cash"] = 21
        self.contracts.validate(result)

    def test_model_cannot_accept_scalar_heads_or_different_reference_pins(self):
        for field, value in [("value_heads", ["shared_win"]), ("reference_lock_sha256", "f" * 64)]:
            message = self.sample("model_metadata")
            message[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.contracts.validate(message)

    def test_json_reader_rejects_duplicate_keys_and_nonfinite_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "message.json"
            for text in ('{"seat":0,"seat":1}', '{"value":NaN}', '{"value":Infinity}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ContractError):
                    read_json(path)


if __name__ == "__main__":
    unittest.main()
