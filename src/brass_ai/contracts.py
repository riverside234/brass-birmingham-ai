"""Validate R0 wire shapes and cross-message invariants, without a game engine.

Schema validation is not a proof of legal moves or hidden-information safety.
The R1 host must construct observations and revalidate complete plans.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "configs/contracts/v0.1.0/wire.schema.json"
RULES_PATH = ROOT / "configs/rules/steam-brass-patch-1.1.0.json"


class ContractError(ValueError):
    """An unsupported or inconsistent boundary message."""


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ContractError(f"non-finite JSON constant: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def canonical_digest(value):
    """SHA-256 of sorted, compact, ASCII JSON; rejects NaN/Infinity."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(content).hexdigest()


def _require(condition, explanation):
    if not condition:
        raise ContractError(explanation)


class Contracts:
    def __init__(self, schema_path=SCHEMA_PATH, rules_path=RULES_PATH):
        self.schema = read_json(schema_path)
        self.rules = read_json(rules_path)
        Draft202012Validator.check_schema(self.schema)
        self.validator = Draft202012Validator(self.schema)
        self.reference_lock_sha256 = canonical_digest(read_json(ROOT / "configs/references/upstream.lock.json"))
        self.compatibility = {
            "contract_version": "0.1.0",
            "contract_sha256": canonical_digest(self.schema),
            "rules_profile_id": self.rules["id"],
            "rules_profile_sha256": canonical_digest(self.rules),
            "player_count": 4,
            "action_schema": "phase-choice/0.1.0",
            "feature_schema": "unencoded/0.1.0",
            "objective": "win-share/1.0.0",
        }

    def validate(self, message):
        # jsonschema's numeric type accepts NaN/Infinity in Python objects.
        # Reject these throughout, including nested values, before validation.
        try:
            canonical_digest(message)
        except (ValueError, TypeError) as exc:
            raise ContractError("message must contain finite JSON values") from exc
        error = next(self.validator.iter_errors(message), None)
        if error is not None:
            raise ContractError(f"wire schema mismatch at {list(error.path)}: {error.message}")
        self._semantics(message)
        return message

    def _semantics(self, value):
        if isinstance(value, list):
            for item in value:
                self._semantics(item)
            return
        if not isinstance(value, dict):
            return
        if "compatibility" in value:
            _require(value["compatibility"] == self.compatibility,
                     "contract/rules/features/objective compatibility mismatch")
        kind = value.get("kind")
        if kind == "player_observation":
            players = value["public_state"]["players"]
            _require([p["seat"] for p in players] == [0, 1, 2, 3],
                     "public players must use stable seat order")
            _require(len(value["private_state"]["own_hand"]) ==
                     players[value["observer_seat"]]["hand_size"],
                     "observer hand size mismatch")
            cards = value["private_state"]["own_hand"]
            _require(len({c["card_id"] for c in cards}) == len(cards), "duplicate own card")
            body = {k: v for k, v in value.items() if k != "observation_id"}
            _require(value["observation_id"] == canonical_digest(body),
                     "observation ID must hash the permitted view, not engine state")
        elif kind == "legal_choices":
            ids = [choice["choice_id"] for choice in value["choices"]]
            _require(len(ids) == len(set(ids)), "duplicate legal choice ID")
        elif kind == "value_prediction":
            _require(math.isclose(sum(value["win_share"]), 1.0, abs_tol=1e-8),
                     "four win shares must sum to one")
            _require(math.isclose(sum(value["expected_rank"]), 10.0, abs_tol=1e-8),
                     "expected ranks use averaged tie places and must sum to ten")
        elif kind == "terminal_result":
            players = value["players"]
            _require([p["seat"] for p in players] == [0, 1, 2, 3],
                     "terminal players must use stable seat order")
            keys = [(p["final_vp"], p["income_level"], p["cash"]) for p in players]
            ranks = [1 + sum(other > key for other in keys) for key in keys]
            _require([p["rank"] for p in players] == ranks, "incorrect official rank/tiebreak")
            winners = ranks.count(1)
            expected = [1.0 / winners if rank == 1 else 0.0 for rank in ranks]
            _require(all(math.isclose(a, b, abs_tol=1e-8)
                         for a, b in zip(value["win_share"], expected)), "incorrect tied win shares")
        elif kind == "model_metadata":
            _require(value["reference_lock_sha256"] == self.reference_lock_sha256,
                     "model metadata uses different reference pins")
        elif kind == "decision_request":
            observation, choices = value["observation"], value["legal_choices"]
            _require(observation["observer_seat"] == observation["context"]["actor_seat"],
                     "decision policy must receive its own actor view")
            _require(observation["context"] == choices["context"] and
                     observation["observation_id"] == choices["observation_id"],
                     "legal choices are bound to another observation/context")
            budget = value["budget"]
            _require(budget["remaining_ms"] <= budget["total_ms"], "turn budget increased")
            _require((budget["mode"] == "fixed_work") == (budget["work_limit"] is not None),
                     "fixed_work requires a work limit; wall_clock uses null")
        elif kind == "decision_response":
            _require((value["status"] == "ok") == (value["plan"] is not None),
                     "only an ok response contains a plan")
            if value["plan"] is not None:
                context = value["context"]
                actions = value["plan"]["actions"]
                if context["phase"] == "forced_liquidation":
                    _require(len(actions) == 1 and actions[0]["action_type"] == "liquidate",
                             "liquidation is its own decision, not an ordinary turn plan")
                else:
                    _require(len(actions) <= context["actions_remaining"], "too many turn actions")
                    _require(all(a["action_type"] != "liquidate" for a in actions),
                             "liquidation action outside its phase")
        for item in value.values():
            self._semantics(item)

    def validate_response(self, request, response):
        self.validate(request)
        self.validate(response)
        _require(request["kind"] == "decision_request" and response["kind"] == "decision_response",
                 "expected a decision request/response pair")
        for field in ("request_id", "policy_id", "compatibility"):
            _require(request[field] == response[field], f"response {field} mismatch")
        _require(request["observation"]["context"] == response["context"], "stale response context")
        _require(request["observation"]["observation_id"] == response["observation_id"],
                 "stale response observation")
        _require(request["budget"]["turn_budget_id"] == response["turn_budget_id"],
                 "response belongs to another turn budget")
        if response["status"] == "ok":
            first = response["plan"]["actions"][0]["choice_ids"][0]
            _require(first in {c["choice_id"] for c in request["legal_choices"]["choices"]},
                     "first choice is absent from this request")
        # Further choices are state-dependent. Only the future R1 host can
        # validate them. Actual deadline receipt checks also belong to the host.
        return response

    def validate_continuation(self, previous, current):
        """Check declared budgets across two ordinary decisions of one turn.

        R1 must additionally measure elapsed time using the coordinator's own
        monotonic clock, including IPC and validation. Worker claims are not clocks.
        """
        self.validate(previous)
        self.validate(current)
        _require(previous["kind"] == current["kind"] == "decision_request", "expected decision requests")
        before, after = previous["budget"], current["budget"]
        for field in ("turn_budget_id", "mode", "total_ms"):
            _require(before[field] == after[field], f"continuation changed {field}")
        _require(after["remaining_ms"] <= before["remaining_ms"], "continuation renewed its time budget")
        if before["mode"] == "fixed_work":
            _require(after["work_limit"] <= before["work_limit"], "continuation renewed its work budget")
        earlier, later = previous["observation"]["context"], current["observation"]["context"]
        for field in ("game_id", "turn_id", "actor_seat", "turn_owner_seat"):
            _require(earlier[field] == later[field], f"continuation changed {field}")
        _require(later["revision"] >= earlier["revision"], "continuation has an older revision")
        _require(previous["policy_id"] == current["policy_id"], "policy changed inside the planned turn")
        for field in ("profile_id", "profile_sha256"):
            _require(previous["intent"][field] == current["intent"][field], "profile changed inside the planned turn")
        return current
