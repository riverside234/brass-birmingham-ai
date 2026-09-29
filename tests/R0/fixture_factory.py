"""Synthetic structural examples only, not game states or legal opening traces."""

from brass_ai.contracts import Contracts, ROOT, canonical_digest, read_json


def examples():
    c = Contracts()
    compat = c.compatibility
    context = {"game_id": "fixture-game", "revision": 7, "decision_id": "decision-7",
               "turn_id": "turn-4", "actor_seat": 0, "turn_owner_seat": 0,
               "phase": "ordinary_action", "actions_remaining": 2}
    players = [{"seat": s, "cash": 17, "income_marker": 10, "income_level": 0,
                "banked_vp": 0, "spent_this_round": 0, "hand_size": 1 if s == 0 else 8,
                "industry_stacks": [], "links_remaining": 14} for s in range(4)]
    observation = {
        "kind": "player_observation", "compatibility": compat, "observer_seat": 0,
        "context": context,
        "public_state": {"era": "canal", "round": 2, "turn_order": [0, 1, 2, 3],
                         "players": players, "industries": [], "links": [], "merchants": [],
                         "coal_market_prices": [], "iron_market_prices": [],
                         "draw_pile_count": 20, "face_down_removed_count": 2,
                         "public_discards": [], "catalog_sha256": "0" * 64},
        "private_state": {"own_hand": [{"card_id": "own-card-0", "kind": "location",
                                        "symbols": ["birmingham"]}], "remembered_events": []},
    }
    observation["observation_id"] = canonical_digest(observation)
    choices = {"kind": "legal_choices", "compatibility": compat, "context": context,
               "observation_id": observation["observation_id"], "choices": [
                   {"choice_id": "action:loan", "choice_kind": "action_type", "target_id": "loan"},
                   {"choice_id": "action:pass", "choice_kind": "action_type", "target_id": "pass"}]}
    action = {"kind": "action_plan", "action_type": "loan", "choice_ids": ["action:loan", "discard:own-card-0"]}
    plan = {"kind": "turn_plan", "actions": [action]}
    value = {"kind": "value_prediction", "compatibility": compat, "observer_seat": 0,
             "observation_id": observation["observation_id"], "seat_ids": [0, 1, 2, 3],
             "win_share": [0.4, 0.35, 0.2, 0.05], "expected_rank": [1.8, 2.2, 2.7, 3.3],
             "expected_final_vp": [130, 125, 119, 104]}
    request = {"kind": "decision_request", "compatibility": compat, "request_id": "request-7",
               "policy_id": "fixture-untrained-policy", "observation": observation, "legal_choices": choices,
               "budget": {"turn_budget_id": "budget-turn-4", "mode": "wall_clock", "total_ms": 3000,
                          "remaining_ms": 2400, "work_limit": None},
               "intent": {"profile_id": "fixture-balanced", "profile_sha256": "0" * 64,
                          "focus": "balanced", "target_ids": []}}
    response = {"kind": "decision_response", "compatibility": compat, "request_id": "request-7",
                "policy_id": "fixture-untrained-policy", "context": context,
                "observation_id": observation["observation_id"], "turn_budget_id": "budget-turn-4",
                "status": "ok", "plan": plan, "elapsed_ms": 20}
    terminal = {"kind": "terminal_result", "compatibility": compat, "game_id": "fixture-game",
                "players": [{"seat": s, "final_vp": [100, 100, 90, 80][s], "income_level": 10,
                             "cash": 20, "rank": [1, 1, 3, 4][s]} for s in range(4)],
                "seat_ids": [0, 1, 2, 3], "win_share": [0.5, 0.5, 0, 0]}
    metadata = {"kind": "model_metadata", "compatibility": compat, "model_id": "fixture-untrained-policy",
                "weights_sha256": "0" * 64,
                "reference_lock_sha256": canonical_digest(read_json(ROOT / "configs/references/upstream.lock.json")),
                "seat_ids": [0, 1, 2, 3], "value_heads": ["win_share", "expected_rank", "expected_final_vp"],
                "parameter_count": 2000000}
    checkpoint = {"kind": "trusted_checkpoint", "visibility": "trusted_host_only", "compatibility": compat,
                  "game_id": "fixture-game", "revision": 7, "checkpoint_id": "fixture-checkpoint",
                  "payload_sha256": "0" * 64, "restore_format": "unimplemented/0.1.0"}
    reset = {"kind": "host_reset", "visibility": "trusted_host_only", "compatibility": compat,
             "game_id": "fixture-game", "setup_id": "fixture-setup", "environment_seed": 19}
    belief = {"kind": "belief_request", "compatibility": compat, "observation": observation,
              "search_seed": 47, "world_count": 8}
    return [observation, choices, action, plan, value, request, response, terminal, metadata, checkpoint, reset, belief]
