"""Broad, constraint-correct card beliefs from an allowlisted observation only.

No checkpoint, game seed, opponent hand or real draw deck is accepted. The public
composition is extracted once from a fresh four-player engine by export-catalog.py.
This is not a behavioral inference model or a search policy.
"""
from collections import Counter
from copy import deepcopy
import random


def sample_world_setup(view, composition, search_seed):
    if view["protocol"] != "0.2.0" or len(view["players"]) != 4 or view["context"]["terminal"]:
        raise ValueError("Expected a nonterminal four-player observation")
    rng = random.Random(search_seed)
    own = view["seat"]
    normal = Counter(composition)
    hands = [[] for _ in range(4)]
    hands[own] = list(view["own_hand"])
    discards = [list(p["discards"]) for p in view["players"]]
    for p in view["players"]:
        seat = p["seat"]
        if seat != own:
            hands[seat].extend(p["known_wild"])
        for card in discards[seat] + hands[seat]:
            if not card.startswith("wild:"):
                normal[card] -= 1
                if normal[card] < 0:
                    raise ValueError("Public card constraints are inconsistent")
    pool = sorted(normal.elements())
    rng.shuffle(pool)

    def take(n):
        if n < 0 or n > len(pool):
            raise ValueError("Unseen card conservation failed")
        values = pool[:n]
        del pool[:n]
        return values

    for p in view["players"]:
        s = p["seat"]
        if s != own:
            hands[s] += take(p["hand_count"] - len(hands[s]))
        if len(hands[s]) != p["hand_count"]:
            raise ValueError("Hand count mismatch")
    # Setup burns are sampled too: never remove their real identities from the pool.
    for p in view["players"]:
        discards[p["seat"]] = take(p["hidden_discards"]) + discards[p["seat"]]
    if len(pool) != view["deck_count"]:
        raise ValueError("Deck count mismatch")
    wild_industry = ["wild:industry"] * view["wild_industry_count"]
    wild_location = ["wild:location"] * view["wild_location_count"]
    for kind, available in (("wild:industry", wild_industry), ("wild:location", wild_location)):
        if len(available) + sum(h.count(kind) for h in hands) != 4:
            raise ValueError("Wild-card conservation failed")
    return {
        "cards": {"hands": hands, "discards": discards, "discard_starts": [0] * 4,
                  "deck": pool, "wild_industry": wild_industry, "wild_location": wild_location},
        "order": list(view["order"]),
        "merchants": [{"id": m["id"], "types": m.get("types")} for m in view["merchants"]],
        "position": deepcopy(view),
    }
