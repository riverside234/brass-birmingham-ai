"""Small original semantic prototypes for R0; not MCTS or a Brass opponent."""

from __future__ import annotations

import math


def vector(values):
    if len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                             or not math.isfinite(v) for v in values):
        raise ValueError("expected four finite stable-seat utilities")
    return tuple(float(v) for v in values)


def choose_for_actor(actor, outcomes):
    """Max-n decision: return a child index and its full utility vector.

    Outcome order supplies a deterministic tie-breaker. No negation occurs
    when actors change or when the same actor makes consecutive decisions.
    """
    if type(actor) is not int or actor not in range(4) or not outcomes:
        raise ValueError("expected a decision owner and at least one outcome")
    values = [vector(outcome) for outcome in outcomes]
    index = max(range(len(values)), key=lambda i: values[i][actor])
    return index, values[index]


def chance_expectation(weighted_outcomes):
    """Exact small-fixture expectation, not OpenSpiel's sampled MCTS solver."""
    if not weighted_outcomes:
        raise ValueError("chance outcomes cannot be empty")
    probabilities = [p for p, _ in weighted_outcomes]
    if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("invalid probability")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-10):
        raise ValueError("chance probabilities must sum to one")
    values = [(p, vector(outcome)) for p, outcome in weighted_outcomes]
    return tuple(sum(p * outcome[seat] for p, outcome in values) for seat in range(4))


def backup_path(actors, returns):
    """Project an outcome on each decision edge's owner, as in n-player backup.

    None is a chance edge and inherits its nearest preceding decision owner.
    An initial chance-only prefix has no decision utility and is rejected here;
    its four-vector expectation is handled by chance_expectation instead.
    """
    values = vector(returns)
    owner = None
    result = []
    for actor in actors:
        if actor is not None:
            if type(actor) is not int or actor not in range(4):
                raise ValueError("invalid decision owner")
            owner = actor
        if owner is None:
            raise ValueError("chance prefix has no decision owner")
        result.append(values[owner])
    return tuple(result)


def to_stable_seats(values, encoded_seats):
    values = vector(values)
    if (len(encoded_seats) != 4 or any(type(s) is not int for s in encoded_seats)
            or sorted(encoded_seats) != [0, 1, 2, 3]):
        raise ValueError("seat mapping must be a four-seat permutation")
    stable = [0.0] * 4
    for seat, value in zip(encoded_seats, values):
        stable[seat] = value
    return tuple(stable)
