#!/usr/bin/env python3
"""Small, synchronous Gumbel IS-MCTS oracle; Python standard library only.

Paper: arXiv:2405.00243v1, Alg. 1/3, Appendix B, Table 7.
See ai-research/GUMBEL_IS_MCTS_REFERENCE.md for scope and explicit conventions.
This finite hidden-type game is not Brass, a neural model, or a timing benchmark.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

VERSION = "gumbel-is-mcts-reference/1"


class Draws:
    """Replay uniform outcomes, not language-specific seeds, when porting to Rust."""

    def __init__(self, seed, tape=None):
        self.rng = random.Random(seed)
        self.tape = None if tape is None else iter(tape)
        self.used = []

    def uniform(self):
        if self.tape is None:
            # Midpoints of 2**52 bins: strictly inside (0, 1), even after rounding.
            value = (self.rng.getrandbits(52) + 0.5) / 2**52
        else:
            try:
                value = next(self.tape)
            except StopIteration as error:
                raise ValueError("replay tape exhausted") from error
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("uniform draw must be finite and strictly inside (0, 1)")
        self.used.append(value)
        return value

    def categorical(self, outcomes):
        if not outcomes or any(not math.isfinite(p) or p < 0 for _, p in outcomes):
            raise ValueError("invalid chance probabilities")
        total = sum(p for _, p in outcomes)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("empty chance mass")
        target, cumulative = self.uniform() * total, 0.0
        for outcome, probability in outcomes:
            cumulative += probability
            if target < cumulative:
                return outcome
        return next(outcome for outcome, p in reversed(outcomes) if p > 0)

    def finish(self):
        sentinel = object()
        if self.tape is not None and next(self.tape, sentinel) is not sentinel:
            raise ValueError("unused replay draws")


def softmax(logits):
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("legal logits must be finite and nonempty")
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


@dataclass(frozen=True)
class Info:
    player: int
    private: tuple
    public: tuple
    actions: tuple

    @property
    def key(self):
        return self.player, self.private, self.public


class Node:
    def __init__(self, info, logits, value, players):
        if (not 0 <= info.player < players or not info.actions
                or tuple(sorted(set(info.actions))) != info.actions
                or len(logits) != len(info.actions)
                or len(value) != players or not all(math.isfinite(v) for v in value)):
            raise ValueError("invalid information set or evaluator response")
        self.info, self.logits, self.value = info, tuple(logits), tuple(value)
        self.prior = softmax(logits)
        self.count = [0] * len(logits)
        self.returns = [[0.0] * players for _ in logits]

    def completed_q(self):
        visited = [i for i, n in enumerate(self.count) if n]
        mixed = self.value[self.info.player]
        if visited:
            # Normalize logits over visited edges directly; avoids 0/0 if their
            # full-legal softmax probabilities underflow, without adding a floor.
            weights = softmax([self.logits[i] for i in visited])
            mean = sum(w * self.returns[i][self.info.player] / self.count[i]
                       for w, i in zip(weights, visited))
            total = sum(self.count)
            mixed = (mixed + total * mean) / (1 + total)
        q = [self.returns[i][self.info.player] / n if n else mixed
             for i, n in enumerate(self.count)]
        return q, mixed

    def scores(self, c1, c2):
        q, _ = self.completed_q()
        scale = c2 * (c1 + max(self.count))
        return [logit + scale * v for logit, v in zip(self.logits, q)]

    def policy(self, c1, c2):
        return softmax(self.scores(c1, c2))

    def select(self, c1, c2):
        # Exact argmin of paper Eq. (3), after removing action-independent terms.
        policy, denominator = self.policy(c1, c2), 1 + sum(self.count)
        return max(range(len(policy)), key=lambda i: policy[i] - self.count[i] / denominator)


def effective_k(requested, legal_count, budget):
    if requested < 1 or legal_count < 1 or budget < 0:
        raise ValueError("K/legal count must be positive; budget must be nonnegative")
    k = 1 << (min(requested, legal_count).bit_length() - 1)
    # Paper's power-of-two schedule needs positive quotas in every round.
    while k > 1 and budget < k * (k.bit_length() - 1):
        k //= 2
    return k


def search(game, *, simulations=200, k=16, seed=7, c1=50.0, c2=0.1, tape=None):
    if any(not math.isfinite(x) or x <= 0 for x in (c1, c2)):
        raise ValueError("c1 and c2 must be positive and finite")
    root_info = game.root_info()
    size = effective_k(k, len(root_info.actions), simulations)
    draws, nodes, lookup, trajectories = Draws(seed, tape), [], {}, []

    def expand(info):
        # Evaluator sees only this actor's permitted information, never a world.
        logits, value = game.evaluate(info)
        node = Node(info, logits, value, game.players)
        lookup[info.key] = len(nodes)
        nodes.append(node)
        return node

    root = expand(root_info)  # Root preparation is separate from trajectory count.
    gumbels = [-math.log(-math.log(draws.uniform())) for _ in root_info.actions]
    survivors = sorted(sorted(range(len(gumbels)),
                              key=lambda i: (-(gumbels[i] + root.logits[i]), i))[:size])

    def simulate(root_edge):
        world = game.sample_root(draws)  # Fresh posterior draw EVERY trajectory.
        if game.information(world) != root_info:
            raise ValueError("sampled world does not match permitted root")
        path, chances = [], []
        for _ in range(256):  # Diagnostic guard, not a value/bootstrap horizon.
            value = game.terminal_returns(world)
            if value is not None:
                leaf = "terminal"
                break
            outcomes = game.chance_outcomes(world)
            if outcomes is not None:
                action = draws.categorical(outcomes)
                chances.append(action)
            else:
                info = game.information(world)
                if info.key not in lookup:
                    value = expand(info).value
                    leaf = len(nodes) - 1
                    break
                node_id = lookup[info.key]
                node = nodes[node_id]
                if info != node.info:
                    raise ValueError("information key collision or inconsistent legal choices")
                edge = root_edge if node_id == 0 else node.select(c1, c2)
                path.append((node_id, edge))
                action = info.actions[edge]
            world = game.step(world, action)
        else:
            raise ValueError("trajectory exceeded diagnostic depth guard")
        if len(value) != game.players or not all(math.isfinite(v) for v in value):
            raise ValueError("invalid leaf returns")
        # Publish once, after successful completion. All vectors use absolute seats.
        for node_id, edge in path:
            nodes[node_id].count[edge] += 1
            for player, utility in enumerate(value):
                nodes[node_id].returns[edge][player] += utility
        trajectories.append({"path": path, "leaf": leaf, "value": value, "chance": chances})

    rounds = []
    levels = size.bit_length() - 1
    while len(survivors) > 1:
        quota = simulations // (len(survivors) * levels)
        before = survivors[:]
        for _ in range(quota):
            for edge in before:
                simulate(edge)
        # Eliminate AFTER the last completed backup, never at selection time.
        scores = [g + s for g, s in zip(gumbels, root.scores(c1, c2))]
        survivors = sorted(sorted(before, key=lambda i: (-scores[i], i))[:len(before) // 2])
        rounds.append({"candidates": before, "quota_per_action": quota,
                       "scores": [scores[i] for i in before], "retained": survivors[:]})
    remainder = simulations - len(trajectories)
    for _ in range(remainder):
        simulate(survivors[0])  # Floor remainder cannot change the surviving action.
    draws.finish()
    return {"version": VERSION, "config": {"simulations": simulations, "k": k,
            "seed": seed, "c1": c1, "c2": c2}, "effective_k": size,
            "selected_action": root_info.actions[survivors[0]],
            "status": "prior_fallback" if simulations == 0 else "completed",
            "rounds": rounds, "remainder_simulations": remainder, "gumbels": gumbels,
            "nodes": [{"info": {"player": n.info.player, "private": n.info.private,
                                 "public": n.info.public, "actions": n.info.actions},
                       "logits": n.logits, "value": n.value, "prior": n.prior,
                       "count": n.count, "returns": n.returns,
                       "q": n.completed_q()[0], "v_mix": n.completed_q()[1],
                       "improved_policy": n.policy(c1, c2)} for n in nodes],
            "trajectories": trajectories, "draws": draws.used}


@dataclass(frozen=True)
class World:
    types: tuple
    moves: tuple = ()
    shock: int | None = None


class HiddenOffer:
    """Finite general-sum toy: offer, private-type replies, then public chance.

    A public 'high' signal has likelihood .8/.2 for opponent 1's types 1/0;
    with a uniform prior, P(type_1=1 | high)=.8. Other types stay uniform.
    This exact toy posterior is deliberately simpler than a Brass belief model.
    """

    def __init__(self, players=4):
        if players not in (2, 4):
            raise ValueError("toy supports two or four players")
        self.players = players

    def root_info(self):
        return Info(0, (0,), (1,), tuple(range(16)))

    def sample_root(self, draws):
        return World((0,) + tuple(int(draws.uniform() < (.8 if i == 1 else .5))
                                 for i in range(1, self.players)))

    def information(self, world):
        actor = len(world.moves)
        return Info(actor, (world.types[actor],), (1,) + world.moves,
                    tuple(range(16)) if actor == 0 else (0, 1))

    def chance_outcomes(self, world):
        return ((-1, .25), (1, .75)) if len(world.moves) == self.players and world.shock is None else None

    def step(self, world, action):
        if len(world.moves) == self.players:
            return World(world.types, world.moves, action)
        return World(world.types, world.moves + (action,))

    def terminal_returns(self, world):
        if world.shock is None:
            return None
        price = (world.moves[0] + 1) / 17
        replies = world.moves[1:]
        root = sum(price + .02 * world.shock if a else -.05 for a in replies) / len(replies)
        others = tuple(.4 + .5 * world.types[i] - price + .01 * world.shock if a else .01
                       for i, a in enumerate(replies, 1))
        return (root,) + others

    def evaluate(self, info):
        if info.player == 0:
            logits = tuple(-abs(a - 7) / 4 for a in info.actions)
        else:
            price = (info.public[1] + 1) / 17
            logits = (0.0, 5 * (.4 + .5 * info.private[0] - price))
        # A deterministic stand-in, not a trained/exact critic. No hidden world.
        return logits, (0.0,) * self.players


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay", type=Path, help="replay and verify a saved JSON trace")
    args = parser.parse_args()
    previous = json.loads(args.replay.read_text(encoding="utf-8")) if args.replay else None
    config = previous["config"] if previous else {"simulations": args.simulations, "k": args.k, "seed": args.seed}
    players = previous["players"] if previous else args.players
    result = search(HiddenOffer(players), **config, tape=previous["draws"] if previous else None)
    result["players"] = players
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if previous is not None and json.loads(encoded) != previous:
        raise ValueError("replay differs from recorded trace")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(json.dumps({"version": VERSION, "action": result["selected_action"],
                      "completed": len(result["trajectories"]), "effective_k": result["effective_k"],
                      "round_work": [len(r["candidates"]) * r["quota_per_action"] for r in result["rounds"]],
                      "remainder": result["remainder_simulations"], "replayed": previous is not None}))


if __name__ == "__main__":
    main()
