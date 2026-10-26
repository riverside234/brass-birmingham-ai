"""G_py: diagnostic full Brass Gumbel driver over the qualified simulator.

Selection equations come from the independently tested small Python oracle.
The simulator is the sealed L09 worker; it has no Gumbel search implementation.
All NN requests use only the acting seat's permitted information. This is a
fixed-count comparator, not the installed caller or a self-play teacher.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gumbel_l10_oracle', ROOT/'scripts/reproduce-gumbel-is-mcts.py')
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)


def canonical(x):
    return json.loads(json.dumps(x, allow_nan=False))


def key(info):
    return json.dumps(info['key'], sort_keys=True, separators=(',', ':'), allow_nan=False)


def information(snapshot, context, ledger):
    actor = snapshot['context']['actor']
    assert actor in range(4) and not snapshot['chance'] and not snapshot['result']
    choices = sorted(snapshot['choices'], key=lambda c: c['id'])
    ctx = dict(context, remaining=ledger.remaining)
    return canonical(dict(actor=actor, actions=[c['id'] for c in choices], key=dict(
        information=snapshot['information'][actor], choices=choices, context=ctx,
        settled=sorted(ledger.settled))))


class Neural:
    def __init__(self, model):
        from brass_ai.learning.features import Encoder
        from brass_ai.native.inference import ReferenceEvaluator
        self.encoder = Encoder()
        self.evaluator = ReferenceEvaluator(model)
        self.replies = []

    def __call__(self, info):
        import numpy as np
        from brass_ai.learning.seat_descriptors import unknown_numpy
        k = info['key']; ctx = k['context']
        assert ctx['stable_descriptors'] is None
        e = self.encoder.encode(k['information']['observation']['view'], k['choices'], ctx['style'], ctx['intent'])
        inputs = dict(states=e.state[None], actions=e.actions[None],
                      legal_mask=np.ones((1, len(e.choice_ids)), bool),
                      stable_to_relative=e.stable_to_relative[None],
                      remaining=np.asarray([ctx['remaining']], np.float32),
                      descriptors=unknown_numpy()[None])
        output = self.evaluator.evaluate(inputs)
        logits = np.asarray(output['policy_logits'][0], np.float32).astype(float).tolist()
        value = np.asarray(output['remaining_utility_4'][0], np.float32).astype(float).tolist()
        self.replies.append(dict(info=copy.deepcopy(info), logits=logits, value=value))
        return logits, value


class Recorded:
    def __init__(self, replies):
        self.replies = iter(replies)
    def __call__(self, info):
        row = next(self.replies)
        assert row['info'] == info
        return row['logits'], row['value']
    def finish(self):
        assert next(self.replies, None) is None


def search(worker, input, config, evaluator, tape=None):
    from brass_ai.learning.reward import CostLedger, terminal_utility
    call = lambda kind, **kw: worker.simulation(0, kind, **kw)
    prepared = call('prepare', view=input['view'], recall=input['recall'])
    base_ledger = CostLedger(input['reward'], **input['ledger'])
    base_ledger.observe(input['view'])
    assert base_ledger.remaining == input['context']['remaining']
    initial = call('sample', random=dict(kind='recorded', indices=[0]*prepared['draw_count']))
    root_info = information(initial['snapshot'], input['context'], base_ledger)
    call('release', handle=initial['handle'])
    assert root_info['key']['information']['observation']['view'] == input['view']
    draws = ref.Draws(config['seed'], tape)
    nodes, infos, lookup, trajectories = [], [], {}, []
    def expand(info):
        logits, value = evaluator(info)
        n = ref.Node(ref.Info(info['actor'], (), (), tuple(info['actions'])), logits, value, 4)
        lookup[key(info)] = len(nodes); nodes.append(n); infos.append(info)
        return n
    root = expand(root_info)
    budget = config['simulations']; size = ref.effective_k(config['k'], len(root_info['actions']), budget)
    gumbels = [-math.log(-math.log(draws.uniform())) for _ in root_info['actions']]
    survivors = sorted(sorted(range(len(gumbels)), key=lambda i: (-(gumbels[i]+root.logits[i]), i))[:size])
    samples = iter(input['shuffle_tape'])
    def simulate(root_edge):
        sampled = call('sample', random=dict(kind='recorded', indices=next(samples)))
        handle, snapshot = sampled['handle'], sampled['snapshot']
        ledger = CostLedger(input['reward'], **base_ledger.snapshot())
        assert information(snapshot, input['context'], ledger) == root_info
        path, costs, chances = [], [], []
        try:
            for _ in range(config['max_depth']):
                if not snapshot['chance']:
                    ledger.observe(snapshot['information'][input['view']['seat']]['observation']['view'])
                if snapshot['result']:
                    value = terminal_utility(snapshot['result'], input['reward']).tolist(); leaf = 'terminal'; break
                if snapshot['chance']:
                    outcomes = sorted(('chance:card:'+c['card'], c['probability']) for c in snapshot['chance']['outcomes'])
                    action = draws.categorical(outcomes); chances.append(action)
                else:
                    info = information(snapshot, input['context'], ledger)
                    identity = key(info)
                    if identity not in lookup:
                        value = list(expand(info).value); leaf = len(nodes)-1; break
                    node_id = lookup[identity]; node = nodes[node_id]
                    assert infos[node_id] == info
                    edge = root_edge if node_id == 0 else node.select(config['c1'], config['c2'])
                    path.append((node_id, edge)); costs.append(ledger.spent.copy())
                    action = info['actions'][edge]
                    ledger.chosen(info['key']['information']['observation']['view'], info['key']['choices'][edge])
                snapshot = call('step', handle=handle, revision=snapshot['context']['revision'], semantic=action)['snapshot']
            else:
                raise ValueError('trajectory exceeded diagnostic depth guard')
            assert len(value) == 4 and all(math.isfinite(v) for v in value)
            edge_values = []
            for (node_id, edge), spent in zip(path, costs):
                future = [value[s]-(ledger.spent[s]-spent[s]) for s in range(4)]
                node = nodes[node_id]; node.count[edge] += 1
                for s in range(4): node.returns[edge][s] += future[s]
                edge_values.append(future)
            trajectories.append(dict(path=path, leaf=leaf, value=value, chance=chances,
                                     edge_values=edge_values, leaf_spent=ledger.spent.tolist()))
        finally:
            call('release', handle=handle)
    rounds = []; levels = size.bit_length()-1
    while len(survivors) > 1:
        before = survivors[:]; quota = budget//(len(before)*levels)
        for _ in range(quota):
            for edge in before: simulate(edge)
        scores = [g+s for g,s in zip(gumbels, root.scores(config['c1'], config['c2']))]
        survivors = sorted(sorted(before, key=lambda i: (-scores[i], i))[:len(before)//2])
        rounds.append(dict(candidates=before, quota_per_action=quota, scores=[scores[i] for i in before], retained=survivors[:]))
    remainder = budget-len(trajectories)
    for _ in range(remainder): simulate(survivors[0])
    assert next(samples, None) is None
    draws.finish()
    return canonical(dict(version='brass-gumbel-is-mcts/count-v1', config=config,
        selected_action=root_info['actions'][survivors[0]], status='prior_fallback' if budget==0 else 'completed',
        effective_k=size, gumbels=gumbels, rounds=rounds, remainder_simulations=remainder,
        trajectories=trajectories, draws=draws.used, nodes=[dict(info=info, logits=n.logits, value=n.value,
        prior=n.prior, count=n.count, returns=n.returns, q=n.completed_q()[0], v_mix=n.completed_q()[1],
        improved_policy=n.policy(config['c1'], config['c2'])) for info,n in zip(infos,nodes)]))
