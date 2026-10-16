"""Isolated W352x6/W448x6 deployment timing, including legal late-game roots.

Initialized weights only; never admits experimental configs into production.
Temporary ONNX files can be removed after preserving measurements and hashes.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
from dataclasses import replace
import json
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch
from torch import nn

from benchmark_deploy_capacity import (BASE_INPUTS, BenchmarkService, Named,
    build_models, collate, compare, context, context_vector, encoder_probe,
    export_model, run_turn, sha, summary, write)
from benchmark_deploy_transport import Probe, active_session, cleanup
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT, require
from brass_ai.learning.model import ResidualBlock
from brass_ai.personality.contracts import CONTEXT_DIM
from brass_ai.personality.model import PersonalityConfig
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.host_runtime import training_host
from brass_ai.searchlab.spiel import BrassSpielGame
from fe_fixtures import benchmark_positions
from profile_grouped import process_memory


PINNED = ('src/brass_ai/learning/model.py',
    'src/brass_ai/learning/utility_model.py',
    'src/brass_ai/personality/model.py',
    'src/brass_ai/personality/search/engine.py',
    'src/brass_ai/personality/laptop/inference.py',
    'src/brass_ai/recovery/encoding.py',
    'src/brass_ai/search/ismcts.py',
    'src/BrassTrainingHost/FastHost.cs',
    'training/runtime/throughput/search/BrassHost.exe',
    'tests/FeatureEngineering/benchmark_deploy_capacity.py',
    'tests/FeatureEngineering/benchmark_deploy_transport.py')


def models():
    core, actor = build_models()['w352']
    torch.manual_seed(610448)
    wider = copy.deepcopy(core)
    wider.config = replace(core.config, hidden=448,
        architecture='benchmark-only/w448-six-blocks/1')
    wider.state_encoder.fusion = nn.Sequential(nn.Linear(748, 448), nn.GELU())
    wider.state_encoder.backbone = nn.Sequential(
        *[ResidualBlock(448) for _ in range(6)], nn.LayerNorm(448))
    wider.policy_scorer = nn.Sequential(nn.Linear(448+128, 128), nn.GELU(), nn.Linear(128, 1))
    wider.outcomes = nn.Sequential(nn.Linear(448, 128), nn.GELU(), nn.Linear(128, 12))
    wider.add_utility_head()
    bigger_actor = copy.deepcopy(actor)
    bigger_actor.core = copy.deepcopy(wider)
    bigger_actor.config = PersonalityConfig(wider.config.to_dict())
    bigger_actor.context_projection = nn.Linear(CONTEXT_DIM, 448, bias=False)
    nn.init.zeros_(bigger_actor.context_projection.weight)
    bigger_actor.goal_trunk = nn.Sequential(nn.Linear(448+CONTEXT_DIM, 128), nn.GELU())
    for a, b in zip(core.state_encoder.projections.parameters(), wider.state_encoder.projections.parameters()):
        require(torch.equal(a, b), 'Shared projection weights changed')
    for a, b in zip(core.action_encoder.parameters(), wider.action_encoder.parameters()):
        require(torch.equal(a, b), 'Shared action encoder changed')
    require(wider.parameter_count == 3671409, 'W448 parameter count differs')
    require(bigger_actor.parameter_count == 3762154, 'W448 actor parameter count differs')
    for model in (wider, bigger_actor):
        try:
            model.config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError('Experimental model was admitted as production')
    return {'w352': (core.eval(), actor.eval()), 'w448': (wider.eval(), bigger_actor.eval())}


def legal_late_positions(out):
    """Fixed cheap family-weighted policy; choose roots before comparing models.

    These are actual legal game states, not authored round-number changes, and
    do not purport to represent strong trained play. No search is used here.
    """
    seed = 914448
    rng = random.Random(seed)
    targets = {(era, round_) for era in ('CanalEra', 'RailEra') for round_ in (6, 8)}
    selected, trace = {}, []
    with RulesHost(training_host(SOURCE_ROOT)) as host:
        host.reset(seed)
        require(host.enable_fastpath(), 'Expected optimized ordinary host')
        decision = host.decision()
        for step in range(1800):
            c = decision['context']
            if c['terminal']:
                break
            view, choices = decision['view'], decision['choices']
            key = (c['era'], c['round'])
            tiles = sum(s.get('tile') is not None for s in view['slots'])
            # Prespecified stress selection: largest legal branching at the
            # start of a full turn in each target round, without model output.
            if key in targets and c['phase'] == 'action' and c['actions_remaining'] == 2:
                if key not in selected or len(choices) > len(selected[key]['choices']):
                    selected[key] = dict(name=f'{key[0]}-round{key[1]}-legal-stress',
                        view=copy.deepcopy(view), choices=copy.deepcopy(choices),
                        origin=dict(seed=seed, step=step, tiles=tiles,
                            links=sum(link.get('owner') is not None for link in view['links']),
                            policy='Fixed family-weighted legal fixture generator; untrained'))
            families = defaultdict(list)
            for choice in choices:
                families[choice['kind']].append(choice)
            cash = view['players'][view['seat']]['cash']
            weights = dict(build=6., sell=9., network=3., develop=1.,
                loan=9. if cash < 15 else .15, scout=.15, **{'pass': .02})
            keys = sorted(families)
            family = rng.choices(keys, weights=[weights.get(k, 1.) for k in keys])[0]
            choice = rng.choice(families[family])
            trace.append(dict(step=step, context=c, choice=choice['id']))
            decision = host.step_decision(c, choice, forced_limit=0)['decision']
        else:
            raise AssertionError('Fixture game exceeded bounded decision cap')
    require(set(selected) == targets, f'Missing late roots: {targets-set(selected)}')
    require(all(r['origin']['tiles'] >= 8 for r in selected.values()), 'Late fixture lacks board occupancy')
    write(out/'fixture-game.json', dict(seed=seed, completed=True, steps=len(trace), trace=trace,
        selection='Maximum root candidates at a full-turn Action in Canal/Rail rounds 6 and 8; fixed before timing.'))
    return [selected[k] for k in sorted(targets)]


def prepare(out):
    require(not out.exists(), 'Use a fresh isolated output directory')
    (out/'models').mkdir(parents=True)
    rows = [r for r in benchmark_positions() if r['name'] in ('settled-next-round', 'rail-action')]
    rows += legal_late_positions(out)
    write(out/'inputs.json', rows)
    game = BrassSpielGame(root=SOURCE_ROOT, fast=True)
    try:
        for i, row in enumerate(rows):
            state = game.from_observation(row['view'], 915000+i)
            require(state.legal == row['choices'], 'Reconstructed legal choices differ')
            print(f"Fixture {row['name']}: {len(row['choices'])} choices, "
                f"{sum(s.get('tile') is not None for s in row['view']['slots'])} tiles", flush=True)
    finally:
        game.close()
    report = dict(format='isolated-w352-w448-deployment/1',
        scope='Initialized weights; throughput/coverage only, not playing strength.',
        platform=platform.platform(), cpu=platform.processor(), python=platform.python_version(),
        torch=torch.__version__, seconds_per_turn=12., tensor_threads=2,
        leaf_batch=1, service_batch_cap='worker count', flush_ms=1.,
        batch_selection='Retain previous measured B1 pending-path winner; microcheck B1/B2/B3 without retuning search.',
        inputs_sha256=sha(out/'inputs.json'), sources={p: sha(SOURCE_ROOT/p) for p in PINNED},
        models={}, micro=[], encoder=encoder_probe(rows))
    model_set = models()
    graphs = {}
    for name, pair in model_set.items():
        report['models'][name] = {}
        for head, model in zip(('utility', 'actor'), pair):
            meta, graph = export_model(model, out/'models'/f'{name}-{head}.onnx', head == 'actor')
            report['models'][name][head] = meta
            graphs[(name, head)] = graph
        print(f'Prepared {name}: utility={pair[0].parameter_count}, actor={pair[1].parameter_count}; export parity passed', flush=True)
        write(out/'prepared.json', report)
    encoder = PreparedEncoder()
    for row in rows:
        encoded = encoder.encode(row['view'], row['choices'])
        for batch_size in (1, 2, 3):
            inputs = dict(zip(BASE_INPUTS[:4], [t.numpy() for t in collate([encoded]*batch_size)]))
            inputs['remaining'] = np.full((batch_size, 4), .075, np.float32)
            for head_index, head in enumerate(('utility', 'actor')):
                feed = dict(inputs)
                if head == 'actor':
                    feed['personality'] = np.stack([context_vector(context())]*batch_size).astype('float32')
                measured = {name: [] for name in model_set}
                for name in model_set:
                    graph = graphs[(name, head)]
                    with torch.inference_mode():
                        actual = graph.run(feed)
                        expected = Named(model_set[name][head_index])(*(torch.from_numpy(feed[k]) for k in graph.inputs))
                        compare({k: v.numpy() for k, v in zip(graph.outputs, expected)}, actual, feed['legal_mask'])
                    one = graph.run({k: v[:1] for k, v in feed.items()})
                    compare({k: v[:1] for k, v in actual.items()}, one, feed['legal_mask'][:1])
                    for _ in range(5):
                        graph.run(feed)
                for repeat in range(30):
                    for name in (('w352', 'w448') if repeat % 2 == 0 else ('w448', 'w352')):
                        began = time.perf_counter()
                        graphs[(name, head)].run(feed)
                        measured[name].append(1000*(time.perf_counter()-began))
                report['micro'].append(dict(case=row['name'], head=head, batch=batch_size,
                    candidates=len(row['choices']), parity=True,
                    models={k: summary(v) for k, v in measured.items()}))
    report['memory'] = process_memory()
    report['fixture_and_real_input_parity_passed'] = True
    write(out/'prepared.json', report)
    print('Preparation complete: six roots; fixed-input inference/batch and encoder parity passed', flush=True)


def search(out):
    require(not (out/'search.json').exists(), 'Do not overwrite completed evidence')
    prepared = json.loads((out/'prepared.json').read_text(encoding='utf-8'))
    require(sha(out/'inputs.json') == prepared['inputs_sha256'], 'Fixture corpus changed')
    require(all(sha(SOURCE_ROOT/p) == h for p, h in prepared['sources'].items()), 'Benchmark source changed after preparation')
    rows = json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    result = dict(format='w352-w448-12s-search/1', scope=prepared['scope'], comparison=[],
        timing_scope='Overlapping aggregate worker timings; do not sum as turn wall time.',
        sampling='One paired seed per root and worker count, alternating model order; six prespecified roots.')
    for workers in (2, 3):
        with BenchmarkService(out, prepared['models'], workers) as service:
            service.config.update(batch_size=workers, flush_ms=1.)
            game = BrassSpielGame(root=SOURCE_ROOT, fast=True)
            try:
                with ExitStack() as stack:
                    sessions = {name: stack.enter_context(active_session(out, service, prepared['models'],
                        name, workers, 1, True)) for name in ('w352', 'w448')}
                    for i, row in enumerate(rows):
                        names = ('w352', 'w448') if (i+workers) % 2 == 0 else ('w448', 'w352')
                        for name in names:
                            probe = Probe()
                            with probe.active():
                                value = run_turn(sessions[name], service, game, row, 916000+i)
                            root = value['decisions'][0]
                            value.update(model=name, workers=workers, leaf_batch=1, fast=True,
                                profile=dict(probe.totals),
                                first_root=dict(choices=root['choices'], visited=sum(n > 0 for n in root['root_visits'].values()),
                                    completed=root['completed'], attempted=root['attempted'], interrupted=root['interrupted']))
                            result['comparison'].append(value)
                            write(out/'search-partial.json', result)
                            m = value['metrics']
                            print(f"{name} W={workers} {row['name']}: completed={m['completed']} "
                                f"interrupted={m['interrupted']} coverage={value['first_root']['visited']}/{root['choices']} "
                                f"labels={value['qualified_labels']} seconds={value['elapsed_seconds']:.3f}", flush=True)
                            require(value['shared_deadline'] and value['turn_actions_complete'], 'Turn completion/deadline check failed')
            finally:
                game.close()
            require(service.error is None, 'Inference service failed')
    require(len(result['comparison']) == 24, 'Incomplete benchmark matrix')
    result['source_unchanged'] = all(sha(SOURCE_ROOT/p) == h for p, h in prepared['sources'].items())
    write(out/'search.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'search', 'all'), default='all')
    parser.add_argument('--delete-models', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts/feature-engineering'), 'Use an isolated benchmark directory')
    torch.set_num_threads(2)
    if args.phase in ('prepare', 'all'):
        prepare(out)
    if args.phase in ('search', 'all'):
        search(out)
    shutil.copyfile(__file__, out/'executed-benchmark.py')
    if args.delete_models:
        cleanup(out)


if __name__ == '__main__':
    main()
