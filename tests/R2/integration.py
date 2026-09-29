"""R2 exit checks against the private, locally owned R1 runtime and traces."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from brass_ai.beliefs import sample_world_setup
from brass_ai.host import RulesHost
from brass_ai.learning.batching import Evaluator, Request, collate, model_id
from brass_ai.learning.checkpoint import load_checkpoint, resume_optimizer, save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, read_json, require
from brass_ai.learning.export import export_onnx
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import read_shards, write_shard
from brass_ai.learning.runner import ModelPolicy, RandomPolicy, collect_games, match_report
from brass_ai.learning.training import on_policy_update, prepared, supervised_updates


def check_sell_beer(root, mono=None):
    """Exercise the inference request boundary on merchant beer and multi-sale."""
    encoder = Encoder()
    model = FourSeatNet(ModelConfig.preset('small'))
    evaluator = Evaluator(model)
    composition = read_json(SOURCE_ROOT / 'configs/contracts/v0.2.0/card-composition.json')
    with RulesHost(Path(root)/'artifacts/r1-host/guarded', mono=mono) as host:
        context = host.reset(1201)
        view = host.observe(context['actor'])
        mat = next(m for m in view['players'][view['seat']]['mat'] if m['type']=='CottonMill')
        for identity in ('Worcester:0','Worcester:1'):
            next(s for s in view['slots'] if s['id']==identity)['tile'] = mat['tiles'].pop(0)
        next(l for l in view['links'] if l['id']=='CanalWorcesterGloucester')['owner'] = view['seat']
        for merchant in view['merchants']:
            if merchant['id'].startswith('Gloucester'):
                merchant.update(types=['CottonMill'],beer=1)
        context = host.reset(9901,sample_world_setup(view,composition,9901))
        steps = [('sell',{'target':'Worcester:0','merchant':'Gloucester:0'}),('beer',{}),
                 ('bonus_develop',{'industry':'Brewery'}),
                 ('sell',{'target':'Worcester:1','merchant':'Gloucester:1'}),('beer',{}),
                 ('bonus_skip',{}),('finish',{})]
        for index,(kind,fields) in enumerate(steps):
            observed, choices = host.observe(context['actor']),host.choices()
            request = Request(f'fixture-{index}',evaluator.model_id,observed,choices,time.monotonic()+30)
            prediction = evaluator.predict([request])[0]
            from brass_ai.learning.batching import accept_response
            accept_response(request,prediction,context['revision'],evaluator.model_id)
            require(len(prediction['win_share'])==4,'Special-phase inference lost a seat')
            selected = next(c for c in choices if c['kind']==kind and all(c.get(k)==v for k,v in fields.items()))
            context = host.apply(context,selected)
        return {'passed':True,'decisions':len(steps),'scenario':'two sales, separate merchant beer, develop/skip bonuses, finish'}


def check_integration(root, *, mono=None):
    root = Path(root)
    directory = root / 'artifacts/r2' / ('exit-' + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    host_directory = root / 'artifacts/r1-host/guarded'
    traces = read_json(root / 'artifacts/r1-host/traces.json')
    composition = read_json(SOURCE_ROOT / 'configs/contracts/v0.2.0/card-composition.json')
    encoder = Encoder()
    torch.manual_seed(31415)
    torch.set_num_threads(4)
    model = FourSeatNet(ModelConfig.preset('medium'))
    evaluator = Evaluator(model, batch_size=16)
    coverage, kinds, seats, encoded_views = set(), set(), set(), []
    counterfactuals = 0
    started = time.monotonic()
    print('R2: checking R1 phase traces, encoding and counterfactual privacy', flush=True)
    with RulesHost(host_directory, mono=mono) as host, RulesHost(host_directory, mono=mono) as branch:
        for entry in traces:
            context = host.call('restore', checkpoint=entry['checkpoint'])
            if context['terminal']:
                continue
            actor = context['actor']
            view, choices = host.observe(actor), host.choices()
            original = encoder.encode(view, choices)
            encoded_views.append(original)
            coverage.add((context['era'], context['phase']))
            kinds.update(c['kind'] for c in choices)
            seats.add(actor)
            for sample_seed in (31001, 31002):
                setup = sample_world_setup(view, composition, sample_seed)
                branch.reset(sample_seed, setup)
                changed_view, changed_choices = branch.observe(actor), branch.choices()
                require(changed_view == view and changed_choices == choices, 'Counterfactual changed permitted information')
                changed = encoder.encode(changed_view, changed_choices)
                np.testing.assert_array_equal(changed.state, original.state)
                np.testing.assert_array_equal(changed.actions, original.actions)
                with torch.inference_mode():
                    outputs = model(*collate([original, changed]))
                for output in outputs:
                    np.testing.assert_allclose(output[0], output[1], atol=1e-6)
                counterfactuals += 1
    require(seats == set(range(4)), 'R1 fixtures did not cover all observer seats')
    require({phase for _, phase in coverage} == {'action','payment','continuation','bonus','debt'}, 'Missing policy phase coverage')
    shape_batches = [[encoded_views[0]], encoded_views[:4],
                     [min(encoded_views, key=lambda e: len(e.choice_ids))]]
    export = export_onnx(model, directory / 'fresh-medium.onnx', shape_batches)
    # CPU measurement is diagnostic, not laptop+Unity or Intel GPU qualification.
    timings = []
    batch = collate([encoded_views[0]])
    with torch.inference_mode():
        model(*batch)
        for _ in range(30):
            begin = time.perf_counter()
            model(*batch)
            timings.append(1000 * (time.perf_counter()-begin))
    print('R2: collecting four random-policy games and four fresh-model games', flush=True)
    random_policy = RandomPolicy()
    random_games = collect_games(host_directory, [2400,2401,2402,2403], [random_policy]*4,
                                 workers=4, mono=mono, sampling_seed=414100)
    write_shard(directory / 'random.jsonl.gz', random_games, {'producer':'R2-exit','mode':'legal-random'})
    neural = ModelPolicy(model, batch_size=16)
    neural_games = collect_games(host_directory, [2500,2501,2502,2503], [neural]*4,
                                 workers=4, mono=mono, sampling_seed=514100)
    write_shard(directory / 'neural.jsonl.gz', neural_games, {'producer':'R2-exit','mode':'fresh-model'})
    roundtrip = read_shards([directory/'random.jsonl.gz',directory/'neural.jsonl.gz'])
    require(len(roundtrip) == 8, 'Replay round trip lost a game')
    trace_kinds = sorted(kinds)
    kinds.update(c['kind'] for g in roundtrip for r in g['records'] for c in r['choices'])
    require(kinds==set(encoder.kinds),'Missing legal-choice kind coverage')
    special_phases = check_sell_beer(root,mono)
    print('R2: value learning, exact resume and one on-policy update', flush=True)
    data = prepared(random_games)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0003)
    metrics = supervised_updates(model,optimizer,data,4,16)
    saved_id = save_checkpoint(directory/'value.pt',model,optimizer,step=4,training={'mode':'exit-smoke'})
    supervised_updates(model,optimizer,data,1,16)
    after_id = model_id(model)
    resumed, raw = load_checkpoint(directory/'value.pt')
    optimizer = torch.optim.AdamW(resumed.parameters(),lr=0.0003)
    resume_optimizer(raw,optimizer)
    supervised_updates(resumed,optimizer,data,1,16)
    require(model_id(resumed) == after_id, 'Real-data optimizer/RNG resume drifted')
    # Use the unchanged producer network: value-trained weights would be off-policy.
    onpolicy_model = copy.deepcopy(neural.evaluator.model).requires_grad_(True)
    opt = torch.optim.AdamW(onpolicy_model.parameters(),lr=0.0001)
    before = model_id(onpolicy_model)
    onpolicy = on_policy_update(onpolicy_model,opt,neural_games,batch_size=16)
    require(model_id(onpolicy_model) != before, 'On-policy update did not change weights')
    save_checkpoint(directory/'onpolicy.pt',onpolicy_model,opt,step=1,training={'mode':'exit-on-policy'})
    trained_export = export_onnx(onpolicy_model,directory/'trained-medium.onnx',shape_batches)
    report = {
        'passed': True, 'artifact_directory': directory.relative_to(root).as_posix(),
        'compatibility': compatibility(), 'parameters': model.parameter_count,
        'state_dim': encoder.state_dim, 'action_dim': encoder.action_dim,
        'trace_positions': len(encoded_views), 'counterfactual_predictions': counterfactuals,
        'observer_seats': sorted(seats), 'era_phases': sorted(coverage), 'choice_kinds': sorted(kinds),
        'trace_choice_kinds':trace_kinds,'special_phase_inference':special_phases,
        'complete_games':8,'recorded_decisions':sum(len(g['records']) for g in roundtrip),
        'random_matches':match_report(random_games),'fresh_model_matches':match_report(neural_games),
        'value_steps':5,'exact_resume':True,'last_value_metrics':metrics[-1],
        'on_policy_update':onpolicy, 'fresh_export':export,'trained_export':trained_export,
        'cpu_forward_ms_p50':float(np.median(timings)),
        'cpu_forward_ms_p95':float(np.percentile(timings,95)),
        'timed_candidates':len(encoded_views[0].choice_ids),
        'torch_version':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'elapsed_seconds':time.monotonic()-started,
        'limitations':['No lab/CUDA/Intel GPU run','No human/legacy strength evaluation',
                       'No search or live-game AI bridge','Linux host not yet qualified'],
    }
    (directory/'integration.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
