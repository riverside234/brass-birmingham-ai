"""Short real-device B01 rules/features/inference checks; no games or updates."""
from __future__ import annotations

import copy
import time
from threading import Event

import numpy as np
import torch

from brass_ai.learning.batching import collate, model_id
from brass_ai.learning.contracts import digest, require
from brass_ai.learning.features import Encoder
from brass_ai.learning.seat_descriptors import descriptor, relative, unknown_numpy
from brass_ai.native.features import NativeFeatureEncoder
from brass_ai.recovery.inference import InferenceService
from .lab_inference import NativeInputClient, Scope


def context_for(model, profile):
    reward = getattr(model.config, 'utility', None)
    cap = reward['insolvency']['max_cost'] if reward is not None else .075
    return dict(model_id=model_id(model), reward_id=digest(reward), profile_id=digest(profile),
                style=[0.] * 4, intent=[0.] * 8, remaining=[cap] * 4, max_cost=cap,
                stable_descriptors=None)


def forward(model, encoded, context, device, descriptors=None):
    kwargs = dict(descriptors=torch.tensor((unknown_numpy() if descriptors is None else descriptors)[None],
                                           dtype=torch.float32, device=device))
    if getattr(model.config, 'utility', None) is not None:
        kwargs['remaining'] = torch.tensor([context['remaining']], dtype=torch.float32, device=device)
    with torch.inference_mode():
        outputs = model(*collate([encoded], device), **kwargs)
        return [torch.softmax(outputs[0].float(), -1).cpu().numpy()[0]] + [x.cpu().numpy()[0] for x in outputs[1:]]


def qualify_device(worker, fixtures, model, second, execution, profile, device):
    require(torch.device(device).type == 'cuda', 'B01 GPU qualification requires actual CUDA')
    oracle = Encoder()
    models = [model.cpu().eval(), second.cpu().eval()]
    gpu = [copy.deepcopy(m).to(device).eval().requires_grad_(False) for m in models]
    contexts = [context_for(m, profile) for m in models]
    require(contexts[0]['model_id'] != contexts[1]['model_id'], 'Mixed-model smoke requires distinct frozen weights')
    atol, rtol = 2e-5, 1e-5
    checks, measurements = 0, []
    native = [NativeFeatureEncoder(worker, s) for s in (0, 1)]
    with InferenceService(execution, device) as service:
        for m, context in zip(models, contexts):
            require(service.register(m) == context['model_id'], 'GPU registration identity differs')
        # Initialize CUDA/model residency outside the 4-second diagnostic request clocks.
        with service.lock:
            for context in contexts:
                service._model(context['model_id'])
        torch.cuda.synchronize(device)
        lease = service.channels(2)
        endpoints = [lease.acquire(), lease.acquire()]
        for endpoint in endpoints:
            endpoint.timeout = 10.0
        try:
            for index, row in enumerate(fixtures['roots']):
                expected = row['expected']; state = expected['state']; actor = state['context']['actor']
                worker.import_state(index % 2, state)
                actual = worker.rules_snapshot(index % 2)
                require(actual['state'] == state and actual['choices'] == expected['choices']
                        and actual['chance'] == expected['chance'], 'Native rules fixture differs: ' + row['id'])
                view = worker.observation(index % 2, actor)['information']['observation']['view']
                require(view == row['view'], 'Native permitted view differs: ' + row['id'])
                choices, context = expected['choices'], contexts[index % 2]
                py = oracle.encode(view, choices, context['style'], context['intent'])
                rs = native[index % 2].encode(view, choices, context)
                for field in ('state', 'actions', 'stable_to_relative'):
                    np.testing.assert_array_equal(getattr(py, field), getattr(rs, field))
                require(py.choice_ids == rs.choice_ids, 'Full legal order differs')
                np.testing.assert_array_equal(rs.descriptors, unknown_numpy())
                # Critic descriptors remain outside actor Features-4. Exercise both
                # CPU/GPU heads and the native stable->relative conversion explicitly.
                stable = [descriptor('learner', 0, [0.] * 28), descriptor('snapshot', 12, [.2] * 28),
                          descriptor('il', 25, [.1] * 28), descriptor('rule_bot', 0, [.3] * 28)]
                explicit = native[index % 2].encode(view, choices, dict(context, stable_descriptors=stable))
                np.testing.assert_array_equal(explicit.state, rs.state)
                np.testing.assert_array_equal(explicit.actions, rs.actions)
                np.testing.assert_array_equal(explicit.descriptors, relative(stable, py.stable_to_relative))
                cpu = forward(models[index % 2], py, context, 'cpu')
                device_outputs = forward(gpu[index % 2], rs, context, device)
                cpu_explicit = forward(models[index % 2], py, context, 'cpu', explicit.descriptors)
                gpu_explicit = forward(gpu[index % 2], explicit, context, device, explicit.descriptors)
                for a, b in list(zip(cpu, device_outputs)) + list(zip(cpu_explicit, gpu_explicit)):
                    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
                np.testing.assert_array_equal(device_outputs[0], gpu_explicit[0])
                # Two frozen models in one real queue/slot submission exercise its
                # existing grouping and model-cache eviction, without new scheduling.
                scope = Scope(index % 2, 'b01-' + row['id'], 0)
                client = NativeInputClient(endpoints[index % 2], native[index % 2], profile, scope)
                deadline = time.monotonic() + 4.0
                ticket = client.submit([(view, choices, c) for c in contexts], deadline=deadline)
                results = client.poll(ticket, wait=True)
                for i, result in enumerate(results):
                    require(result['status'] == 'ok', 'GPU adapter exceeded the 4s request allowance')
                    reference = forward(models[i], py, contexts[i], 'cpu')
                    for name, expected_output in zip(('probabilities', 'win_share', 'placement', 'vp', 'utility'), reference):
                        np.testing.assert_allclose(result[name], expected_output, atol=atol, rtol=rtol)
                checks += 1
                measurements.append(dict(id=row['id'], actor=actor, era=state['context']['era'],
                                         phase=state['context']['phase'], candidates=len(choices),
                                         max_absolute_output_error=max(float(np.max(np.abs(a-b)))
                                                                       for a, b in zip(cpu, device_outputs))))
            # Retirement precedes encoding, and an in-flight cancellation drains
            # the slot before a new epoch can be admitted. No kernel preemption claim.
            stop = Event(); stop.set()
            client = NativeInputClient(endpoints[0], native[0], profile, Scope(0, 'stopped', 0))
            deadline = time.monotonic() + 4.0
            before = worker._sequence
            ticket = client.submit([(view, choices, contexts[0])], deadline=deadline, cancelled=stop)
            require(worker._sequence == before, 'Stopped frame reached native encoding')
            require(client.poll(ticket, wait=True)[0]['status'] == 'cancelled', 'Cancellation did not retire')
            client.advance(Scope(0, 'stopped', 1))
            ticket = client.submit([(view, choices, contexts[0])], deadline=deadline, cancelled=stop)
            require(client.poll(ticket, wait=True)[0]['status'] == 'cancelled', 'Focused epoch cancellation differs')
            expired = NativeInputClient(endpoints[1], native[1], profile, Scope(1, 'expired', 0))
            before = worker._sequence
            ticket = expired.submit([(view, choices, contexts[0])], deadline=time.monotonic()-1)
            require(worker._sequence == before, 'Expired frame reached native encoding')
            require(expired.poll(ticket, wait=True)[0]['status'] == 'expired', 'Expiry did not retire')
            report = service.report()
            require(report['error'] is None and report['positions'] == 2 * checks,
                    'Service error or unexpected GPU work')
        finally:
            for endpoint in endpoints:
                endpoint.close()
    return dict(passed=True, device=device, atol=atol, rtol=rtol, roots=measurements,
                service=report, model_ids=[c['model_id'] for c in contexts],
                matched_search_benchmark=False, teacher_admitted=False,
                scope='real native input/GPU service parity; B04 search/teacher integration remains separate')


def main():
    import argparse
    import json
    from pathlib import Path
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.learning.contracts import read_json
    from brass_ai.native.protocol import NativeWorker
    from .lab_contract import SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('worker', 'fixtures', 'state', 'second', 'profile', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', action='append', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    snapshot = read_json(args.state)
    model, _ = load_checkpoint(snapshot['checkpoint']['path'])
    second, _ = load_checkpoint(args.second)
    with NativeWorker(args.worker, timeout=10) as worker:
        require(worker.hello['build']['source_sha256'] == SOURCE, 'GPU smoke native source differs')
        for index, device in enumerate(args.device):
            report = qualify_device(worker, read_json(args.fixtures), model, second,
                                    snapshot['execution'], read_json(args.profile), device)
            (args.out / f'gpu-{index}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(device, 'passed', len(report['roots']), 'roots', flush=True)


if __name__ == '__main__':
    main()
