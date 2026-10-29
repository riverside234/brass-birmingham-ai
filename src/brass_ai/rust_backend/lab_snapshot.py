"""Capture a committed lab run under its existing run lock; never resume/train."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

from brass_ai.learning.contracts import digest, project_path, read_json, require
from .lab_contract import pin, sha


def tree_digest(value):
    """Stable, typed digest of checkpoint tensors and optimizer/RNG containers."""
    import torch
    h = hashlib.sha256()

    def add(tag, data):
        h.update(tag + len(data).to_bytes(8, 'little') + data)

    def walk(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            add(b't', str((item.dtype, tuple(item.shape))).encode())
            add(b'b', item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            # Module state_dicts (including the local-scoring companion) are
            # OrderedDicts. Hash their mapping contents like ordinary dicts.
            add(b'd', str(len(item)).encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                walk(key); walk(item[key])
        elif type(item) in (tuple, list):
            add(b'l' if type(item) is list else b'u', str(len(item)).encode())
            for child in item:
                walk(child)
        else:
            require(type(item) in (str, int, float, bool, type(None)), 'Unsupported checkpoint digest value')
            add(type(item).__name__.encode(), json.dumps(item, allow_nan=False).encode())
    walk(value)
    return h.hexdigest()


def pending_paths(run, generation):
    pending = []
    directory = run / f'iteration-{generation:04d}'
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        pending.append(str(directory))
    # Include partially written outputs, not just the final checkpoint name.
    pending += [str(p) for p in run.glob(f'generation-{generation+1:04d}.pt*')]
    return sorted(set(pending))


def capture(root, run, *, devices):
    """Caller holds retention.run_lock for this entire operation and qualification."""
    import torch
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.recovery.config import execution_config
    root, run = Path(root).resolve(), Path(run).resolve()
    state_file = run / 'state.json'
    state_pin = pin(state_file)
    state = read_json(state_file)
    generation = state['iteration']
    require(type(generation) is int and generation >= 0 and state['history'], 'No committed learner checkpoint')
    pending = pending_paths(run, generation)
    require(not pending, 'Recover pending work with its existing backend before B01: ' + ', '.join(pending))
    signature = state['signature']
    require(type(signature) is dict and 'implementation' in signature and 'compatibility' in signature,
            'Missing signed source/runtime identity')
    execution = execution_config(signature['config']['execution'])
    host = project_path(root, signature['config']['host'])
    require(host.is_dir(), 'Configured reference host runtime is missing')
    runtime = [pin(p) for p in sorted(host.rglob('*')) if p.is_file()
               and p.suffix in ('.json', '.dll', '.exe', '.so')]
    require(runtime, 'Configured reference host has no runtime files')
    entry = state['history'][-1]
    checkpoint = project_path(root, entry['path'])
    require(checkpoint.is_relative_to(run), 'Learner checkpoint is outside the selected run')
    require(sha(checkpoint) == entry['sha256'], 'Committed learner checkpoint changed')
    # Strict weights-only loading verifies rules/features, architecture, model ID.
    # No optimizer load, RNG restore, source transition or utility exception.
    with torch.random.fork_rng(devices=[]):
        model, raw = load_checkpoint(checkpoint, device='cpu')
    require(raw['training'].get('fresh_run') and project_path(root, raw['training']['fresh_run']) == run,
            'Checkpoint is not the selected run residual learner')
    require(raw['step'] == entry['generation'] == generation and raw['model_id'] == entry['model_id'],
            'Checkpoint generation/model identity differs from committed state')
    known = {raw['training'].get('signature')}
    for transition in state.get('source_transitions', []):
        if transition['checkpoint'] == entry and digest(transition['from_signature']) in known:
            from brass_ai.recovery.resume import check_signature
            # Reuse the original reviewed transition validator; hashes alone do not authorize it.
            target = next((s for s in [signature] + [t['from_signature'] for t in state['source_transitions']]
                           if digest(s) == transition['to_signature']), None)
            require(target is not None, 'Missing reviewed source-transition target')
            check_signature(dict(state, signature=transition['from_signature']), target, reviewed=True)
            known.add(transition['to_signature'])
    initial = generation == 0 and raw['training'].get('mode') == 'step5-fresh-residual-initial'
    require(digest(signature) in known or initial, 'Checkpoint is not bound to the committed run signature')
    require(raw['optimizer'] is not None or generation == 0, 'Committed checkpoint is missing optimizer state')
    frozen = state.get('population', []) + [state.get('evaluation_anchor'), state.get('champion')]
    frozen += list(state.get('champion_mix', {}).get(k) for k in ('anchor', 'previous'))
    population = []
    seen = {str(checkpoint)}
    for item in frozen:
        if item is None:
            continue
        path = project_path(root, item['path'])
        if str(path) in seen:
            continue
        require(sha(path) == item['sha256'], 'Frozen population/anchor changed: ' + str(path))
        population.append(dict(entry=item, file=pin(path)))
        seen.add(str(path))
    require(pin(state_file) == state_pin, 'Committed state changed during snapshot')
    manifest = dict(contract='brass-lab-state/1.0.0', state=state, state_file=state_pin,
                    checkpoint=pin(checkpoint), generation=generation, model_id=raw['model_id'],
                    checkpoint_parts={k: tree_digest(raw[k]) for k in
                                      ('state_dict', 'optimizer', 'rng', 'cuda_rng', 'training')},
                    signature_sha256=digest(signature), config_sha256=digest(signature['config']),
                    population=population, pending_work=pending, execution=execution, host_runtime=runtime,
                    utility_calibration=raw['training'].get('utility_calibration'),
                    teacher_admitted=False, source_transition_performed=False,
                    runtime=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                 cudnn=torch.backends.cudnn.version(), platform=platform.platform(),
                                 cpu_count=os.cpu_count(), cpu_affinity=sorted(os.sched_getaffinity(0))
                                 if hasattr(os, 'sched_getaffinity') else None,
                                 devices=[dict(device=d, name=torch.cuda.get_device_name(d),
                                               capability=torch.cuda.get_device_capability(d)) for d in devices]))
    return manifest, model, raw


def verify_unchanged(snapshot):
    files = [snapshot['state_file'], snapshot['checkpoint']] + [p['file'] for p in snapshot['population']]
    files += snapshot['host_runtime']
    for original in files:
        require(pin(original['path']) == original, 'Lab state changed during B01: ' + original['path'])
    run = Path(snapshot['state_file']['path']).parent
    require(not pending_paths(run, snapshot['generation']), 'New pending work appeared during B01')
