"""Checkpoint-bound B03 preflight and isolated full-state restoration.

This is an explicit diagnostic capability, not a relaxed production resume.
Checkpoints and source state are copied byte-for-byte. Only a sealed path map
resolves their original run binding inside the diagnostic payload directory.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import shutil

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import digest, project_path, require
from brass_ai.recovery.retention import owned, safe, link, _same_payload
from brass_ai.step5.league import implementation_id, write_json
from .lab_snapshot import capture, pending_paths, tree_digest, verify_unchanged

CONTRACT = 'brass-joint7-continuation/1.0.0'
FORK = 'brass-diagnostic-fork/1.0.0'


def target_signature(state):
    """Backend/source only. Never regenerate or upgrade the saved recipe."""
    from brass_ai.recovery.config import validate_config
    config = copy.deepcopy(state['signature']['config'])
    validate_config(config)
    return dict(copy.deepcopy(state['signature']), config=config, implementation=implementation_id())


def inventory(root, run, state, raw):
    """Only committed dependencies; unknown files are neither copied nor deleted."""
    root, run = owned(root, run); files = {}; external = {}

    def add(path, expected=None, required=True):
        path = project_path(root, path)
        require(not link(path), 'Linked continuation dependency')
        if not path.is_file():
            require(not required, 'Missing committed continuation dependency: ' + str(path)); return
        if path.is_relative_to(run): safe(run, path)
        actual = sha(path)
        require(expected is None or expected == actual, 'Changed continuation dependency: ' + str(path))
        item = dict(path=path.relative_to(root).as_posix(), sha256=actual, bytes=path.stat().st_size)
        if path.is_relative_to(run): files[path.relative_to(run).as_posix()] = item
        else: external[item['path']] = item

    add(run / 'state.json')
    latest = state['history'][-1]; add(latest['path'], latest['sha256'])
    live = state.get('population', []) + [state.get('evaluation_anchor'), state.get('champion')]
    live += list(state.get('champion_mix', {}).get(k) for k in ('anchor', 'previous'))
    live += state.get('quality_pool', {}).get('members', [])
    for entry in live:
        if entry is not None and 'path' in entry: add(entry['path'], entry['sha256'])
    # Preserve exact atomic transition backups, including the pre-checkpoint gap.
    for transition in state.get('source_transitions', []):
        path = safe(run, run / transition['backup']); add(path)
        backup = read(path)
        require(digest(backup['signature']) == digest(transition['from_signature'])
                and backup['history'][-1] == transition['checkpoint']
                and (backup['iteration'], backup['games_done']) == (transition['iteration'], transition['games_done']),
                'Source-transition backup differs')
    phase = raw['training'].get('actor_value_phases', {})
    receipts = phase.get('rolling_buffer', [])
    if receipts:
        directory = run / '.ppg-eight-cache'; add(directory / 'owner.json')
        for receipt in receipts:
            stem = f"generation-{receipt['generation']:06d}"
            metadata = directory / (stem + '.json'); add(metadata)
            require(read(metadata) == receipt, 'PPG committed receipt differs')
            for ext, item in receipt['files'].items():
                require('/' not in ext and '\\' not in ext and '..' not in ext, 'Unsafe PPG extension')
                path = directory / (stem + '.' + ext); add(path, item['sha256'])
                require(path.stat().st_size == item['bytes'], 'PPG committed size differs')
    # Legacy one-age cache has its own sealed, source-bound rebuild contract.
    if phase.get('buffer') is not None:
        receipt = phase['buffer']; directory = run / '.ppg-value-cache'
        stem = f"generation-{receipt['generation']:06d}"
        add(directory / 'owner.json'); add(directory / (stem + '.npy'), receipt['sha256'])
        add(directory / (stem + '.json'))
        require(read(directory / (stem + '.json')) == receipt, 'Legacy PPG receipt differs')
    teacher = state.get('teacher_phase')
    if teacher:
        from brass_ai.recovery.teacher_buffer import TeacherBuffer
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        require(teacher == raw['training'].get('teacher_phase'), 'Teacher state/checkpoint differs')
        directory = run / 'teacher-labels'
        cache = teacher['buffer']
        buffer = TeacherBuffer(root, run, buffer_identity(state['signature']['config']),
                               receipts=cache['receipts'], retired=cache['retired'])
        if directory.exists(): add(directory / 'owner.json')
        for receipt in cache['receipts'] + cache['retired']:
            path, sidecar = buffer.paths(receipt)
            if receipt in cache['receipts'] or path.exists() or sidecar.exists():
                buffer.verify(receipt); add(path, receipt['sha256']); add(sidecar)
    # Model/manifest references outside this run keep their original lineage.
    def refs(value, *, required=True):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value and isinstance(value['path'], str):
                add(value['path'], value['sha256'], required=required)
            for child in value.values(): refs(child, required=required)
        elif isinstance(value, list):
            for child in value: refs(child, required=required)
    for key in ('origin', 'imitation', 'il_kl_limit'):
        refs(state.get(key))
    refs(state.get('quality_pool', {}).get('members', []))
    # Retired rating agents may have metadata but no retained checkpoint.
    # All live evidence and active policy references remain mandatory.
    rating = state.get('ratings')
    if rating:
        from brass_ai.recovery.ratings import validate_state, validate_frozen_reference
        validate_state(root, rating, state['signature']['config']['rating'])
        validate_frozen_reference(root, rating)
        refs({k: v for k, v in rating.items() if k != 'agents'})
        refs(rating['agents'], required=False)
    anchor = state.get('policy_anchor')
    if anchor:
        from brass_ai.recovery.policy_anchor import verify_active
        verify_active(root, state, anchor)
        refs(anchor)
        active = anchor.get('active')
        if active is not None:
            refs(read(project_path(root, active['cache']['path'])))
    probe = state.get('il_kl_limit', {}).get('probe')
    if probe:
        metadata = read(project_path(root, probe['path']))
        refs(metadata['cache'])
        for path, expected in metadata['source']['sources'].items(): add(path, expected)
    # Do not copy il-kl/.prepared, unrelated scratch files or stale probe caches.
    source = state['signature'].get('imitation_source')
    if source:
        for key, hash_key in (('path', 'sha256'), ('manifest', 'manifest_sha256')):
            if key in source: add(source[key], source[hash_key])
    if 'diversity' in state['signature']['config']:
        from brass_ai.recovery.domino_fingerprints import Fingerprints
        # Preserve all sealed cached contexts: cheap derived JSON, including
        # historical weights still referenced by the PPG descriptor datasets.
        for path in sorted((run/'domino-fingerprints').glob('*.json')):
            value=read(path)
            require(value['seal']==digest(value['fingerprint']) and path.stem==digest(value['identity']),
                    'Changed DOMiNO fingerprint manifest')
            add(path)
    refs(raw['training'].get('teacher_evaluators',{}))
    for receipt in state.get('teacher_phase',{}).get('buffer',{}).get('receipts',[]):
        refs(receipt.get('evaluators',{}))
    return dict(files=files, external=external)


def preflight(root, run, *, generation, checkpoint_sha256, devices=()):
    """Caller holds the production run lock. No cache construction or commit."""
    from brass_ai.recovery.resume import load_resume
    root, run = owned(root, run)
    initial = read(run / 'state.json')
    require(initial.get('format') in ('step5-recovery-league/2.1.0', 'step5-recovery-league/2.2.0')
            and type(initial.get('iteration')) is int and initial['iteration'] > 0,
            'B03 continuation requires an established joint7 run; completed IL is a separate handoff')
    snapshot, _, raw = capture(root, run, devices=devices)
    state = snapshot['state']
    require(state['iteration'] == generation and state['history'][-1]['sha256'] == checkpoint_sha256,
            'Expected generation/checkpoint differs; refresh qualification at the committed boundary')
    signature = target_signature(state)
    with torch.random.fork_rng(devices=[]):
        model, checked, changed = load_resume(root, run, state, signature, reviewed=True)
    require(_same_payload(raw, checked), 'Preflight checkpoint payload changed')
    require(raw['optimizer'] is not None, 'Continuation requires full PPO Adam, not weights-only input')
    from brass_ai.recovery.utility import reward_spec
    require(getattr(model.config, 'utility', None) == reward_spec(signature['config']),
            'Backend-only continuation cannot adopt a reward change')
    require(raw['training'].get('actor_value_phases', {}).get('options') ==
            {k: signature['config'][k] for k in ('advantage', 'value_phase', 'critic_data', 'seat_descriptors') if k in signature['config']},
            'Finish prior recipe/PPG adoption before backend-only continuation')
    dependencies = inventory(root, run, state, raw)
    verify_unchanged(snapshot)
    receipt = dict(contract=CONTRACT, generation=generation, checkpoint=copy.deepcopy(state['history'][-1]),
        origin_run=run.relative_to(root).as_posix(), source_state_sha256=sha(run / 'state.json'),
        source_signature=digest(state['signature']), target_signature=signature,
        source_changed=changed, dependencies=dependencies, snapshot=snapshot,
        inventory=dict(state={k: tree_digest(v) for k, v in state.items()},
                       checkpoint={k: tree_digest(v) for k, v in raw.items()}),
        allowed_changes=['implementation', 'source-transition metadata', 'source-bound utility readiness'],
        recipe_changed=False, training_advanced=False, production_adoption='Pending B05/B07')
    receipt['sha256'] = digest(receipt)
    return receipt


def verify_dependencies(root, receipt):
    for item in [*receipt['dependencies']['files'].values(), *receipt['dependencies']['external'].values()]:
        path = project_path(root, item['path'])
        require(not link(path) and path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Continuation dependency changed: ' + item['path'])


def create_fork(root, output, receipt):
    root, output = owned(root, output)
    origin = project_path(root, receipt['origin_run'])
    require(receipt['contract'] == CONTRACT and receipt['sha256'] == digest({k: v for k, v in receipt.items() if k != 'sha256'}),
            'Changed continuation receipt')
    require(not output.exists() and not output.is_relative_to(origin) and not origin.is_relative_to(output),
            'Use a new isolated diagnostic directory')
    verify_dependencies(root, receipt)
    require(not pending_paths(origin, receipt['generation']), 'Uncommitted work appeared before diagnostic fork')
    output.mkdir(parents=True)
    payload = output / 'payload'; payload.mkdir()
    for name, item in receipt['dependencies']['files'].items():
        # The original state is evidence, not a runnable destination state.json.
        target = safe(payload, payload / ('source-state.json' if name == 'state.json' else name))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_path(root, item['path']), target)
        require(sha(target) == item['sha256'], 'Diagnostic copy changed')
    verify_dependencies(root, receipt)
    value = dict(contract=FORK, destination=output.relative_to(root).as_posix(), continuation=copy.deepcopy(receipt))
    value['sha256'] = digest(value)
    write_json(output / 'diagnostic-fork.json', value)
    return DiagnosticFork(root, output, generation=receipt['generation'], checkpoint_sha256=receipt['checkpoint']['sha256'])


class DiagnosticFork:
    def __init__(self, root, directory, *, generation, checkpoint_sha256):
        self.root, self.directory = owned(root, directory)
        value = read(safe(self.directory, self.directory / 'diagnostic-fork.json'))
        require(value['contract'] == FORK and value['destination'] == self.directory.relative_to(self.root).as_posix()
                and value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}), 'Invalid diagnostic fork identity')
        self.receipt = value['continuation']; r = self.receipt
        require(r['contract'] == CONTRACT and r['sha256'] == digest({k: v for k, v in r.items() if k != 'sha256'})
                and r['generation'] == generation and r['checkpoint']['sha256'] == checkpoint_sha256,
                'Diagnostic checkpoint/generation binding differs')
        self.origin = project_path(self.root, r['origin_run']); self.run = safe(self.directory, self.directory / 'payload')
        for name, item in r['dependencies']['files'].items():
            path = self.resolve(self.origin / name)
            require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                    'Missing/changed diagnostic dependency: ' + name)
        for item in r['dependencies']['external'].values():
            path = project_path(self.root, item['path'])
            require(path.is_file() and sha(path) == item['sha256'], 'Changed external diagnostic dependency')
        self.state = read(self.run / 'source-state.json')
        require(self.state['iteration'] == generation and self.state['history'][-1] == r['checkpoint']
                and digest(self.state['signature']) == r['source_signature'], 'Diagnostic source state differs')

    def resolve(self, original):
        original = safe(self.origin, original)
        name = original.relative_to(self.origin).as_posix()
        require(name in self.receipt['dependencies']['files'], 'Unmapped diagnostic dependency: ' + name)
        return safe(self.run, self.run / ('source-state.json' if name == 'state.json' else name))

    def authorize(self, root, run, state, signature):
        require(Path(root).resolve() == self.root and Path(run).resolve() == self.run and state == self.state
                and signature == self.receipt['target_signature'] and signature == target_signature(state),
                'Diagnostic capability does not authorize this run/state/source')
        return self.origin


@dataclass
class Restored:
    model: object
    optimizer: object
    phases: object
    companion: object
    teacher: object
    probe: object
    state: dict
    raw: dict
    report: dict
    diversity: object = None


def restore(fork, *, device='cpu', allow_cuda_topology_change=False):
    """Restore real learner/Adam/PPG/auxiliary/RNG; perform zero update steps."""
    from brass_ai.learning.checkpoint import resume_optimizer
    from brass_ai.recovery.resume import load_resume, prepare_opponent_pool
    from brass_ai.recovery.actor_value import TrainingPhases
    from brass_ai.recovery.local_scoring_auxiliary import Companion
    from brass_ai.recovery.teacher_phase import TeacherPhase
    from brass_ai.recovery.exploration import restore as restore_exploration
    state = copy.deepcopy(fork.state); config = state['signature']['config']
    model, raw, changed = load_resume(fork.root, fork.run, state, fork.receipt['target_signature'],
                                     device=device, reviewed=True, diagnostic_fork=fork)
    optimizer = torch.optim.AdamW(model.parameters())
    resume_optimizer(raw, optimizer, restore_rng=False)
    require(_same_payload(optimizer.state_dict(), raw['optimizer']), 'PPO Adam restoration differs')
    phases = TrainingPhases(model, config, raw=raw, generation=state['iteration'], root=fork.root, run=fork.run)
    saved = raw['training']['actor_value_phases']
    pairs = [('options', phases.options), ('optimizer', phases.optimizer.state_dict()), ('gae', phases.state),
             ('beta_clone', phases.beta), ('kl_trips', phases.trips)]
    if 'critic_data' in saved: pairs.append(('critic_data', phases.critic_state))
    for name in ('cache_origin', 'adoption', 'schedule', 'epoch_control'):
        if name in saved: pairs.append((name, getattr(phases, name)))
    if 'rolling_buffer' in saved: pairs.append(('rolling_buffer', phases.buffer.receipts))
    if 'buffer' in saved: pairs.append(('buffer', phases.buffer.receipt))
    if 'critic_evidence' in saved: pairs.append(('critic_evidence', phases.critic_evidence))
    for key, value in pairs: require(_same_payload(value, saved[key]), 'PPG restoration changed ' + key)
    from brass_ai.recovery.snapshot_r2_hotfix import restore as restore_snapshot_r2
    snapshot_overlay = restore_snapshot_r2(state, raw, phases)
    require(not any(getattr(phases, name, None) for name in ('g210_migration', 'ppg_schedule_migration',
                'snapshot_floor_migration', 'snapshot_fixed_migration')), 'Backend continuation repeated a recipe adoption')
    companion = None
    if 'local_scoring_auxiliary' in config:
        companion = Companion(model, config['local_scoring_auxiliary'], raw=raw, generation=state['iteration'])
        require(_same_payload(companion.payload(state['iteration']), raw['training']['local_scoring_auxiliary']),
                'Companion weights/Adams/counters differ')
        phases.local_scoring = companion
    if 'exploration' in raw['training']:
        require(restore_exploration(config, state, raw['training']['exploration']) == raw['training']['exploration'],
                'Exploration state changed')
    diversity = None
    if config.get('diversity',{}).get('mode','off') != 'off':
        from brass_ai.recovery.domino_auxiliary import Diversity
        from brass_ai.recovery.domino_state import regime
        from brass_ai.recovery.local_scoring_auxiliary import bundle_id
        saved_diversity=raw['training'].get('domino')
        require(saved_diversity is not None, 'Backend continuation cannot replay DOMiNO adoption')
        diversity=Diversity(model,config['diversity'],raw=raw,generation=state['iteration'],
            regime=regime(config,state.get('v4_share_gate')))
        restored=dict(config=diversity.config,state=diversity.state,head=diversity.head.state_dict(),
                      optimizers={p:o.state_dict() for p,o in diversity.optimizers.items()})
        if 'search_fit' in saved_diversity:
            restored.update(search_fit=diversity.search_fit,fit_diagnostics=diversity.fit_diagnostics)
        require(_same_payload(restored,{k:v for k,v in saved_diversity.items() if k!='identity'})
                and bundle_id(restored)==saved_diversity['identity'], 'Live DOMiNO restore changed head/Adams/controller/evidence')
        require(state.get('domino_state_digest',digest(diversity.state))==digest(diversity.state), 'DOMiNO state digest differs')
        phases.diversity=diversity
    if 'quality_pool' in state:
        pool, migration = prepare_opponent_pool(fork.root, fork.run, state, raw, config, changed)
        require(migration is None and pool == state['quality_pool'], 'Backend migration changed opponent pool')
    if 'v4_share_gate' in state:
        from brass_ai.recovery.v4_share import validate_state
        validate_state(state['v4_share_gate'], config, state['iteration'], state['games_done'])
    probe = None
    if 'il_kl_limit' in config:
        from brass_ai.recovery.il_kl_limit import validate_state
        from brass_ai.recovery.policy_anchor import verify_checkpoint
        validate_state(state['il_kl_limit'], config['il_kl_limit'], state['iteration'])
        if 'policy_anchor' in state: verify_checkpoint(state, raw)
        else:
            from brass_ai.recovery.g110_hotfix import verify_checkpoint
            verify_checkpoint(state, raw)
        probe = restore_probe(fork, config, state['il_kl_limit']['probe'])
    teacher = TeacherPhase(fork.root, fork.run, config, state, model, raw) if 'teacher_phase' in config else None
    if teacher is not None:teacher.diversity=diversity
    if teacher is not None and 'teacher_phase' in state:
        saved_teacher = state['teacher_phase']
        require(teacher.cursor == saved_teacher['teacher_seed_cursor'] and teacher.buffer.receipts==saved_teacher['buffer']['receipts'],
                'Teacher cursor/committed receipts changed')
        if teacher.buffer.identity==saved_teacher['buffer']['identity']:
            require(teacher.weight == saved_teacher['weight'] and teacher.buffer.payload() == saved_teacher['buffer'],
                    'Teacher weight/retained ages changed')
        else:
            require(teacher.options['mode']=='off' and saved_teacher['weight'] in teacher.weight['archive']
                and all(r in teacher.buffer.retired for r in saved_teacher['buffer']['receipts']),
                'Changed teacher profile must archive calibration and retain rollback shards')
    topology = resume_optimizer(raw, optimizer, allow_cuda_topology_change=allow_cuda_topology_change)
    require(torch.equal(torch.get_rng_state(), raw['rng']), 'CPU RNG restoration differs')
    cuda_checked = False
    if raw['cuda_rng'] and torch.cuda.is_available():
        for index in range(min(len(raw['cuda_rng']), torch.cuda.device_count())):
            require(torch.equal(torch.cuda.get_rng_state(index), raw['cuda_rng'][index]), 'CUDA RNG restoration differs')
        cuda_checked = True
    require(tree_digest(state) == tree_digest(fork.state) and tree_digest(raw['state_dict']) == tree_digest(model.state_dict()),
            'Restoration changed model or progress')
    report = dict(contract=CONTRACT, generation=state['iteration'], next_generation=state['iteration'] + 1,
        checkpoint=copy.deepcopy(fork.receipt['checkpoint']), target_signature=digest(fork.receipt['target_signature']),
        full_state_restored=True, optimizers=['ppo', 'value'] + (['local-scoring-ppo', 'local-scoring-ppg'] if companion else []),
        preserved_ppg_ages=[r['generation'] for r in saved.get('rolling_buffer', [])],
        teacher_state_restored=teacher is not None, cuda_rng_checked=cuda_checked, topology=topology,
        domino_restored=diversity is not None,
        domino_components={k:tree_digest(v) for k,v in restored.items()} if diversity is not None else None,
        diversity_teacher_readiness='missing_retained_intrinsic_fit' if diversity is not None and diversity.search_fit is None else 'requires_frozen_evaluator',
        snapshot_r2_overlay=copy.deepcopy(snapshot_overlay),
        utility_readiness_before=raw['training'].get('utility_calibration'), utility_readiness_after=model.utility_calibration,
        state_digest=tree_digest(state), checkpoint_parts={k: tree_digest(v) for k, v in raw.items()},
        optimizer_steps=0, source_run_modified=False, qualification_scope='Full restoration; next update/commit remains B05')
    if diversity is not None:report['optimizers']+=['intrinsic-ppo','intrinsic-ppg','domino-dual-adam']
    return Restored(model, optimizer, phases, companion, teacher, probe, state, raw, report, diversity)


def restore_probe(fork, config, receipt):
    """Resolve the signed original paths without rewriting probe metadata."""
    import numpy as np
    from brass_ai.recovery.il_kl_probe import FORMAT, Probe, semantic_hash, source_binding, current_config
    path = fork.resolve(project_path(fork.root, receipt['path']))
    require(sha(path) == receipt['sha256'], 'Committed fixed probe manifest changed')
    metadata = read(path)
    require(metadata['format'] == FORMAT and current_config(metadata['config']) == current_config(config['il_kl_limit'])
            and metadata['source'] == source_binding(fork.root, fork.state['signature']['imitation_source']),
            'Diagnostic fixed probe source/config differs')
    cache = fork.resolve(project_path(fork.root, metadata['cache']['path']))
    require(sha(cache) == metadata['cache']['sha256'], 'Diagnostic fixed probe arrays changed')
    with np.load(cache, allow_pickle=False) as data: arrays = {k: data[k] for k in data.files}
    require(metadata['probe_hash'] == receipt['probe_hash'] == semantic_hash(metadata, arrays), 'Probe semantic identity differs')
    return Probe(metadata, arrays, config['il_kl_limit'])


def check_completed_il(root, run):
    from brass_ai.recovery.demonstrations import completed_imitation
    model, origin = completed_imitation(root, run)
    return dict(passed=True, origin=origin, model=model.config.to_dict(),
                scope='Optional completed-IL fresh handoff only; not joint7 continuation or a new PPO run')
