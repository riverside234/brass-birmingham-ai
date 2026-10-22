"""Explicit once-only replacement of joint7's obsolete snapshot EV reference."""
import copy
import re
from pathlib import Path

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import digest, require
from . import critic_data, snapshot_r2

CONTRACT = 'joint7-snapshot-r2-baseline/1.0.0'
KEY = 'snapshot_r2_hotfix'
RUN = 'artifacts/step5/joint7-v11-2-grouped-lr-critic'
OLD_EV = .44701269779353614


def reports(directory, state):
    """Read committed reports/compact moments, never a pending iteration."""
    generation = state['iteration']
    lower = max(1, generation - snapshot_r2.WINDOW + 1)
    selected = {}
    archive = directory / 'retention-history.json'
    if archive.is_file():
        for name, record in read(archive).items():
            match = re.fullmatch(r'iteration-(\d+)/report\.json', name)
            if match and lower <= int(match[1]) + 1 <= generation:
                selected[int(match[1]) + 1] = record['summary']
    for g in range(lower, generation + 1):
        path = directory / f'iteration-{g-1:04d}' / 'report.json'
        if path.is_file():
            selected[g] = read(path)
    entries = {e['generation']: e for e in state['history']}
    for g, report in selected.items():
        require(report['iteration'] + 1 == g and g in entries
                and report['post_update_model_id'] == entries[g]['model_id'],
                'Snapshot baseline report/checkpoint lineage differs')
        if g - 1 in entries:
            require(report['incoming_model_id'] == entries[g-1]['model_id'],
                    'Snapshot baseline incoming model lineage differs')
    require(generation in selected, 'Missing latest committed snapshot report')
    return selected


def from_report(generation, report):
    value = report['value_phase']
    rollout = value['critic_rollout']
    return snapshot_r2.entry(generation, value,
        healthy=rollout['weight_recovery']['health_ok'],
        trained=rollout['snapshot_rows_used'])


def seeded(critic, config, generation, records, *, gae_state=None):
    """Use a contiguous current regime; select the latest ten qualified commits."""
    require(snapshot_r2.finite(critic['learner_ev_baseline'])
            and abs(critic['learner_ev_baseline'] - OLD_EV) < 1e-9,
            'This one-time hotfix requires the original 0.4470127 EV baseline')
    result, migration = snapshot_r2.migrate(critic, config, generation)
    latest = records[generation]
    current = snapshot_r2.regime(latest['value_phase'], latest['actor'], latest.get('exploration_rollout'))
    weight = critic['snapshot_weight']
    # A GAE stage change at the last commit applies to the NEXT collection.
    # Such a new regime has no eligible historic observations.
    if gae_state is not None:
        lam = gae_state['active_lambda']
        current['lambda_by_table'] = dict(selfplay=lam, **{
            k: lam if enabled else 1. for k, enabled in gae_state.get('table_enabled', {}).items()})
    eligible = []
    for g in range(generation, max(0, generation - snapshot_r2.WINDOW), -1):
        report = records.get(g)
        if report is None:
            break
        value = report['value_phase']
        rollout = value['critic_rollout']
        if (rollout['weight_used'] != weight or rollout['weight_next'] != weight
                or snapshot_r2.regime(value, report['actor'], report.get('exploration_rollout')) != current):
            break
        eligible.append(from_report(g, report))
    eligible.reverse()
    controller = result['weight_recovery']
    controller.update(regime=current, entries=eligible,
        stage_origin=eligible[0]['generation'] - 1 if eligible else generation)
    qualified = [e for e in eligible if e['qualified']][-snapshot_r2.MIN_ITERATIONS:]
    evidence = snapshot_r2.summarize(qualified)
    if evidence['supported']:
        controller['reference'] = snapshot_r2.reference(evidence, generation, weight, current)
    snapshot_r2.validate(controller, generation, weight)
    return result, dict(generation=generation, weight=weight, old_ev=OLD_EV,
        baseline=evidence['r2'] if evidence['supported'] else None,
        status='ready' if evidence['supported'] else 'pending_ten_qualifying_iterations',
        evidence=evidence, regime=current, migration=migration,
        required_iterations=snapshot_r2.MIN_ITERATIONS,
        missing_qualifying_iterations=max(0, snapshot_r2.MIN_ITERATIONS - len(qualified)),
        scope='Latest ten qualifying committed iterations at the current weight and regime within twenty; no old EV conversion, no branch or snapshot-seat evidence')


def preview(directory):
    directory = Path(directory)
    state = read(directory / 'state.json')
    from .critic_data import fixed
    if fixed(state['signature']['config'].get('critic_data',{})):
        return dict(generation=state['iteration'],configured_weight=1.,disabled=True,
            status='fixed_weight_controller_disabled',preview_only=True,checkpoint_verified=False)
    records = reports(directory, state)
    latest = records[state['iteration']]['value_phase']['critic_rollout']
    if latest['weight_recovery']['contract'] == snapshot_r2.CONTRACT:
        return dict(generation=state['iteration'], weight=latest['weight_next'],
            current_reference=latest['weight_recovery']['reference_next'],
            evidence=latest['weight_recovery']['learner'], preview_only=True, checkpoint_verified=False,
            status='already_using_r2_controller')
    # This read-only export preview is not used to restore training state.
    critic = critic_data.initial(state['signature']['config']['critic_data'])
    critic.update(generation=state['iteration'], snapshots_active=latest['snapshot_rows_next'],
        snapshot_weight=latest['weight_next'], learner_ev_baseline=latest['phase1_learner_ev'])
    critic['weight_recovery'] = critic_data.recovery_initial(state['iteration'])
    _, result = seeded(critic, state['signature']['config']['critic_data'], state['iteration'], records)
    return dict(result, preview_only=True, checkpoint_verified=False)


def restore(state, raw, phases):
    """An exact state.json overlay until the next normal checkpoint commits."""
    receipt = state.get(KEY)
    if receipt is None:
        return None
    require(receipt['contract'] == CONTRACT and type(receipt['generation']) is int
            and receipt['generation'] <= state['iteration'] == raw['step'],
            'Invalid snapshot R2 hotfix binding')
    if raw['step'] > receipt['generation']:
        from .critic_data import FIXED_STATE
        require(raw['training']['actor_value_phases']['critic_data']['weight_recovery']['contract'] in (snapshot_r2.CONTRACT,FIXED_STATE),
                'Committed checkpoint lost snapshot R2 controller')
        return receipt
    from .critic_data import fixed
    require(not fixed(phases.config['critic_data']), 'Apply the pending R2 hotfix before switching to fixed snapshot weight')
    saved = raw['training']['actor_value_phases']['critic_data']
    require(receipt['checkpoint'] == state['history'][-1]
            and receipt['checkpoint']['model_id'] == raw['model_id']
            and receipt['before_digest'] == digest(saved),
            'Snapshot R2 hotfix is not bound to this committed checkpoint')
    expected, _ = snapshot_r2.migrate(saved, phases.config['critic_data'], raw['step'])
    controller = receipt['controller']
    snapshot_r2.validate(controller, raw['step'], saved['snapshot_weight'])
    require(controller['legacy_ev_baseline'] == saved['learner_ev_baseline']
            and controller['origin'] == raw['step'] and controller['healthy_streak'] == 0
            and controller['last_change_generation'] is None,
            'Snapshot hotfix may only seed evidence and replace its reference')
    selected = [e for e in controller['entries'] if e['qualified']][-snapshot_r2.MIN_ITERATIONS:]
    evidence = snapshot_r2.summarize(selected)
    ref = (snapshot_r2.reference(evidence, raw['step'], saved['snapshot_weight'], controller['regime'])
           if evidence['supported'] else None)
    require(controller['reference'] == ref, 'Snapshot hotfix baseline differs from pooled moments')
    expected['weight_recovery'] = copy.deepcopy(controller)
    require(digest(expected) == receipt['after_digest'], 'Snapshot R2 hotfix payload changed')
    phases.critic_state = expected
    phases.snapshot_r2_migration = copy.deepcopy(receipt['report'])
    return receipt


def run(root, run_path=RUN, *, apply=False, defer_until_ready=False):
    import torch
    from .actor_value import TrainingPhases
    from .retention import owned, run_lock, safe
    from .resume import load_resume
    from brass_ai.step5.league import implementation_id, write_json
    root, directory = owned(root, run_path)
    require(directory.relative_to(root).as_posix() == RUN, 'Hotfix is only for the reviewed joint7 run')
    with run_lock(directory):
        state = read(directory / 'state.json')
        signature = dict(state['signature'], implementation=implementation_id())
        with torch.random.fork_rng(devices=[]):
            model, raw, _ = load_resume(root, directory, state, signature, reviewed=True)
            phases = TrainingPhases(model, signature['config'], raw=raw, generation=state['iteration'],
                                    root=root, run=directory)
        from .critic_data import fixed
        if fixed(signature['config']['critic_data']):
            if KEY in state:restore(state,raw,phases)
            return dict(applied=False,disabled=True,reason='Snapshot weight is fixed; R2 hotfix is inactive',
                snapshot_weight=phases.critic_state['snapshot_weight'])
        if KEY in state:
            restore(state, raw, phases)
            return dict(already_applied=True, receipt=state[KEY],
                        current_reference=phases.critic_state['weight_recovery']['reference'])
        records = reports(directory, state)
        saved = raw['training']['actor_value_phases']['critic_data']
        updated, report = seeded(saved, signature['config']['critic_data'], state['iteration'], records,
                                 gae_state=raw['training']['actor_value_phases']['gae'])
        report.update(applied=apply, deferred=report['baseline'] is None, checkpoint_verified=True)
        require(not apply or report['baseline'] is not None or defer_until_ready,
                'Fewer than ten qualifying iterations; preview only. Use --defer-until-ready to seed available evidence and freeze once ten qualify')
        receipt = dict(contract=CONTRACT, generation=state['iteration'],
            checkpoint=copy.deepcopy(state['history'][-1]), before_digest=digest(saved),
            after_digest=digest(updated), controller=copy.deepcopy(updated['weight_recovery']),
            report=report, reports_digest=digest(records),
            scope='Snapshot critic controller only; checkpoint bytes, weights, both Adams, RNG, GAE, population, progress and recipe unchanged')
        new = copy.deepcopy(state)
        new[KEY] = receipt
        restore(new, raw, phases)
        if apply:
            backup = safe(directory, directory / f"state-before-snapshot-r2-{state['iteration']:04d}.json")
            if backup.exists():
                require(read(backup) == state, 'Snapshot R2 backup differs')
            else:
                write_json(backup, state)
            require(read(directory / 'state.json') == state, 'Run advanced during snapshot R2 hotfix')
            require(sha(safe(directory, root / state['history'][-1]['path'])) == state['history'][-1]['sha256'],
                    'Checkpoint changed during snapshot R2 hotfix')
            write_json(directory / 'state.json', new)
        return report
