"""Explicit same-run continuation across a pinned, reviewed source transition."""
import copy
from pathlib import Path

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import CONFIG_DIR, digest, project_path, require
from brass_ai.learning.model import ModelConfig
from brass_ai.step5.league import write_json, implementation_id
from .retention import safe


def apply_optimizer_config(optimizer, config):
    """Apply the signature-validated settings AFTER Adam restores its groups.

    load_state_dict restores the old LR/decay as well as moments. Changing only
    the AdamW constructor therefore silently ignores a reviewed LR change.
    Keep every moment, step, parameter and other optimizer option intact.
    """
    require(len(optimizer.param_groups)==1,'Step 5 expects a single AdamW group')
    before=[dict(lr=g['lr'],weight_decay=g['weight_decay']) for g in optimizer.param_groups]
    for group in optimizer.param_groups:
        group['lr']=config['learning_rate']
        group['weight_decay']=config['weight_decay']
    return dict(before=before,after=[dict(lr=g['lr'],weight_decay=g['weight_decay']) for g in optimizer.param_groups],
                preserved='Adam moments, steps, other group options and RNG')


def check_signature(state, signature, reviewed=False):
    require(state['format'] == 'step5-recovery-league/2.2.0', 'Unsupported self-play resume format')
    previous = state['signature']
    if previous == signature:
        return False
    require(reviewed, 'Run configuration/source changed. For the reviewed selfplay-v2 source only, '
            'repeat with --resume-reviewed-source; otherwise use a fresh directory')
    unchanged = lambda s: {k: v for k, v in s.items() if k not in ('implementation', 'config')}
    require(unchanged(previous) == unchanged(signature),
            'Reviewed resume requires unchanged config, data/IL, opponents and learning contracts')
    approvals = read(CONFIG_DIR / 'selfplay-resume-sources.json')
    require(approvals['format'] == 'selfplay-resume-sources/1.0.0', 'Unsupported source review receipt')
    transitions = [e for e in approvals['transitions'] if e['from_implementation'] == previous['implementation']
                and e['to_implementation'] == signature['implementation']
                and e['learning_compatibility'] == digest(signature['compatibility'])]
    require(transitions, 'This self-play source transition has not been reviewed')
    config_ok = previous['config'] == signature['config']
    for transition in transitions:
        changes = transition.get('config_changes', {})
        if all(previous['config'].get(k) == v['from'] for k, v in changes.items()):
            expected = dict(previous['config'], **{k: v['to'] for k, v in changes.items()})
            migration=transition.get('config_migration')
            if migration is not None:
                from .config import TIME_CONFIG_MIGRATION,time_budget_config
                require(migration==TIME_CONFIG_MIGRATION,'Unknown reviewed configuration migration')
                expected=time_budget_config(expected)
            config_ok |= expected == signature['config']
    require(config_ok, 'Reviewed resume requires unchanged config except explicitly pinned review changes')
    require(state['iteration'] > 0, 'Reviewed continuation requires a committed PPO generation')
    return True


def load_resume(root, run, state, signature, *, device='cpu', reviewed=False):
    """Read/verify everything before altering state or starting collection."""
    changed = check_signature(state, signature, reviewed)
    for entry in state['history']:
        safe(run, project_path(root, entry['path']))
    entry = state['history'][-1]
    path = safe(run, project_path(root, entry['path']))
    require(sha(path) == entry['sha256'], 'Snapshot changed')
    model, raw = load_checkpoint(path, device=device)
    from .utility import reward_spec
    from brass_ai.learning.utility_model import utility_config
    expected=ModelConfig.preset(signature['config']['model_size'])
    require(model.config in (expected,utility_config(expected,reward_spec(signature['config'])))
            and raw['training'].get('fresh_run') == run.relative_to(root).as_posix(),
            "Resume must use this run's residual checkpoint")
    require(raw['model_id'] == entry['model_id'] and raw['step'] == entry['generation'] == state['iteration'],
            'Resume checkpoint generation/model identity differs')
    bound = raw['training'].get('signature') == digest(state['signature'])
    # A restart after the atomic source transition, before its first new PPO
    # commit, still has the original checkpoint. Accept only that exact binding.
    if not bound:
        bindings={raw['training'].get('signature')}
        for transition in state.get('source_transitions', []):
            if transition['checkpoint']==entry and digest(transition['from_signature']) in bindings:
                bindings.add(transition['to_signature'])
        bound=digest(state['signature']) in bindings
    initial = (state['iteration'] == 0 and raw['training'].get('mode') == 'step5-fresh-residual-initial')
    require(bound or initial, 'Checkpoint is not bound to this run signature')
    if changed:
        # An old partial collection has a different provenance digest. Never mix
        # it into the updated collection or discard it behind the user's back.
        pending = safe(run, run / f"iteration-{state['iteration']:04d}")
        require(not pending.exists() or (pending.is_dir() and not any(pending.iterdir())),
                f'Uncommitted iteration at {pending}; finish/recover it with the previous source before upgrading')
        output = safe(run, run / f"generation-{state['iteration'] + 1:04d}.pt")
        require(not output.exists(), f'Uncommitted checkpoint at {output}; recover it before upgrading')
        # Old full history may already be retired; only the live population and
        # evaluation anchor must still be present. Latest snapshot checked above.
        verified = {entry['path']}
        frozen_entries=state.get('population', []) + [state['evaluation_anchor']]
        if state.get('champion') is not None:frozen_entries.append(state['champion'])
        for frozen in frozen_entries:
            if frozen['path'] in verified:
                continue
            path = safe(run, project_path(root, frozen['path']))
            require(sha(path) == frozen['sha256'], 'Live opponent/evaluation checkpoint changed')
            verified.add(frozen['path'])
    if getattr(model.config,'utility',None) is not None and raw['training'].get('signature')!=digest(signature):
        # The old receipt can predate a post-calibration policy/distillation
        # update. A reviewed source transition preserves its weights and Adam,
        # but does not certify its leaf evaluator under the repaired pipeline.
        # Replace (rather than mutate) the receipt shared with raw checkpoint
        # metadata. This also covers a restart before the first new PPO commit.
        model.utility_calibration=dict(ready=False,reward_id=digest(model.config.utility),
            version='reviewed-source-utility-calibration/1.0.0',
            reason='Reviewed source continuation requires fresh ordinary utility calibration',
            checkpoint_signature=raw['training'].get('signature'),current_signature=digest(signature))
    return model, raw, changed


def evaluation_directory(root, run, state, name):
    """Keep old evaluation evidence intact across an approved source change.

    A committed learner can still have an interrupted or unpublished screen.
    Its old spec must not be silently relabeled with the current implementation,
    nor should it prevent the checkpoint from continuing after a reviewed fix.
    Use a deterministic fresh namespace; the caller still verifies every other
    field of its normal evaluation specification before collecting/reusing data.
    """
    require(isinstance(name,str) and name not in ('','.','..') and Path(name).name==name,
            'Expected a single evaluation directory name')
    directory=safe(run,run/'evaluations'/name)
    spec=directory/'spec.json'
    if not spec.is_file():return directory
    previous=read(spec);current=implementation_id()
    if previous.get('implementation')==current:return directory
    signature=state['signature']
    require(signature['implementation']==current,'Evaluation run has not adopted the current reviewed source')
    # Revalidate actual recorded transitions against pinned review receipts.
    # Several source upgrades can occur before a pending screen is retried.
    known={digest(signature):signature}
    for transition in reversed(state.get('source_transitions',[])):
        target=known.get(transition['to_signature'])
        if target is None:continue
        origin=transition['from_signature']
        check_signature(dict(state,signature=origin),target,reviewed=True)
        known[digest(origin)]=origin
    require(previous.get('implementation') in {s['implementation'] for s in known.values()},
            'Existing evaluation source is not in this run\'s reviewed transition history')
    return safe(run,run/'evaluations'/f'{name}-source-{current[:16]}')


def commit_transition(run, state, signature):
    """Call only after restoring Adam and RNG successfully, under the run lock."""
    previous = copy.deepcopy(state)
    backup = safe(run, run / f"state-before-source-{digest(previous['signature'])[:16]}-{state['iteration']:04d}.json")
    if backup.exists():
        require(read(backup) == previous, 'Source transition backup differs')
    else:
        write_json(backup, previous)
    updated = copy.deepcopy(state)
    updated.setdefault('source_transitions', []).append(dict(
        from_signature=previous['signature'], to_signature=digest(signature),
        checkpoint=copy.deepcopy(state['history'][-1]), iteration=state['iteration'],
        games_done=state['games_done'], backup=backup.name,
        config_changes={k:dict(before=previous['signature']['config'].get(k),after=v)
                        for k,v in signature['config'].items() if previous['signature']['config'].get(k)!=v}))
    updated['signature'] = signature
    from .config import TIME_CONFIG_MIGRATION,time_budget_config
    if signature['config']==time_budget_config(previous['signature']['config']):
        updated['source_transitions'][-1]['config_migration']=TIME_CONFIG_MIGRATION
    from .curriculum import activate_boost
    activate_boost(updated,signature['config'])
    if 'legacy_boost' in updated and 'legacy_boost' not in previous:
        updated['source_transitions'][-1]['legacy_boost']=copy.deepcopy(updated['legacy_boost'])
    old_interval=previous['signature']['config']['search_every_games']
    interval=signature['config']['search_every_games']
    if interval and interval!=old_interval:
        warmup=signature['config']['search_ready_games'];done=state['games_done']
        # Adopt the denser future cadence without replaying past allocations or
        # resetting the warm-up. An already overdue reservation stays due.
        future=warmup if done<warmup else warmup+((done-warmup)//interval+1)*interval
        updated['next_search_games']=min(state['next_search_games'],future)
        updated['source_transitions'][-1]['search_schedule']=dict(
            before=state['next_search_games'],after=updated['next_search_games'])
    write_json(safe(run, run / 'state.json'), updated)
    print(f"Reviewed same-run continuation: generation {state['iteration']}, {state['games_done']} games; "
          'checkpoint, Adam/RNG, population and schedule counters preserved', flush=True)
    changes=updated['source_transitions'][-1]['config_changes']
    if changes:
        print(f'Reviewed configuration changes: {changes}',flush=True)
    return updated
