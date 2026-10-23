"""Explicit same-run continuation across a pinned, reviewed source transition."""
import copy
from pathlib import Path

from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import CONFIG_DIR, digest, project_path, require
from brass_ai.learning.model import ModelConfig
from brass_ai.step5.league import write_json, implementation_id
from .retention import safe
from .source_reviews import expand_source_reviews


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
    if 'learning_rate_warmup' in config:
        from .learning_rate import configure
        warmup=config['learning_rate_warmup']
        configure(optimizer,config['learning_rate'],warmup['selfplay_steps'],warmup['start_factor'],'selfplay')
    if 'ppo_lr_controller' in config:
        from .lr_controller import configure
        configure(optimizer,config['ppo_lr_controller'])
    return dict(before=before,after=[dict(lr=g['lr'],weight_decay=g['weight_decay']) for g in optimizer.param_groups],
                preserved='Adam moments, steps, other group options and RNG')


def check_signature(state, signature, reviewed=False):
    require(state['format'] == 'step5-recovery-league/2.2.0', 'Unsupported self-play resume format')
    previous = state['signature']
    # The latest-model recipe starts from completed IL, never a PPO generation
    # or an old run's population. It also cannot later resume as a gate recipe.
    require(previous['config'].get('selfplay_recipe')==signature['config'].get('selfplay_recipe'),
            'Changing self-play recipe requires a fresh IL run')
    from .config import latest_model_recipe
    if latest_model_recipe(signature['config']):
        require(not any(k in state for k in ('champion','champion_gate','champion_mix')),
                'Latest-model run contains forbidden champion state')
        from .curriculum import boost_window
        boost_window(state,signature['config'])
    if previous == signature:
        return False
    require(reviewed, 'Run configuration/source changed. For the reviewed selfplay-v2 source only, '
            'repeat with --resume-reviewed-source; otherwise use a fresh directory')
    unchanged = lambda s: {k: v for k, v in s.items() if k not in ('implementation', 'config')}
    require(unchanged(previous) == unchanged(signature),
            'Reviewed resume requires unchanged config, data/IL, opponents and learning contracts')
    approvals = expand_source_reviews(read(CONFIG_DIR / 'selfplay-resume-sources.json'))
    transitions = [e for e in approvals['transitions'] if e['from_implementation'] == previous['implementation']
                and e['to_implementation'] == signature['implementation']
                and e['learning_compatibility'] == digest(signature['compatibility'])]
    require(transitions, 'This self-play source transition has not been reviewed')
    config_ok = previous['config'] == signature['config']
    for transition in transitions:
        # An exact unchanged configuration needs no historical upgrade path.
        if config_ok:break
        if transition.get('local_scoring_changes'):
            from .local_scoring_adoption import compatible
            config_ok=compatible(state,signature['config'])
            if config_ok:break
        if transition.get('teacher_value_changes'):
            # Only this separately versioned opt-in block may vary. Explicit
            # reviewed continuation still rejects pending old collections and
            # preserves every unrelated optimizer/curriculum/search setting.
            from .config import teacher_value_config
            teacher_value_config(previous['config'].get('teacher_value'))
            teacher_value_config(signature['config'].get('teacher_value'))
            without=lambda c:{k:v for k,v in c.items() if k!='teacher_value'}
            config_ok |= without(previous['config'])==without(signature['config'])
            if config_ok:break
        changes = transition.get('config_changes', {})
        if all(previous['config'].get(k) == v['from'] for k, v in changes.items()):
            expected = dict(previous['config'], **{k: v['to'] for k, v in changes.items()})
            migrations=transition.get('config_migrations',([transition['config_migration']] if 'config_migration' in transition else []))
            from .config import (TIME_CONFIG_MIGRATION,time_budget_config,
                                 CHAMPION_MIX_CONFIG_MIGRATION,champion_mix_config,
                                 ARCHIVED_EARLY_RAIL_TIME_CONFIG_MIGRATION,archived_early_rail_time_budget_config,
                                 ARCHIVED_PHASE_TIME_CONFIG_MIGRATION,archived_phase_time_budget_config,
                                 ARCHIVED_TIME_CONFIG_MIGRATION,archived_time_budget_config)
            handlers={TIME_CONFIG_MIGRATION:time_budget_config,
                      CHAMPION_MIX_CONFIG_MIGRATION:champion_mix_config,
                      ARCHIVED_EARLY_RAIL_TIME_CONFIG_MIGRATION:archived_early_rail_time_budget_config,
                      ARCHIVED_PHASE_TIME_CONFIG_MIGRATION:archived_phase_time_budget_config,
                      ARCHIVED_TIME_CONFIG_MIGRATION:archived_time_budget_config}
            from .gae import CONFIG_MIGRATION as EV_MIGRATION,evidence_config
            handlers[EV_MIGRATION]=evidence_config
            from .lr_controller import THRESHOLD_MIGRATION,threshold_config
            if THRESHOLD_MIGRATION in migrations:
                require(state['iteration']>=560,'LR threshold hotfix starts from committed generation 560 or later')
            handlers[THRESHOLD_MIGRATION]=threshold_config
            from .quality_config import CONFIG_MIGRATION as PFSP_MIGRATION,sampling_config
            handlers[PFSP_MIGRATION]=sampling_config
            from .config import INSOLVENCY_EVENT_CONFIG_MIGRATION,insolvency_event_config
            handlers[INSOLVENCY_EVENT_CONFIG_MIGRATION]=insolvency_event_config
            from .config import INSOLVENCY_010_CONFIG_MIGRATION,insolvency_010_config
            handlers[INSOLVENCY_010_CONFIG_MIGRATION]=insolvency_010_config
            from .config import INSOLVENCY_020_CONFIG_MIGRATION,insolvency_020_config
            handlers[INSOLVENCY_020_CONFIG_MIGRATION]=insolvency_020_config
            from .config import VP_CAP_CONFIG_MIGRATION,vp_cap_config
            handlers[VP_CAP_CONFIG_MIGRATION]=vp_cap_config
            from .value_phase import CONFIG_MIGRATION as PPG_MIGRATION,refined_options
            handlers[PPG_MIGRATION]=refined_options
            from .value_phase import INTERVAL_MIGRATION,interval_options
            from .gae import HEALTH_MIGRATION,health_options
            handlers[INTERVAL_MIGRATION]=interval_options
            handlers[HEALTH_MIGRATION]=health_options
            from .value_phase import DISABLE_MIGRATION,disabled_options
            handlers[DISABLE_MIGRATION]=disabled_options
            from .config import PPO_MICROBATCH_CONFIG_MIGRATION,ppo_microbatch_config
            handlers[PPO_MICROBATCH_CONFIG_MIGRATION]=ppo_microbatch_config
            from .config import THROUGHPUT_CONFIG_MIGRATION,throughput_config
            handlers[THROUGHPUT_CONFIG_MIGRATION]=throughput_config
            from .critic_data import FLOOR_MIGRATION,floor_config
            handlers[FLOOR_MIGRATION]=floor_config
            from .critic_data import FIXED_MIGRATION,fixed_config
            handlers[FIXED_MIGRATION]=fixed_config
            from .ppg_config import MIGRATION as PPG_REPLAY_MIGRATION,configure as configure_ppg_replay
            handlers[PPG_REPLAY_MIGRATION]=configure_ppg_replay
            from .ppg_config import SHORT_MIGRATION,short_options
            handlers[SHORT_MIGRATION]=short_options
            from .ppg_config import COVERAGE_MIGRATION,coverage_options
            handlers[COVERAGE_MIGRATION]=coverage_options
            from .ppg_config import FIXED_MIGRATION,fixed_options
            handlers[FIXED_MIGRATION]=fixed_options
            from .ppg_phasic import MIGRATION as PHASIC_MIGRATION,configure as phasic_config
            handlers[PHASIC_MIGRATION]=phasic_config
            from .dcpg_config import MIGRATION as DCPG_MIGRATION,configure as dcpg_config
            handlers[DCPG_MIGRATION]=dcpg_config
            from .opening_objective import (MIGRATION as OPENING_MIGRATION,configure as configure_opening,
                ZERO_EXPLORATION_MIGRATION,zero_exploration)
            handlers[OPENING_MIGRATION]=configure_opening
            handlers[ZERO_EXPLORATION_MIGRATION]=zero_exploration
            from .v4_share import (MIGRATION as V4_SHARE_MIGRATION,configure as configure_v4_share,
                LEGACY_MIGRATION as OLD_V4_SHARE_MIGRATION,configure_legacy as configure_old_v4_share)
            handlers[V4_SHARE_MIGRATION]=configure_v4_share
            handlers[OLD_V4_SHARE_MIGRATION]=configure_old_v4_share
            from .panel_gates import MIGRATION as PANEL_MIGRATION,configure as configure_panels
            handlers[PANEL_MIGRATION]=configure_panels
            from .policy_anchor import MIGRATION as ANCHOR_MIGRATION,configure as configure_anchor
            handlers[ANCHOR_MIGRATION]=configure_anchor
            from .exploration import MIGRATION as EXPLORATION_MIGRATION,continuation_config
            handlers[EXPLORATION_MIGRATION]=continuation_config
            from .exploration import FROZEN_MIGRATION,frozen_config
            handlers[FROZEN_MIGRATION]=frozen_config
            from .exploration import GROUP_FLOOR_MIGRATION,group_floor_config
            handlers[GROUP_FLOOR_MIGRATION]=group_floor_config
            from .exploration import ANCHOR_FLOOR_MIGRATION,anchor_floor_config
            handlers[ANCHOR_FLOOR_MIGRATION]=lambda c:anchor_floor_config(c,state)
            from .exploration import HIGH_BRANCH_MIGRATION,high_branch_config
            handlers[HIGH_BRANCH_MIGRATION]=high_branch_config
            for migration in migrations:
                require(migration in handlers,'Unknown reviewed configuration migration')
                expected=handlers[migration](expected)
            if transition.get('teacher_value_changes'):
                expected.pop('teacher_value',None)
                if 'teacher_value' in signature['config']:expected['teacher_value']=signature['config']['teacher_value']
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
    from .utility import reward_spec,reviewed_reward_upgrade
    from brass_ai.learning.utility_model import utility_config
    expected=ModelConfig.preset(signature['config']['model_size'])
    target_reward=reward_spec(signature['config'])
    architecture_ok=model.config in (expected,utility_config(expected,target_reward))
    upgrading=reviewed_reward_upgrade(getattr(model.config,'utility',None),target_reward)
    if upgrading:
        require(model.config==utility_config(expected,model.config.utility),'Reward upgrade architecture differs')
        # The original reward must be bound either to the reviewed source we
        # are leaving or to an exact, pinned transition on this same checkpoint.
        known={digest(state['signature']):state['signature']}
        for transition in reversed(state.get('source_transitions',[])):
            target=known.get(transition['to_signature'])
            if target is None or transition['checkpoint']!=entry:continue
            origin=transition['from_signature']
            check_signature(dict(state,signature=origin),target,reviewed=True)
            known[digest(origin)]=origin
        architecture_ok=any(reward_spec(s['config'])==model.config.utility for s in known.values())
        require(architecture_ok,'Checkpoint reward upgrade has no reviewed origin')
        model._reviewed_cap_upgrade=True
    require(architecture_ok and raw['training'].get('fresh_run') and project_path(root, raw['training']['fresh_run']) == run,
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
        if 'champion_mix' in state:frozen_entries.extend(state['champion_mix'][k] for k in ('anchor','previous'))
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
    return evidence_directory(root,run,state,run/'evaluations'/name)


def evidence_directory(root,run,state,directory):
    """Isolate pending evaluation evidence after a verified source transition.

    Rating slots use the same guard as monitor/champion screens. Matching
    sources still pass through their normal immutable specification checks.
    """
    directory=safe(run,directory)
    spec=safe(run,directory/'spec.json')
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
    return safe(run,directory.with_name(f'{directory.name}-source-{current[:16]}'))


def prepare_opponent_pool(root,run,state,raw,config,source_changed=False):
    """Bind either the generation's pool or its exact atomic migration receipt."""
    from . import quality_pool,pfsp_pool
    pool=state['quality_pool'];quality_pool.validate_state(pool,pool['config'],state['iteration'])
    actual=quality_pool.state_digest(pool);bound=raw['training'].get('quality_pool_digest')
    if actual!=bound:
        matches=[t for t in state.get('source_transitions',[]) if t['checkpoint']==state['history'][-1]
            and t.get('opponent_pool_migration',{}).get('from_digest')==bound
            and t['opponent_pool_migration']['to_digest']==actual]
        require(matches,'Opponent pool state differs from the committed checkpoint')
        for t in matches:
            backup=read(safe(run,run/t['backup']))
            require(digest(backup['signature'])==digest(t['from_signature']) and
                quality_pool.state_digest(backup['quality_pool'])==bound and
                (backup['iteration'],backup['games_done'])==(state['iteration'],state['games_done']),
                'PFSP source transition backup differs')
            migrated,receipt=pfsp_pool.migrate(backup['quality_pool'],pool['config'],state['iteration'],state['games_done'])
            require(receipt==t['opponent_pool_migration'] and quality_pool.state_digest(migrated)==actual,
                'PFSP source transition cannot be reproduced')
    if pool['config']==config['opponent_pool']:
        if pfsp_pool.pfsp(pool):require(pool['games_done']==state['games_done'],'PFSP ordinary progress differs')
        return pool,None
    require(source_changed,'Opponent sampling changes require a reviewed source/config transition')
    return pfsp_pool.migrate(pool,config['opponent_pool'],state['iteration'],state['games_done'])


def commit_transition(run, state, signature,*,opponent_pool=None,pool_migration=None,local_scoring_adoption=None):
    """Call only after restoring Adam and RNG successfully, under the run lock."""
    if local_scoring_adoption is not None and 'local_scoring_adoption' in state:
        # Before the first companion checkpoint, prepare() reuses the durable
        # adoption receipt to restore the old full checkpoint. A subsequent
        # source fix may publish a new signature, but must not replace or reset
        # that receipt (including cleanup progress and the original PPG clock).
        require(state['local_scoring_adoption']==local_scoring_adoption,
                'Local scoring adoption receipt differs; existing adoption cannot be replaced')
    from .v4_share import migrate_state
    share,share_migration=migrate_state(state,signature['config'])
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
    if local_scoring_adoption is not None and 'local_scoring_adoption' not in updated:
        updated['local_scoring_adoption']=copy.deepcopy(local_scoring_adoption)
    if share_migration is not None:
        updated['v4_share_gate']=share
        updated['source_transitions'][-1]['v4_share_migration']=share_migration
    if pool_migration is not None:
        from .quality_pool import state_digest
        require(opponent_pool is not None and state_digest(previous['quality_pool'])==pool_migration['from_digest']
            and state_digest(opponent_pool)==pool_migration['to_digest'],'PFSP transition pool binding differs')
        updated['quality_pool']=copy.deepcopy(opponent_pool)
        updated['source_transitions'][-1]['opponent_pool_migration']=copy.deepcopy(pool_migration)
    from .config import TIME_CONFIG_MIGRATION,time_budget_config
    if signature['config']==time_budget_config(previous['signature']['config']):
        updated['source_transitions'][-1]['config_migration']=TIME_CONFIG_MIGRATION
    from .config import CHAMPION_MIX_CONFIG_MIGRATION,champion_mix_config
    if (champion_mix_config(previous['signature']['config'])!=previous['signature']['config'] and
            signature['config']==champion_mix_config(time_budget_config(previous['signature']['config']))):
        updated['source_transitions'][-1]['config_migration']=CHAMPION_MIX_CONFIG_MIGRATION
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
