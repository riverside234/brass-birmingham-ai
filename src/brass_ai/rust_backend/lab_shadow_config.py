"""Exact user-requested branch-off / 32-table shadow continuation."""
import copy

from brass_ai.learning.contracts import digest,require

BASE_CONFIG='895d866502425dad30f2277b822dba2a15247f9a701d2dd5dd10b7e396de3ca8'
CONTRACT='brass-native-shadow-adoption/1.0.0'


def configure(before,share_target=.25):
    from brass_ai.recovery.search_value_config import adopt
    from brass_ai.recovery.config import validate_config
    require(digest(before)==BASE_CONFIG,'Shadow adoption requires the reviewed restart recipe')
    from brass_ai.recovery.teacher_gradient_qualification import valid_target
    require(valid_target(share_target),'Choose a share_target above 0 and at most 1')
    after=adopt(before)
    after['branching']['rate_per_decision']=0.
    after['teacher_phase']['mode']='shadow'
    after['teacher_phase']['search_value']['stage']='A'
    after['teacher_phase']['term']['share_target']=share_target
    after['teacher_phase']['term']['shadow_iterations']=4
    validate_config(after)
    return after


def is_shadow(config):
    # Recover the exact reviewed base; no other recipe/search/optimizer edits.
    from brass_ai.recovery.collector_runtime import recipe
    from brass_ai.recovery.search_value_schedule import previous_recipe
    from brass_ai.learning.post_transfer_reward import previous as previous_reward
    from brass_ai.recovery.rating_coverage import previous as previous_rating
    from brass_ai.recovery.teacher_phase_config import without_utility_qualification
    from brass_ai.recovery.ordinary_cotton import previous as previous_tables
    from brass_ai.recovery.training_revision_g2090 import previous_opening
    from brass_ai.recovery.training_revision_g3140 import previous as before_value_revision
    from brass_ai.recovery.rating_cadence import previous as before_rating_cadence
    from brass_ai.recovery.training_revision_g3270 import previous as before_teacher_revision
    from brass_ai.recovery.training_revision_g3423 import previous as before_soft_policy
    from brass_ai.recovery.training_revision_g3658 import previous as before_eighth_floor
    from brass_ai.recovery.training_revision_g4138 import previous as before_teacher_search
    from brass_ai.recovery.training_revision_g4475 import previous as before_teacher_training
    from brass_ai.recovery.training_revision_g4551 import previous as before_table_revision
    from brass_ai.recovery.training_revision_g4770 import previous as before_era_head_revision
    try:
        config=previous_tables(previous_opening(before_value_revision(before_rating_cadence(
            before_teacher_revision(before_soft_policy(before_eighth_floor(before_teacher_search(
                before_teacher_training(before_table_revision(before_era_head_revision(recipe(config))))))))))))
        config=previous_rating(previous_recipe(previous_reward(without_utility_qualification(config))))
    except (ValueError,TypeError):return False
    before=copy.deepcopy(config);phase=before.pop('teacher_phase',None)
    if phase is None:return False
    if 'teacher_value' not in before or 'branching' not in before:return False
    before['teacher_value']['enabled']=True
    before['branching']['rate_per_decision']=.005
    try:
        expected=configure(before,phase['term']['share_target'])
        if type(phase['term']['shadow_iterations']) is not int or phase['term']['shadow_iterations'] not in (2,4):return False
        expected['teacher_phase']['term']['shadow_iterations']=phase['term']['shadow_iterations']
        return digest(before)==BASE_CONFIG and config==expected
    except (ValueError,KeyError,TypeError):return False


def qualification_upgrade(before,after):
    """Only lengthen diagnostic measurements; keep pending games and labels."""
    if not is_shadow(before) or not is_shadow(after):return False
    expected=copy.deepcopy(before)
    if expected['teacher_phase']['term']['shadow_iterations']!=2:return False
    expected['teacher_phase']['term']['shadow_iterations']=4
    return expected==after


def collection_signature(signature):
    """Diagnostic window/gradient mode do not change frozen collection or RNG."""
    if not is_active(signature['config']):return signature
    result=copy.deepcopy(signature)
    from brass_ai.recovery.rating_cadence import KEY as RATING_CADENCE
    result['config'].pop(RATING_CADENCE,None)  # rating cost only; no collection setting
    phase=result['config']['teacher_phase']
    phase['mode']='shadow'
    phase['term']['shadow_iterations']=2
    from brass_ai.recovery.adaptive_teacher_beta import KEY as ADAPTIVE_BETA
    phase.pop(ADAPTIVE_BETA,None)
    return result


def is_active(config):
    from .lab_teacher_activation import is_on
    return is_shadow(config) or is_on(config)


def signature_transition(before,after):
    from .lab_replay_compat import QUALIFIED,execution
    old,new=before['config'],after['config']
    from brass_ai.recovery import training_revision_g2090 as g2090
    from brass_ai.recovery import training_revision_g2240 as g2240
    from brass_ai.recovery import training_revision_g2400 as g2400
    if g2090.applied(old) != g2090.applied(new):
        if g2090.applied(old):return False
        try:old=g2090.configure(old)
        except (ValueError, KeyError, TypeError):return False
    if g2240.applied(old) != g2240.applied(new):
        if g2240.applied(old):return False
        try:old=g2240.configure(old)
        except (ValueError, KeyError, TypeError):return False
    if g2400.applied(old) != g2400.applied(new):
        if g2400.applied(old):return False
        try:old=g2400.configure(old)
        except (ValueError, KeyError, TypeError):return False
    from brass_ai.recovery import training_revision_g2654 as g2654
    if g2654.applied(old) != g2654.applied(new):
        if g2654.applied(old):return False
        try:old=g2654.configure(old)
        except (ValueError, KeyError, TypeError):return False
    # g3471: cotton exploration timed within each era; g3528 retimes it (from g3471 or g2654); then kept.
    from brass_ai.recovery import training_revision_g3471 as g3471
    from brass_ai.recovery import training_revision_g3528 as g3528
    if g3471.applied(old) != g3471.applied(new) and not g3528.applied(new):
        if g3471.applied(old):return False
        try:old=g3471.configure(old)
        except (ValueError, KeyError, TypeError):return False
    if g3528.applied(old) != g3528.applied(new):
        if g3528.applied(old):return False
        try:old=g3528.configure(old)
        except (ValueError, KeyError, TypeError):return False
    from brass_ai.recovery import training_revision_g2880 as g2880
    if g2880.applied(old) != g2880.applied(new):
        if g2880.applied(old):return False
        try:old=g2880.configure(old)
        except (ValueError, KeyError, TypeError):return False
    # g3658: the PPG beta floor .25 -> .125 (value phase and search-head Stage C), adopted once; then kept.
    from brass_ai.recovery import training_revision_g3658 as g3658
    if g3658.applied(old) != g3658.applied(new):
        if g3658.applied(old):return False
        try:old=g3658.configure(old)
        except (ValueError, KeyError, TypeError):return False
    # g3140 value revision: its items and the local-scoring weight ladder only move forward.
    from brass_ai.recovery import training_revision_g3140 as g3140
    old=g3140.project(old,new)
    if old is None:return False
    # g3170 rating cadence: adopted once, then kept; it changes only how often ratings run.
    from brass_ai.recovery import rating_cadence
    old=rating_cadence.project(old,new)
    if old is None:return False
    # g3270 teacher revision (build steps 2-3): adopted once, then kept. It changes the
    # collection (40 tables), so pending work is archived, not preserved.
    from brass_ai.recovery import training_revision_g3270 as g3270
    old=g3270.project(old,new)
    if old is None:return False
    # g3423 soft policy head (S1): adopted once, then kept; trainer-only, no collection setting.
    from brass_ai.recovery import training_revision_g3423 as g3423
    old=g3423.project(old,new)
    if old is None:return False
    # g4138 teacher search revision: 44+4 tables, short search 128 at C_scale .6; adopted once, then kept.
    # It changes the collection, so pending work is archived, not preserved.
    from brass_ai.recovery import training_revision_g4138 as g4138
    old=g4138.project(old,new)
    if old is None:return False
    # g4475 teacher training revision: 160 teacher rows per PPO step, search-value head paused; adopted once,
    # then kept; trainer-only, no collection setting.
    from brass_ai.recovery import training_revision_g4475 as g4475
    old=g4475.project(old,new)
    if old is None:return False
    # g4551 table revision: standard PPO self-play tables, 50+4 teacher tables, 176 teacher rows per PPO step;
    # adopted once, then kept. It changes the collection, so pending work is archived, not preserved.
    from brass_ai.recovery import training_revision_g4551 as g4551
    old=g4551.project(old,new)
    if old is None:return False
    # g4770 era-head revision: Rail-end heads restored (PPG only), PPO-game rows on actual maps, the search pool a
    # 4-iteration sample of the last 8; adopted once, then kept; trainer-only (the local-scoring and value-revision
    # changes it makes pass g3140.project above).
    from brass_ai.recovery import training_revision_g4770 as g4770
    old=g4770.project(old,new)
    if old is None:return False
    # Width changes need a converted checkpoint, not an ordinary recipe switch.
    if old.get('utility_hidden',32)!=new.get('utility_hidden',32):return False
    from brass_ai.recovery.ordinary_cotton import KEY as COTTON, LIMITED, NETWORK, RAIL_NETWORK, RETIMED, TIMED
    old_cotton, new_cotton = (c.get('table_plan', {}).get(COTTON) for c in (old, new))
    if old_cotton != new_cotton or (new_cotton in (LIMITED, NETWORK, RAIL_NETWORK, TIMED, RETIMED)) != g2240.applied(new):
        return False
    from brass_ai.learning.post_transfer_reward import previous as previous_reward
    from .lab_teacher_activation import is_on
    from brass_ai.recovery.search_value_schedule import previous_recipe
    old_revised='schedule' in old.get('teacher_phase',{}).get('search_value',{})
    new_revised='schedule' in new.get('teacher_phase',{}).get('search_value',{})
    if old_revised and not new_revised:return False
    from brass_ai.recovery.training_revision_g1294 import compatible
    from brass_ai.recovery.training_revision_g1307 import compatible as lr_floor_compatible
    from brass_ai.recovery import training_revision_g1491 as g1491
    if not compatible(old,new) or not lr_floor_compatible(old,new) or not g1491.compatible(old,new):return False
    # The projection drops the schedule, so its Stage A length must not change here.
    # Nor may its Stage B length, Stage A fit or PPG floor (the g1190 restart schedule).
    def stages(config):
        options=config.get('teacher_phase',{}).get('search_value',{})
        schedule={k:v for k,v in options.get('schedule',{}).items() if k!='validation_during_beta_qualification'}
        return schedule,options.get('head'),options.get('max_epochs')
    if old_revised and stages(old)!=stages(new):return False
    from brass_ai.recovery.teacher_phase_config import without_utility_qualification
    try:old_compare,new_compare=(previous_recipe(previous_reward(without_utility_qualification(c))) for c in (old,new))
    except (ValueError,KeyError,TypeError):return False
    if is_on(new):
        config_ok=False
        if is_on(old) or is_shadow(old) and old['teacher_phase']['term']['shadow_iterations']==g1491.shadow_iterations(old):
            expected=copy.deepcopy(old_compare);expected['teacher_phase']['mode']='on'
            expected['teacher_phase']['term']['share_target']=new['teacher_phase']['term']['share_target']
            config_ok=expected==new_compare
    elif g1491.applied(new) and not g1491.applied(old):
        # The g1491 revision re-measures beta: the on recipe goes back to a two-iteration shadow.
        # Its g1490 restart variant is adopted only by that branch restart, never in a run.
        config_ok=False
        if is_on(old) and is_shadow(new) and g1491.variant_of(new) is None:
            expected=copy.deepcopy(old_compare);expected['teacher_phase']['mode']='shadow'
            expected['teacher_phase']['term'].update(share_target=new['teacher_phase']['term']['share_target'],
                shadow_iterations=g1491.SHADOW_ITERATIONS)
            config_ok=expected==new_compare
    elif is_shadow(old):
        expected=copy.deepcopy(old_compare);expected['teacher_phase']['term']['share_target']=new.get('teacher_phase',{}).get('term',{}).get('share_target')
        if old['teacher_phase']['term']['shadow_iterations']==2:
            expected['teacher_phase']['term']['shadow_iterations']=new.get('teacher_phase',{}).get('term',{}).get('shadow_iterations')
        config_ok=is_shadow(new) and expected==new_compare
    else:config_ok=digest(old)==BASE_CONFIG and is_shadow(new)
    return (before.get('implementation')==after.get('implementation')==QUALIFIED and
        {k:v for k,v in before.items() if k!='config'}=={k:v for k,v in after.items() if k!='config'} and
        config_ok and
        execution()['replay_location_review'] is not None)


def approved(state,signature):
    value=state.get('native_backend',{})
    accepted=(value.get('mode')=='rust' and state.get('iteration',0)>=1257 and
        signature_transition(state['signature'],signature))
    if accepted:
        from brass_ai.learning.post_transfer_reward import previous as previous_reward,transition,authorized
        before,after=state['signature']['config'],signature['config']
        if (before.get('reward'),before.get('insolvency'))!=(after.get('reward'),after.get('insolvency')):
            if not (transition(before,after) and authorized(state,signature)):return False
        from .lab_teacher_activation import is_on,authorize
        if is_on(signature['config']):authorize(state,signature['config'])
    return accepted


def preserves_pending(state,signature):
    """Reviewed diagnostic/on transitions retain the identical frozen collection.

    The loader still checks checkpoint/hash/lineage and unfinished checkpoints;
    collectors revalidate each cached table/family normally. Qualification and
    identical collection bindings exclude unreviewed actor/source/target edits.
    """
    from .lab_teacher_activation import is_on
    return (approved(state,signature) and
        (qualification_upgrade(state['signature']['config'],signature['config']) or is_on(signature['config'])) and
        collection_signature(state['signature'])==collection_signature(signature))


def rebind_backend(previous,updated):
    """Called by the normal transition only after full optimizer/RNG restore."""
    if not approved(previous,updated['signature']):return
    from .lab_shadow import teacher_build
    build=teacher_build(previous)  # Pin the previously qualified executable.
    from brass_ai.learning.post_transfer_reward import is_revised
    if is_revised(updated['signature']['config']):
        from brass_ai.learning.contracts import SOURCE_ROOT
        from .lab_reward_runtime import resolve
        if not is_revised(previous['signature']['config']):build=resolve(SOURCE_ROOT,build)
    from brass_ai.recovery.training_revision_g1491 import tables
    before=previous['native_backend'];after=copy.deepcopy(before)
    after['config_sha256']=digest(updated['signature']['config'])
    after['shadow']=dict(contract=CONTRACT,tables=tables(updated['signature']['config']),
        teacher_build=build['receipt'],generation=previous['iteration'])
    from .lab_teacher_activation import is_on,authorize,CONTRACT as ON
    if is_on(updated['signature']['config']):
        q=authorize(previous,updated['signature']['config'])
        after['teacher']=dict(contract=ON,tables=tables(updated['signature']['config']),gradient_qualification=q['sha256'],
            beta=q['beta'],qualified_generation=q['qualified_generation'],generation=previous['iteration'])
        options=updated['signature']['config']['teacher_phase']['search_value']
        if 'schedule' in options:
            after['teacher'].update(stage_schedule=digest(options['schedule']),
                ppg_beta_floor=updated['signature']['config']['value_phase']['beta_clone_min'])
    else:
        from brass_ai.recovery.training_revision_g1491 import applied
        # A g1491 re-calibration retires the old fixed beta until the next reviewed switch.
        if applied(updated['signature']['config']):after.pop('teacher',None)
    updated['native_backend']=after
    updated['native_backend_switch']=dict(checkpoint=copy.deepcopy(previous['history'][-1]),
        before=copy.deepcopy(before),after=copy.deepcopy(after))


def archive_pending(root,run,state):
    """Lossless one-time rename before changing frozen collection semantics."""
    from brass_ai.bank import read,sha
    from brass_ai.recovery.retention import owned,safe
    from brass_ai.step5.league import write_json
    root,run=owned(root,run);generation=state['iteration']
    pending=safe(run,run/f'iteration-{generation:04d}')
    require(not list(run.glob(f'generation-{generation+1:04d}.pt*')),
        'Uncommitted checkpoint exists; recover it before changing the recipe')
    require(not list((run/'teacher-labels').glob(f'generation-{generation:06d}-*')) and
        not list((run/'search-values').glob(f'{generation:06d}-*')),
        'Uncommitted teacher shards exist; recover the current shadow iteration before changing its target')
    if not pending.exists():return None
    entries=[]
    for path in sorted(pending.rglob('*')):
        safe(run,path)
        if path.is_file():entries.append(dict(path=path.relative_to(pending).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    binding=dict(contract=CONTRACT,generation=generation,checkpoint=state['history'][-1],
        signature=digest(state['signature']),files=entries)
    directory=safe(run,run/'configuration-history'/('shadow-'+str(generation)+'-'+digest(binding)[:16]))
    directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/'pending.json'
    require(not receipt.exists() or read(receipt)==binding,'Pending archive receipt changed')
    write_json(receipt,binding)
    backup=directory/'state-before.json'
    require(not backup.exists() or read(backup)==state,'Pending archive state changed')
    write_json(backup,state)
    destination=safe(run,directory/pending.name)
    require(not destination.exists(),'Pending archive would overwrite data')
    pending.rename(destination)
    print('Preserved old-config pending work at '+str(destination),flush=True)
    return str(receipt)
