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
    try:config=previous_recipe(previous_reward(recipe(config)))
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
    result['config']['teacher_phase']['mode']='shadow'
    result['config']['teacher_phase']['term']['shadow_iterations']=2
    return result


def is_active(config):
    from .lab_teacher_activation import is_on
    return is_shadow(config) or is_on(config)


def signature_transition(before,after):
    from .lab_replay_compat import QUALIFIED,execution
    old,new=before['config'],after['config']
    from brass_ai.learning.post_transfer_reward import previous as previous_reward
    from .lab_teacher_activation import is_on
    from brass_ai.recovery.search_value_schedule import previous_recipe
    old_revised='schedule' in old.get('teacher_phase',{}).get('search_value',{})
    new_revised='schedule' in new.get('teacher_phase',{}).get('search_value',{})
    if old_revised and not new_revised:return False
    try:old_compare,new_compare=previous_recipe(previous_reward(old)),previous_recipe(previous_reward(new))
    except (ValueError,KeyError,TypeError):return False
    if is_on(new):
        config_ok=False
        if is_on(old) or is_shadow(old) and old['teacher_phase']['term']['shadow_iterations']==4:
            expected=copy.deepcopy(old_compare);expected['teacher_phase']['mode']='on'
            expected['teacher_phase']['term']['share_target']=new['teacher_phase']['term']['share_target']
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
    before=previous['native_backend'];after=copy.deepcopy(before)
    after['config_sha256']=digest(updated['signature']['config'])
    after['shadow']=dict(contract=CONTRACT,tables=dict(train=28,held_out=4),
        teacher_build=build['receipt'],generation=previous['iteration'])
    from .lab_teacher_activation import is_on,authorize,CONTRACT as ON
    if is_on(updated['signature']['config']):
        q=authorize(previous,updated['signature']['config'])
        after['teacher']=dict(contract=ON,tables=dict(train=28,held_out=4),gradient_qualification=q['sha256'],
            beta=q['beta'],qualified_generation=q['qualified_generation'],generation=previous['iteration'])
        options=updated['signature']['config']['teacher_phase']['search_value']
        if 'schedule' in options:
            after['teacher'].update(stage_schedule=digest(options['schedule']),ppg_beta_floor=.25)
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
