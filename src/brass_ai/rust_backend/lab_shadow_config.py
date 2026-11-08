"""Exact user-requested branch-off / 32-table shadow continuation."""
import copy

from brass_ai.learning.contracts import digest,require

BASE_CONFIG='895d866502425dad30f2277b822dba2a15247f9a701d2dd5dd10b7e396de3ca8'
CONTRACT='brass-native-shadow-adoption/1.0.0'


def configure(before,share_target=.25):
    from brass_ai.recovery.search_value_config import adopt
    from brass_ai.recovery.config import validate_config
    require(digest(before)==BASE_CONFIG,'Shadow adoption requires the reviewed restart recipe')
    require(type(share_target) in (float,int) and share_target in (.25,.4,.5),'Choose share_target 0.25, 0.4 or 0.5')
    after=adopt(before)
    after['branching']['rate_per_decision']=0.
    after['teacher_phase']['mode']='shadow'
    after['teacher_phase']['search_value']['stage']='A'
    after['teacher_phase']['term']['share_target']=share_target
    validate_config(after)
    return after


def is_shadow(config):
    # Recover the exact reviewed base; no other recipe/search/optimizer edits.
    from brass_ai.recovery.collector_runtime import recipe
    try:config=recipe(config)
    except (ValueError,TypeError):return False
    before=copy.deepcopy(config);phase=before.pop('teacher_phase',None)
    if phase is None:return False
    if 'teacher_value' not in before or 'branching' not in before:return False
    before['teacher_value']['enabled']=True
    before['branching']['rate_per_decision']=.005
    try:return digest(before)==BASE_CONFIG and config==configure(before,phase['term']['share_target'])
    except (ValueError,KeyError,TypeError):return False


def signature_transition(before,after):
    from .lab_replay_compat import QUALIFIED,execution
    old,new=before['config'],after['config']
    if is_shadow(old):
        expected=copy.deepcopy(old);expected['teacher_phase']['term']['share_target']=new.get('teacher_phase',{}).get('term',{}).get('share_target')
        config_ok=is_shadow(new) and expected==new
    else:config_ok=digest(old)==BASE_CONFIG and is_shadow(new)
    return (before.get('implementation')==after.get('implementation')==QUALIFIED and
        {k:v for k,v in before.items() if k!='config'}=={k:v for k,v in after.items() if k!='config'} and
        config_ok and
        execution()['replay_location_review'] is not None)


def approved(state,signature):
    value=state.get('native_backend',{})
    return (value.get('mode')=='rust' and state.get('iteration',0)>=1257 and
        signature_transition(state['signature'],signature))


def rebind_backend(previous,updated):
    """Called by the normal transition only after full optimizer/RNG restore."""
    if not approved(previous,updated['signature']):return
    from .lab_shadow import teacher_build
    build=teacher_build(previous)  # Pin the previously qualified executable.
    before=previous['native_backend'];after=copy.deepcopy(before)
    after['config_sha256']=digest(updated['signature']['config'])
    after['shadow']=dict(contract=CONTRACT,tables=dict(train=28,held_out=4),
        teacher_build=build['receipt'],generation=previous['iteration'])
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
