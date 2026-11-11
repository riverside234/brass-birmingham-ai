"""One-time same-run utility widening and A50 -> A100 at a committed boundary."""
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.learning.utility_width import widen,optimizer_state
from brass_ai.step5.league import write_json
from .retention import owned,run_lock,safe

CONTRACT='utility-width-256-stage-a-100/1.0.0'
KEY='utility_width_upgrade'


def configure(config):
    """Copy only the edited dictionary paths; every other recipe setting survives."""
    from .config import validate_config
    from .training_revision_g1491 import WIDE_RESTART,variant_of
    from .search_value_schedule import restart_schedule
    from brass_ai.rust_backend.lab_shadow_config import is_active
    require(is_active(config) and variant_of(config)==WIDE_RESTART and config.get('utility_hidden',32)==32,
        'Utility upgrade requires the g1190 restart with 256 teacher rows')
    phase=config['teacher_phase'];search=phase['search_value'];schedule=search['schedule']
    require(dict(schedule,validation_during_beta_qualification=True)==restart_schedule(),
        'Utility upgrade requires the current A50/B50/C10 restart schedule')
    result=dict(config,utility_hidden=256,teacher_phase=dict(phase,mode='shadow',skip_utility_qualification=True,
        search_value=dict(search,schedule=dict(schedule,stage_a_iterations=100,validation_during_beta_qualification=True))))
    validate_config(result)
    return result


def pending(run,generation):
    """Only outputs of the uncommitted iteration; committed replay stays in place."""
    paths=[run/f'iteration-{generation:04d}']
    paths.extend(run.glob(f'generation-{generation+1:04d}.pt*'))
    paths.extend((run/'teacher-labels').glob(f'generation-{generation:06d}-*'))
    paths.extend((run/'search-values').glob(f'{generation:06d}-*'))
    paths.extend((run/'.ppg-eight-cache').glob(f'generation-{generation+1:06d}.*'))
    return sorted({safe(run,p).relative_to(run).as_posix() for p in paths if p.exists()})


def archive_pending(run,directory,state):
    """Journal then rename, so a Ctrl-C anywhere in this migration is retryable."""
    path=directory/'pending.json';current=pending(run,state['iteration'])
    previous=read(path) if path.exists() else dict(checkpoint=state['history'][-1],paths=[])
    require(previous['checkpoint']==state['history'][-1],'Utility migration archive belongs to another checkpoint')
    paths=sorted(set(previous['paths'])|set(current))
    write_json(path,dict(checkpoint=previous['checkpoint'],paths=paths))
    for relative in paths:
        source=safe(run,run/relative);destination=safe(run,directory/'pending'/relative)
        if source.exists():
            require(not destination.exists(),'Utility migration archive would overwrite pending work')
            destination.parent.mkdir(parents=True,exist_ok=True)
            source.rename(destination)
        else:require(destination.exists(),'Utility migration pending archive is incomplete')
    return paths


def prepare(root,run,*,expected_generation):
    """Publish a widened checkpoint before atomically selecting it in state.json."""
    from .resume import load_resume
    from .search_value_iteration import stamp
    from .search_value_training import StageMachine
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.rust_backend.lab_backend import validate_state
    from brass_ai.rust_backend.lab_replay_compat import execution
    root,run=owned(root,run)
    with run_lock(run),torch.random.fork_rng(devices=[]):
        state=read(run/'state.json');generation=state['iteration']
        require(generation==expected_generation,'Run advanced; use the committed generation in state.json')
        model,raw,_=load_resume(root,run,state,state['signature'])
        validate_state(state,raw)
        config=state['signature']['config']
        if config.get('utility_hidden',32)==256:
            require(KEY in state and config['teacher_phase']['search_value']['schedule']['stage_a_iterations']==100,
                'Existing wide utility run has no matching upgrade')
            print(f'Utility head already 256; Stage A budget already 100. Continuing g{generation}.',flush=True)
            return state
        config=configure(config)
        saved=raw['training'].get('search_value')
        require(saved is not None and state.get('search_value')==stamp(saved), 'Search head checkpoint/state differs')
        machine=StageMachine(saved=saved['stage'],options=state['signature']['config']['teacher_phase']['search_value'])
        require(machine.state['stage']=='A','Extend Stage A before the search head enters Stage B')
        shapes=widen(model)
        training=dict(raw['training'])
        phases=training.get('actor_value_phases')
        require(phases is not None,'Missing PPG checkpoint state')
        training['actor_value_phases']=dict(phases,optimizer=optimizer_state(phases['optimizer'],model,shapes))
        # Keep labels, the learned search head, Adam, stage counters, readiness and
        # every replay/evidence clock. Only beta is measured again for this model.
        teacher=dict(state['teacher_phase'],weight={})
        backend={k:v for k,v in state['native_backend'].items() if k!='teacher'}
        backend['config_sha256']=digest(config)
        signature=dict(state['signature'],config=config)
        previous=state['history'][-1]
        directory=safe(run,run/'configuration-history'/f'utility256-{generation:04d}')
        output=safe(run,run/f'generation-{generation:04d}-utility256.pt')
        receipt=dict(contract=CONTRACT,generation=generation,previous_checkpoint=previous,
            from_width=32,to_width=256,stage_a_before=50,stage_a_after=100,
            stage_a_completed=machine.state['stage_a_iterations'],backup=(directory/'state-before.json').relative_to(run).as_posix())
        training.update(signature=digest(signature),teacher_phase=teacher,native_backend=backend,
            native_backend_execution=execution(),**{KEY:receipt})
        checkpoint=dict(raw,model_config=model.config.to_dict(),model_id=model_id(model),
            state_dict={k:v.detach() for k,v in model.state_dict().items()},
            optimizer=optimizer_state(raw['optimizer'],model,shapes),training=training)
        directory.mkdir(parents=True,exist_ok=True)
        backup=directory/'state-before.json'
        if backup.exists():require(read(backup)==state,'Utility migration backup differs from committed state')
        else:write_json(backup,state)
        temporary=output.with_name(output.name+'.tmp')
        torch.save(checkpoint,temporary)
        # Strict shape/config/digest check before retiring any interrupted work.
        verified,_=load_checkpoint(temporary)
        require(verified.config.utility_hidden==256,'Utility checkpoint conversion failed')
        del verified
        temporary.replace(output)
        entry=dict(path=output.relative_to(root).as_posix(),sha256=sha(output),model_id=checkpoint['model_id'],generation=generation)
        updated=dict(state,signature=signature,teacher_phase=teacher,native_backend=backend,
            history=[*state['history'][:-1],entry],**{KEY:receipt})
        updated.pop('native_backend_switch',None)
        if 'resume' in state:updated['resume']=[entry if r==previous else r for r in state['resume']]
        validate_state(updated,checkpoint)
        archived=archive_pending(run,directory,state)
        write_json(run/'state.json',updated)
        print(f'Utility head 32 -> 256 at g{generation}; learned outputs, PPO/PPG Adam and RNG preserved. '
            f'Stage A {machine.state["stage_a_iterations"]}/100 (was /50). '
            f'Archived {len(archived)} pending path(s); measuring beta directly in shadow mode.',flush=True)
        return updated
