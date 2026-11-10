"""Restart the Rust teacher run from one committed generation, in a new run folder.

The full checkpoint of that generation may be gone. Its committed state comes from
the verified results export of that generation, and its weights from the retained
rating snapshot (the same checkpoint with only the optimizer, the PPG phase and two
diagnostic payloads removed). The new run keeps the generation count, opponent pool,
ratings, qualified Rust backend, finished entropy anneal and search-head stage, and
starts fresh where only the lost checkpoint held the state:

* new PPO and PPG optimizers (the PPO warmup restarts), empty PPG replay;
* empty teacher labels and a teacher beta measured again over four shadow iterations;
* an empty search-head value buffer, with Stage A counts paused for the first eight
  iterations while it refills.
"""
import copy
import os
from pathlib import Path
import shutil

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import compatibility,digest,project_path,require
from brass_ai.step5.league import write_json
from .retention import owned,safe,active,run_lock

CONTRACT='rust-branch-restart/1.0.0'
RECEIPT='branch-restart.json'
STAGE_A=30
PAUSED_ITERATIONS=8
# Paths that are only records of the source run's own history.
RECORDS=('history','resume','source_transitions','machine_transfer','native_backend_switch')
RECORD_PREFIXES=('/quality_pool/events','/native_backend/adopted_checkpoint')


def recipe(config,*,share_target,stage_a_iterations=STAGE_A):
    """The g1307 recipe without the entropy raise, in shadow mode, with a longer Stage A."""
    from brass_ai.rust_backend.lab_teacher_activation import is_on
    from brass_ai.rust_backend.lab_shadow_config import is_shadow
    from .collector_runtime import recipe as without_runtime
    from .config import validate_config
    from . import entropy_schedule as S,ppg_g210 as P,training_revision_g1307 as lr_floor
    from .search_value_config import max_steps_options
    from .teacher_gradient_qualification import valid_target
    config=without_runtime(config)
    require(is_on(config) and 'schedule' in config['teacher_phase']['search_value'],
        'A branch restart starts from the qualified teacher-on recipe with the stage schedule')
    require(S.RAISE not in config['update'][S.KEY],'The restart recipe has no entropy raise')
    require(valid_target(share_target),'Choose a share target above 0 and at most 1')
    result=copy.deepcopy(config)
    result['teacher_phase']['mode']='shadow'
    result['teacher_phase']['term'].update(share_target=share_target,shadow_iterations=4)
    result=lr_floor.configure(max_steps_options(P.kl_stop_options(result)))
    result['teacher_phase']['search_value']['schedule']['stage_a_iterations']=stage_a_iterations
    validate_config(result)
    require(is_shadow(result),'The restart recipe must pass the reviewed shadow projection')
    return result


def rewrite(value,old,new):
    """Move run-relative strings; tensors and other values are shared, not copied."""
    if isinstance(value,dict):return {k:rewrite(v,old,new) for k,v in value.items()}
    if isinstance(value,list):return [rewrite(v,old,new) for v in value]
    if isinstance(value,str) and value.startswith(old):return new+value[len(old):]
    return value


def receipts(value,prefix,skip=RECORDS,where=''):
    """Every {path, sha256} file receipt inside the run, outside the history records."""
    found=[]
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and value['path'].startswith(prefix) and 'sha256' in value:
            found.append((where,value))
        for k,v in value.items():
            if where=='' and k in skip:continue
            found.extend(receipts(v,prefix,skip,where+'/'+k))
    elif isinstance(value,list):
        for v in value:found.extend(receipts(v,prefix,skip,where+'[]'))
    return found


def place(source,destination,expected):
    """Hard-link (or copy) an immutable file and check its bytes."""
    require(source.is_file() and sha(source)==expected,'Missing/changed source file: '+str(source))
    if destination.exists():
        require(sha(destination)==expected,'Prepared file differs: '+str(destination));return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.tmp')
    temporary.unlink(missing_ok=True)
    try:os.link(source,temporary)
    except OSError:shutil.copyfile(source,temporary)
    try:
        require(sha(temporary)==expected,'Source changed during copy: '+str(source))
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)


def exported(root,snapshot_receipt,relative,expected):
    """The same run-relative file inside the verified export, if it is there with these bytes."""
    where=Path(snapshot_receipt['path']).parent
    path=(where if where.is_absolute() else root/where)/relative
    return path if path.is_file() and sha(path)==expected else None


def gae_state(config,generation,gate):
    """Fresh evidence windows; the committed lambda and per-table gates are kept."""
    from . import gae
    state=gae.initial(config,generation)
    if gate is not None:
        require(gate['generation']==generation,'GAE gate report belongs to another generation')
        state.update(active_lambda=gate['lambda_next'],value_iterations=gate['value_iterations'],
            ev_streak=gate['ev_streak'],healthy_streak=gate['healthy_streak'],stage_origin=gate['stage_origin'])
        for kind,table in gate.get('table_gate',{}).items():
            if kind in state.get('table_enabled',{}) and 'enabled' in table:
                state['table_streaks'][kind]=table['streak'];state['table_enabled'][kind]=table['enabled']
                state['table_origins'][kind]=table['origin']
    return gae.validate_state(state,config,generation)


def ppg_state(config,model,generation,gate):
    """A PPG phase payload at this generation: new Adam, no replay, the recipe's beta floor."""
    from . import critic_data,ppg_g210 as P
    from brass_ai.learning.search_value import ordinary_parameters
    c=config['value_phase'];P.validate(c)
    options={k:copy.deepcopy(config[k]) for k in ('advantage','value_phase','critic_data','seat_descriptors') if k in config}
    critic=dict(critic_data.initial(config['critic_data']),generation=generation)
    critic_data.validate_state(critic,config['critic_data'],generation)
    adam=torch.optim.Adam(ordinary_parameters(model),lr=config['learning_rate']*c['learning_rate_factor'])
    require(not P.two_iteration(c),'The restart recipe uses the every-iteration PPG phase')
    return dict(contract=P.CHECKPOINT,options=options,gae=gae_state(config['advantage'],generation,gate),
        critic_data=critic,beta_clone=c['beta_clone_min'],kl_trips=[],optimizer=adam.state_dict(),
        rolling_buffer=[],cache_origin=generation,adoption=None)


def ppo_optimizer(config,model):
    """A new AdamW exactly as a new run builds it, with its warmup and LR controller."""
    from brass_ai.learning.search_value import ordinary_parameters
    from .learning_rate import configure as warmup
    from .lr_controller import configure as controller
    optimizer=torch.optim.AdamW(ordinary_parameters(model),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    w=config['learning_rate_warmup'];warmup(optimizer,config['learning_rate'],w['selfplay_steps'],w['start_factor'],'selfplay')
    controller(optimizer,config['ppo_lr_controller'])
    return optimizer.state_dict()


def search_payload(saved,generation,identity):
    """Empty value buffer; Stage A counts pause until eight new ages exist after the first one."""
    value=dict(saved,buffer=dict(saved['buffer'],receipts=[],retired=[]),label_ages=[])
    require(saved['generation']==generation and saved['stage']['stage']=='A' and saved['stage']['w']==saved['stage']['scale']==0,
        'The restart keeps the search head in Stage A')
    # The rebuild rule needs ages origin..origin+7. Starting it one generation later keeps
    # the counter paused through the first eight iterations, as requested.
    value['stage']=dict(saved['stage'],buffer_rebuild=dict(origin_generation=generation+1,required_ages=8,
        ready_generation=None,instruction_id=identity))
    return value


def snapshot_state(root,source,snapshot):
    """The committed state of the export, checked against its receipt and the source lineage."""
    snapshot=project_path(root,snapshot) if not Path(snapshot).is_absolute() else Path(snapshot)
    manifest=read(snapshot/'export-manifest.json');run=source.relative_to(root).as_posix()
    require(manifest['run']==run,'Snapshot export belongs to another run')
    entry=next((f for f in manifest['files'] if f['path']=='state.json'),None)
    require(entry is not None and entry['source']==run+'/state.json' and sha(snapshot/'state.json')==entry['sha256'],
        'Snapshot state differs from its export receipt')
    state=read(snapshot/'state.json');generation=state['iteration']
    require(manifest['committed_generation']==generation,'Snapshot generation differs from its export')
    report=next((f for f in manifest['files'] if f['path']==f'iteration-{generation-1:04d}/report.json'),None)
    gate=None
    if report is not None:
        require(sha(snapshot/report['path'])==report['sha256'],'Snapshot report differs from its export receipt')
        gate=read(snapshot/report['path'])['actor']['gae']['gate']
    current=read(safe(source,source/'state.json'))
    require(current['signature']['compatibility']==compatibility()==state['signature']['compatibility'],
        'Source learning layout differs')
    require(current['iteration']>=generation and current['history'][:len(state['history'])]==state['history'],
        'Snapshot is not a committed generation of the source run')
    return state,gate,dict(path=(snapshot/'state.json').relative_to(root).as_posix() if snapshot.is_relative_to(root) else str(snapshot/'state.json'),
        sha256=entry['sha256'],generation=generation,source_iteration=current['iteration'])


def seed_weights(root,source,state,weights=None):
    """The retained snapshot of the committed checkpoint at this generation."""
    from . import quality_pool
    last=state['history'][-1]
    agent=next((a['checkpoint'] for a in state['ratings']['agents'].values()
        if a.get('kind')=='model' and a['checkpoint']['model_id']==last['model_id']),None)
    require(agent is not None,'No retained snapshot of the restart generation')
    path=safe(source,project_path(root,agent['path']))
    if weights is not None:require(project_path(root,weights)==path,'Weights are not the retained snapshot of this generation')
    require(sha(path)==agent['sha256'],'Restart weights changed')
    raw=torch.load(path,map_location='cpu',weights_only=True)
    require(raw['model_id']==last['model_id'] and raw['step']==last['generation']==state['iteration'] and raw['optimizer'] is None
        and raw['training'].get('signature')==digest(state['signature'])
        and raw['training'].get('fresh_run')==source.relative_to(root).as_posix()
        and raw['training'].get('quality_pool_digest')==quality_pool.state_digest(state['quality_pool'])
        and 'search_value' in raw['training'] and 'teacher_phase' in raw['training'],
        'Restart weights are not the committed checkpoint of that generation')
    return path,agent,raw


def prepare(root,source_run,snapshot,run_path,*,share_target=.1,stage_a_iterations=STAGE_A,weights=None,device='cpu'):
    """Build the new run; the source run and its files are never modified."""
    from . import quality_pool,pfsp_pool
    from .search_value_iteration import stamp
    from .il_kl_probe import import_probe
    from .vp_plot import HISTORY_PATH
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.rust_backend.lab_replay_compat import execution
    root,source=owned(root,source_run);_,run=owned(root,run_path)
    require(source!=run and not run.is_relative_to(source) and not source.is_relative_to(run),
        'The new run must be a separate folder, outside the source run')
    require(not active(source),'Stop the source run before preparing the restart')
    require(not (run/'state.json').exists(),'The new run already has training progress')
    state,gate,snapshot_receipt=snapshot_state(root,source,snapshot)
    generation=state['iteration']
    weights_path,weights_receipt,raw=seed_weights(root,source,state,weights)
    old=source.relative_to(root).as_posix()+'/';new=run.relative_to(root).as_posix()+'/'
    config=recipe(state['signature']['config'],share_target=share_target,stage_a_iterations=stage_a_iterations)
    binding=dict(contract=CONTRACT,source_run=old[:-1],destination_run=new[:-1],generation=generation,
        snapshot=snapshot_receipt,weights=weights_receipt,recipe_sha256=digest(config))
    identity=digest(binding)
    with run_lock(run):
        marker=safe(run,run/RECEIPT)
        if marker.exists():require(read(marker)['binding']==binding,'The new run was prepared from another source or recipe')
        else:
            require(all(p.name=='.active-run.json' for p in run.iterdir()),'The new run folder must be empty')
            write_json(marker,dict(binding=binding,complete=False))
        # The signed recipe keeps its own paths (the reviewed projection binds them).
        fresh={k:rewrite(v,old,new) if k!='signature' else copy.deepcopy(v) for k,v in state.items()}
        fresh['signature']=dict(state['signature'],config=config)
        for key in ('source_transitions',):fresh[key]=[]
        for key in ('machine_transfer','native_backend_switch'):fresh.pop(key,None)
        # Copy every file the committed state still refers to; record what is gone.
        missing_weights=set();missing_probes=[];placed=0
        for where,receipt in receipts(fresh,new):
            if where.startswith(('/il_kl_limit',)+RECORD_PREFIXES):continue
            original=project_path(root,old+receipt['path'][len(new):])
            target=safe(run,project_path(root,receipt['path']))
            if not original.is_file():
                # Superseded files (a compacted rating archive, a retired slot) may still be in the export.
                kept=exported(root,snapshot_receipt,receipt['path'][len(new):],receipt['sha256'])
                if kept is not None:place(kept,target,receipt['sha256']);placed+=1;continue
                if where.startswith('/quality_pool/members') and where.endswith('/probe'):missing_probes.append(receipt['path']);continue
                if where.startswith(('/quality_pool/members','/population')):
                    missing_weights.add(Path(receipt['path']).name);continue
                raise ValueError('The source run no longer holds '+str(original)+' and the export does not have it')
            place(original,target,receipt['sha256']);placed+=1
        pool=fresh['quality_pool'];dropped=[]
        if missing_weights:
            from .v4_history import bot
            gone={m['model_id']:m['generation'] for m in pool['members']+fresh['population']
                if not bot(m) and Path(m['path']).name in missing_weights}
            require(all(g!=0 for g in gone.values()),'The pinned IL opponent is missing')
            dropped=[dict(generation=g,model_id=i) for i,g in sorted(gone.items(),key=lambda x:x[1])]
            pool['members']=[m for m in pool['members'] if m['model_id'] not in gone]
            fresh['population']=[e for e in fresh['population'] if e['model_id'] not in gone]
        # The fixed IL probe moves through its own reviewed import.
        probe,probe_receipt=import_probe(root,run,source,state['signature']['imitation_source'],None,
            config['il_kl_limit'],config['update'])
        fresh['il_kl_limit']=dict(fresh['il_kl_limit'],probe=probe_receipt)
        regenerated=[]
        for member in pool['members']:
            if member.get('probe') is None or member['probe']['path'] not in missing_probes:continue
            with torch.random.fork_rng(devices=[]):
                model,_=load_checkpoint(project_path(root,member['path']),device=device)
                values=probe.probabilities(model,config['update'])
            member['probe']=quality_pool.cache_candidate(root,run,dict(model_id=member['model_id']),values,probe)
            regenerated.append(member['generation'])
        if regenerated or dropped:pfsp_pool.seal(pool)
        quality_pool.validate_state(pool,config['opponent_pool'],generation)
        # Fresh teacher labels and beta; the readiness receipt stays.
        teacher=fresh['teacher_phase']
        teacher.update(weight={},buffer=dict(teacher['buffer'],receipts=[],retired=[]))
        search=search_payload(raw['training']['search_value'],generation,identity)
        fresh['search_value']=stamp(search)
        backend=fresh['native_backend'];backend.pop('teacher',None);backend['config_sha256']=digest(config)
        # Rating history is reporting evidence; keep only what precedes the restart.
        history_file=source/HISTORY_PATH
        if history_file.is_file():
            vp=read(history_file)
            initial={k:fresh['history'][0][k] for k in ('path','sha256','model_id')}
            vp.update(initial_checkpoint=initial,committed_generation=generation,
                iterations=[r for r in vp['iterations'] if r['generation']<=generation])
            write_json(safe(run,run/HISTORY_PATH),vp)
        with torch.random.fork_rng(devices=[]):
            model,_=load_checkpoint(weights_path)
            training=rewrite(raw['training'],old,new)
            training.pop('machine_transfer',None)
            training.update(fresh_run=new[:-1],signature=digest(fresh['signature']),
                quality_pool_digest=quality_pool.state_digest(pool),teacher_phase=copy.deepcopy(teacher),
                search_value=search,native_backend=copy.deepcopy(backend),native_backend_execution=execution(),
                il_kl_limit=copy.deepcopy(fresh['il_kl_limit']),actor_value_phases=ppg_state(config,model,generation,gate))
            require(training.get('exploration')==fresh['exploration'] and
                training.get('startup_initialization')==fresh.get('startup_initialization'),
                'Restart weights disagree with the committed exploration/startup state')
            checkpoint=dict(raw,optimizer=ppo_optimizer(config,model),training=training)
            del model
        output=safe(run,run/f'generation-{generation:04d}.pt');temporary=output.with_name(output.name+'.tmp')
        torch.save(checkpoint,temporary);temporary.replace(output)
        entry=dict(path=output.relative_to(root).as_posix(),sha256=sha(output),model_id=raw['model_id'],generation=generation)
        fresh['history'][-1]=entry;fresh['resume']=[copy.deepcopy(entry)]
        summary=dict(binding=binding,complete=True,identity=identity,checkpoint=entry,files_placed=placed,
            dropped_members=dropped,regenerated_probes=regenerated,
            reset=['ppo_adam_and_warmup','ppg_adam_and_replay','gae_evidence_windows','teacher_labels','teacher_beta',
                'search_value_buffer','source_transitions'],
            preserved=['weights','rng','generation','games','opponent_pool','ratings','rust_backend','entropy_anneal',
                'gae_lambda_and_table_gates','search_head_and_stage','teacher_readiness','local_scoring_heads'],
            schedule=dict(shadow_iterations=4,teacher_on_from_iteration=5,stage_a_iterations=stage_a_iterations,
                stage_a_paused_iterations=PAUSED_ITERATIONS,stage_a_count_at_restart=search['stage']['stage_a_iterations']))
        write_json(safe(run,run/'state.json'),fresh)
        write_json(marker,summary)
    return summary
