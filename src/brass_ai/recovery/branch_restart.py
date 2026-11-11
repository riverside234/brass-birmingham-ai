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

A rollback restart (``rollback=...``) keeps the exported state of one generation
but seeds the learner with the retained snapshot of the generation before it,
when that later full checkpoint is gone too. It uses the g1490 restart variant of
the g1491 revision (one shadow iteration re-measures beta), keeps the search head's
committed stage (fitting pauses while its value buffer refills), carries the PPO LR
multiplier of that generation's report, moves the learner's companion heads to the
restart generation without fitting, and applies a rating compaction the source run
already made (exactly as a later export records it) when it deleted those slot files.

The g1190 seed restart (``rollback=dict(revision='g1190', learner_generation=1190)``)
uses the same exported state with a much older retained snapshot of this lineage,
trained before the teacher phase and the post-transfer reward. Its tensors are kept
under the restart reward (placement .1, zero charge): the utility specification is
migrated and its calibration cleared, so the first value phase re-fits it before
any teacher table runs. It has no search head: a fresh one is adopted at Stage A on
the first teaching iteration, under the restart stage schedule (Stage A 50 with the
held-out early stop, Stage B 50, PPG beta floor .5 after Stage A).
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
SEED='g1190'
# The second attempt from the same g1491 state: insolvency charge kept (.0025, cap .025) and 256 teacher rows.
SEED_CHARGED='g1190-charged-256'
SEEDS=(SEED,SEED_CHARGED)
SEED_MIGRATION='seed-restart-reward/1.0.0'
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


def rollback_recipe(config,state,*,share_target,variant=None):
    """The committed teacher-on recipe with the g1490 restart variant of the g1491 revision."""
    from brass_ai.rust_backend.lab_shadow_config import is_shadow
    from .collector_runtime import recipe as without_runtime
    from .config import validate_config
    from . import training_revision_g1491 as g1491
    from .teacher_gradient_qualification import valid_target
    config=without_runtime(config)
    require(valid_target(share_target),'Choose a share target above 0 and at most 1')
    require(not g1491.applied(config),'A rollback restart starts from the recipe committed before the g1491 revision')
    result=g1491.configure(config,state,variant=variant or g1491.RESTART)
    result['teacher_phase']['term']['share_target']=share_target
    validate_config(result)
    require(is_shadow(result),'The rollback recipe must pass the reviewed shadow projection')
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


def clear_partial(run):
    """Remove an unfinished preparation: only links and copies made here (the source files stay)."""
    for child in list(run.iterdir()):
        if child.name=='.active-run.json':continue
        if child.is_symlink() or child.is_file():child.unlink()
        else:shutil.rmtree(safe(run,child))


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


def ppo_optimizer(config,model,multiplier=None):
    """A new AdamW exactly as a new run builds it, with its warmup and LR controller.

    A rollback carries the committed controller multiplier; it applies once the new
    warmup finishes. The measurement window and cooldown start empty.
    """
    from brass_ai.learning.search_value import ordinary_parameters
    from .learning_rate import configure as warmup
    from .lr_controller import configure as controller,KEY,validate_state
    optimizer=torch.optim.AdamW(ordinary_parameters(model),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    w=config['learning_rate_warmup'];warmup(optimizer,config['learning_rate'],w['selfplay_steps'],w['start_factor'],'selfplay')
    controller(optimizer,config['ppo_lr_controller'])
    if multiplier is not None:
        group=optimizer.param_groups[0];group[KEY]['multiplier']=multiplier;validate_state(group[KEY])
    return optimizer.state_dict()


def committed_multiplier(report,config):
    """The PPO LR multiplier the snapshot generation's own update left in force."""
    c=report['actor']['lr_controller']
    require(c['contract']==config['ppo_lr_controller']['contract'] and c['mode']=='apply' and
        type(c['multiplier_after']) in (int,float),'Snapshot report has no PPO LR controller decision')
    return float(c['multiplier_after'])


def generation_fields(value):
    """The exploration state without its two generation stamps."""
    value=copy.deepcopy(value);value.pop('generation',None)
    if isinstance(value.get('entropy_schedule'),dict):value['entropy_schedule'].pop('last_generation',None)
    return value


def rollback_weights(root,source,state,weights=None):
    """The retained snapshot of the generation before the exported one, whose own checkpoint is gone."""
    previous,last=state['history'][-2],state['history'][-1]
    generation=state['iteration']
    require(last['generation']==generation and previous['generation']==generation-1,
        'A rollback restart seeds the learner from the generation just before the exported one')
    agent=next((a['checkpoint'] for a in state['ratings']['agents'].values() if a.get('kind')=='model'
        and a['checkpoint']['model_id']==previous['model_id'] and a['checkpoint'].get('generation')==previous['generation']),None)
    require(agent is not None,f"No retained snapshot of g{previous['generation']}")
    path=safe(source,project_path(root,agent['path']))
    if weights is not None:require(project_path(root,weights)==path,f"Weights are not the retained snapshot of g{previous['generation']}")
    require(sha(path)==agent['sha256'],'Restart weights changed')
    raw=torch.load(path,map_location='cpu',weights_only=True)
    training=raw['training']
    require(raw['model_id']==previous['model_id'] and raw['step']==previous['generation'] and raw['optimizer'] is None
        and training.get('signature')==digest(state['signature'])
        and training.get('fresh_run')==source.relative_to(root).as_posix()
        and 'search_value' in training and 'teacher_phase' in training,
        f"Restart weights are not the committed checkpoint of g{previous['generation']}")
    # One iteration later only the generation stamps of the finished anneal moved.
    require(training.get('exploration',{}).get('generation')==previous['generation'] and
        generation_fields(training['exploration'])==generation_fields(state['exploration']) and
        training.get('startup_initialization')==state.get('startup_initialization') and
        training.get('auxiliary_initialization')==state.get('auxiliary_initialization'),
        'Restart weights disagree with the committed exploration/startup state')
    require(not training.get('policy_anchor') and not training.get('domino'),
        'A rollback restart does not carry policy-anchor or DOMiNO payloads')
    return path,agent,raw


def rollback_search_payload(saved,summary,generation,identity):
    """The learner's search-head optimizer, the committed stage, and an empty value buffer.

    Fitting (and the trunk pass in Stage C) pauses until eight fresh ages exist.
    """
    require(saved['generation']==generation-1 and summary['generation']==generation,
        'Search value payloads are not one generation apart')
    stage=copy.deepcopy(summary['stage'])
    require(all(stage[k]==saved['stage'][k] for k in ('stage','w','scale')),
        'The rollback keeps the committed search-head stage')
    value=dict(saved,generation=generation,buffer=dict(saved['buffer'],receipts=[],retired=[]),label_ages=[])
    value['stage']=dict(stage,buffer_rebuild=dict(origin_generation=generation+1,required_ages=8,
        ready_generation=None,instruction_id=identity))
    return value


def rollback_companion(saved,generation,learner=None):
    """The learner's own local-scoring heads and Adam, moved to the restart generation with no fitting."""
    from .local_scoring_auxiliary import bundle_id
    from . import local_scoring_warmup as warmup
    learner=generation-1 if learner is None else learner
    result={k:v for k,v in saved.items() if k!='identity'}
    require(saved['identity']==bundle_id(result) and saved['generation']==learner<generation,
        'Companion heads are not the snapshot generation\'s own')
    result=dict(result,generation=generation,age=generation-saved['origin']['generation'])
    if 'warmup' in result:
        warmup.validate(saved['warmup'],saved['config'],learner)
        # No fitting step in between: only the generation stamp moves.
        result['warmup']=warmup.validate(dict(saved['warmup'],generation=generation),saved['config'],generation)
    return dict(result,identity=bundle_id(result))


def seed_recipe(config,state,*,share_target,revision=SEED):
    """The g1490 restart recipe with placement .1 restored and the restart stage schedule.

    ``SEED_CHARGED`` also keeps the insolvency charge (.0025, cap .025) and samples 256 teacher rows per step.
    """
    from brass_ai.learning.post_transfer_reward import keep_charge,restore_placement
    from brass_ai.rust_backend.lab_shadow_config import is_shadow
    from .config import validate_config
    from .search_value_schedule import restart_schedule
    from . import training_revision_g1491 as g1491
    require(revision in SEEDS,'Unknown seed restart revision')
    charged=revision==SEED_CHARGED
    result=restore_placement(rollback_recipe(config,state,share_target=share_target,
        variant=g1491.WIDE_RESTART if charged else None))
    if charged:result=keep_charge(result)
    options=result['teacher_phase']['search_value']
    require('schedule' in options,'The seed restart needs the residual stage schedule')
    options['schedule']=dict(restart_schedule(),
        validation_during_beta_qualification=options['schedule']['validation_during_beta_qualification'])
    # The fresh search head is 256 wide (the earlier heads are 32).
    options['head']['hidden']=256
    validate_config(result)
    require(is_shadow(result),'The seed recipe must pass the reviewed shadow projection')
    return result


def seed_snapshot(root,source,state,learner,weights=None):
    """The retained snapshot of an earlier generation of this lineage, from an earlier recipe."""
    entry=next((e for e in state['history'] if e['generation']==learner),None)
    require(type(learner) is int and 0<learner<state['iteration']-1 and entry is not None,
        "The seed generation is not in this run's lineage")
    agent=next((a['checkpoint'] for a in state['ratings']['agents'].values() if a.get('kind')=='model'
        and a['checkpoint']['model_id']==entry['model_id'] and a['checkpoint'].get('generation')==learner),None)
    require(agent is not None,f'No retained snapshot of g{learner}')
    path=safe(source,project_path(root,agent['path']))
    if weights is not None:require(project_path(root,weights)==path,f'Weights are not the retained snapshot of g{learner}')
    require(sha(path)==agent['sha256'],'Restart weights changed')
    raw=torch.load(path,map_location='cpu',weights_only=True)
    require(raw['model_id']==entry['model_id'] and raw['step']==learner and raw['optimizer'] is None,
        f'Restart weights are not the committed checkpoint of g{learner}')
    require(raw['training'].get('search_value') is None and raw['training'].get('search_value_head') is None,
        f'g{learner} already has a search head; this restart adopts a fresh one')
    return path,agent,raw


def seed_model(path,config):
    """The seed tensors under the restart reward: the utility specification moves, nothing else.

    The utility residual keeps its weights; calibration is cleared, so the next value
    phase re-fits it (and readiness) before any teacher table is planned.
    """
    from dataclasses import replace
    from brass_ai.learning.model import ModelConfig
    from brass_ai.learning.reward import specification
    from brass_ai.learning.utility_model import utility_config
    from brass_ai.learning.post_transfer_reward import BEFORE,OLD_COST,RESTORED,ZERO_COST
    from brass_ai.learning.checkpoint import load_checkpoint
    from .utility import reward_spec
    model,_=load_checkpoint(path)
    require(not hasattr(model,'search_value_head'),'The seed model already has a search head')
    spec=reward_spec(config);before=copy.deepcopy(getattr(model.config,'utility',None))
    # The recipe's own charge: zero (first attempt) or the old .0025 / .025 (charged attempt).
    require(before==specification(BEFORE,OLD_COST) and config['insolvency'] in (ZERO_COST,OLD_COST) and
        spec==specification(RESTORED,config['insolvency']),
        'Unreviewed seed reward migration')
    model.config=replace(model.config,utility=copy.deepcopy(spec));model.utility_calibration=None
    require(model.config==utility_config(ModelConfig.preset(config['model_size']),spec),
        'Seed model architecture differs from the recipe')
    return model,dict(contract=SEED_MIGRATION,from_reward=before,to_reward=copy.deepcopy(spec),
        preserved='All tensors, including the utility residual; utility calibration cleared and re-fit by the next value phase')


def compacted_ratings(root,source,ratings,later,old,new,snapshot_receipt):
    """Rating slots the source run archived (and whose files it then deleted) after the export.

    The later export's committed archive must be exactly this state's archive with its
    oldest slots appended, as compact_history writes it. Returns the new rating state
    and the archive file to place, or (ratings, None) when nothing is missing.
    """
    from .ratings import ARCHIVE_FORMAT
    keys=('records','report','spec','panels','bootstrap')
    def present(slot):
        return all(project_path(root,old+slot[k]['path'][len(new):]).is_file() or
            exported(root,snapshot_receipt,slot[k]['path'][len(new):],slot[k]['sha256']) is not None for k in keys)
    gone=[s['generation'] for s in ratings['slots'] if not present(s)]
    if not gone:return ratings,None
    require(later is not None,f'Rating slots {gone} are gone from the source run; pass the later export that archived them')
    later=project_path(root,later) if not Path(later).is_absolute() else Path(later)
    manifest=read(later/'export-manifest.json');run=source.relative_to(root).as_posix()
    require(manifest['run']==run,'Later export belongs to another run')
    entry=next((f for f in manifest['files'] if f['path']=='state.json'),None)
    require(entry is not None and sha(later/'state.json')==entry['sha256'],'Later export state differs from its receipt')
    archive=read(later/'state.json')['ratings']['archive'];relative=archive['path'][len(run)+1:]
    file=next((f for f in manifest['files'] if f['path']==relative),None)
    require(archive['path'].startswith(run+'/') and file is not None and file['sha256']==archive['sha256']==sha(later/relative),
        'Later export lacks its committed rating archive')
    data=read(later/relative)
    before=dict(format=ARCHIVE_FORMAT,slots=[],games=[])
    if ratings.get('archive') is not None:
        name=ratings['archive']['path'][len(new):]
        kept=project_path(root,old+name)
        kept=kept if kept.is_file() and sha(kept)==ratings['archive']['sha256'] else exported(root,snapshot_receipt,name,ratings['archive']['sha256'])
        require(kept is not None,'The exported rating archive is missing')
        before=read(kept)
    added=data['slots'][len(before['slots']):]
    count=len(added);slots=ratings['slots'][:count]
    require(set(data)=={'format','slots','games'} and data['format']==ARCHIVE_FORMAT and
        relative=='ratings/history-'+digest(data)+'.json' and
        data['slots'][:len(before['slots'])]==before['slots'] and data['games'][:len(before['games'])]==before['games'] and
        [a['generation'] for a in added]==[s['generation'] for s in slots] and set(gone)<=set(a['generation'] for a in added) and
        all(a['original_sha256']=={k:s[k]['sha256'] for k in keys} and
            a.get('directory')==Path(s['spec']['path'][len(new):]).parent.as_posix() for a,s in zip(added,slots)) and
        {g['slot'] for g in data['games'][len(before['games']):]}=={s['generation'] for s in slots} and
        all(s['report']!=ratings['latest_report'] for s in slots),
        'The later rating archive is not this state\'s archive plus its oldest slots')
    result=copy.deepcopy(ratings)
    result.update(archive=dict(path=new+relative,sha256=archive['sha256']),slots=ratings['slots'][count:])
    return result,(later/relative,archive['sha256'],[s['generation'] for s in slots])


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
    gate=None;data=None
    if report is not None:
        require(sha(snapshot/report['path'])==report['sha256'],'Snapshot report differs from its export receipt')
        data=read(snapshot/report['path']);gate=data['actor']['gae']['gate']
    current=read(safe(source,source/'state.json'))
    require(current['signature']['compatibility']==compatibility()==state['signature']['compatibility'],
        'Source learning layout differs')
    require(current['iteration']>=generation and current['history'][:len(state['history'])]==state['history'],
        'Snapshot is not a committed generation of the source run')
    return state,gate,dict(path=(snapshot/'state.json').relative_to(root).as_posix() if snapshot.is_relative_to(root) else str(snapshot/'state.json'),
        sha256=entry['sha256'],generation=generation,source_iteration=current['iteration']),data


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


def prepare(root,source_run,snapshot,run_path,*,share_target=.1,stage_a_iterations=STAGE_A,weights=None,device='cpu',
            rollback=None):
    """Build the new run; the source run and its files are never modified.

    ``rollback`` (a dict, optionally with ``later``: a later export of the source run)
    seeds the learner from the generation before the exported one; see the module notes.
    """
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
    state,gate,snapshot_receipt,report=snapshot_state(root,source,snapshot)
    generation=state['iteration']
    old=source.relative_to(root).as_posix()+'/';new=run.relative_to(root).as_posix()+'/'
    seed=None
    if rollback is None:
        weights_path,weights_receipt,raw=seed_weights(root,source,state,weights)
        config=recipe(state['signature']['config'],share_target=share_target,stage_a_iterations=stage_a_iterations)
        multiplier=None
    else:
        require(set(rollback)<={'later','revision','learner_generation'},'Unknown rollback restart option')
        seed=rollback.get('revision')
        require(seed in (None,*SEEDS) and (seed is None)==('learner_generation' not in rollback),'Unknown rollback restart revision')
        if seed is None:
            weights_path,weights_receipt,raw=rollback_weights(root,source,state,weights)
            config=rollback_recipe(state['signature']['config'],state,share_target=share_target)
        else:
            weights_path,weights_receipt,raw=seed_snapshot(root,source,state,rollback['learner_generation'],weights)
            config=seed_recipe(state['signature']['config'],state,share_target=share_target,revision=seed)
        require(report is not None,'A rollback restart needs the snapshot generation\'s report in its export')
        multiplier=committed_multiplier(report,config)
    learner=generation-1 if seed is None else rollback['learner_generation']
    binding=dict(contract=CONTRACT,source_run=old[:-1],destination_run=new[:-1],generation=generation,
        snapshot=snapshot_receipt,weights=weights_receipt,recipe_sha256=digest(config))
    if rollback is not None:
        binding['rollback']=dict(learner_generation=learner,ppo_lr_multiplier=multiplier,
            later=None if rollback.get('later') is None else str(rollback['later']))
        if seed is not None:binding['rollback']['revision']=seed
    identity=digest(binding)
    with run_lock(run):
        marker=safe(run,run/RECEIPT)
        if marker.exists():
            previous=read(marker)
            if previous['binding']!=binding:
                # An attempt that stopped before it wrote state.json holds only copies and links of
                # source files. Another recipe (for example another share target) starts it over.
                require(previous.get('complete') is False,'The new run was prepared from another source or recipe')
                clear_partial(run)
                write_json(marker,dict(binding=binding,complete=False))
        else:
            require(all(p.name=='.active-run.json' for p in run.iterdir()),'The new run folder must be empty')
            write_json(marker,dict(binding=binding,complete=False))
        # The signed recipe keeps its own paths (the reviewed projection binds them).
        fresh={k:rewrite(v,old,new) if k!='signature' else copy.deepcopy(v) for k,v in state.items()}
        fresh['signature']=dict(state['signature'],config=config)
        for key in ('source_transitions',):fresh[key]=[]
        for key in ('machine_transfer','native_backend_switch'):fresh.pop(key,None)
        if seed==SEED_CHARGED:
            # The committed entropy clock holds the applied charge (none); the kept charge is the plan's last step.
            from brass_ai.learning.post_transfer_reward import OLD_COST,ZERO_COST
            schedule=fresh['exploration']['entropy_schedule']
            require(schedule['insolvency']==ZERO_COST and schedule['config']['insolvency_steps'][-1]['insolvency']==OLD_COST
                and schedule['progress']==schedule['config']['ramp_iterations'],
                'The kept charge needs the completed entropy plan with its last step at .0025')
            schedule['insolvency']=copy.deepcopy(OLD_COST)
        compacted=[]
        if rollback is not None:
            fresh['ratings'],archive=compacted_ratings(root,source,fresh['ratings'],rollback.get('later'),old,new,snapshot_receipt)
            if archive is not None:
                place(archive[0],safe(run,project_path(root,fresh['ratings']['archive']['path'])),archive[1]);compacted=archive[2]
        # Copy every file the committed state still refers to; record what is gone.
        missing_weights=set();missing_probes=[];placed=0
        for where,receipt in receipts(fresh,new):
            if where.startswith(('/il_kl_limit',)+RECORD_PREFIXES):continue
            if compacted and where=='/ratings/archive':placed+=1;continue  # placed from the later export
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
        if rollback is None:
            search=search_payload(raw['training']['search_value'],generation,identity)
        else:
            from .teacher_phase_config import buffer_identity
            # The revised recipe's own label identity, so nothing is retired on the first load.
            teacher['buffer']['identity']=buffer_identity(config)
            # A seed without a search head adopts a fresh one at Stage A (no committed head state).
            search=None if seed is not None else rollback_search_payload(raw['training']['search_value'],state['search_value'],generation,identity)
        if search is None:fresh.pop('search_value',None)
        else:fresh['search_value']=stamp(search)
        backend=fresh['native_backend'];backend.pop('teacher',None);backend['config_sha256']=digest(config)
        if rollback is not None:
            from brass_ai.rust_backend.lab_shadow_config import CONTRACT as SHADOW
            from .training_revision_g1491 import tables
            require(backend.get('shadow',{}).get('contract')==SHADOW,'The source run has no reviewed shadow adoption')
            # The same qualified teacher build; the receipt records the revised table counts.
            backend['shadow']=dict(backend['shadow'],tables=tables(config),generation=generation)
        # Rating history is reporting evidence; keep only what precedes the restart.
        history_file=source/HISTORY_PATH
        if history_file.is_file():
            vp=read(history_file)
            initial={k:fresh['history'][0][k] for k in ('path','sha256','model_id')}
            vp.update(initial_checkpoint=initial,committed_generation=generation,
                iterations=[r for r in vp['iterations'] if r['generation']<=generation])
            write_json(safe(run,run/HISTORY_PATH),vp)
        output=safe(run,run/f'generation-{generation:04d}.pt');temporary=output.with_name(output.name+'.tmp')
        migration=None
        if seed is not None:
            from .local_scoring_auxiliary import KEY as COMPANION
            from brass_ai.learning.checkpoint import save_checkpoint
            with torch.random.fork_rng(devices=[]):
                model,migration=seed_model(weights_path,config)
                saved=raw['training']
                # Run-level receipts and schedules come from the exported state; the seed adds
                # only its tensors, RNG and companion heads (moved to this generation, unfitted).
                training=dict(mode=saved.get('mode','step5-population-ppo'),recipe=saved.get('recipe'),parent=saved.get('parent'),
                    initialization=saved.get('initialization','random-lineage'),fresh_run=new[:-1],signature=digest(fresh['signature']),
                    exploration=copy.deepcopy(fresh['exploration']),il_kl_limit=copy.deepcopy(fresh['il_kl_limit']),
                    teacher_phase=copy.deepcopy(teacher),native_backend=copy.deepcopy(backend),native_backend_execution=execution(),
                    actor_value_phases=ppg_state(config,model,generation,gate),quality_pool_digest=quality_pool.state_digest(pool))
                for key in ('startup_initialization','auxiliary_initialization','v4_share_gate'):
                    if key in fresh:training[key]=copy.deepcopy(fresh[key])
                if saved.get(COMPANION) is not None:training[COMPANION]=rollback_companion(saved[COMPANION],generation,learner)
                save_checkpoint(temporary,model,None,step=generation,training=training)
                checkpoint=torch.load(temporary,map_location='cpu',weights_only=True)
                checkpoint.update(optimizer=ppo_optimizer(config,model,multiplier),rng=raw['rng'],cuda_rng=raw.get('cuda_rng',[]))
                del model
        if seed is None:
            with torch.random.fork_rng(devices=[]):
                model,_=load_checkpoint(weights_path)
                training=rewrite(raw['training'],old,new)
                training.pop('machine_transfer',None)
                training.update(fresh_run=new[:-1],signature=digest(fresh['signature']),
                    quality_pool_digest=quality_pool.state_digest(pool),teacher_phase=copy.deepcopy(teacher),
                    search_value=search,native_backend=copy.deepcopy(backend),native_backend_execution=execution(),
                    il_kl_limit=copy.deepcopy(fresh['il_kl_limit']),actor_value_phases=ppg_state(config,model,generation,gate))
                if rollback is not None:
                    # State-level schedules come from the exported generation (checked above);
                    # the learner's own companion heads move to it without fitting.
                    training['exploration']=copy.deepcopy(fresh['exploration'])
                    if 'v4_share_gate' in fresh:training['v4_share_gate']=copy.deepcopy(fresh['v4_share_gate'])
                    from .local_scoring_auxiliary import KEY as COMPANION
                    if training.get(COMPANION) is not None:training[COMPANION]=rollback_companion(training[COMPANION],generation)
                require(training.get('exploration')==fresh['exploration'] and
                    training.get('startup_initialization')==fresh.get('startup_initialization'),
                    'Restart weights disagree with the committed exploration/startup state')
                checkpoint=dict(raw,optimizer=ppo_optimizer(config,model,multiplier),training=training)
                if rollback is not None:checkpoint['step']=generation
                del model
        torch.save(checkpoint,temporary);temporary.replace(output)
        entry=dict(path=output.relative_to(root).as_posix(),sha256=sha(output),model_id=checkpoint['model_id'],generation=generation)
        fresh['history'][-1]=entry;fresh['resume']=[copy.deepcopy(entry)]
        summary=dict(binding=binding,complete=True,identity=identity,checkpoint=entry,files_placed=placed,
            dropped_members=dropped,regenerated_probes=regenerated,
            reset=['ppo_adam_and_warmup','ppg_adam_and_replay','gae_evidence_windows','teacher_labels','teacher_beta',
                'search_value_buffer','source_transitions'],
            preserved=['weights','rng','generation','games','opponent_pool','ratings','rust_backend','entropy_anneal',
                'gae_lambda_and_table_gates','search_head_and_stage','teacher_readiness','local_scoring_heads'],
            schedule=None if search is None else dict(shadow_iterations=4,teacher_on_from_iteration=5,stage_a_iterations=stage_a_iterations,
                stage_a_paused_iterations=PAUSED_ITERATIONS,stage_a_count_at_restart=search['stage']['stage_a_iterations']))
        if seed is not None:
            from .training_revision_g1491 import shadow_iterations,summary as revision
            stages=config['teacher_phase']['search_value']['schedule']
            summary.update(learner=dict(generation=learner,seed_model_id=raw['model_id'],model_id=checkpoint['model_id'],weights=weights_receipt),
                reward_migration=migration,compacted_rating_slots=compacted,ppo_lr_multiplier=multiplier,revision=revision(config),
                schedule=dict(utility_refit_iterations=1,shadow_iterations=shadow_iterations(config),
                    teacher_on_from_iteration=shadow_iterations(config)+2,search_head='fresh, adopted at Stage A on the first teaching iteration',
                    **{k:stages[k] for k in ('stage_a_iterations','stage_a_fit','stage_b_iterations','stage_c_iterations','ppg_beta_floor_after_stage_a')}))
            summary['reset']=summary['reset']+['search_head','utility_calibration','teacher_readiness']
            summary['preserved']=[p for p in summary['preserved'] if p not in ('weights','search_head_and_stage','teacher_readiness')]+[
                f'learner_weights_of_g{learner}','ppo_lr_multiplier']
        elif rollback is not None:
            from .training_revision_g1491 import shadow_iterations,summary as revision
            summary.update(learner=dict(generation=generation-1,model_id=raw['model_id'],weights=weights_receipt),
                compacted_rating_slots=compacted,ppo_lr_multiplier=multiplier,revision=revision(config),
                schedule=dict(shadow_iterations=shadow_iterations(config),teacher_on_from_iteration=shadow_iterations(config)+1,
                    search_head_stage=search['stage']['stage'],search_head_paused_iterations=PAUSED_ITERATIONS))
            summary['preserved']=[p for p in summary['preserved'] if p!='weights']+['learner_weights_of_previous_generation','ppo_lr_multiplier']
        write_json(safe(run,run/'state.json'),fresh)
        write_json(marker,summary)
    return summary
