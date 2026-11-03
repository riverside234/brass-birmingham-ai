"""Verified weights-only initialization in a separate run, with a bounded pool.

Imported opponents have negative local birth generations; original generations
remain explicit in imported_from. Generation zero is still the original IL
opponent/rating origin, not the new learner's seed checkpoint.
"""
import copy
from dataclasses import replace
from pathlib import Path
import shutil
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import compatibility,digest,project_path,require
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.model import ModelConfig
from brass_ai.step5.league import write_json
from .retention import owned,safe,active,run_lock

KEY='fresh_selfplay_seed'
CONTRACT='fresh-selfplay-weights/1.0.0'
IMPORTED='imported-pfsp-member/1.0.0'


def hex64(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_config(config):
    from .neutral_training import enabled
    from .policy_observations import enabled as observations_off
    c=config[KEY]
    require(type(c) is dict and set(c)=={'contract','manifest','sha256'} and c['contract']==CONTRACT
        and type(c['manifest']) is str and hex64(c['sha256']),'Invalid fresh weights seed receipt')
    require(enabled(config) and observations_off(config),'Fresh seed requires neutral contexts and retired AnchorKL')


def validate_import(member):
    origin=member.get('imported_from')
    require(type(origin) is dict and set(origin)=={'contract','source_run','source_generation','source_model_id','history_through'}
        and origin['contract']==IMPORTED and type(origin['source_run']) is str
        and type(origin['source_generation']) is int and type(origin['history_through']) is int
        and 0<origin['source_generation']<=origin['history_through']
        and origin['source_model_id']==member['model_id'] and hex64(origin['source_model_id'])
        and member['generation']==origin['source_generation']-origin['history_through']-1,
        'Invalid imported PFSP generation/identity')
    return origin


def selected(state,seed_generation,history_through):
    """Only the committed active pool is inherited, not every historical file."""
    require(type(seed_generation) is int and type(history_through) is int and
        0<seed_generation<=history_through<=state['iteration'],'Invalid seed/history cutoff')
    entry=next((e for e in state['history'] if e['generation']==seed_generation),None)
    require(entry is not None,'Seed generation is absent from committed history')
    from .v4_history import bot
    members=copy.deepcopy([m for m in state['quality_pool']['members'] if not bot(m) and m['generation']<=history_through])
    require(members and min(m['generation'] for m in members)==0 and
        all(m['generation']>=0 for m in members),'Seed requires a native source pool with its original IL')
    return copy.deepcopy(entry),members


def available_seed(root,source,state,entry):
    candidates=[entry]+state.get('population',[])+[
        a['checkpoint'] for a in state.get('ratings',{}).get('agents',{}).values() if a.get('kind')=='model']
    anchor=state.get('policy_anchor',{}).get('active')
    if anchor:candidates.append(anchor['checkpoint'])
    for candidate in candidates:
        if (candidate['generation'],candidate['model_id'])!=(entry['generation'],entry['model_id']):continue
        path=safe(source,project_path(root,candidate['path']))
        if path.is_file():
            require(sha(path)==candidate['sha256'],'Seed checkpoint changed')
            return copy.deepcopy(candidate)
    raise ValueError(f'No retained full or weights-only checkpoint for g{entry["generation"]}; a report cannot reconstruct weights')


def copy_verified(root,source,destination,receipt):
    path=safe(source,project_path(root,receipt['path']))
    require(path.is_file() and sha(path)==receipt['sha256'],'Missing/changed source: '+str(path))
    if destination.exists():require(sha(destination)==receipt['sha256'],'Prepared destination differs: '+str(destination))
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name(destination.name+'.tmp')
        try:
            shutil.copyfile(path,temporary)
            require(sha(temporary)==receipt['sha256'],'Source changed during copy')
            temporary.replace(destination)
        finally:temporary.unlink(missing_ok=True)
    return dict(receipt,path=destination.relative_to(root).as_posix())


def prepare(root,source_path,run_path,output,*,expected_generation=1360,seed_generation=1290,history_through=1330,no_rehearsal=False,
            trained_restart=False):
    """Copy a self-contained seed bundle; never edit or recover the source run."""
    from . import quality_pool
    from .config import validate_config as validate_recipe
    from .neutral_training import configure
    root,source=owned(root,source_path);_,run=owned(root,run_path)
    require(source!=run and not run.is_relative_to(source) and not source.is_relative_to(run),
        'Fresh run must be a separate sibling, outside the source run')
    output=project_path(root,output)
    require(output.is_relative_to(root/'configs/step5') and output.suffix=='.json','Write config under configs/step5')
    require(not active(source),'Stop the source run before preparing a fresh seed')
    source_state=safe(source,source/'state.json');state_hash=sha(source_state);state=read(source_state)
    require(state['iteration']==expected_generation,'Source committed generation changed')
    require(state['signature']['compatibility']==compatibility(),'Source learning layout differs')
    require(not (run/'state.json').exists(),'Fresh destination already has training progress')
    quality_pool.validate_state(state['quality_pool'],state['signature']['config']['opponent_pool'],state['iteration'])
    last=state['history'][-1];path=safe(source,project_path(root,last['path']))
    require(sha(path)==last['sha256'],'Source committed checkpoint changed')
    raw=torch.load(path,map_location='cpu',weights_only=True)
    require(raw['step']==last['generation']==state['iteration'] and raw['model_id']==last['model_id']
        and raw['training'].get('signature')==digest(state['signature'])
        and raw['training'].get('quality_pool_digest')==quality_pool.state_digest(state['quality_pool']),
        'Source checkpoint/config/pool must be a fully committed generation')
    del raw
    entry,members=selected(state,seed_generation,history_through)
    seed=available_seed(root,source,state,entry)
    config=configure(state['signature']['config']);config.pop(KEY,None)
    # Companion heads are absent from slim population checkpoints. Fresh heads
    # and accepted-work warmup do not depend on the optional scheduling gates.
    from . import local_scoring_auxiliary as auxiliary
    config[auxiliary.KEY]=auxiliary.fresh_config(config[auxiliary.KEY])
    require(type(no_rehearsal) is bool,'Invalid rehearsal override')
    # Bind this opt-in to the new manifest before publishing it. A zero base
    # weight disables the schedule, including its floor, and skips shard loading.
    if no_rehearsal:config['rehearsal_weight']=0.
    from . import trained_startup as startup
    require(type(trained_restart) is bool,'Invalid trained restart override')
    if trained_restart:
        require(seed_generation==1290,'Trained restart requires generation 1290')
        config=startup.configure(config)
    # Start a new seed stream beyond the source's committed ordinary games.
    config['seed']=state['next_seed']
    # Preserve all eligible members, without pretending old games were played
    # in the new run just to unlock the late capacity. The total cap stays 24.
    config['opponent_pool']['max_scored_members'][0]['max']=max(
        config['opponent_pool']['max_scored_members'][0]['max'],len(members)-1)
    validate_recipe(config)
    origin=state['signature']['imitation_source']
    require(origin['model_id']==next(m['model_id'] for m in members if m['generation']==0),
        'Imported IL opponent differs from the completed IL reference')
    # Validate all model identities before creating a destination. No source
    # Adam, RNG, phase state or DOMiNO companions are adopted.
    refs={r['model_id']:r for r in [seed,*members]}
    with torch.random.fork_rng(devices=[]):
        for ref in refs.values():
            path=safe(source,project_path(root,ref['path']))
            require(path.is_file() and sha(path)==ref['sha256'],'Missing/changed source checkpoint: '+str(path))
            model,raw=load_checkpoint(path)
            require(raw['model_id']==ref['model_id'] and raw['step']==ref['generation'],'Imported model identity/generation differs')
            base=model.config.to_dict();base.pop('utility',None)
            require(base==ModelConfig.preset(config['model_size']).to_dict(),'Imported model layout differs')
            del model,raw
    with run_lock(run):
        require(not (run/'state.json').exists(),'Fresh destination advanced')
        marker=safe(run,run/'seed'/'preparation.json')
        binding=dict(contract=CONTRACT,source_run=source.relative_to(root).as_posix(),source_state_sha256=state_hash,
            source_generation=state['iteration'],destination_run=run.relative_to(root).as_posix(),
            seed_generation=seed_generation,history_through=history_through,recipe_sha256=digest(config))
        if marker.exists():require(read(marker)==binding,'Prepared seed belongs to another source/config')
        else:
            require(all(p.name=='.active-run.json' for p in run.iterdir()),'Fresh destination must be empty')
            write_json(marker,binding)
        seed_copy=copy_verified(root,source,safe(run,run/'seed'/('weights-'+seed['model_id']+'.pt')),seed)
        # Reuse the already verified probe without reloading the IL dataset.
        from .il_kl_probe import import_probe
        probe,probe_receipt=import_probe(root,run,source,origin,None,config['il_kl_limit'],config['update'])
        copied=[]
        for member in members:
            checkpoint=copy_verified(root,source,safe(run,run/'population'/('weights-'+member['model_id']+'.pt')),
                quality_pool.reference(member))
            values=quality_pool.cached(root,member,probe)
            cached=quality_pool.cache_candidate(root,run,checkpoint,values,probe)
            copied.append(dict(checkpoint,probe=cached))
        require(sha(source_state)==state_hash and not active(source),'Source advanced during seed preparation')
        manifest=dict(binding,seed_checkpoint=seed_copy,members=copied,probe=probe_receipt,
            **(dict(startup=copy.deepcopy(config[startup.KEY])) if trained_restart else {}),
            imitation_source=copy.deepcopy(origin),excluded_generations=[m['generation'] for m in state['quality_pool']['members']
                if m['generation']>history_through],
            reset=['optimizers','rng','iterations','ordinary_games','PPG_cache','PFSP_scores','gates','DOMiNO'],
            preserved=['seed_model_tensors','original_IL_identity','eligible_frozen_opponents','fixed_pool_probe'])
        manifest_path=safe(run,run/'seed'/'manifest.json')
        require(not manifest_path.exists() or read(manifest_path)==manifest,'Prepared manifest differs')
        write_json(manifest_path,manifest)
        config[KEY]=dict(contract=CONTRACT,manifest=manifest_path.relative_to(root).as_posix(),sha256=sha(manifest_path))
        validate_recipe(config)
        require(not output.exists() or read(output)==config,'Existing output config differs; choose a new --output')
        write_json(output,config)
    return dict(config=output.relative_to(root).as_posix(),run=run.relative_to(root).as_posix(),
        generation=0,seed_generation=seed_generation,seed_model_id=seed['model_id'],history_through=history_through,
        imported_generations=[m['generation'] for m in copied],excluded_generations=manifest['excluded_generations'],
        rehearsal_enabled=bool(config['rehearsal_weight']),rehearsal_weight=config['rehearsal_weight'],
        auxiliary_heads='fresh',auxiliary_warmup=dict(qualifying_iterations=config[auxiliary.KEY]['head_only_iterations'],
            retained_fitting_steps=config[auxiliary.KEY]['minimum_fitting_steps'],ramp_iterations=config[auxiliary.KEY]['ramp_iterations']),
        **(dict(startup=config[startup.KEY],qualification='manual_override_not_measured') if trained_restart else {}),
        source_modified=False,training_started=False)


def load_manifest(root,run,config):
    validate_config(config);c=config[KEY];path=safe(run,project_path(root,c['manifest']))
    require(path==run/'seed'/'manifest.json' and sha(path)==c['sha256'],'Fresh seed manifest changed or belongs to another run')
    m=read(path)
    recipe={k:v for k,v in config.items() if k!=KEY}
    require(m['contract']==CONTRACT and m['destination_run']==run.relative_to(root).as_posix()
        and m['recipe_sha256']==digest(recipe),'Fresh seed recipe/destination differs')
    require(m['seed_checkpoint']['generation']==m['seed_generation']<=m['history_through'],
        'Fresh seed generation differs')
    members=m['members']
    require(len({e['model_id'] for e in members})==len(members) and
        sum(e['generation']==0 for e in members)==1 and
        all(type(e['generation']) is int and 0<=e['generation']<=m['history_through'] for e in members),
        'Imported history exceeds cutoff or repeats a model')
    for ref in [m['seed_checkpoint'],*members]:
        require(sha(safe(run,project_path(root,ref['path'])))==ref['sha256'],'Prepared checkpoint changed')
    from . import trained_startup as startup
    if startup.enabled(config):
        require(m.get('startup')==config[startup.KEY],'Prepared startup settings differ')
    else:require('startup' not in m and 'auxiliary_heads' not in m,'Unexpected trained startup receipt')
    return m


def seed_model(root,run,config,manifest,device):
    """Copy tensors exactly; change reward metadata and invalidate calibration."""
    from .utility import reward_spec
    ref=manifest['seed_checkpoint']
    with torch.random.fork_rng(devices=[]):model,raw=load_checkpoint(project_path(root,ref['path']),device=device)
    require(raw['model_id']==ref['model_id'],'Fresh seed model identity differs')
    base=model.config.to_dict();base.pop('utility',None)
    require(base==ModelConfig.preset(config['model_size']).to_dict() and getattr(model.config,'utility',None) is not None,
        'Fresh PPO seed requires the current utility-critic architecture')
    previous=copy.deepcopy(model.config.utility)
    model.config=replace(model.config,utility=copy.deepcopy(reward_spec(config)))
    model.utility_calibration=None
    # The signed config owns the inventory receipt. Do not link that complete
    # inventory into retention's origin graph: retired imports must become
    # eligible for normal cleanup rather than remaining pinned forever.
    return model,dict(initialization='fresh-selfplay-weights',parent_model_id=ref['model_id'],
        seed_generation=ref['generation'],source_run=manifest['source_run'],seed_checkpoint=ref,
        seed_manifest_sha256=config[KEY]['sha256'],previous_reward=previous,optimizer_reset=True,progress_reset=True)


def initialize_pool(root,run,config,manifest,probe):
    from . import quality_pool,pfsp_pool
    source_anchor=next(m for m in manifest['members'] if m['generation']==0)
    anchor=quality_pool.reference(source_anchor)
    # Imported neural caches keep their verified main/companion payload intact.
    anchor['weights_only']=bool(source_anchor.get('weights_only'))
    pool=quality_pool.initial(anchor,config['opponent_pool'],source_anchor['probe'])
    for member in manifest['members']:
        quality_pool.cached(root,member,probe)  # Verify cache identity/normalization, without inference.
        if member['generation']==0:continue
        ref=quality_pool.reference(member);g=member['generation']
        ref.update(generation=g-manifest['history_through']-1,
            imported_from=dict(contract=IMPORTED,source_run=manifest['source_run'],source_generation=g,
                source_model_id=member['model_id'],history_through=manifest['history_through']))
        pool['members'].append(dict(ref,admitted_iteration=0,probe=copy.deepcopy(member['probe']),
            admission_health=dict(ok=True,scope='Explicit imported membership; new PFSP prior, no inherited health evidence'),
            **pfsp_pool.prior(config['opponent_pool'])))
    pfsp_pool.seal(pool);quality_pool.validate_state(pool,config['opponent_pool'],0)
    return pool,anchor
