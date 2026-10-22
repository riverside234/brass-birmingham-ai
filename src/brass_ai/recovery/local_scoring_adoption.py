"""Durable one-time companion adoption and bounded PPG cache retirement."""
import copy
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import digest,require,project_path
from brass_ai.step5.league import write_json
from .retention import owned,safe
from .local_scoring_auxiliary import KEY,DEFAULT,validate_config

CONTRACT='local-scoring-adoption/1.0.0'


def configure(state,mode='apply'):
    config=copy.deepcopy(state['signature']['config'])
    if KEY in config:
        config[KEY]['mode']=mode
    else:
        # The current run has not adopted the heads yet: combine the requested
        # LR threshold hotfix with that one reviewed g560 transition.
        from .lr_controller import threshold_config
        config=threshold_config(config)
        entry=state['history'][-1]
        config[KEY]=dict(copy.deepcopy(DEFAULT),mode=mode,adoption=dict(generation=state['iteration'],
            checkpoint_sha256=entry['sha256'],model_id=entry['model_id'],fresh_ppg_buffer=True))
    config['branching']['rate_per_decision']=.010
    return config


def compatible(state,new):
    old=state['signature']['config']
    if KEY not in new:return False
    validate_config(new[KEY])
    if KEY not in old:
        return new==configure(state,new[KEY]['mode']) and new[KEY]['mode']!='off'
    expected=copy.deepcopy(old);expected[KEY]['mode']=new[KEY]['mode']
    return expected==new


def audit_files(root,run,generation):
    from .value_buffer import OWNER,FILE
    from .ppg_replay import OWNER as EIGHT_OWNER,FILE as EIGHT_FILE
    root,run=owned(root,run);files=[]
    for name,owner,pattern in (('.ppg-value-cache',OWNER,FILE),('.ppg-eight-cache',EIGHT_OWNER,EIGHT_FILE)):
        folder=safe(run,run/name)
        if not folder.exists():continue
        require(folder.is_dir() and read(safe(folder,folder/'owner.json'))==owner,'Local scoring cleanup owner differs')
        for path in folder.iterdir():
            if path.name=='owner.json':continue
            match=pattern.fullmatch(path.name)
            require(match is not None and safe(folder,path).is_file(),'Unknown/linked auxiliary cache file')
            if int(match[1])<=generation:
                files.append(dict(path=path.relative_to(run).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    return sorted(files,key=lambda x:x['path'])


def prepare(root,run,state,raw,config,*,reviewed=False):
    """Pure preparation. No state publication, checkpoint edit or cache deletion."""
    saved=(raw or {}).get('training',{}).get(KEY)
    if KEY not in config:
        require(saved is None,'Adopted companion must be explicitly retained or disabled')
        return raw,None,False
    c=validate_config(config[KEY]);generation=state['iteration'];origin=c['adoption']
    if saved is not None:return raw,None,False
    if generation==0 and origin is None:return raw,None,False
    require(origin is not None and origin['generation']==generation and c['mode']!='off','Missing active local scoring bundle')
    entry=state['history'][-1]
    require(entry['sha256']==origin['checkpoint_sha256'] and entry['model_id']==origin['model_id'] and
        sha(project_path(root,entry['path']))==origin['checkpoint_sha256'],'Local scoring adoption checkpoint differs')
    old=raw['training']['actor_value_phases']
    from . import ppg_config,ppg_epochs,gae,critic_data
    require(old['contract']=='actor-value-phases/10.0.0' and ppg_config.coverage(config['value_phase'])
        and old['options']=={k:config[k] for k in old['options']},'Local scoring requires unchanged one-SE critic recipe')
    ppg_config.schedule(old['schedule'],generation)
    ppg_epochs.validate(old['epoch_control'],config['value_phase'],phases=old['schedule']['phases'],last_phase=old['schedule']['last_phase'])
    gae.validate_state(old['gae'],config['advantage'],generation)
    critic_data.validate_state(old['critic_data'],config['critic_data'],generation)
    receipt=state.get('local_scoring_adoption')
    if receipt is None:
        require(reviewed and compatible(state,config),'Unreviewed local scoring adoption')
        files=audit_files(root,run,generation)
        available={f['path']:f for f in files}
        for r in old['rolling_buffer']:
            require(read(safe(run,run/f'.ppg-eight-cache/generation-{r["generation"]:06d}.json'))==r,
                'Old PPG metadata differs from committed checkpoint')
            for ext,f in r['files'].items():
                path=f'.ppg-eight-cache/generation-{r["generation"]:06d}.{ext}'
                require(path in available and all(available[path][k]==f[k] for k in ('bytes','sha256')),'Committed old PPG cache changed')
        receipt=dict(contract=CONTRACT,origin=copy.deepcopy(origin),config_sha256=digest(c),
            checkpoint=copy.deepcopy(entry),old_schedule=copy.deepcopy(old['schedule']),
            old_epoch_control=copy.deepcopy(old['epoch_control']),old_cache_receipts_sha256=digest(old['rolling_buffer']),
            old_cache_files=files,new_origin=generation,next_phase=generation+8,
            cleanup_complete=False,cleanup=dict(deleted_files=0,deleted_bytes=0))
    require(receipt['contract']==CONTRACT and receipt['origin']==origin and receipt['config_sha256']==digest(c) and
        receipt['checkpoint']==entry and receipt['old_schedule']==old['schedule'] and
        receipt['old_cache_receipts_sha256']==digest(old['rolling_buffer']),'Adoption reset receipt differs')
    # This explicit receipt authorizes ignoring retired references in the OLD
    # checkpoint until the first new generation commits. Never backdate/reset a
    # generation carrying the companion bundle.
    updated=copy.deepcopy(old)
    updated.update(cache_origin=generation,schedule=ppg_config.initial(generation),rolling_buffer=[],
        epoch_control=dict(ppg_epochs.initial(generation,config['value_phase']),cap=old['epoch_control']['cap']))
    return dict(raw,training=dict(raw['training'],actor_value_phases=updated)),receipt,True


def archive_metadata(run,receipt,raw):
    """Archive receipts, not payload arrays, before durable state adoption."""
    metadata=dict(contract=CONTRACT,origin=receipt['origin'],old_schedule=receipt['old_schedule'],
        old_epoch_control=receipt['old_epoch_control'],rolling_buffer=raw['training']['actor_value_phases']['rolling_buffer'])
    require(digest(metadata['rolling_buffer'])==receipt['old_cache_receipts_sha256'],'Old PPG metadata identity differs')
    path=safe(run,run/'local-scoring'/f'adoption-g{receipt["origin"]["generation"]:06d}.json')
    require(not path.exists() or read(path)==metadata,'Existing adoption metadata differs')
    if not path.exists():write_json(path,metadata)
    return dict(copy.deepcopy(receipt),old_cache_metadata=dict(path=path.relative_to(run).as_posix(),sha256=sha(path),bytes=path.stat().st_size))


def cleanup(root,run,state):
    receipt=state.get('local_scoring_adoption')
    if receipt is None or receipt['cleanup_complete']:return None
    require(read(run/'state.json')==state,'Local scoring cleanup requires durable adopted state')
    if 'old_cache_metadata' in receipt:
        meta=receipt['old_cache_metadata']
        require(sha(safe(run,run/meta['path']))==meta['sha256'],'Archived old PPG metadata changed')
    generation=receipt['origin']['generation']
    found=audit_files(root,run,generation);expected={f['path']:f for f in receipt['old_cache_files']}
    require(all(f==expected.get(f['path']) for f in found),'Old PPG cache changed after adoption')
    # All targets audited before the first unlink. Missing files are a safe
    # crash continuation, but a replaced file is never silently removed.
    for f in found:safe(run,run/f['path']).unlink()
    updated=copy.deepcopy(receipt)
    updated.update(cleanup_complete=True,cleanup=dict(deleted_files=len(expected),
        deleted_bytes=sum(f['bytes'] for f in expected.values()),files_removed_this_start=len(found),
        bytes_removed_this_start=sum(f['bytes'] for f in found)))
    state['local_scoring_adoption']=updated
    write_json(safe(run,run/'state.json'),state)
    return updated
