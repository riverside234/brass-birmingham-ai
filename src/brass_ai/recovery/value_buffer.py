"""One committed iteration of critic-only rows, with bounded disk-backed reads.

The checkpoint owns the receipt. Publication precedes checkpoint/state commit;
deletion follows commit. A missing/corrupt derived cache is rebuilt from sealed
ordinary replays, never silently dropped or admitted to a later PPO update.
"""
import copy
from contextlib import contextmanager
import re
import time

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import compatibility,digest,project_path,require
from brass_ai.learning.features import Encoded,check_encoded,feature_schema
from brass_ai.step5.league import write_json
from .retention import owned,safe
from .updates import Dataset,state_value_model

CONTRACT='ppg-critic-disk-buffer/2.0.0'
# Directory ownership and tensor layout stay unchanged; v3 receipts also bind
# the complete collection's stratified selection, including excluded games.
STRATIFIED_CONTRACT='ppg-critic-disk-buffer/3.0.0'
OWNER=dict(contract=CONTRACT,purpose='disposable previous-iteration critic rows')
TARGETS=('win_share','placement','vp')
FILE=re.compile(r'generation-([0-9]{6,})\.(npy|json)(\.partial|\.tmp)?$')


def enabled(config):
    c=config.get('value_phase',{})
    return c.get('enabled',False) and c.get('buffer_iterations')==2


def retire_disabled(root,run,generation):
    """Retire both owned caches after durable PPO-only config adoption.

    The caller holds the run lock and has restored the checkpoint successfully.
    A reviewed source transition is sufficient; no new PPO iteration is needed.
    """
    root,run=owned(root,run);state=read(run/'state.json')
    require(state['iteration']==generation and state['signature']['config']['value_phase']['enabled'] is False,
        'PPG cache retirement requires committed PPO-only training state')
    return _retire_owned(root,run)


def retire_fresh(root,run,generation,*,reward_id):
    """Explicit fresh-PPG startup only, after durable reviewed config adoption."""
    from .ppg_config import coverage,active
    from .ppg_g210 import active as restored
    from brass_ai.learning.reward import CAPPED_VERSION
    root,run=owned(root,run);state=read(run/'state.json');value=state['signature']['config']['value_phase']
    capped=state['signature']['config'].get('search',{}).get('reward',{}).get('version')==CAPPED_VERSION
    require(state['iteration']==generation and (coverage(value) or restored(value) or (active(value) and capped)) and value['enabled'] is True,
        'Fresh PPG cleanup requires committed one-SE training state')
    from .utility import reward_spec
    require(digest(reward_spec(state['signature']['config']))==reward_id,
        'Fresh PPG cleanup requires the committed reward configuration')
    return _retire_owned(root,run)


def _retire_owned(root,run):
    from .ppg_replay import OWNER as EIGHT_OWNER,FILE as EIGHT_FILE
    files=[]
    directories=[]
    # Audit every file in BOTH caches before deleting anything. No recursion or
    # links: unknown artifacts require inspection and are never treated as cache.
    for name,owner,pattern in (('.ppg-value-cache',OWNER,FILE),('.ppg-eight-cache',EIGHT_OWNER,EIGHT_FILE)):
        directory=safe(run,run/name)
        if not directory.exists():continue
        require(directory.is_dir() and read(safe(directory,directory/'owner.json'))==owner,
            'Unknown PPG cache owner')
        directories.append(name)
        for p in directory.iterdir():
            if p.name=='owner.json':continue
            require(pattern.fullmatch(p.name) is not None and safe(directory,p).is_file(),'Unknown PPG cache artifact')
            files.append(p)
    total=sum(p.stat().st_size for p in files)
    for p in files:p.unlink()
    return dict(deleted_files=[p.name if p.parent.name=='.ppg-value-cache' else
        p.relative_to(run).as_posix() for p in files],deleted_bytes=total,directories=directories)


def row_dtype():
    schema=feature_schema()
    return np.dtype([('state','<f4',(schema['state_dim'],)),
        ('action','<f4',(schema['action_dim'],)),('mapping','<i8',(4,)),
        ('targets','<f4',(3,4)),('utility','<f4',(4,)),('remaining','<f4',(4,)),
        ('seat','<i8'),('game','<i4'),('context','<i4'),('forced','?'),
        ('descriptors','<f4',(4,34)),('role','i1'),('table','i1'),('descriptor_generation','<i8')])


class _Column:
    def __init__(self,rows,field):self.rows,self.field=rows,field
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i][self.field]


class _Record:
    def __init__(self,rows,index):self.rows,self.index=rows,index
    def __getitem__(self,field):return self.rows.field(self.index,field)


class DiskRows:
    def __init__(self,path,receipt):
        from .value_phase import heldout
        self.receipt=receipt
        self.array=np.load(path,mmap_mode='r',allow_pickle=False)
        try:
            require(self.array.dtype==row_dtype() and self.array.shape==(receipt['rows'],),
                    'Invalid PPG cache tensor layout')
            if receipt['contract']==STRATIFIED_CONTRACT:
                from .critic_holdout import validate_selection
                selection=validate_selection(receipt['holdout_selection'])
                require(set(receipt['game_ids'])<=set(selection['games'])-set(selection['heldout_games']),
                    'Held-out or unknown game in PPG cache')
            else:
                require(receipt['contract']==CONTRACT and all(not heldout(g,receipt['holdout_fraction'])
                            for g in receipt['game_ids']),'Held-out game in PPG cache')
        except BaseException:
            self.close();raise

    def __len__(self):return len(self.array)
    def __getitem__(self,i):
        if not 0<=i<len(self):raise IndexError(i)
        return _Record(self,i)

    def field(self,i,field):
        a=self.array[i]
        if field==0:return dict(seat=int(a['seat']),view=dict(context=self.receipt['contexts'][int(a['context'])]),
            _seat_descriptors=a['descriptors'],_critic_role=('learner','snapshot')[int(a['role'])],
            _table_kind=('selfplay','historical','legacy')[int(a['table'])],_descriptor_generation=int(a['descriptor_generation']))
        if field==1:return {k:a['targets'][j] for j,k in enumerate(TARGETS)}
        if field==2:return self.receipt['game_ids'][int(a['game'])],int(a['seat'])
        if field==3:return Encoded(a['state'],a['action'][None,:],a['mapping'],('critic-only',))
        raise IndexError(field)

    def close(self):self.array._mmap.close()


class _Joined:
    def __init__(self,left,right):self.left,self.right=left,right
    def __len__(self):return len(self.left)+len(self.right)
    def __getitem__(self,i):
        if not 0<=i<len(self):raise IndexError(i)
        return self.left[i] if i<len(self.left) else self.right[i-len(self.left)]


class ValueDataset(Dataset):
    """Current rows plus mmap features; only small target columns are resident."""
    def __init__(self,current,previous):
        self.data=_Joined(current.data,previous);self.device=current.device
        self.rows=_Column(self.data,0);self.encoded=_Column(self.data,3)
        self.q=None
        def joined(tensor,name):
            tail=torch.tensor(np.array(previous.array[name]),device=self.device)
            return torch.cat((tensor,tail))
        self.seats=joined(current.seats,'seat')
        self.utility=joined(current.utility,'utility');self.remaining=joined(current.remaining,'remaining')
        self.targets=tuple(torch.cat((t,torch.tensor(np.array(previous.array['targets'][:,j]),device=self.device)))
                           for j,t in enumerate(current.targets))
        self.descriptors=joined(current.descriptors,'descriptors')
        self.roles=current.roles+[('learner','snapshot')[int(r)] for r in previous.array['role']]
        self.tables=current.tables+[('selfplay','historical','legacy')[int(t)] for t in previous.array['table']]
        self.critic_weights=torch.ones(len(self.data),device=self.device)


class DiskBuffer:
    def __init__(self,root,run,config,*,generation,receipt=None,previous_reward=None,previous_value=None):
        from .utility import reward_spec,reviewed_reward_upgrade
        from .value_phase import stratified,refined_options,interval_options
        self.root,self.run=owned(root,run);self.config=config;self.generation=generation
        self.stratified=stratified(config['value_phase'])
        self.reward_id=digest(reward_spec(config))
        self.directory=safe(self.run,self.run/'.ppg-value-cache')
        self.directory.mkdir(parents=True,exist_ok=True)
        owner=safe(self.directory,self.directory/'owner.json')
        temporary=safe(self.directory,self.directory/'owner.json.tmp')
        if not owner.exists():
            existing=list(self.directory.iterdir())
            require(not existing or (existing==[temporary] and read(temporary)==OWNER),'Unowned PPG cache directory')
            write_json(owner,OWNER)
        elif temporary.exists():
            require(read(temporary)==OWNER,'Unknown temporary PPG owner');temporary.unlink()
        require(read(owner)==OWNER,'Unknown PPG cache owner')
        self.receipt=copy.deepcopy(receipt);self.pending=None;self.rebuilt=False
        self.reward_transition=None;self.holdout_transition=None
        require((generation==0)==(receipt is None),'Missing committed PPG buffer receipt')
        if receipt is not None:
            reward_change=receipt.get('reward_id')!=self.reward_id and reviewed_reward_upgrade(previous_reward,reward_spec(config))
            prior_refined=refined_options(dict(value_phase=previous_value)) if previous_value else {}
            holdout_change=bool(previous_value and not stratified(previous_value) and self.stratified and
                config['value_phase'] in (prior_refined['value_phase'],
                    interval_options(prior_refined)['value_phase']))
            if reward_change or holdout_change:
                # Exact reviewed transition: no relabeling of old targets or
                # selection. Retain old files until the next successful commit.
                self.validate_receipt(receipt,generation,reward_id=digest(previous_reward) if reward_change else None,
                    value_options=previous_value if holdout_change else None)
                if reward_change:
                    self.reward_transition=dict(contract='ppg-reward-transition/1.0.0',generation=generation,
                        from_reward_id=receipt['reward_id'],to_reward_id=self.reward_id,excluded_previous_rows=receipt['rows'],
                        scope='One collection uses current rows only; old-target cache retained until next commit')
                if holdout_change:
                    self.holdout_transition=dict(contract='ppg-holdout-transition/1.0.0',generation=generation,
                        excluded_previous_rows=receipt['rows'],
                        scope='Current rows only once at selection upgrade; old cache retained until next commit')
                self.receipt=None
            else:
                self.validate_receipt(receipt,generation)
                self.ensure()
        # Also retires uncommitted output after an interrupted iteration. The
        # committed generation's input is kept until the *next* state commit.
        self.startup_cleanup=self.cleanup(generation)

    def path(self,generation,ext):
        return safe(self.directory,self.directory/f'generation-{generation:06d}.{ext}')

    def write_receipt(self,generation,value):
        self.path(generation,'json.tmp')  # reject linked temporary targets too
        write_json(self.path(generation,'json'),value)

    def validate_receipt(self,r,generation,*,reward_id=None,value_options=None):
        from .value_phase import stratified
        options=self.config['value_phase'] if value_options is None else value_options
        current=stratified(options)
        require(type(r) is dict and r.get('contract')==(STRATIFIED_CONTRACT if current else CONTRACT) and r.get('generation')==generation
                and r.get('compatibility')==digest(compatibility())
                and r.get('reward_id')==(self.reward_id if reward_id is None else reward_id)
                and r.get('holdout_fraction')==options['holdout_fraction']
                and type(r.get('rows')) is int and r['rows']>=0,'Incompatible PPG buffer receipt')
        require(type(r.get('game_ids')) is list and len(set(r['game_ids']))==len(r['game_ids'])
                and type(r.get('contexts')) is list and type(r.get('source')) is dict,
                'Invalid PPG cache identities')
        if current:
            from .critic_holdout import validate_selection
            selection=validate_selection(r.get('holdout_selection'),options['holdout'],options['holdout_fraction'])
            require(set(r['game_ids'])<=set(selection['games'])-set(selection['heldout_games']),
                'Held-out or unknown game in PPG cache receipt')
        else:require('holdout_selection' not in r,'Unexpected stratified PPG receipt')

    def source(self,records,policy_id,kinds,model_id):
        selected=[]
        for r in sorted(records,key=lambda r:r['job']):
            p=safe(self.run,project_path(self.root,r['path']))
            require(p.parent==self.run/f'iteration-{self.generation:04d}','PPG cache needs current ordinary replays')
            selected.append(dict({k:r[k] for k in ('job','game_id','sha256','features')},
                                 path=p.relative_to(self.run).as_posix(),
                                 **({'native_admission':copy.deepcopy(r['native_admission'])} if 'native_admission' in r else {})))
        require(selected and len({r['game_id'] for r in selected})==len(selected),'Invalid PPG source collection')
        return dict(records=selected,policy_id=policy_id,kind_by_group=copy.deepcopy(kinds),model_id=model_id)

    def _write(self,rows,count,generation,source,expected=None,selection=None):
        from .value_phase import heldout
        if self.stratified:
            from .critic_holdout import validate_selection
            if expected is not None:selection=expected['holdout_selection']
            validate_selection(selection,self.config['value_phase']['holdout'],self.config['value_phase']['holdout_fraction'])
            excluded=set(selection['heldout_games'])
        part=self.path(generation,'npy.partial');a=np.lib.format.open_memmap(part,mode='w+',dtype=row_dtype(),shape=(count,))
        games={};contexts={};context_values=[];forced=0;n=0
        try:
            for row,target,identity,encoded in rows:
                require(n<count,'PPG cache row count increased')
                require(identity[0] not in excluded if self.stratified else
                        not heldout(identity[0],self.config['value_phase']['holdout_fraction']),
                        'Held-out row cannot enter PPG buffer')
                if self.stratified:
                    require(selection['games'].get(identity[0])==row.get('_table_kind','selfplay'),
                        'PPG source game/table differs from holdout selection')
                role=row.get('_critic_role','learner')
                require(row['_reward_id']==self.reward_id and role in ('learner','snapshot') and
                        ((role=='learner' and row['behavior']['policy_id']==source['policy_id']) or
                         (role=='snapshot' and source.get('snapshot_rows') and row['behavior']['policy_id']!=source['policy_id'])),
                        'PPG cache source/target identity differs')
                check_encoded(encoded)
                context=row['view']['context'];key=digest(context)
                if key not in contexts:contexts[key]=len(contexts);context_values.append(context)
                entry=a[n];entry['state']=encoded.state;entry['action']=encoded.actions[0]
                entry['mapping']=encoded.stable_to_relative
                entry['targets']=[target[k] for k in TARGETS]
                entry['utility']=row['_reward_return'];entry['remaining']=row['_reward_remaining']
                entry['seat']=row['seat'];entry['game']=games.setdefault(identity[0],len(games));entry['context']=contexts[key]
                entry['forced']=len(encoded.choice_ids)==1;forced+=int(entry['forced'])
                from brass_ai.learning.seat_descriptors import validate,unknown_numpy
                entry['descriptors']=validate(row.get('_seat_descriptors',unknown_numpy()))
                entry['role']=('learner','snapshot').index(role)
                entry['table']=('selfplay','historical','legacy').index(row.get('_table_kind','selfplay'))
                entry['descriptor_generation']=row.get('_descriptor_generation',generation-1)
                require(identity[1]==row['seat'] and 0<=row['seat']<4 and
                        all(np.isfinite(entry[k]).all() for k in ('state','action','targets','utility','remaining')),
                        'Invalid PPG cache features or targets')
                n+=1
            require(n==count,'PPG cache row count decreased');a.flush()
        finally:a._mmap.close()
        result=dict(contract=STRATIFIED_CONTRACT if self.stratified else CONTRACT,generation=generation,compatibility=digest(compatibility()),
            reward_id=self.reward_id,holdout_fraction=self.config['value_phase']['holdout_fraction'],
            **(dict(holdout_selection=copy.deepcopy(selection)) if self.stratified else {}),
            source=copy.deepcopy(source),rows=count,forced_rows=forced,game_ids=list(games),contexts=context_values,
            sha256=sha(part),bytes=part.stat().st_size)
        require(expected is None or result==expected,'Rebuilt PPG cache differs from committed receipt')
        part.replace(self.path(generation,'npy'));self.write_receipt(generation,result)
        return result

    def _rebuild_rows(self):
        from .ingress import load_collection
        from .utility import reward_spec
        from .value_phase import heldout
        source=self.receipt['source'];groups={}
        require(source['records'],'PPG buffer missing and no sealed replay receipts are available')
        for r in source['records']:
            p=safe(self.run,self.run/r['path'])
            require(p.parent==self.run/f'iteration-{self.generation-1:04d}','Wrong PPG rebuild generation')
            groups.setdefault(str(p),[]).append(dict(r,path=str(p)))
        for role in (('learner','snapshot') if source.get('snapshot_rows') else ('learner',)):
            for records in groups.values():
                # Learners then snapshots, in original deterministic shard
                # order, exactly as the staged critic union. Keep one shard
                # resident; an exceptional rebuild may read it twice.
                _,data,_=load_collection(records,source['policy_id'],source['kind_by_group'],
                    insolvency=self.config.get('insolvency'),reward=reward_spec(self.config))
                for d in data:
                    if d[0].get('_critic_role','learner')!=role:continue
                    if self.stratified:
                        selection=self.receipt['holdout_selection']
                        require(selection['games'].get(d[2][0])==d[0].get('_table_kind','selfplay'),
                            'PPG replay differs from committed holdout selection')
                        if d[2][0] not in selection['heldout_games']:yield d
                    elif not heldout(d[2][0],self.receipt['holdout_fraction']):yield d

    def ensure(self):
        r=self.receipt;p=self.path(r['generation'],'npy');meta=self.path(r['generation'],'json')
        if (p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256']):
            # A crash between data publication and receipt publication is safe.
            try:valid=meta.exists() and read(meta)==r
            except (OSError,ValueError):valid=False
            if not valid:self.write_receipt(r['generation'],r)
            return
        self._write(self._rebuild_rows(),r['rows'],r['generation'],r['source'],expected=r)
        self.rebuilt=True

    def stage(self,dataset,train_ids,source,model):
        require(state_value_model(model),'Disk PPG buffer requires stock state-only value heads')
        require(self.pending is None,'PPG cache already staged')
        if self.receipt:
            previous_games=self.receipt.get('holdout_selection',{}).get('games',self.receipt['game_ids'])
            require(not (set(previous_games) & {d[2][0] for d in dataset.data}),
                    'Previous and current PPG games overlap')
        began=time.perf_counter()
        selection=None
        if self.stratified:
            from .critic_holdout import partition
            _,_,report=partition(dataset.data,self.config['value_phase']['holdout_fraction'],self.config['value_phase']['holdout'])
            selection=report['selection']
        self.pending=self._write((dataset.data[i] for i in train_ids),len(train_ids),self.generation+1,source,selection=selection)
        excluded=self.path(self.generation,'npy') if (self.reward_transition or self.holdout_transition) and not self.receipt else None
        excluded_bytes=excluded.stat().st_size if excluded is not None and excluded.is_file() else 0
        return dict(storage='disk-backed one-candidate critic rows',buffer_iterations=2,
            current_generation=self.generation+1,previous_generation=self.generation if self.receipt else None,
            current_rows=len(train_ids),previous_rows=self.receipt['rows'] if self.receipt else 0,
            current_forced_rows=self.pending['forced_rows'],previous_forced_rows=self.receipt['forced_rows'] if self.receipt else 0,
            current_cache_bytes=self.pending['bytes'],previous_cache_bytes=self.receipt['bytes'] if self.receipt else 0,
            excluded_cache_bytes=excluded_bytes,
            peak_cache_bytes=self.pending['bytes']+(self.receipt['bytes'] if self.receipt else 0)+excluded_bytes,
            write_seconds=time.perf_counter()-began,rebuilt_from_replay=self.rebuilt,
            startup_cleanup=self.startup_cleanup,committed_cleanup=None,reward_transition=self.reward_transition,
            holdout_transition=self.holdout_transition)

    @contextmanager
    def previous(self):
        if self.receipt is None:yield None;return
        self.ensure();data=DiskRows(self.path(self.generation,'npy'),self.receipt)
        try:yield data
        finally:data.close()

    def cleanup(self,generation):
        began=time.perf_counter();files=[];removed=[];total=0
        require(read(self.directory/'owner.json')==OWNER,'Unknown PPG cache owner')
        # Audit every target before unlinking anything; never recursive deletion.
        for p in self.directory.iterdir():
            if p.name=='owner.json':continue
            match=FILE.fullmatch(p.name)
            require(match is not None and safe(self.directory,p).is_file(),'Unknown PPG cache artifact')
            if int(match[1])!=generation or match[3]:files.append(p)
        for p in files:
            total+=p.stat().st_size;removed.append(p.name);p.unlink()
        return dict(deleted_files=removed,deleted_bytes=total,seconds=time.perf_counter()-began)

    def committed(self,generation):
        require(self.pending is not None and self.pending['generation']==generation==self.generation+1,
                'PPG cache commit generation differs')
        require(read(self.run/'state.json')['iteration']==generation,'PPG cleanup requires committed training state')
        self.receipt=self.pending;self.pending=None;self.generation=generation
        result=self.cleanup(generation);self.rebuilt=False;self.startup_cleanup=None
        self.reward_transition=None;self.holdout_transition=None
        return result
