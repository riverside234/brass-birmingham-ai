"""Checkpoint-owned eight-age mmap replay. Never a source of PPO trajectories."""
import copy
from contextlib import contextmanager
import re
import time

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import compatibility,digest,require
from brass_ai.learning.features import Encoded,check_encoded,feature_schema
from brass_ai.step5.league import write_json
from .retention import owned,safe
from .value_buffer import row_dtype,DiskRows,_Column
from .updates import Dataset,state_value_model
from . import ppg_sampling

CONTRACT='ppg-eight-disk-buffer/1.0.0'
OWNER=dict(contract=CONTRACT,purpose='sealed MC critic rows and selected complete policy inputs')
FILE=re.compile(r'generation-([0-9]{6,})\.(critic\.npy|actions\.npy|local-scoring\.json|json)(\.partial|\.tmp)?$')


class Rows(DiskRows):
    def __init__(self,path,receipt):
        self.receipt=receipt;self.array=np.load(path,mmap_mode='r',allow_pickle=False)
        self.local={}
        try:
            if 'local-scoring.json' in receipt['files']:
                from .local_scoring_targets import CONTRACT,seal,unpack
                sidecar=path.with_name(path.name.replace('critic.npy','local-scoring.json'))
                require(sha(sidecar)==receipt['files']['local-scoring.json']['sha256'],'Local scoring sidecar changed')
                value=read(sidecar)
                require(value['contract']==CONTRACT and value['generation']==receipt['generation'] and
                    value['source']==digest(receipt['source']) and value['holdout']==digest(receipt['holdout_selection']) and
                    set(value['games'])<=set(receipt['game_ids']),'Local scoring sidecar input/holdout binding differs')
                self.local={g:unpack(e) for g,e in value['games'].items()}
                require(all(g==e['game_id'] and e['sha256']==seal(e) for g,e in self.local.items()),'Invalid cached local scoring envelope')
            require(self.array.dtype==row_dtype() and self.array.shape==(receipt['rows'],),'Invalid eight-age critic layout')
        except BaseException:self.close();raise

    def field(self,i,field):
        value=super().field(i,field)
        if field==0:
            game=self.receipt['game_ids'][int(self.array[i]['game'])]
            if game in self.local:
                value['_local_scoring']=dict(envelope=self.local[game],era=value['view']['context']['era'])
        return value


class Joined:
    def __init__(self,rows):
        self.parts=rows;self.ends=np.cumsum([len(r) for r in rows]).tolist()
    def __len__(self):return self.ends[-1] if self.ends else 0
    def locate(self,i):
        if not 0<=i<len(self):raise IndexError(i)
        import bisect
        p=bisect.bisect_right(self.ends,i)
        return self.parts[p],i-(self.ends[p-1] if p else 0)
    def __getitem__(self,i):
        part,j=self.locate(i);return part[j]

    def column(self,indices,name):
        """Vectorized gathers per mmap, retaining duplicates and request order."""
        indices=np.asarray(indices,dtype=np.int64)
        require(indices.ndim==1 and ((indices>=0)&(indices<len(self))).all(),'Invalid PPG row indices')
        parts=np.searchsorted(self.ends,indices,side='right')
        first=self.parts[0].array[name]
        result=np.empty((len(indices),*first.shape[1:]),dtype=first.dtype)
        for p in np.unique(parts):
            selected=np.flatnonzero(parts==p)
            local=indices[selected]-(self.ends[p-1] if p else 0)
            result[selected]=self.parts[p].array[name][local]
        return result


class Column:
    """Gather only a minibatch of small metadata onto the training device."""
    def __init__(self,data,name,device,head=None):self.data,self.name,self.device,self.head=data,name,device,head
    def __getitem__(self,indices):
        tail=None
        if isinstance(indices,tuple):indices,tail=indices
        if isinstance(indices,torch.Tensor):indices=indices.detach().cpu().tolist()
        if isinstance(indices,int):indices=[indices]
        values=self.data.column(indices,self.name)
        if self.head is not None:values=values[:,self.head]
        t=torch.as_tensor(values,device=self.device)
        if tail is not None:
            if isinstance(tail,torch.Tensor):tail=tail.to(self.device)
            t=t[torch.arange(len(indices),device=self.device),tail]
        return t


class CriticDataset(Dataset):
    def __init__(self,rows,device):
        self.data=Joined(rows);self.device=device;self.rows=_Column(self.data,0);self.encoded=_Column(self.data,3)
        self.q=None
        self.seats=Column(self.data,'seat',device);self.descriptors=Column(self.data,'descriptors',device)
        self.utility=Column(self.data,'utility',device);self.remaining=Column(self.data,'remaining',device)
        self.targets=tuple(Column(self.data,'targets',device,j) for j in range(3))
        self.roles=[('learner','snapshot')[int(x)] for p in rows for x in p.array['role']]
        self.tables=[('selfplay','historical','legacy')[int(x)] for p in rows for x in p.array['table']]

    def microbatches(self,indices,size,cells,bucket=True):
        # This sealed critic format contains exactly one real candidate/row.
        indices=list(indices);size=min(size,cells)
        for start in range(0,len(indices),size):yield indices[start:start+size]

    def batch(self,indices,exploration=0,temperature=1,*,value_only=False):
        require(not exploration,'Critic replay cannot collect exploration probabilities')
        from brass_ai.learning.local_entities import check_action_selectors
        states=self.data.column(indices,'state');actions=self.data.column(indices,'action')
        mapping=self.data.column(indices,'mapping')
        check_action_selectors(actions)
        inputs=tuple(torch.as_tensor(v,device=self.device) for v in
            (states,actions[:,None,:],np.ones((len(indices),1),dtype=np.bool_),mapping))
        return torch.tensor(indices,device=self.device),inputs,torch.zeros((len(indices),1),device=self.device)


class PolicyData:
    def __init__(self,rows,actions,entries):self.rows,self.actions,self.entries=rows,actions,entries
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        if not 0<=i<len(self):raise IndexError(i)
        e=self.entries[i];record=self.rows[e['row']];row=record[0]
        a,b=e['offset'],e['offset']+e['count'];ids=tuple(str(j) for j in range(b-a))
        row=dict(row,choices=range(b-a))
        encoded=record[3]
        return row,record[1],record[2],Encoded(encoded.state,self.actions[a:b],encoded.stable_to_relative,ids)


class PolicyDataset(Dataset):
    def __init__(self,parts,device):
        from .batch_cache import InputAudit,BatchCache
        self.data=Joined(parts);self.device=device;self.rows=_Column(self.data,0);self.encoded=_Column(self.data,3)
        self.q=None;self.input_audit=InputAudit();self.scan_cache=BatchCache(limit=0)


class Buffer:
    def __init__(self,root,run,config,*,generation,receipts=None,origin=0):
        from .utility import reward_spec
        self.root,self.run=owned(root,run);self.config=config;self.generation=generation;self.origin=origin
        self.reward_id=digest(reward_spec(config));self.pending=None
        self.directory=safe(self.run,self.run/'.ppg-eight-cache');self.directory.mkdir(parents=True,exist_ok=True)
        owner=safe(self.directory,self.directory/'owner.json')
        if not owner.exists():
            existing=list(self.directory.iterdir())
            require(not existing or all(p.name=='owner.json.tmp' and read(p)==OWNER for p in existing),'Unowned PPG eight-age cache')
            safe(self.directory,self.directory/'owner.json.tmp')
            write_json(owner,OWNER)
        require(read(owner)==OWNER,'Unknown eight-age PPG owner')
        self.receipts=copy.deepcopy(receipts or [])
        require([r['generation'] for r in self.receipts]==list(range(max(origin+1,generation-7),generation+1)),
            'Missing committed eight-age PPG cache receipts')
        for r in self.receipts:self.verify(r)

    def path(self,generation,ext):return safe(self.directory,self.directory/f'generation-{generation:06d}.{ext}')

    def verify(self,r):
        from .critic_holdout import validate_selection
        require(r.get('contract')==CONTRACT and r.get('reward_id')==self.reward_id
            and r.get('compatibility')==digest(compatibility()) and type(r.get('rows')) is int and r['rows']>0,
            'Incompatible eight-age PPG receipt')
        validate_selection(r['holdout_selection'],self.config['value_phase']['holdout'],self.config['value_phase']['holdout_fraction'])
        require(set(r['game_ids'])<=set(r['holdout_selection']['games']),'Unknown PPG cached game')
        for ext,f in r['files'].items():
            require(ext in ('critic.npy','actions.npy','local-scoring.json'),'Unknown PPG cache file type')
            p=self.path(r['generation'],ext)
            require(p.is_file() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],
                'Missing/corrupt committed eight-age PPG cache; restore its matching cache backup: '+str(p))
        require(read(self.path(r['generation'],'json'))==r,'PPG cache metadata differs from checkpoint')
        if self.config.get('local_scoring_auxiliary',{}).get('mode','off')!='off':
            require('local-scoring.json' in r['files'],'Missing adopted PPG local scoring sidecar')

    def source(self,records,policy_id,kinds,model_id):
        # Reuse the existing checked ordinary-replay provenance builder.
        from .value_buffer import DiskBuffer
        return DiskBuffer.source(self,records,policy_id,kinds,model_id)

    def stage(self,dataset,train_ids,heldout_ids,actor_ids,source,model,selection):
        require(self.pending is None and state_value_model(model),'Eight-age PPG needs an unstaged state-only critic')
        from .critic_holdout import validate_selection
        validate_selection(selection,self.config['value_phase']['holdout'],self.config['value_phase']['holdout_fraction'])
        ids=sorted(set(train_ids)|set(heldout_ids));index={i:j for j,i in enumerate(ids)}
        require(set(train_ids).isdisjoint(heldout_ids) and ids,'Invalid PPG train/holdout partition')
        held=set(selection['heldout_games']);held_ids=set(heldout_ids)
        require(all((dataset.data[i][2][0] in held)==(i in held_ids) for i in ids),'PPG holdout/family leakage')
        games_now=set(selection['games'])
        require(all(games_now.isdisjoint(r['holdout_selection']['games']) for r in self.receipts),'PPG cache game identities overlap')
        # Holdout families enter diagnostics only during this auxiliary phase;
        # even the cloning loss must not train on their observations. Ordinary
        # PPO retains its existing actor/critic holdout semantics.
        training=set(train_ids)
        selected,counts=ppg_sampling.select(dataset.data,[i for i in actor_ids if i in training],self.config['value_phase'])
        require(all(e['index'] in index for e in selected),'Policy row missing critic reference')
        generation=self.generation+1;began=time.perf_counter()
        import shutil
        action_count=sum(len(dataset.data[e['index']][3].choice_ids) for e in selected)
        estimated=len(ids)*row_dtype().itemsize+action_count*feature_schema()['action_dim']*4+16*1024**2
        require(shutil.disk_usage(self.directory).free>=estimated,
            'Insufficient free disk for staged PPG cache; committed caches are never deleted to make room')
        disk_peak=sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        part=self.path(generation,'critic.npy.partial')
        a=np.lib.format.open_memmap(part,mode='w+',dtype=row_dtype(),shape=(len(ids),))
        games={};contexts={};context_values=[]
        try:
            for j,i in enumerate(ids):
                row,target,identity,encoded=dataset.data[i];role=row.get('_critic_role','learner')
                require(role in ('learner','snapshot') and row['_reward_id']==self.reward_id
                    and selection['games'].get(identity[0])==row.get('_table_kind','selfplay')
                    and row.get('behavior',{}).get('component')!='branch_move' and not row.get('_copied_prefix',False)
                    and ((role=='learner' and row['behavior']['policy_id']==source['policy_id']) or
                         (role=='snapshot' and source.get('snapshot_rows') and row['behavior']['policy_id']!=source['policy_id'])),
                    'Invalid cached MC row/source')
                check_encoded(encoded);context=row['view']['context'];key=digest(context)
                if key not in contexts:contexts[key]=len(contexts);context_values.append(copy.deepcopy(context))
                e=a[j];e['state']=encoded.state;e['action']=encoded.actions[0];e['mapping']=encoded.stable_to_relative
                e['targets']=[target[k] for k in ('win_share','placement','vp')]
                e['utility']=row['_reward_return'];e['remaining']=row['_reward_remaining']
                e['seat']=row['seat'];e['game']=games.setdefault(identity[0],len(games));e['context']=contexts[key]
                e['forced']=len(encoded.choice_ids)==1
                from brass_ai.learning.seat_descriptors import validate,unknown_numpy
                e['descriptors']=validate(row.get('_seat_descriptors',unknown_numpy()))
                e['role']=('learner','snapshot').index(role);e['table']=('selfplay','historical','legacy').index(row.get('_table_kind','selfplay'))
                e['descriptor_generation']=row.get('_descriptor_generation',generation-1)
                require(identity[1]==row['seat'] and all(np.isfinite(e[k]).all() for k in ('state','action','targets','utility','remaining')),
                    'Nonfinite/mismatched cached critic row')
            a.flush()
        finally:a._mmap.close()
        disk_peak=max(disk_peak,sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file()))
        part.replace(self.path(generation,'critic.npy'))
        total=action_count
        part=self.path(generation,'actions.npy.partial')
        actions=np.lib.format.open_memmap(part,mode='w+',dtype='<f4',shape=(total,feature_schema()['action_dim']))
        entries=[];offset=0
        try:
            from brass_ai.learning.local_entities import check_action_selectors
            for e in selected:
                encoded=dataset.data[e['index']][3];n=len(encoded.choice_ids)
                # Full legal action distributions are never narrowed in policy caches.
                check_action_selectors(encoded.actions)
                actions[offset:offset+n]=encoded.actions
                entries.append(dict(row=index[e['index']],offset=offset,count=n,**{k:e[k] for k in ('stratum','rank','split')}));offset+=n
            actions.flush()
        finally:actions._mmap.close()
        disk_peak=max(disk_peak,sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file()))
        part.replace(self.path(generation,'actions.npy'))
        files={ext:dict(bytes=self.path(generation,ext).stat().st_size,sha256=sha(self.path(generation,ext))) for ext in ('critic.npy','actions.npy')}
        if 'local_scoring_auxiliary' in self.config:
            from .local_scoring_targets import CONTRACT as TARGET_CONTRACT,seal,pack
            local={};checked=set()
            for i in ids:
                row=dataset.rows[i];label=row.get('_local_scoring');game=dataset.data[i][2][0]
                if label is None:
                    require(self.config['local_scoring_auxiliary']['mode']=='off','New critic row lacks local scoring labels')
                    continue
                envelope=label['envelope']
                require(envelope['game_id']==game,'PPG auxiliary game binding differs')
                token=(game,id(envelope))
                if token not in checked:
                    require(envelope['sha256']==seal(envelope),'PPG auxiliary label binding differs');checked.add(token)
                if game in local:require(local[game]['sha256']==envelope['sha256'],'Mixed era-map sources')
                else:local[game]=envelope
            sidecar=dict(contract=TARGET_CONTRACT,generation=generation,source=digest(source),holdout=digest(selection),games={g:pack(e) for g,e in local.items()})
            self.path(generation,'local-scoring.json.tmp')
            write_json(self.path(generation,'local-scoring.json'),sidecar)
            files['local-scoring.json']=dict(bytes=self.path(generation,'local-scoring.json').stat().st_size,sha256=sha(self.path(generation,'local-scoring.json')))
        r=dict(contract=CONTRACT,generation=generation,compatibility=digest(compatibility()),reward_id=self.reward_id,
            rows=len(ids),game_ids=list(games),contexts=context_values,holdout_selection=copy.deepcopy(selection),
            source=copy.deepcopy(source),policy_entries=entries,policy_population=counts,files=files,
            train_rows=len(train_ids),heldout_rows=len(heldout_ids))
        self.path(generation,'json.tmp')
        write_json(self.path(generation,'json'),r)
        self.pending=(self.receipts+[r])[-8:]
        return dict(buffer_iterations=8,ages=list(range(len(self.pending))),generations=[r['generation'] for r in self.pending],
            rows_by_age={str(generation-x['generation']):x['rows'] for x in self.pending},
            critic_bytes=sum(x['files']['critic.npy']['bytes'] for x in self.pending),
            policy_action_bytes=sum(x['files']['actions.npy']['bytes'] for x in self.pending),
            auxiliary_label_bytes=sum(x['files'].get('local-scoring.json',{}).get('bytes',0) for x in self.pending),
            peak_disk_bytes=max(disk_peak,sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())),
            write_seconds=time.perf_counter()-began,committed_cleanup=None,
            scope='Eight trainable ages; pre-commit disk peak includes the not-yet-expired ninth cache. Mmap features; bounded minibatch transfers; no historical PPO admission.')

    @contextmanager
    def datasets(self,device):
        require(self.pending is not None,'No staged PPG cache')
        rows=[];actions=[];policies=[]
        try:
            for r in reversed(self.pending):
                rows.append(Rows(self.path(r['generation'],'critic.npy'),r))
                actions.append(np.load(self.path(r['generation'],'actions.npy'),mmap_mode='r',allow_pickle=False))
                require(actions[-1].dtype==np.dtype('<f4') and actions[-1].shape[1]==feature_schema()['action_dim'],'Invalid policy action cache')
                policies.append(PolicyData(rows[-1],actions[-1],r['policy_entries']))
            yield CriticDataset(rows,device),PolicyDataset(policies,device),list(reversed(self.pending))
        finally:
            for a in actions:a._mmap.close()
            for r in rows:r.close()

    def committed(self,generation):
        require(self.pending is not None and self.pending[-1]['generation']==generation==self.generation+1
            and read(self.run/'state.json')['iteration']==generation,'PPG cache expiry requires checkpoint/state commit')
        keep={r['generation'] for r in self.pending};files=[]
        require(read(self.directory/'owner.json')==OWNER,'PPG cache owner changed')
        for p in self.directory.iterdir():
            if p.name=='owner.json':continue
            match=FILE.fullmatch(p.name)
            require(match is not None and safe(self.directory,p).is_file(),'Unknown PPG cache artifact')
            if int(match[1]) not in keep or match[3]:files.append(p)
        total=sum(p.stat().st_size for p in files)
        for p in files:p.unlink()
        self.receipts=self.pending;self.pending=None;self.generation=generation
        return dict(deleted_files=[p.name for p in files],deleted_bytes=total)
