"""Derived context fingerprints; fixed probe and policy weights stay immutable."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import digest,require,project_path
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.batching import collate,model_id
from brass_ai.learning.features import Encoded,feature_schema
from brass_ai.step5.policy import context_for
from brass_ai.step5.league import write_json
from .domino_config import CONTEXT,ENCODING,NAMES


def _versions(model):
    return tuple((id(t),t._version) for t in (*model.parameters(),*model.buffers()))


def inventory(root,state):
    """All selectable weights, even when a small collection will not draw them."""
    from brass_ai.learning.model import ARCHITECTURE
    entries={e['model_id']:e for e in state['quality_pool']['members']}
    for mid,e in entries.items():
        p=project_path(root,e['path']);require(p.is_file() and sha(p)==e['sha256'],'DOMiNO unavailable checkpoint: '+str(p))
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            checked,raw=load_checkpoint(p)
        require(raw['model_id']==mid and raw['model_config']['architecture']==ARCHITECTURE,
            'DOMiNO unsupported seven-context checkpoint: '+str(p))
        require(len(feature_schema()['styles'])==4 and feature_schema()['intent_dim']==8,
            'DOMiNO context interface differs: '+str(p))
        del checked,raw
    return entries


class Fingerprints:
    def __init__(self,root,run,probe,model,update,entries,*,neutral=None):
        self.root,self.run,self.probe,self.model,self.update,self.entries=root,run,probe,model,update,entries
        self.current=model_id(model);self.cache={};self.report=dict(hits=0,misses=0,seconds=0.,positions=0)
        self.models={self.current:model}
        self.versions={self.current:_versions(model)}
        self.directory=Path(run)/'domino-fingerprints';self.directory.mkdir(parents=True,exist_ok=True)
        if neutral is not None:
            from brass_ai.learning.state_layout import state_layout
            ix=next(g.indices for g in state_layout().groups if g.name=='policy_context')
            style,intent=context_for('neutral')
            require(all(np.array_equal(e.state[list(ix)],np.asarray(style+intent,dtype=np.float32)) for e in probe.encoded),
                'Neutral fingerprint reuse requires unchanged neutral probe contexts')
            key=self.identity(self.current,0)
            if (self.directory/(digest(key)+'.json')).exists():self.get(self.current,0)
            else:self.save(key,dict(neutral,context_identity=key))

    def identity(self,mid,z):
        require(type(z) is int and 0<=z<7,'Invalid fingerprint context')
        return dict(model_id=mid,strategy_id=z,probe_hash=self.probe.metadata['probe_hash'],
            context_contract=CONTEXT,context_encoding=ENCODING,fingerprint_contract='policy-style-fingerprint/1.0.0')

    def save(self,key,result):
        self.check(key,result)
        token=digest(key);path=self.directory/(token+'.json')
        value=dict(identity=key,fingerprint=result,seal=digest(result))
        if path.exists():require(read(path)==value,'Context fingerprint identity reused with different values')
        else:write_json(path,value)
        self.cache[token]=result
        return result

    def check(self,key,result):
        require(result.get('context_identity')==key and result.get('probe_hash')==key['probe_hash'] and
                result.get('contract')==key['fingerprint_contract'],'Fingerprint context/probe differs')

    def get(self,mid,z):
        from .critic_data import fingerprint
        key=self.identity(mid,z)
        token=digest(key)
        if token in self.cache:self.report['hits']+=1;return self.cache[token]
        path=self.directory/(token+'.json')
        if path.exists():
            value=read(path);require(value['identity']==key and value['seal']==digest(value['fingerprint']), 'Context fingerprint cache changed')
            self.check(key,value['fingerprint'])
            self.cache[token]=value['fingerprint'];self.report['hits']+=1;return self.cache[token]
        start=time.perf_counter();model=self.model
        if mid not in self.models:
            e=self.entries[mid];p=project_path(self.root,e['path']);require(sha(p)==e['sha256'],'Fingerprint checkpoint changed')
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):model,raw=load_checkpoint(p,device=next(self.model.parameters()).device)
            require(raw['model_id']==mid,'Fingerprint model identity differs')
            self.models[mid]=model;self.versions[mid]=_versions(model)
        model=self.models[mid]
        require(_versions(model)==self.versions[mid],'Fingerprint weights changed after collection identity was frozen')
        values=np.empty_like(self.probe.arrays['reference']);style,intent=context_for(NAMES[z])
        from brass_ai.learning.state_layout import state_layout
        context_indices=next(g.indices for g in state_layout().groups if g.name=='policy_context')
        require(len(context_indices)==12,'DOMiNO fingerprint context layout differs')
        was_training=model.training;model.eval()
        try:
            with torch.no_grad():
                for part in self.probe.parts(self.update):
                    encoded=[]
                    for i in part:
                        row=self.probe.encoded[i];state=row.state.copy();state[list(context_indices)]=style+intent
                        encoded.append(Encoded(state,row.actions,row.stable_to_relative,row.choice_ids,row.feature_version))
                    inputs=collate(encoded,next(model.parameters()).device)
                    probs=model(*inputs)[0].float().softmax(-1).cpu().numpy()
                    for j,i in enumerate(part):
                        a,b=map(int,self.probe.arrays['offsets'][i:i+2]);values[a:b]=probs[j,:b-a]
        finally:model.train(was_training)
        result=dict(fingerprint(self.probe,values),context_identity=key)
        self.save(key,result);self.report['misses']+=1;self.report['positions']+=len(self.probe.rows)
        self.report['seconds']+=time.perf_counter()-start
        return result
