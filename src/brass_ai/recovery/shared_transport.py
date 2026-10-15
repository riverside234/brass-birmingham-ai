"""Bounded one-request slots; ownership transfers at queue/pipe boundaries.

Only numeric permitted-observation features and predictions use shared memory.
Model/observer/revision/digest/choice IDs remain in the small control message.
Oversized frames use the original queue transport, never truncate candidates.
"""
from multiprocessing.shared_memory import SharedMemory
import numpy as np
from brass_ai.learning.contracts import require
from brass_ai.learning.features import Encoded,check_encoded

HEADER=64

class SharedSlot:
    def __init__(self,name=None,size=2*1024*1024):
        self.owner=name is None
        self.memory=SharedMemory(name=name,create=self.owner,size=size if self.owner else 0)
        self.size=self.memory.size;self.closed=False

    @property
    def description(self):return (self.memory.name,self.size)

    def close(self):
        if self.closed:return
        self.memory.close()
        if self.owner:
            try:self.memory.unlink()
            except FileNotFoundError:pass
        self.closed=True

    def header(self,token,kind,write=False):
        require(isinstance(token,str) and len(token)==32 and all(c in '0123456789abcdef' for c in token),'Invalid shared token')
        value=('brass-shm/1:'+kind+':'+token).encode('ascii').ljust(HEADER,b'\0')
        if write:self.memory.buf[:HEADER]=value
        else:require(bytes(self.memory.buf[:HEADER])==value,'Stale or misrouted shared slot')

    def put(self,array,offset):
        array=np.asarray(array);offset=(offset+7)//8*8
        require(array.dtype.str in ('<f4','<f8','<i8','<i4'),'Unsupported shared dtype')
        end=offset+array.nbytes;require(end<=self.size,'Shared slot overflow')
        np.ndarray(array.shape,dtype=array.dtype,buffer=self.memory.buf,offset=offset)[:]=array
        return dict(offset=offset,shape=list(array.shape),dtype=array.dtype.str),end

    def get(self,desc,copy=False):
        require(type(desc) is dict and set(desc)=={'offset','shape','dtype'},'Invalid shared array descriptor')
        offset,shape,dtype=desc['offset'],desc['shape'],desc['dtype']
        require(type(offset) is int and offset>=HEADER and offset%8==0 and type(shape) is list and 1<=len(shape)<=2 and
                all(type(n) is int and n>0 for n in shape) and dtype in ('<f4','<f8','<i8','<i4'),'Invalid shared array layout')
        length=1
        for n in shape:length*=n
        require(offset+length*np.dtype(dtype).itemsize<=self.size,'Shared array exceeds slot')
        out=np.ndarray(tuple(shape),dtype=dtype,buffer=self.memory.buf,offset=offset)
        if copy:out=out.copy()
        out.setflags(write=False);return out

    def pack_frames(self,frames,token):
        arrays=[a for f in frames if f['encoded'] is not None for a in
                (f['encoded'].state,f['encoded'].actions,f['encoded'].stable_to_relative)]
        if HEADER+sum(a.nbytes+7 for a in arrays)>self.size:return None
        self.header(token,'input',True);offset=HEADER;result=[]
        for f in frames:
            e=f['encoded'];f=dict(f)
            if e is not None:
                check_encoded(e)
                descriptors=[]
                for a in (e.state,e.actions,e.stable_to_relative):
                    descriptor,offset=self.put(a,offset);descriptors.append(descriptor)
                f['encoded']=dict(shared=descriptors,choice_ids=e.choice_ids,feature_version=e.feature_version)
            result.append(f)
        return result

    def unpack_frames(self,frames,token):
        self.header(token,'input');out=[]
        for f in frames:
            f=dict(f);e=f['encoded']
            if e is not None:
                require(set(e)=={'shared','choice_ids','feature_version'} and len(e['shared'])==3,'Invalid shared encoding')
                f['encoded']=Encoded(*(self.get(d) for d in e['shared']),tuple(e['choice_ids']),e['feature_version'])
                check_encoded(f['encoded'])
            out.append(f)
        return out

    def pack_responses(self,responses,token):
        keys=('probabilities','win_share','placement','vp','utility')
        size=HEADER+sum(8*len(r[k])+7 for r in responses for k in keys if k in r)
        if size>self.size:return None
        self.header(token,'output',True);offset=HEADER;result=[]
        for response in responses:
            response=dict(response)
            for key in keys:
                if key in response:response[key],offset=self.put(np.asarray(response[key],dtype='<f8'),offset)
            result.append(response)
        return result

    def unpack_responses(self,responses,token):
        self.header(token,'output');out=[]
        for response in responses:
            response=dict(response)
            for key in ('probabilities','win_share','placement','vp','utility'):
                if key in response:response[key]=self.get(response[key]).tolist()
            out.append(response)
        return out
