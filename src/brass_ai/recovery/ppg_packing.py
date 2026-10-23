"""Lossless, generation-local column packing. No quantization or encoder cache.

The model still receives the original FP32 tensors. Only verified 0/1 columns
are bit-packed; constants retain their uint32 bit patterns. True floating
columns, targets and remaining costs retain all bits, including signed zero.
"""
import numpy as np

from brass_ai.learning.contracts import require
from brass_ai.learning.features import feature_schema
from brass_ai.learning.local_entities import entity_layout,check_action_selectors
from .dcpg_config import PACKED_CACHE as CONTRACT
from .value_buffer import row_dtype

EXTENSIONS=('meta.npy','games.npy','state-floats.npy','state-bits.npy',
            'critic-actions.npy','policy-actions.npy')
BLOCK=1024
META=np.dtype([('mapping','u1',(4,)),('utility','<f4',(4,)),('remaining','<f4',(4,)),
    ('seat','u1'),('game','<u4'),('context','<u4'),('forced','?'),('role','i1'),('table','i1')])
GAME=np.dtype([('descriptors','<f4',(4,34)),('targets','<f4',(3,4)),('descriptor_generation','<i8')])


def _blocks(n):
    for i in range(0,n,BLOCK):yield slice(i,min(n,i+BLOCK))


def _write(path,dtype,shape,fill):
    part=path.with_name(path.name+'.partial')
    a=np.lib.format.open_memmap(part,mode='w+',dtype=dtype,shape=shape)
    try:fill(a);a.flush()
    finally:a._mmap.close()
    part.replace(path)


def state_layout(values):
    first=np.array(values[0],dtype='<f4').view('<u4')
    constant=np.ones(len(first),dtype=bool);binary=constant.copy()
    for part in _blocks(len(values)):
        bits=np.asarray(values[part],dtype='<f4').view('<u4')
        require(np.isfinite(values[part]).all(),'Nonfinite packed PPG states')
        constant &= np.all(bits==first,axis=0)
        binary &= np.all((bits==0)|(bits==0x3f800000),axis=0)
    binary &= ~constant
    return dict(width=len(first),constant=np.flatnonzero(constant).tolist(),
        constant_bits=first[constant].tolist(),binary=np.flatnonzero(binary).tolist(),
        floating=np.flatnonzero(~(constant|binary)).tolist())


def action_layout(arrays):
    fields=[list(range(v['start'],v['stop'])) for v in entity_layout()['action_fields'].values()]
    selected=sum(fields,[]);signed=False
    for a in arrays:
        for part in _blocks(len(a)):
            x=a[part];check_action_selectors(x)
            signed |= bool(np.signbit(x[:,selected]).any())
    return dict(width=feature_schema()['action_dim'],selectors=fields,
        floating=[i for i in range(feature_schema()['action_dim']) if i not in selected],signed_zero=signed)


def action_dtype(layout):
    fields=[('ids','u1',(3,)),('values','<f4',(len(layout['floating']),))]
    if layout['signed_zero']:fields.append(('signs','u1',((sum(map(len,layout['selectors']))+7)//8,)))
    return np.dtype(fields)


def _pack_actions(target,source,layout):
    for part in _blocks(len(source)):
        x=source[part];target['values'][part]=x[:,layout['floating']]
        for role,columns in enumerate(layout['selectors']):
            values=x[:,columns]
            target['ids'][part,role]=np.where(values.sum(1)>0,values.argmax(1),len(columns))
        if layout['signed_zero']:
            target['signs'][part]=np.packbits(np.signbit(x[:,sum(layout['selectors'],[])]),axis=1,bitorder='little')


def decode_actions(array,indices,layout):
    scalar=np.isscalar(indices);a=array[indices]
    if scalar:a=a[None]
    out=np.zeros((len(a),layout['width']),dtype='<f4');out[:,layout['floating']]=a['values']
    for role,columns in enumerate(layout['selectors']):
        ids=a['ids'][:,role];require((ids<=len(columns)).all(),'Invalid packed action selector')
        rows=np.flatnonzero(ids<len(columns))
        out[rows,np.asarray(columns)[ids[rows]]]=1.
    if layout['signed_zero']:
        columns=sum(layout['selectors'],[])
        signs=np.unpackbits(a['signs'],axis=1,count=len(columns),bitorder='little').astype('<u4')<<31
        out[:,columns]=(out[:,columns].view('<u4')|signs).view('<f4')
    return out[0] if scalar else out


def pack(path,critic,actions,game_count):
    """Pack new, uncommitted staging arrays in bounded blocks; preserve originals
    until every packed array has been written. path(ext) is ownership-checked.
    """
    layout=state_layout(critic['state']);action=action_layout((critic['action'],actions))
    seen=np.zeros(game_count,dtype=bool);games=np.zeros(game_count,dtype=GAME)
    for row in critic:
        g=int(row['game']);require(0<=g<game_count,'Invalid packed game index')
        record=np.zeros((),dtype=GAME)
        # Stored row descriptors are relative; retain the original stable-seat
        # values once per game and recover the exact permutation at read time.
        record['descriptors']=row['descriptors'][row['mapping']]
        record['targets']=row['targets'];record['descriptor_generation']=row['descriptor_generation']
        if seen[g]:require(games[g].tobytes()==record.tobytes(),'Game metadata changed within a PPG cache')
        else:games[g]=record;seen[g]=True
    require(seen.all(),'Missing packed game metadata')
    _write(path('games.npy'),GAME,(game_count,),lambda out:out.__setitem__(slice(None),games))
    def meta(out):
        for part in _blocks(len(critic)):
            for name in META.names:
                x=critic[name][part]
                if name in ('mapping','seat'):require(((x>=0)&(x<4)).all(),'Invalid packed seat mapping')
                out[name][part]=x
    _write(path('meta.npy'),META,(len(critic),),meta)
    def floating(out):
        for part in _blocks(len(critic)):out[part]=critic['state'][part][:,layout['floating']]
    def binary(out):
        for part in _blocks(len(critic)):
            out[part]=np.packbits(critic['state'][part][:,layout['binary']].astype('u1'),axis=1,bitorder='little')
    _write(path('state-floats.npy'),'<f4',(len(critic),len(layout['floating'])),floating)
    _write(path('state-bits.npy'),'u1',(len(critic),(len(layout['binary'])+7)//8),binary)
    _write(path('critic-actions.npy'),action_dtype(action),(len(critic),),lambda out:_pack_actions(out,critic['action'],action))
    _write(path('policy-actions.npy'),action_dtype(action),(len(actions),),lambda out:_pack_actions(out,actions,action))
    return dict(contract=CONTRACT,state=layout,action=action,games=game_count,
        policy_actions=len(actions),dense_array_bytes=critic.nbytes+actions.nbytes,
        packed_array_bytes=sum(path(ext).stat().st_size for ext in EXTENSIONS),
        scope='Bit-exact FP32 reconstruction; constant/binary state columns, categorical action selectors, game-level targets/descriptors. No target quantization or trainable embeddings.')


def validate(layout):
    require(layout.get('contract')==CONTRACT,'Unknown PPG packing contract')
    s=layout['state'];width=feature_schema()['state_dim']
    require(s['width']==width and sorted(s['constant']+s['binary']+s['floating'])==list(range(width))
        and len(s['constant'])==len(s['constant_bits'])
        and all(type(v) is int and 0<=v<2**32 for v in s['constant_bits']),'Invalid packed state layout')
    a=layout['action'];expected=[list(range(v['start'],v['stop'])) for v in entity_layout()['action_fields'].values()]
    require(a['width']==feature_schema()['action_dim'] and a['selectors']==expected
        and a['floating']==[i for i in range(a['width']) if i not in sum(expected,[])]
        and type(a['signed_zero']) is bool,'Invalid packed action layout')


class _Field:
    def __init__(self,array,name):
        self.array,self.name=array,name
        d=row_dtype()[name];self.dtype=d.base;self.shape=(len(array),*d.shape)
    def __getitem__(self,indices):return self.array.column(self.name,indices)
    def __iter__(self):return iter(self.array.column(self.name,slice(None)))


class _Record:
    def __init__(self,array,index):self.array,self.index=array,index
    def __getitem__(self,name):return self.array.column(name,self.index)


class CriticArray:
    """Lazy row compatibility plus vectorized per-column mmap gathers."""
    def __init__(self,path,receipt):
        self.layout=receipt['packing'];validate(self.layout);self.arrays={};self._mmap=self
        self.dtype=row_dtype();self.shape=(receipt['rows'],)
        try:
            for ext in EXTENSIONS[:-1]:self.arrays[ext]=np.load(path(ext),mmap_mode='r',allow_pickle=False)
            s=self.layout['state'];a=self.layout['action']
            self.state_template=np.zeros(s['width'],dtype='<f4')
            self.state_template[s['constant']]=np.asarray(s['constant_bits'],dtype='<u4').view('<f4')
            expected={'meta.npy':(META,self.shape),'games.npy':(GAME,(self.layout['games'],)),
                'state-floats.npy':(np.dtype('<f4'),(len(self),len(s['floating']))),
                'state-bits.npy':(np.dtype('u1'),(len(self),(len(s['binary'])+7)//8)),
                'critic-actions.npy':(action_dtype(a),self.shape)}
            require(all(self.arrays[k].dtype==d and self.arrays[k].shape==shape for k,(d,shape) in expected.items()),
                'Invalid packed critic array shape/dtype')
        except BaseException:self.close();raise
    def __len__(self):return self.shape[0]
    def __getitem__(self,index):
        if isinstance(index,str):return _Field(self,index)
        return _Record(self,index)
    def column(self,name,indices):
        scalar=np.isscalar(indices)
        if name=='state':
            s=self.layout['state'];floats=self.arrays['state-floats.npy'][indices]
            # Dense floating fixtures/columns need no scatter or reconstruction.
            if not s['constant'] and not s['binary']:return floats
            bits=self.arrays['state-bits.npy'][indices]
            if scalar:floats=floats[None];bits=bits[None]
            result=np.broadcast_to(self.state_template,(len(floats),s['width'])).copy()
            result[:,s['floating']]=floats
            result[:,s['binary']]=np.unpackbits(bits,axis=1,count=len(s['binary']),bitorder='little')
            return result[0] if scalar else result
        if name=='action':return decode_actions(self.arrays['critic-actions.npy'],indices,self.layout['action'])
        m=self.arrays['meta.npy']
        if name in META.names:return np.asarray(m[name][indices],dtype=row_dtype()[name].base)
        g=self.arrays['games.npy'][m['game'][indices]]
        if name=='descriptors':
            values=g[name];mapping=m['mapping'][indices]
            if scalar:return values[np.argsort(mapping)]
            return np.take_along_axis(values,np.argsort(mapping,axis=1)[:,:,None],axis=1)
        return g[name]
    def close(self):
        for a in self.arrays.values():a._mmap.close()
        self.arrays={}


class ActionArray:
    def __init__(self,path,layout):
        self.layout=layout['action'];self._mmap=self
        self.array=np.load(path,mmap_mode='r',allow_pickle=False)
        self.shape=(layout['policy_actions'],self.layout['width']);self.dtype=np.dtype('<f4')
        if self.array.dtype!=action_dtype(self.layout) or self.array.shape!=(self.shape[0],):
            self.close();raise ValueError('Invalid packed policy action array')
    def __getitem__(self,indices):return decode_actions(self.array,indices,self.layout)
    def close(self):self.array._mmap.close()
