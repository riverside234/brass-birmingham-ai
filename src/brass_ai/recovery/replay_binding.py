"""Process-local, detached replay capabilities; no serialized trust flags.

Only exact frozen containers and bytes-backed numeric arrays can reuse these
proofs. Mutable/foreign inputs keep the original content checks. Pickling strips
the capability: a receiving process establishes its own boundary once.
"""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib

import numpy as np

from brass_ai.learning.contracts import digest, require
from brass_ai.learning.features import Encoded, check_encoded
from .admitted_inputs import FrozenDict, FrozenList, OwnedTree, _freeze

COUNTS=Counter()
LOCAL=ContextVar('replay-binding-counts',default=None)


def enabled():
    from .collector_runtime import option
    return option('replay_bindings')


def count(name,n=1):
    COUNTS[name]+=n
    local=LOCAL.get()
    if local is not None:local[name]+=n
def report():return dict(COUNTS)
def since(before):return {k:v-before.get(k,0) for k,v in COUNTS.items() if v!=before.get(k,0)}


def compact_metadata(value,memo):
    """Detach only compact outputs, sharing labels but never a game's owner.

    In-process readers do not cross pickle, which normally strips OwnedTree.
    Choice/label/header references must not retain all private board histories.
    Numeric feature arrays and established dtype/layout are left unchanged.
    """
    if not isinstance(value,(dict,list,tuple)):return value
    key=id(value)
    if key in memo:
        require(memo[key][0] is value,'Compact metadata identity changed')
        return memo[key][1]
    if isinstance(value,dict):
        result={};memo[key]=(value,result)
        result.update((k,compact_metadata(v,memo)) for k,v in value.items())
    elif isinstance(value,list):
        result=[];memo[key]=(value,result);result.extend(compact_metadata(v,memo) for v in value)
    else:
        result=tuple(compact_metadata(v,memo) for v in value);memo[key]=(value,result)
    return result


@contextmanager
def scope():
    counts=Counter();token=LOCAL.set(counts)
    try:yield counts
    finally:LOCAL.reset(token)


def _signature(a):return (id(a),a.shape,a.strides,a.dtype.str,a.flags.writeable)


def _bytes_array(a):
    require(isinstance(a,np.ndarray) and a.dtype.kind in 'fiu' and a.dtype.hasobject is False,
            'Unsupported immutable feature array')
    base=a
    while isinstance(base,np.ndarray) and base.base is not None:base=base.base
    if isinstance(base,memoryview):base=base.obj
    if type(base) is bytes and not a.flags.writeable and a.flags.c_contiguous:return a
    raw=a.tobytes(order='C');count('array_detached_bytes',len(raw))
    return np.frombuffer(raw,dtype=a.dtype).reshape(a.shape)


class SealedEncoded(Encoded):
    def __setattr__(self,name,value):
        if getattr(self,'_locked',False):raise TypeError('Sealed training features are immutable')
        object.__setattr__(self,name,value)

    def __reduce_ex__(self,protocol):
        # NumPy IPC can introduce writable aliases. Never serialize the proof.
        return Encoded,(self.state,self.actions,self.stable_to_relative,self.choice_ids,self.feature_version)


def checked_encoded(e):
    if type(e) is not SealedEncoded:return False
    require(e._signature==tuple(_signature(a) for a in (e.state,e.actions,e.stable_to_relative)),
            'Sealed feature layout changed')
    count('array_validation_reused');return True


def seal_encoded(e,*,_checked=False):
    if checked_encoded(e):return e
    check_encoded(e)
    if not _checked:
        from brass_ai.learning.local_entities import check_action_selectors
        require(all(np.isfinite(a).all() for a in (e.state,e.actions,e.stable_to_relative)),
                'Non-finite feature arrays')
        check_action_selectors(e.actions);count('array_validation_executed')
    result=SealedEncoded(*(_bytes_array(a) for a in (e.state,e.actions,e.stable_to_relative)),
                        tuple(e.choice_ids),e.feature_version)
    result._signature=tuple(_signature(a) for a in (result.state,result.actions,result.stable_to_relative))
    result._digests={};result._locked=True
    return result


def encoded_digest(e,field):
    if not checked_encoded(e):return None
    if field not in e._digests:
        a=getattr(e,field);hashed=hashlib.sha256();hashed.update(str((a.shape,a.dtype.str)).encode())
        hashed.update(memoryview(a));e._digests[field]=hashed.digest();count('array_hashes_executed')
    else:count('array_hashes_reused')
    return e._digests[field]


LABELS=('step','seat','chosen','behavior','teacher','seconds')


@dataclass(frozen=True)
class RowBinding:
    row: object
    key: str
    source: str
    label: str


class GameBinding:
    def __init__(self,game,rows):
        self.roots=tuple(game.items());self.rows=rows;self.header=game.header_seal
        self.features=dict(game.features)
        self.memo={}

    def metadata(self,game):
        require(tuple(game)==tuple(k for k,_ in self.roots) and game.header_seal==self.header and
                all(game.get(k) is v for k,v in self.roots),'Validated game metadata changed')
        for child in game.get('branches',()):child.assert_metadata()
        count('metadata_hash_reused')

    def row(self,game,row,*,allow_context=False):
        proof=self.rows.get((row['step'],row['seat']))
        if proof is None:return None
        original=proof.row
        if row['view'] is not original['view'] or row['choices'] is not original['choices']:return None
        def same(a,b):
            return a is b if isinstance(b,(FrozenDict,FrozenList)) else type(a) is type(b) and a==b
        require(all(same(row[k],original[k]) for k in LABELS),'Validated replay record changed')
        require(allow_context or (row['style']==original['style'] and row['intent']==original['intent']),
                'Validated policy context changed')
        require(game.features.get(proof.key) is self.features[proof.key],'Validated features changed')
        checked_encoded(self.features[proof.key]);count('immutable_row_reused')
        return proof


def seal_game(game,*,_owned=False):
    """Detach once after independent admission; callers cannot attest on disk."""
    if not enabled():return game
    if type(getattr(game,'_immutable_binding',None)) is GameBinding:
        game.assert_metadata();return game
    if not _owned:
        game.assert_metadata()
        for row in game['records']:game.encoded(row,None)
    children=[seal_game(child,_owned=_owned) for child in game.get('branches',())]
    rows=[];proofs={};features={};owner=OwnedTree() if _owned else None
    freeze=owner.freeze if owner is not None else _freeze
    for row in game['records']:
        saved=game.seals[(row['step'],row['seat'])]
        key=digest(dict(input=saved['input'],style=row['style'],intent=row['intent']))
        frozen=freeze(row if _owned else dict(row));rows.append(frozen)
        proofs[(row['step'],row['seat'])]=RowBinding(frozen,key,saved['input'],saved['label'])
        features[key]=seal_encoded(game.features[key])
    for k,value in list(game.items()):
        if k=='records':game[k]=FrozenList(rows)
        elif k=='branches':game[k]=FrozenList(children)
        else:game[k]=freeze(value)
    game.features=features;game.seals=_freeze(dict(game.seals))
    game._immutable_binding=GameBinding(game,proofs);count('immutable_games_sealed')
    return game


def memo(game,name,context,compute):
    binding=getattr(game,'_immutable_binding',None)
    if type(binding) is not GameBinding:return compute()
    binding.metadata(game);key=(name,digest(context))
    if key not in binding.memo:
        value=compute()
        binding.memo[key]=tuple(_freeze(v) for v in value) if type(value) is tuple else _freeze(value)
        count(name+'_executed')
    else:count(name+'_reused')
    return binding.memo[key]
