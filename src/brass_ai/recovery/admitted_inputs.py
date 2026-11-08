"""Process-local immutable actor snapshots; never trust a serialized receipt."""
from dataclasses import dataclass
from collections.abc import ItemsView,ValuesView
import hashlib
import json
from brass_ai.learning.batching import Request


def _immutable(*args,**kwargs):
    raise TypeError('Admitted actor inputs are immutable')


class FrozenDict(dict):
    __slots__=('_admission',)
    def __new__(cls,items=()):
        result=dict.__new__(cls);dict.update(result,items);return result
    def __init__(self,items=()):pass
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_immutable
    __setattr__=__delattr__=_immutable
    def __deepcopy__(self,memo):return self
    # IPC/replay readers receive plain containers and must perform their own
    # admission. A receipt is meaningful only for these exact local objects.
    def __reduce_ex__(self,protocol):return dict,(dict(self),)


class FrozenList(list):
    __slots__=('_admission',)
    def __new__(cls,items=()):
        result=list.__new__(cls);list.extend(result,items);return result
    def __init__(self,items=()):pass
    __setitem__=__delitem__=append=clear=extend=insert=pop=remove=reverse=sort=__iadd__=__imul__=_immutable
    __setattr__=__delattr__=_immutable
    def __deepcopy__(self,memo):return self
    def __reduce_ex__(self,protocol):return list,(list(self),)


class OwnedTree:
    """Private JSON reader ownership; expose immutable children on demand.

    Source dictionaries/lists must have no external writable aliases. General
    callers continue to use the eager detached freeze() boundary.
    """
    def __init__(self):self.values={}
    def freeze(self,value):
        if type(value) not in (dict,list):return _freeze(value)
        saved=self.values.get(id(value))
        if saved is not None:
            if saved[0] is not value:raise TypeError('Owned JSON identity changed')
            return saved[1]
        frozen=(OwnedDict if type(value) is dict else OwnedList)(value,self)
        self.values[id(value)]=(value,frozen)
        return frozen


class OwnedDict(FrozenDict):
    __slots__=('_owner',)
    def __new__(cls,value,owner):
        result=dict.__new__(cls);dict.update(result,value);object.__setattr__(result,'_owner',owner);return result
    def __init__(self,value,owner):pass
    def __iter__(self):return dict.__iter__(self)
    def __getitem__(self,key):return self._owner.freeze(dict.__getitem__(self,key))
    def get(self,key,default=None):return self[key] if key in self else default
    def items(self):return ItemsView(self)
    def values(self):return ValuesView(self)
    def copy(self):return dict(self)
    def __or__(self,other):return dict(self)|dict(other) if isinstance(other,dict) else NotImplemented
    def __ror__(self,other):return dict(other)|dict(self) if isinstance(other,dict) else NotImplemented


class OwnedList(FrozenList):
    __slots__=('_owner',)
    def __new__(cls,value,owner):
        result=list.__new__(cls);list.extend(result,value);object.__setattr__(result,'_owner',owner);return result
    def __init__(self,value,owner):pass
    def __iter__(self):return (self._owner.freeze(x) for x in list.__iter__(self))
    def __reversed__(self):return (self[i] for i in range(len(self)-1,-1,-1))
    def __getitem__(self,key):
        if isinstance(key,slice):return FrozenList(self[i] for i in range(*key.indices(len(self))))
        return self._owner.freeze(list.__getitem__(self,key))
    def copy(self):return list(self)
    def __add__(self,other):return list(self)+(list(other) if type(other) is OwnedList else other)
    def __radd__(self,other):return other+list(self)
    def __mul__(self,n):return list(self)*n
    __rmul__=__mul__


@dataclass(frozen=True)
class Admission:
    source: str
    view_id: int
    choices_id: int
    canonical: tuple | None = None


def _freeze(value):
    if type(value) in (FrozenDict,FrozenList,OwnedDict,OwnedList):return value
    if type(value) is dict:return FrozenDict((k,_freeze(v)) for k,v in value.items())
    if type(value) is list:return FrozenList(_freeze(v) for v in value)
    if type(value) in (str,int,float,bool,type(None)):return value
    # Called only after wire admission; never normalize custom wire types.
    raise TypeError('Only admitted JSON containers can be frozen')


def freeze(view,choices,source,*,canonical=None):
    if type(view) not in (dict,FrozenDict,OwnedDict) or type(choices) not in (list,FrozenList,OwnedList):
        raise TypeError('Only admitted JSON roots can be frozen')
    # New root capabilities for each pair; immutable child subtrees may be
    # shared without replacing another in-flight request's admission receipt.
    view=FrozenDict((k,_freeze(v)) for k,v in view.items())
    choices=FrozenList(_freeze(v) for v in choices)
    receipt=Admission(source,id(view),id(choices),canonical)
    object.__setattr__(view,'_admission',receipt)
    object.__setattr__(choices,'_admission',receipt)
    return view,choices


def admitted_digest(view,choices):
    if type(view) is not FrozenDict or type(choices) is not FrozenList:return None
    receipt=getattr(view,'_admission',None)
    if (type(receipt) is Admission and getattr(choices,'_admission',None) is receipt
            and (receipt.view_id,receipt.choices_id)==(id(view),id(choices))):return receipt.source
    return None


@dataclass(frozen=True)
class AdmittedRequest(Request):
    def digest(self):
        admitted=admitted_digest(self.view,self.choices)
        contexts=(self.style,self.intent,self.remaining or ())
        if (admitted is None or type(self.expected_model) is not str or
                not all(type(v) is tuple and all(type(x) in (int,float) for x in v) for v in contexts)):
            return super().digest()
        key=(admitted,id(self.view),id(self.choices),self.expected_model,self.style,self.intent,self.remaining)
        previous=getattr(self,'_digest_memo',None)
        if previous is not None and previous[0]==key:
            from .replay_binding import count
            count('request_hash_reused');return previous[1]
        # Hash the original request representation once, not a new identity
        # formula. Only immutable inputs can reuse it on submit/response checks.
        receipt=self.view._admission
        if receipt.canonical is None:value=super().digest()
        else:
            # The exact original Request JSON/digest, streamed from already
            # admitted immutable canonical subtrees. No alternate hash formula.
            view,choices=receipt.canonical;hashed=hashlib.sha256()
            def small(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            for part in (b'{"choices":',choices,b',"intent":',small(self.intent),b',"model":',small(self.expected_model)):
                hashed.update(part)
            if self.remaining is not None:hashed.update(b',"remaining":');hashed.update(small(self.remaining))
            for part in (b',"style":',small(self.style),b',"view":',view,b'}'):hashed.update(part)
            value=hashed.hexdigest()
            from .replay_binding import count
            count('request_json_reused')
        from .replay_binding import count
        count('request_hash_executed')
        object.__setattr__(self,'_digest_memo',(key,value));return value


def request(*args,**kwargs):
    # Positional view/choices use the same interface as the original Request.
    cls=AdmittedRequest if admitted_digest(args[2],args[3]) is not None else Request
    return cls(*args,**kwargs)
