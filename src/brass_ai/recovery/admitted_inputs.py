"""Process-local immutable actor snapshots; never trust a serialized receipt."""
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Admission:
    source: str
    view_id: int
    choices_id: int


def _freeze(value):
    if type(value) is dict:return FrozenDict((k,_freeze(v)) for k,v in value.items())
    if type(value) is list:return FrozenList(_freeze(v) for v in value)
    if type(value) in (str,int,float,bool,type(None)):return value
    # Called only after wire admission; never normalize custom wire types.
    raise TypeError('Only admitted JSON containers can be frozen')


def freeze(view,choices,source):
    view,choices=_freeze(view),_freeze(choices)
    receipt=Admission(source,id(view),id(choices))
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
        if previous is not None and previous[0]==key:return previous[1]
        # Hash the original request representation once, not a new identity
        # formula. Only immutable inputs can reuse it on submit/response checks.
        value=super().digest();object.__setattr__(self,'_digest_memo',(key,value));return value


def request(*args,**kwargs):
    # Positional view/choices use the same interface as the original Request.
    cls=AdmittedRequest if admitted_digest(args[2],args[3]) is not None else Request
    return cls(*args,**kwargs)
