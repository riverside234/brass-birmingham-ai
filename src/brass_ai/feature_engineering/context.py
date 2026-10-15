"""Complete FE0–FE3 context assembly with separate bounded static/dynamic caches."""
from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType

from brass_ai.learning.contracts import digest,finite_vector,require,read_json,CONFIG_DIR,FEATURE_SCHEMA
from brass_ai.learning.encoding_budget import check_budget
from .contracts import VERSION,RULES,STATE_FIELDS,ACTION_FIELDS,layout_identity,tensorize
from .finance import financial,provisional_order,opportunities
from .industry import realization,development
from .effects import action_effects
from .rules import BoardFacts,income_level
from .static_profile import admit_static


@dataclass(frozen=True)
class DerivedContext:
    version: str
    identity: str
    seat: int
    state: object
    actions: tuple
    choice_ids: tuple
    static_choices: tuple

    def state_tensor(self):return tensorize(self.state,STATE_FIELDS)
    def action_tensors(self):
        # Equivalent card alternatives retain separate policy rows, while their
        # immutable numeric consequence block needs conversion only once.
        converted={}
        for action in self.actions:
            token=id(action)
            if token not in converted:converted[token]=tensorize(action,ACTION_FIELDS)
        return tuple(converted[id(action)] for action in self.actions)


class DerivedContextBuilder:
    def __init__(self,cache_size=64,static_cache_size=8192):
        require(type(cache_size) is int and cache_size>=0 and type(static_cache_size) is int and static_cache_size>=0,'Invalid feature cache bounds')
        self.cache_size,self.static_cache_size=cache_size,static_cache_size
        self.cache=OrderedDict();self.static=OrderedDict();self.schema=layout_identity()
        self.board_schema=read_json(CONFIG_DIR/FEATURE_SCHEMA)
        self.hits=self.misses=self.dynamic_calculations=0

    def build(self,view,choices,style=None,intent=None,*,deadline=None,cancelled=None,validated=False,input_identity=None):
        def admitted():check_budget(deadline,cancelled)
        admitted()
        # Internal encoders reuse ingress admission. Public calls cross the same
        # exact, bounded split-schema validator as replay and inference.
        if not validated:
            from brass_ai.recovery.inputs import BOUNDARY
            input_identity=BOUNDARY.check(view,choices)
        require(view['rules']==RULES and view['context']['actor']==view['seat'] and not view['context']['terminal'],'Derived features require the acting permitted observer')
        admit_static(view,self.board_schema)
        for group in ('slots','links','merchants'):
            actual=[x['id'] for x in view[group]]
            require(len(actual)==len(set(actual)) and set(actual)==set(self.board_schema[group]),'Missing/duplicate required public board facts')
        for player in view['players']:
            require(player['income']==income_level(player['income_marker']),'Inconsistent public income track')
            mats=player['mat']
            require(len(mats)==6 and {m['type'] for m in mats}==set(self.board_schema['industries']),'Missing required full industry stacks')
            require(all(t['seat']==player['seat'] and t['type']==m['type'] for m in mats for t in m['tiles']),'Industry stack owner/type mismatch')
        ids=tuple(c['id'] for c in choices)
        require(ids and len(ids)==len(set(ids)),'Missing/duplicate feature candidates')
        style=[0.]*4 if style is None else style;intent=[0.]*8 if intent is None else intent
        finite_vector(style,4,-1,1);finite_vector(intent,8,-1,1)
        # The facts use no history. Validate it, but do not reserialize its growing
        # contents for this additional cache; active replay/encoder pins are separate.
        inputs={k:v for k,v in view.items() if k!='history'}
        identity=digest(dict(version=VERSION,schema=self.schema,input=input_identity,style=style,intent=intent)) if input_identity else digest(dict(version=VERSION,schema=self.schema,view=inputs,choices=choices,style=style,intent=intent))
        admitted()
        if identity in self.cache:
            self.hits+=1;self.cache.move_to_end(identity);return self.cache[identity]
        from brass_ai.learning.batching import _snapshot_input
        # Reuse the existing exact-type wire snapshotter: immutable scalar facts
        # need no deepcopy dispatch. Containers/aliases are still detached.
        self.misses+=1;detached=_snapshot_input(inputs);detached['history']=[];board=BoardFacts(detached)
        facts=financial(detached)|provisional_order(detached)|opportunities(detached,choices)|realization(board)|development(board)
        require(set(facts)=={f.name for f in STATE_FIELDS},'Incomplete required state feature group')
        actions=[];static_choices=[];dynamic={}
        for choice in choices:
            admitted()
            semantics=(choice['kind'],choice.get('target'),choice.get('industry'),choice.get('merchant'),choice.get('card'),tuple(choice.get('cards',())))
            if semantics not in self.static:self.static[semantics]=semantics
            self.static.move_to_end(semantics);static_choices.append(self.static[semantics])
            while len(self.static)>self.static_cache_size:self.static.popitem(last=False)
            # Money/resource effects do not depend on which equivalent card was
            # discarded. This cache is local to one complete permitted context.
            effect_key=semantics[:4]
            if effect_key not in dynamic:
                effect=action_effects(board,choice)
                require(set(effect)=={f.name for f in ACTION_FIELDS},'Incomplete required action feature group')
                dynamic[effect_key]=MappingProxyType(effect);self.dynamic_calculations+=1
            actions.append(dynamic[effect_key])
        result=DerivedContext(VERSION,identity,view['seat'],MappingProxyType(facts),tuple(actions),ids,tuple(static_choices))
        admitted()
        if self.cache_size:
            self.cache[identity]=result
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return result

    def report(self):
        return dict(hits=self.hits,misses=self.misses,contexts=len(self.cache),static_choices=len(self.static),dynamic_calculations=self.dynamic_calculations)
