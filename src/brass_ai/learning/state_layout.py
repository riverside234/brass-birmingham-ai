"""Immutable index metadata for compact Features-4 and widened grouped v9.

This describes the original emission order; it neither computes nor reorders
encoded features. Both encoders use its dimensions, and models gather its groups.
"""
from dataclasses import dataclass,replace
from functools import lru_cache

from .contracts import CONFIG_DIR,FEATURE_SCHEMA,FEATURE_VERSION,digest,read_json,require

LAYOUT_VERSION='grouped-state-layout/5.0.0'
LAYOUT_FILE='state-groups-5.0.0.json'
GROUP_NAMES=('turn_global','cards_history','player_economy','industry_development',
             'board_resources','current_action','policy_context')
PROJECTION_WIDTHS=(32,64,48,192,224,48)
MODEL_LAYOUT_VERSION='grouped-state-layout/6.0.0'
MODEL_LAYOUT_FILE='state-groups-6.0.0.json'
MODEL_PROJECTION_WIDTHS=(32,64,48,320,352,48)


@dataclass(frozen=True)
class Segment:
    name: str
    group: str
    start: int
    stop: int
    owner: str
    source: str
    components: tuple


@dataclass(frozen=True)
class StateGroup:
    name: str
    indices: tuple
    width: int


@dataclass(frozen=True)
class StateLayout:
    dimension: int
    raw_dimension: int
    feature_schema_sha256: str
    segments: tuple
    groups: tuple
    identity: str=''
    version: str=LAYOUT_VERSION

    @property
    def permutation(self):return tuple(i for g in self.groups for i in g.indices)

    def specification(self,*,with_identity=True,version=None):
        result=dict(format=self.version if version is None else version,feature_version=FEATURE_VERSION,
            feature_schema_sha256=self.feature_schema_sha256,state_dim=self.dimension,
            raw_state_dim=self.raw_dimension,fusion_inputs=sum(g.width for g in self.groups),
            groups=[dict(name=g.name,input_dim=len(g.indices),indices=list(g.indices),projection_width=g.width,
                         learned=g.name!='policy_context') for g in self.groups],
            segments=[dict(name=s.name,group=s.group,start=s.start,stop=s.stop,
                owner=s.owner,source=s.source,components=list(s.components)) for s in self.segments])
        if with_identity:result['layout_sha256']=self.identity
        return result


def build_state_layout(schema):
    """Declarative segments in reference/PreparedEncoder order, with no offsets."""
    segments=[];offset=0
    require(schema['version']==FEATURE_VERSION,'Incompatible feature manifest; explicitly re-encode raw data')
    groups=schema['raw_groups']
    # Features-4 remains byte-identical for existing audited IL datasets. Its
    # projection_widths records the original v8 proposal, not tensor semantics.
    # Active widths belong to the separately versioned model/layout contract;
    # the frozen layout still binds the entire unchanged feature manifest.
    def add(name,group,size,owner='global',source='raw',components=()):
        nonlocal offset
        require(type(size) is int and size>0 and group in GROUP_NAMES,'Invalid state-layout segment')
        segments.append(Segment(name,group,offset,offset+size,owner,source,tuple(components)))
        offset+=size
    cards=schema['card_types'];industries=schema['industries'];kinds=schema['kinds']
    targets=schema['slots']+schema['links']+['merchant:'+m for m in schema['merchants']]+['market']
    tile_fields=tuple(sorted(schema['tile_fields']));tile_size=1+len(tile_fields)
    board_fields=tuple(sorted(schema['board_tile_fields']))
    add('turn.era','turn_global',2,components=('CanalEra','RailEra'))
    add('turn.phase','turn_global',len(schema['phases']),components=schema['phases'])
    for name in ('round','turn','actions_remaining'):add('turn.'+name,'turn_global',1)
    add('turn.owner','turn_global',4,'observer-relative seat')
    for n in range(4):add(f'order.current.{n}','player_economy',4,'observer-relative seat')
    add('hand.counts','cards_history',len(cards),'observer',components=cards)
    for seat in range(4):
        owner=f'observer-relative seat {seat}'
        for field in sorted(schema['player_fields']):add(f'player.{seat}.{field}','player_economy',1,owner)
        add(f'player.{seat}.discards','cards_history',len(cards),owner,components=cards)
        add(f'player.{seat}.known_wild','cards_history',2,owner,components=('wild:industry','wild:location'))
        for kind in industries:
            prefix=f'mat.{seat}.{kind}'
            add(prefix+'.levels',groups['mat'],8,owner,components=tuple(range(1,9)))
            for head in ('head','next'):
                add(prefix+'.'+head,groups['mat'],tile_size,owner,components=('present',)+tile_fields)
    for slot in schema['slots']:
        prefix='slot.'+slot
        add(prefix+'.owner','board_resources',4,'observer-relative seat')
        add(prefix+'.industry','board_resources',len(industries),components=industries)
        add(prefix+'.tile','board_resources',len(board_fields),components=board_fields)
    for link in schema['links']:add('link.'+link,'board_resources',4,'observer-relative seat')
    for merchant in schema['merchants']:
        prefix='merchant.'+merchant
        add(prefix+'.types','board_resources',len(industries),components=industries)
        add(prefix+'.beer','board_resources',1)
    for name in ('coal_market','iron_market','deck_count','wild_industry_count','wild_location_count'):
        add('global.'+name,groups[name],1)
    for name,vocabulary in (('kind',kinds),('target',targets),('merchant',schema['merchants'])):
        add('pending.'+name,'current_action',len(vocabulary),'observer',components=vocabulary)
    for name in ('parts','coal','iron','beer'):add('pending.'+name,'current_action',1,'observer')
    for seat in range(4):
        for era in ('CanalEra','RailEra'):
            add(f'history.{seat}.{era}','cards_history',len(schema['history_kinds']),f'observer-relative seat {seat}',components=schema['history_kinds'])
    # Existing derived Measures appear here, before the unchanged context tail.
    facts={f['name']:f for f in schema['derived']['semantic_state']}
    for field in schema['derived']['state']:
        add(field['name'],field['group'],len(field['components']),facts[field['fact']]['owner'],
            'derived',field['components'])
    add('policy.style','policy_context',4,'observer',components=tuple(range(4)))
    add('policy.intent','policy_context',schema['intent_dim'],'observer',components=tuple(range(schema['intent_dim'])))
    require('state_dim' not in schema or offset==schema['state_dim'],'State grouping does not match the frozen feature manifest')
    grouped=tuple(StateGroup(name,tuple(i for s in segments if s.group==name for i in range(s.start,s.stop)),
        PROJECTION_WIDTHS[n] if n<6 else 12) for n,name in enumerate(GROUP_NAMES))
    permutation=tuple(i for g in grouped for i in g.indices)
    require(len(permutation)==offset and sorted(permutation)==list(range(offset)),
        'State grouping must be complete and disjoint')
    require(grouped[-1].indices==tuple(range(offset-12,offset)),'Policy context must remain the final 12 inputs')
    layout=StateLayout(offset,sum(s.stop-s.start for s in segments if s.source=='raw'),
        digest(schema),tuple(segments),grouped)
    return replace(layout,identity=digest(layout.specification(with_identity=False)))


@lru_cache(maxsize=1)
def state_layout():
    layout=build_state_layout(read_json(CONFIG_DIR/FEATURE_SCHEMA))
    require(layout.specification()==read_json(CONFIG_DIR/LAYOUT_FILE),
        'Frozen grouped state layout differs; do not relabel a layout identity')
    return layout


@lru_cache(maxsize=1)
def model_layout():
    """V11 learned widths; keep the original input/token layout immutable."""
    original=state_layout()
    groups=tuple(replace(g,width=MODEL_PROJECTION_WIDTHS[i] if i<6 else 12)
                 for i,g in enumerate(original.groups))
    layout=replace(original,groups=groups,identity='',version=MODEL_LAYOUT_VERSION)
    layout=replace(layout,identity=digest(layout.specification(with_identity=False,version=MODEL_LAYOUT_VERSION)))
    require(layout.specification(version=MODEL_LAYOUT_VERSION)==read_json(CONFIG_DIR/MODEL_LAYOUT_FILE),
            'Frozen v11 grouped model layout differs')
    return layout
