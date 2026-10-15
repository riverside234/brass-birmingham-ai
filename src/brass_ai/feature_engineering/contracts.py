"""Frozen A–F field ordering/masks/scales for permitted-features/2.0.0."""
from dataclasses import dataclass
from functools import lru_cache
import math

from brass_ai.learning.contracts import require

VERSION='permitted-derived-context/1.0.0'
RULES='steam-brass-patch-1.1.0'
INDUSTRIES=('Brewery','CoalMine','CottonMill','IronWorks','Manufacturer','Pottery')
SALEABLE=('CottonMill','Manufacturer','Pottery')
KINDS=('beer','bonus_develop','bonus_skip','build','coal','develop','finish','iron','liquidate','loan','network','pass','scout','sell')
COMPONENTS=('value','known','applicable','lower','lower_known','upper','upper_known')


@dataclass(frozen=True)
class Measure:
    value: float=0.
    known: bool=False
    applicable: bool=True
    lower: float=0.
    lower_known: bool=False
    upper: float=0.
    upper_known: bool=False

    def __post_init__(self):
        require(all(math.isfinite(v) for v in (self.value,self.lower,self.upper)),'Nonfinite derived fact')
        require(not self.known or self.applicable and self.lower_known and self.upper_known and self.value==self.lower==self.upper,'Malformed exact fact')
        require(not(self.lower_known and self.upper_known) or self.lower<=self.upper,'Inverted derived bounds')
        require(self.applicable or not(self.known or self.lower_known or self.upper_known),'Inapplicable fact has values')

    @classmethod
    @lru_cache(maxsize=512)
    def exact(cls,value):return cls(float(value),True,True,float(value),True,float(value),True)
    @classmethod
    def interval(cls,lower=None,upper=None):
        if lower is not None and lower==upper:return cls.exact(lower)
        return cls(lower=0. if lower is None else float(lower),lower_known=lower is not None,
                   upper=0. if upper is None else float(upper),upper_known=upper is not None)
    @classmethod
    @lru_cache(maxsize=1)
    def absent(cls):return cls(applicable=False)
    def tensor(self,scale):
        return (self.value/scale,float(self.known),float(self.applicable),self.lower/scale,
                float(self.lower_known),self.upper/scale,float(self.upper_known))


@dataclass(frozen=True)
class Field:
    name: str
    scale: float
    owner: str
    source: str
    units: str
    basis: str


TILE_FIELDS=dict(level=8,cash_cost=100,vp=100,beer=8,canal_eligible=1,rail_eligible=1,survives_canal=1)
TARGET_FIELDS=dict(present=1,reachable=1,blocked=1,already_unlocked=1,removals=16,**TILE_FIELDS,
                   delta_cost=100,delta_vp=100,delta_beer=8,delta_canal_eligible=1,delta_rail_eligible=1)


def state_fields():
    fields=[]
    def add(name,scale,owner,source,units='count',basis='current permitted position'):
        fields.append(Field(name,scale,owner,source,units,basis))
    for seat in range(4):
        for name,scale in dict(settlement=1,negative_payment=100,projected_cash=100,shortfall=100,public_debt=100).items():
            add(f'finance.{seat}.{name}',scale,'observer-relative seat '+str(seat),'players cash/income/debt; context phase/era/round','bool' if name=='settlement' else 'GBP','next income settlement, fixed current position; debt projection unknown')
    for seat in range(4):
        add(f'order.{seat}.position',3,'observer-relative seat '+str(seat),'public order; spending','zero-based rank','provisional next round, stable current-order ties; final Rail inapplicable')
        add(f'order.{seat}.spending_difference',100,'observer-relative seat '+str(seat),'public spending','GBP','observer spending minus this player spending')
    add('opportunities.main_action_available',1,'observer','context phase','bool')
    for kind in KINDS:
        for suffix in ('available','targets'):
            add(f'opportunities.{kind}.{suffix}',1 if suffix=='available' else 512,'observer','admitted legal list','bool' if suffix=='available' else 'deduplicated opportunities','current UI phase only; main-action absence during subchoices is inapplicable')
    realization=dict(saleable_count=32,potential_sale_vp=100,beer_needed=32,individually_saleable_count=32,
        individually_saleable_vp=100,unflipped_level_one_count=32,unflipped_level_one_vp=100,
        secured_industry_vp=100,own_action_slots=16,card_supported_actions=16,current_action_in_progress=1)
    for name,scale in realization.items():
        add('realization.'+name,scale,'observer','public tiles/graph/merchants/beer; own hand; context',
            'VP' if 'vp' in name else 'count','individual feasibility, not joint sales; slots include in-progress main action; card support bounds exclude future guarantees')
    for industry in INDUSTRIES:
        for name,scale in dict(present=1,already_unlocked=1,**TILE_FIELDS).items():
            add(f'development.{industry}.current.{name}',scale,'observer','full ordered mat','tile property','head of stack, never skip era-ineligible/undevelopable tiles')
        for era in ('current','next_era'):
            for name,scale in TARGET_FIELDS.items():
                add(f'development.{industry}.{era}_target.{name}',scale,'observer','full ordered mat; era rules','tile property or tile removals','first strictly higher eligible level in stack order; development-only reachability; next_era only in Canal')
    return tuple(fields)


def action_fields():
    fields=[]
    def add(name,scale,owner,source,units,basis):fields.append(Field(name,scale,owner,source,units,basis))
    for name in ('cash_spent','cash_proceeds','cash_after','projected_cash','shortfall','debt_after'):
        add(name,100,'observer','choice, public money/income/market/debt','GBP','incremental chosen action portion, before automatic turn/round advancement')
    for name in ('coal','iron','beer'):
        add(name+'_used',8,'observer','selected source choice','cubes','this one choice only; root resource commitments are separate')
        add('portion_'+name,8,'observer','selected tile/link/part or remaining pending counters','cubes','chosen determined part through its mandatory payments; no unselected second part')
    for name in ('portion_cash_spent','portion_cash_proceeds','portion_cash_after','portion_projected_cash','portion_shortfall'):
        add(name,100,'observer','base charge plus mandatory remaining market/resource facts','GBP','selected determined part from now; excludes previous charges and optional later parts')
    add('merchant_free_develop',1,'observer','selected merchant beer bonus','bool','this choice creates a free development opportunity, not an assumed removal')
    for seat in range(4):
        for name,scale in dict(flips=8,income_marker_delta=100,income_delta=30,immediate_vp=100,secured_era_vp_delta=100).items():
            add(f'effects.{seat}.{name}',scale,'observer-relative seat '+str(seat),'chosen resource/industry plus public owner and income track','VP' if 'vp' in name else 'count or income','chosen action portion only; secured era VP is not immediately awarded VP')
    for name in ('removals_before','removals_after','next_era_removals_before','next_era_removals_after','removed_tiles','immediately_unlocks','free'):
        add('develop.'+name,16 if 'removal' in name else 1,'observer','selected first tile, full mat and era','tile count or bool','one guaranteed removal; do not assume second development or merchant bonus selection')
    for name,scale in TILE_FIELDS.items():
        add('develop.result.'+name,scale,'observer','stack after one removal','tile property','actual resulting stack head; era eligibility remains explicit')
    return tuple(fields)


STATE_FIELDS=state_fields()
ACTION_FIELDS=action_fields()


def tensorize(facts,fields):
    require(set(facts)=={f.name for f in fields},'Incomplete or mixed-version derived context')
    return tuple(x for f in fields for x in facts[f.name].tensor(f.scale))


def specification():
    from dataclasses import asdict
    return dict(version=VERSION,feature_version='permitted-features/2.0.0',production_enabled=True,
        rules=RULES,components=COMPONENTS,seat_order='observer-clockwise-relative',
        status='frozen FE4 layout; state block precedes final style[4]/intent[8], action block follows categorical fields',
        state=[asdict(f) for f in STATE_FIELDS],actions=[asdict(f) for f in ACTION_FIELDS])


@lru_cache(maxsize=1)
def layout_identity():
    from brass_ai.learning.contracts import digest
    return digest(specification())
