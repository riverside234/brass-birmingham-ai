"""Independent personality contracts; never modify the pinned Step 5 schemas."""
from pathlib import Path
import numpy as np

from brass_ai.learning.contracts import (CONFIG_DIR, compatibility, digest, finite_vector,
    keys, read_json, require)

HANDOFF = 'personality-handoff/1.0.0'
MODEL = 'four-seat-personality/2.0.0'
CHECKPOINT = 'personality-checkpoint/1.0.0'
GOALS = 'goal-schema/1.0.0'
CONTEXT = 'personality-context/1.0.0'
TRACE = 'personality-traces/1.0.0'
FAMILIES = ('sale_chain', 'industry_sequence', 'resource_supply', 'network_extension',
            'liquidity_recovery', 'flexible')
INDUSTRIES = ('none', 'CottonMill', 'Manufacturer', 'Pottery', 'Brewery', 'IronWorks', 'CoalMine')
# City areas are public identifiers already present in the rules contract.
REGIONS = ('unspecified',) + tuple(sorted({s.split(':')[0] for s in read_json(CONFIG_DIR/'features-1.0.0.json')['slots']}))
HORIZONS = (1, 2, 4)
PROFILES = ('neutral', 'industrialist', 'networker', 'supplier', 'investor')
FACTORS = dict(family=FAMILIES, industry=INDUSTRIES, region=REGIONS, horizon=HORIZONS)
CONTEXT_DIM = 4 + len(FAMILIES) + len(INDUSTRIES) + len(REGIONS) + len(HORIZONS) + 2


def schema():
    return dict(format=GOALS, factors={k:list(v) for k,v in FACTORS.items()}, profiles=list(PROFILES), context=CONTEXT,
        context_dim=CONTEXT_DIM, confidence_shape=[len(FAMILIES),len(HORIZONS)],
        horizon='Current remaining own turn plus subsequent own turns, same era; unknown negatives masked',
        labels='Public completion proxies, not recovered mental intent or optimality labels',
        input='Original permitted observation plus actual predecision profile/saved intent only')


def source_id():
    return digest({p.name:p.read_text(encoding='utf-8').replace('\r\n','\n')
                   for p in sorted(Path(__file__).parent.glob('*.py'))})


def contract():
    return dict(model=MODEL, goal_schema=digest(schema()), base=compatibility(), source=source_id())


def accepted_contract(recorded):
    """Only exact, explicitly reviewed P0/P1 continuation sources are allowed."""
    current=contract()
    if recorded==current:return None
    path=CONFIG_DIR.parent/'personality/source-migrations.json'
    receipt=read_json(path) if path.is_file() else {}
    require(receipt.get('format')=='personality-source-migrations/1.0.0',
        'Personality contract/source differs; no reviewed migration')
    matches=[entry for entry in receipt.get('migrations',[])
        if entry.get('from_contract')==recorded and entry.get('to_contract')==current]
    require(len(matches)==1 and {k:v for k,v in recorded.items() if k!='source'}==
        {k:v for k,v in current.items() if k!='source'},'Personality contract/source differs')
    require(matches[0].get('kind') in ('p0-p1-persistence-only/1.0.0',
        'p0-p1-audit-continuation/1.0.0'),'Unsupported personality migration')
    return dict(version=matches[0]['kind'],receipt=digest(matches[0]),
        from_source=recorded['source'],to_source=current['source'])


def context(profile='neutral', intent=None):
    value=dict(format=CONTEXT,profile=profile,saved_intent=intent)
    context_vector(value)
    return value


def compatible(family, industry, region):
    if family=='sale_chain': allowed=INDUSTRIES[1:4]
    elif family=='resource_supply': allowed=INDUSTRIES[4:]
    elif family=='industry_sequence': allowed=INDUSTRIES[1:]
    else: allowed=('none',)
    return industry in allowed and (region=='unspecified' if family in ('liquidity_recovery','flexible') else region in REGIONS)


def context_vector(value):
    keys(value,('format','profile','saved_intent'))
    require(value['format']==CONTEXT and value['profile'] in PROFILES,'Unknown personality context')
    result=[float(value['profile']==p) for p in PROFILES[1:]]
    intent=value['saved_intent']
    if intent is None: return np.asarray(result+[0.]*(CONTEXT_DIM-4),dtype=np.float32)
    keys(intent,('schema','family','industry','region','horizon','confidence','age','created_turn','revision_reason'))
    require(intent['schema']==GOALS and type(intent['created_turn']) is str and bool(intent['created_turn'])
        and type(intent['revision_reason']) is str and bool(intent['revision_reason']),'Invalid saved intent provenance')
    for name,vocab in FACTORS.items(): require(intent[name] in vocab,'Unknown goal factor: '+name)
    require(compatible(intent['family'],intent['industry'],intent['region']),'Incompatible goal factors')
    finite_vector([intent['confidence']],1,0,1)
    require(type(intent['age']) is int and 0<=intent['age']<=4,'Invalid own-turn intent age')
    for name,vocab in FACTORS.items(): result.extend(float(intent[name]==v) for v in vocab)
    return np.asarray(result+[intent['confidence'],intent['age']/4],dtype=np.float32)


def local_file(directory, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute(),'Expected bundle-relative path')
    path=(Path(directory)/relative).resolve()
    require(path.is_relative_to(Path(directory).resolve()) and path.is_file(),'Missing/escaping bundle file')
    return path
