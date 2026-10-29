"""Authoritative game-level labels, deliberately outside Features-4 inputs."""
from functools import lru_cache
from brass_ai.learning.contracts import SOURCE_ROOT,digest,keys,read_json,require
from brass_ai.learning.board_topology import topology

CONTRACT='local-scoring-targets/1.0.0'
FIELD='local_scoring_targets'


@lru_cache(maxsize=1)
def identity():
    return dict(contract=CONTRACT,topology=digest(topology()),horizon='current-era',
                ordering='canonical-stable-seat',rules=topology()['rules'])


def seal(envelope):return digest({k:v for k,v in envelope.items() if k!='sha256'})


def bind(native,game,producer):
    envelope=dict(identity(),game_id=game['game_id'],setup_group=game['setup_group'],
        branch_parent=game.get('branch',{}).get('parent_game_id'),producer=producer,receipt=native)
    envelope['sha256']=seal(envelope)
    validate(envelope,game)
    return envelope


def _ints(v,n,lo=-2147483648,hi=2147483647):
    require(type(v) is list and len(v)==n and all(type(x) is int and lo<=x<=hi for x in v),'Invalid scoring integers')


def validate(envelope,game):
    keys(envelope,(*identity(),'game_id','setup_group','branch_parent','producer','receipt','sha256'))
    require(all(envelope[k]==v for k,v in identity().items()),'Local scoring topology/contract differs')
    require(envelope['sha256']==seal(envelope),'Local scoring receipt digest differs')
    require(envelope['game_id']==game['game_id'] and envelope['setup_group']==game['setup_group'] and
            envelope['branch_parent']==game.get('branch',{}).get('parent_game_id'),'Local scoring game/branch binding differs')
    # Producer hashes are immutable provenance, not general host compatibility.
    if envelope['producer'].get('contract') == 'brass-native-scoring-producer/1.0.0':
        from brass_ai.rust_backend.lab_admission import validate_scorer
        validate_scorer(envelope['producer'])
        recorder='rust-local-scoring/1.0.0'
    else:
        keys(envelope['producer'],('runtime_sha256','host_sha256'))
        require(all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
                    for v in envelope['producer'].values()),'Invalid local scoring producer')
        recorder='local-scoring-host/1.0.0'
    validate_receipt(envelope['receipt'],game,producer=recorder)
    return envelope


def validate_receipt(r,game,*,producer):
    """Shared semantic audit; callers separately admit their producer envelope.

    Legacy replay still requires the authored C# envelope. B03's separate native
    replay route binds its typed producer and independent admission receipt.
    """
    require(producer in ('local-scoring-host/1.0.0','rust-local-scoring/1.0.0'),'Unknown scoring recorder')
    keys(r,('contract','rules','producer','start_revision','initial_vp','eras','final_vp'))
    require(r['contract']==CONTRACT and r['rules']==identity()['rules'] and r['producer']==producer,'Scoring recorder differs')
    require(type(r['start_revision']) is int and r['start_revision']>=0,'Invalid scoring start')
    first_revision=game['records'][0]['view']['context'].get('revision')
    require(first_revision is None or r['start_revision']==first_revision,'Scoring capture starts outside this continuation')
    _ints(r['initial_vp'],4);_ints(r['final_vp'],4)
    expected={row['view']['context']['era'] for row in game['records']}
    require(type(r['eras']) is list and [e['era'] for e in r['eras']]==[e for e in ('CanalEra','RailEra') if e in expected],
            'Missing, duplicate or inherited era scoring receipts')
    previous=r['initial_vp'];board=topology()
    for e in r['eras']:
        keys(e,('era','slots','links','before','audit'));_ints(e['before'],4)
        totals={}
        for kind in ('slots','links'):
            rows=e[kind];order=board['slot_order' if kind=='slots' else 'link_order']
            require(type(rows) is list and [x['id'] for x in rows]==order,'Local entity ordering differs')
            sums=[0]*4
            for row in rows:
                keys(row,('id','owner','vp','valid')+(('instance',) if kind=='slots' else ()))
                valid=kind=='slots' or board['link_eras'][row['id']]==e['era'].replace('Era','')
                require(type(row['valid']) is bool and row['valid']==valid,'Local era mask differs')
                require(type(row['owner']) is int and 0<=row['owner']<=4 and type(row['vp']) is int and 0<=row['vp']<=65535,'Local owner/VP bounds')
                require((row['owner']!=0 or row['vp']==0) and (valid or row['owner']==row['vp']==0),'Invalid NONE/masked VP')
                if kind=='slots':
                    require(row['instance'] is None or isinstance(row['instance'],str),'Invalid scoring tile instance')
                    require(row['owner']==0 or row['instance'] is not None,'Owned industry lacks an instance audit')
                if row['owner']:sums[row['owner']-1]+=row['vp']
            totals[kind]=sums
        audit=e['audit'];keys(audit,('industry','network','bonus','bonus_since_recording','outside_adjustment','outside_scoring','after'))
        for v in audit.values():_ints(v,4)
        require(totals['slots']==audit['industry'] and totals['links']==audit['network'],'Local official scoring discrepancy')
        require([e['before'][s]-previous[s] for s in range(4)]==audit['outside_scoring'],'Outside-scoring VP discrepancy')
        require([audit['bonus_since_recording'][s]+audit['outside_adjustment'][s] for s in range(4)]==audit['outside_scoring'],
                'Explicit bonus/adjustment accounting differs')
        require([e['before'][s]+totals['slots'][s]+totals['links'][s] for s in range(4)]==audit['after'],'Era scoring VP discrepancy')
        previous=audit['after']
    final=[next(s['vp'] for s in game['result']['seats'] if s['seat']==i) for i in range(4)]
    require(previous==r['final_vp']==final,'Whole-game local scoring discrepancy')
    return r


def row_target(game,row):
    """Reference one shared era map; never copy its arrays into each decision."""
    envelope=game.get(FIELD)
    if envelope is None:return None
    era=row['view']['context']['era']
    target=next((e for e in envelope['receipt']['eras'] if e['era']==era),None)
    require(target is not None,'Missing current-era scoring target')
    return dict(envelope=envelope,era=era)


def map_for(row):
    value=row.get('_local_scoring')
    if value is None:return None
    return next(e for e in value['envelope']['receipt']['eras'] if e['era']==value['era'])


def pack(envelope):
    """Sidecars omit repeated canonical IDs/field names, preserving exact seals."""
    receipt=envelope['receipt'];eras=[]
    for e in receipt['eras']:
        eras.append(dict(e,slots=[[r['owner'],r['vp'],r['instance']] for r in e['slots']],
            links=[[r['owner'],r['vp'],r['valid']] for r in e['links']]))
    return dict(envelope,receipt=dict(receipt,eras=eras))


def unpack(value):
    board=topology();eras=[]
    for e in value['receipt']['eras']:
        require(len(e['slots'])==49 and len(e['links'])==69,'Invalid packed local scoring dimensions')
        require(all(len(row)==3 for k in ('slots','links') for row in e[k]),'Invalid packed local scoring row')
        slots=[dict(id=name,owner=r[0],vp=r[1],valid=True,instance=r[2]) for name,r in zip(board['slot_order'],e['slots'])]
        links=[dict(id=name,owner=r[0],vp=r[1],valid=r[2]) for name,r in zip(board['link_order'],e['links'])]
        eras.append(dict(e,slots=slots,links=links))
    result=dict(value,receipt=dict(value['receipt'],eras=eras))
    require(result['sha256']==seal(result),'Packed local scoring receipt changed')
    return result
