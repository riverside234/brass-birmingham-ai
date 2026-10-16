"""Pinned v11 model topology; does not change Features-4 tensor semantics."""
from functools import lru_cache
import hashlib

from .contracts import CONFIG_DIR, FEATURE_SCHEMA, SOURCE_ROOT, digest, read_json, require

TOPOLOGY_FILE='local-entity-topology-2.0.0.json'
BOARD_FILE='entity-board-1.0.0.json'


def build_topology():
    board=read_json(CONFIG_DIR/BOARD_FILE);schema=read_json(CONFIG_DIR/FEATURE_SCHEMA)
    require(board['feature_schema_sha256']==digest(schema) and
            board['static_profile_sha256']==digest(schema['static_profile']) and
            board['rules']==schema['static_profile']['rules'],'V11 static board profile differs')
    for name,expected in board['provenance']['sources'].items():
        require(hashlib.sha256((SOURCE_ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,
                'V11 board-definition source changed: '+name)
    pins=read_json(SOURCE_ROOT/'configs/rules/runtime-dependencies-1.1.0.json')
    require({f['name']:f['sha256'] for f in board['provenance']['runtime']['files'] if f['name']!='BrassHost.exe'}==
            {f['name']:f['sha256'] for f in pins['files']},'V11 board rules pins differ')
    for family in ('slots','links','merchants'):
        require([v['id'] for v in board[family]]==schema[family],'V11 ordered board identities differ')
    locations={}
    for slot in board['slots']:
        require(slot['region']==slot['id'].split(':')[0] and
                slot['types']==schema['static_profile']['slot_types'][slot['id']], 'V11 slot location/types differ')
        locations.setdefault(slot['region'],[]).append(slot['id'])
    endpoints={l['id']:l['regions'] for l in board['links']}
    require(all(len(v)<=4 for v in locations.values()),'V11 location exceeds four building slots')
    for name,regions in endpoints.items():
        require(2<=len(regions)<=3 and len(regions)==len(set(regions)), 'V11 invalid link endpoints: '+name)
        require(sum(len(locations.get(r,[])) for r in regions)<=8,'V11 link exceeds eight building slots: '+name)
    return dict(format='grouped-mlp-topology/2.0.0',rules=board['rules'],board_sha256=digest(board),
        location_slots=locations,link_endpoints=endpoints,link_eras={l['id']:l['era'] for l in board['links']},
        slot_order=schema['slots'],link_order=schema['links'],
        ordering='slot: own first, then schema-order siblings; link: native region order, schema-order slots; pad trailing zeros')


@lru_cache(maxsize=1)
def topology():
    result=build_topology()
    require(result==read_json(CONFIG_DIR/TOPOLOGY_FILE),'V11 frozen local topology differs')
    return result


def admit_topology(view):
    """Check omitted static facts, including all three regions of junction links."""
    board=topology()
    require(view['rules']==board['rules'],'V11 topology rules differ')
    require(len(view['slots'])==len(board['slot_order']) and
            {s['id'] for s in view['slots']}==set(board['slot_order']),'V11 topology slot identities differ')
    require(len(view['links'])==len(board['link_order']) and
            {l['id'] for l in view['links']}==set(board['link_order']),'V11 topology link identities differ')
    for link in view['links']:
        expected=board['link_endpoints'][link['id']]
        require(len(link['regions'])==len(expected) and set(link['regions'])==set(expected) and
                link['era']==board['link_eras'][link['id']], 'V11 link topology differs: '+link['id'])
