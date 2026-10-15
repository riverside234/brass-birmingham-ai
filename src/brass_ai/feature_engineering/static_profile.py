"""Admission of omitted constants for the pinned fixed board/tile catalogue.

Raw observations stay complete. Reject unsupported variants instead of encoding
two materially different positions as the same compact tensor.
"""
from brass_ai.learning.contracts import require


def admit_static(view,schema):
    profile=schema['static_profile']
    require(view['rules']==profile['rules'],'Compacted features require the pinned rules profile')
    require(all(p['hidden_discards']==int(view['context']['era']=='CanalEra') for p in view['players']),
            'Compacted features require era-derived hidden discards')
    for slot in view['slots']:
        require(slot['id'] in profile['slot_types'] and
                sorted(slot['types'])==profile['slot_types'][slot['id']],
                'Compacted features require fixed slot types: '+slot['id'])
        tile=slot.get('tile')
        if tile is not None:
            key=tile['type']+':'+str(tile['level'])
            expected=profile['placed_tiles'].get(key)
            require(expected is not None and all(tile[k]==value for k,value in expected.items()),
                    'Compacted features require fixed placed-tile properties: '+key)
    for merchant in view['merchants']:
        require(merchant['id'] in profile['merchant_bonus'] and
                merchant['bonus']==profile['merchant_bonus'][merchant['id']],
                'Compacted features require fixed merchant bonuses: '+merchant['id'])
