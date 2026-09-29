"""Authored schema fixtures, not exported game saves or proprietary rules code."""
import copy

from brass_ai.learning.contracts import CONFIG_DIR, read_json
from brass_ai.learning.replay import split_for


def position(observer=0):
    schema = read_json(CONFIG_DIR / 'features-1.0.0.json')
    hand = ['industry:CoalMine', 'location:Birmingham'] * 4
    players = []
    for seat in range(4):
        mat = []
        for kind in schema['industries']:
            tile = dict(id=f'{seat}:{kind}:0', seat=seat, type=kind, level=2,
                flipped=False, resource=1, vp=4, link_vp=1, income=4, cost=8,
                coal=0, iron=0, beer=0, developable=True, canal_only=False, rail_only=False)
            mat.append({'type': kind, 'tiles': [tile]})
        players.append(dict(seat=seat, cash=17 + seat, income_marker=10, income=0,
            vp=seat * 10, bonus_vp=0, link_tiles=14, hand_count=8, spent=seat,
            discards=[], hidden_discards=1, known_wild=[], mat=mat, debt=0))
    context = dict(protocol='0.2.0', rules='steam-brass-patch-1.1.0', revision=observer,
        actor=observer, turn_owner=observer, phase='action', actions_remaining=1,
        era='CanalEra', round=1, turn=observer + 1, terminal=False)
    view = dict(protocol='0.2.0', rules='steam-brass-patch-1.1.0', seat=observer,
        context=context, own_hand=hand, players=players, order=[0, 1, 2, 3],
        slots=[{'id': x, 'types': ['CottonMill']} for x in schema['slots']],
        links=[{'id': x, 'regions': ['Birmingham', 'Coventry'],
                'era': 'Canal' if x.startswith('Canal') else 'Rail'} for x in schema['links']],
        merchants=[{'id': x, 'types': ['CottonMill'], 'beer': 1, 'bonus': 'VP'} for x in schema['merchants']],
        coal_market=12, iron_market=8, deck_count=28, wild_industry_count=4,
        wild_location_count=4, pending=dict(kind='', target='', merchant='', parts=0, coal=0, iron=0, beer=0), history=[])
    choices = [dict(id='loan-a', kind='loan', card=hand[0]),
               dict(id='pass-a', kind='pass', card=hand[0]),
               dict(id='pass-b', kind='pass', card=hand[1])]
    return view, choices


def result(scores=None):
    scores = scores or [(100, 10, 1), (100, 10, 1), (80, 20, 2), (50, 10, 1)]
    return {'objective': 'win-share/1.0.0', 'seats': [dict(seat=i, vp=score[0],
        income=score[1], cash=score[2], rank=1 + sum(s > score for s in scores),
        utility=1 / scores.count(max(scores)) if score == max(scores) else 0.0)
        for i, score in enumerate(scores)]}


def game():
    records = []
    for seat in range(4):
        view, choices = position(seat)
        records.append(dict(step=seat, seat=seat, view=view, choices=choices, chosen=0,
            behavior=dict(kind='random', policy_id='fixture', probability=1 / 3, temperature=1.0),
            style=[0.0] * 4, intent=[0.0] * 8, teacher=None, seconds=0.01))
    group = 'authored-fixture'
    return dict(game_id='fixture-game', setup_group=group, players=4, split=split_for(group),
                opponents=['fixture'] * 4, result=result(), records=records)


def rotate(view, shift):
    view = copy.deepcopy(view)
    change = lambda seat: (seat + shift) % 4
    view['seat'] = change(view['seat'])
    for key in ('actor', 'turn_owner'):
        view['context'][key] = change(view['context'][key])
    view['order'] = [change(s) for s in view['order']]
    for player in view['players']:
        player['seat'] = change(player['seat'])
        for mat in player['mat']:
            for tile in mat['tiles']:
                tile['seat'] = change(tile['seat'])
                tile['id'] = str(tile['seat']) + tile['id'][1:]
    view['players'].sort(key=lambda p: p['seat'])
    for slot in view['slots']:
        if 'tile' in slot:
            slot['tile']['seat'] = change(slot['tile']['seat'])
    for link in view['links']:
        if 'owner' in link:
            link['owner'] = change(link['owner'])
    for event in view['history']:
        event['seat'] = change(event['seat'])
    return view
