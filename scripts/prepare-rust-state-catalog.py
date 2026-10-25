"""Extract immutable C# semantic definitions; never infer upstream ID mappings."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    args = parser.parse_args()
    data = args.initial.read_bytes()
    state = json.loads(data)
    assert not state['history'] and state['context']['revision'] == 0
    tiles = [t for p in state['players'] for m in p['mat'] for t in m['tiles']]
    assert len(tiles) == len({t['id'] for t in tiles}) == 180
    assert len(state['slots']) == 49 and all('tile' not in s for s in state['slots'])
    assert len(state['links']) == 69 and all('owner' not in s for s in state['links'])
    composition = json.loads((ROOT/'configs/contracts/v0.2.0/card-composition.json').read_text())
    assert sum(composition.values()) == 64
    composition.update({'wild:industry': 4, 'wild:location': 4})
    catalog = dict(source_initial_sha256=hashlib.sha256(data).hexdigest(),
                   tiles=tiles, slots=state['slots'], links=state['links'],
                   merchants=[dict(id=m['id'], bonus=m['bonus']) for m in state['merchants']],
                   merchant_types=[m['types'] for m in state['merchants']], cards=composition)
    dest = ROOT/'native/brass-ai-core/data'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'state-catalog.json').write_text(json.dumps(catalog,indent=2)+'\n',encoding='utf-8')
    tests = ROOT/'native/brass-ai-core/tests/data'
    tests.mkdir(parents=True, exist_ok=True)
    (tests/'initial-state.json').write_bytes(data)
    print('Catalog: 180 tiles, 49 slots, 69 links, 9 merchants, 72 cards')


if __name__ == '__main__': main()
