"""L04 semantic state parity and independent derived-index checks.

The source roots are L02 frozen C# fixtures. A separate diagnostic exporter
verifies their canonical projections before producing these trusted DTOs.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import unittest

from brass_ai.native import NativeWorker, NativeError
from brass_ai.rust_backend.contracts import first_difference

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path(os.environ.get('BRASS_STATE_CAPTURE', ROOT/'artifacts/rust-backend/20261024-l04/states'))
WORKER = Path(os.environ.get('BRASS_NATIVE_WORKER', ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-worker.exe'))
REGRESSIONS=Path(os.environ.get('BRASS_STATE_REGRESSIONS',CAPTURE.parent/'regressions-v2/regressions.json'))


def load_rows():
    receipt = json.loads((CAPTURE/'capture.json').read_text())
    data = (CAPTURE/'states.jsonl.gz').read_bytes()
    assert hashlib.sha256(data).hexdigest() == receipt['states_sha256']
    rows = [json.loads(line) for line in gzip.decompress(data).splitlines()]
    assert len(rows) == receipt['count'] == 546
    for row, entry in zip(rows, receipt['entries']):
        digest = hashlib.sha256(json.dumps(row['state'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert row['id'] == entry['id'] and digest == entry['state_sha256']
    return rows


def indexes(state):
    """Independent graph traversal, not the Rust DSU or its numeric ID mapping."""
    regions = {s['id'].split(':')[0] for s in state['slots']}
    regions.update(r for link in state['links'] for r in link['regions'])
    graph = {r: set() for r in regions}
    owned_slots = [[] for _ in range(4)]; owned_links = [[] for _ in range(4)]
    networks = [set() for _ in range(4)]
    resources = defaultdict(list)
    for link in state['links']:
        if 'owner' in link:
            owner = link['owner']; owned_links[owner].append(link['id'])
            networks[owner].update(link['regions'])
            for region in link['regions']:
                graph[region].update(link['regions'])
    for slot in state['slots']:
        tile = slot.get('tile')
        if tile:
            owned_slots[tile['seat']].append(slot['id'])
            networks[tile['seat']].add(slot['id'].split(':')[0])
            if tile['resource']:
                resources[tile['type']].append(slot['id'])
    groups = []; remaining = set(regions)
    while remaining:
        todo = [min(remaining)]; seen = set(todo)
        while todo:
            for neighbor in graph[todo.pop()] - seen:
                seen.add(neighbor); todo.append(neighbor)
        groups.append(sorted(seen)); remaining -= seen
    return dict(connected_regions=groups, owned_slots=owned_slots, owned_links=owned_links,
                network_regions=[sorted(x) for x in networks], coal_slots=resources['CoalMine'],
                iron_slots=resources['IronWorks'], beer_slots=resources['Brewery'])


class StateParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_rows()
        cls.initial = json.loads((CAPTURE/'initial-state.json').read_text())
        receipt=json.loads((CAPTURE/'capture.json').read_text())
        cls.supplemental=json.loads((CAPTURE/'supplemental.json').read_text())
        assert hashlib.sha256((CAPTURE/'supplemental.json').read_bytes()).hexdigest()==receipt['supplemental_sha256']
        regressions=json.loads(REGRESSIONS.read_text())
        assert regressions['source_capture_sha256']==hashlib.sha256((CAPTURE/'capture.json').read_bytes()).hexdigest()
        cls.corrected=regressions['cases']
        cls.rejected={r['original_id']:r for r in cls.corrected}
        assert len(cls.rejected)==6

    def worker(self):
        worker = NativeWorker(WORKER); self.addCleanup(worker.close); return worker

    def same(self, expected, actual, name):
        self.assertIsNone(first_difference(expected, actual), name + ': ' + str(first_difference(expected, actual)))

    def test_all_frozen_roots_roundtrip_indexes_and_clone_restore(self):
        worker = self.worker(); coverage = Counter(); partitions = Counter(); owners = {}; checks = []
        accepted=[r for r in self.rows if r['id'] not in self.rejected]
        accepted += [dict(id=r['id'],state=r['state'],partition='corrected-authored') for r in self.corrected]
        for i, row in enumerate(accepted):
            state = row['state']; session = i % 2
            with self.subTest(root=row['id']):
                reply = worker.import_state(session, state)
                self.same(state, reply['state'], row['id'] + ' import')
                expected_indexes = indexes(state)
                self.same(expected_indexes, reply['indexes'], row['id'] + ' indexes')
                owners.setdefault(session, reply['owner']); self.assertEqual(owners[session], reply['owner'])
                exported = worker.export_state(session)
                self.same(state, exported['state'], row['id'] + ' export')
                cloned = worker.state_clone_check(session)
                self.assertTrue(cloned['clone_isolated'], row['id'])
                self.same(state, cloned['state'], row['id'] + ' clone original')
                worker.save_state(session)
                worker.import_state(session, self.initial)
                restored = worker.restore_state(session)
                self.same(state, restored['state'], row['id'] + ' restore')
                self.same(expected_indexes, restored['indexes'], row['id'] + ' restored indexes')
                kind = ('chance:' + state['chance']['kind']) if state['chance'] else state['context']['phase']
                coverage[kind] += 1; coverage['era:' + state['context']['era']] += 1
                if state['chance'] and state['chance']['picked']: coverage['partial:' + state['chance']['kind']] += 1
                partitions[row['partition']] += 1; checks.append(row['id'])
        self.assertNotEqual(owners[0], owners[1])
        self.assertEqual(len(checks),len(accepted))
        self.assertEqual(set(coverage) & {'action','payment','continuation','bonus','debt','terminal'},
                         {'action','payment','continuation','bonus','debt','terminal'})
        for key in ('chance:draw','chance:rail-shuffle','era:CanalEra','era:RailEra'):
            self.assertGreater(coverage[key], 0)
        report = dict(format='brass-native-state-parity/1.0.0',build=worker.hello['build'],
                      roots=len(checks),coverage=dict(coverage),partitions=dict(partitions),
                      frozen_roots_roundtripped=540,corrected_roots_roundtripped=6,
                      frozen_roots_rejected=sorted(self.rejected),supplemental_oracle_roots=3,
                      sessions=2,threads_per_session=1,unexplained_differences=0,passed_roots=checks)
        if dest := os.environ.get('BRASS_STATE_PARITY_REPORT'):
            Path(dest).write_text(json.dumps(report,indent=2)+'\n')

    def test_frozen_inconsistent_authored_roots_are_rejected(self):
        worker=self.worker();worker.import_state(0,self.initial);worker.save_state(0)
        for row in self.rows:
            if row['id'] not in self.rejected:continue
            with self.subTest(root=row['id']):
                with self.assertRaises(NativeError) as caught:worker.import_state(0,row['state'])
                self.assertTrue(any(reason in str(caught.exception) for reason in self.rejected[row['id']]['rejection_reasons']))
                self.same(self.initial,worker.export_state(0)['state'],'rejected frozen fixture retains state')
                self.same(self.initial,worker.restore_state(0)['state'],'rejected frozen fixture retains save')

    def test_two_session_save_slots_and_failed_import_are_independent(self):
        worker = self.worker()
        first = self.rows[0]['state']; other = self.rows[-1]['state']
        for session in (0, 1):
            with self.assertRaises(NativeError): worker.export_state(session)
            with self.assertRaises(NativeError): worker.restore_state(session)
        worker.import_state(0, first); worker.save_state(0)
        worker.import_state(1, other); worker.save_state(1)
        worker.import_state(0, other)
        bad = deepcopy(first); bad['cards']['deck'].pop()
        with self.assertRaises(NativeError): worker.import_state(0, bad)
        self.same(other, worker.export_state(0)['state'], 'failed import retained current')
        self.same(first, worker.restore_state(0)['state'], 'failed import retained save')
        self.same(other, worker.restore_state(1)['state'], 'other owner save unchanged')
        worker.import_state(0, other); worker.save_state(0)
        self.same(other, worker.restore_state(0)['state'], 'save slot replaces old snapshot')

    def test_malformed_unsupported_and_inconsistent_states_fail_closed(self):
        worker = self.worker(); worker.import_state(0, self.initial); worker.save_state(0)
        cases = []
        def case(name, edit, source=None):
            value = deepcopy(source or self.initial); edit(value); cases.append((name,value))
        def put(path, value):
            def edit(state):
                for key in path[:-1]: state = state[key]
                state[path[-1]] = value
            return edit
        case('format', put(['format'],'brass-native-state/999'))
        case('rules', put(['rules'],'different-game'))
        case('protocol', put(['context','protocol'],'999'))
        case('3 seats', lambda s:s['players'].pop())
        case('2 card hands', lambda s:s['cards']['hands'].__delitem__(slice(2,None)))
        case('unknown top-level', put(['unsafe'],True))
        case('unknown context', put(['context','unsafe'],True))
        case('unknown tile', put(['players',0,'mat',0,'tiles',0,'id'],'bogus'))
        case('unknown tile field', put(['players',0,'mat',0,'tiles',0,'unsafe'],True))
        case('modified tile definition', put(['players',0,'mat',0,'tiles',0,'cost'],6))
        case('missing tile', lambda s:s['players'][0]['mat'][0]['tiles'].pop())
        case('duplicate tile', lambda s:s['players'][0]['mat'][0]['tiles'].append(deepcopy(s['players'][0]['mat'][0]['tiles'][0])))
        case('removed live tile', lambda s:s['removed'].append(s['players'][0]['mat'][0]['tiles'][0]['id']))
        case('retired live tile', lambda s:s['retired'].append(s['players'][0]['mat'][0]['tiles'][0]['id']))
        case('income', put(['players',0,'income'],30))
        case('income track', put(['players',0,'income_marker'],101))
        case('resource capacity', put(['players',0,'mat',1,'tiles',0,'resource'],255))
        case('link conservation', put(['players',0,'link_tiles'],13))
        case('unknown seat', put(['links',0,'owner'],4))
        case('topology', put(['links',0,'regions'],['Belper','Coventry']))
        case('board count', lambda s:s['slots'].pop())
        case('order duplicate', put(['order'],[0,0,2,3]))
        case('market bounds', put(['coal_market'],255))
        case('merchant conservation', put(['merchants',0,'types'],[]))
        case('phase enum', put(['context','phase'],'unsupported'))
        case('phase chance alias', put(['context','phase'],'chance'))
        case('actor', put(['context','actor'],-1))
        case('actions', put(['context','actions_remaining'],3))
        case('first-round actions', put(['context','actions_remaining'],2))
        case('payment without action', put(['context','phase'],'payment'))
        case('pending while action', put(['pending','kind'],'loan'))
        case('debt without phase', put(['players',0,'debt'],1))
        case('settlement debt without pending', put(['settlement_debts',0],1))
        case('negative settlement debt', put(['settlement_debts',0],-1))
        case('chance flag', put(['context','chance'],True))
        case('terminal flag', put(['context','terminal'],True))
        case('unknown card', put(['cards','hands',0,0],'location:Unknown'))
        case('card conservation', lambda s:s['cards']['deck'].pop())
        case('discard offset', put(['cards','discard_starts',0],1000))
        for kind in ('draw','rail-shuffle'):
            source = next(r['state'] for r in self.rows if r['state']['chance'] and r['state']['chance']['kind']==kind)
            case(kind+' missing pool card',lambda s:s['chance']['pool'].pop(),source)
            case(kind+' impossible needed',put(['chance','needed'],0),source)
            case(kind+' forged history',put(['history',0,'choice','id'],'forged'),source)
            if kind=='rail-shuffle':
                def held_card(s):
                    owner=next(i for i in range(4) if len(s['cards']['discards'][i])>1 and not s['cards']['discards'][i][-1].startswith('wild:'))
                    s['cards']['hands'][owner].append(s['cards']['discards'][owner].pop())
                case('rail pool cannot include a held card',held_card,source)
        terminal = next(r['state'] for r in self.rows if r['state']['context']['terminal'])
        case('terminal utility',put(['result','seats',0,'utility'],.9),terminal)
        multiple=next(r['state'] for r in self.rows if r['state']['context']['phase']=='debt' and r['state']['current_player']!=r['state']['context']['actor'])
        case('wrong debtor order',put(['context','actor'],multiple['current_player']),multiple)
        for name, value in cases:
            with self.subTest(case=name):
                with self.assertRaises(NativeError): worker.import_state(0, value)
                self.same(self.initial, worker.export_state(0)['state'], name+' preserves live state')
                self.same(self.initial, worker.restore_state(0)['state'], name+' preserves save')

    def test_partial_chance_order_and_rail_discard_aliases(self):
        worker = self.worker()
        # Materialize two lawful alternatives of the same paused shuffle. They
        # have identical pool counts but different picked order and future deals.
        shuffle = deepcopy(next(r['state'] for r in self.rows if r['state']['chance'] and r['state']['chance']['kind']=='rail-shuffle'))
        self.assertEqual(shuffle['chance']['picked'], [])
        pool = shuffle['chance']['pool']; first = pool.pop(0)
        second = next(x for x in pool if x != first); pool.remove(second)
        shuffle['chance']['picked'] = [first,second]
        shuffle['context']['revision'] += 2
        a = worker.import_state(0, shuffle)
        reverse = deepcopy(shuffle); reverse['chance']['picked'].reverse()
        b = worker.import_state(1, reverse)
        self.same(shuffle,a['state'],'partial shuffle')
        self.same(reverse,b['state'],'reverse partial shuffle')
        self.assertNotEqual(a['state'],b['state'])
        for session in (0,1): self.assertTrue(worker.state_clone_check(session)['clone_isolated'])
        rail = next(r['state'] for r in self.rows if r['state']['context']['era']=='RailEra' and any(r['state']['cards']['discard_starts']))
        self.same(rail,worker.import_state(0,rail)['state'],'rail aliases')

    def test_additional_oracle_partial_draws_and_settlement_debt(self):
        worker=self.worker();kinds=set()
        for i,row in enumerate(self.supplemental):
            state=row['state'];kinds.add(state['chance']['kind'])
            self.assertTrue(state['chance']['picked'])
            reply=worker.import_state(i%2,state)
            self.same(state,reply['state'],row['id'])
            self.same(indexes(state),reply['indexes'],row['id']+' indexes')
            worker.save_state(i%2);worker.import_state(i%2,self.initial)
            self.same(state,worker.restore_state(i%2)['state'],row['id']+' restore')
            self.assertTrue(worker.state_clone_check(i%2)['clone_isolated'])
        self.assertEqual(kinds,{'draw','rail-shuffle'})
        multi=next(r['state'] for r in self.rows if r['state']['context']['phase']=='debt' and r['state']['current_player']!=r['state']['context']['actor'])
        self.same(multi,worker.import_state(0,multi)['state'],'different debt actor and current player')
        residual=next(r['state'] for r in self.rows if r['state']['context']['chance'] and any(r['state']['settlement_debts']))
        self.assertTrue(all(p['debt']==0 for p in residual['players']))
        self.same(residual,worker.import_state(1,residual)['state'],'residual penalty before rail deal')


if __name__ == '__main__': unittest.main()
