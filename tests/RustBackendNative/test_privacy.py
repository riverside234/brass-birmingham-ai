"""L06 privacy/recall/sampling against sealed C# views and executed frozen Python."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from brass_ai.native import NativeError, NativeWorker
from brass_ai.rust_backend.contracts import first_difference
from brass_ai.searchlab.spiel import BrassSpielState

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path(os.environ.get('BRASS_PRIVACY_CAPTURE', ROOT / 'artifacts/rust-backend/20261025-l06-capture'))
EDGES = Path(os.environ.get('BRASS_PRIVACY_EDGES', ROOT / 'artifacts/rust-backend/20261025-l06-edges'))
RULES = ROOT / 'artifacts/rust-backend/20261025-l05-capture-v2'
FROZEN = ROOT / 'artifacts/rust-backend/20261024-g890-l01-l02/local/fixtures'
WORKER = Path(os.environ.get('BRASS_NATIVE_WORKER', ROOT / 'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-worker.exe'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        yield from map(json.loads, stream)


class ReferenceMemory:
    """Execute the frozen adapter's methods, without creating another rules host."""
    refresh = BrassSpielState.refresh
    is_terminal = BrassSpielState.is_terminal
    information_state_string = BrassSpielState.information_state_string
    observation_string = BrassSpielState.observation_string

    def __init__(self):
        self.context = None
        self.views = [None]*4
        self.recall = [[] for _ in range(4)]
        self.setup_stage = None
        self.pending = None

    def update(self, expected):
        self.context = expected.get('context', expected.get('state', {}).get('context'))
        snapshot = dict(expected, checkpoint={})
        if 'state' in expected:
            snapshot['result'] = expected['state']['result']
        self.refresh(SimpleNamespace(fast=True), snapshot)


class PrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = read(CAPTURE / 'capture.json')
        for name, digest in cls.receipt['files'].items():
            with (CAPTURE/name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest

    def worker(self):
        worker = NativeWorker(WORKER, timeout=30)
        self.addCleanup(worker.close)
        return worker

    def same(self, expected, actual, label):
        if expected == actual:
            return
        diff = first_difference(expected, actual)
        if diff:
            if dest := os.environ.get('BRASS_PRIVACY_FAILURE'):
                Path(dest).write_text(json.dumps(dict(label=label, difference=diff, expected=expected, actual=actual), indent=2)+'\n')
            self.fail(f'{label}: {diff}')

    def information(self, worker, session, reference, label):
        for seat in range(4):
            actual = worker.observation(session, seat)
            expected = dict(observation=json.loads(reference.observation_string(seat)), recall=reference.recall[seat])
            self.same(expected, actual['information'], label+f' seat {seat}')
            self.assertEqual(reference.information_state_string(seat), actual['reference_digest'], label)

    def test_all_roots_four_seat_allowlists(self):
        worker = self.worker()
        count = 0
        for i, row in enumerate(rows(CAPTURE / 'roots.jsonl.gz')):
            state = row['state']; session = i % 2
            worker.import_state(session, state)
            reference = ReferenceMemory()
            expected = dict(state=state, views=row['views'], choices=[], chance=None if not state['chance'] else dict(kind=state['chance']['kind']))
            reference.update(expected)
            self.information(worker, session, reference, row['id'])
            count += 1
        self.assertEqual(count, self.receipt['roots'])

    def test_recorded_draw_worlds_and_sampled_opponent_continuations(self):
        worker = self.worker(); count = steps = 0; groups = Counter()
        for row in rows(CAPTURE / 'samples.jsonl.gz'):
            session = count % 2; label = f'{row["root"]} observer {row["seat"]}'
            receipt = worker.prepare_belief(session, row['view'], row['recall'])
            self.assertEqual(receipt['draw_count'], len(row['draws']))
            self.assertEqual([d['bound'] for d in row['draws']], list(range(receipt['unseen_cards'],1,-1)))
            sample = worker.sample_belief(session, dict(kind='recorded', indices=[d['index'] for d in row['draws']]))
            for field in ('state', 'choices'):
                self.same(row['expected'][field], sample[field], label+' '+field)
            reference = ReferenceMemory(); reference.update(row['expected']); reference.recall[row['seat']] = deepcopy(row['recall'])
            self.information(worker, session, reference, label)
            current = sample['state']
            for step in row['suffix']:
                command = step['command']; revision = current['context']['revision']
                actual = worker.apply_choice(session, revision, command['choice']) if 'choice' in command else worker.resolve_chance(session, revision, command['card'])
                for field in ('state','choices','chance'):
                    self.same(step['expected'][field], actual[field], label+' sampled step '+str(steps)+' '+field)
                reference.update(step['expected']); self.information(worker, session, reference, label+' sampled continuation')
                current = actual['state']; steps += 1
            context = row['view']['context']; groups[f'{context["era"]}:{context["phase"]}:{row["seat"]}'] += 1
            count += 1
        self.assertEqual(count, self.receipt['sampled_worlds']); self.assertEqual(steps, self.receipt['transitions'])
        if dest := os.environ.get('BRASS_PRIVACY_SAMPLING_REPORT'):
            Path(dest).write_text(json.dumps(dict(build=worker.hello['build'], worlds=count, transitions=steps, coverage=dict(groups), unexplained_differences=0),indent=2)+'\n')

    def test_optional_empty_merchants_preserve_wire_identity_through_both_eras(self):
        receipt=read(EDGES/'capture.json')
        with (EDGES/'edges.jsonl.gz').open('rb') as stream:
            self.assertEqual(hashlib.file_digest(stream,'sha256').hexdigest(),receipt['files']['edges.jsonl.gz'])
        worker=self.worker();cases=steps=complete=0
        for row in rows(EDGES/'edges.jsonl.gz'):
            session=cases%2
            worker.prepare_belief(session,row['view'],row['recall'])
            current=worker.sample_belief(session,dict(kind='recorded',indices=[d['index'] for d in row['draws']]))
            self.same(row['expected']['state'],current['state'],row['id']+' import')
            reference=ReferenceMemory();reference.update(row['expected']);reference.recall[row['view']['seat']]=[]
            self.information(worker,session,reference,row['id'])
            for step in row['suffix']:
                command=step['command'];revision=current['state']['context']['revision']
                current=worker.apply_choice(session,revision,command['choice']) if 'choice' in command else worker.resolve_chance(session,revision,command['card'])
                for field in ('state','choices','chance'):self.same(step['expected'][field],current[field],row['id']+' '+field)
                reference.update(step['expected']);self.information(worker,session,reference,row['id']+' continuation')
                steps+=1
            complete+=int(row['complete_game']);cases+=1
            if row['complete_game']:self.assertTrue(current['state']['context']['terminal'])
        self.assertEqual((cases,steps,complete),(receipt['cases'],receipt['transitions'],1))
        if dest:=os.environ.get('BRASS_PRIVACY_EDGE_REPORT'):
            Path(dest).write_text(json.dumps(dict(build=worker.hello['build'],cases=cases,transitions=steps,complete_games=complete,unexplained_differences=0),indent=2)+'\n')

    def test_twenty_complete_games_observations_recall_and_chance_reveals(self):
        worker = self.worker(); transitions = snapshots = chances = 0; games = []
        corpus = read(FROZEN / 'corpus.json')
        native = iter(rows(RULES / 'games.jsonl.gz'))
        for game in corpus['games']:
            path = FROZEN / game['path']
            with path.open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream,'sha256').hexdigest(), game['sha256'])
            stream = iter(rows(path)); header = next(stream); initial = next(native); session = len(games)%2
            self.assertEqual(initial['game'], game['id']); games.append(game['id'])
            worker.import_state(session, initial['initial']['state'])
            reference = ReferenceMemory(); reference.update(header['projection'])
            self.information(worker, session, reference, game['id']+' initial'); snapshots += 1
            current = initial['initial']['state']
            for row in stream:
                if 'command' not in row:
                    continue
                expected = next(native); command = row['command']; revision = current['context']['revision']
                self.assertEqual(expected['command'], command)
                actual = worker.apply_choice(session,revision,command['choice']) if 'choice' in command else worker.resolve_chance(session,revision,command['card'])
                self.same(expected['expected']['state'], actual['state'], game['id']+f' step {transitions}')
                reference.update(row['after']); self.information(worker,session,reference,game['id']+f' step {transitions}')
                current = actual['state']; transitions += 1; snapshots += 1; chances += int(bool(current['chance']))
                if transitions % 97 == 0:
                    saved = [worker.observation(session,s)['information'] for s in range(4)]
                    worker.save_state(session)
                    worker.import_state(session,initial['initial']['state'])
                    worker.restore_state(session)
                    for s in range(4): self.same(saved[s],worker.observation(session,s)['information'],'saved permitted memory')
            print('L06 full-game privacy', game['id'], flush=True)
        self.assertIsNone(next(native,None)); self.assertEqual(transitions,8344)
        if dest := os.environ.get('BRASS_PRIVACY_REPORT'):
            Path(dest).write_text(json.dumps(dict(build=worker.hello['build'],games=games,transitions=transitions,
                snapshots=snapshots,seat_observations=4*snapshots,chance_snapshots=chances,unexplained_differences=0,
                sessions=2,threads_per_session=1),indent=2)+'\n')

    def test_hidden_counterfactuals_recall_independence_and_recorded_behavior(self):
        worker = self.worker(); selected = {}
        for row in rows(CAPTURE/'roots.jsonl.gz'):
            c = row['state']['context']
            if row['views'] and not c['terminal']:
                selected.setdefault((c['era'],c['phase']),row)
        for key,row in selected.items():
            for seat in range(4):
                state = row['state']; changed = deepcopy(state); cards = changed['cards']
                places = []
                for other in range(4):
                    if other != seat:
                        places.extend((cards['hands'][other],i) for i,c in enumerate(cards['hands'][other]) if not c.startswith('wild:'))
                    if state['context']['era']=='CanalEra': places.append((cards['discards'][other],0))
                    else:
                        start=cards['discard_starts'][other]
                        cards['discards'][other][:start]=reversed(cards['discards'][other][:start])
                places.extend((cards['deck'],i) for i in range(len(cards['deck'])))
                values = [p[i] for p,i in places]; random.Random(890+seat).shuffle(values)
                for (p,i),v in zip(places,values): p[i]=v
                cards['hands'][seat].reverse()
                for session,source in enumerate([state,changed]):
                    worker.import_state(session,source)
                a = worker.observation(0,seat); b = worker.observation(1,seat)
                self.same(a['information'],b['information'],str(key)+' private counterfactual')
                self.assertEqual(a['reference_digest'],b['reference_digest'])
                view = a['information']['observation']['view']
                # Same permitted root, different allowed recall: same world,
                # different information-set identity, without extra conditioning.
                recall = a['information']['recall']
                for session,r in enumerate([[],recall]): worker.prepare_belief(session,view,r)
                first = worker.sample_belief(0,dict(kind='chacha12',seed=890,stream=seat))
                second = worker.sample_belief(1,dict(kind='recorded',indices=first['draws']))
                self.same(first['state'],second['state'],'counterfactual sampled world')
                self.assertNotEqual(worker.observation(0,seat)['reference_digest'],worker.observation(1,seat)['reference_digest'])
                for _ in range(8):
                    current = worker.rules_snapshot(0)
                    if current['state']['context']['terminal']: break
                    revision = current['state']['context']['revision']
                    if current['chance']:
                        card = current['chance']['outcomes'][0]['card']
                        results = [worker.resolve_chance(s,revision,card) for s in range(2)]
                    else:
                        choice = next((c for c in current['choices'] if c['kind'] in ('finish','pass','bonus_skip')),current['choices'][0])
                        results = [worker.apply_choice(s,revision,choice['id']) for s in range(2)]
                    for field in ('state','choices','chance'): self.same(results[0][field],results[1][field],'recorded behavior '+field)

    def test_rejected_hidden_payloads_conservation_draws_and_recall_are_atomic(self):
        worker = self.worker(); row = next(rows(CAPTURE/'samples.jsonl.gz')); view = row['view']
        for s in range(2):
            worker.prepare_belief(s,view,row['recall']); worker.sample_belief(s,dict(kind='recorded',indices=[d['index'] for d in row['draws']]))
            worker.save_state(s)
        original = worker.export_state(0)['state']
        memory = [worker.observation(0,s)['information'] for s in range(4)]
        bad = []
        # Unknown fields are rejected throughout the typed allowlist.
        for target in [(),('context',),('players',0),('players',0,'mat',0),('players',0,'mat',0,'tiles',0),('slots',0),('links',0),('merchants',0),('pending',)]:
            v = deepcopy(view); obj = v
            for part in target: obj = obj[part]
            obj['hidden_state'] = {'opponent_hand':['location:Belper']}
            bad.append((v,row['recall']))
        for field,value in [('deck_count',63),('own_hand',view['own_hand'][:-1]),('seat',4),('protocol','unsupported')]:
            v=deepcopy(view);v[field]=value;bad.append((v,row['recall']))
        v=deepcopy(view);v['players'][1]['known_wild']=['location:Belper'];bad.append((v,row['recall']))
        v=deepcopy(view);v['merchants'][0]['types']=None;bad.append((v,row['recall']))
        bad.append((view,[dict(revision=view['context']['revision']+1,hand=view['own_hand'])]))
        bad.append((view,[dict(revision=0,hand=[],secret='hidden')]))
        commands=[lambda v=v,r=r:worker.prepare_belief(0,v,r) for v,r in bad]
        commands += [lambda indices=indices:worker.sample_belief(0,dict(kind='recorded',indices=indices))
                     for indices in ([],[999]*len(row['draws']),[0]*(len(row['draws'])+1))]
        commands += [lambda:worker.call('belief_prepare',session=0,view=view,recall=[],checkpoint=original),
                     lambda:worker.call('belief_sample',session=0,random=dict(kind='chacha12',seed=890,stream=0,live_rng=1))]
        for command in commands:
            with self.assertRaises(NativeError): command()
            self.same(original,worker.export_state(0)['state'],'rejected sample live')
            for s in range(4): self.same(memory[s],worker.observation(0,s)['information'],'rejected sample recall')
            for s in range(2): worker.restore_state(s); self.same(original,worker.export_state(s)['state'],'preserved owner save')
            valid = worker.sample_belief(0,dict(kind='recorded',indices=[d['index'] for d in row['draws']]))
            self.same(original,valid['state'],'prepared root survives rejection')


if __name__ == '__main__':
    unittest.main()
