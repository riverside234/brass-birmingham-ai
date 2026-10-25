"""L07 end-to-end OpenSpiel adapter, frozen corpus and unchanged Python search."""
from collections import Counter
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from brass_ai.native import NativeError, NativeWorker, select_backend
from brass_ai.native.simulation import NativeSpielGame
from brass_ai.searchlab.spiel import action_id, BrassSpielGame
from test_privacy import rows, read, ReferenceMemory, CAPTURE, RULES, ROOT, WORKER, FROZEN


class SimulationTests(unittest.TestCase):
    def games(self):
        worker = NativeWorker(WORKER, timeout=30); self.addCleanup(worker.close)
        games = [NativeSpielGame(worker=worker, session=s, root=ROOT) for s in (0,1)]
        for game in games: self.addCleanup(game.close)
        return games

    def same(self, state, expected, reference=None):
        self.assertEqual(state.context, expected['state']['context'])
        self.assertEqual(state.legal, expected['choices'])
        self.assertEqual(state.pending, expected['chance'])
        self.assertEqual(state.is_terminal(), expected['state']['context']['terminal'])
        self.assertEqual(state.returns(), [s['utility'] for s in expected['state']['result']['seats']] if state.is_terminal() else [0.]*4)
        self.assertEqual(state.rewards(), state.returns() if state.is_terminal() else [0.]*4)
        if reference:
            for s in range(4):
                self.assertEqual(state.observation_string(s), reference.observation_string(s))
                self.assertEqual(state.information_state_string(s), reference.information_state_string(s))
        self.assertEqual(state.legal_actions(state.current_player()), sorted(state.semantic_map()) if not state.is_terminal() else [])

    def initial(self, game, expected):
        state = game.new_initial_state(); dto = expected['state']
        picks = [('order', str(s)) for s in dto['order']]
        picks += [('merchants', json.dumps(m['types'])) for m in dto['merchants']]
        deck = sum(dto['cards']['hands'], []) + sum(dto['cards']['discards'], []) + dto['cards']['deck']
        picks += [('cards', c) for c in deck]
        for stage, item in picks:
            self.assertTrue(state.is_chance_node())
            self.assertAlmostEqual(sum(p for _,p in state.chance_outcomes()), 1.)
            self.assertEqual(json.loads(state.observation_string(0)), dict(view=None,setup=True,chance=None))
            state.apply_action(action_id(f'chance:{stage}:{item}'))
        self.assertEqual(state.trusted_checkpoint()['world']['state'], dto)
        return state

    def test_00_twenty_games_initialization_recall_returns_and_restore(self):
        games=self.games(); state=None; count=steps=restores=0; coverage=Counter()
        def observations():
            for game in read(FROZEN/'corpus.json')['games']:
                path=FROZEN/game['path']
                with path.open('rb') as stream:self.assertEqual(hashlib.file_digest(stream,'sha256').hexdigest(),game['sha256'])
                stream=iter(rows(path));header=next(stream)
                yield game['id'],None,header['projection']
                for row in stream:
                    if 'command' in row:yield game['id'],row['command'],row['after']
        oracle=iter(observations())
        for row in rows(RULES/'games.jsonl.gz'):
            game_id,command,projection=next(oracle)
            self.assertEqual((game_id,command),(row['game'],row.get('command')))
            if 'initial' in row:
                if state: state.close()
                game=games[count%2]; count+=1; expected=row['initial']; state=self.initial(game, expected)
                reference=ReferenceMemory(); reference.update(projection)
            else:
                expected=row['expected']; command=row['command']
                semantic=command.get('choice', 'chance:card:'+command.get('card',''))
                state.apply_action(action_id(semantic)); reference.update(projection); steps+=1
                coverage[state.context['era']+':'+state.context['phase']]+=1
                if steps%97==0:
                    with state.clone() as clone, game.deserialize_state(state.serialize()) as portable:
                        self.same(portable,expected,reference)
                        self.assertEqual(portable.history(),state.history())
                        if state.legal_actions(): clone.apply_action(state.legal_actions()[0])
                        clone.restore(state); self.same(clone,expected,reference)
                        self.assertEqual(clone.trusted_checkpoint()['world']['state'],expected['state'])
                    restores+=1
            self.same(state,expected,reference)
            if state.is_terminal(): print('L07 adapter full game',row['game'],flush=True)
        state.close(); self.assertEqual((count,steps),(20,8344));self.assertIsNone(next(oracle,None))
        if dest:=os.environ.get('BRASS_SIMULATION_REPORT'):
            Path(dest).write_text(json.dumps(dict(games=count,transitions=steps,restores=restores,
                coverage=dict(coverage),build=games[0].worker.hello['build'],unexplained_differences=0),indent=2)+'\n')

    def test_all_roots_clone_portable_checkpoint_and_stale_inputs(self):
        games=self.games(); count=0
        for i,row in enumerate(rows(RULES/'roots.jsonl.gz')):
            expected=row['expected'];game=games[i%2]
            with game.import_state(expected['state']) as state:
                before=state.serialize()
                with state.clone() as clone, game.deserialize_state(before) as restored:
                    self.same(state,expected);self.same(clone,expected);self.same(restored,expected)
                    if state.legal_actions():
                        clone.apply_action(state.legal_actions()[0]); clone.restore(state)
                    self.assertEqual(before,clone.serialize())
                    with self.assertRaises(NativeError):
                        game._call('step',handle=state.handle,revision=999999,semantic='invalid')
                    self.assertEqual(before,state.serialize())
            count+=1
        self.assertEqual(count,546)

    def test_all_reference_sample_seeds_preserve_worlds_and_recall(self):
        games=self.games(); count=0
        for i,row in enumerate(rows(CAPTURE/'samples.jsonl.gz')):
            with games[i%2].from_observation(row['view'],890000+i,row['recall']) as state:
                self.assertEqual(state.trusted_checkpoint()['world']['state'],row['expected']['state'])
                reference=ReferenceMemory();reference.update(row['expected']);reference.recall[row['seat']]=row['recall']
                self.same(state,row['expected'],reference)
                for step in row['suffix']:
                    command=step['command'];state.apply_action(action_id(command.get('choice','chance:card:'+command.get('card',''))))
                    reference.update(step['expected']);self.same(state,step['expected'],reference)
            count+=1
        self.assertEqual(count,1704)

    def test_selection_lifetime_setup_clones_and_bounded_ownership(self):
        games=self.games(); game,other=games
        with self.assertRaises(ValueError):NativeSpielGame(worker=game.worker,session=0)
        with select_backend(backend='rust-simulator',root=ROOT,executable=WORKER) as selected:
            with selected.new_initial_state() as state:
                self.assertTrue(state.is_chance_node())
        state=game.new_initial_state(); first=state.serialize()
        with state.clone() as clone:
            clone.apply_action(clone.legal_actions()[0])
            with game.deserialize_state(clone.serialize()) as replay:self.assertEqual(replay.serialize(),clone.serialize())
            clone.restore(state);self.assertEqual(first,clone.serialize())
        with other.new_initial_state() as foreign:
            with self.assertRaises((ValueError,AssertionError)):state.restore(foreign)
            with self.assertRaises(NativeError):game._call('snapshot',handle=foreign.handle)
        clones=[state.clone() for _ in range(255)]
        with self.assertRaises(NativeError):state.clone()
        for clone in clones:clone.close()
        clone=state.clone(); handle=clone.handle;del clone;gc.collect()
        with self.assertRaises(NativeError):game._call('snapshot',handle=handle)
        malformed=json.loads(first);malformed['checkpoint']['world']['setup']['picked']=[['7'],[],[]]
        with self.assertRaises(NativeError):game.deserialize_state(json.dumps(malformed))
        self.assertEqual(first,state.serialize());state.close()

    def test_unchanged_python_search_matches_reference_and_cancellation(self):
        from brass_ai.search.ismcts import MaskedISMCTS
        from brass_ai.search.common import SearchConfig
        from threading import Event
        native=self.games()[0]
        reference=BrassSpielGame(root=Path(os.environ.get('BRASS_SIMULATION_REFERENCE_ROOT',ROOT)),fast=False);self.addCleanup(reference.close)
        candidates=[];groups=set()
        for row in rows(CAPTURE/'samples.jsonl.gz'):
            view=row['view']; key=(view['context']['era'],view['context']['phase'])
            if view['seat']==view['context']['actor'] and key in {('CanalEra','action'),('RailEra','payment'),('RailEra','action')} and key not in groups:
                candidates.append(row);groups.add(key)
        self.assertEqual(len(candidates),3)
        for row in candidates:
            outputs=[];traces=[]
            for game in (reference,native):
                policy=SimpleNamespace(evaluator=SimpleNamespace(model_id='scripted-l07'),policy_id='scripted-l07',contexts={},utility_remaining={})
                # SimpleNamespace is unhashable; a policy must be an identity key.
                class Policy:
                    pass
                p=Policy();p.__dict__.update(policy.__dict__);p.context_features=lambda view,rng:None
                trace=[]
                def predict(groups):
                    result=[]
                    for _,requests in groups:
                        values=[]
                        for view,choices,_ in requests:
                            trace.append((view,choices))
                            values.append(([1/len(choices)]*len(choices),dict(win_share=[.25]*4)))
                        result.append(values)
                    return result
                search=MaskedISMCTS(game,p,[p],SearchConfig(simulations=4,pending_paths=2),row['seat'],None,random.Random(778),time.monotonic()+120,recall=row['recall'],lanes=[game,game])
                with patch('brass_ai.search.ismcts.predict_groups',predict):stats=search.run(row['view'],row['expected']['choices'])
                self.assertEqual(stats['completed'],4)
                stats.pop('batching');outputs.append(stats);traces.append(trace)
            self.assertEqual(outputs[0],outputs[1]);self.assertEqual(traces[0],traces[1])
            # Cancellation while a generator owns a sampled world releases it;
            # no visit or pending reservation is published.
            cancelled=Event(); search=MaskedISMCTS(native,p,[p],SearchConfig(simulations=4),row['seat'],None,random.Random(3),time.monotonic()+30,recall=row['recall'],cancelled=cancelled)
            def cancel(groups):cancelled.set();return predict(groups)
            with patch('brass_ai.search.ismcts.predict_groups',cancel):stats=search.run(row['view'],row['expected']['choices'])
            self.assertEqual(stats['completed'],0)
            self.assertTrue(all(e.pending==0 for n in search.tree.values() for e in n.edges.values()))
            gc.collect();self.assertEqual(native._handles,set())
