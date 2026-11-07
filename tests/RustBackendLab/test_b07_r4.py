"""Short R4 feature handoff, RNG, reuse, V4 routing and comparison regressions."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from collections import Counter
from types import SimpleNamespace
import numpy as np
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery.replay import check_game, CheckedGame
from brass_ai.rust_backend import lab_admission as A, lab_pool as P, lab_features as F
from brass_ai.rust_backend.lab_collection import route
from test_b03 import family_fixture


class R4(unittest.TestCase):
    def temporary(self):
        t=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(t.cleanup);return Path(t.name)

    def test_actor_features_survive_native_and_old_reference_paths_without_reencoding(self):
        for kind in ('selfplay','historical','legacy'):
            family,binding,config=family_fixture(kind=kind)
            part=family['parent'];part['game']=check_game(part['game']);part['game'].actor_validated=True
            out=self.temporary();publisher=P.Publisher(out,family['producer'],config,'ordinary')
            with patch('brass_ai.recovery.encoding.PreparedEncoder.encode',side_effect=AssertionError('Python re-encoding')):
                handle=P.part_write(out/'parts',part)
                restored=P.part_read(out/'parts',handle)
                self.assertIsInstance(restored['game'],CheckedGame)
                record=publisher.publish(family['frozen_job'],handle,None)
                game=A.read(record['path'],record['native_admission'])
                cold=A.read(record['path'],record['native_admission'],reuse_validation=False)
                for original,new,again in zip(part['game']['records'],game['records'],cold['records'],strict=True):
                    before=part['game'].encoded(original,None)
                    for value,row in ((game,new),(cold,again)):
                        after=value.encoded(row,None)
                        for name in ('state','actions','stable_to_relative'):
                            np.testing.assert_array_equal(getattr(before,name),getattr(after,name))
                self.assertTrue(hasattr(game,'critic_table'))
                self.assertFalse(list((out/'parts').iterdir()))

    def test_feature_handoff_rejects_tamper_context_reorder_and_unsealed_sidecar(self):
        family,_,_=family_fixture();part=family['parent'];part['game']=check_game(part['game'])
        out=self.temporary();handle=P.part_write(out,part)
        data=Path(handle['path'])
        raw=data.read_bytes();data.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaisesRegex(ValueError,'changed'):P.part_read(out,handle)
        data.write_bytes(raw)
        path=out/'cache';receipt=F.write(path,[part['game']],'a'*64)
        changed=copy.deepcopy(part['game']);changed['records'][0]['style'][0]=1.
        with self.assertRaisesRegex(ValueError,'binding'):F.read(path,[changed],receipt,'a'*64)
        self.assertIsNone(F.read(path,[part['game']],None,'a'*64))
        changed=copy.deepcopy(part['game']);changed['records'][0]['chosen']=(changed['records'][0]['chosen']+1)%3
        with self.assertRaisesRegex(ValueError,'record changed'):F.write(out/'changed',[changed],'a'*64)

    def test_mixed_v4_uses_rust_and_keeps_coverage_stratum(self):
        from rust_lab_b06_metrics import choose_jobs
        a,_,_=family_fixture(kind='historical');b,_,_=family_fixture(kind='legacy')
        b['frozen_job']['provenance']['kind']='historical'
        jobs=[a['frozen_job'],b['frozen_job']];before=copy.deepcopy(jobs)
        self.assertTrue(all(route(j)==('rust',None) for j in jobs))
        self.assertEqual(len(choose_jobs(jobs)),2);self.assertEqual(jobs,before)
        jobs[1]['policies'][3]['search_free']=False
        self.assertEqual(route(jobs[1])[0],'reference')

    def test_lab_probability_roundoff_is_bounded_but_actions_and_admission_remain_strict(self):
        from rust_lab_b06_metrics import compare_games
        family,binding,_=family_fixture();game=A.convert(family,binding);changed=copy.deepcopy(game)
        # Exact reported lab pair scale (difference 4.68192948e-7, relative 3.22833888e-6).
        game['records'][0]['behavior']['probability']=.145026
        changed['records'][0]['behavior']['probability']=.145026+4.68192948e-7
        self.assertGreater(compare_games([game],[changed])['probability_comparison']['max_absolute'],0.)
        changed['records'][0]['behavior']['probability']+=1e-4
        with self.assertRaises(AssertionError):compare_games([game],[changed])
        changed=copy.deepcopy(game);changed['records'][0]['chosen']=(changed['records'][0]['chosen']+1)%3
        with self.assertRaises(ValueError):compare_games([game],[changed])
        bad=copy.deepcopy(family);bad['parent']['game']['records'][0]['behavior']['probability']+=1e-7
        bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'denominator'):A.convert(bad,binding)

    def test_session_reuses_policy_process_and_clears_each_seat_and_failed_game(self):
        from brass_ai.rust_backend import lab_session as S
        from brass_ai.step5.policy import NeuralPolicy
        family,_,_=family_fixture();job=family['frozen_job']
        made=[]
        def create(spec,*args):
            evaluator=SimpleNamespace(model_id=spec['model_id'],predict=Mock())
            p=NeuralPolicy(None,evaluator=evaluator,focus=spec['focus'],exploration=spec['exploration'],
                exploration_version=3,opening=spec['opening']);p.close=Mock();made.append(p);return p
        lane=Mock(closed=False,token=None,metrics=Counter())
        with patch.object(S,'Collector',return_value=lane) as spawn,patch.object(S,'make_policy',side_effect=create):
            with S.OrdinarySession(SOURCE_ROOT,'binary',None,{}) as session:
                import random
                old_rng=random.Random(1)
                for index in range(2):
                    costs=Counter()
                    with session.game(job,costs) as (current,policies):
                        self.assertEqual(len(policies[0].contexts),0)
                        self.assertEqual(policies[0].context_counter,0)
                        policies[0].contexts[old_rng]=([1.]*4,[0.]*8)
                        policies[0].context_counter=99
                    self.assertEqual(costs['policy_starts' if index==0 else 'policy_reuses'],4)
                self.assertEqual(spawn.call_count,1);self.assertEqual(len(made),4)
                with self.assertRaisesRegex(ValueError,'failed'):
                    with session.game(job,Counter()):raise ValueError('failed')
                self.assertIsNone(session.lane);self.assertFalse(session.policies)
                self.assertTrue(all(p.close.call_count==1 for p in made))

    def test_original_v4_plan_queue_is_preserved_within_game_and_reset_between_games(self):
        from brass_ai.rust_backend import lab_session as S
        from brass_ai.learning import legacy_policy as L
        import random
        worker=Mock();worker.call.return_value=dict(selected=dict(actions=[['loan','finish']],
            intent=dict(focus='Brewery',age=1)),nodes=3)
        opponent=SimpleNamespace(worker=worker,profile=dict(milliseconds=1000,work_limit=100,tactic='balanced'),
                                 fixed_work=True,close=Mock())
        spec=dict(kind='legacy',policy_id='frozen-v4',search_free=True)
        job=dict(policies=[spec]*4);lane=Mock(closed=False,token=None,metrics=Counter())
        with patch.object(L,'FrozenOpponent',return_value=opponent),patch.object(S,'Collector',return_value=lane):
            with S.OrdinarySession(SOURCE_ROOT,'binary',None,{}) as session:
                for _ in range(2):
                    with session.game(job,Counter()) as (_,policies):
                        p=policies[0];rng=random.Random(12)
                        self.assertFalse(p.contexts)
                        view=dict(seat=0,context=dict(era='CanalEra',round=2,turn=1,revision=0))
                        choices=[dict(id='loan'),dict(id='pass')]
                        self.assertEqual(p.select_batch([(view,choices,rng)])[0],(0,1.,None))
                        self.assertEqual(list(p.contexts[rng]['queue']),['finish'])
                        intent=p.context_features(view,rng)
                        view['context']['revision']=1
                        p.select_batch([(view,[dict(id='finish')],rng)])
                        self.assertEqual(p.context_features(view,rng),intent)
                        self.assertFalse(p.contexts[rng]['queue'])
                self.assertEqual(worker.call.call_count,2)  # one plan per game, no forced-stage replan

    def test_cancellation_retires_inflight_private_v4_process_and_session(self):
        from brass_ai.rust_backend import lab_session as S
        import subprocess,sys,threading
        process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        def close():
            if process.poll() is None:process.kill()
            process.wait(timeout=2)
        self.addCleanup(close)
        policy=SimpleNamespace(contexts={},opponent=SimpleNamespace(worker=SimpleNamespace(process=process)),close=close)
        lane=Mock(closed=False,token=None,metrics=Counter());event=threading.Event()
        with patch.object(S,'Collector',return_value=lane),patch.object(S,'make_policy',return_value=policy):
            with S.OrdinarySession(SOURCE_ROOT,'binary',None,{}) as session:
                with self.assertRaisesRegex(ValueError,'canceled'):
                    with session.game(dict(policies=[dict(kind='legacy',policy_id='v4')]*4),Counter(),event):
                        event.set();process.wait(timeout=2)
                self.assertIsNone(session.lane);self.assertFalse(session.policies)
                lane.close.assert_called()


@unittest.skipUnless(os.environ.get('BRASS_B07_COLLECTOR'),'Set native collector for bounded real transport checks')
class NativeR4(unittest.TestCase):
    def test_complete_collect_lane_keeps_features_and_reuses_real_policies(self):
        from brass_ai.rust_backend.lab_session import OrdinarySession
        from brass_ai.rust_backend.lab_collection import collect_lane
        from brass_ai.recovery.execution import neural_spec
        from brass_ai.recovery.inference import RemoteEvaluator
        import threading
        root=json.loads(Path(os.environ['BRASS_B07_FIXTURES']).read_text())['roots'][-1]
        origin=dict(state=root['expected']['state'],rail_keys=list(range(64)))
        binary=os.environ['BRASS_B07_COLLECTOR'];endpoint=SimpleNamespace(lock=threading.RLock())
        execution=dict(max_decisions=100,batch_size=32,max_candidate_cells=8192)
        spec=neural_spec('f'*64,0.,'neutral',version=3)
        job=dict(index=0,seed=19,sampling_seed=31,policies=[spec]*4,provenance={})
        def predict(evaluator,requests):
            out=[]
            for r in requests:
                evaluator.encoder.encode(r.view,r.choices,r.style,r.intent)
                raw=[0.]*len(r.choices);raw[next((i for i,c in enumerate(r.choices) if c['kind']=='pass'),0)]=1.
                out.append(dict(status='ok',protocol='four-seat-inference/1.0.0',request_id=r.request_id,
                    model_id=r.expected_model,revision=r.view['context']['revision'],observer=r.view['seat'],
                    choice_ids=[c['id'] for c in r.choices],input_digest=r.input_digest,probabilities=raw,
                    win_share=[.25]*4,placement=[2.5]*4,vp=[0.]*4))
            return out
        with patch.object(RemoteEvaluator,'predict',predict),OrdinarySession(SOURCE_ROOT,binary,endpoint,execution) as session:
            for repeat in range(2):
                part=collect_lane(SOURCE_ROOT,binary,None,job,endpoint,execution,origin=origin,session=session)
                game=part['game'];self.assertIsInstance(game,CheckedGame);self.assertTrue(game.actor_validated)
                self.assertEqual(len(game['records']),2)
                for row in game['records']:
                    self.assertIsNotNone(game.encoded(row,None))
                self.assertTrue(any(len(r['choices'])==1 for r in game['records']))
                self.assertEqual(part['timings']['policy_starts' if repeat==0 else 'policy_reuses'],4)
                self.assertIsNone(session.lane.token)

    def test_released_native_process_reuses_no_state_token_or_features(self):
        from brass_ai.rust_backend.lab_transport import Collector,DecisionEncoder
        roots=json.loads(Path(os.environ['BRASS_B07_FIXTURES']).read_text())['roots']
        with Collector(os.environ['BRASS_B07_COLLECTOR']) as lane:
            pid=lane.process.pid;tokens=[];encoder=DecisionEncoder(lane,'same-model')
            for root in (roots[0],roots[-1],roots[0]):
                lane.begin(root['id'],dict(state=root['expected']['state'],rail_keys=list(range(64))),5000)
                tokens.append(lane.token)
                d=lane.current['decision'];self.assertEqual(d['view'],root['view'])
                e=encoder.encode(d['view'],d['choices'],[0.]*4,[0.]*8)
                self.assertEqual(e.choice_ids,tuple(c['id'] for c in d['choices']))
                lane.release();self.assertEqual(lane.process.pid,pid)
            self.assertEqual(len(set(tokens)),3)

    def test_exported_training_arrays_cover_forced_rows_and_destructive_drain(self):
        from brass_ai.rust_backend.lab_transport import Collector
        from brass_ai.learning.features import Encoder
        from rust_lab_b07_transport import walk,LegacyEncoder
        root=json.loads(Path(os.environ['BRASS_B07_FIXTURES']).read_text())['roots'][-1]
        origin=dict(state=root['expected']['state'],rail_keys=list(range(64)));binary=os.environ['BRASS_B07_COLLECTOR']
        trace=walk(binary,origin,limit=96)['trace']
        with Collector(binary,batch_forced=False) as lane:
            lane.begin('retained',origin,1000)
            for row,evidence in trace:
                if len(row['choices'])>1:LegacyEncoder(lane,evidence['model_id']).encode(row['view'],row['choices'],row['style'],row['intent'])
                lane.append(row,evidence)
            receipt=lane.call('finish');rows=[json.loads(bytes(r)) for r,e in lane.records(receipt,drain=True)]
            arrays=list(lane.training_features(receipt,rows,drain=True));encoder=Encoder()
            self.assertEqual(len(arrays),len(rows))
            for r,e in zip(rows,arrays,strict=True):
                py=encoder.encode(r['view'],r['choices'],r['style'],r['intent'])
                for name in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(e,name),getattr(py,name))
            self.assertEqual(lane.call('finish')['records_sha256'],receipt['records_sha256'])
            with self.assertRaisesRegex(ValueError,'cursor'):lane.call('drain_records',token=lane.token,start=0)
            self.assertTrue(lane.closed)
