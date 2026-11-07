"""Short concurrent failure regression: the initiating lane must remain visible."""
import copy
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, Mock, patch

from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest, SOURCE_ROOT
from brass_ai.recovery.teacher_phase import plan_jobs
from brass_ai.recovery.search_value_config import adopt
from brass_ai.recovery.search_value_rows import complete_game
from brass_ai.rust_backend import lab_teacher as teacher
from test_b025 import model_fixture, config
from test_b045 import value_fixture


class CollectionFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_later_lane_failure_is_not_hidden_by_earlier_lane_cancellation(self):
        model=model_fixture().eval()
        jobs=[value_fixture(model,split=s)[1] for s in ('train','heldout')]
        entered=threading.Event();cancel=threading.Event()
        endpoints=[Mock(),Mock()]
        endpoints[0].close.side_effect=ValueError('endpoint cleanup failure')
        service=MagicMock();service.device='cpu'
        service.channels.return_value.acquire.side_effect=endpoints
        cal=dict(contract=teacher.CONTRACT,source=teacher.source_fingerprint(),core_source='core',
            model_id=model_id(model),export_hash=jobs[0]['expected']['export_hash'],
            reward_id=jobs[0]['expected']['reward_id'],device='cpu',profile=digest(jobs[0]['search']),
            samples=[{}],estimate_ns=1000)
        class Worker:
            def __init__(self,executable,request,*args):
                self.split=request['job']['split'];self.failure=None
            def collect(self,*args,**kwargs):
                if self.split=='train':
                    entered.set()
                    if not cancel.wait(3):raise AssertionError('Sibling did not cancel')
                    raise ValueError('secondary cancellation')
                if not entered.wait(3):raise AssertionError('Earlier lane did not start')
                self.failure=dict(stderr='native original failure',verified_turns=12)
                raise ValueError('initiating native error')
        collector=teacher.TeacherCollector('binary','core',[],service,cal,lanes=2,cancel=cancel,model=model)
        with patch.object(teacher,'Pipe'),patch.object(teacher,'Worker',Worker):
            with self.assertRaisesRegex(ValueError,'^initiating native error$'):
                collector.collect(jobs)
        self.assertEqual(len(collector.failures),2)
        first=collector.failures[0]
        self.assertTrue(first['primary']);self.assertEqual(first['split'],'heldout')
        self.assertEqual(first['worker']['verified_turns'],12)
        self.assertIn('initiating native error',first['traceback'])
        self.assertEqual(first['endpoint_cleanup_errors'],['endpoint cleanup failure'])
        self.assertFalse(collector.failures[1]['primary'])
        for endpoint in endpoints:endpoint.close.assert_called_once()

    def test_cleanup_failure_does_not_hide_original_worker_failure(self):
        worker=teacher.Worker.__new__(teacher.Worker)
        worker.cancel=threading.Event();worker.events=Mock();worker.err_reader=Mock()
        worker.stderr=b'native diagnostic'
        worker.events.get.return_value=ValueError('initiating wire failure')
        worker.close=Mock(side_effect=ValueError('cleanup failure'))
        with self.assertRaisesRegex(ValueError,'initiating wire failure'):
            worker.collect({},'core')
        self.assertEqual(worker.failure['close_error'],'cleanup failure')
        self.assertIn('native diagnostic',worker.failure['stderr'])

    def test_value_seal_cache_is_detached_and_cold_restore_matches(self):
        from brass_ai.recovery.search_value_rows import ValueBuffer
        model=model_fixture();row,_,_=value_fixture(model)
        original=copy.deepcopy(row)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            buffer=ValueBuffer(tmp,row['reward_id']);buffer.publish(10,'train',[row])
            row['state'][0]+=1;row['returns'][0][0]+=1;row['job']['model_id']='changed'
            self.assertEqual(buffer.rows(10),[original])
            restored=ValueBuffer(tmp,original['reward_id'],receipts=buffer.receipts)
            self.assertEqual(restored.rows(10),buffer.rows(10))

    def test_value_sealing_charges_each_table_and_still_rejects_an_overrun(self):
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        import numpy as np
        model=model_fixture();c=adopt(config())
        row,job,_=value_fixture(model,split='heldout')
        second=copy.deepcopy(job);second['expected'].update(job_id='second',game_id='second-game')
        tables=[dict(job_id=j['expected']['job_id'],value_rows=[],result={},final_spent=[0.]*4,
            metrics=dict(turns=[dict(profile='fast',post_search_ns=20_000_000)])) for j in (job,second)]
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            phase=SearchValuePhase(tmp,model,c['teacher_phase']['search_value'],row['reward_id'])
            with patch('brass_ai.recovery.search_value_phase.complete_game',side_effect=lambda *a,**k:[dict(row)]), \
                 patch('brass_ai.learning.reward.terminal_utility',return_value=np.zeros(4)), \
                 patch.object(phase.buffer,'publish') as publish:
                # Total batch cost exceeds one turn, but each table stays below
                # its own 500ms cap including the already measured 20ms post.
                with patch('time.monotonic',side_effect=[0.,0.,.32,.32,.64,.64]):
                    phase.admit(tables,[job,second],10)
                self.assertEqual(len(publish.call_args_list),2)
                self.assertAlmostEqual(phase.report['value_admission_seconds'],.64)
                with patch('time.monotonic',side_effect=[0.,0.,.49]):
                    with self.assertRaisesRegex(ValueError,'Value sealing exceeded.*seal=0.490000s'):
                        phase.admit(tables[:1],[job],10)


@unittest.skipUnless(os.environ.get('BRASS_B045_BINARY') and os.environ.get('BRASS_B045_BOOTSTRAP'),
                     'Explicit detached native binary/bootstrap required')
class NativeScriptedTables(unittest.TestCase):
    def test_two_scripted_tables_cover_all_64_turns_without_training(self):
        """Fast wire/rules coverage; scripted values are never training evidence."""
        model=model_fixture().eval();c=adopt(config());c['teacher_phase']['mode']='shadow'
        jobs,_=plan_jobs(c,dict(iteration=1250),model_id=model_id(model),export_hash='a'*64,
            epoch=0,ready=True,native_source=teacher.source_fingerprint())
        jobs=[j for j in jobs if j['split']=='heldout'][:2]
        for job in jobs:
            job['shadow_probe']=True
            job['readiness']=dict(ready=True,reward_id=job['expected']['reward_id'])
        class ScriptedClient:
            wait_seconds=0.
            def __init__(self,*args):pass
            def evaluate(self,meta,payload):
                from brass_ai.rust_backend.lab_teacher_inference import decode
                decode(meta,payload)  # Real tensor dimensions, finite inputs and seat mapping.
                b=meta['binding']
                return dict(binding=b,status='ok',search_value=b['search_value'],
                    logits=[20. if x.startswith('pass|') else 0. for x in b['legal_ids']],
                    utility=[.25]*4,search_utility=[.25]*4,reward_id=b['context']['reward_id'])
        service=MagicMock();service.device='cpu';service.channels.return_value.acquire.side_effect=[Mock(),Mock()]
        core=os.environ['BRASS_B045_CORE']
        calibration=dict(contract=teacher.CONTRACT,source=teacher.source_fingerprint(),core_source=core,
            model_id=model_id(model),export_hash='a'*64,reward_id=jobs[0]['expected']['reward_id'],device='cpu',
            # Zero-count scripted games exercise the full 64-turn protocol and
            # final value sealing quickly. Native unit tests cover real backups.
            profile=digest(jobs[0]['search']),samples=[{'diagnostic_scripted':True}],estimate_ns=4_000_000_000)
        cancel=threading.Event();timer=threading.Timer(30,cancel.set);timer.start()
        collector=teacher.TeacherCollector(os.environ['BRASS_B045_BINARY'],core,
            json.loads(os.environ['BRASS_B045_BOOTSTRAP']),service,calibration,lanes=2,cancel=cancel,model=model)
        try:
            with patch.object(teacher,'Client',ScriptedClient):tables=collector.collect(jobs)
            for table,job in zip(tables,jobs,strict=True):
                accepted,_=collector.verifier.take(table,job,model)
                self.assertEqual(accepted,[])
                rows=complete_game(table['value_rows'],table['result'],table['final_spent'],job)
                self.assertEqual(len(rows),64)
                self.assertEqual(sum(r['profile']=='full' for r in rows),16)
                for turn in table['metrics']['turns']:
                    budget=turn['budget'];full=turn['profile']=='full'
                    self.assertEqual(budget['work_deadline_ns']-budget['started_ns'],4_000_000_000 if full else 800_000_000)
                    self.assertEqual(budget['deadline_ns']-budget['work_deadline_ns'],1_000_000_000 if full else 500_000_000)
            from brass_ai.recovery.search_value_phase import SearchValuePhase
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                phase=SearchValuePhase(tmp,model,c['teacher_phase']['search_value'],jobs[0]['expected']['reward_id'])
                phase.admit(tables,jobs,1250)
                self.assertEqual(len(phase.buffer.rows(1250,'heldout')),128)
                self.assertEqual(phase.buffer.rows(1250,'train'),[])
                self.assertEqual(len(phase.buffer.receipts),2)
                print('Scripted value sealing:',json.dumps(phase.report))
                restored=SearchValuePhase(tmp,model,c['teacher_phase']['search_value'],
                    jobs[0]['expected']['reward_id'],saved=phase.payload())
                self.assertEqual(restored.buffer.rows(1250,'heldout'),phase.buffer.rows(1250,'heldout'))
        except BaseException:
            print(json.dumps([dict(error=f['error'],primary=f['primary'],job_id=f['job_id'],
                verified_turns=(f['worker'] or {}).get('verified_turns'),stderr=(f['worker'] or {}).get('stderr'))
                for f in collector.failures]))
            raise
        finally:timer.cancel()


if __name__=='__main__':unittest.main()
