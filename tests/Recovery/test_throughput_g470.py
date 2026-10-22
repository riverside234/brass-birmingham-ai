"""Matched-model batching, ordered process preparation and reviewed continuation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.config import throughput_config,validate_config,update_config,execution_config
from brass_ai.recovery.inference_batches import ModelBatches
from brass_ai.recovery.ppo_preparation import PreparationPool,Preparation
from brass_ai.recovery.updates import Dataset
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature,apply_optimizer_config
from brass_ai.recovery.branch_scheduler import ReadyWork
from test_categorical_kl_guard import probe_fixture
from test_selfplay_input_caches import StreamingBatches


EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0470-20261022T011816523228Z'


class Throughput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_configuration_change(self):
        old=read(EXPORT/'recorded-config.json');new=validate_config(throughput_config(old))
        self.assertEqual(new['update'],dict(old['update'],microbatch_size=512,max_candidate_cells=131072,preparation_workers=4))
        self.assertEqual(new['execution'],dict(old['execution'],padding_ratio=2.,model_cohorts=False))
        self.assertEqual({k:v for k,v in old.items() if k not in ('update','execution')},
                         {k:v for k,v in new.items() if k not in ('update','execution')})
        self.assertFalse(new['value_phase']['enabled'])
        for bad in (True,1,3,6,16,'4'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):update_config(dict(new['update'],preparation_workers=bad))
        for bad in (True,0,4.1,float('nan')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):execution_config(dict(new['execution'],padding_ratio=bad))

    def test_padding_completes_all_original_model_frames_and_bounds_work(self):
        factory=StreamingBatches();scheduler=ModelBatches(64,131072,3.)
        for i,(model,n) in enumerate([('a',4),('a',4),('a',240),('b',772),('b',1)]):
            scheduler.add((i,0),factory.frame(model,n,submitted=99.99,deadline=200))
        seen=[]
        while scheduler:
            rows=scheduler.take(now=100);seen.extend(key for key,_ in rows)
            self.assertEqual(len({f['model'] for _,f in rows}),1)
            ns=[len(f['encoded'].choice_ids) for _,f in rows]
            self.assertLessEqual(len(ns)*max(ns),131072)
            self.assertLessEqual(len(ns)*max(ns),3*sum(ns))
        self.assertEqual(sorted(seen),[(i,0) for i in range(5)])
        self.assertEqual(scheduler.batches,2)

    def test_full_stream_per_model_metrics_and_iteration_deltas(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.config import EXECUTION_V21 as EXECUTION
        from brass_ai.recovery.inference import InferenceService
        from brass_ai.recovery.telemetry import iteration_utilization
        factory=StreamingBatches()
        with InferenceService(dict(EXECUTION,shared_slot_bytes=0),'cpu') as service:
            ids=[service.register(FourSeatNet(ModelConfig.preset('small')).eval()) for _ in range(2)]
            before=copy.deepcopy(service.report())
            frames=[((i,0),factory.frame(ids[i//2],width=3+i)) for i in range(3)]
            result=service._evaluate(frames)
            for key,frame in frames:self.assertEqual(result[key]['model_id'],frame['model'])
            after=service.report();delta=iteration_utilization(before,after,[],1.,1,EXECUTION)
            self.assertEqual(delta['inference_positions'],3);self.assertEqual(delta['model_loads'],2)
            self.assertEqual(delta['inference_batch_histogram'],{'2':1,'1':1})
            for identity,count in zip(ids,(2,1)):
                m=delta['inference_per_model'][identity]
                self.assertEqual(m['positions'],count)
                self.assertEqual(sum(m['queue_wait_histogram'].values()),count)
                self.assertGreater(m['forward_seconds'],0)
            self.assertGreaterEqual(sum(m['forward_seconds'] for m in delta['inference_per_model'].values()),
                                    delta['inference_phase_deltas']['inference.forward']['seconds'])
            self.assertEqual(iteration_utilization(after,service.report(),[],1.,1,EXECUTION)['inference_per_model'],{})

    def test_cohorts_preserve_fork_priority_long_work_and_all_job_contents(self):
        def packet(i,models,branch=False,kind='selfplay'):
            return dict(jobs=[dict(index=i,branching={} if not branch else {'enabled':True},
                policies=[dict(model_id=m) for m in models],provenance=dict(kind=kind))])
        packets=[packet(0,['c','c']),packet(1,['a','b']),packet(2,['c','c']),packet(3,['a','b'],True),packet(4,['b'],kind='legacy')]
        before=copy.deepcopy(packets);schedule=ReadyWork(packets,{'legacy':10.,'selfplay':1.},True)
        order=[schedule.pop()['jobs'][0]['index'] for _ in packets]
        self.assertEqual(order,[3,4,1,0,2]);self.assertEqual(before,packets)
        self.assertFalse(schedule)

    def test_process_packing_exact_order_masks_mixture_no_rng_and_cleanup(self):
        _,ds,_,cfg,_=probe_fixture((3,1,2,3),microbatch=2)
        parts=[[3,1],[0,2],[1,1],[2,0]]
        before=torch.get_rng_state().clone();expected=list(ds.policy_batches(parts,.08))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp,PreparationPool(4) as pool:
            with Preparation(pool,directory=tmp) as prep:
                actual=list(ds.policy_batches(parts,.08,prep));location=prep.path
                self.assertEqual([p for p,_,_,_ in actual],parts)
                for want,got in zip(expected,actual):
                    self.assertTrue(_same_payload(want,got))
                self.assertLessEqual(prep.metrics['peak_inflight'],4)
                self.assertEqual(prep.metrics['batches'],4)
                self.assertEqual(list(location.glob('batch-*.bin')),[])
                ds.encoded[0].state[0]+=.01
                with self.assertRaisesRegex(ValueError,'inputs changed'):
                    list(ds.policy_batches([[0]],.08,prep))
            self.assertFalse(location.exists());self.assertEqual(list(Path(tmp).iterdir()),[])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_early_stop_retires_prefetched_outputs_before_next_scan(self):
        _,ds,_,_,_=probe_fixture()
        with PreparationPool(4) as pool,Preparation(pool) as prep:
            batches=ds.policy_batches([[0]]*8,.08,prep)
            next(batches);batches.close()
            self.assertFalse(prep.futures)
            self.assertEqual(list(prep.path.glob('batch-*.bin')),[])
            expected=list(ds.policy_batches([[0]],.08))
            actual=list(ds.policy_batches([[0]],.08,prep))
            self.assertTrue(_same_payload(expected,actual))
            self.assertEqual(list(prep.path.glob('batch-*.bin')),[])

    def test_worker_rejects_invalid_selectors_and_cleanup_on_failure(self):
        _,ds,_,_,_=probe_fixture()
        # A nonfinite selector is never admitted, including in a spawned worker.
        ds.encoded[0].actions[:]=np.nan
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp,PreparationPool(4) as pool:
            with self.assertRaises(ValueError):
                with Preparation(pool,directory=tmp) as prep:
                    location=prep.path
                    list(ds.policy_batches([[0]],.08,prep))
            self.assertFalse(location.exists())

    def test_value_only_uses_same_descriptors_targets_and_single_candidate(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        _,ds,_,_,_=probe_fixture((3,1,2,3));net=FourSeatNet(ModelConfig.preset('small'))
        before=copy.deepcopy((ds.targets,ds.descriptors,ds.old));parts=[[2,1,0,3]]
        expected=list(ds.value_batches(parts,net))
        with PreparationPool(4) as pool,Preparation(pool) as prep:
            actual=list(ds.value_batches(parts,net,prep))
        self.assertTrue(_same_payload(expected,actual));self.assertEqual(actual[0][2][1].shape[1],1)
        self.assertTrue(_same_payload(before,(ds.targets,ds.descriptors,ds.old)))

    def test_reviewed_g470_helper_preserves_state_and_rejects_other_edits(self):
        from brass_ai.step5.league import implementation_id,write_json
        from brass_ai.recovery.critic_data import floor_config
        from brass_ai.recovery.exploration import anchor_floor_config
        state=read(EXPORT/'state.json');config=floor_config(anchor_floor_config(throughput_config(state['signature']['config']),state))
        incoming=copy.deepcopy(state)
        signature=dict(state['signature'],implementation=implementation_id(),config=config)
        self.assertTrue(check_signature(state,signature,reviewed=True))
        with self.assertRaises(ValueError):check_signature(state,signature)
        bad=copy.deepcopy(signature);bad['config']['exploration']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        module_spec=importlib.util.spec_from_file_location('throughput_helper',SOURCE_ROOT/'scripts/joint7-throughput.py')
        helper=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True)
            (root/'configs/step5').mkdir(parents=True);write_json(run/'state.json',state)
            output='configs/step5/continue.json'
            self.assertFalse(helper.generate(root,run,output,470)['run_modified'])
            helper.generate(root,run,output,470)
            self.assertEqual(read(root/output),config);self.assertEqual(read(run/'state.json'),incoming)
            with self.assertRaises(ValueError):helper.generate(root,run,output,469)
        self.assertEqual(state,incoming)


if __name__=='__main__':unittest.main()
