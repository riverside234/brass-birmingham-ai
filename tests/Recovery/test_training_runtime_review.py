"""Failure wakeup and flexible lab-GPU continuation regressions."""
import copy
import multiprocessing
import time
import unittest
from unittest.mock import patch

import torch

from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.recovery.resources import CPUAdmission
from brass_ai.recovery.telemetry import Timings


def wait_for_admission(admission,messages):
    messages.put('waiting')
    try:
        with admission.acquire(1,Timings()):messages.put('admitted')
    except ValueError as error:messages.put(str(error))


class RuntimeReviewTests(unittest.TestCase):
    def test_abort_wakes_spawned_reader_and_rejects_future_work(self):
        mp=multiprocessing.get_context('spawn');admission=CPUAdmission(mp,2,1)
        messages=mp.Queue();process=mp.Process(target=wait_for_admission,args=(admission,messages))
        try:
            with admission.acquire(1,Timings()):
                process.start();self.assertEqual(messages.get(timeout=30),'waiting')
                end=time.monotonic()+5
                while admission.report()['queued']!=1 and time.monotonic()<end:time.sleep(.01)
                self.assertEqual(admission.report()['queued'],1)
                admission.abort()
                self.assertEqual(messages.get(timeout=5),'CPU admission aborted')
                process.join(timeout=5);self.assertEqual(process.exitcode,0)
            self.assertEqual(admission.report()['active'],0)
            self.assertTrue(admission.report()['aborted'])
            with self.assertRaisesRegex(ValueError,'CPU admission aborted'):
                with admission.acquire(1,Timings()):self.fail('Aborted gate reused')
            admission.abort()  # Multiple failing pools can signal safely.
        finally:
            admission.abort()
            if process.is_alive():process.terminate();process.join(timeout=5)
            messages.cancel_join_thread();messages.close()

    def fixture(self,count):
        parameter=torch.nn.Parameter(torch.ones(2))
        optimizer=torch.optim.AdamW([parameter],lr=1e-5)
        parameter.square().sum().backward();optimizer.step()
        raw=dict(optimizer=copy.deepcopy(optimizer.state_dict()),rng=torch.get_rng_state().clone(),
                 cuda_rng=[torch.tensor([i],dtype=torch.uint8) for i in range(count)])
        return raw,torch.optim.AdamW([parameter],lr=.01)

    def test_step5_can_shrink_or_expand_visible_gpu_set_without_losing_adam(self):
        for saved,visible in ((2,6),(6,2),(4,4)):
            with self.subTest(saved=saved,visible=visible):
                raw,optimizer=self.fixture(saved)
                with patch.object(torch.cuda,'is_available',return_value=True),\
                     patch.object(torch.cuda,'device_count',return_value=visible),\
                     patch.object(torch.cuda,'set_rng_state') as restore:
                    report=resume_optimizer(raw,optimizer,allow_cuda_topology_change=True)
                self.assertEqual(report,dict(saved_devices=saved,visible_devices=visible,restored_devices=min(saved,visible)))
                self.assertEqual(restore.call_count,min(saved,visible))
                for i,call in enumerate(restore.call_args_list):
                    self.assertTrue(torch.equal(call.args[0],raw['cuda_rng'][i]));self.assertEqual(call.args[1],i)
                self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
                self.assertEqual(optimizer.state_dict()['param_groups'],raw['optimizer']['param_groups'])
                for key,values in raw['optimizer']['state'].items():
                    for name,value in values.items():torch.testing.assert_close(value,optimizer.state_dict()['state'][key][name],rtol=0,atol=0)

    def test_other_checkpoint_callers_retain_strict_topology_default(self):
        raw,optimizer=self.fixture(6)
        with patch.object(torch.cuda,'is_available',return_value=True),\
             patch.object(torch.cuda,'device_count',return_value=2),\
             patch.object(torch.cuda,'set_rng_state') as restore:
            with self.assertRaisesRegex(ValueError,'topology differs'):resume_optimizer(raw,optimizer)
            restore.assert_not_called()

    def test_cpu_continuation_does_not_try_to_restore_invisible_cuda_streams(self):
        raw,optimizer=self.fixture(6)
        with patch.object(torch.cuda,'is_available',return_value=False),\
             patch.object(torch.cuda,'set_rng_state') as restore:
            self.assertIsNone(resume_optimizer(raw,optimizer,allow_cuda_topology_change=True))
            restore.assert_not_called()

    def test_immediate_previous_selfplay_checkpoint_resumes_without_config_changes(self):
        from test_teacher_work_resume import TeacherWorkResumeTests
        from test_throughput_resume import signatures
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.recovery.resume import load_resume,commit_transition
        fixture=TeacherWorkResumeTests();self.addCleanup(fixture.doCleanups)
        def previous_signatures():
            _,new=signatures();old=copy.deepcopy(new)
            old['implementation']='b883eac48cd8e01b1e197ec842ab33461da19aac82082cbe9a8d5466935c080d'
            return old,new
        fixture.signatures=previous_signatures
        run,state,new,original,optimizer=fixture.fixture()
        path=run/'generation-0070.pt';before=path.read_bytes()
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,new,reviewed=True)
        self.assertTrue(changed)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items()))
        restored=torch.optim.AdamW(model.parameters(),lr=new['config']['learning_rate'])
        resume_optimizer(raw,restored,allow_cuda_topology_change=True)
        for key,values in optimizer.state_dict()['state'].items():
            for name,value in values.items():torch.testing.assert_close(value,restored.state_dict()['state'][key][name],rtol=0,atol=0)
        updated=commit_transition(run,state,new)
        self.assertEqual(updated['source_transitions'][-1]['config_changes'],{})
        for key,value in state.items():
            if key!='signature':self.assertEqual(updated[key],value,key)
        self.assertEqual(path.read_bytes(),before)
        self.assertFalse(load_resume(SOURCE_ROOT,run,updated,new)[2])

if __name__=='__main__':unittest.main()
