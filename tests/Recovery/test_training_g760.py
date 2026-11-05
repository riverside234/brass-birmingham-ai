"""Automatic committed reward boundaries and unambiguous CUDA selection."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import entropy_schedule as S,exploration as E
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature,commit_transition,load_resume
from brass_ai.step5.league import implementation_id,write_json
from test_entropy_schedule import commit

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0760-20261105T004442699955Z'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,SOURCE_ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


class AutomaticStages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original=read(BUNDLE/'state.json');cls.old=cls.original['signature']['config']
        cls.new=validate_config(S.automatic_config(cls.old))

    def adopted(self):
        state=copy.deepcopy(self.original)
        state['exploration']=E.restore(self.new,state,state['exploration'])
        state['signature'].update(config=copy.deepcopy(self.new),implementation=implementation_id())
        return state

    def test_adoption_preserves_clock_and_exact_next_stage(self):
        self.assertEqual({k for k in self.new if self.new[k]!=self.old[k]},{'update'})
        self.assertEqual(S.automatic_config(self.new),self.new)
        c=validate_config(S.automatic_next(self.new,self.original))
        self.assertEqual(c['insolvency'],dict(event_cost=.005,max_cost=.05))
        restored=S.restore_run(c,self.original)
        for key in ('progress','held','started_at','reset_count','applied'):
            self.assertEqual(restored[key],self.original['exploration'][S.KEY][key])
        self.assertFalse(S.waiting(restored));self.assertEqual(restored['insolvency_steps_applied'],2)
        receipt=S.automatic_receipt(c,self.original)
        self.assertEqual((receipt['generation'],receipt['progress'],receipt['first_collection_generation']),(760,50,761))
        self.assertEqual(c['search']['reward']['insolvency'],c['insolvency'])
        if 'teacher_phase' in c:
            self.assertEqual(c['teacher_phase']['search']['reward']['insolvency'],c['insolvency'])
        bad=copy.deepcopy(c);bad['update'][S.KEY]['insolvency_steps'][-1]['at_progress']=99
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(c);bad['learning_rate']*=2
        with self.assertRaises(ValueError):S.automatic_receipt(bad,self.original)

    def test_live_boundary_driver_and_stale_launch_config(self):
        from brass_ai.recovery.automatic_training import run as run_stages
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);run=root/'artifacts/run';run.mkdir(parents=True)
            state=self.adopted();write_json(run/'state.json',state)
            calls=[]
            def runner(root,run_path,c,*,iterations,**kwargs):
                state=read(root/run_path/'state.json');calls.append((state['iteration'],c['insolvency'],iterations))
                signature=dict(state['signature'],config=c,implementation=implementation_id())
                if signature!=state['signature']:
                    check_signature(state,signature,reviewed=kwargs['resume_reviewed_source'])
                    state=commit_transition(run,state,signature)
                state['exploration']=E.restore(c,state,state['exploration'])
                reports=[]
                for _ in range(iterations):
                    state,r=commit(state,c);reports.append(r)
                    # Fixture represents a fully committed ordinary iteration.
                    state['history'][-1]=dict(state['history'][-1],generation=state['iteration'])
                    write_json(run/'state.json',state)
                    if S.waiting(state['exploration'][S.KEY]):break
                return dict(completed_iteration=state['iteration'],summaries=reports[-2:])
            r=run_stages(root,'artifacts/run',self.new,runner=runner,iterations=51,resume_reviewed_source=False)
            self.assertEqual(r['completed_iteration'],811)
            self.assertEqual(calls,[(760,dict(event_cost=.005,max_cost=.05),51),
                                   (810,dict(event_cost=.0025,max_cost=.025),1)])
            final=read(run/'state.json');schedule=final['exploration'][S.KEY]
            self.assertEqual((schedule['progress'],schedule['held'],schedule['insolvency_steps_applied']),(100,140,3))
            self.assertEqual(S.follow_committed(self.new,final),final['signature']['config'])
            self.assertEqual(S.automatic_next(final['signature']['config'],final),final['signature']['config'])
            r=run_stages(root,'artifacts/run',self.new,runner=runner,iterations=1,resume_reviewed_source=False)
            self.assertEqual(r['completed_iteration'],812)
            self.assertEqual(len(read(run/'state.json')['source_transitions']),len(self.original['source_transitions'])+2)
            bad=copy.deepcopy(self.new);bad['learning_rate']*=2
            with self.assertRaisesRegex(ValueError,'unrelated'):S.follow_committed(bad,final)

    def test_due_only_from_committed_progress_and_old_manual_contract_preserved(self):
        state=self.adopted();schedule=state['exploration'][S.KEY]
        schedule['progress']=49;S.seal(schedule)
        self.assertEqual(S.automatic_next(self.new,state),self.new)
        held,r=commit(state,self.new,healthy=False)
        self.assertEqual(held['exploration'][S.KEY]['progress'],49)
        self.assertEqual(S.automatic_next(self.new,held),self.new)
        advanced,r=commit(held,self.new)
        self.assertEqual(S.automatic_next(self.new,advanced)['insolvency'],dict(event_cost=.005,max_cost=.05))
        self.assertEqual(S.automatic_next(self.old,self.original),self.old)

    def test_helper_signature_retry_and_pending_rejection(self):
        helper=module('g760_helper','scripts/configure-training-g760.py')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0760.pt';checkpoint.write_bytes(b'committed fixture')
            state=copy.deepcopy(self.original);state['history'][-1].update(path='artifacts/run/generation-0760.pt',sha256=sha(checkpoint))
            write_json(run/'state.json',state);original=(run/'state.json').read_bytes()
            output='configs/step5/g760.json';first=helper.generate(root,'artifacts/run',output,760)
            self.assertEqual(first,helper.generate(root,'artifacts/run',output,760))
            self.assertEqual((run/'state.json').read_bytes(),original)
            c=read(root/output);signature=dict(state['signature'],config=c,implementation=implementation_id())
            committed=commit_transition(run,state,signature)
            self.assertEqual(first,helper.generate(root,'artifacts/run',output,760))
            self.assertEqual(committed['source_transitions'][-1]['automatic_reward_step']['progress'],50)
            with self.assertRaises(ValueError):helper.generate(root,'artifacts/run',output,761)
            (run/'iteration-0760').mkdir();(run/'iteration-0760/partial').write_text('partial')
            with self.assertRaisesRegex(ValueError,'interrupted'):helper.generate(root,'artifacts/run',output,760)

    def test_real_checkpoint_interruption_keeps_weights_Adam_RNG_and_lineage(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.recovery.utility import migrate,reward_spec,calibration_ready
        from brass_ai.recovery.resume import apply_optimizer_config
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);state=copy.deepcopy(self.original);old=copy.deepcopy(self.old);old['model_size']='small'
            state['signature']['config']=old
            model=FourSeatNet(ModelConfig.preset('small'));optimizer=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
            migrate(model,optimizer,reward_spec(old));apply_optimizer_config(optimizer,old)
            p=next(model.parameters());p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
            model.utility_calibration=dict(ready=True,reward_id=digest(reward_spec(old)))
            path=run/'generation-0760.pt'
            identity=save_checkpoint(path,model,optimizer,step=760,training=dict(fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),
                signature=digest(state['signature']),exploration=state['exploration']))
            entry=dict(generation=760,path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity)
            state['history']=[entry];state['population']=[entry];state['evaluation_anchor']=entry
            write_json(run/'state.json',state)
            before=copy.deepcopy((model.state_dict(),optimizer.state_dict(),torch.get_rng_state()))
            c=S.automatic_next(S.automatic_config(old),state);signature=dict(state['signature'],config=c,implementation=implementation_id())
            for retry in range(2):
                loaded,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=True)
                opt=torch.optim.AdamW(loaded.parameters(),lr=old['learning_rate']);resume_optimizer(raw,opt)
                migrate(loaded,opt,reward_spec(c))
                self.assertTrue(_same_payload(before,(loaded.state_dict(),opt.state_dict(),torch.get_rng_state())))
                self.assertFalse(calibration_ready(loaded));self.assertEqual(S.restore_run(c,state)['progress'],50)
                self.assertEqual(changed,retry==0)
                if retry==0:state=commit_transition(run,state,signature)
            self.assertEqual(state['history'],[entry]);self.assertEqual(state['iteration'],760)
            self.assertEqual(len(state['source_transitions']),len(self.original['source_transitions'])+1)


class GPUSelection(unittest.TestCase):
    def test_environment_and_explicit_preimport_selection(self):
        launcher=module('g760_launcher','scripts/step5.py')
        for env,args,expected in [('7',[],'7'),('7,5',[],'7'),('0',['--cuda-visible-devices','7'],'7'),
                ('0',['--cuda-visible-devices=GPU-selected'],'GPU-selected'),
                ('0',['--cuda-visible-devices','7,5','--inference-devices','auto'],'7,5')]:
            with self.subTest(args=args),patch.dict(os.environ,CUDA_VISIBLE_DEVICES=env):
                launcher._limit_default_gpu(args)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],expected)
        # Verify order and inheritance in fresh interpreters, without CUDA.
        code="""import runpy,os,sys,subprocess
m=runpy.run_path('scripts/step5.py')
assert 'torch' not in sys.modules
m['_limit_default_gpu'](['--cuda-visible-devices','7','train'])
assert 'torch' not in sys.modules
print(subprocess.check_output([sys.executable,'-c','import os; print(os.environ[\"CUDA_VISIBLE_DEVICES\"])'],text=True).strip())
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=SOURCE_ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'7')

    def test_physical_token_UUID_reporting_and_wrong_logical_device(self):
        from brass_ai.recovery.telemetry import cuda_selection
        from brass_ai.recovery.inference import validate_inference_devices
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='7',CUDA_DEVICE_ORDER='PCI_BUS_ID'),\
                patch.object(torch.cuda,'current_device',return_value=0),\
                patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(uuid='GPU-seven')) as properties:
            r=cuda_selection('cuda');properties.assert_called_once_with(0)
            self.assertEqual(r,dict(logical_device='cuda:0',visible_devices='7',selected_token='7',uuid='GPU-seven',device_order='PCI_BUS_ID'))
        with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'device_count',return_value=1):
            self.assertEqual(validate_inference_devices(['cuda:0'],'cuda'),['cuda:0'])
            with self.assertRaisesRegex(ValueError,'not visible'):validate_inference_devices(['cuda:7'],'cuda')

    def test_parser_guard_rejects_late_mask_without_preimport_launcher(self):
        from brass_ai.step5 import cli
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'):
            with self.assertRaisesRegex(ValueError,'before importing Torch'):
                cli.main(['--root',str(SOURCE_ROOT),'--cuda-visible-devices','7','preflight'])


if __name__=='__main__':unittest.main()
