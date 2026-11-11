"""Explicit utility bypass still requires measured beta and persists on resume."""
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery import branch_restart as B,teacher_gradient_qualification as Q
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.collector_runtime import recipe
from brass_ai.recovery.teacher_phase import TeacherPhase,readiness,plan_jobs
from brass_ai.recovery.teacher_phase_config import buffer_identity,without_utility_qualification
from brass_ai.recovery.utility import reward_spec
from brass_ai.rust_backend import lab_shadow_config as C,lab_teacher_activation as A
from brass_ai.rust_backend.lab_teacher import validate_job
from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
from brass_ai.step5.league import write_json
from test_g1190_restart import committed,PHASES


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class SkipUtility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.state,before=committed()
        self.shadow=B.seed_recipe(before,self.state,share_target=.05,revision=B.SEED_CHARGED)
        self.skip=dict(self.shadow,teacher_phase=dict(self.shadow['teacher_phase'],skip_utility_qualification=True))
        self.state.update(iteration=1495,signature=dict(self.state['signature'],config=self.shadow),
                          teacher_phase=dict(weight={},readiness=None))
        self.state['exploration']['entropy_schedule']['insolvency']=self.shadow['insolvency']

    def test_flag_changes_only_readiness_and_keeps_batch_reward_beta_and_labels(self):
        validate_config(self.skip);self.assertTrue(C.is_shadow(self.skip))
        on=A.configure(self.skip,.05,revise_stages=True)
        self.assertTrue(A.is_on(on));self.assertTrue(on['teacher_phase']['skip_utility_qualification'])
        self.assertEqual(without_utility_qualification(self.skip),self.shadow)
        self.assertEqual(buffer_identity(self.skip),buffer_identity(self.shadow))
        self.assertEqual(self.skip['teacher_phase']['term']['rows_per_step'],256)
        self.assertEqual(self.skip['insolvency'],dict(event_cost=.0025,max_cost=.025))
        self.assertNotIn('skip_utility_qualification',self.shadow['teacher_phase'])
        signature=lambda c:dict(self.state['signature'],config=c)
        self.assertTrue(C.signature_transition(signature(self.shadow),signature(self.skip)))
        self.assertTrue(C.signature_transition(signature(self.skip),signature(on)))

    def test_skipped_receipt_never_becomes_a_measured_utility_qualification(self):
        for current in (None,dict(ready=False,reward_id='r'),dict(ready=True,reward_id='old')):
            receipt,ready=readiness(None,current,'r',skip=True)
            self.assertTrue(ready);self.assertTrue(receipt['ready'])
            self.assertEqual(receipt['utility_qualification'],'skipped')
            self.assertIs(receipt['utility_calibration'],current)
            self.assertFalse(readiness(receipt,None,'r')[1])
            again,ready=readiness(receipt,None,'r',skip=True)
            self.assertEqual(again,receipt);self.assertTrue(ready)
        current=dict(ready=True,reward_id='r')
        receipt,ready=readiness(None,current,'r')
        self.assertTrue(ready);self.assertIs(receipt,current)

    def test_teacher_collects_immediately_and_on_resume_without_a_utility_fit(self):
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(self.skip)))
        state=dict(iteration=1495,history=[dict(generation=1495,sha256='a'*64)])
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            run=Path(tmp);raw=None
            for utility in (None,dict(ready=False,reward_id=digest(model.config.utility))):
                model.utility_calibration=utility
                phase=TeacherPhase(ROOT,run,self.skip,state,model,raw)
                jobs=phase.begin(state,model,export_hash='a'*64,epoch=state['iteration'],native_source=source_fingerprint())
                self.assertEqual(len(jobs),36)
                self.assertEqual(sum(j['split']=='train' for j in jobs),32)
                for job in jobs:
                    validate_job(job)
                    self.assertEqual(job['readiness']['utility_qualification'],'skipped')
                self.assertIs(model.utility_calibration,utility)
                self.assertTrue(phase.finish(model,{})['ready'])
                payload=phase.payload(state['iteration']+1)
                state=dict(state,iteration=state['iteration']+1,teacher_phase=payload,teacher_seed_cursor=phase.cursor)
                raw=dict(training=dict(teacher_phase=payload))
        self.assertEqual(plan_jobs(self.shadow,self.state,model_id='m',export_hash='a'*64,epoch=1495,ready=False)[0],[])

    def test_activation_still_requires_full_measured_beta(self):
        on=A.configure(self.skip,.05,revise_stages=True)
        state=self.state
        state['exploration']['entropy_schedule']['last_generation']=state['exploration']['generation']=1496
        state['iteration']=1496
        with self.assertRaisesRegex(ValueError,'qualified full shadow'):A.authorize(state,on)
        term=SimpleNamespace(config=self.skip['teacher_phase'],mode='shadow',generation=1495,qualification=None,
            qualification_context=dict(tables={'train':32,'heldout':4},labels_by_phase={p:30 for p in PHASES},
                                       model_id='m1495',export_hash='h1495'),beta=1.,calibrated=False,
            steps=[dict(raw_share=v,cosine=.01,actor_norm=1.,teacher_norm=v,share=v) for v in (4.,5.,6.)])
        weight={};Q.finish(term,weight)
        state['teacher_phase']['weight']=weight
        state['history']=[dict(generation=g,model_id=f'm{g}',sha256=f'h{g}') for g in (1495,1496)]
        self.assertTrue(A.authorize(state,on)['passed'])
        with self.assertRaisesRegex(ValueError,'utility readiness'):A.authorize(state,A.configure(self.shadow,.05,revise_stages=True))

    def test_command_runs_beta_first_then_remaining_training_and_keeps_skip(self):
        driver=script('continue-rust-g1491');shared=driver.shared();calls=[]
        self.state['iteration']=1491
        def continuation(args,**kw):
            calls.append(kw);self.assertTrue(args.skip_utility_qualification)
            self.assertEqual(args.run,'artifacts/step5/joint7-v11-2-g1190-restart-3')
            config=self.skip if kw['mode']=='shadow' else A.configure(self.skip,.05,revise_stages=True)
            return dict(self.state,iteration=kw['expected']+kw['iterations'],signature=dict(self.state['signature'],config=config),
                        teacher_phase=dict(readiness=None,weight=dict(gradient_qualification=dict(passed=True))))
        argv=['continue-rust-g1190-restart-2.py','--run','artifacts/step5/joint7-v11-2-g1190-restart-3',
              '--expected-generation','1491','--iterations','120',
              '--share-target','0.05','--skip-utility-qualification']
        with patch.object(driver,'shared',return_value=shared),patch.object(driver,'state_of',return_value=self.state),\
                patch.object(driver,'ensure_cpu'),patch.object(shared,'continuation',side_effect=continuation),\
                patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()) as out:
            driver.main(run='artifacts/step5/joint7-v11-2-g1190-restart-2',teacher_fast=96,share_target_required=True)
        self.assertEqual([(c['mode'],c['expected'],c['iterations']) for c in calls],[('shadow',1491,1),('on',1492,119)])
        self.assertNotIn('re-fits the utility',out.getvalue())
        saved=dict(self.state,signature=dict(self.state['signature'],config=self.skip))
        self.assertTrue(driver.ready(saved));self.assertEqual(driver.calibrated(saved),(False,1))
        self.assertFalse(driver.ready(self.state));self.assertEqual(driver.calibrated(self.state),(False,2))

    def test_prepare_command_selects_original_g1190_weights_in_the_fresh_run(self):
        prepare=script('prepare-rust-g1190-restart-2')
        argv=['prepare-rust-g1190-restart-2.py','--run','artifacts/step5/joint7-v11-2-g1190-restart-3',
              '--share-target','0.05']
        with patch.object(sys,'argv',argv),patch.object(B,'prepare',return_value={}) as build,redirect_stdout(io.StringIO()):
            prepare.main()
        args,kwargs=build.call_args
        self.assertEqual(args[3],'artifacts/step5/joint7-v11-2-g1190-restart-3')
        self.assertEqual(kwargs['share_target'],.05)
        self.assertEqual(kwargs['rollback']['learner_generation'],1190)
        self.assertEqual(kwargs['rollback']['revision'],B.SEED_CHARGED)
        self.assertEqual(kwargs['weights'],prepare.SOURCE+'/population/weights-d872ea925d6643ee3a3eeec99a52d16c6a78c2de1f757617f1aaaa47d175f864-aux-dd8dcbfaf5350001.pt')

    def test_prepare_writes_skip_and_keeps_it_without_repeating_the_flag(self):
        shared=script('continue-rust-shadow')
        # A short root keeps this generated config path below Windows MAX_PATH.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'artifacts/step5/restart';run.mkdir(parents=True)
            checkpoint=run/'generation-1495.pt';checkpoint.write_bytes(b'committed checkpoint fixture')
            state=dict(self.state,history=[dict(generation=1495,path=checkpoint.relative_to(root).as_posix(),sha256=sha(checkpoint))])
            write_json(run/'state.json',state)
            with patch('brass_ai.rust_backend.lab_backend.validate_state'),patch('brass_ai.rust_backend.lab_shadow.teacher_build'):
                _,path,remaining=shared.prepare(root,run,1495,1,.05,mode='shadow',revision='g1491',
                                                inference=None,skip_utility_qualification=True)
                prepared=read(path);validate_config(prepared)
                self.assertEqual(remaining,1);self.assertTrue(prepared['teacher_phase']['skip_utility_qualification'])
                self.assertEqual(prepared['teacher_phase']['term']['shadow_iterations'],1)
                state=dict(state,signature=dict(state['signature'],config=recipe(prepared)))
                write_json(run/'state.json',state)
                _,again,_=shared.prepare(root,run,1495,1,.05,mode='shadow',revision='g1491',inference=None)
                self.assertEqual(path,again);self.assertEqual(read(again),prepared)


if __name__=='__main__':unittest.main()
