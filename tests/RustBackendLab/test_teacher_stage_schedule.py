"""Unconditional stage scheduling and real small checkpoint/Adam regressions."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.search_value import ordinary_parameters
from brass_ai.recovery import search_value_schedule as S,ppg_g210 as P,resume
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.search_value_training import StageMachine,Companion,unconditional_pass
from brass_ai.recovery.search_value_phase import SearchValuePhase
from brass_ai.recovery.search_value_iteration import SearchValueIteration,stamp
from brass_ai.recovery.teacher_phase import plan_jobs
from brass_ai.recovery.retention import _same_payload
from brass_ai.rust_backend import lab_teacher_activation as A,lab_shadow_config as C,lab_backend as B
from brass_ai.step5.league import write_json
from test_b025 import model_fixture
from test_b045 import value_fixture
import test_teacher_on_stage_a as activation_fixture

EXPORT=ROOT/'results/joint7-v11-2-g1290-trained-restart-g1280-20261108T173525364834Z'


class Schedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.state=read(EXPORT/'state.json')
        cls.old=cls.state['signature']['config'];cls.new=A.configure(cls.old,revise_stages=True)

    def test_current_results_explain_stalled_untrained_stage_and_fixed_beta_is_retained(self):
        self.assertEqual(self.state['iteration'],1280)
        self.assertEqual(self.state['search_value']['stage']['stage'],'A')
        for generation in (1278,1279):
            report=read(EXPORT/f'iteration-{generation}/report.json')
            self.assertFalse(report['search_value']['updated']);self.assertEqual(report['search_value']['delta']['mean'],0)
            self.assertTrue(all(x['skipped'] for x in report['search_value']['staleness'].values()))
        self.assertEqual(A.authorize(self.state,self.new)['beta'],.16110139609854138)

    def test_exact_config_scope_and_rejection_of_unreviewed_changes(self):
        validate_config(self.new);self.assertTrue(A.is_on(self.new))
        self.assertEqual(S.previous_recipe(self.new),S.previous_recipe(self.old))
        self.assertEqual(self.new['value_phase']['beta_clone_min'],.25)
        self.assertEqual(self.new['value_phase']['kl_cap'],.003)
        self.assertEqual(self.new['teacher_phase']['search_value']['head_to_head']['games'],0)
        signature=dict(self.state['signature'],config=self.new)
        self.assertTrue(C.signature_transition(self.state['signature'],signature))
        self.assertFalse(C.preserves_pending(self.state,signature))
        for field,value in [('stage_a_iterations',9),('stage_b_iterations',10),('validation_interval',1),('stage_c_iterations',11)]:
            bad=copy.deepcopy(self.new);bad['teacher_phase']['search_value']['schedule'][field]=value
            with self.assertRaises(ValueError):validate_config(bad)
        self.assertFalse(C.signature_transition(signature,self.state['signature']))

    def test_four_heldout_games_only_on_tenth_completed_iteration_and_seed_stream_preserved(self):
        arguments=dict(model_id='a'*64,export_hash='b'*64,epoch=1,ready=True)
        for g in range(1280,1300):
            state=dict(self.state,iteration=g)
            jobs,cursor=plan_jobs(self.new,state,**arguments);old,old_cursor=plan_jobs(self.old,state,**arguments)
            self.assertEqual(len(jobs),32 if (g+1)%10==0 else 28)
            self.assertEqual([j for j in jobs if j['split']=='train'],[j for j in old if j['split']=='train'])
            self.assertEqual(cursor,old_cursor)
            self.assertEqual(sum(j['split']=='heldout' for j in jobs),4 if (g+1)%10==0 else 0)

    def test_beta_qualification_override_runs_all_four_games_every_round_and_can_be_disabled(self):
        shadow=copy.deepcopy(self.new);shadow['teacher_phase']['mode']='shadow'
        state=copy.deepcopy(self.state);state['teacher_phase']['weight'].pop('gradient_qualification')
        for g in range(1280,1284):
            plan=S.validation_plan(shadow,dict(state,iteration=g))
            self.assertEqual(plan['games'],4);self.assertTrue(plan['forced_by_beta_qualification'])
        shadow['teacher_phase']['search_value']['schedule']['validation_during_beta_qualification']=False
        validate_config(shadow);self.assertTrue(C.is_shadow(shadow))
        self.assertEqual(S.validation_plan(shadow,state)['games'],0)
        self.assertEqual(S.validation_plan(shadow,dict(state,iteration=1289))['games'],4)
        shadow['teacher_phase']['search_value']['schedule']['validation_during_beta_qualification']=True
        self.assertEqual(S.validation_plan(shadow,self.state)['games'],0)  # already qualified

    def test_a10_b20_c10_unconditional_and_restart_exact_with_no_h2h_or_rollback(self):
        options=self.new['teacher_phase']['search_value']
        machine=StageMachine(saved=self.state['search_value']['stage'],options=options)
        self.assertEqual(machine.state['stage_a_iterations'],0)
        diagnostic=dict(recorded=False,improvement=-999.)
        for step,g in enumerate(range(1280,1322),1):
            before=copy.deepcopy(machine.state)
            state=machine.advance(g,diagnostic,ordinary=None,annealing=True)
            self.assertEqual(state,StageMachine(saved=before,options=options).advance(g,diagnostic))
            machine=StageMachine(saved=state,options=options)
            self.assertNotEqual(state['stage'],'head_to_head')
            if step<10:self.assertEqual(state['stage'],'A')
            elif step<30:
                self.assertEqual(state['stage'],'B');self.assertAlmostEqual(state['w'],max(0,step-10)*.05)
            else:
                self.assertEqual(state['stage'],'C');self.assertEqual(state['w'],1)
                self.assertAlmostEqual(state['scale'],min(1,max(0,step-30)*.1))
            with self.assertRaises(ValueError):machine.advance(g,diagnostic)

    def test_old_head_to_head_is_skipped_and_malformed_committed_counters_fail(self):
        options=self.new['teacher_phase']['search_value'];old=copy.deepcopy(self.state['search_value']['stage'])
        old.update(stage='head_to_head',w=1.)
        machine=StageMachine(saved=old,options=options);self.assertEqual(machine.state['stage'],'C')
        for key,value in [('stage_a_iterations',9),('stage_b_iterations',19)]:
            bad=dict(machine.state,**{key:value})
            with self.assertRaises(ValueError):StageMachine(saved=bad,options=options)

    def test_stage_b_collection_uses_first_point_zero_five_and_reaches_one_on_twentieth(self):
        for completed,expected in ((0,.05),(9,.5),(19,1.)):
            state=copy.deepcopy(self.state)
            state['search_value']['stage'].update(stage='B',w=completed*.05)
            jobs,_=plan_jobs(self.new,state,model_id='a'*64,export_hash='b'*64,epoch=1,ready=True)
            self.assertTrue(all(j['search_value']['w']==expected for j in jobs))


class Updates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name);self.model=model_fixture()
        config=A.configure(read(EXPORT/'recorded-config.json'),revise_stages=True)
        self.options=config['teacher_phase']['search_value']
        self.phase=SearchValuePhase(self.directory/'values',self.model,self.options,digest(self.model.config.utility))
        self.row,_,_=value_fixture(self.model,generation=1280)
        self.row.update(reward_id=self.phase.reward_id,terminal_utility=self.row['u'])
        self.phase.buffer.publish(1280,'train',[self.row],game_id=self.row['game_id'])

    def test_head_trains_directly_even_when_all_policy_ages_would_be_stale(self):
        ordinary=torch.optim.AdamW(ordinary_parameters(self.model));weights=copy.deepcopy(self.model.state_dict())
        adam=copy.deepcopy(ordinary.state_dict());rng=torch.get_rng_state().clone()
        with patch.object(self.phase,'stale_ages',side_effect=AssertionError('No head policy-staleness gate')),\
                patch('brass_ai.recovery.search_value_phase.clustered_gate',side_effect=AssertionError('No validation gate')),\
                patch('brass_ai.recovery.search_value_phase.predict_rows',side_effect=AssertionError('No unscheduled validation')):
            report=self.phase.after_ppg(1280,mode='on',annealing=True)
        self.assertEqual(report['update']['steps'],32);self.assertTrue(report['updated'])
        self.assertEqual(report['stage']['stage_a_iterations'],1);self.assertFalse(report['validation']['recorded'])
        self.assertEqual(report['stage']['w'],0);self.assertEqual(report['stage']['scale'],0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(_same_payload(adam,ordinary.state_dict()))
        for name,value in weights.items():
            if not name.startswith('search_value_head.'):self.assertTrue(torch.equal(value,self.model.state_dict()[name]),name)
        self.assertTrue(any(not torch.equal(value,self.model.state_dict()[name]) for name,value in weights.items() if name.startswith('search_value_head.')))

    def test_return_buffer_keeps_eight_ages_and_legacy_duplicate_labels_are_dropped(self):
        old=self.phase.payload();old['label_ages']=[dict(generation=1280,rows=[{'obsolete':'full policy view'}])]
        restored=SearchValuePhase(self.directory/'values',self.model,self.options,self.phase.reward_id,saved=old)
        self.assertEqual(restored.label_ages,[]);self.assertEqual(restored.payload()['label_ages'],[])
        for g in (1272,1273):
            row=dict(self.row,generation=g,game_id=f'age-{g}')
            restored.buffer.publish(g,'train',[row],game_id=row['game_id'])
        rows=restored.buffer.rows(1280)
        self.assertEqual({r['generation'] for r in rows},{1273,1280})

    def test_diagnostics_use_four_current_games_only_never_old_cached_validation(self):
        for game in range(4):
            rows=[dict(self.row,split='heldout',index=i,game_id=f'heldout-{game}') for i in range(1,65)]
            self.phase.buffer.publish(1280,'heldout',rows,game_id=rows[0]['game_id'])
        first=self.phase.after_ppg(1280,mode='on')
        self.assertEqual(first['validation']['games'],4);self.assertEqual(first['validation']['rows'],256)
        self.assertEqual(first['stage']['stage'],'A')
        with patch('brass_ai.recovery.search_value_phase.predict_rows',side_effect=AssertionError('No stale validation reuse')):
            second=self.phase.after_ppg(1281,mode='on')
        self.assertFalse(second['validation']['recorded'])

    def test_strict_checkpoint_restores_head_adam_rng_and_stage_counters(self):
        self.phase.after_ppg(1280,mode='on');ordinary=torch.optim.AdamW(ordinary_parameters(self.model))
        saved=self.phase.payload();path=self.directory/'strict.pt'
        save_checkpoint(path,self.model,ordinary,step=1281,training=dict(search_value=saved))
        restored,raw=load_checkpoint(path);other=torch.optim.AdamW(ordinary_parameters(restored));resume_optimizer(raw,other)
        phase=SearchValuePhase(self.directory/'values',restored,self.options,self.phase.reward_id,saved=raw['training']['search_value'])
        self.assertTrue(_same_payload(phase.companion.snapshot(),self.phase.companion.snapshot()))
        self.assertEqual(phase.machine.state,self.phase.machine.state);self.assertEqual(phase.companion.steps,32)
        self.assertTrue(_same_payload(self.model.state_dict(),restored.state_dict()))

    def guard(self,probe):
        optimizer=torch.optim.Adam(ordinary_parameters(self.model),lr=1e-5)
        return dict(value_optimizer=optimizer,scale=.1,cloning_loss=lambda:next(self.model.parameters()).sum()*0,
            measure_joint_kl=probe,beta=.25,rate_for_step=lambda *_:1e-5,kl_cap=.003,
            measure_ordinary=lambda:(_ for _ in ()).throw(AssertionError('No validation qualification')))

    def test_stage_c_keeps_all_under_cap_intervals_without_validation_qualifications(self):
        companion=self.phase.companion;guard=self.guard(lambda:.0025)
        result=unconditional_pass(companion,[self.row],**guard)
        self.assertEqual(result['steps'],32);self.assertTrue(all(i['accepted'] for i in result['intervals']))
        self.assertEqual(result['cap'],.003);self.assertFalse(result['validation_gates'])

    def test_stage_c_over_cap_only_restores_rejected_interval_and_both_adams(self):
        companion=self.phase.companion;guard=self.guard(iter([.001,.0031]).__next__)
        before=copy.deepcopy((self.model.state_dict(),guard['value_optimizer'].state_dict(),companion.snapshot()))
        result=unconditional_pass(companion,[self.row],**guard)
        self.assertEqual(result['steps'],0);self.assertEqual(result['intervals'][0]['reason'],'joint-kl-cap')
        self.assertTrue(_same_payload(before,(self.model.state_dict(),guard['value_optimizer'].state_dict(),companion.snapshot())))

    def test_stage_c_second_interval_rejection_keeps_first_accepted_interval(self):
        companion=self.phase.companion;guard=self.guard(iter([.001,.0025,.004]).__next__)
        result=unconditional_pass(companion,[self.row],**guard)
        self.assertEqual(result['steps'],16);self.assertEqual(companion.steps,16)
        self.assertEqual([r['accepted'] for r in result['intervals']],[True,False])

    def test_no_kl_budget_and_interruption_preserve_weights_and_adams(self):
        companion=self.phase.companion;guard=self.guard(lambda:.003)
        before=copy.deepcopy((self.model.state_dict(),guard['value_optimizer'].state_dict(),companion.snapshot()))
        self.assertEqual(unconditional_pass(companion,[self.row],**guard)['steps'],0)
        probe=iter([.001,RuntimeError('cancelled trunk pass')])
        def measure():
            value=next(probe)
            if isinstance(value,Exception):raise value
            return value
        guard['measure_joint_kl']=measure
        with self.assertRaisesRegex(RuntimeError,'cancelled'):unconditional_pass(companion,[self.row],**guard)
        self.assertTrue(_same_payload(before,(self.model.state_dict(),guard['value_optimizer'].state_dict(),companion.snapshot())))

    def test_first_stage_c_post_ppg_pass_uses_point_one_and_advances_even_without_a_probe(self):
        import test_b05
        config=test_b05.recipe();config['teacher_phase']['search_value']=self.options
        iteration=SearchValueIteration(self.directory/'committed',config,dict(iteration=1280),self.model)
        phase=iteration.phase
        phase.machine.state.update(stage='C',stage_a_iterations=10,stage_b_iterations=20,w=1.,scale=0.)
        phase.buffer.publish(1280,'train',[self.row],game_id=self.row['game_id'])
        iteration.prepare_ppg(None,1281);guard=self.guard(lambda:.0025);guard.pop('scale')
        iteration._finish(guard)
        self.assertEqual(iteration.report['training_scale'],.1);self.assertEqual(iteration.report['stage']['scale'],.1)
        self.assertEqual(iteration.report['update']['steps'],32);self.assertEqual(iteration.report['stage']['stage'],'C')
        phase.machine.state['generation']=1279;phase.machine.state['scale']=0.
        iteration.report=None
        report=iteration.finish_ppg()
        self.assertFalse(report['updated']);self.assertEqual(report['stage']['scale'],.1)


class Resume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);activation_fixture.Activation.setUpClass()

    def test_public_teacher_command_defaults_to_revised_schedule_and_exports_it(self):
        import sys,os
        launcher=activation_fixture.Activation.launcher(self)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            run=Path(tmp);state=read(EXPORT/'state.json')
            config=A.configure(state['signature']['config'],revise_stages=True);path=run/'revised.json';write_json(path,config)
            final=copy.deepcopy(state);final['iteration']=1281;final['signature']=dict(state['signature'],config=config)
            final['history'][-1]=dict(final['history'][-1],generation=1281)
            def train(argv,log):
                self.assertEqual(argv[argv.index('--expected-generation')+1],'1280')
                self.assertEqual(argv[argv.index('--iterations')+1],'1')
                self.assertIn('--resume-reviewed-source',argv);write_json(run/'state.json',final)
            with patch.object(launcher,'ROOT',run),patch.object(launcher,'prepare',return_value=(state,path,1)) as prepare,\
                    patch.object(launcher,'run_training',side_effect=train),patch('rust_lab_results.export',return_value=run/'results'),\
                    patch.object(sys,'argv',['continue-rust-teacher.py','--run',str(run),'--expected-generation','1280']),\
                    patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                launcher.main(default_mode='on',default_stage_revision=True)
                prepare.assert_called_once_with(run,str(run),1280,1,.25,mode='on',revise_stages=True)
            report=read(next((run/'artifacts/rust-backend').glob('*/qualification.json')))
            self.assertTrue(report['passed']);self.assertEqual(report['stage_schedule'],S.defaults())

    def test_real_loader_transition_retry_and_pending_archive_do_not_reset_primary_checkpoint(self):
        case=activation_fixture.Activation('test_actual_qualification_fixed_beta_and_runtime_anneal_completion')
        case.setUp();self.addCleanup(case.doCleanups);f=case.f
        target=dict(f.after,config=A.configure(f.after['config'],revise_stages=True))
        with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):f.load(target)
        launcher=case.launcher();destination=ROOT/'configs/step5'/(f.run.name+'-rust-teacher-on-stages-v2-share025.json')
        self.addCleanup(destination.unlink,missing_ok=True)
        with patch.object(B,'validate_state'),patch('brass_ai.rust_backend.lab_shadow.teacher_build'):
            _,config,remaining=launcher.prepare(ROOT,str(f.run),1261,1,mode='on',revise_stages=True)
        self.assertEqual(remaining,1);self.assertTrue(A.is_on(read(config)))
        archives=list((f.run/'configuration-history').glob('*/iteration-1261'))
        self.assertEqual(len(archives),1)
        self.assertEqual({p.relative_to(archives[0]).as_posix():p.read_bytes() for p in archives[0].rglob('*') if p.is_file()},f.files)
        checksum=sha(f.checkpoint);model,raw,_=f.load(target)
        self.assertTrue(_same_payload(dict(raw['state_dict']),dict(model.state_dict())))
        with patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value={'receipt':{'existing':'prepared teacher'}}):
            updated=resume.commit_transition(f.run,f.state,target)
        self.assertEqual(updated['native_backend']['teacher']['ppg_beta_floor'],.25)
        for _ in range(2):
            restored,retry,changed=f.load(target,state=updated,reviewed=False)
            self.assertFalse(changed);self.assertTrue(_same_payload(restored.state_dict(),model.state_dict()))
            self.assertEqual(sha(f.checkpoint),checksum);self.assertEqual(retry['rng'].tolist(),raw['rng'].tolist())

    def test_ppg_floor_adoption_preserves_adam_and_replay_and_never_reapplies_reset(self):
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.recovery.utility import reward_spec
        old=read(EXPORT/'recorded-config.json');model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(old)))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            run=Path(tmp);phase=TrainingPhases(model,old,root=ROOT,run=run)
            p=next(model.parameters());p.grad=torch.ones_like(p);phase.optimizer.step();phase.optimizer.zero_grad(set_to_none=True)
            phase.buffer.pending=[];phase.beta=.125;phase.trips=[True,False]
            raw=dict(model_config=model.config.to_dict(),training={KEY:phase.payload()})
            before=copy.deepcopy((model.state_dict(),phase.optimizer.state_dict(),torch.get_rng_state(),phase.state,phase.critic_state))
            new=P.teacher_floor_options(old)
            revised=TrainingPhases(model,new,raw=raw,root=ROOT,run=run)
            self.assertEqual(revised.beta,.25);self.assertEqual(revised.trips,phase.trips)
            self.assertTrue(_same_payload(before,(model.state_dict(),revised.optimizer.state_dict(),torch.get_rng_state(),revised.state,revised.critic_state)))
            revised.buffer.pending=[];revised.beta=.75
            again=TrainingPhases(model,new,raw=dict(training={KEY:revised.payload()}),root=ROOT,run=run)
            self.assertEqual(again.beta,.75)
            report=dict(kl_guard_tripped=False,optimizer_steps=32,probe_kl=.0001)
            self.assertEqual(P.beta_after_phase(.25,[],report,new['value_phase'])[0],.25)


if __name__=='__main__':unittest.main()
