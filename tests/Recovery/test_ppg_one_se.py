"""Coverage retention, anchored tolerance, fresh PPG replay and .020 reward adoption."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.reward import CostLedger
from brass_ai.recovery import ppg_config as P,ppg_auxiliary as A,ppg_epochs,critic_data,gae
from brass_ai.recovery.actor_value import TrainingPhases,KEY,ppo_evidence
from brass_ai.recovery.config import insolvency_010_config,insolvency_020_config,latest_model_config,validate_config
from brass_ai.recovery.ppg_selection import Selection
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import reward_spec,migrate,event_cost_upgrade
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_eight import setup,stage_only

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0540-20261022T144917069131Z'
spec=importlib.util.spec_from_file_location('g540_helper',SOURCE_ROOT/'scripts/joint7-g540-ppg.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


class Choice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.model=torch.nn.Linear(1,1);self.opt=torch.optim.Adam(self.model.parameters(),lr=.001)
        self.games=dict.fromkeys('abcd',1.)
        self.s=Selection(self.model,self.opt,self.games)

    def step(self,games,rows,budget):
        for p in self.model.parameters():p.grad=torch.ones_like(p)
        self.opt.step();self.opt.zero_grad(set_to_none=True)
        weights=copy.deepcopy(self.model.state_dict());adam=copy.deepcopy(self.opt.state_dict())
        chosen=self.s.consider(self.model,self.opt,games,rows,[0]*budget,budget,budget)
        return chosen,weights,adam

    def test_neutral_keeps_more_distinct_data_with_matching_adam(self):
        candidate=dict(zip('abcd',(1.03,.99,1.03,.99)))
        chosen,weights,adam=self.step(candidate,[1,2,3],2)
        self.assertTrue(chosen);self.assertEqual(self.s.minimum_games,self.games)
        self.s.restore(self.model,self.opt)
        self.assertTrue(_same_payload(self.model.state_dict(),weights))
        self.assertTrue(_same_payload(self.opt.state_dict(),adam))
        self.assertEqual(self.s.report()['neutral_intervals'],1)
        self.assertGreater(self.s.report()['retained_loss'],self.s.report()['minimum_loss'])

    def test_no_ratchet_and_new_minimum_rechecks_previous_selection(self):
        self.step(dict(zip('abcd',(1.03,.99,1.03,.99))),[1,2,3],2)
        chosen,_,_=self.step(dict(zip('abcd',(1.06,1.02,1.06,1.02))),list(range(10)),4)
        self.assertFalse(chosen);self.assertEqual(self.s.budget_step,2)
        chosen,_,_=self.step(dict.fromkeys('abcd',.9),[1],6)
        self.assertTrue(chosen);self.assertEqual(self.s.minimum_games,dict.fromkeys('abcd',.9))
        self.assertEqual(self.s.rows,[1]);self.assertEqual(self.s.budget_step,6)

    def test_ties_prefer_lower_loss_over_repeated_presentations(self):
        self.step(dict(zip('abcd',(1.03,.99,1.03,.99))),[1,2,3],2)
        chosen,_,_=self.step(dict(zip('abcd',(1.035,.995,1.035,.995))),[1,2,3]*2,4)
        self.assertFalse(chosen)
        chosen,_,_=self.step(dict.fromkeys('abcd',.99),[1,2,3],6)
        self.assertTrue(chosen);self.assertEqual(self.s.budget_step,6)

    def test_paired_game_not_row_support_and_one_se_boundary(self):
        self.assertEqual(A.paired(self.games,self.games)['games'],4)
        self.assertFalse(A.paired(self.games,dict(zip('abcd',(1.03,.99,1.03,.99))))['reject'])
        self.assertTrue(A.paired(self.games,dict.fromkeys('abcd',1.001))['reject'])
        boundary=A.paired({'a':0.,'b':0.},{'a':0.,'b':.02})
        self.assertAlmostEqual(boundary['mean_change'],boundary['standard_error'])
        self.assertFalse(boundary['reject'])
        with self.assertRaises(ValueError):A.paired(self.games,{'a':float('nan')})


class Fit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def run_fit(self,trip=None):
        model,opt,cfg=setup();cfg=P.coverage_options(cfg)
        cfg['value_phase'].update(batch_size=128,acceptance_steps=2)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            for g in range(1,9):
                stage_only(model,p,g)
                if g<8:write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            calls=[];saved={};step=p.optimizer.step;probes=A.Reference.probes
            def steps():return max((int(s['step']) for s in p.optimizer.state.values()),default=0)
            def recording(*args,**kwargs):
                result=step(*args,**kwargs);calls.append(1)
                if len(calls)==4:
                    saved.update(weights=copy.deepcopy(model.state_dict()),adam=copy.deepcopy(p.optimizer.state_dict()))
                return result
            def fake_losses(net,ds,ids,update):
                names=sorted({ds.data[i][2][0] for i in ids});signs={g:(.02 if j%2 else -.02) for j,g in enumerate(names)}
                n=steps();base=1. if n<2 else .98 if n<4 else .981 if n==4 else .99
                return {i:base+(signs[ds.data[i][2][0]] if n==4 else 0.) for i in ids}
            def guarded(ref,net):
                result=probes(ref,net)
                if trip and steps()>=5:result[trip]=.003
                return result
            with patch.object(p.optimizer,'step',recording),patch.object(A,'losses',fake_losses),patch.object(A.Reference,'probes',guarded):
                report=A.fit(model,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                    cfg['value_phase'],cfg['update'],beta=1.,snapshot_weight=1.,dropout=None,
                    run_phase=True,ppo_work=p.ppo_work,epoch_cap=2)
            self.assertEqual(report['optimizer_steps'],4)
            self.assertTrue(_same_payload(model.state_dict(),saved['weights']))
            self.assertTrue(_same_payload(p.optimizer.state_dict(),saved['adam']))
            self.assertEqual(report['best_budget_step'],2)
            self.assertEqual(report['selection']['retained_budget_step'],4)
            self.assertEqual(report['distinct_value_rows_used'],512)
            self.assertEqual(report['acceptance_games'],48)
            self.assertEqual(report['acceptance_generations'],list(range(8,0,-1)))
            self.assertLessEqual(report['probe_kl'],.002);self.assertLessEqual(report['age0_probe_kl'],.002)
            return report

    def test_neutral_does_not_stop_but_outside_one_se_does(self):
        r=self.run_fit();self.assertEqual(r['attempted_steps'],6)
        self.assertEqual([i['outcome'] for i in r['interval_reports']],['improved','neutral','failed'])
        self.assertEqual(r['early_stop_reason'],'validation_patience')

    def test_each_kl_guard_independently_restores_coverage_state(self):
        for metric in ('overall','age0'):
            with self.subTest(metric=metric):
                r=self.run_fit(metric);self.assertTrue(r['kl_guard_tripped'])
                self.assertEqual(r['attempted_steps'],5)
                self.assertEqual(r['early_stop_reason'],'policy_kl_cap_no_value_improvement')


class Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_config_defaults_and_archived_migration(self):
        old=read(EXPORT/'recorded-config.json');new=helper.configure(old);validate_config(new)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'insolvency','search','value_phase'})
        self.assertEqual(new['insolvency'],dict(event_cost=.020,max_cost=.080))
        self.assertEqual(new['search']['reward']['insolvency'],new['insolvency'])
        self.assertEqual(new['value_phase']['epoch_controller']['minimum'],2)
        self.assertEqual(helper.configure(new),new)
        for key in ('exploration','entropy_control','branching','ppo_lr_controller','advantage'):
            self.assertEqual(new[key],old[key])
        older=copy.deepcopy(old);older['insolvency']['event_cost']=.0075
        older['search']['reward']['insolvency']['event_cost']=.0075
        self.assertEqual(insolvency_010_config(older)['insolvency']['event_cost'],.010)
        self.assertEqual(insolvency_010_config(old),old)
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(fresh['value_phase'],new['value_phase']);self.assertEqual(fresh['insolvency'],new['insolvency'])
        bad=copy.deepcopy(new);bad['value_phase']['selection']['standard_errors']=2.
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(new);bad['value_phase']['epoch_controller']['minimum']=1
        with self.assertRaises(ValueError):validate_config(bad)

    def test_event_cost_cap_and_explicit_reward_migration(self):
        old=read(EXPORT/'recorded-config.json');new=helper.configure(old)
        self.assertTrue(event_cost_upgrade(reward_spec(old),reward_spec(new)))
        ledger=CostLedger(reward_spec(new))
        for i in range(6):
            ledger.charge(0);self.assertAlmostEqual(ledger.spent[0],min((i+1)*.020,.080))
        self.assertEqual(ledger.spent[1:].tolist(),[0.,0.,0.])
        bad=copy.deepcopy(new);bad['insolvency']['max_cost']=.09
        self.assertFalse(event_cost_upgrade(reward_spec(old),reward_spec(bad)))

    def test_epoch_controller_floor_two_and_existing_raise_rule(self):
        c=helper.configure(read(EXPORT/'recorded-config.json'))['value_phase']
        state=ppg_epochs.initial(540,c);decisions=[]
        for fraction in (.1,.1,.1,.1,1.,1.,1.,1.):
            report=dict(scheduled=True,epoch_cap_used=state['cap'],steps_per_epoch=100,
                planned_steps=state['cap']*100,best_budget_step=int(state['cap']*100*fraction),kl_guard_tripped=False)
            state,result=ppg_epochs.advance(state,report,c,state['last_phase']+8)
            decisions.append(result['decision'])
            self.assertGreaterEqual(state['cap'],2)
        self.assertEqual(state['cap'],3);self.assertIn('up',decisions)

    def test_fresh_reset_cleanup_and_restart_preserve_training_state(self):
        model,opt,cfg=setup();old=critic_data.fixed_config(P.short_options(insolvency_010_config(cfg)))
        model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            for g in range(1,9):
                stage_only(model,p,g);write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            p.state=gae.initial(old['advantage'],8);p.critic_state['generation']=8;p.critic_evidence=ppo_evidence(8)
            for optimizer in (opt,p.optimizer):
                param=next(model.parameters());param.grad=torch.ones_like(param);optimizer.step();optimizer.zero_grad(set_to_none=True)
            clock=P.initial(0)
            for g in range(1,9):clock=P.advance(clock,g,P.due(clock))
            epoch=dict(ppg_epochs.initial(0,old['value_phase']),cap=1,phases=1,last_phase=8,window_origin=1)
            saved=dict(contract=p.contract,options=p.options,gae=p.state,critic_data=p.critic_state,
                critic_evidence=p.critic_evidence,beta_clone=8.,kl_trips=[False],optimizer=p.optimizer.state_dict(),
                schedule=clock,epoch_control=epoch,rolling_buffer=p.buffer.receipts)
            raw=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(saved)})
            new=P.coverage_options(insolvency_020_config(old))
            weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError,'explicit reward migration'):migrate(model,opt,reward_spec(new))
            model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(new))
            restored=TrainingPhases(model,new,raw=raw,generation=8,root=SOURCE_ROOT,run=run)
            self.assertEqual(restored.schedule,P.initial(8));self.assertEqual(restored.epoch_control['cap'],2)
            self.assertEqual(restored.state,p.state);self.assertEqual(restored.critic_state,p.critic_state)
            self.assertTrue(_same_payload(restored.optimizer.state_dict(),p.optimizer.state_dict()))
            self.assertTrue(_same_payload(opt.state_dict(),adam));self.assertTrue(_same_payload(model.state_dict(),weights))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(restored.buffer.receipts)
            self.assertTrue(p.buffer.path(8,'critic.npy').exists())
            write_json(run/'state.json',dict(iteration=8,signature=dict(config=old)))
            with self.assertRaisesRegex(ValueError,'committed one-SE'):restored.startup(8)
            self.assertTrue(p.buffer.path(8,'critic.npy').exists())
            write_json(run/'state.json',dict(iteration=8,signature=dict(config=new)))
            # Audit all cache paths before deleting any; unknown files fail closed.
            unknown=restored.buffer.directory/'user-file.txt';unknown.write_text('keep')
            with self.assertRaisesRegex(ValueError,'Unknown PPG cache artifact'):restored.startup(8)
            self.assertTrue(p.buffer.path(8,'critic.npy').exists());unknown.unlink()
            cleanup=restored.startup(8);self.assertTrue(cleanup['deleted_files'])
            self.assertEqual(cleanup['next_phase'],16);self.assertFalse(p.buffer.path(8,'critic.npy').exists())
            repeat=TrainingPhases(model,new,raw=raw,generation=8,root=SOURCE_ROOT,run=run)
            self.assertEqual(repeat.startup(8)['deleted_files'],[])
            stage_only(model,restored,9)
            restored.pending_schedule=P.advance(restored.schedule,9,False)
            restored.pending_epoch_control=copy.deepcopy(restored.epoch_control)
            restored.state=gae.initial(new['advantage'],9);restored.critic_state['generation']=9
            checkpoint=dict(model_config=model.config.to_dict(),training={KEY:restored.payload()})
            write_json(run/'state.json',dict(iteration=9,signature=dict(config=new)));restored.buffer.committed(9)
            again=TrainingPhases(model,new,raw=checkpoint,generation=9,root=SOURCE_ROOT,run=run)
            self.assertIsNone(again.startup(9));self.assertEqual(again.cache_origin,8)
            self.assertEqual([r['generation'] for r in again.buffer.receipts],[9])
            self.assertTrue(again.buffer.path(9,'critic.npy').exists());self.assertEqual(again.schedule['origin'],8)

    def test_actual_g540_signature_and_idempotent_helper(self):
        from brass_ai.recovery.resume import check_signature
        state=read(EXPORT/'state.json');new=helper.configure(state['signature']['config'])
        target=dict(state['signature'],config=new,implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        bad=copy.deepcopy(target);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True)
            write_json(run/'state.json',state);checksum=sha(run/'state.json')
            for _ in range(2):
                result=helper.generate(root,run,'configs/step5/new.json',540)
                self.assertFalse(result['run_modified']);self.assertEqual(result['first_fresh_ppg_generation'],548)
                self.assertEqual(sha(run/'state.json'),checksum)

    def test_interrupted_source_adoption_restores_original_checkpoint_exactly(self):
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.recovery.resume import load_resume,commit_transition
        model,opt,cfg=setup();old=read(EXPORT/'recorded-config.json');old['model_size']='small'
        model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(old))
        options={k:copy.deepcopy(old[k]) for k in ('advantage','value_phase','critic_data','seat_descriptors')}
        value_adam=torch.optim.Adam(model.parameters(),lr=old['value_phase']['learning_rate'])
        for optimizer in (opt,value_adam):
            param=next(model.parameters());param.grad=torch.ones_like(param);optimizer.step();optimizer.zero_grad(set_to_none=True)
        # Authored old checkpoint immediately after PPG enablement at g1.
        saved=dict(contract='actor-value-phases/9.0.0',options=options,gae=gae.initial(old['advantage'],1),
            critic_data=dict(critic_data.initial(old['critic_data']),generation=1),critic_evidence=ppo_evidence(1),
            beta_clone=8.,kl_trips=[],optimizer=value_adam.state_dict(),schedule=P.initial(1),
            epoch_control=dict(ppg_epochs.initial(1,old['value_phase']),cap=1),rolling_buffer=[])
        signature=dict(read(EXPORT/'state.json')['signature'],config=old)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True);path=run/'generation-0001.pt'
            model_id=save_checkpoint(path,model,opt,step=1,training={KEY:saved,'signature':digest(signature),
                'fresh_run':run.relative_to(root).as_posix()})
            checksum=sha(path);entry=dict(path=path.relative_to(root).as_posix(),sha256=checksum,model_id=model_id,generation=1)
            state=dict(format='step5-recovery-league/2.2.0',iteration=1,games_done=320,signature=signature,
                history=[entry],population=[],evaluation_anchor=entry)
            write_json(run/'state.json',state);new=helper.configure(old)
            target=dict(signature,config=new,implementation=implementation_id())
            def restore():
                net,raw,_=load_resume(root,run,state,target,reviewed=True)
                adam=torch.optim.AdamW(net.parameters());resume_optimizer(raw,adam)
                migrate(net,adam,reward_spec(new))
                phase=TrainingPhases(net,new,raw=raw,generation=1,root=root,run=run)
                return net,adam,phase,torch.get_rng_state().clone()
            net,adam,a,rng=restore();state=commit_transition(run,state,target);a.startup(1)
            other,other_adam,b,rng2=restore();b.startup(1)
            self.assertTrue(_same_payload(net.state_dict(),other.state_dict()))
            self.assertTrue(_same_payload(adam.state_dict(),other_adam.state_dict()))
            self.assertTrue(_same_payload(a.optimizer.state_dict(),b.optimizer.state_dict()))
            self.assertTrue(torch.equal(rng,rng2));self.assertEqual(a.state,b.state)
            self.assertEqual(a.epoch_control['cap'],2);self.assertEqual(a.schedule,P.initial(1))
            self.assertEqual(sha(path),checksum);self.assertEqual(state['iteration'],1);self.assertEqual(state['games_done'],320)


if __name__=='__main__':unittest.main()
