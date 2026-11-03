"""Entropy/reward restart boundaries, coefficient provenance and bounded fitting."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import entropy_schedule as S,exploration as E
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.training_revision_g570 import configure as g570
from brass_ai.recovery.utility import reward_spec,migrate,calibration_ready
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.league import Behavior
from brass_ai.step5.league import write_json,implementation_id
from test_training_g570 import BUNDLE


def commit(state,config,*,healthy=True,measurement=None):
    result=copy.deepcopy(state)
    ex,report=E.advance(state['exploration'],config,measurement or {},generation=state['iteration']+1,
        warmup_done=True,healthy=healthy)
    result.update(iteration=state['iteration']+1,exploration=ex,signature=dict(state['signature'],config=config))
    return result,report


class Anneal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original=read(BUNDLE/'state.json');cls.off=g570(cls.original['signature']['config'])
        cls.on=validate_config(S.configure(cls.off,cls.original,'on',start='next'))

    def start(self,config=None):
        config=config or self.on;state=copy.deepcopy(self.original)
        state['exploration']=E.restore(config,state,state['exploration'])
        state['signature']['config']=copy.deepcopy(config)
        return state

    def progress(self,n):
        state=self.start()
        for _ in range(n):state,_=commit(state,self.on)
        return state

    def test_config_scope_reward_binding_and_strict_admission(self):
        self.assertEqual(self.on['update']['entropy_schedule']['start_generation'],571)
        self.assertEqual(self.on['insolvency'],dict(event_cost=.01,max_cost=.075))
        self.assertEqual({k for k in self.on if self.on[k]!=self.off[k]},
            {'update','insolvency','search'})
        self.assertEqual(self.on['exploration'],.02)
        self.assertEqual(self.on['search']['reward'],reward_spec(self.on))
        from brass_ai.recovery.teacher_phase_config import defaults
        with_teacher=copy.deepcopy(self.off);with_teacher['teacher_phase']=defaults(reward_spec(with_teacher))
        scheduled=S.configure(with_teacher,self.original,'on',start='next')
        self.assertEqual(scheduled['teacher_phase']['search']['reward'],reward_spec(scheduled))
        self.assertEqual(self.on['opening'],E.THREE_OPENING)
        self.assertEqual(self.on['branching'],E.FORK_T15_BRANCH)
        for key,value in [('ramp_iterations',0),('measure_every',False),('shape','cosine'),
            ('advance_on','all-iterations'),('start_generation',None)]:
            bad=copy.deepcopy(self.on);bad['update'][S.KEY][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        for key,value in [('start',.0025),('final',.006)]:
            bad=copy.deepcopy(self.on);bad['update'][S.KEY][key]['canal_action']=value
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(self.on);bad['update'][S.KEY]['insolvency_steps'][1]['insolvency']['event_cost']=.008
        with self.assertRaises(ValueError):validate_config(bad)
        for key in ('search','teacher_phase'):
            bad=copy.deepcopy(scheduled)
            target=bad['search'] if key=='search' else bad['teacher_phase']['search']
            target['reward']['insolvency']['event_cost']=.02
            with self.assertRaises(ValueError):validate_config(bad)
        with self.assertRaisesRegex(ValueError,'past'):S.configure(self.off,self.original,'on',start=570)
        bad=copy.deepcopy(self.on);bad['insolvency']=dict(event_cost=.005,max_cost=.05)
        S._reward(bad,bad['insolvency'])
        with self.assertRaises(ValueError):S.restore(bad,None,570)

    def test_linear_healthy_commits_midpoint_hold_step_two_and_final(self):
        state=self.start();baseline=copy.deepcopy(state['exploration'])
        values=[]
        for k in range(50):
            settings=E.effective(self.on,state['exploration'],state['iteration'])
            values.append(settings['alphas']['canal_action'])
            state,report=commit(state,self.on)
            self.assertEqual(report['groups']['canal_action']['effective_alpha'],values[-1])
            self.assertEqual(report[S.KEY]['effective_alpha'],settings['alphas'])
        self.assertEqual(values[0],.005);self.assertAlmostEqual(values[25],.004)
        self.assertEqual(state['exploration'][S.KEY]['progress'],50)
        self.assertTrue(report[S.KEY]['waiting_for_insolvency_step'])
        before=copy.deepcopy(state)
        state,report=commit(state,self.on)
        self.assertEqual((report[S.KEY]['progress'],report[S.KEY]['reason']),(50,'waiting_for_insolvency_step'))
        self.assertAlmostEqual(report[S.KEY]['effective_alpha']['canal_action'],.003)
        self.assertEqual(self.on['insolvency'],dict(event_cost=.01,max_cost=.075))
        for key in ('baseline','baseline_rows','armed_at','groups'):
            self.assertEqual(state['exploration'][key],baseline[key])
        new=validate_config(S.configure(self.on,state,'insolvency-step'))
        self.assertEqual(new['insolvency'],dict(event_cost=.005,max_cost=.05))
        state['exploration']=E.restore(new,state,state['exploration'])
        state['signature']['config']=new
        self.assertEqual(state['exploration'][S.KEY]['progress'],50)
        for k in range(50,102):
            alpha=E.effective(new,state['exploration'],state['iteration'])['alphas']['rail_other']
            if k==75:self.assertAlmostEqual(alpha,.002)
            if k>=100:self.assertEqual(alpha,.001)
            state,report=commit(state,new)
        self.assertEqual(report[S.KEY]['progress'],100)
        self.assertEqual(report[S.KEY]['reason'],'complete')
        self.assertEqual(report[S.KEY]['insolvency_steps_applied'],2)
        self.assertEqual(before['iteration'],620)

    def test_hold_resume_interrupt_off_on_reset_and_explicit_reverse(self):
        state=self.progress(7);original=copy.deepcopy(state)
        rejected,report=commit(state,self.on,healthy=False)
        self.assertEqual(report[S.KEY]['progress'],7);self.assertEqual(report[S.KEY]['held'],1)
        # Computing a candidate commit never advances the persisted input.
        self.assertEqual(state,original)
        restored=E.restore(self.on,json.loads(json.dumps(state)),json.loads(json.dumps(state['exploration'])))
        again=copy.deepcopy(state);again['exploration']=restored
        self.assertEqual(commit(state,self.on),commit(again,self.on))
        off=validate_config(S.configure(self.on,state,'off'))
        self.assertEqual(off['insolvency'],self.on['insolvency'])
        state['exploration']=E.restore(off,state,state['exploration']);state['signature']['config']=off
        for _ in range(3):
            state,report=commit(state,off)
            self.assertEqual(report[S.KEY]['progress'],7)
            self.assertEqual(report[S.KEY]['effective_alpha'],dict.fromkeys(E.GROUPS,.005))
        on=S.configure(off,state,'on',start='next')
        state['exploration']=E.restore(on,state,state['exploration']);state['signature']['config']=on
        state,report=commit(state,on);self.assertEqual(report[S.KEY]['progress'],8)
        reset=S.configure(on,state,'on',start='next',reset=True)
        reset_state=E.restore(reset,state,state['exploration'])
        self.assertEqual(reset_state[S.KEY]['progress'],0);self.assertEqual(reset_state[S.KEY]['reset_count'],1)
        self.assertEqual(reset_state,E.restore(reset,state,state['exploration']))
        with self.assertRaisesRegex(ValueError,'not due'):S.configure(self.on,self.progress(49),'insolvency-step')
        at50=self.progress(50);step2=S.configure(self.on,at50,'insolvency-step')
        at50['exploration']=E.restore(step2,at50,at50['exploration']);at50['signature']['config']=step2
        reverse=S.configure(step2,at50,'insolvency-step',reverse=True)
        self.assertEqual(reverse['insolvency'],self.on['insolvency'])
        back=E.restore(reverse,at50,at50['exploration']);self.assertEqual(back[S.KEY]['progress'],50)
        self.assertTrue(back[S.KEY]['waiting_for_insolvency_step'])
        with self.assertRaises(ValueError):S.configure(reverse,dict(at50,exploration=back),'insolvency-step',reverse=True)

    def test_exact_entropy_loss_provenance_and_off_equivalence(self):
        from brass_ai.recovery.updates import entropy_term
        state=self.progress(25);settings=E.effective(self.on,state['exploration'],state['iteration'])
        behavior=Behavior('model',.02,settings)
        rows=[dict(view=dict(context=dict(era=era,phase=phase)))
            for era in ('CanalEra','RailEra') for phase in ('action','loan')]
        dataset=SimpleNamespace(rows=rows);logits=torch.tensor([[1.,2.,-torch.inf]]*4,requires_grad=True)
        raw=logits.softmax(-1)
        entropy,loss=entropy_term(logits,raw,dataset,list(range(4)),behavior,self.on['update'])
        torch.testing.assert_close(loss,.004*entropy)
        self.assertTrue(torch.isfinite(loss).all());loss.sum().backward();self.assertTrue(torch.isfinite(logits.grad).all())
        behavior.entropy_alphas=dict.fromkeys(E.GROUPS,.005)
        with self.assertRaisesRegex(ValueError,'differ'):entropy_term(logits,raw,dataset,range(4),behavior,self.on['update'])
        behavior=Behavior('model',.02,settings);behavior.entropy_schedule['config_id']='stale'
        with self.assertRaisesRegex(ValueError,'receipt'):entropy_term(logits,raw,dataset,range(4),behavior,self.on['update'])
        offstate=self.start(self.off);offsettings=E.effective(self.off,offstate['exploration'],570)
        _,loss=entropy_term(logits,raw,dataset,range(4),Behavior('m',.02,offsettings),self.off['update'])
        torch.testing.assert_close(loss,.005*entropy)

    def test_report_only_probe_no_KL_no_RNG_baseline_change_and_cadence(self):
        from brass_ai.recovery.il_kl_probe import Probe
        from brass_ai.recovery.history import compact
        rows=[dict(group=g,kinds=['build','loan'],industries=['cotton',None]) for g in E.GROUPS]
        probe=object.__new__(Probe);probe.rows=rows;probe.arrays=dict(offsets=np.arange(0,10,2));probe._input_bytes=0
        before=torch.get_rng_state().clone()
        with patch.object(probe,'probabilities',return_value=np.array([.25,.75]*4,dtype=np.float32)) as probabilities,\
             patch.object(probe,'measure',side_effect=AssertionError('AnchorKL forbidden')):
            report,values=probe.entropy_only(None,{})
        probabilities.assert_called_once();self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertNotIn('kl_mean',report);self.assertFalse(report['anchor_kl_measured'])
        self.assertAlmostEqual(report['family_shares']['build'],.25)
        self.assertEqual(report['industry_shares'],{'cotton':1.})
        self.assertEqual([g for g in range(571,582) if S.measurement_due(self.on,g)],[575,580])
        self.assertFalse(S.measurement_due(self.off,575))
        state=self.start();baseline=copy.deepcopy(state['exploration'])
        state,out=commit(state,self.on,measurement=report)
        for key in ('baseline','baseline_rows','armed_at','groups'):self.assertEqual(state['exploration'][key],baseline[key])
        small=compact(dict(entropy_schedule=out[S.KEY],entropy_measurement=report,timings=dict(entropy_probe_seconds=.1)))
        self.assertEqual(small['entropy_measurement'],report);self.assertEqual(small['entropy_schedule'],out[S.KEY])
        self.assertEqual(small['timings']['entropy_probe_seconds'],.1)

    def test_current_fresh_generation_does_not_inherit_schedule(self):
        from brass_ai.recovery.config import latest_model_config
        fresh=latest_model_config(self.on,self.on['imitation_checkpoint'])
        validate_config(fresh)
        self.assertNotIn(S.KEY,fresh['update'])
        self.assertEqual(fresh['update']['fixed_entropy'],E.FIXED_ENTROPY_005)
        self.assertEqual(fresh['insolvency'],dict(event_cost=.02,max_cost=.075))
        self.assertEqual(fresh['exploration'],.05)

    def test_current_LR_health_and_KL_guards_remain_independent(self):
        from brass_ai.recovery import lr_controller as L
        from test_ppo_lr_controller import measurements
        self.assertEqual(self.on['ppo_lr_controller'],self.original['signature']['config']['ppo_lr_controller'])
        cfg=self.on['ppo_lr_controller'];health=dict(contract='selfplay-economic-health/3.0.0',ok=True,reasons=[])
        m=measurements(economic_health=health,anchor_distance=999.,
            calibrated_anchor=dict(ok=False,anchor_distance=999.,reasons=['stale_anchor']))
        s=L.initial_state(cfg)
        for _ in range(2):s,_=L.decide(s,m)
        next_state,report=L.decide(s,m)
        self.assertEqual(report['decision'],'up');self.assertIsNone(report['anchor_distance'])
        self.assertEqual((next_state,report),L.decide(copy.deepcopy(s),m))
        bad=dict(m,economic_health=dict(health,ok=False,reasons=['vp_per_seat']))
        self.assertEqual(L.decide(s,bad)[1]['reason'],'health_check_failed')
        self.assertEqual(L.decide(s,dict(bad,kl=.02))[1]['decision'],'down')
        self.assertEqual(L.decide(s,dict(m,early_stop=True,steps_done=1))[1]['decision'],'down')

    def test_signed_helper_transitions_and_precommit_retry(self):
        from brass_ai.recovery.resume import check_signature,commit_transition,entropy_receipt_preserved
        state=copy.deepcopy(self.original)
        signature=dict(state['signature'],config=self.on,implementation=implementation_id())
        check_signature(state,signature,reviewed=True)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);write_json(run/'state.json',state)
            committed=commit_transition(run,state,signature)
            restored=E.restore(self.on,committed,state['exploration'])
            self.assertEqual(restored,self.start()['exploration'])
            self.assertEqual(S.configure(self.on,committed,'on',start='next'),self.on)
            # A second reviewed boundary toggle before PPO must replay step one,
            # without inventing a second checkpoint or returning to the old cost.
            off=S.configure(self.on,committed,'off')
            target=dict(signature,config=off)
            check_signature(committed,target,reviewed=True)
            toggled=commit_transition(run,committed,target)
            after=E.restore(off,toggled,state['exploration'])
            self.assertEqual(after[S.KEY]['progress'],0)
            self.assertEqual(after[S.KEY]['insolvency_steps_applied'],1)
            self.assertEqual(after[S.KEY]['applied'],dict.fromkeys(E.GROUPS,.005))
        at50=self.progress(50);at50['signature']['implementation']=implementation_id()
        step2=S.configure(self.on,at50,'insolvency-step')
        check_signature(at50,dict(at50['signature'],config=step2),reviewed=True)
        off=S.configure(self.on,at50,'off')
        check_signature(at50,dict(at50['signature'],config=off),reviewed=True)
        raw=dict(training=dict(signature=digest(at50['signature'])))
        self.assertTrue(entropy_receipt_preserved(at50,dict(at50['signature'],config=off),raw,at50['history'][-1]))
        self.assertFalse(entropy_receipt_preserved(at50,dict(at50['signature'],config=step2),raw,at50['history'][-1]))
        bad=copy.deepcopy(off);bad['exploration']=.08
        with self.assertRaises(ValueError):check_signature(at50,dict(at50['signature'],config=bad),reviewed=True)
        with self.assertRaises(ValueError):E.restore(self.off,self.start(),self.start()['exploration'])

    def test_generic_helper_read_only_due_check_and_pending_rejection(self):
        spec=importlib.util.spec_from_file_location('entropy_helper',SOURCE_ROOT/'scripts/configure-entropy-schedule.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0620.pt';checkpoint.write_bytes(b'committed fixture')
            state=self.progress(50);state['signature']['implementation']=implementation_id()
            state['history'][-1].update(path='artifacts/run/generation-0620.pt',generation=620,sha256=sha(checkpoint))
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes()
            output='configs/step5/step2.json'
            report=helper.generate(root,'artifacts/run',output,620,'insolvency-step')
            self.assertEqual(report,helper.generate(root,'artifacts/run',output,620,'insolvency-step'))
            self.assertEqual(report['entropy_schedule']['progress'],50)
            self.assertEqual(report['entropy_schedule']['insolvency_steps_applied'],2)
            self.assertEqual((run/'state.json').read_bytes(),before)
            with self.assertRaises(ValueError):helper.generate(root,'artifacts/run',output,619,'insolvency-step')
            pending=run/'iteration-0620';pending.mkdir();(pending/'game.txt').write_text('partial')
            with self.assertRaisesRegex(ValueError,'interrupted'):helper.generate(root,'artifacts/run',output,620,'insolvency-step')

    def test_real_checkpoint_reward_resume_preserves_weights_Adam_RNG_and_progress(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.recovery.resume import load_resume,commit_transition
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);state=copy.deepcopy(self.original);old=copy.deepcopy(state['signature']['config'])
            old['model_size']='small';state['signature']['config']=old
            model=FourSeatNet(ModelConfig.preset('small'));optimizer=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
            migrate(model,optimizer,reward_spec(old))
            p=next(model.parameters());p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
            model.utility_calibration=dict(ready=True,reward_id=digest(reward_spec(old)))
            path=run/'generation-0570.pt'
            identity=save_checkpoint(path,model,optimizer,step=570,training=dict(fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),
                signature=digest(state['signature']),exploration=state['exploration']))
            entry=dict(generation=570,path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity)
            state['history']=[entry];state['population']=[entry];state['evaluation_anchor']=entry
            write_json(run/'state.json',state)
            expected=copy.deepcopy((model.state_dict(),optimizer.state_dict(),torch.get_rng_state()))
            new=S.configure(g570(old),state,'on',start='next')
            signature=dict(state['signature'],config=new,implementation=implementation_id())
            loaded,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=True)
            self.assertTrue(changed)
            resumed=torch.optim.AdamW(loaded.parameters(),lr=old['learning_rate']);resume_optimizer(raw,resumed)
            self.assertTrue(_same_payload(expected,(loaded.state_dict(),resumed.state_dict(),torch.get_rng_state())))
            self.assertFalse(calibration_ready(loaded));migrate(loaded,resumed,reward_spec(new))
            self.assertTrue(_same_payload(expected,(loaded.state_dict(),resumed.state_dict(),torch.get_rng_state())))
            committed=commit_transition(run,state,signature)
            # Simulate interruption before g571: the complete checkpoint remains g570.
            loaded2,raw2,changed2=load_resume(SOURCE_ROOT,run,committed,signature,reviewed=True)
            self.assertFalse(changed2);resumed2=torch.optim.AdamW(loaded2.parameters(),lr=old['learning_rate'])
            resume_optimizer(raw2,resumed2);migrate(loaded2,resumed2,reward_spec(new))
            self.assertTrue(_same_payload(expected,(loaded2.state_dict(),resumed2.state_dict(),torch.get_rng_state())))
            ex=E.restore(new,committed,raw2['training']['exploration'])
            self.assertEqual((ex[S.KEY]['progress'],ex[S.KEY]['insolvency_steps_applied']),(0,1))
            self.assertEqual((committed['iteration'],committed['games_done']),(570,182400))

    def test_reward_migration_cache_refresh_and_small_PPG_smoke(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        from brass_ai.learning.batching import model_id
        from brass_ai.recovery import gae
        from test_ppg_g210 import prepare
        old=copy.deepcopy(self.original['signature']['config']);old['model_size']='small';old['update']['preparation_workers']=0
        old['value_phase'].update(batch_size=128,clone_rows=24,clone_batch_size=8)
        model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
        migrate(model,opt,reward_spec(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);phase=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            phase.local_scoring=Companion(model,old['local_scoring_auxiliary'],raw=dict(model_id=model_id(model)))
            prepare(model,phase,1,companion=True)
            phase.buffer.stage(phase.critic_dataset,phase.train_ids,phase.heldout_ids,phase.actor_ids,
                phase.collection_source,model,phase.split['selection'])
            phase.state=gae.initial(old['advantage'],1);phase.critic_state['generation']=1
            raw=dict(model_config=model.config.to_dict(),training={KEY:phase.payload(),
                'local_scoring_auxiliary':phase.local_scoring.payload(1)})
            write_json(run/'state.json',dict(iteration=1,signature=dict(config=old)));phase.buffer.committed(1)
            before=copy.deepcopy((model.state_dict(),opt.state_dict(),phase.optimizer.state_dict(),phase.state,torch.get_rng_state()))
            config=S.configure(g570(old),dict(iteration=1),'on',start='next')
            model._reviewed_cap_upgrade=True;change=migrate(model,opt,reward_spec(config))
            self.assertIsNotNone(change);self.assertFalse(calibration_ready(model))
            resumed=TrainingPhases(model,config,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(resumed.buffer.receipts,[])
            self.assertEqual(resumed.g210_migration['invalidated_generations'],[1])
            self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),resumed.optimizer.state_dict(),resumed.state,torch.get_rng_state())))
            with self.assertRaisesRegex(ValueError,'committed reward'):resumed.startup(1)
            write_json(run/'state.json',dict(iteration=1,signature=dict(config=config)))
            cleanup=resumed.startup(1);self.assertGreater(cleanup['deleted_bytes'],0)
            resumed.local_scoring=Companion(model,config['local_scoring_auxiliary'],raw=raw,generation=1)
            prepare(model,resumed,2,companion=True);report=resumed.fit(model)
            self.assertGreater(report['attempted_steps'],0);self.assertLessEqual(report['probe_kl'],.002)
            self.assertEqual([r['generation'] for r in resumed.buffer.pending],[2])
            print('entropy/reward isolated PPG smoke:',{k:report[k] for k in ('optimizer_steps','attempted_steps','probe_kl','early_stop_reason')})


if __name__=='__main__':unittest.main()
