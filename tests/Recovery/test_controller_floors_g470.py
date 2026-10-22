"""Anchor-event entropy floors and recoverable zero snapshot loss, without resets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import critic_data as C,exploration as E,snapshot_r2 as R,value_phase
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import latest_model_config,validate_config
from brass_ai.recovery.retention import _same_payload
from test_snapshot_r2 import observe,value_report

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0470-20261022T011816523228Z'


class Floors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.state=read(EXPORT/'state.json');self.old=self.state['signature']['config']
        self.cfg=validate_config(C.floor_config(E.anchor_floor_config(self.old,self.state)))

    def test_exact_config_defaults_and_strict_validation(self):
        self.assertEqual({k for k in self.cfg if self.cfg[k]!=self.old[k]}, {'critic_data','entropy_control'})
        self.assertEqual(C.floor_config(E.anchor_floor_config(self.cfg,self.state)),self.cfg)
        self.assertEqual(self.cfg['critic_data']['rollout']['reduced_snapshot_weight'],0.)
        fresh=latest_model_config(self.old,self.old['imitation_checkpoint'])
        self.assertEqual(fresh['entropy_control'],E.ANCHOR_ENTROPY)
        self.assertEqual({k:v for k,v in fresh['entropy_control'].items() if k!='promotion_origin'},
                         {k:v for k,v in self.cfg['entropy_control'].items() if k!='promotion_origin'})
        self.assertEqual(fresh['critic_data'],C.FIXED_DEFAULT)  # g503; archived zero-floor migration stays exact.
        for key,value in (('floor_drop_per_anchor_promotion',.03),('floor_decay_per_iteration',.999)):
            bad=copy.deepcopy(self.cfg);bad['entropy_control'][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        for floor in (.1,-.1,True):
            bad=copy.deepcopy(self.cfg);bad['critic_data']['rollout']['reduced_snapshot_weight']=floor
            with self.assertRaises(ValueError):validate_config(bad)
        # Archived signed configurations retain their old meanings.
        validate_config(self.old)
        self.assertEqual(E.group_floor_config(self.old),self.old)

    def promoted(self,anchor,index):
        result=copy.deepcopy(anchor)
        previous=result['events'][-1]['reference_generation'] if result['events'] else 0
        result['events'].append(dict(decision='promote',generation=470+index,reference_generation=previous+1,
                                    evidence=dict(points_delta_ci95=[1.,2.])))
        return result

    def test_entropy_starts_at_current_floors_without_any_iteration_decay(self):
        before=copy.deepcopy(self.state)
        saved=E.restore(self.cfg,self.state,copy.deepcopy(self.state['exploration']))
        self.assertEqual(saved['floor_anchor_promotions'],0)
        for k in self.state['exploration']:
            if k!='contract':self.assertEqual(saved[k],self.state['exploration'][k])
        initial=E.floors(self.old,self.state['exploration'],470)
        self.assertEqual(E.floors(self.cfg,saved,470),initial)
        measurement=dict(groups={g:dict(raw_entropy=v,normalized_entropy=.3) for g,v in initial.items()})
        anchor=self.state['policy_anchor']
        first,report=E.advance(saved,self.cfg,measurement,generation=471,warmup_done=True,healthy=True,policy_anchor=anchor)
        for g,v in initial.items():self.assertEqual(report['groups'][g]['floor'],v)
        resumed=E.restore(self.cfg,dict(iteration=471,exploration=first,policy_anchor=anchor),json.loads(json.dumps(first)))
        self.assertEqual(resumed,first)
        self.assertEqual(E.advance(first,self.cfg,measurement,generation=472,warmup_done=True,healthy=True,policy_anchor=anchor),
                         E.advance(resumed,self.cfg,measurement,generation=472,warmup_done=True,healthy=True,policy_anchor=anchor))
        self.assertEqual(E.floors(self.cfg,saved,100000),initial)
        self.assertEqual(self.state,before)
        with self.assertRaises(ValueError):E.restore(self.old,dict(iteration=471,exploration=first),first)

    def test_anchor_promotions_drop_absolute_two_percent_once_and_clamp_by_group(self):
        saved=E.restore(self.cfg,self.state,copy.deepcopy(self.state['exploration']))
        baseline=copy.deepcopy(saved['baseline']);groups=copy.deepcopy(saved['groups'])
        anchor=self.state['policy_anchor']
        for i in range(1,9):
            anchor=self.promoted(anchor,i)
            saved,transition=E.anchor_floor_transition(self.cfg,saved,anchor)
            self.assertEqual(saved['floor_anchor_promotions'],i)
            for g,r in E.floor_ratios(self.cfg,saved).items():
                start=.95 if g.endswith('_action') else .90
                minimum=.85 if g.endswith('_action') else .80
                self.assertEqual(r,max(minimum,round(start-.02*i,12)))
                self.assertEqual(E.floors(self.cfg,saved,500)[g],baseline[g]*r)
            self.assertEqual(E.anchor_floor_transition(self.cfg,saved,anchor),(saved,None))
            self.assertTrue(transition['baseline_preserved'])
        self.assertEqual(saved['baseline'],baseline);self.assertEqual(saved['groups'],groups)
        with self.assertRaisesRegex(ValueError,'backwards'):
            E.sync_anchor_floor(self.cfg,saved,self.state['policy_anchor'])

    def test_pending_promotion_before_first_new_checkpoint_survives_restart(self):
        # State has published the promotion, but the model/entropy checkpoint
        # still predates adoption. Its signed config origin cannot move forward.
        anchor=self.promoted(self.state['policy_anchor'],0)
        state=dict(self.state,policy_anchor=anchor)
        before=copy.deepcopy(state);checkpoint=copy.deepcopy(state['exploration'])
        restored=E.restore(self.cfg,state,checkpoint)
        self.assertEqual(restored['floor_anchor_promotions'],1)
        self.assertEqual(E.restore(self.cfg,state,checkpoint),restored)
        self.assertEqual(state,before)
        self.assertEqual(E.anchor_floor_config(self.cfg,state),self.cfg)
        invalid=copy.deepcopy(anchor);invalid['events'][-1]['decision']='hold'
        with self.assertRaisesRegex(ValueError,'Unqualified'):
            E.restore(self.cfg,dict(state,policy_anchor=invalid),checkpoint)
        invalid=copy.deepcopy(anchor);invalid['events'][0]['reason']='tampered'
        with self.assertRaisesRegex(ValueError,'signed origin'):
            E.restore(self.cfg,dict(state,policy_anchor=invalid),checkpoint)

    def test_fresh_entropy_waits_for_baseline_and_health_remains_required(self):
        cfg=E.configure(self.cfg);s=E.initial(cfg,0)
        measurement=dict(groups={g:dict(raw_entropy=2.,normalized_entropy=.3) for g in E.GROUPS})
        self.assertEqual(E.floors(cfg,s,0),dict.fromkeys(E.GROUPS))
        for g in range(1,11):s,report=E.advance(s,cfg,measurement,generation=g,warmup_done=True,healthy=True)
        self.assertEqual(s['armed_at'],10)
        self.assertEqual(E.floors(cfg,s,10),{g:2*cfg['entropy_control']['floor_ratio_by_group'][g] for g in E.GROUPS})
        low=dict(groups={g:dict(raw_entropy=.1,normalized_entropy=.1) for g in E.GROUPS})
        s2,r=E.advance(s,cfg,low,generation=11,warmup_done=True,healthy=False)
        self.assertEqual([x['alpha'] for x in s2['groups'].values()],[x['alpha'] for x in s['groups'].values()])
        self.assertTrue(all(x['reason']=='unhealthy' for x in r['groups'].values()))

    def test_real_qualified_anchor_publication_and_pending_restore_apply_once(self):
        from brass_ai.step5.league import write_json
        from brass_ai.recovery import policy_anchor as A,panel_gates
        from test_policy_anchor import evaluated,measurement
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);state,cfg,_,_,_=evaluated(root)
            cfg=E.group_floor_config(E.frozen_config(cfg))
            saved=E.initial(cfg,0)
            measure=dict(groups={g:dict(raw_entropy=2.,normalized_entropy=.3) for g in E.GROUPS})
            for generation in range(1,11):saved,_=E.advance(saved,cfg,measure,generation=generation,warmup_done=True,healthy=True)
            saved['generation']=30;state['exploration']=copy.deepcopy(saved)
            cfg=E.anchor_floor_config(cfg,state)
            state['ratings']['panel_gates']=panel_gates.prepare(root,state,cfg)
            write_json(root/'state.json',state)
            restored=E.restore(cfg,state,saved);model=torch.nn.Linear(1,1)
            class Probe:
                def measure(self,*args,**kw):return dict(measurement(.2),original_il=measurement(1.1))
            with patch.object(A,'cache',return_value=dict(path='cache.json',sha256='a'*64)),\
                    patch.object(A,'values',return_value=None),patch.object(A,'verify_active'):
                _,decision=A.prepare(root,root,state,cfg,Probe(),model)
                self.assertEqual(decision['decision'],'promote')
                after,transition=E.anchor_floor_transition(cfg,restored,state['policy_anchor'])
                self.assertEqual(transition['floor_ratio_after']['rail_action'],.93)
                self.assertEqual(transition['floor_ratio_after']['rail_other'],.88)
                # Disk state still contains the old entropy checkpoint, but
                # the published anchor event is sufficient to replay the drop.
                restarted=E.restore(cfg,read(root/'state.json'),saved)
                self.assertEqual(restarted,after)
                self.assertEqual(restarted['baseline'],saved['baseline'])
                self.assertEqual(restarted['baseline_rows'],saved['baseline_rows'])
                self.assertEqual(restarted['groups'],saved['groups'])
                _,again=A.prepare(root,root,state,cfg,Probe(),model)
                self.assertEqual(again['decision'],'hold')
                self.assertEqual(E.anchor_floor_transition(cfg,after,state['policy_anchor']),(after,None))

    def test_zero_floor_cut_and_supported_learner_only_recovery(self):
        c=self.cfg['critic_data'];s=C.initial(c)
        s.update(generation=10,snapshots_active=True,snapshot_weight=.5,activation_generation=1,
                 weight_recovery=R.initial(10,.5))
        regime={'source':'ppo'};s['weight_recovery']['regime']=regime
        ref=R.reference(R.summarize([observe(g) for g in range(1,11)]),10,.5,regime)
        s['weight_recovery']['reference']=copy.deepcopy(ref)
        for g in range(11,21):
            s,r=C.advance(s,c,generation=g,report=value_report(g,bias=1.),healthy=True,
                          fingerprint_now=None,ppo_snapshot_rows=10,regime=regime)
        self.assertEqual(s['snapshot_weight'],0.)
        self.assertEqual(r['decision'],'reduce_snapshot_weight')
        self.assertEqual(s['weight_recovery']['reference'],ref)
        self.assertEqual(s['weight_recovery']['entries'],[])
        for g in range(21,35):
            s,r=C.advance(s,c,generation=g,report=value_report(g),healthy=True,
                          fingerprint_now=None,ppo_snapshot_rows=0,regime=regime)
            self.assertFalse(r['snapshot_rows_used']);self.assertTrue(r['zero_weight_learner_observation'])
            if g<34:self.assertEqual(s['snapshot_weight'],0.)
        self.assertEqual(s['snapshot_weight'],.5)
        self.assertEqual(r['decision'],'increase_snapshot_weight')
        # Above zero, ordinary work alone cannot pretend snapshot exposure.
        s,r=C.advance(s,c,generation=35,report=value_report(35),healthy=True,
                      fingerprint_now=None,ppo_snapshot_rows=0,regime=regime)
        self.assertFalse(r['weight_recovery']['current_qualifying'])

    def test_zero_weight_without_training_or_with_failed_health_cannot_recover(self):
        c=self.cfg['critic_data'];s=C.initial(c)
        s.update(snapshots_active=True,snapshot_weight=0.,weight_recovery=R.initial(0,0.))
        for g in range(1,16):
            s,r=C.advance(s,c,generation=g,report=value_report(g,trained=False),healthy=True,
                          fingerprint_now=None,regime={'source':'ppo'})
        self.assertEqual(s['snapshot_weight'],0.);self.assertIsNone(s['weight_recovery']['reference'])
        self.assertEqual(r['weight_recovery']['learner']['qualifying_iterations'],0)

    def test_checkpoint_option_upgrade_keeps_weight_evidence_adam_and_rng(self):
        model=torch.nn.Linear(2,2);phase=TrainingPhases(model,self.old)
        phase.critic_state.update(snapshot_weight=.5,snapshots_active=True,weight_recovery=R.initial(0,.5))
        model(torch.ones(1,2)).sum().backward();phase.optimizer.step();phase.optimizer.zero_grad()
        saved=phase.payload();before=copy.deepcopy(saved);rng=torch.get_rng_state().clone()
        restored=TrainingPhases(model,self.cfg,raw={'training':{KEY:saved}},generation=0)
        expected=copy.deepcopy(before);expected['options']=C.floor_config(expected['options'])
        self.assertTrue(_same_payload(restored.payload(),expected))
        self.assertTrue(_same_payload(saved,before));self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(restored.critic_state['snapshot_weight'],.5)
        self.assertEqual(restored.snapshot_floor_migration['floor_after'],0.)
        again=TrainingPhases(model,self.cfg,raw={'training':{KEY:restored.payload()}},generation=0)
        self.assertIsNone(again.snapshot_floor_migration)
        self.assertTrue(_same_payload(restored.payload(),again.payload()))

    def test_real_ppo_zero_weight_excludes_snapshot_training_without_disabling_gae(self):
        from test_critic_descriptors import mixed_fixture
        from test_gae_value_phase import ppo
        from test_ppg_window import upgraded
        model,opt,cfg,behavior,data,snap,_=mixed_fixture()
        cfg=C.floor_config(value_phase.disabled_options(upgraded(cfg)))
        phase=TrainingPhases(model,cfg);phase.snapshot_data=snap
        phase.critic_state.update(snapshot_weight=0.,weight_recovery=R.initial(0,0.))
        report=ppo(model,opt,cfg,behavior,data,phase)
        self.assertGreater(report['optimizer_steps'],0)
        self.assertGreater(report['critic_rows']['learner_distinct_used'],0)
        self.assertEqual(report['critic_rows']['snapshot_distinct_used'],0)
        self.assertTrue(all(phase.critic_dataset.roles[i]=='learner' for i in phase.train_ids))
        value=phase.fit(model);self.assertFalse(value['enabled']);self.assertGreater(value['critic_training']['optimizer_steps'],0)
        phase.finish(generation=1,update=report,anchor_gate={'ok':True},rating={'ok':True})
        self.assertEqual(phase.state['generation'],1)


if __name__=='__main__':unittest.main()
