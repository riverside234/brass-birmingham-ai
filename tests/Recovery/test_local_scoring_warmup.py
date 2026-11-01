"""Fresh companion heads advance only with committed, KL-safe PPG fitting."""
import copy
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import local_scoring_auxiliary as A,local_scoring_warmup as W,updates
from brass_ai.recovery.retention import _same_payload
from test_gae_value_phase import fixture
from test_local_scoring import annotate


def setup():
    model,opt,c,(_,data)=fixture(games=4);annotate(data)
    ds=updates.Dataset(data,'cpu');config=A.fresh_config(A.DEFAULT)
    return model,opt,ds,A.Companion(model,config,raw=dict(model_id=model_id(model)))


def fit_step(model,ds,a,ids=(0,1)):
    model.zero_grad(set_to_none=True);a.begin('ppg')
    ix,x,_=ds.value_batch(ids,model);_,loss=a.forward(model,x,ds,ix)
    if loss is not None:(loss*ds.critic_weights[ix]).mean().backward()
    a.step('ppg')


class AcceptedWarmup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_normal_hidden_zero_outputs_exact_main_and_rng(self):
        m,_,_,_=setup();before=copy.deepcopy(m.state_dict())
        rng=torch.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state()
        cfg=A.fresh_config(A.DEFAULT);a=A.Companion(m,cfg,raw=dict(model_id=model_id(m)))
        b=A.Companion(m,cfg,raw=dict(model_id=model_id(m)))
        self.assertTrue(_same_payload(before,m.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]))
        self.assertTrue(_same_payload(a.heads.state_dict(),b.heads.state_dict()))
        for head in (a.heads.slots,a.heads.links):
            self.assertTrue(head.global_projection.weight.any());self.assertTrue(head.entity_projection.weight.any())
            for layer in (head.owner,head.vp):
                self.assertFalse(layer.weight.any());self.assertFalse(layer.bias.any())
        self.assertFalse(any(o.state for o in a.optimizers.values()))
        self.assertEqual(a.config['optimizer']['learning_rate'],1e-4)

    def test_thresholds_ramp_empty_commits_and_restart(self):
        c=A.fresh_config(A.DEFAULT);s=W.initial()
        s=W.advance(s,c,1,31,2);self.assertEqual(W.scale(s,c),0.)
        s=W.advance(s,c,2,0,0);self.assertEqual(s['qualifying_iterations'],1)
        s=W.advance(s,c,3,1,1);self.assertEqual(W.scale(s,c),.2)
        for g,steps,scale in ((4,0,.2),(5,1,.4),(6,1,.6),(7,0,.6),(8,1,.8),(9,1,1.)):
            resumed=copy.deepcopy(s)
            s=W.advance(s,c,g,steps,int(steps>0))
            self.assertEqual(s,W.advance(resumed,c,g,steps,int(steps>0)))
            self.assertAlmostEqual(W.scale(s,c),scale)
        self.assertEqual(s['qualifying_iterations'],6);self.assertEqual(s['fitting_steps'],36)
        s=W.initial()
        for g in range(1,4):s=W.advance(s,c,g,1,1)
        self.assertEqual(W.scale(s,c),0.)  # iteration minimum alone is insufficient
        c.update(head_only_iterations=1,minimum_fitting_steps=1,ramp_iterations=2);A.validate_config(c)
        self.assertEqual(W.scale(W.advance(W.initial(),c,1,1,1),c),.5)
        for k in ('head_only_iterations','minimum_fitting_steps','ramp_iterations'):
            for bad in (0,True,1.5):
                with self.assertRaises(ValueError):A.validate_config(dict(c,**{k:bad}))
        with self.assertRaises(ValueError):W.advance(W.initial(),c,1,1,0)

    def test_encoder_and_descriptor_gradient_scale_heads_unscaled_and_preservation(self):
        from brass_ai.recovery.value_preservation import FrozenValues
        m,_,ds,a=setup();m.eval();fit_step(m,ds,a)
        ix,x,_=ds.value_batch([0,1],m)
        # Nonzero output weights make this a meaningful gradient isolation test.
        self.assertTrue(a.heads.slots.owner.weight.any())
        states=[W.initial()];s=W.advance(W.initial(),a.config,1,16,2)
        s=W.advance(s,a.config,2,16,2);states.append(s)
        for g in range(3,7):s=W.advance(s,a.config,g,1,1);states.append(s)
        def grads(preservation=False):
            m.zero_grad(set_to_none=True);a.begin('ppo' if preservation else 'ppg')
            loss=(frozen.losses(m,x,ix)['local'] if preservation else a.forward(m,x,ds,ix)[1])
            loss.mean().backward()
            return ({n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None},
                {n:p.grad.detach().clone() for n,p in a.heads.named_parameters() if p.grad is not None})
        for preservation in (False,True):
            if preservation:
                # Include all companion outputs; perturb a head after the snapshot.
                frozen=FrozenValues(m,ds,dict(microbatch_size=64,max_candidate_cells=131072,precision='fp32'),a)
                with torch.no_grad():a.heads.slots.owner.bias[0].add_(.2);a.heads.links.vp.bias.add_(.1)
            a.warmup=states[-1];main_full,head_full=grads(preservation)
            for prefix in ('state_encoder','entity_encoder','descriptor_encoder'):
                self.assertGreater(sum(float(v.abs().sum()) for n,v in main_full.items() if n.startswith(prefix)),0.)
            for state in states:
                a.warmup=state;main,heads=grads(preservation);factor=a.scale
                for n,g in main.items():torch.testing.assert_close(g,main_full[n]*factor,atol=2e-8,rtol=2e-4)
                for n,g in heads.items():torch.testing.assert_close(g,head_full[n],atol=1e-9,rtol=1e-5)
                if factor==0:self.assertFalse(any(g.any() for g in main.values()))

    def test_only_retained_ppg_counts_payload_preview_is_pure(self):
        m,_,ds,a=setup();start=a.snapshot()
        fit_step(m,ds,a);a.restore(start)
        self.assertEqual(a.fitting_steps,0);self.assertFalse(a.fitting_used)
        # PPO preservation can move head Adam without fitting terminal targets.
        a.begin('ppo');a.preservation_used(ds,[0,1])
        sum(p.square().sum() for p in a.heads.parameters()).backward();a.step('ppo')
        self.assertEqual(a.retained_steps['ppo'],1);self.assertEqual(a.fitting_steps,0)
        a.committed(1);self.assertEqual(a.warmup,W.advance(W.initial(),a.config,1,0,0))
        for _ in range(16):fit_step(m,ds,a)
        # Rejected final work is rolled back with Adam, rows AND evidence.
        kept=a.snapshot();fit_step(m,ds,a);a.restore(kept)
        first=a.payload(2);self.assertEqual(a.generation,1);self.assertEqual(a.warmup['fitting_steps'],0)
        a.committed(2);self.assertEqual(a.payload(2)['warmup'],first['warmup']);self.assertEqual(a.scale,0.)
        a=A.Companion(m,a.config,raw=dict(training={A.KEY:first}),generation=2)
        for _ in range(16):fit_step(m,ds,a)
        report=a.report()['warmup'];self.assertEqual(report['retained_fitting_steps'],16)
        self.assertEqual(report['distinct_fitting_games'],len({ds.data[i][2][0] for i in (0,1)}))
        next_payload=a.payload(3);self.assertEqual(a.scale,0.);self.assertEqual(next_payload['warmup']['encoder_stage'],1)
        resumed=A.Companion(m,a.config,raw=dict(training={A.KEY:next_payload}),generation=3)
        a.committed(3);self.assertEqual(a.scale,.2);self.assertTrue(_same_payload(a.payload(3),resumed.payload(3)))
        self.assertEqual(a.retained_steps['ppg'],32);self.assertEqual(a.warmup['fitting_steps'],32)
        with self.assertRaisesRegex(ValueError,'restricted to PPG'):
            a.begin('ppo');ix,x,_=ds.value_batch([0,1],m);a.forward(m,x,ds,ix)
        bad=copy.deepcopy(next_payload);bad.pop('warmup');bad['identity']=A.bundle_id({k:v for k,v in bad.items() if k!='identity'})
        with self.assertRaises(ValueError):A.Companion(m,a.config,raw=dict(training={A.KEY:bad}),generation=3)

    def test_missing_and_zero_weight_targets_do_not_advance(self):
        m,_,ds,a=setup();ds.critic_weights.zero_();fit_step(m,ds,a)
        self.assertEqual(a.fitting_steps,0);self.assertFalse(a.optimizers['ppg'].state)
        ds.critic_weights.fill_(1.)
        for row in ds.rows:row.pop('_local_scoring',None)
        fit_step(m,ds,a);a.committed(1)
        self.assertEqual(a.warmup['qualifying_iterations'],0);self.assertEqual(a.scale,0.)

    def test_bounded_diagnostics_occupied_slot_and_link_metrics(self):
        m,_,ds,a=setup();before=copy.deepcopy(m.state_dict())
        report=a.diagnostics(m,ds,range(len(ds.data)),maximum=4)
        self.assertEqual(report['rows'],4);self.assertTrue(_same_payload(before,m.state_dict()))
        self.assertEqual({k.rsplit('/',1)[-1] for k in report['groups']},{'slots','links'})
        for g in report['groups'].values():
            self.assertGreater(g['occupied']['entities'],0)
            self.assertAlmostEqual(g['owner_ce'],np.log(5),places=6)
            self.assertAlmostEqual(g['occupied']['owner_ce'],np.log(5),places=6)
            self.assertTrue(np.isfinite(g['occupied']['vp_mae']))
        self.assertEqual(a.fitting_steps,0)

    def test_delayed_kl_and_loss_rejection_keep_exact_work_and_lr_factors(self):
        from test_training_g1120_kl32 import Revision
        # Existing production two-age fit exercises loss16/KL32 rollback. Run
        # both keep-16 and keep-zero cases with the new companion contract.
        import test_training_g1120_kl32 as suite
        from brass_ai.recovery import value_phase as V,ppg_g210 as G
        old=suite.Companion;seen=[]
        def companion(model,cfg,*args,**kwargs):
            a=old(model,A.fresh_config(cfg),*args,**kwargs);seen.append(a);return a
        with patch.object(suite,'Companion',side_effect=companion):
            for kept in (16,0):
                Revision().guard_case(.001 if kept else .003,kept)
                a=seen[-1];self.assertEqual(a.fitting_steps,kept)
                self.assertEqual(a.report()['warmup']['after_commit']['qualifying_iterations'],int(kept>0))
                if kept:
                    self.assertAlmostEqual(a.optimizers['ppg'].param_groups[0]['lr'],1e-4)
                else:self.assertFalse(a.optimizers['ppg'].state)


if __name__=='__main__':unittest.main()
