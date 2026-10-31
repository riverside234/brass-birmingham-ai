"""Production-shaped admission, GPU-write and PPG evidence regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import torch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import updates,value_phase
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.domino_auxiliary import Diversity,PPGCompanions,row_key
from brass_ai.recovery.retention import _same_payload
from test_domino import fixture,collect
from test_domino_audit import ready


class Review(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def prepared(self,run):
        m,o,c=fixture();p=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
        p.diversity=Diversity(m,c['diversity'],regime='g1330-review');ready(p.diversity)
        _,rows=collect(m,p,1,games=42);ds=updates.Dataset(rows,'cpu');p.prepare(ds)
        return m,o,c,p,ds

    def test_snapshot_weights_exact_without_rowwise_device_writes(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            _,_,_,p,ds=self.prepared(Path(tmp))
            for weight in (0.,.5,1.):
                p.critic_state['snapshot_weight']=weight;p.critic_state['snapshots_active']=True
                writes=[];original=torch.Tensor.__setitem__
                def write(t,key,value):
                    writes.append(key);return original(t,key,value)
                before=torch.get_rng_state().clone()
                with patch.object(torch.Tensor,'__setitem__',write):p.prepare(ds)
                expected=torch.tensor([weight if role=='snapshot' else 1. for role in p.critic_dataset.roles])
                torch.testing.assert_close(p.critic_dataset.critic_weights,expected,rtol=0,atol=0)
                self.assertEqual(writes,[]);self.assertTrue(torch.equal(before,torch.get_rng_state()))
                self.assertTrue(set(p.train_ids).isdisjoint(p.heldout_ids))
                if weight==0:self.assertTrue(all(p.critic_dataset.roles[i]=='learner' for i in p.train_ids))
                self.assertEqual(set(p.split['preparation_seconds']),{'dataset','weights','split','local_inventory','diversity'})

    def test_intrinsic_loss_is_not_reported_as_local_head_loss(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            m,_,c,p,_=self.prepared(Path(tmp));ds=p.critic_dataset;d=p.diversity
            ids=[i for i in p.train_ids if row_key(ds,i) in d.train_keys][:8]
            ix,inputs,_=ds.value_batch(ids,m)
            with torch.no_grad():d.head[-1].bias.fill_(.2)
            local=Mock()
            local.forward.side_effect=lambda *args,**kwargs:(updates.value_output(*args,**kwargs),torch.full((len(ids),),.25))
            wrapper=PPGCompanions(local,d,p.optimizer,.5);wrapper.batch(ds,ids)
            _,combined=wrapper.forward(m,inputs,ds,ix)
            self.assertTrue(bool((combined>.25).all()))
            weights=torch.ones(len(ids));wrapper.account_loss(combined,weights,len(ids),'ppg')
            reported=local.account_loss.call_args.args[0]
            torch.testing.assert_close(reported,torch.full_like(reported,.25),rtol=0,atol=0)
            self.assertFalse(reported.requires_grad)
            combined.mean().backward();self.assertGreater(float(d.head[-1].bias.grad.abs().sum()),0.)

    def test_rejected_unchecked_interval_reports_unknown_kl_and_zero_retained_fit(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            m,_,c,p,ds=self.prepared(Path(tmp));d=p.diversity;critic=p.critic_dataset
            wrapper=PPGCompanions(None,d,p.optimizer,.5)
            original=value_phase.metrics;calls=[]
            def reject(*args,**kwargs):
                out=original(*args,**kwargs);calls.append(out)
                if len(calls)==3:out=dict(out,loss=out['loss']+100.)
                return out
            before=copy.deepcopy((m.state_dict(),p.optimizer.state_dict(),d.snapshot()))
            actor=[i for i in p.learner_train_ids if not ds.forced_cpu[i]]
            cfg=dict(c['value_phase'],learning_rate=1e-5,batch_size=128,max_epochs=1,acceptance_steps=1,kl_check_steps=32)
            with patch.object(value_phase,'metrics',side_effect=reject):
                report=value_phase.fit(m,p.optimizer,ds,p.learner_train_ids,p.heldout_ids,actor,actor[:8],cfg,c['update'],beta=1.,
                    value_dataset=critic,value_train_ids=p.train_ids,validation_dataset=critic,companion=wrapper)
            self.assertEqual(report['attempted_steps'],1);self.assertEqual(report['optimizer_steps'],0)
            self.assertEqual(report['early_stop_reason'],'heldout_loss_increased')
            self.assertIsNone(report['maximum_attempted_probe_kl']);self.assertEqual(report['attempted_probe_kl_checks'],0)
            self.assertEqual(report['probe_kl'],0.);self.assertEqual(report['kl_checks'],[])
            self.assertTrue(_same_payload(before,(m.state_dict(),p.optimizer.state_dict(),d.snapshot())))
            fit=wrapper.report();self.assertEqual(fit['attempted_steps'],1);self.assertGreater(fit['attempted_distinct_rows'],0)
            self.assertEqual(fit['retained_steps'],0);self.assertEqual(fit['retained_distinct_rows'],0)
            self.assertEqual(fit['status'],'no_retained_fit')
            d.diagnostics(m,critic,p.heldout_ids,c['update']);held=d.report['intrinsic_heldout']
            self.assertEqual(held['prediction_rms'],0.);self.assertGreater(held['target_rms'],0.)

    def test_retained_fit_accounting_follows_exact_selected_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            m,_,_,p,_=self.prepared(Path(tmp));d=p.diversity;ds=p.critic_dataset
            ids=[i for i in p.train_ids if row_key(ds,i) in d.train_keys][:8]
            wrapper=PPGCompanions(None,d,p.optimizer,.5)
            for part in (ids[:4],ids[2:]):
                wrapper.begin('ppg');wrapper.batch(ds,part);ix,inputs,_=ds.value_batch(part,m)
                _,loss=wrapper.forward(m,inputs,ds,ix);loss.mean().backward();wrapper.step('ppg')
                if part==ids[:4]:saved=wrapper.snapshot()
            before=torch.get_rng_state().clone();wrapper.restore(saved)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            fit=wrapper.report()
            self.assertEqual((fit['attempted_steps'],fit['retained_steps']),(2,1))
            self.assertEqual((fit['attempted_presentations'],fit['retained_presentations']),(10,4))
            self.assertEqual((fit['attempted_distinct_rows'],fit['retained_distinct_rows']),(8,4))
            self.assertEqual(fit['status'],'retained_fit')

    def test_live_g1330_config_has_a_narrow_source_only_continuation(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        path=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1330-20261031T024940047516Z/state.json'
        state=json.loads(path.read_text());before=copy.deepcopy(state)
        self.assertEqual(state['iteration'],1330)
        target=dict(state['signature'],implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True));self.assertEqual(state,before)
        self.assertEqual(target['config']['diversity']['adoption']['generation'],1326)
        self.assertEqual(set(target['config']['update']['fixed_entropy']['weights'].values()),{.10})
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,target)
        changed=copy.deepcopy(target);changed['config']['exploration']=.079
        with self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)


if __name__=='__main__':unittest.main()
