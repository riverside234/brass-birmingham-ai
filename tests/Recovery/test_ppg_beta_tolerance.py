"""Adaptive beta retries use paired-game tolerance without accepting unsafe policy states."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import dcpg_config as D,ppg_auxiliary as A
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.step5.league import write_json
from test_ppg_eight import setup,stage_only


class BetaTolerance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model,_,cls.cfg=setup();cls.cfg=D.beta_options(D.configure(cls.cfg))
        cls.cfg['value_phase']['batch_size']=128
        cls.phase=TrainingPhases(cls.model,cls.cfg,root=SOURCE_ROOT,run=Path(cls.tmp.name))
        for g in range(1,9):
            stage_only(cls.model,cls.phase,g)
            if g<8:
                write_json(Path(cls.tmp.name)/'state.json',dict(iteration=g))
                cls.phase.buffer.committed(g)
        cls.weights=copy.deepcopy(cls.model.state_dict())
        cls.adam=copy.deepcopy(cls.phase.optimizer.state_dict())

    def run_phase(self,standard_errors,*,metric='age0',beta=1.,adaptive=True):
        self.model.load_state_dict(self.weights);self.phase.optimizer.load_state_dict(self.adam)
        cfg=copy.deepcopy(self.cfg['value_phase'])
        if not adaptive:
            cfg['contract']=D.CONTRACT;cfg.pop('beta_control')
        checks=[];unsafe=[False];real_probe=A.Reference.probes
        def probes(ref,model):
            result=real_probe(ref,model);checks.append(result)
            if len(checks)==1:
                result[metric]=.003;unsafe[0]=True
            return result
        def losses(model,ds,ids,update):
            if not unsafe[0]:return {i:1. for i in ids}
            unsafe[0]=False
            games=sorted({ds.data[i][2][0] for i in ids})
            noise=np.array([-.03125 if j%2 else .03125 for j in range(len(games))])
            noise-=noise.mean()
            se=float(noise.std(ddof=1)/np.sqrt(len(games)))
            changes=dict(zip(games,noise+standard_errors*se))
            return {i:1.+changes[ds.data[i][2][0]] for i in ids}
        def short_intervals(train,config,cap):
            ids=[i for rows in train.values() for i in rows]
            yield 0,0,[ids[:8],ids[8:16]],True
        with (patch.object(A,'losses',losses),patch.object(A.Reference,'probes',probes),
              patch.object(A,'intervals',short_intervals)):
            report=A.fit(self.model,self.phase.optimizer,self.phase.buffer,self.phase.critic_dataset,
                self.phase.train_ids,self.phase.heldout_ids,cfg,self.cfg['update'],beta=beta,
                snapshot_weight=1.,dropout=None,run_phase=True,ppo_work=self.phase.ppo_work,
                epoch_cap=5,ppo_peak=D.rate_receipt(6e-6,8,48))
        self.assertLessEqual(report['probe_kl'],.002)
        self.assertLessEqual(report['age0_probe_kl'],.002)
        self.assertTrue(all(c['safe'] for c in report['retention_checks']))
        return report,next(i for i in report['interval_reports'] if i['reason']=='policy_kl_cap')

    def test_small_increase_between_one_and_one_point_five_SE_retries_both_guards(self):
        for metric in ('overall','age0'):
            with self.subTest(metric=metric):
                r,trip=self.run_phase(1.25,metric=metric)
                paired=trip['unsafe_paired']
                self.assertGreater(paired['mean_change'],paired['standard_error'])
                self.assertLess(paired['mean_change'],1.5*paired['standard_error'])
                self.assertFalse(paired['improved']);self.assertFalse(paired['reject'])
                self.assertTrue(trip['retry']);self.assertEqual(r['beta_retries'],1)
                self.assertEqual(r['beta_clone_final'],2.);self.assertGreater(r['optimizer_steps'],0)
                self.assertEqual(D.beta_after_phase(r,self.cfg['value_phase'])[0],2.)

    def test_loss_above_allowance_stops_without_beta_increase(self):
        r,trip=self.run_phase(1.75)
        self.assertTrue(trip['unsafe_paired']['reject']);self.assertFalse(trip['retry'])
        self.assertEqual(r['beta_clone_final'],1.);self.assertEqual(r['optimizer_steps'],0)
        self.assertEqual(r['early_stop_reason'],'policy_kl_cap_heldout_loss_rejected')

    def test_unchanged_mean_loss_is_eligible(self):
        r,trip=self.run_phase(0.)
        self.assertAlmostEqual(trip['unsafe_paired']['mean_change'],0.)
        self.assertTrue(trip['retry']);self.assertEqual(r['beta_clone_final'],2.)

    def test_maximum_beta_still_stops(self):
        r,trip=self.run_phase(1.25,beta=32.)
        self.assertFalse(trip['retry']);self.assertEqual(r['beta_clone_final'],32.)
        self.assertEqual(r['early_stop_reason'],'policy_kl_cap_beta_max')

    def test_archived_strict_improvement_recipe_is_unchanged(self):
        r,trip=self.run_phase(1.25,adaptive=False)
        self.assertFalse(trip['unsafe_paired']['reject']);self.assertFalse(trip['retry'])
        self.assertEqual(r['beta_clone_final'],1.)
        self.assertEqual(r['early_stop_reason'],'policy_kl_cap_no_value_improvement')


if __name__=='__main__':unittest.main()
