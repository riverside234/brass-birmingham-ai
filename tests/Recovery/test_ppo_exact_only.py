"""2048-row exact control, diagnostic-only sampled KL and preserved safety checks."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppo_kl_control as K,updates,lr_controller as LR
from brass_ai.recovery.config import validate_config,update_config,latest_model_config
from brass_ai.recovery.review_metrics import CategoricalProbe
from test_categorical_kl_guard import GuardTests,probe_fixture
import test_ppo_lr_controller as lr_fixtures


class ExactOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_narrow_versioned_config_and_fresh_defaults(self):
        old=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0650-20261023T202940235880Z/state.json')['signature']['config']
        new=K.configure(old);validate_config(new)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'update'})
        self.assertEqual(new['update'],dict(old['update'],kl_control=K.DEFAULT))
        self.assertEqual(new,K.configure(new));self.assertFalse(K.active(old['update']))
        self.assertTrue(K.active(latest_model_config(old,old['imitation_checkpoint'])['update']))
        for block in (dict(K.DEFAULT,sampled_kl='stop'),dict(K.DEFAULT,probe_rows=1024),dict(K.DEFAULT,extra=True)):
            with self.assertRaises(ValueError):update_config(dict(new['update'],kl_control=block))
        for field,value in (('actor_nonforced',False),('kl_probe_rows',1024)):
            with self.assertRaises(ValueError):update_config(dict(new['update'],**{field:value}))

    def test_2048_fixed_nonforced_rows_sum_complete_action_distributions(self):
        model,ds,b,c,initial=probe_fixture((3,3,3,1)*700,microbatch=128)
        del initial
        eligible=(~ds.forced).nonzero().flatten().tolist()
        ids=updates.phase_probe(ds,eligible,2048)
        self.assertEqual(len(ids),2048);self.assertEqual(len(set(ids)),2048)
        self.assertTrue(all(len(ds.rows[i]['choices'])>1 for i in ids))
        c=dict(c,kl_control=K.DEFAULT);rng=torch.get_rng_state().clone()
        probe=CategoricalProbe(model,ds,ids,b,c)
        frozen=[p.clone() for _,p in probe.parts]
        self.assertAlmostEqual(probe.measure_step(model)['categorical_kl'],0.,places=7)
        with torch.no_grad():model.logits.copy_(torch.log(torch.tensor([.2,.79,.01])))
        result=probe.measure_step(model);expected=0.
        for j,(_,old) in enumerate(probe.parts):
            _,_,ix,inputs,q=probe._batch(j)
            with torch.no_grad():p=updates.row_probabilities(ds,ix,model(*inputs)[0],q,b)
            a=old.double().numpy();n=p.double().numpy()
            expected+=np.sum(a*(np.log(np.maximum(a,1e-30))-np.log(np.maximum(n,1e-30))))
            self.assertTrue(torch.equal(old,frozen[j]))
        self.assertEqual(result['positions'],2048)
        self.assertAlmostEqual(result['categorical_kl'],expected/2048,places=6)
        self.assertGreater(result['categorical_kl'],.1)
        self.assertAlmostEqual(result['sampled_kl'],0.,places=7)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_sampled_kl_cannot_stop_or_trigger_lr_cut_but_exact_kl_can(self):
        h=GuardTests()
        report,opt,_=h.run_update(exact=.001,sampled=.10,full=.10,exact_only=True)
        self.assertEqual(report['optimizer_steps'],8);self.assertFalse(report['kl_early_stop'])
        self.assertEqual(report['kl_guard']['contract'],K.CONTRACT)
        self.assertFalse(report['kl_guard']['sampled_kl_stop_enabled'])
        self.assertEqual(report['epochs'][-1]['post_update_kl'],.10)
        cfg=K.configure(lr_fixtures.config())
        report['categorical_probe']['sampled_kl']=100.
        with patch.object(LR,'collection_signature',return_value='a'*64),patch.object(
            LR,'game_health',return_value=dict(selfplay_vp_per_seat=100.,zero_vp_fraction=0.)):
            result=LR.finish_iteration(opt,report,cfg,{},[],None,{},'anchor')
        self.assertEqual(result['kl'],report['categorical_probe']['categorical_kl'])
        self.assertEqual(result['decision'],'hold');self.assertEqual(result['kl_signal'],'exact-categorical-probe')
        self.assertEqual(result['requested_probe_positions'],2048);self.assertFalse(result['sampled_kl_stop_enabled'])
        invalid=copy.deepcopy(report);invalid['kl_guard']['stop_reason']='sampled_full_dataset'
        with self.assertRaisesRegex(ValueError,'incompatible sampled-KL stop'):
            LR.finish_iteration(opt,invalid,cfg,{},[],None,{},'anchor')
        stopped,_,_=h.run_update(exact=.02,sampled=0.,full=0.,exact_only=True)
        self.assertEqual(stopped['optimizer_steps'],1);self.assertTrue(stopped['kl_early_stop'])
        self.assertEqual(stopped['kl_guard']['stop_reason'],'categorical_probe')

    def test_nonfinite_rejection_and_archived_secondary_guard_remain(self):
        h=GuardTests()
        with self.assertRaisesRegex(ValueError,'Nonfinite full-dataset PPO sampled KL'):
            h.run_update(exact=.001,full=float('nan'),exact_only=True)
        legacy,_,_=h.run_update(exact=.001,full=.02)
        self.assertEqual(legacy['optimizer_steps'],4)
        self.assertEqual(legacy['kl_guard']['stop_reason'],'sampled_full_dataset')

    def test_temperature1_and_actual_behavior_exact_guards_remain_independent(self):
        from test_gae_value_phase import fixture,ppo
        from test_opening_ppo import opening_rows
        from brass_ai.recovery.opening_objective import DEFAULT as OPENING
        for metric,reason in [('categorical_kl','temperature1_categorical_probe'),
            ('behavior_categorical_kl','behavior_categorical_probe')]:
            model,opt,cfg,(_,data)=fixture(games=3,decisions=3)
            behavior,data=opening_rows(model,data);cfg=K.configure(cfg)
            cfg['update']['opening_objective']=copy.deepcopy(OPENING)
            measured=dict(categorical_kl=0.,behavior_categorical_kl=0.,sampled_kl=0.)
            measured[metric]=.02
            with patch.object(CategoricalProbe,'measure_step',return_value=measured):
                report=ppo(model,opt,cfg,behavior,data)
            self.assertEqual(report['optimizer_steps'],1);self.assertEqual(report['kl_guard']['stop_reason'],reason)
            self.assertEqual(report['kl_guard']['contract'],K.CONTRACT)
            self.assertTrue(report['opening_objective']['behavior_verified'])


if __name__=='__main__':unittest.main()
