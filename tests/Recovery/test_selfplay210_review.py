"""Current exported recipe/state and g210 collection/guard evidence."""
import copy
import unittest
from pathlib import Path
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import exploration as E
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0210-20261020T145156876449Z'


class G210ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state=read(EXPORT/'state.json')
        cls.latest=read(EXPORT/'iteration-0209/report.json')

    def test_exact_source_only_continuation_preserves_stage_and_recipe(self):
        before=copy.deepcopy(self.state);cfg=self.state['signature']['config']
        validate_config(cfg)
        self.assertEqual(cfg,read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-exploration.json'))
        target=dict(self.state['signature'],implementation=implementation_id())
        self.assertTrue(check_signature(self.state,target,reviewed=True))
        with self.assertRaisesRegex(ValueError,'changed'):check_signature(self.state,target)
        restored=E.restore(cfg,self.state,copy.deepcopy(self.state['exploration']))
        effective=E.effective(cfg,restored,210)
        self.assertEqual((restored['origin'],restored['armed_at'],effective['age']),(185,197,25))
        self.assertEqual(effective['branching'],E.LEGACY_BRANCH)
        self.assertEqual(self.state,before)
        for k in ('exploration','learning_rate'):
            bad=copy.deepcopy(target);bad['config'][k]*=1.1
            with self.assertRaises(ValueError):check_signature(self.state,bad,reviewed=True)

    def test_latest_ppg_rejection_is_loss_not_kl_and_health_has_no_il_action_checks(self):
        value=self.latest['value_phase']
        self.assertEqual(value['early_stop_reason'],'heldout_loss_increased')
        self.assertFalse(value['kl_guard_tripped'])
        self.assertLess(value['maximum_attempted_probe_kl'],.002)
        health=self.latest['il_kl_limit']['health']
        self.assertTrue(health['ok'])
        self.assertEqual(set(health['checks']),{'vp_per_seat','passes_per_seat','zero_vp_fraction','insolvency_fraction'})
        self.assertNotIn('behavior',health)
        self.assertFalse(self.latest['actor']['kl_early_stop'])

    def test_branches_are_separate_and_actual_samples_reach_ppo(self):
        metrics=self.latest['exploration_metrics'];branch=metrics['branching']
        self.assertEqual(self.latest['health_counts']['tables'],320)
        self.assertEqual(branch['games'],64)
        self.assertEqual(metrics['pass_with_alternative_exploration_picks'],0)
        self.assertEqual(metrics['tempered_decisions'],2*metrics['seat_games'])
        self.assertEqual(branch['extra_actor_rows'],self.latest['actor']['branch_sample_usage']['actor_distinct_used'])
        self.assertEqual(self.state['games_done'],210*320)


if __name__=='__main__':unittest.main()
