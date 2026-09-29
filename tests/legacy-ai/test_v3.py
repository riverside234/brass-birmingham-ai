"""V3 cash and observation contracts through the actual four-player worker."""
import copy
import unittest
from test_legacy import ROOT, COMPOSITION
from brass_ai.legacy import LegacyWorker
from brass_ai.beliefs import sample_world_setup


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.w = LegacyWorker(ROOT)
    @classmethod
    def tearDownClass(cls): cls.w.close()

    def test_v3_flags_are_explicit_and_invalid_versions_fail(self):
        context = self.w.call('reset', seed=1301)
        view = self.w.call('observe', seat=context['actor'])
        profile = self.w.profile('balanced-v3')
        self.assertEqual((1, 1), (profile['cash_policy'], profile['conversion_policy']))
        for flag in ('cash_policy', 'conversion_policy'):
            invalid = dict(profile, **{flag: 2})
            with self.assertRaisesRegex(RuntimeError, 'policy_contract'):
                self.w.call('plan', observation=view, choices=self.w.call('choices'), profile=invalid)

    def test_all_v3_styles_protect_income_at_exhausted_budget(self):
        for style in ('balanced', 'rail-tempo', 'cotton', 'goods-beer', 'supplier', 'pottery-flex'):
            context = self.w.call('reset', seed=1301)
            view = self.w.call('observe', seat=context['actor'])
            view['context'].update(round=2, actions_remaining=1)
            view['players'][view['seat']].update(cash=10, income_marker=0, income=-10)
            context = self.w.call('reset', seed=1, setup=sample_world_setup(view, COMPOSITION, 7))
            view = self.w.call('observe', seat=context['actor'])
            before = self.w.call('hash')
            profile = self.w.profile(style+'-v3'); profile.update(work_limit=1, milliseconds=0)
            result = self.w.call('plan', observation=view, choices=self.w.call('choices'), profile=profile)
            self.assertEqual(before, self.w.call('hash'))
            self.assertEqual(0, result['selected']['cash']['shortfall'])
            self.assertEqual(1, len(result['selected']['actions']))
            self.w.call('commit', game_id=context['game_id'], revision=context['revision'],
                        choices=result['selected']['actions'][0])

    def test_fixed_work_v3_decisions_are_hidden_world_invariant_all_seats(self):
        context = self.w.call('reset', seed=1301)
        for _ in range(4):
            view = self.w.call('observe', seat=context['actor'])
            profile = self.w.profile('balanced-v3'); profile['work_limit'] = 64
            first = self.w.call('plan', observation=view, choices=self.w.call('choices'),
                                profile=profile, fixed_work=True)
            context = self.w.call('reset', seed=1, setup=sample_world_setup(view, COMPOSITION, 999))
            second = self.w.call('plan', observation=self.w.call('observe', seat=context['actor']),
                                 choices=self.w.call('choices'), profile=profile, fixed_work=True)
            self.assertEqual(first['selected'], second['selected'])
            self.assertEqual(first['nodes'], second['nodes'])
            legal = self.w.call('choices')
            context = self.w.call('commit', game_id=context['game_id'], revision=context['revision'],
                                 choices=[next(c['id'] for c in legal if c['kind']=='pass'), 'finish|||||'])


if __name__ == '__main__': unittest.main()
