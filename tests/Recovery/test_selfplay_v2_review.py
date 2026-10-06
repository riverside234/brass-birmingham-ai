"""Exact validation/rehearsal parity and trustworthy post-IL diagnostics."""
import copy
import math
import unittest
from unittest.mock import patch

import torch

from fixtures import position
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.runtime_contracts import validate
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.recovery.inputs import InputBoundary
from brass_ai.recovery.telemetry import iteration_utilization
from brass_ai.recovery import updates
import test_revision as revision
from test_selfplay_review import teacher_games


class SelfplayV2ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def assert_parity(self,view,choices,boundary=None):
        def accepted(f):
            try:f();return True
            except (ValueError,TypeError,KeyError,IndexError):return False
            except Exception as error:
                from jsonschema.exceptions import ValidationError
                if isinstance(error,ValidationError):return False
                raise
        expected=accepted(lambda:(validate('observation',view),validate('choices',choices)))
        actual=accepted(lambda:(boundary or InputBoundary()).check(view,choices))
        self.assertEqual(actual,expected)
        return actual

    def test_cached_parts_match_pinned_schema_and_reject_mutations(self):
        view,choices=position();boundary=InputBoundary()
        self.assertTrue(self.assert_parity(view,choices,boundary))
        changed=copy.deepcopy(view);changed['context']['revision']+=1
        self.assertTrue(self.assert_parity(changed,choices,boundary))
        self.assertGreater(boundary.report()['part_cache_hits'],0)
        mutations=[
            lambda v,c:v.update(opponents_hands=[]),
            lambda v,c:v['context'].update(game_id='f'*32),
            lambda v,c:v['players'][1].update(own_hand=[]),
            lambda v,c:v['players'][0]['mat'][0]['tiles'][0].update(secret=1),
            lambda v,c:v['players'][0]['mat'][0]['tiles'][0].update(level=True),
            lambda v,c:v['players'][0]['mat'].pop(),
            lambda v,c:v['players'].pop(),
            lambda v,c:v['order'].__setitem__(1,v['order'][0]),
            lambda v,c:v['own_hand'].pop(),
            lambda v,c:v['players'][0].update(seat=1),
            lambda v,c:v['links'][0].update(owner=4),
            lambda v,c:v['merchants'][0].update(beer=2),
            lambda v,c:v['slots'][0].update(private=True),
            lambda v,c:v['history'].append(dict(seat=0,era='CanalEra',choice=dict(id='a',kind='finish',secret=1))),
            lambda v,c:v.update(history=None),
            lambda v,c:v.update(players={}),
            lambda v,c:c[0].update(secret=1),
            lambda v,c:c[0].update(kind=[]),
            lambda v,c:c.__setitem__(0,None),
        ]
        for change in mutations:
            v,c=copy.deepcopy(view),copy.deepcopy(choices);change(v,c)
            self.assertFalse(self.assert_parity(v,c,boundary))
        self.assertTrue(self.assert_parity(view,choices,boundary))

    def test_all_choice_branches_keep_optional_fields_and_multiplicity(self):
        view,_=position();card='location:Birmingham'
        choices=[dict(id='a',kind=k,card=card) for k in ('pass','loan')]+[
            dict(id='b',kind='build',card=card,target='x',industry='Pottery'),
            dict(id='c',kind='develop',industry='Pottery'),
            dict(id='d',kind='network',target='x'),
            dict(id='e',kind='sell',target='x',merchant='m'),
            dict(id='f',kind='scout',card=card,cards=[card,card]),
            *[dict(id='g',kind=k,target='x') for k in ('coal','iron','beer','liquidate')],
            *[dict(id='h',kind=k) for k in ('finish','bonus_skip')],
            dict(id='i',kind='bonus_develop',industry='Pottery')]
        boundary=InputBoundary(frames=1,events=2,parts=3)
        for choice in choices:
            self.assertTrue(self.assert_parity(view,[choice],boundary))
            broken=dict(choice,kind='unknown')
            self.assertFalse(self.assert_parity(view,[broken],boundary))
        self.assertLessEqual(boundary.report()['retained_parts'],3)
        self.assertLessEqual(boundary.report()['retained_frames'],1)
        self.assertTrue(self.assert_parity(view,choices,boundary))
        # Constructing the split schemas must not mutate the authoritative one.
        invalid=copy.deepcopy(view);invalid['players'][0]['mat'][0]['tiles'][0]['level']=0
        self.assertFalse(self.assert_parity(invalid,choices))

    def test_cached_rehearsal_is_reused_and_keeps_identical_ppo_update(self):
        model,policy,game=revision.RevisionTests().onpolicy({1})
        data=updates.prepared([game]);cached=updates.rehearsal_dataset(data,'cpu')
        self.assertIs(updates.rehearsal_dataset(cached,'cpu'),cached)
        left,right=copy.deepcopy(model),copy.deepcopy(model)
        cfg=dict(UPDATE_V21,effective_batch_size=4,critic_batch_size=4,microbatch_size=2,epochs=1)
        torch.manual_seed(781)
        a=updates.ppo_update(left,torch.optim.AdamW(left.parameters(),lr=1e-6),[game],policy,cfg,
                             rehearsal=data,rehearsal_weight=.1)
        init=updates.Dataset.__init__;calls=[]
        def tracked(instance,*args,**kwargs):calls.append(kwargs.get('policy',True));init(instance,*args,**kwargs)
        torch.manual_seed(781)
        with patch.object(updates.Dataset,'__init__',tracked):
            b=updates.ppo_update(right,torch.optim.AdamW(right.parameters(),lr=1e-6),[game],policy,cfg,
                                 rehearsal=cached,rehearsal_weight=.1)
        self.assertEqual(calls,[True]) # Only the new on-policy collection is prepared.
        self.assertEqual(a['epochs'],b['epochs'])
        self.assertTrue(all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items()))
        self.assertIsNone(updates.rehearsal_dataset(None,'cpu'))
        with self.assertRaisesRegex(ValueError,'supervised'):
            updates.rehearsal_dataset(updates.Dataset(data,'cpu'),'cpu')

    def test_teacher_accuracy_entropy_and_kl_are_measured(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        games=teacher_games(2)
        for game in games:
            for row in game['records']:row['chosen']=0
        model=FourSeatNet(ModelConfig.preset('small'))
        with torch.no_grad():
            for p in model.parameters():p.zero_()
        report=updates.supervised_epoch(model,None,updates.prepared(games),UPDATE_V21,training=False,kind='teacher')
        self.assertEqual(report['meaningful_top1'],1.)
        self.assertEqual(report['root_action_top1'],1.)
        self.assertEqual(report['root_family_top1'],1.)
        self.assertEqual(sum(r['decisions'] for r in report['by_teacher_era_phase_action'].values()),32)
        self.assertAlmostEqual(report['policy_loss'],math.log(3),places=5)
        self.assertAlmostEqual(report['teacher_kl'],report['policy_loss']-report['teacher_entropy'],places=5)
        result=updates.distill_update(model,torch.optim.AdamW(model.parameters(),lr=1e-5),games,UPDATE_V21)
        for name in ('fit_before','fit_after'):
            self.assertEqual(result[name]['root_action_top1'],1.)
            self.assertAlmostEqual(result[name]['teacher_entropy'],report['teacher_entropy'],places=5)
        self.assertFalse(result['value_loss_training'])
        self.assertEqual(result['max_observed_reuse'],2)

    def test_cpu_busy_time_is_separate_and_missing_is_not_zero(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')['execution']
        records=[dict(actor_python_cpu_seconds=.7),dict(actor_python_cpu_seconds=.5)]
        report=iteration_utilization({}, {},records,2.,2,cfg)
        self.assertAlmostEqual(report['actor_python_cpu_seconds'],1.2)
        self.assertAlmostEqual(report['actor_python_average_cores'],.6)
        report=iteration_utilization({}, {},records+[dict(cached=True)],2.,3,cfg)
        self.assertIsNone(report['actor_python_cpu_seconds'])
        self.assertIsNone(report['actor_python_average_cores'])
        self.assertEqual(report['actor_cpu_recorded_games'],2)

    def test_stable_lab_learning_and_search_settings_are_preserved(self):
        for path in (SOURCE_ROOT/'configs/step5').glob('lab*.json'):
            cfg=read(path)
            self.assertEqual((cfg['learning_rate'],cfg['update']['target_kl']),(1e-5,.01))
            self.assertEqual((cfg['search']['seconds'],cfg['execution']['search_workers_per_turn']),(12.,4))
            self.assertEqual((cfg['search_ready_games'],cfg['search_every_games'],cfg['search_wall_fraction']),(640,320,.5))


if __name__=='__main__':unittest.main()
