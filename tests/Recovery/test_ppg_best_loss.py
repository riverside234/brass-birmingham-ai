"""Minimum-loss retention is independent of paired-SE stopping and row coverage."""
import copy
import json
import unittest

import torch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.ppg_selection import Selection,MINIMUM_DEFAULT,RELAXED_DEFAULT
from brass_ai.recovery.retention import _same_payload


class Companion:
    def __init__(self):self.value=dict(weights=torch.tensor([0.]),adam_step=0)
    def snapshot(self):return copy.deepcopy(self.value)
    def restore(self,value):self.value=copy.deepcopy(value)


class BestLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.model=torch.nn.Linear(1,1);self.opt=torch.optim.Adam(self.model.parameters(),lr=.001)
        self.aux=Companion();self.games=dict.fromkeys('abcd',1.)
        self.selection=Selection(self.model,self.opt,self.games,companion=self.aux,config=MINIMUM_DEFAULT)

    def step(self,games,rows,budget):
        for p in self.model.parameters():p.grad=torch.ones_like(p)
        self.opt.step();self.opt.zero_grad(set_to_none=True)
        self.aux.value=dict(weights=torch.tensor([float(budget)]),adam_step=budget)
        expected=self.snapshot()
        chosen=self.selection.consider(self.model,self.opt,games,rows,[0]*budget,budget,budget)
        return chosen,expected

    def snapshot(self):
        return copy.deepcopy(dict(weights=self.model.state_dict(),adam=self.opt.state_dict(),aux=self.aux.snapshot()))

    def restore(self,expected):
        self.selection.restore(self.model,self.opt)
        self.assertTrue(_same_payload(self.snapshot(),expected))
        r=self.selection.report();self.assertEqual(r['retained_loss'],r['minimum_loss'])
        self.assertEqual(r['retained_budget_step'],r['minimum_budget_step'])

    def test_more_data_inside_tolerance_cannot_replace_best_matching_states(self):
        chosen,best=self.step(dict.fromkeys('abcd',.9),[1,2],16);self.assertTrue(chosen)
        later=dict(zip('abcd',(.93,.89,.93,.89)))
        self.assertFalse(self.selection.paired(self.selection.minimum_games,later)['reject'])
        chosen,_=self.step(later,list(range(50)),32);self.assertFalse(chosen)
        self.restore(best);self.assertEqual(self.selection.steps,16);self.assertEqual(self.selection.rows,[1,2])

    def test_no_improvement_returns_post_PPO_start_and_zero_retained_work(self):
        start=self.snapshot()
        worse=dict(zip('abcd',(1.03,.99,1.03,.99)))
        self.assertFalse(self.selection.paired(self.games,worse)['reject'])
        self.assertFalse(self.step(worse,list(range(50)),16)[0])
        self.assertFalse(self.step(self.games,list(range(100)),32)[0])
        self.restore(start);self.assertEqual(self.selection.steps,0);self.assertEqual(self.selection.rows,[])

    def test_equal_loss_keeps_earliest_but_lower_loss_wins_with_fewer_rows(self):
        chosen,first=self.step(dict.fromkeys('abcd',.9),list(range(100)),16);self.assertTrue(chosen)
        self.assertFalse(self.step(dict.fromkeys('abcd',.9),list(range(200)),32)[0])
        self.restore(first);self.assertEqual(self.selection.budget_step,16)
        chosen,last=self.step(dict.fromkeys('abcd',.899),[0],48);self.assertTrue(chosen)
        self.restore(last);self.assertEqual(self.selection.rows,[0]);self.assertEqual(self.selection.budget_step,48)

    def test_new_phase_uses_its_own_heldout_set(self):
        self.step(dict.fromkeys('abcd',.9),[0],16)
        next_games=dict.fromkeys(('new-1','new-2'),5.)
        self.selection=Selection(self.model,self.opt,next_games,companion=self.aux,config=MINIMUM_DEFAULT)
        chosen,best=self.step(dict.fromkeys(next_games,4.9),[1],16)
        self.assertTrue(chosen);self.restore(best);self.assertEqual(self.selection.initial_loss,5.)

    def test_archived_loss_curves_select_547_start_555_112_and_595_64(self):
        path=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0620-20261023T133150568578Z/retention-history.json'
        history=json.loads(path.read_text())
        for generation,expected_step in ((547,0),(555,112),(595,64)):
            value=history[f'iteration-{generation:04d}/report.json']['summary']['value_phase']
            logged=value['selection'];reference=logged['initial_loss']
            selection=Selection(self.model,self.opt,{'aggregate':reference},config=MINIMUM_DEFAULT)
            for interval in value['interval_reports']:
                if 'budget_step' not in interval or 'paired' not in interval:continue
                observed=reference+interval['paired']['mean_change'];budget=interval['budget_step']
                selection.consider(self.model,self.opt,{'aggregate':observed},list(range(budget)),[],budget,budget)
                if interval['paired']['improved']:reference=observed
            self.assertEqual(selection.budget_step,expected_step)
            self.assertAlmostEqual(selection.report()['minimum_loss'],logged['minimum_loss'],places=12)

    def test_archived_coverage_contract_keeps_its_old_behavior(self):
        old=Selection(self.model,self.opt,self.games,config=RELAXED_DEFAULT)
        self.assertTrue(old.consider(self.model,self.opt,dict(zip('abcd',(1.03,.99,1.03,.99))),[1,2],[],16,16))
        self.assertGreater(old.report()['retained_loss'],old.report()['minimum_loss'])


if __name__=='__main__':unittest.main()
