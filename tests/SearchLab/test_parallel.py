"""Parallel trees must select the acting seat and confirm only complete batches."""
import time
import unittest

import numpy as np

from brass_ai.recovery.parallel import ParallelS3
from brass_ai.searchlab.s3 import TreeConfig
from brass_ai.step5.search import SearchConfig,confirmed_teacher


class PrivateWorker:
    def __init__(self):
        self.base=type('Base',(),{'request_deadline':None})();self.rollouts=[];self.samples=[]
    def trajectory(self,view,plan,seed,context,tree,deadline,frozen=False):
        self.samples.append((view['seat'],tuple(plan),seed,frozen,id(tree)))
        return np.asarray([.8,.1,.05,.05] if plan==['a'] else [.05,.05,.1,.8])
    def close(self): pass


class ParallelTreeTests(unittest.TestCase):
    def planner(self):
        policy=object.__new__(ParallelS3)
        policy.policy_id='test-private-root';policy.host=type('Host',(),{'close':lambda s:None})()
        policy.config=SearchConfig(proposals=2,finalists=2,pilot_samples=1,confirmation_samples=2)
        policy.tree_config=TreeConfig(simulations=8,future_turns=1)
        policy.telemetry=dict(searches=0,tree_simulations=0,confirmed=0,fresh_worlds=0,incomplete=0,unconfirmed_incumbents=0)
        policy.propose=lambda *args:[['a'],['b']]
        policy.context_features=lambda *args:([0.]*4,[0.]*8)
        return policy.configure(2,'lab',lambda i:PrivateWorker())

    def test_four_seat_backups_and_fresh_matched_confirmation(self):
        import random
        policy=self.planner()
        try:
            view=dict(seat=3);choices=[dict(id='a'),dict(id='b')]
            chosen,plans,values,value=policy.search(view,choices,random.Random(778),time.monotonic()+5)
            self.assertEqual(chosen,['b'])
            self.assertEqual(sum(policy.last_tree['root_counts']),8)
            teacher=confirmed_teacher(choices,plans,values,[],value,2,.1)
            self.assertIsNotNone(teacher);self.assertGreater(teacher['confidence'],0)
            exploration={s[2] for w in policy.replicas for s in w.samples if not s[3]}
            fresh={s[2] for w in policy.replicas for s in w.samples if s[3]}
            self.assertTrue(exploration.isdisjoint(fresh))
            for worker in policy.replicas:
                for seed in {s[2] for s in worker.samples if s[3]}:
                    samples=[s for s in worker.samples if s[2]==seed and s[3]]
                    self.assertEqual({s[1] for s in samples},{('a',),('b',)})
                    self.assertEqual(len({s[4] for s in samples}),1)
                self.assertIsNone(worker.base.request_deadline)
        finally: policy.close()

    def test_timeout_has_legal_incumbent_without_teacher(self):
        import random
        policy=self.planner()
        try:
            chosen,plans,values,value=policy.search(dict(seat=3),[dict(id='a'),dict(id='b')],random.Random(1),time.monotonic()-1)
            self.assertIn(chosen,[['a'],['b']]);self.assertEqual(value,0)
            self.assertIsNone(confirmed_teacher([dict(id='a'),dict(id='b')],plans,values,[],value,2,.1))
        finally: policy.close()
