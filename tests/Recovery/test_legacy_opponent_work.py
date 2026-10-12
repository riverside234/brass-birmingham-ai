"""Training opponent quotas leave frozen v4, demonstrations and evaluation intact."""
import copy
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fixtures import position
from brass_ai.learning.contracts import digest
from brass_ai.recovery import legacy_search as legacy
from brass_ai.recovery.config import LEGACY_BUDGET,LEGACY_OPPONENT_WORK_BUDGET


class LegacyOpponentWorkTests(unittest.TestCase):
    def setUp(self):
        self.clock=SimpleNamespace(now=100.)
        self.profile=dict(mode='masked_ismcts',simulations=100000,work_limit=1000000,
            milliseconds=1000,tactic='balanced')
        self.payload=dict(profiles={'style':self.profile},runtime_source_paths=['src/BrassMaskedSearch/RootForest.cs'])
        self.calls=[];self.delay=0.;self.excess=False;self.partial=False
        outer=self
        class Worker:
            def call(self,operation,**request):
                outer.calls.append((operation,copy.deepcopy(request)))
                p=request['profile'];count=0 if p['incumbent_only'] else p['simulations']*request['search_workers']
                if outer.partial and count:count=4
                if outer.excess:count+=1
                outer.clock.now+=outer.delay
                return dict(mode='masked_ismcts',failures=0,nodes=20,overrun=outer.delay>p['milliseconds']/1000,
                    selected=dict(actions=[[request['choices'][0]['id']],[request['choices'][-1]['id']]],intent={'age':0}),
                    search=dict(opponent_nodes=0,completed=count,interrupted=0,root={},plans={}))
        class Frozen:
            def __init__(self,*args,**kwargs):self.profile=outer.profile;self.worker=Worker()
            def close(self):pass
        for mock in (patch.object(legacy,'admission'),patch.object(legacy,'load_bank',side_effect=lambda *args:{'payload':self.payload}),
                     patch.object(legacy,'FrozenOpponent',Frozen),patch.object(legacy.time,'monotonic',side_effect=lambda:self.clock.now)):
            mock.start();self.addCleanup(mock.stop)
        self.budget=dict(LEGACY_BUDGET,search_workers=4)
        self.work=copy.deepcopy(LEGACY_OPPONENT_WORK_BUDGET)
        self.rng=random.Random(1)

    def policy(self,work=True,**kwargs):
        policy=legacy.LabLegacyPolicy(Path('.'),'bank/style',self.budget,
            work_budget=self.work if work else None,**kwargs)
        self.addCleanup(policy.close)
        return policy

    def test_identity_and_scope_keep_old_teachers_and_evaluation_unchanged(self):
        original=legacy.legacy_spec('.', 'bank/style',self.budget)
        expected='masked-v3-forest/'+digest(dict(parent='bank/style',budget=self.budget,search_free=False,version=3))
        self.assertEqual(original['policy_id'],expected);self.assertNotIn('work_budget',original)
        self.assertEqual(legacy.legacy_spec('.','bank/style',self.budget,purpose='evaluation')['policy_id'],expected)
        changed=legacy.legacy_spec('.','bank/style',self.budget,work_budget=self.work)
        self.assertNotEqual(changed['policy_id'],expected);self.assertEqual(changed['work_budget'],self.work)
        self.assertEqual(legacy.legacy_spec('.','bank/style',self.budget,purpose='evaluation',work_budget=self.work)['policy_id'],changed['policy_id'])
        reference=legacy.legacy_spec('.','bank/style',dict(self.budget,role='reference'),work_budget=self.work)
        self.assertNotEqual(reference['policy_id'],changed['policy_id'])
        self.assertEqual(reference['budget']['role'],'reference')
        with self.assertRaisesRegex(ValueError,'lab policy'):
            legacy.legacy_spec('.','bank/style',dict(self.budget,role='laptop'),work_budget=self.work)
        fast=legacy.legacy_spec('.','bank/style',self.budget,work_budget=self.work,search_free=True)
        self.assertNotEqual(fast['policy_id'],changed['policy_id'])
        old_work=dict(contract='legacy-opponent-turn-work/1.0.0',mode='simulations',simulations_per_turn=16)
        with self.assertRaisesRegex(ValueError,'lab policy'):
            legacy.legacy_spec('.','bank/style',self.budget,work_budget=old_work,search_free=True)
        with self.assertRaisesRegex(ValueError,'divide across workers'):
            legacy.legacy_spec('.','bank/style',self.budget,work_budget=dict(self.work,simulations_per_turn=18))
        with self.assertRaisesRegex(ValueError,'four-attempt blocks'):
            legacy.legacy_spec('.','bank/style',self.budget,work_budget=dict(self.work,simulations_per_turn=8))
        self.payload['runtime_source_paths']=[]
        with self.assertRaisesRegex(ValueError,'native root forest'):
            legacy.legacy_spec('.','bank/style',self.budget,work_budget=self.work)

    def test_forced_first_then_complete_plan_uses_one_shared_native_cap(self):
        policy=self.policy();view,choices=position();before=copy.deepcopy(self.profile)
        policy.select_batch([(view,choices[:1],self.rng)])
        self.assertEqual(self.calls,[])
        view['context']['revision']+=1
        self.assertEqual(policy.select_batch([(view,choices,self.rng)])[0][0],0)
        view['context']['revision']+=1;view['context']['actions_remaining']=1
        self.assertEqual(policy.select_batch([(view,choices,self.rng)])[0][0],2)
        self.assertEqual(len(self.calls),1)
        op,request=self.calls[0]
        self.assertEqual((op,request['search_workers'],request['profile']['simulations']),('plan_forest',4,4))
        self.assertIs(request['fixed_work'],False)
        self.assertEqual(request['profile']['milliseconds'],120000.)
        self.assertEqual(request['profile']['work_limit'],4096+4*513)
        self.assertEqual(self.profile,before)
        self.assertEqual((policy.telemetry['allocated_simulations'],policy.telemetry['attempted_simulations'],policy.telemetry['completed']),(16,16,16))
        self.assertEqual(policy.telemetry['work_exhausted_turns'],1)

    def test_replans_and_duplicate_revision_never_renew_turn_work_or_clock(self):
        policy=self.policy();view,choices=position()
        policy.select_batch([(view,choices,self.rng)])
        self.clock.now+=.2
        # A stale callback invalidates its queue, not the already-issued quota.
        policy.select_batch([(view,choices,self.rng)])
        view['context']['revision']+=2
        policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(len(self.calls),3)
        self.assertTrue(all(r['profile']['incumbent_only'] for _,r in self.calls[1:]))
        self.assertAlmostEqual(self.calls[1][1]['profile']['milliseconds'],119800.)
        self.assertEqual(policy.telemetry['turns'],1)
        self.assertEqual(policy.telemetry['allocated_simulations'],16)
        self.assertEqual(policy.telemetry['attempted_simulations'],16)
        self.assertEqual(policy.telemetry['incumbent_only_decisions'],2)

    def test_debt_does_not_spend_search_and_next_turn_receives_only_its_own_quota(self):
        policy=self.policy();view,choices=position();view['context']['phase']='debt'
        policy.select_batch([(view,choices,self.rng)])
        self.assertTrue(self.calls[0][1]['profile']['incumbent_only'])
        self.assertEqual(policy.telemetry['allocated_simulations'],0)
        view['context'].update(phase='action',revision=2)
        policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(policy.telemetry['allocated_simulations'],16)
        view['context']['turn']+=1;view['context']['revision']+=1
        policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(policy.telemetry['allocated_simulations'],32)
        self.assertEqual(policy.telemetry['turns'],2)

    def test_safety_clock_retained_and_native_overallocation_rejected(self):
        self.work=dict(contract='legacy-opponent-turn-work/1.0.0',mode='simulations',simulations_per_turn=16)
        policy=self.policy();view,choices=position();self.partial=True;self.delay=1.1
        policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(policy.telemetry['attempted_simulations'],4)
        self.assertEqual(policy.telemetry['deadline_limited_turns'],1)
        self.assertEqual(policy.telemetry['work_exhausted_turns'],0)
        view['context']['revision']+=2;self.delay=0
        policy.select_batch([(view,choices,self.rng)])
        self.assertTrue(self.calls[-1][1]['profile']['incumbent_only'])
        self.assertEqual(self.calls[-1][1]['profile']['milliseconds'],0)
        self.assertIs(self.calls[-1][1]['fixed_work'],False)
        self.excess=True;view['context']['turn']+=1;view['context']['revision']+=1;self.partial=False
        with self.assertRaisesRegex(ValueError,'shared turn allowance'):
            policy.select_batch([(view,choices,self.rng)])

    def test_count_mode_exceeds_reference_second_and_rejects_incomplete_work(self):
        policy=self.policy();view,choices=position();self.delay=2.
        policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(policy.telemetry['attempted_simulations'],16)
        self.assertEqual(policy.telemetry['over_budget_turns'],0)
        self.assertEqual(policy.telemetry['watchdog_failures'],0)
        self.partial=True;view['context']['turn']+=1;view['context']['revision']+=1
        with self.assertRaisesRegex(TimeoutError,'allocated=16, attempted=4'):
            policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(policy.telemetry['incomplete_work_failures'],1)
        self.assertEqual(policy.telemetry['watchdog_failures'],0)
        self.partial=False;self.delay=121.;view['context']['turn']+=1;view['context']['revision']+=1
        with self.assertRaisesRegex(TimeoutError,'watchdog_expired=True'):
            policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(policy.telemetry['watchdog_failures'],1)

    def test_active_search_free_is_bounded_incumbent_only_without_reference_clock(self):
        policy=self.policy(search_free=True);view,choices=position();self.delay=2.
        policy.select_batch([(view,choices,self.rng)])
        request=self.calls[0][1]
        self.assertEqual(request['search_workers'],1)
        self.assertEqual(request['profile']['work_limit'],4096)
        self.assertEqual(request['profile']['milliseconds'],120000.)
        self.assertTrue(request['profile']['incumbent_only'])
        self.assertEqual(policy.telemetry['allocated_simulations'],0)
        self.assertEqual(policy.telemetry['attempted_simulations'],0)
        self.assertEqual(policy.telemetry['over_budget_turns'],0)

    def test_replan_node_allowance_subtracts_aggregate_work_in_forest_units(self):
        policy=self.policy();view,choices=position()
        policy.select_batch([(view,choices,self.rng)])
        state=policy.contexts[self.rng]
        state['work']=7000;state['queue'].clear();view['context']['revision']+=1
        policy.select_batch([(view,choices,self.rng)])
        request=self.calls[-1][1]
        self.assertTrue(request['profile']['incumbent_only'])
        self.assertEqual(request['profile']['work_limit'],(4*(4096+4*513)-7000)//4)
        self.assertEqual(request['profile']['work_limit'],4398)
        self.assertEqual(policy.telemetry['allocated_simulations'],16)

    def test_absent_budget_keeps_time_mode_and_search_free_behavior(self):
        view,choices=position();policy=self.policy(work=False)
        policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(self.calls[-1][1]['profile']['simulations'],100000)
        self.assertNotIn('allocated_simulations',policy.telemetry)
        fast=self.policy(work=False,search_free=True)
        fast.select_batch([(view,choices,random.Random(3))])
        self.assertTrue(self.calls[-1][1]['profile']['incumbent_only'])
        self.assertEqual(fast.telemetry['completed'],0)


if __name__=='__main__':unittest.main()
