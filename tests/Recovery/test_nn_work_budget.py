"""Lab NN evaluation uses one count budget per real turn, not teacher phases."""
import copy
import random
import unittest
import weakref
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from fixtures import game,position,result
from brass_ai.recovery import teacher_schedule as schedule
from brass_ai.recovery.config import NN_WORK_BUDGET,teacher_search_config
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.execution import _run_turns
from brass_ai.recovery.telemetry import Timings
from brass_ai.search.ismcts import MaskedSearchPolicy
from test_teacher_schedule import Clock
from test_teacher_work import WorkDecider


class WorkPlanner(schedule.WorkTurns,WorkDecider):
    kind='external';policy_id='unit-work-policy'
    def __init__(self,clock):
        self.clock=clock;self.calls=[];self.mode='success';self.next_views=[]
        self.teacher_states=weakref.WeakKeyDictionary();self.cancelled=Event()
        self.nn_work_budget=copy.deepcopy(NN_WORK_BUDGET)
        self.teacher_work_budget=dict(contract='teacher-turn-work/1.1.0',mode='simulations',
            watchdog_seconds=NN_WORK_BUDGET['watchdog_seconds'],
            simulations_per_phase={p:NN_WORK_BUDGET['simulations_per_turn'] for p,_,_,_ in schedule.PHASES})
        self.config=teacher_search_config(dict(turn_interval=1,max_searches_per_seat=0))
        self.base=SimpleNamespace(request_deadline=None,select_batch=lambda inputs:[(2,1.,None)]*len(inputs))


class NNWorkTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.clock.now=100.
        fake=patch.object(schedule.time,'monotonic',self.clock);fake.start();self.addCleanup(fake.stop)
        fake=patch('brass_ai.step5.league.implementation_id',return_value='a'*64);fake.start();self.addCleanup(fake.stop)
        self.planner=WorkPlanner(self.clock);self.rng=random.Random(9)
        self.planner.begin_work_game(self.rng,0)

    def test_every_turn_searches_once_without_exporting_teacher_labels(self):
        view,choices=position();view['context'].update(actions_remaining=2)
        self.assertFalse(hasattr(self.planner,'begin_teacher_game'))
        for turn in (1,2):
            view=copy.deepcopy(view);view['context'].update(turn=turn,round=turn,revision=turn*10)
            self.planner.admit_work_turn(view['context'],self.rng)
            self.assertIsNone(self.planner.select_batch([(view,choices[:1],self.rng)])[0][2])
            before=len(self.planner.calls)
            self.planner.mode='slow_normal'
            self.assertEqual(self.planner.select_batch([(view,choices,self.rng)])[0],(0,1.,None))
            self.assertEqual(len(self.planner.calls),before+1)
            # Second action/subchoices cannot reset the count or relabel search.
            view['context'].update(actions_remaining=1,revision=view['context']['revision']+1)
            self.planner.admit_work_turn(view['context'],self.rng)
            self.assertIsNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
            self.assertEqual(len(self.planner.calls),before+1)
            self.planner.finish_work_turn(self.rng)
        counts=self.planner.work_report(self.rng)['phases']['canal_early']
        self.assertEqual((counts['scheduled_turns'],counts['actual_searches']),(2,2))
        self.assertEqual((counts['allocated_simulations'],counts['attempted_simulations']),(32,32))
        self.assertEqual(counts['timeouts'],0)
        self.assertGreater(counts['elapsed_seconds'],24.)

    def test_execution_starts_watchdog_after_cpu_admission_and_shares_both_actions(self):
        view,choices=position();view['context'].update(actions_remaining=2)
        contexts=[];planner=self.planner;clock=self.clock;events=[]
        for phase,remaining in [('action',2),('payment',1),('action',1),('continuation',0)]:
            item=copy.deepcopy(view);item['context'].update(phase=phase,actions_remaining=remaining,revision=len(contexts))
            contexts.append(item)
        planner.next_views=contexts[1:]
        class Lease:
            first=True
            def update(self,key,workers):
                self.assert_workers=workers
                if self.first:clock.now+=30;self.first=False;events.append('admission')
        class Host:
            index=0
            def observe(self,seat):events.append('observe');return contexts[self.index]
            def choices(self):return choices
            def apply(self,context,choice):
                clock.now+=.5;self.index+=1
                return contexts[self.index]['context'] if self.index<len(contexts) else dict(context,terminal=True)
            def call(self,name):return result()
        episode=game();episode['records']=[];lease=Lease()
        _run_turns(Host(),{},[planner]*4,Timings(),5,contexts[0]['context'],episode,
            [self.rng]*4,[],PreparedEncoder(),lease)
        self.assertEqual(events[0],'admission');self.assertEqual(lease.assert_workers,4)
        self.assertEqual(planner.calls,[(248.8,16)])
        self.assertTrue(all(row['teacher'] is None for row in episode['records']))
        report=planner.work_report(self.rng)['phases']['canal_early']
        self.assertEqual((report['scheduled_turns'],report['actual_searches']),(1,1))
        self.assertEqual(report['continuation_choices'],3)

    def test_failed_search_never_retries_and_next_turn_has_a_new_allowance(self):
        view,choices=position();self.planner.mode='raise'
        self.planner.admit_work_turn(view['context'],self.rng)
        for _ in range(3):self.assertIsNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
        self.assertEqual(len(self.planner.calls),1)
        view['context'].update(turn=view['context']['turn']+1,round=2)
        self.planner.mode='success';self.planner.admit_work_turn(view['context'],self.rng)
        self.planner.select_batch([(view,choices,self.rng)])
        counts=self.planner.work_report(self.rng)['phases']['canal_early']
        self.assertEqual((counts['actual_searches'],counts['failures']),(2,1))

    def test_existing_cadence_and_search_cap_ignore_debt_and_subchoices(self):
        self.planner.config=teacher_search_config(dict(turn_interval=4,max_searches_per_seat=2))
        view,choices=position();view['context'].update(turn=0,phase='debt')
        self.assertEqual(self.planner.work_workers(view['context'],self.rng),1)
        self.planner.admit_work_turn(view['context'],self.rng)
        self.planner.select_batch([(view,choices,self.rng)])
        self.assertEqual(self.planner.teacher_states[self.rng]['turns'],0)
        for turn in range(1,11):
            view['context'].update(turn=turn,phase='action',round=(turn-1)%8+1,
                era='CanalEra' if turn<=8 else 'RailEra',revision=turn*10)
            self.assertEqual(self.planner.work_workers(view['context'],self.rng),4 if turn in (1,5) else 1)
            self.planner.admit_work_turn(view['context'],self.rng)
            self.planner.select_batch([(view,choices,self.rng)])
            view['context']['phase']='payment'
            self.planner.admit_work_turn(view['context'],self.rng)
            self.planner.select_batch([(view,choices,self.rng)])
            self.planner.finish_work_turn(self.rng)
        report=self.planner.work_report(self.rng)
        self.assertEqual((report['turns'],report['search_turns'],len(self.planner.calls)),(10,2,2))
        self.assertEqual((report['turn_interval'],report['max_searches_per_seat']),(4,2))

    def test_wrapper_has_separate_identity_without_mutating_default_policy_contract(self):
        policy=object.__new__(MaskedSearchPolicy);policy.config=teacher_search_config({});policy.policy_id='original'
        old=policy.config.to_dict();wrapped=schedule.work_search(policy,NN_WORK_BUDGET)
        self.assertIs(wrapped,policy);self.assertIsInstance(wrapped,schedule.WorkMasked)
        self.assertFalse(hasattr(wrapped,'begin_teacher_game'))
        self.assertEqual(policy.config.to_dict(),old)
        self.assertTrue(policy.policy_id.startswith('step5-work-search/'))
        self.assertEqual(schedule.turn_seconds(policy),120.)


if __name__=='__main__':unittest.main()
