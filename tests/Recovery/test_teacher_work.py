"""Whole-turn attempted-work caps without worker/action multiplication."""
import copy
import random
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event,RLock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fixtures import position,game
from brass_ai.learning.replay import validate_teacher
from brass_ai.recovery import teacher_schedule as schedule
from brass_ai.recovery.config import TEACHER_WORK_BUDGET,NN_WORK_BUDGET,teacher_search_config
from brass_ai.recovery.parallel import ParallelMasked,simulation_quotas
from brass_ai.search.common import SearchConfig,SearchIncomplete
from brass_ai.search.ismcts import MaskedISMCTS,information_key,visit_teacher
from test_teacher_schedule import Clock,evidence


class WorkDecider:
    def decide(self,view,choices,rng,deadline,recall,tree,cache):
        self.calls.append((deadline,self.config.simulations))
        self.clock.now+=.2
        if self.mode=='slow_normal':self.clock.now+=14.
        if self.mode=='timeout':self.clock.now=deadline+2.
        if self.mode=='work_cutoff':self.clock.now=deadline+.1
        if self.mode=='raise':raise SearchIncomplete('failed root')
        stats=evidence(view,choices,recall,self.mode!='coverage')
        stats['batching']=dict(attempted=self.config.simulations-int(self.mode=='partial'))
        self.last_statistics=stats
        self.last_continuation={information_key(v,choices,recall):choices[1]['id'] for v in self.next_views}
        teacher=visit_teacher(choices,stats,[.6,.3,.1],model='a'*64,observer=view['seat'],config=self.config)
        return 0,1.,teacher


class WorkPlanner(schedule.TeacherTurns,WorkDecider):
    def __init__(self,clock):
        self.clock=clock;self.calls=[];self.mode='success';self.next_views=[]
        self.teacher_states=weakref.WeakKeyDictionary();self.cancelled=Event()
        self.teacher_work_budget=copy.deepcopy(TEACHER_WORK_BUDGET)
        self.config=teacher_search_config({})
        self.base=SimpleNamespace(request_deadline=None,select_batch=lambda inputs:[(2,1.,None)]*len(inputs))


class TeacherWorkTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.clock.now=100.
        self.clock_patch=patch.object(schedule.time,'monotonic',self.clock);self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        source=patch('brass_ai.step5.league.implementation_id',return_value='b'*64);source.start();self.addCleanup(source.stop)
        self.planner=WorkPlanner(self.clock);self.rng=random.Random(19)
        self.planner.begin_teacher_game(self.rng,0,1200,3400)

    def view(self,phase='canal_early'):
        view,choices=position()
        view['context'].update(era=next(e for p,e,_,_ in schedule.PHASES if p==phase),
            round=self.planner.teacher_states[self.rng]['rounds'][phase],actions_remaining=2)
        return view,choices

    def test_one_work_quota_covers_both_actions_and_subchoices_without_waiting(self):
        view,choices=self.view();self.planner.admit_teacher_turn(view['context'],self.rng)
        self.assertEqual(self.planner.select_batch([(view,choices[:1],self.rng)])[0],(0,1.,None))
        self.assertEqual(self.planner.calls,[])
        for phase,remaining in [('payment',1),('action',1),('continuation',0)]:
            future=copy.deepcopy(view);future['context'].update(phase=phase,actions_remaining=remaining,
                revision=len(self.planner.next_views)+1)
            self.planner.next_views.append(future)
        root=self.planner.select_batch([(view,choices,self.rng)])[0]
        self.assertIsNotNone(root[2]);validate_teacher(root[2],len(choices))
        for future in self.planner.next_views:
            self.assertEqual(self.planner.select_batch([(future,choices,self.rng)])[0],(1,1.,None))
        self.assertEqual(self.planner.calls,[(218.8,32)])
        self.assertEqual(self.planner.config.simulations,100000)
        counts=self.planner.teacher_report(self.rng)['phases']['canal_early']
        self.assertEqual((counts['allocated_simulations'],counts['attempted_simulations']),(32,32))
        self.assertEqual((counts['actual_searches'],counts['qualified_samples'],counts['work_exhausted_turns']),(1,1,1))
        self.assertEqual((counts['timeouts'],counts['deadline_limited_turns']),(0,0))
        self.assertEqual(counts['continuation_choices'],3)
        self.assertAlmostEqual(counts['turn_elapsed_seconds'],.2)

    def test_failed_work_is_not_recycled_and_late_rail_still_reserved(self):
        self.planner.mode='coverage'
        view,choices=self.view();self.planner.admit_teacher_turn(view['context'],self.rng)
        for _ in range(3):self.assertIsNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
        self.assertEqual(len(self.planner.calls),1)
        self.planner.mode='success'
        view,choices=self.view('rail_late');self.planner.admit_teacher_turn(view['context'],self.rng)
        self.assertIsNotNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
        report=self.planner.teacher_report(self.rng)['phases']
        self.assertEqual(report['canal_early']['qualified_samples'],0)
        self.assertEqual(report['canal_early']['insufficient_coverage'],1)
        self.assertEqual(report['rail_late']['allocated_simulations'],16)
        self.assertEqual(report['rail_late']['qualified_samples'],1)

    def test_safety_deadline_and_exception_are_not_successful_work(self):
        for mode in ('timeout','raise'):
            with self.subTest(mode=mode):
                self.clock.now=100.;self.planner=WorkPlanner(self.clock);self.planner.mode=mode
                self.planner.begin_teacher_game(self.rng,0,1200,3400)
                view,choices=self.view();self.planner.admit_teacher_turn(view['context'],self.rng)
                self.assertIsNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
                self.planner.select_batch([(view,choices,self.rng)])
                counts=self.planner.teacher_report(self.rng)['phases']['canal_early']
                self.assertEqual((counts['qualified_samples'],counts['actual_searches']),(0,1))
                self.assertEqual(counts['deadline_limited_turns'],1)
                self.assertEqual(counts['work_exhausted_turns'],int(mode=='timeout'))
                self.assertEqual(self.planner.config.simulations,100000)
                self.assertIsNone(self.planner.teacher_simulation_allowance)

    def test_work_cutoff_preserves_supported_evidence_before_hard_deadline(self):
        self.planner.teacher_work_budget=dict(contract='teacher-turn-work/1.0.0',mode='simulations',
            simulations_per_phase=copy.deepcopy(TEACHER_WORK_BUDGET['simulations_per_phase']))
        self.planner.mode='work_cutoff'
        view,choices=self.view();self.planner.admit_teacher_turn(view['context'],self.rng)
        self.assertIsNotNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
        counts=self.planner.teacher_report(self.rng)['phases']['canal_early']
        self.assertEqual((counts['qualified_samples'],counts['timeouts'],counts['deadline_limited_turns']),(1,0,1))

    def test_count_primary_watchdog_and_partial_work_do_not_qualify(self):
        for mode,reason in [('work_cutoff','timeouts'),('partial','insufficient_coverage')]:
            with self.subTest(mode=mode):
                self.clock.now=100.;self.planner=WorkPlanner(self.clock);self.planner.mode=mode
                self.planner.begin_teacher_game(self.rng,0,1200,3400)
                view,choices=self.view();self.planner.admit_teacher_turn(view['context'],self.rng)
                self.assertIsNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
                self.planner.select_batch([(view,choices,self.rng)])
                counts=self.planner.teacher_report(self.rng)['phases']['canal_early']
                self.assertEqual((counts['actual_searches'],counts['qualified_samples'],counts[reason]),(1,0,1))

    def test_simulation_budget_can_run_past_old_twelve_second_clock(self):
        self.planner.mode='slow_normal'
        view,choices=self.view();self.planner.admit_teacher_turn(view['context'],self.rng)
        self.assertIsNotNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
        counts=self.planner.teacher_report(self.rng)['phases']['canal_early']
        self.assertGreater(counts['elapsed_seconds'],12.)
        self.assertEqual((counts['attempted_simulations'],counts['qualified_samples']),(32,1))
        self.assertEqual((counts['timeouts'],counts['deadline_limited_turns']),(0,0))

    def test_old_explicit_work_contract_retains_its_original_clock(self):
        self.planner.teacher_work_budget=dict(contract='teacher-turn-work/1.0.0',mode='simulations',
            simulations_per_phase=copy.deepcopy(TEACHER_WORK_BUDGET['simulations_per_phase']))
        view,choices=self.view();self.planner.admit_teacher_turn(view['context'],self.rng)
        self.planner.select_batch([(view,choices,self.rng)])
        self.assertEqual(self.planner.calls,[(110.8,32)])

    def test_receipts_enforce_shared_quota_and_preserve_legacy_counter_checks(self):
        view,choices=self.view();self.planner.admit_teacher_turn(view['context'],self.rng)
        teacher=self.planner.select_batch([(view,choices,self.rng)])[0][2]
        reports=[self.planner.teacher_report(self.rng)]
        for seat in range(1,4):
            rng=random.Random(seat);self.planner.begin_teacher_game(rng,seat,1200,3400)
            reports.append(self.planner.teacher_report(rng))
        episode=game();episode['records']=[dict(seat=0,view=view,choices=choices,teacher=teacher)]
        records=[dict(game_id=episode['game_id'],teacher_phases=reports)]
        self.assertEqual(schedule.collection_report(records,[episode])['phases']['canal_early']['allocated_simulations'],32)
        corrupt=copy.deepcopy(records)
        corrupt[0]['teacher_phases'][0]['phases']['canal_early']['attempted_simulations']=33
        with self.assertRaisesRegex(ValueError,'shared turn allowance'):schedule.collection_report(corrupt,[episode])
        legacy=copy.deepcopy(records)
        for report in legacy[0]['teacher_phases']:
            report.pop('work_budget')
            for counts in report['phases'].values():
                for key in schedule.WORK_COUNTERS:counts.pop(key)
        self.assertEqual(schedule.collection_report(legacy,[episode])['phases']['canal_early']['qualified_samples'],1)
        legacy[0]['teacher_phases'][0]['phases']['canal_early'].pop('timeouts')
        with self.assertRaises(KeyError):schedule.collection_report(legacy,[episode])

    def forest(self,total,fail=False):
        view,choices=self.view();calls=[]
        class Replica:
            def __init__(self,index):
                self.index=index;self.base=SimpleNamespace(contexts={});self.config=SearchConfig(simulations=100000)
            def decide(self,view,choices,rng,deadline,recall,tree,cache):
                count=self.config.simulations;calls.append((self.index,count,deadline))
                if fail and self.index==0:raise SearchIncomplete('fixture interruption')
                stats=evidence(view,choices,recall)
                # Split each private quota across two exact legal roots.
                counts=[(count+1)//2,count//2,0]
                stats['completed']=count;stats['batching']=dict(attempted=count)
                stats['root']={c['id']:dict(visits=n,returns=[n*.25]*4,available=n)
                    for c,n in zip(choices,counts) if n}
                for plan in stats['plans'].values():plan['visits']=counts[0];plan['returns']=[counts[0]*.25]*4
                self.last_statistics=stats
        planner=object.__new__(ParallelMasked);planner.config=teacher_search_config({});planner.decision_lock=RLock()
        planner.replicas=[Replica(i) for i in range(4)];planner.teacher_simulation_allowance=total
        planner.base=SimpleNamespace(request_deadline=None,evaluator=SimpleNamespace(model_id='a'*64),
            predict=lambda inputs:[(np.array([.6,.3,.1]),{})])
        planner.incumbent=lambda *args:{};planner.context_features=lambda *args:([0.]*4,[0.]*8)
        planner.telemetry=dict(searches=0,simulations=0,incomplete=0,max_pending=0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            planner.pool=pool
            if fail:
                with self.assertRaises(SearchIncomplete):planner.decide(view,choices,self.rng,110.8,[],{},{} )
                result=None
            else:result=planner.decide(view,choices,self.rng,110.8,[],{},{})
        self.assertTrue(all(w.config.simulations==100000 for w in planner.replicas))
        return planner,sorted(calls),result

    def test_four_worker_quotas_sum_to_turn_cap_and_labels_use_aggregate_cap(self):
        planner,calls,result=self.forest(19)
        self.assertEqual([c[1] for c in calls],[5,5,5,4])
        self.assertEqual({c[2] for c in calls},{110.8})
        self.assertEqual(planner.last_statistics['batching']['attempted'],19)
        self.assertEqual(result[2]['search']['simulations'],19)
        validate_teacher(result[2],3)
        self.assertEqual(simulation_quotas(2,4,10),[1,1,0,0])
        _,calls,_=self.forest(2)
        self.assertEqual(len(calls),2);self.assertEqual(sum(c[1] for c in calls),2)
        with self.assertRaisesRegex(ValueError,'worker cap'):simulation_quotas(41,4,10)

    def test_private_worker_caps_restore_after_all_workers_retire_on_failure(self):
        _,calls,_=self.forest(19,fail=True)
        self.assertEqual(len(calls),4);self.assertEqual(sum(c[1] for c in calls),19)

    def test_kernel_reports_attempts_even_when_trajectories_fail(self):
        for failed in (False,True):
            bot=object.__new__(MaskedISMCTS);bot.config=SimpleNamespace(simulations=7,pending_paths=2)
            bot.lanes=[None,None];bot.deadline=110.;bot.completed=bot.interrupted=bot.max_pending=0;bot.metrics={}
            def trajectory(*args):
                if failed:raise SearchIncomplete('incomplete path')
                bot.completed+=1
                if False:yield None
            bot.trajectory=trajectory;bot.statistics=lambda choices:dict(completed=bot.completed,
                interrupted=bot.interrupted,batching=bot.metrics)
            result=bot.run({},[])
            self.assertEqual(result['batching']['attempted'],7)
            self.assertEqual(result['completed'],0 if failed else 7)
            self.assertEqual(result['interrupted'],7 if failed else 0)


if __name__=='__main__':unittest.main()
