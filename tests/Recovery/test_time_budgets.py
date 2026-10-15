"""Shared training clocks, legal retirement, archived data and exact resume."""
import copy
import random
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event,RLock,get_ident
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from fixtures import position,game
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.replay import validate_teacher
from brass_ai.recovery import teacher_schedule as schedule
from brass_ai.recovery.config import (TEACHER_TIME_BUDGET,NN_TIME_BUDGET,LEGACY_TIME_BUDGET,
    TEACHER_WORK_BUDGET,LEGACY_OPPONENT_WORK_BUDGET,teacher_search_config,
    teacher_work_budget_config,nn_work_budget_config,legacy_opponent_work_budget_config,
    time_budget_config,validate_config,SEARCH_WALL_FRACTION,TEACHER_PHASE_TIME_BUDGET_V3,latest_model_recipe)
from brass_ai.recovery.legacy_search import LabLegacyPolicy
from brass_ai.recovery.resume import check_signature
from brass_ai.search.common import SearchIncomplete
from brass_ai.search.ismcts import MaskedISMCTS,information_key,visit_teacher
from brass_ai.step5.evaluation import evaluation_config
from brass_ai.step5.league import implementation_id
from test_teacher_schedule import Clock,evidence
import test_legacy_opponent_work as legacy_fixture


class Decider:
    def decide(self,view,choices,rng,deadline,recall,tree,cache):
        self.calls.append(deadline)
        if self.mode=='exception':raise SearchIncomplete('inference expired')
        if self.mode=='expiry':self.clock.now=deadline
        elif self.mode=='overrun':self.clock.now=deadline+.3
        else:self.clock.now+=.01
        self.last_statistics=evidence(view,choices,recall,self.mode!='coverage')
        self.last_statistics['batching']=dict(attempted=self.last_statistics['completed'])
        self.last_continuation={information_key(v,choices,recall):choices[1]['id'] for v in self.next_views}
        teacher=visit_teacher(choices,self.last_statistics,[.6,.3,.1],model='a'*64,observer=view['seat'],config=self.config)
        return (1 if self.mode=='fallback' else 0),1.,teacher


class Planner(schedule.TeacherTurns,Decider):
    def __init__(self,clock):
        self.clock=clock;self.mode='expiry';self.calls=[];self.next_views=[]
        self.config=teacher_search_config({});self.teacher_work_budget=copy.deepcopy(TEACHER_TIME_BUDGET)
        self.teacher_states=weakref.WeakKeyDictionary();self.cancelled=Event()
        self.base=SimpleNamespace(request_deadline=None,select_batch=lambda inputs:[(2,1.,None)]*len(inputs))


class TimeTeacherTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.clock.now=100.;self.rng=random.Random(9)
        for mock in (patch.object(schedule.time,'monotonic',self.clock),
                     patch('brass_ai.step5.league.implementation_id',return_value='b'*64)):
            mock.start();self.addCleanup(mock.stop)
        self.policy=Planner(self.clock);self.policy.begin_teacher_game(self.rng,0,21,33)

    def view(self,phase='canal_early'):
        view,choices=position();view['context'].update(
            era=next(e for p,e,_,_ in schedule.PHASES if p==phase),
            round=self.policy.teacher_states[self.rng]['rounds'][phase],actions_remaining=2)
        return view,choices

    def test_admission_starts_single_deadline_and_expiry_keeps_partial_evidence(self):
        view,choices=self.view();self.clock.now+=40 # CPU admission queue
        self.policy.admit_teacher_turn(view['context'],self.rng)
        current=self.policy.teacher_states[self.rng]['current'];self.assertEqual(current['deadline'],149.)
        for phase in ('payment','action','continuation'):
            future=copy.deepcopy(view);future['context'].update(phase=phase,revision=len(self.policy.next_views)+1,actions_remaining=1)
            self.policy.next_views.append(future)
        chosen,_,teacher=self.policy.select_batch([(view,choices,self.rng)])[0]
        validate_teacher(teacher,len(choices))
        for future in self.policy.next_views:
            self.policy.admit_teacher_turn(future['context'],self.rng)
            self.clock.now+=.01
            self.assertEqual(self.policy.select_batch([(future,choices,self.rng)])[0],(1,1.,None))
            self.assertEqual(current['deadline'],149.)
        self.assertEqual(self.policy.calls,[148.85])
        stats=self.policy.teacher_report(self.rng)['phases']['canal_early']
        self.assertEqual(stats['attempted_simulations'],5) # Former quota was 32.
        self.assertEqual(stats['qualified_samples'],1);self.assertEqual(stats['timeouts'],0)
        self.assertEqual(stats['allocated_simulations'],0)
        self.assertEqual(stats['time_budget_exhausted_turns'],1)

    def test_early_return_does_not_wait_or_repeat_search(self):
        view,choices=self.view();self.policy.mode='fast'
        self.policy.admit_teacher_turn(view['context'],self.rng)
        self.policy.select_batch([(view,choices,self.rng)])
        self.assertAlmostEqual(self.clock.now,100.01)
        self.policy.select_batch([(view,choices,self.rng)])
        self.assertEqual(len(self.policy.calls),1)

    def test_forced_choices_late_phase_and_failure_quality_gates(self):
        for mode in ('overrun','exception','coverage','fallback','cancelled'):
            with self.subTest(mode=mode):
                self.clock.now=100.;self.policy=Planner(self.clock);self.policy.mode=mode
                self.policy.begin_teacher_game(self.rng,0,21,33);view,choices=self.view()
                self.policy.admit_teacher_turn(view['context'],self.rng)
                self.assertEqual(self.policy.select_batch([(view,choices[:1],self.rng)])[0],(0,1.,None))
                self.assertFalse(self.policy.calls)
                if mode=='cancelled':self.policy.cancelled.set()
                self.assertIsNone(self.policy.select_batch([(view,choices,self.rng)])[0][2])
                self.policy.cancelled.clear();self.policy.mode='fast'
                view,choices=self.view('rail_late');self.policy.admit_teacher_turn(view['context'],self.rng)
                self.assertIsNotNone(self.policy.select_batch([(view,choices,self.rng)])[0][2])
                stats=self.policy.teacher_report(self.rng)['phases']
                self.assertEqual(stats['canal_early']['qualified_samples'],0)
                self.assertEqual(stats['rail_late']['qualified_samples'],1)

    def test_half_cap_uses_old_accumulated_cost_and_defers_without_reset(self):
        self.assertIsNone(SEARCH_WALL_FRACTION)
        self.assertEqual(schedule.search_allowance(100.,80.,.5),20.)
        self.assertEqual(schedule.search_allowance(100.,300.,.5),0.)
        self.assertEqual(schedule.search_allowance(400.,300.,.5),100.)
        self.assertEqual(schedule.search_allowance(100.,80.,.2),0.)
        with self.assertRaises(ValueError):schedule.search_allowance(10.,0.,1.)

    def test_four_workers_share_one_deadline_and_retire_a_late_owner(self):
        view,choices=self.view();deadlines=[];retired=[]
        class Replica:
            def __init__(self,index):self.index=index;self.base=SimpleNamespace(contexts={})
            def decide(self,view,choices,rng,deadline,recall,tree,cache):
                try:
                    deadlines.append(deadline)
                    if self.index==3:raise SearchIncomplete('late worker')
                    self.last_statistics=evidence(view,choices,recall)
                    self.last_statistics['batching']=dict(attempted=5)
                finally:retired.append(self.index)
        planner=object.__new__(schedule.ScheduledParallelMasked)
        planner.teacher_states=weakref.WeakKeyDictionary();planner.cancelled=Event()
        planner.teacher_work_budget=TEACHER_TIME_BUDGET;planner.time_budget_primary=True
        planner.config=teacher_search_config({});planner.decision_lock=RLock()
        planner.replicas=[Replica(i) for i in range(4)]
        planner.base=SimpleNamespace(request_deadline=None,evaluator=SimpleNamespace(model_id='a'*64),
            predict=lambda inputs:[(np.array([.6,.3,.1]),{})])
        planner.incumbent=lambda *args:{};planner.context_features=lambda *args:([0.]*4,[0.]*8)
        planner.telemetry=dict(searches=0,simulations=0,incomplete=0,max_pending=0)
        planner.begin_teacher_game(self.rng,0,21,33);planner.admit_teacher_turn(view['context'],self.rng)
        with ThreadPoolExecutor(max_workers=4) as pool:
            planner.pool=pool
            self.assertIsNotNone(planner.select_batch([(view,choices,self.rng)])[0][2])
        self.assertEqual(deadlines,[108.85]*4);self.assertEqual(set(retired),set(range(4)))
        self.assertEqual(planner.last_statistics['completed'],15)

    def test_best_once_completed_plan_survives_without_a_teacher_label(self):
        view,choices=self.view();clock=self.clock
        class Result:
            def decide(self,*args):
                self.last_statistics=evidence(view,choices,[])
                for plan in self.last_statistics['plans'].values():plan['visits']=1
                return 2,1.,None
        class Salvage(schedule.TurnSearch,Result):pass
        planner=Salvage();planner.config=self.policy.config;planner.time_budget_primary=True
        planner.cancelled=Event();planner.last_continuation={}
        selected,_,teacher=planner.decide(view,choices,self.rng,clock.now+12.)
        self.assertEqual(selected,0);self.assertTrue(planner.last_continuation);self.assertIsNone(teacher)

    def test_time_primary_loop_can_exceed_old_count_and_stops_without_waits(self):
        clock=self.clock
        bot=object.__new__(MaskedISMCTS);bot.config=teacher_search_config({'simulations':1,'pending_paths':1})
        bot.time_budget_primary=True;bot.deadline=100.035;bot.lanes=[None]
        bot.cancelled=None;bot.owner=get_ident()
        bot.completed=bot.interrupted=bot.max_pending=0;bot.metrics={}
        def trajectory(*args):
            clock.now+=.01;bot.completed+=1
            yield from ()
        bot.trajectory=trajectory;bot.statistics=lambda choices:bot.metrics
        result=bot.run({},[])
        self.assertEqual(result['attempted'],4);self.assertEqual(bot.completed,4)
        self.assertEqual(bot.interrupted,0);self.assertAlmostEqual(clock.now,100.04)

    def test_expired_inflight_generators_are_retired_once_without_backup(self):
        clock=self.clock;retired=[]
        bot=object.__new__(MaskedISMCTS);bot.config=teacher_search_config({'simulations':1,'pending_paths':2})
        bot.time_budget_primary=True;bot.deadline=100.1;bot.lanes=[None,None]
        bot.cancelled=None;bot.owner=get_ident()
        bot.completed=bot.interrupted=bot.max_pending=0;bot.metrics=dict(coalesced=0)
        def trajectory(*args):
            try:
                clock.now+=.051
                yield ('request',None,{},[],None)
                raise AssertionError('Expired work resumed')
            finally:retired.append(True)
        bot.trajectory=trajectory;bot.statistics=lambda choices:bot.metrics
        bot.run({},[])
        self.assertEqual(len(retired),2);self.assertEqual(bot.completed,0)
        self.assertEqual(bot.metrics['attempted'],2);self.assertEqual(bot.interrupted,2)

    def test_collection_admission_defers_whole_batches_and_never_retries(self):
        from test_selfplay_70_review import TeacherCollectionTests
        fixture=TeacherCollectionTests()
        (report,next_due,cursor,spent),jobs,calls=fixture.run_fixture(ordinary=100.,fraction=.5)
        self.assertEqual((len(jobs),len(calls),next_due,cursor,spent),(0,0,800,18000000,10.))
        self.assertTrue(report['budget']['deferred']);self.assertEqual(report['budget']['wall_fraction'],.5)
        (report,next_due,_,spent),jobs,calls=fixture.run_fixture(ordinary=1430.,fraction=.5)
        self.assertEqual((len(jobs),len(calls),next_due,spent),(5,1,960,210.))
        self.assertTrue(report['budget']['deferred'])
        (report,next_due,_,spent),jobs,calls=fixture.run_fixture(ordinary=10000.,failed=True)
        self.assertEqual((len(jobs),len(calls),next_due,spent),(10,2,1120,410.))
        self.assertFalse(report['updated'])


class TimeLegacyTests(unittest.TestCase):
    def setUp(self):
        # Reuse the old frozen-worker double, without inheriting fixed-work tests.
        legacy_fixture.LegacyOpponentWorkTests.setUp(self);self.work=copy.deepcopy(LEGACY_TIME_BUDGET)
    policy=legacy_fixture.LegacyOpponentWorkTests.policy

    def test_time_expiry_below_old_quota_is_not_a_game_failure(self):
        p=self.policy();view,choices=position();self.partial=True;self.delay=.981
        self.clock.now=140.;p.admit_legacy_turn(view['context'],self.rng)
        self.assertEqual(p.select_batch([(view,choices,self.rng)])[0][0],0)
        self.assertAlmostEqual(self.calls[0][1]['profile']['milliseconds'],980.)
        self.assertEqual(self.calls[0][1]['search_workers'],4)
        self.assertEqual(p.telemetry['attempted_simulations'],4)
        self.assertEqual(p.telemetry['failures'],0)
        self.clock.now=141.01;view['context']['revision']+=1;view['context']['actions_remaining']=1
        self.assertEqual(p.select_batch([(view,choices,self.rng)])[0][0],2)
        self.assertEqual(len(self.calls),1)
        p.contexts[self.rng]['queue'].clear();view['context']['revision']+=1
        p.select_batch([(view,choices,self.rng)])
        self.assertEqual(len(self.calls),1);self.assertEqual(p.contexts[self.rng]['deadline'],141.)

    def test_forced_choice_and_duplicate_revision_never_reset_deadline(self):
        p=self.policy();view,choices=position();p.admit_legacy_turn(view['context'],self.rng)
        p.select_batch([(view,choices[:1],self.rng)]);self.clock.now+=.3
        p.select_batch([(view,choices,self.rng)])
        self.assertAlmostEqual(self.calls[0][1]['profile']['milliseconds'],680.)
        self.clock.now+=1.;p.admit_legacy_turn(view['context'],self.rng)
        p.select_batch([(view,choices,self.rng)])
        self.assertEqual(len(self.calls),1)

    def test_search_free_keeps_its_old_incumbent_limits(self):
        p=self.policy(search_free=True);view,choices=position();p.select_batch([(view,choices,self.rng)])
        request=self.calls[0][1]
        self.assertTrue(request['profile']['incumbent_only'])
        self.assertEqual(request['profile']['work_limit'],4096)
        self.assertEqual(request['profile']['milliseconds'],120000.)


class TimeConfigTests(unittest.TestCase):
    def test_active_training_and_evaluation_presets_and_old_contracts(self):
        for path in (SOURCE_ROOT/'configs/step5').glob('*.json'):
            if path.name.startswith('previous-'):continue
            cfg=read(path)
            if cfg['format']=='step5-training/2.2.0':
                validate_config(cfg);self.assertEqual(time_budget_config(cfg),cfg)
                self.assertEqual(cfg['teacher_work_budget'],TEACHER_TIME_BUDGET if latest_model_recipe(cfg) else TEACHER_PHASE_TIME_BUDGET_V3)
                self.assertIsNone(cfg['search_wall_fraction'])
            else:self.assertEqual(evaluation_config(cfg),cfg);self.assertEqual(cfg['nn_work_budget'],NN_TIME_BUDGET)
            self.assertEqual(cfg['legacy_opponent_work_budget'],LEGACY_TIME_BUDGET)
            self.assertEqual(cfg['search']['seconds'],12.)
            self.assertEqual(cfg['legacy_budget']['seconds'],1.)
        self.assertEqual(teacher_work_budget_config(TEACHER_WORK_BUDGET),TEACHER_WORK_BUDGET)
        self.assertEqual(legacy_opponent_work_budget_config(LEGACY_OPPONENT_WORK_BUDGET),LEGACY_OPPONENT_WORK_BUDGET)

    def test_no_mixed_or_multiplied_budget_contract(self):
        for value,validator in ((TEACHER_TIME_BUDGET,teacher_work_budget_config),(NN_TIME_BUDGET,nn_work_budget_config),
                                (LEGACY_TIME_BUDGET,legacy_opponent_work_budget_config)):
            for changed in (dict(value,simulations_per_turn=16),dict(value,seconds_per_turn=value.get('seconds_per_turn',12)*4),
                            dict(value,seconds_per_turn=True),dict(value,mode='simulations')):
                with self.assertRaises(ValueError):validator(changed)
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        for fraction in (0,True,.91,1.,float('nan')):
            with self.assertRaises(ValueError):time_budget_config(dict(cfg,search_wall_fraction=fraction))


if __name__=='__main__':unittest.main()
