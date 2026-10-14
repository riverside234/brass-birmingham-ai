"""Phase clocks, unlimited cumulative admission, distinct rows and exact resume."""
import copy
import json
import random
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event,RLock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import teacher_schedule as schedule
from brass_ai.recovery.config import (TEACHER_TIME_BUDGET,TEACHER_TIME_BUDGET_V1,TEACHER_WORK_BUDGET,
    TEACHER_PHASE_TIME_BUDGET_V1,archived_phase_time_budget_config,
    TEACHER_PHASE_TIME_BUDGET_V2,archived_early_rail_time_budget_config,teacher_time_seconds,
    UPDATE_V21,SEARCH_WALL_FRACTION,time_budget_config,archived_time_budget_config,
    teacher_work_budget_config,teacher_search_config,validate_config)
from brass_ai.recovery.history import compact
from brass_ai.recovery.teacher_collection import summarize
from brass_ai.recovery.updates import distill_update
from test_selfplay_review import teacher_games
from test_teacher_schedule import evidence
import test_time_budgets as clock_fixture
import test_time_resume as resume_fixture


class PhaseClockTests(unittest.TestCase):
    setUp=clock_fixture.TimeTeacherTests.setUp
    view=clock_fixture.TimeTeacherTests.view

    def test_exact_phase_definitions_and_one_clock_across_actions(self):
        self.assertEqual(schedule.PHASES,(('canal_early','CanalEra',1,4),('canal_late','CanalEra',5,8),
            ('rail_early','RailEra',1,6),('rail_late','RailEra',7,8)))
        for phase,_,_,_ in schedule.PHASES:
            self.clock.now=200.;view,choices=self.view(phase)
            self.policy.admit_teacher_turn(view['context'],self.rng)
            current=self.policy.teacher_states[self.rng]['current']
            seconds={'canal_early':12.,'canal_late':16.,'rail_early':20.,'rail_late':20.}[phase]
            self.assertEqual(current['deadline'],200.+seconds)
            self.assertIsNone(self.policy.select_batch([(view,choices[:1],self.rng)])[0][2])
            future=copy.deepcopy(view);future['context'].update(phase='continuation',revision=1,actions_remaining=1)
            self.policy.next_views=[future]
            label=self.policy.select_batch([(view,choices,self.rng)])[0][2]
            self.assertIsNotNone(label);self.assertEqual(label['search']['seconds'],seconds)
            self.policy.admit_teacher_turn(future['context'],self.rng)
            self.assertEqual(self.policy.select_batch([(future,choices,self.rng)])[0],(1,1.,None))
            self.assertEqual(current['deadline'],200.+seconds)
            self.assertAlmostEqual(self.policy.calls[-1],200.+seconds-.2)
            self.policy.finish_teacher_turn(self.rng)
        report=self.policy.teacher_report(self.rng)
        self.assertEqual(len(self.policy.calls),4);self.assertIsNone(report['seconds_per_turn'])
        self.assertEqual(report['seconds_per_phase'],TEACHER_TIME_BUDGET['seconds_per_phase'])
        counts=report['phases']['rail_late']
        self.assertEqual((counts['allocated_seconds'],counts['completed_searches'],counts['qualified_samples']),(20.,1,1))
        self.assertEqual((counts['legal_root_actions'],counts['visited_root_actions'],counts['revisited_root_actions']),(3,2,2))

    def test_four_workers_share_phase_deadlines(self):
        for phase,seconds in (('canal_late',16.),('rail_early',20.),('rail_late',20.)):
            with self.subTest(phase=phase):self.check_worker_deadlines(phase,seconds)

    def check_worker_deadlines(self,phase,seconds):
        view,choices=self.view(phase);deadlines=[]
        class Replica:
            def __init__(self):self.base=SimpleNamespace(contexts={})
            def decide(self,view,choices,rng,deadline,recall,tree,cache):
                deadlines.append(deadline);self.last_statistics=evidence(view,choices,recall)
                self.last_statistics['batching']=dict(attempted=5)
        planner=object.__new__(schedule.ScheduledParallelMasked)
        planner.teacher_states=weakref.WeakKeyDictionary();planner.cancelled=Event();planner.decision_lock=RLock()
        planner.teacher_work_budget=TEACHER_TIME_BUDGET;planner.time_budget_primary=True
        planner.config=teacher_search_config({});planner.replicas=[Replica() for _ in range(4)]
        planner.base=SimpleNamespace(request_deadline=None,evaluator=SimpleNamespace(model_id='a'*64),
            predict=lambda inputs:[(np.array([.6,.3,.1]),{})])
        planner.incumbent=lambda *args:{};planner.context_features=lambda *args:([0.]*4,[0.]*8)
        planner.telemetry=dict(searches=0,simulations=0,incomplete=0,max_pending=0)
        planner.begin_teacher_game(self.rng,0,21,33);planner.admit_teacher_turn(view['context'],self.rng)
        with ThreadPoolExecutor(max_workers=4) as pool:
            planner.pool=pool;label=planner.select_batch([(view,choices,self.rng)])[0][2]
        self.assertEqual(deadlines,[100.+seconds-.2]*4);self.assertEqual(label['search']['seconds'],seconds)
        self.assertEqual(planner.teacher_states[self.rng]['current']['deadline'],100.+seconds)

    def test_late_failure_timeout_and_cancel_still_disqualify(self):
        for mode in ('overrun','exception','cancelled'):
            with self.subTest(mode=mode):
                self.setUp();view,choices=self.view('rail_late');self.policy.mode=mode
                self.policy.admit_teacher_turn(view['context'],self.rng)
                if mode=='cancelled':self.policy.cancelled.set()
                self.assertIsNone(self.policy.select_batch([(view,choices,self.rng)])[0][2])
                self.assertEqual(self.policy.teacher_report(self.rng)['phases']['rail_late']['failures'],1)


class PhaseConfigTests(unittest.TestCase):
    def test_generic_nn_search_keeps_uniform_twelve_seconds_in_late_rail(self):
        from brass_ai.search.ismcts import MaskedSearchPolicy
        from brass_ai.recovery.config import NN_TIME_BUDGET
        policy=object.__new__(MaskedSearchPolicy);policy.config=teacher_search_config({});policy.policy_id='fixture'
        schedule.work_search(policy,NN_TIME_BUDGET)
        for phase,_,_,_ in schedule.PHASES:self.assertEqual(schedule.turn_seconds(policy,phase),12.)

    def test_defaults_validation_archives_and_migration(self):
        self.assertIsNone(SEARCH_WALL_FRACTION)
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json');validate_config(cfg)
        self.assertIsNone(cfg['search_wall_fraction']);self.assertEqual(cfg['search_games'],5)
        self.assertEqual(cfg['distill_steps'],8);self.assertEqual(time_budget_config(cfg),cfg)
        self.assertIsNone(schedule.search_allowance(0.,1e12))
        self.assertEqual(schedule.search_allowance(100.,80.,.5),20.)
        for value in (None,.2,.5,.9):
            current=dict(cfg,search_wall_fraction=value)
            validate_config(current);self.assertEqual(time_budget_config(current),current)
        for value in (True,False,0.,-1.,.91,float('inf'),float('nan'),'None'):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_config(dict(cfg,search_wall_fraction=value))
        old=copy.deepcopy(cfg);old.update(search_games=4,search_wall_fraction=.5,teacher_work_budget=TEACHER_TIME_BUDGET_V1)
        original=copy.deepcopy(old);self.assertEqual(time_budget_config(old),cfg);self.assertEqual(old,original)
        self.assertEqual(archived_time_budget_config(old),old)
        teacher_work_budget_config(TEACHER_TIME_BUDGET_V1);teacher_work_budget_config(TEACHER_WORK_BUDGET)

    def test_collection_cost_and_phase_schema(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        self.assertEqual(schedule.estimate_collection(cfg,5.),1385.) # 5 * (4*(12+16+20+20) + 5)
        self.assertEqual(teacher_time_seconds(TEACHER_TIME_BUDGET),20.)
        missing=dict(cfg);missing.pop('teacher_work_budget')
        self.assertEqual(schedule.estimate_collection(missing,5.),1385.)
        self.assertEqual(schedule.estimate_collection(dict(cfg,teacher_work_budget=TEACHER_TIME_BUDGET_V1),5.),985.)
        for phase in TEACHER_TIME_BUDGET['seconds_per_phase']:
            broken=copy.deepcopy(TEACHER_TIME_BUDGET);del broken['seconds_per_phase'][phase]
            with self.assertRaises(ValueError):teacher_work_budget_config(broken)
        for value in (None,True,0.,-1.,121.,float('inf'),float('nan')):
            broken=copy.deepcopy(TEACHER_TIME_BUDGET);broken['seconds_per_phase']['rail_late']=value
            with self.assertRaises(ValueError):teacher_work_budget_config(broken)

    def test_phase_upgrades_preserve_archived_receipts_and_custom_settings(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        prior=dict(cfg,teacher_work_budget=copy.deepcopy(TEACHER_PHASE_TIME_BUDGET_V1))
        original=copy.deepcopy(prior)
        self.assertEqual(time_budget_config(prior),cfg);self.assertEqual(prior,original)
        self.assertEqual(archived_phase_time_budget_config(prior),prior)
        uniform=dict(prior,teacher_work_budget=TEACHER_TIME_BUDGET_V1,search_games=4,search_wall_fraction=.5)
        self.assertEqual(archived_phase_time_budget_config(uniform),prior)
        preceding=dict(prior,teacher_work_budget=copy.deepcopy(TEACHER_PHASE_TIME_BUDGET_V2))
        preceding_original=copy.deepcopy(preceding)
        self.assertEqual(archived_early_rail_time_budget_config(uniform),preceding)
        self.assertEqual(archived_early_rail_time_budget_config(prior),preceding)
        self.assertEqual(archived_early_rail_time_budget_config(preceding),preceding)
        self.assertEqual(time_budget_config(preceding),cfg);self.assertEqual(preceding,preceding_original)
        numeric=dict(prior,search_wall_fraction=.3)
        self.assertEqual(time_budget_config(numeric),dict(cfg,search_wall_fraction=.3))
        custom=copy.deepcopy(prior);custom['teacher_work_budget']['seconds_per_phase']['rail_early']=18.
        self.assertEqual(time_budget_config(custom),custom)

    def test_disabled_allowance_admits_two_five_game_collections_even_without_ordinary_credit(self):
        from test_selfplay_70_review import TeacherCollectionTests
        fixture=TeacherCollectionTests();original=fixture.fixture
        def configured():
            cfg,state,*rest=original();cfg['search_wall_fraction']=None;cfg['search_games']=5
            state['search_seconds']=1e9
            return cfg,state,*rest
        fixture.fixture=configured
        (report,next_due,cursor,spent),jobs,calls=fixture.run_fixture(ordinary=0.)
        self.assertEqual((len(jobs),len(calls),next_due),(10,2,1120))
        for key in ('seed','sampling_seed','path','index'):self.assertEqual(len({j[key] for j in jobs}),10)
        self.assertEqual([call.kwargs['steps'] for call in calls],[8,8])
        self.assertEqual(spent,1e9+400.)
        self.assertFalse(report['budget']['wall_fraction_enabled']);self.assertFalse(report['budget']['deferred'])
        self.assertIsNone(report['budget']['available_seconds']);self.assertIsNone(report['budget']['remaining_seconds'])
        json.dumps(report,allow_nan=False)

    def test_disabled_allowance_does_not_bypass_calibration_and_is_reported(self):
        from test_selfplay_70_review import TeacherCollectionTests
        from brass_ai.recovery.teacher_collection import collect_due
        cfg,state,service,heavy,_,_,jobs=TeacherCollectionTests().fixture()
        model=SimpleNamespace(config=SimpleNamespace(utility=cfg['search']['reward']),utility_calibration=None)
        report,next_due,cursor,spent=collect_due(SOURCE_ROOT,SOURCE_ROOT/'artifacts/mock',SOURCE_ROOT/'artifacts/mock/iteration-0070',
            cfg,state,{},model,None,service,heavy,done=960,cursor=18000000,ordinary_seconds=0.,ordinary_game_seconds=1.)
        self.assertEqual((jobs,next_due,cursor,spent),([],800,18000000,10.))
        self.assertTrue(report['budget']['deferred']);self.assertFalse(report['budget']['admitted'])
        self.assertIsNone(report['budget']['available_seconds']);json.dumps(report,allow_nan=False)


class DistinctUsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def run_update(self,count,steps=8):
        model=FourSeatNet(ModelConfig.preset('small'))
        return distill_update(model,torch.optim.AdamW(model.parameters(),lr=1e-5),teacher_games(count),UPDATE_V21,steps=steps)

    def test_five_games_use_all_80_positions_before_reuse(self):
        report=self.run_update(5)
        self.assertEqual((report['teacher_rows'],report['distinct_labels_used'],report['unused_qualified_labels']),(80,80,0))
        self.assertEqual((report['optimizer_steps'],report['sample_presentations'],report['max_observed_reuse']),(8,128,2))
        self.assertTrue(all(n==20 for n in report['used_samples_by_phase'].values()))
        self.assertEqual(sum(report['sample_presentations_by_phase'].values()),128)
        self.assertGreater(report['teacher_policy_drift']['categorical_kl'],0.)
        self.assertEqual(report['teacher_policy_drift']['positions'],80)

    def test_unused_rows_and_partial_batches_are_reported(self):
        report=self.run_update(5,steps=3)
        self.assertEqual((report['distinct_labels_used'],report['unused_qualified_labels'],report['sample_presentations']),(48,32,48))
        report=self.run_update(10)
        self.assertEqual((report['teacher_rows'],report['distinct_labels_used'],report['unused_qualified_labels']),(160,128,32))
        model=FourSeatNet(ModelConfig.preset('small'));games=teacher_games(5);games[-1]['records'][-1]['teacher']=None
        report=distill_update(model,torch.optim.AdamW(model.parameters()),games,UPDATE_V21)
        self.assertEqual((report['teacher_rows'],report['distinct_labels_used'],report['sample_presentations']),(79,79,127))

    def test_sparse_rejection_reports_zero_usage_and_duplicate_positions_fail(self):
        model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters())
        games=teacher_games(5)
        for g in games:
            for row in g['records']:
                if schedule.phase_for(row['view']['context'])=='rail_late':row['teacher']=None
        report=distill_update(model,opt,games,UPDATE_V21)
        self.assertFalse(report['updated']);self.assertEqual(report['distinct_labels_used'],0)
        self.assertEqual(report['unused_qualified_labels'],60);self.assertIsNone(report['teacher_policy_drift'])
        games=teacher_games(5);games[0]['records'].append(copy.deepcopy(games[0]['records'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate qualified'):distill_update(model,opt,games,UPDATE_V21)

    def test_retention_keeps_per_collection_usage_coverage_and_drift(self):
        batch=self.run_update(5);batch.update(collection=[],teaching=dict(contract=schedule.CONTRACT,
            phases={p:dict(schedule.counters(),insufficient_coverage=1,visited_root_actions=7) for p,_,_,_ in schedule.PHASES}))
        report=summarize([batch]);out=compact(dict(search=report))['search']
        self.assertEqual(out['distinct_labels_used'],80);self.assertEqual(out['unused_qualified_labels'],0)
        saved=out['collection_summaries'][0]
        self.assertEqual(saved['teacher_policy_drift'],batch['teacher_policy_drift'])
        self.assertEqual(saved['sample_usage']['used_samples_by_phase'],batch['used_samples_by_phase'])
        self.assertEqual(saved['phases']['rail_late']['insufficient_coverage'],1)


class CurrentTeacherResumeTests(unittest.TestCase):
    previous_implementation='3db61ed932779f2cd8d8e149509ab0c1b7232af63102213f17d5666c2d338130'
    previous_budget=TEACHER_TIME_BUDGET_V1
    previous_games=4
    previous_fraction=.5

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        from test_teacher_work_resume import TeacherWorkResumeTests
        from brass_ai.learning.contracts import compatibility
        from brass_ai.step5.league import implementation_id
        def signatures():
            cfg=read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json');cfg['model_size']='small'
            new=dict(config=cfg,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
            old=copy.deepcopy(new);old['implementation']=self.previous_implementation
            old['config'].update(teacher_work_budget=copy.deepcopy(self.previous_budget),
                search_games=self.previous_games,search_wall_fraction=self.previous_fraction)
            return old,new
        fixture=TeacherWorkResumeTests();fixture.signatures=signatures
        self.addCleanup(fixture.doCleanups)
        return fixture.fixture()

    test_weights_adam_rng_counters_and_search_debt_survive=resume_fixture.TimeResumeTests.test_weights_adam_rng_counters_and_search_debt_survive
    test_partial_iteration_and_old_teacher_targets_cannot_be_reused=resume_fixture.TimeResumeTests.test_partial_iteration_and_old_teacher_targets_cannot_be_reused

    def test_unrelated_learning_gate_opponent_or_budget_changes_are_rejected(self):
        from brass_ai.recovery.resume import check_signature
        _,state,new,_,_=self.fixture()
        self.assertEqual(new['config']['champion'],state['signature']['config']['champion'])
        self.assertEqual(new['config']['legacy_opponent_work_budget'],state['signature']['config']['legacy_opponent_work_budget'])
        for section,key in (('update','target_kl'),('reward','vp'),('champion','every_games'),
                            ('legacy_opponent_work_budget','seconds_per_turn')):
            changed=copy.deepcopy(new);changed['config'][section][key]*=2
            with self.assertRaises(ValueError):check_signature(state,changed,True)
        changed=copy.deepcopy(new);changed['config']['teacher_work_budget']['seconds_per_phase']['rail_late']=36.
        with self.assertRaises(ValueError):check_signature(state,changed,True)


class EarlyRailResumeTests(CurrentTeacherResumeTests):
    previous_implementation='48831bc83887a8d6beb563fd908dff18ee6b180dd8d7da6824724f4c53d9e596'
    previous_budget=TEACHER_PHASE_TIME_BUDGET_V1
    previous_games=5
    previous_fraction=None

    def test_historical_and_current_source_transitions_compose(self):
        from brass_ai.recovery.resume import check_signature
        _,state,new,_,_=self.fixture();prior=state['signature']
        old=copy.deepcopy(prior);old['implementation']=CurrentTeacherResumeTests.previous_implementation
        old['config'].update(teacher_work_budget=TEACHER_TIME_BUDGET_V1,search_games=4,search_wall_fraction=.5)
        self.assertTrue(check_signature(dict(state,signature=old),prior,True))
        self.assertTrue(check_signature(state,new,True))
        self.assertTrue(check_signature(dict(state,signature=old),new,True))
        with self.assertRaises(ValueError):
            check_signature(dict(state,signature=old),dict(prior,config=new['config']),True)
        changes={k for k in prior['config'] if prior['config'][k]!=new['config'][k]}
        self.assertEqual(changes,{'teacher_work_budget'})


class RevisedPhaseResumeTests(CurrentTeacherResumeTests):
    previous_implementation='201a2b00d5e5c1c8485cfa26fd2d3e062375d510a4bbae74257dba99e54b6e9d'
    previous_budget=TEACHER_PHASE_TIME_BUDGET_V2
    previous_games=5
    previous_fraction=None

    def test_historical_early_rail_receipt_cannot_approve_new_clocks(self):
        from brass_ai.recovery.resume import check_signature
        _,state,new,_,_=self.fixture();prior=state['signature']
        old=copy.deepcopy(prior);old['implementation']=EarlyRailResumeTests.previous_implementation
        old['config']['teacher_work_budget']=copy.deepcopy(TEACHER_PHASE_TIME_BUDGET_V1)
        self.assertTrue(check_signature(dict(state,signature=old),prior,True))
        self.assertTrue(check_signature(state,new,True))
        self.assertTrue(check_signature(dict(state,signature=old),new,True))
        with self.assertRaises(ValueError):
            check_signature(dict(state,signature=old),dict(prior,config=new['config']),True)
        changes={k for k in prior['config'] if prior['config'][k]!=new['config'][k]}
        self.assertEqual(changes,{'teacher_work_budget'})


if __name__=='__main__':unittest.main()
