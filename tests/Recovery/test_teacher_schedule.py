"""Training-only phase reservations, shared clocks, evidence and IL handoff."""
import contextlib
import copy
import io
import random
import tempfile
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fixtures import game,position,result
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.search.common import SearchConfig,SearchIncomplete
from brass_ai.search.ismcts import information_key,visit_teacher
from brass_ai.recovery import teacher_schedule as schedule
from brass_ai.recovery.config import (SEARCH_EVERY_GAMES,SEARCH_WALL_FRACTION,SEARCH_SECONDS,
    SEARCH_WORKERS,UPDATE_V21,validate_config,teacher_search_config)
from brass_ai.recovery.execution import _run_turns
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.telemetry import Timings


class Clock:
    now=100.
    def __call__(self):return self.now


def evidence(view,choices,recall,coverage=True):
    visits=[3,2,0] if coverage else [3,0,0]
    trace=[[information_key(view,choices,recall),choices[0]['id']]];key=digest(trace)
    return dict(completed=sum(visits),interrupted=0,opponent_nodes=0,opponent_choices=0,max_pending=1,
        expansion=[True]*len(choices),root={c['id']:dict(visits=n,returns=[n*.25]*4,available=n)
            for c,n in zip(choices,visits) if n},
        plans={key:dict(signature=key,first=choices[0]['id'],trace=trace,visits=3,returns=[.75]*4)})


class Decider:
    def decide(self,view,choices,rng,deadline,recall,tree,cache):
        self.calls.append(deadline)
        if self.mode=='raise':raise SearchIncomplete('no time remaining')
        self.last_statistics=evidence(view,choices,recall,self.mode!='coverage')
        self.last_continuation={}
        teacher=visit_teacher(choices,self.last_statistics,[.6,.3,.1],model='a'*64,observer=view['seat'],config=self.config)
        if self.mode=='timeout':self.clock.now=deadline+1
        return (1 if self.mode=='fallback' else 0),1.,teacher


class Planner(schedule.TeacherTurns,Decider):
    kind='external';policy_id='unit-teacher'
    def __init__(self,clock):
        self.clock=clock;self.calls=[];self.mode='success'
        self.teacher_states=weakref.WeakKeyDictionary();self.cancelled=Event()
        self.config=SearchConfig(seconds=4.,workers=SEARCH_WORKERS)
        self.base=SimpleNamespace(request_deadline=None,select_batch=lambda inputs:[(0,1.,None)]*len(inputs))


class TeacherScheduleTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.clock_patch=patch.object(schedule.time,'monotonic',self.clock);self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.source_patch=patch('brass_ai.step5.league.implementation_id',return_value='b'*64);self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.planner=Planner(self.clock);self.rng=random.Random(19)
        self.planner.begin_teacher_game(self.rng,0,1200,3400)

    def view(self,phase):
        view,choices=position();view['context'].update(era=next(e for p,e,_,_ in schedule.PHASES if p==phase),
            round=self.planner.teacher_states[self.rng]['rounds'][phase],actions_remaining=2)
        return view,choices

    def select(self,phase='canal_early'):
        view,choices=self.view(phase);self.planner.admit_teacher_turn(view['context'],self.rng)
        return view,choices,self.planner.select_batch([(view,choices,self.rng)])[0]

    def test_seeded_seat_specific_reservations_ignore_policy_rng_and_order(self):
        expected={s:schedule.reserved_rounds(1200,3400,s) for s in range(4)}
        for s in reversed(range(4)):
            for _ in range(100):self.rng.random()
            self.assertEqual(expected[s],schedule.reserved_rounds(1200,3400,s))
            for phase,_,first,last in schedule.PHASES:self.assertTrue(first<=expected[s][phase]<=last)
        self.assertGreater(len({tuple(v.values()) for v in expected.values()}),1)
        self.assertNotEqual(expected,{s:schedule.reserved_rounds(1200,3401,s) for s in range(4)})

    def test_early_failures_do_not_spend_late_rail_and_never_retry(self):
        for phase,_,_,_ in schedule.PHASES:
            self.clock.now+=10;self.planner.mode='coverage' if phase!='rail_late' else 'success'
            view,choices,selected=self.select(phase)
            if phase!='rail_late':
                self.assertIsNone(selected[2]);before=len(self.planner.calls)
                self.planner.select_batch([(view,choices,self.rng)])
                self.assertEqual(len(self.planner.calls),before)
            else:self.assertIsNotNone(selected[2])
            self.planner.finish_teacher_turn(self.rng)
            # Even another turn number in the selected round cannot respend it.
            view['context']['turn']+=1;self.planner.admit_teacher_turn(view['context'],self.rng)
            self.assertIsNone(self.planner.teacher_states[self.rng]['current'])
        report=self.planner.teacher_report(self.rng)
        self.assertEqual(len(self.planner.calls),4)
        self.assertEqual(report['phases']['rail_late']['qualified_samples'],1)
        self.assertEqual(report['phases']['rail_late']['successful_turns'],1)
        for phase in ('canal_early','canal_late','rail_early'):
            self.assertEqual(report['phases'][phase]['failures'],1)
            self.assertEqual(report['phases'][phase]['successful_turns'],0)

    def test_forced_choices_are_not_successful_teaching(self):
        view,choices=self.view('rail_late');self.planner.admit_teacher_turn(view['context'],self.rng)
        self.assertEqual(self.planner.select_batch([(view,choices[:1],self.rng)])[0],(0,1.,None))
        self.assertEqual(self.planner.calls,[])
        report=self.planner.teacher_report(self.rng)['phases']['rail_late']
        self.assertEqual(report['forced_choices'],1);self.assertEqual(report['qualified_samples'],0)
        self.assertEqual(report['failed_turns'],1)

    def test_timeouts_fallbacks_and_inadequate_coverage_never_label(self):
        for mode,reason in [('timeout','timeouts'),('fallback','fallbacks'),('coverage','insufficient_coverage'),('raise','timeouts')]:
            with self.subTest(mode=mode):
                self.planner.begin_teacher_game(self.rng,0,1200,3400);self.planner.mode=mode
                _,_,selected=self.select();report=self.planner.teacher_report(self.rng)['phases']['canal_early']
                self.assertIsNone(selected[2]);self.assertEqual(report['qualified_samples'],0)
                self.assertEqual(report[reason],1);self.assertEqual(report['failures'],1)
                self.assertEqual(report['actual_searches'],1);self.assertEqual(report['successful_turns'],0)

    def test_both_actions_and_subchoices_share_deadline_after_cpu_admission(self):
        view,choices=self.view('canal_early');events=[];contexts=[]
        for phase,remaining in [('action',2),('payment',1),('action',1),('continuation',0)]:
            v=copy.deepcopy(view);v['context'].update(phase=phase,actions_remaining=remaining,revision=len(contexts))
            contexts.append(v)
        planner=self.planner;clock=self.clock
        class Lease:
            first=True
            def update(self,key,workers):
                self.test_workers=workers
                if self.first:clock.now+=30;self.first=False;events.append('admitted')
        class Host:
            index=0
            def observe(self,seat):events.append('observe');return contexts[self.index]
            def choices(self):return choices
            def apply(self,context,choice):
                clock.now+=.5;self.index+=1
                return contexts[self.index]['context'] if self.index<len(contexts) else dict(context,terminal=True)
            def call(self,name):return result()
        episode=game();episode['records']=[]
        _run_turns(Host(),{},[planner]*4,Timings(),5,contexts[0]['context'],episode,
            [self.rng]*4,[],PreparedEncoder(),Lease())
        self.assertEqual(events[0],'admitted')
        self.assertEqual(len(planner.calls),4)
        self.assertEqual(set(planner.calls),{133.6}) # 130 after admission + 4 s, 400 ms finalization reserve
        report=planner.teacher_report(self.rng)['phases']['canal_early']
        self.assertEqual(report['scheduled_turns'],1);self.assertEqual(report['actual_searches'],4)
        self.assertEqual(report['turn_elapsed_seconds'],2.)

    def test_four_private_workers_receive_identical_remaining_deadline(self):
        view,choices=self.view('canal_early');deadlines=[]
        class Replica:
            def __init__(self):self.base=SimpleNamespace(contexts={})
            def decide(self,view,choices,rng,deadline,recall,tree,cache):
                deadlines.append(deadline);self.last_statistics=evidence(view,choices,recall)
        planner=object.__new__(schedule.ScheduledParallelMasked)
        planner.teacher_states=weakref.WeakKeyDictionary();planner.cancelled=Event()
        planner.config=teacher_search_config({});planner.decision_lock=RLock()
        planner.replicas=[Replica() for _ in range(4)]
        planner.base=SimpleNamespace(request_deadline=None,evaluator=SimpleNamespace(model_id='a'*64),
            predict=lambda inputs:[(np.array([.6,.3,.1]),{})])
        planner.incumbent=lambda *args:{};planner.context_features=lambda *args:([0.]*4,[0.]*8)
        planner.telemetry=dict(searches=0,simulations=0,incomplete=0,max_pending=0)
        planner.begin_teacher_game(self.rng,0,1200,3400);planner.admit_teacher_turn(view['context'],self.rng)
        with ThreadPoolExecutor(max_workers=4) as pool:
            planner.pool=pool
            self.assertIsNotNone(planner.select_batch([(view,choices,self.rng)])[0][2])
        self.assertEqual(deadlines,[110.8]*4) # 12 seconds shared, including the 1.2 s reserve
        self.assertEqual(planner.teacher_report(self.rng)['phases']['canal_early']['qualified_samples'],1)

    def test_exhausted_budget_does_not_restart_and_debt_is_not_reserved(self):
        view,choices=self.view('canal_early');context=view['context']
        self.assertEqual(self.planner.teacher_workers(context,self.rng),4)
        context['phase']='debt';self.planner.admit_teacher_turn(context,self.rng)
        self.assertIsNone(self.planner.teacher_states[self.rng]['current'])
        self.assertEqual(self.planner.teacher_workers(context,self.rng),1)
        context['phase']='action';self.planner.admit_teacher_turn(context,self.rng)
        self.clock.now+=4.1
        for _ in range(3):self.assertIsNone(self.planner.select_batch([(view,choices,self.rng)])[0][2])
        report=self.planner.teacher_report(self.rng)['phases']['canal_early']
        self.assertEqual(self.planner.calls,[]);self.assertEqual(report['timeouts'],1)
        self.assertEqual(report['actual_searches'],0)

    def test_receipts_match_qualified_replay_and_reject_unscheduled_labels(self):
        view,choices,(_,_,teacher)=self.select('rail_late')
        reports=[self.planner.teacher_report(self.rng)]
        for seat in range(1,4):
            rng=random.Random(seat);self.planner.begin_teacher_game(rng,seat,1200,3400)
            reports.append(self.planner.teacher_report(rng))
        episode=game();episode['records']=[dict(seat=0,view=view,choices=choices,teacher=teacher)]
        records=[dict(game_id=episode['game_id'],teacher_phases=reports)]
        report=schedule.collection_report(records,[episode])
        self.assertEqual(report['phases']['rail_late']['scheduled_turns'],4)
        self.assertEqual(report['phases']['rail_late']['qualified_samples'],1)
        episode['records'][0]['view']['context']['round']=6
        with self.assertRaisesRegex(ValueError,'outside reserved'):schedule.collection_report(records,[episode])

    def test_active_presets_and_160_game_expanded_budget_default_lower_overrides(self):
        self.assertEqual((SEARCH_EVERY_GAMES,SEARCH_WALL_FRACTION,SEARCH_SECONDS,SEARCH_WORKERS),(160,7/9,12.,4))
        for path in (SOURCE_ROOT/'configs/step5').glob('lab*.json'):
            cfg=read(path);validate_config(cfg)
            self.assertEqual((cfg['search_ready_games'],cfg['search_every_games'],cfg['search_wall_fraction']),(640,160,7/9))
            self.assertEqual((cfg['search_games'],cfg['distill_steps'],cfg['search']['seconds'],cfg['execution']['search_workers_per_turn']),(4,8,12.,4))
            self.assertEqual((cfg['search']['pending_paths'],cfg['search']['batch_size']),(2,2))
            self.assertEqual(cfg['legacy_budget']['seconds'],1.)
            self.assertEqual(schedule.estimate_collection(cfg,5.),788.) # 4 games * (16 turns * 12 s + 5 s)
        self.assertAlmostEqual(schedule.search_allowance(100,0),350.)
        self.assertAlmostEqual(schedule.search_allowance(100,30),320.)
        self.assertEqual(schedule.search_allowance(100,360),0.)
        self.assertAlmostEqual(schedule.search_allowance(100,0,.25),100/3)
        self.assertAlmostEqual(schedule.search_allowance(100,10,.20),15.)
        self.assertEqual(schedule.search_allowance(100,30,.20),0.)
        # Generic evaluation scheduling and clocks are unchanged.
        self.assertEqual((SearchConfig().seconds,SearchConfig().turn_interval),(1.,4))
        self.assertEqual((teacher_search_config({}).seconds,teacher_search_config({}).workers),(12.,4))
        self.assertEqual((teacher_search_config({}).pending_paths,teacher_search_config({}).batch_size),(2,2))

    def test_search_wall_fraction_limit_and_admission_at_expanded_budget(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        for fraction in (.20,.25,.50,.70,7/9):
            validate_config(dict(cfg,search_wall_fraction=fraction))
        for fraction in (0.,-.1,.8,1.,1.5,float('nan')):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError,'search fraction'):
                    validate_config(dict(cfg,search_wall_fraction=fraction))
                with self.assertRaisesRegex(ValueError,'search wall allowance'):
                    schedule.search_allowance(100,0,fraction)
        estimate=schedule.estimate_collection(cfg,5.)
        self.assertLess(estimate,schedule.search_allowance(400,0))
        self.assertGreater(estimate,schedule.search_allowance(400,0,.50))
        self.assertGreater(estimate,schedule.search_allowance(400,700))
        for name in ('smoke.json','recovery-smoke.json'):
            self.assertEqual(validate_config(read(SOURCE_ROOT/'configs/step5'/name))['search_wall_fraction'],7/9)


class HandoffCompatibilityTests(unittest.TestCase):
    def test_distillation_keeps_eight_step_ceiling_and_ignores_unlabelled_choices(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.updates import distill_update
        from test_selfplay_review import teacher_games
        torch.set_num_threads(2);torch.manual_seed(913)
        games=teacher_games()
        for episode in games:
            row=copy.deepcopy(episode['records'][-1]);row['teacher']=None;row['step']+=1
            row['view']['context']['revision']+=1;episode['records'].append(row)
        model=FourSeatNet(ModelConfig.preset('small'))
        report=distill_update(model,torch.optim.AdamW(model.parameters(),lr=1e-5),games,
            dict(UPDATE_V21,effective_batch_size=2048,microbatch_size=8),steps=8,weight=.25)
        self.assertTrue(report['updated']);self.assertEqual(report['teacher_rows'],64)
        self.assertEqual(report['optimizer_steps'],8)
        self.assertEqual(report['max_observed_reuse'],2)

    def test_completed_pre_revision_il_is_accepted_but_unknown_source_is_not(self):
        from test_imitation import ImitationTests
        from brass_ai.recovery import demonstrations as demos
        from brass_ai.recovery.updates import prepared
        torch.set_num_threads(2)
        entry=read(SOURCE_ROOT/'configs/learning/imitation-handoff-sources.json')['compatible_sources'][0]
        self.assertEqual(entry['learning_compatibility'],digest(compatibility()))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            directory=Path(temp);manifest=ImitationTests().fixture_manifest(directory)
            validation=game();validation['game_id']='validation'
            rows={'train':prepared([game()],neutral=True),'validation':prepared([validation],neutral=True)}
            with patch.object(demos,'load_dataset',side_effect=lambda root,path,split,**kwargs:(rows[split],{})),contextlib.redirect_stdout(io.StringIO()):
                for name,source in [('approved',entry['implementation']),('unknown','f'*64)]:
                    with patch.object(demos,'implementation_id',return_value=source):
                        demos.train_bc(SOURCE_ROOT,directory/name,manifest,None,
                            dict(UPDATE_V21,effective_batch_size=4,microbatch_size=2),epochs=1,min_epochs=1,patience=1)
            model,origin=demos.completed_imitation(SOURCE_ROOT,directory/'approved')
            self.assertEqual(model.parameter_count,1840109)
            self.assertEqual(origin['manifest_sha256'],demos.sha(manifest))
            with self.assertRaisesRegex(ValueError,'reviewed compatible source'):
                demos.completed_imitation(SOURCE_ROOT,directory/'unknown')


if __name__=='__main__':unittest.main()
