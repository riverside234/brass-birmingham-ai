"""Seat coverage, teacher retirement, fixed-anchor alarms and diagnostic parity."""
from collections import Counter
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility
from brass_ai.learning.runner import match_report
from brass_ai.recovery.config import CURRICULUM, table_kind, table_rotation, teacher_search_config
from brass_ai.recovery import diagnostics, teacher_schedule as schedule
from brass_ai.recovery.league import schedule_evaluations
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import implementation_id, pottery_metrics, write_json
from fixtures import game, position
from test_teacher_schedule import Clock, Planner, Decider


class Selfplay31ReviewTests(unittest.TestCase):
    def test_each_curriculum_kind_covers_all_seats_and_resume_offsets(self):
        for start in (0, 20000, 100000):
            counts=Counter((table_kind(i,CURRICULUM),table_rotation(i)) for i in range(start,start+320))
            kinds={kind for kind,_ in counts}
            for kind in kinds:
                self.assertEqual(len({counts[kind,s] for s in range(4)}),1,(start,kind,counts))
            # Stopping and resuming an iteration never restarts a rotation.
            full=[table_rotation(i) for i in range(start,start+320)]
            pieces=[table_rotation(i) for i in range(start,start+117)]+[table_rotation(i) for i in range(start+117,start+320)]
            self.assertEqual(full,pieces)
        # Retain the original 5%-v4 seat-aliasing regression with its old recipe.
        archived=read(SOURCE_ROOT/'configs/step5/lab.json')['curriculum']
        old={i%4 for i in range(10000,10320) if table_kind(i,archived)=='legacy'}
        new={table_rotation(i) for i in range(10000,10320) if table_kind(i,archived)=='legacy'}
        self.assertEqual(len(old),1);self.assertEqual(new,set(range(4)))

    def test_retirement_reserve_stays_inside_one_clock_and_late_labels_fail(self):
        self.assertEqual(schedule.finalization_reserve(12),1.2)
        self.assertEqual(schedule.finalization_reserve(30),1.2)
        self.assertAlmostEqual(schedule.finalization_reserve(1),.1)
        for retirement,qualified in ((1.0,True),(1.3,False)):
            clock=Clock();clock.now=100.;planner=Planner(clock);planner.config=teacher_search_config({})
            rng=random.Random(19);planner.begin_teacher_game(rng,0,1200,3400)
            view,choices=position();view['context']['round']=planner.teacher_states[rng]['rounds']['canal_early']
            def finish(policy,*args):
                result=original(policy,*args);clock.now=args[3]+retirement;return result
            original=Decider.decide
            with patch.object(schedule.time,'monotonic',clock),patch.object(Decider,'decide',finish):
                planner.admit_teacher_turn(view['context'],rng)
                current=planner.teacher_states[rng]['current']
                self.assertEqual((current['started'],current['deadline']),(100.,112.))
                chosen=planner.select_batch([(view,choices,rng)])[0]
                self.assertEqual(chosen[2] is not None,qualified)
                counts=planner.teacher_report(rng)['phases']['canal_early']
                self.assertEqual(counts['qualified_samples'],int(qualified))
                self.assertEqual(counts['timeouts'],int(not qualified))

    def test_one_diagnostic_pass_per_composition_keeps_totals(self):
        games=[copy.deepcopy(game()) for _ in range(2)]
        for i,g in enumerate(games):g['game_id']=f'game-{i}';g['setup_group']=f'group-{i}'
        kinds={'group-0':'selfplay','group-1':'historical'}
        def seat_totals(count):
            return {str(seat):dict(seat_games=count,win_share_sum=count*win,
                placement_sum=count*rank,vp_sum=count*vp,vp_square_sum=count*vp**2)
                for seat,(win,rank,vp) in enumerate(((.5,1,100),(.5,1,100),(0.,3,80),(0.,4,50)))}
        expected=dict(matches=match_report(games),economy=diagnostics.economy(games),diversity=pottery_metrics(games),
            learner_by_seat=seat_totals(2),
            by_composition={k:dict(matches=match_report([g]),economy=diagnostics.economy([g]),
                                  diversity=pottery_metrics([g]),learner_by_seat=seat_totals(1))
                            for k,g in zip(kinds.values(),games)},
            measured_policy='frozen incoming collection policy, before this update')
        with patch.object(diagnostics,'economy',wraps=diagnostics.economy) as economy:
            actual=diagnostics.summarize(games,None,kinds)
        self.assertEqual(economy.call_count,2)
        self.assertEqual(actual,expected)

    def test_monitor_uses_fixed_initial_anchor_without_more_games(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp)
            anchor=dict(path='artifacts/fake/initial.pt',model_id='anchor')
            recent=dict(path='artifacts/fake/recent.pt',model_id='recent')
            candidate=dict(path='artifacts/fake/new.pt',model_id='candidate')
            state=dict(signature=dict(config=cfg),games_done=6400,history=[anchor,recent,candidate],
                       population=[anchor,recent,candidate],evaluation_anchor=anchor,evaluations=[])
            def evaluate(root,directory,selected,references,settings,**kwargs):
                self.assertEqual(selected,candidate['path']);self.assertEqual(references,[anchor['path']])
                self.assertEqual(settings['setups'],1);self.assertEqual(settings['lineups'],['recent-neural'])
                write_json(directory/'spec.json',dict(config=settings));write_json(directory/'report.json',{})
                return dict(failed_games=0,games=4)
            with patch('brass_ai.recovery.evaluation.evaluate',side_effect=evaluate) as measured:
                schedule_evaluations(SOURCE_ROOT,run,state,'cpu',None)
                schedule_evaluations(SOURCE_ROOT,run,state,'cpu',None)
            self.assertEqual(measured.call_count,1);self.assertEqual(state['evaluations'][0]['games'],4)

    def test_completed_32_generation_run_has_reviewed_combined_reward_continuation(self):
        state=read(SOURCE_ROOT/'results/selfplay-v2-31/state.json');old=state['signature']
        self.assertEqual((state['iteration'],state['games_done']),(32,10240))
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json');cfg['rehearsal_manifest']=old['imitation_source']['manifest']
        for key in ('learning_rate','update','curriculum','imitation'):
            self.assertEqual(cfg[key],old['config'][key])
        self.assertEqual(old['config']['search']['contract'],'masked-ismcts/1.1.0')
        self.assertEqual(cfg['search'],dict(old['config']['search'],
            contract='masked-ismcts/1.2.0',reward=reward_spec(cfg),utility_scale=1.2))
        self.assertEqual(compatibility(),old['compatibility'])
        signature=dict(old,config=cfg,implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,True))
        with self.assertRaisesRegex(ValueError,'unchanged'):
            check_signature(state,dict(signature,config=dict(cfg,learning_rate=2e-5)),True)


if __name__=='__main__':unittest.main()
