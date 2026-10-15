"""Focused reward, incumbent selection, on-policy attribution and history checks."""
import copy
from itertools import combinations
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.reward import specification,terminal_utility,replay_returns
from brass_ai.recovery.champion import decision,jobs_for,gate_rows,selfplay_policies,initialize,schedule
from brass_ai.recovery.config import CHAMPION as CHAMPION_STAGED,CHAMPION_LEGACY as CHAMPION,RETENTION,INSOLVENCY,validate_config
from brass_ai.recovery.history import compact_history
from brass_ai.recovery.retention import plan,prune
from brass_ai.recovery.execution import neural_spec
from brass_ai.step5.league import write_json,seeds_for
from test_combined_reward import scored
from test_insolvency_recovery import row


def rows(win=.75):
    return [dict(seed=s,candidate_seats=list(p),win=win,candidate_vp=100.,incumbent_vp=90.,
                 candidate_rank=2.,incumbent_rank=3.) for s in range(8) for p in combinations(range(4),2)]


class ChampionTests(unittest.TestCase):
    def test_reward_defaults_linear_vp_and_causal_new_cap(self):
        spec=specification();self.assertEqual(spec['insolvency'],INSOLVENCY)
        self.assertEqual((spec['win'],spec['placement'],spec['vp']),(1.,.1,.001))
        for vp in (50,200,100000):
            a=scored([(vp,0,0)]*4);b=scored([(vp+10,0,0)]*4)
            self.assertAlmostEqual(terminal_utility(b,spec)[0]-terminal_utility(a,spec)[0],.01)
        episode=[r for n in range(1,18) for r in (row(0,round=n),row(0,round=n,phase='debt',kind='liquidate',debts={0:5}))]
        labels,summary=replay_returns(episode,scored([(100,0,0)]*4),spec)
        self.assertAlmostEqual(summary['cost'][0],.075)
        self.assertAlmostEqual(labels[1]['utility'][0]-labels[0]['utility'][0],.005)
        self.assertEqual(labels[-1]['remaining'][0],0)

    def test_presets_match_reward_and_preserve_other_settings(self):
        for name in ('lab','lab-16x2','lab-24x4','lab-small','lab-reference','smoke','recovery-smoke'):
            cfg=validate_config(read(SOURCE_ROOT/f'configs/step5/{name}.json'))
            self.assertEqual(cfg['search']['reward'],specification(cfg['reward'],cfg['insolvency']))
            self.assertEqual(cfg['champion'],dict(CHAMPION_STAGED,enabled=True));self.assertEqual(cfg['search_wall_fraction'],None)
            if name.startswith('lab'):
                self.assertEqual((cfg['search']['seconds'],cfg['execution']['search_workers_per_turn']),(12.,4))
                self.assertEqual((cfg['learning_rate'],cfg['search_every_games'],cfg['search_games'],cfg['distill_steps']),(1e-5,160,5,8))

    def test_gate_balance_ties_and_uncertainty(self):
        self.assertTrue(decision(rows(),CHAMPION)['accepted'])
        self.assertFalse(decision(rows(.5),CHAMPION)['accepted'])
        self.assertFalse(decision(rows(.54),CHAMPION)['accepted'])
        uncertain=rows()
        for r in uncertain:r['win']=float(r['seed']<5)
        self.assertGreater(decision(uncertain,CHAMPION)['candidate_win_share'],.55)
        self.assertFalse(decision(uncertain,CHAMPION)['accepted'])
        with self.assertRaisesRegex(ValueError,'six seat pairs'):decision(rows()[:-1],CHAMPION)

    def test_scheduler_and_results_use_seats_not_completion_order(self):
        seeds=seeds_for(1916400,8,'validation')[0]
        jobs=jobs_for(seeds,dict(model_id='a'),dict(model_id='b'),Path('unused'),'spec')
        again=jobs_for(seeds,dict(model_id='a'),dict(model_id='b'),Path('unused'),'spec')
        self.assertEqual(jobs,again);self.assertEqual(len(jobs),48)
        games=[];records=[]
        for j in jobs:
            # All four players tie officially: candidate family gets .5, not 1.
            games.append(dict(game_id=str(j['index']),split='validation',setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=j['seed'])),
                opponents=[p['policy_id'] for p in j['policies']],result=scored([(100,0,0)]*4)))
            records.append(dict(job=j['index'],game_id=str(j['index'])))
        result=gate_rows(list(reversed(games)),records[::2]+records[1::2],jobs)
        self.assertEqual([r['win'] for r in result],[.5]*48)
        self.assertFalse(decision(result,CHAMPION)['accepted'])
        with self.assertRaisesRegex(ValueError,'Incomplete'):gate_rows(games,records[:-1],jobs)

    def test_only_latest_learner_actions_enter_ppo(self):
        learner=neural_spec('latest',.05,'mixed');lineup=selfplay_policies(learner,dict(model_id='best'))
        self.assertEqual(lineup.count(learner),1)
        self.assertEqual([p['model_id'] for p in lineup],['latest','best','best','best'])
        self.assertTrue(all(p['policy_id']!=learner['policy_id'] for p in lineup[1:]))

    def test_promotion_rejection_and_completed_gate_resume(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);entries=[]
            for gen in (0,1):
                path=run/f'generation-{gen:04d}.pt';path.write_bytes(str(gen).encode())
                entries.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=str(gen),generation=gen))
            cfg=read(SOURCE_ROOT/'configs/step5/lab.json');cfg['champion']=copy.deepcopy(CHAMPION)
            state=dict(signature=dict(config=cfg),games_done=0,history=entries[:1],evaluation_anchor=entries[0])
            with patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,existing=None:e):
                initialize(SOURCE_ROOT,run,state)
                self.assertEqual((run/'best.pt').read_bytes(),b'0')
                state.update(games_done=6400,history=entries)
                service=SimpleNamespace(models={'0':None,'1':None});actor=SimpleNamespace(collect=lambda jobs:[])
                with patch('brass_ai.recovery.champion.read_shards',return_value=[]),patch('brass_ai.recovery.champion.gate_rows',return_value=rows()):
                    won=schedule(SOURCE_ROOT,run,state,service,actor)
                self.assertTrue(won['accepted']);self.assertEqual((run/'best.pt').read_bytes(),b'1')
                self.assertEqual(state['champion']['generation'],1);self.assertEqual(state['champion_gate']['next_games'],12800)
                self.assertIsNone(schedule(SOURCE_ROOT,run,state,service,actor))
                # Crash after report publication, before state promotion: re-read
                # the same completed gate; never collect another favorable batch.
                state['champion']=entries[0];state['champion_gate']['next_games']=6400
                actor.collect=lambda jobs:self.fail('Completed gate was collected again')
                self.assertTrue(schedule(SOURCE_ROOT,run,state,service,actor)['accepted'])
                # A later rejected candidate leaves the immutable best intact.
                state['history']=list(reversed(entries));state['games_done']=12800
                rollout_before=copy.deepcopy(state['champion_mix'])
                actor.collect=lambda jobs:[]
                with patch('brass_ai.recovery.champion.read_shards',return_value=[]),patch('brass_ai.recovery.champion.gate_rows',return_value=rows(.5)):
                    self.assertFalse(schedule(SOURCE_ROOT,run,state,service,actor)['accepted'])
                self.assertEqual((run/'best.pt').read_bytes(),b'1')
                self.assertEqual(state['champion_mix'],rollout_before)

    def test_compact_history_migrates_old_files_is_bounded_and_idempotent(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);archive={}
            for i in range(300):
                archive[f'iteration-{i:04d}/report.json']=dict(sha256=str(i),summary=dict(iteration=i,games_done=320*(i+1),
                    actor=dict(epochs=[dict(value=.2,post_update_kl=.001,utility_prediction=dict(mse=.3,mae=.4))]),
                    search=dict(collections=[dict(huge='x'*1000)]*10),inference=dict(models=['x'*1000]*10)))
            write_json(run/'retention-history.json',archive);before=(run/'retention-history.json').stat().st_size
            compact_history(run,limit=128);data=read(run/'retention-history.json')
            self.assertEqual(len(data),128);self.assertIn('iteration-0299/report.json',data)
            self.assertNotIn('iteration-0000/report.json',data)
            self.assertLess((run/'retention-history.json').stat().st_size,before/10)
            saved=(run/'retention-history.json').read_bytes();compact_history(run,limit=128)
            self.assertEqual(saved,(run/'retention-history.json').read_bytes())
            # Cleanup with no expiring artifacts still compacts a legacy archive.
            write_json(run/'retention-history.json',archive)
            prune(SOURCE_ROOT,run,dict(RETENTION,history_reports=3),apply=True)
            self.assertEqual(len(read(run/'retention-history.json')),3)

    def test_retention_protects_champion_outside_population(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);best=run/'population/weights-best.pt';best.parent.mkdir();best.write_bytes(b'best')
            old=run/'population/weights-old.pt';old.write_bytes(b'old')
            entry=dict(path=best.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(best),model_id='best',generation=1)
            write_json(run/'state.json',dict(format='step5-recovery-league/2.2.0',iteration=200,history=[],population=[],
                champion=entry,signature=dict(config=dict(retention=RETENTION))))
            planned=plan(SOURCE_ROOT,run)
            self.assertNotIn('population/weights-best.pt',[e['path'] for e in planned['entries']])
            self.assertIn('population/weights-old.pt',[e['path'] for e in planned['entries']])


if __name__=='__main__':unittest.main()
