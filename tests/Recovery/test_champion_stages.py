"""Two-stage selection thresholds, frozen panels, restart and evidence contracts."""
import copy
from itertools import combinations
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.qualification import grouped_interval
from brass_ai.recovery.champion import CONTRACT,decision,initialize,schedule
from brass_ai.recovery.config import CHAMPION as DEFAULT_CHAMPION,CHAMPION_LEGACY,RETENTION,champion_config,validate_config
from brass_ai.recovery.history import compact
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.retention import plan
from brass_ai.step5.league import implementation_id,write_json
from test_combined_reward import scored


# Exercise the explicitly enabled historical recipe, not the new gate-free default.
CHAMPION=dict(DEFAULT_CHAMPION,enabled=True)


def outcomes(screen=.56,confirmation=.60,complete=True):
    return [dict(seed=s,candidate_seats=list(pair),win=screen if s<8 else confirmation,
        candidate_vp=100.,incumbent_vp=90.,candidate_rank=2.,incumbent_rank=3.)
        for s in range(72 if complete else 8) for pair in combinations(range(4),2)]


class Actor:
    """Exercise real job/result attribution with official four-player winners."""
    def __init__(self,screen_extra=3,confirmation_extra=19):
        self.calls=[];self.games={};self.screen_extra=screen_extra;self.confirmation_extra=confirmation_extra
        self.fail_confirmation=False;self.drop_confirmation=False

    def collect(self,jobs):
        self.calls.append(copy.deepcopy(jobs))
        if jobs[0]['index']==48 and self.fail_confirmation:raise RuntimeError('Interrupted confirmation')
        records=[]
        for job in jobs:
            setup,seat_pair=divmod(job['index'],6)
            extra=setup<self.screen_extra if setup<8 else setup-8<self.confirmation_extra
            pair=job['provenance']['candidate_seats'];other=[s for s in range(4) if s not in pair]
            winner=pair[0] if seat_pair<3+extra else other[0]
            scores=[(100 if s==winner else 90,0,0) for s in range(4)]
            game=dict(game_id=str(job['index']),split='validation',
                setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed'])),
                opponents=[p['policy_id'] for p in job['policies']],result=scored(scores))
            self.games[job['path']]=game
            records.append(dict(job=job['index'],game_id=game['game_id'],path=job['path']))
        if jobs[0]['index']==48 and self.drop_confirmation:records.pop()
        return list(reversed(records))

    def read(self,paths,**_):return [self.games[p] for p in paths]


class ChampionStageTests(unittest.TestCase):
    def fixture(self,**actor_options):
        temp=tempfile.TemporaryDirectory(prefix='champion-stage-',dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(temp.cleanup);run=Path(temp.name).resolve();entries=[]
        for gen in (0,1):
            path=run/f'generation-{gen:04d}.pt';path.write_bytes(str(gen).encode())
            entries.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=str(gen),generation=gen))
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        state=dict(format='step5-recovery-league/2.2.0',iteration=1,signature=dict(config=cfg),
                   games_done=0,history=entries[:1],evaluation_anchor=entries[0])
        self.enterContext(patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,existing=None:e))
        initialize(SOURCE_ROOT,run,state);state.update(games_done=6400,history=entries)
        write_json(run/'state.json',state)
        actor=Actor(**actor_options);self.enterContext(patch('brass_ai.recovery.champion.read_shards',side_effect=actor.read))
        return run,state,SimpleNamespace(models={'0':None,'1':None}),actor

    def test_explicit_protocol_and_preset_only_change_the_gate(self):
        old=read(SOURCE_ROOT/'configs/step5/joint7-g150-control.json')
        new=validate_config(read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json'))
        self.assertEqual(new,dict(old,champion=CHAMPION))
        self.assertEqual((CHAMPION['setups'],CHAMPION['confirmation_setups']), (8,64))
        self.assertEqual((CHAMPION['screen_min_win_share'],CHAMPION['min_win_share'],CHAMPION['min_ci_lower']),(.56,.55,.49))
        self.assertEqual(champion_config(CHAMPION_LEGACY),CHAMPION_LEGACY)
        for change in (dict(protocol='unknown'),dict(setups=16),dict(confirmation_setups=32),
                       dict(screen_min_win_share=float('nan')),dict(min_ci_lower=1.),dict(unexpected=True)):
            with self.subTest(change=change),self.assertRaises(ValueError):champion_config(dict(CHAMPION,**change))

    def test_screen_56_percent_is_inclusive_but_never_promotes(self):
        for share,passed in ((.55,False),(.559999,False),(.56,True),(1.,True)):
            result=decision(outcomes(screen=share,complete=False),CHAMPION)
            self.assertEqual(result['screen']['passed'],passed)
            self.assertEqual(result['stage'],'confirmation_required' if passed else 'screen_rejected')
            self.assertFalse(result['accepted']);self.assertEqual((result['setups'],result['games']),(8,48))

    def test_combined_55_is_inclusive_and_lower_49_is_strict(self):
        rows=outcomes()
        for index,row in enumerate(rows):row['win']=1. if index<237 else (.6 if index==237 else 0.)
        # 237.6 / 432 rounds to the exact same float as the .55 threshold.
        for lower,accepted in ((.49,False),(.490001,True),(.51,True)):
            with patch('brass_ai.recovery.champion.grouped_interval',return_value=[lower,.62]):
                result=decision(rows,CHAMPION)
            self.assertEqual(result['candidate_win_share'],.55);self.assertEqual(result['accepted'],accepted)
        with patch('brass_ai.recovery.champion.grouped_interval',return_value=[.52,.62]):
            below=copy.deepcopy(rows);below[237]['win']=.599
            self.assertFalse(decision(below,CHAMPION)['accepted'])
        with patch('brass_ai.recovery.champion.grouped_interval',return_value=None):
            self.assertFalse(decision(rows,CHAMPION)['accepted'])

    def test_all_72_setups_are_pooled_and_bootstrapped_as_clusters(self):
        rows=outcomes(screen=.75,confirmation=.54)
        result=decision(rows,CHAMPION)
        self.assertAlmostEqual(result['candidate_win_share'],(8*.75+64*.54)/72)
        self.assertEqual(result['setup_bootstrap_95'],grouped_interval(rows,'win',samples=4000))
        self.assertEqual((result['setups'],result['games'],result['stage']),(72,432,'combined'))
        self.assertEqual(result['confirmation']['games'],384)
        self.assertEqual(decision(list(reversed(rows)),CHAMPION),result)

    def test_partial_duplicate_and_unscreened_panels_are_rejected(self):
        rows=outcomes()
        bad_panels=[rows[:-1],rows[:54],rows+[rows[-1]],rows[:-1]+[rows[-2]],outcomes(screen=.55)]
        for panel in bad_panels:
            with self.subTest(size=len(panel)),self.assertRaises(ValueError):decision(panel,CHAMPION)
        rows[0]['win']=float('nan')
        with self.assertRaisesRegex(ValueError,'win share'):decision(rows,CHAMPION)

    def test_historical_result_schema_and_decision_remain_exact(self):
        path=SOURCE_ROOT/'results/v4-selfplay-joint-g0150-20261013T173011108713Z'
        reports=list(path.glob('evaluations/champion-*/report.json'))
        self.assertTrue(reports)
        for path in reports:
            report=read(path);spec=report.get('spec') or read(path.parent/'spec.json')
            self.assertEqual(decision(report['outcomes'],spec['config']),report['result'])

    def test_screen_rejection_collects_only_48_and_retains_incumbent(self):
        run,state,service,actor=self.fixture(screen_extra=0)
        result=schedule(SOURCE_ROOT,run,state,service,actor)
        self.assertEqual([len(j) for j in actor.calls],[48]);self.assertFalse(result['accepted'])
        self.assertEqual(result['stage'],'screen_rejected')
        self.assertEqual(state['champion']['generation'],0)
        self.assertEqual((run/'best.pt').read_bytes(),b'0')
        self.assertEqual(state['champion_gate']['next_games'],12800)
        self.assertFalse((run/'evaluations/champion-00006400/confirmation-report.json').exists())

    def test_confirmation_runs_full_384_new_games_with_fixed_pure_policies(self):
        run,state,service,actor=self.fixture();before=copy.deepcopy(state)
        result=schedule(SOURCE_ROOT,run,state,service,actor)
        self.assertEqual([len(j) for j in actor.calls],[48,384]);self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['candidate_win_share'],238/432)
        first,second=actor.calls
        self.assertFalse({j['seed'] for j in first}&{j['seed'] for j in second})
        self.assertEqual([j['index'] for j in first+second],list(range(432)))
        self.assertEqual(len({j['path'] for j in first+second}),432)
        for job in first+second:
            self.assertEqual(sorted(p['model_id'] for p in job['policies']),['0','0','1','1'])
            self.assertTrue(all(p['kind']=='neural' and p['exploration']==0 and p['focus']=='neutral' for p in job['policies']))
        report=read(SOURCE_ROOT/result['report'])
        self.assertEqual(len(report['spec']['seeds']),72)
        self.assertEqual(report['result'],decision(report['outcomes'],report['spec']['config']))
        self.assertEqual((run/'best.pt').read_bytes(),b'1')
        # Exercise downstream accepted-parent verification, without changing it.
        from brass_ai.personality.handoff import selection_evidence
        entry,gate=selection_evidence(state,read(run/'best.json'),report,report['spec'])
        self.assertEqual(entry,state['champion']);self.assertEqual(gate,result)
        # A crash after report publication must reuse its exact result.
        self.assertEqual(schedule(SOURCE_ROOT,run,before,service,actor),result)
        self.assertEqual(len(actor.calls),2)

    def test_failed_combined_gate_never_adds_more_setups(self):
        run,state,service,actor=self.fixture(confirmation_extra=0)
        result=schedule(SOURCE_ROOT,run,state,service,actor)
        self.assertFalse(result['accepted']);self.assertEqual(result['stage'],'combined')
        self.assertEqual([len(j) for j in actor.calls],[48,384]);self.assertEqual(result['setups'],72)
        self.assertEqual(state['champion']['generation'],0)

    def test_interrupted_confirmation_reuses_screen_and_same_new_seeds(self):
        run,state,service,actor=self.fixture();before=copy.deepcopy(state)
        actor.fail_confirmation=True
        with self.assertRaisesRegex(RuntimeError,'Interrupted'):schedule(SOURCE_ROOT,run,state,service,actor)
        self.assertEqual(state,before);self.assertEqual(read(run/'state.json'),before)
        directory=run/'evaluations/champion-00006400'
        screen=(directory/'screen-report.json').read_bytes()
        self.assertFalse((directory/'report.json').exists())
        actor.fail_confirmation=False
        result=schedule(SOURCE_ROOT,run,state,service,actor)
        self.assertTrue(result['accepted']);self.assertEqual([len(j) for j in actor.calls],[48,384,384])
        self.assertEqual(actor.calls[1],actor.calls[2]);self.assertEqual((directory/'screen-report.json').read_bytes(),screen)

    def test_missing_confirmation_receipt_cannot_publish_or_promote(self):
        run,state,service,actor=self.fixture();before=copy.deepcopy(state);actor.drop_confirmation=True
        with self.assertRaisesRegex(ValueError,'Incomplete champion receipts'):schedule(SOURCE_ROOT,run,state,service,actor)
        self.assertEqual(state,before)
        self.assertFalse((run/'evaluations/champion-00006400/report.json').exists())

    def test_cached_screen_is_bound_to_spec_and_setup_identities(self):
        run,state,service,actor=self.fixture();actor.fail_confirmation=True
        with self.assertRaises(RuntimeError):schedule(SOURCE_ROOT,run,state,service,actor)
        path=run/'evaluations/champion-00006400/screen-report.json';original=read(path)
        changed=copy.deepcopy(original);changed['spec_id']='different'
        write_json(path,changed)
        with self.assertRaisesRegex(ValueError,'specification changed'):schedule(SOURCE_ROOT,run,state,service,actor)
        changed=copy.deepcopy(original)
        for row in changed['outcomes']:row['seed']+=100000
        write_json(path,changed)
        with self.assertRaisesRegex(ValueError,'setup identities changed'):schedule(SOURCE_ROOT,run,state,service,actor)
        self.assertEqual(len(actor.calls),2)

    def test_compaction_preserves_stage_counts_thresholds_and_pooled_interval(self):
        rows=outcomes();result=decision(rows,CHAMPION)
        report=dict(format=CONTRACT,spec=dict(config=CHAMPION),result=result,outcomes=rows)
        kept=compact(report)
        for gate in (kept['champion_gate'],kept['evaluation_evidence']['champion']):
            for key in ('stage','games','setups','screen','confirmation','threshold','min_ci_lower','setup_bootstrap_95'):
                self.assertEqual(gate[key],result[key])
        self.assertEqual(kept['evaluation_evidence']['champion']['outcomes'],rows)

    def test_retention_keeps_unfinished_stage_and_retires_old_completed_stages(self):
        temp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(temp.cleanup)
        root=Path(temp.name).resolve();run=root/'artifacts/league'
        write_json(run/'state.json',dict(format='step5-recovery-league/2.2.0',iteration=4,history=[],
            signature=dict(config=dict(retention=RETENTION))))
        for index in range(4):
            directory=run/f'evaluations/champion-{index:08d}'
            write_json(directory/'spec.json',dict(config=CHAMPION))
            write_json(directory/'screen-report.json',dict(outcomes=[]))
            if index<3:
                write_json(directory/'report.json',dict(format=CONTRACT))
                write_json(directory/'confirmation-report.json',dict(outcomes=[]))
        retired={e['path'] for e in plan(root,run)['entries']}
        self.assertIn('evaluations/champion-00000000/screen-report.json',retired)
        self.assertIn('evaluations/champion-00000000/confirmation-report.json',retired)
        self.assertFalse(any('champion-00000003' in path for path in retired))

    def test_generation150_resume_requires_review_and_exact_new_configuration(self):
        state=read(SOURCE_ROOT/'results/v4-selfplay-joint-g0150-20261013T173011108713Z/state.json')
        cfg=read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json')
        signature=dict(state['signature'],config=cfg,implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,reviewed=True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        for key,value in (('seed',123),('min_win_share',.54),('screen_min_win_share',.55),('min_ci_lower',.48)):
            altered=copy.deepcopy(signature);altered['config']['champion'][key]=value
            with self.assertRaisesRegex(ValueError,'unchanged config'):check_signature(state,altered,reviewed=True)

    def test_generation150_new_gate_preserves_checkpoint_adam_rng_and_gate_counters(self):
        import torch
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.recovery.resume import load_resume,commit_transition,apply_optimizer_config
        torch.set_num_threads(2)
        state=read(SOURCE_ROOT/'results/v4-selfplay-joint-g0150-20261013T173011108713Z/state.json')
        cfg=read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json')
        state['signature']['config']['model_size']=cfg['model_size']='small'
        model=UtilityNet(utility_config(ModelConfig.preset('small'),cfg['search']['reward']))
        model.utility_calibration=dict(ready=True,reward_id=digest(cfg['search']['reward']))
        original=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        sum(p.square().mean() for p in model.parameters()).backward();original.step()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp).resolve();path=run/'generation-0150.pt'
            identity=save_checkpoint(path,model,original,step=150,training=dict(
                fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(state['signature']),mode='step5-population-ppo'))
            entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity,generation=150)
            state.update(history=[entry],population=[entry],resume=[entry],evaluation_anchor=entry,champion=entry)
            write_json(run/'state.json',state);before=path.read_bytes()
            signature=dict(state['signature'],config=cfg,implementation=implementation_id())
            restored,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=True)
            self.assertTrue(changed)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            optimizer=torch.optim.AdamW(restored.parameters(),lr=cfg['learning_rate'])
            resume_optimizer(raw,optimizer);apply_optimizer_config(optimizer,cfg)
            self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
            for key,values in original.state_dict()['state'].items():
                for name,value in values.items():torch.testing.assert_close(value,optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
            updated=commit_transition(run,state,signature)
            for key,value in state.items():
                if key not in ('signature','source_transitions'):self.assertEqual(updated[key],value,key)
            self.assertEqual(updated['source_transitions'][-1]['config_changes']['champion'],
                             dict(before=CHAMPION_LEGACY,after=CHAMPION))
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(load_resume(SOURCE_ROOT,run,updated,signature)[2])


if __name__=='__main__':unittest.main()
