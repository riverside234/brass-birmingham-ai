"""Small real-host replay and branch-specific future receipt checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from brass_ai.bank import sha
from brass_ai.host import RulesHost,HostError
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery.host_runtime import local_scoring_host
from brass_ai.recovery import local_scoring_targets as targets


class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_complete_collector_replay_seal_and_old_replay_admission(self):
        from brass_ai.recovery.execution import run_game
        from brass_ai.recovery.telemetry import Timings
        from brass_ai.recovery.replay import write_shard,read_shards,check_game
        from brass_ai.recovery.updates import prepared
        class Pass:
            # Authored deterministic model-policy fixture; tests reader/labels,
            # not probability reconstruction or playing strength.
            kind='model';policy_id='local-scoring-native-fixture';temperature=1.
            def select_batch(self,batch):
                return [(next((j for j,c in enumerate(choices) if c['kind']=='pass'),0),1.,None) for _,choices,_ in batch]
        directory=local_scoring_host(SOURCE_ROOT)
        producer=dict(runtime_sha256=sha(directory/'runtime.json'),host_sha256=sha(directory/'BrassHost.exe'))
        provenance=dict(purpose='training',local_scoring=dict(targets.identity(),producer=producer))
        job=dict(seed=153,sampling_seed=99,provenance=provenance)
        with RulesHost(directory) as host:game=run_game(host,job,[Pass() for _ in range(4)],Timings(),1000)
        self.assertEqual([e['era'] for e in game[targets.FIELD]['receipt']['eras']],['CanalEra','RailEra'])
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'native.jsonl.gz';write_shard(path,[game],provenance,cache=True)
            loaded=read_shards([path])[0];data=prepared([loaded])
            self.assertTrue(all('_local_scoring' in r for r,*_ in data))
            self.assertEqual(len({id(r['_local_scoring']['envelope']) for r,*_ in data}),1)
            from brass_ai.recovery.ingress import _prepare_inner
            from brass_ai.recovery.replay import cache_receipt
            from brass_ai.recovery.updates import Dataset
            from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT
            from test_gae_value_phase import fixture
            record=dict(path=str(path),sha256=sha(path),features=cache_receipt(path),game_id=game['game_id'])
            # This is the real compact CPU-reader boundary, previously bypassed
            # by the direct prepared() smoke. No mocks of admission/compaction.
            _,compact,_,_=_prepare_inner(([record],Pass.policy_id,{game['setup_group']:'selfplay'},None,None))
            revisions=[r['view']['context']['revision'] for r,*_ in compact]
            self.assertEqual(revisions,[r['view']['context']['revision'] for r,*_ in data])
            self.assertEqual(len(set(revisions)),len(compact))
            self.assertEqual(len({id(r['_local_scoring']['envelope']) for r,*_ in compact}),1)
            model,*_=fixture(games=2);companion=Companion(model,copy.deepcopy(DEFAULT));ds=Dataset(compact,'cpu',policy=False)
            companion.inventory(ds,range(len(compact)));self.assertEqual(len(companion.available),len(compact))
            companion.begin('ppo');ix,x,_=ds.value_batch([0,1],model)
            _,loss=companion.forward(model,x,ds,ix);loss.mean().backward();companion.step('ppo')
            self.assertEqual(companion.usage()['rows'],2)
            tampered=copy.deepcopy(game);tampered[targets.FIELD]['receipt']['eras'][0]['slots'][0]['owner']=1
            with self.assertRaises(ValueError):check_game(tampered)
            legacy=copy.deepcopy(dict(game));legacy.pop(targets.FIELD);check_game(legacy)

    def test_planner_checks_frozen_checkpoint_before_local_recorder_import(self):
        from test_legacy_profile_cycle import LegacyProfiles
        from brass_ai.recovery.curriculum import training_table
        fixture=LegacyProfiles();fixture.setUp();self.addCleanup(fixture.doCleanups)
        cfg=copy.deepcopy(fixture.config);cfg['games_per_iteration']=1
        first=next(i for i in range(100) if training_table(i,cfg,None)[0]=='historical')
        state=dict(copy.deepcopy(fixture.state),games_done=first)
        from brass_ai.recovery.local_scoring_auxiliary import DEFAULT
        directory=local_scoring_host(SOURCE_ROOT)
        for active in (False,True):
            config=copy.deepcopy(cfg)
            if active:config['local_scoring_auxiliary']=copy.deepcopy(DEFAULT)
            planned=fixture.plan(state,config)
            job=planned['jobs'][0];self.assertEqual(job['provenance']['kind'],'historical')
            self.assertEqual('local_scoring' in job['provenance'],active)
            if active:self.assertEqual(job['provenance']['local_scoring']['producer']['host_sha256'],sha(directory/'BrassHost.exe'))

    def test_rail_fork_restoration_has_only_own_future_and_requires_completion(self):
        directory=local_scoring_host(SOURCE_ROOT)
        with RulesHost(directory) as h:
            c=h.reset(154);h.call('local_scoring_enable')
            with self.assertRaises(HostError):h.call('local_scoring_targets')
            for _ in range(500):
                if c['era']=='RailEra' and c['phase']=='action':break
                choices=h.choices();choice=next((x for x in choices if x['kind']=='pass'),choices[0]);c=h.apply(c,choice)
            self.assertEqual(c['era'],'RailEra');checkpoint=h.call('checkpoint');fork_revision=c['revision']
            def finish():
                c=h.call('context')
                for _ in range(500):
                    if c['terminal']:break
                    choices=h.choices();c=h.apply(c,next((x for x in choices if x['kind']=='pass'),choices[0]))
                self.assertTrue(c['terminal']);return h.call('local_scoring_targets')
            parent=finish()
            h.call('fork_restore',checkpoint=checkpoint,shuffle_seed=17);h.call('local_scoring_enable',fresh=True)
            # Save and restore the branch, including its private capture origin.
            branch_cp=h.call('checkpoint');h.call('restore',checkpoint=branch_cp)
            branch=finish()
            self.assertEqual([e['era'] for e in parent['eras']],['CanalEra','RailEra'])
            self.assertEqual([e['era'] for e in branch['eras']],['RailEra'])
            self.assertEqual(branch['start_revision'],fork_revision)


if __name__=='__main__':unittest.main()
