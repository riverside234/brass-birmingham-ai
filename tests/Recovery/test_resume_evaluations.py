"""Reviewed-source continuation of committed learners and pending screens."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.resume import evaluation_directory,load_resume,commit_transition
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import write_json


OLD='826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16'
MID='1'*64
CURRENT='2'*64


class EvaluationResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def temp(self):
        item=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(item.cleanup)
        return Path(item.name).resolve()

    def signatures(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json');cfg['model_size']='small'
        old=dict(config=cfg,compatibility=compatibility(),implementation=OLD,legacy_pool={},teachers={})
        return [dict(old,implementation=x) for x in (OLD,MID,CURRENT)]

    def approvals(self,signatures):
        receipts=dict(format='selfplay-resume-sources/1.0.0',transitions=[
            dict(from_implementation=a['implementation'],to_implementation=b['implementation'],
                 learning_compatibility=digest(b['compatibility'])) for a,b in zip(signatures,signatures[1:])])
        return patch('brass_ai.recovery.resume.read',side_effect=lambda p:
            receipts if Path(p).name=='selfplay-resume-sources.json' else read(p))

    def state(self,signatures):
        return dict(format='step5-recovery-league/2.2.0',iteration=1,signature=signatures[-1],
            source_transitions=[dict(from_signature=a,to_signature=digest(b)) for a,b in zip(signatures,signatures[1:])])

    def test_pending_and_completed_old_screens_use_fresh_deterministic_namespace(self):
        run=self.temp();sigs=self.signatures();state=self.state(sigs)
        for name in ('monitor-0001','champion-00006400'):
            old=run/'evaluations'/name;write_json(old/'spec.json',dict(implementation=OLD))
            (old/'cached-game.jsonl.gz').write_bytes(b'old immutable evidence')
            write_json(old/'report.json',dict(completed=True))
            before={p.name:p.read_bytes() for p in old.iterdir()}
            with self.approvals(sigs),patch('brass_ai.recovery.resume.implementation_id',return_value=CURRENT):
                selected=evaluation_directory(SOURCE_ROOT,run,state,name)
                self.assertEqual(selected.name,name+'-source-'+CURRENT[:16])
                write_json(selected/'spec.json',dict(implementation=CURRENT))
                self.assertEqual(selected,evaluation_directory(SOURCE_ROOT,run,state,name))
            self.assertEqual(before,{p.name:p.read_bytes() for p in old.iterdir()})

    def test_unknown_source_and_unreviewed_config_cannot_select_new_namespace(self):
        run=self.temp();sigs=self.signatures();state=self.state(sigs)
        name='broader-0001';write_json(run/'evaluations'/name/'spec.json',dict(implementation='unknown'))
        with self.approvals(sigs),patch('brass_ai.recovery.resume.implementation_id',return_value=CURRENT):
            with self.assertRaisesRegex(ValueError,'reviewed transition history'):
                evaluation_directory(SOURCE_ROOT,run,state,name)
            write_json(run/'evaluations'/name/'spec.json',dict(implementation=OLD))
            state['signature']=copy.deepcopy(state['signature']);state['signature']['config']['learning_rate']=1e-2
            # A fabricated target cannot connect the recorded, approved chain.
            with self.assertRaisesRegex(ValueError,'reviewed transition history'):
                evaluation_directory(SOURCE_ROOT,run,state,name)
            with self.assertRaisesRegex(ValueError,'single evaluation'):
                evaluation_directory(SOURCE_ROOT,run,state,'../outside')

    def test_matching_source_still_uses_original_directory_for_normal_spec_check(self):
        run=self.temp();sigs=self.signatures();state=self.state(sigs)
        directory=run/'evaluations'/'monitor-0001'
        write_json(directory/'spec.json',dict(implementation=CURRENT,config=dict(changed=True)))
        with patch('brass_ai.recovery.resume.implementation_id',return_value=CURRENT):
            self.assertEqual(evaluation_directory(SOURCE_ROOT,run,state,'monitor-0001'),directory)
        # Other spec fields remain the caller's exact equality checks; the
        # helper must not create a new directory merely for changed config.

    def test_champion_source_upgrade_preserves_old_report_and_reuses_only_new_report(self):
        from brass_ai.recovery.champion import initialize,schedule
        from brass_ai.recovery.config import CHAMPION_LEGACY
        from test_champion import rows
        run=self.temp();sigs=self.signatures()[::2];entries=[]
        for sig in sigs:sig['config']['champion']=copy.deepcopy(CHAMPION_LEGACY)
        for generation in (0,1):
            path=run/f'generation-{generation:04d}.pt';path.write_bytes(str(generation).encode())
            entries.append(dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),
                model_id=str(generation),generation=generation))
        state=dict(format='step5-recovery-league/2.2.0',iteration=1,signature=sigs[0],games_done=0,
            history=entries[:1],evaluation_anchor=entries[0])
        service=SimpleNamespace(models={'0':None,'1':None});actor=SimpleNamespace(collect=lambda jobs:[])
        with patch('brass_ai.recovery.retention.slim_checkpoint',side_effect=lambda root,run,e,existing=None:e):
            initialize(SOURCE_ROOT,run,state);state.update(history=entries,games_done=6400)
            before=copy.deepcopy(state)
            with patch('brass_ai.recovery.champion.implementation_id',return_value=OLD),\
                    patch('brass_ai.recovery.resume.implementation_id',return_value=OLD),\
                    patch('brass_ai.recovery.champion.read_shards',return_value=[]),\
                    patch('brass_ai.recovery.champion.gate_rows',return_value=rows()):
                schedule(SOURCE_ROOT,run,state,service,actor)
            old=run/'evaluations/champion-00006400';old_bytes={p.name:p.read_bytes() for p in old.iterdir()}
            state=copy.deepcopy(before);state['signature']=sigs[-1]
            state['source_transitions']=[dict(from_signature=sigs[0],to_signature=digest(sigs[-1]))]
            uncommitted=copy.deepcopy(state);calls=[]
            actor.collect=lambda jobs:calls.append(jobs) or []
            with self.approvals(sigs),patch('brass_ai.recovery.champion.implementation_id',return_value=CURRENT),\
                    patch('brass_ai.recovery.resume.implementation_id',return_value=CURRENT),\
                    patch('brass_ai.recovery.champion.read_shards',return_value=[]),\
                    patch('brass_ai.recovery.champion.gate_rows',return_value=rows()):
                self.assertTrue(schedule(SOURCE_ROOT,run,state,service,actor)['accepted'])
                self.assertEqual(len(calls),1)
                self.assertTrue(all('-source-'+CURRENT[:16] in j['path'] for j in calls[0]))
                actor.collect=lambda jobs:self.fail('Completed current-source gate collected again')
                self.assertTrue(schedule(SOURCE_ROOT,run,uncommitted,service,actor)['accepted'])
            self.assertEqual(old_bytes,{p.name:p.read_bytes() for p in old.iterdir()})

    def test_namespaced_development_screens_share_retention_group(self):
        from brass_ai.recovery.config import RETENTION
        from brass_ai.recovery.retention import plan
        run=self.temp()
        write_json(run/'state.json',dict(format='step5-recovery-league/2.2.0',iteration=4,history=[],
            signature=dict(config=dict(retention=RETENTION))))
        names=['monitor-0001-source-'+OLD[:16],'monitor-0002','monitor-0003-source-'+CURRENT[:16]]
        for name in names:
            directory=run/'evaluations'/name
            write_json(directory/'spec.json',dict(config=dict(panel='development',setups=1)))
            write_json(directory/'report.json',dict(format='step5-evaluation-report/1.0.0',
                spec=dict(config=dict(panel='development',setups=1))))
        retired={e['path'] for e in plan(SOURCE_ROOT,run)['entries']}
        self.assertIn(f'evaluations/{names[0]}/report.json',retired)
        self.assertFalse(any(f'evaluations/{name}/report.json' in retired for name in names[1:]))

    def headed_fixture(self):
        run=self.temp();sigs=self.signatures()[::2]
        state=dict(format='step5-recovery-league/2.2.0',signature=sigs[0],iteration=1,games_done=320,
            next_search_games=640,ordinary_seconds=12.,search_seconds=0.)
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(sigs[0]['config'])))
        model.utility_calibration=dict(ready=True,reward_id=digest(model.config.utility),scope='old source receipt')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        sum(p.square().mean() for p in model.parameters()).backward();optimizer.step()
        path=run/'generation-0001.pt'
        mid=save_checkpoint(path,model,optimizer,step=1,training=dict(mode='step5-population-ppo',
            signature=digest(sigs[0]),fresh_run=run.relative_to(SOURCE_ROOT).as_posix()))
        entry=dict(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=mid,generation=1)
        state.update(history=[entry],population=[entry],evaluation_anchor=entry,resume=[entry])
        write_json(run/'state.json',state)
        return run,sigs,state,model,optimizer

    def test_old_headed_resume_clears_readiness_preserving_weights_adam_rng_and_receipt(self):
        run,sigs,state,original,optimizer=self.headed_fixture()
        with self.approvals(sigs):
            model,raw,changed=load_resume(SOURCE_ROOT,run,state,sigs[-1],reviewed=True)
            self.assertTrue(changed);self.assertFalse(model.utility_calibration['ready'])
            self.assertTrue(raw['training']['utility_calibration']['ready'])
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items()))
            restored=torch.optim.AdamW(model.parameters(),lr=1e-5);resume_optimizer(raw,restored)
            self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
            self.assertEqual(restored.state_dict()['param_groups'],optimizer.state_dict()['param_groups'])
            for key,values in optimizer.state_dict()['state'].items():
                for field,value in values.items():self.assertTrue(torch.equal(value,restored.state_dict()['state'][key][field]))
            updated=commit_transition(run,state,sigs[-1])
            again,second,changed=load_resume(SOURCE_ROOT,run,updated,sigs[-1])
            self.assertFalse(changed);self.assertFalse(again.utility_calibration['ready'])
            self.assertTrue(second['training']['utility_calibration']['ready'])

    def test_champion_outside_population_checked_before_source_commit(self):
        run,sigs,state,_,_=self.headed_fixture()
        path=run/'champion.pt';path.write_bytes(b'changed')
        state['champion']=dict(state['history'][0],path=path.relative_to(SOURCE_ROOT).as_posix())
        before=(run/'state.json').read_bytes()
        with self.approvals(sigs),self.assertRaisesRegex(ValueError,'checkpoint changed'):
            load_resume(SOURCE_ROOT,run,state,sigs[-1],reviewed=True)
        self.assertEqual(before,(run/'state.json').read_bytes())


if __name__=='__main__':unittest.main()
