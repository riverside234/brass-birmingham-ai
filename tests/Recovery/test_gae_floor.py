"""Reviewed g253 lambda floor, exact continuation and isolated optimizer smoke."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import gae
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature,gae_floor_receipt_preserved
from brass_ai.step5.league import implementation_id,write_json
from test_ppg_window import tick

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0253-20261102T075610938330Z'


class Floor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.run_state=read(BUNDLE/'state.json')
        cls.old=cls.run_state['signature']['config']
        cls.new=gae.floor_options(cls.old)

    def test_strict_config_and_only_requested_change(self):
        validate_config(self.new)
        self.assertEqual({k for k in self.old if self.old[k]!=self.new[k]},{'advantage'})
        self.assertEqual(self.old['advantage']['contract'],gae.TRAINED_CONTRACT)
        self.assertEqual(gae.floor_options(self.new),self.new)
        for value in (.95,.99,True,None,float('nan')):
            c=copy.deepcopy(self.new);c['advantage']['lambda_floor']=value
            with self.assertRaises(ValueError):validate_config(c)
        bad=copy.deepcopy(self.old);bad['advantage']['lambda_floor']=.98
        with self.assertRaises(ValueError):validate_config(bad)

    def test_healthy_windows_cannot_lower_floor_and_legacy_unchanged(self):
        for cfg,expected in ((self.old['advantage'],.95),(self.new['advantage'],.98)):
            s=gae.initial(cfg)
            for g in range(1,61):
                s,r=tick(s,cfg,steps=8)
                self.assertGreaterEqual(s['active_lambda'],gae.lambda_floor(cfg))
            self.assertEqual(s['active_lambda'],expected)
            self.assertEqual(r['lambda_floor'],expected)
            self.assertEqual(r['reason'],'final_stage')
            gae.validate_state(json.loads(json.dumps(s)),cfg,60)

    def test_hotfix_preserves_mixed_gate_evidence_and_counters(self):
        c=self.old['advantage'];s=gae.initial(c)
        for g in range(1,31):s,_=tick(s,c,steps=8)
        s['table_enabled']=dict(historical=True,legacy=False)
        s['table_origins']=dict(historical=21,legacy=None)
        before=copy.deepcopy(s);new,r=gae.migrate_floor(s,c,self.new['advantage'],30)
        self.assertEqual(s,before);self.assertEqual((r['lambda_before'],r['lambda_after']),(.95,.98))
        for k in s:
            if k not in ('active_lambda','stage_origin','healthy_streak'):self.assertEqual(new[k],s[k],k)
        self.assertEqual(new['stage_origin'],30)
        with self.assertRaises(ValueError):gae.validate_state(s,self.new['advantage'],30)
        new,r=tick(new,self.new['advantage'])
        self.assertEqual(r['table_gate']['historical']['lambda_used'],.98)
        self.assertEqual(r['table_gate']['legacy']['lambda_used'],1.)
        self.assertEqual(r['table_gate']['historical']['origin'],21)

    def test_trained_start_health_regression_using_available_g253_recipe(self):
        # The old g1360 export used by that fixture is no longer retained here.
        # Run its original guard assertions with the available, identical GAE-3 block.
        import test_trained_startup as startup
        with patch.object(startup,'config',return_value=copy.deepcopy(self.old)):
            startup.Startup().test_gae_98_fresh_evidence_twenty_observations_and_guards()

    def test_exported_signature_review_rejects_unrelated_changes_and_retry(self):
        s=copy.deepcopy(self.run_state);target=dict(s['signature'],config=self.new,implementation=implementation_id())
        self.assertTrue(check_signature(s,target,reviewed=True))
        with self.assertRaises(ValueError):check_signature(s,target)
        bad=copy.deepcopy(target);bad['config']['exploration']=.08
        with self.assertRaises(ValueError):check_signature(s,bad,reviewed=True)
        entry=s['history'][-1];raw=dict(training=dict(signature=digest(s['signature'])))
        self.assertTrue(gae_floor_receipt_preserved(s,target,raw,entry))
        self.assertFalse(gae_floor_receipt_preserved(s,bad,raw,entry))
        s.setdefault('source_transitions',[]).append(dict(checkpoint=entry,from_signature=s['signature'],to_signature=digest(target)))
        s['signature']=target
        self.assertFalse(check_signature(s,target,reviewed=True))
        self.assertTrue(gae_floor_receipt_preserved(s,target,raw,entry))

    def test_real_value_adam_checkpoint_restore_and_small_smoke(self):
        # Isolated CPU optimizer smoke; no native games or production checkpoints.
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from test_mlp_v11 import inputs
        from brass_ai.recovery import ppg_replay,ppg_g210
        model=FourSeatNet(ModelConfig.preset('small'))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp, patch.object(ppg_replay,'Buffer') as buffer:
            buffer.return_value.pending=[]
            p=TrainingPhases(model,self.old,root=Path(tmp),run=Path(tmp),generation=0)
            states,actions,mask,mapping=inputs(2,3)
            loss=model(states,actions,mask,mapping)[1].square().mean()
            loss.backward();p.optimizer.step();p.optimizer.zero_grad()
            p.state=gae.initial(self.old['advantage'],253);p.state['active_lambda']=.95
            from brass_ai.recovery import critic_data
            p.critic_state=dict(critic_data.initial(self.old['critic_data']),generation=253)
            saved=ppg_g210.payload(p);raw=dict(training={KEY:saved})
            path=Path(tmp)/'checkpoint.pt';torch.save(dict(weights=model.state_dict(),raw=raw),path)
            loaded=torch.load(path,weights_only=False);rng=torch.get_rng_state().clone()
            before=copy.deepcopy(model.state_dict())
            q=TrainingPhases(model,self.new,raw=loaded['raw'],generation=253,root=Path(tmp),run=Path(tmp))
            self.assertEqual(q.state['active_lambda'],.98)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(_same_payload(before,model.state_dict()))
            self.assertTrue(_same_payload(p.optimizer.state_dict(),q.optimizer.state_dict()))
            self.assertEqual(q.beta,p.beta);self.assertEqual(q.cache_origin,p.cache_origin)
            self.assertEqual(q.critic_state,p.critic_state)
            self.assertEqual(q.buffer,buffer.return_value)
            restored=ppg_g210.payload(q)
            again=TrainingPhases(model,self.new,raw=dict(training={KEY:restored}),generation=253,root=Path(tmp),run=Path(tmp))
            self.assertIsNone(again.gae_floor_migration);self.assertEqual(again.state,q.state)
            self.assertTrue(_same_payload(again.optimizer.state_dict(),q.optimizer.state_dict()))
            # Real advantage preparation must use .98; terminal MC targets stay unchanged.
            data=[(dict(view=dict(context=dict(era='CanalEra',round=2,phase='action'))),None,('g',0),None)]*2
            returns=torch.tensor([.8,1.]);mc=returns-torch.tensor([.2,.3])
            advantage,report=gae.compute(data,[0,1],mc,returns,q.state['active_lambda'],[.98,.95,.9])
            self.assertAlmostEqual(float(advantage[0]),-.1+.98*.7,places=6)
            self.assertEqual(report['lambda'],.98)
            model(states,actions,mask,mapping)[1].square().mean().backward();again.optimizer.step()
            self.assertTrue(all(int(v['step'])==2 for v in again.optimizer.state.values()))

    def test_helper_retry_stale_generation_and_missing_checkpoint(self):
        spec=importlib.util.spec_from_file_location('configure_floor',SOURCE_ROOT/'scripts/configure-gae-floor.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0253.pt';checkpoint.write_bytes(b'authored checksum fixture')
            s=copy.deepcopy(self.run_state);s['history'][-1].update(path='artifacts/run/generation-0253.pt',sha256=sha(checkpoint))
            write_json(run/'state.json',s);before=sha(run/'state.json')
            output='configs/step5/floor.json'
            a=helper.generate(root,'artifacts/run',output,253)
            self.assertEqual(a,helper.generate(root,'artifacts/run',output,253))
            self.assertEqual(read(root/output),self.new);self.assertEqual(before,sha(run/'state.json'))
            with self.assertRaisesRegex(ValueError,'Stale'):helper.generate(root,'artifacts/run',output,252)
            checkpoint.unlink()
            with self.assertRaisesRegex(ValueError,'missing/changed'):helper.generate(root,'artifacts/run',output,253)


if __name__=='__main__':unittest.main()
