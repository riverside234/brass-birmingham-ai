"""Every-iteration PPG and one-time beta hotfix continuation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import ppg_g210 as P,gae
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition,apply_optimizer_config
from brass_ai.step5.league import write_json,implementation_id

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g1247-20261107T033857653329Z'

class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original=read(BUNDLE/'state.json');cls.old=cls.original['signature']['config']
        cls.new=validate_config(P.every_iteration_options(cls.old))

    def test_scope_rate_and_limits(self):
        self.assertEqual(P.every_iteration_options(self.new),self.new)
        self.assertEqual({k for k in self.old if self.old[k]!=self.new[k]},{'value_phase'})
        a,b=self.old['value_phase'],self.new['value_phase']
        self.assertEqual({k for k in a if a[k]!=b[k]},{'contract','phase_interval','learning_rate_factor'})
        self.assertEqual((b['phase_interval'],b['learning_rate_factor'],b['beta_clone_min']),(1,.65,.125))
        self.assertEqual(P.rate_receipt(7.5e-6,1248,20,b)['peak_lr'],7.5e-6*.65)
        for key,value in [('phase_interval',2),('learning_rate_factor',.5),('kl_cap',.004)]:
            with self.assertRaises(ValueError):P.validate(dict(b,**{key:value}))
        self.assertEqual(P.beta_after_phase(.125,[],dict(kl_guard_tripped=True,optimizer_steps=16),b)[0],.1875)

    def test_one_time_hotfix_replay_restart_and_smoke(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.utility import migrate,reward_spec
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from test_ppg_g210 import prepare
        from test_gae_value_phase import finish
        from test_value_buffer import commit
        old=copy.deepcopy(self.old);old['model_size']='small';old['update']['preparation_workers']=0
        old['value_phase'].update(batch_size=128,clone_rows=24,clone_batch_size=8)
        torch.manual_seed(42)
        model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
        migrate(model,opt,reward_spec(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            prepare(model,p,1,companion=True);p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,p.actor_ids,p.collection_source,model,p.split['selection'])
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1;p.beta=.25;p.trips=[True,False]
            raw=dict(model_config=model.config.to_dict(),training={KEY:p.payload()})
            write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            before=copy.deepcopy((model.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            new=P.every_iteration_options(old)
            for _ in range(2):
                q=TrainingPhases(model,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
                self.assertEqual(q.beta,.125);self.assertEqual(q.trips,[True,False])
                self.assertEqual(q.adoption['beta_before'],.25);self.assertEqual(q.adoption['next_phase'],2)
                self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
                self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            for generation in (2,3):
                prepare(model,q,generation,rate=7.5e-6,companion=True);r=q.fit(model)
                self.assertTrue(r['scheduled']);self.assertEqual(r['phase_interval'],1)
                self.assertGreater(r['optimizer_steps'],0);self.assertLessEqual(r['probe_kl'],.003)
                self.assertEqual(r['peak_learning_rate'],7.5e-6*.65)
                update=dict(categorical_probe=dict(categorical_kl=0.),controller_warmup_done=True,optimizer_steps=1,kl_early_stop=False,gae={})
                finish(q,update,generation)
                # Simulate a later adaptive increase; restart must not reapply the reset.
                q.beta=.1875;payload=q.payload();commit(run,q,generation)
                resumed=TrainingPhases(model,new,raw=dict(training={KEY:payload}),generation=generation,root=SOURCE_ROOT,run=run)
                self.assertEqual(resumed.beta,.1875);self.assertEqual(resumed.trips,q.trips)
                self.assertTrue(_same_payload(resumed.optimizer.state_dict(),q.optimizer.state_dict()))
                self.assertEqual([r['generation'] for r in resumed.buffer.receipts],[generation-1,generation])
                q=resumed
                print('g1247 PPG smoke',generation,{k:r[k] for k in ('optimizer_steps','attempted_steps','probe_kl','peak_learning_rate')})

    def test_real_checkpoint_and_retry_safe_helper(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.recovery.utility import migrate,reward_spec
        spec=importlib.util.spec_from_file_location('g1247_helper',SOURCE_ROOT/'scripts/configure-training-g1247.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);state=copy.deepcopy(self.original);old=state['signature']['config'];old['model_size']='small'
            model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
            migrate(model,opt,reward_spec(old));apply_optimizer_config(opt,old)
            p=next(model.parameters());p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
            model.utility_calibration=dict(ready=True,reward_id=digest(reward_spec(old)))
            path=run/'generation-1247.pt';identity=save_checkpoint(path,model,opt,step=1247,
                training=dict(fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(state['signature']),
                              exploration=state['exploration']))
            entry=dict(generation=1247,path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity)
            state['history']=[entry];state['population']=[entry];state['evaluation_anchor']=entry
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes();checksum=sha(path)
            output=SOURCE_ROOT/'configs/step5'/('test-g1247-'+run.name+'.json');self.addCleanup(lambda:output.unlink(missing_ok=True))
            args=(SOURCE_ROOT,str(run),str(output),1247)
            a=helper.generate(*args);self.assertEqual(a,helper.generate(*args));self.assertEqual((run/'state.json').read_bytes(),before)
            new=read(output);signature=dict(state['signature'],config=new,implementation=implementation_id())
            with self.assertRaises(ValueError):check_signature(state,signature)
            bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
            saved=torch.load(path,map_location='cpu',weights_only=True)
            for retry in range(2):
                model,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=retry==0)
                opt=torch.optim.AdamW(model.parameters(),lr=new['learning_rate']);resume_optimizer(raw,opt);apply_optimizer_config(opt,new)
                self.assertTrue(_same_payload(saved['state_dict'],dict(model.state_dict())))
                self.assertTrue(_same_payload(saved['optimizer'],opt.state_dict()))
                self.assertTrue(torch.equal(saved['rng'],torch.get_rng_state()))
                self.assertEqual(model.utility_calibration,saved['training']['utility_calibration'])
                self.assertEqual(sha(path),checksum)
                if retry==0:state=commit_transition(run,state,signature)
            self.assertEqual(a,helper.generate(*args))
            bad_output=dict(new,learning_rate=new['learning_rate']*2);write_json(output,bad_output)
            with self.assertRaisesRegex(ValueError,'output differs'):helper.generate(*args)
            write_json(output,new)
            with self.assertRaisesRegex(ValueError,'generation differs'):helper.generate(*args[:3],1248)
            (run/'iteration-1247').mkdir();(run/'iteration-1247/partial').write_text('pending')
            with self.assertRaisesRegex(ValueError,'interrupted'):helper.generate(*args)


