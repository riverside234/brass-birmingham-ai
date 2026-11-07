"""Two-iteration PPG: exact adoption, restart, skipped phases and bounded fitting."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import ppg_g210 as P,gae
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition,apply_optimizer_config
from brass_ai.step5.league import write_json,implementation_id

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g1241-20261107T023636701265Z'


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original=read(BUNDLE/'state.json');cls.old=cls.original['signature']['config']
        cls.new=validate_config(P.two_iteration_options(cls.old))

    def test_exact_options_and_strict_validation(self):
        self.assertEqual(P.two_iteration_options(self.new),self.new)
        self.assertEqual({k for k in self.old if self.old[k]!=self.new[k]},{'value_phase'})
        a,b=self.old['value_phase'],self.new['value_phase']
        self.assertEqual({k for k in a if a[k]!=b[k]},
            {'contract','beta_clone_min','learning_rate_factor','phase_interval'})
        self.assertEqual((b['beta_clone_min'],b['learning_rate_factor'],b['phase_interval']),(.125,.5,2))
        for k,v in [('phase_interval',2.),('beta_clone_min',.1),('learning_rate_factor',.65),
                    ('buffer_iterations',4),('kl_cap',.004),('acceptance_steps',8)]:
            with self.assertRaises(ValueError):P.validate(dict(b,**{k:v}))
        for step,want in [(1,.1),(16,1.),(200,.001)]:
            self.assertAlmostEqual(P.rate_factor(step,200,b),want)
        for rate in (3e-6,7.5e-6,1.8e-5):
            self.assertEqual(P.rate_receipt(rate,1243,20,b)['peak_lr'],rate*.5)
            self.assertEqual(P.rate_receipt(rate,1243,20,a)['peak_lr'],rate*.65)

    def test_beta_floor_and_only_actual_phases_enter_trip_window(self):
        c=self.new['value_phase'];report=dict(scheduled=True,kl_guard_tripped=False,optimizer_steps=16,probe_kl=.001)
        beta,trips,r=P.beta_after_phase(.25,[True,False],report,c)
        self.assertEqual((beta,trips,r['reason']),(.125,[True,False,False],'retained_KL_below_half_cap'))
        self.assertEqual(P.beta_after_phase(beta,trips,report,c)[0],.125)
        skipped=dict(scheduled=False,kl_guard_tripped=False,optimizer_steps=0)
        self.assertEqual(P.beta_after_phase(beta,trips,skipped,c)[:2],(beta,trips))
        self.assertEqual(P.beta_after_phase(beta,trips,skipped,c)[2]['reason'],'not_due')
        trip=dict(report,kl_guard_tripped=True)
        self.assertEqual(P.beta_after_phase(.125,[],trip,c)[0],.1875)
        self.assertEqual(P.beta_after_phase(.125,[True,True],trip,c)[0],.25)
        for bad in (dict(skipped,optimizer_steps=1),dict(skipped,kl_guard_tripped=True)):
            with self.assertRaises(ValueError):P.beta_after_phase(beta,trips,bad,c)

    def test_committed_clock_boundaries(self):
        c=self.new['value_phase']
        for generation,next_phase in [(1241,1243),(1242,1243),(1243,1245),(1244,1245)]:
            self.assertEqual(P.phase_clock(1241,generation,c)['next_phase'],next_phase)
        for origin,g in [(True,1241),(1242,1241),(-1,0),(1,2.)]:
            with self.assertRaises(ValueError):P.phase_clock(origin,g,c)

    def test_real_checkpoint_and_retry_safe_helper(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.recovery.utility import migrate,reward_spec
        spec=importlib.util.spec_from_file_location('g1241_helper',SOURCE_ROOT/'scripts/configure-training-g1241.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);state=copy.deepcopy(self.original);old=state['signature']['config'];old['model_size']='small'
            model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
            migrate(model,opt,reward_spec(old));apply_optimizer_config(opt,old)
            p=next(model.parameters());p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
            model.utility_calibration=dict(ready=True,reward_id=digest(reward_spec(old)))
            path=run/'generation-1241.pt';identity=save_checkpoint(path,model,opt,step=1241,
                training=dict(fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(state['signature']),
                              exploration=state['exploration']))
            entry=dict(generation=1241,path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity)
            state['history']=[entry];state['population']=[entry];state['evaluation_anchor']=entry
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes();checksum=sha(path)
            output=SOURCE_ROOT/'configs/step5'/('test-g1241-'+run.name+'.json');self.addCleanup(lambda:output.unlink(missing_ok=True))
            args=(SOURCE_ROOT,str(run),str(output),1241)
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
            with self.assertRaisesRegex(ValueError,'generation differs'):helper.generate(*args[:3],1242)
            (run/'iteration-1241').mkdir();(run/'iteration-1241/partial').write_text('pending')
            with self.assertRaisesRegex(ValueError,'interrupted'):helper.generate(*args)

    def test_cache_adoption_skip_restart_and_isolated_PPG_smoke(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.utility import migrate,reward_spec
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        from brass_ai.learning.batching import model_id
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
            p.local_scoring=Companion(model,old['local_scoring_auxiliary'],raw=dict(model_id=model_id(model)))
            prepare(model,p,1,companion=True)
            p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,p.actor_ids,p.collection_source,model,p.split['selection'])
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1;p.beta=.25;p.trips=[True,False]
            raw=dict(model_config=model.config.to_dict(),training={KEY:p.payload(),'local_scoring_auxiliary':p.local_scoring.payload(1)})
            write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            before=copy.deepcopy((model.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            new=P.two_iteration_options(old);q=TrainingPhases(model,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual((q.beta,q.trips,q.phase_origin),(.25,[True,False],1));self.assertIsNone(q.startup(1))
            self.assertEqual(q.buffer.receipts,p.buffer.receipts)
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            q.local_scoring=Companion(model,new['local_scoring_auxiliary'],raw=raw,generation=1)
            update=dict(categorical_probe=dict(categorical_kl=0.),controller_warmup_done=True,optimizer_steps=1,kl_early_stop=False,gae={})
            for generation in (2,3,4):
                prepare(model,q,generation,companion=True,rate=7.5e-6)
                before=copy.deepcopy((model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),{k:o.state_dict() for k,o in q.local_scoring.optimizers.items()},torch.get_rng_state()))
                trip_before=list(q.trips)
                if generation%2==0:
                    with patch.object(q.buffer,'datasets',side_effect=AssertionError('Skipped phase opened old replay')):
                        r=q.fit(model)
                    self.assertFalse(r['scheduled']);self.assertEqual(r['optimizer_steps'],0)
                    self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),{k:o.state_dict() for k,o in q.local_scoring.optimizers.items()},torch.get_rng_state())))
                    self.assertEqual(r['gate_critic_training'],dict(source='ppg',optimizer_steps=0))
                else:
                    r=q.fit(model);self.assertTrue(r['scheduled']);self.assertGreater(r['optimizer_steps'],0)
                    self.assertEqual(r['acceptance_generations'],[2,3]);self.assertLessEqual(r['probe_kl'],.003)
                    self.assertEqual(r['peak_learning_rate'],7.5e-6*.5)
                    self.assertAlmostEqual(r['attempted_learning_rates'][0],7.5e-6*.5*.1)
                    print('g1241 isolated PPG smoke:',{k:r[k] for k in ('optimizer_steps','attempted_steps','probe_kl','peak_learning_rate')})
                finish(q,copy.deepcopy(update),generation)
                if generation%2==0:self.assertEqual(q.trips,trip_before)
                payload=q.payload();aux=q.local_scoring.payload(generation);commit(run,q,generation)
                resumed=TrainingPhases(model,new,raw=dict(training={KEY:payload}),generation=generation,root=SOURCE_ROOT,run=run)
                self.assertEqual((resumed.beta,resumed.trips,resumed.phase_origin),(q.beta,q.trips,1))
                self.assertTrue(_same_payload(q.optimizer.state_dict(),resumed.optimizer.state_dict()))
                self.assertEqual([r['generation'] for r in resumed.buffer.receipts],[generation-1,generation])
                self.assertEqual(P.phase_clock(resumed.phase_origin,generation,new['value_phase'])['next_phase'],3 if generation==2 else 5)
                resumed.local_scoring=Companion(model,new['local_scoring_auxiliary'],raw=dict(training={'local_scoring_auxiliary':aux}),generation=generation)
                q=resumed
            corrupt=copy.deepcopy(payload);corrupt['phase_origin']=5
            with self.assertRaisesRegex(ValueError,'phase origin'):
                TrainingPhases(model,new,raw=dict(training={KEY:corrupt}),generation=4,root=SOURCE_ROOT,run=run)


if __name__=='__main__':unittest.main()
