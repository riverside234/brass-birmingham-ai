"""g611 exact adoption, schedule continuity, beta feedback and isolated PPG."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import entropy_schedule as S,exploration as E,ppg_g210 as P,updates
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.training_revision_g611 import configure
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id,write_json
from test_entropy_schedule import commit

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0611-20261104T073052792185Z'


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original=read(BUNDLE/'state.json');cls.old=cls.original['signature']['config']
        cls.new=validate_config(configure(cls.old))

    def start(self):
        s=copy.deepcopy(self.original)
        s['exploration']=E.restore(self.new,s,s['exploration'])
        s['signature'].update(config=copy.deepcopy(self.new),implementation=implementation_id())
        return s

    def at(self,k):
        s=self.start()
        while s['exploration'][S.KEY]['progress']<k:
            if S.waiting(s['exploration'][S.KEY]):
                c=S.configure(s['signature']['config'],s,'insolvency-step')
                s['exploration']=E.restore(c,s,s['exploration']);s['signature']['config']=c
            s,_=commit(s,s['signature']['config'])
        return s

    def test_config_scope_old_contracts_and_strict_bounds(self):
        self.assertEqual(validate_config(copy.deepcopy(self.old)),self.old)
        self.assertEqual(configure(self.new),self.new)
        self.assertEqual({k for k in self.new if self.new[k]!=self.old[k]}, {'update','exploration','value_phase'})
        self.assertEqual(self.new['exploration'],0.)
        self.assertEqual(self.new['insolvency'],dict(event_cost=.01,max_cost=.075))
        self.assertEqual(self.new['value_phase'],dict(self.old['value_phase'],contract=P.QUARTER_CONTRACT,
            beta_clone_min=.25,kl_cap=.003))
        for key,value in [('beta_clone_min',.125),('kl_cap',.004),('phase_interval',2),('kl_check_steps',16)]:
            bad=copy.deepcopy(self.new);bad['value_phase'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(self.new);bad['update'][S.KEY]['insolvency_steps'][-1]['at_progress']=75
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(self.new);bad['exploration']=.02
        with self.assertRaises(ValueError):validate_config(bad)

    def test_schedule_keeps_k41_baselines_and_waits_at_50_and_100(self):
        s=self.start();saved=s['exploration'][S.KEY]
        self.assertEqual((saved['progress'],saved['held'],saved['reset_count']),(41,0,0))
        self.assertEqual(saved['applied'],dict.fromkeys(E.GROUPS,.00336))
        for key in ('baseline','baseline_rows','groups','origin'):
            self.assertEqual(s['exploration'][key],self.original['exploration'][key])
        with self.assertRaisesRegex(ValueError,'not due'):S.configure(self.new,s,'insolvency-step')
        s=self.at(50);c=s['signature']['config']
        held,report=commit(s,c)
        self.assertEqual(report[S.KEY]['reason'],'waiting_for_insolvency_step')
        self.assertEqual(held['exploration'][S.KEY]['progress'],50)
        self.assertEqual(c['insolvency'],self.old['insolvency'])
        c=S.configure(c,s,'insolvency-step')
        self.assertEqual(c['insolvency'],dict(event_cost=.005,max_cost=.05))
        s=self.at(99)
        with self.assertRaisesRegex(ValueError,'not due'):S.configure(s['signature']['config'],s,'insolvency-step')
        s=self.at(100);c=s['signature']['config']
        self.assertTrue(s['exploration'][S.KEY]['waiting_for_insolvency_step'])
        self.assertEqual(s['exploration'][S.KEY]['applied'],dict.fromkeys(E.GROUPS,.001))
        step3=validate_config(S.configure(c,s,'insolvency-step'))
        self.assertEqual(step3['insolvency'],dict(event_cost=.0025,max_cost=.025))
        s['exploration']=E.restore(step3,s,s['exploration']);s['signature']['config']=step3
        self.assertEqual(s['exploration'][S.KEY]['insolvency_steps_applied'],3)
        self.assertFalse(s['exploration'][S.KEY]['waiting_for_insolvency_step'])
        _,report=commit(s,step3);self.assertEqual(report[S.KEY]['reason'],'complete')
        reverse=S.configure(step3,s,'insolvency-step',reverse=True)
        self.assertEqual(reverse['insolvency'],c['insolvency'])
        self.assertEqual(S.restore_run(reverse,s)['progress'],100)

    def test_receipt_rejects_old_schedule_and_unauthorized_rebaseline(self):
        s=self.start();receipt=S.receipt(self.new,s['exploration'])
        self.assertEqual(receipt['contract'],S.EXTENDED_CONTRACT)
        behavior=SimpleNamespace(entropy_schedule=receipt,entropy_alphas=receipt['alphas'])
        self.assertEqual(S.behavior_weights(self.new['update'],behavior),receipt['alphas'])
        behavior.entropy_schedule=S.receipt(self.old,self.original['exploration'])
        with self.assertRaises(ValueError):S.behavior_weights(self.new['update'],behavior)
        bad=copy.deepcopy(self.new);bad['update'][S.KEY]['final']['rail_other']=.0005
        with self.assertRaises(ValueError):S.restore_run(bad,self.original)

    def test_beta_ladder_strict_half_cap_empty_and_escalation(self):
        beta=1.;trips=[];c=self.new['value_phase']
        for expected in (.5,.25,.25):
            beta,trips,r=P.beta_after_phase(beta,trips,dict(kl_guard_tripped=False,optimizer_steps=1,probe_kl=.001499),c)
            self.assertEqual(beta,expected);self.assertEqual(r['threshold'],.0015)
        for kl in (.0015,.0025,.003):
            self.assertEqual(P.beta_after_phase(1.,[],dict(kl_guard_tripped=False,optimizer_steps=1,probe_kl=kl),c)[0],1.)
        self.assertEqual(P.beta_after_phase(1.,[],dict(kl_guard_tripped=False,optimizer_steps=0),c)[0],1.)
        self.assertEqual(P.beta_after_phase(.25,[True,True],dict(kl_guard_tripped=True,optimizer_steps=0),c)[0],.5)
        self.assertEqual(P.beta_after_phase(32.,[True,True],dict(kl_guard_tripped=True,optimizer_steps=0),c)[0],32.)
        self.assertEqual(P.beta_after_phase(1.,[],dict(kl_guard_tripped=False,optimizer_steps=1,probe_kl=.0009),self.old['value_phase'])[0],1.)

    def test_zero_epsilon_real_behavior_and_tempered_openings(self):
        from test_gae_value_phase import fixture
        from test_opening_ppo import opening_rows
        from brass_ai.recovery import opening_objective as O
        from brass_ai.recovery.telemetry import Timings
        model,_,cfg,(_,data)=fixture(games=3,decisions=4)
        parent=[d for d in data if d[2]==data[1][2]]
        branch=[(dict(r,_branch=True),t,(g+'/branch/test',s),e) for r,t,(g,s),e in parent]
        behavior,normal=opening_rows(model,parent,0.,opening=E.THREE_OPENING,objective=O.FORK_T15)
        _,branched=opening_rows(model,branch,0.,opening=E.FORK_OPENING,objective=O.FORK_T15)
        for row,*_ in branched:row['behavior']['policy_id']=behavior.policy_id
        ds=updates.Dataset(normal+branched,'cpu');ds.opening_reference=O.Reference(ds,behavior,O.FORK_T15)
        self.assertEqual(ds.opening_reference.mask.tolist(),[True,True,True,False,True,True,False,False])
        updates.policy_measure(model,ds,range(len(ds.rows)),behavior,cfg['update'],Timings(),verify=True)
        _,stale=opening_rows(model,parent,.02,opening=E.THREE_OPENING,objective=O.FORK_T15)
        with self.assertRaises(ValueError):updates.verify_behavior_metadata(updates.Dataset(stale,'cpu'),behavior)

    def test_003_guard_accepts_0028_and_rolls_back_above_cap_with_Adam(self):
        from brass_ai.recovery.value_phase import DeferredKL
        from unittest.mock import Mock
        for retained_kl,kept in ((.0028,1),(.00301,0)):
            model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
            initial=copy.deepcopy((model.state_dict(),optimizer.state_dict()))
            reference=Mock();reference.measure.side_effect=[.0031,retained_kl]
            report={};guard=DeferredKL(model,optimizer,None,reference,[0],self.new['value_phase']['kl_cap'],report)
            model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
            accepted=copy.deepcopy((model.state_dict(),optimizer.state_dict()));guard.accept(16,1,None)
            model(torch.ones(1,2)).sum().backward();optimizer.step()
            self.assertGreater(guard.measure(32,'periodic'),.003)
            count,kl=guard.finish();self.assertEqual(count,kept);self.assertLessEqual(kl,.003)
            self.assertTrue(_same_payload(accepted if kept else initial,(model.state_dict(),optimizer.state_dict())))

    def test_signed_resume_and_helpers_preserve_generation_and_retry(self):
        from brass_ai.recovery.resume import check_signature,commit_transition
        sig=dict(self.original['signature'],config=self.new,implementation=implementation_id())
        check_signature(self.original,sig,reviewed=True)
        spec=importlib.util.spec_from_file_location('g611_helper',SOURCE_ROOT/'scripts/configure-training-g611.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0611.pt';checkpoint.write_bytes(b'committed fixture')
            state=copy.deepcopy(self.original);state['history'][-1].update(path='artifacts/run/generation-0611.pt',sha256=sha(checkpoint))
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes()
            output='configs/step5/g611.json'
            first=helper.generate(root,'artifacts/run',output,611)
            self.assertEqual(first,helper.generate(root,'artifacts/run',output,611))
            self.assertEqual((run/'state.json').read_bytes(),before)
            self.assertEqual(read(root/output),self.new)
            committed=commit_transition(run,state,sig)
            self.assertEqual(S.restore_run(self.new,committed),S.restore_run(self.new,state))
            self.assertEqual(helper.generate(root,'artifacts/run',output,611),first)
            for key in ('iteration','games_done','population','next_seed','quality_pool','exploration'):
                self.assertEqual(committed[key],state[key])
            bad=copy.deepcopy(self.new);bad['learning_rate']=.001
            with self.assertRaises(ValueError):check_signature(state,dict(sig,config=bad),reviewed=True)
            with self.assertRaises(ValueError):helper.generate(root,'artifacts/run',output,612)
            (run/'iteration-0611').mkdir();(run/'iteration-0611/game').write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError,'interrupted'):helper.generate(root,'artifacts/run',output,611)

    def test_actual_checkpoint_resume_and_later_reward_receipts(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.recovery.utility import migrate,reward_spec,calibration_ready
        from brass_ai.recovery.resume import load_resume,commit_transition,check_signature
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);state=copy.deepcopy(self.original);old=copy.deepcopy(self.old)
            old['model_size']='small';state['signature']['config']=old
            model=FourSeatNet(ModelConfig.preset('small'));optimizer=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
            migrate(model,optimizer,reward_spec(old))
            p=next(model.parameters());p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
            model.utility_calibration=dict(ready=True,reward_id=digest(reward_spec(old)))
            path=run/'generation-0611.pt'
            identity=save_checkpoint(path,model,optimizer,step=611,training=dict(fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),
                signature=digest(state['signature']),exploration=state['exploration']))
            entry=dict(generation=611,path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity)
            state['history']=[entry];state['population']=[entry];state['evaluation_anchor']=entry
            write_json(run/'state.json',state)
            expected=copy.deepcopy((model.state_dict(),optimizer.state_dict(),torch.get_rng_state()))
            new=configure(old);sig=dict(state['signature'],config=new,implementation=implementation_id())
            loaded,raw,changed=load_resume(SOURCE_ROOT,run,state,sig,reviewed=True)
            self.assertTrue(changed);resumed=torch.optim.AdamW(loaded.parameters(),lr=old['learning_rate']);resume_optimizer(raw,resumed)
            self.assertTrue(calibration_ready(loaded))
            self.assertTrue(_same_payload(expected,(loaded.state_dict(),resumed.state_dict(),torch.get_rng_state())))
            committed=commit_transition(run,state,sig)
            loaded,raw,changed=load_resume(SOURCE_ROOT,run,committed,sig,reviewed=True)
            self.assertFalse(changed);self.assertTrue(calibration_ready(loaded))
            self.assertEqual(S.restore_run(new,committed)['progress'],41)
        for k in (50,100):
            s=self.at(k);c=S.configure(s['signature']['config'],s,'insolvency-step')
            self.assertTrue(check_signature(s,dict(s['signature'],config=c),reviewed=True))

    def test_PPG_adoption_keeps_cache_beta_optimizers_and_small_smoke(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.utility import migrate,reward_spec
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        from brass_ai.learning.batching import model_id
        from brass_ai.recovery import gae
        from test_ppg_g210 import prepare
        from test_gae_value_phase import finish
        from test_value_buffer import commit as commit_cache
        old=copy.deepcopy(self.old);old['model_size']='small';old['update']['preparation_workers']=0
        old['value_phase'].update(batch_size=128,clone_rows=24,clone_batch_size=8)
        model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
        migrate(model,opt,reward_spec(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);phase=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            phase.local_scoring=Companion(model,old['local_scoring_auxiliary'],raw=dict(model_id=model_id(model)))
            prepare(model,phase,1,companion=True)
            phase.buffer.stage(phase.critic_dataset,phase.train_ids,phase.heldout_ids,phase.actor_ids,
                phase.collection_source,model,phase.split['selection'])
            phase.state=gae.initial(old['advantage'],1);phase.critic_state['generation']=1
            phase.beta=1.;phase.trips=[True,False]
            raw=dict(model_config=model.config.to_dict(),training={KEY:phase.payload(),
                'local_scoring_auxiliary':phase.local_scoring.payload(1)})
            write_json(run/'state.json',dict(iteration=1));phase.buffer.committed(1)
            before=copy.deepcopy((model.state_dict(),opt.state_dict(),phase.optimizer.state_dict(),phase.state,phase.critic_state,torch.get_rng_state()))
            hashes={p.name:sha(p) for p in phase.buffer.directory.iterdir()}
            new=configure(old);q=TrainingPhases(model,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual((q.beta,q.trips),(1.,[True,False]));self.assertIsNone(q.startup(1))
            self.assertEqual(q.buffer.receipts,phase.buffer.receipts)
            self.assertEqual(hashes,{p.name:sha(p) for p in phase.buffer.directory.iterdir()})
            self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            q.local_scoring=Companion(model,new['local_scoring_auxiliary'],raw=raw,generation=1)
            prepare(model,q,2,companion=True);r=q.fit(model)
            self.assertEqual(r['kl_cap'],.003);self.assertGreater(r['optimizer_steps'],0);self.assertLessEqual(r['probe_kl'],.003)
            update=dict(categorical_probe=dict(categorical_kl=0.),controller_warmup_done=True,optimizer_steps=1,kl_early_stop=False,gae={})
            finish(q,update,2);payload=q.payload();commit_cache(run,q,2)
            resumed=TrainingPhases(model,new,raw=dict(training={KEY:payload}),generation=2,root=SOURCE_ROOT,run=run)
            self.assertEqual((resumed.beta,resumed.trips),(q.beta,q.trips))
            self.assertEqual([r['generation'] for r in resumed.buffer.receipts],[1,2])
            self.assertTrue(_same_payload(q.optimizer.state_dict(),resumed.optimizer.state_dict()))
            print('g611 isolated PPG smoke:',{k:r[k] for k in ('optimizer_steps','attempted_steps','probe_kl','kl_cap','beta_clone_next','early_stop_reason')})

    def test_third_reward_step_preserves_tensors_and_retires_cache_after_commit(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.utility import migrate,reward_spec,event_cost_upgrade,calibration_ready
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery import gae
        from test_ppg_g210 import prepare
        c=copy.deepcopy(self.new);c['model_size']='small';c['update']['preparation_workers']=0
        c['local_scoring_auxiliary']['mode']='off'
        c['value_phase'].update(batch_size=128,clone_rows=24,clone_batch_size=8)
        S._reward(c,dict(event_cost=.005,max_cost=.05))
        model=FourSeatNet(ModelConfig.preset('small'));optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'])
        migrate(model,optimizer,reward_spec(c))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);p=TrainingPhases(model,c,root=SOURCE_ROOT,run=run);prepare(model,p,1)
            p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,p.actor_ids,p.collection_source,model,p.split['selection'])
            p.state=gae.initial(c['advantage'],1);p.critic_state['generation']=1
            raw=dict(model_config=model.config.to_dict(),training={KEY:p.payload()})
            write_json(run/'state.json',dict(iteration=1,signature=dict(config=c)));p.buffer.committed(1)
            before=copy.deepcopy((model.state_dict(),optimizer.state_dict(),p.optimizer.state_dict(),torch.get_rng_state()))
            target=copy.deepcopy(c);S._reward(target,dict(event_cost=.0025,max_cost=.025))
            self.assertTrue(event_cost_upgrade(reward_spec(c),reward_spec(target)))
            bad=copy.deepcopy(target);S._reward(bad,dict(event_cost=.002,max_cost=.025))
            self.assertFalse(event_cost_upgrade(reward_spec(c),reward_spec(bad)))
            model._reviewed_cap_upgrade=True;migrate(model,optimizer,reward_spec(target))
            self.assertFalse(calibration_ready(model))
            q=TrainingPhases(model,target,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.buffer.receipts,[]);self.assertEqual(q.g210_migration['invalidated_generations'],[1])
            self.assertTrue(_same_payload(before,(model.state_dict(),optimizer.state_dict(),q.optimizer.state_dict(),torch.get_rng_state())))
            with self.assertRaisesRegex(ValueError,'committed reward'):q.startup(1)
            write_json(run/'state.json',dict(iteration=1,signature=dict(config=target)))
            self.assertGreater(q.startup(1)['deleted_bytes'],0)
            self.assertEqual(q.startup(1)['deleted_files'],[])


if __name__=='__main__':unittest.main()
