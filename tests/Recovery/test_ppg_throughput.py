"""Frozen-input parity, vectorized replay, delayed guard transactions and resume."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_config as P,ppg_auxiliary as A,ppg_epochs,ppg_replay,updates,value_phase,utility,gae
from brass_ai.recovery.actor_value import TrainingPhases,KEY,ppo_evidence
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_ppg_eight import setup,stage_only,collect
from test_gae_value_phase import fixture
from test_local_scoring import annotate


class Throughput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_calibration_frozen_features_preserve_weights_Adam_rng_and_subset(self):
        from brass_ai.recovery import calibration_cache
        m,opt,c,(_,data)=fixture(games=3)
        ds=updates.Dataset(data,'cpu');indices=[i for i in range(len(data)) if i%3]
        original=copy.deepcopy(m.state_dict());initial_adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        results=[];build=calibration_cache.build
        for cached in (False,True):
            m.load_state_dict(original);m.utility_calibration=None;opt.load_state_dict(copy.deepcopy(initial_adam));torch.set_rng_state(rng)
            with patch.object(calibration_cache,'build',build if cached else lambda *a:None),patch.object(m,'encode_state',wraps=m.encode_state) as enc:
                r=utility.calibrate(m,opt,ds,c['update'],indices=indices)
            results.append((copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone(),r['receipt'],enc.call_count))
        self.assertLess(results[1][-1],results[0][-1])
        for k in original:torch.testing.assert_close(results[0][0][k],results[1][0][k],rtol=2e-5,atol=1e-7)
        self.assertTrue(torch.equal(results[0][2],results[1][2]));self.assertEqual(results[0][3]['steps'],results[1][3]['steps'])
        for k,state in results[0][1]['state'].items():
            for name,t in state.items():torch.testing.assert_close(t,results[1][1]['state'][k][name],rtol=2e-5,atol=1e-7)
        head={id(p) for p in m.utility_head.parameters()}
        for k,p in m.named_parameters():
            if id(p) not in head:self.assertTrue(torch.equal(p,original[k]))
        # A new call after trunk movement builds fresh features; no global cache.
        m.utility_calibration=None
        with torch.no_grad():next(m.parameters()).add_(.001)
        with patch.object(m,'encode_state',wraps=m.encode_state) as enc:utility.calibrate(m,opt,ds,c['update'],indices=indices)
        self.assertGreater(enc.call_count,0)

    def test_calibration_build_failure_restores_gradient_flags_rates_and_Adam(self):
        m,opt,c,(_,data)=fixture(games=2);ds=updates.Dataset(data,'cpu')
        next(m.parameters()).requires_grad_(False)
        flags=[p.requires_grad for p in m.parameters()];adam=copy.deepcopy(opt.state_dict());weights=copy.deepcopy(m.state_dict())
        with patch.object(m,'encode_state',side_effect=RuntimeError('cache build failed')):
            with self.assertRaisesRegex(RuntimeError,'cache build failed'):utility.calibrate(m,opt,ds,c['update'])
        self.assertEqual(flags,[p.requires_grad for p in m.parameters()])
        self.assertTrue(_same_payload(adam,opt.state_dict()));self.assertTrue(_same_payload(weights,m.state_dict()))

    def test_cached_policy_inputs_raw_KL_gradients_duplicates_and_masks(self):
        m,_,c,(_,data)=fixture(games=2);ds=updates.Dataset(data,'cpu');ids=list(range(len(data)))
        began=time.perf_counter();ref=value_phase.RawPolicyReference(m,ds,ids,c['update']);elapsed=time.perf_counter()-began
        self.assertGreaterEqual(ref.cache.build_seconds,0.);self.assertLessEqual(ref.cache.build_seconds,elapsed)
        rng=torch.get_rng_state().clone()
        selected=[5,0,5,3,8,7]
        with torch.no_grad():m.policy_scorer[-1].weight.add_(.02)
        parts=list(ds.microbatches(selected,8,131072))
        self.assertEqual(parts,list(ref.microbatches(selected)))
        for part in parts:
            inputs=ref.inputs(part);uncached=ds.batch(part)[1]
            for a,b in zip(inputs,uncached):self.assertTrue(torch.equal(a,b))
            m.zero_grad();actual=ref.loss(part,inputs,m);actual.sum().backward()
            grad={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None}
            m.zero_grad();logits=m(*uncached)[0].float();old=torch.zeros_like(logits)
            for j,i in enumerate(part):old[j,:len(ref.rows[i])]=ref.rows[i]
            expected=(old*(old.clamp_min(1e-30).log()-torch.log_softmax(logits,-1).masked_fill(~uncached[2],0.))).sum(-1)
            expected.sum().backward();torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for k,p in m.named_parameters():
                if k in grad:torch.testing.assert_close(p.grad,grad[k],rtol=0,atol=0)
            self.assertFalse(ref.cache.old.requires_grad)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        # Cache owns the admitted inputs and does not observe later caller writes.
        saved=ref.inputs([0])[0].clone();data[0][3].state[0]+=1
        self.assertTrue(torch.equal(saved,ref.inputs([0])[0]))

    def test_fixed_contract_strict_and_no_epoch_adaptation(self):
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0580-20261023T004923397197Z/state.json')
        old=state['signature']['config'];new=P.fixed_options(old)
        from brass_ai.recovery.config import validate_config
        validate_config(new)
        self.assertEqual({k:v for k,v in old.items() if k!='value_phase'},{k:v for k,v in new.items() if k!='value_phase'})
        self.assertEqual(new,P.fixed_options(new));c=new['value_phase']
        for k,v in (('max_epochs',4),('epoch_controller',P.DEFAULT['epoch_controller']),('kl_check_steps',16),('kl_check_steps',True),('acceptance_steps',16)):
            bad=copy.deepcopy(c);bad[k]=v
            with self.assertRaises(ValueError):P.validate(bad)
        s=ppg_epochs.initial(560,c)
        for g in range(568,641,8):
            s,r=ppg_epochs.advance(s,dict(scheduled=True,epoch_cap_used=5,best_budget_step=1),c,g)
            self.assertEqual(s['cap'],5);self.assertEqual(r['decision'],'fixed');self.assertNotIn('window',s)

    def test_paired_1_5_SE_is_used_by_selection_and_zero_SE_still_rejects(self):
        from brass_ai.recovery.ppg_selection import Selection,RELAXED_DEFAULT
        before={'a':1.,'b':1.};after={'a':1.02,'b':1.22}
        self.assertTrue(A.paired(before,after)['reject'])
        self.assertFalse(A.paired(before,after,standard_errors=1.5)['reject'])
        self.assertTrue(A.paired(before,{'a':1.1,'b':1.1},standard_errors=1.5)['reject'])
        net=torch.nn.Linear(1,1);opt=torch.optim.Adam(net.parameters())
        selection=Selection(net,opt,before,config=RELAXED_DEFAULT)
        self.assertTrue(selection.consider(net,opt,after,[1,2],[],32,32))
        selection.restore(net,opt)
        self.assertEqual(selection.report()['retained_vs_minimum']['standard_errors'],1.5)

    def test_vectorized_mmap_batches_columns_order_and_duplicates(self):
        m,_,c=setup();c=P.coverage_options(c)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);phase=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
            for g in range(1,3):
                stage_only(m,phase,g)
                if g==1:write_json(run/'state.json',dict(iteration=g));phase.buffer.committed(g)
            with phase.buffer.datasets('cpu') as (ds,policy,receipts):
                n=len(ds.data.parts[0]);ids=[n+3,1,n+3,n-1,0]
                fast=ds.batch(ids)[1];slow=updates.Dataset.batch(ds,ids,value_only=True)[1]
                for a,b in zip(fast,slow):self.assertTrue(torch.equal(a,b))
                for field in ('state','mapping','descriptors','remaining','utility','targets','seat'):
                    expected=np.array([ds.data.locate(i)[0].array[ds.data.locate(i)[1]][field] for i in ids])
                    np.testing.assert_array_equal(ds.data.column(ids,field),expected)
                self.assertEqual(list(ds.microbatches(ids,3,10)),[ids[:3],ids[3:]])
                self.assertEqual(ds.utility[torch.tensor(ids),torch.tensor([0,1,2,3,0])].shape,(5,))

    def test_auxiliary_stats_deferred_without_changing_gradients(self):
        from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT
        m,_,_,(_,data)=fixture(games=2);annotate(data);ds=updates.Dataset(data,'cpu')
        a=Companion(m,copy.deepcopy(DEFAULT));a.age=7
        results=[]
        for deferred in (False,True):
            a.reset_report();a.begin('ppg');m.zero_grad();a.heads.zero_grad()
            def immediate(name,value):a.stats[name]+=float(value.detach().double())
            fn=a.accumulate if deferred else immediate
            with patch.object(a,'accumulate',fn):
                ix,inputs,_=ds.value_batch([0,2,4],m);_,loss=a.forward(m,inputs,ds,ix)
                a.account_loss(loss,ds.critic_weights[ix],3,'ppg');loss.sum().backward()
            report=a.report();results.append((report,{k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None}))
            self.assertTrue(all(not t.requires_grad and t.numel()==1 for t in a.device_stats.values()))
        for key,value in results[0][0]['training'].items():
            if not key.endswith('seconds'):self.assertEqual(value,results[1][0]['training'][key])
        self.assertTrue(_same_payload(results[0][1],results[1][1]))

    def test_32_step_guard_rolls_back_entire_interval_and_both_Adams(self):
        from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT
        m,ppo,c=setup();c=P.fixed_options(c);c['value_phase']['batch_size']=16
        c['local_scoring_auxiliary']=copy.deepcopy(DEFAULT)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);phase=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
            companion=Companion(m,c['local_scoring_auxiliary']);companion.age=7
            for g in range(1,9):
                _,data=collect(m,phase,g,branches=True);annotate(data);annotate(phase.snapshot_data)
                ds=updates.Dataset(data,'cpu');phase.prepare(ds)
                actor=[i for i,d in enumerate(data) if len(d[3].choice_ids)>1]
                phase.after_ppo(m,actor,actor[:4],{},critic_steps=1,critic_rows=len(phase.train_ids))
                phase.buffer.stage(phase.critic_dataset,phase.train_ids,phase.heldout_ids,actor,phase.collection_source,m,phase.split['selection'])
                if g<8:write_json(run/'state.json',dict(iteration=g));phase.buffer.committed(g)
            weights=copy.deepcopy(m.state_dict());adam=copy.deepcopy(phase.optimizer.state_dict());aux=companion.snapshot()
            ppo_adam=copy.deepcopy(ppo.state_dict());calls=[];probe=A.Reference.probes
            def guarded(ref,net):
                n=max((int(s['step']) for s in phase.optimizer.state.values()),default=0)
                calls.append(n);r=probe(ref,net)
                if n:r['age0']=.003
                return r
            with patch.object(A.Reference,'probes',guarded),patch.object(A,'losses',side_effect=
                lambda model,ds,ids,update:{i:1.+.1*max((int(s['step']) for s in phase.optimizer.state.values()),default=0) for i in ids}):
                report=A.fit(m,phase.optimizer,phase.buffer,phase.critic_dataset,phase.train_ids,phase.heldout_ids,
                    c['value_phase'],c['update'],beta=8.,snapshot_weight=1.,dropout=None,run_phase=True,
                    ppo_work=phase.ppo_work,epoch_cap=5,companion=companion)
            self.assertEqual(calls,[32,0]);self.assertEqual(report['attempted_steps'],32)
            self.assertEqual(report['optimizer_steps'],0);self.assertEqual(report['policy_guard_checks'],2)
            self.assertTrue(_same_payload(weights,m.state_dict()));self.assertTrue(_same_payload(adam,phase.optimizer.state_dict()))
            self.assertTrue(_same_payload(aux,companion.snapshot()));self.assertTrue(_same_payload(ppo_adam,ppo.state_dict()))

    def test_short_final_interval_is_checked_and_fixed_budget_not_minimum(self):
        m,_,c=setup();c=P.fixed_options(c);c['value_phase']['batch_size']=4096
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);phase=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
            for g in range(1,9):
                stage_only(m,phase,g)
                if g<8:write_json(run/'state.json',dict(iteration=g));phase.buffer.committed(g)
            calls=[];real=A.Reference.probes
            def guarded(ref,net):calls.append(1);return real(ref,net)
            with patch.object(A.Reference,'probes',guarded),patch.object(A,'losses',side_effect=
                lambda model,ds,ids,update:{i:1.+.1*max((int(s['step']) for s in phase.optimizer.state.values()),default=0) for i in ids}):
                r=A.fit(m,phase.optimizer,phase.buffer,phase.critic_dataset,phase.train_ids,phase.heldout_ids,
                    c['value_phase'],c['update'],beta=8.,snapshot_weight=1.,dropout=None,run_phase=True,ppo_work=phase.ppo_work)
            self.assertEqual(r['attempted_steps'],1);self.assertEqual(r['planned_steps'],5)
            self.assertEqual(r['optimizer_steps'],0);self.assertEqual(len(calls),2)
            self.assertEqual(r['early_stop_reason'],'validation_patience')

    def test_fixed_migration_keeps_committed_cache_clock_Adam_rng_and_gates(self):
        from brass_ai.recovery import critic_data
        m,ppo,old=setup();old=critic_data.fixed_config(P.coverage_options(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);phase=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
            for g in range(1,5):
                stage_only(m,phase,g);write_json(run/'state.json',dict(iteration=g));phase.buffer.committed(g)
            clock=P.initial(0)
            for g in range(1,5):clock=P.advance(clock,g,False)
            phase.state=gae.initial(old['advantage'],4);phase.critic_state['generation']=4
            phase.critic_evidence=ppo_evidence(0)
            for opt in (ppo,phase.optimizer):
                p=next(m.parameters());p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
            saved=dict(contract=phase.contract,options=phase.options,gae=phase.state,critic_data=phase.critic_state,
                critic_evidence=phase.critic_evidence,beta_clone=8.,kl_trips=[False],optimizer=phase.optimizer.state_dict(),
                schedule=clock,epoch_control=ppg_epochs.initial(0,old['value_phase']),rolling_buffer=phase.buffer.receipts,cache_origin=0)
            raw=dict(model_config=m.config.to_dict(),training={KEY:copy.deepcopy(saved)})
            before=copy.deepcopy(raw);rng=torch.get_rng_state().clone();weights=copy.deepcopy(m.state_dict())
            new=P.fixed_options(old);resumed=TrainingPhases(m,new,raw=raw,generation=4,root=SOURCE_ROOT,run=run)
            self.assertTrue(_same_payload(before,raw));self.assertTrue(_same_payload(weights,m.state_dict()))
            self.assertTrue(_same_payload(phase.optimizer.state_dict(),resumed.optimizer.state_dict()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(resumed.schedule,clock)
            self.assertEqual(resumed.buffer.receipts,phase.buffer.receipts);self.assertEqual(resumed.state,phase.state)
            self.assertEqual(resumed.critic_state,phase.critic_state);self.assertEqual(resumed.epoch_control['cap'],5)
            self.assertIsNone(resumed.startup(4));self.assertEqual(resumed.cache_origin,0)
            # The source/config transition can be retried from the same old checkpoint.
            again=TrainingPhases(m,new,raw=raw,generation=4,root=SOURCE_ROOT,run=run)
            self.assertEqual(again.schedule,resumed.schedule);self.assertEqual(again.epoch_control,resumed.epoch_control)

    def test_g580_reviewed_config_helper_is_narrow_and_leaves_run_untouched(self):
        from brass_ai.bank import sha
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0580-20261023T004923397197Z/state.json')
        target=dict(state['signature'],config=P.fixed_options(state['signature']['config']),implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,target)
        bad=copy.deepcopy(target);bad['config']['exploration']=.02
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        spec=importlib.util.spec_from_file_location('ppg_throughput_helper',SOURCE_ROOT/'scripts/configure-ppg-throughput.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0580.pt';checkpoint.write_bytes(b'authored helper checksum fixture')
            state['history'][-1]=dict(state['history'][-1],path=checkpoint.relative_to(root).as_posix(),sha256=sha(checkpoint))
            write_json(run/'state.json',state);before={p.name:sha(p) for p in run.iterdir()}
            for _ in range(2):
                result=helper.generate(root,run,'configs/step5/new.json',580)
                self.assertFalse(result['run_modified']);self.assertEqual(result['heldout_standard_errors'],1.5)
                self.assertEqual({p.name:sha(p) for p in run.iterdir()},before)
            with self.assertRaisesRegex(ValueError,'Stale'):helper.generate(root,run,'configs/step5/new.json',579)


if __name__=='__main__':unittest.main()
