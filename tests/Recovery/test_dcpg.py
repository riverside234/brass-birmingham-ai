"""Delayed critic objectives, split checks, immutable targets and continuation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import dcpg_config as D,ppg_config as P,ppg_phasic as F,ppg_auxiliary as A
from brass_ai.recovery import value_preservation as V,updates,gae,ppg_epochs,value_phase,critic_data
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_ppg_eight import setup,stage_only,collect
from test_gae_value_phase import fixture,ppo
from test_local_scoring import annotate

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0590-20261023T035130866327Z'


class Delayed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_config_narrow_strict_defaults_and_fixed_dynamic_rate(self):
        from brass_ai.recovery.config import validate_config,latest_model_config
        old=read(EXPORT/'state.json')['signature']['config'];new=D.configure(old)
        validate_config(new);self.assertEqual(D.configure(new),new)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'update','value_phase'})
        self.assertEqual({k:v for k,v in new['update'].items() if k!='value_gradient'},
            {k:v for k,v in old['update'].items() if k!='value_gradient'})
        self.assertEqual(latest_model_config(old,old['imitation_checkpoint'])['value_phase'],D.fast_options(D.sixteen_options(D.beta_options(new)))['value_phase'])
        c=new['value_phase'];self.assertEqual((c['phase_interval'],c['buffer_iterations'],c['validation_iterations']),(8,8,8))
        for rate in (3e-6,6.25e-6,1.8e-5):
            receipt=D.rate_receipt(rate,591,48)
            for s in (1,16,32,499,500):self.assertEqual(P.phase_rate(c,s,500,peak=receipt['peak_lr']),rate*.35)
        for key,value in [('phase_interval',4),('acceptance_steps',32),('kl_check_steps',16),('learning_rate_factor',1.),('max_epochs',2)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):P.validate(bad)
        bad=copy.deepcopy(new);bad['value_phase']['enabled']=False
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(new);bad['update']['value_gradient']['beta']=True
        with self.assertRaises(ValueError):validate_config(bad)

    def test_g590_cadence_preserves_origin_phase_counts_and_restart(self):
        report=read(EXPORT/'iteration-0589/report.json')
        old=dict(report['value_phase']['schedule']);old.pop('next_phase')
        new=D.migrate_clock(old,590)
        for k in ('origin','last_phase','phases','generation'):self.assertEqual(new[k],old[k])
        self.assertEqual(P.next_phase(new),596);seen=[]
        for g in range(591,611):
            new=json.loads(json.dumps(new));due=P.due(new)
            if due:seen.append(g)
            new=P.advance(new,g,due)
        self.assertEqual(seen,[596,604]);self.assertEqual(D.migrate_clock(new,610),new)
        fresh=P.initial(0,D.configure(setup()[2])['value_phase']);seen=[]
        for g in range(1,25):
            if P.due(fresh):seen.append(g)
            fresh=P.advance(fresh,g,P.due(fresh))
        self.assertEqual(seen,[8,16,24])
        with self.assertRaises(ValueError):P.schedule(dict(new,phases=new['phases']+1),610)

    def test_frozen_true_outputs_zero_loss_no_targets_and_gradient_paths(self):
        from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT
        model,_,cfg,(_,rows)=fixture(games=2);annotate(rows)
        aux=Companion(model,copy.deepcopy(DEFAULT));aux.age=8
        with torch.no_grad():
            model.outcomes[0].weight[:,-32:].add_(.01)
            model.utility_head[0].weight[:,-32:].add_(.01)
            model.utility_head[-1].weight.add_(.01)
            for p in aux.heads.parameters():p.add_(.01)
        ds=updates.Dataset(rows,'cpu');rng=torch.get_rng_state().clone()
        ref=V.FrozenValues(model,ds,cfg['update'],aux)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(ref.values),{'win','rank','vp','utility','slots_owner','slots_vp','links_owner','links_vp'})
        self.assertTrue(all(not v.requires_grad and v.device.type=='cpu' for v in ref.values.values()))
        ids=[5,0,5,3];ix,inputs,_=ds.value_batch(ids,model)
        out=updates.value_output(model,inputs,ds,ix)
        for k,v in zip(('win','rank','vp','utility'),out[1:]):torch.testing.assert_close(ref.selected(ix,'cpu')[k],v,rtol=1e-6,atol=5e-5 if k=='vp' else 1e-6)
        losses=ref.losses(model,inputs,ix)
        for v in losses.values():self.assertLess(float(v.detach().abs().max()),1e-6)
        frozen=copy.deepcopy(ref.values)
        ds.utility.fill_(500);[t.fill_(-999) for t in ds.targets]
        for row in ds.rows:row.pop('_local_scoring',None)
        self.assertTrue(_same_payload(losses,ref.losses(model,inputs,ix)))
        with torch.no_grad():
            for p in model.state_encoder.parameters():p.add_(.002)
        model.zero_grad(set_to_none=True);aux.heads.zero_grad(set_to_none=True)
        v=ref.losses(model,inputs,ix);(.5*v['core']+v['local']).mean().backward()
        for module in (model.state_encoder,model.entity_encoder,model.descriptor_encoder,model.outcomes,model.utility_head,aux.heads):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertTrue(_same_payload(frozen,ref.values))
        self.assertTrue(all(p.grad is None for p in model.policy_scorer.parameters()))

    def test_soft_win_KL_and_existing_regression_scales(self):
        old=dict(win=torch.tensor([[.1,.2,.3,.4]]),rank=torch.tensor([[1.,2.,3.,4.]]),
            vp=torch.tensor([[20.,30.,40.,50.]]),utility=torch.zeros(1,4))
        logits=torch.tensor([[.1,.4,.3,.2]],requires_grad=True)
        current=dict(old,win=logits.softmax(-1),rank=old['rank']+.3,vp=old['vp']+10,utility=torch.ones(1,4)*.2)
        loss=V.divergences(current,old)
        grad=torch.autograd.grad(loss['win'].sum(),logits,retain_graph=True)[0]
        ce=-(old['win']*logits.log_softmax(-1)).sum()
        torch.testing.assert_close(grad,torch.autograd.grad(ce,logits)[0])
        self.assertAlmostEqual(float(loss['rank']),.005,places=6)
        self.assertAlmostEqual(float(loss['vp']),.005,places=6)
        self.assertAlmostEqual(float(loss['utility']),.04,places=6)

    def test_gradient_telemetry_is_exact_and_does_not_change_gradients(self):
        m,_,cfg,(_,rows)=fixture(games=2);ds=updates.Dataset(rows,'cpu')
        ix,inputs,_=ds.value_batch(range(6),m);ref=V.FrozenValues(m,ds,cfg['update'])
        with torch.no_grad():next(m.state_encoder.parameters()).add_(.001)
        m.zero_grad(set_to_none=True);meter=V.TrunkGradients(m)
        actor=m(*inputs)[0][:,0].mean();actor_grad=torch.autograd.grad(actor,meter.parameters,retain_graph=True,allow_unused=True)
        meter.add_actor(actor);actor.backward();meter.before_value()
        loss=.5*ref.losses(m,inputs,ix)['core'].mean()
        value_grad=torch.autograd.grad(loss,meter.parameters,retain_graph=True,allow_unused=True)
        loss.backward()
        r=meter.finish(1)
        flat=lambda gs:torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(meter.parameters,gs)])
        a,v=flat(actor_grad),flat(value_grad)
        # Per-parameter and concatenated float32 reductions need not have the
        # same last bits; compare relative error rather than decimal places.
        torch.testing.assert_close(torch.tensor(r['actor_norm']),a.norm(),rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(torch.tensor(r['preservation_norm']),v.norm(),rtol=1e-5,atol=1e-7)
        self.assertAlmostEqual(r['cosine'],float(torch.dot(a,v)/(a.norm()*v.norm())),places=4)
        for p,a_grad,v_grad in zip(meter.parameters,actor_grad,value_grad):
            expected=(torch.zeros_like(p) if a_grad is None else a_grad)+(torch.zeros_like(p) if v_grad is None else v_grad)
            torch.testing.assert_close(p.grad,expected,rtol=1e-6,atol=1e-7)

    def test_ppo_has_no_fresh_fitting_uses_frozen_GAE_and_actual_lr(self):
        from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT
        from brass_ai.recovery import utility
        from brass_ai.recovery.learning_rate import KEY as WARMUP
        from brass_ai.recovery.lr_controller import KEY as CONTROLLER
        model,opt,cfg=setup();cfg=D.configure(cfg)
        opt.param_groups[0][WARMUP]['completed_steps']=100;opt.param_groups[0][CONTROLLER]['multiplier']=1.5
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=Path(tmp))
            phase.local_scoring=Companion(model,copy.deepcopy(DEFAULT));phase.local_scoring.age=8
            b,rows=collect(model,phase,1,branches=True);annotate(rows);annotate(phase.snapshot_data)
            expected={};original=phase.advantages
            def advantages(mc,actor):
                ds=phase.dataset;ix=torch.arange(len(ds.data));values=[]
                with torch.no_grad():
                    for start in range(0,len(ds.data),64):
                        ids=list(range(start,min(start+64,len(ds.data))));idx,inputs,_=ds.value_batch(ids,model)
                        values.append(updates.value_output(model,inputs,ds,idx)[4])
                baseline=torch.cat(values)
                torch.testing.assert_close(mc,(ds.utility-baseline)[ix,ds.seats],rtol=1e-6,atol=1e-6)
                expected['verified']=True;return original(mc,actor)
            with (patch.object(utility,'calibrate',side_effect=AssertionError('fresh calibration')),
                 patch.object(updates,'value_losses',side_effect=AssertionError('fresh target loss')),
                 patch.object(phase.local_scoring,'forward',side_effect=AssertionError('fresh local targets')),
                 patch.object(phase,'advantages',side_effect=advantages)):
                r=ppo(model,opt,cfg,b,rows,phase)
            self.assertTrue(expected['verified']);v=r['value_preservation']
            self.assertEqual(v['frozen_rows'],len(rows)+len(phase.snapshot_data))
            self.assertGreater(v['frozen_by_role']['snapshot'],0);self.assertGreater(r['forced_decisions'],0)
            self.assertEqual(len(v['gradients']),r['optimizer_steps']);self.assertEqual(v['fresh_target_steps'],0)
            self.assertEqual(r['utility_calibration']['steps'],0)
            self.assertEqual(phase.ppo_peak['peak_lr'],r['learning_rates'][0]*.35)
            opt.param_groups[0]['lr']=1e-9
            value=phase.fit(model)
            self.assertEqual(value['peak_learning_rate'],r['learning_rates'][0]*.35)
            self.assertEqual(value_phase.critic_work(value)['optimizer_steps'],0)
            self.assertEqual(value_phase.critic_work(value)['source'],'ppg')
            self.assertIn('value_sag',value)

    def test_migration_preserves_Adams_rng_gates_cached_targets_and_repeat_resume(self):
        model,ppo_opt,old=setup();old=critic_data.fixed_config(F.configure(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            for g in range(1,5):
                stage_only(model,p,g);write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            for opt in (ppo_opt,p.optimizer):
                param=next(model.parameters());param.grad=torch.ones_like(param);opt.step();opt.zero_grad(set_to_none=True)
            p.state=gae.initial(old['advantage'],4);p.critic_state['generation']=4
            clock=P.initial(0,old['value_phase'])
            for g in range(1,5):clock=P.advance(clock,g,False)
            saved=dict(contract=p.contract,options=p.options,gae=p.state,critic_data=p.critic_state,
                critic_evidence=p.critic_evidence,beta_clone=8.,kl_trips=[False],optimizer=p.optimizer.state_dict(),
                schedule=clock,epoch_control=ppg_epochs.initial(0,old['value_phase']),rolling_buffer=p.buffer.receipts,cache_origin=0)
            raw=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(saved)})
            raw_before=copy.deepcopy(raw);weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(ppo_opt.state_dict());rng=torch.get_rng_state().clone()
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            new=D.configure(old);q=TrainingPhases(model,new,raw=raw,generation=4,root=SOURCE_ROOT,run=run)
            self.assertTrue(_same_payload(raw,raw_before));self.assertTrue(_same_payload(weights,model.state_dict()))
            self.assertTrue(_same_payload(adam,ppo_opt.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(_same_payload(q.optimizer.state_dict(),saved['optimizer']))
            self.assertEqual(q.state,saved['gae']);self.assertEqual(q.critic_state,saved['critic_data'])
            self.assertEqual(q.buffer.receipts,saved['rolling_buffer']);self.assertEqual(q.beta,8.)
            self.assertEqual(q.trips,[False]);self.assertEqual(P.next_phase(q.schedule),8)
            self.assertIsNone(q.startup(4));self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            migrated=dict(saved,options=q.options,schedule=q.schedule)
            again=TrainingPhases(model,new,raw=dict(training={KEY:migrated}),generation=4,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.schedule,again.schedule);self.assertIsNone(again.rolling_migration)

    def split_fit(self,unsafe_selected=False):
        model,ppo_opt,cfg=setup();cfg=D.configure(cfg);cfg['value_phase']['batch_size']=16
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            for g in range(1,9):
                stage_only(model,p,g)
                if g<8:write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            initial=copy.deepcopy(model.state_dict());ppo_adam=copy.deepcopy(ppo_opt.state_dict())
            checks=[];loss_checks=[];saved={};real_step=p.optimizer.step;real_probe=A.Reference.probes
            def steps():return max((int(s['step']) for s in p.optimizer.state.values()),default=0)
            def step(*args,**kwargs):
                result=real_step(*args,**kwargs)
                if steps() in (16,32):saved[steps()]=(copy.deepcopy(model.state_dict()),copy.deepcopy(p.optimizer.state_dict()))
                return result
            def losses(net,ds,ids,update):
                n=steps();loss_checks.append(n)
                # Fail at 32 to select an unchecked 16-step state; otherwise
                # exercise the ordinary 16/32/48 loss and 32-step KL cadence.
                base=(1. if n==0 else .9 if n<32 else 1.1) if unsafe_selected else (
                    1. if n==0 else .9 if n<32 else .89 if n<48 else 1.1)
                return {i:base for i in ids}
            def probes(ref,net):
                n=steps();checks.append(n);r=real_probe(ref,net)
                r['overall']=0.;r['age0']=.003 if unsafe_selected and n==16 else 0.
                return r
            with patch.object(p.optimizer,'step',step),patch.object(A,'losses',losses),patch.object(A.Reference,'probes',probes):
                r=A.fit(model,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                    cfg['value_phase'],cfg['update'],beta=8.,snapshot_weight=1.,dropout=None,run_phase=True,
                    ppo_work=p.ppo_work,epoch_cap=5,ppo_peak=D.rate_receipt(6.25e-6,8,48))
            self.assertEqual(r['acceptance_generations'],list(range(8,0,-1)))
            self.assertEqual(r['applied_lr_min'],.35*6.25e-6);self.assertEqual(r['applied_lr_min'],r['applied_lr_max'])
            self.assertEqual(r['scheduled_policy_checks'],1)
            self.assertTrue(_same_payload(ppo_adam,ppo_opt.state_dict()))
            if unsafe_selected:
                self.assertEqual(checks,[32,16,0]);self.assertEqual(r['rejected_unverified_selections'],1)
                self.assertEqual(r['optimizer_steps'],0);self.assertTrue(_same_payload(initial,model.state_dict()))
                self.assertEqual(r['age0_probe_kl'],0.);self.assertTrue(r['kl_guard_tripped'])
            else:
                self.assertEqual(checks,[32,32,32]);self.assertEqual(r['heldout_checks'],3)
                self.assertEqual(r['attempted_steps'],48);self.assertEqual(r['optimizer_steps'],32)
                self.assertTrue(_same_payload(model.state_dict(),saved[32][0]))
                self.assertTrue(_same_payload(p.optimizer.state_dict(),saved[32][1]))
            return r

    def test_loss_every_16_KL_every_32_and_constant_rate(self):self.split_fit()

    def test_unchecked_16_step_selection_never_bypasses_KL(self):self.split_fit(True)

    def test_g590_config_helper_narrow_review_and_no_run_writes(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        state=read(EXPORT/'state.json');target=dict(state['signature'],config=D.configure(state['signature']['config']),implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        bad=copy.deepcopy(target);bad['config']['exploration']=.02
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        spec=importlib.util.spec_from_file_location('dcpg_helper',SOURCE_ROOT/'scripts/configure-dcpg.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True)
            ck=run/'generation-0590.pt';ck.write_bytes(b'authored checkpoint checksum fixture')
            state['history'][-1]=dict(state['history'][-1],path=ck.relative_to(root).as_posix(),sha256=sha(ck))
            write_json(run/'state.json',state);before={p.name:sha(p) for p in run.iterdir()}
            for _ in range(2):
                result=helper.generate(root,run,'configs/step5/new.json',590)
                self.assertFalse(result['run_modified']);self.assertEqual(result['phase_interval'],8)
                self.assertEqual(before,{p.name:sha(p) for p in run.iterdir()})
            with self.assertRaisesRegex(ValueError,'Stale'):helper.generate(root,run,'configs/step5/new.json',589)


if __name__=='__main__':unittest.main()
