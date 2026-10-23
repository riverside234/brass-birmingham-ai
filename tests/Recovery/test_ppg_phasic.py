"""Four-iteration cadence, dynamic peaks, stop-gradients and exact continuation."""
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
from brass_ai.recovery import ppg_config as P,ppg_phasic as F,ppg_auxiliary as A,ppg_epochs,updates,gae,critic_data
from brass_ai.recovery.actor_value import TrainingPhases,KEY,ppo_evidence
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_ppg_eight import setup,stage_only
from test_gae_value_phase import fixture
from test_local_scoring import annotate

RESULT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0580-20261023T004923397197Z'


class Phasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_config_defaults_strict_and_only_requested_changes(self):
        from brass_ai.recovery.config import validate_config,latest_model_config
        state=read(RESULT/'state.json');old=state['signature']['config'];new=F.configure(old)
        validate_config(new);self.assertEqual(F.configure(new),new)
        for k,v in old.items():
            if k not in ('update','value_phase'):self.assertEqual(v,new[k])
        self.assertEqual({k:v for k,v in new['update'].items() if k!='value_gradient'},old['update'])
        c=new['value_phase'];self.assertEqual((c['phase_interval'],c['buffer_iterations'],c['validation_iterations']),(4,8,8))
        self.assertEqual(c['acceptance_steps'],32);self.assertEqual(c['max_epochs'],5)
        self.assertEqual(c['selection']['standard_errors'],1.5);self.assertIsNone(c['learning_rate'])
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(fresh['value_phase'],new['value_phase']);self.assertEqual(fresh['update']['value_gradient'],F.GRADIENT)
        for key,value in [('phase_interval',8),('phase_interval',True),('buffer_iterations',4),('validation_iterations',4),
                          ('learning_rate',3e-6),('learning_rate_source','configured-base')]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):P.validate(bad)
        bad=copy.deepcopy(new);bad['update'].pop('value_gradient')
        with self.assertRaises(ValueError):validate_config(bad)

    def test_fresh_schedule_waits_for_eight_ages_then_every_four(self):
        c=F.options({'value_phase':P.fixed_options({'value_phase':dict(copy.deepcopy(P.DEFAULT),enabled=True)})['value_phase']})['value_phase']
        s=P.initial(0,c);ran=[]
        for g in range(1,21):
            self.assertEqual(P.due(s),P.due(json.loads(json.dumps(s))))
            if P.due(s):ran.append(g)
            s=P.advance(s,g,P.due(s))
        self.assertEqual(ran,[8,12,16,20]);self.assertEqual(s['phases'],4)
        with self.assertRaises(ValueError):P.advance(s,22,False)

    def test_g580_clock_preserves_completed_phases_and_restarts_without_skips(self):
        old=dict(contract=P.MIGRATION,origin=560,generation=580,last_phase=576,phases=2)
        s=F.migrate_clock(old,580);self.assertEqual(s['origin'],560);self.assertEqual(s['last_phase'],576)
        self.assertEqual(s['phases'],2);self.assertEqual(P.next_phase(s),584)
        ran=[]
        for g in range(581,601):
            if P.due(s):ran.append(g)
            s=P.advance(json.loads(json.dumps(s)),g,P.due(s))
        self.assertEqual(ran,[584,588,592,596,600]);self.assertEqual(s['phases'],7)
        self.assertEqual(F.migrate_clock(s,600),s)
        bad=copy.deepcopy(s);bad['phases']+=1
        with self.assertRaises(ValueError):P.schedule(bad,600)
        for generation in (577,583):
            start=dict(old,generation=generation)
            self.assertEqual(P.next_phase(F.migrate_clock(start,generation)),min(generation+4,584))

    def test_shuffle_is_full_uniform_and_new_each_epoch_with_rng_replay(self):
        _,_,cfg=setup();cfg=F.configure(cfg);train={a:list(range(100*a,100*a+50+a)) for a in range(8)}
        rng=torch.get_rng_state().clone()
        first=A.epoch_order(train,cfg['value_phase']);second=A.epoch_order(train,cfg['value_phase'])
        expected=sorted(i for group in train.values() for i in group)
        self.assertEqual(sorted(first[0]),expected);self.assertEqual(sorted(second[0]),expected)
        self.assertNotEqual(first[0],second[0]);self.assertEqual(len(set(first[0])),len(expected))
        torch.set_rng_state(rng);self.assertEqual(A.epoch_order(train,cfg['value_phase']),first)

    def test_linked_rate_uses_ppo_peak_with_unchanged_curve_and_retry_positions(self):
        _,_,cfg=setup();c=F.configure(cfg)['value_phase']
        for peak in (3e-6,6.25e-6,1.8e-5):
            rates=[P.phase_rate(c,s,500,peak=peak) for s in range(1,501)]
            self.assertAlmostEqual(rates[0],.1*peak);self.assertAlmostEqual(rates[31],peak)
            self.assertAlmostEqual(rates[-1],.3*peak)
            self.assertEqual(P.phase_rate(c,53,500,peak=peak),rates[52])
        with self.assertRaises(ValueError):P.phase_rate(c,1,500)
        with self.assertRaises(ValueError):P.phase_rate(c,1,500,peak=float('nan'))

    def test_peak_capture_precedes_next_lr_decision_and_no_steps_have_no_peak(self):
        from brass_ai.recovery.learning_rate import KEY as WARMUP
        from brass_ai.recovery.lr_controller import KEY as CONTROLLER
        from test_ppg_eight import collect
        from test_gae_value_phase import ppo
        model,opt,cfg=setup();cfg=F.configure(cfg)
        opt.param_groups[0][WARMUP]['completed_steps']=100
        opt.param_groups[0][CONTROLLER]['multiplier']=1.5
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=Path(tmp))
            behavior,data=collect(model,phase,1)
            report=ppo(model,opt,cfg,behavior,data,phase)
            used=report['learning_rates'][0]
            self.assertEqual(used,1.5*cfg['learning_rate'])
            # A post-PPO controller cut may alter the group before phase.fit.
            # The phase must retain the actual PPO rate, not reread that group.
            opt.param_groups[0][CONTROLLER]['multiplier']=.5
            opt.param_groups[0]['lr']=.5*cfg['learning_rate']
            value=phase.fit(model)
            self.assertEqual(value['peak_learning_rate'],used)
            self.assertEqual(value['learning_rate_source']['ppo_steps'],report['optimizer_steps'])
            phase.after_ppo(model,[],[],{},ppo_rate=None,ppo_steps=0)
            self.assertIsNone(phase.ppo_peak)
            self.assertEqual(phase.optimizer.param_groups[0][F.RATE_KEY]['source'],'initial-ppo-base')

    def test_value_stop_gradient_preserves_predictions_trains_heads_and_restores_ppg_path(self):
        model,_,_,(_,rows)=fixture(games=2);ds=updates.Dataset(rows,'cpu')
        ix,inputs,_=ds.value_batch(range(8),model)
        full=updates.value_output(model,inputs,ds,ix)
        detached=updates.value_output(model,inputs,ds,ix,detach=True)
        for a,b in zip(full[1:],detached[1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        def loss(out):return (updates.value_losses(out,tuple(t[ix] for t in ds.targets))[0]+(out[4]-ds.utility[ix]).square().mean(-1)).mean()
        model.zero_grad(set_to_none=True);loss(detached).backward()
        for name,param in model.named_parameters():
            if name.startswith(('state_encoder.','entity_encoder.','entity_projections.','action_encoder.','policy_scorer.')):
                self.assertIsNone(param.grad,name)
        for module in (model.outcomes,model.utility_head):self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        model.zero_grad(set_to_none=True);loss(updates.value_output(model,inputs,ds,ix)).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.state_encoder.parameters()))
        model.zero_grad(set_to_none=True);model(*inputs)[0][:,0].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.state_encoder.parameters()))

    def test_companion_does_not_bypass_detach_and_keeps_ppg_gradients(self):
        from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT
        model,_,_,(_,rows)=fixture(games=2);annotate(rows);ds=updates.Dataset(rows,'cpu')
        ix,inputs,_=ds.value_batch(range(8),model);aux=Companion(model,copy.deepcopy(DEFAULT));aux.age=8
        with torch.no_grad():
            for p in aux.heads.parameters():p.add_(.01)
        for mode in ('apply','shadow','off'):
            aux.config['mode']=mode;aux.active=mode!='off'
            model.zero_grad(set_to_none=True);aux.begin('ppo')
            out,local=aux.forward(model,inputs,ds,ix,detach=True)
            loss=updates.value_losses(out,tuple(t[ix] for t in ds.targets))[0]+(out[4]-ds.utility[ix]).square().mean(-1)
            if local is not None:loss=loss+local
            loss.mean().backward()
            self.assertTrue(all(p.grad is None for p in model.state_encoder.parameters()),mode)
            self.assertTrue(all(p.grad is None for p in model.entity_encoder.parameters()),mode)
        aux.config['mode']='apply';aux.active=True;aux.begin('ppg');model.zero_grad(set_to_none=True)
        _,local=aux.forward(model,inputs,ds,ix);local.mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.state_encoder.parameters()))

    def test_migration_preserves_cache_Adam_RNG_gates_and_is_idempotent(self):
        model,ppo,old=setup();old=critic_data.fixed_config(P.coverage_options(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);phase=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            for g in range(1,5):
                stage_only(model,phase,g);write_json(run/'state.json',dict(iteration=g));phase.buffer.committed(g)
            clock=P.initial(0)
            for g in range(1,5):clock=P.advance(clock,g,False)
            phase.state=gae.initial(old['advantage'],4);phase.critic_state['generation']=4
            for opt in (ppo,phase.optimizer):
                p=next(model.parameters());p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
            saved=dict(contract=phase.contract,options=phase.options,gae=phase.state,critic_data=phase.critic_state,
                critic_evidence=ppo_evidence(0),beta_clone=8.,kl_trips=[False],optimizer=phase.optimizer.state_dict(),
                schedule=clock,epoch_control=ppg_epochs.initial(0,old['value_phase']),rolling_buffer=phase.buffer.receipts,cache_origin=0)
            raw=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(saved)})
            before=copy.deepcopy(raw);weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(ppo.state_dict());rng=torch.get_rng_state().clone()
            files={p.name:sha(p) for p in phase.buffer.directory.iterdir()}
            new=F.configure(old);resumed=TrainingPhases(model,new,raw=raw,generation=4,root=SOURCE_ROOT,run=run)
            self.assertTrue(_same_payload(raw,before));self.assertTrue(_same_payload(weights,model.state_dict()))
            self.assertTrue(_same_payload(adam,ppo.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            actual=copy.deepcopy(resumed.optimizer.state_dict());actual['param_groups'][0].pop(F.RATE_KEY)
            self.assertTrue(_same_payload(actual,saved['optimizer']))
            self.assertEqual(resumed.state,saved['gae']);self.assertEqual(resumed.critic_state,saved['critic_data'])
            self.assertEqual(resumed.buffer.receipts,saved['rolling_buffer']);self.assertEqual(resumed.cache_origin,0)
            self.assertEqual(resumed.beta,8.);self.assertEqual(resumed.trips,[False]);self.assertEqual(P.next_phase(resumed.schedule),8)
            self.assertIsNone(resumed.startup(4));self.assertEqual(files,{p.name:sha(p) for p in phase.buffer.directory.iterdir()})
            again=TrainingPhases(model,new,raw=raw,generation=4,root=SOURCE_ROOT,run=run)
            self.assertEqual(again.schedule,resumed.schedule);self.assertTrue(_same_payload(again.optimizer.state_dict(),resumed.optimizer.state_dict()))
            # After the first new checkpoint, never repeat migration or restart cadence.
            migrated=dict(saved,options=resumed.options,schedule=resumed.schedule,epoch_control=resumed.epoch_control,optimizer=resumed.optimizer.state_dict())
            final=TrainingPhases(model,new,raw=dict(training={KEY:migrated}),generation=4,root=SOURCE_ROOT,run=run)
            self.assertEqual(final.schedule,resumed.schedule);self.assertIsNone(final.rolling_migration)

    def test_g580_helper_reviews_narrow_transition_and_does_not_edit_run(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        state=read(RESULT/'state.json');target=dict(state['signature'],config=F.configure(state['signature']['config']),implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        bad=copy.deepcopy(target);bad['config']['exploration']=.02
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        spec=importlib.util.spec_from_file_location('phasic_helper',SOURCE_ROOT/'scripts/configure-ppg-phasic.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True)
            ck=run/'generation-0580.pt';ck.write_bytes(b'authored checksum fixture')
            state['history'][-1]=dict(state['history'][-1],path=ck.relative_to(root).as_posix(),sha256=sha(ck));write_json(run/'state.json',state)
            before={p.name:sha(p) for p in run.iterdir()}
            for _ in range(2):
                result=helper.generate(root,run,'configs/step5/new.json',580)
                self.assertFalse(result['run_modified']);self.assertEqual(result['phase_interval'],4)
                self.assertEqual(before,{p.name:sha(p) for p in run.iterdir()})
            with self.assertRaisesRegex(ValueError,'Stale'):helper.generate(root,run,'configs/step5/new.json',579)


if __name__=='__main__':unittest.main()
