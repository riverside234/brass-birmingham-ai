"""Current-run beta reset, bounded decreases, and restart-safe feedback."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import dcpg_config as D,ppg_config as P,ppg_epochs,gae,critic_data
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_ppg_eight import setup,stage_only

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0600-20261023T061313188381Z'


class Feedback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_thresholds_bounds_no_waiting_or_rejected_phase_decrease(self):
        c=D.beta_options(D.configure(setup()[2]))['value_phase']
        base=dict(beta_clone_final=8.,scheduled=True,optimizer_steps=32,
            kl_guard_tripped=False,beta_retries=0,probe_kl=.0005,age0_probe_kl=.0009)
        self.assertEqual(D.beta_after_phase(base,c)[0],4.)
        for changes in (dict(scheduled=False),dict(optimizer_steps=0),dict(kl_guard_tripped=True),
                        dict(beta_retries=1),dict(probe_kl=.001),dict(age0_probe_kl=.001),
                        dict(probe_kl=.0015),dict(age0_probe_kl=.0015)):
            self.assertEqual(D.beta_after_phase(dict(base,**changes),c)[0],8.)
        for beta,expected in ((1.,1.),(1.5,1.),(2.,1.),(32.,16.)):
            self.assertEqual(D.beta_after_phase(dict(base,beta_clone_final=beta),c)[0],expected)
        for invalid in (None,float('nan'),-.01,.003):
            with self.assertRaises(ValueError):D.beta_after_phase(dict(base,age0_probe_kl=invalid),c)

    def test_config_review_and_g600_helper_leave_checkpoint_unchanged(self):
        from brass_ai.recovery.config import validate_config,latest_model_config
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        s=read(EXPORT/'state.json');old=s['signature']['config'];new=D.beta_options(old)
        validate_config(new);self.assertEqual(D.beta_options(new),new)
        self.assertEqual({k for k in new if new[k]!=old[k]},{'value_phase'})
        self.assertEqual({k for k in new['value_phase'] if new['value_phase'][k]!=old['value_phase'].get(k)},
            {'contract','beta_control'})
        self.assertEqual(latest_model_config(old,old['imitation_checkpoint'])['value_phase'],new['value_phase'])
        target=dict(s['signature'],config=new,implementation=implementation_id())
        self.assertTrue(check_signature(s,target,reviewed=True))
        with self.assertRaises(ValueError):check_signature(s,target)
        bad=copy.deepcopy(target);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(s,bad,reviewed=True)
        for changes in (dict(beta_clone=2.),dict(beta_clone_max=1.),dict(beta_control={})):
            with self.assertRaises(ValueError):P.validate(dict(new['value_phase'],**changes))
        report=read(EXPORT/'iteration-0599/report.json');clock=dict(report['value_phase']['schedule']);clock.pop('next_phase')
        self.assertEqual(report['value_phase']['beta_clone_next'],8.)
        self.assertEqual(P.next_phase(clock),604)
        spec=importlib.util.spec_from_file_location('beta_helper',SOURCE_ROOT/'scripts/configure-dcpg.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True)
            ck=run/'generation-0600.pt';ck.write_bytes(b'checksum fixture')
            s['history'][-1]=dict(s['history'][-1],path=ck.relative_to(root).as_posix(),sha256=sha(ck))
            write_json(run/'state.json',s);before={p.name:sha(p) for p in run.iterdir()}
            for _ in range(2):
                result=helper.generate(root,run,'configs/step5/beta1.json',600,beta_feedback=True)
                self.assertEqual(result['beta_mode'],'adaptive-up-down')
                self.assertFalse(result['run_modified'])
                self.assertEqual(read(root/'configs/step5/beta1.json'),new)
                self.assertEqual(before,{p.name:sha(p) for p in run.iterdir()})
            with self.assertRaisesRegex(ValueError,'Stale'):
                helper.generate(root,run,'configs/step5/beta1.json',599,beta_feedback=True)

    def test_resume_resets_effective_beta_once_and_preserves_state(self):
        model,ppo_opt,old=setup();old=critic_data.fixed_config(D.configure(old))
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
                critic_evidence=p.critic_evidence,beta_clone=8.,kl_trips=[True,False],optimizer=p.optimizer.state_dict(),
                schedule=clock,epoch_control=ppg_epochs.initial(0,old['value_phase']),rolling_buffer=p.buffer.receipts,cache_origin=0)
            raw=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(saved)})
            before=copy.deepcopy(raw);weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(ppo_opt.state_dict())
            rng=torch.get_rng_state().clone();hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            new=D.beta_options(old)
            for _ in range(2):
                q=TrainingPhases(model,new,raw=raw,generation=4,root=SOURCE_ROOT,run=run)
                self.assertEqual(q.beta,1.);self.assertEqual(q.rolling_migration['beta_before'],8.)
                self.assertEqual(q.schedule,clock);self.assertEqual(q.trips,[True,False])
                self.assertEqual(q.buffer.receipts,saved['rolling_buffer'])
                self.assertEqual(q.state,saved['gae']);self.assertEqual(q.critic_state,saved['critic_data'])
                self.assertTrue(_same_payload(q.optimizer.state_dict(),saved['optimizer']))
                self.assertIsNone(q.startup(4))
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            self.assertTrue(_same_payload(raw,before));self.assertTrue(_same_payload(weights,model.state_dict()))
            self.assertTrue(_same_payload(adam,ppo_opt.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            # A later adapted beta must survive an ordinary restart unchanged.
            migrated=dict(saved,options=q.options,beta_clone=4.)
            again=TrainingPhases(model,new,raw=dict(training={KEY:migrated}),generation=4,root=SOURCE_ROOT,run=run)
            self.assertEqual(again.beta,4.);self.assertIsNone(again.rolling_migration)
            again.value_report=dict(beta_clone_final=4.,scheduled=True,optimizer_steps=32,
                kl_guard_tripped=False,beta_retries=0,probe_kl=.0002,age0_probe_kl=.0008,early_stop_reason=None)
            again.critic_state=None
            with patch('brass_ai.recovery.gae.advance',return_value=(dict(again.state,generation=5),{})):
                again.finish(generation=5,update=dict(categorical_probe=dict(categorical_kl=.001),
                    controller_warmup_done=True,optimizer_steps=1,kl_early_stop=False,gae={}),anchor_gate=dict(ok=True),rating={})
            self.assertEqual(again.beta,2.);self.assertEqual(again.value_report['beta_decision'],'down')
            self.assertEqual(again.value_report['beta_clone_final'],4.)

    def test_isolated_phase_keeps_KL_retries_and_safe_retention(self):
        # Run the real auxiliary optimizer and existing rollback regression
        # under the new beta contract, starting from 1 rather than 8.
        from test_dcpg import Delayed
        from brass_ai.recovery import ppg_auxiliary as A
        original_configure=D.configure;original_fit=A.fit
        reports=[]
        def configure(c):return D.beta_options(original_configure(c))
        def fit(*args,**kwargs):
            kwargs['beta']=1.;r=original_fit(*args,**kwargs);reports.append(r);return r
        with patch.object(D,'configure',configure),patch.object(A,'fit',fit):
            Delayed().split_fit(unsafe_selected=True)
        self.assertEqual(reports[0]['beta_mode'],'adaptive-up-down')
        self.assertEqual(reports[0]['beta_clone_final'],1.)
        self.assertTrue(reports[0]['kl_guard_tripped'])
        self.assertEqual(D.beta_after_phase(reports[0],D.beta_options(original_configure(setup()[2]))['value_phase'])[0],1.)

    def test_new_contract_still_raises_beta_on_KL_trip(self):
        from brass_ai.recovery import ppg_auxiliary as A
        model,_,cfg=setup();cfg=D.beta_options(D.configure(cfg));cfg['value_phase']['batch_size']=128
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            for g in range(1,9):
                stage_only(model,p,g)
                if g<8:write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            checks=[];real_probe=A.Reference.probes
            def steps():return max((int(s['step']) for s in p.optimizer.state.values()),default=0)
            def losses(net,ds,ids,update):return {i:1.-steps()*.001 for i in ids}
            def probes(ref,net):
                result=real_probe(ref,net);checks.append(steps())
                # One injected cap hit with improving value loss, then normal
                # exact complete-policy measurements for the bounded retry.
                if len(checks)==1:result['age0']=.003
                return result
            with patch.object(A,'losses',losses),patch.object(A.Reference,'probes',probes):
                r=A.fit(model,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                    cfg['value_phase'],cfg['update'],beta=1.,snapshot_weight=1.,dropout=None,
                    run_phase=True,ppo_work=p.ppo_work,epoch_cap=5,ppo_peak=D.rate_receipt(6e-6,8,48))
            self.assertEqual(r['beta_retries'],1);self.assertEqual(r['beta_clone_final'],2.)
            self.assertGreater(r['optimizer_steps'],0)
            self.assertLessEqual(r['probe_kl'],.002);self.assertLessEqual(r['age0_probe_kl'],.002)
            self.assertEqual(D.beta_after_phase(r,cfg['value_phase'])[0],2.)


if __name__=='__main__':unittest.main()
