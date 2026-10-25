"""g950 PPG schedule, one-time beta adoption and exact continuation regressions."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_g210 as G,value_phase as V,gae,snapshot_r2
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_g210 import prepare
from test_ppg_eight import stage_only
from test_training_g850 import smoke_config as prior_config
from test_gae_value_phase import finish
from test_value_buffer import commit

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0950-20261025T053358889475Z'


def smoke_config():
    m,opt,c=prior_config()
    return m,opt,G.linear_options(c)


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_config_only_changes_ppg_and_fresh_default(self):
        old=read(EXPORT/'recorded-config.json');new=validate_config(G.linear_options(old))
        self.assertEqual({k for k in new if old[k]!=new[k]},{'value_phase'})
        self.assertEqual({k for k in new['value_phase'] if old['value_phase'][k]!=new['value_phase'][k]},
            {'contract','rate_schedule','beta_clone_min'})
        self.assertEqual(G.linear_options(new),new)
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(fresh['value_phase'],new['value_phase'])
        for k,v in [('beta_clone_min',.5),('max_epochs',3),('phase_interval',2),('validation_iterations',1)]:
            with self.assertRaises(ValueError):G.validate(dict(new['value_phase'],**{k:v}))
        for k,v in [('end_factor',0.),('end_factor',.3),('warmup_steps',16.),('start_factor',0.)]:
            bad=copy.deepcopy(new['value_phase']);bad['rate_schedule'][k]=v
            with self.assertRaises(ValueError):G.validate(bad)
        self.assertEqual(G.validate(old['value_phase'])['beta_clone_min'],.5)

    def test_linear_warmup_cosine_endpoint_and_beta_floor(self):
        c=G.LINEAR_DEFAULT;rates=[G.rate_factor(i,138,c) for i in range(1,139)]
        self.assertEqual(rates[0],.1);self.assertEqual(rates[15],1.);self.assertEqual(rates[-1],.001)
        for i in range(1,16):self.assertAlmostEqual(rates[i]-rates[i-1],.9/15)
        self.assertEqual(rates[15:],sorted(rates[15:],reverse=True))
        self.assertAlmostEqual(G.rate_factor(77,138,c),.5005)
        self.assertAlmostEqual(G.rate_factor(4,4,c),.28)  # No compressed warmup for small fixtures.
        report=dict(kl_guard_tripped=False,optimizer_steps=1,probe_kl=.0005)
        self.assertEqual(G.beta_after_phase(1.,[],report,c)[0],1.)
        self.assertEqual(G.beta_after_phase(2.,[],report,c)[0],1.)
        self.assertEqual(G.beta_after_phase(1.,[True,True],dict(report,kl_guard_tripped=True),c)[0],2.)
        self.assertEqual(G.beta_after_phase(1.,[],dict(report,optimizer_steps=0),c)[0],1.)
        with self.assertRaises(ValueError):G.beta_after_phase(.5,[],report,c)

    def test_adoption_changes_beta_once_keeps_cache_rng_adam_and_gates(self):
        m,opt,old=prior_config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
            stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
            for optimizer in (opt,p.optimizer):
                optimizer.zero_grad(set_to_none=True)
                next(m.parameters()).grad=torch.zeros_like(next(m.parameters()));optimizer.step()
            p.beta=.5;p.trips=[True,False,True];p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            raw=dict(training={KEY:p.payload()});original_raw=copy.deepcopy(raw);new=G.linear_options(old)
            original=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            q=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.beta,1.);self.assertEqual(q.trips,p.trips)
            self.assertEqual(q.cache_origin,p.cache_origin);self.assertEqual(q.buffer.receipts,p.buffer.receipts)
            self.assertIsNone(q.startup(1));self.assertIsNone(q.g210_migration)
            self.assertEqual(q.ppg_schedule_migration['origin'],1)
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            self.assertTrue(_same_payload(raw,original_raw))
            self.assertTrue(_same_payload(original,(m.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            # Retry before the first committed v16 checkpoint reapplies the same hotfix.
            retry=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(retry.adoption,q.adoption);self.assertEqual(retry.beta,1.)
            # A committed v16 payload restores adaptive beta, not the one-time value.
            q.buffer.pending=copy.deepcopy(q.buffer.receipts);q.beta=2.
            resumed=TrainingPhases(m,new,raw=dict(training={KEY:q.payload()}),generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(resumed.beta,2.);self.assertIsNone(resumed.ppg_schedule_migration)
            self.assertEqual(resumed.adoption,q.adoption)
            bad=copy.deepcopy(raw);bad['training'][KEY]['beta_clone']=float('nan')
            with self.assertRaises(ValueError):TrainingPhases(m,new,raw=bad,generation=1,root=SOURCE_ROOT,run=run)

    def test_two_epochs_endpoint_and_next_phase_restart(self):
        m,opt,c=smoke_config();c['value_phase']['batch_size']=4
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,c,root=SOURCE_ROOT,run=run);prepare(m,p,1)
            metric=V.metrics
            def improving(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence'):
                    step=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1.-.0001*step
                return result
            with patch.object(V,'metrics',improving),patch.object(V.RawPolicyReference,'measure',return_value=0.):r=p.fit(m)
            self.assertEqual(r['optimizer_steps'],r['planned_steps']);self.assertEqual(r['completed_epochs'],2)
            self.assertGreater(r['planned_steps'],16)
            peak=r['peak_learning_rate'];rates=r['attempted_learning_rates']
            self.assertAlmostEqual(rates[0],peak*.1);self.assertEqual(rates[15],peak)
            self.assertAlmostEqual(rates[-1],peak*.001);self.assertGreater(rates[-1],0)
            self.assertEqual(p.optimizer.param_groups[0]['lr'],rates[-1])
            update=dict(categorical_probe=dict(categorical_kl=0.),controller_warmup_done=True,optimizer_steps=1,kl_early_stop=False,gae={})
            finish(p,update,1);saved=copy.deepcopy(p.payload());commit(run,p,1)
            resumed=TrainingPhases(m,c,raw=dict(training={KEY:saved}),generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(resumed.optimizer.param_groups[0]['lr'],rates[-1])
            prepare(m,resumed,2,rate=8e-6)
            second=resumed.fit(m)
            self.assertAlmostEqual(second['attempted_learning_rates'][0],8e-6*.65*.1)
            self.assertEqual(second['acceptance_generations'],[1,2])
            self.assertEqual(sum(x['rows'] for x in second['acceptance_support']),second['acceptance_heldout_rows'])
            self.assertEqual(second['heldout_rows'],len(resumed.heldout_ids))
            bad=copy.deepcopy(saved);bad['optimizer']['param_groups'][0]['lr']=0.
            with self.assertRaises(ValueError):TrainingPhases(m,c,raw=dict(training={KEY:bad}),generation=1,root=SOURCE_ROOT,run=run)

    def test_sixteen_step_kl_rollback_restores_model_and_adam_lr(self):
        m,opt,c=smoke_config();c['value_phase']['batch_size']=2
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1)
            metric=V.metrics;capture={}
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence'):
                    step=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1.-.001*step
                    if step==16:capture.update(model=copy.deepcopy(m.state_dict()),adam=copy.deepcopy(p.optimizer.state_dict()))
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',side_effect=[.001,.003,0.]):r=p.fit(m)
            self.assertEqual(r['attempted_steps'],32);self.assertEqual(r['optimizer_steps'],16)
            self.assertTrue(_same_payload(capture['model'],m.state_dict()))
            self.assertTrue(_same_payload(capture['adam'],p.optimizer.state_dict()))
            self.assertEqual([x['step'] for x in r['kl_checks']],[16,32])

    def test_helper_review_and_read_only_exact_generation(self):
        spec=importlib.util.spec_from_file_location('g950_helper',SOURCE_ROOT/'scripts/configure-training-g950.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        state=read(EXPORT/'state.json')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
            checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated-checkpoint-stand-in')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            before=(sha(run/'state.json'),sha(checkpoint))
            report=helper.generate(root,run,'configs/step5/g950.json',950)
            self.assertEqual(report,helper.generate(root,run,'configs/step5/g950.json',950))
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
            self.assertEqual(report['beta_on_adoption'],1.)
            signature=dict(state['signature'],implementation=implementation_id(),config=G.linear_options(state['signature']['config']))
            self.assertTrue(check_signature(state,signature,reviewed=True))
            signature['config']['learning_rate']*=2
            with self.assertRaises(ValueError):check_signature(state,signature,reviewed=True)
            with self.assertRaises(ValueError):helper.generate(root,run,'configs/step5/stale.json',949)


if __name__=='__main__':unittest.main()
