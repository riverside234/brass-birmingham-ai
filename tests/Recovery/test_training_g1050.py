"""Half-beta adoption and loss8/KL16 continuation regressions."""
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
from test_training_g950 import smoke_config as prior_config
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT as AUX

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1050-20261025T163535686370Z'


def smoke_config():
    m,opt,c=prior_config()
    return m,opt,G.short_options(c)


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_config_only_changes_loss_interval_and_floor_and_identity(self):
        old=read(EXPORT/'recorded-config.json');new=validate_config(G.short_options(old))
        self.assertEqual({k for k in new if old[k]!=new[k]},{'value_phase'})
        self.assertEqual({k for k in new['value_phase'] if old['value_phase'][k]!=new['value_phase'][k]},
            {'contract','acceptance_steps','beta_clone_min'})
        self.assertEqual(G.short_options(new),new)
        self.assertEqual(latest_model_config(old,old['imitation_checkpoint'])['value_phase'],new['value_phase'])
        for k,v in [('beta_clone_min',1.),('acceptance_steps',16),('kl_check_steps',8),('validation_iterations',1)]:
            with self.assertRaises(ValueError):G.validate(dict(new['value_phase'],**{k:v}))
        self.assertEqual([G.rate_factor(i,142,new['value_phase']) for i in range(1,143)],
            [G.rate_factor(i,142,old['value_phase']) for i in range(1,143)])

    def test_beta_half_floor_and_existing_up_down_rules(self):
        c=G.SHORT_DEFAULT;r=dict(kl_guard_tripped=False,optimizer_steps=8,probe_kl=.0009)
        self.assertEqual(G.beta_after_phase(1.,[],r,c)[0],.5)
        self.assertEqual(G.beta_after_phase(.5,[],r,c)[0],.5)
        self.assertEqual(G.beta_after_phase(.5,[True,True],dict(r,kl_guard_tripped=True),c)[0],1.)
        self.assertEqual(G.beta_after_phase(1.,[],dict(r,optimizer_steps=0),c)[0],1.)

    def test_adoption_is_once_preserves_cache_bytes_adams_rng_and_evidence(self):
        m,opt,old=prior_config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
            stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
            for optimizer in (opt,p.optimizer):
                optimizer.zero_grad(set_to_none=True)
                next(m.parameters()).grad=torch.zeros_like(next(m.parameters()));optimizer.step()
            p.beta=1.;p.trips=[True,False,True];p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            raw=dict(training={KEY:p.payload()});original_raw=copy.deepcopy(raw);new=G.short_options(old)
            original=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            q=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.beta,.5);self.assertEqual(q.trips,p.trips)
            self.assertEqual(q.buffer.receipts,p.buffer.receipts);self.assertEqual(q.cache_origin,p.cache_origin)
            self.assertIsNone(q.startup(1));self.assertIsNone(q.g210_migration)
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            self.assertTrue(_same_payload(raw,original_raw))
            self.assertTrue(_same_payload(original,(m.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            retry=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(retry.adoption,q.adoption);self.assertEqual(retry.beta,.5)
            q.buffer.pending=copy.deepcopy(q.buffer.receipts);q.beta=2.
            resumed=TrainingPhases(m,new,raw=dict(training={KEY:q.payload()}),generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(resumed.beta,2.);self.assertIsNone(resumed.ppg_schedule_migration)
            self.assertEqual(resumed.adoption,q.adoption)
            bad=copy.deepcopy(raw);bad['training'][KEY]['beta_clone']=float('nan')
            with self.assertRaises(ValueError):TrainingPhases(m,new,raw=bad,generation=1,root=SOURCE_ROOT,run=run)

    def test_step16_kl_failure_certifies_and_restores_step8_with_matching_adams(self):
        for pending_kl,kept in ((.001,8),(.003,0)):
            with self.subTest(pending_kl=pending_kl):self.guard_case(pending_kl,kept)

    def guard_case(self,pending_kl,kept):
        m,opt,c=smoke_config();c['value_phase']['batch_size']=2;c['local_scoring_auxiliary']=copy.deepcopy(AUX)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));p.local_scoring=Companion(m,AUX)
            prepare(m,p,1,companion=True)
            metric=V.metrics;captures={}
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence'):
                    step=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1.-.001*step
                    captures.setdefault(step,copy.deepcopy((m.state_dict(),p.optimizer.state_dict(),p.local_scoring.snapshot())))
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',side_effect=[.003,pending_kl,0.]):r=p.fit(m)
            self.assertEqual(r['attempted_steps'],16);self.assertEqual(r['optimizer_steps'],kept)
            self.assertEqual([i['step'] for i in r['kl_checks']],[16,8])
            self.assertEqual([i['steps'] for i in r['interval_reports']],[8,8])
            self.assertTrue(_same_payload(captures[kept],(m.state_dict(),p.optimizer.state_dict(),p.local_scoring.snapshot())))
            self.assertEqual(r['value_presentations'],kept*2);self.assertLessEqual(r['probe_kl'],.002)

    def test_safe_later_kl_does_not_certify_loss_rejected_step8(self):
        m,opt,c=smoke_config();c['value_phase']['batch_size']=2
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1)
            metric=V.metrics;capture={}
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence'):
                    step=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1. if step==0 else .9 if step==8 else 1.1
                    if step==0:capture.setdefault('initial',copy.deepcopy((m.state_dict(),p.optimizer.state_dict())))
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',side_effect=[.001,.003,0.]):r=p.fit(m)
            self.assertEqual(r['optimizer_steps'],0);self.assertEqual([x['step'] for x in r['kl_checks']],[16,8])
            self.assertTrue(_same_payload(capture['initial'],(m.state_dict(),p.optimizer.state_dict())))

    def test_helper_is_generation_bound_narrow_and_read_only(self):
        spec=importlib.util.spec_from_file_location('g1050_helper',SOURCE_ROOT/'scripts/configure-training-g1050.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);state=read(EXPORT/'state.json')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
            checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated-checkpoint-stand-in')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            before=(sha(run/'state.json'),sha(checkpoint))
            report=helper.generate(root,run,'configs/step5/g1050.json',1050)
            self.assertEqual(report,helper.generate(root,run,'configs/step5/g1050.json',1050))
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)));self.assertEqual(report['beta_on_adoption'],.5)
            signature=dict(state['signature'],implementation=implementation_id(),config=G.short_options(state['signature']['config']))
            self.assertTrue(check_signature(state,signature,reviewed=True))
            signature['config']['learning_rate']*=2
            with self.assertRaises(ValueError):check_signature(state,signature,reviewed=True)
            with self.assertRaises(ValueError):helper.generate(root,run,'configs/step5/stale.json',1049)


if __name__=='__main__':unittest.main()
