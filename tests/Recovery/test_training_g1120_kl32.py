"""Loss16/KL32 safety, direct g1120 adoption and restart regressions."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_g210 as G,value_phase as V,gae,snapshot_r2
from brass_ai.recovery.training_revision_g1120 import configure,configure_kl32
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT as AUX
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import write_json,implementation_id
from test_training_g1050 import smoke_config as saved_config
from test_training_g1120 import smoke_config as previous_config,EXPORT
from test_ppg_g210 import prepare
from test_ppg_eight import stage_only


def smoke_config():
    model,opt,c=previous_config()
    return model,opt,G.kl32_options(c)


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_direct_and_intermediate_config_adoption_are_equal_and_narrow(self):
        old=read(EXPORT/'recorded-config.json');previous=configure(old);new=validate_config(configure_kl32(old))
        self.assertEqual(new,configure_kl32(previous));self.assertEqual(new,configure_kl32(new))
        self.assertEqual({k for k in previous if previous[k]!=new[k]},{'value_phase'})
        self.assertEqual(new['value_phase'],dict(previous['value_phase'],contract=G.KL32_CONTRACT,kl_check_steps=32))
        self.assertEqual(latest_model_config(old,old['imitation_checkpoint'])['value_phase'],new['value_phase'])
        for key,value in [('kl_check_steps',16),('acceptance_steps',8),('kl_cap',.003),('beta_clone_min',1.)]:
            with self.assertRaises(ValueError):G.validate(dict(new['value_phase'],**{key:value}))
        self.assertEqual([G.rate_factor(i,142,new['value_phase']) for i in range(1,143)],
            [G.rate_factor(i,142,previous['value_phase']) for i in range(1,143)])

    def test_direct_v17_and_v18_resume_preserve_beta_cache_adams_rng_and_gates(self):
        for factory in (saved_config,previous_config):
            with self.subTest(factory=factory.__module__):
                m,opt,old=factory()
                with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                    run=Path(tmp);p=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
                    stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
                    p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1
                    p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
                    for optimizer in (opt,p.optimizer):
                        optimizer.zero_grad(set_to_none=True);next(m.parameters()).grad=torch.zeros_like(next(m.parameters()));optimizer.step()
                    p.beta=.5;p.trips=[True,False];p.buffer.pending=copy.deepcopy(p.buffer.receipts)
                    raw=dict(training={KEY:p.payload()});original=copy.deepcopy(raw);new=G.kl32_options(old)
                    before=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
                    hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
                    q=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
                    self.assertEqual(q.beta,.5);self.assertEqual(q.trips,p.trips)
                    self.assertEqual(q.buffer.receipts,p.buffer.receipts);self.assertEqual(q.cache_origin,p.cache_origin)
                    self.assertIsNone(q.startup(1));self.assertIsNone(q.g210_migration)
                    self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
                    self.assertTrue(_same_payload(original,raw))
                    self.assertTrue(_same_payload(before,(m.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
                    retry=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
                    self.assertEqual(retry.adoption,q.adoption)
                    q.buffer.pending=copy.deepcopy(q.buffer.receipts);q.beta=2.
                    resumed=TrainingPhases(m,new,raw=dict(training={KEY:q.payload()}),generation=1,root=SOURCE_ROOT,run=run)
                    self.assertEqual(resumed.beta,2.);self.assertIsNone(resumed.ppg_schedule_migration)

    def test_failed_32_step_guard_checks_provisional_16_and_restores_all_adams(self):
        for pending_kl,kept in ((.001,16),(.003,0)):
            with self.subTest(pending_kl=pending_kl):self.guard_case(pending_kl,kept)

    def guard_case(self,pending_kl,kept,*,loss_reject=False):
        m,opt,c=smoke_config();c['value_phase']['batch_size']=2;c['local_scoring_auxiliary']=copy.deepcopy(AUX)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));p.local_scoring=Companion(m,AUX)
            prepare(m,p,1,companion=True);metric=V.metrics;captures={}
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence'):
                    step=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1.1 if loss_reject and step==32 else 1.-.001*step
                    captures.setdefault(step,copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.local_scoring.snapshot())))
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',side_effect=[.001 if loss_reject else .003,pending_kl,0.]):r=p.fit(m)
            self.assertEqual(r['attempted_steps'],32);self.assertEqual(r['optimizer_steps'],kept)
            self.assertEqual([(i['step'],i['scope']) for i in r['kl_checks']],[(32,'periodic'),(16,'retention')])
            self.assertEqual([i['steps'] for i in r['interval_reports']],[16,16])
            self.assertTrue(r['kl_guard_tripped']);self.assertLessEqual(r['probe_kl'],.002)
            self.assertTrue(_same_payload(captures[kept],(m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.local_scoring.snapshot())))
            self.assertEqual(r['value_presentations'],kept*2)
            self.assertEqual(r['accepted_intervals'],int(kept>0))

    def test_safe_32_probe_cannot_certify_unsafe_16_loss_checkpoint(self):
        self.guard_case(.003,0,loss_reject=True)

    def test_kl_cadence_spans_epochs_and_checks_short_tails(self):
        m,opt,c=smoke_config();c['value_phase']['batch_size']=4
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1);metric=V.metrics
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence'):result['loss']=1.-.001*max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',return_value=0.):r=p.fit(m)
            per_epoch=math.ceil(r['train_rows']/4)
            expected=sorted(set(range(32,per_epoch*2+1,32))|{per_epoch,per_epoch*2})
            self.assertEqual([v['step'] for v in r['kl_checks']],expected);self.assertIn(32,expected)
            self.assertEqual(r['optimizer_steps'],per_epoch*2);self.assertEqual(r['completed_epochs'],2)

    def test_helper_direct_g1120_and_interrupted_adoption_are_read_only(self):
        spec=importlib.util.spec_from_file_location('g1120_kl32_helper',SOURCE_ROOT/'scripts/configure-training-g1120.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);original=read(EXPORT/'state.json')
        for adopted in (False,True):
            state=copy.deepcopy(original)
            if adopted:
                state['signature'].update(config=configure(state['signature']['config']),
                    implementation='4199ffb44025664806548ace38b27470c477719b1fcc9f3e11ef0f9b402ceb4e')
            with self.subTest(adopted=adopted),tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
                root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
                checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated checkpoint stand-in')
                state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
                before=(sha(run/'state.json'),sha(checkpoint));output='configs/step5/g1120-kl32.json'
                report=helper.generate(root,run,output,1120)
                self.assertEqual(report,helper.generate(root,run,output,1120))
                self.assertEqual(report['value_phase']['kl_check_steps'],32);self.assertFalse(report['beta_reset'])
                self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
                signature=dict(state['signature'],implementation=implementation_id(),config=configure_kl32(state['signature']['config']))
                self.assertTrue(check_signature(state,signature,reviewed=True))
                signature['config']['exploration']=0.
                with self.assertRaises(ValueError):check_signature(state,signature,reviewed=True)
                with self.assertRaises(ValueError):helper.generate(root,run,output,1119)


if __name__=='__main__':unittest.main()
