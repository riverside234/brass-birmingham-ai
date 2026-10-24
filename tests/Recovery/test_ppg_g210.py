"""g210 semantics, narrow g660 adoption, beta feedback and transaction guards."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_g210 as G,value_phase as V,gae,ppg_config,updates
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT as AUX
from brass_ai.step5.league import write_json
from test_ppg_dense_eight import config as prior_config
from test_ppg_eight import collect,stage_only
from test_gae_value_phase import finish
from test_value_buffer import commit
from test_local_scoring import annotate

EXPORT='results/joint7-v11-2-grouped-lr-critic-g0660-20261023T225307387849Z'


def config():
    m,opt,c=prior_config();c=G.options(c)
    c['value_phase'].update(batch_size=128,clone_rows=64,clone_batch_size=8)
    return m,opt,c


def prepare(model,phase,g,*,companion=False,rate=6e-6,steps=1):
    b,data=collect(model,phase,g,branches=True)
    if companion:annotate(data);annotate(phase.snapshot_data)
    ds=updates.Dataset(data,'cpu');phase.prepare(ds)
    actor=[i for i in range(len(data)) if len(data[i][3].choice_ids)>1]
    phase.after_ppo(model,actor,actor[:12],{},ppo_rate=rate,ppo_steps=steps)
    return b,data


class Restored(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_g210_options_with_requested_beta_and_rate_changes(self):
        original=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0210-20261020T145156876449Z/recorded-config.json')['value_phase']
        old=read(SOURCE_ROOT/EXPORT/'state.json')['signature']['config'];new=validate_config(G.options(old))
        self.assertEqual({k for k in old if old[k]!=new[k]},{'value_phase'})
        c=new['value_phase']
        for k in original:
            if k not in ('contract','beta_clone','beta_clone_max','learning_rate'):
                self.assertEqual(c[k],original[k],k)
        self.assertEqual(c['max_epochs'],3);self.assertEqual(c['beta_clone'],4.)
        self.assertEqual(c['phase_interval'],1);self.assertEqual(c['validation_iterations'],1)
        self.assertEqual(c['kl_check_steps'],32)
        self.assertEqual(c['learning_rate_factor'],.65);self.assertEqual(G.options(new),new)
        for k,v in [('phase_interval',2),('buffer_iterations',8),('acceptance_steps',32),('kl_cap',None),
                    ('max_epochs',7),('beta_clone_min',4.),('beta_clone_max',64.),('learning_rate_factor',.35),('kl_check_steps',1)]:
            with self.assertRaises(ValueError):V.validate(dict(c,**{k:v}))

    def test_beta_old_three_of_five_up_new_low_kl_down_and_clamps(self):
        c=G.DEFAULT;beta=4.;trips=[]
        for trip,expected in [(True,4.),(False,4.),(True,4.),(False,4.),(True,8.)]:
            beta,trips,r=G.beta_after_phase(beta,trips,dict(kl_guard_tripped=trip,optimizer_steps=1,probe_kl=.001),c)
            self.assertEqual(beta,expected)
        for expected in (4.,2.,1.,1.):
            beta,trips,r=G.beta_after_phase(beta,trips,dict(kl_guard_tripped=False,optimizer_steps=1,probe_kl=.000999),c)
            self.assertEqual(beta,expected)
        self.assertEqual(G.beta_after_phase(32.,[True,True],dict(kl_guard_tripped=True,optimizer_steps=1,probe_kl=0.),c)[0],32.)
        self.assertEqual(G.beta_after_phase(4.,[],dict(kl_guard_tripped=False,optimizer_steps=0,probe_kl=0.),c)[0],4.)
        self.assertEqual(G.beta_after_phase(4.,[],dict(kl_guard_tripped=True,optimizer_steps=1,probe_kl=0.),c)[0],4.)
        for bad in (-1.,float('nan')):
            with self.assertRaises(ValueError):G.beta_after_phase(4.,[],dict(kl_guard_tripped=False,optimizer_steps=1,probe_kl=bad),c)

    def test_dynamic_rate_and_no_rate_no_update(self):
        for rate in (3e-6,6e-6,1.8e-5):self.assertEqual(G.rate_receipt(rate,661,20)['peak_lr'],rate*.65)
        m,opt,c=config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1,rate=None,steps=0)
            before=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),torch.get_rng_state()))
            r=p.fit(m);self.assertEqual(r['early_stop_reason'],'no_applied_ppo_rate');self.assertEqual(r['optimizer_steps'],0)
            self.assertTrue(_same_payload(before,(m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),torch.get_rng_state())))

    def test_adoption_resets_beta_once_preserves_all_adams_rng_and_gate_state(self):
        m,opt,old=prior_config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
            stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1
            from brass_ai.recovery import snapshot_r2
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
            p.schedule=ppg_config.advance(p.schedule,1,False);p.beta=16.
            p.buffer.pending=copy.deepcopy(p.buffer.receipts);p.pending_schedule=p.schedule;p.pending_epoch_control=p.epoch_control
            raw=dict(training={KEY:p.payload()});new=G.options(old)
            original=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            q=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.beta,4.);self.assertEqual(q.trips,[])
            self.assertTrue(_same_payload(original,(m.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            with self.assertRaises((ValueError,KeyError)):q.startup(1)
            write_json(run/'state.json',dict(iteration=1,signature=dict(config=new)))
            self.assertTrue(q.startup(1)['deleted_files'])
            again=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(again.beta,4.);self.assertEqual(again.startup(1)['deleted_files'],[])
            prepare(m,q,2);r=q.fit(m)
            update=dict(categorical_probe=dict(categorical_kl=0.),controller_warmup_done=True,optimizer_steps=1,kl_early_stop=False,gae={})
            finish(q,update,2);payload=q.payload();commit(run,q,2)
            resumed=TrainingPhases(m,new,raw=dict(training={KEY:payload}),generation=2,root=SOURCE_ROOT,run=run)
            self.assertEqual(resumed.beta,q.beta);self.assertEqual(resumed.trips,q.trips)
            self.assertIsNone(resumed.startup(2));self.assertEqual(len(resumed.buffer.receipts),1)

    def test_32_step_kl_checks_pending_loss_state_and_restores_adams_and_companion(self):
        for pending_kl,kept in ((.003,0),(.0015,16)):
            with self.subTest(pending_kl=pending_kl):self._guard_case(pending_kl,kept)

    def _guard_case(self,pending_kl,kept):
        m,opt,c=config();c['local_scoring_auxiliary']=copy.deepcopy(AUX)
        c['value_phase']['batch_size']=2
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));p.local_scoring=Companion(m,AUX)
            prepare(m,p,1,companion=True)
            p.optimizer.param_groups[0]['lr']=6e-6*.65
            from brass_ai.recovery.ppg_phasic import RATE_KEY
            p.optimizer.param_groups[0][RATE_KEY]=copy.deepcopy(p.ppo_peak)
            def states():return copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.local_scoring.heads.state_dict(),p.local_scoring.optimizers['ppg'].state_dict()))
            before=states();capture={};metric=V.metrics
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if ds is p.critic_dataset and not kw.get('unknown'):
                    count=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1.-.001*count
                    if count==16:capture[16]=states()
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',side_effect=[.003,pending_kl,0.]):r=p.fit(m)
            self.assertEqual(r['attempted_steps'],32);self.assertEqual(r['optimizer_steps'],kept);self.assertTrue(r['kl_guard_tripped'])
            self.assertEqual([(v['step'],v['scope']) for v in r['kl_checks']],[(32,'periodic'),(16,'retention')])
            self.assertTrue(_same_payload(capture[16] if kept else before,states()))
            self.assertEqual(r['value_presentations'],kept*2)
            self.assertEqual(r['accepted_intervals'],int(bool(kept)))
            self.assertEqual(r['heldout_after']['loss'],1.-.001*kept)
            self.assertEqual(r['epoch_reports'][0]['accepted_steps'],kept)

    def test_kl_cadence_spans_epochs_and_checks_short_tails(self):
        m,opt,c=config();c['value_phase']['batch_size']=8
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1)
            metric=V.metrics
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if ds is p.critic_dataset and not kw.get('unknown'):
                    result['loss']=1.-.001*max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',return_value=0.):r=p.fit(m)
            import math
            per_epoch=math.ceil(r['train_rows']/8)
            expected=sorted(set(range(32,per_epoch*3+1,32))|{per_epoch,per_epoch*2,per_epoch*3})
            self.assertEqual([v['step'] for v in r['kl_checks']],expected)
            self.assertIn(32,expected);self.assertEqual(r['optimizer_steps'],per_epoch*3)
            self.assertEqual(r['completed_epochs'],3)

    def test_loss_rejection_restores_last_interval_and_no_SE_tolerance(self):
        m,opt,c=config();c['value_phase']['batch_size']=4
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1)
            metric=V.metrics;capture={};real_step=p.optimizer.step
            def count():return max((int(s['step']) for s in p.optimizer.state.values()),default=0)
            def step(*a,**kw):
                out=real_step(*a,**kw)
                if count()==16:capture.update(model=copy.deepcopy(m.state_dict()),adam=copy.deepcopy(p.optimizer.state_dict()))
                return out
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if ds is p.critic_dataset and not kw.get('unknown'):
                    result['loss']=1. if count()==0 else .9 if count()==16 else .900001
                return result
            with patch.object(p.optimizer,'step',step),patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',return_value=0.):r=p.fit(m)
            self.assertEqual(r['optimizer_steps'],16);self.assertEqual(r['early_stop_reason'],'heldout_loss_increased')
            self.assertTrue(_same_payload(capture['model'],m.state_dict()));self.assertTrue(_same_payload(capture['adam'],p.optimizer.state_dict()))
            self.assertEqual(r['interval_reports'][1]['heldout_loss_before'],.9)
            self.assertEqual(r['kl_checks'][-1]['step'],16)
            self.assertEqual(r['kl_checks'][-1]['scope'],'retention')

    def test_safe_later_probe_cannot_certify_rejected_interval_start(self):
        m,opt,c=config();c['value_phase']['batch_size']=4
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1)
            before=copy.deepcopy(m.state_dict());metric=V.metrics
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if ds is p.critic_dataset and not kw.get('unknown'):
                    count=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1. if count==0 else .9 if count==16 else .900001
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',side_effect=[.001,.003,0.]):r=p.fit(m)
            self.assertEqual(r['optimizer_steps'],0);self.assertTrue(r['kl_guard_tripped'])
            self.assertEqual(r['kl_checks'][0]['scope'],'epoch_tail');self.assertTrue(r['kl_checks'][0]['passed'])
            self.assertEqual(r['kl_checks'][1]['step'],16);self.assertFalse(r['kl_checks'][1]['passed'])
            self.assertEqual(r['heldout_after']['loss'],1.);self.assertTrue(_same_payload(before,m.state_dict()))

    def test_g660_helper_is_idempotent_narrow_and_preserves_run(self):
        spec=importlib.util.spec_from_file_location('g210_config',SOURCE_ROOT/'scripts/configure-ppg-g210.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        state=read(SOURCE_ROOT/EXPORT/'state.json')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
            checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated full-checkpoint stand-in')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            before=(sha(run/'state.json'),sha(checkpoint))
            report=helper.generate(root,run,'configs/step5/g210.json',660)
            self.assertEqual(report['first_phase'],661);self.assertEqual(report['beta_on_adoption'],4.)
            self.assertEqual(report['kl_check_steps'],32)
            self.assertEqual(helper.generate(root,run,'configs/step5/g210.json',660),report)
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
            new=read(root/'configs/step5/g210.json')
            self.assertEqual({k for k in new if new[k]!=state['signature']['config'][k]},{'value_phase'})
            from brass_ai.recovery.resume import check_signature
            from brass_ai.step5.league import implementation_id
            bad=copy.deepcopy(new);bad['exploration']=.08
            with self.assertRaises(ValueError):check_signature(state,dict(state['signature'],implementation=implementation_id(),config=bad),reviewed=True)
            with self.assertRaises(ValueError):helper.generate(root,run,'configs/step5/stale.json',659)


if __name__=='__main__':unittest.main()
