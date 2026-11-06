"""Sparse exact PPO checks, disabled logging scans, and reviewed continuation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import ppo_kl_control as K,updates,lr_controller as LR
from brass_ai.recovery.config import update_config,validate_config,latest_model_config
from brass_ai.recovery.history import compact
from brass_ai.recovery.resume import check_signature
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.review_metrics import CategoricalProbe
from brass_ai.step5.league import implementation_id,write_json
from test_categorical_kl_guard import GuardTests
import test_ppo_lr_controller as L

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0921-20261105T213224468369Z'


class SparseProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def run_update(self,*,epochs=2,batch_size=2,stop_step=None,metric='categorical_kl',nan_step=None):
        model,opt,b,data,c=GuardTests().fixture()
        c.update(kl_control=copy.deepcopy(K.SPARSE_DEFAULT),epochs=epochs,effective_batch_size=batch_size)
        steps=[];checks=[];scans=[]
        real_step=updates.scheduled_step;real_measure=CategoricalProbe.measure_step;real_scan=updates.policy_measure
        def step(*a,**kw):
            real_step(*a,**kw);steps.append(len(steps)+1)
        def measure(probe,*a,**kw):
            checks.append(len(steps));result=real_measure(probe,*a,**kw)
            result['categorical_kl']=.001;result['behavior_categorical_kl']=.001
            if len(steps)==stop_step:result[metric]=.02
            if len(steps)==nan_step:
                raise ValueError('Nonfinite PPO KL probe; refusing further updates')
            return result
        def scan(*a,**kw):
            verify=kw.get('verify',a[6] if len(a)>6 else False)
            self.assertTrue(verify,'A logging-only full scan ran')
            scans.append(len(steps));return real_scan(*a,**kw)
        with patch.object(updates,'scheduled_step',side_effect=step),patch.object(CategoricalProbe,'measure_step',measure),\
                patch.object(updates,'policy_measure',side_effect=scan):
            report=updates.ppo_update(model,opt,[dict(split='train')],b,c,learner_data=data)
        self.assertEqual(scans,[0])
        return report,opt,checks

    def test_first_fourth_and_epoch_tails_including_short_epoch(self):
        for batch,expected in [(2,[1,4,8]),(3,[1,3,4,6]),(16,[1,2])]:
            with self.subTest(batch=batch):
                r,_,checks=self.run_update(batch_size=batch)
                self.assertEqual(checks,expected)
                self.assertEqual(r['kl_guard']['probe_check_steps'],expected)
                self.assertEqual(r['kl_guard']['probe_checks'],len(expected))
                self.assertEqual(r['kl_guard']['full_dataset_checks'],0)
                self.assertTrue(all(e['post_update_kl'] is None for e in r['epochs']))
                self.assertTrue(torch.isfinite(torch.tensor(r['categorical_probe']['categorical_kl'])))

    def test_48_steps_have_13_checks(self):
        r,_,checks=self.run_update(epochs=6,batch_size=1)
        self.assertEqual(r['optimizer_steps'],48)
        self.assertEqual(checks,[1,*range(4,49,4)])

    def test_both_exact_guards_stop_at_first_periodic_or_tail_check(self):
        for metric in ('categorical_kl','behavior_categorical_kl'):
            for step,batch in [(1,2),(4,2),(3,3),(6,3)]:
                with self.subTest(metric=metric,step=step):
                    r,_,checks=self.run_update(stop_step=step,metric=metric,batch_size=batch)
                    self.assertTrue(r['kl_early_stop']);self.assertEqual(r['optimizer_steps'],step)
                    self.assertEqual(checks[-1],step)

    def test_mandatory_behavior_reconstruction_rejects_tampering(self):
        m,o,b,data,c=GuardTests().fixture();c['kl_control']=copy.deepcopy(K.SPARSE_DEFAULT)
        data[0][0]['behavior']['probability']*=.5
        with patch.object(updates,'scheduled_step') as step,self.assertRaises(ValueError):
            updates.ppo_update(m,o,[dict(split='train')],b,c,learner_data=data)
        step.assert_not_called()

    def test_nonfinite_at_scheduled_check_aborts(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite PPO KL probe'):self.run_update(nan_step=4)

    def test_tempered_openings_keep_raw_and_behavior_guards_separate(self):
        from test_gae_value_phase import fixture,ppo
        from test_opening_ppo import opening_rows
        from brass_ai.recovery.opening_objective import DEFAULT as OPENING
        for metric,reason in [('categorical_kl','temperature1_categorical_probe'),
                              ('behavior_categorical_kl','behavior_categorical_probe')]:
            model,opt,cfg,(_,data)=fixture(games=3,decisions=3)
            behavior,data=opening_rows(model,data);cfg=K.configure_sparse(K.configure(cfg))
            cfg['update']['opening_objective']=copy.deepcopy(OPENING)
            measured=dict(categorical_kl=0.,behavior_categorical_kl=0.,sampled_kl=0.)
            measured[metric]=.02
            with patch.object(CategoricalProbe,'measure_step',return_value=measured):
                report=ppo(model,opt,cfg,behavior,data)
            self.assertEqual(report['optimizer_steps'],1)
            self.assertEqual(report['kl_guard']['stop_reason'],reason)
            self.assertTrue(report['opening_objective']['behavior_verified'])

    def test_skipped_observation_preserves_updates_and_rng_when_no_guard_trips(self):
        final=[]
        for block in (K.DEFAULT,K.SPARSE_DEFAULT):
            m,o,b,data,c=GuardTests().fixture();c['kl_control']=copy.deepcopy(block)
            r=updates.ppo_update(m,o,[dict(split='train')],b,c,learner_data=data)
            self.assertFalse(r['kl_early_stop'])
            final.append((m.state_dict(),o.state_dict(),torch.get_rng_state(),r['categorical_probe']))
        self.assertTrue(_same_payload(*final))

    def test_lr_and_compact_report_use_final_exact_probe_and_null_full_scan(self):
        r,opt,_=self.run_update();cfg=K.configure_sparse(K.configure(L.config()))
        with patch.object(LR,'collection_signature',return_value='a'*64),patch.object(
                LR,'game_health',return_value=dict(selfplay_vp_per_seat=100.,zero_vp_fraction=0.)):
            feedback=LR.finish_iteration(opt,r,cfg,{},[],None,{},'anchor')
        self.assertEqual(feedback['kl'],r['categorical_probe']['categorical_kl'])
        history=compact(dict(actor=r))['actor']
        self.assertEqual(history['kl_guard']['probe_check_steps'],[1,4,8])
        self.assertIsNone(history['epochs'][-1]['post_update_kl']);digest(history)
        self.assertIn('KLsample_probe=',K.sampled_console(r));self.assertIn('KLfull=disabled',K.sampled_console(r))

    def test_exact_versioned_config_and_fresh_defaults(self):
        state=read(BUNDLE/'state.json');old=state['signature']['config']
        new=K.configure_sparse(old);validate_config(new)
        expected=copy.deepcopy(old);expected['update']['kl_control']=copy.deepcopy(K.SPARSE_DEFAULT)
        self.assertEqual(new,expected);self.assertEqual(K.configure_sparse(new),new)
        base=read(SOURCE_ROOT/'configs/step5/joint7-latest.json')
        fresh=latest_model_config(base,base['imitation_checkpoint'])
        self.assertEqual(fresh['update']['kl_control'],K.SPARSE_DEFAULT)
        for key,value in [('probe_check_steps',0),('probe_check_steps',True),('probe_check_steps',4.),
                          ('full_dataset_kl',0),('full_dataset_kl',True),('probe_rows',1024),('extra',True)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                update_config(dict(new['update'],kl_control=dict(K.SPARSE_DEFAULT,**{key:value})))

    def test_reviewed_transition_is_narrow(self):
        state=read(BUNDLE/'state.json');signature=dict(state['signature'],
            config=K.configure_sparse(state['signature']['config']),implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,reviewed=True))
        with self.assertRaises(ValueError):check_signature(state,signature)
        bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)

    def test_generator_retry_missing_checkpoint_and_pending_work(self):
        spec=importlib.util.spec_from_file_location('sparse_helper',SOURCE_ROOT/'scripts/configure-ppo-probes.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0921.pt';checkpoint.write_bytes(b'isolated committed fixture')
            state=read(BUNDLE/'state.json');state['history'][-1].update(path='artifacts/run/generation-0921.pt',sha256=sha(checkpoint))
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes()
            args=(root,'artifacts/run','configs/step5/sparse.json',921)
            self.assertEqual(helper.generate(*args),helper.generate(*args))
            self.assertEqual((run/'state.json').read_bytes(),before)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint'):helper.generate(*args)
            checkpoint.write_bytes(b'isolated committed fixture')
            (run/'iteration-0921').mkdir();(run/'iteration-0921/partial').write_text('pending')
            with self.assertRaisesRegex(ValueError,'interrupted'):helper.generate(*args)

    def test_real_checkpoint_adoption_preserves_Adam_rng_and_retry_state(self):
        from test_compact_il_resume import CompactILResumeTests
        from brass_ai.learning.checkpoint import resume_optimizer
        from brass_ai.recovery.resume import load_resume,commit_transition,apply_optimizer_config
        helper=CompactILResumeTests();helper.original=read(BUNDLE/'state.json');helper.current=implementation_id()
        self.addCleanup(helper.doCleanups)
        run,state,signature=helper.fixture();signature['config']=K.configure_sparse(signature['config'])
        original_state=copy.deepcopy(state);path=SOURCE_ROOT/state['history'][-1]['path']
        before=torch.load(path,map_location='cpu',weights_only=True);original_hash=sha(path)
        for retry in range(2):
            model,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=retry==0)
            opt=torch.optim.AdamW(model.parameters(),lr=signature['config']['learning_rate'])
            resume_optimizer(raw,opt);apply_optimizer_config(opt,signature['config'])
            self.assertTrue(_same_payload(before['state_dict'],dict(model.state_dict())))
            self.assertTrue(_same_payload(before['optimizer'],opt.state_dict()))
            self.assertTrue(_same_payload(before['training'],raw['training']))
            self.assertTrue(torch.equal(before['rng'],torch.get_rng_state()))
            self.assertEqual(model.utility_calibration,before['training']['utility_calibration'])
            self.assertEqual(changed,retry==0);self.assertEqual(sha(path),original_hash)
            if retry==0:state=commit_transition(run,state,signature)
        for key in original_state.keys()-{'signature','source_transitions'}:
            self.assertEqual(state[key],original_state[key])


if __name__=='__main__':unittest.main()
