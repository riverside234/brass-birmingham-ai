"""One-time KL repair, exact reward continuation and old-target PPG exclusion."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import g110_hotfix as fix,il_kl_limit,value_buffer,updates,value_phase
from brass_ai.recovery.config import insolvency_event_config,champion_mix_config,validate_config
from brass_ai.recovery.utility import event_cost_upgrade,cap_upgrade,migrate,reward_spec
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_gae_value_phase import fixture,examples,config
from test_value_buffer import source


def hotfix_state():
    c=config();c['insolvency']=dict(event_cost=.005,max_cost=.075)
    c['search']['reward']['insolvency']=copy.deepcopy(c['insolvency'])
    limit=il_kl_limit.initial_state(c['il_kl_limit'],dict(probe_hash='same'))
    limit.update(limit=.152587890625,last_generation=110,last_update=105,blocked=True,
        healthy_window=[.31]*10,last_healthy_generation=110,kl_last_healthy=.315)
    return dict(format='step5-recovery-league/2.2.0',iteration=110,games_done=35200,next_seed=123,
        history=[dict(generation=110,model_id=fix.MODEL,path='artifacts/run/generation-0110.pt',sha256='same')],
        signature=dict(config=c,implementation='old-source'),il_kl_limit=limit,population=['unchanged'])


class Review110(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_hotfix_is_exact_limit_only_and_idempotent(self):
        state=hotfix_state();before=copy.deepcopy(state);after,changed=fix.prepare(state)
        self.assertTrue(changed);self.assertEqual(state,before)
        self.assertEqual(after['il_kl_limit']['limit'],.2288818359375)
        self.assertEqual(after[fix.KEY]['group_limit_after'],.457763671875)
        expected=copy.deepcopy(state['il_kl_limit']);expected['limit']*=1.5
        self.assertEqual(after['il_kl_limit'],expected)
        self.assertEqual({k for k in after if after[k]!=state.get(k)},{'il_kl_limit',fix.KEY})
        again,changed=fix.prepare(after);self.assertFalse(changed);self.assertEqual(again,after)
        raw=dict(step=110,model_id=fix.MODEL,training=dict(il_kl_limit=state['il_kl_limit']))
        self.assertEqual(fix.verify_checkpoint(after,raw),after[fix.KEY])
        self.assertTrue(after['il_kl_limit']['blocked'])
        after['iteration']=111;raw['step']=111;raw['training']['il_kl_limit']=after['il_kl_limit']
        self.assertIsNone(fix.verify_checkpoint(after,raw))
        self.assertFalse(fix.prepare(after)[1])

    def test_wrong_generation_identity_or_unrecorded_override_fails(self):
        state=hotfix_state();raw=dict(step=110,model_id=fix.MODEL,training=dict(il_kl_limit=state['il_kl_limit']))
        after,_=fix.prepare(state)
        for mutate in (lambda s:s.pop(fix.KEY),lambda s:s['il_kl_limit'].update(blocked=False),
                       lambda s:s['il_kl_limit'].update(limit=.4),lambda s:s['history'][-1].update(sha256='changed')):
            bad=copy.deepcopy(after);mutate(bad)
            with self.assertRaises(ValueError):fix.verify_checkpoint(bad,raw)
        for key,value in (('iteration',109),('iteration',111)):
            with self.assertRaises(ValueError):fix.prepare(dict(state,**{key:value}))
        state['history'][-1]['model_id']='unreviewed'
        with self.assertRaises(ValueError):fix.prepare(state)

    def test_helper_preview_apply_and_retry_never_edit_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            state=hotfix_state();checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'unchanged')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            raw=dict(step=110,model_id=fix.MODEL,training=dict(il_kl_limit=state['il_kl_limit']))
            target='configs/step5/continuation.json'
            with patch('brass_ai.recovery.resume.load_resume',return_value=(None,raw,True)):
                result=fix.run(root,run,target);self.assertFalse(result['applied'])
                self.assertEqual(read(run/'state.json'),state);self.assertFalse((root/target).exists())
                fix.run(root,run,target,apply=True);saved=read(run/'state.json')
                result=fix.run(root,run,target,apply=True);self.assertTrue(result['already_applied'])
                self.assertEqual(saved,read(run/'state.json'))
                self.assertEqual(read(run/'state-before-ilkl-g110-x1.5.json'),state)
                self.assertEqual(read(root/target),insolvency_event_config(state['signature']['config']))
            self.assertEqual(checkpoint.read_bytes(),b'unchanged')

    def test_new_reward_is_explicit_and_old_cap_migration_stays_exact(self):
        old=hotfix_state()['signature']['config'];new=validate_config(insolvency_event_config(old))
        self.assertEqual(new['insolvency'],dict(event_cost=.0075,max_cost=.075))
        self.assertEqual({k for k in old if old[k]!=new[k]},{'insolvency','search'})
        self.assertTrue(event_cost_upgrade(reward_spec(old),reward_spec(new)))
        self.assertEqual(new['il_kl_limit']['initial_limit'],.15)
        oldest=copy.deepcopy(old);oldest['insolvency']['max_cost']=.025
        oldest['search']['reward']['insolvency']['max_cost']=.025
        capped=champion_mix_config(oldest)
        self.assertEqual(capped['insolvency'],old['insolvency'])
        self.assertTrue(cap_upgrade(reward_spec(oldest),reward_spec(capped)))
        invalid=copy.deepcopy(new);invalid['reward']['vp']*=2
        self.assertFalse(event_cost_upgrade(reward_spec(old),reward_spec(invalid)))

    def test_new_cost_reaches_unchanged_cap_after_ten_events(self):
        from brass_ai.learning.reward import CostLedger
        ledger=CostLedger(reward_spec(config()))
        ledger.charge(0);self.assertAlmostEqual(ledger.spent[0],.0075)
        for _ in range(9):ledger.charge(0)
        self.assertAlmostEqual(ledger.spent[0],.075)
        for _ in range(10):ledger.charge(0)
        self.assertAlmostEqual(ledger.spent[0],.075)

    def test_actual_g110_signature_allows_only_reviewed_reward_change(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        path=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0110-20261019T233943412893Z/state.json'
        state=read(path);signature=copy.deepcopy(state['signature'])
        signature.update(implementation=implementation_id(),config=insolvency_event_config(signature['config']))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        self.assertTrue(check_signature(state,signature,reviewed=True))
        bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)

    def test_reward_migration_preserves_weights_Adam_and_RNG(self):
        model,opt,cfg,_=fixture();old=hotfix_state()['signature']['config'];before_spec=reward_spec(old)
        model.config=replace(model.config,utility=before_spec)
        # Populate real Adam moments rather than checking only an empty state.
        p=next(model.parameters());p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
        weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,'explicit reward migration'):migrate(model,opt,reward_spec(cfg))
        model._reviewed_cap_upgrade=True;report=migrate(model,opt,reward_spec(cfg))
        self.assertEqual(report['version'],'insolvency-event-cost/1.0.0')
        self.assertTrue(_same_payload(weights,model.state_dict()));self.assertTrue(_same_payload(adam,opt.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertIsNone(model.utility_calibration)

    def test_old_reward_cache_excluded_once_and_deleted_only_after_next_commit(self):
        model,_,cfg,_=fixture();cfg['value_phase']['buffer_iterations']=2
        old=copy.deepcopy(cfg);old['insolvency']=dict(event_cost=.005,max_cost=.075)
        old['search']['reward']['insolvency']=copy.deepcopy(old['insolvency'])
        model.config=replace(model.config,utility=reward_spec(old))
        behavior,data=examples(model);ds=updates.Dataset(data,'cpu');ids,_,_=value_phase.split(data,.05)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp)/'run';buf=value_buffer.DiskBuffer(SOURCE_ROOT,run,old,generation=0)
            buf.stage(ds,ids,source(behavior),model);receipt=copy.deepcopy(buf.pending)
            write_json(run/'state.json',dict(iteration=1));buf.committed(1);old_sha=sha(buf.path(1,'npy'))
            with self.assertRaisesRegex(ValueError,'Incompatible PPG'):
                value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=1,receipt=receipt)
            new=value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=1,receipt=receipt,previous_reward=reward_spec(old))
            self.assertIsNone(new.receipt);self.assertEqual(sha(buf.path(1,'npy')),old_sha)
            with new.previous() as previous:self.assertIsNone(previous)
            model.config=replace(model.config,utility=reward_spec(cfg))
            behavior,data=examples(model,prefix='next-');ds=updates.Dataset(data,'cpu');ids,_,_=value_phase.split(data,.05)
            report=new.stage(ds,ids,source(behavior),model)
            self.assertEqual(report['previous_rows'],0);self.assertEqual(report['reward_transition']['excluded_previous_rows'],receipt['rows'])
            self.assertEqual(report['excluded_cache_bytes'],receipt['bytes'])
            self.assertEqual(report['peak_cache_bytes'],report['current_cache_bytes']+receipt['bytes'])
            self.assertTrue(buf.path(1,'npy').exists())
            write_json(run/'state.json',dict(iteration=2));new.committed(2)
            self.assertFalse(buf.path(1,'npy').exists())
            restored=value_buffer.DiskBuffer(SOURCE_ROOT,run,cfg,generation=2,receipt=new.receipt)
            with restored.previous() as previous:self.assertEqual(len(previous),len(ids))


if __name__=='__main__':unittest.main()
