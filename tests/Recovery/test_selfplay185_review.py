"""Sales are diagnostics only; admitted g185 continuation retains active state."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import il_kl_limit as limits,il_kl_probe as probes,lr_controller as lr
from brass_ai.recovery.config import latest_model_config,validate_config
from brass_ai.recovery.history import compact
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id
import test_auto_il_kl as auto_tests
from test_auto_il_kl import baseline,counts,measurement,ToyPolicy
from test_ppo_lr_controller import measurements

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0185-20261020T090519451283Z'


def legacy_state():
    s=baseline()
    s['contract']=s['config']['contract']=limits.LEGACY_CONTRACT
    s['config']['sells_ratio']=.7
    s['baseline']['metrics']['sells_per_seat']=2.
    s['baseline']['thresholds']['sells_per_seat']=1.4
    return s


class SaleRemoval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_new_contract_and_old_signed_configs_are_strict(self):
        self.assertNotIn('sells_ratio',limits.DEFAULT)
        cfg=read(SOURCE_ROOT/'configs/step5/joint7-latest.json')
        self.assertEqual(cfg['il_kl_limit'],limits.DEFAULT);validate_config(cfg)
        old=read(EXPORT/'recorded-config.json');before=copy.deepcopy(old)
        validate_config(old)
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(fresh['il_kl_limit'],limits.current_config(old['il_kl_limit']))
        self.assertEqual(old,before)
        for value in (dict(limits.DEFAULT,sells_ratio=.7),dict(limits.DEFAULT,extra=1),
                      dict(limits.DEFAULT,contract=limits.LEGACY_CONTRACT)):
            with self.assertRaisesRegex(ValueError,'contract/fields'):limits.validate(value)
        changed=copy.deepcopy(old);changed['il_kl_limit']['pass_factor']=0
        with self.assertRaises(ValueError):validate_config(changed)

    def test_sales_cannot_change_limit_health_or_reports_even_when_missing(self):
        original=legacy_state();before=copy.deepcopy(original);reference=None
        for sale in (0,1,999999,None,-1,float('nan'),'unused','missing'):
            s=copy.deepcopy(original)
            # All archived/current sales inputs are deliberately adversarial.
            for row in s['baseline_rows']+s['rolling']:
                row['sells']=sale
            s['baseline']['metrics']['sells_per_seat']=sale
            s['baseline']['thresholds']['sells_per_seat']=sale
            current=counts(sells=sale)
            if sale=='missing':current.pop('sells')
            observed=measurement(behavior=dict(ok=False,sales=dict(ok=False),passing=dict(ok=True)))
            new,r=limits.advance(s,observed,current,generation=11,warmup_done=True)
            self.assertTrue(r['health']['ok']);self.assertTrue(r['lr_gate']['ok'])
            self.assertEqual(set(r['health']['checks']),set(limits.METRICS))
            self.assertNotIn('sells',json.dumps(r['health']))
            self.assertNotIn('sells',json.dumps(r['baseline']))
            self.assertNotIn('sales',json.dumps(r['measurement']))
            json.dumps(new,allow_nan=False)
            if reference is None:reference=(new,r)
            else:self.assertEqual((new,r),reference)
        self.assertEqual(original,before)

    def test_existing_non_sale_guards_and_baseline_tamper_checks_remain(self):
        s=legacy_state()
        for changes in (dict(vp_sum=0),dict(passes_with_alternative=1000),
                        dict(zero_vp_seats=20),dict(insolvent_seats=80),dict(vp_sum=float('nan'))):
            rows=[counts(sells=999999,**changes)]*5
            self.assertFalse(limits.health(s['baseline'],limits.rates(rows),dict(passing=dict(ok=True)))['ok'])
        self.assertTrue(limits.health(s['baseline'],limits.rates([counts(sells=0)]*5),
                                      dict(ok=True,passing=dict(ok=False)))['ok'])
        for key in limits.METRICS:
            bad=copy.deepcopy(s);bad['baseline']['thresholds'][key]+=1
            with self.assertRaisesRegex(ValueError,'baseline differs'):
                limits.validate_state(bad,bad['config'])

    def test_shared_evidence_reaches_lr_gae_snapshot_and_both_pool_samplers(self):
        from test_selfplay55_review import phase_fixture
        from test_snapshot_weight_recovery import active
        from test_quality_pool import pool as old_pool,entry
        from test_pfsp_pool import pool as pfsp_pool
        from brass_ai.recovery import quality_pool
        s=legacy_state()
        _,r=limits.advance(s,measurement(behavior=dict(ok=False,sales=dict(ok=False),passing=dict(ok=False))),
                           counts(sells=0),generation=11,warmup_done=True)
        control=lr.initial_state(lr.AUTO_CONTROLLER)
        for _ in range(3):control,d=lr.decide(control,measurements(kl=.001,calibrated_anchor=r['lr_gate']))
        self.assertEqual(d['decision'],'up')
        phase,update=phase_fixture(85);_,phase.critic_state=active()
        phase.state['active_lambda']=.98
        update['critic_rows']=dict(snapshot_distinct_used=8)
        gate=phase.finish(generation=86,update=update,anchor_gate=r['lr_gate'],rating=dict(ok=True))
        self.assertTrue(gate['healthy']);self.assertEqual(gate['healthy_streak'],1)
        self.assertEqual(phase.critic_state['weight_recovery']['healthy_streak'],1)
        for pool in (old_pool(),pfsp_pool()):
            pool['generation']=100;pool['health_window']=[True]*5
            distances={m['model_id']:.1 for m in pool['members']}
            _,event=quality_pool.admission(pool,entry(100),r['health'],distances,lambda a,b:.1)
            self.assertTrue(event['admitted'],event)
        self.assertEqual(compact(dict(il_kl_limit=r))['il_kl_limit'],r)

    def test_real_g185_evidence_advances_without_rebaseline_or_limit_reset(self):
        state=read(EXPORT/'state.json');s=state['il_kl_limit'];before=copy.deepcopy(s)
        report=read(EXPORT/'iteration-0184/report.json')
        limits.validate_state(s,state['signature']['config']['il_kl_limit'],185)
        current=dict(report['health_counts'],sells=0)
        new,r=limits.advance(s,report['il_kl_limit']['measurement'],current,generation=186,warmup_done=True)
        self.assertTrue(r['health']['ok']);self.assertEqual(s,before)
        self.assertEqual(new['baseline'],limits.baseline_evidence(s['baseline']))
        self.assertEqual((new['limit'],new['last_update'],new['blocked']),
                         (s['limit'],s['last_update'],s['blocked']))
        self.assertEqual((new['probe'],new['config']),(s['probe'],s['config']))
        self.assertEqual(new['last_generation'],186)
        restored=json.loads(json.dumps(new,allow_nan=False))
        self.assertEqual(limits.advance(new,measurement(.41),current,generation=187,warmup_done=True),
                         limits.advance(restored,measurement(.41),current,generation=187,warmup_done=True))

    def test_reviewed_g185_signature_keeps_all_settings_and_freshness_rules(self):
        state=read(EXPORT/'state.json');before=copy.deepcopy(state)
        signature=copy.deepcopy(state['signature']);signature['implementation']=implementation_id()
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        self.assertTrue(check_signature(state,signature,True))
        for key,change in (('learning_rate',1e-5),('teacher_enabled',True)):
            bad=copy.deepcopy(signature);bad['config'][key]=change
            with self.assertRaisesRegex(ValueError,'unchanged config'):check_signature(state,bad,True)
        self.assertEqual(state,before)
        self.assertEqual((state['iteration'],state['games_done']),(185,59200))

    def test_old_probe_reused_without_loading_shards_or_changing_identity(self):
        rows=auto_tests.ProbeTests().data();model=ToyPolicy();old=legacy_state()['config']
        update=dict(microbatch_size=3,max_candidate_cells=100)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);data=root/'validation';data.write_bytes(b'fixed validation')
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(entries=[dict(split='validation',path='validation',sha256=sha(data))])))
            origin=dict(manifest='manifest.json',manifest_sha256=sha(manifest),model_id='a'*64,sha256='b'*64)
            with patch('brass_ai.recovery.demonstrations.load_dataset',return_value=(rows,{})):
                probe,receipt=probes.load(root,root/'run',origin,model,old,update,1)
            hashes={p:sha(p) for p in (root/receipt['path'],root/probe.metadata['cache']['path'])}
            with patch('brass_ai.recovery.demonstrations.load_dataset',side_effect=AssertionError('reloaded')):
                for config in (old,limits.current_config(old)):
                    loaded,again=probes.load(root,root/'run',origin,model,config,update,1,receipt)
                    self.assertEqual(again,receipt)
                    self.assertEqual(loaded.measure(model,update)['kl_mean'],0.)
            self.assertEqual({p:sha(p) for p in hashes},hashes)
            for changes in (dict(probe_seed=777),dict(pass_allowance=.7)):
                with self.assertRaisesRegex(ValueError,'source/config differs'):
                    probes.load(root,root/'run',origin,model,dict(limits.DEFAULT,**changes),update,1,receipt)

    def test_retired_threshold_and_active_threshold_distinction(self):
        current=counts(sells=0)
        results=[]
        for ratio in (.01,.7,1.):
            s=legacy_state();s['config']['sells_ratio']=ratio
            _,r=limits.advance(s,measurement(),current,generation=11,warmup_done=True)
            results.append(r)
        self.assertEqual(results[0],results[1]);self.assertEqual(results[1],results[2])
        s=legacy_state();s['blocked']=True
        new,r=limits.advance(s,measurement(.14),current,generation=11,warmup_done=True)
        self.assertTrue(new['blocked']);self.assertFalse(r['lr_gate']['ok'])
        self.assertEqual(r['lr_gate']['reasons'],['anchor_recovery_latch'])


if __name__=='__main__':unittest.main()
