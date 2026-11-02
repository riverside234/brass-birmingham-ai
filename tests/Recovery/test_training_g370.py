"""g370 continuation: unchanged entropy, direct GAE, T2 forks and beta floor."""
import copy
import importlib.util
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import gae,exploration as E,branching as B,ppg_g210 as G,critic_data
from brass_ai.recovery.training_revision_g370 import configure
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature,gae_floor_receipt_preserved
from brass_ai.step5.league import implementation_id,write_json
from test_ppg_window import tick

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g0370-20261102T221517838215Z'


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.state=read(BUNDLE/'state.json');cls.old=cls.state['signature']['config'];cls.new=configure(cls.old)

    def test_exact_requested_settings_and_strict_validation(self):
        validate_config(self.old);validate_config(self.new)
        self.assertEqual(configure(self.new),self.new)
        self.assertEqual({k for k in self.old if self.new[k]!=self.old[k]},{'branching','advantage','value_phase'})
        self.assertEqual(self.new['update'],self.old['update'])
        self.assertEqual(self.new['update']['fixed_entropy']['weights'],dict.fromkeys(E.GROUPS,.005))
        self.assertEqual(self.new['opening'],E.THREE_OPENING)
        self.assertEqual(self.new['exploration'],.05)
        c=self.new['value_phase']
        self.assertEqual((c['beta_clone_min'],c['acceptance_steps'],c['kl_check_steps'],c['buffer_iterations'],c['phase_interval']),(1.,16,32,2,1))
        self.assertEqual([G.rate_factor(i,200,c) for i in range(1,201)],
            [G.rate_factor(i,200,self.old['value_phase']) for i in range(1,201)])
        for section,key,value in [('branching','rate_per_decision',.01),('branching','move_distribution',{'temperature_1':1.}),
                ('advantage','lambda',.98),('advantage','lambda_floor',.98),('value_phase','beta_clone_min',.5),
                ('update','entropy_weight',.01)]:
            bad=copy.deepcopy(self.new);bad[section][key]=value
            with self.assertRaises(ValueError):validate_config(bad)

    def test_direct_lambda_without_gate_and_independent_evidence_preserved(self):
        s=gae.initial(self.old['advantage'],370)
        s.update(value_iterations=120,stage_origin=253,table_enabled=dict(historical=True,legacy=False),
            table_origins=dict(historical=21,legacy=None))
        before=copy.deepcopy(s)
        new,receipt=gae.migrate_direct_095(s,self.old['advantage'],self.new['advantage'],370)
        self.assertEqual(s,before);self.assertEqual((receipt['lambda_before'],receipt['lambda_after']),(.98,.95))
        for k in s:
            if k not in ('active_lambda','stage_origin','healthy_streak'):self.assertEqual(new[k],s[k],k)
        self.assertEqual(new['stage_origin'],370)
        for g in range(371,401):
            new,r=tick(new,self.new['advantage'],healthy=False,rating=False,steps=0)
            self.assertEqual((r['lambda_used'],r['lambda_next'],r['lambda_floor']),(.95,.95,.95))
            self.assertEqual(r['table_gate']['historical']['lambda_used'],.95)
            self.assertEqual(r['table_gate']['legacy']['lambda_used'],1.)
        with self.assertRaises(ValueError):gae.validate_state(before,self.new['advantage'],370)

    def test_t2_only_sampling_legal_support_probability_and_replay_seed(self):
        raw=np.array([.5,.4,.1,0.]);q=np.array([0.,0.,0.,1.]);contract=E.T2_ONLY_BRANCH['contract']
        p=np.sqrt(raw);p[0]=0;p/=p.sum()
        for seed in range(50):
            a=B.alternative(raw,q,0,random.Random(seed),contract=contract)
            self.assertEqual(a,B.alternative(raw,q,0,random.Random(seed),contract=contract))
            self.assertEqual((a['component'],a['component_probability']),('temperature_2',1.))
            self.assertIn(a['chosen'],(1,2));self.assertAlmostEqual(a['probability'],p[a['chosen']])
            self.assertAlmostEqual(a['marginal_probability'],a['probability'])
        self.assertIsNone(B.alternative(np.array([1.,0.]),np.array([0.,1.]),0,random.Random(1),contract=contract))
        self.assertNotEqual(B.seed_for('p',4,contract),B.seed_for('p',4,E.NEUTRAL_BRANCH['contract']))

    def test_branch_opening_reset_and_effective_settings(self):
        from test_training_g1120 import Revision as Openings
        with patch.object(E,'THREE_BRANCH',E.T2_ONLY_BRANCH):
            Openings().test_fork_skips_three_and_resets_all_four_seats()
        s=E.restore(self.new,self.state,copy.deepcopy(self.state['exploration']))
        result=E.effective(self.new,s,370)
        self.assertEqual(result['alphas'],dict.fromkeys(E.GROUPS,.005))
        self.assertEqual(result['branching'],E.T2_ONLY_BRANCH)
        self.assertEqual(result['opening'],E.THREE_OPENING)
        self.assertEqual(s,self.state['exploration'])

    def test_beta_feedback_never_drops_below_one_and_keeps_KL_increases(self):
        c=self.new['value_phase'];beta=2.;trips=[]
        for expected in (1.,1.,1.):
            beta,trips,r=G.beta_after_phase(beta,trips,dict(optimizer_steps=1,kl_guard_tripped=False,probe_kl=.0001),c)
            self.assertEqual(beta,expected)
        beta,trips,r=G.beta_after_phase(1.,[True,True],dict(optimizer_steps=1,kl_guard_tripped=True,probe_kl=.003),c)
        self.assertEqual(beta,2.)

    def test_real_two_age_disk_cache_adam_rng_restart_and_bounded_ppg_smoke(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.utility import migrate,reward_spec
        from brass_ai.learning.batching import model_id
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        from test_ppg_g210 import prepare
        old=copy.deepcopy(self.old);old['model_size']='small';old['update']['preparation_workers']=0
        old['value_phase'].update(batch_size=128,clone_rows=24,clone_batch_size=8)
        model=FourSeatNet(ModelConfig.preset('small'))
        opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'],weight_decay=old['weight_decay'])
        migrate(model,opt,reward_spec(old));new=configure(old)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            aux=Companion(model,old['local_scoring_auxiliary'],raw=dict(model_id=model_id(model)))
            p.local_scoring=aux
            for g in (1,2):
                prepare(model,p,g,companion=True)
                p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,p.actor_ids,p.collection_source,model,p.split['selection'])
                write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
                p.state=gae.initial(old['advantage'],g);p.critic_state=dict(critic_data.initial(old['critic_data']),generation=g)
                p.release()
            for optimizer in (opt,p.optimizer):
                next(model.parameters()).grad=torch.zeros_like(next(model.parameters()));optimizer.step();optimizer.zero_grad(set_to_none=True)
            p.buffer.pending=copy.deepcopy(p.buffer.receipts);p.trips=[True,False]
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            for beta in (.5,1.,4.):
                p.beta=beta;raw=dict(model_config=model.config.to_dict(),training={KEY:copy.deepcopy(p.payload())})
                original=copy.deepcopy(raw)
                before=copy.deepcopy((model.state_dict(),opt.state_dict(),p.optimizer.state_dict(),torch.get_rng_state()))
                q=TrainingPhases(model,new,raw=raw,generation=2,root=SOURCE_ROOT,run=run)
                self.assertEqual(q.beta,max(1.,beta));self.assertEqual(q.trips,p.trips);self.assertEqual(q.state['active_lambda'],.95)
                self.assertEqual(q.buffer.receipts,p.buffer.receipts);self.assertEqual(q.critic_state,p.critic_state)
                self.assertIsNone(q.startup(2));self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
                self.assertTrue(_same_payload(raw,original))
                self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),torch.get_rng_state())))
                q.buffer.pending=copy.deepcopy(q.buffer.receipts);saved=q.payload();q.buffer.pending=None
                again=TrainingPhases(model,new,raw=dict(training={KEY:saved}),generation=2,root=SOURCE_ROOT,run=run)
                self.assertEqual(again.state,q.state);self.assertEqual(again.beta,q.beta)
                self.assertIsNone(again.gae_floor_migration);self.assertIsNone(again.ppg_schedule_migration)
                self.assertTrue(_same_payload(again.optimizer.state_dict(),q.optimizer.state_dict()))
            q.local_scoring=aux;prepare(model,q,3,companion=True)
            result=q.fit(model)
            self.assertEqual(result['beta_clone'],4.)
            self.assertGreater(result['attempted_steps'],0)
            self.assertLessEqual(result['optimizer_steps'],result['attempted_steps'])
            self.assertEqual(q.options['value_phase']['kl_cap'],.002)
            print('g370 bounded PPG smoke:',{k:result.get(k) for k in ('optimizer_steps','attempted_steps','probe_kl','early_stop_reason')},flush=True)

    def test_source_review_and_precommit_retry_preserve_calibration(self):
        s=copy.deepcopy(self.state);target=dict(s['signature'],implementation=implementation_id(),config=self.new)
        self.assertTrue(check_signature(s,target,reviewed=True))
        with self.assertRaises(ValueError):check_signature(s,target)
        bad=copy.deepcopy(target);bad['config']['exploration']=.08
        with self.assertRaises(ValueError):check_signature(s,bad,reviewed=True)
        entry=s['history'][-1];raw=dict(training=dict(signature=digest(s['signature'])))
        self.assertTrue(gae_floor_receipt_preserved(s,target,raw,entry));self.assertFalse(gae_floor_receipt_preserved(s,bad,raw,entry))
        s.setdefault('source_transitions',[]).append(dict(checkpoint=entry,from_signature=s['signature'],to_signature=digest(target)))
        s['signature']=target
        self.assertFalse(check_signature(s,target,reviewed=True));self.assertTrue(gae_floor_receipt_preserved(s,target,raw,entry))

    def test_config_helper_retry_readonly_and_integrity_checks(self):
        spec=importlib.util.spec_from_file_location('g370_helper',SOURCE_ROOT/'scripts/configure-training-g370.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-0370.pt';checkpoint.write_bytes(b'authored checksum fixture')
            s=copy.deepcopy(self.state);s['history'][-1].update(path='artifacts/run/generation-0370.pt',sha256=sha(checkpoint))
            write_json(run/'state.json',s);before=sha(run/'state.json');output='configs/step5/g370.json'
            a=helper.generate(root,'artifacts/run',output,370)
            self.assertEqual(a,helper.generate(root,'artifacts/run',output,370))
            self.assertEqual(read(root/output),self.new);self.assertEqual(before,sha(run/'state.json'))
            with self.assertRaisesRegex(ValueError,'Stale'):helper.generate(root,'artifacts/run',output,369)
            checkpoint.unlink()
            with self.assertRaisesRegex(ValueError,'missing/changed'):helper.generate(root,'artifacts/run',output,370)


if __name__=='__main__':unittest.main()
