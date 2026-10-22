"""g503 explicit reward/fork/fixed-critic adoption and restart invariants."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.reward import CostLedger
from brass_ai.recovery import critic_data as C,exploration as E,snapshot_r2 as R,value_phase
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import (insolvency_010_config,insolvency_event_config,
    latest_model_config,validate_config)
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import migrate,reward_spec,event_cost_upgrade
from brass_ai.recovery.history import compact

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0503-20261022T051624803527Z'
spec=importlib.util.spec_from_file_location('joint7_g503',SOURCE_ROOT/'scripts/joint7-g503.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


class Review503(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.state=read(EXPORT/'state.json');cls.old=cls.state['signature']['config']
        cls.config=helper.configure(cls.old)

    def test_exact_requested_settings_and_baselines_unchanged(self):
        old=copy.deepcopy(self.old);state=copy.deepcopy(self.state);c=self.config
        self.assertEqual({k for k in old if old[k]!=c[k]},{'insolvency','search','branching','critic_data'})
        self.assertEqual(c['insolvency'],dict(event_cost=.010,max_cost=.075))
        self.assertEqual(c['search']['reward']['insolvency'],c['insolvency'])
        self.assertEqual(c['branching'],E.HIGH_BRANCH)
        self.assertEqual(c['critic_data'],C.FIXED_DEFAULT)
        self.assertEqual(c['entropy_control'],old['entropy_control'])
        restored=E.restore(c,state,state['exploration'])
        self.assertEqual(restored,state['exploration'])
        self.assertEqual(restored['baseline'],state['exploration']['baseline'])
        self.assertEqual(E.floors(c,restored,503),E.floors(old,state['exploration'],503))
        self.assertEqual(state,self.state);self.assertEqual(old,self.old)
        self.assertEqual(helper.configure(c),c)
        self.assertFalse(c['teacher_enabled']);self.assertFalse(c['value_phase']['enabled'])

    def test_fresh_generation_has_new_defaults_but_old_migrations_stay_exact(self):
        fresh=latest_model_config(self.old,self.old['imitation_checkpoint'])
        self.assertEqual(fresh['insolvency'],dict(event_cost=.020,max_cost=.080))
        self.assertEqual(fresh['branching'],E.HIGH_BRANCH)
        self.assertEqual(fresh['critic_data'],C.FIXED_DEFAULT)
        self.assertEqual(fresh['entropy_control']['promotion_origin']['generation'],0)
        old=copy.deepcopy(self.old);old['insolvency']['event_cost']=.005
        old['search']['reward']['insolvency']['event_cost']=.005
        self.assertEqual(insolvency_event_config(old)['insolvency']['event_cost'],.0075)
        self.assertEqual(C.floor_config(self.config),self.config)
        for field,value in [('snapshot_weight',.5),('snapshot_weight',True),('snapshot_rows',False)]:
            bad=copy.deepcopy(self.config);bad['critic_data'][field]=value
            with self.assertRaisesRegex(ValueError,'Fixed snapshot'):validate_config(bad)

    def test_new_cost_caps_on_eighth_event_and_leaves_official_targets_alone(self):
        ledger=CostLedger(reward_spec(self.config))
        for i in range(12):
            ledger.charge(0)
            self.assertAlmostEqual(ledger.spent[0],min((i+1)*.010,.075))
        self.assertEqual(ledger.spent[1:].tolist(), [0.,0.,0.])
        self.assertTrue(event_cost_upgrade(reward_spec(self.old),reward_spec(self.config)))
        bad=copy.deepcopy(self.config);bad['reward']['vp']*=2
        self.assertFalse(event_cost_upgrade(reward_spec(self.old),reward_spec(bad)))
        from test_insolvency_recovery import row
        from brass_ai.recovery.insolvency import returns
        rows=[row(),row(phase='debt',kind='liquidate',debts={0:5})]
        original=copy.deepcopy(rows)
        costs,_=returns(rows,self.config['insolvency'])
        self.assertEqual(costs,[.010,0]);self.assertEqual(rows,original)

    def test_checkpoint_transition_preserves_weights_both_adams_gae_and_rng(self):
        from test_gae_value_phase import fixture
        model,opt,cfg,_=fixture()
        cfg=C.floor_config(value_phase.disabled_options(cfg));old=TrainingPhases(model,cfg)
        old.critic_state.update(snapshot_weight=0.,snapshots_active=True,activation_generation=0,
            weight_recovery=R.initial(0,0.))
        p=next(model.parameters());p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
        p.grad=torch.ones_like(p);old.optimizer.step();old.optimizer.zero_grad(set_to_none=True)
        weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
        saved=old.payload();before=copy.deepcopy(saved);rng=torch.get_rng_state().clone()
        target=C.fixed_config(insolvency_010_config(cfg))
        with self.assertRaisesRegex(ValueError,'explicit reward migration'):migrate(model,opt,reward_spec(target))
        model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(target))
        phase=TrainingPhases(model,target,raw={'training':{KEY:saved}})
        self.assertEqual(phase.critic_state['snapshot_weight'],1.)
        self.assertTrue(phase.critic_state['snapshots_active'])
        self.assertEqual(phase.critic_state['weight_recovery']['previous']['state'],before['critic_data'])
        self.assertEqual(phase.state,before['gae'])
        self.assertTrue(_same_payload(phase.optimizer.state_dict(),before['optimizer']))
        self.assertTrue(_same_payload(model.state_dict(),weights));self.assertTrue(_same_payload(opt.state_dict(),adam))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(_same_payload(saved,before))
        again=TrainingPhases(model,target,raw={'training':{KEY:phase.payload()}})
        self.assertIsNone(again.snapshot_fixed_migration)
        self.assertTrue(_same_payload(again.payload(),phase.payload()))

    def test_fixed_mode_never_recovers_or_decays_and_keeps_diagnostics(self):
        from test_snapshot_r2 import value_report
        c=self.config['critic_data'];s=C.initial(c)
        for generation in range(1,45):
            with patch.object(R,'advance',side_effect=AssertionError('Disabled controller ran')):
                s,report=C.advance(s,c,generation=generation,report=value_report(generation,bias=2.),
                    healthy=False,fingerprint_now=None,ppo_snapshot_rows=5)
            self.assertEqual(s['snapshot_weight'],1.);self.assertTrue(s['snapshots_active'])
            self.assertIsNone(report['weight_recovery']);self.assertEqual(report['weight_control'],'fixed')
            self.assertLess(report['learner_r2'],0.)
        self.assertEqual(s['weight_recovery'],dict(contract=C.FIXED_STATE,origin=0,previous=None))
        self.assertEqual(compact(dict(value_phase={'critic_rollout':report}))['value_phase']['critic_rollout'],report)

    def test_real_ppo_uses_snapshots_immediately_and_keeps_gae_active(self):
        from test_critic_descriptors import mixed_fixture
        from test_gae_value_phase import ppo
        from test_ppg_window import upgraded
        model,opt,cfg,behavior,data,snap,_=mixed_fixture()
        cfg=C.floor_config(value_phase.disabled_options(upgraded(cfg)))
        old=TrainingPhases(model,cfg)
        old.critic_state.update(snapshot_weight=0.,weight_recovery=R.initial(0,0.))
        cfg=C.fixed_config(cfg);phase=TrainingPhases(model,cfg,raw={'training':{KEY:old.payload()}})
        phase.snapshot_data=snap
        report=ppo(model,opt,cfg,behavior,data,phase)
        self.assertGreater(report['critic_rows']['snapshot_distinct_used'],0)
        self.assertEqual(report['critic_rows']['snapshot_weight'],1.)
        self.assertTrue(all(float(phase.critic_dataset.critic_weights[i])==1. for i in phase.train_ids))
        held={phase.critic_dataset.data[i][2][0] for i in phase.heldout_ids}
        train={phase.critic_dataset.data[i][2][0] for i in phase.train_ids}
        self.assertTrue(held.isdisjoint(train))
        value=phase.fit(model);self.assertFalse(value['enabled'])
        phase.finish(generation=1,update=report,anchor_gate={'ok':True},rating={'ok':True})
        self.assertEqual(phase.state['generation'],1)
        self.assertEqual(value['critic_rollout']['weight_next'],1.)
        self.assertEqual(report['gae']['gate']['critic_training']['source'],'ppo')

    def test_archived_r2_hotfix_accepts_future_fixed_checkpoints(self):
        from brass_ai.recovery.snapshot_r2_hotfix import restore
        from types import SimpleNamespace
        state=copy.deepcopy(self.state);raw=dict(step=503,training={KEY:{'critic_data':C.initial(C.FIXED_DEFAULT)}})
        phase=SimpleNamespace(config=self.config,critic_state=copy.deepcopy(raw['training'][KEY]['critic_data']))
        before=copy.deepcopy(phase.critic_state)
        self.assertEqual(restore(state,raw,phase),state['snapshot_r2_hotfix'])
        self.assertEqual(phase.critic_state,before)

    def test_disabled_r2_preview_does_not_seed_or_load_old_reports(self):
        from brass_ai.recovery.snapshot_r2_hotfix import preview
        from brass_ai.step5.league import write_json
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            folder=Path(tmp)
            write_json(folder/'state.json',dict(self.state,signature=dict(self.state['signature'],config=self.config)))
            with patch('brass_ai.recovery.snapshot_r2_hotfix.reports',side_effect=AssertionError('Disabled controller reports')):
                result=preview(folder)
            self.assertTrue(result['disabled']);self.assertEqual(result['configured_weight'],1.)

    def test_actual_g503_r2_history_archived_without_resetting_its_reference(self):
        history=read(EXPORT/'retention-history.json')
        reports={v['summary']['iteration']+1:v['summary'] for v in history.values()}
        for path in EXPORT.glob('iteration-*/report.json'):
            r=read(path);reports[r['iteration']+1]=r
        latest=reports[503]['value_phase']['critic_rollout']['weight_recovery']
        old=C.initial(self.old['critic_data'])
        recovery=R.initial(213,0.)
        recovery.update(stage_origin=471,last_change_generation=471,
            reference=copy.deepcopy(latest['reference_next']),
            regime=R.regime(reports[503]['value_phase'],reports[503]['actor'],reports[503]['exploration_rollout']),
            entries=[R.entry(g,reports[g]['value_phase'],healthy=True,trained=True) for g in range(484,504)])
        old.update(generation=503,snapshot_weight=0.,snapshots_active=True,activation_generation=1,weight_recovery=recovery)
        C.validate_state(old,self.old['critic_data'],503)
        new=C.fix_weight(old,self.old['critic_data'],self.config['critic_data'],503)
        self.assertEqual(new['weight_recovery']['previous']['state'],old)
        self.assertEqual(new['weight_recovery']['previous']['state']['weight_recovery']['reference'],latest['reference_next'])
        self.assertEqual(new['weight_recovery']['origin'],503)

    def test_signature_requires_review_and_rejects_unrelated_changes(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        signature=dict(self.state['signature'],config=self.config,implementation=implementation_id())
        self.assertTrue(check_signature(self.state,signature,reviewed=True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(self.state,signature)
        for key in ('learning_rate','exploration'):
            bad=copy.deepcopy(signature);bad['config'][key]*=2
            with self.assertRaises(ValueError):check_signature(self.state,bad,reviewed=True)

    def test_config_helper_is_repeatable_and_never_changes_run(self):
        from brass_ai.step5.league import write_json
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True)
            write_json(run/'state.json',self.state);before=(run/'state.json').read_bytes()
            output='configs/step5/review503.json'
            for _ in range(2):
                result=helper.generate(root,run,output,503)
                self.assertFalse(result['run_modified'])
                self.assertEqual(read(root/output),self.config)
                self.assertEqual((run/'state.json').read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'Unexpected committed'):helper.generate(root,run,output,502)

    def test_actual_resume_archives_partial_collection_and_survives_precommit_restart(self):
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.learning.contracts import digest
        from brass_ai.bank import sha
        from brass_ai.step5.league import implementation_id,write_json
        from brass_ai.recovery.resume import load_resume,commit_transition
        from brass_ai.recovery.interrupted import archive_uncommitted_iteration
        from test_gae_value_phase import fixture
        model,opt,_,_=fixture()
        old=copy.deepcopy(self.old);old['model_size']='small'
        signature=dict(copy.deepcopy(self.state['signature']),config=old)
        phase=TrainingPhases(model,old)
        phase.state['generation']=1
        phase.state['table_evidence']['origin']=1
        phase.state['stage_evidence']['origin']=1
        phase.critic_state.update(generation=1,snapshot_weight=0.,snapshots_active=True,
            activation_generation=1,weight_recovery=R.initial(1,0.))
        # Nonempty PPO/value Adam states make preservation assertions meaningful.
        p=next(model.parameters());p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
        p.grad=torch.ones_like(p);phase.optimizer.step();phase.optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/run';run.mkdir(parents=True)
            path=run/'generation-0001.pt';relative=run.relative_to(root).as_posix()
            mid=save_checkpoint(path,model,opt,step=1,training={KEY:phase.payload(),
                'signature':digest(signature),'fresh_run':relative})
            original=sha(path);entry=dict(path=path.relative_to(root).as_posix(),sha256=original,model_id=mid,generation=1)
            state=dict(format='step5-recovery-league/2.2.0',iteration=1,games_done=320,
                signature=signature,history=[entry],population=[],evaluation_anchor=entry)
            write_json(run/'state.json',state)
            pending=run/'iteration-0001';pending.mkdir();(pending/'partial.bin').write_bytes(b'partial-evidence')
            new=helper.configure(old);target=dict(signature,config=new,implementation=implementation_id())
            with self.assertRaisesRegex(ValueError,'Uncommitted iteration'):load_resume(root,run,state,target,reviewed=True)
            result=archive_uncommitted_iteration(root,run,expected_iteration=1,apply=True)
            self.assertEqual(result['status'],'archived');self.assertEqual(read(run/'state.json'),state)
            self.assertEqual((run/result['archive']/'iteration-0001/partial.bin').read_bytes(),b'partial-evidence')
            loaded,raw,_=load_resume(root,run,state,target,reviewed=True)
            other=torch.optim.AdamW(loaded.parameters());resume_optimizer(raw,other)
            before=copy.deepcopy(other.state_dict());migrate(loaded,other,reward_spec(new))
            first=TrainingPhases(loaded,new,raw=raw,generation=1)
            self.assertEqual(first.critic_state['snapshot_weight'],1.)
            self.assertTrue(_same_payload(before,other.state_dict()))
            state=commit_transition(run,state,target)
            # Restart after source publication, before the first new model commit.
            again,raw,_=load_resume(root,run,state,target,reviewed=True)
            last=torch.optim.AdamW(again.parameters());resume_optimizer(raw,last)
            migrate(again,last,reward_spec(new))
            repeated=TrainingPhases(again,new,raw=raw,generation=1)
            self.assertTrue(_same_payload(first.payload(),repeated.payload()))
            self.assertTrue(_same_payload(other.state_dict(),last.state_dict()))
            self.assertEqual(sha(path),original)


if __name__=='__main__':unittest.main()
