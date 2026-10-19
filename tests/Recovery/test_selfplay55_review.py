"""Generation-55 regressions: adoption gates, clone escalation and frozen ratings."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.legacy import LegacyWorker
from brass_ai.recovery import gae,critic_data,ratings
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.execution import _collection_recipe,make_policy
from brass_ai.recovery.history import compact
from brass_ai.recovery.legacy_search import frozen_rating_runtime
from brass_ai.recovery.review_metrics import gate_diagnostics
from brass_ai.search.legacy_runtime import directory,source_hash,CONTRACT
from brass_ai.step5.league import write_json

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0055-20261019T053031361382Z'


def phase_fixture(generation=0):
    cfg=read(EXPORT/'recorded-config.json');p=object.__new__(TrainingPhases)
    p.config=cfg;p.options={k:copy.deepcopy(cfg[k]) for k in ('advantage','value_phase')}
    p.state=gae.initial(cfg['advantage']);p.state.update(generation=generation,
        value_iterations=generation,ev_streak=min(generation,4))
    p.critic_state=critic_data.initial(cfg['critic_data']);p.critic_state['generation']=generation
    p.fingerprint=None;p.beta=1.;p.trips=[];p.evidence_migration=None
    p.value_report=dict(optimizer_steps=1,kl_guard_tripped=False,
        descriptor_gain=dict(learner_ev_gain=.01),heldout_after=dict(groups={
            'role:learner':dict(explained_variance=.6),
            'table:selfplay/role:learner':dict(explained_variance=.6)}))
    update=dict(gae={},categorical_probe=dict(categorical_kl=.003),controller_warmup_done=True,
                optimizer_steps=1,kl_early_stop=False)
    return p,update


class AdoptionAndDiagnostics(unittest.TestCase):
    def test_generation55_explicit_source_transition_preserves_config_probe_and_progress(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        state=read(EXPORT/'state.json');before=copy.deepcopy(state)
        signature=copy.deepcopy(state['signature']);signature['implementation']=implementation_id()
        self.assertEqual(state['iteration'],55);self.assertEqual(state['games_done'],17600)
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        self.assertTrue(check_signature(state,signature,True))
        changed=copy.deepcopy(signature);changed['config']['value_phase']['beta_clone']=2.
        with self.assertRaisesRegex(ValueError,'unchanged config'):check_signature(state,changed,True)
        self.assertEqual(state,before)

    def test_calibrated_distance_only_gates_lr_not_healthy_gae_or_critic_adoption(self):
        p,u=phase_fixture(9);rng=torch.get_rng_state().clone()
        gate=dict(ok=False,reasons=['anchor_budget_exceeded','anchor_recovery_latch'],
                  anchor_distance=.2,anchor_budget=.05,health=dict(ok=True,reasons=[]))
        original=copy.deepcopy(gate)
        report=p.finish(generation=10,update=u,anchor_gate=gate,rating=dict(ok=True))
        self.assertEqual(p.state['active_lambda'],.98)
        self.assertTrue(p.critic_state['snapshots_active'])
        self.assertTrue(report['healthy']);self.assertFalse(report['anchor_gate']['ok'])
        self.assertTrue(report['training_health']['ok'])
        self.assertEqual(gate,original);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_behavior_failure_still_blocks_both_adoptions_even_with_healthy_economy(self):
        r=read(EXPORT/'iteration-0054/report.json');gate=r['il_kl_limit']['lr_gate']
        self.assertTrue(all(c['ok'] for c in gate['health']['checks'].values()))
        self.assertIn('behavior_check_failed',gate['health']['reasons'])
        p,u=phase_fixture(9);report=p.finish(generation=10,update=u,anchor_gate=gate,rating=dict(ok=True))
        self.assertEqual(p.state['active_lambda'],1.)
        self.assertFalse(p.critic_state['snapshots_active']);self.assertFalse(report['healthy'])

    def test_noncalibrated_gate_and_missing_evidence_remain_fail_closed(self):
        for gate in (None,dict(ok=False),dict(ok=True,health=None)):
            p,u=phase_fixture(9);p.finish(generation=10,update=u,anchor_gate=gate,rating=dict(ok=True))
            self.assertEqual(p.state['active_lambda'],1.)
            self.assertFalse(p.critic_state['snapshots_active'])

    def test_no_repeat_beta_raise_without_new_cap_trip_and_resume_matches(self):
        p,u=phase_fixture()
        for g,trip in enumerate([True,True,True,False,False],1):
            p.value_report['kl_guard_tripped']=trip
            p.finish(generation=g,update=u,anchor_gate=dict(ok=True),rating=dict(ok=True))
            self.assertEqual(p.beta,1. if g<3 else 2.)
        self.assertEqual(p.value_report['beta_reason'],'no_new_kl_trip')
        resumed=copy.deepcopy(p);before=torch.get_rng_state().clone()
        for phase in (p,resumed):
            phase.value_report['kl_guard_tripped']=True
            phase.finish(generation=6,update=copy.deepcopy(u),anchor_gate=dict(ok=True),rating=dict(ok=True))
        self.assertEqual(p.beta,4.);self.assertEqual(p.state,resumed.state)
        self.assertEqual(p.trips,resumed.trips);self.assertEqual(p.value_report,resumed.value_report)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_existing_beta_eight_is_preserved_without_rewriting_history(self):
        p,u=phase_fixture(55);p.beta=8.;p.trips=[False]*5
        p.finish(generation=56,update=u,anchor_gate=dict(ok=True),rating=dict(ok=True))
        self.assertEqual(p.beta,8.);self.assertEqual(p.value_report['beta_decision'],'hold')

    def test_report_exposes_stall_causes_without_relaxing_or_mutating_health(self):
        report=read(EXPORT/'iteration-0054/report.json');before=copy.deepcopy(report)
        result=gate_diagnostics(report['actor'],report['value_phase'],report['il_kl_limit'],report['quality_pool'])
        self.assertNotIn('sale_failures',result);self.assertNotIn('sale_positions',result)
        self.assertEqual(result['scored_history_members_used'],0);self.assertTrue(result['history_il_only'])
        self.assertEqual(set(result['warnings']),{'history_pool_il_only','il_behavior_check_failed','il_distance_alarm'})
        self.assertEqual(result['value_accepted_steps'],41)
        self.assertEqual(result['value_accepted_fraction'],.5)
        self.assertIsNone(result['heldout_by_table']['legacy']['ev'])
        self.assertIsNone(result['heldout_by_table']['legacy']['games'])  # unavailable in the old export
        self.assertEqual(report,before)
        report['gate_diagnostics']=result
        self.assertEqual(compact(report)['gate_diagnostics'],result)

    def test_critic_support_counts_complete_games_not_correlated_rows(self):
        from test_gae_value_phase import fixture
        from brass_ai.recovery.value_phase import metrics,split
        from brass_ai.recovery.updates import Dataset
        torch.set_num_threads(2)
        model,_,cfg,(_,data)=fixture();ds=Dataset(data,'cpu')
        _,held,_=split(data,.05);report=metrics(model,ds,held,cfg['update'])
        self.assertEqual(report['games'],1)
        for group in report['groups'].values():
            self.assertEqual(group['games'],1);self.assertGreater(group['rows'],group['games'])


class FrozenRating(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.parent='bank-test/balanced'
        profile=self.root/'configs/legacy-ai/balanced.json';write_json(profile,dict(style='balanced'))
        recipe=dict(contract=CONTRACT,bank='bank-test',sources={'old.cs':'a'*64},
                    profiles={'balanced':source_hash(profile)},adapter='b'*64)
        self.ref=dict(id='search-'+digest(recipe)[:24],recipe=recipe)
        self.base=directory(self.root,self.ref['id']);self.runtime=self.base/'runtime';self.runtime.mkdir(parents=True)
        (self.runtime/'BrassLegacyAI.exe').write_bytes(b'frozen authored executable')
        (self.runtime/'dependency.dll').write_bytes(b'fixed rules dependency')
        pins=[dict(name=p.name,sha256=sha(p)) for p in self.runtime.iterdir()]
        # Simulate newer working sources without changing the recorded binary.
        (self.root/'old.cs').write_text('new source',encoding='utf-8')
        write_json(self.runtime/'runtime.json',dict(files=pins,sources=[dict(path='old.cs',sha256='a'*64)]))
        write_json(self.root/'configs/rules/runtime-dependencies-1.1.0.json',
                   dict(files=[p for p in pins if p['name']!='BrassLegacyAI.exe']))
        write_json(self.base/'search.json',dict(reference=self.ref,runtime_sha256=sha(self.runtime/'runtime.json')))

    def test_frozen_reference_checks_receipt_binary_and_profile_but_not_new_sources(self):
        runtime,pin=frozen_rating_runtime(self.root,self.ref,self.parent)
        self.assertEqual(runtime,self.runtime);self.assertEqual(pin,sha(runtime/'runtime.json'))
        with self.assertRaisesRegex(ValueError,'source change'):
            LegacyWorker(self.root,runtime_directory=runtime)
        process=SimpleNamespace(stdin=io.StringIO(),stdout=[],stderr=[])
        with patch('brass_ai.legacy.subprocess.Popen',return_value=process) as launch:
            worker=LegacyWorker(self.root,runtime_directory=runtime,frozen_manifest_sha256=pin)
        self.assertIs(worker.process,process);launch.assert_called_once()

    def test_changed_manifest_or_executable_is_rejected_before_launch(self):
        runtime,pin=frozen_rating_runtime(self.root,self.ref,self.parent)
        with patch('brass_ai.legacy.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError,'manifest changed'):
                LegacyWorker(self.root,runtime_directory=runtime,frozen_manifest_sha256='0'*64)
            (runtime/'BrassLegacyAI.exe').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'binary/dependency changed'):
                frozen_rating_runtime(self.root,self.ref,self.parent)
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                LegacyWorker(self.root,runtime_directory=runtime,frozen_manifest_sha256=pin)
            launch.assert_not_called()

    def test_changed_profile_missing_runtime_and_bad_receipt_fail_closed(self):
        profile=self.root/'configs/legacy-ai/balanced.json';profile.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'profile changed'):frozen_rating_runtime(self.root,self.ref,self.parent)
        other=copy.deepcopy(self.ref);other['recipe']['adapter']='c'*64
        other['id']='search-'+digest(other['recipe'])[:24]
        with self.assertRaisesRegex(ValueError,'retain artifacts'):frozen_rating_runtime(self.root,other,self.parent)
        (self.runtime/'runtime.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'manifest changed'):frozen_rating_runtime(self.root,self.ref,self.parent)

    def test_marker_only_on_rating_jobs_no_policy_relabelling_or_spec_mutation(self):
        spec=read(EXPORT/'ratings/slots/generation-000050/spec.json');before=copy.deepcopy(spec)
        jobs=ratings.jobs_for(spec,self.root)
        self.assertEqual(spec,before)
        v4=spec['agents']['v4']['policy']
        for job in jobs:
            _collection_recipe([job],{})
            for p in job['policies']:
                if p['kind']=='lab-legacy':
                    self.assertTrue(p['frozen_rating']);self.assertEqual(p['policy_id'],v4['policy_id'])
                    with patch('brass_ai.recovery.legacy_search.LabLegacyPolicy',
                               return_value=SimpleNamespace(policy_id=p['policy_id'])) as factory:
                        make_policy(p,self.root,None,None,None)
                    self.assertTrue(factory.call_args.kwargs['frozen_rating'])
                else:self.assertNotIn('frozen_rating',p)
        job=next(j for j in jobs if any(p.get('frozen_rating') for p in j['policies']))
        job=copy.deepcopy(job);job['provenance']['purpose']='training'
        with self.assertRaisesRegex(ValueError,'restricted'):_collection_recipe([job],{})

    def test_startup_checks_fixed_reference_and_disabled_rating_skips(self):
        policy=dict(parent=self.parent,search_runtime=self.ref,search_free=True)
        rating=dict(config=dict(enabled=True),agents=dict(v4=dict(policy=policy)))
        ratings.validate_frozen_reference(self.root,rating)
        (self.runtime/'BrassLegacyAI.exe').unlink()
        with self.assertRaisesRegex(ValueError,'binary/dependency changed'):
            ratings.validate_frozen_reference(self.root,rating)
        rating['config']['enabled']=False;ratings.validate_frozen_reference(self.root,rating)


if __name__=='__main__':unittest.main()
