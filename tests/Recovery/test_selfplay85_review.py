"""Real-log gate replay and reporting regressions from the generation-85 review."""
import copy
import io
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import gae,critic_data
from brass_ai.recovery.history import compact
from brass_ai.recovery.review_metrics import gate_diagnostics,mixed_ev_console,mixed_ev_status

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0085-20261019T172419550588Z'


def reports():
    rows={int(k.split('/')[0].split('-')[1])+1:v['summary']
          for k,v in read(EXPORT/'retention-history.json').items()}
    for path in EXPORT.glob('iteration-*/report.json'):
        row=read(path);rows[row['iteration']+1]=row
    return rows


class ExportReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=reports();cls.config=read(EXPORT/'recorded-config.json')['advantage']

    def test_exact_decisions_window_restart_and_actual_extension_age(self):
        # The generation-55 migration starts its evidence window at 55. Rebuild
        # only the gate's scalar checkpoint from its committed report; no model
        # weights, tensor buffers or training replay are claimed to be present.
        last=self.rows[55]['actor']['gae']['gate'];s=gae.initial(self.config,55)
        s.update(active_lambda=last['lambda_next'],**{k:last[k] for k in
            ('value_iterations','ev_streak','healthy_streak','stage_origin')})
        rng=torch.get_rng_state().clone();resume=None
        for generation in range(56,86):
            row=self.rows[generation];expected=row['actor']['gae']['gate']
            before=copy.deepcopy(s)
            kwargs=dict(generation=generation,value_report=row['value_phase'],
                health_ok=expected['training_health']['ok'],rating=expected['rating'])
            s,r=gae.advance(s,self.config,**kwargs)
            self.assertEqual(before['generation'],generation-1)
            for key in ('decision','reason','lambda_used','lambda_next','value_iterations',
                        'heldout_ev','ev_streak','healthy','healthy_streak','stage_origin'):
                self.assertEqual(r[key],expected[key],(generation,key))
            for kind in ('historical','legacy'):
                actual=r['table_gate'][kind];old=expected['table_gate'][kind]
                for key in ('enabled','origin','streak','lambda_used','lambda_next','decision','evidence'):
                    self.assertEqual(actual[key],old[key],(generation,kind,key))
                if generation<81:
                    self.assertIsNone(actual['effective_stage_origin'])
                    self.assertEqual(actual['iterations_since_extension'],0)
                    self.assertFalse(actual['thirty_iteration_review_due'])
            if resume is not None:
                resume,rr=gae.advance(resume,self.config,**kwargs)
                self.assertEqual(resume,s);self.assertEqual(rr,r)
            if generation==70:
                cache=io.BytesIO();torch.save(s,cache);cache.seek(0)
                resume=torch.load(cache,weights_only=False)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(s['active_lambda'],.98)
        self.assertEqual(s['table_origins'],dict(historical=65,legacy=66))
        self.assertEqual([e['generation'] for e in s['table_evidence']['entries']],list(range(66,86)))
        for kind in ('historical','legacy'):
            self.assertEqual(r['table_gate'][kind]['effective_stage_origin'],81)
            self.assertEqual(r['table_gate'][kind]['iterations_since_extension'],4)

    def test_latest_report_exposes_supported_history_decline_not_one_game_v4_ev(self):
        row=copy.deepcopy(self.rows[85]);before=copy.deepcopy(row)
        d=gate_diagnostics(row['actor'],row['value_phase'],row['il_kl_limit'],row['quality_pool'])
        self.assertIn('mixed_ev_below_tolerance:historical',d['warnings'])
        self.assertNotIn('mixed_ev_below_tolerance:legacy',d['warnings'])
        self.assertEqual(d['mixed_ev_status']['historical']['status'],'below_tolerance')
        self.assertAlmostEqual(d['mixed_ev_status']['historical']['ev_gap'],-.08593933082131522)
        self.assertEqual(d['mixed_ev_status']['historical']['games'],150)
        self.assertEqual(d['mixed_ev_status']['legacy']['status'],'ok')
        self.assertEqual(d['mixed_ev_status']['legacy']['games'],16)
        self.assertLess(d['heldout_by_table']['legacy']['ev'],-13.)
        self.assertEqual(d['heldout_by_table']['legacy']['games'],1)
        self.assertIn('historical:0.280vs0.366[below_tolerance]',mixed_ev_console(d))
        self.assertIn('legacy:0.390vs0.353[ok]',mixed_ev_console(d))
        self.assertEqual(row,before)
        row['gate_diagnostics']=d
        self.assertEqual(compact(row)['gate_diagnostics'],d)

    def test_reviewed_continuation_requires_exact_source_and_unchanged_recipe(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        state=read(EXPORT/'state.json');before=copy.deepcopy(state)
        signature=copy.deepcopy(state['signature']);signature['implementation']=implementation_id()
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        self.assertTrue(check_signature(state,signature,True))
        for key,value in (('learning_rate',9e-6),('exploration',.1)):
            changed=copy.deepcopy(signature);changed['config'][key]=value
            with self.assertRaisesRegex(ValueError,'unchanged config'):check_signature(state,changed,True)
        self.assertEqual(state,before)


class SnapshotExposure(unittest.TestCase):
    def state(self,generation=59,weight=1.):
        config=copy.deepcopy(critic_data.DEFAULT);state=critic_data.initial(config)
        state.update(generation=generation,snapshots_active=True,snapshot_weight=weight,
            learner_ev_baseline=.44701269779353614,activation_generation=56)
        return config,state

    def test_real_generation60_cannot_reduce_before_any_snapshot_training(self):
        row=reports()[60];cfg,state=self.state();before=copy.deepcopy(state)
        self.assertEqual(row['actor']['critic_rows']['snapshot_distinct_used'],0)
        self.assertEqual(row['value_phase']['role_usage']['snapshot']['distinct_used'],0)
        state,r=critic_data.advance(state,cfg,generation=60,report=row['value_phase'],
            healthy=True,fingerprint_now=None)
        self.assertEqual(state['snapshot_weight'],1.)
        self.assertEqual(r['reason'],'no_snapshot_training')
        self.assertFalse(r['snapshot_rows_used']);self.assertTrue(r['snapshot_rows_enabled'])
        self.assertEqual(before['snapshot_weight'],1.)

    def test_actual_ppo_exposure_counts_even_when_ppg_rejected(self):
        row=reports()[61];cfg,state=self.state(generation=60)
        self.assertEqual(row['value_phase']['optimizer_steps'],0)
        count=row['actor']['critic_rows']['snapshot_distinct_used']
        self.assertGreater(count,0)
        state,r=critic_data.advance(state,cfg,generation=61,report=row['value_phase'],
            healthy=True,fingerprint_now=None,ppo_snapshot_rows=count)
        self.assertEqual(state['snapshot_weight'],.5)
        self.assertTrue(r['snapshot_rows_used']);self.assertEqual(r['ppo_snapshot_distinct_used'],count)
        self.assertEqual(r['value_snapshot_distinct_used'],0)

    def test_only_accepted_value_rows_count_not_available_or_attempted(self):
        cfg,state=self.state()
        report=dict(optimizer_steps=0,attempted_steps=3,train_rows=100,
            role_usage=dict(snapshot=dict(distinct_used=0)),
            heldout_after=dict(groups={'role:learner':dict(explained_variance=.2)}))
        same,r=critic_data.advance(state,cfg,generation=60,report=report,healthy=True,fingerprint_now=None)
        self.assertEqual(same['snapshot_weight'],1.);self.assertFalse(r['snapshot_rows_used'])
        report.update(optimizer_steps=1,role_usage=dict(snapshot=dict(distinct_used=40)))
        updated,r=critic_data.advance(state,cfg,generation=60,report=report,healthy=True,fingerprint_now=None)
        self.assertEqual(updated['snapshot_weight'],.5)
        self.assertEqual(r['value_snapshot_distinct_used'],40)
        self.assertEqual(r['ppo_snapshot_distinct_used'],0)

    def test_current_saved_half_weight_and_restart_are_preserved(self):
        cfg,state=self.state(generation=85,weight=.5)
        buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0)
        resumed=torch.load(buffer,weights_only=False);rng=torch.get_rng_state().clone()
        kwargs=dict(generation=86,report=dict(optimizer_steps=0),healthy=True,fingerprint_now=None)
        result=critic_data.advance(state,cfg,**kwargs)
        self.assertEqual(result,critic_data.advance(resumed,cfg,**kwargs))
        self.assertEqual(result[0]['snapshot_weight'],.5)
        self.assertEqual(result[0]['activation_generation'],56)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_training_phase_forwards_actual_ppo_usage_and_ignores_empty_update(self):
        from test_selfplay55_review import phase_fixture
        rows=reports()
        for steps,expected in ((38,.5),(0,1.)):
            p,u=phase_fixture(60);_,p.critic_state=self.state(generation=60)
            p.value_report=copy.deepcopy(rows[61]['value_phase'])
            u.update(optimizer_steps=steps,critic_rows=dict(snapshot_distinct_used=12284))
            p.finish(generation=61,update=u,anchor_gate=dict(ok=True),rating=dict(ok=True))
            self.assertEqual(p.critic_state['snapshot_weight'],expected)
            self.assertEqual(p.value_report['critic_rollout']['ppo_snapshot_distinct_used'],12284 if steps else 0)


class ReportingEdges(unittest.TestCase):
    def test_no_evidence_is_unavailable_not_zero_or_regression(self):
        self.assertEqual(mixed_ev_status({}),{})
        self.assertEqual(mixed_ev_console({}),'')
        table=dict(enabled=True,lambda_used=.98,lambda_next=.98,evidence=dict(
            supported=False,ev_ok=False,mixed=dict(games=0,explained_variance=None),
            selfplay=dict(games=0,explained_variance=None),reasons=['insufficient_games']))
        actor=dict(gae=dict(gate=dict(table_gate=dict(legacy=table))))
        d=gate_diagnostics(actor)
        self.assertEqual(d['mixed_ev_status']['legacy']['status'],'insufficient_support')
        self.assertEqual(d['warnings'],[])
        self.assertIn('legacy:navsna[insufficient_support]',mixed_ev_console(d))
        # A scalar EV with inadequate distinct-game support cannot issue a
        # supported regression warning, even if it is negative.
        table['evidence']['mixed']['explained_variance']=-100.
        table['evidence']['selfplay']['explained_variance']=.6
        self.assertEqual(gate_diagnostics(actor)['warnings'],[])

    def test_qualified_at_lambda_one_does_not_count_as_active_regression(self):
        table=dict(enabled=True,lambda_used=1.,lambda_next=1.,evidence=dict(
            supported=True,ev_ok=False,mixed=dict(games=10,explained_variance=.2),
            selfplay=dict(games=100,explained_variance=.6)))
        actor=dict(gae=dict(gate=dict(table_gate=dict(historical=table))))
        self.assertEqual(gate_diagnostics(actor)['warnings'],[])
        table['lambda_next']=.98
        self.assertIn('mixed_ev_below_tolerance:historical',gate_diagnostics(actor)['warnings'])

    def test_review_age_uses_later_of_qualification_and_current_lambda_stage(self):
        cfg=copy.deepcopy(gae.DEFAULT);cfg['table_gate']=copy.deepcopy(gae.ARCHIVED_TABLE_GATE)
        cfg['gate']['mode']='fixed';cfg['lambda']=.98
        s=gae.initial(cfg);s.update(generation=109,stage_origin=80,table_enabled=dict(historical=True,legacy=True),
            table_origins=dict(historical=20,legacy=90))
        s,r=gae.advance(s,cfg,generation=110,value_report={},health_ok=False,rating={})
        self.assertEqual(r['table_gate']['historical']['iterations_since_extension'],30)
        self.assertTrue(r['table_gate']['historical']['thirty_iteration_review_due'])
        self.assertEqual(r['table_gate']['legacy']['iterations_since_extension'],20)
        self.assertFalse(r['table_gate']['legacy']['thirty_iteration_review_due'])
        # A subsequent global stage change starts its own exposure review age.
        cfg['lambda']=.95;s.update(active_lambda=.95,stage_origin=110)
        _,r=gae.advance(s,cfg,generation=111,value_report={},health_ok=False,rating={})
        self.assertEqual(r['table_gate']['historical']['iterations_since_extension'],1)
        self.assertEqual(r['table_gate']['legacy']['iterations_since_extension'],1)


if __name__=='__main__':unittest.main()
