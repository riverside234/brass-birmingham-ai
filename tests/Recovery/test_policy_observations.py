"""Neutral critic fingerprints, economic-only LR, and pool-only fixed probes."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.recovery import policy_observations as obs,lr_controller as lr,il_kl_limit as limits
from brass_ai.recovery import critic_data,quality_pool,exploration,updates
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.utility import reward_spec
from brass_ai.recovery.domino_auxiliary import Diversity
from test_ppo_lr_controller import measurements


def state1340():
    return read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1340-20261031T210153391863Z/state.json')


class Observations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_strict_explicit_config_preserves_recipe_and_fresh_defaults(self):
        old=state1340()['signature']['config'];saved=copy.deepcopy(old)
        new=validate_config(obs.configure(old))
        self.assertEqual(old,saved);self.assertEqual(obs.configure(new),new)
        self.assertEqual({k for k in new if new[k]!=old.get(k)},{obs.KEY,'ppo_lr_controller'})
        c=copy.deepcopy(new['ppo_lr_controller'])
        c.update({k:old['ppo_lr_controller'][k] for k in ('contract','anchor_kl_budget','anchor_group_kl_budget')})
        self.assertEqual(c,old['ppo_lr_controller'])
        for origin in (old,new):
            fresh=latest_model_config(origin,origin['imitation_checkpoint'])
            self.assertEqual(fresh[obs.KEY],obs.DEFAULT)
            self.assertEqual(latest_model_config(fresh,fresh['imitation_checkpoint']),fresh)
        for field,value in (('anchor_kl','on'),('seat_fingerprints','old'),('extra',True)):
            bad=copy.deepcopy(new);bad[obs.KEY][field]=value
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(new);bad['ppo_lr_controller']['anchor_kl_budget']=.15
        with self.assertRaises(ValueError):validate_config(bad)

    def test_economic_baseline_and_progress_preserved_without_stale_kl(self):
        state=state1340()['il_kl_limit'];before=copy.deepcopy(state)
        state['blocked']=True;state['limit']=.05;state['healthy_window']=[999.]
        counts=state['baseline_rows'][-1]
        new,report=obs.advance_health(state,counts,generation=1341,warmup_done=True)
        self.assertEqual(new['baseline'],before['baseline'])
        self.assertEqual(new['baseline_rows'],before['baseline_rows'])
        self.assertEqual(new['rolling'],(state['rolling']+[counts])[-5:])
        for k in ('limit','blocked','healthy_window','last_update','kl_last_healthy','probe'):
            self.assertEqual(new[k],state[k])
        self.assertIsNone(report['measurement']);self.assertFalse(report['anchor_kl_enabled'])
        expected=limits.health(limits.baseline_evidence(state['baseline']),limits.rates(new['rolling']))
        self.assertEqual(report['health'],expected)
        self.assertEqual(report['lr_gate']['ok'],expected['ok'])
        self.assertIsNone(report['lr_gate']['anchor_distance'])
        with self.assertRaisesRegex(ValueError,'skipped/repeated'):obs.advance_health(new,counts,generation=1341,warmup_done=True)
        _,missing=obs.advance_health(new,None,generation=1342,warmup_done=True)
        self.assertFalse(missing['health']['ok'])

    def test_lr_ignores_stale_anchor_and_preserves_other_guards_and_restart(self):
        cfg=obs.configure(state1340()['signature']['config'])['ppo_lr_controller']
        health=dict(contract='selfplay-economic-health/3.0.0',ok=True,reasons=[])
        measurement=measurements(economic_health=health,anchor_distance=999.,
            calibrated_anchor=dict(ok=False,anchor_distance=999.,reasons=['old_anchor']))
        s=lr.initial_state(cfg)
        for _ in range(2):s,r=lr.decide(s,measurement)
        restarted=copy.deepcopy(s);n,r=lr.decide(s,measurement)
        self.assertEqual((n,r),lr.decide(restarted,measurement));self.assertEqual(r['decision'],'up')
        self.assertIsNone(r['anchor_distance'])
        m=dict(measurement,economic_health=dict(health,ok=False,reasons=['vp_per_seat']))
        _,r=lr.decide(s,m);self.assertEqual(r['reason'],'health_check_failed')
        _,r=lr.decide(s,dict(m,kl=.02));self.assertEqual(r['decision'],'down')
        _,r=lr.decide(s,dict(m,early_stop=True,steps_done=1));self.assertEqual(r['decision'],'down')
        _,r=lr.decide(s,dict(measurement,warmup_done=False));self.assertEqual(r['reason'],'warmup')
        with self.assertRaisesRegex(ValueError,'fresh economic'):lr.decide(s,dict(measurement,economic_health=None))

    def test_neutral_plans_preserve_roles_rotation_age_and_context(self):
        from brass_ai.recovery.execution import neural_spec
        from brass_ai.recovery.domino_sampling import assignments
        members=[dict(model_id='snap',generation=4),dict(model_id='il',generation=0)]
        policies=[neural_spec('current',.05,'mixed'),neural_spec('snap'),neural_spec('il'),dict(kind='legacy',policy_id='bot')]
        provider=Mock();provider.get.side_effect=AssertionError('fingerprint inference')
        with patch.object(critic_data,'fingerprint',side_effect=AssertionError('fingerprint computation')):
            for rotation in range(4):
                strategy=assignments(policies,rotation,seed=123,index=19,learner_id=policies[0]['policy_id'],
                    current='current',checkpoints={'current':'a'*64,'snap':'b'*64,'il':'c'*64})
                table=critic_data.planned_table(policies,rotation,'current',9,members,None,None,
                    learner_policy_id=policies[0]['policy_id'],strategies=strategy,context_fingerprints=provider,neutral_fingerprints=True)
                self.assertEqual(table['fingerprint_mode'],'neutral')
                self.assertFalse(np.asarray(table['stable'])[:,6:].any())
                self.assertEqual(table['seats'][rotation]['role'],'learner')
                self.assertAlmostEqual(table['stable'][(rotation+1)%4][5],np.log1p(5)/np.log1p(256),places=6)
                self.assertEqual(table['seats'][rotation]['strategy'],strategy['seats'][rotation])
                game=dict(opponents=[p['policy_id'] for p in policies[-rotation:]+policies[:-rotation]] if rotation else [p['policy_id'] for p in policies],records=[])
                critic_data.check_table(table,game,policies[0]['policy_id'])
                bad=copy.deepcopy(table);bad['stable'][0][6]=.1
                with self.assertRaisesRegex(ValueError,'Neutral fingerprint'):critic_data.check_table(bad,game,policies[0]['policy_id'])

    def test_cached_descriptors_zeroed_at_boundary_without_weights_or_adam_or_rng_changes(self):
        from test_critic_descriptors import mixed_fixture
        model,opt,cfg,_,data,snapshot,_=mixed_fixture();cfg=obs.configure(dict(cfg,entropy_control={'mode':'off'}))
        # Populate Adam to exercise exact continuation, without a training loop.
        next(model.parameters()).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
        ds=updates.Dataset(data+snapshot,'cpu');original=ds.descriptors.clone()
        weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        obs.install(model,cfg);obs.install(model,cfg)
        self.assertTrue(_same_payload(weights,model.state_dict()));self.assertTrue(_same_payload(adam,opt.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        seen=[];handle=model.descriptor_encoder[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        inputs=ds.descriptors.clone().requires_grad_(True);model.descriptor_encoder(inputs).sum().backward();handle.remove()
        self.assertTrue(torch.equal(seen[0][...,:6],original[...,:6]));self.assertEqual(seen[0].shape[-1],34)
        self.assertFalse(seen[0][...,6:].any());self.assertFalse(inputs.grad[...,6:].any())
        self.assertGreater(float(inputs.grad[...,:6].abs().sum()),0)
        self.assertTrue(torch.equal(ds.descriptors,original))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'checkpoint.pt';save_checkpoint(path,model,opt,step=1340,training={'parent':'unchanged','games_done':428800})
            checksum=sha(path);other,raw=load_checkpoint(path);other_opt=torch.optim.AdamW(other.parameters())
            resume_optimizer(raw,other_opt);before=torch.get_rng_state().clone();obs.install(other,cfg)
            self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(sha(path),checksum)
            self.assertTrue(_same_payload(model.state_dict(),other.state_dict()));self.assertTrue(_same_payload(opt.state_dict(),other_opt.state_dict()))
            self.assertTrue(torch.equal(model.descriptor_encoder(original),other.descriptor_encoder(original)))
            self.assertEqual(raw['training']['parent'],'unchanged');self.assertEqual(raw['training']['games_done'],428800)

    def test_pool_probe_only_due_healthy_distinct_attempts_and_no_fingerprints(self):
        from test_pfsp_pool import pool
        from test_quality_pool import entry
        p=pool(2,generation=101);probe=Mock();probe.arrays={'offsets':np.array([0,2])};probe.config={'probability_floor':1e-6}
        probe.probabilities.return_value=np.array([.9,.1]);candidate=entry(102)
        health=dict(health=dict(ok=True))
        with (patch.object(critic_data,'fingerprint',side_effect=AssertionError('fingerprint computation')),
             patch.object(quality_pool,'cached',return_value=np.array([.1,.9])),
             patch.object(quality_pool,'cache_candidate',return_value={'path':'new.npz'})):
            _,event=quality_pool.finish(None,None,p,None,candidate,probe,{},health,compute_fingerprints=False)
            self.assertEqual(event['reason'],'not_due');probe.probabilities.assert_not_called()
            p['generation']=105
            quality_pool.finish(None,None,p,None,candidate,probe,{},dict(health=dict(ok=False)),compute_fingerprints=False)
            probe.probabilities.assert_not_called()
            p['members'][0]['fingerprint']={'values':[.5]*28}
            out,event=quality_pool.finish(None,None,p,None,candidate,probe,{},health,compute_fingerprints=False)
            self.assertTrue(event['admitted']);probe.probabilities.assert_called_once()
            self.assertNotIn('fingerprint',out['members'][-1]);self.assertEqual(out['config'],p['config'])

    def test_fixed_entropy_branch_stage_needs_no_retired_probe(self):
        cfg=obs.configure(state1340()['signature']['config']);s=exploration.initial(cfg,0)
        self.assertIsNone(s['baseline'])
        self.assertIsNotNone(exploration.effective(cfg,s,20)['branching'])
        saved=state1340()['exploration'];before=copy.deepcopy(saved)
        new,report=exploration.advance(saved,cfg,{},generation=1341,warmup_done=True,healthy=True)
        self.assertEqual(new['baseline'],before['baseline']);self.assertEqual(new['baseline_rows'],before['baseline_rows'])
        self.assertEqual(exploration.effective_alphas(cfg,new),cfg['update']['fixed_entropy']['weights'])
        self.assertTrue(all(g['raw_entropy'] is None for g in report['groups'].values()))

    def test_complete_planner_keeps_pfsp_models_seeds_contexts_and_composition(self):
        from test_pfsp_pool import pool
        from brass_ai.recovery import pfsp_pool,domino_config,panel_gates
        from brass_ai.recovery.ordinary import plan_collection
        c=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');c['diversity']=domino_config.defaults()
        c['entropy_control']['mode']='off';value=pool(4,generation=85);fp=dict(values=[1/8]*16+[1/6]*12)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);path=root/'weights.pt';path.write_bytes(b'planner fixture')
            for m in value['members']:m.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),fingerprint=fp)
            pfsp_pool.seal(value)
            state=dict(iteration=85,games_done=27200,next_seed=c['seed'],context_counter=81600,quality_pool=value,
                population=[quality_pool.reference(m) for m in value['members']],history=[quality_pool.reference(value['members'][0])],
                evaluation_anchor=quality_pool.reference(value['members'][0]),ratings=dict(panel_gates=panel_gates.initial(85)))
            before=copy.deepcopy(state);service=Mock();service.models={m['model_id']:None for m in value['members']}
            bots={str(i):dict(kind='legacy',policy_id=str(i),search_free=True) for i in range(5)}
            provider=Mock();provider.get.return_value=fp
            def plan(cfg,**kwargs):return plan_collection(SOURCE_ROOT,root,state,cfg,dict(config=cfg),service,'current',list(bots),bots,bots,None,**kwargs)
            old=plan(c,fingerprints=dict(learner=fp,rule_bot=fp,contexts=provider));provider.get.side_effect=AssertionError('retired fingerprints')
            new=plan(obs.configure(c));self.assertEqual(new,plan(obs.configure(c)));self.assertEqual(state,before)
            for key in ('composition','quality_planning','quality_tables','context_counter','cursor'):
                if key in old:self.assertEqual(new[key],old[key])
            self.assertEqual(len(new['jobs']),320)
            for a,b in zip(old['jobs'],new['jobs']):
                for key in ('policies','seed','rotation'):self.assertEqual(a[key],b[key])
                self.assertEqual(a['provenance']['diversity_assignment'],b['provenance']['diversity_assignment'])
                self.assertEqual(b['provenance']['policy_observations'],obs.DEFAULT)


class Smoke(unittest.TestCase):
    def test_isolated_ppo_ppg_with_old_cached_descriptors_and_no_policy_probe(self):
        from test_domino import fixture,collect
        from brass_ai.recovery.training_revision_g1340 import recipe
        from brass_ai.recovery.il_kl_probe import Probe
        from brass_ai.recovery import policy_anchor
        torch.set_num_threads(2);m,opt,c=fixture();c=obs.configure(recipe(c));m.config=replace(m.config,utility=reward_spec(c))
        lr.configure(opt,c['ppo_lr_controller'])
        with (tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp,
             patch.object(Probe,'measure',side_effect=AssertionError('AnchorKL measured')),
             patch.object(Probe,'probabilities',side_effect=AssertionError('fixed-probe diagnostic')),
             patch.object(policy_anchor,'prepare',side_effect=AssertionError('AnchorKL calibration'))):
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));p.diversity=Diversity(m,c['diversity'],regime='neutral-smoke')
            b,rows=collect(m,p,1,games=42);seen=[]
            handle=m.descriptor_encoder[0].register_forward_pre_hook(lambda module,args:seen.append(bool(args[0][...,6:].any())))
            reward=reward_spec(c);r=updates.ppo_update(m,opt,[dict(split='train')],b,c['update'],learner_data=rows,
                insolvency=reward['insolvency'],reward=reward,phases=p)
            value=p.fit(m);handle.remove()
            self.assertEqual(r['optimizer_steps'],1);self.assertTrue(seen);self.assertFalse(any(seen))
            self.assertIsNotNone(r['categorical_probe']['categorical_kl'])
            print('Neutral fingerprints isolated smoke:',dict(ppo_steps=r['optimizer_steps'],ppg_steps=value['optimizer_steps'],
                ppo_kl=r['categorical_probe']['categorical_kl'],ppg_kl=value['probe_kl'],zero_fingerprint_forwards=len(seen)),flush=True)


if __name__=='__main__':unittest.main()
