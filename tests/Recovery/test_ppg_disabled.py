"""Eight-age PPG -> PPO-only continuation and owned-cache startup retirement."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import gae,critic_data,snapshot_r2,ppg_config,ppg_epochs,ppg_replay,value_buffer,value_phase
from brass_ai.recovery.actor_value import TrainingPhases,KEY,PPO_ONLY_CONTRACT
from brass_ai.recovery.config import validate_config,latest_model_config,ppo_microbatch_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_eight import setup,collect
from test_gae_value_phase import ppo,finish
from test_ppo_only import measurement
from test_health_ev_window import report as ev_report

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0460-20261021T234253728877Z'


def rolling_checkpoint(preset='small'):
    """Authored current-architecture state with nonempty optimizers and gate evidence."""
    model,opt,cfg=setup(preset);disabled=value_phase.disabled_options(cfg)
    p=TrainingPhases(model,disabled)
    for optimizer in (opt,p.optimizer):
        parameter=next(model.parameters());parameter.grad=torch.ones_like(parameter)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    p.optimizer.param_groups[0]['lr']=cfg['value_phase']['learning_rate']*.3
    p.state=gae.initial(cfg['advantage'],440)
    p.state.update(active_lambda=.95,table_enabled=dict(historical=True,legacy=True),
        table_origins=dict(historical=65,legacy=66))
    for g in range(441,461):
        value=measurement();value['heldout_after']=ev_report(g)['heldout_after']
        p.state,_=gae.advance(p.state,cfg['advantage'],generation=g,value_report=value,
            health_ok=True,rating=dict(ok=True))
    p.critic_state=critic_data.initial(cfg['critic_data'])
    p.critic_state.update(generation=460,snapshots_active=True,snapshot_weight=.75,activation_generation=15)
    p.critic_state['weight_recovery']=snapshot_r2.initial(440,.75)
    r=p.critic_state['weight_recovery'];r['regime']=dict(source='ppg')
    r['entries']=[snapshot_r2.entry(g,ev_report(g,games=3),healthy=True,trained=True) for g in range(441,461)]
    r['reference']=snapshot_r2.reference(snapshot_r2.summarize(r['entries']),460,.5,r['regime'])
    p.beta=32.;p.trips=[True,False,True]
    saved=p.payload();saved.update(contract='actor-value-phases/9.0.0',
        options=dict(saved['options'],value_phase=copy.deepcopy(cfg['value_phase'])))
    clock=ppg_config.initial(418);epochs=ppg_epochs.initial(418,cfg['value_phase'])
    for g in range(419,461):
        due=ppg_config.due(clock);clock=ppg_config.advance(clock,g,due)
        if due:
            epochs,_=ppg_epochs.advance(epochs,dict(scheduled=True,epoch_cap_used=epochs['cap'],
                steps_per_epoch=10,planned_steps=epochs['cap']*10,best_budget_step=12,
                kl_guard_tripped=False),cfg['value_phase'],g)
    saved.update(schedule=clock,epoch_control=epochs,
        rolling_buffer=[dict(generation=g,missing_derived_files=True) for g in range(453,461)])
    return model,opt,cfg,dict(training={KEY:saved})


def caches(run):
    total=0
    for name,owner,filename in (('.ppg-value-cache',value_buffer.OWNER,'generation-000210.npy'),
            ('.ppg-eight-cache',ppg_replay.OWNER,'generation-000460.critic.npy')):
        path=run/name;path.mkdir(parents=True);write_json(path/'owner.json',owner)
        (path/filename).write_bytes(b'old cache');total+=9
        if name=='.ppg-eight-cache':
            for suffix in ('actions.npy','json','critic.npy.partial','json.tmp'):
                (path/('generation-000461.'+suffix)).write_bytes(b'pending');total+=7
    return total


class DisabledEight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_defaults_and_exact_g460_config_review(self):
        state=read(EXPORT/'state.json');old=state['signature']['config'];new=value_phase.disabled_options(old)
        validate_config(new)
        self.assertEqual([k for k in old if old[k]!=new[k]],['value_phase'])
        self.assertEqual(new['value_phase'],dict(old['value_phase'],enabled=False))
        self.assertFalse(ppg_config.DEFAULT['enabled'])
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        self.assertTrue(fresh['value_phase']['enabled']);self.assertEqual(fresh['value_phase']['holdout'],old['value_phase']['holdout'])
        self.assertEqual(fresh['value_phase']['contract'],ppg_config.COVERAGE_CONTRACT)
        self.assertEqual((fresh['value_phase']['acceptance_steps'],fresh['value_phase']['patience']),(16,1))
        self.assertEqual(fresh['update']['microbatch_size'],512)
        self.assertEqual(fresh['update']['max_candidate_cells'],131072)
        self.assertEqual(fresh['update']['preparation_workers'],4)
        self.assertEqual(ppg_config.configure(new),old)
        target=dict(state['signature'],config=new,implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        with self.assertRaises(ValueError):check_signature(state,target)
        bad=copy.deepcopy(target);bad['config']['exploration']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        combined=ppo_microbatch_config(new);validate_config(combined)
        self.assertEqual(combined['update'],dict(old['update'],microbatch_size=2048,max_candidate_cells=524288))
        self.assertEqual(combined['imitation'],old['imitation'])
        self.assertTrue(check_signature(state,dict(target,config=combined),reviewed=True))
        bad=copy.deepcopy(combined);bad['update']['effective_batch_size']*=2
        with self.assertRaises(ValueError):check_signature(state,dict(target,config=bad),reviewed=True)
        for enabled in (0,1,None,'false'):
            with self.assertRaises(ValueError):ppg_config.validate(dict(new['value_phase'],enabled=enabled))

    def test_g460_checkpoint_migration_keeps_adams_rng_and_PPO_gate_windows(self):
        model,opt,cfg,raw=rolling_checkpoint();new=value_phase.disabled_options(cfg)
        before=copy.deepcopy(raw);weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        with patch.object(ppg_replay,'Buffer',side_effect=AssertionError('No old replay required')), \
             patch.object(value_buffer,'DiskBuffer',side_effect=AssertionError('No old replay required')):
            p=TrainingPhases(model,new,raw=raw,generation=460)
        self.assertEqual(p.contract,PPO_ONLY_CONTRACT);self.assertFalse(p.rolling);self.assertIsNone(p.buffer)
        self.assertEqual(p.state,raw['training'][KEY]['gae'])
        self.assertEqual(p.critic_state,raw['training'][KEY]['critic_data'])
        self.assertEqual(p.beta,32.);self.assertEqual(p.trips,[True,False,True])
        self.assertTrue(_same_payload(p.optimizer.state_dict(),raw['training'][KEY]['optimizer']))
        self.assertTrue(_same_payload(raw,before));self.assertTrue(_same_payload(weights,model.state_dict()))
        self.assertTrue(_same_payload(adam,opt.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for state in (raw,dict(training={KEY:p.payload()})):
            restored=TrainingPhases(model,new,raw=state,generation=460)
            self.assertTrue(_same_payload(restored.payload(),p.payload()))
        bad=copy.deepcopy(new);bad['value_phase']['learning_rate']*=2
        with self.assertRaisesRegex(ValueError,'configuration differs'):
            TrainingPhases(model,bad,raw=raw,generation=460)

    def test_startup_cleans_both_caches_before_new_iteration_and_is_restart_safe(self):
        model,opt,cfg,raw=rolling_checkpoint();new=value_phase.disabled_options(cfg)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);total=caches(run)
            protected=run/'generation-0460.pt';protected.write_bytes(b'preserve checkpoint')
            fixed=run/'il-kl';fixed.mkdir();(fixed/'probe.npy').write_bytes(b'preserve probe')
            p=TrainingPhases(model,new,raw=raw,generation=460,root=SOURCE_ROOT,run=run)
            write_json(run/'state.json',dict(iteration=460,signature=dict(config=cfg)))
            with self.assertRaises(ValueError):p.startup(460)
            write_json(run/'state.json',dict(iteration=460,signature=dict(config=new)))
            with patch.object(ppg_replay,'Buffer',side_effect=AssertionError('No cache read')):
                r=p.startup(460)
                restarted=TrainingPhases(model,new,raw=raw,generation=460,root=SOURCE_ROOT,run=run)
                self.assertEqual(restarted.startup(460)['deleted_files'],[])
            self.assertEqual(r['deleted_bytes'],total);self.assertEqual(len(r['deleted_files']),6)
            self.assertEqual(protected.read_bytes(),b'preserve checkpoint')
            self.assertEqual((fixed/'probe.npy').read_bytes(),b'preserve probe')
            self.assertEqual(read(run/'ppg-disabled-cache-cleanup.json')['deleted_bytes'],total)
            for name in ('.ppg-eight-cache','.ppg-value-cache'):
                self.assertEqual([p.name for p in (run/name).iterdir()],['owner.json'])

    def test_cleanup_audits_both_directories_before_deleting_and_rejects_links(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);caches(run)
            write_json(run/'state.json',dict(iteration=460,signature=dict(config=dict(value_phase=dict(enabled=False)))))
            unknown=run/'.ppg-eight-cache'/'unknown.pt';unknown.write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError,'Unknown PPG cache artifact'):
                value_buffer.retire_disabled(SOURCE_ROOT,run,460)
            old=run/'.ppg-value-cache'/'generation-000210.npy';self.assertTrue(old.exists());unknown.unlink()
            from brass_ai.recovery import retention
            real=retention.link
            with patch.object(retention,'link',side_effect=lambda p: Path(p)==old or real(p)):
                with self.assertRaisesRegex(ValueError,'Linked retention target'):
                    value_buffer.retire_disabled(SOURCE_ROOT,run,460)
            self.assertTrue(old.exists())
            write_json(run/'.ppg-eight-cache'/'owner.json',dict(contract='unknown'))
            with self.assertRaisesRegex(ValueError,'Unknown PPG cache owner'):
                value_buffer.retire_disabled(SOURCE_ROOT,run,460)
            self.assertTrue(old.exists())

    def test_fresh_PPO_only_startup_retires_owned_stale_auxiliary_caches(self):
        model,opt,cfg=setup();cfg=value_phase.disabled_options(cfg)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);expected=caches(run)
            write_json(run/'state.json',dict(iteration=0,signature=dict(config=cfg)))
            phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            result=phase.startup(0)
            self.assertEqual(result['deleted_bytes'],expected)
            self.assertEqual(phase.state['generation'],0);self.assertIsNone(phase.buffer)
            self.assertEqual(phase.startup(0)['deleted_files'],[])

    def test_disabled_v5_real_PPO_and_gates_without_clones_or_cache(self):
        model,opt,cfg,raw=rolling_checkpoint();new=value_phase.disabled_options(cfg)
        p=TrainingPhases(model,new,raw=raw,generation=460)
        behavior,data=collect(model,p,461,branches=True)
        actor=ppo(model,opt,new,behavior,data,p)
        weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());value_adam=copy.deepcopy(p.optimizer.state_dict());rng=torch.get_rng_state().clone()
        with patch.object(p.optimizer,'step',side_effect=AssertionError('PPG step')), \
             patch.object(value_phase,'RawPolicyReference',side_effect=AssertionError('PPG clones')):
            report=p.fit(model)
        self.assertFalse(report['enabled']);self.assertEqual(report['optimizer_steps'],0)
        self.assertEqual(value_phase.critic_work(report),p.ppo_work)
        self.assertGreater(p.ppo_work['optimizer_steps'],0)
        self.assertNotIn('schedule',report);self.assertNotIn('buffer',report)
        self.assertTrue(_same_payload(weights,model.state_dict()));self.assertTrue(_same_payload(adam,opt.state_dict()))
        self.assertTrue(_same_payload(value_adam,p.optimizer.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        finish(p,actor,461)
        self.assertEqual(actor['gae']['gate']['stage_evidence']['source'],'ppo')
        self.assertEqual(p.state['active_lambda'],.95);self.assertEqual(p.state['value_iterations'],21)

    def test_config_helper_is_locked_idempotent_and_never_edits_run(self):
        spec=importlib.util.spec_from_file_location('ppg_off_helper',SOURCE_ROOT/'scripts/joint7-ppo-only.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            state=read(EXPORT/'state.json');write_json(run/'state.json',state);before=sha(run/'state.json')
            output='configs/step5/ppo-off.json'
            result=helper.generate(root,run,output,460)
            self.assertFalse(result['run_modified']);self.assertEqual(result['changed_blocks'],['value_phase'])
            self.assertEqual(result,helper.generate(root,run,output,460))
            self.assertEqual(sha(run/'state.json'),before)
            result=helper.generate(root,run,'configs/step5/ppo-off-2048.json',460,2048,524288)
            self.assertEqual(set(result['changed_blocks']),{'value_phase','update'})
            actual=read(root/'configs/step5/ppo-off-2048.json')
            self.assertEqual(actual,read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only-g460.json'))
            with self.assertRaises(ValueError):helper.generate(root,run,output,460,4096)
            with self.assertRaises(ValueError):helper.generate(root,run,output,460,2048,1048576)
            with self.assertRaisesRegex(ValueError,'Unexpected committed generation'):helper.generate(root,run,output,418)
            with self.assertRaises(ValueError):helper.generate(root,run,'artifacts/unsafe.json',460)

    def test_2048_microbatch_still_respects_candidate_cells_and_coverage(self):
        from types import SimpleNamespace
        from brass_ai.recovery.updates import Dataset
        sizes=[4]*2049+[128]*2048+[772]*1500
        rows=[(None,None,None,SimpleNamespace(choice_ids=range(n))) for n in sizes]
        ds=object.__new__(Dataset);ds.data=rows;ds.encoded=[r[3] for r in rows]
        parts=list(ds.microbatches(range(len(rows)),2048,524288,True))
        self.assertEqual(sorted(i for part in parts for i in part),list(range(len(rows))))
        self.assertEqual(len(parts[0]),2048)
        self.assertTrue(all(len(part)<=2048 and len(part)*max(sizes[i] for i in part)<=524288 for part in parts))
        self.assertTrue(any(len(part)<2048 for part in parts if any(sizes[i]==772 for i in part)))
        self.assertTrue(any(len(part)==524288//772 for part in parts))


if __name__=='__main__':unittest.main()
