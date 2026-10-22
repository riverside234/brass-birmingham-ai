"""Full checkpoint adoption and interrupted cleanup without rewriting a run."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.recovery import local_scoring_adoption as adoption,local_scoring_auxiliary as auxiliary
from brass_ai.recovery import ppg_config,ppg_epochs,ppg_replay,gae,critic_data
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_ppg_eight import setup


class Adoption(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name);model,opt,cfg=setup();cfg=ppg_config.coverage_options(cfg)
        from brass_ai.recovery.exploration import HIGH_BRANCH
        cfg['branching']=copy.deepcopy(HIGH_BRANCH)
        cfg=critic_data.fixed_config(cfg)
        phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=self.run)
        parameter=next(model.parameters());parameter.grad=torch.ones_like(parameter)
        opt.step();phase.optimizer.step();opt.zero_grad();phase.optimizer.zero_grad()
        generation=560;cache=self.run/'.ppg-eight-cache';old_files={}
        for ext in ('critic.npy','actions.npy'):
            path=cache/f'generation-000560.{ext}';path.write_bytes(b'owned fixture payload '+ext.encode())
            old_files[ext]=dict(bytes=path.stat().st_size,sha256=sha(path))
        old_receipt=dict(generation=560,files=old_files)
        write_json(cache/'generation-000560.json',old_receipt)
        critic=critic_data.initial(cfg['critic_data']);critic['generation']=generation
        clock=dict(ppg_config.initial(540),generation=560,last_phase=556,phases=2)
        epochs=ppg_epochs.initial(540,cfg['value_phase'])
        for g in (548,556):
            epochs,_=ppg_epochs.advance(epochs,dict(scheduled=True,epoch_cap_used=2,steps_per_epoch=4,
                planned_steps=8,best_budget_step=4,kl_guard_tripped=False),cfg['value_phase'],g)
        # This controller's full shape and safety history are validated below.
        saved=dict(contract='actor-value-phases/10.0.0',options=phase.options,gae=gae.initial(cfg['advantage'],generation),
            beta_clone=8.,kl_trips=[True,False],optimizer=phase.optimizer.state_dict(),critic_data=critic,
            critic_evidence=phase.critic_evidence,schedule=clock,cache_origin=540,epoch_control=epochs,rolling_buffer=[old_receipt])
        checkpoint=self.run/'generation-0560.pt'
        identity=save_checkpoint(checkpoint,model,opt,step=generation,training={KEY:saved,'test_controller':{'multiplier':1.2,'window':[.003,.004]}})
        self.state=dict(iteration=generation,games_done=179200,next_seed=202020,context_counter=202,
            signature=dict(config=cfg),history=[dict(generation=generation,path=checkpoint.relative_to(SOURCE_ROOT).as_posix(),model_id=identity,sha256=sha(checkpoint))])
        write_json(self.run/'state.json',self.state)
        self.model,self.raw=load_checkpoint(checkpoint);self.old_sha=sha(checkpoint);self.checkpoint=checkpoint
        self.cfg=adoption.configure(self.state)

    def prepare(self):return adoption.prepare(SOURCE_ROOT,self.run,self.state,self.raw,self.cfg,reviewed=True)

    def test_resume_preserves_Adam_RNG_gates_and_requires_eight_fresh_commits(self):
        raw=copy.deepcopy(self.raw);rng=torch.get_rng_state().clone()
        phase_raw,receipt,adopting=self.prepare()
        p=TrainingPhases(self.model,self.cfg,raw=phase_raw,generation=560,root=SOURCE_ROOT,run=self.run)
        heads=auxiliary.Companion(self.model,self.cfg[auxiliary.KEY],raw=self.raw,generation=560,adopting=adopting)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(_same_payload(raw,self.raw))
        self.assertEqual(p.beta,8.);self.assertEqual(p.trips,[True,False]);self.assertEqual(p.schedule,ppg_config.initial(560))
        self.assertTrue(_same_payload(p.optimizer.state_dict(),raw['training'][KEY]['optimizer']))
        self.assertEqual(p.state,raw['training'][KEY]['gae']);self.assertEqual(p.critic_state,raw['training'][KEY]['critic_data'])
        self.assertEqual(sha(self.checkpoint),self.old_sha);self.assertEqual(heads.scale,0.)
        clock=p.schedule;due=[]
        for g in range(561,577):
            ran=ppg_config.due(clock)
            if ran:due.append(g)
            clock=ppg_config.advance(clock,g,ran)
        self.assertEqual(due,[568,576])
        # Nothing is deleted before durable adoption.
        self.assertTrue((self.run/'.ppg-eight-cache/generation-000560.critic.npy').exists())

    def test_phase_peak_lr_roundoff_does_not_break_exact_resume(self):
        raw,_,_=self.prepare();base=self.cfg['value_phase']['learning_rate']
        rate=base*(1+2e-16)
        raw['training'][KEY]['optimizer']['param_groups'][0]['lr']=rate
        phase=TrainingPhases(self.model,self.cfg,raw=raw,generation=560,root=SOURCE_ROOT,run=self.run)
        self.assertEqual(phase.optimizer.param_groups[0]['lr'],rate)
        raw['training'][KEY]['optimizer']['param_groups'][0]['lr']=base*1.001
        with self.assertRaisesRegex(ValueError,'optimizer settings'):TrainingPhases(self.model,self.cfg,raw=raw,generation=560,root=SOURCE_ROOT,run=self.run)
        a=auxiliary.Companion(self.model,self.cfg[auxiliary.KEY],raw=self.raw,generation=560,adopting=True)
        a.begin('ppg',rate_factor=1+2e-16)
        self.assertEqual(a.optimizers['ppg'].param_groups[0]['lr'],self.cfg[auxiliary.KEY]['optimizer']['learning_rate'])

    def test_crash_cleanup_restart_never_deletes_new_generations(self):
        transformed,receipt,_=self.prepare()
        receipt=adoption.archive_metadata(self.run,receipt,self.raw)
        self.assertEqual(adoption.archive_metadata(self.run,receipt,self.raw),receipt)
        archived=self.run/receipt['old_cache_metadata']['path']
        self.assertEqual(read(archived)['rolling_buffer'],self.raw['training'][KEY]['rolling_buffer'])
        with self.assertRaisesRegex(ValueError,'durable'):adoption.cleanup(SOURCE_ROOT,self.run,dict(self.state,local_scoring_adoption=receipt))
        self.state['signature']['config']=self.cfg;self.state['local_scoring_adoption']=receipt
        write_json(self.run/'state.json',self.state)
        # A crash after one old payload deletion is allowed; no second reset.
        (self.run/'.ppg-eight-cache/generation-000560.critic.npy').unlink()
        new=self.run/'.ppg-eight-cache/generation-000561.critic.npy';new.write_bytes(b'new committed generation')
        again,r,adopting=adoption.prepare(SOURCE_ROOT,self.run,self.state,self.raw,self.cfg)
        self.assertTrue(_same_payload(again,transformed));self.assertTrue(adopting)
        result=adoption.cleanup(SOURCE_ROOT,self.run,self.state)
        self.assertTrue(result['cleanup_complete']);self.assertTrue(new.exists())
        self.assertIsNone(adoption.cleanup(SOURCE_ROOT,self.run,self.state))
        self.assertTrue(archived.exists())
        self.assertEqual(sha(self.checkpoint),self.old_sha)

    def test_unknown_cache_aborts_before_any_delete_and_no_unrelated_config_change(self):
        existing=self.run/'.ppg-eight-cache/generation-000560.critic.npy'
        unknown=self.run/'.ppg-eight-cache/do-not-delete.txt';unknown.write_text('not a cache')
        with self.assertRaisesRegex(ValueError,'Unknown'):self.prepare()
        self.assertTrue(existing.exists());unknown.unlink()
        self.assertTrue(adoption.compatible(self.state,self.cfg))
        bad=copy.deepcopy(self.cfg);bad['learning_rate']*=2
        self.assertFalse(adoption.compatible(self.state,bad))
        bad=copy.deepcopy(self.cfg);bad[auxiliary.KEY]['adoption']['checkpoint_sha256']='f'*64
        self.assertFalse(adoption.compatible(self.state,bad))

    def test_g560_review_route_helper_and_mode_only_changes(self):
        import importlib.util
        from brass_ai.recovery.resume import check_signature
        from brass_ai.recovery.config import validate_config,latest_model_config
        from brass_ai.step5.league import implementation_id
        export=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0560-20261022T202410715648Z'
        state=read(export/'state.json');config=adoption.configure(state);validate_config(config)
        self.assertEqual({k for k in config if config.get(k)!=state['signature']['config'].get(k)},{auxiliary.KEY,'branching','ppo_lr_controller'})
        self.assertEqual(config['ppo_lr_controller']['low_ratio'],.5)
        expected=copy.deepcopy(state['signature']['config']['ppo_lr_controller']);expected['low_ratio']=.5
        self.assertEqual(config['ppo_lr_controller'],expected)
        signature=dict(state['signature'],config=config,implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,reviewed=True))
        bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
        with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        adopted=dict(state,signature=signature)
        for mode in ('off','shadow','apply'):
            changed=adoption.configure(adopted,mode)
            check_signature(adopted,dict(signature,config=changed),reviewed=True)
        fresh=latest_model_config(state['signature']['config'],state['signature']['config']['imitation_checkpoint'])
        self.assertEqual(fresh[auxiliary.KEY]['mode'],'apply');self.assertIsNone(fresh[auxiliary.KEY]['adoption'])
        self.assertEqual(fresh['branching']['rate_per_decision'],.01)
        spec=importlib.util.spec_from_file_location('local_helper',SOURCE_ROOT/'scripts/configure-local-scoring-aux.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        # Bind the real fixture file to an authored state using the actual g560
        # config/source. The production helper still verifies its bytes.
        state['history'][-1]=self.state['history'][-1];write_json(self.run/'state.json',state)
        checksum=sha(self.run/'state.json');output=SOURCE_ROOT/'configs/step5'/('local-test-'+self.run.name+'.json')
        self.addCleanup(lambda:output.unlink(missing_ok=True))
        for _ in range(2):
            preview=helper.generate(SOURCE_ROOT,self.run,output,560,fresh_ppg_buffer=True)
            self.assertFalse(preview['run_modified']);self.assertEqual(preview['next_fresh_ppg'],568)
            self.assertEqual(preview['lr_raise_below_ratio'],.5);self.assertEqual(preview['lr_raise_below_kl'],.005)
            self.assertEqual(sha(self.run/'state.json'),checksum)
        with self.assertRaisesRegex(ValueError,'Stale'):helper.generate(SOURCE_ROOT,self.run,output,559,fresh_ppg_buffer=True)

    def test_diagnostics_fix_resumes_already_adopted_source_without_config_reset(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        export=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0560-20261022T202410715648Z'
        state=read(export/'state.json');config=adoption.configure(state)
        state['signature']=dict(state['signature'],config=config,
            implementation='c33ff1ba936905fe06c4b6436d316a5073c9049be4a8d8d0e9c734f87e071314')
        before=copy.deepcopy(state)
        target=dict(state['signature'],implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        self.assertEqual(state,before)
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):
            check_signature(state,target)
        for key in ('learning_rate','exploration'):
            bad=copy.deepcopy(target);bad['config'][key]+=.001
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)


if __name__=='__main__':unittest.main()
