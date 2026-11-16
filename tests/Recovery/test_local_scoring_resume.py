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

    def bind_source_checkpoint(self):
        """Bind the reduced fixture like the real pre-adoption g560 checkpoint."""
        from brass_ai.learning.contracts import compatibility
        config=self.state['signature']['config']
        self.state.update(format='step5-recovery-league/2.2.0',next_search_games=180000,
            signature=dict(config=config,compatibility=compatibility(),
                implementation='b95f45e2b0be6672a4efb7a3cabf9e2fab3e152a667e7f229b6c09ab5cb01c71'))
        if config.get('legacy_boost') is not None:
            self.state['legacy_boost']=dict(start_games=0,
                until_games=sum(s['games'] for s in config['legacy_boost']['stages']),config=digest(config['legacy_boost']))
        optimizer=torch.optim.AdamW(self.model.parameters());resume_optimizer(self.raw,optimizer)
        save_checkpoint(self.checkpoint,self.model,optimizer,step=560,training=dict(self.raw['training'],
            signature=digest(self.state['signature']),fresh_run=self.run.relative_to(SOURCE_ROOT).as_posix()))
        self.old_sha=sha(self.checkpoint);self.state['history'][-1]['sha256']=self.old_sha
        self.state['population']=copy.deepcopy(self.state['history'])
        self.state['evaluation_anchor']=copy.deepcopy(self.state['history'][-1])
        self.model,self.raw=load_checkpoint(self.checkpoint);self.cfg=adoption.configure(self.state)
        write_json(self.run/'state.json',self.state)

    def commit_initial_adoption(self):
        from brass_ai.recovery.resume import load_resume,commit_transition
        self.bind_source_checkpoint()
        signature=dict(self.state['signature'],config=self.cfg,
            implementation='c33ff1ba936905fe06c4b6436d316a5073c9049be4a8d8d0e9c734f87e071314')
        model,raw,changed=load_resume(SOURCE_ROOT,self.run,self.state,signature,reviewed=True)
        transformed,receipt,_=adoption.prepare(SOURCE_ROOT,self.run,self.state,raw,self.cfg,reviewed=changed)
        receipt=adoption.archive_metadata(self.run,receipt,raw)
        self.state=commit_transition(self.run,self.state,signature,local_scoring_adoption=receipt)
        return transformed

    def test_source_commit_preserves_pending_cleanup_and_rejects_changed_adoption_receipts(self):
        from brass_ai.recovery.resume import commit_transition
        from brass_ai.step5.league import implementation_id
        self.commit_initial_adoption();durable=copy.deepcopy(self.state)
        receipt=durable['local_scoring_adoption'];self.assertFalse(receipt['cleanup_complete'])
        target=dict(durable['signature'],implementation=implementation_id())
        before={p:sha(p) for p in self.run.rglob('*') if p.is_file()}
        for field,value in (('new_origin',561),('next_phase',569),('cleanup_complete',True)):
            bad=copy.deepcopy(receipt);bad[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'adoption receipt differs'):
                commit_transition(self.run,durable,target,local_scoring_adoption=bad)
            self.assertEqual({p:sha(p) for p in self.run.rglob('*') if p.is_file()},before)
        committed=commit_transition(self.run,durable,target,local_scoring_adoption=copy.deepcopy(receipt))
        self.assertEqual(committed['local_scoring_adoption'],receipt)
        cleaned=adoption.cleanup(SOURCE_ROOT,self.run,committed)
        self.assertTrue(cleaned['cleanup_complete']);self.assertEqual(cleaned['new_origin'],560)
        self.assertIsNone(adoption.cleanup(SOURCE_ROOT,self.run,committed))

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

if __name__=='__main__':unittest.main()
