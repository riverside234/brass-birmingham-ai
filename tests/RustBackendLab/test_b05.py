"""Bounded B05 commit, optimizer ownership, true PPG boundary and rollback."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint, resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.search_value import ordinary_parameters
from brass_ai.recovery.search_value_config import adopt
from brass_ai.recovery.search_value_iteration import SearchValueIteration, stamp
from brass_ai.recovery.search_value_training import StageMachine
from brass_ai.recovery.search_value_rows import ValueBuffer
from brass_ai.rust_backend.lab_snapshot import tree_digest
from test_b025 import config, model_fixture
from test_b045 import value_fixture


def recipe(mode='on'):
    c=adopt(config());c['teacher_phase']['mode']=mode
    c['teacher_phase']['search_value']['stage']='A'
    return c


class Lifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def directory(self):
        t=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(t.cleanup)
        return Path(t.name)

    def test_off_and_shadow_preserve_weights_gradients_rng_and_optimizers(self):
        for mode in ('off','shadow'):
            m=model_fixture();opt=torch.optim.AdamW(ordinary_parameters(m))
            next(m.parameters()).grad=torch.ones_like(next(m.parameters()))
            before=tree_digest((m.state_dict(),opt.state_dict(),torch.get_rng_state(),[p.grad for p in m.parameters()]))
            phase=SearchValueIteration(self.directory(),recipe(mode),dict(iteration=10),m)
            phases=SimpleNamespace();phase.prepare_ppg(phases,11);r=phase.finish_ppg()
            self.assertIsNone(phases.post_value_fit);self.assertFalse(r['updated'])
            self.assertEqual(before,tree_digest((m.state_dict(),opt.state_dict(),torch.get_rng_state(),[p.grad for p in m.parameters()])))

    def test_three_commits_restore_head_adams_stage_and_old_value_age(self):
        root=self.directory();m=model_fixture();opt=torch.optim.AdamW(ordinary_parameters(m),lr=1e-5)
        state=dict(iteration=10);c=recipe();phase=SearchValueIteration(root,c,state,m)
        row,_,_=value_fixture(m);row['terminal_utility']=row['u']
        phase.phase.buffer.publish(10,'train',[row],game_id=row['game_id'])
        for generation in (11,12,13):
            phase.phase.companion.head_only([row],generation=generation,steps=1)
            payload=phase.payload(generation)
            path=root/f'g{generation}.pt'
            save_checkpoint(path,m,opt,step=generation,training=dict(search_value=payload))
            restored,raw=load_checkpoint(path)
            # Checkpoint before state commit cannot be paired with old state.
            with self.assertRaisesRegex(ValueError,'checkpoint/state'):
                SearchValueIteration(root,c,state,restored,raw)
            state=dict(iteration=generation,search_value=stamp(payload))
            phase.committed(state)
            other=torch.optim.AdamW(ordinary_parameters(restored),lr=1e-5);resume_optimizer(raw,other)
            again=SearchValueIteration(root,c,state,restored,raw)
            self.assertEqual(tree_digest(phase.phase.companion.snapshot()),tree_digest(again.phase.companion.snapshot()))
            self.assertEqual(tree_digest(opt.state_dict()),tree_digest(other.state_dict()))
            self.assertEqual(again.phase.buffer.rows(generation),[row])
            head={id(p) for p in restored.search_value_head.parameters()}
            self.assertFalse(head & {id(p) for g in other.param_groups for p in g['params']})
            bad=copy.deepcopy(raw);bad['training']['search_value']['companion']=None
            with self.assertRaisesRegex(ValueError,'checkpoint/state'):
                SearchValueIteration(root,c,state,restored,bad)
            from brass_ai.recovery.local_scoring_auxiliary import bundle_id
            missing=copy.deepcopy(payload);companion=missing['companion'];companion['optimizer']['state'].clear()
            companion['identity']=bundle_id({k:v for k,v in companion.items() if k!='identity'})
            with self.assertRaisesRegex(ValueError,'Adam moments'):
                SearchValueIteration(root,c,dict(iteration=generation,search_value=stamp(missing)),restored,
                    dict(training=dict(search_value=missing)))
            m,opt,phase=restored,other,again

    def test_all_stage_transitions_resume_exactly_without_manual_advancement(self):
        m=StageMachine();gate=dict(passed=True,mean=.1,lower=.01,upper=.2,full_window=True)
        ordinary=dict(critic_ev=.4,ppo_kl=.01,entropy=.5,economy=.8)
        for g in range(44):
            before=copy.deepcopy(m.state)
            h2h=dict(games=256,passed=True,upper=.6) if m.state['stage']=='head_to_head' else None
            after=m.advance(g,gate,ordinary=ordinary,h2h=h2h)
            retry=StageMachine(saved=before)
            self.assertEqual(after,retry.advance(g,gate,ordinary=ordinary,h2h=h2h))
            m=StageMachine(saved=after)
            with self.assertRaises(ValueError):m.advance(g,gate)
        self.assertEqual(m.state['stage'],'C');self.assertEqual(m.state['scale'],1.)
        m.advance(44,gate,ordinary=dict(ordinary,entropy=10.))
        self.assertEqual(m.state['stage'],'B');self.assertEqual(m.state['scale'],0.)

    def test_stage_guard_and_exception_restore_full_post_ppg_state(self):
        m=model_fixture();phase=SearchValueIteration(self.directory(),recipe(),dict(iteration=10),m)
        opt=torch.optim.Adam(ordinary_parameters(m),lr=1e-5)
        phase.phase.machine.state.update(stage='C',scale=.1)
        phase.prepare_ppg(None,11)
        before=tree_digest((m.state_dict(),opt.state_dict(),phase.phase.companion.snapshot()))
        def change(*a,**k):
            next(m.parameters()).data.add_(1)
            phase.phase.companion.steps+=1
            phase.phase.machine.state.update(stage='B',scale=0.)
            return dict(updated=True)
        with patch.object(phase.phase,'after_ppg',side_effect=change):phase._finish(dict(value_optimizer=opt))
        self.assertTrue(phase.report['stage_guard_rollback']);self.assertEqual(phase.phase.machine.state['stage'],'B')
        self.assertEqual(before,tree_digest((m.state_dict(),opt.state_dict(),phase.phase.companion.snapshot())))
        phase.phase.machine.state.update(stage='C',scale=.1)
        def fail(*a,**k):change();raise RuntimeError('interrupted update')
        with patch.object(phase.phase,'after_ppg',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):phase._finish(dict(value_optimizer=opt))
        self.assertEqual(phase.phase.machine.state['stage'],'C')
        self.assertEqual(before,tree_digest((m.state_dict(),opt.state_dict(),phase.phase.companion.snapshot())))

    def test_head_only_interrupt_restores_weights_adam_and_stage(self):
        m=model_fixture();phase=SearchValueIteration(self.directory(),recipe(),dict(iteration=10),m)
        row,_,_=value_fixture(m);row['terminal_utility']=row['u']
        phase.prepare_ppg(None,11)
        before=tree_digest((m.state_dict(),phase.phase.payload()))
        def fail(*a,**k):
            phase.phase.companion.head_only([row],generation=11,steps=1)
            phase.phase.machine.state.update(stage='B')
            raise RuntimeError('interrupted head update')
        with patch.object(phase.phase,'after_ppg',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):phase.finish_ppg()
        self.assertEqual(before,tree_digest((m.state_dict(),phase.phase.payload())))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_unknown_partial_duplicate_and_missing_value_files_fail_closed(self):
        root=self.directory();m=model_fixture();row,_,_=value_fixture(m)
        b=ValueBuffer(root,row['reward_id']);b.publish(10,'train',[row])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            ValueBuffer(root,row['reward_id'],receipts=b.receipts*2)
        stray=root/'unknown.partial';stray.write_bytes(b'interrupted')
        with self.assertRaisesRegex(ValueError,'Unknown/uncommitted'):
            ValueBuffer(root,row['reward_id'],receipts=b.receipts)
        stray.unlink();(root/b.receipts[0]['file']).unlink()
        with self.assertRaises(OSError):ValueBuffer(root,row['reward_id'],receipts=b.receipts)

    def test_post_ppg_ages_by_collection_generation_not_next_checkpoint(self):
        m=model_fixture();phase=SearchValueIteration(self.directory(),recipe(),dict(iteration=10),m)
        row,_,_=value_fixture(m)
        for generation in range(3,11):
            value=dict(row,generation=generation,game_id='fixture-'+str(generation))
            phase.phase.buffer.publish(generation,'heldout',[dict(value,split='heldout')])
        phase.prepare_ppg(None,11)
        def finish(generation,**kwargs):
            self.assertEqual(generation,10)
            self.assertEqual(len(phase.phase.buffer.rows(generation,'heldout')),8)
            return dict(updated=False)
        with patch.object(phase.phase,'after_ppg',side_effect=finish):phase.finish_ppg()
        self.assertEqual(phase.payload(11)['generation'],11)

    def test_static_profile_change_retires_labels_and_beta_together(self):
        from brass_ai.recovery.teacher_phase import TeacherPhase
        from test_b025 import example
        from brass_ai.learning.batching import model_id
        root=self.directory();m=model_fixture();c=config()
        state=dict(iteration=10,history=[dict(generation=10,model_id=model_id(m))])
        before=TeacherPhase(SOURCE_ROOT,root,c,state,m)
        row,e=example(m)
        before.buffer.publish(10,'train',[row],model_id=e['model_id'],reward_id=e['reward_id'])
        before.weight=dict(beta=.123,calibrated=True);saved=before.payload(11)
        state.update(iteration=11,teacher_phase=saved,teacher_seed_cursor=before.cursor)
        with patch('brass_ai.recovery.teacher_phase.buffer_identity',return_value='reviewed-new-static-profile'):
            after=TeacherPhase(SOURCE_ROOT,root,c,state,m,dict(training=dict(teacher_phase=saved)))
        self.assertEqual(after.weight,{});self.assertEqual(after.buffer.rows(11),[])
        self.assertEqual(len(after.buffer.payload()['retired']),1)
        self.assertTrue((root/saved['buffer']['receipts'][0]['path']).exists())


class PPGBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_task_teacher_uses_real_ordinary_actor_gradient(self):
        from brass_ai.rust_backend.lab_b05_smoke import task_teacher_step
        self.assertTrue(task_teacher_step('cpu')['passed'])

    def test_callback_uses_same_pre_ppg_policy_and_does_not_change_ordinary_selection(self):
        import test_ppg_g210 as fixture
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery import value_phase
        m,opt,c=fixture.config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as t:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(t));fixture.prepare(m,p,1)
            before=value_phase.RawPolicyReference(m,p.dataset,p.actor_ids,c['update'])
            seen=[]
            def post(**kw):
                seen.append(kw['report']['optimizer_steps'])
                self.assertIs(kw['optimizer'],p.optimizer)
                for i,probability in kw['reference'].rows.items():
                    torch.testing.assert_close(probability,before.rows[i],atol=1e-7,rtol=1e-6)
                self.assertAlmostEqual(kw['reference'].measure(m,kw['probe_ids']),kw['report']['probe_kl'],places=7)
                # The callback runs before the pooled dataset's owners close.
                self.assertGreater(value_phase.metrics(m,kw['dataset'],kw['heldout_ids'][:1],c['update'])['rows'],0)
            p.post_value_fit=post;p.fit(m)
            self.assertEqual(len(seen),1)

    def test_latest_recipe_short_update_commit_and_reload(self):
        import json
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.learning.model import ModelConfig
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        from brass_ai.rust_backend.lab_b05_smoke import ordinary_step
        c=json.loads((SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g1250-20261107T041141451177Z/recorded-config.json').read_text())
        c['model_size']='small'
        m=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(c)))
        opt=torch.optim.AdamW(ordinary_parameters(m),lr=c['learning_rate'])
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as t:
            path=Path(t);p=TrainingPhases(m,c,root=SOURCE_ROOT,run=path)
            companion=Companion(m,c['local_scoring_auxiliary']);p.local_scoring=companion
            restored=SimpleNamespace(model=m,optimizer=opt,phases=p,companion=companion,
                state=dict(iteration=0),raw=dict(training={}))
            report=ordinary_step(restored,c,path)
            self.assertTrue(report['checkpoint_restored']);self.assertGreater(report['ppo_steps'],0)


if __name__=='__main__':unittest.main()
