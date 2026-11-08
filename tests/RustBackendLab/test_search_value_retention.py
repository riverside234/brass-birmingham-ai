"""Small loss-retention and fresh-buffer recovery regressions; no benchmark."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT as ROOT, digest
from brass_ai.learning.search_value import ordinary_parameters
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.search_value_phase import SearchValuePhase
from brass_ai.recovery.search_value_retention import reserve_games, HeldoutLoss
from brass_ai.recovery.search_value_training import unconditional_pass
from brass_ai.rust_backend.lab_teacher_activation import configure
from brass_ai.bank import read
from test_b025 import model_fixture
from test_b045 import value_fixture


class Retention(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.model=model_fixture()
        options=configure(read(ROOT/'results/joint7-v11-2-g1290-trained-restart-g1284-20261108T193427364056Z/recorded-config.json'),
            revise_stages=True)['teacher_phase']['search_value']
        self.phase=SearchValuePhase(Path(self.tmp.name)/'values',self.model,options,digest(self.model.config.utility))
        self.row,_,_=value_fixture(self.model,generation=1284)
        self.row.update(reward_id=self.phase.reward_id,terminal_utility=self.row['u'])

    def rows(self):
        return [dict(self.row,game_id=f'game-{age}-{game}',generation=1284-age,index=i)
            for age in range(8) for game in range(2) for i in range(64)]

    def guard(self, kl=lambda:.001):
        return dict(value_optimizer=torch.optim.Adam(ordinary_parameters(self.model),lr=1e-5),
            scale=.1,cloning_loss=lambda:next(self.model.parameters()).sum()*0,
            measure_joint_kl=kl,beta=.25,rate_for_step=lambda *_:1e-5)

    def test_whole_game_partition_repeats_without_consuming_global_rng_or_mutating_rows(self):
        rows=self.rows();before=copy.deepcopy(rows);rng=torch.get_rng_state();numpy=np.random.get_state()
        train,held,receipt=reserve_games(rows,1284,4504)
        again=reserve_games(list(reversed(rows)),1284,4504)
        self.assertEqual(receipt['games'],again[2]['games']);self.assertEqual(len(held),512)
        self.assertEqual(len(train),512);self.assertTrue({r['game_id'] for r in train}.isdisjoint(r['game_id'] for r in held))
        self.assertEqual(rows,before);self.assertTrue(all(r['split']=='heldout' for r in held))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        after=np.random.get_state();self.assertEqual(numpy[0],after[0]);np.testing.assert_array_equal(numpy[1],after[1]);self.assertEqual(numpy[2:],after[2:])
        self.assertNotEqual(receipt['games'],reserve_games(rows,1284,4505)[2]['games'])

    def test_incomplete_duplicate_or_wrong_age_games_rejected_and_small_window_deferred(self):
        rows=self.rows()
        for invalid in (rows[:-1],rows+[rows[0]], [dict(r,generation=1276) for r in rows], [dict(r,split='heldout') for r in rows]):
            with self.assertRaises(ValueError):reserve_games(invalid,1284,4504)
        self.assertFalse(reserve_games(rows[:512],1284,4504)[2]['ready'])

    def test_packed_holdout_matches_independent_predictions_and_has_no_gradients(self):
        _,held,_=reserve_games(self.rows(),1284,4504)
        from brass_ai.recovery.search_value_training import predict_rows
        main,delta=predict_rows(self.model,held)
        expected=float(np.mean((main+delta-np.asarray([r['target'] for r in held]))**2))
        self.assertAlmostEqual(HeldoutLoss(self.model,held)(),expected,places=6)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_loss_increase_restores_weights_both_adams_and_companion_steps(self):
        train,held,_=reserve_games(self.rows(),1284,4504);guard=self.guard();c=self.phase.companion
        before=copy.deepcopy((self.model.state_dict(),guard['value_optimizer'].state_dict(),c.snapshot()))
        with patch('brass_ai.recovery.search_value_retention.HeldoutLoss',return_value=iter([1.,1.1]).__next__):
            result=unconditional_pass(c,train[:1],heldout=held,**guard)
        self.assertEqual(result['steps'],0);self.assertEqual(result['early_stop_reason'],'heldout_loss_increased')
        self.assertTrue(_same_payload(before,(self.model.state_dict(),guard['value_optimizer'].state_dict(),c.snapshot())))

    def test_second_interval_loss_rejection_keeps_first_accepted_interval_exactly(self):
        train,held,_=reserve_games(self.rows(),1284,4504);guard=self.guard();c=self.phase.companion;snap=[]
        def kl():
            snap.append(copy.deepcopy((self.model.state_dict(),guard['value_optimizer'].state_dict(),c.snapshot())))
            return .001
        guard['measure_joint_kl']=kl
        with patch('brass_ai.recovery.search_value_retention.HeldoutLoss',return_value=iter([1.,.9,1.]).__next__):
            result=unconditional_pass(c,train[:1],heldout=held,**guard)
        self.assertEqual(result['steps'],16);self.assertEqual(c.steps,16)
        self.assertEqual(result['heldout_loss_after'],.9)
        self.assertTrue(_same_payload(snap[1],(self.model.state_dict(),guard['value_optimizer'].state_dict(),c.snapshot())))

    def test_loss_ties_retained_hard_kl_and_cancellation_still_restore(self):
        train,held,_=reserve_games(self.rows(),1284,4504);guard=self.guard();c=self.phase.companion
        with patch('brass_ai.recovery.search_value_retention.HeldoutLoss',return_value=lambda:1.):
            result=unconditional_pass(c,train[:1],heldout=held,**guard)
        self.assertEqual(result['steps'],32)
        for failure in ('kl','cancel'):
            before=copy.deepcopy((self.model.state_dict(),guard['value_optimizer'].state_dict(),c.snapshot()))
            values=iter([.001,.004])
            guard['measure_joint_kl']=(values.__next__ if failure=='kl' else lambda:next(values) if c.steps==32 else (_ for _ in ()).throw(RuntimeError('cancelled')))
            with patch('brass_ai.recovery.search_value_retention.HeldoutLoss',return_value=lambda:1.):
                if failure=='cancel':
                    with self.assertRaisesRegex(RuntimeError,'cancelled'):unconditional_pass(c,train[:1],heldout=held,**guard)
                else:self.assertEqual(unconditional_pass(c,train[:1],heldout=held,**guard)['steps'],0)
            self.assertTrue(_same_payload(before,(self.model.state_dict(),guard['value_optimizer'].state_dict(),c.snapshot())))

    def test_stage_a_counter_and_adam_pause_until_eight_fresh_ages_and_resume_once(self):
        p=self.phase;p.machine.state['stage_a_iterations']=4
        p.machine.state['buffer_rebuild']=dict(origin_generation=1284,required_ages=8,ready_generation=None,instruction_id='test')
        before=copy.deepcopy((self.model.state_dict(),p.companion.snapshot()))
        for generation in range(1284,1292):
            rows=[dict(self.row,generation=generation,game_id=f'fresh-{generation}',index=i) for i in range(64)]
            p.buffer.publish(generation,'train',rows,game_id=rows[0]['game_id'])
            result=p.after_ppg(generation,mode='on')
            if generation<1291:
                self.assertFalse(result['updated']);self.assertEqual(result['stage']['stage_a_iterations'],4)
                self.assertTrue(_same_payload(before,(self.model.state_dict(),p.companion.snapshot())))
                saved=p.payload();p=SearchValuePhase(Path(self.tmp.name)/'values',self.model,p.options,p.reward_id,saved=saved)
            else:
                self.assertTrue(result['updated']);self.assertEqual(result['stage']['stage_a_iterations'],5)
                self.assertEqual(result['stage']['buffer_rebuild']['ready_generation'],1291)

    def test_real_ppg_adam_and_state_restore_with_missing_old_buffers_then_keep_fresh_age(self):
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery.transfer_resume import ppg_payload
        from brass_ai.recovery import gae
        from brass_ai.step5.league import write_json
        from test_ppg_g210 import prepare
        config=read(ROOT/'results/joint7-v11-2-g1290-trained-restart-g1284-20261108T193427364056Z/recorded-config.json')
        config['model_size']='small';config['update']['preparation_workers']=0
        model=UtilityNet(utility_config(ModelConfig.preset('small'),reward_spec(config)))
        source=Path(self.tmp.name)/'source';target=Path(self.tmp.name)/'destination'
        p=TrainingPhases(model,config,root=ROOT,run=source)
        parameter=next(model.parameters());parameter.grad=torch.ones_like(parameter)
        p.optimizer.step();p.optimizer.zero_grad(set_to_none=True)
        prepare(model,p,1,companion=True)
        p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,p.actor_ids,p.collection_source,model,p.split['selection'])
        p.state=gae.initial(config['advantage'],1);p.critic_state['generation']=1;p.beta=.75;p.trips=[True,False]
        raw=dict(model_config=model.config.to_dict(),training={KEY:p.payload()})
        before=copy.deepcopy((model.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
        projected=ppg_payload(raw,dict(origin_generation=1),True)
        q=TrainingPhases(model,config,raw=projected,generation=1,root=ROOT,run=target)
        self.assertEqual(q.buffer.receipts,[]);self.assertEqual(q.cache_origin,1)
        self.assertEqual((q.beta,q.trips),(.75,[True,False]))
        self.assertTrue(_same_payload(before,(model.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
        prepare(model,q,2,companion=True)
        q.buffer.stage(q.critic_dataset,q.train_ids,q.heldout_ids,q.actor_ids,q.collection_source,model,q.split['selection'])
        q.state=gae.initial(config['advantage'],2);q.critic_state['generation']=2
        saved=dict(training={KEY:q.payload()});write_json(target/'state.json',dict(iteration=2));q.buffer.committed(2)
        resumed=TrainingPhases(model,config,raw=saved,generation=2,root=ROOT,run=target)
        self.assertEqual([r['generation'] for r in resumed.buffer.receipts],[2]);self.assertEqual(resumed.cache_origin,1)
        self.assertTrue(_same_payload(q.optimizer.state_dict(),resumed.optimizer.state_dict()))


if __name__=='__main__':unittest.main()
