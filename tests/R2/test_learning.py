import copy
import contextlib
import gzip
import hashlib
import itertools
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from jsonschema import ValidationError

from brass_ai.learning.batching import Evaluator, Request, accept_response, collate, model_id
from brass_ai.learning.checkpoint import load_checkpoint, resume_optimizer, save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, assert_compatible, compatibility, decode_json, load_config, project_path, read_json
from brass_ai.learning.export import export_onnx, load_export, INPUTS, OUTPUTS
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import (examples, read_shards, split_for, terminal_targets,
    validate_game, validate_teacher, write_shard)
from brass_ai.learning.training import losses, on_policy_update, prepared, supervised_updates
from brass_ai.learning.runner import ModelPolicy
from brass_ai.learning.cli import main as cli_main, parser as cli_parser
from fixtures import game, position, result, rotate


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(902)
        self.encoder = Encoder()
        self.view, self.choices = position()
        self.encoded = self.encoder.encode(self.view, self.choices)
        self.model = FourSeatNet(ModelConfig.preset('small')).eval()

    def test_size_presets_and_fresh_initialization(self):
        # Features 2 requires fresh residual weights with the explicit 3M cap.
        for size, expected in [('small', 957709), ('medium', 2580429)]:
            model = FourSeatNet(ModelConfig.preset(size))
            self.assertEqual(model.config.architecture, 'brass-four-seat-mlp-resnet/2.0.0')
            self.assertEqual(model.parameter_count, expected)
            self.assertLessEqual(model.parameter_count, 3000000)
        with self.assertRaisesRegex(ValueError, 'version-2 residual model preset'):
            ModelConfig.preset('large')
        second = FourSeatNet(ModelConfig.preset('small'))
        self.assertNotEqual(model_id(self.model), model_id(second))

    def test_shapes_four_stable_outputs_and_single_choice(self):
        one = self.encoder.encode(self.view, self.choices[:1])
        batch = collate([self.encoded, one])
        output = self.model(*batch)
        self.assertEqual(output[0].shape, (2, 3))
        for head in output[1:]:
            self.assertEqual(head.shape, (2, 4))
        np.testing.assert_allclose(output[1].sum(-1).detach(), [1, 1], atol=1e-6)
        probabilities = torch.softmax(output[0], -1)
        np.testing.assert_array_equal(probabilities[1].detach(), [1, 0, 0])

    def test_batch_and_candidate_permutation_equivalence(self):
        shuffled = self.encoder.encode(self.view, self.choices[::-1])
        with torch.inference_mode():
            batched = self.model(*collate([self.encoded, shuffled]))
            single = self.model(*collate([self.encoded]))
        np.testing.assert_allclose(batched[0][1].numpy()[::-1], single[0][0], atol=2e-6)
        for head,(a,b) in enumerate(zip(batched[1:],single[1:])):
            # Different FP32 matrix shapes differ by a few ulps. The wider
            # state encoder can produce up to 0.00005 VP of rounding noise.
            np.testing.assert_allclose(a.numpy(),b.numpy().repeat(2,axis=0),atol=5e-5 if head==2 else 1e-5)

    def test_clockwise_rotation_encoding_and_outputs(self):
        with torch.inference_mode():
            baseline = self.model(*collate([self.encoded]))
        for shift in range(4):
            encoded = self.encoder.encode(rotate(self.view, shift), self.choices)
            np.testing.assert_array_equal(self.encoded.state, encoded.state)
            with torch.inference_mode():
                actual = self.model(*collate([encoded]))
            for a, b in zip(actual[1:], baseline[1:]):
                np.testing.assert_allclose(a.numpy(), np.roll(b.numpy(), shift, axis=1), atol=1e-6)

    def test_all_turn_orders_and_observers_preserve_stable_mapping(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()
            self.model.outcomes[-1].bias[:4].copy_(torch.tensor([-2., -1., 0., 1.]))
        base = torch.softmax(torch.tensor([-2., -1., 0., 1.]), -1).numpy()
        for observer, order in itertools.product(range(4), itertools.permutations(range(4))):
            view, choices = position(observer)
            view['order'] = list(order)
            with torch.inference_mode():
                win = self.model(*collate([self.encoder.encode(view, choices)]))[1][0].numpy()
            np.testing.assert_allclose(win, [base[(s-observer)%4] for s in range(4)], atol=1e-6)

    def test_private_fields_and_inactive_observer_rejected(self):
        for path, key, value in [((), 'seed', 2), (('players', 1), 'hand', ['location:Derby']),
                                 (('context',), 'game_id', 'a'*32), ((), 'checkpoint', {})]:
            view = copy.deepcopy(self.view)
            node = view
            for part in path:
                node = node[part]
            node[key] = value
            with self.assertRaises((ValidationError, ValueError)):
                self.encoder.encode(view, self.choices)
        self.view['context']['actor'] = 1
        with self.assertRaises(ValueError):
            self.encoder.encode(self.view, self.choices)

    def test_transport_revision_and_choice_ids_are_not_features(self):
        view, choices = copy.deepcopy(self.view), copy.deepcopy(self.choices)
        view['context']['revision'] += 800
        for i, choice in enumerate(choices):
            choice['id'] = f'unrelated-id-{i}'
        altered = self.encoder.encode(view, choices)
        np.testing.assert_array_equal(altered.state, self.encoded.state)
        np.testing.assert_array_equal(altered.actions, self.encoded.actions)

    def test_scout_multiplicity_and_all_choice_kinds(self):
        choice = {'id':'scout', 'kind':'scout', 'card':'industry:CoalMine',
                  'cards':['industry:CoalMine', 'location:Birmingham']}
        encoded = self.encoder.encode(self.view, [choice])
        offset = self.encoder.base_action_dim - len(self.encoder.cards)
        self.assertAlmostEqual(encoded.actions[0, offset + self.encoder.cards.index('industry:CoalMine')], 2/3)
        kinds = [{'id':k, 'kind':k} for k in ('finish','bonus_skip')]
        kinds += [dict(id='x',kind='bonus_develop',industry='Pottery')]
        kinds += [dict(id='x',kind=k,target=self.encoder.schema['slots'][0]) for k in ('beer','coal','iron','liquidate')]
        for choice in kinds:
            # Categorical vocabulary coverage does not make these fabricated
            # subchoices legal in this opening. FE3 tests native consequences.
            self.assertTrue(np.isfinite(self.encoder.choice_features(choice)).all())

    def test_terminal_ties_ranks_and_no_sign_flip(self):
        targets = terminal_targets(result())
        self.assertEqual(targets['win_share'], [0.5,0.5,0,0])
        self.assertEqual(targets['placement'], [1,1,3,4])
        different = result([(50,2,3),(50,3,0),(50,3,1),(49,30,99)])
        self.assertEqual(terminal_targets(different)['win_share'], [0,0,1,0])
        rows = examples([game()])
        self.assertEqual(len({r[2] for r in rows}),4)
        self.assertTrue(all(r[1] == targets for r in rows))
        broken = result()
        broken['seats'][0]['rank'] = 2
        with self.assertRaises(ValueError): terminal_targets(broken)

    def test_teacher_masks_unexamined_choices_and_keeps_losers(self):
        record = game()['records'][0]
        record['teacher'] = dict(source='fresh-finalist/1.0.0', probabilities=[1.,0.,0.],
            coverage=[True,True,False], confidence=1.,samples=16,horizon='next-root-turn')
        validate_teacher(record['teacher'],3)
        logits = torch.zeros((1,3), requires_grad=True)
        heads = (logits, torch.full((1,4),0.25), torch.full((1,4),2.5), torch.zeros((1,4)))
        terms = losses(heads,[record],[terminal_targets(result())])
        terms['teacher'].sum().backward()
        self.assertEqual(logits.grad[0,2].item(),0)
        self.assertLess(logits.grad[0,0].item(),0)
        self.assertEqual(len(prepared([game()])),4)
        record['teacher']['source'] = 'visit-counts'
        with self.assertRaises(ValueError): validate_teacher(record['teacher'],3)

    def test_replay_roundtrip_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'shard.gz'
            write_shard(path,[game()],{'source':'authored-fixture'})
            self.assertEqual(read_shards([path]),[game()])
            with self.assertRaises(ValueError): read_shards([path,path])
            with self.assertRaises(ValueError): write_shard(path,[game()],{})
        for mutate in (lambda g:g.update(players=2), lambda g:g.update(result=None),
                       lambda g:g['records'][0].update(chosen=999),
                       lambda g:g['records'][0]['view'].update(deck_order=[]),
                       lambda g:g['records'][0]['behavior'].update(probability=float('nan')),
                       lambda g:g['records'][0]['behavior'].update(policy_id='changed')):
            value=game(); mutate(value)
            with self.assertRaises((ValueError,ValidationError)): validate_game(value)

    def test_compatibility_rejects_scalar_two_player_and_rules_drift(self):
        for key,value in [('players',2),('heads',['scalar']),('rules','stock'),
                          ('features_sha256','0'*64),('wire_sha256','0'*64)]:
            changed=compatibility(); changed[key]=value
            with self.assertRaises(ValueError): assert_compatible(changed)
        with self.assertRaises(ValueError):
            ModelConfig.from_dict({**self.model.config.to_dict(),'players':3})

    def test_held_out_opponent_replay_is_rejected_before_training(self):
        sample = game()
        sample['setup_group'] = next(str(i) for i in range(100) if split_for(str(i)) == 'train')
        sample['split'] = 'train'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'held-out.gz'
            write_shard(path, [sample], {'producer': 'R3-bank/1.0.0', 'purpose': 'evaluation'})
            self.assertEqual(1, len(read_shards([path])))  # Evaluation/export remains available.
            with self.assertRaisesRegex(ValueError, 'Evaluation-only'):
                read_shards([path], partition='train')
            with self.assertRaisesRegex(ValueError, 'Evaluation-only'):
                cli_main(['--project-root', temp, '--device', 'cpu', '--size', 'small',
                          'train', '--shards', 'held-out.gz', '--steps', '1',
                          '--checkpoint', 'should-not-exist.pt'])
            self.assertFalse((Path(temp) / 'should-not-exist.pt').exists())
            allowed = Path(temp) / 'training.gz'
            write_shard(allowed, [sample], {'producer': 'R3-bank/1.0.0', 'purpose': 'training'})
            self.assertEqual(1, len(read_shards([allowed], partition='train')))

    def test_lab_paths_and_game_level_partition(self):
        self.assertEqual(load_config()['project_root'],'/data/not_backed_up/yxu209/ai-training/')
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(project_path(temp,'artifacts/r2').is_relative_to(Path(temp)))
            with self.assertRaises(ValueError): project_path(temp,'../outside')
            config=load_config(); config['players']=3
            path=Path(temp)/'config.json'; path.write_text(json.dumps(config))
            with self.assertRaises(ValueError): load_config(path)
        self.assertEqual(split_for('setup-A'),split_for('setup-A'))
        self.assertEqual({split_for(str(i)) for i in range(100)}, {'train','validation','test'})

    def test_reject_ambiguous_json(self):
        for text in ('{"players":4,"players":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError): decode_json(text)

    def test_r2_reference_provenance_and_license(self):
        record=read_json(SOURCE_ROOT/'configs/references/r2-components.json')
        lock=read_json(SOURCE_ROOT/'configs/references/upstream.lock.json')
        upstream=next(r for r in lock['references'] if r['id']=='daiyu')
        self.assertEqual(record['commit'],upstream['commit'])
        self.assertEqual(hashlib.sha256((SOURCE_ROOT/record['license_text']).read_bytes()).hexdigest(),record['license_sha256'])
        self.assertTrue((SOURCE_ROOT/record['notice']).is_file())
        for component in record['components']:
            self.assertIn(component['source'],[f['path'] for f in upstream['files']])
            self.assertTrue((SOURCE_ROOT/component['destination']).is_file())

    def test_batch_routing_deadline_cancellation_and_frozen_model(self):
        evaluator=Evaluator(self.model,batch_size=2)
        deadline=time.monotonic()+60
        request=Request('live',evaluator.model_id,self.view,self.choices,deadline)
        event=threading.Event(); event.set()
        expired=Request('expired',evaluator.model_id,self.view,self.choices,time.monotonic()-1)
        cancelled=Request('cancelled',evaluator.model_id,self.view,self.choices,deadline,cancelled=event)
        responses=evaluator.predict([request,expired,cancelled])
        self.assertEqual([r['status'] for r in responses],['ok','expired','cancelled'])
        accept_response(request,responses[0],self.view['context']['revision'],evaluator.model_id)
        with self.assertRaises(ValueError): accept_response(request,responses[0],99,evaluator.model_id)
        with self.assertRaises(ValueError): evaluator.predict([request,request])
        altered=copy.deepcopy(responses[0]); altered['input_digest']='0'*64
        with self.assertRaises(ValueError): accept_response(request,altered,self.view['context']['revision'],evaluator.model_id)
        before=evaluator.model_id
        with torch.no_grad(): next(self.model.parameters()).add_(1)
        self.assertEqual(evaluator.model_id,before)
        self.assertNotEqual(model_id(self.model),before)

    def test_request_snapshots_and_rejects_later_mutation(self):
        evaluator=Evaluator(self.model)
        request=Request('snapshot',evaluator.model_id,self.view,self.choices,time.monotonic()+60)
        self.view['players'][0]['cash']=999
        self.assertNotEqual(request.view['players'][0]['cash'],999)
        request.view['players'][0]['cash']=777
        with self.assertRaises(ValueError): evaluator.predict([request])

    def test_expired_or_cancelled_native_results_are_discarded(self):
        evaluator=Evaluator(self.model)
        event=threading.Event()
        request=Request('cancel-late',evaluator.model_id,self.view,self.choices,time.monotonic()+60,cancelled=event)
        hook=evaluator.model.register_forward_hook(lambda *_: event.set())
        response=evaluator.predict([request])[0]
        self.assertEqual(response['status'],'cancelled')
        self.assertNotIn('probabilities',response)
        hook.remove()
        now=[10.0]
        hook=evaluator.model.register_forward_hook(lambda *_: now.__setitem__(0,12.0))
        with patch('brass_ai.learning.batching.time.monotonic',side_effect=lambda:now[0]):
            response=evaluator.predict([Request('late',evaluator.model_id,self.view,self.choices,11.0)])[0]
        self.assertEqual(response['status'],'expired')
        self.assertNotIn('win_share',response)
        hook.remove()

    def test_bounded_batch_splitting_and_model_policy_interface(self):
        import random
        evaluator=Evaluator(self.model,batch_size=2,max_candidate_cells=5)
        requests=[Request(str(i),evaluator.model_id,self.view,self.choices,time.monotonic()+60) for i in range(3)]
        responses=evaluator.predict(requests)
        self.assertEqual([r['request_id'] for r in responses],['0','1','2'])
        self.assertTrue(all(r['status']=='ok' for r in responses))
        policy=ModelPolicy(self.model,batch_size=2)
        selected,probability,teacher=policy.select_batch([(self.view,self.choices,random.Random(9))])[0]
        self.assertIn(selected,range(3))
        self.assertGreater(probability,0)
        self.assertIsNone(teacher)

    def test_finite_training_checkpoint_resume_is_exact(self):
        data=prepared([game()])
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=0.0003)
        metrics=supervised_updates(self.model,optimizer,data,2,4)
        self.assertTrue(all(np.isfinite(list(x.values())).all() for x in metrics))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.pt'
            identity=save_checkpoint(path,self.model,optimizer,step=2)
            supervised_updates(self.model,optimizer,data,1,4)
            expected=model_id(self.model)
            resumed,raw=load_checkpoint(path)
            self.assertEqual(model_id(resumed),identity)
            opt=torch.optim.AdamW(resumed.parameters(),lr=0.0003)
            resume_optimizer(raw,opt)
            supervised_updates(resumed,opt,data,1,4)
            self.assertEqual(model_id(resumed),expected)
            raw['state_dict']['outcomes.2.bias']=torch.zeros(1)
            torch.save(raw,path)
            with self.assertRaises((ValueError,RuntimeError)):load_checkpoint(path)

    def test_onpolicy_rejects_random_or_stale_trajectories(self):
        optimizer=torch.optim.AdamW(self.model.parameters())
        before=model_id(self.model)
        with self.assertRaises(ValueError): on_policy_update(self.model,optimizer,[game()])
        self.assertEqual(model_id(self.model),before)

    def test_onnx_dynamic_shapes_and_checked_manifest(self):
        one=self.encoder.encode(self.view,self.choices[:1])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.onnx'
            report=export_onnx(self.model,path,[[self.encoded],[one,self.encoded],[one]])
            self.assertLess(report['parity_max_absolute_error'],2e-4)
            session,manifest=load_export(path)
            output=session.run(OUTPUTS,{n:t.numpy() for n,t in zip(INPUTS,collate([one]))})
            self.assertEqual(output[1].shape,(1,4))
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaises(ValueError):load_export(path)

    def test_cli_fresh_train_resume_export_and_four_seat_arguments(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            base=['--project-root',temp,'--device','cpu','--size','small']
            first=cli_main(base+['init','--checkpoint','artifacts/fresh.pt'])
            sample=game()
            sample['setup_group']=next(str(i) for i in range(100) if split_for(str(i))=='train')
            sample['split']='train'
            sample['records'][-1]['choices']=sample['records'][-1]['choices'][:1]
            sample['records'][-1]['behavior']['probability']=1.0
            write_shard(Path(temp)/'artifacts/sample.gz',[sample],{'source':'authored-cli-fixture'})
            args=base+['train','--shards','artifacts/sample.gz','--steps','1']
            trained=cli_main(args+['--from-checkpoint','artifacts/fresh.pt','--checkpoint','artifacts/trained.pt'])
            resumed=cli_main(args+['--resume','--from-checkpoint','artifacts/trained.pt','--checkpoint','artifacts/resumed.pt'])
            self.assertEqual(resumed['step'],2)
            self.assertNotEqual(trained['model_id'],first['model_id'])
            exported=cli_main(base+['export','--checkpoint','artifacts/resumed.pt',
                '--shards','artifacts/sample.gz','--output','artifacts/model.onnx'])
            self.assertEqual(exported['model_id'],resumed['model_id'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli_parser().parse_args(['evaluate','--opponents','random','random','random'])
