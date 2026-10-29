"""Fast development tests, using fake native/service endpoints; no qualification."""
from pathlib import Path
import queue
import json
import tempfile
from threading import Event, RLock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json
from brass_ai.learning.features import FEATURE_VERSION
from brass_ai.learning.seat_descriptors import unknown_numpy
from brass_ai.rust_backend.lab_contract import (ARCHIVE, SOURCE, admit, admit_search_request,
    extract_new, native_fingerprint, sha, verify_inputs)
from brass_ai.rust_backend.lab_inference import NativeInputClient, Scope
from brass_ai.rust_backend.lab_snapshot import capture, pending_paths, tree_digest, verify_unchanged


PROFILE = read_json(SOURCE_ROOT / 'configs/search/lab-search-v1.json')


class ContractTests(unittest.TestCase):
    def unpack_inputs(self, temp):
        inputs = Path(temp) / 'inputs'
        extract_new(SOURCE_ROOT / 'distribution/rust-lab-b01/b01-search-v1-inputs.zip', inputs)
        return inputs

    def test_search_v1_inputs_bind_current_installed_evidence_and_source(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            inputs = self.unpack_inputs(temp)
            self.assertEqual(verify_inputs(inputs, PROFILE)['native_source_sha256'], SOURCE)
            self.assertEqual(sha(inputs / 'source.zip'), ARCHIVE)
            extract_new(inputs / 'source.zip', Path(temp) / 'checkout')
            self.assertEqual(native_fingerprint(Path(temp) / 'checkout/native'), SOURCE)

    def test_rehashed_input_manifest_cannot_relabel_old_methods_or_broken_receipts(self):
        mutations = [('deployment.json', 'method', 'search_v2_focus_only'),
                     ('local-qualification.json', 'build_manifest_sha256', 'changed'),
                     ('policy-k8-t15.json', 'method', 'search_v2'),
                     ('candidate.json', 'native_source_sha256', 'old-core')]
        for name, key, value in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
                inputs = self.unpack_inputs(temp)
                record = read_json(inputs / name); record[key] = value
                (inputs / name).write_text(json.dumps(record), encoding='utf-8')
                manifest = read_json(inputs / 'inputs.json'); manifest['files'][name] = sha(inputs / name)
                (inputs / 'inputs.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify_inputs(inputs, PROFILE)

    def test_omitted_pin_or_old_input_contract_cannot_bypass_verification(self):
        for old_contract in (False, True):
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
                inputs = self.unpack_inputs(temp)
                manifest = read_json(inputs / 'inputs.json')
                if old_contract:
                    manifest['contract'] = 'brass-lab-b01-inputs/1.0.0'
                else:
                    manifest['files'].pop('installed-smoke.json')
                (inputs / 'inputs.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify_inputs(inputs, PROFILE)

    def test_only_exact_explicit_search_v1_profile(self):
        admit(PROFILE)
        for key, value in [('method', 'search_v2'), ('method', 'search_v2_focus_only'),
                           ('method', 'search_v2_no_switch'), ('method', None),
                           ('v205_enabled', True), ('v205_enabled', 0), ('focus_policy', 'adaptive-focus/2.0.0'),
                           ('session_contract', 'unknown'), ('native_source_sha256', 'old')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                admit(dict(PROFILE, **{key: value}))
        for key in PROFILE:
            incomplete = dict(PROFILE); incomplete.pop(key)
            with self.assertRaises(ValueError):
                admit(incomplete)
        with self.assertRaises(ValueError):
            admit(dict(PROFILE, enabled=True))
        with self.assertRaises(ValueError):
            admit(read_json(SOURCE_ROOT / 'configs/search/lab-cache-only.json'))

    def test_method_overrides_cannot_reach_future_dispatch(self):
        admit_search_request(PROFILE, {'method': 'search_v1'})
        for value in ({}, {'method': 'search_v2'}, {'method': 'search_v2_focus_only'},
                      {'method': 'search_v2_no_switch'}, {'method': 'search_v1', 'reuse': False},
                      {'method': 'search_v1', 'focus_policy': None},
                      {'method': 'search_v1', 'composite_contract': None}):
            with self.assertRaises(ValueError):
                admit_search_request(PROFILE, value)

    def test_extract_rejects_paths_duplicates_and_symlinks(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            temp = Path(temp)
            for i, name in enumerate(('../outside', '/absolute', 'C:/file', 'a\\b')):
                archive = temp / f'{i}.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    z.writestr(name, 'bad')
                if '\\' in name:
                    archive.write_bytes(archive.read_bytes().replace(b'a/b', b'a\\b'))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    extract_new(archive, temp / f'out{i}')
                self.assertFalse((temp / f'out{i}').exists())
            archive = temp / 'link.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                link = zipfile.ZipInfo('link'); link.create_system = 3; link.external_attr = 0o120777 << 16
                z.writestr(link, '../outside')
            with self.assertRaises(ValueError):
                extract_new(archive, temp / 'links')

    def test_fingerprint_ignores_build_output_but_binds_config_and_source(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            temp = Path(temp); (temp / 'lib.rs').write_text('one')
            before = native_fingerprint(temp)
            (temp / 'target').mkdir(); (temp / 'target/generated.rs').write_text('noise')
            self.assertEqual(before, native_fingerprint(temp))
            (temp / 'Cargo.toml').write_text('[workspace]')
            self.assertNotEqual(before, native_fingerprint(temp))

    def test_pending_collection_or_partial_checkpoint_blocks_entry(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            temp = Path(temp)
            self.assertEqual(pending_paths(temp, 12), [])
            (temp / 'generation-0013.pt.tmp').write_text('partial')
            (temp / 'iteration-0012').mkdir(); (temp / 'iteration-0012/jobs.json').write_text('{}')
            self.assertEqual(len(pending_paths(temp, 12)), 2)

    def test_optimizer_rng_and_tensor_shape_hashes_are_distinct(self):
        import torch
        a = torch.tensor([1, 2], dtype=torch.int64)
        self.assertEqual(tree_digest({'x': a, 1: [2]}), tree_digest({1: [2], 'x': a.clone()}))
        self.assertNotEqual(tree_digest(a), tree_digest(a.reshape(1, 2)))
        self.assertNotEqual(tree_digest([a]), tree_digest((a,)))
        self.assertNotEqual(tree_digest({'step': 2}), tree_digest({'step': 3}))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = SimpleNamespace(lock=RLock(), pending=None, slot=lambda: None,
            requests=queue.Queue(), connection=Mock(), timeout=.01, lane=3)
        self.native = SimpleNamespace(state=np.zeros(3128, np.float32), actions=np.zeros((2, 327), np.float32),
            stable_to_relative=np.arange(4), choice_ids=('a', 'b'), feature_version=FEATURE_VERSION,
            descriptors=unknown_numpy(), revision=17, observer=0)
        self.encoder = Mock(); self.encoder.encode.return_value = self.native
        self.client = NativeInputClient(self.endpoint, self.encoder, PROFILE, Scope(0, 'turn', 0))
        self.view = dict(context=dict(revision=17), seat=0)
        self.choices = [dict(id='a'), dict(id='b')]
        self.context = dict(model_id='m', reward_id='reward', profile_id='p', style=[0.] * 4,
            intent=[0.] * 8, remaining=[.075] * 4, max_cost=.075, stable_descriptors=None)
        self.rows = [(self.view, self.choices, self.context)]
        self.deadline = time.monotonic() + 4

    def submit(self, **kwargs):
        return self.client.submit(self.rows, deadline=self.deadline, **kwargs)

    def reply(self, ticket):
        r = ticket.rows[0]; b = r['binding']
        value = dict(token=ticket.token, responses=[dict(protocol='four-seat-inference/1.0.0',
            request_id=r['request_id'], model_id=b['model'], revision=b['revision'], observer=b['observer'],
            input_digest=r['digest'], status='ok', choice_ids=['a', 'b'], probabilities=[.3, .7],
            win_share=[.25] * 4, placement=[2.5] * 4, vp=[3.] * 4, utility=[.4] * 4, reward_id='reward')])
        self.endpoint.connection.poll.return_value = True
        self.endpoint.connection.recv.return_value = value
        return value

    def test_roundtrip_and_one_unreplenished_deadline(self):
        ticket = self.submit(); self.reply(ticket)
        with self.assertRaises(ValueError):
            self.client.advance(Scope(0, 'turn', 1))
        result = self.client.poll(ticket, wait=True)[0]
        self.assertEqual(result['native_binding']['scope']['epoch'], 0)
        self.client.advance(Scope(0, 'turn', 1))
        with self.assertRaises(ValueError):
            self.client.submit(self.rows, deadline=self.deadline + 1)
        next_ticket = self.submit(); self.reply(next_ticket)
        self.assertEqual(self.client.poll(next_ticket)[0]['native_binding']['scope']['epoch'], 1)
        with self.assertRaises(ValueError):
            self.client.poll(ticket)

    def test_stopped_before_encoding(self):
        cancelled = Event(); cancelled.set()
        ticket = self.submit(cancelled=cancelled); self.reply(ticket)
        result = self.client.poll(ticket)[0]
        self.assertEqual(result['status'], 'cancelled'); self.assertNotIn('probabilities', result)
        self.encoder.encode.assert_not_called()

    def test_expired_before_encoding(self):
        ticket = self.client.submit(self.rows, deadline=time.monotonic() - 1); self.reply(ticket)
        self.assertEqual(self.client.poll(ticket)[0]['status'], 'expired')
        self.encoder.encode.assert_not_called()

    def test_cancel_after_dispatch_drains_slot_before_epoch(self):
        cancel = Event(); ticket = self.submit(cancelled=cancel); cancel.set(); self.reply(ticket)
        with self.assertRaises(ValueError):
            self.submit()
        self.assertEqual(self.client.poll(ticket)[0]['status'], 'cancelled')
        self.assertIsNone(self.endpoint.pending)
        self.client.advance(Scope(0, 'turn', 1))

    def test_misrouted_reply_or_wrong_legal_order_retires_lane(self):
        for key, value in [('model_id', 'other'), ('observer', 1), ('revision', 18), ('input_digest', 'old'),
                           ('request_id', 'other'), ('choice_ids', ['b', 'a']), ('reward_id', 'wrong'),
                           ('utility', [float('nan')] * 4)]:
            self.setUp(); ticket = self.submit(); reply = self.reply(ticket); reply['responses'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.client.poll(ticket)
            self.assertTrue(self.client.retired)

    def test_device_failure_and_timeout_never_reuse_lane(self):
        ticket = self.submit(); reply = self.reply(ticket); reply['error'] = 'CUDA failure'
        with self.assertRaises(ValueError):
            self.client.poll(ticket)
        self.assertTrue(self.client.retired)
        self.setUp(); ticket = self.submit(); self.endpoint.connection.poll.return_value = False
        self.assertIsNone(self.client.poll(ticket))
        with self.assertRaises(TimeoutError):
            self.client.poll(ticket, wait=True)
        with self.assertRaises(ValueError):
            self.client.advance(Scope(0, 'turn', 1))

    def test_model_or_context_change_inside_request_is_rejected(self):
        for key, value in [('model_id', 'new-weights'), ('profile_id', 'new-profile'), ('reward_id', 'new-reward')]:
            self.setUp(); ticket = self.submit(); self.reply(ticket); self.client.poll(ticket)
            self.client.advance(Scope(0, 'turn', 1)); self.context[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.submit()

    def test_critic_descriptors_cannot_enter_search(self):
        self.context['stable_descriptors'] = unknown_numpy().tolist()
        with self.assertRaises(ValueError):
            self.submit()
        self.encoder.encode.assert_not_called()

    def test_native_input_is_detached_before_encoder_or_caller_can_mutate_it(self):
        def encode(view, choices, context):
            self.view['context']['revision'] = 99
            self.choices.reverse()
            self.context['remaining'][0] = 0
            self.assertEqual(view['context']['revision'], 17)
            self.assertEqual([c['id'] for c in choices], ['a', 'b'])
            self.assertEqual(context['remaining'][0], .075)
            return self.native
        self.encoder.encode.side_effect = encode
        ticket = self.submit(); self.reply(ticket)
        self.assertEqual(self.client.poll(ticket)[0]['revision'], 17)

    def test_slot_not_reused_on_partial_or_foreign_batch(self):
        for kind in ('token', 'partial'):
            self.setUp(); ticket = self.submit(); reply = self.reply(ticket)
            if kind == 'token': reply['token'] = 'old'
            else: reply['responses'] = []
            with self.assertRaises(ValueError):
                self.client.poll(ticket)
            self.assertTrue(self.client.retired)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        import torch
        from brass_ai.recovery.config import EXECUTION
        self.temp = tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.run = self.root / 'run'; self.run.mkdir()
        self.host = self.root / 'runtime'; self.host.mkdir(); (self.host / 'runtime.json').write_text('{}')
        self.checkpoint = self.run / 'generation-0003.pt'; self.checkpoint.write_bytes(b'mock checkpoint')
        self.signature = dict(config=dict(execution=dict(EXECUTION), host='runtime', teacher_enabled=False),
                              implementation='frozen-source', compatibility={})
        entry = dict(path='run/generation-0003.pt', generation=3, model_id='frozen', sha256=sha(self.checkpoint))
        self.state = dict(iteration=3, signature=self.signature, history=[entry], population=[], games_done=12)
        (self.run / 'state.json').write_text(json.dumps(self.state), encoding='utf-8')
        self.raw = dict(step=3, model_id='frozen', optimizer={'moment': torch.ones(2)}, state_dict={'w': torch.ones(2)},
                        rng=torch.tensor([1, 2], dtype=torch.uint8), cuda_rng=[],
                        training=dict(signature=digest(self.signature), fresh_run='run', ppg={'beta': 1.5}))
        self.loader = patch('brass_ai.learning.checkpoint.load_checkpoint', return_value=(object(), self.raw))
        self.loader.start(); self.addCleanup(self.loader.stop)

    def test_committed_snapshot_pins_recipe_and_all_checkpoint_state_without_updates(self):
        before = sha(self.run / 'state.json')
        receipt, _, _ = capture(self.root, self.run, devices=[])
        self.assertEqual(receipt['state'], self.state)
        self.assertFalse(receipt['teacher_admitted'])
        self.assertEqual(set(receipt['checkpoint_parts']), {'state_dict', 'optimizer', 'rng', 'cuda_rng', 'training'})
        self.assertEqual(receipt['checkpoint_parts']['training'], tree_digest(self.raw['training']))
        self.assertEqual(sha(self.run / 'state.json'), before)
        verify_unchanged(receipt)
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_unchanged(receipt)

    def test_pending_work_rejected_before_checkpoint_load(self):
        (self.run / 'iteration-0003').mkdir(); (self.run / 'iteration-0003/jobs.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'pending work'):
            capture(self.root, self.run, devices=[])

    def test_bad_generation_signature_or_missing_optimizer_never_admitted(self):
        for key, value in [('step', 4), ('model_id', 'different'), ('optimizer', None)]:
            old = self.raw[key]; self.raw[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                capture(self.root, self.run, devices=[])
            self.raw[key] = old
        self.raw['training']['signature'] = 'unreviewed'
        with self.assertRaisesRegex(ValueError, 'not bound'):
            capture(self.root, self.run, devices=[])


if __name__ == '__main__':
    unittest.main()
