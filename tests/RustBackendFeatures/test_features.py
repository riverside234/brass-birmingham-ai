"""Real two-owner diagnostic transport tests; corpus/NN parity is separately sealed."""
import copy
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from brass_ai.native.protocol import NativeWorker, NativeError, WorkerFailure
from brass_ai.native.features import NativeFeatureEncoder
from brass_ai.native.simulation import NativeSpielGame

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(os.environ.get('BRASS_NATIVE_WORKER', ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-worker.exe'))


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(ROOT/'artifacts/rust-backend/20261025-l06-capture/roots.jsonl.gz', 'rt', encoding='utf-8') as f:
            row = json.loads(next(f))
        cls.state = row['state']; cls.view = row['views'][row['state']['context']['actor']]
        with gzip.open(ROOT/'artifacts/rust-backend/20261025-l05-capture-v2/roots.jsonl.gz', 'rt', encoding='utf-8') as f:
            cls.choices = json.loads(next(f))['expected']['choices']
        cls.ctx = dict(model_id='test', reward_id='test-reward', profile_id='test-profile', style=[0.]*4,
                       intent=[0.]*8, remaining=[0., .01, .05, .075], max_cost=.075, stable_descriptors=None)

    def worker(self):
        w = NativeWorker(WORKER, timeout=60); self.addCleanup(w.close); return w

    def test_both_owners_complete_pagination_and_native_handles(self):
        w = self.worker()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda s: NativeFeatureEncoder(w, s).encode(self.view, self.choices, self.ctx), (0, 1)))
        self.assertNotEqual(results[0].encoding_id, results[1].encoding_id)
        self.assertGreater(len(results[0].choice_ids), 64)
        np.testing.assert_array_equal(results[0].actions, results[1].actions)
        for s in (0, 1):
            with NativeSpielGame(worker=w, session=s) as game:
                state = game.import_state(self.state)
                encoded = NativeFeatureEncoder(w, s).from_state(state, self.ctx)
                np.testing.assert_array_equal(encoded.state, results[s].state)
                np.testing.assert_array_equal(encoded.actions, results[s].actions)
                with self.assertRaises(ValueError):
                    NativeFeatureEncoder(w, 1-s).from_state(state, self.ctx)

    def test_changed_context_does_not_reuse_an_old_result(self):
        w = self.worker(); e = NativeFeatureEncoder(w)
        a = e.encode(self.view, self.choices, self.ctx)
        ctx = dict(self.ctx, profile_id='new', model_id='frozen-opponent', reward_id='different-reward',
                   style=[1., -.5, .25, 0.], remaining=[.075, 0., 0., 0.])
        b = e.encode(self.view, self.choices, ctx)
        self.assertGreater(b.encoding_id, a.encoding_id)
        np.testing.assert_array_equal(b.state[:-12], a.state[:-12])
        np.testing.assert_array_equal(b.state[-12:-8], ctx['style'])
        np.testing.assert_array_equal(b.remaining, np.asarray(ctx['remaining'], dtype=np.float32))
        self.assertFalse(np.array_equal(a.state, b.state))

    def test_pages_reject_old_foreign_or_failed_encodings(self):
        w = self.worker()
        def prepare(**kw):
            return w.call('features', session=0, command=dict(kind='encode', view=self.view, choices=self.choices,
                          context=dict(self.ctx, **kw)))['data']['encoding_id']
        old = prepare(); new = prepare(profile_id='new')
        for owner, sequence in [(0, old), (1, new)]:
            with self.assertRaises(NativeError):
                w.call('features', session=owner, command=dict(kind='page', encoding_id=sequence, start=0, limit=64))
        with self.assertRaises(NativeError): prepare(remaining=[.1]*4)
        with self.assertRaises(NativeError):
            w.call('features', session=0, command=dict(kind='page', encoding_id=new, start=0, limit=64))
        self.assertGreater(prepare(), new)

    def test_invalid_shape_unknown_metadata_and_nonacting_view(self):
        w = self.worker(); e = NativeFeatureEncoder(w)
        for field, value in [('style', [0.]*3), ('intent', [0.]*9), ('model_id', ''),
                             ('stable_descriptors', [[0.]*34]*4), ('remaining', [-1.]*4)]:
            with self.subTest(field=field), self.assertRaises(NativeError):
                e.encode(self.view, self.choices, dict(self.ctx, **{field:value}))
        invalid = copy.deepcopy(self.view); invalid['seat'] = (invalid['seat']+1) % 4
        with self.assertRaises(NativeError): e.encode(invalid, self.choices, self.ctx)
        invalid = copy.deepcopy(self.choices); invalid[0]['id'] += ':forged'
        with self.assertRaises(NativeError): e.encode(self.view, invalid, self.ctx)
        with self.assertRaises(NativeError): e.encode(self.view, self.choices, dict(self.ctx, folded=True))
        self.assertIsNotNone(e.encode(self.view, self.choices, self.ctx))

    def test_corrupt_candidate_order_retires_worker(self):
        w = self.worker(); original = w.call
        def corrupt(*a, **kw):
            result = original(*a, **kw)
            if result['status'] == 'features': result['data']['tensors']['choice_ids'].reverse()
            return result
        with patch.object(w, 'call', side_effect=corrupt), self.assertRaises(WorkerFailure):
            NativeFeatureEncoder(w).encode(self.view, self.choices, self.ctx)
        self.assertTrue(w._closed)

    def test_public_bounds_and_wide_money_match_oracle(self):
        from brass_ai.learning.features import Encoder
        w = self.worker(); e = NativeFeatureEncoder(w)
        for field, value in [('protocol', 'unknown'), ('own_hand', []), ('wild_industry_count', 5)]:
            invalid = copy.deepcopy(self.view); invalid[field] = value
            with self.subTest(field=field), self.assertRaises(NativeError):
                e.encode(invalid, self.choices, self.ctx)
        for field in ('cash', 'spent', 'debt', 'bonus_vp'):
            invalid = copy.deepcopy(self.view); invalid['players'][0][field] = -1
            with self.subTest(field=field), self.assertRaises(NativeError):
                e.encode(invalid, self.choices, self.ctx)
        view = copy.deepcopy(self.view)
        view['players'][view['seat']]['cash'] = 2**31-1
        view['players'][view['seat']]['spent'] = 2**31-1
        py = Encoder().encode(view, self.choices)
        rs = e.encode(view, self.choices, self.ctx)
        np.testing.assert_array_equal(py.state, rs.state)
        np.testing.assert_array_equal(py.actions, rs.actions)

    def test_malformed_numeric_response_retires_worker(self):
        w = self.worker(); original = w.call
        def corrupt(*a, **kw):
            result = original(*a, **kw)
            if result['status'] == 'features': result['data']['tensors']['actions'][0] = 'invalid'
            return result
        with patch.object(w, 'call', side_effect=corrupt), self.assertRaises(WorkerFailure):
            NativeFeatureEncoder(w).encode(self.view, self.choices, self.ctx)
        self.assertTrue(w._closed)


if __name__ == '__main__': unittest.main()
