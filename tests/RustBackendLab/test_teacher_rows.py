"""Teacher row preparation in helper processes, faster freezing, PPO transport and PPG prefetch."""
import copy
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch

from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT
from brass_ai.recovery import admitted_inputs as A
from brass_ai.recovery import ppo_preparation as P
from brass_ai.recovery.teacher_phase_config import TIMING
from brass_ai.rust_backend import lab_teacher_finalization as F
from brass_ai.rust_backend import lab_teacher_rows as W
from test_b025 import model_fixture, example
from test_b045 import value_fixture


def reference_freeze(value):
    """The former recursive implementation."""
    if type(value) in (A.FrozenDict, A.FrozenList, A.OwnedDict, A.OwnedList):return value
    if type(value) is dict:return A.FrozenDict((k, reference_freeze(v)) for k, v in value.items())
    if type(value) is list:return A.FrozenList(reference_freeze(v) for v in value)
    if type(value) in (str, int, float, bool, type(None)):return value
    raise TypeError('Only admitted JSON containers can be frozen')


def shape(value):
    """Container types and leaves, recursively, for an exact structural comparison."""
    if isinstance(value, dict):return (type(value).__name__, {k: shape(v) for k, v in value.items()})
    if isinstance(value, list):return (type(value).__name__, [shape(v) for v in value])
    return (type(value).__name__, value)


class Freeze(unittest.TestCase):
    def test_same_frozen_structure_as_the_recursive_version(self):
        model = model_fixture().eval(); _, job, _ = value_fixture(model)
        row, _ = example(model, job=job, round=job['reservations']['0']['canal_early'])
        nested = dict(a=[1, 2., 'x', None, True], b=[dict(c=[[1, 2], []], d={}), 3], e=[], f=dict(g=[dict()]))
        nested['h'] = A.FrozenList([1, 2]); nested['i'] = [A.FrozenDict(dict(j=1))]
        for value in (copy.deepcopy(row.row), nested, [], {}, [[1], [2.5, 'z']]):
            with self.subTest(kind=type(value).__name__):
                self.assertEqual(shape(A._freeze(value)), shape(reference_freeze(value)))
        frozen = A._freeze(nested)
        self.assertIs(frozen['h'], nested['h']); self.assertIs(frozen['i'][0], nested['i'][0])
        with self.assertRaises(TypeError):frozen['a'].append(1)

    def test_non_json_values_are_still_refused(self):
        for value in ([1, (2,)], dict(a=np.float64(1.)), [np.int64(1)], (1, 2), dict(a=dict(b={1, 2}))):
            with self.subTest(value=repr(value)), self.assertRaises(TypeError):A._freeze(value)


class RowPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.pool = W.RowPool(2)

    @classmethod
    def tearDownClass(cls):
        cls.pool.shutdown()

    def fixture(self, row_pool=None):
        model = model_fixture().eval(); _, job, _ = value_fixture(model)
        row, _ = example(model, job=job, round=job['reservations']['0']['canal_early'])
        history = [dict(generation=10, model_id=model_id(model))]
        verifier = F.TurnVerifier(model, history, [job], threading.Event(), row_pool=row_pool)
        meta = dict(job=job['expected'], timing=TIMING, index=1, profile='full',
                    remaining_ns=900_000_000, native_post_ns=10_000_000, search_ns=800_000_000,
                    rows=[copy.deepcopy(row.row)])
        return model, job, verifier, meta

    def test_helper_rows_match_in_process_admission_exactly(self):
        _, job, inline, meta = self.fixture()
        _, _, pooled, again = self.fixture(self.pool)
        self.assertEqual(inline.verify(meta, time.monotonic())['status'], 'ok')
        self.assertEqual(pooled.verify(again, time.monotonic())['status'], 'ok')
        key = job['expected']['job_id']
        expected, actual = inline.cache[key], pooled.cache[key]
        self.assertEqual(dict(expected['rejected']), dict(actual['rejected']))
        self.assertEqual(len(expected['accepted']), len(actual['accepted']))
        for want, got in zip(expected['accepted'], actual['accepted'], strict=True):
            self.assertEqual(shape(want.row), shape(got.row)); self.assertEqual(want.identity, got.identity)
            for name in ('state', 'actions', 'stable_to_relative'):
                a, b = getattr(want.encoded, name), getattr(got.encoded, name)
                self.assertEqual(a.dtype, b.dtype); np.testing.assert_array_equal(a, b)
                self.assertFalse(b.flags.writeable)
            self.assertEqual((want.encoded.choice_ids, want.encoded.feature_version),
                             (got.encoded.choice_ids, got.encoded.feature_version))
        # The LabelRow owns the trainer's own frozen row, not a copy from the helper.
        self.assertIs(actual['accepted'][0].row, again['rows'][0])
        self.assertEqual(pooled.rows.report()['turns'], 1)

    def test_helper_refusal_reaches_the_turn(self):
        _, _, pooled, meta = self.fixture(self.pool)
        meta['rows'][0]['teacher']['target'][0] += .5
        with self.assertRaisesRegex(ValueError, 'Teacher target'):pooled.verify(meta, time.monotonic())
        self.assertTrue(all(not r['accepted'] for r in pooled.cache.values()))

    def test_unknown_lineage_is_refused_in_the_helper(self):
        model, job, _, meta = self.fixture()
        stranger = F.TurnVerifier(model, [dict(generation=11, model_id='0'*64)], [job], threading.Event(), row_pool=self.pool)
        with self.assertRaisesRegex(ValueError, 'lineage'):stranger.verify(meta, time.monotonic())

    def test_cancellation_stops_a_waiting_turn(self):
        _, _, pooled, meta = self.fixture(self.pool)
        pooled.cancel.set()
        with self.assertRaisesRegex(ValueError, 'cancelled'):pooled.verify(meta, time.monotonic())

    def test_a_table_always_uses_the_same_helper(self):
        routes = {W.route('job-%d' % i, 4) for i in range(64)}
        self.assertEqual(routes, {0, 1, 2, 3})
        self.assertEqual([W.route('same', 4) for _ in range(5)], [W.route('same', 4)]*5)

    def test_a_dead_helper_falls_back_to_identical_in_process_rows(self):
        import signal
        pool = W.RowPool(1)
        try:
            _, job, pooled, meta = self.fixture(pool)
            _, _, inline, again = self.fixture()
            pid = pool.ready[0].result(timeout=120)
            pooled.rows.configured[0].result(timeout=60)
            os.kill(pid, signal.SIGTERM)
            self.assertEqual(pooled.verify(meta, time.monotonic())['status'], 'ok')
            self.assertTrue(pooled.rows.unavailable and pool.broken and pooled.rows.report()['fallback'])
            self.assertEqual(inline.verify(again, time.monotonic())['status'], 'ok')
            key = job['expected']['job_id']
            for want, got in zip(inline.cache[key]['accepted'], pooled.cache[key]['accepted'], strict=True):
                np.testing.assert_array_equal(want.encoded.state, got.encoded.state)
                np.testing.assert_array_equal(want.encoded.actions, got.encoded.actions)
        finally:
            pool.shutdown()

    def test_diagnostic_turns_stay_in_process(self):
        model, job, _, _ = self.fixture()
        verifier = F.TurnVerifier(model, [], [job], threading.Event(), diagnostic=True, row_pool=self.pool)
        self.assertIsNone(verifier.rows)


class Transport(unittest.TestCase):
    def test_ram_needs_room_in_tmpfs_and_memory(self):
        big = P.RAM_MIN_FREE
        self.assertTrue(P.roomy(big, big))
        for free, memory in ((big-1, big), (big, big-1), (None, big), (big, None)):
            with self.subTest(free=free, memory=memory):self.assertFalse(P.roomy(free, memory))

    def test_windows_and_missing_tmpfs_keep_the_given_folder(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as folder:
            with mock.patch.object(P, 'RAM_DIRECTORY', Path(folder)/'missing'):
                self.assertEqual(P.transport_directory(folder), (Path(folder), 'disk'))
            if os.name != 'posix':
                self.assertEqual(P.transport_directory(folder), (Path(folder), 'disk'))

    @unittest.skipUnless(hasattr(os, 'getuid'), 'POSIX process ownership')
    def test_stale_folders_of_dead_trainers_are_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dead = root/(P.PREFIX+'999999999-x'); live = root/(P.PREFIX+f'{os.getpid()}-y'); other = root/'unrelated'
            for path in (dead, live, other):path.mkdir()
            P.reap_stale(root)
            self.assertFalse(dead.exists()); self.assertTrue(live.exists()); self.assertTrue(other.exists())

    def test_owned_tensors_copy_once_and_keep_dtype(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        mapped = np.frombuffer(source.tobytes(), dtype=np.uint8)
        view = P.view(mapped, (0, (3, 4), '<f4'))
        tensor = P.owned_tensor(view, pinned=False)
        self.assertEqual(tensor.dtype, torch.float32); np.testing.assert_array_equal(tensor.numpy(), source)
        if torch.cuda.is_available():
            for array in (view, np.array([True, False]), np.arange(3, dtype=np.int64)):
                pinned = P.owned_tensor(array, pinned=True)
                self.assertTrue(pinned.is_pinned()); np.testing.assert_array_equal(pinned.numpy(), array)

    def test_preparation_reports_its_transport(self):
        with P.PreparationPool(4) as pool, P.Preparation(pool) as prep:
            self.assertIn(prep.metrics['transport'], ('ram', 'disk'))
            self.assertTrue(prep.path.name.startswith(f'{P.PREFIX}{os.getpid()}-'))
            self.assertEqual(prep.report()['transport'], prep.metrics['transport'])


class Prefetch(unittest.TestCase):
    def buffer(self, files):
        import shutil
        from brass_ai.recovery.ppg_replay import Buffer
        value = Buffer.__new__(Buffer)
        value.window = 2; value.directory = Path(tempfile.mkdtemp(dir=ROOT/'artifacts'))
        self.addCleanup(shutil.rmtree, value.directory, True)
        value.receipts = [dict(generation=g, files={ext: {} for ext in files}) for g in (4, 5)]
        for receipt in value.receipts:
            for ext in files:value.path(receipt['generation'], ext).write_bytes(b'x')
        return value

    def test_only_the_previous_critic_files_are_read_ahead(self):
        value = self.buffer(('critic.npy', 'actions.npy', 'local-scoring.json'))
        calls = []
        with mock.patch.object(os, 'posix_fadvise', create=True, side_effect=lambda fd, *a: calls.append(a)), \
             mock.patch.object(os, 'POSIX_FADV_WILLNEED', 3, create=True):
            self.assertEqual(value.prefetch(), 1)
            value.prefetching.join(5)
        self.assertEqual(calls, [(0, 0, 3)])

    def test_no_hint_without_posix_fadvise(self):
        value = self.buffer(('critic.npy',))
        if not hasattr(os, 'posix_fadvise'):
            self.assertIsNone(value.prefetch())
        value.receipts = []
        self.assertIsNone(value.prefetch())


if __name__ == '__main__':
    unittest.main()
