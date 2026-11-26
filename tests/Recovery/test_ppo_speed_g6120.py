"""g6120 code-only speedups: every replaced path, restored here, gives the same tensors, files and updates."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT, digest, require
from brass_ai.recovery import ppg_cache_writer as W, updates as U, value_preservation as V


# ---- the paths g6120 replaced, as they were ---------------------------------------------------------------

def reference_outputs(model, inputs, ds, ix, companion=None, diversity=None):
    from brass_ai.learning.seat_descriptors import critic_inputs
    ids = ix.detach().cpu().tolist()
    encoded = model.encode_state(inputs[0])
    embeddings = model.descriptor_embeddings(encoded, critic_inputs(ds, ids))
    heads = model.official_heads(encoded, inputs[3], embeddings)
    utility = model.predict_utility(encoded.context, inputs[3], heads, ds.remaining[ix], embeddings)
    result = dict(win=heads[0].float(), rank=heads[1].float(), vp=heads[2].float(), utility=utility.float())
    if diversity is not None and diversity.active:
        result['intrinsic'] = diversity.prediction(model, inputs, ds, ix, encoded=encoded, embeddings=embeddings)
    if companion is not None and companion.active:
        preserved = companion.preservation_weights()
        for kind, (owner, vp) in companion.heads(encoded, embeddings, companion.scale).items():
            if kind not in preserved: continue
            result[kind + '_owner'] = owner.float().softmax(-1)
            result[kind + '_vp'] = vp.float()
    return result


def reference_frozen_init(self, model, ds, update, companion=None, executor=None, diversity=None):
    from brass_ai.recovery.value_phase import value_parts
    self.ds = ds; self.update = update; self.companion = companion; self.values = {}; self.gradients = []
    self.diversity = diversity
    started = time.perf_counter(); was_training = model.training; model.eval()
    self.rows = len(ds.data)
    try:
        with torch.no_grad():
            for part, ix, inputs in ds.value_batches(value_parts(ds, range(self.rows), update, model), model, executor):
                with U.autocast(update, ds.device): out = V.outputs(model, inputs, ds, ix, companion, diversity)
                for k, v in out.items():
                    require(bool(torch.isfinite(v).all()), 'Nonfinite frozen critic output')
                    if k not in self.values: self.values[k] = torch.empty((self.rows, *v.shape[1:]), dtype=torch.float32, device='cpu')
                    self.values[k][part] = v.detach().cpu()
    finally: model.train(was_training)
    self.seconds = time.perf_counter() - started


def reference_selected(self, indices, device):
    ids = indices.detach().cpu() if isinstance(indices, torch.Tensor) else indices
    return {k: v[ids].to(device) for k, v in self.values.items()}


def reference_entropy_weights(dataset, indices, behavior, device):
    from brass_ai.recovery.exploration import group
    ids = indices.detach().cpu().tolist() if isinstance(indices, torch.Tensor) else indices
    return torch.tensor([behavior.entropy_alphas[group(dataset.rows[i]['view']['context'])] for i in ids], device=device)


def reference_write_critic_file(array, dataset, ids, source, reward_id, selection, generation, *, block_rows=256, writev=None):
    """The pre-g6120 writer: rows copied into the mapping (contexts keyed by their digest), then flushed."""
    with mock.patch.multiple(W, context_key=digest, context_copy=copy.deepcopy):
        games, contexts, report = W.write_critic(array, dataset, ids, source, reward_id, selection, generation, block_rows=block_rows)
    array.flush()
    return games, contexts, report


def strip(value):
    """A report without wall times, storage labels or the new device-row metrics."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items()
                if k not in ('timings', 'seconds', 'memory', 'storage', 'device_rows', 'snapshot_seconds', 'cpu_preparation')
                and not k.endswith('_seconds')}
    if isinstance(value, list): return [strip(v) for v in value]
    return value


def differences(a, b, path=''):
    """Dotted paths where two reports differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b), key=str) for p in differences(a.get(k), b.get(k), f'{path}.{k}')]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for j, (x, y) in enumerate(zip(a, b)) for p in differences(x, y, f'{path}[{j}]')]
    return [] if a == b else [path]


def same_tensors(case, a, b):
    case.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        case.assertEqual(x.dtype, y.dtype); case.assertEqual(tuple(x.shape), tuple(y.shape))
        case.assertTrue(torch.equal(x, y))


# ---- tests -----------------------------------------------------------------------------------------------

class DeviceRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts'); self.addCleanup(self.tmp.cleanup)

    def phases(self, model, cfg, name):
        from brass_ai.recovery.actor_value import TrainingPhases
        return TrainingPhases(model, cfg, root=SOURCE_ROOT, run=Path(self.tmp.name) / name)

    def test_gathered_batches_equal_collated_ones(self):
        from test_ppg_eight import setup, collect
        torch.manual_seed(3); m, opt, cfg = setup()
        behavior, data = collect(m, self.phases(m, cfg, 'a'), 1)
        rng = np.random.default_rng(5); n = len(data)
        parts = list(U.Dataset(data, 'cpu').microbatches(range(n), 8, 131072))
        parts += [rng.choice(n, 7).tolist(), rng.choice(n, 13).tolist(), [0, 0, n - 1]]  # unsorted, with repeats
        for exploration in (0, .25):
            for value_only in (False, True):
                with self.subTest(exploration=exploration, value_only=value_only):
                    collated = list(U.Dataset(data, 'cpu').policy_batches(parts, exploration, None, value_only=value_only))
                    with mock.patch.object(U.Dataset, 'device_rows_on_cpu', True):
                        ds = U.Dataset(data, 'cpu')
                        gathered = list(ds.policy_batches(parts, exploration, None, value_only=value_only))
                    self.assertIsNotNone(ds._device_rows.actions if not value_only else ds._device_rows.first)
                    self.assertEqual(len(collated), len(gathered))
                    for (p, ix, inputs, q), (p2, ix2, inputs2, q2) in zip(collated, gathered):
                        self.assertEqual(p, p2)
                        same_tensors(self, (ix, *inputs, q), (ix2, *inputs2, q2))

    def test_ppo_update_equals_the_collated_path_with_host_frozen_values(self):
        from test_ppg_eight import setup, collect
        from brass_ai.recovery import dcpg_config
        from brass_ai.recovery.updates import ppo_update
        from brass_ai.recovery.utility import reward_spec
        torch.manual_seed(1); m, opt, cfg = setup(); cfg = dcpg_config.configure(cfg)
        cfg['update'].update(epochs=2, effective_batch_size=128, critic_batch_size=128)
        other = copy.deepcopy(m); second = torch.optim.AdamW(other.parameters()); second.load_state_dict(copy.deepcopy(opt.state_dict()))
        a, b = self.phases(m, cfg, 'new'), self.phases(other, cfg, 'old')
        behavior, data = collect(m, a, 1)
        b.snapshot_data = copy.deepcopy(a.snapshot_data); b.collection_source = copy.deepcopy(a.collection_source)
        reward = reward_spec(cfg); kwargs = dict(learner_data=data, insolvency=reward['insolvency'], reward=reward)
        rng = torch.get_rng_state().clone()
        with mock.patch.object(U.Dataset, 'device_rows_on_cpu', True):
            new = ppo_update(m, opt, [dict(split='train')], behavior, cfg['update'], phases=a, **kwargs)
        self.assertGreater(new['device_rows']['actor']['batches'], 0)
        self.assertGreater(new['device_rows']['critic']['batches'], 0)
        torch.set_rng_state(rng)
        with mock.patch.object(V, 'outputs', reference_outputs), \
             mock.patch.object(V.FrozenValues, '__init__', reference_frozen_init), \
             mock.patch.object(V.FrozenValues, 'selected', reference_selected):
            old = ppo_update(other, second, [dict(split='train')], behavior, cfg['update'], phases=b, **kwargs)
        self.assertIsNone(old['device_rows']['actor'])
        # Only diagnostics of the replaced path differ: gathered batches never enter the host scan cache, and each
        # row is admitted once instead of being re-audited at every collation.
        self.assertEqual([p for p in differences(strip(new), strip(old)) if not p.startswith('.input_cache.scans.')
                          and p != '.input_cache.admission.selector_reuses'], [])
        self.assertEqual(new['input_cache']['scans']['entries'], 0)
        self.assertLess(new['input_cache']['admission']['selector_reuses'], old['input_cache']['admission']['selector_reuses'])
        for (k, x), (_, y) in zip(m.state_dict().items(), other.state_dict().items()):
            self.assertTrue(torch.equal(x, y), k)
        for x, y in zip(opt.state_dict()['state'].values(), second.state_dict()['state'].values()):
            for name in x: self.assertTrue(torch.equal(x[name], y[name]))
        # PPG reads host copies of what PPO froze.
        self.assertEqual(sorted(a.frozen_outputs), sorted(b.frozen_outputs))
        for k in a.frozen_outputs:
            self.assertEqual(a.frozen_outputs[k].device.type, 'cpu')
            self.assertTrue(torch.equal(a.frozen_outputs[k], b.frozen_outputs[k]))

    def test_entropy_group_weights_equal_the_per_row_lookup(self):
        from test_ppg_eight import setup, collect
        torch.manual_seed(3); m, opt, cfg = setup()
        _, data = collect(m, self.phases(m, cfg, 'e'), 1)
        ds = U.Dataset(data, 'cpu')
        from brass_ai.recovery.exploration import group
        names = sorted({group(r['view']['context']) for r in ds.rows})
        behavior = mock.Mock(raw_entropy=True, entropy_alphas={name: .01 * (j + 1) / 3 for j, name in enumerate(names)})
        config = dict(entropy_weight=.01)
        rng = np.random.default_rng(2)
        for part in (list(range(len(data))), rng.choice(len(data), 9).tolist()):
            logits = torch.randn(len(part), 5)
            entropy, bonus = U.entropy_term(logits, None, ds, part, behavior, config)
            expected = reference_entropy_weights(ds, part, behavior, logits.device) * entropy
            self.assertTrue(torch.equal(bonus, expected))
            _, bonus = U.entropy_term(logits, None, ds, torch.as_tensor(part), behavior, config)
            self.assertTrue(torch.equal(bonus, expected))


class TeacherStaging(unittest.TestCase):
    def rows(self, count=24):
        from brass_ai.learning.features import Encoded, FEATURE_VERSION, feature_schema
        from brass_ai.recovery.teacher_labels import LabelRow
        schema = feature_schema(); rng = np.random.default_rng(4); rows = []
        for w in rng.integers(2, 9, count):
            actions = np.zeros((int(w), schema['action_dim']), np.float32); actions[:, 0] = 1.
            e = Encoded(rng.standard_normal(schema['state_dim']).astype(np.float32), actions,
                        np.arange(4, dtype=np.int64), tuple(str(i) for i in range(int(w))), FEATURE_VERSION)
            rows.append(LabelRow(dict(teacher=dict(target=[1. / int(w)] * int(w), legal_ids=list(range(int(w))))), e))
        return rows

    def test_admitted_rows_stage_the_same_tensors_without_a_second_audit(self):
        from brass_ai.recovery.batch_cache import InputAudit
        from brass_ai.recovery.teacher_buffer import admitted
        from brass_ai.recovery.teacher_term import StagedRows
        rows = self.rows(); bonus = [np.linspace(0, 1, len(r.label['target'])).astype(np.float32) for r in rows]
        audit = InputAudit(); plain = StagedRows(rows, 'cpu', audit, values=bonus)
        self.assertEqual(audit.checks, len(rows))
        sealed = [admitted(r) for r in rows]
        self.assertIs(sealed[0].label, rows[0].label)
        detached = admitted(rows[0], detach=True)
        self.assertEqual(detached.label, rows[0].label); self.assertIsNot(detached.label, rows[0].label)
        audit = InputAudit(); staged = StagedRows(sealed, 'cpu', audit, values=bonus)
        self.assertEqual((audit.checks, audit.hits), (0, 0))
        same_tensors(self, (plain.states, plain.actions, plain.targets, plain.offsets, plain.lengths),
                     (staged.states, staged.actions, staged.targets, staged.offsets, staged.lengths))
        same_tensors(self, plain.batch([3, 0, 7]), staged.batch([3, 0, 7]))
        # Admission still refuses a multi-hot selector, once, when the row enters the buffer.
        from brass_ai.learning.local_entities import entity_layout
        field = next(iter(entity_layout()['action_fields'].values()))
        bad = rows[1].encoded.actions.copy(); bad[0, field['start']:field['start'] + 2] = 1.
        from brass_ai.learning.features import Encoded
        from brass_ai.recovery.teacher_labels import LabelRow
        e = rows[1].encoded
        with self.assertRaisesRegex(ValueError, 'one-hot'):
            admitted(LabelRow(rows[1].row, Encoded(e.state, bad, e.stable_to_relative, e.choice_ids, e.feature_version)))


class CriticCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_context_keys_partition_like_their_digests(self):
        base = dict(protocol='0.2.0', rules='steam-brass-patch-1.1.0', revision=7, actor=1, turn_owner=1, phase='action',
                    actions_remaining=2, era='CanalEra', round=3, turn=2, terminal=False, chance=False)
        variants = [base, dict(reversed(list(base.items()))), dict(base, revision=8), dict(base, terminal=0),
                    dict(base, chance=None), dict(base, round=3.0), dict(base, extra=[1, 2]), dict(base, extra=[1, 2.0])]
        for a in variants:
            for b in variants:
                self.assertEqual(W.context_key(a) == W.context_key(b), digest(a) == digest(b), (a, b))
            copied = W.context_copy(a)
            self.assertEqual(copied, copy.deepcopy(a)); self.assertEqual(list(copied), list(a)); self.assertIsNot(copied, a)

    def test_gather_writes_complete_after_partial_writes(self):
        import io
        rng = np.random.default_rng(3)
        buffers = [rng.integers(0, 256, int(n), dtype=np.uint8) for n in rng.integers(1, 40, 25)]
        sink = io.BytesIO(); calls = []
        def writev(fd, views):  # at most 3 buffers and 17 bytes per call, as a short write may do
            calls.append(len(views)); data = b''.join(bytes(v) for v in views)[:17]
            sink.write(data); return len(data)
        with mock.patch.object(W, 'IOV_MAX', 3):
            W.write_buffers(0, buffers, writev=writev)
        self.assertEqual(sink.getvalue(), b''.join(b.tobytes() for b in buffers))
        self.assertLessEqual(max(calls), 3)
        with self.assertRaisesRegex(ValueError, 'no progress'):
            W.write_buffers(0, buffers, writev=lambda fd, views: 0)

    def test_file_writer_bytes_equal_the_array_writer(self):
        from test_value_work_optimization import Cache
        from brass_ai.learning.features import Encoded
        from brass_ai.recovery.replay_binding import seal_encoded
        from brass_ai.recovery.value_buffer import row_dtype
        ds, source, reward, selection = Cache().fixture()
        # Sealed and plain rows, and a float64 state, which takes the structured path for its block.
        sealed = [(r, t, i, seal_encoded(e) if j % 2 else e) for j, (r, t, i, e) in enumerate(ds.data)]
        r, t, i, e = sealed[2]
        sealed[2] = (r, t, i, Encoded(e.state.astype(np.float64), e.actions, e.stable_to_relative, e.choice_ids, e.feature_version))
        ds.data = sealed; ids = [3, 0, 1, 2, 0, 1]  # blocks: sealed+plain, float64 (structured), plain+sealed
        expected = np.empty(len(ids), dtype=row_dtype())
        games, contexts, _ = W.write_critic(expected, ds, ids, source, reward, selection, 12, block_rows=2)
        for writev in (None, 'short'):
            with self.subTest(writev=writev), tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
                path = Path(tmp) / 'c.npy'
                a = np.lib.format.open_memmap(path, mode='w+', dtype=row_dtype(), shape=(len(ids),)); offset = a.offset
                import os
                short = (lambda fd, views: os.write(fd, bytes(views[0])[:1000])) if writev == 'short' else None
                try: g, c, report = W.write_critic_file(a, ds, ids, source, reward, selection, 12, block_rows=2, writev=short)
                finally: a._mmap.close()
                data = path.read_bytes()
                self.assertEqual(data[offset:], expected.tobytes()); self.assertEqual((g, c), (games, contexts))
                self.assertEqual(np.load(path).tobytes(), expected.tobytes())
                self.assertEqual((report['direct_rows'], report['structured_rows']), (4, 2))
                self.assertEqual(report['source_finite_rows'], 2)  # the plain rows written directly
        # A plain row's non-finite state is still refused.
        r, t, i, e = ds.data[0]; bad = e.state.copy(); bad[5] = np.nan
        ds.data[0] = (r, t, i, Encoded(bad, e.actions, e.stable_to_relative, e.choice_ids, e.feature_version))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            a = np.lib.format.open_memmap(Path(tmp) / 'c.npy', mode='w+', dtype=row_dtype(), shape=(2,))
            try:
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    W.write_critic_file(a, ds, [0, 1], source, reward, selection, 12)
            finally: a._mmap.close()

    def test_staged_cache_files_and_receipt_are_unchanged(self):
        from test_ppg_eight import setup, stage_only
        from brass_ai.recovery.actor_value import TrainingPhases
        results = []
        for name in ('new', 'old'):
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
                torch.manual_seed(7); model, opt, cfg = setup()
                p = TrainingPhases(model, cfg, root=SOURCE_ROOT, run=Path(tmp) / 'run')
                with mock.patch.object(W, 'write_critic_file', reference_write_critic_file) if name == 'old' else mock.MagicMock():
                    report = stage_only(model, p, 1)
                receipt = p.buffer.pending[-1]
                data = p.buffer.path(receipt['generation'], 'critic.npy').read_bytes()
                results.append((receipt, data, report['serialization']))
                # A restart reopens the cache by its sizes and receipts.
                from brass_ai.recovery import ppg_replay
                ppg_replay.Buffer(SOURCE_ROOT, Path(tmp) / 'run', cfg, generation=1, receipts=[receipt])
        (new, new_bytes, timing), (old, old_bytes, _) = results
        self.assertEqual(new_bytes, old_bytes)
        for key in ('contexts', 'game_ids', 'rows', 'policy_entries'):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new['files'], old['files'])
        self.assertTrue(all(set(f) == {'bytes'} for f in new['files'].values()))  # sizes only, no SHA-256
        self.assertEqual(new['files']['critic.npy']['bytes'], len(new_bytes))
        self.assertEqual(timing['writer'], 'gather-write' if hasattr(__import__('os'), 'writev') else 'joined-write')
        self.assertGreater(len(new['contexts']), 1)


class ValueRows(unittest.TestCase):
    def test_reads_take_the_cached_rows_and_never_hash_the_shard(self):
        from brass_ai.recovery.search_value_rows import ValueBuffer
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as temp:
            values = Path(temp) / 'search-values'; values.mkdir()
            buffer = ValueBuffer(values, 'reward')
            row = dict(generation=5, split='train', reward_id='reward', game_id='teacher/g', index=0, state=[0., 1.])
            receipt = buffer.publish(5, 'train', [row], game_id='teacher/g')
            path = values / receipt['file']
            cached = buffer._read(receipt)
            from brass_ai.recovery import search_value_rows as R
            with mock.patch.object(R, 'hashlib', mock.Mock(sha256=mock.Mock(side_effect=AssertionError('hashed')))):
                self.assertIs(buffer._read(receipt), cached)  # the in-memory copy, no file read or hash
                restored = ValueBuffer(values, 'reward', receipts=[receipt])  # a cold read parses without hashing
            self.assertEqual(restored._read(receipt)[0]['index'], 0)
            path.write_bytes(b'damaged on disk')
            self.assertIs(buffer._read(receipt), cached)
            with self.assertRaises(OSError): ValueBuffer(values, 'reward', receipts=[receipt])  # gzip's own check


if __name__ == '__main__':
    unittest.main()
