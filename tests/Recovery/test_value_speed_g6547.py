"""g6547 value-phase speed: state-only value rows run as one device batch per step (value_phase.JOINED_ROWS).

- `joined_value_batches` joins the CPU workers' parts on the device: the same rows in order, the same inputs, and the
  critic replay's prepared columns (targets, utility, descriptors, seats) attached for the joined batch only.
- A whole PPG phase (the g4770 fixture: era heads, ownership stop, search rows, Rail-end) with joined batches reports
  the numbers of the unjoined phase, up to float rounding; only the era heads' target-cache counters (diagnostics of
  a cache the per-phase tables replaced) count the joined batches.
"""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import value_phase
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.local_scoring_auxiliary import DEFAULT as AUX
from brass_ai.recovery.ppo_preparation import Preparation, PreparationPool
from brass_ai.step5.league import write_json


def numbers(value, path=''):
    skip = ('seconds', 'timings', '_ns', 'wall', 'prefetch', 'cpu_preparation', 'worker_pids', 'opened_after', 'step_profile')
    if isinstance(value, dict):
        for k, v in value.items():
            if not any(s in str(k) for s in skip): yield from numbers(v, f'{path}.{k}')
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): yield from numbers(v, f'{path}[{i}]')
    elif isinstance(value, (bool, int, float, str)) or value is None:
        yield path, value


class Joined(unittest.TestCase):
    def test_worker_parts_join_with_their_prepared_columns(self):
        from test_ppg_dense_eight import config
        from test_ppg_eight import stage_only
        from test_local_scoring import annotate
        import test_ppg_eight as base
        m, _, c = config(); c['local_scoring_auxiliary'] = copy.deepcopy(AUX)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            run = Path(tmp); p = TrainingPhases(m, c, root=SOURCE_ROOT, run=run)
            original = base.collect
            def labeled(*args, **kwargs):
                b, data = original(*args, **kwargs); annotate(data); annotate(p.snapshot_data); return b, data
            with mock.patch.object(base, 'collect', labeled):
                for g in (1, 2):
                    stage_only(m, p, g)
                    if g == 1: write_json(run / 'state.json', dict(iteration=g)); p.buffer.committed(g)
            with p.buffer.datasets('cpu') as (ds, _, _), PreparationPool(4) as pool, Preparation(pool, directory=run) as prep:
                n = len(ds.data.parts[0])
                parts = [[n + 2, 0, n + 2], [1, n + 1], [3], [0, 2], [n, 1, 4]]
                single = []
                for part, ix, inputs in ds.value_batches(parts, m, prep):
                    single.append((part, ix.clone(), tuple(t.clone() for t in inputs),
                                   tuple(t[ix].clone() for t in ds.targets), ds.utility[ix].clone(),
                                   ds.descriptors[ix].clone(), ds.seats[ix].clone()))
                joined = []
                for part, ix, inputs in value_phase.joined_value_batches(ds, parts, m, executor=prep, rows=6):
                    self.assertIsNotNone(ds.data.prepared)  # the joined columns serve this batch's lookups
                    joined.append((part, ix, inputs, tuple(t[ix] for t in ds.targets), ds.utility[ix],
                                   ds.descriptors[ix], ds.seats[ix], ds.descriptors[part]))
                self.assertIsNone(ds.data.prepared)
                # 3+2+1 rows, then 2+3: two joined batches of the five parts
                self.assertEqual([j[0] for j in joined], [parts[0] + parts[1] + parts[2], parts[3] + parts[4]])
                for (part, ix, inputs, targets, utility, descriptors, seats, by_rows), group in zip(joined, ((0, 1, 2), (3, 4))):
                    cat = lambda k: torch.cat([single[g][k] for g in group])
                    self.assertTrue(torch.equal(ix, cat(1)))
                    for j, t in enumerate(inputs): self.assertTrue(torch.equal(t, torch.cat([single[g][2][j] for g in group])))
                    for j, t in enumerate(targets): self.assertTrue(torch.equal(t, torch.cat([single[g][3][j] for g in group])))
                    self.assertTrue(torch.equal(utility, cat(4)) and torch.equal(descriptors, cat(5)) and torch.equal(seats, cat(6)))
                    self.assertTrue(torch.equal(by_rows, descriptors))  # the host row list hits the same columns
                gen = value_phase.joined_value_batches(ds, parts, m, executor=prep, rows=6); next(gen); gen.close()
                self.assertIsNone(ds.data.prepared); self.assertFalse(prep.futures)
                # off: the parts pass through unchanged
                self.assertEqual([part for part, _, _ in value_phase.joined_value_batches(ds, parts, m, executor=prep, rows=0)], parts)


class Corrections(unittest.TestCase):
    def test_table_arrays_equal_the_per_row_corrections(self):
        """search_rows.search_corrections (per-table arrays) against its per-row form, bitwise: gaps in the turn
        order, extra rows past an era's last root, a table without a Rail map, every seat rotation, Rail-end on/off."""
        import numpy as np
        from brass_ai.recovery import search_rows as R
        rng = np.random.default_rng(6547); stable = {}
        def map_arrays(target, mapping):
            if target not in stable: stable[target] = rng.random((R.ENTITIES, R.WIDTH))
            return stable[target], None
        tables = []
        for t in range(9):
            eras = ('CanalEra', 'RailEra') if t % 4 else ('CanalEra',)
            maps = {era: ('map', t, era) for era in eras}
            turns = [k for k in range(12 + t) if not (t % 3 == 0 and k in (3, 7))]  # gaps
            rows = []
            for k in turns:
                seat = int(rng.integers(4))
                rows.append(dict(index=k, turn_key=['CanalEra' if k < 6 or 'RailEra' not in maps else 'RailEra', k],
                                 mapping=[(s - seat) % 4 for s in range(4)]))
            for e in range(15):
                root = rows[int(rng.integers(len(rows)))]; seat = int(rng.integers(4))
                rows.append(dict(kind='develop', index=root['index'] + (e % 5 == 0), revision=e, turn_key=root['turn_key'],
                                 mapping=[(s - seat) % 4 for s in range(4)]))
            frozen = {j: rng.random((R.ENTITIES, R.WIDTH)).astype(np.float32) for j, r in enumerate(rows) if R.kind(r) == 'turn'}
            ends = {j: rng.random((R.ENTITIES, R.WIDTH)).astype(np.float32) for j, r in enumerate(rows)
                    if R.kind(r) == 'turn' and r['turn_key'][0] == 'CanalEra'}
            tables.append((rows, maps, frozen, ends))
        with mock.patch.object(R, 'map_arrays', map_arrays):
            for lam in (.95, .5, 1.):
                for rail_end in (False, True):
                    new, old = R.search_corrections(tables, lam, rail_end=rail_end), R._search_corrections_rows(tables, lam, rail_end=rail_end)
                    self.assertEqual(new.dtype, old.dtype)
                    self.assertTrue(np.array_equal(new, old), (lam, rail_end))
                    if lam < 1: self.assertGreater(np.count_nonzero(new), 0)  # lambda 1: corrections of zero
                    if rail_end and lam < 1: self.assertGreater(np.count_nonzero(new[..., R.WIDTH:]), 0)
            short = [(rows, maps, frozen) for rows, maps, frozen, _ in tables]  # no Rail-end predictions
            self.assertTrue(np.array_equal(R.search_corrections(short, .9), R._search_corrections_rows(short, .9)))


class DeviceMemory(unittest.TestCase):
    """g6547: device rows only for PPO's own datasets, one budget for all live stores, and nothing kept after PPO."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts'); self.addCleanup(self.tmp.cleanup)

    def data(self, name, configure=None):
        from test_ppg_eight import setup, collect
        torch.manual_seed(3); m, opt, cfg = setup()
        if configure is not None: cfg = configure(cfg)
        phases = TrainingPhases(m, cfg, root=SOURCE_ROOT, run=Path(self.tmp.name) / name)
        behavior, data = collect(m, phases, 1)
        return m, opt, cfg, phases, behavior, data

    def test_only_opted_in_datasets_stage_and_share_one_budget(self):
        from brass_ai.recovery import updates as U
        _, _, _, _, _, data = self.data('a')
        parts = [list(range(8))]
        with mock.patch.object(U.Dataset, 'device_rows_on_cpu', True):
            plain = U.Dataset(data, 'cpu'); list(plain.policy_batches(parts))
            self.assertIsNone(getattr(plain, '_device_rows', None))  # probes, drift checks, calibration: collated
            first = U.Dataset(data, 'cpu'); first.device_rows_allowed = True; list(first.policy_batches(parts))
            held = first._device_rows._bytes(); self.assertGreater(held, 0)
            with mock.patch.object(U.DeviceRows, 'LIMIT_BYTES', held + 1):
                second = U.Dataset(data, 'cpu'); second.device_rows_allowed = True; list(second.policy_batches(parts))
                self.assertEqual(second._device_rows.failed, 'over_limit')  # the budget counts the first store
            U.release_device_rows()
            self.assertEqual(first._device_rows._bytes(), 0); self.assertEqual(first._device_rows.failed, 'released')
            expected = list(U.Dataset(data, 'cpu').policy_batches(parts))
            again = list(first.policy_batches(parts))  # a released store serves the original path, the same batches
            self.assertEqual(first._device_rows._bytes(), 0)
            for (p, ix, inputs, q), (p2, ix2, inputs2, q2) in zip(expected, again):
                self.assertEqual(p, p2)
                for a, b in zip((ix, *inputs, q), (ix2, *inputs2, q2)): self.assertTrue(torch.equal(a, b))

    def test_ppo_releases_its_device_rows(self):
        from brass_ai.recovery import dcpg_config, updates as U
        from brass_ai.recovery.utility import reward_spec
        m, opt, cfg, phases, behavior, data = self.data('ppo', dcpg_config.configure)
        cfg['update'].update(epochs=1, effective_batch_size=128, critic_batch_size=128)
        reward = reward_spec(cfg)
        with mock.patch.object(U.Dataset, 'device_rows_on_cpu', True):
            report = U.ppo_update(m, opt, [dict(split='train')], behavior, cfg['update'], phases=phases, learner_data=data,
                                  insolvency=reward['insolvency'], reward=reward)
        self.assertGreater(report['device_rows']['actor']['batches'], 0)
        self.assertGreater(report['device_rows']['actor']['bytes'], 0)  # reported as staged during PPO
        self.assertTrue(all(store._bytes() == 0 for store in list(U.DeviceRows.live)))  # and freed when it returned

    def test_phase_memory_is_off_without_cuda(self):
        from brass_ai.recovery.gpu_memory import Phases
        phases = Phases('cpu'); phases.mark('ppo')
        self.assertIsNone(phases.take())


class PublicationDecode(unittest.TestCase):
    def test_canonical_dumps_and_freeze_equal_their_plain_formulas(self):
        import json
        from brass_ai.recovery import admitted_inputs as A, canonical_inputs as C
        self.assertIsNotNone(C._FAST)
        values = [{'b': [1, 2.5, -0.0, 1e-07, 3e300, True, None, {'é': 'ü'}], 'a': 'q"\\\n☃', 'c': {'z': {}, 'y': []}},
                  [[1, [2, [3, {'k': [4.25]}]]], 'x'], 'text', 7, 1.0, None, False, [], {}]
        for value in values:
            self.assertEqual(C.dumps(value), json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
            self.assertEqual(C.dumps(A._freeze(value)), C.dumps(value))
        for bad in (float('nan'), [float('inf')], {'a': -float('inf')}):
            with self.assertRaises(ValueError): C.dumps(bad)
        with self.assertRaises(TypeError): C.dumps({'a': object()})
        def kinds(a, b):
            if type(b) is dict: return type(a) is A.FrozenDict and list(a) == list(b) and all(kinds(a[k], b[k]) for k in b)
            if type(b) is list: return type(a) is A.FrozenList and len(a) == len(b) and all(kinds(x, y) for x, y in zip(a, b))
            return type(a) is type(b) and a == b
        for value in values: self.assertTrue(kinds(A._freeze(value), value))
        frozen = A._freeze({'a': [1, {'b': 2}]})
        with self.assertRaises(TypeError): frozen['a'] = 1
        with self.assertRaises(TypeError): frozen['a'].append(3)
        with self.assertRaises(TypeError): A._freeze({'a': object()})


class Phase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """The g4770 fixture phase with two-row microbatches (several parts per step), joined and not joined."""
        import test_era_head_revision_g4770 as fixture
        import test_ppg_g210
        original = test_ppg_g210.config
        def config():
            m, opt, c = original(); c['update']['microbatch_size'] = 2
            return m, opt, c
        cls.calls = {}
        source = value_phase.joined_value_batches
        def counted(dataset, parts, model, **kw):
            for item in source(dataset, parts, model, **kw):
                cls.calls[value_phase.JOINED_ROWS] = cls.calls.get(value_phase.JOINED_ROWS, 0) + 1
                yield item
        with mock.patch.object(test_ppg_g210, 'config', config), mock.patch.object(value_phase, 'joined_value_batches', counted):
            cls.joined = fixture.Phase.run_phase(None)
            with mock.patch.object(value_phase, 'JOINED_ROWS', 0):
                cls.single = fixture.Phase.run_phase(None)

    def test_the_joined_phase_reports_the_unjoined_numbers(self):
        joined, single = dict(numbers(self.joined)), dict(numbers(self.single))
        self.assertEqual(set(joined), set(single))
        counters = ('target_cache_hits', 'target_cache_misses')
        differ = {}
        for k, a in joined.items():
            b = single[k]
            if any(c in k for c in counters): continue
            if isinstance(a, float) and isinstance(b, float):
                if math.isnan(a) and math.isnan(b): continue
                if not math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6): differ[k] = (a, b)
            elif a != b: differ[k] = (a, b)
        self.assertEqual(differ, {})
        self.assertGreater(self.joined['attempted_steps'], 0)
        self.assertGreater(self.joined['_moved']['owner.canal_slots.weight'], 0.)
        self.assertIsNotNone(self.joined['ownership_stop']['before'])

    def test_steps_and_scans_run_fewer_larger_batches(self):
        self.assertLess(self.calls[4096], self.calls[0] / 2)


if __name__ == '__main__':
    unittest.main()
