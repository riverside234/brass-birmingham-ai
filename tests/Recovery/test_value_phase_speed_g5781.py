"""g5781 value-phase speed: the host paths that replace per-row work give exactly the phase they replace.

- Descriptor dropout is decided once per game (`seat_descriptors.dropout_mask`), not by a SHA-256 and a replay-row
  read per row; the PPG critic replay maps rows to games with its game column (`game_codes`).
- Branch-family counts come from that column too (`value_phase.branch_rows`).
- The era heads take the batch's host rows and a host copy of the critic weights instead of device read-backs, and
  upload their targets with pinned non-blocking copies (`local_scoring_v3.upload`).
- The era heads gather every row's targets from per-phase device tables (`local_scoring_v3.EraTargets`) for the critic
  replay and the search rows, with the target cache's counters replayed exactly.
"""
import math
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
import torch

from brass_ai.learning import seat_descriptors as S
from brass_ai.recovery import local_scoring_v3 as v3
from brass_ai.recovery import ppg_replay
from brass_ai.recovery import value_phase


def reference_inputs(dataset, indices, *, dropout=None, unknown=False, index=None):
    """`critic_inputs` before g5781: a new device index, and `drop_game` with a replay-row read per row."""
    ix = torch.as_tensor(indices, device=dataset.device)
    if not hasattr(dataset, 'descriptors'):
        return S.unknown_tensor(torch.empty((len(indices),), device=dataset.device))
    descriptors = dataset.descriptors[ix]
    if unknown: return S.unknown_tensor(descriptors)
    if dropout is not None:
        generation, epoch, fraction, phase = dropout
        mask = [S.drop_game(dataset.data[i][2][0], generation, epoch, fraction, phase) for i in indices]
        if any(mask):
            descriptors = descriptors.clone()
            descriptors[torch.tensor(mask, device=dataset.device)] = S.unknown_tensor(descriptors)[0]
    return descriptors


def reference_paths():
    """Every replaced path restored: per-row dropout and branch counts, device read-backs, pageable uploads."""
    return [mock.patch.object(S, 'critic_inputs', reference_inputs),
            mock.patch.object(value_phase, 'branch_rows', lambda ds, rows: sum('/branch/' in ds.data[i][2][0] for i in rows)),
            mock.patch.object(value_phase, 'host_ids', lambda companion, part: {}),
            mock.patch.object(v3, 'upload', lambda values, device: torch.as_tensor(values, device=device)),
            mock.patch.object(v3, 'host_weights', lambda ds, ids: ds.critic_weights[torch.as_tensor(ids)].detach().cpu().tolist()),
            mock.patch.object(ppg_replay.Column, 'host_indices', False),
            # the era heads' per-phase target tables off: every batch builds its targets from its metadata
            mock.patch.object(v3, '_critic_rows', lambda ds: None),
            mock.patch.object(v3, '_meta_rows', lambda ds: None)]


def numbers(value, path=''):
    """Every number of a report, by path, without wall times or process identities."""
    skip = ('seconds', 'timings', '_ns', 'wall', 'prefetch', 'cpu_preparation', 'worker_pids', 'opened_after', 'step_profile')
    if isinstance(value, dict):
        for k, v in value.items():
            if not any(s in str(k) for s in skip): yield from numbers(v, f'{path}.{k}')
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): yield from numbers(v, f'{path}[{i}]')
    elif isinstance(value, (bool, int, float)) or value is None:
        yield path, value
    elif isinstance(value, str):
        yield path, value


class Paths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """One g4770 PPG phase (ppg_g210.fit) with the new paths, checking them on the real critic replay inside the
        phase; and the same phase with every replaced path restored."""
        import test_era_head_revision_g4770 as fixture
        cls.checks = {}
        original = value_phase.fit

        def fit(model, optimizer, dataset, train_ids, heldout_ids, actor_ids, probe_ids, config, update, **kw):
            ds = kw.get('value_dataset')
            if isinstance(ds, ppg_replay.CriticDataset) and 'replay' not in cls.checks:
                cls.checks['replay'] = cls.inspect(ds, list(kw['value_train_ids']), list(heldout_ids))
            return original(model, optimizer, dataset, train_ids, heldout_ids, actor_ids, probe_ids, config, update, **kw)

        with mock.patch.object(value_phase, 'fit', fit):
            cls.new = fixture.Phase.run_phase(None)
        with ExitStack() as stack:
            for patch in reference_paths(): stack.enter_context(patch)
            cls.old = fixture.Phase.run_phase(None)

    @staticmethod
    def inspect(ds, train_ids, heldout_ids):
        rng = np.random.default_rng(5)
        rows = [int(i) for i in rng.choice(np.asarray(train_ids + heldout_ids), 300)]
        found = dict(rows=len(rows), dropped=0, branch=0)
        names = ds.game_names()
        found['codes'] = all(names[c] == ds.data[i][2][0] for i, c in zip(rows, ds.game_codes(rows)))
        same = True
        for generation, epoch, fraction in ((7, 0, .25), (7, 1, .25), (8, 0, .5), (9, 2, 1.)):
            dropout = (generation, epoch, fraction, 'value')
            mask = S.dropout_mask(ds, rows, dropout)
            expected = [S.drop_game(ds.data[i][2][0], *dropout) for i in rows]
            same &= mask.tolist() == expected
            found['dropped'] += int(mask.sum())
            new, old = S.critic_inputs(ds, rows, dropout=dropout), reference_inputs(ds, rows, dropout=dropout)
            same &= torch.equal(new, old)
        same &= torch.equal(S.critic_inputs(ds, rows), reference_inputs(ds, rows))
        found['inputs'] = same
        found['branch'] = value_phase.branch_rows(ds, rows)
        found['branch_reference'] = sum('/branch/' in ds.data[i][2][0] for i in rows)
        found['branch_set'] = value_phase.branch_rows(ds, set(rows)) == sum('/branch/' in ds.data[i][2][0] for i in set(rows))
        kept = ds.critic_weights
        ds.critic_weights = torch.tensor([.5 if k % 3 else 1. for k in range(len(ds.data))])
        found['weights'] = v3.host_weights(ds, rows) == ds.critic_weights[torch.tensor(rows)].tolist()
        ds.critic_weights = torch.ones(len(ds.data))  # a new tensor: the host copy follows it
        found['weights_replaced'] = v3.host_weights(ds, rows) == [1.] * len(rows)
        ds.critic_weights = kept
        return found

    def test_dropout_decided_per_game_equals_the_per_row_rule(self):
        replay = self.checks['replay']
        self.assertTrue(replay['codes'])
        self.assertTrue(replay['inputs'])
        self.assertGreater(replay['dropped'], 0)

    def test_branch_counts_and_host_weights_equal_the_row_reads(self):
        replay = self.checks['replay']
        self.assertEqual(replay['branch'], replay['branch_reference'])
        self.assertGreater(replay['branch_reference'], 0)
        self.assertTrue(replay['branch_set'])
        self.assertTrue(replay['weights'] and replay['weights_replaced'])

    def test_in_memory_dataset_memoizes_the_same_decisions(self):
        from brass_ai.recovery import updates
        from test_ppg_g210 import config
        import test_ppg_g210
        dataset = type('Rows', (), {})()
        dataset.data = [(None, None, (f'game-{k // 3}', k % 4)) for k in range(60)]
        dataset.device = 'cpu'
        dataset.descriptors = torch.rand(60, 4, S.DIM)
        rows = list(range(60))[::-1]
        dropout = (11, 0, .4, 'ppo')
        self.assertEqual(S.dropout_mask(dataset, rows, dropout).tolist(),
                         [S.drop_game(dataset.data[i][2][0], *dropout) for i in rows])
        self.assertTrue(torch.equal(S.critic_inputs(dataset, rows, dropout=dropout),
                                    reference_inputs(dataset, rows, dropout=dropout)))
        self.assertTrue(torch.equal(S.critic_inputs(dataset, rows, dropout=dropout, index=torch.tensor(rows)),
                                    reference_inputs(dataset, rows, dropout=dropout)))

    def test_the_phase_equals_the_phase_with_the_replaced_paths(self):
        new, old = dict(numbers(self.new)), dict(numbers(self.old))
        self.assertEqual(set(new), set(old))
        differ = {k: (new[k], old[k]) for k in new if not (new[k] == old[k] or (
            isinstance(new[k], float) and isinstance(old[k], float) and math.isnan(new[k]) and math.isnan(old[k])))}
        self.assertEqual(differ, {})
        self.assertGreater(self.new['attempted_steps'], 0)
        self.assertGreater(self.new['_training'].get('target_table_seconds', 0.), 0.)  # the tables were used
        self.assertNotIn('target_table_seconds', self.old['_training'])
        self.assertGreater(self.new['_training']['target_cache_misses'], 0)
        profile = self.new['step_profile']  # g5781: every part of a joint step is timed
        self.assertEqual(profile['steps'], self.new['attempted_steps'])
        self.assertTrue({'value_rows', 'clone_rows', 'clip_and_device_wait', 'optimizer'} <= set(profile['seconds']))
        self.assertGreater(self.new['_moved']['owner.canal_slots.weight'], 0.)


if __name__ == '__main__':
    unittest.main()
