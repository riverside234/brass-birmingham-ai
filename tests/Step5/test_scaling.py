"""Focused checks for scaled collection, memory bounds and run migration."""
import copy
from collections import Counter
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.replay import split_for
from brass_ai.learning.training import prepared
from brass_ai.step5.league import collection_batches, table_kind, train, validate_config
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.step5.updates import bounded_batches, ppo_update, distill_update
from fixtures import game


def onpolicy_game(policies, seed=510001):
    result = game()
    group = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=seed))
    result.update(game_id='scaling-' + str(seed), setup_group=group, split=split_for(group),
                  opponents=[p.policy_id for p in policies])
    for row in result['records']:
        policy = policies[row['seat']]
        rng = random.Random(row['seat'])
        selected, probability, teacher = policy.select_batch([(row['view'], row['choices'], rng)])[0]
        style, intent = policy.context_features(row['view'], rng)
        row.update(chosen=selected, teacher=teacher, style=style, intent=intent,
                   behavior=dict(kind='model', policy_id=policy.policy_id, probability=probability, temperature=1.0))
    return result


class ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_presets_and_strict_new_fields(self):
        large = validate_config(read(SOURCE_ROOT / 'configs/step5/previous-lab.json'))
        small = validate_config(read(SOURCE_ROOT / 'configs/step5/lab-small.json'))
        self.assertEqual((large['games_per_iteration'], large['workers'], large['minibatch_size']), (320, 32, 512))
        self.assertEqual(small['format'],'step5-training/2.1.0')
        self.assertEqual(small['games_per_iteration'],160)
        self.assertEqual((small['execution']['processes'],small['execution']['hosts_per_process']),(32,1))
        for field, bad in [('minibatch_size', 0), ('max_candidate_cells', -1), ('search_workers', True),
                           ('parallel_hosts', 1), ('group_tables', 'yes')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(dict(large, **{field: bad}))
        with self.assertRaises(ValueError):
            validate_config(dict(large, num_workers=64))

    def test_grouping_fills_workers_and_keeps_exact_opponent_quotas(self):
        for offset in (0, 7, 53):
            batches = list(collection_batches(offset, 320, 32, 20, True))
            indices = [i for _, batch in batches for i in batch]
            self.assertEqual(sorted(indices), list(range(320)))
            self.assertEqual(Counter({kind: sum(len(b) for k, b in batches if k == kind) for kind in ('selfplay', 'historical', 'legacy')}),
                             Counter(table_kind(offset + i) for i in range(320)))
            self.assertTrue(any(len(b) == 32 for kind, b in batches if kind == 'selfplay'))
            self.assertTrue(all(len(b) <= 32 and all(table_kind(offset + i) == kind for i in b) for kind, b in batches))
        consecutive = list(collection_batches(0, 320, 32, 20, False))
        self.assertEqual(max(len(b) for _, b in consecutive), 14)

    def test_padded_cell_limit_preserves_all_rows(self):
        from types import SimpleNamespace
        data = [(None, None, None, SimpleNamespace(choice_ids=list(range(n)))) for n in (2, 5, 3, 8)]
        batches = list(bounded_batches(data, 4, 10))
        self.assertEqual([(start, len(batch)) for start, batch in batches], [(0, 2), (2, 1), (3, 1)])
        self.assertEqual([entry for _, batch in batches for entry in batch], data)
        self.assertEqual(len(list(bounded_batches(data, 4))), 1)
        with self.assertRaises(ValueError):
            list(bounded_batches(data, 4, 7))

    def test_microbatch_size_preserves_full_epoch_update(self):
        from brass_ai.step5.league import seeds_for
        torch.manual_seed(181)
        first = FourSeatNet(ModelConfig.preset('small'))
        second = copy.deepcopy(first)
        policy = NeuralPolicy(first)
        games = [onpolicy_game([policy] * 4, seeds_for(551000, 1)[0][0])]
        cells = max(len(e[3].choice_ids) for e in prepared(games))
        a = ppo_update(first, torch.optim.AdamW(first.parameters(), lr=1e-4), games, policy, epochs=2, batch_size=512)
        b = ppo_update(second, torch.optim.AdamW(second.parameters(), lr=1e-4), games, policy,
                       epochs=2, batch_size=512, max_candidate_cells=cells)
        self.assertEqual(a['optimizer_steps'], b['optimizer_steps'])
        self.assertEqual(a['learner_decisions'], b['learner_decisions'])
        self.assertLessEqual(b['max_padded_candidate_cells'], cells)
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-3, atol=2e-6)

    def test_fork_preserves_weights_optimizer_history_and_setup_cursor(self):
        # This gate concerns the historical execution-only migration. Recipe
        # 2.1 deliberately starts fresh optimizer moments in a separate run.
        config = read(SOURCE_ROOT / 'configs/step5/smoke.json')
        config.update(model_size='small', games_per_iteration=1, workers=1, threads=2,
                      legacy_every=0, search_every=0, ppo_epochs=1)
        seen = []

        def collect(root, path, host, seeds, policies, config, provenance, **kwargs):
            seen.extend(seeds)
            return [onpolicy_game(policies, seed) for seed in seeds]

        temporary_root = SOURCE_ROOT / 'artifacts/step5'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            base = Path(temporary)
            source, target = base / 'source', base / 'scaled'
            with patch('brass_ai.step5.league.collect_cached', side_effect=collect):
                train(SOURCE_ROOT, source, config, iterations=1)
                original = (source / 'state.json').read_bytes()
                parent = read(source / 'state.json')
                _, parent_raw = load_checkpoint(SOURCE_ROOT / parent['history'][-1]['path'])
                large = dict(config, format='step5-training/1.1.0', minibatch_size=16,
                             max_candidate_cells=8192, parallel_hosts=True, group_tables=True, search_workers=2)
                train(SOURCE_ROOT, target, large, from_run=source, iterations=1)
                child = read(target / 'state.json')
                _, initial = load_checkpoint(target / 'generation-0000.pt')
                self.assertEqual(initial['model_id'], parent_raw['model_id'])
                torch.testing.assert_close(initial['rng'], parent_raw['rng'], rtol=0, atol=0)
                self.assertEqual(initial['optimizer']['param_groups'], parent_raw['optimizer']['param_groups'])
                for key, state in parent_raw['optimizer']['state'].items():
                    for name, value in state.items():
                        torch.testing.assert_close(initial['optimizer']['state'][key][name], value, rtol=0, atol=0)
                self.assertEqual(child['origin']['iteration'], parent['iteration'])
                self.assertEqual(child['games_done'], parent['games_done'] + 1)
                self.assertEqual(child['history'][0], parent['history'][0])
                self.assertGreaterEqual(seen[1], parent['next_seed'])
                self.assertEqual((source / 'state.json').read_bytes(), original)
                train(SOURCE_ROOT, target, large, iterations=1)
                self.assertEqual(read(target / 'state.json')['iteration'], 2)
                self.assertEqual(len(set(seen)), len(seen))

    def test_distillation_memory_splits_preserve_teacher_weights(self):
        from brass_ai.step5.league import seeds_for
        torch.manual_seed(271)
        first = FourSeatNet(ModelConfig.preset('small'))
        second = copy.deepcopy(first)
        policy = NeuralPolicy(first)
        games = [onpolicy_game([policy] * 4, seeds_for(551000, 1)[0][0])]
        for row in games[0]['records']:
            row['teacher'] = dict(source='fresh-finalist/1.0.0', probabilities=[.8, .2, 0.0],
                coverage=[True, True, False], confidence=.75, samples=4,
                horizon='complete-current-turn/actual-next-root-decision/neural-v1')
        torch.manual_seed(721)
        a = distill_update(first, torch.optim.AdamW(first.parameters(), lr=1e-4), games, steps=2, batch_size=512)
        torch.manual_seed(721)
        b = distill_update(second, torch.optim.AdamW(second.parameters(), lr=1e-4), games,
                           steps=2, batch_size=512, max_candidate_cells=3)
        self.assertTrue(a['updated'] and b['updated'])
        self.assertEqual(a['teacher_rows'], b['teacher_rows'])
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-3, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
