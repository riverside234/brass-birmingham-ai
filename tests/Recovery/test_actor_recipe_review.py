"""Durable actor cache reuse must bind the actual execution inputs."""
import copy
import gzip
import json
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import Mock

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.learning.replay import split_for
from brass_ai.recovery.execution import ActorPool, neural_spec
from brass_ai.recovery.replay import read_shards, write_shard
from review_fixture import current_game as game


class ActorRecipeReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.pool = ActorPool.__new__(ActorPool)
        self.pool.width = self.pool.lanes = 1
        self.pool.heavy = False
        self.pool.config = dict(shard_games=1, feature_cache=False, max_decisions=3000)
        self.pool.service = Mock(); self.pool.service.error = None
        self.pool.processes = []
        self.pool.start = Mock(); self.pool.jobs = Mock(); self.pool.results = queue.Queue()
        self.packets = []
        self.pool.jobs.put.side_effect = self.publish

    def job(self):
        return dict(index=0, seed=17100000, sampling_seed=1701, rotation=0,
            policies=[dict(kind='masked_ismcts', model_id='a'*64, opponents=['a'*64],
                search=dict(seconds=1., simulations=2), focus='neutral')]*4, focuses={'0': 'neutral'},
            path=str(self.directory/'actor-0.jsonl.gz'),
            provenance=dict(purpose='training', kind='masked-ismcts-teacher', spec='frozen-run'))

    def publish(self, packet):
        self.packets.append(packet)
        records = []
        for offset, job in enumerate(packet['jobs']):
            episode = copy.deepcopy(game())
            episode['game_id'] = f'collection-{len(self.packets)}-{offset}'
            episode['setup_group'] = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=job['seed']))
            episode['split'] = split_for(episode['setup_group'])
            path = Path(packet['prefix']+f'-part-{packet["part_start"]+offset:04d}.jsonl.gz')
            proof = dict(purpose='training', collection=packet['identity'],
                jobs=[dict(index=job['index'], seed=job['seed'], provenance=job['provenance'])])
            if 'recipe' in packet: proof['recipe'] = packet['recipe']
            write_shard(path, [episode], proof, cache=False)
            records.append(dict(job=job['index'], game_id=episode['game_id'], path=str(path),
                sha256=sha(path), cached=False, features=None))
        self.pool.results.put(dict(records=records))

    def test_changed_policy_model_cannot_reuse_teacher_collection(self):
        job = self.job(); before = self.pool.collect([job])[0]
        changed = copy.deepcopy(job)
        for policy in changed['policies']: policy['model_id'] = 'b'*64
        after = self.pool.collect([changed])[0]
        self.assertFalse(after['cached'])
        self.assertNotEqual(before['path'], after['path'])

    def test_changed_focus_cannot_reuse_ordinary_collection(self):
        job = self.job(); job['provenance']['kind'] = 'selfplay'
        job['policies'] = [neural_spec('a'*64, focus='mixed')]*4
        self.pool.collect([job])
        changed = copy.deepcopy(job); changed['focuses']['0'] = 'coal'
        self.assertFalse(self.pool.collect([changed])[0]['cached'])

    def test_changed_rollout_opponents_cannot_reuse_teacher_collection(self):
        job = self.job(); self.pool.collect([job])
        changed = copy.deepcopy(job)
        for policy in changed['policies']: policy['opponents'] = ['a'*64, 'b'*64]
        self.assertFalse(self.pool.collect([changed])[0]['cached'])

    def test_sampling_rotation_and_execution_inputs_are_bound(self):
        job = self.job(); self.pool.collect([job])
        for field, value in [('sampling_seed', 1702), ('rotation', 1), ('rng_index', 1)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(job); changed[field] = value
                self.assertFalse(self.pool.collect([changed])[0]['cached'])
        self.pool.config['search_workers_per_turn'] = 4
        self.assertFalse(self.pool.collect([job])[0]['cached'])

    def test_exact_recipe_reuses_new_cache_and_preserves_raw_bytes(self):
        job = self.job(); before = self.pool.collect([job])[0]
        raw = Path(before['path']).read_bytes()
        after = self.pool.collect([copy.deepcopy(job)])[0]
        self.assertTrue(after['cached'])
        self.assertEqual(before['path'], after['path'])
        self.assertEqual(len(self.packets), 1)
        self.assertEqual(Path(before['path']).read_bytes(), raw)

    def test_effective_defaults_and_output_basename_do_not_change_recipe(self):
        job = self.job(); job.pop('rotation'); job.pop('focuses')
        before = self.pool.collect([job])[0]
        changed = dict(job, rotation=0, focuses={}, rng_index=0,
            path=str(self.directory/'relocated-output-name.jsonl.gz'))
        after = self.pool.collect([changed])[0]
        self.assertTrue(after['cached'])
        self.assertEqual(before['path'], after['path'])

    def test_legacy_cache_is_readable_but_not_execution_proof(self):
        job = self.job(); identity = digest([job['provenance']])
        self.publish(dict(jobs=[job], identity=identity,
            prefix=str(self.directory/('games-'+identity[:16])), part_start=0))
        legacy = self.pool.results.get()['records'][0]
        raw = Path(legacy['path']).read_bytes()
        self.assertEqual(len(read_shards([legacy['path']])), 1)
        fresh = self.pool.collect([job])[0]
        self.assertFalse(fresh['cached'])
        self.assertNotEqual(fresh['path'], legacy['path'])
        self.assertEqual(Path(legacy['path']).read_bytes(), raw)

    def test_changed_recipe_metadata_is_rejected(self):
        job = self.job(); record = self.pool.collect([job])[0]
        path = Path(record['path'])
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            header = json.loads(next(stream)); body = stream.read()
        # A collection digest cannot attest a different or absent recipe.
        header['provenance']['recipe'] = {'format': 'wrong'}
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            stream.write(json.dumps(header)+'\n'+body)
        with self.assertRaisesRegex(ValueError, 'recipe'):
            self.pool.collect([job])


if __name__ == '__main__': unittest.main()
