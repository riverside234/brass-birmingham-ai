"""Lossless IL disk transport, actual ingress and unchanged dense training inputs."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, decode_json
from brass_ai.recovery import demonstrations as demos, il_replay_storage as storage
from brass_ai.recovery.replay import cache_receipt, read_shards, write_shard
from brass_ai.step5.league import write_json


def proof():
    return dict(purpose='training', kind='v3-demonstration', seed=1201, replay_format=storage.FORMAT)


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def toy_game():
    rows = []
    for i in range(4):
        view = dict(history=[dict(turn=j, choice='build') for j in range(i)],
                    players=[dict(seat=s, cash=10+i if s == 0 else 10) for s in range(4)],
                    slots=[dict(id='slot', tile=None if i < 2 else dict(level=2, flipped=i == 3))],
                    links=[dict(owner=None if i < 1 else 0)], merchants=[], own_hand=['Birmingham'])
        rows.append(dict(view=view, behavior=dict(kind='external', probability=.3), teacher=None,
                         chosen=0, choices=[dict(id='build', kind='build')], seat=i, step=i))
    return dict(game_id='synthetic', records=rows, result=dict(vp=[1, 2, 3, 4]))


class CompactCodecTests(unittest.TestCase):
    def test_exact_roundtrip_without_mutating_input(self):
        game = toy_game(); before = encoded(game)
        packed = storage.pack(game)
        restored = storage.unpack(decode_json(encoded(packed)))
        self.assertEqual(encoded(restored), before)
        self.assertEqual(encoded(game), before)
        self.assertEqual([r['view']['history'] for r in packed['game']['records']], [0, 1, 2, 3])

    def test_nonprefix_history_is_preserved_inline(self):
        game = toy_game(); game['records'][1]['view']['history'] = [dict(private='different observer')]
        packed = storage.pack(game)
        self.assertIsInstance(packed['game']['records'][1]['view']['history'], list)
        self.assertEqual(encoded(storage.unpack(decode_json(encoded(packed)))), encoded(game))

    def test_shape_type_key_order_signed_zero_and_missing_field_changes(self):
        game = toy_game()
        values = [dict(a=[0, True, -0.0], b=1), dict(b=1.0, a=[False, 1, 0.0]), [], [None]]
        for row, value in zip(game['records'], values):
            row['view']['players'] = value
        del game['records'][1]['view']['links']
        self.assertEqual(encoded(storage.unpack(decode_json(encoded(storage.pack(game))))), encoded(game))

    def test_decoded_rows_do_not_alias_other_rows_or_payload(self):
        payload = decode_json(encoded(storage.pack(toy_game()))); before = encoded(payload)
        game = storage.unpack(payload)
        game['records'][0]['view']['players'][1]['cash'] = 999
        game['records'][1]['view']['history'][0]['choice'] = 'changed'
        self.assertEqual(game['records'][1]['view']['players'][1]['cash'], 10)
        self.assertEqual(game['records'][2]['view']['history'][0]['choice'], 'build')
        self.assertEqual(encoded(payload), before)

    def test_malformed_prefix_delta_and_duplicate_indices_rejected(self):
        for field, edit in [('history', -1), ('history', 999), ('history', True),
                            ('players', None), ('players', [9, []]), ('players', [True, []]),
                            ('players', [2, [[0, [0, 1]], [0, [0, 2]]]]),
                            ('players', [2, [[999, [0, 1]]]]), ('players', [1, {'missing': [0, 0]}])]:
            with self.subTest(field=field, edit=edit):
                value = decode_json(encoded(storage.pack(toy_game())))
                index = 0 if edit is None else 1
                value['game']['records'][index]['view'][field] = edit
                with self.assertRaises(ValueError):storage.unpack(value)
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            decode_json('{"history":[],"history":[],"game":{}}')

    def test_neural_teacher_and_branch_data_are_not_compacted(self):
        for change in ('neural', 'teacher', 'branch'):
            game = toy_game()
            if change == 'neural':game['records'][0]['behavior']['kind'] = 'model'
            elif change == 'teacher':game['records'][0]['teacher'] = dict(source='search')
            else:game['branches'] = []
            with self.assertRaises(ValueError):storage.pack(game)

    def test_storage_requires_explicit_consistent_il_provenance(self):
        self.assertEqual(storage.storage_format(dict(purpose='training', kind='selfplay')), storage.LEGACY_FORMAT)
        self.assertEqual(storage.storage_format(proof()), storage.FORMAT)
        grouped = dict(purpose='training', jobs=[dict(provenance=proof())]*2)
        self.assertEqual(storage.storage_format(grouped), storage.FORMAT)
        for invalid in [dict(proof(), kind='selfplay'), dict(proof(), purpose='evaluation'),
                        dict(proof(), replay_format='unknown'), dict(grouped, jobs=[]),
                        dict(grouped, jobs=[dict(provenance=proof()), dict(provenance={})])]:
            with self.subTest(proof=invalid), self.assertRaises(ValueError):storage.storage_format(invalid)


class CompactIngressTests(unittest.TestCase):
    def setUp(self):
        from review_fixture import current_game
        self.game = current_game()
        for row in self.game['records']:row['behavior']['kind'] = 'external'
        self.temp = tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write_pair(self, cache=True):
        paths = [self.directory/'legacy.jsonl.gz', self.directory/'compact.jsonl.gz']
        write_shard(paths[0], [self.game], dict(purpose='training'), cache=cache)
        write_shard(paths[1], [self.game], proof(), cache=cache)
        return paths

    def test_legacy_and_compact_ingress_dense_features_targets_and_receipts_match(self):
        paths = self.write_pair()
        tasks = []
        for path in paths:
            receipt = cache_receipt(path)
            self.assertEqual(receipt['replay_sha256'], sha(path))
            game = read_shards([path], heuristic_only=True, receipts={str(path): receipt})[0]
            self.assertEqual(encoded(game), encoded(self.game))
            entry = dict(sha256=sha(path), game_id=game['game_id'], features=receipt,
                         teachers={p:1 for p in game['opponents']})
            tasks.append((str(path), [entry], game['split'], set(game['opponents']), True))
        left, right = [demos._prepared_shard(task) for task in tasks]
        for a, b in zip(left, right):
            self.assertEqual(a[:3], b[:3])
            for field in ('state', 'actions', 'stable_to_relative'):
                np.testing.assert_array_equal(getattr(a[3], field), getattr(b[3], field))
        path = paths[1]; path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):demos._prepared_shard(tasks[1])

    def test_decoded_inputs_still_receive_validation_and_bad_header_fails(self):
        path = self.write_pair(cache=False)[1]
        with gzip.open(path, 'rt', encoding='utf-8') as stream:header, payload = map(decode_json, stream)
        payload['game']['records'][0]['chosen'] = 100000
        def save():
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                stream.write(encoded(header)+'\n'+encoded(payload)+'\n')
        save()
        with self.assertRaisesRegex(ValueError, 'Illegal chosen index'):read_shards([path], heuristic_only=True)
        header['provenance'].pop('replay_format'); save()
        with self.assertRaisesRegex(ValueError, 'pinned storage provenance'):read_shards([path], heuristic_only=True)
        header['format'] = storage.FORMAT.replace('1.0.0', '999.0.0'); save()
        with self.assertRaisesRegex(ValueError, 'Unsupported replay format'):read_shards([path], heuristic_only=True)

    def test_rejects_overwrite_and_separates_games(self):
        second = copy.deepcopy(self.game); second['game_id'] = 'second'
        path = self.directory/'multi.jsonl.gz'
        write_shard(path, [self.game, second], proof(), cache=False)
        self.assertEqual([g['game_id'] for g in read_shards([path])], [self.game['game_id'], 'second'])
        before = sha(path)
        with self.assertRaisesRegex(ValueError, 'Existing or empty'):write_shard(path, [self.game], proof())
        self.assertEqual(sha(path), before)

    def test_failed_compaction_does_not_publish_partial_file(self):
        path = self.directory/'failure.jsonl.gz'
        with patch.object(storage, 'pack', side_effect=ValueError('injected codec failure')):
            with self.assertRaisesRegex(ValueError, 'injected'):write_shard(path, [self.game], proof())
        self.assertFalse(path.exists())
        self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_parallel_disk_preparation_and_warm_cache_hit(self):
        from brass_ai.recovery import prepared_store
        tasks = []
        for i in range(2):
            game = copy.deepcopy(self.game); game['game_id'] += '-'+str(i)
            path = self.directory/f'{i}.jsonl.gz'
            write_shard(path, [game], proof())
            entry = dict(sha256=sha(path), game_id=game['game_id'], features=cache_receipt(path))
            tasks.append((str(path), [entry], game['split'], set(game['opponents']), True))
        directory = self.directory/'prepared'
        first = prepared_store.prepare(tasks, directory, 2, self.game['split'])
        self.addCleanup(first.close)
        second = prepared_store.prepare(tasks, directory, 2, self.game['split'])
        self.addCleanup(second.close)
        self.assertEqual(len(first), len(second))
        # A valid prepared hit verifies source/output hashes but never expands
        # a raw replay or calls the codec again during minibatch reads.
        with patch.object(storage, 'unpack', side_effect=AssertionError('unexpected expansion')):
            for task in tasks:
                key = prepared_store.task_key(task, prepared_store.preparation_identity())
                _, hit = prepared_store._prepare_job((task, str(directory), key))
                self.assertTrue(hit)
            for a, b in zip(first, second):
                for field in ('state', 'actions', 'stable_to_relative'):
                    np.testing.assert_array_equal(getattr(a[3], field), getattr(b[3], field))

    def test_generation_defaults_pin_compact_format_and_retry_does_not_rewrite(self):
        # Exercise real generation planning/publication with frozen synthetic
        # games, without spending time on new teacher searches.
        out = self.directory/'new-il'; actor = Mock()
        def collect(jobs):
            results = []
            for job in jobs:
                self.assertEqual(job['provenance']['replay_format'], storage.FORMAT)
                path = Path(job['path'])
                cached = path.exists()
                if not cached:write_shard(path, [self.game], job['provenance'], cache=False)
                results.append(dict(path=str(path), sha256=sha(path), cached=cached, features=None))
            return results
        actor.collect.side_effect = collect
        cm = Mock(); cm.__enter__ = Mock(return_value=actor); cm.__exit__ = Mock(return_value=False)
        service = Mock(); service.__enter__ = Mock(return_value=Mock()); service.__exit__ = Mock(return_value=False)
        with patch.object(demos, 'execution_config'), patch.object(demos, 'admission'), \
             patch.object(demos, 'ActorPool', return_value=cm), patch.object(demos, 'InferenceService', return_value=service), \
             patch('brass_ai.recovery.legacy_search.legacy_spec', return_value=dict(policy_id='fixture')):
            first = demos.generate(SOURCE_ROOT, out, 1, {})
            again = demos.generate(SOURCE_ROOT, out, 1, {})
            self.assertEqual(first['replays'][0]['sha256'], again['replays'][0]['sha256'])
            self.assertTrue(again['replays'][0]['cached'])
            self.assertEqual(first['replay_storage']['bytes'], Path(first['replays'][0]['path']).stat().st_size)
            spec = json.loads((out/'spec.json').read_text()); spec['replay_format'] = storage.LEGACY_FORMAT
            write_json(out/'spec.json', spec)
            with self.assertRaisesRegex(ValueError, 'spec changed'):demos.generate(SOURCE_ROOT, out, 1, {})

    def test_feature_reaudit_can_read_compact_source_without_rewriting_it(self):
        from brass_ai.feature_engineering.data import convert
        path = self.write_pair(cache=False)[1]; before = sha(path)
        entry = dict(path=path.relative_to(SOURCE_ROOT).as_posix(), sha256=before,
                     game_id=self.game['game_id'], split=self.game['split'], setup_group=self.game['setup_group'],
                     semantic_sha256='a'*64, transitions_verified=True)
        manifest = self.directory/'manifest.json'
        write_json(manifest, dict(format='bc-dataset/1.0.0', compatibility=compatibility(),
            label_kind='qualified-v3-choice/1.0.0', context='neutral-predecision/1.0.0',
            teachers=self.game['opponents'], entries=[entry]))
        def audited(args):
            target = Path(args[1])
            self.assertEqual(encoded(read_shards([target], heuristic_only=True)[0]), encoded(self.game))
            return [dict(entry, path=target.relative_to(SOURCE_ROOT).as_posix(), sha256=sha(target))], []
        with patch.object(demos, '_audit_path', side_effect=audited) as audit:
            result = convert(SOURCE_ROOT, manifest, self.directory/'converted')
        audit.assert_called_once()
        self.assertEqual(result['games'], 1)
        self.assertEqual(sha(path), before)


if __name__ == '__main__':unittest.main()
