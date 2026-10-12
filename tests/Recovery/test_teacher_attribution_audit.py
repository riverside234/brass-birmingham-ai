"""Search labels must bind to the replay row even with attested feature arrays."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fixtures import game
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility
from brass_ai.learning.replay import validate_teacher
from brass_ai.learning.reward import specification
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.replay import (cache_receipt, check_game, read_shards,
                                      write_cache, write_shard)
from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import visit_teacher


class TeacherAttributionAuditTests(unittest.TestCase):
    def teacher_game(self, combined):
        replay = game()
        row = replay['records'][0]
        config = SearchConfig(contract='masked-ismcts/1.2.0', reward=specification()) if combined else SearchConfig()
        stats = dict(root={c['id']: dict(visits=n) for c, n in zip(row['choices'], (4, 2, 1))},
                     expansion=[True] * 3)
        with patch('brass_ai.step5.league.implementation_id', return_value='b' * 64):
            row['teacher'] = visit_teacher(row['choices'], stats, [.6, .3, .1],
                model='a' * 64, observer=row['seat'], config=config)
        return replay

    def corrupt(self, replay, field):
        replay = copy.deepcopy(replay)
        teacher = replay['records'][0]['teacher']
        if field == 'observer':
            teacher['observer'] = 1
        else:
            teacher['legal_ids'].reverse()
        # Each teacher is valid in isolation; only its attachment to this row is wrong.
        validate_teacher(teacher, 3)
        return replay

    def write_raw(self, path, replay):
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(format='four-seat-replay/1.0.0',
                compatibility=compatibility(), provenance=dict(purpose='training'))) + '\n')
            stream.write(json.dumps(replay) + '\n')

    def test_raw_shard_rejects_teacher_order_and_observer(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
            for combined in (False, True):
                for field in ('observer', 'legal_ids'):
                    with self.subTest(combined=combined, field=field):
                        path = Path(directory) / 'raw.jsonl.gz'
                        self.write_raw(path, self.corrupt(self.teacher_game(combined), field))
                        with self.assertRaisesRegex(ValueError, 'Teacher action order/observer mismatch'):
                            read_shards([path])

    def test_attested_cache_rejects_teacher_order_and_observer_before_sealing(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
            for combined in (False, True):
                valid = check_game(self.teacher_game(combined))
                for field in ('observer', 'legal_ids'):
                    with self.subTest(combined=combined, field=field):
                        path = Path(directory) / 'cached.jsonl.gz'
                        self.write_raw(path, self.corrupt(valid, field))
                        # Features remain exact when only labels change. Even an independent
                        # receipt for those arrays cannot attest row/teacher attribution.
                        write_cache(path, [valid])
                        receipt = cache_receipt(path)
                        with patch('brass_ai.recovery.replay.BOUNDARY.check',
                                   side_effect=AssertionError('Attested arrays were not reused')):
                            with self.assertRaisesRegex(ValueError, 'Teacher action order/observer mismatch'):
                                read_shards([path], receipts={str(path): receipt})

    def test_supported_teachers_roundtrip_and_remain_sealed(self):
        cases = [game(), self.teacher_game(False), self.teacher_game(True), game()]
        cases[-1]['records'][0]['teacher'] = dict(source='fresh-finalist/1.0.0',
            probabilities=[1., 0., 0.], coverage=[True, False, False],
            confidence=1., samples=1, horizon='fixture')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
            for index, replay in enumerate(cases):
                with self.subTest(source=replay['records'][0]['teacher']):
                    path = Path(directory) / f'valid-{index}.jsonl.gz'
                    write_shard(path, [replay], dict(purpose='training'))
                    for receipts in (None, {str(path): cache_receipt(path)}):
                        checked = read_shards([path], receipts=receipts)[0]
                        self.assertEqual(checked, replay)
                        row = checked['records'][0]
                        checked.encoded(row, PreparedEncoder())
                        row['seconds'] += 1
                        with self.assertRaisesRegex(ValueError, 'Validated replay record changed'):
                            checked.encoded(row, PreparedEncoder())


if __name__ == '__main__':
    unittest.main()
