"""Document the accepted JSON-equivalent list/tuple cache-validation gap.

Uncached schema validation remains strict. Warm hashes, features and replay seals
intentionally do not distinguish container types with identical JSON content.
"""
import copy
import unittest
from unittest.mock import patch

from fixtures import game, position
from brass_ai.learning.contracts import digest
from brass_ai.recovery import encoding, inputs, replay
from brass_ai.runtime_contracts import validate


class CacheTypeReviewTests(unittest.TestCase):
    def fixture(self):
        view, choices = position()
        choices.append(dict(id='scout', kind='scout', card=view['own_hand'][0], cards=view['own_hand'][1:3]))
        view['history'] = [dict(seat=0, era='CanalEra', choice=copy.deepcopy(choices[-1])) for _ in range(65)]
        validate('observation', view); validate('choices', choices)
        return view, choices

    def mutations(self):
        return [
            lambda v,c: v.update(own_hand=tuple(v['own_hand'])),
            lambda v,c: v.update(history=tuple(v['history'])),
            lambda v,c: v['players'][0].update(discards=tuple(v['players'][0]['discards'])),
            lambda v,c: v['players'][0]['mat'][0].update(tiles=tuple(v['players'][0]['mat'][0]['tiles'])),
            lambda v,c: v['links'][0].update(regions=tuple(v['links'][0]['regions'])),
            lambda v,c: c[-1].update(cards=tuple(c[-1]['cards'])),
            lambda v,c: v['history'][32]['choice'].update(cards=tuple(v['history'][32]['choice']['cards'])),
        ]

    def test_warm_frame_accepts_known_json_equivalent_container_gap(self):
        view, choices = self.fixture(); boundary = inputs.InputBoundary()
        expected = boundary.check(view, choices)
        for mutate in self.mutations():
            bad, bad_choices = copy.deepcopy((view, choices)); mutate(bad, bad_choices)
            self.assertEqual(digest(dict(view=bad, choices=bad_choices)), expected)
            with self.subTest(mutation=mutate), self.assertRaises(Exception):
                validate('observation', bad); validate('choices', bad_choices)
            before = boundary.report()
            with self.subTest(mutation=mutate): self.assertEqual(boundary.check(bad, bad_choices), expected)
            self.assertEqual(boundary.report()['frame_cache_hits'], before['frame_cache_hits'] + 1)
        self.assertEqual(boundary.check(view, tuple(choices)), expected)
        self.assertEqual(boundary.check(view, choices), expected)

    def test_uncached_schema_rejects_malformed_containers(self):
        # No earlier equivalent cache entry: the original validators still
        # reject tuples, including nested choices and history events.
        view, choices = self.fixture()
        for mutate in self.mutations():
            bad, bad_choices = copy.deepcopy((view, choices)); mutate(bad, bad_choices)
            with self.subTest(mutation=mutate), self.assertRaises(Exception):
                inputs.InputBoundary(frames=0, parts=0, events=0).check(bad, bad_choices)
        with self.assertRaises(Exception): inputs.InputBoundary().check(view, tuple(choices))

    def test_frame_miss_retains_outer_checks_and_known_nested_cache_gap(self):
        for parts in (16384, 0):
            view, choices = self.fixture(); boundary = inputs.InputBoundary(frames=0, parts=parts)
            boundary.check(view, choices)
            for index, mutate in enumerate(self.mutations()):
                bad, bad_choices = copy.deepcopy((view, choices)); mutate(bad, bad_choices)
                bad['context']['revision'] += 1
                with self.subTest(parts=parts, mutation=index):
                    if index < 2 or (parts == 0 and index < 6):
                        with self.assertRaises(Exception): boundary.check(bad, bad_choices)
                    else:
                        self.assertEqual(boundary.check(bad, bad_choices), digest(dict(view=bad, choices=bad_choices)))

    def test_feature_cache_accepts_known_container_gap_on_both_ingress_paths(self):
        for validated in (False, True):
            view, choices = self.fixture(); encoder = encoding.PreparedEncoder()
            with patch.object(encoding, 'BOUNDARY', inputs.InputBoundary()):
                expected = encoder.encode(view, choices)
                view['players'][0]['mat'][0]['tiles'] = tuple(view['players'][0]['mat'][0]['tiles'])
                with self.subTest(validated=validated):
                    self.assertIs(encoder.encode(view, choices, validated=validated), expected)

    def test_attested_replay_accepts_known_json_equivalent_input_gap(self):
        checked = replay.check_game(game()); row = checked['records'][0]
        before = replay.row_key(row)
        row['choices'] = tuple(row['choices'])
        self.assertEqual(digest(dict(view=row['view'], choices=row['choices'])), checked.seals[(0,0)]['input'])
        checked.encoded(row, encoding.PreparedEncoder())
        self.assertEqual(replay.row_key(row), before)
        row['choices'] = list(row['choices'])
        self.assertEqual(replay.row_key(row), before)
        checked.encoded(row, encoding.PreparedEncoder())


if __name__ == '__main__': unittest.main()
