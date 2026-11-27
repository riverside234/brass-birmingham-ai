"""g6547: the soft policy weight 8 -> 4 through the reviewed g3423 weight change (--soft-policy-weight), on the
g4970 run's weight history, and the g4970 wrapper keeping the run's weight unless it is given."""
import copy
import unittest
from unittest import mock


def block():
    """The g6547 export's soft_policy block: 8 at g3423, 6 at g3471, 12 at g3528, 8 again at g4970."""
    from brass_ai.recovery import training_revision_g3423 as S
    return dict(copy.deepcopy(S.SETTINGS), from_generation=3423, weight=8., weight_change=dict(
        generation=4970, before=12., previous=dict(generation=3528, before=6., previous=dict(generation=3471, before=8.))))


class Weight(unittest.TestCase):
    def test_eight_to_four_at_g6547(self):
        from brass_ai.recovery import training_revision_g3423 as S
        config = {S.KEY: S.validate(block())}
        with mock.patch('brass_ai.recovery.config.validate_config', lambda c: c):
            four = S.set_weight(config, 4, 6547)
            self.assertEqual(S.set_weight(four, 4, 6557), four)  # kept by the next continuations
        b = four[S.KEY]
        self.assertEqual((b['weight'], b['weight_change']['generation'], b['weight_change']['before']), (4., 6547, 8.))
        self.assertEqual(b['weight_change']['previous'], config[S.KEY]['weight_change'])
        self.assertEqual(S.previous_settings(b), config[S.KEY])  # the committed head's stamp at g6547
        self.assertEqual(S.project(config, four), four); self.assertIsNone(S.project(four, config))
        self.assertEqual(S.tag(four), 'soft-t4-w4-s1')

    def test_the_g4970_wrapper_keeps_the_run_weight_unless_given(self):
        import ast
        import re
        from pathlib import Path
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.recovery import training_revision_g3423 as S
        text = (Path(SOURCE_ROOT) / 'scripts' / 'continue-rust-g4970.py').read_text()
        choices = re.search(r"'--soft-policy-weight', type=float, choices=(\([^)]*\))", text).group(1)
        self.assertEqual(ast.literal_eval(choices), S.WEIGHTS)
        self.assertIn("state['signature']['config']['soft_policy']['weight']", text)
        self.assertIn("'--soft-policy-weight', f'{weight:g}'", text)
        self.assertNotIn("'--soft-policy-weight', '8'", text)


if __name__ == '__main__':
    unittest.main()
