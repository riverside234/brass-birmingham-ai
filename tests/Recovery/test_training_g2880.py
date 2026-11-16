"""g2880: search-value returns use lambda .95; buffered games are retargeted, not retired."""
import copy
import unittest

import numpy as np

from brass_ai.recovery import training_revision_g2654 as rail, training_revision_g2880 as G


def g2654():
    from test_training_g2654 import g2600
    return rail.configure(g2600(0))


class Recipe(unittest.TestCase):
    def test_switch_validates_and_cannot_revert(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        before = g2654()
        self.assertEqual(before['teacher_phase']['search_value']['target']['lambda'], .8)
        after = validate_config(G.configure(before))
        self.assertTrue(G.applied(after))
        self.assertIs(G.configure(after), after)
        self.assertEqual(after['teacher_phase']['search_value']['target']['lambda'], .95)
        changed = copy.deepcopy(after); changed['teacher_phase']['search_value']['target']['lambda'] = .8
        self.assertEqual(changed, before)
        self.assertEqual(buffer_identity(before), buffer_identity(after))
        sig = lambda c: dict(config=c, implementation=QUALIFIED)
        self.assertTrue(signature_transition(sig(before), sig(after)))
        self.assertFalse(signature_transition(sig(after), sig(before)))
        with self.assertRaisesRegex(ValueError, 'g2654'):
            from test_training_g2654 import g2600
            G.configure(g2600(0))

    def test_committed_stage_moves_to_95_only(self):
        from brass_ai.recovery.search_value_training import StageMachine
        options = G.configure(g2654())['teacher_phase']['search_value']
        saved = StageMachine(options=g2654()['teacher_phase']['search_value']).state
        self.assertEqual(saved['lam'], .8)
        machine = StageMachine(saved=copy.deepcopy(saved), options=options)
        self.assertEqual(machine.state['lam'], .95)
        with self.assertRaisesRegex(ValueError, 'buffer retirement'):
            machine.choose_lambda(.8)

    def test_buffered_games_are_retargeted_at_the_current_lambda(self):
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        rng = np.random.default_rng(2880)
        game = [dict(game_id='g', index=i, u=rng.normal(size=4).tolist(), r=rng.normal(size=4).tolist(),
                     terminal_utility=[1., 0., 0., -1.], target=[9.] * 4, target_lambda=.8) for i in range(1, 5)]
        phase = SearchValuePhase.__new__(SearchValuePhase)
        targets = {}
        for lam in (.8, .95):
            phase.machine = type('M', (), dict(state=dict(lam=lam)))()
            rows = phase._targets(copy.deepcopy(game))
            self.assertTrue(all(r['target_lambda'] == lam for r in rows))
            targets[lam] = np.asarray([r['target'] for r in rows])
        self.assertFalse(np.allclose(targets[.8], targets[.95]))
        # The last turn bootstraps to the realized outcome under either lambda.
        np.testing.assert_allclose(targets[.8][-1], targets[.95][-1])


if __name__ == '__main__':
    unittest.main()
