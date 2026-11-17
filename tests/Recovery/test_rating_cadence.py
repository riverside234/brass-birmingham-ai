"""g3170 rating cadence: PL ratings every 20 iterations from a committed generation."""
import copy
import unittest

from brass_ai.recovery import rating_cadence as R


def g3140():
    from test_training_g3140 import g2880
    from brass_ai.recovery import training_revision_g3140 as G
    return G.configure(g2880(), G.parse('time,validation'), 3140)


def signature(config):
    from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
    return dict(config=config, implementation=QUALIFIED)


class Cadence(unittest.TestCase):
    def test_slots_after_the_boundary_run_every_twenty(self):
        config = R.configure(g3140(), 20, 3170)
        self.assertEqual(config['rating'], g3140()['rating'])  # the rating block and its slot grid are unchanged
        due = [g for g in range(3100, 3260, 10) if R.due(config, g)]
        self.assertEqual(due, [3100, 3110, 3120, 3130, 3140, 3150, 3160, 3170, 3190, 3210, 3230, 3250])
        self.assertFalse(R.due(config, 3175)); self.assertTrue(R.due(g3140(), 3180))
        # Reference setups still rotate one position per rated slot.
        ordinals = [R.ordinal(config, g) for g in due]
        self.assertEqual(ordinals, list(range(310, 310+len(due))))
        self.assertEqual([R.ordinal(g3140(), g) for g in (3160, 3170, 3180)], [316, 317, 318])

    def test_recipe_is_recognized_and_only_adopted_once(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_shadow_config import collection_signature, is_shadow, signature_transition
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before = g3140(); after = validate_config(R.configure(before, 20, 3170))
        self.assertEqual(R.configure(after, 20, 3180), after)  # idempotent within one command
        self.assertEqual((is_shadow(after), is_on(after)), (is_shadow(before), is_on(before)))
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        changed = copy.deepcopy(after); changed[R.KEY]['from_generation'] = 3190
        self.assertFalse(signature_transition(signature(after), signature(changed)))
        # It changes no collection setting.
        if is_on(before):
            self.assertEqual(collection_signature(signature(before)), collection_signature(signature(after)))
        with self.assertRaisesRegex(ValueError, 'rated generation'):R.configure(before, 20, 3171)
        with self.assertRaisesRegex(ValueError, 'rating-every'):R.configure(before, 30, 3170)
        bad = copy.deepcopy(after); bad[R.KEY]['every_iterations'] = 15
        with self.assertRaisesRegex(ValueError, 'cadence'):validate_config(bad)


if __name__ == '__main__':
    unittest.main()
