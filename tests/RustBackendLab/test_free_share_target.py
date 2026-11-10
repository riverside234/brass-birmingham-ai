import copy
import unittest

from brass_ai.recovery import teacher_gradient_qualification as Q


class FreeShareTarget(unittest.TestCase):
    def test_any_share_above_zero_and_at_most_one_is_a_valid_target(self):
        for target in (.05, .1, .25, .33, .4, .5, .75, 1, 1.):
            self.assertTrue(Q.valid_target(target), target)
        for target in (0, 0., -.1, 1.01, float('nan'), float('inf'), True, '.2', None):
            self.assertFalse(Q.valid_target(target), repr(target))

    def passed(self, target, raw=.2):
        observations = [dict(generation=g) for g in range(10, 14)]
        return Q.seal(dict(contract=Q.CONTRACT, passed=True, target=target, observations=observations, required_iterations=4,
                           raw_median=raw, beta=Q.fixed_beta(target, raw), qualified_generation=14, clamped=False,
                           reason='ok', scope='s'))

    def test_a_qualified_weight_follows_a_new_target_through_the_measured_median(self):
        receipt = self.passed(.25)
        self.assertEqual(Q.restore(receipt, .25), receipt)
        changed = Q.restore(receipt, .1)
        self.assertEqual(changed['target'], .1)
        self.assertEqual(changed['raw_median'], .2)
        self.assertAlmostEqual(changed['beta'], .5)
        self.assertEqual(changed, Q.seal(changed))
        self.assertEqual(Q.restore(changed, .1), changed)

    def test_a_saved_receipt_with_a_free_target_restores(self):
        receipt = self.passed(.33)
        self.assertEqual(Q.restore(receipt, .33), receipt)
        bad = copy.deepcopy(receipt)
        bad['target'] = 1.5
        with self.assertRaises(ValueError):
            Q.restore(Q.seal(bad), 1.5)



if __name__ == '__main__':
    unittest.main()
