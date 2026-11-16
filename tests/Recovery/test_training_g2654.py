"""g2654: Rail-era cotton builds below level three are not targeted (PPO and teacher)."""
import copy
import unittest

from brass_ai.recovery import ordinary_cotton as C, teacher_cotton as T, training_revision_g1491 as R
from brass_ai.recovery import teacher_search_retune as retune, training_revision_g2654 as G
from brass_ai.step5.policy import network_cotton_candidates


def g2600(weight=1):
    from test_training_g2600 import g2400
    return R.configure_teacher_games(retune.configure(g2400(), g2600=True), weight)


def view(era, level):
    seat = 0
    tiles = [dict(level=level), dict(level=level + 1)]
    return dict(seat=seat, context=dict(phase='action', era=era),
                players=[dict(mat=[dict(type='CottonMill', tiles=tiles)]) for _ in range(4)],
                slots=[dict(id='A:0', types=['CottonMill'])], links=[dict(id='AB', regions=['A', 'B'])])


CHOICES = [dict(id='net', kind='network', target='AB'), dict(id='build', kind='build', industry='CottonMill'),
           dict(id='dev', kind='develop', industry='CottonMill'), dict(id='loan', kind='loan')]


class Rule(unittest.TestCase):
    def test_rail_level_two_builds_leave_the_candidates(self):
        for era, level, minimum, build in (('CanalEra', 1, 2, False), ('CanalEra', 2, 2, True), ('RailEra', 2, 2, True),
                                           ('RailEra', 2, 3, False), ('RailEra', 3, 3, True), ('CanalEra', 2, 3, True)):
            rail_minimum = minimum if era == 'RailEra' else 2
            got = network_cotton_candidates(view(era, level), CHOICES, minimum_build_level=rail_minimum)
            self.assertEqual(1 in got, build, (era, level, minimum))
            self.assertIn(0, got); self.assertIn(2, got)

    def test_ppo_tables_use_the_rule_only_under_the_new_contract(self):
        def table(settings):
            i = next(i for i in range(40) if C.planned_assignment(settings, i, 4400))
            proof = dict(kind='selfplay', purpose='training', ordinary_game_index=i, sampling_seed=4400,
                         **{C.KEY: C.planned_assignment(settings, i, 4400)})
            return C.TableExploration(proof)
        old, new = table(C.NETWORK), table(C.RAIL_NETWORK)
        for actor, rail_build in ((old, True), (new, False)):
            v = view('RailEra', 2); v['seat'] = actor.seats[0]
            self.assertEqual(1 in actor.targets(v, CHOICES), rail_build)
            v['context']['era'] = 'CanalEra'
            self.assertIn(1, actor.targets(v, CHOICES))
        self.assertEqual(C.RAIL_NETWORK, dict(C.NETWORK, contract=C.RAIL_NETWORK_CONTRACT, rail_minimum_build_level=3))

    def test_teacher_settings_match_the_native_contract(self):
        settings = T.settings(contract=T.RAIL_NETWORK_CONTRACT)
        self.assertEqual(settings, dict(T.settings(contract=T.NETWORK_CONTRACT), contract=T.RAIL_NETWORK_CONTRACT,
                                        rail_minimum_build_level=3))
        self.assertEqual(T.PROBABILITIES[T.RAIL_NETWORK_CONTRACT], .3)


class Recipe(unittest.TestCase):
    def test_switch_keeps_labels_and_allows_the_w0_blend(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before = g2600(1)
        after = R.configure_teacher_games(G.configure(before), 0)
        validate_config(after)
        self.assertTrue(G.applied(after) and is_on(after))
        self.assertIs(G.configure(after), after)
        self.assertEqual(buffer_identity(before), buffer_identity(after))
        self.assertEqual(after['table_plan'][C.KEY], C.RAIL_NETWORK)
        self.assertEqual(R.cotton_settings(after)['table_fraction'], R.cotton_settings(before)['table_fraction'])
        sig = lambda c: dict(config=c, implementation=QUALIFIED)
        self.assertTrue(signature_transition(sig(before), sig(after)))
        self.assertFalse(signature_transition(sig(after), sig(before)))
        with self.assertRaisesRegex(ValueError, 'g2600 cotton-network'):
            from test_training_g2600 import g2400
            G.configure(g2400())
        state = dict(iteration=2654, signature=dict(config=after), search_value=dict(stage='C', w=1.))
        jobs, _ = plan_jobs(after, state, model_id='m', export_hash='e', epoch=2654, ready=True)
        cotton = [j for j in jobs if T.enabled(j)]
        self.assertTrue(cotton)
        for job in jobs:
            R.validate_job(job)
            self.assertEqual(job['search_value']['w'], 0.)
        for job in cotton:
            self.assertEqual(T.job_settings(job)['contract'], T.RAIL_NETWORK_CONTRACT)
        changed = copy.deepcopy(cotton[0]); changed['teacher_revision']['exploration']['cotton']['rail_minimum_build_level'] = 2
        with self.assertRaises(ValueError):
            R.validate_job(changed)


if __name__ == '__main__':
    unittest.main()
