"""g3528: cotton exploration retimed within each era (PPO and teacher cotton tables)."""
import copy
import unittest

from brass_ai.recovery import ordinary_cotton as C, teacher_cotton as T, training_revision_g1491 as R
from brass_ai.recovery import training_revision_g2654 as G2654, training_revision_g3471 as G3471
from brass_ai.recovery import training_revision_g3528 as G
from brass_ai.step5.policy import COTTON_TIMINGS, cotton_permissions, cotton_timing, era_action
from test_cotton_timing_g3471 import CHOICES, table, view

# (era, round, actions remaining) -> (the seat's action in the era, explores, cotton develop, cotton-city network)
ACTIONS = [
    ('CanalEra', 1, 1, 1, True, True, True), ('CanalEra', 2, 2, 2, True, True, True), ('CanalEra', 7, 1, 13, True, True, True),
    ('CanalEra', 8, 2, 14, True, True, False), ('CanalEra', 8, 1, 15, False, False, False),
    ('RailEra', 1, 2, 1, False, False, False), ('RailEra', 1, 1, 2, False, False, False), ('RailEra', 2, 2, 3, True, True, True),
    ('RailEra', 7, 1, 14, True, True, True), ('RailEra', 8, 2, 15, True, False, False), ('RailEra', 8, 1, 16, False, False, False),
]


class Rule(unittest.TestCase):
    def test_canal_one_explores_everything_and_canal_fourteen_develops_too(self):
        for era, round_, left, index, explores, develop, network in ACTIONS:
            context = view(era, round_, left)['context']
            self.assertEqual(era_action(context), index)
            self.assertEqual(cotton_permissions(context, COTTON_TIMINGS['g3528']), (explores, develop, network), (era, round_, left))
        # Only Canal 1 and Canal 14 differ from g3471, whose rule is unchanged.
        changed = [(era, round_, left) for era, round_, left, _, explores, develop, network in ACTIONS
                   if cotton_permissions(view(era, round_, left)['context'], COTTON_TIMINGS['g3471']) != (explores, develop, network)]
        self.assertEqual(changed, [('CanalEra', 1, 1), ('CanalEra', 8, 2)])
        self.assertEqual(cotton_timing(view('CanalEra', 1, 1)['context']), (False, False))

    def test_ppo_cotton_tables_follow_the_retimed_rule_under_the_new_contract(self):
        retimed, timed = table(C.RETIMED), table(C.TIMED)
        for era, round_, left, _, explores, develop, network in ACTIONS:
            for phase in ('action', 'continuation'):
                v = view(era, round_, left, seat=retimed.seats[0], phase=phase)
                want = [i for i, on in ((0, explores and phase == 'action'), (1, explores and develop), (2, explores and network)) if on]
                self.assertEqual(retimed.targets(v, CHOICES), want, (era, round_, left, phase))
        v = view('CanalEra', 1, 1, seat=timed.seats[0])
        self.assertEqual(timed.targets(v, CHOICES), [])  # a g3471 table keeps its own rule
        self.assertEqual(C.RETIMED, dict(C.RAIL_NETWORK, contract=C.RETIMED_CONTRACT,
                                         quiet_actions=dict(CanalEra=[15], RailEra=[1, 2, 16]),
                                         no_develop_actions=dict(CanalEra=[15], RailEra=[15, 16]),
                                         no_network_actions=dict(CanalEra=[14, 15], RailEra=[15, 16])))
        self.assertIn(C.RETIMED_CONTRACT, C.NETWORK_CONTRACTS)
        self.assertEqual(retimed.rail_build_level, C.RAIL_MINIMUM_BUILD_LEVEL)

    def test_teacher_labels_follow_the_retimed_rule_and_the_settings_match_the_native_mode(self):
        settings = T.settings(contract=T.RETIMED_NETWORK_CONTRACT)
        self.assertEqual(settings, dict(T.settings(contract=T.RAIL_NETWORK_CONTRACT), contract=T.RETIMED_NETWORK_CONTRACT,
                                        quiet_actions=dict(CanalEra=[15], RailEra=[1, 2, 16]),
                                        no_develop_actions=dict(CanalEra=[15], RailEra=[15, 16]),
                                        no_network_actions=dict(CanalEra=[14, 15], RailEra=[15, 16])))
        counts = [6, 3, 1, 0]

        def label(contract, cotton_visits, covered):  # the planned and played cotton build, chosen by the exploration
            probability = .7 + .3 * counts[0] / cotton_visits
            return dict(seat=0, counts=counts, played_action='build', explored=True, cotton_exploration=dict(
                contract=contract, targeted=True, probability=probability, used_before=0, network_used_before=0,
                cotton_visits=cotton_visits, total_visits=10, covered_locations=covered, planned_action='build'))
        new, old = T.RETIMED_NETWORK_CONTRACT, T.TIMED_NETWORK_CONTRACT
        T.validate_label(label(new, 10, 3), CHOICES, view=view('CanalEra', 1, 1))  # Canal 1: build, develop and network
        with self.assertRaisesRegex(ValueError, 'quiet action'):
            T.validate_label(label(old, 10, 3), CHOICES, view=view('CanalEra', 1, 1))  # g3471 labels keep their rule
        T.validate_label(label(new, 9, 2), CHOICES, view=view('CanalEra', 8, 2))   # Canal 14: build and develop
        with self.assertRaisesRegex(ValueError, 'coverage'):
            T.validate_label(label(new, 10, 3), CHOICES, view=view('CanalEra', 8, 2))  # no network at Canal 14
        with self.assertRaisesRegex(ValueError, 'coverage'):
            T.validate_label(label(old, 9, 2), CHOICES, view=view('CanalEra', 8, 2))   # g3471: builds only there
        T.validate_label(label(new, 6, 1), CHOICES, view=view('RailEra', 8, 2))    # Rail 15: builds only
        for era, round_, left in (('CanalEra', 8, 1), ('RailEra', 1, 2), ('RailEra', 1, 1), ('RailEra', 8, 1)):
            with self.assertRaisesRegex(ValueError, 'quiet action'):
                T.validate_label(label(new, 6, 1), CHOICES, view=view(era, round_, left))


class Recipe(unittest.TestCase):
    def test_the_retiming_follows_the_timed_or_the_rail_cotton_recipe(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        from test_training_g2654 import g2600
        rail = G2654.configure(g2600(1)); timed = G3471.configure(rail)
        after = G.configure(timed)
        validate_config(after)
        self.assertEqual(G.configure(rail), after)  # straight from g2654 too
        self.assertTrue(G.applied(after) and G2654.applied(after) and is_on(after))
        self.assertFalse(G3471.applied(after))
        self.assertIs(G.configure(after), after)
        self.assertEqual(buffer_identity(timed), buffer_identity(after))
        self.assertEqual(after['table_plan'][C.KEY], C.RETIMED)
        self.assertEqual(R.cotton_settings(after)['table_fraction'], R.cotton_settings(timed)['table_fraction'])
        self.assertEqual({k for k in after if after[k] != timed[k]}, {'table_plan', 'teacher_phase'})
        self.assertEqual(G.summary(after)['no_network_actions'], dict(CanalEra=[14, 15], RailEra=[15, 16]))
        with self.assertRaisesRegex(ValueError, 'g2654'):
            G.configure(g2600(1))
        state = dict(iteration=3528, signature=dict(config=after), search_value=dict(stage='C', w=1.))
        jobs, _ = plan_jobs(after, state, model_id='m', export_hash='e', epoch=3528, ready=True)
        cotton = [j for j in jobs if T.enabled(j)]
        self.assertTrue(cotton)
        for job in jobs: R.validate_job(job)
        for job in cotton: self.assertEqual(T.job_settings(job)['contract'], T.RETIMED_NETWORK_CONTRACT)
        changed = copy.deepcopy(cotton[0]); changed['teacher_revision']['exploration']['cotton']['no_network_actions']['CanalEra'] = [15]
        with self.assertRaises(ValueError): R.validate_job(changed)

    def test_the_recipe_transition_is_reviewed_forward_only(self):
        from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from test_training_g2654 import g2600
        rail = G2654.configure(g2600(1)); timed = G3471.configure(rail); after = G.configure(timed)
        sig = lambda c: dict(config=c, implementation=QUALIFIED)
        self.assertTrue(signature_transition(sig(timed), sig(after)))
        self.assertTrue(signature_transition(sig(rail), sig(after)))
        self.assertFalse(signature_transition(sig(after), sig(timed)))
        self.assertFalse(signature_transition(sig(after), sig(rail)))
        self.assertTrue(signature_transition(sig(after), sig(after)))

    def test_the_continuation_flag_moves_to_the_retimed_rule(self):
        from pathlib import Path
        from brass_ai.learning.contracts import SOURCE_ROOT
        text = (Path(SOURCE_ROOT) / 'scripts' / 'continue-rust-shadow.py').read_text()
        self.assertIn('config=validate_config(g3528.configure(config))', text)


if __name__ == '__main__':
    unittest.main()
