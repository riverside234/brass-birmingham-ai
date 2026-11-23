"""g3471: cotton exploration timed within each era (PPO and teacher), and self-play VP by table type."""
import copy
import unittest
from types import SimpleNamespace

from brass_ai.recovery import ordinary_cotton as C, teacher_cotton as T, training_revision_g1491 as R
from brass_ai.recovery import training_revision_g2654 as G2654, training_revision_g3471 as G
from brass_ai.step5.policy import cotton_timing, era_action

# (era, round, actions remaining) -> (the seat's action in the era, explores, develop and network too)
ACTIONS = [
    ('CanalEra', 1, 1, 1, False, False), ('CanalEra', 2, 2, 2, True, True), ('CanalEra', 7, 1, 13, True, True),
    ('CanalEra', 8, 2, 14, True, False), ('CanalEra', 8, 1, 15, False, False),
    ('RailEra', 1, 2, 1, False, False), ('RailEra', 1, 1, 2, False, False), ('RailEra', 2, 2, 3, True, True),
    ('RailEra', 7, 1, 14, True, True), ('RailEra', 8, 2, 15, True, False), ('RailEra', 8, 1, 16, False, False),
]
CHOICES = [dict(id='build', kind='build', industry='CottonMill'), dict(id='dev', kind='develop', industry='CottonMill'),
           dict(id='net', kind='network', target='AB'), dict(id='loan', kind='loan')]


def view(era, round_, left, *, seat=0, phase='action'):
    tiles = [dict(level=3), dict(level=4)]
    return dict(seat=seat, context=dict(phase=phase, era=era, round=round_, actions_remaining=left),
                players=[dict(mat=[dict(type='CottonMill', tiles=tiles)]) for _ in range(4)],
                slots=[dict(id='A:0', types=['CottonMill'])], links=[dict(id='AB', regions=['A', 'B'])])


def table(settings, seed=4400):
    i = next(i for i in range(40) if C.planned_assignment(settings, i, seed))
    proof = dict(kind='selfplay', purpose='training', ordinary_game_index=i, sampling_seed=seed,
                 **{C.KEY: C.planned_assignment(settings, i, seed)})
    return C.TableExploration(proof)


class Rule(unittest.TestCase):
    def test_each_era_is_quiet_at_its_edges_and_builds_only_at_its_end(self):
        for era, round_, left, index, explores, wide in ACTIONS:
            context = view(era, round_, left)['context']
            self.assertEqual(era_action(context), index)
            self.assertEqual(cotton_timing(context), (explores, wide), (era, round_, left))
        # Every action of both eras: 15 Canal actions (one in round 1), 16 Rail actions.
        canal = sorted({era_action(dict(era='CanalEra', round=r, actions_remaining=a)) for r in range(1, 9) for a in (1, 2)
                        if not (r == 1 and a == 2)})
        rail = sorted({era_action(dict(era='RailEra', round=r, actions_remaining=a)) for r in range(1, 9) for a in (1, 2)})
        self.assertEqual((canal, rail), (list(range(1, 16)), list(range(1, 17))))

    def test_ppo_cotton_tables_follow_the_timing_under_the_new_contract_only(self):
        timed, plain = table(C.TIMED), table(C.RAIL_NETWORK)
        for era, round_, left, _, explores, wide in ACTIONS:
            expected = ([0, 1, 2] if wide else [0]) if explores else []
            for phase in ('action', 'continuation'):
                v = view(era, round_, left, seat=timed.seats[0], phase=phase)
                want = expected if phase == 'action' else [i for i in expected if i != 0]  # builds are action-phase only
                self.assertEqual(timed.targets(v, CHOICES), want, (era, round_, left, phase))
                v['seat'] = plain.seats[0]
                self.assertEqual(plain.targets(v, CHOICES), [0, 1, 2] if phase == 'action' else [1, 2])
        self.assertEqual(C.TIMED, dict(C.RAIL_NETWORK, contract=C.TIMED_CONTRACT,
                                       quiet_actions=dict(CanalEra=[1, 15], RailEra=[1, 2, 16]),
                                       build_only_actions=dict(CanalEra=[14, 15], RailEra=[15, 16])))
        self.assertIn(C.TIMED_CONTRACT, C.NETWORK_CONTRACTS)

    def test_teacher_labels_follow_the_timing_and_the_settings_match_the_native_mode(self):
        settings = T.settings(contract=T.TIMED_NETWORK_CONTRACT)
        self.assertEqual(settings, dict(T.settings(contract=T.RAIL_NETWORK_CONTRACT), contract=T.TIMED_NETWORK_CONTRACT,
                                        quiet_actions=dict(CanalEra=[1, 15], RailEra=[1, 2, 16]),
                                        build_only_actions=dict(CanalEra=[14, 15], RailEra=[15, 16])))
        counts = [6, 3, 1, 0]

        def label(cotton_visits, covered):  # the planned and played cotton build, chosen by the exploration
            probability = .7 + .3 * counts[0] / cotton_visits  # planned base 1, plus the cotton component's share
            return dict(seat=0, counts=counts, played_action='build', explored=True, cotton_exploration=dict(
                contract=T.TIMED_NETWORK_CONTRACT, targeted=True, probability=probability, used_before=0, network_used_before=0,
                cotton_visits=cotton_visits, total_visits=10, covered_locations=covered, planned_action='build'))
        T.validate_label(label(10, 3), CHOICES, view=view('CanalEra', 4, 2))       # develop and network count too
        T.validate_label(label(6, 1), CHOICES, view=view('CanalEra', 8, 2))        # build-only: the build alone
        with self.assertRaisesRegex(ValueError, 'coverage'):
            T.validate_label(label(10, 3), CHOICES, view=view('RailEra', 8, 2))
        for era, round_, left in (('CanalEra', 1, 1), ('CanalEra', 8, 1), ('RailEra', 1, 2), ('RailEra', 1, 1), ('RailEra', 8, 1)):
            with self.assertRaisesRegex(ValueError, 'quiet action'):
                T.validate_label(label(6, 1), CHOICES, view=view(era, round_, left))


class Recipe(unittest.TestCase):
    def test_the_timing_extends_the_rail_cotton_recipe_for_both_table_kinds(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        from test_training_g2654 import g2600
        before = G2654.configure(g2600(1))
        with self.assertRaisesRegex(ValueError, 'g2654'):
            G.configure(g2600(1))
        after = G.configure(before)
        validate_config(after)
        self.assertTrue(G.applied(after) and G2654.applied(after) and is_on(after))
        self.assertIs(G.configure(after), after)
        self.assertEqual(buffer_identity(before), buffer_identity(after))
        self.assertEqual(after['table_plan'][C.KEY], C.TIMED)
        self.assertEqual(R.cotton_settings(after)['table_fraction'], R.cotton_settings(before)['table_fraction'])
        self.assertEqual({k for k in after if after[k] != before[k]}, {'table_plan', 'teacher_phase'})
        state = dict(iteration=3471, signature=dict(config=after), search_value=dict(stage='C', w=1.))
        jobs, _ = plan_jobs(after, state, model_id='m', export_hash='e', epoch=3471, ready=True)
        cotton = [j for j in jobs if T.enabled(j)]
        self.assertTrue(cotton)
        for job in jobs: R.validate_job(job)
        for job in cotton: self.assertEqual(T.job_settings(job)['contract'], T.TIMED_NETWORK_CONTRACT)
        changed = copy.deepcopy(cotton[0]); changed['teacher_revision']['exploration']['cotton']['quiet_actions']['RailEra'] = [1, 16]
        with self.assertRaises(ValueError): R.validate_job(changed)

    def test_the_recipe_transition_is_reviewed_forward_only(self):
        from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        from test_training_g2654 import g2600
        before = G2654.configure(g2600(1)); after = G.configure(before)
        sig = lambda c: dict(config=c, implementation=QUALIFIED)
        self.assertTrue(signature_transition(sig(before), sig(after)))
        self.assertFalse(signature_transition(sig(after), sig(before)))


class SelfPlayVP(unittest.TestCase):
    def test_learner_vp_is_kept_by_table_type_without_split_curves(self):
        from brass_ai.recovery import diagnostics, vp_plot

        class Game(dict):
            pass
        def game(name, vps, cotton):
            g = Game(game_id=name, records=[], opponents=['p'] * 4,
                     result=dict(seats=[dict(seat=s, vp=v, rank=1 + sorted(vps, reverse=True).index(v)) for s, v in enumerate(vps)]))
            if cotton: g.cotton_seats = (1,)
            return g
        from unittest import mock
        targets = lambda result: dict(vp=[s['vp'] for s in result['seats']])
        with mock.patch.object(diagnostics, 'terminal_targets', targets):
            out = diagnostics.economy([game('a', [100, 90, 80, 70], False), game('b', [120, 110, 100, 90], True),
                                       game('c', [140, 130, 120, 110], False)], {'p'}, selfplay=True)
        groups = out['cotton_by_selfplay']
        self.assertEqual((groups['standard_exploration']['seat_games'], groups['standard_exploration']['vp_sum']), (8, 840.))
        self.assertEqual((groups['cotton_exploration']['seat_games'], groups['cotton_exploration']['vp_sum']), (4, 420.))
        merged = diagnostics.merge_summaries
        self.assertIsNotNone(merged)
        rows = []
        for generation in range(1, 21):
            report = dict(iteration=generation - 1, health_by_composition=dict(selfplay=dict(vp_sum=1260., seat_games=12, tables=3)),
                          economy=dict(cotton_by_selfplay=copy.deepcopy(groups)))
            rows.append(dict(generation=generation, **vp_plot._totals(report)))
        history = dict(format=vp_plot.FORMAT, initial_checkpoint={}, committed_generation=20, iterations=rows)
        data = vp_plot.groups(history, 20)
        self.assertEqual([d['by_table']['standard']['mean_vp'] for d in data], [105., 105.])
        self.assertEqual([d['by_table']['cotton']['mean_vp'] for d in data], [105., 105.])
        # A history row saved without the split (older reports) leaves the group's split empty.
        older = copy.deepcopy(history); older['iterations'][0].pop('by_table')
        self.assertIsNone(vp_plot.groups(older, 20)[0]['by_table'])
        restored = vp_plot._totals({'health_by_composition': {'selfplay': rows[0]}, 'by_table': rows[0]['by_table']})
        self.assertEqual(restored['by_table'], rows[0]['by_table'])
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'progress_vp.svg'
            vp_plot.render(history, out_path, generation=20)
            svg = out_path.read_text(encoding='utf-8')
        for panel in ('history', 'recent'):
            self.assertIn(f'vp-{panel}-line', svg)
            self.assertNotIn(f'vp-{panel}-cotton-line', svg)
            self.assertNotIn(f'vp-{panel}-standard-line', svg)
        self.assertNotIn('Cotton exploration tables', svg)
        self.assertNotIn('Standard tables', svg)


if __name__ == '__main__':
    unittest.main()
