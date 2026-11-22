"""g4551 table revision: standard PPO self-play tables (teacher cotton tables kept), 50+4 teacher tables and 176
teacher label rows per PPO step, plus the 64-lane cap and the CPU bundles without the search head."""
import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from brass_ai.recovery import training_revision_g4551 as R


def before():
    from test_teacher_training_g4475 import before as g4138_recipe
    from brass_ai.recovery import training_revision_g4475 as g4475
    return g4475.configure(g4138_recipe(), 4475)


def signature(config):
    from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
    return dict(config=config, implementation=QUALIFIED)


def plan(config, generation=4552):
    from brass_ai.recovery.teacher_phase import plan_jobs
    state = dict(iteration=generation, history=[], signature=signature(config))
    return plan_jobs(config, state, model_id='m'*64, export_hash='a'*64, epoch=generation, ready=True)[0]


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before(); cls.after = R.configure(cls.before, 4551)

    def test_it_overlays_the_tables_rows_and_selfplay_and_keeps_the_labels(self):
        from brass_ai.recovery import ordinary_cotton, training_revision_g3270 as g3270
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual(after['teacher_phase'], before['teacher_phase'])  # the qualified recipe is untouched
        self.assertEqual(after['table_plan'], before['table_plan'])
        self.assertEqual(R.previous(after), before)
        self.assertTrue(is_on(before) and is_on(after))  # recipe recognition peels the block
        self.assertEqual(buffer_identity(after), buffer_identity(before))  # labels and beta carry over
        self.assertEqual(R.configure(after, 4552), after)  # adopted once
        options = lambda config: g3270.phase_options(config, config['teacher_phase'])
        self.assertEqual((options(after)['term']['rows_per_step'], options(before)['term']['rows_per_step']), (176, 160))
        self.assertEqual((options(after)['tables'], options(before)['tables']),
                         (dict(train=50, held_out=4), dict(train=44, held_out=4)))
        self.assertEqual(options(after)['buffer'], options(before)['buffer'])
        self.assertIsNone(ordinary_cotton.selfplay_settings(after))
        self.assertEqual(ordinary_cotton.selfplay_settings(before), before['table_plan'][ordinary_cotton.KEY])
        self.assertTrue(ordinary_cotton.enabled(after))  # the committed block still names the cotton rule
        self.assertEqual(R.tag(after), 'teacher54-rows176-selfplay-standard')
        bad = copy.deepcopy(after); bad[R.KEY]['tables'] = dict(train=60, held_out=8)
        with self.assertRaisesRegex(ValueError, 'g4551 table revision'):validate_config(bad)
        from test_teacher_training_g4475 import before as without_g4475
        with self.assertRaisesRegex(ValueError, 'g4475 teacher training revision'):R.configure(without_g4475(), 4551)

    def test_it_is_adopted_once_forward_only(self):
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        changed = copy.deepcopy(after); changed[R.KEY]['from_generation'] = 4560
        self.assertFalse(signature_transition(signature(after), signature(changed)))

    def test_54_teacher_tables_keep_their_cotton_tables(self):
        from brass_ai.recovery.collector_runtime import MAX_WORKERS
        from brass_ai.recovery.teacher_cotton import enabled
        old, new = plan(self.before), plan(self.after)
        self.assertEqual(Counter(j['split'] for j in old), Counter(train=44, heldout=4))
        self.assertEqual(Counter(j['split'] for j in new), Counter(train=50, heldout=4))
        self.assertLessEqual(len(new), MAX_WORKERS)  # one CPU lane each
        # The teacher's cotton rule is unchanged: 3/20 of the train tables, one of the four held-out tables.
        for generation in (4552, 4553, 4554):
            jobs = plan(self.after, generation)
            cotton = Counter(j['split'] for j in jobs if enabled(j))
            self.assertIn(cotton['train'], (7, 8)); self.assertEqual(cotton['heldout'], 1)
        # The train seeds continue the same stream: the first 44 are the tables the old count would have run.
        self.assertEqual([j['seed'] for j in new if j['split'] == 'train'][:44], [j['seed'] for j in old if j['split'] == 'train'])
        self.assertEqual(len({j['expected']['job_id'] for j in new}), 54)


class SelfPlay(unittest.TestCase):
    """The admission audit refuses a self-play cotton seat once the revision is adopted."""

    def binding(self, config, index):
        from brass_ai.recovery import ordinary_cotton
        from brass_ai.rust_backend import lab_admission
        settings = config['table_plan'][ordinary_cotton.KEY]
        spec = dict(kind='neural', policy_id='p'*64, exploration_version=3)
        proof = dict(kind='selfplay', purpose='training', ordinary_game_index=index, sampling_seed=31, spec='d'*64,
                     learner_behavior=dict(model_id='m'*64, policy_id='p'*64),
                     ppo_opening_objective=config['update'].get('opening_objective'))
        if config.get('seat_contexts') is not None: proof['seat_contexts'] = config['seat_contexts']
        if 'critic_data' in config: proof['critic_table'] = {}
        if config.get('local_scoring_auxiliary', {}).get('mode', 'off') != 'off': proof['local_scoring'] = {}
        assignment = ordinary_cotton.planned_assignment(settings, index, 31)
        job = lambda cotton: dict(index=0, seed=19, sampling_seed=31, policies=[spec]*4,
                                  provenance=dict(proof, **({ordinary_cotton.KEY: assignment} if cotton else {})))
        producer = dict(build_sha256='b'*64, b01_sha256='c'*64, signature='d'*64,
                        profile=lab_admission.METHOD, v205_enabled=False)
        return lambda cotton: lab_admission.binding(job(cotton), producer, generation=4552, model_id='m'*64, config=config)

    def test_every_selfplay_table_is_standard(self):
        from brass_ai.recovery import ordinary_cotton
        old_recipe = before(); new_recipe = R.configure(old_recipe, 4551)
        settings = old_recipe['table_plan'][ordinary_cotton.KEY]
        index = next(i for i in range(100) if ordinary_cotton.planned_assignment(settings, i, 31))
        old, new = self.binding(old_recipe, index), self.binding(new_recipe, index)
        self.assertTrue(old(True)['sha256'])  # the cotton table as planned before
        with self.assertRaisesRegex(ValueError, 'Ordinary cotton job differs'):old(False)
        self.assertTrue(new(False)['sha256'])  # the same table is standard after
        with self.assertRaisesRegex(ValueError, 'Ordinary cotton job differs'):new(True)

    def test_the_vp_report_and_chart_take_a_split_with_no_cotton_tables(self):
        from brass_ai.recovery import vp_plot
        groups = dict(standard_exploration=dict(tables=224, builds=41, flips=37, seat_games=896, vp_sum=121700.),
                      cotton_exploration=dict(tables=0, builds=0, flips=0, seat_games=0, vp_sum=0.))
        rows = [dict(generation=g, **vp_plot._totals(dict(health_by_composition=dict(selfplay=dict(vp_sum=121700., seat_games=896, tables=224)),
                                                          economy=dict(cotton_by_selfplay=copy.deepcopy(groups)))))
                for g in range(1, 21)]
        history = dict(format=vp_plot.FORMAT, initial_checkpoint={}, committed_generation=20, iterations=rows)
        data = vp_plot.groups(history, 20)
        self.assertEqual([d['by_table']['cotton']['mean_vp'] for d in data], [None, None])
        with tempfile.TemporaryDirectory() as tmp:
            vp_plot.render(history, Path(tmp)/'progress_vp.svg', generation=20)
            self.assertIn('vp-recent-standard-line', (Path(tmp)/'progress_vp.svg').read_text(encoding='utf-8'))


class Workers(unittest.TestCase):
    def test_cpu_lanes_are_capped_at_64(self):
        from brass_ai.recovery.collector_runtime import MAX_WORKERS, cpu_block, validate_cpu
        self.assertEqual(MAX_WORKERS, 64)
        validate_cpu(cpu_block(workers=64))
        with self.assertRaisesRegex(ValueError, 'native_inference'):validate_cpu(cpu_block(workers=65))
        import argparse
        import importlib.util
        from brass_ai.learning.contracts import SOURCE_ROOT
        spec = importlib.util.spec_from_file_location('continue_rust_shadow', SOURCE_ROOT/'scripts/continue-rust-shadow.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertEqual(module.workers_arg('64'), 64)
        with self.assertRaises(argparse.ArgumentTypeError):module.workers_arg('96')
        self.assertEqual(module.parser().parse_args(['--expected-generation', '4551']).workers, 64)


class SearchDelta(unittest.TestCase):
    def test_only_a_blending_teacher_job_needs_the_search_head(self):
        from brass_ai.learning.search_value import binding
        from brass_ai.rust_backend.lab_cpu_models import needs_search_delta
        self.assertFalse(needs_search_delta([dict(search_value=binding(0))] * 3))
        self.assertFalse(needs_search_delta([dict()]))  # tables with no search-value binding
        self.assertTrue(needs_search_delta([dict(search_value=binding(0)), dict(search_value=binding(.5))]))
        self.assertTrue(needs_search_delta([dict(search_value=binding(0), seat_values=[binding(0), binding(1)] * 2)]))
        jobs = plan(R.configure(before(), 4551))
        self.assertEqual(needs_search_delta(jobs), any(j['search_value']['w'] > 0 for j in jobs if 'search_value' in j))


if __name__ == '__main__':
    unittest.main()
