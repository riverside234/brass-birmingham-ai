"""The lab qualification's own checks, on fixtures. Its stages run on the lab."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import collector_runtime as R

spec = importlib.util.spec_from_file_location('qualify_rust_cpu', SOURCE_ROOT/'scripts/qualify-rust-cpu-inference.py')
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)
BUDGET = R.cpu_block()['teacher_budget']


def search(profile, n, k, completed=None, hit=False):
    return dict(profile=profile, frozen_n=n, effective_k=k, completed=n if completed is None else completed, deadline_hit=hit, elapsed_ns=10**9)


def tables(*rows):
    return [dict(metrics=dict(searches=list(rows)))]


class Budgets(unittest.TestCase):
    def test_production_budget_is_the_requested_one(self):
        self.assertEqual(BUDGET, dict(mode='count', full=640, full_k=16, fast=96, fast_k=8, c_scale=.1))

    def run_checks(self, *rows):
        checks = {}
        summary = Q.search_summary(tables(*rows), BUDGET)
        Q.check_searches(checks, summary)
        return checks, summary

    def test_a_correct_table_passes(self):
        checks, summary = self.run_checks(search('full', 640, 16), search('full', 640, 8), search('fast', 96, 8), search('fast', 96, 4))
        self.assertTrue(all(c['passed'] for c in checks.values()), checks)
        self.assertEqual(summary['full']['frozen_n'], {'640': 2})
        self.assertEqual(summary['fast']['searches'], 2)

    def test_zero_quota_turns_are_not_searches(self):
        _, summary = self.run_checks(search('full', 640, 16), search('full', 0, 0), search('fast', 96, 8))
        self.assertEqual(summary['full']['searches'], 1)

    def test_split_scale_checks_the_applied_scale_for_each_search_profile(self):
        budget = dict(BUDGET, c_scale=.15, fast_c_scale=.5)
        full = dict(search('full', 640, 16), c_scale=.15)
        for value in (.5, .15, None):
            fast = dict(search('fast', 96, 8), c_scale=value)
            checks = {}
            Q.check_searches(checks, Q.search_summary(tables(full, fast), budget))
            self.assertTrue(checks['full_uses_c_scale']['passed'])
            self.assertEqual(checks['fast_uses_c_scale']['passed'], value == .5)

    def test_every_wrong_budget_fails_by_name(self):
        for rows, failed in [((search('full', 256, 16), search('fast', 96, 8)), 'full_uses_the_budget'),
                             ((search('full', 640, 16), search('fast', 48, 8)), 'fast_uses_the_budget'),
                             ((search('full', 640, 16), search('fast', 96, 4)), 'fast_reaches_k'),
                             ((search('full', 640, 32), search('fast', 96, 8)), 'full_stays_within_k'),
                             ((search('full', 640, 16),), 'fast_searches_ran')]:
            with self.subTest(failed=failed):
                checks, _ = self.run_checks(*rows)
                self.assertFalse(checks[failed]['passed'])

    def test_mixed_long_scales_are_bound_to_each_table_group(self):
        from brass_ai.recovery.training_revision_g1491 import COTTON_SPLIT_SEARCH

        budget = dict(BUDGET, **COTTON_SPLIT_SEARCH)
        fast = dict(search('fast', 96, 8), c_scale=.5)
        jobs = [{}, dict(teacher_revision=dict(exploration=dict(cotton={})))]
        for cotton_scale in (.1, .2):
            games = [*tables(dict(search('full', 640, 8), c_scale=.2), fast),
                     *tables(dict(search('full', 640, 8), c_scale=cotton_scale), fast)]
            checks = {}
            Q.check_searches(checks, Q.search_summary(games, budget, jobs=jobs))
            self.assertTrue(checks['full_standard_uses_c_scale']['passed'])
            self.assertTrue(checks['fast_uses_c_scale']['passed'])
            self.assertEqual(checks['full_cotton_uses_c_scale']['passed'], cotton_scale == .1)

    def test_searches_stopped_by_the_safety_cap_fail(self):
        rows = [search('full', 640, 16, completed=500, hit=True)]*5+[search('full', 640, 16)]*95+[search('fast', 96, 8)]
        checks, summary = self.run_checks(*rows)
        self.assertFalse(checks['full_rarely_hits_the_safety_cap']['passed'])
        self.assertEqual(summary['full']['deadline_hits'], 5)


def record(step, chosen=0, probability=.5, choices=2, seat=0):
    return dict(step=step, seat=seat, chosen=chosen, choices=[dict(id=str(i)) for i in range(choices)],
                behavior=dict(probability=probability))


def game(*rows, game_id='g'):
    return dict(game_id=game_id, records=list(rows))


class Matching(unittest.TestCase):
    def test_identical_games_match(self):
        a = game(record(0), record(1, choices=1, probability=1.))
        result = Q.match_games([a], [a])
        self.assertEqual((result['games'], result['diverged'], result['rows'], result['forced_rows']), (1, 0, 1, 1))

    def test_rounding_within_bound_passes(self):
        result = Q.match_games([game(record(0, probability=.5))], [game(record(0, probability=.5004))])
        self.assertEqual(result['tolerance_failure_count'], 0)
        self.assertAlmostEqual(result['max_absolute'], .0004)

    def test_a_probability_outside_the_bound_is_reported(self):
        result = Q.match_games([game(record(3, probability=.5))], [game(record(3, probability=.52))])
        self.assertEqual(result['tolerance_failure_count'], 1)
        self.assertEqual(result['tolerance_failures'][0]['step'], 3)

    def test_a_sample_flip_counts_and_stops_the_comparison(self):
        left = game(record(0), record(1, chosen=0), record(2, probability=.1))
        right = game(record(0), record(1, chosen=1), record(2, probability=.9))
        result = Q.match_games([left], [right])
        self.assertEqual(result['diverged'], 1)
        self.assertEqual(result['tolerance_failure_count'], 0)
        self.assertEqual(result['rows'], 2)

    def test_different_game_lengths_count_as_divergence(self):
        result = Q.match_games([game(record(0))], [game(record(0), record(1))])
        self.assertEqual(result['diverged'], 1)

    def test_branches_are_compared_too(self):
        parent = game(record(0))
        branch = dict(game(record(0, chosen=1), game_id='b'))
        result = Q.match_games([dict(parent, branches=[game(record(0), game_id='b')])], [dict(parent, branches=[branch])])
        self.assertEqual((result['games'], result['diverged']), (2, 1))


class RatedGames(unittest.TestCase):
    def test_arms_pair_by_setup_whatever_their_random_ids(self):
        left = [game(record(0), game_id='z'), game(record(0, choices=3), game_id='a')]
        left[0]['setup_group'], left[1]['setup_group'] = 'one', 'two'
        right = [game(record(0, choices=3), game_id='y'), game(record(0), game_id='b')]
        right[0]['setup_group'], right[1]['setup_group'] = 'two', 'one'
        self.assertEqual(Q.match_games(Q.by_setup(left), Q.by_setup(right))['diverged'], 0)
        self.assertEqual(Q.match_games(sorted(left, key=lambda g: g['game_id']), sorted(right, key=lambda g: g['game_id']))['diverged'], 2)


class BlendError(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_cpu_inference import headed_model
        torch.set_num_threads(2)
        cls.model = headed_model()

    def row(self, weight):
        from brass_ai.learning.search_value import correction
        state = torch.zeros((1, self.model.config.state_dim))
        mapping = torch.tensor([[0, 1, 2, 3]])
        remaining = torch.zeros((1, 4))
        with torch.inference_mode():
            encoded = self.model.encode_state(state)
            utility = self.model.heads_from_encoded(encoded, mapping, remaining)[3].float()
            delta = correction(self.model, encoded, mapping, remaining).float()
        value = (utility+torch.tensor(weight, dtype=torch.float32)*delta)[0].double().tolist()
        return dict(state=state[0].tolist(), mapping=[0, 1, 2, 3], remaining=[0.]*4, v=value), delta

    def test_a_correct_blend_has_no_error(self):
        row, delta = self.row(.25)
        self.assertLess(Q.blend_error(self.model, row, .25), 1e-6)
        self.assertGreater(float(delta.abs().max()), 1e-5)

    def test_a_missing_blend_is_detected(self):
        row, delta = self.row(1.)
        self.assertGreater(Q.blend_error(self.model, row, 0.), float(delta.abs().max())*.5)


class TeacherJobs(unittest.TestCase):
    def test_requested_blends_do_not_inherit_the_training_stage(self):
        from brass_ai.recovery.search_value_config import defaults
        config = json.loads((SOURCE_ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-rust-teacher-on-stages-v2-share025.json').read_text())
        for scheduled in (True, False):
            for saved in (None, dict(stage=dict(stage='A', w=0.)), dict(stage=dict(stage='B', w=.35)),
                          dict(stage=dict(stage='C', w=1.)), dict(stage='B', w=.35)):
                with self.subTest(scheduled=scheduled, saved=saved):
                    recipe = copy.deepcopy(config)
                    if not scheduled:
                        recipe['teacher_phase']['search_value'] = dict(defaults(), stage='A')
                    state = dict(iteration=1307, teacher_seed_cursor=48925454)
                    if saved is not None:
                        state['search_value'] = copy.deepcopy(saved)
                    c = SimpleNamespace(config=recipe, readiness=dict(ready=True),
                        snapshot=dict(state=state, model_id='a'*64, checkpoint=dict(sha256='b'*64)))
                    before = copy.deepcopy(c)
                    jobs = []
                    for number, weight in enumerate(Q.BLENDS):
                        count = 8 if weight == 0 else 4
                        batch = Q.teacher_jobs(c, weight, count, 100000*number)
                        self.assertEqual(len(batch), count)
                        self.assertEqual({job['search_value']['w'] for job in batch}, {weight})
                        self.assertTrue(all(job['split'] == 'train' and job['readiness'] == c.readiness for job in batch))
                        jobs.extend(batch)
                    self.assertEqual(len({job['expected']['job_id'] for job in jobs}), 16)
                    self.assertEqual(c.snapshot, before.snapshot)
                    self.assertEqual(c.config, before.config)


class CpuView(unittest.TestCase):
    def test_the_evidence_shows_every_view_of_the_cpu_count(self):
        view = Q.cpu_view()
        self.assertTrue({'cpu_count', 'affinity', 'online', 'smt_active', 'cgroup_cpuset', 'allowed_list', 'nproc', 'nproc_all',
                         'slurm_cpus_per_task', 'omp_num_threads'} <= set(view))
        self.assertEqual(view['cpu_count'], os.cpu_count())
        json.dumps(view)


def table(long=(), short=(), receipt=None, job=None):
    def row(label, **fields):
        return dict(teacher=dict(label, **fields))
    rows = [row(dict(explored=True) if e else {}) for e in long] + [row(dict(profile='fast'), **({'explored': True} if e else {})) for e in short]
    metrics = dict(cpu_inference=dict(revision=receipt) if receipt else {})
    return dict(rows=rows, metrics=metrics)


def receipt(job, long=3, short=5, sampled=(3, 2)):
    return dict(job=job, short_labels=short, decisions=dict(long=long+2, short=short+9), sampled=dict(long=sampled[0], short=sampled[1]),
                changed=dict(long=1, short=1))


class G1491Teacher(unittest.TestCase):
    job = dict(contract='g1491-job', short_labels=True)

    def verdict(self, *tables_):
        checks, jobs = {}, [dict(teacher_revision=self.job)]*len(tables_)
        Q.check_revision(checks, jobs, list(tables_), Q.label_summary(list(tables_)))
        return {k: v['passed'] for k, v in checks.items()}

    def good(self):
        return table(long=[True]*3, short=[True]*2+[False]*3, receipt=receipt(self.job))

    def test_a_table_with_both_label_kinds_explored_as_reviewed_passes(self):
        verdict = self.verdict(self.good())
        self.assertTrue(all(verdict.values()), verdict)
        summary = Q.label_summary([self.good(), self.good()])
        self.assertEqual((summary['full'], summary['fast']), (dict(labels=6, explored=6), dict(labels=10, explored=4)))
        self.assertEqual(summary['worker']['short_labels'], 10)

    def test_a_long_only_restart_qualifies_with_no_short_labels(self):
        self.job = dict(self.job, short_labels=False)
        verdict = self.verdict(table(long=[True]*3, receipt=receipt(self.job, short=0)))
        self.assertTrue(all(verdict.values()), verdict)

    def test_the_revised_qualification_runs_as_many_tables_as_an_iteration(self):
        from brass_ai.recovery import training_revision_g1491 as g1491
        self.assertEqual(set(Q.TABLES_PER_BLEND_REVISED), set(Q.BLENDS))
        self.assertEqual(sum(Q.TABLES_PER_BLEND_REVISED.values()), g1491.TABLES['train'])
        self.assertEqual(sum(Q.TABLES_PER_BLEND.values()), 16)

    def test_each_departure_fails_by_name(self):
        for name, bad in [('short_labels_produced', table(long=[True]*3, receipt=receipt(self.job, short=0))),
                          ('long_labels_produced', table(short=[True, False], receipt=receipt(self.job, long=0, short=2, sampled=(0, 1)))),
                          ('every_long_label_is_explored', table(long=[True, True, False], short=[True, False], receipt=receipt(self.job, short=2, sampled=(3, 1)))),
                          ('short_labels_explore_only_openings', table(long=[True], short=[True]*4, receipt=receipt(self.job, long=1, short=4, sampled=(1, 4)))),
                          ('short_labels_explore_only_openings', table(long=[True], short=[False]*4, receipt=receipt(self.job, long=1, short=4, sampled=(1, 0)))),
                          ('worker_echoed_the_revision', table(long=[True], short=[True, False], receipt=receipt(dict(self.job, short_labels=False), long=1, short=2))),
                          ('worker_counts_match_the_labels', table(long=[True], short=[True, False], receipt=receipt(self.job, long=1, short=3))),
                          ('worker_counts_match_the_labels', table(long=[True]*3, short=[True, False], receipt=receipt(self.job, long=3, short=2, sampled=(1, 1)))),
                          ('worker_counts_match_the_labels', table(long=[True], short=[True, False]))]:
            with self.subTest(failed=name, labels=len(bad['rows'])):
                self.assertFalse(self.verdict(bad)[name])


EXPORT = SOURCE_ROOT/'results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z'


@unittest.skipUnless((EXPORT/'state.json').exists(), 'Needs the g1491 export')
class G1491Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from brass_ai.bank import read
        cls.state = read(EXPORT/'state.json')
        cls.config = cls.state['signature']['config']

    def test_the_committed_g1491_run_is_qualified_under_the_revision_by_default(self):
        from brass_ai.recovery import training_revision_g1491 as g1491
        for choice in ('auto', 'g1491'):
            with self.subTest(choice=choice):
                revision, config = Q.teacher_recipe(self.config, self.state, choice)
                self.assertEqual(revision, g1491.CONTRACT)
                self.assertTrue(g1491.applied(config))
                self.assertEqual(config['execution'], self.config['execution'])
        self.assertFalse(g1491.applied(self.config))
        self.assertEqual(Q.teacher_recipe(self.config, self.state, 'none'), (None, self.config))
        self.assertIsNone(Q.teacher_recipe(self.config, dict(self.state, iteration=1490), 'auto')[0])

    def test_an_adopted_run_is_used_as_it_is(self):
        _, once = Q.teacher_recipe(self.config, self.state, 'auto')
        self.assertIs(Q.teacher_recipe(once, self.state, 'auto')[1], once)

    def test_planned_tables_carry_the_revision_and_the_run_is_untouched(self):
        revision, config = Q.teacher_recipe(self.config, self.state, 'auto')
        before = copy.deepcopy((self.config, self.state))
        c = SimpleNamespace(config=self.config, teacher_config=config, readiness=dict(ready=True),
                            snapshot=dict(state=self.state, model_id=self.state['history'][-1]['model_id'], checkpoint=dict(sha256='b'*64)))
        revised = Q.teacher_jobs(c, 0., 4, 0)
        self.assertEqual(len(revised), 4)
        self.assertTrue(all(j['split'] == 'train' and j['teacher_revision']['short_labels'] is True for j in revised))
        plain = Q.teacher_jobs(SimpleNamespace(**dict(vars(c), teacher_config=self.config)), 0., 4, 0)
        self.assertTrue(all('teacher_revision' not in j for j in plain))
        self.assertEqual((self.config, self.state), before)


class League(unittest.TestCase):
    def test_entries_cover_learner_pool_anchor_and_population_once(self):
        make = lambda name: dict(model_id=name, path=name, sha256=name)
        learner, member, anchor, member2 = (make(x) for x in 'lmap')
        # The real league holds a frozen v4 bot member, which has no checkpoint file.
        bot = dict(kind='v4-history', model_id='v4-history-x', generation=570)
        state = dict(history=[make('old'), learner], quality_pool=dict(members=[member, member2, anchor, bot]), evaluation_anchor=anchor,
                     champion=None, population=[member2])
        result = Q.league_entries(SimpleNamespace(snapshot=dict(state=state)))
        self.assertEqual(sorted(result), ['a', 'l', 'm', 'p'])

    def test_representatives_prefer_the_learner_for_the_search_head(self):
        league = dict(x=dict(search_delta=True, utility=True), l=dict(search_delta=True, utility=True), u=dict(search_delta=False, utility=True),
                      a=dict(search_delta=False, utility=False))
        kinds = Q.representatives(SimpleNamespace(league=league, snapshot=dict(model_id='l')))
        self.assertEqual(kinds, dict(search_head='l', utility='u', anchor='a'))


class Reporting(unittest.TestCase):
    def test_a_stage_passes_only_if_every_check_passes(self):
        checks = {}
        Q.expect(checks, 'a', True)
        self.assertTrue(Q.finish(checks)['passed'])
        Q.expect(checks, 'b', False, 7)
        result = Q.finish(checks, extra=1)
        self.assertFalse(result['passed'])
        self.assertEqual(result['extra'], 1)

    def test_parity_scales_with_large_heads_and_still_catches_small_ones(self):
        self.assertLess(Q.scaled_error([100.01], [100.]), 1.)
        self.assertGreater(Q.scaled_error([.101], [.1]), 1.)
        self.assertGreater(Q.scaled_error([100.1], [100.]), 1.)
        self.assertEqual(Q.scaled_error([1., 2.], [1., 2.]), 0.)

    def test_a_failed_collection_leaves_a_working_gate_for_later_stages(self):
        import multiprocessing
        from brass_ai.recovery.resources import CPUAdmission
        mp = multiprocessing.get_context('spawn')
        original = CPUAdmission(mp, 4, 1)
        service = SimpleNamespace(mp=mp, cpu=original)
        Q.restore_admission(service, dict(cpu_budget=8, cpu_reserve=0))
        self.assertIs(service.cpu, original)
        original.abort()
        Q.restore_admission(service, dict(cpu_budget=8, cpu_reserve=0))
        self.assertIsNot(service.cpu, original)
        self.assertFalse(service.cpu.aborted.value)
        self.assertEqual(service.cpu.capacity, 8)

    def test_spread_of_nothing_is_none(self):
        self.assertIsNone(Q.spread([]))
        self.assertEqual(Q.spread([3, 1, 2])['p50'], 2)

    def test_startup_stage_requires_measurements_from_every_lane(self):
        phases = dict(mono_bootstrap_seconds=dict(lanes=2), total_seconds=dict(lanes=2))
        breakdown = dict(processes=2, lanes=2, lanes_reported=2, phases=phases, waits_for_all_lanes=False, reply_channel_seconds=0.,
                         first_lane_ready_seconds=1., last_lane_ready_seconds=2., process_start_seconds=.1)
        good = Q.startup(SimpleNamespace(ordinary_report=dict(startup_breakdown=breakdown), ordinary_usage=None))
        self.assertTrue(good['passed'], good)
        bad = Q.startup(SimpleNamespace(ordinary_report=dict(startup_breakdown=dict(breakdown, lanes_reported=1)), ordinary_usage=None))
        self.assertFalse(bad['passed'])
        self.assertFalse(Q.startup(SimpleNamespace(ordinary_report=None))['passed'])
        slow = Q.startup(SimpleNamespace(ordinary_report=dict(startup_breakdown=dict(breakdown, process_start_seconds=1.4)), ordinary_usage=None))
        self.assertFalse(slow['checks']['processes_start_without_waiting']['passed'])


class CrashedIteration(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name)/'run'
        (self.run_dir/'iteration-1290/teacher-shadow').mkdir(parents=True)
        (self.run_dir/'iteration-1290/teacher-shadow/table.json.gz').write_bytes(b'finished table')
        (self.run_dir/'state.json').write_text(json.dumps(dict(iteration=1290)))
        (self.run_dir/'teacher-labels').mkdir()
        (self.run_dir/'teacher-labels/generation-001290-train.jsonl.gz').write_bytes(b'published shard')
        (self.run_dir/'teacher-labels/generation-001289-train.jsonl.gz').write_bytes(b'committed shard')

    def test_inventory_covers_the_iteration_and_its_uncommitted_shards_only(self):
        files = Q.pending_inventory(self.run_dir, 1290)
        self.assertEqual(sorted(files), ['iteration-1290/teacher-shadow/table.json.gz', 'teacher-labels/generation-001290-train.jsonl.gz'])

    def test_a_missing_iteration_is_empty(self):
        self.assertEqual(Q.pending_inventory(self.run_dir, 7), {})

    def test_capture_runs_past_a_crashed_iteration(self):
        from brass_ai.rust_backend import lab_snapshot as S
        def capture(root, run, devices):
            self.assertEqual(S.pending_paths(Path(run), 1290), [])
            return dict(generation=1290), 'model', 'raw'
        with patch.object(S, 'capture', side_effect=capture):
            snapshot, model, raw, pending = Q.capture_run(self.run_dir, 'cuda:0')
        self.assertEqual((model, raw), ('model', 'raw'))
        self.assertEqual([Path(p).name for p in snapshot['pending_work']], ['iteration-1290'])
        self.assertEqual(len(pending['files']), 2)
        self.assertEqual(len(S.pending_paths(self.run_dir, 1290)), 1)

    def test_an_unfinished_checkpoint_still_stops_it(self):
        from brass_ai.rust_backend import lab_snapshot as S
        (self.run_dir/'generation-1291.pt.tmp').write_bytes(b'partial')
        with patch.object(S, 'capture', side_effect=AssertionError('must not capture')), self.assertRaisesRegex(ValueError, 'unfinished checkpoint'):
            Q.capture_run(self.run_dir, 'cuda:0')

    def pending(self):
        return dict(paths=[], generation=1290, files=Q.pending_inventory(self.run_dir, 1290))

    def test_an_untouched_iteration_confirms(self):
        from brass_ai.rust_backend import lab_snapshot as S
        with patch.object(S, 'verify_unchanged') as verify:
            Q.confirm_unchanged(dict(), self.pending(), self.run_dir)
        verify.assert_called_once()

    def test_a_changed_iteration_is_reported(self):
        from brass_ai.rust_backend import lab_snapshot as S
        pending = self.pending()
        (self.run_dir/'iteration-1290/teacher-shadow/new.json').write_bytes(b'x')
        with patch.object(S, 'verify_unchanged'), self.assertRaisesRegex(ValueError, 'changed during qualification'):
            Q.confirm_unchanged(dict(), pending, self.run_dir)


class Retry(unittest.TestCase):
    identity = dict(contract='c', implementation='i', cpu_build='b', checkpoint='k')

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.current = self.base/'9-cpu-qualification-now'
        self.current.mkdir()

    def attempt(self, stamp, stages, identity=None):
        directory = self.base/(stamp+'-cpu-qualification-x')
        directory.mkdir()
        (directory/'qualification.json').write_text(json.dumps(dict(identity=identity or self.identity, stages=stages)))
        for name, stage in stages.items():
            if 'skipped' not in stage:
                (directory/(name+'.json')).write_text(json.dumps(dict(passed=stage['passed'], marker=stamp)))

    def test_a_passed_stage_is_reused(self):
        self.attempt('1', dict(teacher=dict(passed=True), rating=dict(passed=False)))
        found = Q.passed_stages(self.base, self.current, self.identity)
        self.assertEqual(sorted(found), ['teacher'])
        self.assertEqual(found['teacher']['result']['marker'], '1')

    def test_the_newest_attempt_decides(self):
        self.attempt('1', dict(teacher=dict(passed=True), rating=dict(passed=False)))
        self.attempt('2', dict(teacher=dict(passed=False), rating=dict(passed=True)))
        self.assertEqual(sorted(Q.passed_stages(self.base, self.current, self.identity)), ['rating'])

    def test_other_code_builds_or_checkpoints_never_count(self):
        self.attempt('1', dict(teacher=dict(passed=True)), identity=dict(self.identity, checkpoint='other'))
        self.assertEqual(Q.passed_stages(self.base, self.current, self.identity), {})

    def test_environment_and_bundles_always_run_again(self):
        self.attempt('1', dict(environment=dict(passed=True), bundles=dict(passed=True), startup=dict(passed=True)))
        self.assertEqual(Q.passed_stages(self.base, self.current, self.identity), {})

    def test_a_reviewed_source_change_keeps_only_its_unaffected_stages(self):
        before = dict(self.identity, implementation='before')
        after = dict(self.identity, implementation='after')
        self.attempt('1', dict(teacher=dict(passed=True), ordinary=dict(passed=True)), identity=before)
        self.assertEqual(Q.reusable_stages(self.base, self.current, after), {})
        with patch.dict(Q.REVIEWED_CHANGES, after=dict(previous='before', stages=('teacher',))):
            self.assertEqual(sorted(Q.reusable_stages(self.base, self.current, after)), ['teacher'])

    def test_a_skipped_stage_is_not_a_pass(self):
        self.attempt('1', dict(ordinary=dict(passed=False, skipped='Needs bundles')))
        self.assertEqual(Q.passed_stages(self.base, self.current, self.identity), {})


class PreparedSuite(unittest.TestCase):
    def log(self, text):
        from brass_ai.rust_backend.lab_contract import pin
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)/'regressions.log'
        path.write_text(text)
        return dict(regressions=pin(str(path)))

    def test_a_passing_prepare_is_reused(self):
        value = self.log('Ran 43 tests in 300s\n\nOK (skipped=3)\n')
        self.assertEqual(Q.prepared_suite(value), value['regressions'])

    def test_a_failed_or_mostly_skipped_prepare_is_not(self):
        for text in ('Ran 43 tests in 300s\n\nFAILED (errors=1)\n', 'Ran 43 tests in 1s\n\nOK (skipped=20)\n', 'Ran 10 tests in 1s\n\nOK\n'):
            with self.subTest(text=text):
                self.assertIsNone(Q.prepared_suite(self.log(text)))

    def test_a_missing_or_changed_log_is_not(self):
        value = self.log('Ran 43 tests in 300s\n\nOK (skipped=3)\n')
        Path(value['regressions']['path']).write_text('Ran 43 tests in 300s\n\nOK\n')
        self.assertIsNone(Q.prepared_suite(value))
        self.assertIsNone(Q.prepared_suite(dict(regressions=None)))


@unittest.skipUnless(os.environ.get('BRASS_CPU_TEACHER') and os.environ.get('BRASS_CPU_BOOTSTRAP'), 'Needs built workers and a bootstrap')
class StagesOnFixtures(unittest.TestCase):
    """The bundle, parity and teacher stages, with fixture models and a small budget."""
    @classmethod
    def setUpClass(cls):
        from brass_ai.bank import sha
        from brass_ai.learning.batching import model_id
        from brass_ai.learning.checkpoint import save_checkpoint
        from brass_ai.learning.contracts import digest
        from brass_ai.recovery.inference import InferenceService
        from brass_ai.recovery.search_value_config import adopt
        from brass_ai.rust_backend.lab_contract import pin
        from test_b025 import config, model_fixture
        from test_cpu_inference import anchor_model, headed_model
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        learner, plain, anchor = headed_model(), model_fixture().eval(), anchor_model()
        identity = model_id(learner)
        entries = []
        for name, model in (('learner', learner), ('plain', plain), ('anchor', anchor)):
            path = root/(name+'.pt')
            save_checkpoint(path, model, step=10)
            entries.append(dict(path=str(path), sha256=sha(path), model_id=model_id(model)))
        c = config()
        c = adopt(c)
        c['teacher_phase']['mode'] = 'shadow'
        c['teacher_phase']['search_value']['stage'] = 'A'
        reward = digest(learner.config.utility)
        state = dict(iteration=10, history=[dict(generation=10, model_id=identity)], teacher_phase={},
                     quality_pool=dict(members=entries[1:]), evaluation_anchor=entries[2], population=[], champion=None)
        state['history'][-1].update(entries[0])
        execution = dict(json.loads((SOURCE_ROOT/'configs/step5/recovery-smoke.json').read_text())['execution'], processes=2, hosts_per_process=1)
        block = R.cpu_block(workers=4, teacher_budget=dict(mode='count', full=64, full_k=16, fast=24, fast_k=8))
        args = SimpleNamespace(mono='mono', device='cpu', workers=4, expected_generation=10)
        runtime = pin(os.environ['BRASS_CPU_RUNTIME'])
        receipt = dict(path=os.environ['BRASS_CPU_TEACHER'], sha256='0'*64)
        value = dict(runtime=dict(path=runtime['path'], sha256=runtime['sha256']), teacher=dict(path=os.environ['BRASS_CPU_TEACHER']),
                     teacher_execution_core=os.environ['BRASS_CPU_CORE'],
                     collector=dict(path=os.environ['BRASS_CPU_COLLECTOR']), receipt=receipt)
        cls.service = InferenceService(execution, 'cpu')
        cls.service.__enter__()
        cls.service.register(learner)
        cls.c = SimpleNamespace(args=args, out=root/'out', snapshot=dict(state=state, model_id=identity, checkpoint=dict(sha256='a'*64), generation=10),
                                model=learner, raw=dict(training=dict(utility_calibration=dict(ready=True, reward_id=reward))), config=c, block=block,
                                cpu_settings=dict(R.DEFAULTS, native_inference=block), service=cls.service, cpu_value=value,
                                fixtures=os.environ['BRASS_CPU_FIXTURES'])
        cls.c.out.mkdir()
        cls.build = dict(core_source=os.environ['BRASS_CPU_CORE'], bootstrap=dict(binary=dict(path=os.environ['BRASS_CPU_BOOTSTRAP'])))
        cls.value = value
        cls.root = root
        previous, Q.ROOT = Q.ROOT, root
        try:
            cls.bundle_result = Q.bundles(cls.c)
        finally:
            Q.ROOT = previous

    @classmethod
    def tearDownClass(cls):
        cls.service.__exit__(None, None, None)
        cls.directory.cleanup()

    def test_bundles_cover_all_three_model_kinds(self):
        result = self.bundle_result
        self.assertTrue(result['passed'], result)
        self.assertEqual(len(result['models']), 3)
        self.assertEqual(sum(not m['utility'] for m in result['models'].values()), 1)
        self.assertEqual(sum(m['search_delta'] for m in result['models'].values()), 1)

    def test_parity_covers_all_three_model_kinds(self):
        result = Q.parity(self.c)
        self.assertTrue(result['passed'], result)
        self.assertEqual(set(result['kinds']), {'search_head', 'utility', 'anchor'})
        for kind in result['kinds'].values():
            self.assertLessEqual(kind['worst_scaled_error'], 1.)

    def test_teacher_stage_runs_real_tables_at_each_blend(self):
        with patch('brass_ai.rust_backend.lab_shadow.teacher_build', return_value=self.build), \
                patch('brass_ai.rust_backend.lab_cpu_runtime.resolve', return_value=self.value), \
                patch.object(Q, 'ROOT', self.root):
            result = Q.teacher(self.c)
        self.assertTrue(result['passed'], result['checks'])
        self.assertEqual(result['tables'], 16)
        self.assertEqual(result['searches']['full']['frozen_n'], {'64': result['searches']['full']['searches']})
        self.assertIn('8', result['searches']['fast']['effective_k'])
        self.assertEqual(set(result['blend_effect']), {'0.25', '1.0'})
        self.assertEqual(result['mono_bootstrap_seconds']['samples'], 16)
        self.assertNotIn('worker', result['labels'])
        self.assertIsNone(result['revision'])

    def test_teacher_stage_under_the_g1491_revision_labels_short_searches_and_explores(self):
        from brass_ai.recovery import training_revision_g1491 as g1491
        planned = Q.teacher_jobs

        def revised(*args):
            jobs = planned(*args)
            for job in jobs:
                job['teacher_revision'] = g1491._job_block(g1491.EXPLORATION)
            return jobs
        with patch('brass_ai.rust_backend.lab_shadow.teacher_build', return_value=self.build),                 patch('brass_ai.rust_backend.lab_cpu_runtime.resolve', return_value=self.value),                 patch.object(Q, 'teacher_jobs', revised), patch.object(Q, 'ROOT', self.root):
            result = Q.teacher(self.c)
        self.assertTrue(result['passed'], {k: v for k, v in result['checks'].items() if not v['passed']})
        labels = result['labels']
        self.assertGreater(labels['fast']['labels'], 0)
        self.assertEqual(labels['full']['explored'], labels['full']['labels'])
        self.assertEqual(result['revision'], g1491._job_block(g1491.EXPLORATION))
        self.assertGreater(result['verification']['deferred_rows'], 0)
        self.assertEqual(result['verification']['deferred_logit_check']['samples'], 16)
        self.assertEqual(result['budget']['fast'], 24)


if __name__ == '__main__':
    unittest.main()
