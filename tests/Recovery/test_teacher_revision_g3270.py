"""g3270 teacher revision (build steps 2-3): recipe, label window, rebuilt targets, batches, search-value rotation,
and the code-only collection changes (spooled teacher output, collection GC window, parity sample)."""
import copy
import gc
import json
import os
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.recovery import training_revision_g3270 as R


def before():
    from test_rating_cadence import g3140
    from brass_ai.recovery import rating_cadence
    from brass_ai.recovery.config import validate_config
    return validate_config(rating_cadence.configure(g3140(), 20, 3170))


def signature(config):
    from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
    return dict(config=config, implementation=QUALIFIED)


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before(); cls.after = R.configure(cls.before, 3270)

    def test_it_overlays_the_qualified_teacher_recipe(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual(after['teacher_phase'], before['teacher_phase'])  # the qualified recipe is untouched
        self.assertEqual(R.previous(after), before)
        self.assertTrue(is_on(before) and is_on(after))
        self.assertEqual(buffer_identity(after), buffer_identity(before))  # labels and beta carry over
        self.assertEqual(R.configure(after, 3271), after)  # adopted once
        self.assertEqual(R.tables(after), dict(train=36, held_out=4))
        self.assertEqual(R.tables(before), dict(train=32, held_out=4))
        options = R.phase_options(after, after['teacher_phase'])
        self.assertEqual((options['tables'], options['buffer']['iterations'], options['buffer']['age_weights'],
                          options['term']['rows_per_step']), (dict(train=36, held_out=4), 8, [8, 7, 6, 5, 4, 3, 2, 1], 128))
        self.assertIs(R.phase_options(before, before['teacher_phase']), before['teacher_phase'])
        from brass_ai.recovery.search_value_config import validate as search_value
        values = search_value(R.search_value_options(after, after['teacher_phase']['search_value']))
        self.assertEqual((values['steps_per_iteration'], values['trunk']['intervals'], values['trunk']['interval_steps']), (32, 2, 16))
        self.assertEqual(R.value_settings(after)['buffer_iterations'], 16)
        self.assertEqual(R.term_settings(after)['label_scale'], .2)
        self.assertIsNone(R.term_settings(before)); self.assertIsNone(R.parity_rows(before))
        bad = copy.deepcopy(after); bad[R.KEY]['rows_per_step'] = 256
        with self.assertRaisesRegex(ValueError, 'g3270 teacher revision'):validate_config(bad)
        other = copy.deepcopy(before); other['teacher_phase']['term']['rows_per_step'] = 256
        with self.assertRaisesRegex(ValueError, 'search-refresh recipe'):R.configure(other, 3270)

    def test_it_is_adopted_once_and_changes_the_collection(self):
        from brass_ai.rust_backend.lab_shadow_config import collection_signature, signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        changed = copy.deepcopy(after); changed[R.KEY]['from_generation'] = 3280
        self.assertFalse(signature_transition(signature(after), signature(changed)))
        self.assertNotEqual(collection_signature(signature(before)), collection_signature(signature(after)))

    def test_beta_is_remeasured_once_at_adoption(self):
        from brass_ai.recovery.adaptive_teacher_beta import REMEASUREMENT, pending
        weight = dict(beta=.0278, calibrated=True)
        state = dict(iteration=3270, signature=signature(self.after),
                     source_transitions=[dict(iteration=3270, from_signature=signature(self.before))])
        requested = R.teacher_weight(self.after, state, weight)
        self.assertEqual(requested[REMEASUREMENT], dict(requested_generation=3270, previous_beta=.0278))
        self.assertTrue(pending(requested))
        self.assertEqual(R.teacher_weight(self.after, state, requested), requested)  # recovery keeps one request
        measured = dict(requested, **{REMEASUREMENT: dict(requested[REMEASUREMENT], measured_generation=3270)})
        later = dict(state, iteration=3271)
        self.assertEqual(R.teacher_weight(self.after, later, measured), measured)
        self.assertEqual(R.teacher_weight(self.before, dict(state, signature=signature(self.before)), weight), weight)

    def test_forty_tables_with_four_validating_every_iteration(self):
        from collections import Counter
        from brass_ai.recovery.search_value_schedule import validation_plan
        from brass_ai.recovery.teacher_phase import plan_jobs
        for config, expected in ((self.after, {3270: (36, 4), 3271: (36, 4), 3274: (36, 4)}),
                                 (self.before, {3270: (32, 0), 3271: (32, 0), 3274: (32, 4)})):
            for generation, (train, heldout) in expected.items():
                state = dict(iteration=generation, history=[], signature=signature(config))
                self.assertEqual(validation_plan(config, state)['due'], heldout > 0)
                jobs, _ = plan_jobs(config, state, model_id='m'*64, export_hash='a'*64, epoch=generation, ready=True)
                counts = Counter(j['split'] for j in jobs)
                self.assertEqual((counts['train'], counts['heldout']), (train, heldout))
        state = dict(iteration=3271, history=[], signature=signature(self.after))
        self.assertEqual(validation_plan(self.after, state)['reason'], 'every-iteration')


class Buffer(unittest.TestCase):
    def buffer(self, lag):
        from brass_ai.recovery.teacher_buffer import TeacherBuffer
        buffer = TeacherBuffer.__new__(TeacherBuffer)
        buffer.identity, buffer.c_scale, buffer.iterations, buffer.retired = 'id', .2, 8, []
        buffer.mixed_scales, buffer.heldout_lag = None, lag
        buffer.pending = [dict(profile_digest='id', generation=g, split=s, c_scale=.2)
                          for g in range(3260, 3271) for s in ('train', 'heldout')]
        return buffer

    def test_eight_ages_and_validation_labels_train_from_the_next_iteration(self):
        buffer = self.buffer(1)
        keep = buffer.window(3270)
        self.assertEqual(sorted(r['generation'] for r in keep if r['split'] == 'train'), list(range(3263, 3271)))
        self.assertEqual(sorted(r['generation'] for r in keep if r['split'] == 'heldout'), list(range(3262, 3271)))
        trained = {(r['generation'], r['split']): buffer.age(r, 3270) for r in keep if buffer.trains(r, 3270)}
        self.assertEqual(trained[(3270, 'train')], 0)
        self.assertEqual(trained[(3269, 'heldout')], 0)  # age counted from the iteration it starts training
        self.assertEqual(trained[(3262, 'heldout')], 7)
        self.assertNotIn((3270, 'heldout'), trained)  # this iteration's validation labels wait
        self.assertEqual(max(trained.values()), 7)
        old = self.buffer(None); old.iterations = 4
        self.assertFalse(any(old.trains(r, 3270) for r in old.window(3270) if r['split'] == 'heldout'))

    def test_slim_rows_keep_the_label_and_encoding_only(self):
        from brass_ai.recovery.teacher_buffer import slim
        from brass_ai.recovery.teacher_labels import LabelRow
        label = dict(phase='canal_early', target=[1.]); encoded = object()
        row = LabelRow(dict(view=dict(history=[1, 2, 3]), choices=[], teacher=label), encoded)
        small = slim(row)
        self.assertIs(small.label, label); self.assertIs(small.encoded, encoded)
        self.assertNotIn('view', small.row)


def label_row(model, *, seat=0, round=1, width=3, counts=(1, 0, 0), means=(.8, 0., 0.), c2=.1):
    """A b025 fixture label over the first `width` choices of the fixture position."""
    from test_b025 import example
    from brass_ai.recovery.encoding import PreparedEncoder
    from brass_ai.recovery import teacher_labels as L
    from brass_ai.recovery.teacher_term import policy_logits
    row, _ = example(model, seat=seat, round=round)
    choices = row.row['choices'][:width]
    encoded = PreparedEncoder().encode(row.row['view'], choices)
    base = dict(row.row, choices=choices, teacher=dict(row.label, legal_ids=[c['id'] for c in choices]))
    logits = policy_logits(model, [L.LabelRow(base, encoded)])[0].tolist()
    counts, means = list(counts[:width]), list(means[:width])
    target, mixed = L.improved_policy(logits, counts, means, .2, c2=c2)
    label = dict(base['teacher'], logits=logits, counts=counts, mean_returns=means, v_mix=mixed, c2=c2,
                 max_n=max(counts), target=target.astype(np.float32).tolist(),
                 played_action=choices[-1]['id'])
    return L.LabelRow(dict(base, teacher=label), encoded)


class Term(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_b025 import config, model_fixture
        from brass_ai.recovery import training_revision_g1491 as g1491
        cls.model = model_fixture()
        options = copy.deepcopy(config()['teacher_phase'])
        options.update(mode='on', buffer=copy.deepcopy(R.SETTINGS['buffer']))
        options['term']['rows_per_step'] = 4
        options[g1491.KEY] = g1491.block(variant=g1491.SEARCH_REFRESH)
        cls.options = options
        rows = []
        for i in range(20):
            rows.append((i % 8, label_row(cls.model, seat=i % 4, round=1 + i % 8, width=2 + i % 2,
                                          counts=(2 + i % 3, 1, 0), means=(.6, .1 * (i % 5), 0.))))
        cls.rows = rows

    def term(self, rows=None):
        from brass_ai.recovery.teacher_term import TeacherTerm
        return TeacherTerm(self.model, self.options, self.rows if rows is None else rows, seed=5, generation=3270,
                           state=dict(beta=1., calibrated=True), revision=R.term_settings(dict(teacher_revision=1)))

    def test_the_bonus_rebuilds_the_exported_target_from_the_stored_prior(self):
        from brass_ai.recovery.teacher_labels import softmax
        from brass_ai.recovery.teacher_term import search_bonus
        for _, row in self.rows:
            label = row.label
            rebuilt = softmax(np.asarray(label['logits']) + search_bonus(label, label['c2']))
            np.testing.assert_allclose(rebuilt, label['target'], atol=2e-6)
        # At the .20 label scale the same statistics give a sharper push than the stored .10 target.
        label = self.rows[0][1].label
        sharper = softmax(np.asarray(label['logits']) + search_bonus(label, .2))
        self.assertGreater(sharper.max(), max(label['target']))

    def test_the_loss_is_the_kl_from_the_target_rebuilt_on_the_current_policy(self):
        from brass_ai.recovery import teacher_term as M
        from brass_ai.recovery.value_preservation import TrunkGradients
        term = self.term(); term.plan(3)
        indices = term.batches[0]; selected = [term.rows[i] for i in indices]
        parameters = tuple(p for n, p in self.model.named_parameters() if n.startswith(M.ACTOR_PREFIXES) and p.requires_grad)
        with M.policy_context(self.model):
            logits = M.forward_policy(self.model, selected).float()
            bonus = M.target_tensor(selected, logits, [term.bonus[i] for i in indices])
            target = (logits.detach() + bonus).softmax(-1)
            logp = logits.log_softmax(-1).masked_fill(~torch.isfinite(logits), 0.)
            formula = (torch.special.xlogy(target, target) - target * logp).sum(-1).mean()
            expected = torch.autograd.grad(formula, parameters, allow_unused=True)
        self.model.zero_grad(set_to_none=True)
        term.apply(self.model, TrunkGradients(self.model), 1)
        self.assertAlmostEqual(term.steps[-1]['loss'], float(formula.detach()), places=5)
        for p, g in zip(parameters, expected):
            if g is None: continue
            np.testing.assert_allclose(p.grad.detach().numpy(), g.numpy(), rtol=1e-4, atol=1e-6)
        self.model.zero_grad(set_to_none=True)
        self.assertEqual(term.report()['revision']['targets'], 'rebuilt-from-current-policy')

    def test_batches_draw_each_row_once_grouped_by_legal_width(self):
        from brass_ai.recovery.value_preservation import TrunkGradients
        term = self.term(); term.plan(3)
        drawn = [i for b in term.batches for i in b]
        self.assertEqual((len(term.batches), len(drawn), len(set(drawn))), (3, 12, 12))
        widths = sorted(len(term.rows[i].label['legal_ids']) for i in drawn)
        mixed = sum(len({len(term.rows[i].label['legal_ids']) for i in b}) > 1 for b in term.batches)
        self.assertLessEqual(mixed, 1)  # sorted by width before the cut: at most one batch spans two widths
        again = self.term(); again.plan(3)
        self.assertEqual(again.batches, term.batches)  # fixed by seed and generation
        for step in (1, 2, 3):
            term.apply(self.model, TrunkGradients(self.model), step); self.model.zero_grad(set_to_none=True)
        report = term.report()
        self.assertEqual((report['presentations'], report['distinct_rows'], report['revision']['repeated_rows']), (12, 12, 0))
        self.assertEqual(report['revision']['plan']['drawn_rows'], 12)
        # A pool smaller than the update's demand: every row once, in smaller batches; later steps get none.
        small = self.term(); small.plan(10)
        drawn = [i for b in small.batches for i in b]
        self.assertEqual((sorted(drawn), small.plan_report['batch_rows']), (list(range(20)), 2))
        small.apply(self.model, TrunkGradients(self.model), 11); self.model.zero_grad(set_to_none=True)
        self.assertEqual(small.steps, [])
        self.assertEqual(widths, sorted(widths))

    def test_staleness_reads_a_fixed_sample_of_each_age(self):
        from brass_ai.recovery.teacher_term import staleness_sample
        rows = list(range(1000))
        a = staleness_sample(rows, 256, seed=1, generation=3270, age=3)
        self.assertEqual(len(a), 256); self.assertEqual(a, sorted(a))
        self.assertEqual(a, staleness_sample(rows, 256, seed=1, generation=3270, age=3))
        self.assertNotEqual(a, staleness_sample(rows, 256, seed=1, generation=3270, age=4))
        self.assertEqual(staleness_sample(rows[:10], 256, seed=1, generation=3270, age=3), rows[:10])
        term = self.term()
        self.assertEqual(sum(s['rows'] for s in term.staleness.values()), 20)
        self.assertTrue(all('sampled' in s for s in term.staleness.values()))

    def test_the_ppo_update_plans_the_batches_for_its_own_steps(self):
        import random
        from fixtures import game
        from test_b025 import current_board
        from brass_ai.recovery.config import UPDATE_V21
        from brass_ai.recovery.teacher_term import TeacherTerm
        from brass_ai.recovery.updates import ppo_update
        from brass_ai.step5.policy import NeuralPolicy
        model = copy.deepcopy(self.model); policy = NeuralPolicy(model, focus='neutral', exploration=.05)
        self.addCleanup(policy.close)
        g = game(); g['split'] = 'train'
        for i, row in enumerate(g['records']):
            current_board(row['view'])
            index, p, _ = policy.select_batch([(row['view'], row['choices'], random.Random(i))])[0]
            row.update(chosen=index, behavior=dict(kind='model', policy_id=policy.policy_id, probability=p, temperature=1.))
        cfg = dict(UPDATE_V21, epochs=2, effective_batch_size=2, critic_batch_size=2, microbatch_size=2, target_kl=10.)
        options = dict(copy.deepcopy(self.options), mode='shadow')
        term = TeacherTerm(model, options, self.rows, seed=5, generation=3270, revision=R.term_settings(dict(teacher_revision=1)))
        report = ppo_update(model, torch.optim.AdamW(model.parameters(), lr=1e-5), [g], policy, cfg, teacher_term=term)
        plan = term.plan_report
        self.assertEqual(plan['planned_steps'], report['optimizer_steps'])
        out = term.report()
        self.assertEqual(out['revision']['repeated_rows'], 0)
        self.assertEqual(out['presentations'], min(plan['demand'], plan['pool_rows']))
        self.assertEqual(len(term.steps), min(report['optimizer_steps'], plan['batches']))

    def test_movement_is_measured_on_the_same_rows(self):
        from brass_ai.recovery.teacher_term import movement
        rows = [row for _, row in self.rows[:4]]
        before = [np.asarray(r.label['logits']) for r in rows]
        self.assertEqual(movement(rows, before, before)['all'], 0.)
        after = [l + np.arange(len(l)) for l in before]
        self.assertGreater(movement(rows, before, after)['all'], 0.)


def heldout_game(generation, game, split='heldout'):
    return [dict(split=split, generation=generation, game_id=f'teacher/{generation}-{game}', index=i, u=[0.]*4)
            for i in range(1, 65)]


class SearchValue(unittest.TestCase):
    def test_the_held_out_set_fills_to_32_then_retires_into_training(self):
        from brass_ai.recovery.search_value_retention import rotating_holdout
        games = {g: [heldout_game(g, k) for k in range(4)] for g in (3254, 3259, 3264, 3269, *range(3270, 3282))}
        seen = {}
        for generation in range(3270, 3282):
            rows = [r for g, group in games.items() if g <= generation and generation - g < 20 for game in group for r in game]
            validation, retired, report = rotating_holdout(rows, generation, games=32, minimum=16, train_window=16)
            seen[generation] = (report['available_games'], len({r['game_id'] for r in validation}),
                                report['retired_training_games'], report['retired_generations'])
            self.assertTrue(all(r['split'] == 'train' and r['source_split'] == 'heldout' for r in retired))
            self.assertTrue({r['game_id'] for r in validation}.isdisjoint(r['game_id'] for r in retired))
            latest = sorted({(r['generation'], r['game_id']) for r in validation})
            self.assertEqual(latest[-1][0], generation)
        self.assertEqual(seen[3270][:3], (20, 20, 0))   # the 16 held-out games plus this iteration's 4
        self.assertEqual(seen[3273][:3], (32, 32, 0))
        self.assertEqual(seen[3275][:3], (36, 32, 0))   # g3259 leaves at age 16: too old to train
        self.assertEqual(seen[3276][1:], (32, 4, [3264]))
        self.assertEqual(seen[3277][1:], (32, 8, [3264, 3269]))
        self.assertEqual(seen[3281][1:], (32, 20, [3269, 3270, 3271, 3272, 3273]))  # ages 8-12 of the 16-iteration buffer
        empty = rotating_holdout([r for g in (3269,) for game in games[g] for r in game], 3270, games=32, minimum=16)
        self.assertEqual((empty[0], empty[2]['ready']), ([], False))

    def test_compact_states_feed_identical_float32_values(self):
        from brass_ai.recovery.search_value_rows import ValueBuffer, compact
        state = [float(i) / 7 for i in range(3128)]
        row = compact(dict(state=state, u=[1.]*4))
        self.assertEqual(row['state'].dtype, np.float32); self.assertFalse(row['state'].flags.writeable)
        np.testing.assert_array_equal(np.asarray([row['state']]), np.asarray([state], dtype=np.float32))
        np.testing.assert_array_equal(torch.as_tensor(np.asarray([row['state']]), dtype=torch.float32).numpy(),
                                      torch.as_tensor(np.asarray([state]), dtype=torch.float32).numpy())
        with tempfile.TemporaryDirectory() as directory:
            buffer = ValueBuffer(Path(directory)/'values', 'r', train_iterations=16, heldout_iterations=20)
            self.assertEqual(buffer.windows, dict(train=16, heldout=20))
            published = dict(generation=5, split='train', reward_id='r', game_id='teacher/x', index=1, state=state, u=[0.]*4)
            buffer.publish(5, 'train', [published], game_id='teacher/x')
            cached = buffer.rows(5, 'train')
            self.assertEqual(cached[0]['state'].dtype, np.float32)
            self.assertEqual(json.loads(__import__('gzip').decompress((Path(directory)/'values'/buffer.receipts[0]['file']).read_bytes()))[0]['state'], state)
            with self.assertRaisesRegex(ValueError, 'training window'):ValueBuffer(Path(directory)/'v2', 'r', train_iterations=12)


CHILD = textwrap.dedent('''
    import json, struct, sys, time
    n = struct.unpack('<I', sys.stdin.buffer.read(4))[0]; json.loads(sys.stdin.buffer.read(n))
    out = sys.stdout.buffer
    def frame(meta):
        m = json.dumps(meta).encode(); body = struct.pack('<I', len(m)) + m
        return struct.pack('<I', len(body)) + body
    out.write(frame(dict(event='hello'))); out.flush()
    time.sleep(.2)
    began = time.perf_counter()
    big = frame(dict(event='turn', index=1, blob='x' * (3 * 1024 * 1024)))
    out.write(big[:len(big) // 2]); out.flush()
    out.write(big[len(big) // 2:]); out.flush()
    out.write(frame(dict(event='finished', write_seconds=time.perf_counter() - began))); out.flush()
''')


class Spool(unittest.TestCase):
    def worker(self, script, stream=True):
        from brass_ai.rust_backend import lab_teacher as T
        real = subprocess.Popen
        def popen(args, **kwargs): return real([sys.executable, '-c', script], **kwargs)
        request = dict(job={}, turn_handoff=T.STREAM_TURNS if stream else None)
        with patch.object(T.subprocess, 'Popen', popen):
            return T.Worker('teacher', request, None, threading.Event())

    def events(self, worker, count, timeout=30):
        out = []
        while len(out) < count:
            event = worker.events.get(timeout=timeout)
            if isinstance(event, BaseException): raise event
            out.append(event[0])
        return out

    def test_a_streamed_table_never_waits_for_a_busy_trainer(self):
        data = np.random.default_rng(1).random(3_000_000).tolist()
        worker = self.worker(CHILD)
        try:
            self.assertIsNotNone(worker.spool)
            self.assertEqual(self.events(worker, 1)[0]['event'], 'hello')
            held = time.perf_counter(); sorted(data); held = time.perf_counter() - held  # one C call: no thread runs
            events = self.events(worker, 2)
            self.assertEqual([e['event'] for e in events], ['turn', 'finished'])
            self.assertEqual(len(events[0]['blob']), 3 * 1024 * 1024)
            if held > .5: self.assertLess(events[1]['write_seconds'], held / 2)
            path = worker.spool.path
        finally:
            worker.close()
        self.assertFalse(os.path.exists(path))

    def test_the_pipe_path_and_a_truncated_output(self):
        with patch.dict(os.environ, {'BRASS_TEACHER_SPOOL': '0'}):
            worker = self.worker(CHILD)
        try:
            self.assertIsNone(worker.spool)
            self.assertEqual([e['event'] for e in self.events(worker, 3)], ['hello', 'turn', 'finished'])
        finally:
            worker.close()
        truncated = CHILD.replace("out.write(big[len(big) // 2:]); out.flush()", "raise SystemExit(0)")
        worker = self.worker(truncated)
        try:
            self.assertEqual(self.events(worker, 1)[0]['event'], 'hello')
            with self.assertRaisesRegex(ValueError, 'closed output'):self.events(worker, 1)
        finally:
            worker.close()


class Collection(unittest.TestCase):
    def test_the_gc_window_freezes_and_times_collector_passes(self):
        from brass_ai.recovery.collection_gc import Window
        keep = [[i] for i in range(10000)]
        window = Window(); window.start()
        try:
            self.assertGreater(gc.get_freeze_count(), 0)
            gc.collect()
        finally:
            report = window.stop()
        self.assertEqual(gc.get_freeze_count(), 0)
        self.assertTrue(report['freeze']); self.assertGreaterEqual(report['frozen_objects'], len(keep))
        self.assertGreaterEqual(report['collections'].get('2', 0), 1)
        with patch.dict(os.environ, {'BRASS_COLLECTION_GC_FREEZE': '0'}):
            window.start(); report = window.stop()
        self.assertFalse(report['freeze']); self.assertEqual(gc.get_freeze_count(), 0)
        self.assertNotIn(window._callback, gc.callbacks)

    def test_parity_checks_a_fixed_sample_over_the_phases(self):
        from brass_ai.rust_backend.lab_teacher_finalization import parity_sample
        class Row:
            def __init__(self, i, phase): self.identity, self.label = f'{i:03d}', dict(phase=phase)
        phases = ('canal_early', 'canal_late', 'rail_early', 'rail_late')
        rows = [Row(i, phases[i % 4]) for i in range(31)]
        picked = parity_sample(rows, 8)
        self.assertEqual(len(picked), 8)
        self.assertEqual(sorted(r.label['phase'] for r in picked), sorted(phases * 2))
        self.assertEqual(picked, parity_sample(list(reversed(rows)), 8))
        self.assertEqual(parity_sample(rows, None), rows)


if __name__ == '__main__':
    unittest.main()
