"""B02.5 / T1 bounded fixture qualification. No native games or training run."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read, sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint, resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, digest
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery import teacher_phase_config as C, teacher_labels as L
from brass_ai.recovery.teacher_buffer import TeacherBuffer, sample_indices
from brass_ai.recovery.teacher_phase import TeacherPhase, readiness, plan_jobs, collection_order
from brass_ai.recovery.teacher_term import TeacherTerm, policy_logits, forward_policy
from brass_ai.recovery.config import TEACHER_VALUE, UPDATE_V21, validate_config
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature, load_resume, commit_transition, teacher_receipt_preserved
from brass_ai.recovery.updates import ppo_update
from brass_ai.recovery.value_preservation import TrunkGradients
from brass_ai.step5.league import implementation_id, write_json
from brass_ai.step5.policy import NeuralPolicy
from fixtures import position, game

BASE_IMPLEMENTATION = '603d1d236ca3da4511467956a9b656b41a4f03017d814b1abd8f060467a4c05c'


def config():
    c = read(SOURCE_ROOT / 'results/joint7-v11-2-grouped-lr-critic-g1320-20261027T075134827445Z/recorded-config.json')
    c['model_size'] = 'small'
    return C.adopt(c)


def model_fixture():
    torch.manual_seed(924)
    return UtilityNet(utility_config(ModelConfig.preset('small'), config()['teacher_phase']['search']['reward']))


def current_board(view):
    from brass_ai.learning.board_topology import topology
    from brass_ai.learning.features import feature_schema
    board = topology(); static = feature_schema()['static_profile']
    for link in view['links']:
        link['regions'] = list(board['link_endpoints'][link['id']]); link['era'] = board['link_eras'][link['id']]
    for slot in view['slots']: slot['types'] = static['slot_types'][slot['id']]
    for merchant in view['merchants']: merchant['bonus'] = static['merchant_bonus'][merchant['id']]
    return view


def example(model, *, generation=10, seat=0, kind='root', job=None, round=1):
    view, choices = position(seat); current_board(view); view['context']['round'] = round
    identity = dict(job_id='fixture-job', game_id='fixture-game', generation=generation,
        model_id=model_id(model), export_hash='a' * 64, reward_id=digest(model.config.utility),
        source_fingerprint=C.SOURCE, epoch=7) if job is None else copy.deepcopy(job['expected'])
    encoded = PreparedEncoder().encode(view, choices)
    row = dict(seat=seat, view=view, choices=choices, teacher=dict(legal_ids=[c['id'] for c in choices]))
    logits = policy_logits(model, [L.LabelRow(row, encoded)])[0].tolist()
    counts, means = [1, 0, 0], [.8, 0., 0.]
    target, mixed = L.improved_policy(logits, counts, means, .2)
    label = dict(identity, source=C.LABEL_SOURCE, method='search_v1', algorithm=C.ALGORITHM,
        session=C.SESSION, v205_enabled=False, seat=seat,
        turn_key=[view['context'][k] for k in ('era', 'round', 'turn', 'turn_owner')],
        revision=view['context']['revision'], kind=kind, phase='canal_early' if round <= 4 else 'canal_late',
        legal_ids=[c['id'] for c in choices], logits=logits, counts=counts, mean_returns=means,
        node_value=.2, v_mix=mixed, c1=50, c2=.1, max_n=1, target=target.astype(np.float32).tolist(),
        requested_k=16, effective_k=2, frozen_n=100, completed_trajectories=1, node_trajectories=1,
        completed_rounds=0, deadline_hit=True, overrun_seconds=.7, played_action=choices[-1]['id'],
        plan_supported=False, fallback_played=True, gumbel_draw_id='fixture-draw', cancelled=False)
    row['teacher'] = label
    return L.LabelRow(row, encoded), identity


def admit(model, row, expected):
    return L.admit_rows([row.row], model=model, expected=expected,
        history=[dict(generation=expected['generation'], model_id=expected['model_id'])], admitted_sources={C.SOURCE})[0]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def temp_run(self):
        temporary = tempfile.TemporaryDirectory(prefix='b025-', dir=SOURCE_ROOT / 'artifacts')
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)


class Contracts(Base):
    def test_defaults_historical_validation_and_strict_settings(self):
        c = config(); validate_config(c)
        self.assertEqual(c['teacher_phase']['mode'], 'off'); self.assertFalse(TEACHER_VALUE['enabled'])
        old = copy.deepcopy(c); old.pop('teacher_phase'); old['teacher_value']['enabled'] = True
        validate_config(old)  # Saved historical recipes do not change.
        for path, value in [('teacher_enabled', True), ('teacher_phase.mode', 'auto'),
                ('teacher_phase.search.k', 8), ('teacher_phase.search.pending_paths', 2),
                ('teacher_phase.search.v205_enabled', True), ('teacher_phase.search.method', 'search_v2'),
                ('teacher_phase.search.reuse', True), ('teacher_phase.buffer.age_weights', [1, 2, 3, 4]),
                ('teacher_phase.buffer.age_weights', [4, 3, 0, 1]), ('teacher_phase.term.share_target', 0),
                ('teacher_phase.term.rows_per_step', c['update']['effective_batch_size'] + 1)]:
            bad = copy.deepcopy(c); keys = path.split('.'); target = bad
            for k in keys[:-1]: target = target[k]
            target[keys[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError): validate_config(bad)
        c['teacher_phase']['mode'] = 'shadow'; c['teacher_value']['enabled'] = True
        with self.assertRaises(ValueError): validate_config(c)

    def test_toggle_and_invalidation_boundaries(self):
        a = config(); b = copy.deepcopy(a); b['teacher_phase']['mode'] = 'on'
        b['teacher_phase']['term']['share_target'] = .4
        self.assertTrue(C.toggle_compatible(a, b)); self.assertEqual(C.buffer_identity(a), C.buffer_identity(b))
        for change in ('learning_rate', 'teacher_value'):
            bad = copy.deepcopy(b)
            if change == 'learning_rate': bad[change] *= .5
            else: bad[change]['diagnostics'] = True
            self.assertFalse(C.toggle_compatible(a, bad))
        b['teacher_phase']['term']['rows_per_step'] = 32
        self.assertFalse(C.toggle_compatible(a, b))
        b = copy.deepcopy(a); b['learning_rate'] *= .5
        self.assertEqual(C.buffer_identity(a), C.buffer_identity(b))

    def test_equations_reference_zero_evidence_and_all_legal(self):
        l = [math.log(.2), math.log(.3), math.log(.5)]
        p, mixed = L.improved_policy(l, [2, 1, 0], [.2, .8, 0.], .6)
        self.assertAlmostEqual(mixed, .57)
        np.testing.assert_allclose(p, L.softmax(np.array(l) + 5.2 * np.array([.2, .8, .57])), atol=1e-15)
        np.testing.assert_allclose(L.improved_policy(l, [0, 0, 0], [0., 0., 0.], .6)[0], [.2, .3, .5])
        p, _ = L.improved_policy([0., .2, -.3, .5], [1, 0, 0, 0], [1., 0., 0., 0.], 0.)
        self.assertAlmostEqual(math.log(p[1] / p[2]), .5)
        spec = importlib.util.spec_from_file_location('teacher_gumbel_reference', SOURCE_ROOT / 'scripts/reproduce-gumbel-is-mcts.py')
        import sys
        ref = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ref; spec.loader.exec_module(ref)
        node = ref.Node(ref.Info(2, (), (), (0, 1, 2)), tuple(l), (9., 8., .6, 7.), 4)
        node.count[:] = [2, 1, 0]; node.returns[:] = [[100., -100., .4, 50.], [100., -100., .8, 50.], [0.] * 4]
        np.testing.assert_allclose(L.softmax(node.scores(50, .1)), L.improved_policy(l, node.count, [.2, .8, 0.], .6)[0])
        trace = read(SOURCE_ROOT / 'tests/GumbelISMCTS/fixtures/reference-v1.json')
        for n in trace['nodes']:
            actor = n['info']['player']; counts = n['count']
            means = [total[actor] / count if count else 0. for count, total in zip(counts, n['returns'])]
            target, mixed = L.improved_policy(n['logits'], counts, means, n['value'][actor])
            np.testing.assert_allclose(target, n['improved_policy'], atol=1e-14)
            self.assertAlmostEqual(mixed, n['v_mix'])

    def test_partial_round_overrun_fallback_and_single_visit_admitted(self):
        model = model_fixture()
        for kind in L.KINDS:
            row, expected = example(model, kind=kind)
            self.assertEqual(len(admit(model, row, expected)), 1)
        for why in ('cancelled', 'no_evidence'):
            row, expected = example(model)
            if why == 'cancelled': row.label['cancelled'] = True
            else:
                row.label.update(counts=[0, 0, 0], max_n=0, node_trajectories=0,
                    target=L.softmax(row.label['logits']).tolist(), v_mix=.2)
            self.assertEqual(admit(model, row, expected), [])

    def test_tampered_identity_legal_observer_target_and_logits_rejected(self):
        model = model_fixture(); row, expected = example(model)
        changes = dict(target=[1., 0., 0.], legal_ids=list(reversed(row.label['legal_ids'])), seat=1,
            reward_id='wrong', model_id='wrong', epoch=8, source_fingerprint='wrong', export_hash='wrong',
            revision=99, counts=[True, 0, 0], node_value=float('nan'), logits=[float('inf'), 0., 0.])
        for key, value in changes.items():
            bad = copy.deepcopy(row); bad.label[key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)): admit(model, bad, expected)
        bad = copy.deepcopy(row); bad.label['logits'][0] += .01
        target, mixed = L.improved_policy(bad.label['logits'], bad.label['counts'], bad.label['mean_returns'], .2)
        bad.label.update(target=target.tolist(), v_mix=mixed)
        with self.assertRaisesRegex(ValueError, 'logits parity'): admit(model, bad, expected)

    def test_readiness_null_false_reward_and_seed_isolation(self):
        ready = dict(ready=True, reward_id='r')
        receipt = ready
        for _ in range(67): receipt, ok = readiness(receipt, None, 'r'); self.assertTrue(ok)
        receipt, ok = readiness(receipt, dict(ready=False, reward_id='r'), 'r'); self.assertFalse(ok)
        self.assertFalse(readiness(receipt, None, 'r')[1]); self.assertFalse(readiness(ready, None, 'new')[1])
        c = config(); c['teacher_phase']['mode'] = 'shadow'; state = dict(iteration=1324, next_seed=12345)
        original = copy.deepcopy(state)
        jobs, cursor = plan_jobs(c, state, model_id='m', export_hash='a' * 64, epoch=2, ready=True)
        self.assertEqual(state, original); self.assertEqual(len(jobs), 32)
        self.assertEqual(sum(j['split'] == 'train' for j in jobs), 28)
        self.assertEqual(len({j['seed'] for j in jobs}), 32)
        ordinary = [dict(seed=5), dict(seed=6)]; self.assertEqual(collection_order(jobs, ordinary)[-2:], ordinary)
        self.assertEqual((jobs, cursor), plan_jobs(c, state, model_id='m', export_hash='a' * 64, epoch=2, ready=True))
        c['teacher_phase']['mode'] = 'off'
        self.assertEqual(plan_jobs(c, state, model_id='m', export_hash='a' * 64, epoch=2, ready=True)[0], [])


class Buffer(Base):
    def test_weighted_draws_independent_rng_and_missing_ages(self):
        before = torch.get_rng_state().clone(); counts = np.zeros(4)
        for step in range(3000): counts[sample_indices([0, 1, 2, 3], [4, 3, 2, 1], 1, seed=7, generation=4, step=step)[0]] += 1
        np.testing.assert_allclose(counts / counts.sum(), [.4, .3, .2, .1], atol=.035)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        a = sample_indices([0, 2, 2], [4, 3, 2, 1], 64, seed=7, generation=4, step=2)
        self.assertEqual(len(a), len(set(a))); self.assertEqual(a, sample_indices([0, 2, 2], [4, 3, 2, 1], 64, seed=7, generation=4, step=2))

    def test_seals_resume_window_commit_cleanup_and_unknown_files(self):
        run = self.temp_run(); m = model_fixture(); c = config(); b = TeacherBuffer(SOURCE_ROOT, run, C.buffer_identity(c))
        for generation in range(10, 15):
            r, expected = example(m, generation=generation)
            b.publish(generation, 'train', admit(m, r, expected), model_id=expected['model_id'], reward_id=expected['reward_id'])
        unknown = b.directory / 'keep.txt'; unknown.write_text('owned by user', encoding='utf-8')
        self.assertEqual([a for a, _ in b.rows(14)], [0, 1, 2, 3])
        old = b.directory / 'generation-000010-train.jsonl.gz'; self.assertTrue(old.exists())
        with self.assertRaises((ValueError, FileNotFoundError)): b.committed(15)
        payload = b.payload(); write_json(run / 'state.json', dict(iteration=15, teacher_phase=dict(buffer=payload)))
        b.committed(15); self.assertFalse(old.exists()); self.assertTrue(unknown.exists())
        rebuilt = TeacherBuffer(SOURCE_ROOT, run, C.buffer_identity(c), receipts=payload['receipts'], retired=payload['retired'])
        self.assertEqual([r.identity for _, r in b.rows(14)], [r.identity for _, r in rebuilt.rows(14)])
        (run / payload['receipts'][0]['path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'missing/changed'):
            TeacherBuffer(SOURCE_ROOT, run, C.buffer_identity(c), receipts=payload['receipts'])

    def test_invalidation_and_heldout_never_trains(self):
        run = self.temp_run(); m = model_fixture(); b = TeacherBuffer(SOURCE_ROOT, run, 'old')
        r, e = example(m); rows = admit(m, r, e)
        for split in ('train', 'heldout'): b.publish(10, split, rows, model_id=e['model_id'], reward_id=e['reward_id'])
        self.assertEqual(len(b.rows(10)), 1)
        new = TeacherBuffer(SOURCE_ROOT, run, 'new-reward-or-schema', receipts=b.payload()['receipts'])
        self.assertEqual(new.rows(10), [])
        self.assertEqual(len(new.payload()['retired']), 2)

    def test_staleness_skips_age_and_beta_calibration_share_scaling(self):
        m = model_fixture(); r, _ = example(m); c = config()['teacher_phase']; c['mode'] = 'shadow'
        # Stored prior equals target: zero correction, so any positive drift expires this age.
        stale = copy.deepcopy(r); stale.label['logits'] = [10., -10., -10.]
        stale.label['target'] = L.softmax(stale.label['logits']).tolist()
        term = TeacherTerm(m, c, [(0, r), (3, stale)], seed=1, generation=10)
        self.assertEqual(term.ages, [0]); self.assertTrue(term.staleness['3']['skipped'])
        term.steps = [dict(raw_share=2., share=2.)]; state = term.finish()
        c['mode'] = 'on'; term = TeacherTerm(m, c, [(0, r)], seed=1, generation=11, state=state)
        self.assertAlmostEqual(term.beta, .125); state = term.finish()
        c['term']['share_target'] = .4
        self.assertAlmostEqual(TeacherTerm(m, c, [], seed=1, generation=12, state=state).beta, .2)


class PPO(Base):
    def fixture(self, mode):
        m = model_fixture(); policy = NeuralPolicy(m, focus='neutral', exploration=.05)
        self.addCleanup(policy.close)
        g = game(); g['split'] = 'train'
        for i, row in enumerate(g['records']):
            current_board(row['view'])
            index, p, _ = policy.select_batch([(row['view'], row['choices'], random.Random(i))])[0]
            row.update(chosen=index, behavior=dict(kind='model', policy_id=policy.policy_id, probability=p, temperature=1.))
        cfg = dict(UPDATE_V21, epochs=2, effective_batch_size=2, critic_batch_size=2, microbatch_size=2)
        options = config()['teacher_phase']; options['mode'] = mode
        rows = [example(m, seat=s)[0] for s in range(4)]
        term = TeacherTerm(m, options, [(0, r) for r in rows], seed=1, generation=10)
        return m, policy, g, cfg, term

    def test_shadow_is_bit_exact_weights_adam_rng_and_loss(self):
        m, p, g, cfg, term = self.fixture('shadow'); other = copy.deepcopy(m)
        a = torch.optim.AdamW(m.parameters(), lr=1e-5); b = torch.optim.AdamW(other.parameters(), lr=1e-5)
        rng = torch.get_rng_state().clone()
        baseline = ppo_update(m, a, [g], p, cfg); after = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        measured = ppo_update(other, b, [g], p, cfg, teacher_term=term)
        self.assertTrue(_same_payload(m.state_dict(), other.state_dict()))
        self.assertTrue(_same_payload(a.state_dict(), b.state_dict())); self.assertTrue(torch.equal(after, torch.get_rng_state()))
        self.assertEqual(baseline['epochs'], measured['epochs']); self.assertEqual(len(term.steps), measured['optimizer_steps'])

    def test_policy_gradient_excludes_critic_and_descriptor_heads(self):
        m = model_fixture(); row, _ = example(m); c = config()['teacher_phase']; c['mode'] = 'on'
        term = TeacherTerm(m, c, [(0, row)], seed=1, generation=10)
        actor = TrunkGradients(m); term.apply(m, actor, 1)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.state_encoder.parameters()))
        for module in (m.outcomes, m.utility_head, m.descriptor_encoder):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        # The companion local-scoring heads are not passed to this module at all.
        self.assertTrue(any(p.grad is not None for p in m.policy_scorer.parameters()))

    def test_on_uses_existing_optimizer_and_kl_stop(self):
        m, p, g, cfg, term = self.fixture('on')
        cfg.update(epochs=5, target_kl=1e-12)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        report = ppo_update(m, opt, [g], p, cfg, teacher_term=term)
        self.assertTrue(report['kl_early_stop']); self.assertEqual(report['kl_guard']['probe_checks'], report['optimizer_steps'])
        self.assertEqual(len(term.steps), report['optimizer_steps'])
        self.assertEqual({int(s['step']) for s in opt.state.values()}, {report['optimizer_steps']})

    def test_teacher_rows_cannot_enter_ordinary_ppo(self):
        m, p, g, cfg, term = self.fixture('off'); g['kind'] = 'teacher_phase'
        with self.assertRaisesRegex(ValueError, 'Teacher tables'): ppo_update(m, torch.optim.AdamW(m.parameters()), [g], p, cfg)

    def test_teacher_cannot_update_without_an_ordinary_kl_probe(self):
        m, p, g, cfg, term = self.fixture('on')
        for r in g['records']:
            r['choices'] = r['choices'][:1]; r['chosen'] = 0; r['behavior']['probability'] = 1.
        with self.assertRaisesRegex(ValueError, 'nonforced ordinary PPO KL probe'):
            ppo_update(m, torch.optim.AdamW(m.parameters()), [g], p, cfg, teacher_term=term)

    def test_dcpg_shadow_preserves_shared_encoder_gradients(self):
        from test_ppg_eight import setup, collect
        from brass_ai.recovery import dcpg_config
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.utility import reward_spec
        m, opt, cfg = setup(); cfg = dcpg_config.configure(cfg)
        cfg['update'].update(epochs=1, effective_batch_size=128, critic_batch_size=128)
        other = copy.deepcopy(m); second = torch.optim.AdamW(other.parameters()); second.load_state_dict(copy.deepcopy(opt.state_dict()))
        a = TrainingPhases(m, cfg, root=SOURCE_ROOT, run=self.temp_run())
        b = TrainingPhases(other, cfg, root=SOURCE_ROOT, run=self.temp_run())
        behavior, data = collect(m, a, 1); b.snapshot_data = copy.deepcopy(a.snapshot_data); b.collection_source = copy.deepcopy(a.collection_source)
        options = C.defaults(m.config.utility); options['mode'] = 'shadow'
        term = TeacherTerm(other, options, [(0, example(other, seat=s)[0]) for s in range(4)], seed=1, generation=10)
        reward = reward_spec(cfg); kwargs = dict(learner_data=data, insolvency=reward['insolvency'], reward=reward)
        rng = torch.get_rng_state().clone()
        first = ppo_update(m, opt, [dict(split='train')], behavior, cfg['update'], phases=a, **kwargs)
        after = torch.get_rng_state().clone(); torch.set_rng_state(rng)
        last = ppo_update(other, second, [dict(split='train')], behavior, cfg['update'], phases=b, teacher_term=term, **kwargs)
        self.assertTrue(_same_payload(m.state_dict(), other.state_dict()))
        self.assertTrue(_same_payload(opt.state_dict(), second.state_dict())); self.assertTrue(torch.equal(after, torch.get_rng_state()))
        self.assertEqual(first['value_preservation']['gradients'], last['value_preservation']['gradients'])
        self.assertEqual(len(term.steps), last['optimizer_steps'])

    def test_full_checkpoint_restart_reproduces_combined_update(self):
        m, p, g, cfg, term = self.fixture('on'); opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
        ppo_update(m, opt, [g], p, cfg, teacher_term=term); weight = term.finish()
        run = self.temp_run(); path = run / 'fixture.pt'
        save_checkpoint(path, m, opt, step=11, training=dict(teacher_phase_weight=weight))
        other, raw = load_checkpoint(path); second = torch.optim.AdamW(other.parameters(), lr=1e-5)
        resume_optimizer(raw, second)
        # Build the next frozen ordinary replay once, then use identical fixed data.
        policy = NeuralPolicy(m, focus='neutral', exploration=.05)
        self.addCleanup(policy.close)
        for i, row in enumerate(g['records']):
            index, prob, _ = policy.select_batch([(row['view'], row['choices'], random.Random(i))])[0]
            row.update(chosen=index, behavior=dict(kind='model', policy_id=policy.policy_id, probability=prob, temperature=1.))
        options = config()['teacher_phase']; options['mode'] = 'on'
        rows = [(0, example(m, generation=11, seat=s)[0]) for s in range(4)]
        a = TeacherTerm(m, options, rows, seed=1, generation=11, state=weight)
        b = TeacherTerm(other, options, rows, seed=1, generation=11, state=raw['training']['teacher_phase_weight'])
        rng = torch.get_rng_state().clone(); ppo_update(m, opt, [g], policy, cfg, teacher_term=a); after = torch.get_rng_state().clone()
        torch.set_rng_state(rng); ppo_update(other, second, [g], policy, cfg, teacher_term=b)
        self.assertTrue(_same_payload(m.state_dict(), other.state_dict()))
        self.assertTrue(_same_payload(opt.state_dict(), second.state_dict())); self.assertTrue(torch.equal(after, torch.get_rng_state()))
        self.assertEqual(a.finish(), b.finish())


class Resume(Base):
    def signature(self, config, implementation=None):
        return dict(config=config, implementation=implementation or implementation_id(),
                    compatibility=compatibility(), legacy_pool={}, teachers={})

    def test_exact_adoption_and_reviewed_self_toggle(self):
        old = config(); old.pop('teacher_phase'); old['teacher_value']['enabled'] = True
        before = self.signature(old, BASE_IMPLEMENTATION); after = self.signature(C.adopt(old))
        state = dict(format='step5-recovery-league/2.2.0', iteration=1324, signature=before)
        self.assertTrue(check_signature(state, after, True))
        with self.assertRaises(ValueError): check_signature(state, after)
        state['signature'] = after; toggled = copy.deepcopy(after)
        toggled['config']['teacher_phase']['mode'] = 'on'
        self.assertTrue(check_signature(state, toggled, True))
        toggled['config']['learning_rate'] *= .9
        with self.assertRaises(ValueError): check_signature(state, toggled, True)

    def test_receipt_preserved_through_toggle_crash_chain_only(self):
        before = self.signature(config()); after = copy.deepcopy(before)
        after['config']['teacher_phase']['mode'] = 'shadow'
        entry = dict(model_id='model', generation=10)
        raw = dict(training=dict(signature=digest(before)))
        state = dict(signature=before)
        self.assertTrue(teacher_receipt_preserved(state, after, raw, entry))
        state = dict(signature=after, source_transitions=[dict(checkpoint=entry, from_signature=before, to_signature=digest(after))])
        self.assertTrue(teacher_receipt_preserved(state, after, raw, entry))
        changed = copy.deepcopy(after); changed['implementation'] = 'changed'
        self.assertFalse(teacher_receipt_preserved(state, changed, raw, entry))
        changed = copy.deepcopy(after); changed['config']['learning_rate'] *= .5
        self.assertFalse(teacher_receipt_preserved(state, changed, raw, entry))

    def test_full_resume_receipt_mode_toggle_keeps_weights_adam_rng(self):
        before = self.signature(config()); after = copy.deepcopy(before); after['config']['teacher_phase']['mode'] = 'shadow'
        run = self.temp_run(); original = model_fixture()
        original.utility_calibration = dict(ready=True, reward_id=digest(original.config.utility))
        optimizer = torch.optim.AdamW(original.parameters(), lr=1e-5)
        sum(p.square().mean() for p in original.parameters()).backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        path = run / 'generation-0070.pt'
        identity = save_checkpoint(path, original, optimizer, step=70,
            training=dict(mode='step5-population-ppo', fresh_run=run.relative_to(SOURCE_ROOT).as_posix(), signature=digest(before)))
        entry = dict(path=path.relative_to(SOURCE_ROOT).as_posix(), sha256=sha(path), model_id=identity, generation=70)
        state = dict(format='step5-recovery-league/2.2.0', signature=before, iteration=70,
            games_done=22400, history=[entry], population=[entry], evaluation_anchor=entry)
        write_json(run / 'state.json', state)
        model, raw, changed = load_resume(SOURCE_ROOT, run, state, after, reviewed=True)
        self.assertTrue(changed); self.assertTrue(model.utility_calibration['ready'])
        self.assertTrue(_same_payload(model.state_dict(), original.state_dict()))
        other = torch.optim.AdamW(model.parameters()); resume_optimizer(raw, other)
        self.assertTrue(_same_payload(optimizer.state_dict(), other.state_dict()))
        state = commit_transition(run, state, after)
        model, _, _ = load_resume(SOURCE_ROOT, run, state, after)
        self.assertTrue(model.utility_calibration['ready'])

    def test_helper_is_readonly_idempotent_generation_hash_and_adoption_bound(self):
        spec = importlib.util.spec_from_file_location('configure_teacher', SOURCE_ROOT / 'scripts/configure-teacher-phase.py')
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        run = self.temp_run(); root = run; run = root / 'artifacts/test'; run.mkdir(parents=True)
        checkpoint = run / 'generation-1324.pt'; checkpoint.write_bytes(b'fixture checkpoint')
        old = config(); old.pop('teacher_phase'); old['teacher_value']['enabled'] = True
        state = dict(format='step5-recovery-league/2.2.0', iteration=1324, signature=self.signature(old, BASE_IMPLEMENTATION),
            history=[dict(generation=1324, path='artifacts/test/generation-1324.pt', sha256=sha(checkpoint))])
        write_json(run / 'state.json', state); before = sha(run / 'state.json')
        output = 'configs/step5/teacher-off.json'
        a = helper.generate(root, run, output, 1324, 'off')
        self.assertEqual(a, helper.generate(root, run, output, 1324, 'off'))
        self.assertEqual(before, sha(run / 'state.json'))
        with self.assertRaisesRegex(ValueError, 'adoption'): helper.generate(root, run, output, 1324, 'on')
        with self.assertRaisesRegex(ValueError, 'generation'): helper.generate(root, run, output, 1323, 'off')
        checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint'): helper.generate(root, run, output, 1324, 'off')


class Lifecycle(Base):
    def test_fixture_collection_barrier_partial_buffer_and_resume(self):
        run = self.temp_run(); c = config(); c['teacher_phase']['mode'] = 'shadow'; m = model_fixture()
        m.utility_calibration = dict(ready=True, reward_id=digest(m.config.utility))
        state = dict(iteration=10, history=[dict(generation=10, model_id=model_id(m))], next_seed=456)
        phase = TeacherPhase(SOURCE_ROOT, run, c, state, m)
        jobs = phase.begin(state, m, export_hash='a' * 64, epoch=1)
        # 32 planned jobs; two table results have one authored row each. No games run.
        results = []
        for i, job in enumerate(jobs):
            r, _ = example(m, job=job, round=job['reservations']['0']['canal_early'])
            results.append(dict(job_id=job['expected']['job_id'], rows=[r.row] if i in (0, 28) else [],
                                metrics=dict(fixture=True)))
        phase.admit_collection(results, m); term = phase.prepare(m)
        self.assertIsNotNone(term); self.assertEqual(len(term.rows), 1); self.assertEqual(len(phase.heldout), 1)
        report = phase.finish(m, dict(optimizer_steps=0)); payload = phase.payload(11)
        self.assertEqual(report['collection']['labels'], 2); self.assertEqual(report['heldout']['delta']['all'], 0.)
        state.update(iteration=11, teacher_phase=payload, teacher_seed_cursor=phase.cursor)
        write_json(run / 'state.json', state); phase.committed(11)
        m.utility_calibration = None
        restored = TeacherPhase(SOURCE_ROOT, run, c, state, m, raw=dict(training=dict(teacher_phase=payload)))
        self.assertTrue(restored.ready)
        off = copy.deepcopy(c); off['teacher_phase']['mode'] = 'off'
        stopped = TeacherPhase(SOURCE_ROOT, run, off, state, m, raw=dict(training=dict(teacher_phase=payload)))
        self.assertEqual(stopped.begin(state, m), []); self.assertIsNone(stopped.prepare(m))
        self.assertEqual(state['next_seed'], 456)
        bad = copy.deepcopy(state); bad['teacher_seed_cursor'] += 1
        with self.assertRaisesRegex(ValueError, 'generation/seed'):
            TeacherPhase(SOURCE_ROOT, run, c, bad, m, raw=dict(training=dict(teacher_phase=payload)))
        raw = dict(training=dict(teacher_phase=copy.deepcopy(payload))); raw['training']['teacher_phase']['weight']['beta'] = 99
        with self.assertRaisesRegex(ValueError, 'checkpoint/state'):
            TeacherPhase(SOURCE_ROOT, run, c, state, m, raw=raw)

    def test_live_enablement_fails_before_any_collection_or_state_write(self):
        from brass_ai.recovery.league import _train
        c = config(); c['teacher_phase']['mode'] = 'on'; run = self.temp_run()
        with self.assertRaisesRegex(ValueError, 'B05/B07 teacher update/adoption qualification is Pending'):
            _train(SOURCE_ROOT, run, c)
        self.assertEqual(list(run.iterdir()), [])

    def test_lifecycle_rejects_wrong_result_set_and_retains_ready_buffer(self):
        run = self.temp_run(); c = config(); c['teacher_phase']['mode'] = 'shadow'; m = model_fixture()
        m.utility_calibration = dict(ready=True, reward_id=digest(m.config.utility))
        state = dict(iteration=10, history=[dict(generation=10, model_id=model_id(m))])
        phase = TeacherPhase(SOURCE_ROOT, run, c, state, m); phase.begin(state, m, export_hash='a' * 64, epoch=1)
        with self.assertRaisesRegex(ValueError, 'Missing/duplicate'): phase.admit_collection([], m)
        r, e = example(m)
        phase.buffer.publish(10, 'train', admit(m, r, e), model_id=e['model_id'], reward_id=e['reward_id'])
        m.utility_calibration = dict(ready=False, reward_id=digest(m.config.utility))
        self.assertEqual(phase.begin(state, m), [])
        self.assertIsNotNone(phase.prepare(m))  # Deferral applies to new collection, not valid old rows.


if __name__ == '__main__': unittest.main()
