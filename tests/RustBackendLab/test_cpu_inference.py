"""Focused CPU-inference regressions: export parity, worker parity and bindings.

Native cases run against the built workers when BRASS_CPU_* is set, which
scripts/prepare-rust-cpu-inference.py does. No training or long games run here.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery import collector_runtime as R
from brass_ai.recovery.teacher_labels import logit_tolerance
from brass_ai.rust_backend.lab_cpu_inference import softmax32
from brass_ai.rust_backend.lab_cpu_overlay import PATCHES, apply, fingerprint
from test_b025 import config, model_fixture

NATIVE = bool(os.environ.get('BRASS_CPU_TEACHER'))


def headed_model():
    from brass_ai.learning.search_value import install
    model = model_fixture().eval()
    head = install(model)
    with torch.no_grad():
        torch.manual_seed(4711)
        for parameter in head.parameters():
            parameter.add_(torch.randn_like(parameter)*.02)
    return model


def anchor_model():
    """A league anchor: the imitation-learning network has no utility head."""
    from brass_ai.learning.model import FourSeatNet, ModelConfig
    torch.manual_seed(925)
    return FourSeatNet(ModelConfig.preset('small')).eval()


class Configuration(unittest.TestCase):
    def test_absent_block_keeps_the_exact_execution_object(self):
        execution = dict(processes=24, hosts_per_process=2, cpu_budget=64, cpu_reserve=4)
        with R.scope(R.from_config({})):
            self.assertIs(R.effective_execution(execution), execution)

    def test_block_is_stripped_from_the_signature_and_sets_the_64_lane_topology(self):
        base = dict(update={}, execution=dict(processes=24, hosts_per_process=2, cpu_budget=64, cpu_reserve=4))
        configured = R.configured(base, dict(R.from_config(base), native_inference=R.cpu_block()))
        self.assertEqual(R.recipe(configured), base)
        with R.scope(R.from_config(configured)):
            effective = R.effective_execution(R.recipe(configured)['execution'])
        self.assertEqual((effective['processes'], effective['hosts_per_process']), (64, 1))
        self.assertEqual((effective['cpu_budget'], effective['cpu_reserve']), (64, 0))

    def test_default_block_uses_fixed_count_budgets(self):
        block = R.cpu_block()
        R.validate_cpu(block)
        self.assertEqual(block['teacher_budget'], dict(mode='count', full=640, full_k=16, fast=96, fast_k=8, c_scale=.1))
        self.assertEqual({block[k] for k in ('ordinary', 'teacher', 'rating')}, {'cpu-ort'})

    def test_python_and_native_safety_caps_stay_equal(self):
        import re
        from brass_ai.rust_backend.lab_cpu_runtime import COUNT_CAP_PER_TRAJECTORY_NS, count_search_limits
        text = (SOURCE_ROOT/'lab-native-cpu/src/cpu_eval.rs').read_text(encoding='utf-8')
        native = int(re.search(r'COUNT_CAP_PER_TRAJECTORY_NS: u64 = ([0-9_]+);', text).group(1).replace('_', ''))
        self.assertEqual(native, COUNT_CAP_PER_TRAJECTORY_NS)
        self.assertEqual(count_search_limits(dict(budget=R.cpu_block()['teacher_budget'])), dict(full=38_400_000_000, fast=5_760_000_000))
        self.assertIsNone(count_search_limits(None))
        self.assertIsNone(count_search_limits(dict(budget=dict(mode='timed'))))

    def test_a_recorded_budget_without_k_stays_readable(self):
        R.validate_cpu(dict(R.cpu_block(), teacher_budget=dict(mode='count', full=256, fast=48)))

    def test_invalid_blocks_are_rejected(self):
        full = dict(mode='count', full=640, full_k=16, fast=96, fast_k=8)
        for change in (dict(workers=0), dict(pinning='sometimes'), dict(teacher_budget=dict(mode='count', full=8, fast=48)),
                       dict(teacher_budget=dict(mode='count', full=256)), dict(teacher='tpu'), dict(extra=1),
                       dict(teacher_budget=dict(full, full_k=4)), dict(teacher_budget=dict(full, fast_k=3)),
                       dict(teacher_budget={k: v for k, v in full.items() if k != 'fast_k'})):
            with self.assertRaises(ValueError):
                R.validate_cpu(dict(R.cpu_block(), **change))
        with self.assertRaises(ValueError):
            R.validate(dict(native_inference=dict(R.cpu_block(), contract='old')))

    def test_settings_require_every_default_key(self):
        with self.assertRaises(ValueError):
            R.install(dict(collector_validation='production_fast'))

    def test_continuation_flags_select_cpu_gpu_or_the_recorded_choice(self):
        import importlib.util
        path = SOURCE_ROOT/'scripts/continue-rust-shadow.py'
        spec = importlib.util.spec_from_file_location('continue_rust_shadow_fixture', path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        parser = launcher.parser(inference='cpu')
        options = ['--expected-generation', '1730']
        block = launcher.inference_choice(parser.parse_args(options))
        self.assertEqual(block, R.cpu_block())
        unit_scale = launcher.inference_choice(parser.parse_args([*options, '--c-scale', '1']))
        self.assertEqual(unit_scale['teacher_budget']['c_scale'], 1.)
        self.assertIsNone(launcher.inference_choice(parser.parse_args([*options, '--inference', 'gpu'])))
        self.assertEqual(launcher.inference_choice(parser.parse_args([*options, '--inference', 'keep'])), 'keep')
        timed = launcher.inference_choice(parser.parse_args([*options, '--teacher-budget', 'timed']))
        self.assertEqual(timed['teacher_budget'], dict(mode='timed'))

    def test_removing_the_block_returns_the_exact_gpu_settings(self):
        base = dict(update={}, execution=dict(processes=24, hosts_per_process=2))
        selected = R.configured(base, dict(R.from_config(base), native_inference=R.cpu_block()))
        rolled_back = R.configured(selected, {k: v for k, v in R.from_config(selected).items() if k != 'native_inference'})
        self.assertNotIn('native_inference', rolled_back['execution'])
        self.assertEqual(R.recipe(rolled_back), base)


class Startup(unittest.TestCase):
    def test_breakdown_reports_every_phase_and_the_lane_spread(self):
        from brass_ai.rust_backend.lab_pool import STARTUP, startup_breakdown
        lanes = [dict(spawn_seconds=.1*i, python_import_seconds=1., lane_wait_seconds=0., lease_seconds=0., session_import_seconds=.2,
                      mono_spawn_seconds=.01, mono_bootstrap_seconds=.15+.01*i, total_seconds=2.+i, ready_time=100.+i) for i in range(4)]
        report = startup_breakdown(lanes, 99., width=4, lanes=4, barrier=False, process_start_seconds=.5, channel_seconds=0.)
        self.assertEqual(report['contract'], STARTUP)
        self.assertEqual((report['first_lane_ready_seconds'], report['last_lane_ready_seconds']), (1., 4.))
        mono = report['phases']['mono_bootstrap_seconds']
        self.assertAlmostEqual(mono['min'], .15)
        self.assertAlmostEqual(mono['max'], .18)
        self.assertEqual(mono['lanes'], 4)
        self.assertEqual(len(report['per_lane']), 4)
        self.assertNotIn('ready_time', report['per_lane'][0])
        self.assertFalse(report['waits_for_all_lanes'])

    def test_no_lane_reports_still_produce_a_receipt(self):
        from brass_ai.rust_backend.lab_pool import startup_breakdown
        report = startup_breakdown([], 0., width=0, lanes=0, barrier=False, process_start_seconds=0., channel_seconds=0.)
        self.assertIsNone(report['first_lane_ready_seconds'])
        self.assertEqual(report['phases'], {})

    def test_a_lane_receives_only_what_fits_in_a_pipe(self):
        import pickle
        from brass_ai.rust_backend.lab_pool import lane_build
        full = dict(source='x'*500000, collector=dict(path='/c', sha256='0'*64), b01=dict(sha256='1'*64),
                    bootstrap=dict(binary=dict(path='/b', sha256='2'*64), dependencies=[dict(path='d'*200)]*300))
        small = lane_build(full)
        self.assertEqual(small, dict(bootstrap=dict(binary=dict(path='/b')), collector=dict(path='/c')))
        self.assertLess(len(pickle.dumps(small)), 4096)
        fixture = dict(fixture='part.json', fail=False)
        self.assertIs(lane_build(fixture), fixture)

    def test_a_cpu_lane_needs_no_service_channel(self):
        from brass_ai.recovery.inference import endpoint_encoder
        from brass_ai.rust_backend.lab_cpu_inference import NoServiceEndpoint
        endpoint = NoServiceEndpoint()
        self.assertIs(endpoint_encoder(endpoint), endpoint_encoder(endpoint))
        endpoint.close()

    def test_a_lane_ready_message_never_counts_as_a_job_result(self):
        from brass_ai.rust_backend.lab_pool import NativePool
        pool = NativePool.__new__(NativePool)
        pool.stop, pool.processes = threading.Event(), []
        pool.service = SimpleNamespace(error=None)
        seen = []
        messages = [dict(ready=True, startup=dict(total_seconds=1.)), dict(result='job')]
        connection = SimpleNamespace(poll=lambda timeout: True, recv=lambda: messages.pop(0))
        self.assertEqual(pool.receive(connection, on_ready=seen.append), dict(result='job'))
        self.assertEqual(seen[0]['startup'], dict(total_seconds=1.))
        messages[:] = [dict(ready=True, startup={})]
        self.assertEqual(pool.receive(connection)['ready'], True)


class Probability(unittest.TestCase):
    def test_float32_softmax_matches_torch(self):
        rng = np.random.default_rng(11)
        for n in (1, 2, 7, 257, 1000):
            logits = rng.normal(0, 6, n).astype(np.float32)
            expected = torch.softmax(torch.from_numpy(logits), -1).numpy()
            actual = softmax32(logits)
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=4e-12)
            self.assertLess(abs(float(actual.sum())-1), 1e-5)

    def test_invalid_logits_are_rejected(self):
        for bad in ([], [np.nan, 1.], [np.inf, 1.]):
            with self.assertRaises(ValueError):
                softmax32(np.asarray(bad, dtype=np.float32))


class Tolerance(unittest.TestCase):
    def test_gpu_labels_keep_their_original_bound(self):
        self.assertEqual(logit_tolerance({}), (2e-5, 1e-5))

    def test_cpu_labels_use_the_cross_runtime_export_bound(self):
        self.assertEqual(logit_tolerance(dict(inference_backend='cpu-ort')), (2e-4, 2e-4))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            logit_tolerance(dict(inference_backend='cpu-other'))


class Overlay(unittest.TestCase):
    def test_every_anchor_matches_the_gpu_source_exactly_once(self):
        for name, changes in PATCHES.items():
            text = (SOURCE_ROOT/'lab-native'/name).read_bytes().replace(b'\r\n', b'\n').decode()
            for before, _ in changes:
                self.assertEqual(text.count(before), 1, name+': '+before[:50])

    def test_overlay_leaves_the_gpu_tree_untouched_and_is_stable(self):
        original = {p: p.read_bytes() for p in (SOURCE_ROOT/'lab-native').rglob('*') if p.is_file() and 'target' not in p.parts}
        with tempfile.TemporaryDirectory() as directory:
            result = apply(Path(directory)/'lab-native')
            self.assertEqual(result['source'], fingerprint())
            self.assertIn('cpu_eval', (Path(directory)/'lab-native/src/lib.rs').read_text())
            with self.assertRaises(ValueError):
                apply(Path(directory)/'lab-native')
        self.assertEqual(original, {p: p.read_bytes() for p in original})

    def test_overlay_refuses_the_real_source_directory(self):
        with self.assertRaises(ValueError):
            apply(SOURCE_ROOT/'lab-native')


class Bundles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = headed_model()
        cls.directory = tempfile.TemporaryDirectory()
        from brass_ai.rust_backend.lab_cpu_models import export_bundle
        cls.path = Path(cls.directory.name)/'bundle'
        before = torch.random.get_rng_state()
        cls.manifest = export_bundle(cls.model, cls.path)
        cls.rng_unchanged = torch.equal(before, torch.random.get_rng_state())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_export_binds_model_reward_and_six_outputs(self):
        m = self.manifest
        self.assertEqual(m['model_id'], model_id(self.model))
        self.assertEqual(m['reward_id'], digest(self.model.config.to_dict()['utility']))
        self.assertEqual(m['outputs'][-1], 'search_delta_4')
        self.assertTrue(m['parity']['passed'])
        self.assertLess(max(m['parity']['errors'].values()), 2e-4)

    def test_export_does_not_consume_trainer_randomness(self):
        self.assertTrue(self.rng_unchanged)

    def test_anchor_without_a_utility_head_exports_a_zero_head(self):
        import onnxruntime as ort
        from brass_ai.learning.onnx_tools import cpu_options
        from brass_ai.personality.laptop.export import samples
        from brass_ai.rust_backend.lab_cpu_models import NO_REWARD, export_bundle
        model = anchor_model()
        self.assertIsNone(getattr(model.config, 'utility', None))
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_bundle(model, Path(directory)/'bundle')
            self.assertEqual((manifest['utility'], manifest['reward_id'], manifest['max_cost']), (False, NO_REWARD, 0.))
            self.assertEqual(manifest['outputs'][-1], 'remaining_utility_4')
            self.assertFalse(manifest['search_delta'])
            self.assertLess(max(manifest['parity']['errors'].values()), 2e-4)
            session = ort.InferenceSession(str(Path(directory)/'bundle/model.onnx'), cpu_options(1), providers=['CPUExecutionProvider'])
            config = dict(model.config.to_dict(), utility=dict(insolvency=dict(max_cost=1.)))
            for batch in samples(config, False)[:2]:
                out = session.run(None, batch)
                self.assertTrue((out[4] == 0).all())
                inputs = (torch.from_numpy(batch[k]) for k in ('states', 'actions', 'legal_mask', 'stable_to_relative'))
                with torch.inference_mode():
                    logits, win, placement, vp = model(*inputs)
                finite = torch.isfinite(logits).numpy()
                np.testing.assert_allclose(out[0][finite], logits.numpy()[finite], rtol=2e-4, atol=2e-4)
                for index, head in enumerate((win, placement, vp), 1):
                    np.testing.assert_allclose(out[index], head.numpy(), rtol=2e-4, atol=2e-4)

    def test_rating_reply_omits_utility_for_an_anchor_exactly_like_the_service(self):
        from brass_ai.bank import sha
        from brass_ai.learning.batching import Request, accept_response, collate, predict_encoded
        from brass_ai.rust_backend import lab_cpu_inference as I
        from brass_ai.rust_backend.lab_cpu_models import export_bundle
        from test_b025 import example
        model = anchor_model()
        row, _ = example(self.model)
        identity = model_id(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bundle'
            export_bundle(model, path)
            receipt = dict(manifest=str((path/'manifest.json').resolve()), sha256=sha(path/'manifest.json'))
            encoder = SimpleNamespace(encode=lambda *a, **k: row.encoded)
            endpoint = SimpleNamespace(lock=threading.RLock(), _encoder=encoder, encoding_cache=None)
            evaluator = I.OrtEvaluator(endpoint, identity, {identity: receipt})
            view, choices = row.row['view'], row.row['choices']
            request = Request('r1', identity, view, choices, __import__('time').monotonic()+60)
            reply = evaluator.predict([request])[0]
            I.SESSIONS.values.clear()
        accept_response(request, reply, view['context']['revision'], identity)
        self.assertNotIn('utility', reply)
        self.assertNotIn('reward_id', reply)
        with torch.inference_mode():
            arrays = predict_encoded(model, [row.encoded], 'cpu', 8192, None)
        self.assertEqual(len(arrays), 4)
        np.testing.assert_allclose(reply['probabilities'], arrays[0][0, :len(choices)], rtol=3e-3, atol=3e-5)
        for index, name in enumerate(('win_share', 'placement', 'vp'), 1):
            np.testing.assert_allclose(reply[name], arrays[index][0], rtol=2e-4, atol=2e-4)

    def test_search_delta_matches_the_gpu_correction(self):
        import onnxruntime as ort
        from brass_ai.learning.onnx_tools import cpu_options
        from brass_ai.learning.search_value import correction
        from brass_ai.personality.laptop.export import samples
        session = ort.InferenceSession(str(self.path/'model.onnx'), cpu_options(1), providers=['CPUExecutionProvider'])
        for batch in samples(self.model.config.to_dict(), False)[:2]:
            out = session.run(None, batch)
            states, mapping, remaining = (torch.from_numpy(batch[k]) for k in ('states', 'stable_to_relative', 'remaining'))
            with torch.inference_mode():
                delta = correction(self.model, self.model.encode_state(states), mapping, remaining).numpy()
            self.assertGreater(np.abs(delta).max(), 0)
            np.testing.assert_allclose(out[-1], delta, rtol=2e-4, atol=2e-4)

    def test_cache_reuses_exports_and_rejects_foreign_bundles(self):
        from brass_ai.rust_backend import lab_cpu_models as M
        identity = model_id(self.model)
        with tempfile.TemporaryDirectory() as root:
            base = M.bundle_root(root)
            base.mkdir(parents=True)
            import shutil
            shutil.copytree(self.path, base/identity)
            bundles, exported = M.ensure(root, {}, [identity])
            self.assertEqual(exported, 0)
            self.assertEqual(bundles[identity]['manifest'], str((base/identity/'manifest.json').resolve()))
            with self.assertRaises(ValueError):
                M.ensure(root, {}, ['0'*64])
            manifest = json.loads((base/identity/'manifest.json').read_text())
            shutil.copytree(base/identity, base/('1'*64))
            with self.assertRaises(ValueError):
                M.ensure(root, {}, ['1'*64])
            self.assertEqual(manifest['model_id'], identity)

    def test_pruning_keeps_active_and_recent_bundles(self):
        from brass_ai.rust_backend import lab_cpu_models as M
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            for i in range(6):
                directory = base/str(i)
                directory.mkdir()
                (directory/'manifest.json').write_text('{}')
                os.utime(directory/'manifest.json', (i+1, i+1))
            M.prune(base, {'0'}, limit=3)
            self.assertEqual({p.name for p in base.iterdir()}, {'0', '4', '5'})

    def test_rating_evaluator_matches_the_gpu_response_shape(self):
        from brass_ai.rust_backend import lab_cpu_inference as I
        from brass_ai.learning.batching import Request, accept_response
        import shutil
        from test_b025 import example
        row, _ = example(self.model)
        identity = model_id(self.model)
        receipt = dict(manifest=str((self.path/'manifest.json').resolve()),
                       sha256=__import__('brass_ai.bank', fromlist=['sha']).sha(self.path/'manifest.json'))
        encoder = SimpleNamespace(encode=lambda *a, **k: row.encoded)
        endpoint = SimpleNamespace(lock=threading.RLock(), _encoder=encoder, encoding_cache=None)
        evaluator = I.OrtEvaluator(endpoint, identity, {identity: receipt})
        view, choices = row.row['view'], row.row['choices']
        request = Request('r1', identity, view, choices, __import__('time').monotonic()+60)
        reply = evaluator.predict([request])[0]
        accept_response(request, reply, view['context']['revision'], identity)
        from brass_ai.rust_backend.lab_teacher_inference import predict
        cap = self.model.config.utility['insolvency']['max_cost']
        with torch.inference_mode():
            arrays = predict(self.model, [row.encoded], 'cpu', 8192, torch.full((1, 4), cap, dtype=torch.float32))
        np.testing.assert_allclose(reply['probabilities'], arrays[0][0, :len(choices)], rtol=3e-3, atol=3e-5)
        for index, name in enumerate(('win_share', 'placement', 'vp', 'utility'), 1):
            np.testing.assert_allclose(reply[name], arrays[index][0], rtol=2e-4, atol=2e-4)


@unittest.skipUnless(NATIVE, 'Set BRASS_CPU_* to run the built workers')
class NativeWorkers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = headed_model()
        cls.anchor = anchor_model()
        cls.directory = tempfile.TemporaryDirectory()
        from brass_ai.rust_backend.lab_cpu_models import ensure
        models = {model_id(cls.model): cls.model, model_id(cls.anchor): cls.anchor}
        cls.bundles, _ = ensure(cls.directory.name, models, list(models))
        cls.identity = model_id(cls.model)
        cls.anchor_id = model_id(cls.anchor)
        from brass_ai.rust_backend.lab_contract import pin
        runtime = pin(os.environ['BRASS_CPU_RUNTIME'])
        cls.runtime = dict(path=runtime['path'], sha256=runtime['sha256'])
        cls.roots = json.loads(Path(os.environ['BRASS_CPU_FIXTURES']).read_text())['roots']

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def collector(self):
        from brass_ai.rust_backend.lab_transport import Collector
        lane = Collector(os.environ['BRASS_CPU_COLLECTOR'])
        self.assertEqual(lane.hello['cpu_inference'], R.CPU_CONTRACT)
        lane.call('configure_cpu', contract=R.CPU_CONTRACT, runtime=self.runtime, model_cache=2)
        return lane

    def test_collector_outputs_match_pytorch_and_gpu_features(self):
        from brass_ai.rust_backend.lab_transport import DecisionEncoder, feature_context
        from brass_ai.rust_backend.lab_teacher_inference import predict
        wide = max(self.roots, key=lambda r: len(r['expected']['choices']))
        with self.collector() as lane:
            for root in (wide, self.roots[0]):
                lane.begin(root['id'], dict(state=root['expected']['state'], rail_keys=list(range(64))), 5000)
                self.assertTrue(lane.call('load_model', model_id=self.identity, bundle=self.bundles[self.identity])['model_id'])
                d = lane.current['decision']
                style, intent = [0.]*4, [0.]*8
                encoded = DecisionEncoder(lane, self.identity).encode(d['view'], d['choices'], style, intent)
                page = lane.call('infer', token=lane.token, revision=d['view']['context']['revision'],
                                 context=feature_context(self.identity, style, intent))
                self.assertEqual(page['wire'], 'ordinary-infer/1')
                self.assertEqual(page['choice_ids'], [c['id'] for c in d['choices']])
                n = len(d['choices'])
                values = np.frombuffer(page['_arrays'], '<f4')
                self.assertEqual(len(values), n+16)
                cap = self.model.config.utility['insolvency']['max_cost']
                with torch.inference_mode():
                    arrays = predict(self.model, [encoded], 'cpu', 8192, torch.full((1, 4), cap, dtype=torch.float32))
                np.testing.assert_allclose(values[:n], arrays[-1][0, :n], rtol=2e-4, atol=2e-4)
                for index in range(4):
                    np.testing.assert_allclose(values[n+4*index:n+4*index+4], arrays[index+1][0], rtol=2e-4, atol=2e-4)
                lane.release()
            report = lane.call('cpu_report')
            self.assertEqual(report['positions'], 2)
            self.assertEqual(report['loads'], 1)

    def test_collector_serves_an_anchor_with_no_utility_head(self):
        from brass_ai.learning.batching import collate
        from brass_ai.rust_backend.lab_transport import DecisionEncoder, feature_context
        wide = max(self.roots, key=lambda r: len(r['expected']['choices']))
        with self.collector() as lane:
            lane.begin(wide['id'], dict(state=wide['expected']['state'], rail_keys=list(range(64))), 5000)
            loaded = lane.call('load_model', model_id=self.anchor_id, bundle=self.bundles[self.anchor_id])
            self.assertEqual(loaded['receipt']['reward_id'], '')
            d = lane.current['decision']
            style, intent = [0.]*4, [0.]*8
            encoded = DecisionEncoder(lane, self.anchor_id).encode(d['view'], d['choices'], style, intent)
            page = lane.call('infer', token=lane.token, revision=d['view']['context']['revision'],
                             context=feature_context(self.anchor_id, style, intent))
            n = len(d['choices'])
            values = np.frombuffer(page['_arrays'], '<f4')
            with torch.inference_mode():
                logits, win, placement, vp = self.anchor(*collate([encoded], 'cpu'))
            np.testing.assert_allclose(values[:n], logits[0].numpy(), rtol=2e-4, atol=2e-4)
            for index, head in enumerate((win, placement, vp)):
                np.testing.assert_allclose(values[n+4*index:n+4*index+4], head[0].numpy(), rtol=2e-4, atol=2e-4)
            self.assertTrue((values[n+12:n+16] == 0).all())
            lane.release()

    def test_a_game_mixes_anchors_and_utility_models_like_the_league(self):
        """The exact shape that stopped the first lab run."""
        from brass_ai.rust_backend.lab_session import OrdinarySession
        from brass_ai.rust_backend.lab_collection import collect_lane
        from brass_ai.recovery.execution import neural_spec
        native = Path(os.environ['BRASS_CPU_NATIVE'])
        state = json.loads((native/'brass-ai-core/tests/data/initial-state.json').read_text())
        origin = dict(state=state, rail_keys=list(range(64)))
        binary = os.environ['BRASS_CPU_COLLECTOR']
        endpoint = SimpleNamespace(lock=threading.RLock())
        execution = dict(max_decisions=5000, batch_size=32, max_candidate_cells=8192)
        learner, anchor = (neural_spec(i, 0., 'neutral', version=3) for i in (self.identity, self.anchor_id))
        job = dict(index=0, seed=19, sampling_seed=31, policies=[learner, anchor, learner, anchor], provenance={})
        cpu = dict(contract=R.CPU_CONTRACT, runtime=self.runtime, bundles=self.bundles, model_cache=2, pinning='none')
        with OrdinarySession(SOURCE_ROOT, binary, endpoint, execution, cpu=cpu) as session:
            first = collect_lane(SOURCE_ROOT, binary, None, job, endpoint, execution, origin=origin, session=session)
        records = first['game']['records']
        self.assertEqual({r['behavior']['policy_id'] for r in records}, {learner['policy_id'], anchor['policy_id']})
        self.assertGreater(first['timings']['cpu_model_setup_seconds'], 0)

    def test_wrong_manifest_pin_and_unloaded_model_fail_closed(self):
        root = self.roots[0]
        from brass_ai.rust_backend.lab_transport import feature_context
        with self.collector() as lane:
            bad = dict(self.bundles[self.identity], sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'differs from its pin'):
                lane.call('load_model', model_id=self.identity, bundle=bad)
        with self.collector() as lane:
            lane.begin('x', dict(state=root['expected']['state'], rail_keys=list(range(64))), 5000)
            d = lane.current['decision']
            with self.assertRaisesRegex(ValueError, 'not loaded'):
                lane.call('infer', token=lane.token, revision=d['view']['context']['revision'],
                          context=feature_context(self.identity, [0.]*4, [0.]*8))

    def test_complete_collect_lane_samples_through_cpu_inference(self):
        from brass_ai.rust_backend.lab_session import OrdinarySession
        from brass_ai.rust_backend.lab_collection import collect_lane
        from brass_ai.recovery.execution import neural_spec
        root = self.roots[-1]
        origin = dict(state=root['expected']['state'], rail_keys=list(range(64)))
        binary = os.environ['BRASS_CPU_COLLECTOR']
        endpoint = SimpleNamespace(lock=threading.RLock())
        execution = dict(max_decisions=100, batch_size=32, max_candidate_cells=8192)
        spec = neural_spec(self.identity, 0., 'neutral', version=3)
        job = dict(index=0, seed=19, sampling_seed=31, policies=[spec]*4, provenance={})
        cpu = dict(contract=R.CPU_CONTRACT, runtime=self.runtime, bundles=self.bundles, model_cache=2, pinning='none')
        with OrdinarySession(SOURCE_ROOT, binary, endpoint, execution, cpu=cpu) as session:
            first = collect_lane(SOURCE_ROOT, binary, None, job, endpoint, execution, origin=origin, session=session)
            second = collect_lane(SOURCE_ROOT, binary, None, job, endpoint, execution, origin=origin, session=session)
        records = first['game']['records']
        self.assertTrue(records)
        self.assertEqual([r['chosen'] for r in records], [r['chosen'] for r in second['game']['records']])
        self.assertEqual([r['behavior'] for r in records], [r['behavior'] for r in second['game']['records']])
        self.assertTrue(all(0 < r['behavior']['probability'] <= 1 for r in records))
        self.assertGreater(first['timings']['ipc_cpu_forward_ns'], 0)
        self.assertEqual(second['timings']['policy_reuses'], 4)

    def teacher_request(self, job, state, budget=None):
        from brass_ai.rust_backend.lab_teacher import CONTRACT
        return dict(contract=CONTRACT, op='run', job=job, origin=dict(state=state, rail_keys=list(range(64))),
                    core_source=os.environ['BRASS_CPU_CORE'], estimate_ns=1000, calibration_id='fixed-count',
                    inference=dict(contract=R.CPU_CONTRACT, runtime=self.runtime, bundle=self.bundles[self.identity],
                                   budget=budget or dict(mode='count', full=64, fast=8)))

    def test_teacher_table_uses_count_budget_and_passes_gpu_verification(self):
        from test_b04 import job as frozen_job
        from brass_ai.rust_backend.lab_teacher import Worker
        from brass_ai.rust_backend.lab_teacher_finalization import TurnVerifier
        native = Path(os.environ['BRASS_CPU_NATIVE'])
        state = json.loads((native/'brass-ai-core/tests/data/initial-state.json').read_text())
        job = frozen_job(self.model)
        cancel = threading.Event()
        history = [dict(generation=10, model_id=self.identity)]
        verifier = TurnVerifier(self.model, history, [job], cancel)
        worker = Worker(os.environ['BRASS_CPU_TEACHER'], self.teacher_request(job, state), None, cancel)
        result = worker.collect(job['expected'], os.environ['BRASS_CPU_CORE'], verifier=verifier)
        metrics = result['metrics']
        self.assertEqual(metrics['reserved_turns'], 16)
        self.assertGreater(metrics['labels'], 0)
        searches = [s for s in metrics['searches'] if s['frozen_n'] > 0]
        self.assertTrue(searches)
        self.assertTrue(all(s['frozen_n'] == 64 for s in searches))
        self.assertTrue(all(s['completed'] == 64 for s in searches))
        self.assertEqual(metrics['cpu_inference']['backend'], 'cpu-ort')
        self.assertGreater(metrics['cpu_inference']['positions'], 0)
        self.assertTrue(all(r['teacher']['inference_backend'] == 'cpu-ort' for r in result['rows']))
        accepted, rejected = verifier.take(result, job, self.model)
        self.assertEqual(len(accepted)+sum(rejected.values()), len(result['rows']))
        self.assertTrue(accepted)

    def test_teacher_count_search_is_deterministic(self):
        from test_b04 import job as frozen_job
        from brass_ai.rust_backend.lab_teacher import Worker
        from brass_ai.rust_backend.lab_teacher_finalization import TurnVerifier
        native = Path(os.environ['BRASS_CPU_NATIVE'])
        state = json.loads((native/'brass-ai-core/tests/data/initial-state.json').read_text())
        job = frozen_job(self.model)
        rows = []
        for _ in range(2):
            cancel = threading.Event()
            verifier = TurnVerifier(self.model, [dict(generation=10, model_id=self.identity)], [job], cancel)
            worker = Worker(os.environ['BRASS_CPU_TEACHER'], self.teacher_request(job, state), None, cancel)
            rows.append(worker.collect(job['expected'], os.environ['BRASS_CPU_CORE'], verifier=verifier)['rows'])
        strip = lambda items: [{k: r['teacher'][k] for k in ('counts', 'logits', 'target', 'played_action')} for r in items]
        self.assertEqual(strip(rows[0]), strip(rows[1]))

    def all_seat_job(self, weight):
        from brass_ai.recovery.search_value_config import adopt
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        c = adopt(config())
        c['teacher_phase']['mode'] = 'shadow'
        c['teacher_phase']['search_value']['stage'] = 'B'
        state = dict(iteration=10, search_value=dict(stage=dict(stage='B', w=weight)))
        jobs, _ = plan_jobs(c, state, model_id=self.identity, export_hash='a'*64, epoch=1, ready=True,
                            native_source=source_fingerprint())
        job = jobs[0]
        job['readiness'] = dict(ready=True, reward_id=job['expected']['reward_id'])
        return job

    def run_all_seat(self, weight, budget, revision=None, *, split='train'):
        from brass_ai.rust_backend.lab_teacher import Worker
        from brass_ai.rust_backend.lab_teacher_finalization import TurnVerifier
        native = Path(os.environ['BRASS_CPU_NATIVE'])
        state = json.loads((native/'brass-ai-core/tests/data/initial-state.json').read_text())
        job = self.all_seat_job(weight)
        job['split'] = split
        if revision is not None:job['teacher_revision'] = revision
        self.assertEqual(job['search_value']['w'], weight)
        cancel = threading.Event()
        from brass_ai.rust_backend.lab_cpu_runtime import count_search_limits
        verifier = TurnVerifier(self.model, [dict(generation=10, model_id=self.identity)], [job], cancel,
                                search_limits=count_search_limits(dict(budget=budget)))
        worker = Worker(os.environ['BRASS_CPU_TEACHER'], self.teacher_request(job, state, budget), None, cancel)
        result = worker.collect(job['expected'], os.environ['BRASS_CPU_CORE'], verifier=verifier)
        return job, result, verifier

    def root_heads(self, row):
        """The GPU path's float32 utility and search-head correction at one root."""
        from brass_ai.learning.search_value import correction
        states = torch.as_tensor(row['state'], dtype=torch.float32)[None]
        mapping = torch.as_tensor(row['mapping'])[None]
        remaining = torch.as_tensor(row['remaining'], dtype=torch.float32)[None]
        with torch.inference_mode():
            encoded = self.model.encode_state(states)
            utility = self.model.heads_from_encoded(encoded, mapping, remaining)[3].float()
            delta = correction(self.model, encoded, mapping, remaining).float()
        return utility, delta

    def test_a_unit_c_scale_reaches_the_search_and_its_labels(self):
        from brass_ai.recovery.teacher_labels import improved_policy, validate_improved_policy
        budget = dict(mode='count', full=64, full_k=16, fast=24, fast_k=8, c_scale=1.)
        job, result, verifier = self.run_all_seat(0., budget)
        self.assertEqual(result['metrics']['cpu_inference']['budget'], budget)
        self.assertGreater(len(result['rows']), 0)
        for row in result['rows']:
            label = row['teacher']
            self.assertEqual((label['c1'], label['c2']), (50, 1.))
            self.assertIsNotNone(validate_improved_policy(label, row['choices']))
            target, _ = improved_policy(label['logits'], label['counts'], label['mean_returns'], label['node_value'], 50, 1.)
            np.testing.assert_allclose(target, label['target'], atol=1e-5)
        # The same counts under the old constant would give a much softer target.
        label = result['rows'][0]['teacher']
        gentle, _ = improved_policy(label['logits'], label['counts'], label['mean_returns'], label['node_value'], 50, .1)
        self.assertGreater(max(label['target']), float(gentle.max()))

    def test_retuned_budget_reaches_long_search_and_its_labels(self):
        from brass_ai.recovery import training_revision_g1491 as G
        from brass_ai.recovery.teacher_labels import validate_improved_policy

        budget = dict(mode='count', full=64, fast=24, fast_k=8, **G.RETUNED_SEARCH)
        revision = G._job_block(G.block(variant=G.SEARCH_REFRESH)['exploration'], short_labels=False)
        job, result, verifier = self.run_all_seat(0., budget, revision)
        metrics = result['metrics']
        self.assertEqual(metrics['cpu_inference']['budget'], budget)
        self.assertTrue(result['rows'])
        for profile, scale in (('full', .15), ('fast', .5)):
            self.assertEqual({s['c_scale'] for s in metrics['searches']
                              if s['profile'] == profile and s['frozen_n'] > 0}, {scale})
        self.assertEqual(max(s['effective_k'] for s in metrics['searches'] if s['profile'] == 'full'), 8)
        self.assertTrue(all(s['completed'] == s['frozen_n'] for s in metrics['searches'] if s['profile'] == 'full'))
        for row in result['rows']:
            label = row['teacher']
            self.assertEqual((label['requested_k'], label['c2']), (8, .15))
            self.assertIsNotNone(validate_improved_policy(label, row['choices']))
        self.assertTrue(verifier.take(result, job, self.model)[0])

    def test_mixed_teacher_games_apply_the_weight_and_report_actions(self):
        from brass_ai.recovery import training_revision_g1491 as G
        from brass_ai.recovery.teacher_phase import action_summary

        budget = dict(mode='count', full=64, fast=24, fast_k=8, **G.RETUNED_SEARCH)
        for weight, split, biased in ((0, 'heldout', True), (1, 'train', False)):
            with self.subTest(weight=weight, split=split):
                revision = G._job_block(G.EXPLORATION, short_labels=False, search_head_weight=weight, cotton_table=biased)
                job, result, verifier = self.run_all_seat(weight, budget, revision, split=split)
                G.validate_job(job)
                self.assertTrue(verifier.take(result, job, self.model)[0])
                for row in result['value_rows'][:6]:
                    self.assertEqual(row['search_value'], job['search_value'])
                    utility, delta = self.root_heads(row)
                    self.assertGreater(float(delta.abs().max()), 4e-4)
                    np.testing.assert_allclose(row['v'], (utility + weight*delta)[0].double().numpy(), atol=2e-4, rtol=2e-4)
                metrics = result['metrics']
                receipt = metrics['cpu_inference']['revision']
                report = action_summary([metrics], [job], weight)
                totals = report['total']
                self.assertEqual(report['search_head_weight'], weight)
                group = 'cotton_exploration' if biased else 'standard_exploration'
                self.assertEqual(report['by_split'][split][group], totals)
                for kind in ('builds', 'development_actions', 'network_actions'):
                    self.assertGreater(totals[kind], 0)
                self.assertGreaterEqual(totals['developed_tiles'], totals['development_actions'])
                self.assertGreaterEqual(totals['links_laid'], totals['network_actions'])
                self.assertGreater(receipt['sampled']['long'], 0)
                self.assertGreater(receipt['sampled']['short'], 0)
                self.assertEqual('cotton' in receipt, biased)
                if biased:
                    cotton = receipt['cotton']
                    self.assertTrue(any(row['teacher'].get('cotton_exploration') for row in result['rows']))
                    for kind in ('builds', 'develops'):
                        self.assertEqual(totals[f'cotton_{kind}'], sum(map(sum, cotton[f'{kind}_by_era_seat'])))
                else:
                    self.assertTrue(all('cotton_exploration' not in row['teacher'] for row in result['rows']))

    def test_all_seat_table_blends_the_search_head_like_the_gpu_path(self):
        budget = dict(mode='count', full=64, full_k=16, fast=24, fast_k=8)
        job, result, verifier = self.run_all_seat(.45, budget)
        self.assertEqual(job['search_value']['value_source'], 'corrected')
        rows, metrics = result['value_rows'], result['metrics']
        self.assertEqual(len(rows), metrics['searched_turns'])
        self.assertGreater(len(rows), 16)
        self.assertEqual({r['profile'] for r in rows}, {'full', 'fast'})
        self.assertEqual({str(s['frozen_n']) for s in metrics['searches'] if s['frozen_n'] > 0 and s['profile'] == 'full'}, {'64'})
        self.assertEqual({str(s['frozen_n']) for s in metrics['searches'] if s['frozen_n'] > 0 and s['profile'] == 'fast'}, {'24'})
        self.assertTrue(all(s['completed'] == s['frozen_n'] for s in metrics['searches'] if s['frozen_n'] > 0))
        self.assertTrue(all(s['effective_k'] <= (16 if s['profile'] == 'full' else 8)
                            for s in metrics['searches'] if s['frozen_n'] > 0))
        self.assertEqual(max(s['effective_k'] for s in metrics['searches'] if s['frozen_n'] > 0 and s['profile'] == 'fast'), 8)
        self.assertEqual(metrics['cpu_inference']['budget'], dict(budget, c_scale=.1))
        self.assertTrue(all(r['search_value'] == job['search_value'] for r in rows))
        for row in rows[:6]:
            utility, delta = self.root_heads(row)
            self.assertGreater(float(delta.abs().max()), 1e-5)
            blended = utility+torch.tensor(.45, dtype=torch.float32)*delta
            np.testing.assert_allclose(row['v'], blended[0].double().numpy(), rtol=2e-4, atol=2e-4)
            # The unblended value differs by more than the bound, so a dropped head fails.
            self.assertGreater(float((blended-utility).abs().max()), 4e-4)
        accepted, rejected = verifier.take(result, job, self.model)
        self.assertEqual(len(accepted)+sum(rejected.values()), len(result['rows']))
        self.assertTrue(accepted)

    def test_zero_blend_table_keeps_the_exact_utility(self):
        job, result, _ = self.run_all_seat(0., dict(mode='count', full=64, full_k=16, fast=24, fast_k=8))
        self.assertEqual(job['search_value']['value_source'], 'main')
        for row in result['value_rows'][:6]:
            utility, _ = self.root_heads(row)
            np.testing.assert_allclose(row['v'], utility[0].double().numpy(), rtol=2e-4, atol=2e-4)

    def test_a_g1491_table_labels_short_searches_and_explores_from_visit_counts(self):
        from brass_ai.recovery import training_revision_g1491 as R
        from brass_ai.recovery.teacher_labels import check_turn, label_profile, validate_improved_policy
        budget = dict(mode='count', full=64, full_k=16, fast=32, fast_k=8, c_scale=1., fast_c_scale=.5)
        revision = R._job_block(R.EXPLORATION)
        job, result, verifier = self.run_all_seat(0., budget, revision)
        rows, metrics = result['rows'], result['metrics']
        long = [r for r in rows if label_profile(r['teacher']) == 'full']
        short = [r for r in rows if label_profile(r['teacher']) == 'fast']
        self.assertTrue(long and short)
        self.assertGreater(len(short), len(long))
        self.assertEqual(metrics['reserved_turns'], 16)
        for row in rows:
            label = row['teacher']
            self.assertIsNotNone(validate_improved_policy(label, row['choices']))
            check_turn(label, row['view']['context']['round'], job)
            fast = label_profile(label) == 'fast'
            self.assertEqual((label['requested_k'], label['c2']), (8, .5) if fast else (16, 1.))
            self.assertLessEqual(label['effective_k'], label['requested_k'])
        # Every long-search action is sampled from its root visit counts; short ones only in each seat's opening.
        self.assertTrue(all(r['teacher'].get('explored') is True for r in long))
        opening = [r for r in short if r['teacher'].get('explored')]
        self.assertTrue(opening and len(opening) < len(short))
        self.assertTrue(all(r['teacher']['played_action'] in r['teacher']['legal_ids'] for r in rows))
        receipt = metrics['cpu_inference']['revision']
        self.assertEqual(receipt['job'], revision)
        self.assertEqual(receipt['short_labels'], len(short))
        self.assertGreater(receipt['changed']['long'], 0)
        self.assertGreaterEqual(receipt['sampled']['long'], len(long))
        self.assertGreaterEqual(receipt['decisions']['short'], len(short))
        self.assertGreater(receipt['decisions']['short'], receipt['sampled']['short'])
        # The worker already ran the table's one batched logit check as it finished; take() only reports it.
        settled = verifier.settle(job['expected']['job_id'])
        self.assertEqual(settled['rows'], len(rows))
        accepted, rejected = verifier.take(result, job, self.model)
        self.assertEqual(len(accepted)+sum(rejected.values()), len(rows))
        # Every row's logit parity ran once, as the table finished: no turn waited for the shared model lock.
        self.assertEqual(result['metrics']['deferred_logit_check'], settled)
        # The same job plays the same explored game again.
        _, again, _ = self.run_all_seat(0., budget, revision)
        played = lambda table: [(r['teacher']['turn_key'], r['teacher']['played_action'], r['teacher'].get('explored')) for r in table['rows']]
        self.assertEqual(played(result), played(again))
        # Without the revision the same binary labels long searches only and never explores.
        _, plain, _ = self.run_all_seat(0., budget)
        self.assertTrue(all('profile' not in r['teacher'] and 'explored' not in r['teacher'] for r in plain['rows']))
        self.assertNotIn('revision', plain['metrics']['cpu_inference'])

    def test_a_g1490_restart_table_labels_long_searches_only_and_still_explores(self):
        from brass_ai.recovery import training_revision_g1491 as R
        from brass_ai.recovery.teacher_cotton import validate_receipt
        from brass_ai.recovery.teacher_labels import check_turn, label_profile, validate_improved_policy
        budget = dict(mode='count', full=64, full_k=16, fast=32, fast_k=8, c_scale=1.)
        revision = R._job_block(R.block(variant=R.RESTART)['exploration'], short_labels=False)
        job, result, verifier = self.run_all_seat(0., budget, revision)
        self.assertEqual(R.validate_job(job), revision)
        rows, metrics = result['rows'], result['metrics']
        self.assertTrue(rows)
        self.assertTrue(all(label_profile(r['teacher']) == 'full' for r in rows))
        for row in rows:
            self.assertIsNotNone(validate_improved_policy(row['teacher'], row['choices']))
            check_turn(row['teacher'], row['view']['context']['round'], job)
            self.assertIs(row['teacher'].get('explored'), True)
        receipt = metrics['cpu_inference']['revision']
        self.assertEqual((receipt['job'], receipt['short_labels']), (revision, 0))
        # Short searches still explore in each seat's opening; they just leave no label.
        self.assertGreater(receipt['sampled']['short'], 0)
        self.assertGreater(receipt['decisions']['short'], receipt['sampled']['short'])
        validate_receipt(receipt['cotton'], job, rows)
        settled = verifier.settle(job['expected']['job_id'])
        self.assertEqual(settled['rows'], len(rows))
        accepted, rejected = verifier.take(result, job, self.model)
        self.assertEqual(len(accepted)+sum(rejected.values()), len(rows))

    def test_a_two_hundredths_blend_from_the_fifty_step_stage_b_is_accepted(self):
        budget = dict(mode='count', full=64, full_k=16, fast=24, fast_k=8)
        job, result, verifier = self.run_all_seat(.02, budget)
        self.assertEqual((job['search_value']['w'], job['search_value']['value_source']), (.02, 'corrected'))
        rows = result['value_rows']
        self.assertTrue(rows and all(r['search_value'] == job['search_value'] for r in rows))
        for row in rows[:4]:
            utility, delta = self.root_heads(row)
            blended = utility+torch.tensor(.02, dtype=torch.float32)*delta
            np.testing.assert_allclose(row['v'], blended[0].double().numpy(), rtol=2e-4, atol=2e-4)
        accepted, rejected = verifier.take(result, job, self.model)
        self.assertEqual(len(accepted)+sum(rejected.values()), len(result['rows']))

    def test_blended_search_differs_from_the_unblended_one(self):
        budget = dict(mode='count', full=64, full_k=16, fast=24, fast_k=8)
        _, zero, _ = self.run_all_seat(0., budget)
        _, blended, _ = self.run_all_seat(.45, budget)
        self.assertNotEqual([r['counts'] for r in zero['value_rows']], [r['counts'] for r in blended['value_rows']])

    @unittest.skip('Fixture job is not all-seat, so its fast probe cannot produce valid labels')
    def test_timed_budget_calibrates_on_every_lane_and_runs_one_turn(self):
        from test_b04 import job as frozen_job
        from brass_ai.rust_backend.lab_teacher import CONTRACT, Worker, calibrate_cpu
        from brass_ai.rust_backend.lab_teacher_finalization import TurnVerifier
        from brass_ai.learning.contracts import digest as seal
        native = Path(os.environ['BRASS_CPU_NATIVE'])
        state = json.loads((native/'brass-ai-core/tests/data/initial-state.json').read_text())
        origin = dict(state=state, rail_keys=list(range(64)))
        job = frozen_job(self.model)
        inference = dict(contract=R.CPU_CONTRACT, runtime=self.runtime, bundle=self.bundles[self.identity], budget=dict(mode='timed'))
        calibration = calibrate_cpu(os.environ['BRASS_CPU_TEACHER'], os.environ['BRASS_CPU_CORE'], [origin], job, inference, [None, None])
        self.assertEqual(calibration['device'], 'cpu-ort')
        self.assertGreater(calibration['estimate_ns'], 1000)
        self.assertEqual({s['lane'] for s in calibration['samples']}, {0, 1})
        request = dict(contract=CONTRACT, op='turn_probe', job=job, origin=origin, core_source=os.environ['BRASS_CPU_CORE'],
                       probe_profile='fast', estimate_ns=calibration['estimate_ns'], calibration_id=seal(calibration), inference=inference)
        cancel = threading.Event()
        verifier = TurnVerifier(self.model, [dict(generation=10, model_id=self.identity)], [job], cancel, diagnostic=True)
        result = Worker(os.environ['BRASS_CPU_TEACHER'], request, None, cancel).collect(
            job['expected'], os.environ['BRASS_CPU_CORE'], verifier=verifier, turn_probe=True)
        self.assertEqual(len(result['metrics']['turns']), 1)
        self.assertGreater(result['metrics']['searches'][0]['completed'], 0)

    def test_teacher_rejects_a_foreign_model_bundle(self):
        from test_b04 import job as frozen_job
        from brass_ai.rust_backend.lab_teacher import Worker
        native = Path(os.environ['BRASS_CPU_NATIVE'])
        state = json.loads((native/'brass-ai-core/tests/data/initial-state.json').read_text())
        job = copy.deepcopy(frozen_job(self.model))
        job['expected']['model_id'] = '0'*64
        job['seats'] = ['0'*64]*4
        worker = Worker(os.environ['BRASS_CPU_TEACHER'], self.teacher_request(job, state), None, threading.Event())
        with self.assertRaises(ValueError):
            worker.collect(job['expected'], os.environ['BRASS_CPU_CORE'], verifier=SimpleNamespace())
        self.assertIn('differs from the frozen job', worker.stderr.decode())


def fixture_cpu_actors(connections, root, build, execution, mono, lease, cpu, stop, directory, inference, core, started):
    """Spawned lane that starts like a CPU lane but answers from a fixture part."""
    from concurrent.futures import ThreadPoolExecutor
    from brass_ai.rust_backend import lab_pool as P
    assert lease is None and inference['contract'] == R.CPU_CONTRACT
    def lane(connection):
        connection.send(dict(ready=True, startup=dict(spawn_seconds=.1, python_import_seconds=.2, lane_wait_seconds=0., lease_seconds=0.,
            session_import_seconds=.01, mono_spawn_seconds=.01, mono_bootstrap_seconds=.15, total_seconds=.5, ready_time=__import__('time').time())))
        while not stop.is_set():
            if not connection.poll(.05):
                continue
            command = connection.recv()
            if command is None:
                return
            part = json.loads(Path(build['fixture']).read_text())
            connection.send(dict(result=P.part_write(directory, part)))
    with ThreadPoolExecutor(max_workers=len(connections)) as pool:
        list(pool.map(lane, connections))


EXPORT = SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g1290-20261109T041246972332Z'


@unittest.skipUnless((EXPORT/'state.json').is_file(), 'Needs the g1290 export')
class TrainingEntry(unittest.TestCase):
    def test_the_on_mode_check_sees_the_signed_recipe_with_a_64_lane_topology(self):
        """The exact call that stopped the second lab run."""
        from unittest.mock import patch
        from brass_ai.bank import read
        from brass_ai.rust_backend import lab_backend as B, lab_pool as P
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        state = read(EXPORT/'state.json')
        config = state['signature']['config']
        self.assertTrue(is_on(config))
        settings = dict(R.from_config(config), native_inference=R.cpu_block())
        with R.scope(settings):
            effective = R.effective_execution(config['execution'])
        self.assertEqual(effective['processes'], 64)
        self.assertFalse(is_on(dict(config, execution=effective)))
        seen = {}
        def collect(pool, jobs, on_records=None):
            seen.update(execution=pool.execution, config=pool.config)
            pool.last_collection_report = dict(wall_seconds=0.)
            return ['record']
        build = dict(_receipt_sha256='0'*64, b01=dict(sha256='1'*64))
        cpu = dict(contract=R.CPU_CONTRACT)
        service = SimpleNamespace(cpu=None, error=None)
        with R.scope(settings), patch('brass_ai.recovery.execution.ActorPool'), \
                patch.object(B, 'validate_state', return_value=dict(mode='rust', build=dict(path='x'))), \
                patch.object(B, 'check_build', return_value=build), \
                patch('brass_ai.rust_backend.lab_perf_runtime.resolve', return_value=build), \
                patch('brass_ai.rust_backend.lab_cpu_runtime.ordinary_plan', return_value=(build, cpu)), \
                patch.object(P.NativePool, 'collect', collect):
            pool = B.RoutedPool(SOURCE_ROOT, 'host', service, config, state, mono='mono', execution=effective)
            self.assertEqual(pool.collect([dict(provenance=dict(purpose='training'))]), ['record'])
        self.assertIs(seen['execution'], effective)
        self.assertIs(seen['config'], config)


class CpuPool(unittest.TestCase):
    def test_cpu_lanes_need_no_channels_and_report_their_startup(self):
        import multiprocessing
        from unittest.mock import Mock, patch
        from brass_ai.recovery.resources import CPUAdmission
        from brass_ai.rust_backend import lab_pool as P
        from brass_ai.step5.league import write_json
        from test_b03 import family_fixture
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            out = Path(directory)
            f, _, c = family_fixture()
            job = f['frozen_job']
            job['path'] = str(out/'job.jsonl.gz')
            write_json(out/'fixture.json', f['parent'])
            cpu = CPUAdmission(multiprocessing.get_context('spawn'), 4, 1)
            service = SimpleNamespace(cpu=cpu, error=None, channels=Mock(side_effect=AssertionError('CPU lanes lease no channels')),
                                      release_channels=Mock())
            execution = dict(processes=1, hosts_per_process=1, worker_threads=1, cpu_budget=4, cpu_reserve=1)
            build = dict(_receipt_sha256=f['producer']['build_sha256'], b01={'sha256': f['producer']['b01_sha256']}, fixture=str(out/'fixture.json'))
            settings = dict(contract=R.CPU_CONTRACT, pinning='none', model_cache=8, runtime=dict(path='runtime', sha256='0'*64),
                            bundles={}, exported_bundles=0)
            pool = P.NativePool(SOURCE_ROOT, service, execution, build, c, f['producer']['signature'], scope='ordinary', cpu=settings)
            with patch.object(P, '_actors', fixture_cpu_actors):
                records = pool.collect([job])
            self.assertEqual(len(records), 1)
            service.channels.assert_not_called()
            service.release_channels.assert_not_called()
            report = pool.last_collection_report
            breakdown = report['startup_breakdown']
            self.assertEqual(breakdown['lanes_reported'], 1)
            self.assertFalse(breakdown['waits_for_all_lanes'])
            self.assertEqual(breakdown['phases']['mono_bootstrap_seconds']['p50'], .15)
            self.assertGreater(report['actor_startup_seconds'], 0)
            self.assertEqual(report['collector_runtime']['cpu_inference']['contract'], R.CPU_CONTRACT)
            self.assertFalse(pool.processes)


@unittest.skipUnless(NATIVE and os.environ.get('BRASS_CPU_BOOTSTRAP'), 'Needs the built workers and a bootstrap')
class RealLane(unittest.TestCase):
    def test_a_spawned_cpu_lane_reports_mono_bootstrap_and_finishes_a_game(self):
        """The real lane: spawn, Mono probe, Rust collector, CPU evaluator, raw export."""
        import multiprocessing
        from brass_ai.recovery.execution import neural_spec
        from brass_ai.recovery.resources import CPUAdmission
        from brass_ai.rust_backend import lab_pool as P, lab_shared
        from brass_ai.rust_backend.lab_contract import pin
        from brass_ai.rust_backend.lab_cpu_models import ensure
        model = headed_model()
        identity = model_id(model)
        mp = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as directory:
            bundles, _ = ensure(directory, {identity: model}, [identity])
            runtime = pin(os.environ['BRASS_CPU_RUNTIME'])
            cpu = dict(contract=R.CPU_CONTRACT, runtime=dict(path=runtime['path'], sha256=runtime['sha256']), bundles=bundles,
                       exported_bundles=1, model_cache=2, pinning='none')
            build = dict(bootstrap=dict(binary=dict(path=os.environ['BRASS_CPU_BOOTSTRAP'])), collector=dict(path=os.environ['BRASS_CPU_COLLECTOR']))
            execution = dict(max_decisions=5000, batch_size=32, max_candidate_cells=8192, worker_threads=1, processes=1, hosts_per_process=1,
                             collector_validation='production_fast', replay_bindings=True, native_inference=R.cpu_block(workers=1))
            spec = neural_spec(identity, 0., 'neutral', version=3)
            job = dict(index=0, seed=19, sampling_seed=31, policies=[spec]*4, provenance={}, stable_game_id='lane-fixture')
            parent, child = mp.Pipe()
            stop = mp.Event()
            admission = CPUAdmission(mp, 2, 0)
            process = mp.Process(target=P._actors, args=([child], str(SOURCE_ROOT), build, execution, 'mono', None, admission, stop,
                                                         str(Path(directory)/'parts'), cpu, None, __import__('time').time()))
            registry = lab_shared.Registry(stop)
            process.start()
            child.close()
            try:
                self.assertTrue(parent.poll(120), 'lane never became ready')
                ready = parent.recv()
                self.assertTrue(ready['ready'], ready)
                startup = ready['startup']
                self.assertGreater(startup['mono_bootstrap_seconds'], 0)
                self.assertGreater(startup['total_seconds'], startup['mono_bootstrap_seconds'])
                # Strict games are audited in full. Training validates most games the fast way.
                for number, validation in enumerate((None, dict(semantic=False, branch=False))):
                    parent.send(dict(job=dict(job, seed=19+number, stable_game_id=f'lane-fixture-{number}'), continuation=None, validation=validation))
                    result = None
                    while result is None:
                        self.assertTrue(parent.poll(240), 'lane stopped answering')
                        message = parent.recv()
                        self.assertNotIn('error', message, message.get('error'))
                        if 'shared_allocate' in message:
                            parent.send(dict(shared_lease=registry.allocate(message['shared_allocate']['size'])))
                        elif 'result' in message:
                            result = message['result']
                    self.assertEqual(result['format'], lab_shared.FORMAT)
                    self.assertGreater(result['size'], 0)
            finally:
                stop.set()
                parent.send(None)
                process.join(30)
                if process.is_alive():
                    process.terminate()
                registry.close()


@unittest.skipUnless(os.environ.get('BRASS_CPU_RATING_GAME'), 'Set BRASS_CPU_RATING_GAME for a real hosted game')
class RatingGame(unittest.TestCase):
    """One complete game through the real rating actor pool, on each backend."""
    def play(self, directory, model, *, cpu):
        from brass_ai.bank import read
        from brass_ai.recovery.execution import ActorPool, neural_spec
        from brass_ai.recovery.inference import InferenceService
        from brass_ai.recovery.replay import read_shards
        from brass_ai.step5.league import seeds_for
        config = read(SOURCE_ROOT/'configs/step5/recovery-smoke.json')
        execution = dict(config['execution'], processes=1, hosts_per_process=1, heavy_processes=1)
        seed = seeds_for(22517000, 1, 'validation')[0][0]
        with InferenceService(execution, 'cpu') as service:
            identity = service.register(model)
            spec = neural_spec(identity, 0., 'neutral')
            provenance = dict(producer='plackett-luce-rating/1.0.0', purpose='evaluation', kind='rating',
                              seed=seed, sampling_seed=99017000, rotation=0, spec=digest(spec))
            job = dict(index=0, seed=seed, rotation=0, sampling_seed=99017000, policies=[spec]*4,
                       path=str(Path(directory)/'game.jsonl.gz'), provenance=provenance)
            with ActorPool(SOURCE_ROOT, config['host'], service, execution, mono=os.environ.get('BRASS_CPU_MONO'),
                           cpu_inference=cpu) as pool:
                records = pool.collect([job])
            positions = service.metrics['positions']
        games = read_shards([r['path'] for r in records], partition='validation',
                            receipts={r['path']: r.get('features') for r in records})
        self.assertEqual(len(games), 1)
        return games[0], positions

    def test_cpu_rating_game_matches_the_service_path_and_never_uses_it(self):
        from brass_ai.rust_backend.lab_cpu_models import ensure
        model = model_fixture().eval()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as games:
            bundles, _ = ensure(root, {model_id(model): model}, [model_id(model)])
            cpu = dict(contract=R.CPU_CONTRACT, bundles=bundles, exported_bundles=1)
            reference, used = self.play(Path(games)/'service', model, cpu=None)
            self.assertGreater(used, 0)
            actual, used = self.play(Path(games)/'cpu', model, cpu=cpu)
            self.assertEqual(used, 0)
        self.assertEqual(actual['result'], reference['result'])
        self.assertEqual([r['chosen'] for r in actual['records']], [r['chosen'] for r in reference['records']])
        for ours, theirs in zip(actual['records'], reference['records']):
            self.assertAlmostEqual(ours['behavior']['probability'], theirs['behavior']['probability'], delta=2e-3)


if __name__ == '__main__':
    unittest.main()
