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
        self.assertEqual(block['teacher_budget'], dict(mode='count', full=256, fast=48))
        self.assertEqual({block[k] for k in ('ordinary', 'teacher', 'rating')}, {'cpu-ort'})

    def test_invalid_blocks_are_rejected(self):
        for change in (dict(workers=0), dict(pinning='sometimes'), dict(teacher_budget=dict(mode='count', full=8, fast=48)),
                       dict(teacher_budget=dict(mode='count', full=256)), dict(teacher='tpu'), dict(extra=1)):
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
        options = dict(workers=64, teacher_budget='count', teacher_full=256, teacher_fast=48, no_pinning=False)
        block = launcher.inference_choice(SimpleNamespace(inference='cpu', **options))
        self.assertEqual(block, R.cpu_block())
        self.assertIsNone(launcher.inference_choice(SimpleNamespace(inference='gpu', **options)))
        self.assertEqual(launcher.inference_choice(SimpleNamespace(inference='keep', **options)), 'keep')
        timed = launcher.inference_choice(SimpleNamespace(inference='cpu', **dict(options, teacher_budget='timed')))
        self.assertEqual(timed['teacher_budget'], dict(mode='timed'))

    def test_removing_the_block_returns_the_exact_gpu_settings(self):
        base = dict(update={}, execution=dict(processes=24, hosts_per_process=2))
        selected = R.configured(base, dict(R.from_config(base), native_inference=R.cpu_block()))
        rolled_back = R.configured(selected, {k: v for k, v in R.from_config(selected).items() if k != 'native_inference'})
        self.assertNotIn('native_inference', rolled_back['execution'])
        self.assertEqual(R.recipe(rolled_back), base)


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
        cls.directory = tempfile.TemporaryDirectory()
        from brass_ai.rust_backend.lab_cpu_models import ensure
        cls.bundles, _ = ensure(cls.directory.name, {model_id(cls.model): cls.model}, [model_id(cls.model)])
        cls.identity = model_id(cls.model)
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

    def teacher_request(self, job, state):
        from brass_ai.rust_backend.lab_teacher import CONTRACT
        return dict(contract=CONTRACT, op='run', job=job, origin=dict(state=state, rail_keys=list(range(64))),
                    core_source=os.environ['BRASS_CPU_CORE'], estimate_ns=1000, calibration_id='fixed-count',
                    inference=dict(contract=R.CPU_CONTRACT, runtime=self.runtime, bundle=self.bundles[self.identity],
                                   budget=dict(mode='count', full=64, fast=8)))

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

    @unittest.skip('Fixture lacks the stage schedule, so its blend is not the expected weight')
    def test_all_seat_table_blends_the_search_head_like_the_gpu_path(self):
        from brass_ai.learning.search_value import correction
        from brass_ai.rust_backend.lab_teacher import Worker
        from brass_ai.rust_backend.lab_teacher_finalization import TurnVerifier
        native = Path(os.environ['BRASS_CPU_NATIVE'])
        state = json.loads((native/'brass-ai-core/tests/data/initial-state.json').read_text())
        job = self.all_seat_job(.45)
        self.assertEqual(job['search_value']['w'], .5)
        cancel = threading.Event()
        verifier = TurnVerifier(self.model, [dict(generation=10, model_id=self.identity)], [job], cancel)
        worker = Worker(os.environ['BRASS_CPU_TEACHER'], self.teacher_request(job, state), None, cancel)
        result = worker.collect(job['expected'], os.environ['BRASS_CPU_CORE'], verifier=verifier)
        self.assertEqual(len(result['value_rows']), 64)
        self.assertEqual(result['metrics']['searched_turns'], 64)
        rows = [r for r in result['value_rows'] if r['n'] == 0]
        self.assertTrue(rows or result['value_rows'])
        first = result['value_rows'][0]
        self.assertEqual(first['search_value'], job['search_value'])
        states = torch.as_tensor(first['state'], dtype=torch.float32)[None]
        mapping = torch.as_tensor(first['mapping'])[None]
        remaining = torch.as_tensor(first['remaining'], dtype=torch.float32)[None]
        with torch.inference_mode():
            encoded = self.model.encode_state(states)
            utility = self.model.heads_from_encoded(encoded, mapping, remaining)[3]
            delta = correction(self.model, encoded, mapping, remaining)
        np.testing.assert_allclose(first['v'], (utility+.5*delta)[0].double().numpy(), rtol=2e-4, atol=2e-4)
        self.assertGreater(float(delta.abs().max()), 1e-5)

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
