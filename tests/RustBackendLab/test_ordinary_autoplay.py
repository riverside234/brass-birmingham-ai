"""Build step 9: ordinary games played inside the CPU collector (lab_autoplay).

Routing and hand-off checks run anywhere. The native parity case plays one
production-like cotton table through both paths and needs BRASS_CPU_COLLECTOR
and BRASS_CPU_BOOTSTRAP (two games, fixture weights).
"""
import os
from pathlib import Path
import random
import tempfile
import threading
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ordinary_cotton
from brass_ai.rust_backend import lab_autoplay as A
from brass_ai.rust_backend.lab_cpu_overlay import PATCHES, apply

OPENING = dict(contract='opening-temperature/4.0.0', temperature=1.3, action_decisions_per_seat=3, seats='neural')


def cotton_job(identity):
    from brass_ai.recovery.execution import neural_spec
    spec = neural_spec(identity, 0., 'mixed', version=3, opening=OPENING)
    index, sampling = 16, 134381537
    assignment = ordinary_cotton.assignment(index, sampling, probability=ordinary_cotton.LIMITED['probability'],
                                            contract=ordinary_cotton.RAIL_NETWORK_CONTRACT)
    proof = dict(kind='selfplay', purpose='training', ordinary_game_index=index, sampling_seed=sampling,
                 **{ordinary_cotton.KEY: assignment})
    return dict(index=index, seed=20309165, sampling_seed=sampling, policies=[spec]*4, provenance=proof,
                stable_game_id='autoplay-parity', rotation=0)


class FakeLane:
    hello = dict(autoplay=A.CONTRACT)


def policies(lane, *, focus='neutral', version=3):
    from brass_ai.step5.policy import NeuralPolicy
    from brass_ai.rust_backend.lab_cpu_inference import NativeCpuEvaluator
    result = []
    for _ in range(4):
        evaluator = NativeCpuEvaluator(lane, 'm'*64, {})
        evaluator.model_id = 'm'*64
        with patch('brass_ai.step5.policy.policy_identity', return_value='step5-neural/p'):
            p = NeuralPolicy(None, evaluator=evaluator, exploration=.1, focus=focus, exploration_version=version,
                             opening=OPENING if version == 3 else None)
        result.append(p)
    return result


class Routing(unittest.TestCase):
    def setUp(self):
        self.lane = FakeLane()
        self.rngs = [random.Random(s) for s in range(4)]
        self.job = dict(provenance={}, policies=[])
        self.cotton = SimpleNamespace(seats=(), probability=0.)
        self.options = dict(continuation=None, compare=None, raw_export=True, validation=dict(mode='production_fast'),
                            diagnostic=False)

    def request(self, assigned, **changes):
        return A.request(self.job, assigned, self.rngs, self.cotton, self.lane, **dict(self.options, **changes))

    def test_production_seats_hand_over_their_rng_state_without_a_draw(self):
        before = [r.getstate() for r in self.rngs]
        settings, reason = self.request(policies(self.lane))
        self.assertIsNone(reason)
        self.assertEqual([r.getstate() for r in self.rngs], before)
        self.assertEqual(settings['contract'], A.CONTRACT)
        seat = settings['seats'][0]
        self.assertEqual(seat['rng'], list(before[0][1]))
        self.assertEqual((seat['exploration'], seat['temperature'], seat['opening']['temperature']), (.1, 1., 1.3))
        self.assertEqual((len(seat['style']), len(seat['intent'])), (4, 8))
        self.assertIsNone(settings['cotton'])

    def test_unsupported_games_keep_the_python_loop(self):
        assigned = policies(self.lane)
        self.assertEqual(self.request(assigned, validation=None)[1], 'strict')
        self.assertEqual(self.request(assigned, validation=dict(mode='strict'))[1], 'strict')
        self.assertEqual(self.request(assigned, compare=print)[1], 'route')
        self.assertEqual(self.request(assigned, raw_export=False)[1], 'route')
        self.job['branching'] = dict(contract='x')
        self.assertEqual(self.request(assigned)[1], 'route')
        del self.job['branching']
        self.assertEqual(self.request(policies(self.lane, version=2))[1], 'policy')
        self.assertEqual(self.request(policies(FakeLane()))[1], 'policy')  # another lane's evaluator
        self.assertEqual(self.request(policies(self.lane, focus='mixed'))[1], 'context')
        self.lane.hello = {}
        self.assertEqual(self.request(assigned)[1], 'collector')
        self.lane.hello = dict(autoplay=A.CONTRACT)
        with patch.dict(os.environ, {A.SWITCH: '0'}):
            self.assertEqual(self.request(assigned)[1], 'disabled')
        with patch.dict(os.environ, {A.SWITCH: 'yes'}), self.assertRaises(ValueError):
            self.request(assigned)

    def test_cotton_tables_carry_their_recipe(self):
        job = cotton_job('m'*64)
        self.job = job
        self.cotton = ordinary_cotton.TableExploration(job['provenance'])
        settings, _ = self.request(policies(self.lane))
        self.assertEqual(settings['cotton'], dict(contract=ordinary_cotton.RAIL_NETWORK_CONTRACT,
                         seats=list(self.cotton.seats), probability=ordinary_cotton.LIMITED['probability']))

    def test_play_restores_each_seat_rng_and_records_worker_time(self):
        states = [list(random.Random(10+s).getstate()[1]) for s in range(4)]
        reply = dict(wire=A.CONTRACT, token='t', rows=10, forced_rows=4, inferences=6, forward_ns=7, policy_ns=2*10**9,
                     elapsed_ns=3*10**9, rng=states)
        lane = SimpleNamespace(token='t', metrics=Counter(), autoplay=lambda settings, timeout: reply)
        timings = Counter()
        A.play(lane, {}, self.rngs, timings)
        self.assertEqual([list(r.getstate()[1]) for r in self.rngs], states)
        self.assertEqual((timings['autoplay_games'], timings['autoplay_decisions'], timings['policy_seconds'],
                          timings['native_step_seconds'], lane.metrics['cpu_forward_ns']), (1, 10, 2., 1., 7))
        with self.assertRaises(ValueError):
            A.play(SimpleNamespace(token='other', metrics=Counter(), autoplay=lambda settings, timeout: reply),
                   {}, self.rngs, timings)


class Overlay(unittest.TestCase):
    def test_the_collector_overlay_adds_the_autoplay_operation(self):
        main = dict(PATCHES['src/main.rs'])
        self.assertTrue(any('"autoplay" =>' in after for after in main.values()))
        with tempfile.TemporaryDirectory() as directory:
            apply(Path(directory)/'lab-native')
            self.assertTrue((Path(directory)/'lab-native/src/cpu_autoplay.rs').is_file())
            self.assertIn('pub mod cpu_autoplay;', (Path(directory)/'lab-native/src/lib.rs').read_text())
            self.assertIn('cpu_autoplay::CONTRACT', (Path(directory)/'lab-native/src/main.rs').read_text())
        rust = (SOURCE_ROOT/'lab-native-cpu/src/cpu_autoplay.rs').read_text()
        self.assertIn('pub const CONTRACT: &str = "%s";' % A.CONTRACT, rust)


@unittest.skipUnless(os.environ.get('BRASS_CPU_COLLECTOR') and os.environ.get('BRASS_CPU_BOOTSTRAP'),
                     'Needs the built CPU collector and a bootstrap')
class NativeParity(unittest.TestCase):
    """One cotton table through the Python loop and through autoplay: identical rows."""

    def play(self, job, autoplay, bundles):
        from brass_ai.recovery import collector_runtime as R
        from brass_ai.recovery.collector_runtime import from_config, install, recipe
        from brass_ai.rust_backend import lab_perf, lab_shared
        from brass_ai.rust_backend.lab_collection import collect_lane
        from brass_ai.rust_backend.lab_contract import pin
        from brass_ai.rust_backend.lab_cpu_inference import NoServiceEndpoint
        from brass_ai.rust_backend.lab_session import OrdinarySession
        from brass_ai.rust_backend.lab_transport import Pipe
        runtime = pin(os.environ['BRASS_CPU_RUNTIME'])
        cpu = dict(contract=R.CPU_CONTRACT, runtime=dict(path=runtime['path'], sha256=runtime['sha256']), bundles=bundles,
                   exported_bundles=1, model_cache=2, pinning='none')
        execution = dict(max_decisions=5000, batch_size=32, max_candidate_cells=8192, worker_threads=1, processes=1,
                         hosts_per_process=1, collector_validation='production_fast', replay_bindings=True,
                         native_inference=R.cpu_block(workers=1))
        install(from_config(dict(execution=execution))); execution = recipe(dict(execution=execution))['execution']
        endpoint = NoServiceEndpoint()
        command = [os.environ['BRASS_CPU_BOOTSTRAP']]
        if os.name != 'nt':
            command.insert(0, os.environ.get('BRASS_CPU_MONO', 'mono'))
        with patch.dict(os.environ, {A.SWITCH: '1' if autoplay else '0'}), Pipe(command, binary=False) as bootstrap, \
                OrdinarySession(str(SOURCE_ROOT), os.environ['BRASS_CPU_COLLECTOR'], endpoint, execution, None, cpu=cpu) as session:
            bootstrap.timeout = 120; bootstrap.call('init', seed=1); bootstrap.timeout = 15
            validation = lab_perf.selection(job, ordinal=1, first=False, selected_mode='production_fast')
            part = collect_lane(str(SOURCE_ROOT), os.environ['BRASS_CPU_COLLECTOR'], bootstrap, job, endpoint, execution,
                                session=session, raw_export=True, validation=validation)
        registry = lab_shared.Registry(threading.Event())
        handle = lab_shared.write(part, registry.allocate)
        try:
            return part.metadata['timings'], lab_shared.decode(lab_shared.read(handle))
        finally:
            registry.release(handle)

    def test_autoplay_rows_equal_the_python_loop_and_pass_the_semantic_audit(self):
        import numpy as np
        from brass_ai.learning.batching import model_id
        from brass_ai.rust_backend.lab_cpu_models import ensure
        from brass_ai.rust_backend.lab_families import validate_part
        from test_cpu_inference import headed_model
        model = headed_model(); identity = model_id(model)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'tmp') as directory:
            bundles, _ = ensure(directory, {identity: model}, [identity])
            job = cotton_job(identity)
            (before, python), (after, auto) = self.play(job, False, bundles), self.play(job, True, bundles)
        self.assertNotIn('autoplay_games', before)
        self.assertEqual(after['autoplay_games'], 1)
        a, b = python['game']['records'], auto['game']['records']
        self.assertEqual(len(a), len(b))
        for x, y, ex, ey in zip(a, b, python['evidence'], auto['evidence'], strict=True):
            for key in ('step', 'seat', 'view', 'choices', 'chosen', 'style', 'intent', 'teacher'):
                self.assertEqual(x[key], y[key], (x['step'], key))
            # Each path samples from its own float32 softmax (NumPy and Rust exp differ in the last bit), so
            # denominators agree to the raw outputs' precision; within a path the audit below holds them to 1e-12.
            self.assertAlmostEqual(x['behavior']['probability'], y['behavior']['probability'], delta=1e-6)
            self.assertEqual({k: v for k, v in x['behavior'].items() if k != 'probability'},
                             {k: v for k, v in y['behavior'].items() if k != 'probability'})
            np.testing.assert_allclose(ex['raw_probabilities'], ey['raw_probabilities'], rtol=0, atol=1e-6)
        for e, f in zip(python['game']._native_arrays, auto['game']._native_arrays, strict=True):
            np.testing.assert_array_equal(e.state, f.state)
            np.testing.assert_array_equal(e.actions, f.actions)
        self.assertEqual((python['game']['result'], python['scoring']), (auto['game']['result'], auto['scoring']))
        # The publisher's full audit replays all four seats' draws against the autoplayed rows.
        validate_part(dict(auto, branch_expected=False), job, semantic=True)


if __name__ == '__main__':
    unittest.main()
