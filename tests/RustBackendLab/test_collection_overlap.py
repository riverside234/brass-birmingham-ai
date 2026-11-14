"""Teacher tables beside ordinary games: CPU hand-over, failures and c_scale."""
import math
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from brass_ai.learning.contracts import digest
from brass_ai.recovery import collector_runtime as R
from brass_ai.rust_backend.lab_cpu_runtime import CorePlan
from brass_ai.rust_backend.lab_pool import Lanes
from brass_ai.rust_backend.lab_shadow import TeacherRun
from brass_ai.rust_backend import lab_teacher as T
from brass_ai.rust_backend.lab_teacher_overlay import CONTRACT


class Plan(unittest.TestCase):
    def test_physical_cores_precede_their_smt_siblings(self):
        from pathlib import Path
        from brass_ai.rust_backend import lab_cpu_runtime as CPU
        def topology(path):
            cpu = int(path.parents[1].name.removeprefix('cpu'))
            return str(cpu//4) if path.name=='physical_package_id' else f'{cpu//2*2}-{cpu//2*2+1}'
        with mock.patch.object(CPU.os,'sched_getaffinity',return_value=set(range(8)),create=True), \
                mock.patch.object(Path,'read_text',autospec=True,side_effect=topology):
            self.assertEqual(CPU.core_allocation(8),[0,4,2,6,1,5,3,7])

    def test_small_ordinary_batches_prefer_free_cpus(self):
        from pathlib import Path
        from brass_ai.rust_backend import lab_cpu_runtime as CPU

        plan = CorePlan((0, 1))
        with mock.patch.object(CPU.os, 'sched_getaffinity', return_value={0, 1, 2, 3}, create=True), \
                mock.patch.object(Path, 'read_text', return_value='0'):
            self.assertEqual(CPU.core_allocation(2, held=plan.holds), [2, 3])
            plan.release(0)
            self.assertEqual(CPU.core_allocation(2, held=plan.holds), [0, 2])

    def test_a_released_cpu_stops_being_held_and_reaches_a_later_listener(self):
        plan = CorePlan((3, 7))
        self.assertTrue(plan.holds(3))
        plan.release(3)
        plan.release(3)
        self.assertFalse(plan.holds(3))
        seen = []
        plan.attach(seen.append, lambda error: None)
        self.assertEqual(seen, [3])
        plan.release(7)
        self.assertEqual(seen, [3, 7])

    def test_cpus_outside_the_plan_are_ignored(self):
        plan = CorePlan((1,))
        seen = []
        plan.attach(seen.append, lambda error: None)
        plan.release(2)
        self.assertEqual(seen, [])

    def test_a_failure_reaches_the_ordinary_pool_even_when_it_attaches_later(self):
        plan = CorePlan((1,))
        error = RuntimeError('table failed')
        plan.fail(error)
        raised = []
        plan.attach(lambda core: None, raised.append)
        self.assertEqual(raised, [error])

    def test_detached_pools_receive_nothing(self):
        plan = CorePlan((1,))
        seen = []
        plan.attach(seen.append, seen.append)
        plan.detach()
        plan.release(1)
        plan.fail(RuntimeError('late'))
        self.assertEqual(seen, [])


class LaneQueue(unittest.TestCase):
    def test_a_lane_on_a_held_cpu_waits_for_its_table(self):
        plan = CorePlan((5,))
        lanes = Lanes(plan)
        lanes.add('free', 6)
        lanes.add('parked', 5)
        self.assertEqual(lanes.get(), 'free')
        self.assertEqual(lanes.parked_lanes, 1)
        with self.assertRaises(queue.Empty):
            lanes.queue.get_nowait()
        plan.attach(lanes.release, lambda error: None)
        plan.release(5)
        self.assertEqual(lanes.get(), 'parked')

    def test_a_table_that_ended_before_the_lane_existed_never_parks_it(self):
        plan = CorePlan((5,))
        plan.release(5)
        lanes = Lanes(plan)
        lanes.add('lane', 5)
        self.assertEqual(lanes.parked_lanes, 0)
        self.assertEqual(lanes.get(), 'lane')

    def test_without_a_plan_or_pinning_every_lane_is_free(self):
        for plan, core in ((None, 4), (CorePlan((4,)), None)):
            lanes = Lanes(plan)
            lanes.add('lane', core)
            self.assertEqual(lanes.get(), 'lane')

    def test_a_teacher_release_defers_process_start_until_an_ordinary_job_claims_it(self):
        plan = CorePlan((5,))
        lanes = Lanes(plan)
        start = mock.Mock()
        lanes.add('lazy', 5, start=start)
        plan.attach(lanes.release, lambda error: None)
        self.assertEqual(lanes.capacity, 0)
        plan.release(5)
        self.assertEqual(lanes.capacity, 1)
        start.assert_not_called()
        lanes.add('already running', 6)
        self.assertEqual(lanes.get(), 'already running')
        start.assert_not_called()
        connection = lanes.get()
        self.assertEqual(connection, 'lazy')
        lanes.put(connection)
        self.assertEqual(lanes.get(), 'lazy')
        start.assert_called_once()


class Background(unittest.TestCase):
    def run_of(self, collect):
        finished = []
        plan = CorePlan((1,))
        collector = SimpleNamespace(cancel=threading.Event())
        return TeacherRun(collector, plan, collect, finished.append), finished, plan, collector

    def test_finish_returns_the_tables_to_the_admission_step(self):
        run, finished, _, _ = self.run_of(lambda: ['table'])
        run.finish()
        self.assertEqual(finished, [['table']])

    def test_a_table_failure_is_reraised_and_stops_ordinary_games_at_once(self):
        error = RuntimeError('native teacher failed')
        started = threading.Event()
        def collect():
            started.wait(5)
            raise error
        run, finished, plan, _ = self.run_of(collect)
        raised = []
        plan.attach(lambda core: None, raised.append)
        started.set()
        with self.assertRaises(RuntimeError) as caught:
            run.finish()
        self.assertIs(caught.exception, error)
        self.assertEqual(raised, [error])
        self.assertEqual(finished, [])

    def test_abort_cancels_running_tables_and_waits_for_them(self):
        cancelled = threading.Event()
        def collect():
            cancelled.wait(5)
            return []
        run, _, _, collector = self.run_of(collect)
        with mock.patch.object(collector, 'cancel') as cancel:
            cancel.set.side_effect = cancelled.set
            run.abort()
        self.assertFalse(run.thread.is_alive())

    def test_finishing_twice_is_refused(self):
        run, _, _, _ = self.run_of(lambda: [])
        run.finish()
        with self.assertRaises(ValueError):
            run.finish()


class Barrier(unittest.TestCase):
    """The real ShadowCollector on the CPU path, with the Rust workers replaced."""
    def setUp(self):
        import copy
        import tempfile
        from pathlib import Path
        from brass_ai.bank import read
        from brass_ai.recovery import entropy_schedule as E
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.learning.batching import model_id
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.learning.utility_model import UtilityNet, utility_config
        from brass_ai.rust_backend import lab_backend as B, lab_cpu_runtime as CPU, lab_shadow as S, lab_shadow_config as C
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        from brass_ai.step5.league import write_json
        self.root = Path(SOURCE_ROOT)
        config = C.configure(read(self.root/'configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json'))
        schedule = copy.deepcopy(config['update'][E.KEY])
        self.state = dict(iteration=1257, signature={'config': config}, native_backend={'build': {}},
            history=[dict(generation=1257, sha256='a'*64)],
            exploration={E.KEY: E.seal(dict(contract=E.STATE, config=schedule, started_at=schedule['start_generation'], progress=100, held=0,
                                            last_generation=1257, insolvency=copy.deepcopy(config['insolvency']),
                                            insolvency_steps_applied=E.step_count(schedule, config['insolvency']), reset_count=0))})
        self.model = UtilityNet(utility_config(ModelConfig.preset('small'), config['teacher_phase']['search']['reward']))
        self.state['history'][-1]['model_id'] = model_id(self.model)
        self.jobs, _ = plan_jobs(config, self.state, model_id=model_id(self.model), export_hash='a'*64, epoch=1257, ready=True,
                                 native_source=source_fingerprint())
        for job in self.jobs:
            job['readiness'] = dict(ready=True, reward_id=job['expected']['reward_id'])
        temp = tempfile.TemporaryDirectory(dir=self.root/'artifacts')
        self.addCleanup(temp.cleanup)
        self.run_path = Path(temp.name)
        write_json(self.run_path/'b01/inputs/fixtures.json',
                   dict(roots=[dict(view=dict(context=dict(phase='action')), expected=dict(choices=[1, 2], state={}))]))
        build = dict(binary=dict(path='fake'), core_source='qualified', bootstrap=dict(binary=dict(path='fake')))
        plan = dict(binary='binary', core_source='qualified', inference=dict(budget=dict(R.TEACHER_BUDGET)),
                    receipt={}, exported_bundles=0)
        self.gate = threading.Event()
        self.created = []
        for patcher in (mock.patch.object(S, 'teacher_build', return_value=build),
                        mock.patch.object(B, 'checked_pin', return_value=dict(b01=dict(path=str(self.run_path/'b01/qualification.json')))),
                        mock.patch.object(CPU, 'teacher_plan', return_value=plan),
                        mock.patch.object(T, 'count_calibration', return_value=dict(estimate_ns=1)),
                        mock.patch.object(T, 'TeacherCollector', side_effect=self.collector)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.gate.set)
        self.collector_module = S
        self.admitted = []
        self.phase = SimpleNamespace(options={'mode': 'shadow'}, receipt={'ready': True}, jobs=self.jobs, search_value=None, metrics={})
        self.phase.begin = mock.Mock(return_value=self.jobs)
        def admit(results, model, *, verifier=None):
            self.admitted.append(len(results))
            self.phase.metrics = {'tables': {'train': 28, 'heldout': 4}}
        self.phase.admit_collection = mock.Mock(side_effect=admit)
        self.service = SimpleNamespace(metrics={}, cpu=SimpleNamespace(capacity=64))

    def collector(self, *args, **kwargs):
        worker = SimpleNamespace(verifier=SimpleNamespace(cache={}), inference=None, cancel=threading.Event())
        self.created.append(kwargs)
        def collect(batch, *, on_result):
            self.gate.wait(10)
            output = []
            for job in batch:
                result = dict(job_id=job['expected']['job_id'], rows=[], metrics=dict(turns=[dict(profile='full', post_search_ns=0)]))
                worker.verifier.cache[result['job_id']] = {}
                result['metrics']['shadow_cache_seconds'] = on_result(result, job)
                output.append(result)
            return output
        worker.collect = collect
        return worker

    def begin(self, overlap):
        import torch
        runtime = self.collector_module.ShadowCollector(self.root, self.run_path, self.state)
        before = torch.get_rng_state().clone()
        outcome = runtime.begin(self.phase, self.state, self.model, self.service, self.run_path/'iteration-1257', overlap=overlap)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        return outcome

    def test_overlapped_tables_run_in_the_background_and_are_admitted_at_finish(self):
        with R.scope(dict(R.DEFAULTS, native_inference=R.cpu_block())):
            run = self.begin(True)
            self.assertIsNotNone(run)
            self.assertIsNotNone(run.core_plan)
            self.assertIs(self.created[0]['core_plan'], run.core_plan)
            self.assertEqual(self.admitted, [])
            self.gate.set()
            run.thread.join(10)
            self.assertEqual(self.admitted, [])
            run.finish()
        self.assertEqual(self.admitted, [len(self.jobs)])
        self.assertTrue(self.phase.metrics['overlapped_with_ordinary'])
        self.assertEqual(self.phase.metrics['new_tables'], len(self.jobs))

    def test_the_sequential_barrier_finishes_before_returning(self):
        self.gate.set()
        with R.scope(dict(R.DEFAULTS, native_inference=R.cpu_block())):
            self.assertIsNone(self.begin(False))
        self.assertIsNone(self.created[0]['core_plan'])
        self.assertEqual(self.admitted, [len(self.jobs)])
        self.assertFalse(self.phase.metrics['overlapped_with_ordinary'])

    def test_a_timed_budget_never_overlaps(self):
        self.gate.set()
        with R.scope(dict(R.DEFAULTS, native_inference=R.cpu_block(teacher_budget=dict(mode='timed')))):
            with mock.patch.object(T, 'calibrate_cpu', return_value=dict(estimate_ns=1)):
                self.assertIsNone(self.begin(True))
        self.assertIsNone(self.created[0]['core_plan'])


class Tables(unittest.TestCase):
    """The real TeacherCollector with its native worker replaced."""
    def collect(self, count, lanes, *, defer=False):
        producer = 'producer'
        search = dict(profile=1)
        jobs = [dict(expected=dict(job_id=str(i), model_id='m', export_hash='e', reward_id='r'), search=search, split='train', seed=i)
                for i in range(count)]
        if defer:
            for job in jobs:job['teacher_revision'] = {'test': True}
        calibration = dict(contract=CONTRACT, source=producer, core_source='core', model_id='m', export_hash='e', reward_id='r',
                           device='cpu-ort', profile=digest(search), samples=[1], estimate_ns=1)
        plan = CorePlan((10, 11, 12)[:lanes])
        released = []
        plan.attach(released.append, lambda error: None)
        gates = {str(i): threading.Event() for i in range(count)}
        class Worker:
            def __init__(self, executable, request, client, cancel, cpu=None):
                self.job = request['job'];self.failure = None
            def collect(self, expected, core, verifier=None, defer_logits=False):
                assert defer_logits is defer
                gates[self.job['expected']['job_id']].wait(10)
                return dict(job_id=self.job['expected']['job_id'], rows=[], metrics={}, result={}, value_rows=[], final_spent=None)
        class Bootstrap:
            def __init__(self, *args, **kwargs):pass
            def __enter__(self):return self
            def __exit__(self, *args):return False
            def call(self, op, seed):return {}
        class Admission:
            def acquire(self, workers, timings):
                from contextlib import nullcontext
                return nullcontext()
        service = SimpleNamespace(cpu=Admission(), device='cuda')
        collector = T.TeacherCollector('binary', 'core', ['bootstrap'], service, calibration, lanes=lanes, model=object(), history=[],
                                       inference=dict(budget=dict(mode='count', full=64, fast=24)), cores=(10, 11, 12)[:lanes], core_plan=plan)
        settled = threading.Event()
        def check(*args, **kwargs):settled.wait(10)
        def verifier(model, history, jobs, cancel, **kwargs):
            from contextlib import nullcontext
            from types import SimpleNamespace
            row = SimpleNamespace(label={'legal_ids': ['pass']})
            return SimpleNamespace(cache={job['expected']['job_id']: dict(deferred=[row]) for job in jobs},
                                   cancel=cancel, model=model, _policy_lock=nullcontext, _check_model=lambda: None)
        # The worker threads outlive this method, so the fakes must stay installed.
        for patcher in (mock.patch.object(T, 'Worker', Worker), mock.patch.object(T, 'Pipe', Bootstrap),
                        mock.patch.object(T, '_validate_job', lambda job, source: None), mock.patch.object(T, 'source_fingerprint', lambda: producer),
                        mock.patch.object(T, 'Client', lambda *args: SimpleNamespace(wait_seconds=0.)),
                        mock.patch('brass_ai.rust_backend.lab_teacher_finalization.check_logits', check),
                        mock.patch('brass_ai.rust_backend.lab_teacher_finalization.TurnVerifier',
                                   verifier)):
            patcher.start();self.addCleanup(patcher.stop)
        result = []
        thread = threading.Thread(target=lambda: result.extend(collector.collect(jobs)))
        thread.start()
        return thread, gates, released, result, settled

    def wait_for(self, released, count):
        for _ in range(500):
            if len(released) >= count:
                return
            threading.Event().wait(.01)
        self.fail(f'Only {len(released)} of {count} CPUs were released')

    def test_each_table_frees_its_cpu_as_soon_as_it_ends(self):
        thread, gates, released, _, _ = self.collect(2, 2)
        gates['1'].set()
        self.wait_for(released, 1)
        self.assertEqual(len(released), 1)
        gates['0'].set()
        thread.join(10)
        self.assertEqual(sorted(released), [10, 11])

    def test_a_cpu_stays_held_while_another_table_still_needs_it(self):
        thread, gates, released, result, _ = self.collect(2, 1)
        gates['0'].set()
        threading.Event().wait(.2)
        self.assertEqual(released, [])
        gates['1'].set()
        thread.join(10)
        self.assertEqual(released, [10])
        self.assertEqual(len(result), 2)

    def test_search_cpu_is_available_while_the_table_waits_for_logit_parity(self):
        thread, gates, released, result, settled = self.collect(1, 1, defer=True)
        try:
            gates['0'].set()
            self.wait_for(released, 1)
            self.assertEqual(released, [10])
            self.assertTrue(thread.is_alive())
            self.assertEqual(result, [])
        finally:
            settled.set()
            thread.join(10)
        self.assertEqual(len(result), 1)

    def test_queued_search_reuses_its_cpu_while_verification_waits(self):
        thread, gates, released, result, settled = self.collect(2, 1, defer=True)
        try:
            gates['0'].set()
            gates['1'].set()
            self.wait_for(released, 1)
            self.assertTrue(thread.is_alive())
            self.assertEqual(result, [])
        finally:
            settled.set()
            thread.join(10)
        self.assertEqual([item['job_id'] for item in result], ['0', '1'])


class LogitBatching(unittest.TestCase):
    def run_verification(self, error=None):
        from contextlib import nullcontext
        from brass_ai.rust_backend import lab_teacher_finalization as F
        rows = {str(i):[SimpleNamespace(label=dict(legal_ids=list(range(width)))) for width in (8,20,129)]
                for i in range(2)}
        records = {key:dict(deferred=value) for key,value in rows.items()}
        verifier = SimpleNamespace(cache=records,cancel=threading.Event(),model=object(),
                                   _policy_lock=nullcontext,_check_model=lambda:None)
        with mock.patch.object(F,'check_logits',side_effect=error) as verify:
            batcher = F.TableLogitBatcher(verifier,2,pending=4)
            batcher.FLUSH_SECONDS = .2
            with batcher:
                futures = [batcher.submit(key) for key in rows]
                for future in futures:
                    if error is None:
                        future.result(timeout=3)
                    else:
                        self.assertIs(future.exception(timeout=3),error)
            self.assertFalse(batcher.thread.is_alive())
        return rows,records,verify

    def test_both_tables_are_checked_and_settled_in_one_batch(self):
        rows,records,verify = self.run_verification()
        verify.assert_called_once()
        _, checked = verify.call_args.args
        self.assertCountEqual(map(id,checked),[id(row) for value in rows.values() for row in value])
        self.assertEqual([len(r.label['legal_ids']) for r in checked],[8,8,20,20,129,129])
        self.assertEqual([record['settled']['rows'] for record in records.values()],[3,3])

    def test_failed_batch_settles_no_table_and_wakes_both_waiters(self):
        _,records,_ = self.run_verification(ValueError('teacher logits differ'))
        self.assertTrue(all('settled' not in record for record in records.values()))


class CScale(unittest.TestCase):
    def test_the_block_defaults_to_point_one_and_records_only_supported_values(self):
        self.assertEqual(R.cpu_block()['teacher_budget']['c_scale'], .1)
        budget = dict(mode='count', full=64, fast=8)
        for value in (.1, .15, .5, 1.):
            R.validate_cpu(dict(R.cpu_block(), teacher_budget=dict(budget, full_k=16, fast_k=4, c_scale=value)))
        for bad in (0., 2., True):
            with self.assertRaises(ValueError):
                R.validate_cpu(dict(R.cpu_block(), teacher_budget=dict(budget, full_k=16, fast_k=4, c_scale=bad)))

    def test_split_scale_uses_the_committed_profile_and_legacy_uses_one_scale(self):
        from brass_ai.recovery import training_revision_g1491 as G
        config = dict(teacher_phase={G.KEY:dict(G.block(variant=G.SEARCH_REFRESH), count_search=G.RETUNED_SEARCH)})
        budget = dict(R.TEACHER_BUDGET, **G.RETUNED_SEARCH)
        block = R.cpu_block(teacher_budget=budget)
        R.validate_cpu(block)
        R.validate_teacher_budget(config, block)
        with R.scope(dict(R.DEFAULTS, native_inference=block)):
            self.assertEqual(R.teacher_c_scale(), .15)
            self.assertEqual(R.teacher_c_scale(fast=True), .5)
        common = {k:v for k,v in budget.items() if k!='fast_c_scale'}
        self.assertEqual(R.teacher_c_scale(common, fast=True), .15)
        with self.assertRaisesRegex(ValueError, 'committed teacher search profile'):
            R.validate_teacher_budget(config, R.cpu_block(teacher_budget=common))
        with self.assertRaises(ValueError):
            R.validate_cpu(R.cpu_block(teacher_budget=dict(budget, fast_c_scale=True)))
        with self.assertRaisesRegex(ValueError, 'long-only policy-label buffer'):
            R.validate_teacher_budget(dict(teacher_phase={G.KEY:G.block()}), block)

    def test_blocks_recorded_before_this_change_stay_valid_and_sequential(self):
        old = {k: v for k, v in R.cpu_block().items() if k not in ('overlap', 'readers')}
        old['teacher_budget'] = dict(mode='count', full=640, full_k=16, fast=96, fast_k=8)
        R.validate_cpu(old)
        with R.scope(dict(R.DEFAULTS, native_inference=old)):
            self.assertFalse(R.overlap_collection())
            self.assertEqual(R.teacher_c_scale(), .1)
            self.assertEqual(R.reader_workers(16), 16)

    def test_the_new_block_overlaps_and_uses_more_readers(self):
        with R.scope(dict(R.DEFAULTS, native_inference=R.cpu_block())):
            self.assertTrue(R.overlap_collection())
            self.assertEqual(R.teacher_c_scale(), .1)
            self.assertEqual(R.reader_workers(16), 24)

    def test_the_gpu_path_keeps_the_historical_constant_and_sequence(self):
        with R.scope(dict(R.DEFAULTS)):
            self.assertFalse(R.overlap_collection())
            self.assertEqual(R.teacher_c_scale(), .1)
            self.assertEqual(R.reader_workers(16), 16)

    def test_a_timed_budget_keeps_the_historical_constant(self):
        block = R.cpu_block(teacher_budget=dict(mode='timed'))
        R.validate_cpu(block)
        with R.scope(dict(R.DEFAULTS, native_inference=block)):
            self.assertEqual(R.teacher_c_scale(), .1)


class Labels(unittest.TestCase):
    def test_a_larger_c_scale_trusts_the_search_values_more(self):
        from brass_ai.recovery.teacher_labels import improved_policy
        logits, counts, means = [0., 0., 0.], [40, 40, 40], [.1, .2, .3]
        gentle, _ = improved_policy(logits, counts, means, .2, 50, .1)
        sharp, _ = improved_policy(logits, counts, means, .2, 50, 1.)
        self.assertGreater(sharp[2], gentle[2])
        expected = np.exp(np.array(logits) + 1.*(50+40)*np.array(means))
        expected /= expected.sum()
        np.testing.assert_allclose(sharp, expected, atol=1e-12)

    def test_only_the_reviewed_constants_are_accepted(self):
        from brass_ai.recovery.teacher_labels import improved_policy
        for c1, c2 in ((50, .25), (50, 2.), (49, 1.), (50, True), (True, 1.)):
            with self.assertRaises(ValueError):
                improved_policy([0., 0.], [1, 1], [.1, .2], .1, c1, c2)


class Window(unittest.TestCase):
    def test_mixed_scale_shards_survive_reopen_and_aging(self):
        import tempfile
        from pathlib import Path
        from brass_ai.recovery.teacher_buffer import TeacherBuffer
        from brass_ai.recovery.teacher_labels import IDENTITY, LabelRow

        with tempfile.TemporaryDirectory(dir=str(Path.cwd()/'artifacts')) as temp:
            root = Path(temp)
            run = root/'artifacts/step5/run'
            run.mkdir(parents=True)
            buffer = TeacherBuffer(root, run, 'mixed', c_scale=.2, cotton_c_scale=.1)
            rows = []
            for seat, scale in enumerate((.1, .2)):
                label = dict.fromkeys(IDENTITY)
                label.update(generation=9, model_id='m', reward_id='r', c2=scale,
                             seat=seat, turn_key=[], revision=0, kind='root')
                rows.append(LabelRow(dict(teacher=label), None))
            receipt = buffer.publish(9, 'train', rows, model_id='m', reward_id='r')
            self.assertEqual(receipt['c_scales'], [.1, .2])
            reopened = TeacherBuffer(root, run, 'mixed', c_scale=.2, cotton_c_scale=.1, receipts=[receipt])
            self.assertEqual(reopened.window(10), [receipt])
            self.assertEqual(reopened.window(13), [])
            self.assertEqual(reopened.retired, [receipt])

    def buffer(self, root, c_scale, receipts=()):
        from brass_ai.recovery.teacher_buffer import TeacherBuffer
        return TeacherBuffer(root, root/'artifacts/step5/run', 'identity', receipts=receipts, c_scale=c_scale)

    def receipt(self, buffer, generation, scale):
        import json
        from brass_ai.bank import sha
        from brass_ai.step5.league import write_json
        buffer.owner(create=True)
        name = f'generation-{generation:06d}-train.jsonl.gz'
        (buffer.directory/name).write_bytes(b'x'+str(generation).encode())
        receipt = dict(contract='teacher-label-buffer/1.0.0', path='teacher-labels/'+name, generation=generation, split='train',
                       rows=1, model_id='m', reward_id='r', profile_digest='identity', sha256=sha(buffer.directory/name))
        if scale != .1:
            receipt['c_scale'] = scale
        write_json(buffer.directory/(name+'.receipt.json'), receipt)
        return receipt

    def test_changing_c_scale_retires_older_labels_but_keeps_the_current_generation(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=str(Path.cwd()/'artifacts')) as temp:
            root = Path(temp)
            (root/'artifacts/step5/run').mkdir(parents=True)
            first = self.buffer(root, .1)
            old = [self.receipt(first, g, .1) for g in (8, 9, 10)]
            buffer = self.buffer(root, 1., old)
            kept = buffer.window(10)
            self.assertEqual([r['generation'] for r in kept], [10])
            self.assertEqual(sorted(r['generation'] for r in buffer.retired), [8, 9])

    def test_matching_c_scale_keeps_the_whole_window(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=str(Path.cwd()/'artifacts')) as temp:
            root = Path(temp)
            (root/'artifacts/step5/run').mkdir(parents=True)
            first = self.buffer(root, 1.)
            old = [self.receipt(first, g, 1.) for g in (8, 9, 10)]
            buffer = self.buffer(root, 1., old)
            self.assertEqual(len(buffer.window(10)), 3)

    def test_labels_from_before_the_change_read_as_the_historical_constant(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=str(Path.cwd()/'artifacts')) as temp:
            root = Path(temp)
            (root/'artifacts/step5/run').mkdir(parents=True)
            first = self.buffer(root, .1)
            old = [self.receipt(first, 9, .1)]
            self.assertNotIn('c_scale', old[0])
            self.assertEqual(len(self.buffer(root, .1, old).window(10)), 1)


if __name__ == '__main__':
    unittest.main()
