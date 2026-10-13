"""Active Step 5 forest failure retirement; no models or native hosts."""
from concurrent.futures import ThreadPoolExecutor
import random
from threading import Event, RLock, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fixtures import position
from brass_ai.recovery.parallel import ParallelMasked
from brass_ai.recovery.config import teacher_search_config


class ForestSubmissionTests(unittest.TestCase):
    def test_partial_submission_failure_drains_already_started_worker(self):
        entered = Event(); release = Event(); retired = Event(); returned = Event()
        errors = []

        class Replica:
            def __init__(self):
                self.base = SimpleNamespace(contexts={})

            def decide(self, *args):
                entered.set()
                try:
                    if not release.wait(5):
                        raise TimeoutError('Test failed to release the private worker')
                finally:
                    retired.set()

        with ThreadPoolExecutor(max_workers=1) as workers:
            class PartialPool:
                count = 0

                def submit(self, *args):
                    self.count += 1
                    if self.count == 2:
                        if not entered.wait(5):
                            raise TimeoutError('Private worker never started')
                        raise RuntimeError('Injected second forest submission failure')
                    return workers.submit(*args)

                def shutdown(self, **kwargs):
                    return workers.shutdown(**kwargs)

            planner = object.__new__(ParallelMasked)
            planner.config = teacher_search_config({})
            planner.decision_lock = RLock(); planner.replicas = [Replica(), Replica()]
            planner.pool = PartialPool()
            planner.base = SimpleNamespace(request_deadline=None,
                predict=lambda inputs: [(np.array([.6, .3, .1]), {})])
            planner.incumbent = lambda *args: {}
            planner.context_features = lambda *args: ([0.]*4, [0.]*8)

            def decide():
                try:
                    planner.decide(*position(), random.Random(1), float('inf'))
                except BaseException as error:
                    errors.append(error)
                finally:
                    returned.set()

            caller = Thread(target=decide)
            caller.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertFalse(returned.wait(.1), 'Actor returned while its private worker still owned a reply lane')
            finally:
                release.set(); caller.join(5)
            self.assertFalse(caller.is_alive())
            self.assertTrue(retired.is_set())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertIn('second forest submission', str(errors[0]))

    def test_thread_start_failure_retires_enqueued_worker_without_returned_future(self):
        entered = [Event(), Event()]; release = [Event(), Event()]
        retired = [Event(), Event()]; returned = Event(); errors = []

        class Replica:
            def __init__(self, index):
                self.index = index; self.base = SimpleNamespace(contexts={})

            def decide(self, *args):
                entered[self.index].set()
                try:
                    if not release[self.index].wait(5):
                        raise TimeoutError('Test failed to release the private worker')
                finally:
                    retired[self.index].set()

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='audit-private') as workers:
            planner = object.__new__(ParallelMasked)
            planner.config = teacher_search_config({})
            planner.decision_lock = RLock(); planner.replicas = [Replica(0), Replica(1)]
            planner.pool = workers
            planner.base = SimpleNamespace(request_deadline=None,
                predict=lambda inputs: [(np.array([.6, .3, .1]), {})])
            planner.incumbent = lambda *args: {}
            planner.context_features = lambda *args: ([0.]*4, [0.]*8)

            def decide():
                try:
                    planner.decide(*position(), random.Random(1), float('inf'))
                except BaseException as error:
                    errors.append(error)
                finally:
                    returned.set()

            original_start = Thread.start

            def start(thread, *args, **kwargs):
                if thread.name == 'audit-private_1':
                    # submit() has queued replica 1 but has not returned its
                    # future. Let the first thread pick up that orphaned work.
                    release[0].set()
                    if not entered[1].wait(5):
                        raise TimeoutError('Enqueued private worker never started')
                    raise RuntimeError('Injected actual thread startup failure')
                return original_start(thread, *args, **kwargs)

            with patch.object(Thread, 'start', start):
                caller = Thread(target=decide)
                caller.start()
                try:
                    self.assertTrue(entered[1].wait(5))
                    self.assertFalse(returned.wait(.1),
                        'Actor returned while an unreturned worker still owned a reply lane')
                finally:
                    for event in release: event.set()
                    caller.join(5)
            self.assertFalse(caller.is_alive())
            self.assertTrue(all(event.is_set() for event in retired))
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertIn('actual thread startup', str(errors[0]))


if __name__ == '__main__':
    unittest.main()
