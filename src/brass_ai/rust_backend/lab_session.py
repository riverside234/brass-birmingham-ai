"""One serial lane's bounded, reusable collector and original policy adapters."""
from collections import OrderedDict
from contextlib import contextmanager
import time
import threading
import numpy as np
from brass_ai.learning.contracts import digest, require
from brass_ai.recovery.execution import make_policy
from brass_ai.step5.policy import NeuralPolicy
from .lab_transport import Collector, DecisionEncoder


class OrdinarySession:
    def __init__(self, root, executable, endpoint, execution, mono=None, cpu=None):
        self.root, self.executable, self.endpoint = root, executable, endpoint
        self.execution, self.mono = execution, mono
        self.cpu = cpu
        self.lane = None
        self.policies = OrderedDict()
        self.active = False
        self.timings = None

    @contextmanager
    def game(self, job, timings, cancel=None, compare=None):
        require(not self.active, 'Ordinary session is already in use')
        self.active = True; self.timings = timings
        finished=threading.Event();watcher=None
        try:
            require(cancel is None or not cancel.is_set(), 'Collection canceled before game')
            began = time.monotonic()
            if self.lane is None:
                self.lane = Collector(self.executable)
                timings['collector_process_starts'] += 1
                if self.cpu is not None:
                    require(self.lane.hello.get('cpu_inference') == self.cpu['contract'], 'Collector lacks CPU inference')
                    self.lane.call('configure_cpu', contract=self.cpu['contract'], runtime=self.cpu['runtime'],
                                   model_cache=self.cpu['model_cache'])
            else:
                require(not self.lane.closed and self.lane.token is None, 'Unreleased collector game')
                timings['collector_process_reuses'] += 1
                self.lane.metrics.clear()
            timings['native_startup_seconds'] += time.monotonic()-began
            self.lane.cancel_event = cancel
            self.lane.batch_forced = compare is None
            policies = []
            for index, spec in enumerate(job['policies']):
                key = (index, digest(spec))
                p = self.policies.pop(key, None)
                if p is None:
                    began = time.monotonic()
                    p = make_policy(spec, self.root, self.endpoint, self.execution, self.mono)
                    timings['policy_starts'] += 1
                    timings['policy_startup_seconds'] += time.monotonic()-began
                    if type(p) is NeuralPolicy:
                        if self.cpu is not None:
                            from .lab_cpu_inference import NativeCpuEvaluator
                            require(p.evaluator.model_id in self.cpu['bundles'], 'Ordinary model has no CPU bundle')
                            p.evaluator = NativeCpuEvaluator(self.lane, p.evaluator.model_id,
                                                             self.cpu['bundles'][p.evaluator.model_id])
                        predict = p.evaluator.predict
                        def recorded(requests, p=p, predict=predict):
                            began = time.monotonic(); replies = predict(requests)
                            self.timings['inference_envelope_seconds'] += time.monotonic()-began
                            require(len(replies)==len(requests), 'Incomplete ordinary inference batch')
                            p.native_raw = np.asarray(replies[0]['probabilities'], dtype=np.float64)
                            return replies
                        p.evaluator.predict = recorded
                else:
                    timings['policy_reuses'] += 1
                self.policies[key] = p
                # Fresh RNGs bind each seat. Clear all turn/intent/queue/counter
                # state; do not close/recreate the C# worker or inference endpoint.
                for name in ('contexts','action_counts','last_behavior','last_raw','utility_remaining'):
                    if hasattr(p, name): getattr(p, name).clear()
                if type(p) is NeuralPolicy:
                    p.context_counter, p.context_offset, p.request_deadline = 0, None, None
                    p.opening = spec.get('opening')
                    if self.cpu is not None:
                        p.evaluator.begin_game(self.lane)
                    else:
                        p.evaluator.encoder = DecisionEncoder(self.lane, p.evaluator.model_id)
                    p.native_raw = None
                policies.append(p)
            while len(self.policies)>12:
                _, old = self.policies.popitem(last=False); old.close()
            if cancel is not None:
                def stop_workers():
                    while not finished.wait(.05):
                        if cancel.is_set():
                            # A C# call is synchronous. Kill only this private
                            # lane's workers so its original client wakes on EOF.
                            for p in policies:
                                workers=getattr(p,'workers',[getattr(p,'opponent',None)])
                                for w in workers:
                                    if w is not None:
                                        process=w.worker.process
                                        if process.poll() is None:
                                            try:process.kill()
                                            except OSError:pass
                            return
                watcher=threading.Thread(target=stop_workers,daemon=True);watcher.start()
            yield self.lane, policies
            require(cancel is None or not cancel.is_set(), 'Collection canceled after game')
            require(self.lane.token is None, 'Completed collector game was not released')
        except BaseException:
            self.close()  # Never reuse a canceled/failed process or plan queue.
            raise
        finally:
            finished.set()
            if watcher is not None:watcher.join(timeout=1)
            self.active = False; self.timings = None

    def close(self):
        if self.lane is not None: self.lane.close(); self.lane = None
        for p in self.policies.values(): p.close()
        self.policies.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
