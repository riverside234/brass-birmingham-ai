"""Spawn-safe admission for complete CPU turns, before their search clock starts.

Private forests acquire their whole worker group atomically. Tables may queue;
the table count is not a promise that all forests run at once. A failed actor
aborts collection rather than allowing its incomplete game into replay.
"""
from contextlib import contextmanager
import time
from brass_ai.learning.contracts import require


class CPUAdmission:
    def __init__(self, mp, budget, reserve):
        require(0 <= reserve < budget, 'Invalid CPU reserve')
        self.capacity = budget-reserve
        self.condition = mp.Condition()
        self.used = mp.Value('i', 0)
        self.peak = mp.Value('i', 0)
        self.next_ticket = mp.Value('q', 0)
        self.serving = mp.Value('q', 0)

    @contextmanager
    def acquire(self, workers, timings):
        require(1 <= workers <= self.capacity, 'CPU group exceeds admission capacity')
        began = time.monotonic()
        with self.condition:
            ticket = self.next_ticket.value
            self.next_ticket.value += 1
            while ticket != self.serving.value or self.used.value+workers > self.capacity:
                self.condition.wait(timeout=1)
            self.used.value += workers
            self.peak.value = max(self.peak.value, self.used.value)
            self.serving.value += 1
            self.condition.notify_all()
        timings.observe('cpu.admission.wait', time.monotonic()-began)
        try:
            yield
        finally:
            with self.condition:
                self.used.value -= workers
                require(self.used.value >= 0, 'CPU lease underflow')
                self.condition.notify_all()

    def report(self):
        with self.condition:
            return dict(capacity=self.capacity, active=self.used.value,
                        peak=self.peak.value, queued=self.next_ticket.value-self.serving.value)


class TurnLease:
    def __init__(self, admission, timings):
        self.admission, self.timings = admission, timings
        self.key = self.lease = None

    def update(self, key, workers):
        if key == self.key:
            return
        self.close()
        if self.admission is not None:
            self.lease = self.admission.acquire(workers, self.timings)
            self.lease.__enter__()
        self.key = key

    def close(self):
        if self.lease is not None:
            self.lease.__exit__(None, None, None)
        self.key = self.lease = None
