"""Fixed-count tables are acknowledged before their turn is verified; failures still stop them."""
import queue
import threading
import unittest
from unittest import mock

from brass_ai.rust_backend import lab_teacher as T
from brass_ai.rust_backend.lab_teacher_overlay import CONTRACT


EXPECTED = {'job_id': 'frozen', 'model_id': 'm'}
CORE = 'c'*64


def worker(counted, events):
    w = T.Worker.__new__(T.Worker)
    w.client, w.cancel, w.job, w.counted = None, threading.Event(), {}, counted
    w.events, w.stop, w.stderr = queue.Queue(), threading.Event(), bytearray()
    w.process = mock.Mock(**{'poll.return_value': 0})
    w.reader, w.err_reader = mock.Mock(), mock.Mock()
    w.log = []
    w.send = lambda value: w.log.append(('ack', value['index'], value['status']))
    for meta in events:
        w.events.put((meta, b'', 0.))
    return w


def game(turns=2):
    events = [dict(event='hello', contract=CONTRACT, job=EXPECTED, core=dict(source_sha256=CORE),
                   method='search_v1', v205_enabled=False)]
    events += [dict(event='turn', job=EXPECTED, index=i, rows=[], value_row=None) for i in range(1, turns+1)]
    events.append(dict(event='finished', job=EXPECTED, ordinary_training=False, result={},
                       metrics=dict(labels=0, reserved_turns=16), value_rows=[]))
    return events


class Verifier:
    def __init__(self, log, fail_at=None):
        self.log, self.fail_at = log, fail_at

    def verify(self, meta, received):
        self.log.append(('verify', meta['index']))
        if meta['index'] == self.fail_at:
            raise ValueError('Teacher logits parity failed')
        return dict(job=meta['job'], index=meta['index'], status='ok')


class Acknowledgement(unittest.TestCase):
    def test_a_fixed_count_turn_is_acknowledged_first_then_verified_in_order(self):
        w = worker(True, game())
        result = w.collect(EXPECTED, CORE, verifier=Verifier(w.log))
        self.assertEqual(w.log, [('ack', 1, 'ok'), ('verify', 1), ('ack', 2, 'ok'), ('verify', 2)])
        self.assertEqual(result['job_id'], 'frozen')

    def test_a_timed_turn_still_waits_for_verification(self):
        w = worker(False, game())
        w.collect(EXPECTED, CORE, verifier=Verifier(w.log))
        self.assertEqual(w.log, [('verify', 1), ('ack', 1, 'ok'), ('verify', 2), ('ack', 2, 'ok')])

    def test_a_failed_check_stops_the_table_before_its_result(self):
        w = worker(True, game(3))
        with self.assertRaisesRegex(ValueError, 'parity'):
            w.collect(EXPECTED, CORE, verifier=Verifier(w.log, fail_at=2))
        self.assertEqual(w.log[-1], ('verify', 2))
        self.assertNotIn(('verify', 3), w.log)
        self.assertEqual(w.failure['type'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
