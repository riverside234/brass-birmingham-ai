"""Publisher decode with shared subtrees, canonical receipts and pre-issued shared leases.

Every fast path must equal the plain formula exactly: rows equal json.loads of their bytes
(values, types and key order), digests equal contracts.digest, replay bytes equal
json.dumps(sort_keys=True).
"""
import copy
import gzip
import hashlib
import json
import pickle
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery.admitted_inputs import FrozenDict, admitted_digest
from brass_ai.recovery.canonical_inputs import dumps, dumps_family, input_digest, receipt
from brass_ai.rust_backend import lab_shared as S
from brass_ai.rust_backend.lab_delta import Decoder, SharedRows


def delta_pages(rows, evidence, *, token='authored-token'):
    """The Rust ordinary-records-delta/1 encoding of these rows (ordinary_delta.rs)."""
    pages = []; previous = b''; lengths = []; payload = bytearray(); start = 0
    mark = '\x00split-here\x00'
    for index, (row, proof) in enumerate(zip(rows, evidence, strict=True)):
        shell = dict(row, view=dict(row['view'], history=mark))
        text = json.dumps(shell, separators=(',', ':'), ensure_ascii=False).encode()
        token_bytes = json.dumps(mark, ensure_ascii=False).encode()
        prefix, found, suffix = text.partition(token_bytes)
        assert found and token_bytes not in suffix
        history = json.dumps(row['view']['history'], separators=(',', ':'), ensure_ascii=False).encode()
        base = 0
        while base < min(len(previous), len(history)) and previous[base] == history[base]:base += 1
        tail = history[base:]; proof_bytes = json.dumps(proof, separators=(',', ':')).encode()
        payload += prefix + tail + suffix + proof_bytes
        lengths.append([len(prefix), len(tail), len(suffix), len(proof_bytes), base])
        previous = history
        if len(lengths) == 64 or index == len(rows)-1:
            end = index+1
            pages.append((dict(wire='ordinary-records-delta/1', token=token, start=start, end=end, rows=len(rows),
                               lengths=lengths, bytes=len(payload)), bytes(payload)))
            start = end; lengths = []; payload = bytearray()
    return pages


def full_rows(rows):
    """Exact bytes of each row as the delta wire restores them."""
    out = []
    for row in rows:
        out.append(json.dumps(row, separators=(',', ':'), ensure_ascii=False).encode())
    return out


def synthetic_rows():
    """Growing and rewritten histories, non-ASCII text and JSON type flips between rows."""
    rows = []; history = []
    player = lambda cash, flag: dict(seat=0, cash=cash, mat=[dict(level=1, built=flag)]*3)
    for step in range(40):
        if step and step % 7 == 0 and history:
            history[-1] = dict(history[-1], note='café rewritten')  # an event changes in place
        if step % 3 != 2:history.append(dict(seat=step % 4, era='CanalEra', choice=dict(kind='build', n=step), city='café'))
        # 1 / 1.0 / true are == in Python but not the same JSON: never shared across rows.
        flip = [1, 1.0, True, 1, 1][step % 5]
        view = dict(protocol='p', seat=step % 4, context=dict(revision=step, phase='action'),
                    players=[player(10 + (step // 4), step % 2 == 0), player(5, False), player(7, flip is True), player(3, False)],
                    slots=[dict(id='a', owner=None)] * 5, links=[flip], merchants=[], pending=None,
                    history=copy.deepcopy(history), own_hand=['x'])
        choices = [dict(id='c%d' % (step % 3), kind='pass')] if step % 4 else [dict(id='only', kind='pass')]
        rows.append(dict(step=step, seat=step % 4, view=view, choices=choices, chosen=0,
                         behavior=dict(kind='model', probability=0.5 if step % 2 else 1.0), style=[0.]*4,
                         intent=[flip]*2, teacher=None, seconds=0.25))
    return rows


class SharedDecode(unittest.TestCase):
    def decode(self, rows):
        evidence = [dict(kind='ordinary', n=i) for i in range(len(rows))]
        decoder = Decoder(); shared = SharedRows(); out = []
        for meta, raw in delta_pages(rows, evidence):
            for prefix, history, suffix, base, _ in decoder.split_records(meta, raw):
                out.append(shared.row(prefix, history, suffix, base))
        return out, shared

    def test_rows_equal_plain_parse_with_types_and_order(self):
        rows = synthetic_rows(); decoded, shared = self.decode(rows)
        self.assertEqual(shared.fallbacks, 0); self.assertGreater(shared.reused, shared.parsed)
        for raw, row in zip(full_rows(rows), decoded, strict=True):
            plain = json.loads(raw)
            self.assertEqual(row, plain)
            # Key order and the exact JSON type of every value (1 vs 1.0 vs true).
            self.assertEqual(json.dumps(row, ensure_ascii=False).encode(), json.dumps(plain, ensure_ascii=False).encode())
            self.assertIs(type(row), dict); self.assertIs(type(row['behavior']), dict)
        # Unchanged events and players are the same immutable objects in consecutive rows.
        self.assertIs(decoded[2]['view']['history'][0], decoded[3]['view']['history'][0])
        self.assertIs(decoded[2]['view']['players'][1], decoded[3]['view']['players'][1])
        with self.assertRaises(TypeError):decoded[5]['view']['history'][0]['seat'] = 9
        with self.assertRaises(TypeError):decoded[5]['view']['players'].append(None)

    def test_receipts_equal_plain_digest_and_never_admit_inputs(self):
        rows = synthetic_rows(); decoded, _ = self.decode(rows)
        for plain, row in zip(rows, decoded, strict=True):
            expected = digest(dict(view=plain['view'], choices=plain['choices']))
            self.assertIsNotNone(receipt(row['view'], row['choices']))
            self.assertEqual(input_digest(row['view'], row['choices']), expected)
            self.assertEqual(input_digest(plain['view'], plain['choices']), expected)
            # The input boundary still validates these views: no actor Admission.
            self.assertIsNone(admitted_digest(row['view'], row['choices']))
        # Another row's choices or a copy carry no receipt; a pickled view is plain.
        self.assertIsNone(receipt(decoded[3]['view'], decoded[4]['choices']) if decoded[3]['choices'] is not decoded[4]['choices'] else None)
        restored = pickle.loads(pickle.dumps(decoded[9]['view']))
        self.assertIs(type(restored), dict); self.assertEqual(restored, rows[9]['view'])

    def test_edited_history_still_digests_by_content(self):
        rows = synthetic_rows(); decoded, _ = self.decode(rows)
        view = dict(decoded[-1]['view'], history=list(decoded[-1]['view']['history']))
        view['history'][4] = dict(view['history'][4], extra=1)
        self.assertEqual(input_digest(view, decoded[-1]['choices']), digest(dict(view=view, choices=decoded[-1]['choices'])))

    def test_family_bytes_equal_sorted_json(self):
        rows = synthetic_rows(); decoded, _ = self.decode(rows)
        child = dict(game=dict(records=decoded[:5], game_id='g/branch'), evidence=[{}]*5)
        family = dict(format='f', parent=dict(game=dict(records=decoded, game_id='g', result=[1, 2]), evidence=[1]*len(decoded)),
                      child=child, training_admitted=False)
        self.assertEqual(dumps_family(family), dumps(family))
        plain = json.loads(dumps(family))
        self.assertEqual(dumps_family(plain), dumps(plain))

    def test_unexpected_bytes_take_the_plain_parse(self):
        rows = synthetic_rows()[:6]; evidence = [{}]*6
        pages = delta_pages(rows, evidence); meta, raw = pages[0]
        decoder = Decoder(); shared = SharedRows(); out = []
        for i, (prefix, history, suffix, base, _) in enumerate(decoder.split_records(meta, raw)):
            if i == 3:suffix = bytes(suffix).replace(b'"protocol"', b'"protocol" ', 1) if b'"protocol"' in bytes(suffix) else suffix
            if i == 4:prefix = bytes(prefix).replace(b'{', b'{ ', 1)  # whitespace: valid JSON, not the producer's form
            out.append((shared.row(prefix, history, suffix, base), json.loads(b''.join((prefix, history, suffix)))))
        self.assertGreaterEqual(shared.fallbacks, 1)
        for row, plain in out:self.assertEqual(json.dumps(row), json.dumps(plain))


class PreissuedLeases(unittest.TestCase):
    def setUp(self):
        self.cancel = threading.Event(); self.registry = S.Registry(self.cancel); self.addCleanup(self.registry.close)

    def test_written_lease_shrinks_and_reads_back(self):
        part = S.Export(dict(branch_expected=False), [], [b'abc'*100, b'z'*50])
        with patch.object(S, 'CHUNK', 128), patch.object(S, 'OFFER', 1024):
            offered = self.registry.reserve()
            self.assertEqual(self.registry.used, 1024)
            handle = S.write(part, lease=offered)
            self.assertTrue(handle['transfer_timings']['shared_preissued'])
            self.registry.settle(offered, handle)
            self.assertEqual(self.registry.used, handle['size']); self.registry.verify(handle)
            restored = S.read(handle)
            self.assertEqual([bytes(x) for x in restored.features], [b'abc'*100, b'z'*50])
            kept = {c['name'] for c in handle['chunks']}
            for chunk in offered['chunks']:
                if chunk['name'] not in kept:
                    with self.assertRaises(FileNotFoundError):S.shared_memory.SharedMemory(name=chunk['name'])
            self.assertEqual(self.registry.offer_size(), -(-(handle['size']*5//4)//2**20)*2**20)
        self.registry.release(handle); self.assertEqual(self.registry.used, 0)
        self.assertEqual(self.registry.report()['preissued_leases']['used'], 1)

    def test_export_too_large_falls_back_and_releases_the_offer(self):
        part = S.Export(dict(branch_expected=False), [], [b'q'*5000])
        with patch.object(S, 'CHUNK', 512), patch.object(S, 'OFFER', 1024):
            offered = self.registry.reserve()
            self.assertIsNone(S.write(part, lease=offered))
            handle = S.write(part, self.registry.allocate)
            self.registry.settle(offered, handle); self.registry.observe(handle)
            self.assertEqual(self.registry.used, handle['size'])
            self.assertEqual(self.registry.report()['preissued_leases']['too_small'], 1)
            self.assertEqual(bytes(S.read(handle).features[0]), b'q'*5000)
        self.registry.release(handle)

    def test_tampered_descriptor_is_rejected(self):
        part = S.Export(dict(branch_expected=False), [], [b'abc'*100])
        with patch.object(S, 'CHUNK', 128):
            handle = S.write(part, self.registry.allocate)
            for change in ({'size': handle['size']+1}, {'chunks': handle['chunks'][:-1]},
                           {'chunks': [dict(c, size=c['size']-1) for c in handle['chunks']]}):
                with self.assertRaisesRegex(ValueError, 'identity|exceeds'):self.registry.verify(dict(handle, **change))
            with self.assertRaisesRegex(ValueError, 'descriptor'):S.read(dict(handle, sha256='0'*64))
            swapped = dict(handle, chunk_hashes=list(reversed(handle['chunk_hashes'])))
            with self.assertRaisesRegex(ValueError, 'descriptor|changed'):S.read(swapped)
        self.registry.release(handle)

    def test_unused_offer_for_a_file_part_is_released(self):
        offered = self.registry.reserve(); self.registry.settle(offered, dict(path='x', sha256='0'*64))
        self.assertEqual(self.registry.used, 0); self.assertEqual(self.registry.report()['active_leases'], 0)


class DeltaPublication(unittest.TestCase):
    """A fixture family through the delta wire publishes the plain wire's exact replay bytes."""
    def test_delta_and_plain_wire_publish_identical_replays(self):
        import sys
        sys.path.insert(0, str(SOURCE_ROOT/'scripts/legacy/rust-lab'))
        from test_b03 import family_fixture
        from test_collector_performance import exported
        from brass_ai.rust_backend import lab_pool as P
        f, _, c = family_fixture(decisions=32)
        replays = []
        for wire, sharing in (('plain', '1'), ('delta', '1'), ('delta', '0')):
            part = exported(copy.deepcopy(f['parent']))
            if wire == 'delta':
                decoder = Decoder(); rows = []; proofs = []
                for meta, raw in part.records:
                    for row, proof in decoder.records(meta, raw):rows.append(json.loads(bytes(row))); proofs.append(json.loads(bytes(proof)))
                pages = delta_pages(rows, proofs, token=part.records[0][0]['token'])
                part = S.Export(part.metadata, pages, part.features)
            registry = S.Registry(threading.Event()); self.addCleanup(registry.close)
            handle = S.write(part, registry.allocate)
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as out, patch.dict('os.environ', {S.SHARED_ROWS: sharing}):
                record = P.Publisher(Path(out), f['producer'], c, 'ordinary').publish(f['frozen_job'], handle, None)
                replays.append(gzip.decompress(Path(record['path']).read_bytes()))
                if wire == 'delta':
                    expected = dict(disabled=True) if sharing == '0' else None
                    sharing_report = record['publication_phases']['decode_sharing']
                    if expected:self.assertEqual(sharing_report, expected)
                    else:self.assertEqual(sharing_report['fallbacks'], 0)
            registry.release(handle)
        self.assertEqual(replays[0], replays[1]); self.assertEqual(replays[0], replays[2])
        with patch.dict('os.environ', {S.PREISSUED: 'x'}), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            S.switch(S.PREISSUED)


if __name__ == '__main__':
    unittest.main()
