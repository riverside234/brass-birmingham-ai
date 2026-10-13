"""Small authored P5 audit regressions; these fixtures never qualify a controller."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from test_difficulty import named, panel, record
from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, read_json
from brass_ai.personality.difficulty import collection, contracts as c, evaluation, panels
from brass_ai.step5.league import seeds_for, write_json


AUDIT_TMP = SOURCE_ROOT/'artifacts/p56-audit-20261012/tmp'


class P5AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AUDIT_TMP.mkdir(parents=True, exist_ok=True)

    def test_crashed_panel_job_cannot_be_recollected_in_another_chunk(self):
        value = named(); frozen = dict(panel(value), pool_id='pool')
        ready = dict(model=dict(model_id='fixture'))
        handoff = dict(signature=dict(config=dict(execution={})))
        with tempfile.TemporaryDirectory(dir=AUDIT_TMP) as tmp:
            base = Path(tmp); directory = base/'panel'; directory.mkdir()
            write_json(directory/'manifest.json', frozen)
            with patch.object(panels, 'load', return_value=(frozen, value)), \
                 patch.object(collection, 'load_ready', return_value=(None, [], ready, handoff, 'pool')), \
                 patch.object(collection, 'SharedInference', side_effect=RuntimeError('injected after reservation')):
                with self.assertRaisesRegex(RuntimeError, 'injected after reservation'):
                    collection.execute(SOURCE_ROOT, directory, base/'first', games=1, tables=1)
                with self.assertRaisesRegex(ValueError, 'reserved'):
                    collection.execute(SOURCE_ROOT, directory, base/'retry', games=1, tables=1)
                self.assertFalse((base/'retry').exists())

    def test_loaded_report_rechecks_failure_stock_and_timing_summaries(self):
        value = named(); frozen = panel(value); game = record(frozen['jobs'][0])
        game['failure'] = dict(kind='stock', type='Fixture', message='failed')
        game['result'] = None; game['baseline_failures'] = [game['failure']]
        with tempfile.TemporaryDirectory(dir=AUDIT_TMP) as tmp:
            base = Path(tmp); write_json(base/'manifest.json', frozen)
            def chunks(*args):
                return frozen, value, iter([game]), {0}, []
            with patch.object(collection, 'read_chunks', side_effect=chunks), \
                 patch.object(evaluation, 'public_packet', return_value={}):
                original = evaluation.check(SOURCE_ROOT, base, [], base/'checked')
                self.assertEqual(evaluation.load_report(SOURCE_ROOT, base/'checked'), original)
                for key, replacement in [('failures', []), ('stock_failures', []),
                                         ('timing', dict(original['timing'], game_seconds=0.)),
                                         ('environment', dict(device='different-fixture-cpu'))]:
                    changed = copy.deepcopy(original); changed[key] = replacement
                    write_json(base/'checked/report.json', c.sealed({k:v for k,v in changed.items() if k!='id'}))
                    with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'pinned evidence'):
                        evaluation.load_report(SOURCE_ROOT, base/'checked')

    def test_reservation_overlap_is_atomic_and_disjoint_chunks_remain_available(self):
        value = named(); frozen = panel(value)
        with tempfile.TemporaryDirectory(dir=AUDIT_TMP) as tmp:
            base = Path(tmp); directory = base/'panel'; directory.mkdir()
            write_json(directory/'manifest.json', frozen)
            collection.reserve(SOURCE_ROOT, directory, base/'first', frozen, frozen['jobs'][:2])
            before = (directory/'reservations.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'reserved'):
                collection.reserve(SOURCE_ROOT, directory, base/'overlap', frozen, frozen['jobs'][1:3])
            self.assertEqual((directory/'reservations.json').read_bytes(), before)
            self.assertFalse((base/'overlap').exists())
            collection.reserve(SOURCE_ROOT, directory, base/'second', frozen, frozen['jobs'][2:3])
            self.assertEqual(set(read_json(directory/'reservations.json')['jobs']), {'0','1','2'})

    def test_concurrent_reservation_has_one_owner(self):
        value = named(); frozen = panel(value)
        with tempfile.TemporaryDirectory(dir=AUDIT_TMP) as tmp:
            base = Path(tmp); directory = base/'panel'; directory.mkdir()
            write_json(directory/'manifest.json', frozen)
            def claim(name):
                try:
                    collection.reserve(SOURCE_ROOT, directory, base/name, frozen, frozen['jobs'][:1])
                    return True
                except ValueError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as workers:
                results = list(workers.map(claim, ['first', 'second']))
            self.assertEqual(results.count(True), 1)
            self.assertEqual(len(read_json(directory/'reservations.json')['jobs']), 1)

    def test_completed_chunk_must_own_panel_reservation(self):
        value = named(); frozen = panel(value)
        seeds, _ = seeds_for(151000000, 2, 'validation')
        frozen['jobs'] = panels.jobs_for(value, seeds, 2, frozen['profiles'], repetitions=1)
        with tempfile.TemporaryDirectory(dir=AUDIT_TMP) as tmp:
            base = Path(tmp); directory = base/'panel'; directory.mkdir()
            write_json(directory/'manifest.json', frozen)
            chunk = base/'chunk'
            collection.reserve(SOURCE_ROOT, directory, chunk, frozen, frozen['jobs'][:1])
            game = c.sealed(dict(record(frozen['jobs'][0]), format=c.RECORD,
                                 contract=c.identity(), panel_id=frozen['id']))
            write_json(chunk/'game.json', game)
            receipt = c.sealed(dict(format=c.CHUNK, contract=c.identity(), panel_id=frozen['id'],
                fixture=True, fixture_direct=False, tables=1, environment=game['environment'],
                panel_sha256=sha(directory/'manifest.json'),
                records=[dict(job=0, path='game.json', sha256=sha(chunk/'game.json'), id=game['id'])]))
            write_json(chunk/'manifest.json', receipt)
            shutil.copytree(chunk, base/'substitute')
            with patch.object(panels, 'load', return_value=(frozen, value)):
                self.assertEqual(list(collection.read_chunks(SOURCE_ROOT, directory, [chunk])[2]), [game])
                with self.assertRaisesRegex(ValueError, 'own these reserved jobs'):
                    list(collection.read_chunks(SOURCE_ROOT, directory, [base/'substitute'])[2])

    def test_channel_preparation_failure_closes_inference_service(self):
        value = named(); frozen = dict(panel(value), pool_id='pool')
        ready = dict(model=dict(model_id='fixture'))
        handoff = dict(signature=dict(config=dict(execution={})))
        class Service:
            closed = False
            def __init__(self, *args, **kwargs):
                pass
            def register(self, model):
                return 'fixture'
            def channels(self, count):
                raise RuntimeError('injected channel allocation failure')
            def __exit__(self, *args):
                self.closed = True
        service = Service()
        with tempfile.TemporaryDirectory(dir=AUDIT_TMP) as tmp:
            base = Path(tmp); directory = base/'panel'; directory.mkdir()
            write_json(directory/'manifest.json', frozen)
            with patch.object(panels, 'load', return_value=(frozen, value)), \
                 patch.object(collection, 'load_ready', return_value=(None, [], ready, handoff, 'pool')), \
                 patch.object(collection, 'SharedInference', return_value=service):
                with self.assertRaisesRegex(RuntimeError, 'channel allocation failure'):
                    collection.execute(SOURCE_ROOT, directory, base/'chunk', games=1, tables=1)
            self.assertTrue(service.closed)


if __name__ == '__main__':
    unittest.main()
