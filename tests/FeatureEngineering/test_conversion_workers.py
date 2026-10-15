"""Worker scheduling/commit checks; native serial/parallel parity is a CLI smoke."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from brass_ai import feature_conversion as conversion
from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, compatibility, read_json


class WorkerAllowanceTests(unittest.TestCase):
    def test_64_core_lab_defaults_to_48_and_caps_at_shards(self):
        with patch.object(conversion, 'available_cpus', return_value=64):
            self.assertEqual(conversion.worker_count(None, 1000), 48)
            self.assertEqual(conversion.worker_count(48, 1000), 48)
            self.assertEqual(conversion.worker_count(48, 3), 3)
            self.assertEqual(conversion.worker_count(1, 1000), 1)

    def test_explicit_workers_respect_available_allocation(self):
        with patch.object(conversion, 'available_cpus', return_value=6):
            self.assertEqual(conversion.worker_count(None, 100), 6)
            self.assertEqual(conversion.worker_count(48, 100), 6)

    def test_invalid_counts_and_empty_dataset_are_rejected(self):
        for count in (0, -1, 49, 64, True, 2.5, '48'):
            with self.subTest(count=count), self.assertRaises(ValueError):
                conversion.worker_count(count, 100)
        with self.assertRaises(ValueError):
            conversion.worker_count(2, 0)

    def test_scheduler_and_affinity_are_respected(self):
        with patch.object(conversion.os, 'cpu_count', return_value=64), \
             patch.object(conversion.os, 'process_cpu_count', return_value=64, create=True), \
             patch.object(conversion.os, 'sched_getaffinity', return_value=set(range(32)), create=True), \
             patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '24'}):
            self.assertEqual(conversion.available_cpus(), 24)
            with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': 'bad'}):
                self.assertEqual(conversion.available_cpus(), 32)
            with patch.object(conversion.os, 'sched_getaffinity', side_effect=OSError):
                self.assertEqual(conversion.available_cpus(), 24)


class ConversionCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def source(self, count=2):
        entries = []
        for index in range(count):
            path = self.root/f'input-{index}.gz'
            path.write_bytes(f'fixture-{index}'.encode())
            entries.append(dict(path=path.name, sha256=sha(path), game_id=f'g-{index}',
                setup_group=f'setup-{index}', split='train' if index % 2 == 0 else 'validation',
                semantic_sha256=str(index)*64, transitions_verified=True))
        manifest = dict(format='bc-dataset/1.0.0', compatibility=compatibility(),
            label_kind='qualified-v3-choice/1.0.0', context='neutral-predecision/1.0.0',
            teachers=['fixture'], calibration=None, entries=entries, excluded=[])
        path = self.root/'source.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return path, manifest

    def part(self, task):
        root, output, index, name, entries, header, mono = task
        self.assertNotIn('entries', header)
        self.assertNotIn('excluded', header)
        (Path(output)/'work').mkdir(parents=True, exist_ok=True)
        return dict(entries=[dict(e, path=f'output/raw/{index}.gz') for e in entries],
            inputs=[dict(path=name, sha256=sha(Path(root)/name), labels=[])],
            preserved='fixture', source=name, pid=100+index, seconds=0.01)

    def threads(self, **kwargs):
        self.assertEqual(kwargs['mp_context'].get_start_method(), 'spawn')
        self.assertIs(kwargs['initializer'], conversion._worker_init)
        return ThreadPoolExecutor(max_workers=kwargs['max_workers'])

    def test_out_of_order_completion_keeps_manifest_order_and_inputs_unchanged(self):
        source, original = self.source()
        before = {e['path']: sha(self.root/e['path']) for e in original['entries']}
        barrier = threading.Barrier(2)
        later = threading.Event()
        completed = []

        def run(task):
            part = self.part(task)
            barrier.wait(timeout=10)
            if task[2] == 0:
                self.assertTrue(later.wait(timeout=10))
            completed.append(task[2])
            if task[2] == 1:
                later.set()
            return part

        with patch.object(conversion, 'available_cpus', return_value=64), \
             patch.object(conversion, 'ProcessPoolExecutor', side_effect=self.threads), \
             patch.object(conversion, '_convert_shard', side_effect=run):
            receipt = conversion.convert(self.root, source, 'output', workers=48)
        result = read_json(self.root/'output/manifest.json')
        self.assertEqual(completed, [1, 0])
        self.assertEqual([e['game_id'] for e in result['entries']], ['g-0', 'g-1'])
        self.assertEqual(result['audit_workers'], 2)
        self.assertEqual(result['teachers'], original['teachers'])
        self.assertEqual(result['calibration'], original['calibration'])
        self.assertEqual(result['compatibility'], original['compatibility'])
        self.assertEqual(receipt['source_manifest_sha256'], sha(source))
        self.assertEqual(receipt['execution']['worker_pids'], [100, 101])
        self.assertEqual(receipt['execution']['requested_workers'], 48)
        self.assertEqual(receipt['partitions'], dict(train=1, validation=1))
        self.assertEqual(before, {name: sha(self.root/name) for name in before})
        self.assertFalse((self.root/'output/work').exists())

    def test_queue_is_bounded(self):
        source, _ = self.source(9)
        queued = []
        real_wait = conversion.wait

        def observe(pending, **kwargs):
            queued.append(len(pending))
            return real_wait(pending, **kwargs)

        with patch.object(conversion, 'available_cpus', return_value=64), \
             patch.object(conversion, 'ProcessPoolExecutor', side_effect=self.threads), \
             patch.object(conversion, '_convert_shard', side_effect=self.part), \
             patch.object(conversion, 'wait', side_effect=observe):
            conversion.convert(self.root, source, 'output', workers=2)
        self.assertEqual(max(queued), 4)

    def test_worker_error_never_commits_final_manifest(self):
        source, _ = self.source()

        def run(task):
            if task[2] == 1:
                raise ValueError('native transition mismatch')
            return self.part(task)

        with patch.object(conversion, 'available_cpus', return_value=64), \
             patch.object(conversion, 'ProcessPoolExecutor', side_effect=self.threads), \
             patch.object(conversion, '_convert_shard', side_effect=run), \
             self.assertRaisesRegex(ValueError, 'native transition mismatch'):
            conversion.convert(self.root, source, 'output', workers=2)
        self.assertFalse((self.root/'output/manifest.json').exists())
        self.assertFalse((self.root/'output/conversion.json').exists())
        self.assertFalse(read_json(self.root/'output/failure.json')['committed'])

    def test_duplicate_game_or_semantics_rejected_across_workers(self):
        source, _ = self.source()
        for field in ('game_id', 'semantic_sha256'):
            with self.subTest(field=field):
                def run(task):
                    part = self.part(task)
                    part['entries'][0][field] = 'duplicate'
                    return part
                output = 'output-'+field
                with patch.object(conversion, '_convert_shard', side_effect=run), \
                     self.assertRaisesRegex(ValueError, 'Duplicate'):
                    conversion.convert(self.root, source, output, workers=1)
                self.assertFalse((self.root/output/'manifest.json').exists())

    def test_changed_source_manifest_is_not_committed(self):
        source, original = self.source(1)

        def run(task):
            part = self.part(task)
            source.write_text(json.dumps(dict(original, changed=True)), encoding='utf-8')
            return part

        with patch.object(conversion, '_convert_shard', side_effect=run), \
             self.assertRaisesRegex(ValueError, 'manifest changed'):
            conversion.convert(self.root, source, 'output', workers=1)
        self.assertFalse((self.root/'output/manifest.json').exists())

    def test_existing_output_and_bad_workers_do_not_write(self):
        source, _ = self.source()
        output = self.root/'existing'
        output.mkdir()
        (output/'keep.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'new output'):
            conversion.convert(self.root, source, output, workers=2)
        self.assertEqual(list(output.iterdir()), [output/'keep.txt'])
        with self.assertRaisesRegex(ValueError, 'workers'):
            conversion.convert(self.root, source, 'invalid-workers', workers=49)
        self.assertFalse((self.root/'invalid-workers').exists())

    def test_worker_reuses_serial_audit_and_relocates_cache_without_changing_bytes(self):
        from brass_ai.recovery.replay import _cache_paths
        source, manifest = self.source(1)
        del source
        output = self.root/'output'
        original = manifest['entries'][0]

        def serial(root, job, temporary, *, mono):
            single = read_json(job)
            self.assertEqual(single['entries'], [original])
            self.assertEqual(single['teachers'], manifest['teachers'])
            self.assertEqual(mono, 'test-mono')
            raw = temporary/'raw/shard-000000.jsonl.gz'
            raw.parent.mkdir(parents=True)
            for index, path in enumerate((raw, *_cache_paths(raw))):
                path.write_bytes(f'audited-file-{index}'.encode())
            entry = dict(original, path=raw.relative_to(root).as_posix())
            (temporary/'manifest.json').write_text(json.dumps(dict(entries=[entry])))
            (temporary/'conversion.json').write_text('{}')
            return dict(inputs=[dict(path=original['path'], sha256=original['sha256'], labels=[])],
                        preserved='all raw facts and labels')

        with patch('brass_ai.feature_engineering.data.convert', side_effect=serial) as audit:
            part = conversion._convert_shard((str(self.root), str(output), 7, original['path'],
                [original], {k: v for k, v in manifest.items() if k != 'entries'}, 'test-mono'))
        self.assertEqual(audit.call_count, 1)
        target = output/'raw/shard-000007.jsonl.gz'
        self.assertEqual(part['entries'][0]['path'], target.relative_to(self.root).as_posix())
        for index, path in enumerate((target, *_cache_paths(target))):
            self.assertEqual(path.read_bytes(), f'audited-file-{index}'.encode())
        self.assertEqual(list((output/'work').iterdir()), [])

    def test_spawned_workers_reject_neural_raw_labels_without_publishing(self):
        source, manifest = self.source()
        for entry in manifest['entries']:
            path = self.root/entry['path']
            raw = dict(game_id=entry['game_id'], split=entry['split'], setup_group=entry['setup_group'],
                       records=[dict(behavior=dict(kind='model'), teacher=None)])
            header = dict(format='four-seat-replay/1.0.0', compatibility=compatibility(),
                          provenance=dict(purpose='training', seed=1))
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                stream.write(json.dumps(header)+'\n'+json.dumps(raw)+'\n')
            entry['sha256'] = sha(path)
        source.write_text(json.dumps(manifest), encoding='utf-8')
        before = {e['path']: e['sha256'] for e in manifest['entries']}
        with patch.object(conversion, 'available_cpus', return_value=64), \
             self.assertRaisesRegex(ValueError, 'Neural behavior/search teachers'):
            conversion.convert(self.root, source, 'output', workers=2)
        self.assertFalse((self.root/'output/manifest.json').exists())
        self.assertFalse((self.root/'output/conversion.json').exists())
        self.assertFalse(read_json(self.root/'output/failure.json')['committed'])
        self.assertEqual(before, {name: sha(self.root/name) for name in before})


if __name__ == '__main__':
    unittest.main()
