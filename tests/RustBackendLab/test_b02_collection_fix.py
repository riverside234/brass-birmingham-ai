"""Short checks for runtime newline identity and bounded smoke concurrency."""
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, Mock

from brass_ai.legacy import LegacyWorker
from brass_ai.search import legacy_runtime as shared
from brass_ai.rust_backend.lab_b02_smoke import worker_settings, preflight_reference, reference_part
import test_shared_search_prebuilt


class RuntimeNewlines(test_shared_search_prebuilt.SharedPrebuiltTests):
    def worker(self, frozen=None):
        runtime=shared.directory(self.root,self.ref['id'])/'runtime'
        pins=self.root/'configs/rules/runtime-dependencies-1.1.0.json'
        pins.parent.mkdir(parents=True,exist_ok=True)
        shared.write_json(pins,dict(files=self.dependencies))
        return LegacyWorker(self.root,runtime_directory=runtime,frozen_manifest_sha256=frozen)

    def test_newline_change_after_install_preserves_runtime_and_launches(self):
        self.install();runtime=shared.runtime_directory(self.root,self.ref)
        before={p.name:shared.sha(p) for p in runtime.iterdir() if p.is_file()}
        self.source.write_bytes(b'// fixed strategy\r\n')
        with patch('brass_ai.legacy.subprocess.Popen',side_effect=RuntimeError('verified spawn')):
            with self.assertRaisesRegex(RuntimeError,'verified spawn'):self.worker()
        self.assertEqual(before,{p.name:shared.sha(p) for p in runtime.iterdir() if p.is_file()})

    def test_real_source_change_does_not_launch(self):
        self.install();self.source.write_bytes(b'// changed strategy\r\n')
        with patch('brass_ai.legacy.subprocess.Popen') as spawn:
            with self.assertRaises(ValueError):self.worker()
            spawn.assert_not_called()

    def test_frozen_reference_still_uses_its_exact_manifest_and_binary(self):
        self.install();runtime=shared.runtime_directory(self.root,self.ref)
        frozen=shared.sha(runtime/'runtime.json');self.source.write_bytes(b'// newer unrelated source\n')
        with patch('brass_ai.legacy.subprocess.Popen',side_effect=RuntimeError('verified frozen spawn')):
            with self.assertRaisesRegex(RuntimeError,'verified frozen spawn'):self.worker(frozen)
            with self.assertRaisesRegex(ValueError,'manifest changed'):self.worker('f'*64)

    def test_binary_and_manifest_tampering_are_still_rejected(self):
        self.install();runtime=shared.runtime_directory(self.root,self.ref)
        self.source.write_bytes(b'// fixed strategy\r\n')
        with patch('brass_ai.legacy.subprocess.Popen') as spawn:
            exe=runtime/'BrassLegacyAI.exe';original=exe.read_bytes();exe.write_bytes(b'bad')
            with self.assertRaises(ValueError):self.worker()
            exe.write_bytes(original)
            manifest=runtime/'runtime.json';manifest.write_bytes(manifest.read_bytes()+b' ')
            with self.assertRaises(ValueError):self.worker()
            spawn.assert_not_called()


class SmokeWorkers(TestCase):
    def jobs(self):
        from test_b02 import job
        jobs=[job(i) for i in range(3)]
        from brass_ai.recovery.exploration import THREE_BRANCH
        jobs[0]['branching']=THREE_BRANCH
        return jobs

    def test_cap_60_uses_all_runnable_jobs_without_extra_games(self):
        jobs=self.jobs()
        result=worker_settings(dict(cpu_budget=64,cpu_reserve=4),dict(legacy_budget={'global_workers':60}),jobs)
        self.assertEqual(result['effective_cap'],60);self.assertEqual(result['lanes'],4)
        self.assertEqual(len(jobs),3)

    def test_lower_capacity_and_explicit_limit_are_respected(self):
        for requested,capacity,expected in [(60,2,2),(1,60,1),(60,60,4)]:
            result=worker_settings(dict(cpu_budget=capacity+4,cpu_reserve=4),dict(legacy_budget={'global_workers':60}),self.jobs(),requested)
            self.assertEqual(result['lanes'],expected)
        for value in (0,61,True):
            with self.assertRaises(ValueError):worker_settings({},dict(legacy_budget={'global_workers':60}),self.jobs(),value)

    def test_reference_preflight_closes_each_unique_policy(self):
        policy=Mock();jobs=[dict(policies=[dict(kind='lab-legacy',policy_id='v4')]*4)]
        with patch('brass_ai.recovery.execution.make_policy',return_value=policy) as make:
            self.assertEqual(preflight_reference('.',jobs,{},'mono'),['v4'])
        make.assert_called_once();policy.close.assert_called_once()

    def test_partial_reference_policy_setup_closes_previous_policy_and_host(self):
        from contextlib import nullcontext
        policy=Mock();host=Mock();host.__enter__=Mock(return_value=host);host.__exit__=Mock(return_value=False)
        with patch('brass_ai.host.RulesHost',return_value=host), \
             patch('brass_ai.recovery.host_runtime.training_host',return_value=Path('.')), \
             patch('brass_ai.recovery.execution.make_policy',side_effect=[policy,ValueError('setup failure')]):
            with self.assertRaisesRegex(ValueError,'setup failure'):
                reference_part('.',dict(provenance={},policies=[{},{}]),None,{},'mono')
        policy.close.assert_called_once();host.__exit__.assert_called_once()
