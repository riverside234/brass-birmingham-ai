"""Real local remotes cover automatic publication and isolation from code Git."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import results_git as G


class ResultsGitTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts', prefix='results-push-test-')
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.remote = self.base / 'remote.git'
        self.root = self.base / 'code'
        self.root.mkdir()
        env = patch.dict(os.environ, GIT_AUTHOR_NAME='Report test', GIT_AUTHOR_EMAIL='reports@example.invalid',
                         GIT_COMMITTER_NAME='Report test', GIT_COMMITTER_EMAIL='reports@example.invalid')
        env.start()
        self.addCleanup(env.stop)
        self.cmd('init', '--bare', '--initial-branch=main', str(self.remote), cwd=self.base)
        self.cmd('init', '--initial-branch=main')
        (self.root / '.gitignore').write_text('/results/\n/artifacts/\n')
        (self.root / 'code.txt').write_text('code\n')
        self.cmd('add', '.gitignore', 'code.txt')
        self.cmd('commit', '-m', 'Code only')
        self.run = 'artifacts/step5/example'
        run = self.root / self.run
        (run / 'iteration-0000').mkdir(parents=True)
        self.state = dict(format='step5-recovery-league/2.2.0', iteration=1, games_done=320,
                          signature=dict(config={}, implementation='test-source'))
        (run / 'state.json').write_text(json.dumps(self.state))
        (run / 'retention-history.json').write_text('{}')
        (run / 'iteration-0000/report.json').write_bytes(b'{"iteration":0}\r\n')

    def cmd(self, *args, cwd=None):
        return subprocess.check_output(['git', '-c', 'core.longpaths=true', '-C', str(cwd or self.root), *args],
                                       stderr=subprocess.STDOUT).decode().strip()

    def cli(self, output, *extra):
        return subprocess.run([sys.executable, str(ROOT / 'scripts/export-training-reports.py'),
            '--root', str(self.root), '--run', self.run, '--output', 'results/' + output,
            '--continue', '--results-remote', str(self.remote), *extra], capture_output=True, text=True)

    def test_default_cli_publishes_and_leaves_main_index_and_artifacts_unchanged(self):
        (self.root / 'code.txt').write_text('staged code stays staged\n')
        self.cmd('add', 'code.txt')
        index = (self.root / '.git/index').read_bytes()
        original = {p: p.read_bytes() for p in (self.root / 'artifacts').rglob('*') if p.is_file()}
        main = self.cmd('rev-parse', 'HEAD')
        result = self.cli('first')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Uploaded results commit', result.stdout)
        self.assertEqual(main, self.cmd('rev-parse', 'HEAD'))
        self.assertEqual(index, (self.root / '.git/index').read_bytes())
        self.assertEqual(original, {p: p.read_bytes() for p in original})
        self.assertEqual(self.cmd('show', 'main:first/iteration-0000/report.json', cwd=self.remote), '{"iteration":0}')
        raw = subprocess.check_output(['git', '-C', str(self.remote), 'show', 'main:first/iteration-0000/report.json'])
        self.assertEqual(raw, b'{"iteration":0}\r\n')
        self.assertEqual(self.cmd('show', 'main:.gitattributes', cwd=self.remote), '* -text')
        self.assertFalse(self.cmd('ls-files', 'results'))

    def test_fresh_checkout_downloads_previous_bundles_before_incremental_export(self):
        first = self.cli('first')
        self.assertEqual(first.returncode, 0, first.stderr)
        second_root = self.base / 'second'
        second_root.mkdir()
        (second_root / self.run / 'iteration-0000').mkdir(parents=True)
        for path in (self.root / self.run).rglob('*'):
            if path.is_file():
                target = second_root / path.relative_to(self.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/export-training-reports.py'),
            '--root', str(second_root), '--run', self.run, '--output', 'results/second', '--continue',
            '--results-remote', str(self.remote)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((second_root / 'results/second/export-manifest.json').read_text())
        self.assertTrue(manifest['reused_files'])
        self.assertTrue(all(row['previous_export'] == 'results/first' for row in manifest['reused_files']))
        self.assertTrue((second_root / 'results/first/state.json').is_file())
        self.assertIn('second/export-manifest.json', self.cmd('ls-tree', '-r', '--name-only', 'main', cwd=self.remote))

    def test_local_export_does_not_create_git_or_contact_remote(self):
        result = self.cli('local-only', '--no-push')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / 'results/.git').exists())
        self.assertIsNone(G.head(self.remote))

    def test_parent_repository_and_wrong_origin_are_rejected(self):
        (self.root / 'results').mkdir()
        with self.assertRaisesRegex(ValueError, 'own .git'):
            G.verify(self.root, str(self.remote))
        directory = G.prepare(self.root, str(self.remote))
        with self.assertRaisesRegex(ValueError, 'origin'):
            G.verify(self.root, 'https://github.com/another/repository.git')
        self.cmd('config', 'remote.origin.pushurl', 'https://github.com/another/repository.git', cwd=directory)
        with self.assertRaisesRegex(ValueError, 'origin'):
            G.verify(self.root, str(self.remote))

    def test_main_tracked_results_are_rejected_before_nested_repo_creation(self):
        (self.root / 'results').mkdir()
        (self.root / 'results/old.txt').write_text('keep')
        self.cmd('add', '-f', 'results/old.txt')
        self.cmd('commit', '-m', 'Old tracked results')
        with self.assertRaisesRegex(ValueError, 'still tracks'):
            G.prepare(self.root, str(self.remote))
        self.assertFalse((self.root / 'results/.git').exists())
        self.assertEqual((self.root / 'results/old.txt').read_text(), 'keep')

    def test_failed_push_keeps_export_and_commit_then_next_export_retries(self):
        module_spec = importlib.util.spec_from_file_location('export_git_smoke', ROOT / 'scripts/export-training-reports.py')
        reports = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(reports)
        original = G.git
        def fail_push(directory, *args, **kwargs):
            if args[0] == 'push':
                raise ValueError('simulated rejected push')
            return original(directory, *args, **kwargs)
        with G.transaction(self.root, str(self.remote)) as directory:
            output, manifest = reports.export(self.root, self.run, 'results/retained')
            (output / 'unrelated-checkpoint.pt').write_bytes(b'never publish arbitrary files')
            with patch.object(G, 'git', side_effect=fail_push), self.assertRaisesRegex(ValueError, 'rejected push'):
                G.publish(self.root, directory, output, manifest, str(self.remote))
        commit = G.head(directory)
        self.assertIsNotNone(commit)
        self.assertTrue((output / 'export-manifest.json').is_file())
        result = self.cli('retry')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.cmd('merge-base', commit, 'main', cwd=self.remote), commit)
        self.assertNotIn('unrelated-checkpoint.pt', self.cmd('ls-tree', '-r', '--name-only', 'main', cwd=self.remote))

    def test_changed_export_is_rejected_before_commit(self):
        module_spec = importlib.util.spec_from_file_location('export_git_changed', ROOT / 'scripts/export-training-reports.py')
        reports = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(reports)
        with G.transaction(self.root, str(self.remote)) as directory:
            output, manifest = reports.export(self.root, self.run, 'results/changed')
            (output / 'state.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed before Git publication'):
                G.publish(self.root, directory, output, manifest, str(self.remote))
        self.assertIsNone(G.head(directory))
        self.assertFalse(self.cmd('diff', '--cached', '--name-only', cwd=directory))

    def test_staged_results_and_concurrent_export_are_preserved_and_rejected(self):
        result = self.cli('first')
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = self.root / 'results'
        (directory / 'manual.txt').write_text('manual staged file')
        self.cmd('add', 'manual.txt', cwd=directory)
        before = self.cmd('diff', '--cached', cwd=directory)
        with self.assertRaisesRegex(ValueError, 'staged changes'):
            with G.transaction(self.root, str(self.remote)):
                self.fail('must reject')
        self.assertEqual(before, self.cmd('diff', '--cached', cwd=directory))
        self.cmd('reset', '--', 'manual.txt', cwd=directory)
        with G.transaction(self.root, str(self.remote)):
            with self.assertRaisesRegex(ValueError, 'Another report export'):
                with G.transaction(self.root, str(self.remote)):
                    self.fail('must reject')
        self.assertFalse((directory / '.git/report-export.lock').exists())


if __name__ == '__main__':
    unittest.main()
