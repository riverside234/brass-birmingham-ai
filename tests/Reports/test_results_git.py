"""Real local remotes cover automatic publication and isolation from code Git."""
import importlib.util
import contextlib
import io
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


class RemoteIdentityTests(unittest.TestCase):
    def test_github_transport_and_auth_variants_share_one_repository_identity(self):
        for remote in (
            G.REMOTE, 'https://github.com/riverside234/ai-training-results/',
            'https://riverside234@github.com/riverside234/ai-training-results.git',
            'https://user:unit_test_only@github.com/riverside234/ai-training-results.git',
            'https://user%40name:unit%2Ftest@github.com:443/RIVERSIDE234/AI-TRAINING-RESULTS.GIT/',
            'git@GitHub.com:Riverside234/ai-training-results.git',
            'ssh://git@github.com:22/riverside234/ai-training-results.git',
        ):
            self.assertEqual(G.identity(remote), G.identity(G.REMOTE))

    def test_other_repositories_hosts_and_url_modifiers_do_not_match(self):
        for remote in (
            'https://github.com/riverside234/brass-birmingham-ai.git',
            'https://github.com/another/ai-training-results.git',
            'https://github.com.invalid/riverside234/ai-training-results.git',
            'https://github.com@example.invalid/riverside234/ai-training-results.git',
            G.REMOTE + '?other', G.REMOTE + '#other',
            'https://github.com:444/riverside234/ai-training-results.git',
            'ssh://git@github.com:2222/riverside234/ai-training-results.git',
            'https://github.com:invalid/riverside234/ai-training-results.git',
        ):
            self.assertNotEqual(G.identity(remote), G.identity(G.REMOTE))

    def test_remote_diagnostics_hide_credentials_and_query_fragment_values(self):
        value = G.display('https://private_user:unit_test_only@github.com/another/repository.git?unit_query#unit_fragment')
        self.assertIn('github.com/another/repository.git', value)
        for secret in ('private_user', 'unit_test_only', 'unit_query', 'unit_fragment'):
            self.assertNotIn(secret, value)
        self.assertEqual(G.display('https://user:unit_test_only@github.com:invalid/path'), '<invalid remote URL>')


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

    def test_parent_repository_is_rejected_but_existing_origin_does_not_select_target(self):
        (self.root / 'results').mkdir()
        with self.assertRaisesRegex(ValueError, 'own .git'):
            G.verify(self.root, str(self.remote))
        directory = G.prepare(self.root, str(self.remote))
        self.cmd('remote', 'set-url', 'origin', 'https://github.com/another/repository.git', cwd=directory)
        self.cmd('config', 'remote.origin.pushurl', 'https://github.com/another/repository.git', cwd=directory)
        before = (directory / '.git/config').read_bytes()
        self.assertEqual(G.verify(self.root, str(self.remote)), directory)
        self.assertEqual(before, (directory / '.git/config').read_bytes())

    def test_authenticated_github_origin_and_equivalent_ssh_push_pass_without_config_changes(self):
        directory = G.prepare(self.root, str(self.remote))
        index = (self.root / '.git/index').read_bytes()
        self.cmd('remote', 'set-url', 'origin',
                 'https://user:unit_test_only@github.com/riverside234/ai-training-results.git', cwd=directory)
        self.cmd('config', 'remote.origin.pushurl',
                 'ssh://git@github.com:22/riverside234/ai-training-results.git', cwd=directory)
        config = (directory / '.git/config').read_bytes()
        self.assertEqual(G.verify(self.root), directory)
        self.assertEqual(config, (directory / '.git/config').read_bytes())
        self.assertEqual(index, (self.root / '.git/index').read_bytes())

    def test_git_authenticated_url_rewrite_is_recognized_without_network(self):
        directory = G.prepare(self.root, str(self.remote))
        self.cmd('remote', 'set-url', 'origin', G.REMOTE, cwd=directory)
        self.cmd('config', 'url.https://user:unit_test_only@github.com/.insteadOf',
                 'https://github.com/', cwd=directory)
        before = (directory / '.git/config').read_bytes()
        self.assertEqual(G.verify(self.root), directory)
        self.assertEqual(before, (directory / '.git/config').read_bytes())

    def test_push_instead_of_cannot_redirect_the_explicit_scripted_target(self):
        directory = G.prepare(self.root, str(self.remote))
        token = 'unit_test_only_never_print'
        self.cmd('config', f'url.https://user:{token}@github.com/another/repository.git.pushInsteadOf',
                 str(self.remote), cwd=directory)
        before = (directory / '.git/config').read_bytes()
        result = self.cli('explicit-push')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('explicit-push/export-manifest.json',
                      self.cmd('ls-tree', '-r', '--name-only', 'main', cwd=self.remote))
        self.assertNotIn(token, result.stdout + result.stderr)
        self.assertEqual(before, (directory / '.git/config').read_bytes())

    def test_fetch_rewrite_to_another_repository_is_rejected(self):
        directory = G.prepare(self.root, str(self.remote))
        self.cmd('config', 'url.https://github.com/another/repository.git.insteadOf',
                 G.REMOTE, cwd=directory)
        before = (directory / '.git/config').read_bytes()
        with self.assertRaisesRegex(ValueError, 'fetch/push rewrite does not identify'):
            G.verify(self.root)
        self.assertEqual(before, (directory / '.git/config').read_bytes())

    def test_cli_default_uses_scripted_target_not_existing_origin_or_pushurl(self):
        directory = G.prepare(self.root, str(self.remote))
        unrelated = self.base / 'unrelated.git'
        self.cmd('init', '--bare', '--initial-branch=main', str(unrelated), cwd=self.base)
        self.cmd('remote', 'set-url', 'origin', str(unrelated), cwd=directory)
        self.cmd('config', 'remote.origin.pushurl', str(unrelated), cwd=directory)
        before = (directory / '.git/config').read_bytes()
        spec = importlib.util.spec_from_file_location('export_selected_default', ROOT / 'scripts/export-training-reports.py')
        reports = importlib.util.module_from_spec(spec);spec.loader.exec_module(reports)
        argv = ['export-training-reports.py', '--root', str(self.root), '--run', self.run,
                '--output', 'results/scripted-target', '--continue']
        with patch.object(G, 'REMOTE', str(self.remote)), patch.object(sys, 'argv', argv), \
                contextlib.redirect_stdout(io.StringIO()):
            reports.main()
        self.assertIn('scripted-target/export-manifest.json',
                      self.cmd('ls-tree', '-r', '--name-only', 'main', cwd=self.remote))
        self.assertIsNone(G.head(unrelated))
        self.assertEqual(before, (directory / '.git/config').read_bytes())

    def test_publication_without_origin_uses_selected_target(self):
        directory = G.prepare(self.root, str(self.remote))
        self.cmd('remote', 'remove', 'origin', cwd=directory)
        result = self.cli('without-origin')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('without-origin/export-manifest.json',
                      self.cmd('ls-tree', '-r', '--name-only', 'main', cwd=self.remote))
        self.assertEqual(self.cmd('remote', cwd=directory), '')

    def remote_commit(self, directory, parent, message):
        tree = self.cmd('rev-parse', parent + '^{tree}', cwd=directory)
        return G.git(directory, 'commit-tree', tree, '-p', parent, data=message.encode()).stdout.decode().strip()

    def test_selected_target_fast_forward_works_with_unrelated_origin(self):
        first = self.cli('first')
        self.assertEqual(first.returncode, 0, first.stderr)
        directory = self.root / 'results'
        parent = G.head(self.remote)
        advanced = self.remote_commit(self.remote, parent, 'External results advance\n')
        self.cmd('update-ref', 'refs/heads/main', advanced, cwd=self.remote)
        self.cmd('remote', 'set-url', 'origin', 'https://example.invalid/unrelated.git', cwd=directory)
        config = (directory / '.git/config').read_bytes()
        result = self.cli('after-advance')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.cmd('merge-base', advanced, 'HEAD', cwd=directory), advanced)
        self.assertEqual(G.head(directory), G.head(self.remote))
        self.assertEqual(config, (directory / '.git/config').read_bytes())

    def test_selected_target_divergence_rejects_before_export_and_keeps_both_histories(self):
        first = self.cli('first')
        self.assertEqual(first.returncode, 0, first.stderr)
        directory = self.root / 'results';parent = G.head(self.remote)
        local = self.remote_commit(directory, parent, 'Local results advance\n')
        external = self.remote_commit(self.remote, parent, 'External results advance\n')
        self.cmd('update-ref', 'refs/heads/main', local, cwd=directory)
        self.cmd('update-ref', 'refs/heads/main', external, cwd=self.remote)
        before = (directory / '.git/index').read_bytes()
        result = self.cli('divergent')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('history diverged', result.stderr)
        self.assertFalse((directory / 'divergent').exists())
        self.assertEqual(G.head(directory), local);self.assertEqual(G.head(self.remote), external)
        self.assertEqual(before, (directory / '.git/index').read_bytes())

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

    def test_unreferenced_changed_old_state_does_not_block_or_get_uploaded(self):
        first = self.cli('first')
        self.assertEqual(first.returncode, 0, first.stderr)
        old_state = self.root / 'results/first/state.json'
        original = old_state.read_bytes()
        changed = original + b'\n'
        old_state.write_bytes(changed)
        result = self.cli('next')
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads((self.root / 'results/next/export-manifest.json').read_text())
        self.assertIn('state.json', [entry['path'] for entry in receipt['files']])
        self.assertTrue(receipt['reused_files'])
        self.assertNotIn('state.json', [entry['path'] for entry in receipt['reused_files']])
        self.assertEqual(old_state.read_bytes(), changed)
        remote_state = subprocess.check_output(['git', '-C', str(self.remote), 'show', 'main:first/state.json'])
        self.assertEqual(remote_state, original)

    def test_reused_file_changed_after_export_still_rejects_before_staging(self):
        first = self.cli('first')
        self.assertEqual(first.returncode, 0, first.stderr)
        module_spec = importlib.util.spec_from_file_location('export_reuse_changed', ROOT / 'scripts/export-training-reports.py')
        reports = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(reports)
        with G.transaction(self.root, str(self.remote)) as directory:
            before = G.head(directory)
            output, receipt = reports.export(self.root, self.run, 'results/reused-changed', continue_export=True)
            entry = receipt['reused_files'][0]
            dependency = self.root / entry['previous_export'] / entry['previous_path']
            dependency.write_bytes(dependency.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'changed before Git publication'):
                G.publish(self.root, directory, output, receipt, str(self.remote))
            self.assertEqual(G.head(directory), before)
            self.assertFalse(self.cmd('diff', '--cached', '--name-only', cwd=directory))

    def test_unpublished_cache_with_unrelated_bad_file_keeps_flat_reuse_after_clone(self):
        module_spec = importlib.util.spec_from_file_location('export_unpublished_cache', ROOT / 'scripts/export-training-reports.py')
        reports = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(reports)
        output, _ = reports.export(self.root, self.run, 'results/local-cache')
        state = output / 'state.json'
        state.write_bytes(state.read_bytes() + b'\n')
        result = self.cli('published')
        self.assertEqual(result.returncode, 0, result.stderr)
        names = self.cmd('ls-tree', '-r', '--name-only', 'main', cwd=self.remote)
        self.assertNotIn('local-cache/state.json', names)
        self.assertIn('local-cache/export-manifest.json', names)
        self.assertIn('local-cache/iteration-0000/report.json', names)
        # A fresh clone can reuse the real referenced file using the old index
        # without needing an unrelated, invalid file from that same cache.
        fresh = self.base / 'fresh'
        fresh.mkdir()
        G.prepare(fresh, str(self.remote))
        copied = fresh / self.run
        for file in (self.root / self.run).rglob('*'):
            if file.is_file():
                target = copied / file.relative_to(self.root / self.run)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(file.read_bytes())
        _, receipt = reports.export(fresh, self.run, 'results/fresh-next', continue_export=True)
        self.assertTrue(any(entry['previous_export'] == 'results/local-cache'
                            and entry['previous_path'] == 'iteration-0000/report.json'
                            for entry in receipt['reused_files']))

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
