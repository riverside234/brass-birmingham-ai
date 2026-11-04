"""Fake private credentials cover real Git helper IPC without network access."""
import contextlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_results_git as fixtures
ROOT = fixtures.ROOT
import results_git as G
import results_git_auth as A


TOKEN = 'unit_test_only_not_a_real_token'


class AuthTests(unittest.TestCase):
    cmd = fixtures.ResultsGitTests.cmd
    cli = fixtures.ResultsGitTests.cli

    def setUp(self):
        fixtures.ResultsGitTests.setUp(self)
        path = self.root / '.gitignore'
        path.write_text(path.read_text() + '.env.*\n')
        self.cmd('add', '.gitignore')
        self.cmd('commit', '-m', 'Ignore private local credentials')

    def configure(self):
        return A.configure(self.root, 'riverside234', TOKEN)

    def test_private_file_is_ignored_and_not_stored_in_git_config(self):
        index = (self.root / '.git/index').read_bytes()
        path = self.configure()
        self.assertEqual(A.read(path), ('riverside234', TOKEN))
        self.assertEqual(self.cmd('check-ignore', A.NAME), A.NAME)
        self.assertFalse(self.cmd('status', '--short'))
        self.assertEqual((self.root / '.git/index').read_bytes(), index)
        self.assertNotIn(TOKEN, (self.root / '.git/config').read_text())
        if os.name != 'nt':
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_real_git_credential_protocol_uses_file_not_token_arguments(self):
        self.configure()
        original = subprocess.run
        calls = []
        def record(command, **kwargs):
            calls.append((command, kwargs.get('env', {})))
            return original(command, **kwargs)
        with patch.object(G.subprocess, 'run', side_effect=record):
            result = G.git(self.root, 'credential', 'fill',
                data=('url=' + G.REMOTE + '\n\n').encode(), auth_root=self.root, auth_remote=G.REMOTE)
        self.assertIn(b'username=riverside234\n', result.stdout)
        self.assertIn(('password=' + TOKEN).encode(), result.stdout)
        for command, env in calls:
            self.assertNotIn(TOKEN, str(command))
            self.assertNotIn(TOKEN, str(env))
            self.assertEqual(env['GIT_TERMINAL_PROMPT'], '0')
            self.assertEqual(env['GIT_TRACE_REDACT'], '1')

    def test_credentials_are_not_returned_for_another_repository_or_host(self):
        self.configure()
        for url in ('https://github.com/riverside234/brass-birmingham-ai.git',
                    'https://example.invalid/riverside234/ai-training-results.git'):
            with self.subTest(url=url), self.assertRaises(ValueError) as error:
                G.git(self.root, 'credential', 'fill', data=('url=' + url + '\n\n').encode(),
                      auth_root=self.root, auth_remote=G.REMOTE)
            self.assertNotIn(TOKEN, str(error.exception))
        for remote in (str(self.remote), 'git@github.com:riverside234/ai-training-results.git',
                       'https://github.com/another/repository.git'):
            env = {}
            self.assertEqual(A.git_options(self.root, remote, env), [])
            self.assertNotIn(A.ENV, env)

    def test_invalid_url_and_dotenv_never_disclose_the_token(self):
        path = self.configure()
        for value in (f'https://riverside234:{TOKEN}@example.invalid/repo.git',
                      f'https://riverside234:{TOKEN}@github.com/another/repository.git',
                      '$(echo ' + TOKEN + ')'):
            path.write_text(A.KEY + '=' + value + '\n')
            with self.assertRaises(ValueError) as error:
                A.read(path)
            self.assertNotIn(TOKEN, str(error.exception))
        for value in (A.KEY + '=invalid\n' + A.KEY + '=invalid\n', 'UNKNOWN=' + TOKEN):
            path.write_text(value)
            with self.assertRaises(ValueError) as error:
                A.read(path)
            self.assertNotIn(TOKEN, str(error.exception))

    def test_refuses_tracked_or_unignored_credential_files(self):
        path = self.configure()
        before = path.read_bytes()
        self.cmd('add', '-f', A.NAME)  # Fake test credentials, temporary repo only.
        self.cmd('commit', '-m', 'Auth safety fixture')
        with self.assertRaisesRegex(ValueError, 'already tracked'):
            A.configure(self.root, 'riverside234', 'different_fake_token')
        self.assertEqual(path.read_bytes(), before)
        self.cmd('rm', '--cached', A.NAME)
        self.cmd('commit', '-m', 'Untrack fixture')
        path.unlink()
        ignore = self.root / '.gitignore'
        ignore.write_text('/results/\n/artifacts/\n')
        with self.assertRaisesRegex(ValueError, 'gitignore'):
            self.configure()
        self.assertFalse(path.exists())

    def test_interactive_setup_hides_token_and_non_get_actions_return_nothing(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['results_git_auth.py', 'configure', '--root', str(self.root)]), \
                patch.object(sys.stdin, 'isatty', return_value=True), patch('builtins.input', return_value='riverside234'), \
                patch.object(A.getpass, 'getpass', return_value=TOKEN), contextlib.redirect_stdout(output):
            self.assertEqual(A.main(), 0)
        self.assertNotIn(TOKEN, output.getvalue())
        self.assertIn('Saved ignored', output.getvalue())
        for action in ('store', 'erase'):
            stdout = io.StringIO()
            A.helper(action, io.StringIO(''), stdout)
            self.assertFalse(stdout.getvalue())

    def test_push_only_retries_without_reading_training_data_or_new_export(self):
        self.configure()
        first = self.cli('first')
        self.assertEqual(first.returncode, 0, first.stderr)
        (self.root / self.run / 'state.json').write_text('invalid source must not be read')
        before = sorted(path.name for path in (self.root / 'results').iterdir())
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/export-training-reports.py'),
            '--root', str(self.root), '--push-only', '--results-remote', str(self.remote)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Uploaded results commit', result.stdout)
        self.assertNotIn(TOKEN, result.stdout + result.stderr)
        self.assertEqual(before, sorted(path.name for path in (self.root / 'results').iterdir()))
        self.assertNotIn(A.NAME, self.cmd('ls-tree', '-r', '--name-only', 'main', cwd=self.remote))
        config = (self.root / 'results/.git/config').read_text()
        self.assertNotIn(TOKEN, config)
        self.assertNotIn('credential.helper', config)


if __name__ == '__main__':
    unittest.main()
