"""A clean shell must resolve B01's actual fixtures, regardless of its cwd."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from brass_ai.learning.contracts import SOURCE_ROOT

spec = importlib.util.spec_from_file_location('b01_environment', SOURCE_ROOT/'scripts/qualify-rust-lab.py')
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)


class Environment(unittest.TestCase):
    def test_unrelated_environment_is_preserved_without_mutating_the_caller(self):
        original = {'PYTHONPATH': 'unrelated-checkout', 'CUDA_VISIBLE_DEVICES': '2', 'CUSTOM_FLAG': 'keep'}
        env = qualifier.python_environment(original)
        self.assertEqual(original['PYTHONPATH'], 'unrelated-checkout')
        self.assertNotIn('unrelated-checkout', env['PYTHONPATH'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
        self.assertEqual(env['CUSTOM_FLAG'], 'keep')

    def test_clean_subprocess_finds_all_fixture_dependencies_from_an_unrelated_cwd(self):
        env = qualifier.python_environment()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as folder:
            # The test must not inherit the developer shell's broad PYTHONPATH.
            query = ('import importlib.util,json; '
                     'print(json.dumps({n:importlib.util.find_spec(n).origin '
                     'for n in ("fixtures","review_fixture","fe_fixtures","test_b01")}))')
            run = subprocess.run([sys.executable, '-c', query], cwd=folder, env=env,
                                 capture_output=True, text=True, timeout=15, check=True)
        found = json.loads(run.stdout)
        for module, name in {'fixtures': 'tests/R2/fixtures.py',
                             'review_fixture': 'tests/Recovery/review_fixture.py',
                             'fe_fixtures': 'tests/FeatureEngineering/fe_fixtures.py',
                             'test_b01': 'tests/RustBackendLab/test_b01.py'}.items():
            self.assertEqual(Path(found[module]).resolve(), (SOURCE_ROOT/name).resolve())


if __name__ == '__main__':
    unittest.main()
