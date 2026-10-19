"""Isolated candidate freezing must not overwrite a live/default runtime."""
import json
from pathlib import Path
import tempfile
import unittest

from brass_ai.bank import freeze,load_bank,sha

ARTIFACTS=Path(__file__).resolve().parents[2]/'artifacts'


class IsolatedBankTests(unittest.TestCase):
    def fixture(self,root):
        def write(name,data):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(data,encoding='utf-8');return path
        source=write('src/policy.cs','candidate source')
        executable=write('candidate/runtime/BrassLegacyAI.exe','candidate executable fixture')
        write('artifacts/legacy-ai/runtime/BrassLegacyAI.exe','keep the existing runtime')
        write('artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe','compiler fixture')
        write('configs/legacy-ai/balanced-test.json',json.dumps(dict(player_count=4)))
        manifest=dict(format='legacy-runtime/0.1.0',rules='steam-brass-patch-1.1.0',protocol='0.2.0',
                      sources=[dict(path='src/policy.cs',sha256=sha(source))],
                      files=[dict(name='BrassLegacyAI.exe',sha256=sha(executable))])
        write('candidate/runtime/runtime.json',json.dumps(manifest))
        return executable

    def test_freeze_uses_isolated_runtime(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temp:
            root=Path(temp);exe=self.fixture(root)
            bank=freeze(root,['balanced-test'],runtime_directory='candidate/runtime')
            load_bank(root,bank)
            self.assertEqual((root/'artifacts/legacy-ai/runtime/BrassLegacyAI.exe').read_text(),'keep the existing runtime')
            self.assertEqual((root/'artifacts/legacy-ai/bank'/bank/'runtime/BrassLegacyAI.exe').read_bytes(),exe.read_bytes())
            self.assertEqual(freeze(root,['balanced-test'],runtime_directory='candidate/runtime'),bank)

    def test_bad_runtime_is_rejected_before_freeze(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temp:
            root=Path(temp);exe=self.fixture(root);exe.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Runtime changed before freeze'):
                freeze(root,['balanced-test'],runtime_directory='candidate/runtime')
            self.assertFalse((root/'training/opponents/bank').exists())


if __name__=='__main__':unittest.main()
