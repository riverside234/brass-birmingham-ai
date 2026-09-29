"""Verify Git inclusion and loadable authored assets, without staging or pushing."""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from brass_ai.bank import load_bank, read, sha, require
from check_qualification_reports import check_reports

qualification = check_reports(root)
distribution = read(root / 'training/runtime/distribution.json')
manifest = load_bank(root, distribution['bank'])
paths = [item['path'] for item in distribution['assets']]
paths += ['src/brass_ai/learning/model.py', 'src/brass_ai/learning/features.py',
          'src/brass_ai/learning/runner.py', 'src/brass_ai/legacy.py',
          'src/BrassLegacyAI/Harness.cs', 'configs/step5/lab.json',
          'training/opponents/current.json', 'training/opponents/pool.json',
          'training/runtime/distribution.json', 'training/runtime/parity.json',
          'requirements-step5.txt']
paths += ['training/opponents/bank/' + distribution['bank'] + '/manifest.json']
paths += ['training/opponents/bank/' + distribution['bank'] + '/source/' + p
          for p in manifest['payload']['source_files']]
paths += [p.relative_to(root).as_posix() for p in (root / 'training/models/releases').glob('*') if p.is_file()]
pool = read(root / 'training/opponents/pool.json')
paths += sorted({entry['qualification'] for purpose in ('training', 'evaluation')
                 for entry in pool[purpose].values()})
for item in distribution['assets']:
    require(sha(root / item['path']) == item['sha256'], 'Distributed asset hash mismatch')
for path in paths:
    require((root / path).is_file(), 'Missing Git asset: ' + path)
    check = subprocess.run(['git', 'check-ignore', '--quiet', '--no-index', '--', path], cwd=root)
    require(check.returncode == 1, 'Required authored asset is ignored or Git check failed: ' + path)
private = 'artifacts/step5-transfer/private-rules-runtime.zip'
check = subprocess.run(['git', 'check-ignore', '--quiet', '--no-index', '--', private], cwd=root)
require(check.returncode == 0, 'Private runtime bundle must remain ignored')
tracked = subprocess.run(['git', 'ls-files', '--', *paths], cwd=root, text=True, capture_output=True, check=True)
print(json.dumps(dict(passed=True, required_paths=len(paths),
                     already_tracked=len(tracked.stdout.splitlines()),
                     private_bundle_ignored=True, bank=distribution['bank'],
                     qualification=qualification), indent=2))
