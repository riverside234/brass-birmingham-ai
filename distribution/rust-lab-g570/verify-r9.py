"""Check fresh r9 extraction, unchanged runtime and exact prerequisite reuse."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/rust-backend/g570-b03-topology-package'
sys.path.insert(0, str(ROOT/'src'))
from brass_ai.rust_backend.lab_contract import sha
from brass_ai.rust_backend.lab_transfer import verify, DIRECTORY
from brass_ai.rust_backend.lab_prerequisite_reuse import reviewed, verify_delta


def read(path): return json.loads(path.read_bytes())
def write(path, value): path.write_bytes((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def relevant(name):
    return Path(name).suffix in ('.py', '.json', '.cs', '.toml', '.dll', '.exe', '.so') and (
        name.startswith(('src/', 'configs/', 'training/runtime/')) or name in (
            'scripts/build-training-host.py', 'scripts/build-search-host.py', 'scripts/build-local-scoring-host.py'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = read(ROOT/DIRECTORY/'bundle.json'); manifest = read(ROOT/DIRECTORY/'transfer-manifest.json')
    assert receipt['path'].endswith('g570-r9.zip')
    stages = [verify(s) for s in ('b01', 'b02', 'b025', 'b03', 'b04')]
    with zipfile.ZipFile(ROOT/DIRECTORY/'rust-lab-b01-b04-g570-r8.zip') as previous:
        old = json.loads(previous.read(DIRECTORY+'/transfer-manifest.json'))
        inventory = json.loads(previous.read('artifacts/rust-backend/g570-b03-cuda/lab-evidence/evidence/b01/python-runtime-sources.json'))
        before = {i['path']: i['sha256'] for i in inventory if relevant(i['path'])}; after = dict(before)
        assert manifest['implementation'] == old['implementation'] and manifest['producer'] == old['producer']
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts', prefix='r9-extracted-') as directory:
            destination = Path(directory)
            with zipfile.ZipFile(ROOT/receipt['path']) as archive:
                assert archive.testzip() is None
                for name, expected in manifest['files'].items():
                    raw = archive.read(name)
                    assert raw == (ROOT/name).read_bytes(), name
                    assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == expected, name
                    if name.startswith(('src/', 'configs/', 'lab-native/', 'training/')) or name == DIRECTORY+'/prerequisite-reuse.json':
                        assert raw == previous.read(name), name
                    if relevant(name): after[name] = hashlib.sha256(raw).hexdigest()
                for name, expected in manifest['archives'].items():
                    assert hashlib.sha256(archive.read(name)).hexdigest() == expected
                assert not any(n.startswith('results/rust-lab-qualification/') and n != 'results/rust-lab-qualification/README.md'
                               for n in archive.namelist())
                archive.extractall(destination)
            verify_delta(before, after, reviewed())
            shutil.copyfile(ROOT/DIRECTORY/'bundle.json', destination/DIRECTORY/'bundle.json')
            shutil.copyfile(ROOT/receipt['path'], destination/receipt['path'])
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(destination/n) for n in (
                'src', 'scripts', 'tests/R2', 'tests/Recovery', 'tests/FeatureEngineering', 'tests/RustBackendLab')), CUDA_VISIBLE_DEVICES='')
            commands = [
                ('integrity', [sys.executable, '-X', 'utf8', '-c',
                    'from brass_ai.rust_backend.lab_transfer import verify; from brass_ai.rust_backend.lab_prerequisite_reuse import reviewed; print([verify(s)["milestone"] for s in ("b01","b02","b025","b03","b04")]); print(reviewed()["scope"])']),
                ('regression', [sys.executable, '-X', 'utf8', '-m', 'unittest', '-v',
                    'test_b03_topology', 'test_b03_cuda_restore', 'test_qualification_runner', 'test_lab_transfer'])]
            for name, command in commands:
                print('Fresh r9: '+name, flush=True)
                with (OUT/(name+'.log')).open('wb') as log:
                    result = subprocess.run(command, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                assert result.returncode == 0, name
    report = dict(passed=True, archive=receipt, fresh_extraction=True, transfer_stages=stages,
        exact_lab_inventory_overlay=True, runtime_identical_to_r8=True, original_results_untouched=True,
        prerequisites_repeated=False, scope='Package and CPU tests only; original B01/B02/B02.5 passes retained; B03/B04 lab Pending',
        logs={p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__), OUT/'integrity.log', OUT/'regression.log')})
    write(OUT/'package-check.json', report); write(ROOT/DIRECTORY/'package-check.json', report)
    print(json.dumps(dict(passed=True, runtime_unchanged=True, reuse_inventory=True, prerequisites_repeated=False)))


if __name__ == '__main__': main()
