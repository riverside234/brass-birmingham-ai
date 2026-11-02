"""Short interrupted-install/recovery/rollback check in a disposable local game."""
from pathlib import Path
import json
import runpy
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
H = runpy.run_path(str(ROOT / 'scripts/fix-sell-replay-speed.py'))
STUDY, OUT, NAMES = (H[k] for k in ('STUDY', 'OUT', 'NAMES'))
read, write, sha, pins = (H[k] for k in ('read', 'write', 'sha', 'pins'))


def main():
    directory = STUDY / 'install-tests'; assert not directory.exists(), 'Preserve installation tests'
    game = directory / 'game'; managed = game / 'BrassBirminghamUnity_Data/Managed'; managed.mkdir(parents=True)
    base = H['BASE']
    for n in NAMES: shutil.copyfile(base / 'Managed' / n, managed / n)
    shutil.copytree(ROOT / 'BrassNative', game / 'BrassNative')
    receipt = read(ROOT / 'artifacts/patch-installation.json'); assert receipt['version'] == '1.4.48-rust-local'
    receipt['gameRoot'] = str(game)
    data = game / 'user-data'; data.mkdir(); (data / 'save-sentinel.json').write_text('{"keep":"unchanged"}\n')
    receipt['saveDataPaths'] = [str(data)]
    record = game / 'artifacts/patch-installation.json'; record.parent.mkdir(parents=True); write(record, receipt)
    before = dict(files={n: sha(managed / n) for n in NAMES}, receipt=sha(record), native=pins(game / 'BrassNative'), saves=pins(data))
    phases = []
    for action, interrupt in (('Install', True), ('Recover', False), ('Install', False), ('Rollback', False)):
        args = [shutil.which('pwsh'), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts/install-sell-replay-speed.ps1'), '-GameRoot', str(game), '-Action', action]
        if interrupt: args.append('-TestInterrupt')
        result = subprocess.run(args, capture_output=True, text=True, timeout=55, creationflags=subprocess.CREATE_NO_WINDOW)
        label = action.lower() + ('-interrupted' if interrupt else '')
        (directory / (label + '.log')).write_text(result.stdout + result.stderr)
        assert bool(result.returncode) == interrupt, (action, result.stdout, result.stderr)
        if interrupt:
            assert 'TEST INTERRUPT: replay-speed files published' in result.stderr
            assert (game / 'artifacts/sell-replay-speed-install.pending').exists()
        if action in ('Recover', 'Rollback'):
            assert before['files'] == {n: sha(managed / n) for n in NAMES} and sha(record) == before['receipt']
        if action == 'Install' and not interrupt: assert read(record)['version'] == '1.4.49-rust-local'
        assert pins(data) == before['saves'] and pins(game / 'BrassNative') == before['native']
        phases.append(dict(phase=label, passed=True, exit_code=result.returncode))
    write(STUDY / 'upgrade-test.json', dict(passed=True, phases=phases,
        candidate_sha256=sha(OUT / 'build-manifest.json'), qualification_sha256=sha(OUT / 'qualification.json'),
        installer_sha256=sha(ROOT / 'scripts/install-sell-replay-speed.ps1'), exact_rollback=True, user_data_unchanged=True, native_unchanged=True))
    print(json.dumps(dict(passed=True, phases=phases)), flush=True)


if __name__ == '__main__': main()
