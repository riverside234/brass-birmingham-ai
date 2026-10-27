"""Build the narrow 1.4.37 hotfix over the sealed 1.4.36 local release."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/rust-backend/20261026-local-release/candidate'
OUT = ROOT / 'artifacts/city-cancel-fix/candidate'
SCOPE = ['src/BrassPatch/TurnHistory/ActionHistoryFixes.cs',
         'src/BrassPatch/TurnHistory/manifest.json',
         'src/BrassPatch/Liquidation/LiquidationFixes.cs']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    assert not (OUT / 'qualification.json').exists(), 'Do not overwrite a qualified candidate'
    baseline = read(BASE / 'build-manifest.json')
    qualified = read(BASE / 'qualification.json')
    assert qualified['passed'] and qualified['local_release_qualified']
    assert qualified['build_manifest_sha256'] == sha(BASE / 'build-manifest.json')
    runtime, managed = OUT / 'runtime', OUT / 'Managed'
    managed.mkdir(parents=True, exist_ok=True)
    for pin in baseline['sourcePins']:
        src, dst = BASE / 'runtime' / pin['path'], runtime / pin['path']
        assert src.resolve().is_relative_to(BASE) and dst.resolve().is_relative_to(runtime)
        assert sha(src) == pin['sha256'].lower(), pin['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for name in SCOPE:
        dst = runtime / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dst)
    for src in (BASE / 'Managed').glob('*.dll'):
        if not any(word in src.name for word in ('Tests', 'Regression')):
            shutil.copyfile(src, managed / src.name)
    for pin in baseline['files']:
        assert sha(managed / pin['name']) == pin['sha256'].lower()
    csc = ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    args = ['/nologo', '/target:library', '/langversion:7.3', '/nostdlib+', '/deterministic+', '/optimize+',
            f'/out:"{managed / "BrassPatch.dll"}"']
    args += [f'/reference:"{p}"' for p in managed.glob('*.dll') if p.name not in ('BrassPatch.dll', 'BrassLegacyAI.dll')]
    args += [f'"{p}"' for p in sorted((runtime / 'src/BrassPatch').rglob('*.cs'))]
    response = OUT / 'helper.rsp'
    response.write_text('\n'.join(args), encoding='utf-8-sig')
    subprocess.run([str(csc), '@' + str(response)], check=True, cwd=ROOT)
    original = OUT / 'baseline/Assembly-CSharp.dll'
    original.parent.mkdir(exist_ok=True)
    shutil.copyfile(BASE / 'Managed/Assembly-CSharp.dll', original)
    inventory = ROOT / 'artifacts/network-input-build/Managed/Assembly-CSharp.dll.manifest.json'
    assert sha(Path(str(inventory).removesuffix('.manifest.json'))) == sha(original)
    shutil.copyfile(inventory, str(original) + '.manifest.json')
    subprocess.run([str(ROOT / 'artifacts/patch-build/BrassPatch.Weaver.exe'), '--overlay', str(original),
                    str(managed / 'BrassPatch.dll'), str(runtime / 'src/BrassPatch/TurnHistory'),
                    str(managed / 'Assembly-CSharp.dll'), sha(original)], check=True, cwd=ROOT)
    # The native engine, NN model, ORT provider and AI executor remain byte-identical.
    shutil.copytree(BASE / 'BrassNative', OUT / 'BrassNative', dirs_exist_ok=True)
    manifest = dict(baseline)
    manifest.update(version='1.4.37-rust-local', baseVersion=baseline['version'], baseFiles=baseline['files'],
                    base_qualification_sha256=sha(BASE / 'qualification.json'), sourceScope=SCOPE,
                    save_policy='completed-turn-checkpoint/1.0.0', local_release_qualified=False,
                    sourcePins=[dict(path=p.relative_to(runtime).as_posix(), sha256=sha(p)) for p in sorted(runtime.rglob('*')) if p.is_file()],
                    files=[dict(name=n, sha256=sha(managed / n)) for n in ('Assembly-CSharp.dll', 'BrassPatch.dll', 'BrassLegacyAI.dll')])
    write(OUT / 'build-manifest.json', manifest)
    print('Built 1.4.37 city Cancel/Undo and completed-turn checkpoint candidate', flush=True)


if __name__ == '__main__':
    main()
