"""Restore explicit search_v1 over 1.4.41, preserving its UI and worker."""
from pathlib import Path
import runpy
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
B = runpy.run_path(str(ROOT / 'scripts/build-nn-search-choice.py'))
sha, read, write, pins = (B[k] for k in ('sha', 'read', 'write', 'pins'))
BASE = ROOT / 'artifacts/nn-temperature/candidate'
OUT = ROOT / 'artifacts/search-v1-default/candidate'
SCOPE = ['src/BrassLocalAI/Native/NativePackage.cs']
POLICIES = ['policy.json', 'policy-k8.json', 'policy-t15.json', 'policy-k8-t15.json']


def main():
    assert not OUT.exists(), 'Preserve candidate inputs/evidence'
    installed = read(ROOT / 'artifacts/patch-installation.json')
    baseline = read(BASE / 'build-manifest.json')
    assert installed['version'] == baseline['version'] == '1.4.41-rust-local'
    assert read(BASE / 'qualification.json')['passed']
    assert sha(BASE / 'qualification.json') == installed['nnTemperatureQualificationSha256'].lower()
    assert pins(ROOT / 'BrassNative') == baseline['nativeFiles']
    for pin in installed['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / pin['name']) == pin['sha256'].lower()
        assert sha(BASE / 'Managed' / pin['name']) == pin['sha256'].lower()
    runtime, managed = OUT / 'runtime', OUT / 'Managed'
    managed.mkdir(parents=True)
    for pin in baseline['sourcePins']:
        src, dst = BASE / 'runtime' / pin['path'], runtime / pin['path']
        assert sha(src) == pin['sha256']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for name in SCOPE:
        shutil.copyfile(ROOT / name, runtime / name)
    for p in (ROOT / 'BrassBirminghamUnity_Data/Managed').glob('*.dll'):
        if not any(s in p.name for s in ('Tests', 'Regression')):
            shutil.copyfile(p, managed / p.name)
    args = ['/nologo', '/target:library', '/langversion:7.3', '/nostdlib+', '/deterministic+', '/optimize+', f'/out:"{managed}/BrassLegacyAI.dll"']
    args += [f'/reference:"{p}"' for p in managed.glob('*.dll') if p.name != 'BrassLegacyAI.dll']
    for folder in ('BrassLocalAI', 'BrassMaskedSearch', 'BrassPersonalityGame'):
        args += [f'"{p}"' for p in sorted((runtime / 'src' / folder).rglob('*.cs'))]
    for key in ('balanced', 'rail-tempo', 'cotton', 'goods-beer', 'supplier', 'pottery-flex'):
        for suffix in ('v2', 'v3', 'v3-masked', 'v3-masked-strength', 'v3-masked-era', 'v3-masked-industry', 'v3-masked-economy'):
            args.append(f'/resource:"{runtime}/configs/legacy-ai/{key}-{suffix}.json",BrassLegacyAI.Profiles.{key}-{suffix}.json')
        args.append(f'/resource:"{runtime}/configs/legacy-ai/archive/1.4.0/{key}-v3.json",BrassLegacyAI.Profiles.{key}-v3-1.4.0.json')
    rsp = OUT / 'legacy.rsp'
    rsp.write_text('\n'.join(args), encoding='utf-8-sig')
    subprocess.run([str(ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'), '@' + str(rsp)], check=True)
    shutil.copytree(ROOT / 'BrassNative', OUT / 'BrassNative')
    package = OUT / 'BrassNative/alpha-g890'
    for name in POLICIES:
        policy = read(package / name)
        assert policy['method'] == 'search_v2_focus_only'
        policy['method'] = 'search_v1'
        write(package / name, policy)
    build = read(package / 'build.json')
    for pin in build['files']:
        pin['sha256'] = sha(package / pin['path'])
    write(package / 'build.json', build)
    write(OUT / 'build-manifest.json', dict(
        version='1.4.42-rust-local', baseVersion=installed['version'], stockSha256=installed['stockSha256'],
        baseFiles=installed['files'], baseNative=pins(ROOT / 'BrassNative'), sourceScope=SCOPE, sourcePins=pins(runtime),
        files=[dict(name=n, sha256=sha(managed / n)) for n in ('Assembly-CSharp.dll', 'BrassPatch.dll', 'BrassLegacyAI.dll')],
        nativeFiles=pins(OUT / 'BrassNative'), native_build_sha256=sha(package / 'build.json'),
        native_source_sha256=build['source_sha256'], search_method='search_v1', v205_enabled=False,
        search_k_choices=[8, 16], default_search_k=16, temperature_choices=[1.0, 1.5], default_temperature=1.0,
        save_policy='completed-turn-checkpoint/1.0.0', baseline_receipt_sha256=sha(ROOT / 'artifacts/patch-installation.json')))
    shutil.copyfile(ROOT / 'artifacts/patch-installation.json', OUT.parent / 'before-installation.json')
    print(OUT)


if __name__ == '__main__':
    main()
