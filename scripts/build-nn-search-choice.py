"""Build the NN K selector over the exact installed cache-only release.

Reuse the installed Rust binary/model, and overlay only the scoped C# sources
on its frozen managed source tree. Never pick up unrelated working-tree edits.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/search-v2/20261028/v205-cache-candidate'
OUT = ROOT / 'artifacts/nn-search-choice/candidate'
SCOPE = [
    'src/BrassLocalAI/Policy/PlayerSelectionDraft.cs',
    'src/BrassLocalAI/Policy/LocalSetupPreferences.cs',
    'src/BrassLocalAI/Unity/LocalMenu.cs',
    'src/BrassLocalAI/Unity/AiSelectionPanel.cs',
    'src/BrassLocalAI/Unity/LocalTurnBridge.cs',
    'src/BrassLocalAI/Native/NativeGameState.cs',
    'src/BrassLocalAI/Native/NativePackage.cs',
    'src/BrassLocalAI/Native/NativeProcess.cs',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def pins(directory):
    return [dict(path=p.relative_to(directory).as_posix(), sha256=sha(p))
            for p in sorted(directory.rglob('*')) if p.is_file()]


def main():
    assert not (OUT / 'qualification.json').exists(), 'Preserve qualified candidates'
    baseline = read(BASE / 'build-manifest.json')
    installed = read(ROOT / 'artifacts/patch-installation.json')
    assert installed['version'] == '1.4.39-rust-local'
    assert installed['searchMethod'] == 'search_v2_focus_only' and not installed['v205Enabled']
    assert sha(BASE / 'build-manifest.json') == installed['searchV2CandidateSha256'].lower()
    for pin in installed['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / pin['name']) == pin['sha256'].lower()
        assert sha(BASE / 'Managed' / pin['name']) == pin['sha256'].lower()
    runtime, managed = OUT / 'runtime', OUT / 'Managed'
    managed.mkdir(parents=True, exist_ok=True)
    for pin in baseline['sourcePins']:
        src, dst = BASE / 'runtime' / pin['path'], runtime / pin['path']
        assert sha(src) == pin['sha256'].lower()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for name in SCOPE:
        shutil.copyfile(ROOT / name, runtime / name)
    for src in (ROOT / 'BrassBirminghamUnity_Data/Managed').glob('*.dll'):
        if not any(word in src.name for word in ('Tests', 'Regression')):
            shutil.copyfile(src, managed / src.name)
    args = ['/nologo', '/target:library', '/langversion:7.3', '/nostdlib+',
            '/deterministic+', '/optimize+', f'/out:"{managed}/BrassLegacyAI.dll"']
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
    package = OUT / 'BrassNative/alpha-g890'
    shutil.copytree(ROOT / 'BrassNative', OUT / 'BrassNative', dirs_exist_ok=True)
    policy = read(package / 'policy.json')
    assert policy['method'] == 'search_v2_focus_only'
    assert policy['k_by_allowance_ms'] == read(ROOT / 'configs/search/local-gumbel-k.json')
    assert set(policy['k_by_allowance_ms'].values()) == {16}
    policy['k_by_allowance_ms'] = read(ROOT / 'configs/search/local-gumbel-k8.json')
    assert set(policy['k_by_allowance_ms'].values()) == {8}
    write(package / 'policy-k8.json', policy)
    build = read(package / 'build.json')
    for pin in build['files']:
        assert sha(package / pin['path']) == pin['sha256'].lower()
    build['files'].append(dict(path='policy-k8.json', sha256=sha(package / 'policy-k8.json')))
    write(package / 'build.json', build)
    write(OUT / 'build-manifest.json', dict(
        version='1.4.40-rust-local', baseVersion=installed['version'], stockSha256=installed['stockSha256'],
        baseFiles=installed['files'], baseNative=pins(ROOT / 'BrassNative'),
        sourceScope=SCOPE, sourcePins=pins(runtime),
        files=[dict(name=n, sha256=sha(managed / n)) for n in ('Assembly-CSharp.dll', 'BrassPatch.dll', 'BrassLegacyAI.dll')],
        nativeFiles=pins(OUT / 'BrassNative'), native_build_sha256=sha(package / 'build.json'),
        native_source_sha256=build['source_sha256'], search_method='search_v2_focus_only', v205_enabled=False,
        search_k_choices=[8, 16], default_search_k=16, save_policy='completed-turn-checkpoint/1.0.0',
        baseline_receipt_sha256=sha(ROOT / 'artifacts/patch-installation.json')))
    shutil.copyfile(ROOT / 'artifacts/patch-installation.json', OUT.parent / 'before-installation.json')
    print(OUT)


if __name__ == '__main__':
    main()
