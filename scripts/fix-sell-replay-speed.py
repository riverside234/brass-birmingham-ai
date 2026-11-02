"""Small replay-input overlay on the frozen, installed Beta g1290 release."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/sell-replay-speed/20261101'
BASE = ROOT / 'artifacts/nn-beta/20261101/release-1.4.48'
OUT = STUDY / 'release-1.4.49'
PLAYER = STUDY / 'player'
CSC = ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
SCOPE = ['src/BrassPatch/Presentation/ReplaySpeed.cs', 'src/BrassPatch/Presentation/replay-speed.patches.json']
NAMES = ('Assembly-CSharp.dll', 'BrassPatch.dll', 'BrassLegacyAI.dll')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
def pins(path): return [dict(path=p.relative_to(path).as_posix(), sha256=sha(p)) for p in sorted(path.rglob('*')) if p.is_file()]


def compile_cs(output, sources, references, extra=()):
    rsp = output.with_suffix(output.suffix + '.rsp')
    args = ['/nologo', '/target:' + ('exe' if output.suffix == '.exe' else 'library'), '/langversion:7.3', '/deterministic+', '/optimize+', *extra, f'/out:"{output}"']
    args += [f'/reference:"{p}"' for p in references] + [f'"{p}"' for p in sources]
    rsp.write_text('\n'.join(args), encoding='utf-8-sig')
    result = subprocess.run([str(CSC), '@' + str(rsp)], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    output.with_suffix(output.suffix + '.build.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode: raise RuntimeError(result.stdout + result.stderr)


def build():
    assert not OUT.exists(), 'Preserve candidates'
    old = read(BASE / 'build-manifest.json'); installed = read(ROOT / 'artifacts/patch-installation.json')
    assert installed['version'] == old['version'] == '1.4.48-rust-local'
    assert read(BASE / 'qualification.json')['passed']
    for p in old['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / p['name']) == p['sha256'] == sha(BASE / 'Managed' / p['name'])
    runtime = OUT / 'runtime'; managed = OUT / 'Managed'
    shutil.copytree(BASE / 'runtime', runtime); shutil.copytree(BASE / 'Managed', managed)
    for p in old['sourcePins']: assert sha(runtime / p['path']) == p['sha256']
    for name in SCOPE:
        dst = runtime / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dst)
    references = [p for p in managed.glob('*.dll') if p.name not in ('BrassPatch.dll', 'BrassLegacyAI.dll') and 'Tests' not in p.name and 'Regression' not in p.name]
    compile_cs(managed / 'BrassPatch.dll', sorted((runtime / 'src/BrassPatch').rglob('*.cs')), references, ['/nostdlib+'])
    tools = OUT / 'tools'; tools.mkdir()
    weaver_source = tools / 'Program.cs'; shutil.copyfile(ROOT / 'src/BrassPatch.Weaver/Program.cs', weaver_source)
    cecil = ROOT / 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll'
    for source in (cecil, managed / 'Newtonsoft.Json.dll'): shutil.copyfile(source, tools / source.name)
    compile_cs(tools / 'BrassPatch.Weaver.exe', [weaver_source], [cecil, managed / 'Newtonsoft.Json.dll'])
    source = tools / 'baseline.dll'; shutil.copyfile(BASE / 'Managed/Assembly-CSharp.dll', source)
    write(source.with_suffix('.dll.manifest.json'), dict(methods=[], baseline_sha256=sha(source),
        scope='Additional overlay only; all existing bodies retained by the weaver allowlist check.'))
    manifests = tools / 'overlay'; manifests.mkdir(); shutil.copyfile(ROOT / SCOPE[1], manifests / 'replay-speed.json')
    subprocess.run([str(tools / 'BrassPatch.Weaver.exe'), '--overlay', str(source), str(managed / 'BrassPatch.dll'), str(manifests), str(managed / 'Assembly-CSharp.dll'), sha(source)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    manifest = dict(version='1.4.49-rust-local', baseVersion=old['version'], stockSha256=old['stockSha256'],
        baseFiles=old['files'], files=[dict(name=n, sha256=sha(managed / n)) for n in NAMES],
        nativeFiles=pins(ROOT / 'BrassNative'), sourceScope=SCOPE, sourcePins=pins(runtime),
        installed_baseline_receipt_sha256=sha(ROOT / 'artifacts/patch-installation.json'),
        method_scope=['InterfaceAdapters.Common.IAnimation Frameworks.View.Animation.DOTweenAnimationSequence::Play()'])
    assert manifest['files'][2] == old['files'][2]
    write(OUT / 'build-manifest.json', manifest)
    print('Built scoped 1.4.49 replay-speed candidate', flush=True)


def player():
    if PLAYER.exists(): return
    PLAYER.mkdir(parents=True)
    def link_or_copy(source, destination):
        try: os.link(source, destination)
        except OSError: shutil.copyfile(source, destination)
    for name in ('BrassBirminghamUnity.exe', 'UnityCrashHandler64.exe', 'UnityPlayer.dll'):
        shutil.copyfile(ROOT / name, PLAYER / name)
    shutil.copytree(ROOT / 'MonoBleedingEdge', PLAYER / 'MonoBleedingEdge', copy_function=link_or_copy)
    data = PLAYER / 'BrassBirminghamUnity_Data'; data.mkdir()
    for source in (ROOT / 'BrassBirminghamUnity_Data').iterdir():
        if source.name == 'Managed': continue
        if source.is_dir(): shutil.copytree(source, data / source.name, copy_function=link_or_copy)
        elif source.is_file(): link_or_copy(source, data / source.name)
    shutil.copytree(BASE / 'Managed', data / 'Managed')
    (PLAYER / 'artifacts/patch-build/Managed').mkdir(parents=True)


def run(label, installed=False):
    target = STUDY / label
    if target.exists():
        assert not (target / 'unity.json').exists() or not read(target / 'unity.json')['passed'], 'Preserve passing evidence'
        attempt = 1
        while (STUDY / (label + '-failed-' + str(attempt))).exists(): attempt += 1
        retained = STUDY / (label + '-failed-' + str(attempt))
        assert target.resolve().parent == retained.resolve().parent == STUDY.resolve()
        target.rename(retained)
    target.mkdir(parents=True)
    if installed:
        game = ROOT
    else:
        player(); game = PLAYER
        source = BASE if label == 'baseline' else OUT
        for n in NAMES: shutil.copyfile(source / 'Managed' / n, game / 'BrassBirminghamUnity_Data/Managed' / n)
    managed = game / 'BrassBirminghamUnity_Data/Managed'
    testroot = game / 'artifacts/patch-build'
    tests = ['tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs',
             'tests/BrassPatch/Presentation/' + ('PresentationLifecycleTests.cs' if label == 'lifecycle' else 'SellReplaySpeedTests.cs')]
    frozen = []
    for name in tests:
        dst = target / 'test-source' / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dst); frozen.append(dst)
    dll = target / 'LocalAiUnityTests.dll'
    compile_cs(dll, frozen, [p for p in managed.glob('*.dll') if 'Tests' not in p.name and 'Regression' not in p.name], ['/nostdlib+'])
    touched = [testroot / 'Managed/LocalAiUnityTests.dll', testroot / 'local-ai-native.json']
    prior = {}
    for p in touched:
        if p.exists():
            saved = target / ('prior-' + p.name); shutil.copyfile(p, saved); prior[p] = saved
    try:
        shutil.copyfile(dll, touched[0]); write(touched[1], dict(passed=False, status='running-sell-speed-' + label))
        args = [str(game / 'BrassBirminghamUnity.exe'), '-batchmode', '-screen-width', '1920', '-screen-height', '1080', '-screen-fullscreen', '0', '-logFile', str(target / 'unity.log'), '--brass-local-ai-test']
        if label == 'baseline': args.append('--brass-sell-speed-baseline')
        result = subprocess.run(args, cwd=game, timeout=55, creationflags=subprocess.CREATE_NO_WINDOW)
        shutil.copyfile(touched[1], target / 'unity.json')
        evidence = read(target / 'unity.json')
        write(target / 'test-build.json', dict(testedFiles=[dict(name=n, sha256=sha(managed / n)) for n in NAMES],
            sources=[dict(path=p.relative_to(target / 'test-source').as_posix(), sha256=sha(p)) for p in frozen], testDllSha256=sha(dll)))
        print(json.dumps(dict(label=label, exit_code=result.returncode, passed=evidence.get('passed'), checks=evidence.get('checks'),
            reproductions=evidence.get('reproductions'), error=evidence.get('error'), failed=[c for c in evidence.get('cases', []) if c.get('passed') is False][:8])), flush=True)
        assert result.returncode == 0 and evidence['passed'], 'Unity regression failed: ' + label
        return evidence
    finally:
        for p in touched:
            if p in prior: shutil.copyfile(prior[p], p)
            elif p.exists(): p.unlink()


def qualify():
    assert not (OUT / 'qualification.json').exists()
    m = read(OUT / 'build-manifest.json'); old = read(BASE / 'build-manifest.json')
    previous = {p['path']: p['sha256'] for p in old['sourcePins']}
    assert {p['path'] for p in m['sourcePins'] if previous.get(p['path']) != p['sha256']} == set(SCOPE)
    for p in m['sourcePins']: assert sha(OUT / 'runtime' / p['path']) == p['sha256']
    for name in SCOPE: assert (ROOT / name).read_bytes() == (OUT / 'runtime' / name).read_bytes()
    for p in m['files']: assert sha(OUT / 'Managed' / p['name']) == p['sha256']
    assert sha(OUT / 'Managed/BrassLegacyAI.dll') == sha(BASE / 'Managed/BrassLegacyAI.dll')
    assert pins(ROOT / 'BrassNative') == m['nativeFiles']
    reports = []
    for label in ('baseline', 'fixed', 'lifecycle'):
        d = STUDY / label; e = read(d / 'unity.json'); t = read(d / 'test-build.json')
        assert e['passed'] and not e['nativeErrors']
        assert t['testDllSha256'] == sha(d / 'LocalAiUnityTests.dll')
        for p in t['sources']: assert sha(d / 'test-source' / p['path']) == p['sha256']
        assert t['testedFiles'] == (old if label == 'baseline' else m)['files']
        if label == 'baseline': assert e['reproductions'] == 28
        if label == 'fixed': assert len(e['timings']) == 12 and all(abs(v['ratio'] - 2) < .04 for v in e['timings'])
        reports.append(dict(label=label, checks=e['checks'], report_sha256=sha(d / 'unity.json'), test_build_sha256=sha(d / 'test-build.json')))
    write(OUT / 'qualification.json', dict(format='brass-sell-replay-speed/1.0.0', passed=True,
        version=m['version'], build_manifest_sha256=sha(OUT / 'build-manifest.json'), source_scope=SCOPE,
        method_scope=m['method_scope'], evidence=reports, preserved_ai=True, preserved_native=True))
    print('Qualified 1.4.49 sell replay input fix', flush=True)


def protected():
    result = []
    receipt = read(ROOT / 'artifacts/patch-installation.json')
    for path in receipt['saveDataPaths']:
        directory = Path(path)
        files = [dict(path=p.relative_to(directory).as_posix(), sha256=sha(p))
                 for folder in ('games', 'GameHistoryStatistics')
                 for p in sorted((directory / folder).rglob('*')) if p.is_file()]
        result.append(dict(directory=str(directory), files=files))
    return result


def snapshot():
    target = STUDY / 'protected-before.json'; assert not target.exists()
    write(target, dict(users=protected(), native=pins(ROOT / 'BrassNative'), legacy_sha256=sha(ROOT / 'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll')))
    print('Recorded installed models and user save/history fingerprints', flush=True)


def audit():
    before = read(STUDY / 'protected-before.json'); m = read(OUT / 'build-manifest.json')
    r = read(ROOT / 'artifacts/patch-installation.json'); u = read(STUDY / 'installed/unity.json')
    t = read(STUDY / 'installed/test-build.json'); upgrade = read(STUDY / 'upgrade-test.json')
    assert r['version'] == m['version'] and r['files'] == m['files'] == t['testedFiles']
    for p in m['files']: assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / p['name']) == p['sha256']
    assert u['passed'] and u['checks'] == 324 and not u['nativeErrors']
    assert upgrade['passed'] and upgrade['installer_sha256'] == sha(ROOT / 'scripts/install-sell-replay-speed.ps1')
    assert upgrade['qualification_sha256'] == sha(OUT / 'qualification.json')
    assert before['users'] == protected() and before['native'] == pins(ROOT / 'BrassNative') == m['nativeFiles']
    assert before['legacy_sha256'] == sha(ROOT / 'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll')
    for name in SCOPE: assert (ROOT / name).read_bytes() == (OUT / 'runtime' / name).read_bytes()
    summary = dict(passed=True, version=r['version'], installed_checks=u['checks'],
        native_files_unchanged=len(m['nativeFiles']), user_files_unchanged=sum(len(v['files']) for v in before['users']),
        installed_report_sha256=sha(STUDY / 'installed/unity.json'), qualification_sha256=sha(OUT / 'qualification.json'),
        upgrade_test_sha256=sha(STUDY / 'upgrade-test.json'),
        timing_ratios=[v['ratio'] for v in u['timings']])
    write(STUDY / 'installed-verification.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('build', 'baseline', 'fixed', 'lifecycle', 'qualify', 'installed', 'snapshot', 'audit'))
    stage = parser.parse_args().stage; STUDY.mkdir(parents=True, exist_ok=True)
    run(stage, installed=stage == 'installed') if stage in ('baseline', 'fixed', 'lifecycle', 'installed') else globals()[stage]()
