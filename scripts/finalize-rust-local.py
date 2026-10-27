"""Finish the user-authorized local release in bounded, resumable groups.

The release scope is this Windows machine and the accepted temporary g890 NN.
Statistical strength claims and lab/teacher admission are separate capabilities.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import runpy
import shutil
import os
import statistics
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/rust-backend/20261026-local-release'
PRIOR = ROOT / 'artifacts/rust-backend/20261026-l15'
S = runpy.run_path(str(ROOT / 'scripts/qualify-rust-release.py'))
read, write, sha, pin, run = (S[k] for k in ('read','write','sha','pin','run'))


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    old = read(PRIOR/'acceptance.json')
    assert old['smoke_passed'] and not old['local_release_qualified']
    for entry in old['files']:
        S['check_pin'](entry)
    write(OUT/'protocol.json', dict(format='brass-local-release-scope/1.0.0',
        requested='Finish L14/L15, install final implementation and mark local migration complete (Full release).',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='User local Windows x64 game; unchanged accepted temporary g890 model; not a statistical strength certification or lab deployment.',
        test_scope='Short groups, <=90 seconds per player. Complete real-search games split at ordinary turn boundaries using trusted test event journals.',
        games=['four-seat human + three NN, actual 4-second search','two-seat human + V4, actual 4-second search'],
        matched_allowance_ms=4000, sessions=2, compute_threads_each=1,
        retain=['L14 validation benefit/exception','Rust copy/allocation optimization','ORT 1.22.1 CPU rc.10','human-End-only saves'],
        historical_gates='Do not rewrite historical failures. Short local acceptance does not establish population tails, statistical noninferiority or other CPU support.',
        required=['current pinned correctness evidence','normal-clock complete search games','allocation investigation with explicit counter validity',
                  'verified L13 upgrade/rollback/recovery','installed file/model/readiness verification','save preservation']))
    write(OUT/'before.json', dict(installed=S['installed_pins'](), prior_acceptance=pin(PRIOR/'acceptance.json')))
    shutil.copytree(PRIOR/'candidate', OUT/'candidate')
    # Only packaging identity changes. Reuse exactly the runtime binaries that
    # passed L15; never relabel a newly compiled binary with historical tests.
    manifest=read(OUT/'candidate/build-manifest.json')
    manifest.update(version='1.4.36-rust-local',baseVersion='1.4.35-alpha-test',
                    local_release_qualified=False,release_scope='local-game-g890',
                    tested_runtime_origin=pin(PRIOR/'candidate/build-manifest.json'))
    write(OUT/'candidate/build-manifest.json',manifest)
    print('Prepared release scope and byte-identical L15 runtime',flush=True)


def segment(family, number, attempt):
    mode=family+'-live';folder=OUT/'segments';folder.mkdir(exist_ok=True)
    destination=folder/f'{family}-{number:02d}.json'
    assert not destination.exists()
    prior=folder/f'{family}-{number-1:02d}.json'
    assert number==1 or prior.exists()
    env=dict(os.environ,BRASS_RELEASE_SEGMENT_IN='' if number==1 else str(prior),BRASS_RELEASE_SEGMENT_OUT=str(destination))
    label=f'{family}-{number:02d}-{attempt}'
    run(OUT,label,[S['powershell'](),'-NoProfile','-File',ROOT/'tests/RustBackendLocal/run-unity.ps1',
        '-BuildDirectory',(OUT/'candidate').relative_to(ROOT),'-EvidenceDirectory',(OUT/(label+'-evidence')).relative_to(ROOT),
        '-MaxSeconds','90','-ReleaseCase',mode],env,timeout=110)
    data=read(destination)
    assert data['passed'] and data['saves_after_fixture_setup']==data['human_turns']
    print(json.dumps({k:data[k] for k in ('mode','turns','terminal','seconds','ai_turns','debt_phases')}),flush=True)


def summarize_games():
    result={}
    for family in ('nn','v4'):
        paths=sorted((OUT/'segments').glob(f'{family}-*.json'))
        rows=[];last=0;reports=[];saves=0;seconds=0;debt=0
        for number,path in enumerate(paths,1):
            data=read(path)
            assert data['passed'] and data['mode']==family+'-live' and data['turns']==last+len(data['trace'])
            assert data['saves_after_fixture_setup']==data['human_turns']
            assert number==1 or data['prior']==str(paths[number-2])
            if rows:assert data['trace'][0]['before']==rows[-1]['after']
            assert [r['turn'] for r in data['trace']]==list(range(last+1,data['turns']+1))
            rows.extend(data['trace']);last=data['turns'];seconds+=data['seconds'];debt+=data['debt_phases'];saves+=data['saves_after_fixture_setup']
            evidence=[p for p in OUT.glob(f'{family}-{number:02d}-*-evidence/unity.release.json') if read(p).get('passed')]
            assert len(evidence)==1
            e=evidence[0].parent;test=read(e/'test-build.json')
            assert test['candidateManifestSha256']==sha(OUT/'candidate/build-manifest.json')
            assert read(e/'unity.json')['passed'] and sha(e/'LocalAiUnityTests.dll')==test['testDllSha256']
            for entry in test['sources']:assert sha(e/'test-source'/entry['path'])==entry['sha256']
            reports += [pin(path),pin(Path(str(path)+'.events')),pin(Path(str(path)+'.metadata')),pin(e/'test-build.json'),pin(e/'unity.json')]
        assert paths and read(paths[-1])['terminal'] and {r['era'] for r in rows}=={'CanalEra','RailEra'}
        assert all(r['human'] or r['allowance_ms']==4000 for r in rows)
        replies=[r for t in rows for r in t['replies']]
        assert all(r['retired'] and r['response_ready_ns']<r['deadline_ns'] for r in replies)
        lat=[r['delivery_ms'] for r in rows if not r['human']]
        result[family]=dict(passed=True,turns=last,actions=sum(r['actions'] for r in rows),ai_turns=len(lat),human_end_saves=saves,
            debt_phases=debt,game_execution_seconds=seconds,segments=len(paths),terminal=True,scores=read(paths[-1])['scores'],
            delivery_ms=dict(median=statistics.median(lat),maximum=max(lat),observed_p95=sorted(lat)[__import__('math').ceil(len(lat)*.95)-1]),
            native_replies=len(replies),supported=sum(r['provenance']=='pooled_search_supported' for r in replies),
            native_action_families=dict(Counter(a[0].split('|')[0] for r in replies for a in r['actions'])),
            limitations='One game, scripted human Pass input, cold initialization each short segment. Not statistical strength or uninterrupted-game memory evidence.',evidence=reports)
    write(OUT/'complete-games.json',result)
    return result


def qualify():
    build=OUT/'candidate';assert not (build/'qualification.json').exists()
    summarize_games()
    allocation_dir=OUT/'allocation-final'
    allocation=read(allocation_dir/'receipt.json');assert allocation['passed']
    for entry in allocation['source']:S['check_pin'](entry)
    manifest=read(build/'build-manifest.json');origin=read(PRIOR/'candidate/build-manifest.json')
    assert manifest['files']==origin['files'] and manifest['native_source_sha256']==S['SUPPORT']['native_fingerprint']()
    for entry in origin['files']:assert sha(build/'Managed'/entry['name'])==entry['sha256']
    for entry in origin['sourcePins']:assert sha(build/'runtime'/entry['path'])==entry['sha256']
    for entry in origin['sourceScope']:assert sha(ROOT/entry)==sha(build/'runtime'/entry)
    for p in (PRIOR/'candidate/BrassNative').rglob('*'):
        if p.is_file():assert sha(p)==sha(build/'BrassNative'/p.relative_to(PRIOR/'candidate/BrassNative'))
    # Keep the tested manifest before adding release-only packaging metadata.
    shutil.copyfile(build/'build-manifest.json',OUT/'tested-build-manifest.json')
    manifest.update(local_release_qualified=True,tested_manifest=pin(OUT/'tested-build-manifest.json'))
    write(build/'build-manifest.json',manifest)
    readme=build/'BrassNative/alpha-g890/README.txt'
    readme.write_text(readme.read_text().replace('Alpha Test g890 local Windows x64 runtime, L13. Not an L14/L15 release qualification.',
        'Local Rust backend release 1.4.36, qualified on the user Windows x64 host. Alpha Test uses the temporary g890 NN.'),encoding='utf-8')
    evidence=build/'release-evidence';evidence.mkdir()
    for name in ('protocol.json','before.json','complete-games.json','tested-build-manifest.json'):
        shutil.copyfile(OUT/name,evidence/name)
    shutil.copyfile(allocation_dir/'receipt.json',evidence/'allocation-receipt.json')
    shutil.copyfile(PRIOR/'acceptance.json',evidence/'l15-mechanical-acceptance.json')
    shutil.copyfile(ROOT/'artifacts/rust-backend/20261026-l14-smoke/assessment.json',evidence/'l14-assessment.json')
    for name in ('install-rust-local-release.ps1','test-rust-local-upgrade.ps1','finalize-rust-local.py','measure-mono-allocation.py'):
        shutil.copyfile(ROOT/'scripts'/name,evidence/name)
    support=read(PRIOR/'acceptance.json')['support_matrix']
    support['v4']='Existing two/three/four-seat settings; two-seat normal-clock complete search game, 3/4-seat mechanics and preserved V4 qualification.'
    qualification=dict(format='brass-local-release/1.0.0',scope='local-game-g890',passed=True,local_release_qualified=True,
        local_migration_complete=True,final_model=False,part2_entry_eligible=True,lab_deployed=False,teacher_admitted=False,
        support=support,provider=read(PRIOR/'acceptance.json')['provider'],
        build_manifest_sha256=sha(build/'build-manifest.json'),native_source_sha256=manifest['native_source_sha256'],
        prior_acceptance=pin(PRIOR/'acceptance.json'),games=pin(OUT/'complete-games.json'),allocation=pin(allocation_dir/'receipt.json'),
        scope_decision='2026-10-26 user-requested full local implementation/deployment; bounded verification retained. Historical statistical strength and population-tail screens are not represented as passed.',
        limitations=['Temporary g890 model; final NN requires fresh admission.',
            'Only measured Windows x64 host qualified; standard four-seat NN tables with human; V4 existing 2/3/4-seat support.',
            'No statistical strength/noninferiority or population-tail guarantee. Historical performance screen exceptions retained.',
            'Managed validation allocations measured on exact Mono embedding host; full Unity presentation/engine allocation totals unavailable.',
            'Lab Linux/GPU/training/teacher qualification remains separate.'],
        files=[dict(path=p.relative_to(build).as_posix(),sha256=sha(p)) for p in sorted(build.rglob('*')) if p.is_file()])
    write(build/'qualification.json',qualification)
    write(build/'final-verification.json',dict(passed=True,qualification_sha256=sha(build/'qualification.json'),files_verified=len(qualification['files'])))
    print('Sealed local-game release for disposable upgrade testing',flush=True)


def save_pins():
    record=read(ROOT/'artifacts/patch-installation.json')
    return [dict(path=str(p),sha256=sha(p)) for directory in record['saveDataPaths'] for p in sorted(Path(directory).rglob('*')) if p.is_file()]


def before_install():
    assert read(OUT/'upgrade-test/receipt.json')['passed']
    assert not (OUT/'saves-before.json').exists()
    for entry in read(OUT/'before.json')['installed']:S['check_pin'](entry)
    write(OUT/'saves-before.json',save_pins())
    print('Preinstall runtime and save hashes recorded',flush=True)


def verify_installed():
    record=read(ROOT/'artifacts/patch-installation.json');build=OUT/'candidate';manifest=read(build/'build-manifest.json')
    assert record['localReleaseQualified'] and record['version']==manifest['version']
    assert record['qualificationSha256'].lower()==sha(build/'qualification.json')
    for entry in manifest['files']:assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/entry['name'])==entry['sha256']
    native=build/'BrassNative'
    for p in native.rglob('*'):
        if p.is_file():assert sha(p)==sha(ROOT/'BrassNative'/p.relative_to(native))
    assert len([p for p in native.rglob('*') if p.is_file()])==len([p for p in (ROOT/'BrassNative').rglob('*') if p.is_file()])
    assert save_pins()==read(OUT/'saves-before.json')
    worker=runpy.run_path(str(ROOT/'tests/RustBackendLocal/test_local.py'))['Worker'](ROOT/'BrassNative/alpha-g890')
    try:
        hello=worker.hello['data'];assert hello['provider']=='CPUExecutionProvider'
        assert hello['build']['source_sha256']==manifest['native_source_sha256']
        write(OUT/'installed-readiness.json',hello)
    finally:worker.close()
    write(OUT/'installed-verification.json',dict(passed=True,installed=True,local_release_qualified=True,local_migration_complete=True,
        final_model=False,lab_deployed=False,version=record['version'],save_files_unchanged=len(save_pins()),
        installation=pin(ROOT/'artifacts/patch-installation.json'),qualification=pin(build/'qualification.json'),
        readiness=pin(OUT/'installed-readiness.json'),recovery_directory=record['localReleaseRecovery'],upgrade_tests=pin(OUT/'upgrade-test/receipt.json')))
    print('Installed runtime, model, readiness and every existing save verified',flush=True)


def seal():
    assert not (OUT/'release-acceptance.json').exists()
    assert read(OUT/'installed-verification.json')['passed'] and read(OUT/'upgrade-test/receipt.json')['passed']
    package=OUT/'candidate';q=read(package/'qualification.json')
    for entry in q['files']:assert sha(package/entry['path'])==entry['sha256']
    assert sha(ROOT/'scripts/install-rust-local-release.ps1')==read(OUT/'upgrade-test/receipt.json')['installerSha256'].lower()
    sources=[ROOT/'scripts'/n for n in ('finalize-rust-local.py','install-rust-local-release.ps1','test-rust-local-upgrade.ps1','measure-mono-allocation.py')]
    sources += [ROOT/'tests/RustBackendLocal'/n for n in ('ReleaseSegmentUnityTests.cs','NativeReleaseUnityTests.cs','run-unity.ps1','AllocationHost.c','AllocationBench.cs')]
    write(OUT/'release-acceptance.json',dict(format='brass-local-final-acceptance/1.0.0',passed=True,installed=True,
        local_migration_complete=True,local_release_qualified=True,part2_entry_eligible=True,lab_deployed=False,final_model=False,
        qualification=pin(package/'qualification.json'),installed_verification=pin(OUT/'installed-verification.json'),
        upgrade=pin(OUT/'upgrade-test/receipt.json'),games=pin(OUT/'complete-games.json'),allocation=pin(OUT/'allocation-final/receipt.json'),
        limits=q['limitations'],source_pins=[pin(p) for p in sources],
        files=[pin(p) for p in sorted(OUT.rglob('*')) if p.is_file()]))
    audit()


def audit():
    record=read(OUT/'release-acceptance.json')
    for entry in record['files']+record['source_pins']:S['check_pin'](entry)
    write(OUT/'final-verification.json',dict(passed=True,acceptance=pin(OUT/'release-acceptance.json'),files_verified=len(record['files']),
        local_release_qualified=True,installed=True,lab_deployed=False))
    print('Full local release evidence verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','segment','qualify','before-install','verify-installed','seal','audit'])
    p.add_argument('--family',choices=['nn','v4']);p.add_argument('--number',type=int);p.add_argument('--attempt',default='a')
    a=p.parse_args()
    if a.mode=='segment':segment(a.family,a.number,a.attempt)
    else:globals()[a.mode.replace('-','_')]()
