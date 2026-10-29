"""Local full search allowance plus one bounded delivery second; no lab changes."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(ROOT/'scripts/build-nn-search-choice.py'))
read,write,sha,pins=(H[k] for k in ('read','write','sha','pins'))
STUDY=ROOT/'artifacts/search-delivery-grace/20261029'
BASE=ROOT/'artifacts/host-delivery/20261029/release-1.4.44'
OUT=STUDY/'release-1.4.45'
NATIVE=STUDY/'native-build'
FIXTURES=ROOT/'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures'
SCOPE=['src/BrassLocalAI/Native/NativeProcess.cs','src/BrassLocalAI/Native/NativeGameState.cs',
    'src/BrassLocalAI/Unity/LocalTurnBridge.cs','src/BrassLocalAI/Unity/AiSelectionPanel.cs']


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'protocol.json').exists():return
    installed=read(ROOT/'artifacts/patch-installation.json')
    assert installed['version']=='1.4.44-rust-local'
    assert pins(ROOT/'BrassNative')==read(BASE/'build-manifest.json')['nativeFiles']
    write(STUDY/'before-installation.json',installed)
    write(STUDY/'protocol.json',dict(format='brass-search-delivery-grace-study/1.0.0',
        base_version=installed['version'],baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),
        search_allowances_ms=[4000,8000,10000,12000],search_k=[8,16,32],delivery_grace_ms=1000,
        sessions=2,threads_per_session=1,method='search_v1',v205_enabled=False,temperature=1.0,
        probe='Pause only test bridge Update until search cutoff; mark a completed test response delivered at resume time. Native timing is retained separately.',
        promotion='Every late result completes two actions/save-load within search+1000ms and with >=100ms clock headroom; reject an already-expired reply before mutation.',
        scope='Short independent single-turn probes; no tournament or complete-game run.'))


def native():
    assert not NATIVE.exists()
    spec=importlib.util.spec_from_file_location('grace_native_builder',ROOT/'scripts/build-nn-k32-native.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=NATIVE;m.main()


def package():
    assert not OUT.exists()
    m=read(BASE/'build-manifest.json');n=read(NATIVE/'native-build.json')
    assert n['passed']
    for pin in m['files']:assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/pin['name'])==pin['sha256']
    with zipfile.ZipFile(BASE/'BrassNative/alpha-g890/native-source.zip') as a,zipfile.ZipFile(NATIVE/'native-source.zip') as b:
        old={p:a.read(p) for p in a.namelist() if not p.endswith('/')};new={p:b.read(p) for p in b.namelist() if not p.endswith('/')}
    assert set(old)==set(new) and {p for p in old if old[p]!=new[p]}==set(n['scope'])
    for folder in ('runtime','Managed','BrassNative'):shutil.copytree(BASE/folder,OUT/folder)
    for name in SCOPE:shutil.copyfile(ROOT/name,OUT/'runtime'/name)
    rsp=OUT/'legacy.rsp'
    rsp.write_text((BASE/'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(BASE),str(OUT)),encoding='utf-8-sig')
    subprocess.run([str(ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'),'@'+str(rsp)],check=True)
    p=OUT/'BrassNative/alpha-g890'
    for name in ('brass-ai-local.exe','native-source.zip'):shutil.copyfile(NATIVE/name,p/name)
    build=read(p/'build.json');build['source_sha256']=n['source_sha256']
    for pin in build['files']:
        if pin['path']=='brass-ai-local.exe':pin['sha256']=sha(p/pin['path'])
        else:assert sha(p/pin['path'])==pin['sha256']
    write(p/'build.json',build)
    m.update(version='1.4.45-rust-local',baseVersion='1.4.44-rust-local',sourceScope=SCOPE,sourcePins=pins(OUT/'runtime'),
        files=[dict(name=p['name'],sha256=sha(OUT/'Managed'/p['name'])) for p in m['files']],nativeFiles=pins(OUT/'BrassNative'),
        native_source_sha256=n['source_sha256'],native_build_sha256=sha(p/'build.json'),delivery_grace_ms=1000,
        native_build_receipt_sha256=sha(NATIVE/'native-build.json'),deadline_policy='search-plus-delivery/1.0.0',
        installed_baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),
        baseFiles=read(BASE/'build-manifest.json')['files'],baseNative=read(BASE/'build-manifest.json')['nativeFiles'])
    write(OUT/'build-manifest.json',m)
    print(OUT,flush=True)


def smoke(label,allowance,k,late=True,scenario='normal'):
    out=STUDY/label;assert not out.exists()
    fixture='live-92781-RailEra-2.json'
    if allowance==8000:fixture='live-92781-CanalEra-2.json'
    if allowance==10000:fixture='live-92782-RailEra-7.json'
    hold=allowance+1050 if scenario=='expired' else allowance if late else 0
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',BRASS_SEARCH_V2_RESULT=str(out/'result.json'),
        BRASS_SEARCH_V2_EVENTS=str(FIXTURES/fixture),BRASS_SEARCH_V2_MS=str(allowance),BRASS_SEARCH_K=str(k),
        BRASS_SEARCH_V2_SCENARIO=scenario,BRASS_NN_TEMPERATURE='1.0',BRASS_SEARCH_HOLD_UNTIL_MS=str(hold))
    env.pop('BRASS_L14_TRACE',None)
    command=[shutil.which('pwsh'),'-NoProfile','-File',str(ROOT/'tests/RustBackendLocal/run-unity.ps1'),
        '-BuildDirectory',str(OUT.relative_to(ROOT)),'-EvidenceDirectory',str(out.relative_to(ROOT)),'-SearchV2','-MaxSeconds','60']
    with (STUDY/(label+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=85)
    r,u=read(out/'result.json'),read(out/'unity.json');assert r['passed'] and u['passed']
    assert r['delivery_grace_ms']==1000 and r['total_allowance_ms']==allowance+1000
    assert r['readiness']['host_delivery_reserve_policy']=='search-plus-delivery/1.0.0'
    assert r['search_k']==k and r['allowance_ms']==allowance and r['search_method']=='search_v1' and not r['focused']
    assert all(p['requested_k']==k and p['sessions']==2 and p['threads_per_session']==1 and p['response_ready_ns']<p['deadline_ns'] for p in r['replies'])
    if scenario=='expired':assert not r['commits']
    else:
        assert len(r['commits'])==2 and r['commits'][-1]['remaining']==0
        assert r['turn_clock_ms']<allowance+1000 and r['delivery_ms']<allowance+1000
        if late:assert r['late_delivery_resumed_ms']>=allowance and r['delivery_tail_after_resume_ms']<1000
    headroom=allowance+1000-r['turn_clock_ms']
    write(out/'receipt.json',dict(passed=True,scenario=scenario,search_k=k,search_ms=allowance,delivery_grace_ms=1000,
        total_ms=allowance+1000,checks=u['checks'],delivery_ms=r['delivery_ms'],clock_ms=r['turn_clock_ms'],headroom_ms=headroom,
        tail_ms=r['delivery_tail_after_resume_ms'],late_probe=late,result_sha256=sha(out/'result.json'),
        build_manifest_sha256=sha(OUT/'build-manifest.json'),test_build_sha256=sha(out/'test-build.json')))
    print(label,'delivery_ms',round(r['delivery_ms'],2),'tail_ms',round(r['delivery_tail_after_resume_ms'] or 0,2),'headroom_ms',round(headroom,2),flush=True)


def matrix(k):
    for allowance in (4000,8000,10000,12000):smoke(f'late-{allowance}-k{k}',allowance,k)


def negative():
    smoke('expired-4000-k16',4000,16,True,'expired')
    smoke('normal-4000-k8',4000,8,False)
    smoke('normal-4000-k32',4000,32,False)


def qualify():
    assert not (OUT/'qualification.json').exists()
    rows=[];evidence=[]
    def verified_test(directory):
        t=read(directory/'test-build.json')
        assert t['candidateManifestSha256']==sha(OUT/'build-manifest.json')
        assert t['testDllSha256']==sha(directory/'LocalAiUnityTests.dll')
        for p in t['sources']:assert sha(directory/'test-source'/p['path'])==p['sha256']
        assert read(directory/'unity.json')['passed']
        evidence.append(dict(directory=str(directory.relative_to(ROOT)),
            test_build_sha256=sha(directory/'test-build.json'),unity_sha256=sha(directory/'unity.json')))
    for k in (8,16,32):
        for allowance in (4000,8000,10000,12000):
            d=STUDY/f'late-{allowance}-k{k}';r=read(d/'receipt.json')
            verified_test(d)
            assert r['passed'] and r['headroom_ms']>=100 and r['tail_ms']<1000
            assert r['build_manifest_sha256']==sha(OUT/'build-manifest.json')
            assert r['result_sha256']==sha(d/'result.json') and r['test_build_sha256']==sha(d/'test-build.json')
            rows.append(r)
    for name in ('expired-4000-k16','normal-4000-k8','normal-4000-k32'):
        d=STUDY/name;r=read(d/'receipt.json');verified_test(d)
        assert r['passed'] and r['result_sha256']==sha(d/'result.json') and r['test_build_sha256']==sha(d/'test-build.json')
    assert read(STUDY/'budget-checks/result.json')['passed']
    verified_test(STUDY/'lifecycle')
    lifecycle=read(STUDY/'lifecycle/unity.json');assert lifecycle['passed']
    old=read(BASE/'build-manifest.json');m=read(OUT/'build-manifest.json')
    previous={p['path']:p['sha256'] for p in old['sourcePins']}
    changed=[p['path'] for p in m['sourcePins'] if previous[p['path']]!=p['sha256']]
    assert set(changed)==set(SCOPE)
    assert pins(OUT/'BrassNative')==m['nativeFiles']
    for p in m['files']:
        assert sha(OUT/'Managed'/p['name'])==p['sha256']
        if p['name']!='BrassLegacyAI.dll':assert p['sha256']==next(x['sha256'] for x in old['files'] if x['name']==p['name'])
    analysis=dict(passed=True,decision='Use full selected search time plus one additional delivery second for every K.',
        deadline_policy='search-plus-delivery/1.0.0',rows=rows,delivery_grace_ms=1000,
        evidence=evidence,budget_result_sha256=sha(STUDY/'budget-checks/result.json'),
        minimum_headroom_ms=min(r['headroom_ms'] for r in rows),maximum_delivery_tail_ms=max(r['tail_ms'] for r in rows),
        lifecycle_checks=lifecycle['checks'],budget_checks=read(STUDY/'budget-checks/result.json')['checks'],
        limitations=['Short single-turn probes, not a strength or population latency-tail guarantee.',
            'Late probes inject transport timing only; native timing and unmodified search clocks remain separately recorded.',
            'Initial model loading and initial board preparation remain outside the search clock as before.'])
    write(STUDY/'assessment.json',analysis)
    write(OUT/'qualification.json',dict(format='brass-search-delivery-grace-release/1.0.0',passed=True,version=m['version'],
        build_manifest_sha256=sha(OUT/'build-manifest.json'),assessment_sha256=sha(STUDY/'assessment.json'),
        native_build_receipt_sha256=sha(NATIVE/'native-build.json'),delivery_grace_ms=1000,
        shared_rust_core_unchanged=True,source_scope=SCOPE,files=pins(OUT)))
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('native','package','k8','k16','k32','negative','qualify'))
    stage=parser.parse_args().stage;prepare()
    matrix(int(stage[1:])) if stage.startswith('k') else globals()[stage]()
