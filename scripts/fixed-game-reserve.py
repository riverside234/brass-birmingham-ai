"""Short admission-only reserve study; isolated candidates and frozen evidence."""
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
STUDY=ROOT/'artifacts/fixed-reserve/20261029'
BASE=ROOT/'artifacts/host-delivery/20261029/release-1.4.44'
OUT=STUDY/'release-1.4.45-v2'
NATIVE=STUDY/'native-build-v2'
FIXTURES=ROOT/'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures'
RESERVE_MS={8:600,16:600,32:800}


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True)
    p=STUDY/'protocol.json'
    if p.exists():return
    installed=read(ROOT/'artifacts/patch-installation.json')
    assert installed['version']=='1.4.44-rust-local' and read(BASE/'qualification.json')['passed']
    assert pins(ROOT/'BrassNative')==read(BASE/'build-manifest.json')['nativeFiles']
    write(STUDY/'before-installation.json',installed)
    write(p,dict(format='brass-fixed-reserve-study/1.0.0',base_version=installed['version'],
        installed_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),baseline_manifest_sha256=sha(BASE/'build-manifest.json'),
        allowances_ms=[4000,8000,10000,12000],search_k=[8,16,32],sessions=2,threads_per_session=1,
        method='search_v1',v205_enabled=False,temperature=1.0,candidate_reserve_ms_by_k=RESERVE_MS,
        minimum_final_clock_headroom_ms=100,probe='Hold only the test bridge Update until work cutoff; clocks and native jobs continue.',
        scope='One short late-consumption turn per candidate setting, three baseline extremes; no full game or tournament.',
        decision='Install fixed-per-K reserves only if all short cases finish both actions/save-load within the unchanged allowance with >=100ms clock headroom. Preserve every failure.'))


def smoke(build,label,allowance,k,fixture,hold=True):
    out=STUDY/label;assert not out.exists(),'Preserve existing evidence'
    reserve=RESERVE_MS[k] if build==OUT else max(allowance//10,800 if k==32 else 0)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',BRASS_SEARCH_V2_RESULT=str(out/'result.json'),
        BRASS_SEARCH_V2_EVENTS=str(FIXTURES/fixture),BRASS_SEARCH_V2_MS=str(allowance),BRASS_SEARCH_K=str(k),
        BRASS_SEARCH_V2_SCENARIO='normal',BRASS_NN_TEMPERATURE='1.0',BRASS_SEARCH_HOLD_UNTIL_MS=str(allowance-reserve if hold else 0))
    env.pop('BRASS_L14_TRACE',None)
    command=[shutil.which('pwsh'),'-NoProfile','-File',str(ROOT/'tests/RustBackendLocal/run-unity.ps1'),
        '-BuildDirectory',str(build.relative_to(ROOT)),'-EvidenceDirectory',str(out.relative_to(ROOT)),'-SearchV2','-MaxSeconds','60']
    with (STUDY/(label+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=85)
    r,u=read(out/'result.json'),read(out/'unity.json')
    assert r['passed'] and u['passed'] and r['search_k']==k and r['allowance_ms']==allowance
    assert r['search_method']=='search_v1' and not r['focused']
    assert len(r['commits'])==2 and r['commits'][-1]['remaining']==0
    assert r['replies'] and all(p['requested_k']==k and p['sessions']==2 and p['threads_per_session']==1
        and p['response_ready_ns']<p['deadline_ns'] for p in r['replies'])
    if hold:assert r['late_delivery_resumed_ms']>=allowance-reserve
    headroom=allowance-r['turn_clock_ms']
    write(out/'receipt.json',dict(passed=True,setting=f'{allowance}/K{k}',reserve_ms=reserve,
        checks=u['checks'],fixture=fixture,fixture_sha256=sha(FIXTURES/fixture),
        build_manifest_sha256=sha(build/'build-manifest.json'),result_sha256=sha(out/'result.json'),
        test_build_sha256=sha(out/'test-build.json'),delivery_ms=r['delivery_ms'],turn_clock_ms=r['turn_clock_ms'],
        headroom_ms=headroom,late_delivery_probe=hold,tail_ms=r['delivery_tail_after_resume_ms']))
    print(label,'delivery_ms',round(r['delivery_ms'],2),'tail_ms',round(r['delivery_tail_after_resume_ms'] or 0,2),
        'headroom_ms',round(headroom,2),flush=True)


def baseline():
    for allowance,k in [(4000,16),(12000,16),(12000,32)]:
        smoke(BASE,f'baseline-{allowance}-k{k}',allowance,k,'live-92781-RailEra-2.json')


def native():
    assert not NATIVE.exists() or not any(NATIVE.iterdir()),'Preserve existing native build evidence'
    spec=importlib.util.spec_from_file_location('fixed_local_native_builder',ROOT/'scripts/build-nn-k32-native.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=NATIVE;m.main()


def package():
    assert not OUT.exists()
    m=read(BASE/'build-manifest.json');n=read(NATIVE/'native-build.json')
    assert n['passed'] and sha(NATIVE/'brass-ai-local.exe')==n['binary_sha256']
    for pin in m['files']:assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/pin['name'])==pin['sha256']
    with zipfile.ZipFile(BASE/'BrassNative/alpha-g890/native-source.zip') as a,zipfile.ZipFile(NATIVE/'native-source.zip') as b:
        old={p:a.read(p) for p in a.namelist() if not p.endswith('/')};new={p:b.read(p) for p in b.namelist() if not p.endswith('/')}
    assert set(old)==set(new) and {p for p in old if old[p]!=new[p]}==set(n['scope'])
    for folder in ('Managed','runtime','BrassNative'):shutil.copytree(BASE/folder,OUT/folder)
    p=OUT/'BrassNative/alpha-g890'
    for name in ('brass-ai-local.exe','native-source.zip'):shutil.copyfile(NATIVE/name,p/name)
    build=read(p/'build.json');build['source_sha256']=n['source_sha256']
    for pin in build['files']:
        if pin['path']=='brass-ai-local.exe':pin['sha256']=sha(p/pin['path'])
        else:assert sha(p/pin['path'])==pin['sha256']
    write(p/'build.json',build)
    (OUT/'legacy.rsp').write_text((BASE/'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(BASE),str(OUT)),encoding='utf-8-sig')
    m.update(version='1.4.45-rust-local',baseVersion='1.4.44-rust-local',nativeFiles=pins(OUT/'BrassNative'),
        native_source_sha256=n['source_sha256'],native_build_sha256=sha(p/'build.json'),
        native_build_receipt_sha256=sha(NATIVE/'native-build.json'),fixed_reserve_ms_by_k=RESERVE_MS,
        installed_baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),baseFiles=read(BASE/'build-manifest.json')['files'],
        baseNative=read(BASE/'build-manifest.json')['nativeFiles'])
    write(OUT/'build-manifest.json',m)
    print(OUT,flush=True)


def matrix(k):
    for allowance in (4000,8000,10000,12000):
        # Keep actual K32 branching at every allowance. K8/16 additionally cover
        # Canal and long-history late-Rail captures.
        fixture='live-92781-RailEra-2.json'
        if k in (8,16) and allowance==8000:fixture='live-92781-CanalEra-2.json'
        if k in (8,16) and allowance==10000:fixture='live-92782-RailEra-7.json'
        smoke(OUT,f'candidate-v2-{allowance}-k{k}',allowance,k,fixture)


def qualify():
    assert not (OUT/'qualification.json').exists()
    rows=[]
    for k in RESERVE_MS:
        for allowance in (4000,8000,10000,12000):
            d=STUDY/f'candidate-v2-{allowance}-k{k}';receipt=read(d/'receipt.json');r=read(d/'result.json')
            assert receipt['passed'] and receipt['headroom_ms']>=100 and r['delivery_ms']<allowance
            assert receipt['build_manifest_sha256']==sha(OUT/'build-manifest.json')
            assert r['readiness']['host_delivery_reserve_policy']=='fixed-per-k/admission-only'
            assert r['readiness']['host_delivery_reserve_ns_by_k']=={str(key):v*1000000 for key,v in RESERVE_MS.items()}
            if k==32:assert any(o['effective_k']==32 for p in r['replies'] for o in p['replay']['owners'])
            rows.append(receipt)
    m=read(OUT/'build-manifest.json');old=read(BASE/'build-manifest.json')
    assert m['files']==old['files'] and m['sourcePins']==old['sourcePins']
    assert pins(OUT/'BrassNative')==m['nativeFiles']
    analysis=dict(passed=True,decision='Adopt fixed reserves per K across all allowances',reserve_ms_by_k=RESERVE_MS,
        rows=rows,baseline=[read(STUDY/f'baseline-{a}-k{k}/receipt.json') for a,k in [(4000,16),(12000,16),(12000,32)]],
        limitations=['One bounded late-consumption probe per setting, not a statistical latency-tail guarantee.',
            'The test bridge is paused until the work cutoff; actual native computation and original clocks keep running.',
            'K32 keeps a separate larger reserve; no untested common reserve is forced across K values.'])
    write(STUDY/'assessment.json',analysis)
    write(OUT/'qualification.json',dict(format='brass-fixed-reserve-release/1.0.0',passed=True,version=m['version'],
        build_manifest_sha256=sha(OUT/'build-manifest.json'),assessment_sha256=sha(STUDY/'assessment.json'),
        native_build_receipt_sha256=sha(NATIVE/'native-build.json'),fixed_reserve_ms_by_k=RESERVE_MS,
        shared_rust_core_unchanged=True,managed_unchanged=True,original_deadlines_unchanged=True,files=pins(OUT)))
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('baseline','native','package','k8','k16','k32','qualify'))
    stage=parser.parse_args().stage;prepare()
    matrix(int(stage[1:])) if stage.startswith('k') else globals()[stage]()
