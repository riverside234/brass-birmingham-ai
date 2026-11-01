"""Restore one neutral NN menu option; preserve installed engine and signed saves."""
from pathlib import Path
import argparse
import os
import runpy
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(ROOT/'scripts/build-nn-search-choice.py'))
sha,read,write,pins=(H[k] for k in ('sha','read','write','pins'))
BASE=ROOT/'artifacts/nn-industry-focus/20261031/release-1.4.46'
STUDY=ROOT/'artifacts/nn-neutral-ui/20261031'
OUT=STUDY/'release-1.4.47'
SCOPE=['src/BrassLocalAI/Policy/AiSelectionCatalog.cs','src/BrassLocalAI/Policy/LocalSetupPreferences.cs',
       'src/BrassLocalAI/Unity/AiSelectionPanel.cs','src/BrassLocalAI/Unity/LocalMenu.cs']


def package():
    assert not OUT.exists(),'Preserve candidate evidence'
    installed=read(ROOT/'artifacts/patch-installation.json');old=read(BASE/'build-manifest.json')
    assert installed['version']==old['version']=='1.4.46-rust-local'
    assert read(BASE/'qualification.json')['passed'] and pins(ROOT/'BrassNative')==old['nativeFiles']
    for p in old['files']:
        assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/p['name'])==p['sha256']
    for folder in ('runtime','Managed','BrassNative'):shutil.copytree(BASE/folder,OUT/folder)
    for name in SCOPE:shutil.copyfile(ROOT/name,OUT/'runtime'/name)
    rsp=OUT/'legacy.rsp';rsp.write_text((BASE/'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(BASE),str(OUT)),encoding='utf-8-sig')
    subprocess.run([str(ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'),'@'+str(rsp)],check=True)
    old.update(version='1.4.47-rust-local',baseVersion=installed['version'],baseFiles=installed['files'],
        baseNative=pins(ROOT/'BrassNative'),sourceScope=SCOPE,sourcePins=pins(OUT/'runtime'),
        files=[dict(name=p['name'],sha256=sha(OUT/'Managed'/p['name'])) for p in old['files']],
        installed_baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),
        nn_ui_choices=['alpha-test'],nn_ui_focuses=['neutral'],default_nn_focus='neutral',native_unchanged=True)
    write(OUT/'build-manifest.json',old)
    shutil.copyfile(ROOT/'artifacts/patch-installation.json',STUDY/'before-installation.json')
    user_files=[dict(path=str(p),sha256=sha(p)) for directory in installed['saveDataPaths']
                for p in Path(directory).rglob('*') if p.is_file()]
    write(STUDY/'user-data-before.json',dict(files=user_files))
    print(OUT,flush=True)


def unity(label):
    d=STUDY/label;assert not d.exists(),'Preserve smoke evidence'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('BRASS_INDUSTRY_FOCUS_TEST',None);env.pop('BRASS_L14_TRACE',None)
    command=[shutil.which('pwsh'),'-NoProfile','-File',str(ROOT/'tests/RustBackendLocal/run-unity.ps1'),
             '-BuildDirectory',str(OUT.relative_to(ROOT)),'-EvidenceDirectory',str(d.relative_to(ROOT)),'-MaxSeconds','60']
    if label.startswith('ui-'):command+=['-SetupPreferences',label[3:]]
    else:
        env.update(BRASS_SEARCH_V2_RESULT=str(d/'result.json'),
            BRASS_SEARCH_V2_EVENTS=str(ROOT/'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json'),
            BRASS_SEARCH_V2_MS='4000',BRASS_SEARCH_K='16',BRASS_NN_TEMPERATURE='1.0',BRASS_NN_FOCUS='neutral',
            BRASS_SEARCH_V2_SCENARIO='normal',BRASS_SEARCH_HOLD_UNTIL_MS='0')
        command+=['-SearchV2']
    with (STUDY/(label+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=85)
    u=read(d/'unity.json');assert u['passed'] and not u['nativeErrors']
    if label=='live-neutral':
        r=read(d/'result.json');assert r['passed'] and r['industry_focus']=='neutral' and r['search_method']=='search_v1'
        assert len(r['commits'])==2 and r['commits'][-1]['remaining']==0 and r['turn_clock_ms']<5000
    print(label,u['checks'],'checks',flush=True)


def qualify():
    assert not (OUT/'qualification.json').exists()
    m=read(OUT/'build-manifest.json');old=read(BASE/'build-manifest.json');previous={p['path']:p['sha256'] for p in old['sourcePins']}
    assert {p['path'] for p in m['sourcePins'] if previous.get(p['path'])!=p['sha256']}==set(SCOPE)
    for p in m['sourcePins']:assert sha(OUT/'runtime'/p['path'])==p['sha256']
    assert pins(OUT/'BrassNative')==m['nativeFiles']==m['baseNative']==pins(ROOT/'BrassNative')
    for p in m['files']:
        assert sha(OUT/'Managed'/p['name'])==p['sha256']
        if p['name']!='BrassLegacyAI.dll':assert p in old['files']
    evidence=[]
    for label in ('ui-write','ui-read','live-neutral'):
        d=STUDY/label;t=read(d/'test-build.json');u=read(d/'unity.json')
        assert u['passed'] and not u['nativeErrors']
        assert t['candidateManifestSha256']==sha(OUT/'build-manifest.json') and t['testDllSha256']==sha(d/'LocalAiUnityTests.dll')
        for p in t['sources']:assert sha(d/'test-source'/p['path'])==p['sha256']
        evidence.append(dict(label=label,checks=u['checks'],unity_sha256=sha(d/'unity.json'),test_build_sha256=sha(d/'test-build.json')))
    write(OUT/'qualification.json',dict(format='brass-nn-neutral-ui-release/1.0.0',passed=True,version=m['version'],
        build_manifest_sha256=sha(OUT/'build-manifest.json'),native_unchanged=True,ui_choices=['alpha-test'],default_focus='neutral',
        evidence=evidence,files=pins(OUT)))
    print('Qualified single neutral NN UI',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('package','ui-write','ui-read','live-neutral','qualify'))
    stage=parser.parse_args().stage;STUDY.mkdir(parents=True,exist_ok=True)
    unity(stage) if stage.startswith(('ui-','live-')) else globals()[stage]()
