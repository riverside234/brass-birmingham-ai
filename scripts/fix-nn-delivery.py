"""Scoped managed delivery fix over the installed 1.4.51 release."""
from pathlib import Path
import json, os, runpy, shutil, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(ROOT/'scripts/fix-sell-replay-speed.py'))
sha,read,write,pins,compile_cs=(H[k] for k in ('sha','read','write','pins','compile_cs'))
STUDY=ROOT/'artifacts/nn-delivery/20261105'
BASE=ROOT/'artifacts/mixed-ai/20261102/release-1.4.51'
OUT=STUDY/'release-1.4.52'
SCOPE=['src/BrassLocalAI/'+p for p in ('Native/NativeProcess.cs','Native/NativeValidation.cs','Unity/LocalTurnBridge.cs')]
NAMES=('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')

def build():
    assert not OUT.exists(), 'Preserve previous build'
    old=read(BASE/'build-manifest.json'); installed=read(ROOT/'artifacts/patch-installation.json')
    assert installed['version']==old['version']=='1.4.51-rust-local'
    for p in installed['files']:
        assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/p['name'])==p['sha256']==sha(BASE/'Managed'/p['name'])
    shutil.copytree(BASE/'runtime',OUT/'runtime');shutil.copytree(BASE/'Managed',OUT/'Managed')
    for name in SCOPE: shutil.copyfile(ROOT/name,OUT/'runtime'/name)
    rsp=(BASE/'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(BASE),str(OUT))
    (OUT/'legacy.rsp').write_text(rsp,encoding='utf-8-sig')
    subprocess.run([str(H['CSC']),'@'+str(OUT/'legacy.rsp')],check=True)
    write(OUT/'build-manifest.json',dict(version='1.4.52-rust-local',baseVersion=installed['version'],stockSha256=old['stockSha256'],
        baseFiles=installed['files'],files=[dict(name=n,sha256=sha(OUT/'Managed'/n)) for n in NAMES],nativeFiles=pins(ROOT/'BrassNative'),
        sourceScope=SCOPE,sourcePins=pins(OUT/'runtime'),installed_baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),
        search_method='search_v1',v205_enabled=False,delivery_grace_ms=1000))
    write(STUDY/'protected-before.json',H['protected']())
    # Copy only; never open the user's save for mutation.
    source=Path(os.environ['USERPROFILE'])/'AppData/LocalLow/Cublo/Brass Birmingham/games/local/local_game/adaffe44-9206-42c5-b237-1feff4f8c19b/events.storage'
    data=read(source);data['$values']=[e for e in data['$values'] if e['Metadata']['Revision']<=729]
    write(STUDY/'reported-events.json',data)
    failure=read(ROOT/'artifacts/native-failures/native-failure-d226086cb4214ba7be572506aaa97251.json')
    write(STUDY/'validation-fixtures.json',[dict(name='reported-beta',input=json.loads(failure['request']['input_json']),actions=failure['response']['actions'])])
    print('Built managed-only 1.4.52 candidate',flush=True)

def smoke(label, allowance=4000, hold=0, scenario='normal'):
    target=STUDY/label;assert not target.exists();target.mkdir(parents=True)
    installed=label.startswith('installed');game=ROOT if installed else ROOT/'artifacts/card-tracker-player'
    managed=game/'BrassBirminghamUnity_Data/Managed'
    if not installed:
        source=BASE if label.startswith('baseline') else OUT
        for n in NAMES:shutil.copyfile(source/'Managed'/n,managed/n)
    tests=['tests/BrassLocalAI/LocalAiUnityTests.cs','tests/BrassLocalAI/AiExecutionResourceTests.cs']+[
        'tests/RustBackendLocal/'+n+'.cs' for n in ('NativeLocalUnityTests','NativeMatchedUnityTests','NativeValidationLifecycleTests','NativeReleaseUnityTests','ReleaseSegmentUnityTests','SearchV2UnityTests')]
    sources=[]
    for name in tests:
        dest=target/'test-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);sources.append(dest)
    dll=target/'LocalAiUnityTests.dll'
    compile_cs(dll,sources,[p for p in managed.glob('*.dll') if 'Tests' not in p.name and 'Regression' not in p.name],['/nostdlib+'])
    staging=game/'artifacts/patch-build';touched=[staging/'Managed/LocalAiUnityTests.dll',staging/'local-ai-native.json',staging/'local-ai-cold-menu.png']
    prior={}
    for p in touched:
        if p.exists():
            saved=target/('prior-'+p.name);shutil.copyfile(p,saved);prior[p]=saved
    env={k:v for k,v in os.environ.items() if not k.startswith('BRASS_')}
    env.update(BRASS_SEARCH_V2_RESULT=str(target/'result.json'),BRASS_SEARCH_V2_EVENTS=str(STUDY/'reported-events.json'),
        BRASS_SEARCH_V2_MS=str(allowance),BRASS_SEARCH_V2_SCENARIO=scenario,BRASS_SEARCH_K='16',BRASS_NN_TEMPERATURE='1.0',BRASS_NN_MODEL='beta',
        BRASS_SEARCH_HOLD_UNTIL_MS=str(hold))
    try:
        shutil.copyfile(dll,touched[0]);write(touched[1],dict(passed=False))
        p=subprocess.run([str(game/'BrassBirminghamUnity.exe'),'-batchmode','-screen-fullscreen','0','-screen-width','1280','-screen-height','720',
            '-logFile',str(target/'unity.log'),'--brass-local-ai-test','--brass-native-matched-test','--brass-search-v2-test'],
            cwd=game,env=env,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
        shutil.copyfile(touched[1],target/'unity.json')
        u=read(target/'unity.json');write(target/'test-build.json',dict(files=[dict(name=n,sha256=sha(managed/n)) for n in NAMES],sources=pins(target/'test-source')))
        print(json.dumps(dict(label=label,passed=u['passed'],error=u.get('error'),checks=u.get('checks'))),flush=True)
        assert p.returncode==0 and u['passed'] and not u['nativeErrors']
    finally:
        for p in touched:
            if p in prior:shutil.copyfile(prior[p],p)
            elif p.exists():p.unlink()

def qualify():
    m=read(OUT/'build-manifest.json');old=read(BASE/'build-manifest.json')
    previous={p['path']:p['sha256'] for p in old['sourcePins']}
    assert {p['path'] for p in m['sourcePins'] if previous.get(p['path'])!=p['sha256']}==set(SCOPE)
    for name in SCOPE:assert sha(ROOT/name)==sha(OUT/'runtime'/name)
    assert pins(ROOT/'BrassNative')==m['nativeFiles'] and H['protected']()==read(STUDY/'protected-before.json')
    evidence=[]
    for label in ('candidate-4s','candidate-12s','candidate-cancel','candidate-expired','candidate-load'):
        u=read(STUDY/label/'unity.json');t=read(STUDY/label/'test-build.json')
        assert u['passed'] and not u['nativeErrors'] and t['files']==m['files']
        evidence.append(dict(label=label,sha256=sha(STUDY/label/'unity.json'),checks=u['checks']))
    for name in ('validation-events','checkpoint-tests'):
        assert read(STUDY/'contracts'/(name+'.json'))['passed']
    write(OUT/'qualification.json',dict(format='brass-nn-delivery-release/1.0.0',passed=True,
        build_manifest_sha256=sha(OUT/'build-manifest.json'),evidence=evidence,native_unchanged=True))
    print('Qualified delivery patch',flush=True)

def upgrade():
    fixture=STUDY/'install-rehearsal';assert not fixture.exists()
    managed=fixture/'BrassBirminghamUnity_Data/Managed';managed.mkdir(parents=True)
    for n in NAMES:shutil.copyfile(BASE/'Managed'/n,managed/n)
    shutil.copytree(ROOT/'BrassNative',fixture/'BrassNative')
    prior=read(ROOT/'artifacts/patch-installation.json');prior['gameRoot']=str(fixture);prior['saveDataPaths']=[str(fixture/'saves')]
    record=fixture/'artifacts/patch-installation.json';record.parent.mkdir();write(record,prior);original=sha(record)
    command=['pwsh','-NoProfile','-File',str(ROOT/'scripts/install-nn-delivery.ps1'),'-GameRoot',str(fixture)]
    for action,extra,success in [('Install',[],True),('Rollback',[],True),('Install',['-TestInterrupt'],False),('Recover',[],True)]:
        p=subprocess.run(command+['-Action',action]+extra,capture_output=True,text=True)
        (fixture/(action+('-interrupt' if extra else '')+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
        assert (p.returncode==0)==success,p.stdout+p.stderr
        if not success:assert 'TEST INTERRUPT' in p.stderr
        if action in ('Rollback','Recover'):
            assert sha(record)==original
            for f in prior['files']:assert sha(managed/f['name'])==f['sha256']
    write(STUDY/'upgrade-test.json',dict(passed=True,install_rollback=True,interrupted_recovery=True,
        installer_sha256=sha(ROOT/'scripts/install-nn-delivery.ps1')))
    print('Installer and recovery passed',flush=True)

def audit():
    m=read(OUT/'build-manifest.json');record=ROOT/'artifacts/patch-installation.json';i=read(record)
    assert i['version']==m['version'] and i['files']==m['files']
    for f in m['files']:assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/f['name'])==f['sha256']
    assert pins(ROOT/'BrassNative')==m['nativeFiles'] and H['protected']()==read(STUDY/'protected-before.json')
    u=read(STUDY/'installed-12s'/'unity.json');r=read(STUDY/'installed-12s'/'result.json')
    assert u['passed'] and not u['nativeErrors'] and r['passed'] and r['commits'][-1]['remaining']==0 and r['turn_clock_ms']<13000
    assert read(STUDY/'installed-12s'/'test-build.json')['files']==m['files']
    verification=STUDY/'installed-verification.json'
    write(verification,dict(passed=True,version=i['version'],checks=u['checks'],native_unchanged=True,user_saves_unchanged=True,
        delivery_tail_ms=r['delivery_tail_after_resume_ms'],turn_clock_ms=r['turn_clock_ms'],result_sha256=sha(STUDY/'installed-12s'/'result.json')))
    i.update(nnDeliveryInstalledSmoke='passed/Beta-reported-position-two-actions-save-load',nnDeliveryVerification=str(verification),nnDeliveryVerificationSha256=sha(verification))
    temporary=record.with_suffix('.nn-delivery.tmp');write(temporary,i);temporary.replace(record)
    print(json.dumps(read(verification)),flush=True)

if __name__=='__main__':
    if sys.argv[1] in ('build','qualify','upgrade','audit'):globals()[sys.argv[1]]()
    else:smoke(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 4000,int(sys.argv[3]) if len(sys.argv)>3 else 0,sys.argv[4] if len(sys.argv)>4 else 'normal')
