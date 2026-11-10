"""Build, qualify and audit beta-2 without changing existing NN packages."""
from pathlib import Path
import argparse
import json
import os
import runpy
import shutil
import subprocess
import zipfile

H = runpy.run_path(str(Path(__file__).with_name('build-nn-beta2.py')))
ROOT, STUDY, OUT, BASE, BUNDLE, MODEL, CHECKPOINT, BACKEND, PROFILE, DIRECTORY, VERSION, SCOPE, NATIVE_SCOPE, POLICIES = (
    H[k] for k in ('ROOT','STUDY','OUT','BASE','BUNDLE','MODEL','CHECKPOINT','BACKEND','PROFILE','DIRECTORY','VERSION','SCOPE','NATIVE_SCOPE','POLICIES'))
sha, read, write, pins, compile_cs = (H[k] for k in ('sha','read','write','pins','compile_cs'))
NAMES = ('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')


def native():
    out = STUDY/'native-build'
    assert not out.exists(), 'Preserve native evidence'
    out.mkdir(parents=True)
    support = runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
    env, cargo = support['tool_environment']()
    checkout = out/'n'
    original = ROOT/'BrassNative/il-v11-2/native-source.zip'
    with zipfile.ZipFile(original) as z:
        assert all((checkout/n).resolve().is_relative_to(checkout.resolve()) for n in z.namelist())
        z.extractall(checkout)
    formatter = ROOT/'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin/rustfmt.exe'
    for name in NATIVE_SCOPE:
        shutil.copyfile(ROOT/name, checkout/name)
        if name.endswith('.rs'):
            subprocess.run([str(formatter),'--edition','2021','--config','skip_children=true',str(checkout/name)],env=env,check=True)
    env.update(CARGO_TARGET_DIR=str(ROOT/'artifacts/nn-temperature/target'), CARGO_INCREMENTAL='0')
    checks=[]
    for label, args in (
        ('build',['build','--release','--bin','brass-ai-local','--features','local-beta2']),
        ('worker-tests',['test','--release','-p','brass-ai-worker','--lib','--features','local-beta2'])):
        print(label,flush=True)
        log=out/(label+'.log')
        with log.open('wb') as f:
            subprocess.run([str(cargo),*args,'--offline','--locked'],cwd=checkout/'native',env=env,
                stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        checks.append(dict(stage=label,passed=True,log_sha256=sha(log)))
        print(log.read_text()[-1500:],flush=True)
    support['native_fingerprint'].__globals__['NATIVE']=checkout/'native'
    identity=support['native_fingerprint']()
    shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'x86_64-pc-windows-gnu/release/brass-ai-local.exe',out/'brass-ai-local.exe')
    with zipfile.ZipFile(out/'native-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(checkout.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(checkout).as_posix())
    with zipfile.ZipFile(original) as before, zipfile.ZipFile(out/'native-source.zip') as after:
        a,b={n:before.read(n) for n in before.namelist()},{n:after.read(n) for n in after.namelist()}
        changed=sorted(n for n in a.keys()|b.keys() if a.get(n)!=b.get(n))
        assert changed==sorted(NATIVE_SCOPE), changed
    write(out/'native-build.json',dict(passed=True,source_sha256=identity,scope=NATIVE_SCOPE,checks=checks,
        binary_sha256=sha(out/'brass-ai-local.exe'),source_archive_sha256=sha(out/'native-source.zip'),
        base_archive_sha256=sha(original),feature='local-beta2'))


def package():
    assert not OUT.exists(), 'Preserve candidate'
    installed=read(ROOT/'artifacts/patch-installation.json')
    base=read(BASE/'build-manifest.json')
    assert installed['version']=='1.4.54-rust-local'
    for f in installed['files']:
        assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/f['name'])==f['sha256'].lower()==sha(BASE/'Managed'/f['name'])
    model=read(STUDY/'model/manifest.json');n=read(STUDY/'native-build/native-build.json')
    assert n['passed'] and model['model_id']==MODEL and model['checkpoint_sha256']==CHECKPOINT
    for p in base['sourcePins']:assert sha(BASE/'runtime'/p['path'])==p['sha256']
    for directory in ('runtime','Managed'):shutil.copytree(BASE/directory,OUT/directory)
    shutil.copytree(ROOT/'BrassNative',OUT/'BrassNative')
    for name in SCOPE:shutil.copyfile(ROOT/name,OUT/'runtime'/name)
    # Reuse the exact last managed build's response file, with its references
    # redirected to the current managed/source snapshot (including tracker fix).
    rsp_base=ROOT/'artifacts/nn-delivery/20261105/release-1.4.52'
    rsp=(rsp_base/'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(rsp_base),str(OUT))
    (OUT/'legacy.rsp').write_text(rsp,encoding='utf-8-sig')
    subprocess.run([str(H['HELPERS']['CSC']),'@'+str(OUT/'legacy.rsp')],check=True)
    package=OUT/'BrassNative'/DIRECTORY;package.mkdir()
    # All existing search settings come from the installed package, including c2=1.
    template=ROOT/'BrassNative/alpha-g890'
    for name in POLICIES:shutil.copyfile(template/name,package/name)
    for name in ('brass-ai-local.exe','native-source.zip'):shutil.copyfile(STUDY/'native-build'/name,package/name)
    shutil.copytree(STUDY/'model',package/'model')
    shutil.copyfile(BUNDLE/'manifest.json',package/'model/checkpoint-manifest.json')
    for name in ('LICENSE','NOTICE','THIRD_PARTY_NOTICES.txt'):
        if (template/name).is_file():shutil.copyfile(template/name,package/name)
    warm=read(template/'warmup.json');reward=read(STUDY/'model/export-receipt.json')['reward']
    warm['context'].update(model_id=MODEL,profile_id=PROFILE,reward_id=model['reward_id'],
        max_cost=reward['insolvency']['max_cost'],remaining=[reward['insolvency']['max_cost']]*4,style=[0.]*4,intent=[0.]*8)
    assert warm['ledger']['spent']==[0.]*4, 'Warmup needs exact debt conversion'
    warm['reward']=reward
    write(package/'warmup.json',warm)
    acceptance=dict(version='brass-local-utility-acceptance/1.0.0',scope='part1-local-beta2-test',decision='user-2026-11-10',
        model_id=MODEL,export_sha256=model['export_sha256'],checkpoint_sha256=CHECKPOINT,utility_calibration=None,final_model=False)
    for name in POLICIES:
        p=read(package/name)
        p.update(backend=BACKEND,model_manifest_sha256=sha(package/'model/manifest.json'),acceptance=acceptance,
                 warmup_sha256=sha(package/'warmup.json'))
        assert p['method']=='search_v1' and p['config']['c2']==1.0
        write(package/name,p)
    build=read(template/'build.json');build.update(backend=BACKEND,scope='local-beta2-test',source_sha256=n['source_sha256'])
    for p in build['files']:p['sha256']=sha(package/p['path'])
    write(package/'build.json',build)
    (package/'README.txt').write_text('beta-2: exact selected g1190 checkpoint. search_v1, ONNX Runtime CPU, two one-thread sessions.\n'
        'Original policy/utility weights and reward retained; no fitting or new calibration receipt.\n'
        'Reward: win=1, placement=.1, VP=.001 cap180; insolvency=.0025 cap.025.\n'
        'Sources: native-source.zip and model/export-source.zip. Existing dependency notices apply.\n'
        'Rebuild native/: cargo build --offline --locked --release --bin brass-ai-local --features local-beta2\n',encoding='utf-8')
    write(OUT/'build-manifest.json',dict(version=VERSION,baseVersion=installed['version'],stockSha256=installed['stockSha256'],
        baseFiles=installed['files'],baseNative=pins(ROOT/'BrassNative'),sourceScope=SCOPE,sourcePins=pins(OUT/'runtime'),
        files=[dict(name=n,sha256=sha(OUT/'Managed'/n)) for n in NAMES],nativeFiles=pins(OUT/'BrassNative'),
        installed_baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),
        beta2_model_id=MODEL,beta2_checkpoint_sha256=CHECKPOINT,beta2_export_sha256=model['export_sha256'],
        beta2_native_build_sha256=sha(package/'build.json'),beta2_reward=reward,
        nn_ui_choices=['alpha-test','beta','il','beta-2'],search_method='search_v1',v205_enabled=False))
    write(STUDY/'protected-before.json',H['HELPERS']['protected']())
    shutil.copyfile(ROOT/'artifacts/patch-installation.json',STUDY/'before-installation.json')
    print(OUT,flush=True)


def smoke(label):
    target=STUDY/label;assert not target.exists(),'Preserve smoke evidence'
    target.mkdir()
    installed=label.startswith('installed');ui='ui-' in label
    game=ROOT if installed else ROOT/'artifacts/card-tracker-player'
    managed=game/'BrassBirminghamUnity_Data/Managed';manifest=read(OUT/'build-manifest.json')
    if not installed:
        for n in NAMES:shutil.copyfile(OUT/'Managed'/n,managed/n)
        if pins(game/'BrassNative')!=manifest['nativeFiles']:
            current=(game/'BrassNative').resolve()
            retained=(game/'artifacts'/('native-retained-beta2-'+label)).resolve()
            assert current.is_relative_to(game.resolve()) and retained.is_relative_to(game.resolve()) and not retained.exists()
            if current.exists():current.rename(retained)
            shutil.copytree(OUT/'BrassNative',current)
    assert pins(game/'BrassNative')==manifest['nativeFiles']
    tests=['tests/BrassLocalAI/LocalAiUnityTests.cs','tests/BrassLocalAI/AiExecutionResourceTests.cs']+[
        'tests/RustBackendLocal/'+n+'.cs' for n in ('NativeLocalUnityTests','NativeMatchedUnityTests','NativeValidationLifecycleTests',
                                                  'NativeReleaseUnityTests','ReleaseSegmentUnityTests','SearchV2UnityTests')]
    if ui:tests+=['tests/BrassLocalAI/MixedAiUnityTests.cs','tests/BrassLocalAI/Beta2UiUnityTests.cs']
    sources=[]
    for name in tests:
        dest=target/'test-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);sources.append(dest)
    dll=target/'LocalAiUnityTests.dll'
    compile_cs(dll,sources,[p for p in managed.glob('*.dll') if 'Tests' not in p.name and 'Regression' not in p.name],['/nostdlib+'])
    staging=game/'artifacts/patch-build';touched=[staging/'Managed/LocalAiUnityTests.dll',staging/'local-ai-native.json',staging/'local-ai-cold-menu.png']
    prior={}
    for p in touched:
        if p.exists():saved=target/('prior-'+p.name);shutil.copyfile(p,saved);prior[p]=saved
    env={k:v for k,v in os.environ.items() if not k.startswith('BRASS_')}
    args=['--brass-local-ai-test']
    if ui:
        env.update(BRASS_BETA_UI_TEST='read' if label.endswith('read') else 'write',BRASS_BETA2_PREFERENCES=str(STUDY/'preferences.json'))
    else:
        scenario='cancel' if 'cancel' in label else 'load' if 'load' in label else 'normal'
        selected='beta' if label=='live-old-beta' else 'beta-2'
        fixture='live-92781-RailEra-2.json' if installed else 'live-92781-CanalEra-2.json'
        env.update(BRASS_SEARCH_V2_RESULT=str(target/'result.json'),
            BRASS_SEARCH_V2_EVENTS=str(ROOT/'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures'/fixture),
            BRASS_SEARCH_V2_MS='4000',BRASS_SEARCH_V2_SCENARIO=scenario,BRASS_SEARCH_K='16',BRASS_NN_TEMPERATURE='1.0',
            BRASS_NN_MODEL=selected,BRASS_SEARCH_HOLD_UNTIL_MS='0')
        args+=['--brass-native-matched-test','--brass-search-v2-test']
    try:
        shutil.copyfile(dll,touched[0]);write(touched[1],dict(passed=False,status='beta2-running'))
        p=subprocess.run([str(game/'BrassBirminghamUnity.exe'),'-batchmode','-screen-fullscreen','0','-screen-width','1920',
            '-screen-height','1080','-logFile',str(target/'unity.log'),*args],cwd=game,env=env,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
        shutil.copyfile(touched[1],target/'unity.json');u=read(target/'unity.json')
        write(target/'test-build.json',dict(files=[dict(name=n,sha256=sha(managed/n)) for n in NAMES],
            candidateManifestSha256=sha(OUT/'build-manifest.json'),sources=pins(target/'test-source'),testDllSha256=sha(dll)))
        print(json.dumps(dict(label=label,passed=u['passed'],checks=u['checks'],error=u.get('error'))),flush=True)
        assert p.returncode==0 and u['passed'] and not u['nativeErrors']
        if not ui:
            r=read(target/'result.json');assert r['passed'] and r['nn_model']==selected and r['search_method']=='search_v1'
            expected=MODEL if selected=='beta-2' else read(ROOT/'BrassNative/beta-g1290/model/manifest.json')['model_id']
            assert all(x['model_id']==expected and x['sessions']==2 and x['threads_per_session']==1 for x in r['replies'])
            if scenario=='normal':assert len(r['commits'])==2 and r['commits'][-1]['remaining']==0
    finally:
        for p in touched:
            if p in prior:shutil.copyfile(prior[p],p)
            elif p.exists():p.unlink()


def qualify():
    assert not (OUT/'qualification.json').exists()
    m=read(OUT/'build-manifest.json');base=read(BASE/'build-manifest.json')
    previous={p['path']:p['sha256'] for p in base['sourcePins']}
    assert {p['path'] for p in m['sourcePins'] if previous.get(p['path'])!=p['sha256']}==set(SCOPE)
    for p in m['sourcePins']:assert sha(OUT/'runtime'/p['path'])==p['sha256']
    assert pins(OUT/'BrassNative')==m['nativeFiles']
    for p in m['baseNative']:assert p in m['nativeFiles'] and sha(ROOT/'BrassNative'/p['path'])==p['sha256']
    assert sha(BUNDLE/'checkpoint.pt')==CHECKPOINT and read(STUDY/'model/export-receipt.json')['passed']
    evidence=[]
    for label in ('ui-write','ui-read','live-beta2','cancel-beta2','load-beta2','live-old-beta'):
        u=read(STUDY/label/'unity.json');t=read(STUDY/label/'test-build.json')
        assert u['passed'] and not u['nativeErrors'] and t['candidateManifestSha256']==sha(OUT/'build-manifest.json')
        evidence.append(dict(label=label,checks=u['checks'],sha256=sha(STUDY/label/'unity.json')))
    worker=runpy.run_path(str(ROOT/'scripts/build-nn-c-scale.py'))['worker_readiness']
    readiness=[]
    for policy in POLICIES:
        ready=worker(OUT/'BrassNative'/DIRECTORY/policy)
        assert ready['model_id']==MODEL and ready['sessions']==2 and ready['threads_per_session']==1
        readiness.append(dict(policy=policy,ready=ready))
    assert H['HELPERS']['protected']()==read(STUDY/'protected-before.json')
    write(STUDY/'readiness.json',readiness)
    (OUT/'evidence').mkdir()
    shutil.copyfile(ROOT/'artifacts/card-tracker-player/artifacts/patch-build/beta2-1080-largest.png',OUT/'evidence/beta2-1080-largest.png')
    write(OUT/'qualification.json',dict(format='brass-nn-beta2-release/1.0.0',passed=True,version=VERSION,
        build_manifest_sha256=sha(OUT/'build-manifest.json'),beta2_native_build_receipt_sha256=sha(STUDY/'native-build/native-build.json'),
        export_receipt_sha256=sha(STUDY/'model/export-receipt.json'),beta2_model_id=MODEL,beta2_checkpoint_sha256=CHECKPOINT,
        evidence=evidence,readiness_sha256=sha(STUDY/'readiness.json'),files=pins(OUT),
        scope='Short functional checks, exact checkpoint/reward identity, parity, UI and save/load; no strength claim.'))
    print('Qualified beta-2',flush=True)


def upgrade():
    fixture=STUDY/'install-tests/recovery';assert not fixture.exists()
    managed=fixture/'BrassBirminghamUnity_Data/Managed';managed.mkdir(parents=True)
    for name in NAMES:shutil.copyfile(ROOT/'BrassBirminghamUnity_Data/Managed'/name,managed/name)
    shutil.copytree(ROOT/'BrassNative',fixture/'BrassNative')
    prior=read(ROOT/'artifacts/patch-installation.json');assert prior['version']=='1.4.54-rust-local'
    prior.update(gameRoot=str(fixture),saveDataPaths=[str(fixture/'saves')])
    record=fixture/'artifacts/patch-installation.json';record.parent.mkdir();write(record,prior);original=sha(record)
    command=['pwsh','-NoProfile','-File',str(ROOT/'scripts/install-nn-beta2.ps1'),'-GameRoot',str(fixture)]
    for action,extra,success in [('Install',[],True),('Rollback',[],True),('Install',['-TestInterrupt'],False),('Recover',[],True)]:
        p=subprocess.run(command+['-Action',action]+extra,capture_output=True,text=True)
        (fixture/(action+('-interrupt' if extra else '')+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
        assert (p.returncode==0)==success,p.stdout+p.stderr
        if not success:assert 'TEST INTERRUPT' in p.stderr
        if action in ('Rollback','Recover'):
            assert sha(record)==original
            for f in prior['files']:assert sha(managed/f['name'])==f['sha256'].lower()
            assert pins(fixture/'BrassNative')==read(OUT/'build-manifest.json')['baseNative']
    write(STUDY/'upgrade-test.json',dict(passed=True,install_rollback=True,interrupted_recovery=True,
        installer_sha256=sha(ROOT/'scripts/install-nn-beta2.ps1')))
    print('Install, rollback and interrupted recovery passed',flush=True)


def audit():
    m=read(OUT/'build-manifest.json');record=ROOT/'artifacts/patch-installation.json';installed=read(record)
    assert installed['version']==VERSION and installed['files']==m['files']
    for p in m['files']:assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/p['name'])==p['sha256']
    assert pins(ROOT/'BrassNative')==m['nativeFiles']
    assert H['HELPERS']['protected']()==read(STUDY/'protected-before.json')
    checks=0
    for label in ('installed-beta2','installed-ui-read'):
        u=read(STUDY/label/'unity.json');t=read(STUDY/label/'test-build.json')
        assert u['passed'] and not u['nativeErrors'] and t['files']==m['files'];checks+=u['checks']
    result=read(STUDY/'installed-beta2/result.json')
    assert result['commits'][-1]['remaining']==0 and result['nn_model']=='beta-2'
    verification=STUDY/'installed-verification.json'
    write(verification,dict(passed=True,version=VERSION,model_id=MODEL,checkpoint_sha256=CHECKPOINT,
        export_sha256=m['beta2_export_sha256'],checks=checks,allowance_ms=4000,search_k=16,
        turn_clock_ms=result['turn_clock_ms'],native_files=len(m['nativeFiles']),
        prior_packages_unchanged=True,user_data_unchanged=True,qualification_sha256=sha(OUT/'qualification.json'),
        upgrade_test_sha256=sha(STUDY/'upgrade-test.json'),result_sha256=sha(STUDY/'installed-beta2/result.json')))
    # The installer owns atomic receipt publication; final metadata uses its helper too.
    command=". './scripts/patch-install-common.ps1'\n$r=Read-BrassJson './artifacts/patch-installation.json'\n"
    command+="$r|Add-Member -NotePropertyName nnBeta2InstalledSmoke -NotePropertyValue 'passed/4s-K16/turn-save-load-and-ui' -Force\n"
    command+="$r|Add-Member -NotePropertyName nnBeta2InstalledVerification -NotePropertyValue 'artifacts/nn-beta2/20261110/installed-verification.json' -Force\n"
    command+="Write-BrassJsonAtomic './artifacts/patch-installation.json' $r"
    subprocess.run(['pwsh','-NoProfile','-Command',command],cwd=ROOT,check=True)
    print(json.dumps(read(verification)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('native','package','ui-write','ui-read','live-beta2','cancel-beta2','load-beta2','live-old-beta','qualify','upgrade','audit','installed-beta2','installed-ui-read'))
    args=parser.parse_args();STUDY.mkdir(parents=True,exist_ok=True)
    if args.stage in ('native','package','qualify','upgrade','audit'):globals()[args.stage]()
    else:smoke(args.stage)
