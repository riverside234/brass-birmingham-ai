"""Export and qualify the exact g1290 test NN as the local Beta option."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/nn-beta/20261101'
BASE=ROOT/'artifacts/nn-neutral-ui/20261031/release-1.4.47'
OUT=STUDY/'release-1.4.48'
BUNDLE=ROOT/'test_nn/joint7-v11-2-grouped-lr-critic-1b96e228-g1290-ed8edb039a12-d50f4bc95d51'
MODEL='73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff'
CHECKPOINT='ed8edb039a1257d591cc0d2e2b87eb7a2d7aa411ce0f7710a2d7f8ad487dc3b0'
CALIBRATION='f56bd03a91e661c5be0f941765ec9420c4729ef6caca0f8d294d61b3d3550670'
H=runpy.run_path(str(ROOT/'scripts/build-nn-search-choice.py'))
sha,read,write,pins=(H[k] for k in ('sha','read','write','pins'))
SCOPE=['src/BrassLocalAI/Policy/AiSelectionCatalog.cs','src/BrassLocalAI/Policy/LocalSetupPreferences.cs',
       'src/BrassLocalAI/Unity/LocalMenu.cs','src/BrassLocalAI/Unity/AiSelectionPanel.cs']
SCOPE+=['src/BrassLocalAI/Native/'+n+'.cs' for n in ('NativeModel','NativePackage','NativeProcess','NativeGameState','NativeInput','NativeValidation')]


def export():
    sys.path[:0]=[str(ROOT/'src')]+[str(ROOT/p) for p in ('artifacts/r2-pythondeps','artifacts/r4-pythondeps',
        'artifacts/r0-pythonlibs','artifacts/r0-validation-deps','artifacts/onnx-export-deps')]
    import torch
    import onnxruntime as ort
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.personality.laptop.export import export as onnx_export
    from brass_ai.native.features import SCHEMA,GROUP,ENTITY
    model_dir=STUDY/'model';assert not model_dir.exists(),'Preserve model export'
    assert sha(BUNDLE/'checkpoint.pt')==CHECKPOINT
    assert read(BUNDLE/'manifest.json')['derived_checkpoint']['generation']==1290
    assert ort.__version__=='1.22.1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,raw=load_checkpoint(BUNDLE/'checkpoint.pt')
    calibration=read(BUNDLE/'calibration.json')
    assert raw['model_id']==MODEL and model.utility_calibration==calibration['calibration_receipt']
    assert model.utility_calibration['request_id']==CALIBRATION and model.utility_calibration['weights_unchanged']
    assert model.utility_calibration['ready'] and model.utility_calibration['steps']==0
    model_dir.mkdir(parents=True)
    sources=[p for folder in ('src/brass_ai/learning','src/brass_ai/personality/laptop','src/brass_ai/native')
             for p in sorted((ROOT/folder).rglob('*.py'))]+[Path(__file__).resolve()]
    source_pins=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sources]
    start=time.perf_counter();meta=onnx_export(model.eval(),model_dir/'model.onnx')
    assert meta['model_id']==MODEL and meta['parity']['passed']
    for p in source_pins:assert sha(ROOT/p['path'])==p['sha256'],'Export source changed during execution'
    with zipfile.ZipFile(model_dir/'export-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in source_pins:z.write(ROOT/p['path'],p['path'])
    runtime=Path(ort.__file__).parent/'capi/onnxruntime.dll'
    shutil.copyfile(runtime,model_dir/'onnxruntime.dll')
    shutil.copyfile(BUNDLE/'calibration.json',model_dir/'calibration.json')
    manifest=dict(format='brass-native-model/1.0.0',model_id=MODEL,checkpoint_sha256=CHECKPOINT,
        export_sha256=sha(model_dir/'model.onnx'),export_file='model.onnx',metadata_sha256=sha(model_dir/'model.json'),
        runtime_file='onnxruntime.dll',runtime_sha256=sha(runtime),runtime_version=ort.__version__,binding_version='2.0.0-rc.10',
        architecture=meta['config']['architecture'],feature_schema=SCHEMA,group_layout=GROUP,entity_layout=ENTITY,
        reward_id=meta['reward_id'],max_cost=meta['config']['utility']['insolvency']['max_cost'],precision='float32',opset=17,
        inputs=meta['inputs'],outputs=meta['outputs'],mechanical_only=False,utility_calibration=model.utility_calibration,
        intra_threads=1,inter_threads=1,sequential=True,provider='CPUExecutionProvider',optimization='all-exact-gelu',
        source_archive_sha256=sha(model_dir/'export-source.zip'))
    write(model_dir/'manifest.json',manifest)
    assert sha(BUNDLE/'checkpoint.pt')==CHECKPOINT
    write(model_dir/'export-receipt.json',dict(passed=True,generation=1290,model_id=MODEL,checkpoint_sha256=CHECKPOINT,
        export_sha256=manifest['export_sha256'],manifest_sha256=sha(model_dir/'manifest.json'),parity=meta['parity'],
        checkpoint_unchanged=True,seconds=time.perf_counter()-start,parameters=model.parameter_count,source_pins=source_pins))
    print(json.dumps(dict(passed=True,model_id=MODEL,export_sha256=manifest['export_sha256'],parity=meta['parity'])),flush=True)


def native():
    out=STUDY/'native-build';assert not out.exists(),'Preserve native build'
    out.mkdir(parents=True)
    support=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'));env,cargo=support['tool_environment']()
    scope=['native/brass-ai-worker/Cargo.toml','native/brass-ai-worker/src/lib.rs',
           'native/brass-ai-worker/src/local.rs','native/brass-ai-worker/src/local_model.rs',
           'native/brass-ai-inference/src/manifest.rs']
    formatter=ROOT/'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin/rustfmt.exe'
    checkout=out/'n'
    with zipfile.ZipFile(ROOT/'BrassNative/alpha-g890/native-source.zip') as z:
        assert all((checkout/n).resolve().is_relative_to(checkout.resolve()) for n in z.namelist());z.extractall(checkout)
    for name in scope:shutil.copyfile(ROOT/name,checkout/name)
    for name in scope:
        if name.endswith('.rs'):subprocess.run([str(formatter),'--edition','2021','--config','skip_children=true',str(checkout/name)],env=env,check=True)
    env.update(CARGO_TARGET_DIR=str(ROOT/'artifacts/nn-temperature/target'),CARGO_INCREMENTAL='0')
    stages=[]
    for label,args in [('build',['build','--release','--bin','brass-ai-local','--features','local-beta']),
                       ('beta-worker-tests',['test','--release','-p','brass-ai-worker','--lib','--features','local-beta']),
                       ('inference-contract-tests',['test','--release','-p','brass-ai-inference','--test','contracts'])]:
        print(label,flush=True)
        log=out/(label+'.log')
        with log.open('wb') as f:
            subprocess.run([str(cargo),*args,'--offline','--locked'],cwd=checkout/'native',env=env,
                stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        print(log.read_text()[-1800:],flush=True);stages.append(dict(stage=label,passed=True,log_sha256=sha(log)))
    support['native_fingerprint'].__globals__['NATIVE']=checkout/'native';identity=support['native_fingerprint']()
    shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'x86_64-pc-windows-gnu/release/brass-ai-local.exe',out/'brass-ai-local.exe')
    with zipfile.ZipFile(out/'native-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(checkout.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(checkout).as_posix())
    write(out/'native-build.json',dict(passed=True,source_sha256=identity,scope=scope,checks=stages,
        binary_sha256=sha(out/'brass-ai-local.exe'),source_archive_sha256=sha(out/'native-source.zip'),feature='local-beta'))


def package():
    assert not OUT.exists(),'Preserve candidate'
    installed=read(ROOT/'artifacts/patch-installation.json');old=read(BASE/'build-manifest.json')
    assert installed['version']==old['version']=='1.4.47-rust-local' and read(BASE/'qualification.json')['passed']
    assert pins(ROOT/'BrassNative')==old['nativeFiles']
    for p in old['files']:assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/p['name'])==p['sha256']
    n=read(STUDY/'native-build/native-build.json');model=read(STUDY/'model/manifest.json');assert n['passed']
    assert model['model_id']==MODEL and model['checkpoint_sha256']==CHECKPOINT
    for folder in ('runtime','Managed','BrassNative'):shutil.copytree(BASE/folder,OUT/folder)
    for name in SCOPE:shutil.copyfile(ROOT/name,OUT/'runtime'/name)
    rsp=OUT/'legacy.rsp';rsp.write_text((BASE/'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(BASE),str(OUT))+
        '\n"'+str(OUT/'runtime/src/BrassLocalAI/Native/NativeModel.cs')+'"\n',encoding='utf-8-sig')
    subprocess.run([str(ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'),'@'+str(rsp)],check=True)
    alpha=OUT/'BrassNative/alpha-g890';beta=OUT/'BrassNative/beta-g1290'
    shutil.copytree(alpha,beta)
    for name in ('brass-ai-local.exe','native-source.zip'):shutil.copyfile(STUDY/'native-build'/name,beta/name)
    for p in (STUDY/'model').iterdir():
        if p.is_file():shutil.copyfile(p,beta/'model'/p.name)
    shutil.copyfile(BUNDLE/'manifest.json',beta/'model/checkpoint-manifest.json')
    warm=read(beta/'warmup.json');warm['context'].update(model_id=MODEL,profile_id='beta-g1290-neutral/1.0.0',style=[0.]*4,intent=[0.]*8)
    write(beta/'warmup.json',warm)
    acceptance=dict(version='brass-local-utility-acceptance/1.0.0',scope='part1-local-beta-test',decision='user-2026-11-01',
        model_id=MODEL,export_sha256=model['export_sha256'],checkpoint_sha256=CHECKPOINT,utility_calibration=model['utility_calibration'],final_model=False)
    names=['policy.json','policy-k8.json','policy-k32.json','policy-t15.json','policy-k8-t15.json','policy-k32-t15.json']
    for name in names:
        policy=read(beta/name);policy.update(backend='brass-gumbel-is-mcts/beta-g1290-v1',
            model_manifest_sha256=sha(beta/'model/manifest.json'),acceptance=acceptance,warmup_sha256=sha(beta/'warmup.json'))
        assert policy['method']=='search_v1' and len(json.dumps(policy,indent=2).encode())<65536
        write(beta/name,policy)
    build=read(beta/'build.json');build.update(backend='brass-gumbel-is-mcts/beta-g1290-v1',scope='local-beta-test',source_sha256=n['source_sha256'])
    for p in build['files']:p['sha256']=sha(beta/p['path'])
    write(beta/'build.json',build)
    (beta/'README.txt').write_text('Beta g1290 test NN. Neutral context. search_v1, V205 disabled.\n'
        'Two independent ONNX Runtime CPU sessions; one compute thread per session.\n'
        'Verified checkpoint and read-only utility receipt: model/checkpoint-manifest.json and model/calibration.json.\n'
        'Build native/: cargo build --locked --release --bin brass-ai-local --features local-beta --target x86_64-pc-windows-gnu\n'
        'Corresponding sources: native-source.zip and model/export-source.zip. Existing dependency notices apply.\n',encoding='utf-8')
    old.update(version='1.4.48-rust-local',baseVersion=installed['version'],baseFiles=installed['files'],baseNative=pins(ROOT/'BrassNative'),
        sourceScope=SCOPE,sourcePins=pins(OUT/'runtime'),files=[dict(name=p['name'],sha256=sha(OUT/'Managed'/p['name'])) for p in old['files']],
        nativeFiles=pins(OUT/'BrassNative'),installed_baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),
        beta_model_id=MODEL,beta_checkpoint_sha256=CHECKPOINT,beta_export_sha256=model['export_sha256'],
        beta_native_source_sha256=n['source_sha256'],beta_native_build_sha256=sha(beta/'build.json'),
        beta_native_build_receipt_sha256=sha(STUDY/'native-build/native-build.json'),nn_ui_choices=['alpha-test','beta'],
        nn_ui_focuses=['neutral'],default_nn_focus='neutral',native_unchanged=False)
    write(OUT/'build-manifest.json',old);shutil.copyfile(ROOT/'artifacts/patch-installation.json',STUDY/'before-installation.json')
    write(STUDY/'user-data-before.json',dict(files=[dict(path=str(p),sha256=sha(p)) for directory in installed['saveDataPaths']
        for p in Path(directory).rglob('*') if p.is_file()]))
    print(OUT,flush=True)


def unity(label):
    d=STUDY/label;assert not d.exists(),'Preserve smoke evidence'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('BRASS_INDUSTRY_FOCUS_TEST',None);env.pop('BRASS_L14_TRACE',None)
    command=[shutil.which('pwsh'),'-NoProfile','-File',str(ROOT/'tests/RustBackendLocal/run-unity.ps1'),
        '-BuildDirectory',str(OUT.relative_to(ROOT)),'-EvidenceDirectory',str(d.relative_to(ROOT)),'-MaxSeconds','60']
    if label.startswith('ui-'):
        env['BRASS_BETA_UI_TEST']=label[3:];command+=['-BetaUi']
    elif label=='legacy-ui':
        env.pop('BRASS_BETA_UI_TEST',None);command+=['-SetupPreferences','write']
    else:
        model='alpha-test' if label=='live-alpha' else 'beta';scenario='cancel' if label=='cancel-beta' else 'normal'
        env.pop('BRASS_BETA_UI_TEST',None)
        env.update(BRASS_SEARCH_V2_RESULT=str(d/'result.json'),
            BRASS_SEARCH_V2_EVENTS=str(ROOT/'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json'),
            BRASS_SEARCH_V2_MS='4000',BRASS_SEARCH_K='16',BRASS_NN_TEMPERATURE='1.0',BRASS_NN_FOCUS='neutral',
            BRASS_NN_MODEL=model,BRASS_SEARCH_V2_SCENARIO=scenario,BRASS_SEARCH_HOLD_UNTIL_MS='0')
        command+=['-SearchV2']
    with (STUDY/(label+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=85)
    u=read(d/'unity.json');assert u['passed'] and not u['nativeErrors']
    if label in ('live-alpha','live-beta','cancel-beta'):
        r=read(d/'result.json');assert r['passed'] and r['nn_model']==model and r['search_method']=='search_v1'
        assert all(x['model_id']==(MODEL if model=='beta' else read(BASE/'BrassNative/alpha-g890/model/manifest.json')['model_id']) for x in r['replies'])
        if scenario=='normal':assert len(r['commits'])==2 and r['commits'][-1]['remaining']==0 and r['turn_clock_ms']<5000
        else:assert r['commits'][-1]['remaining']==1
    print(label,u['checks'],'checks',flush=True)


def qualify():
    assert not (OUT/'qualification.json').exists()
    m=read(OUT/'build-manifest.json');old=read(BASE/'build-manifest.json');previous={p['path']:p['sha256'] for p in old['sourcePins']}
    assert {p['path'] for p in m['sourcePins'] if previous.get(p['path'])!=p['sha256']}==set(SCOPE)
    for p in m['sourcePins']:assert sha(OUT/'runtime'/p['path'])==p['sha256']
    for p in SCOPE:assert (ROOT/p).read_bytes()==(OUT/'runtime'/p).read_bytes()
    for p in m['files']:
        assert sha(OUT/'Managed'/p['name'])==p['sha256']
        if p['name']!='BrassLegacyAI.dll':assert p in old['files']
    assert pins(OUT/'BrassNative')==m['nativeFiles']
    for p in m['baseNative']:assert p in m['nativeFiles'] and sha(ROOT/'BrassNative'/p['path'])==p['sha256']
    assert len(m['nativeFiles'])==len(m['baseNative'])+len([p for p in m['nativeFiles'] if p['path'].startswith('beta-g1290/')])
    assert sha(BUNDLE/'checkpoint.pt')==CHECKPOINT and read(STUDY/'model/export-receipt.json')['passed']
    assert read(OUT/'BrassNative/beta-g1290/model/manifest.json')['model_id']==MODEL
    native=read(STUDY/'native-build/native-build.json');assert native['passed'] and native['feature']=='local-beta'
    support=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'));env,_=support['tool_environment']()
    formatter=ROOT/'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin/rustfmt.exe'
    for name in native['scope']:
        archived=(STUDY/'native-build/n'/name).read_bytes()
        if name.endswith('.rs'):
            actual=subprocess.run([str(formatter),'--edition','2021','--emit','stdout','--config','skip_children=true'],
                input=(ROOT/name).read_bytes(),capture_output=True,env=env,check=True).stdout
        else:actual=(ROOT/name).read_bytes()
        assert actual.replace(b'\r\n',b'\n')==archived.replace(b'\r\n',b'\n'),'Native source differs: '+name
    evidence=[]
    for label in ('ui-write','ui-read','legacy-ui','live-beta','cancel-beta','live-alpha'):
        d=STUDY/label;t=read(d/'test-build.json');u=read(d/'unity.json')
        assert u['passed'] and not u['nativeErrors']
        assert t['candidateManifestSha256']==sha(OUT/'build-manifest.json') and t['testDllSha256']==sha(d/'LocalAiUnityTests.dll')
        for p in t['sources']:assert sha(d/'test-source'/p['path'])==p['sha256']
        evidence.append(dict(label=label,checks=u['checks'],unity_sha256=sha(d/'unity.json'),test_build_sha256=sha(d/'test-build.json')))
    images=OUT/'evidence';images.mkdir()
    for name in ('beta-1080-largest.png',):
        shutil.copyfile(ROOT/'artifacts/card-tracker-player/artifacts/patch-build'/name,images/name)
    write(OUT/'qualification.json',dict(format='brass-nn-beta-release/1.0.0',passed=True,version=m['version'],
        build_manifest_sha256=sha(OUT/'build-manifest.json'),beta_native_build_receipt_sha256=sha(STUDY/'native-build/native-build.json'),
        export_receipt_sha256=sha(STUDY/'model/export-receipt.json'),beta_model_id=MODEL,beta_checkpoint_sha256=CHECKPOINT,
        existing_alpha_package_unchanged=True,source_scope=SCOPE,evidence=evidence,files=pins(OUT),
        scope='Short local game functional qualification; no strength or long-session claim.'))
    print('Qualified exact g1290 Beta package; Alpha preserved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('export','native','package','ui-write','ui-read','legacy-ui','live-beta','live-alpha','cancel-beta','qualify'))
    stage=parser.parse_args().stage;STUDY.mkdir(parents=True,exist_ok=True)
    unity(stage) if stage.startswith(('ui-','live-','cancel-')) or stage=='legacy-ui' else globals()[stage]()
