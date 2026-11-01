"""Build and smoke a scoped local-game industry tag patch from installed sources.

No training/lab source or model is packaged. Existing release inputs are immutable.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import queue
import runpy
import shutil
import struct
import subprocess
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]
H = runpy.run_path(str(ROOT / 'scripts/build-nn-search-choice.py'))
sha, read, write, pins = (H[k] for k in ('sha', 'read', 'write', 'pins'))
STUDY = ROOT / 'artifacts/nn-industry-focus/20261031'
BASE = ROOT / 'artifacts/nn-search-ui/remove-k32/candidate'
OUT = STUDY / 'release-1.4.46'
NATIVE = STUDY / 'native-build'
SCOPE = ['src/BrassLocalAI/Native/' + n + '.cs' for n in
         ('NativeFocus', 'NativeInput', 'NativeGameState', 'NativeProcess', 'NativeValidation')]
SCOPE += ['src/BrassLocalAI/Policy/AiSelectionCatalog.cs', 'src/BrassLocalAI/Policy/LocalSetupPreferences.cs',
          'src/BrassLocalAI/Unity/LocalMenu.cs', 'src/BrassLocalAI/Unity/AiSelectionPanel.cs']


def native():
    assert not NATIVE.exists(), 'Preserve previous native build evidence'
    NATIVE.mkdir(parents=True)
    support = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
    env, cargo = support['tool_environment']()
    source = 'native/brass-ai-worker/src/local.rs'
    formatter = ROOT / 'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin/rustfmt.exe'
    subprocess.run([str(formatter), '--edition', '2021', '--config', 'skip_children=true', str(ROOT/source)], env=env, check=True)
    checkout = NATIVE / 'n'
    archive = ROOT / 'BrassNative/alpha-g890/native-source.zip'
    with zipfile.ZipFile(archive) as z:
        assert all((checkout/n).resolve().is_relative_to(checkout.resolve()) for n in z.namelist())
        z.extractall(checkout)
    shutil.copyfile(ROOT/source, checkout/source)
    env['CARGO_TARGET_DIR'] = str(ROOT / 'artifacts/nn-temperature/target')
    env['CARGO_INCREMENTAL'] = '0'
    records = []
    for label, args in [('build', ['build', '--release', '--bin', 'brass-ai-local']),
                        ('worker-tests', ['test', '--release', '-p', 'brass-ai-worker', '--lib'])]:
        log = NATIVE / (label+'.log')
        print(label, flush=True)
        with log.open('wb') as f:
            subprocess.run([str(cargo), *args, '--offline', '--locked'], cwd=checkout/'native', env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        print(log.read_text()[-2200:], flush=True)
        records.append(dict(stage=label, passed=True, log_sha256=sha(log)))
    support['native_fingerprint'].__globals__['NATIVE'] = checkout/'native'
    source_sha = support['native_fingerprint']()
    shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'x86_64-pc-windows-gnu/release/brass-ai-local.exe', NATIVE/'brass-ai-local.exe')
    with zipfile.ZipFile(NATIVE/'native-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(checkout.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(checkout).as_posix())
    write(NATIVE/'native-build.json', dict(passed=True, source_sha256=source_sha, scope=[source], checks=records,
          binary_sha256=sha(NATIVE/'brass-ai-local.exe'), source_archive_sha256=sha(NATIVE/'native-source.zip'),
          baseline_source_archive_sha256=sha(archive), toolchain='1.99.1', profile='release/locked/offline/x86-64'))


def package():
    assert not OUT.exists(), 'Preserve previous candidate'
    installed = read(ROOT/'artifacts/patch-installation.json')
    old = read(BASE/'build-manifest.json'); n = read(NATIVE/'native-build.json')
    assert installed['version']=='1.4.45-rust-local' and installed['searchMethod']=='search_v1' and not installed['v205Enabled']
    assert n['passed'] and pins(ROOT/'BrassNative')==old['nativeFiles']
    for p in installed['files']:
        assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/p['name'])==p['sha256'].lower()
        assert next(x['sha256'] for x in old['files'] if x['name']==p['name'])==p['sha256'].lower()
    for folder in ('runtime', 'Managed', 'BrassNative'): shutil.copytree(BASE/folder, OUT/folder)
    for name in SCOPE: shutil.copyfile(ROOT/name, OUT/'runtime'/name)
    rsp = OUT/'legacy.rsp'
    rsp.write_text((BASE/'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(BASE),str(OUT)) +
                   '\n"'+str(OUT/'runtime/src/BrassLocalAI/Native/NativeFocus.cs')+'"\n', encoding='utf-8-sig')
    subprocess.run([str(ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'), '@'+str(rsp)], check=True)
    pkg = OUT/'BrassNative/alpha-g890'
    for name in ('brass-ai-local.exe', 'native-source.zip'): shutil.copyfile(NATIVE/name, pkg/name)
    build = read(pkg/'build.json'); build['source_sha256'] = n['source_sha256']
    for p in build['files']:
        if p['path']=='brass-ai-local.exe': p['sha256'] = sha(pkg/p['path'])
        else: assert sha(pkg/p['path'])==p['sha256']
    write(pkg/'build.json',build)
    old.update(version='1.4.46-rust-local', baseVersion=installed['version'], sourceScope=SCOPE,
               sourcePins=pins(OUT/'runtime'), files=[dict(name=p['name'],sha256=sha(OUT/'Managed'/p['name'])) for p in old['files']],
               nativeFiles=pins(OUT/'BrassNative'), native_source_sha256=n['source_sha256'], native_build_sha256=sha(pkg/'build.json'),
               native_build_receipt_sha256=sha(NATIVE/'native-build.json'), installed_baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),
               baseFiles=installed['files'], baseNative=pins(ROOT/'BrassNative'),
               industry_focuses=['neutral','cotton','manufacturer','pottery','brewery','iron','coal'])
    write(OUT/'build-manifest.json',old)
    shutil.copyfile(ROOT/'artifacts/patch-installation.json', STUDY/'before-installation.json')
    print(OUT,flush=True)


def unity(label, *, phase=None, focus=None):
    destination = STUDY/label
    assert not destination.exists(), 'Preserve smoke evidence'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    command = [shutil.which('pwsh'), '-NoProfile', '-File', str(ROOT/'tests/RustBackendLocal/run-unity.ps1'),
               '-BuildDirectory', str(OUT.relative_to(ROOT)), '-EvidenceDirectory', str(destination.relative_to(ROOT)), '-MaxSeconds','60']
    if phase:
        env['BRASS_INDUSTRY_FOCUS_TEST'] = phase
        command += ['-IndustryFocus']
    else:
        env.update(BRASS_SEARCH_V2_RESULT=str(destination/'result.json'),
                   BRASS_SEARCH_V2_EVENTS=str(ROOT/'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json'),
                   BRASS_SEARCH_V2_MS='4000', BRASS_SEARCH_K='16', BRASS_NN_TEMPERATURE='1.0',
                   BRASS_NN_FOCUS=focus, BRASS_SEARCH_V2_SCENARIO='normal', BRASS_SEARCH_HOLD_UNTIL_MS='0')
        env.pop('BRASS_INDUSTRY_FOCUS_TEST',None); env.pop('BRASS_L14_TRACE',None)
        command += ['-SearchV2']
    with (STUDY/(label+'.log')).open('w',encoding='utf-8') as f:
        subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=85)
    u = read(destination/'unity.json'); assert u['passed'] and not u['nativeErrors']
    if focus:
        r = read(destination/'result.json')
        assert r['passed'] and r['industry_focus']==focus and r['search_method']=='search_v1'
        assert len(r['commits'])==2 and r['commits'][-1]['remaining']==0 and r['turn_clock_ms']<5000
        assert all(x['industry_focus']==focus and x['sessions']==2 and x['threads_per_session']==1 for x in r['replies'])
    print(label,u['checks'],'checks',flush=True)


def protocol():
    destination = STUDY/'protocol-smoke.json'
    assert not destination.exists()
    pkg = OUT/'BrassNative/alpha-g890'; template=read(pkg/'warmup.json')
    p = subprocess.Popen([str(pkg/'brass-ai-local.exe'),str(pkg/'policy.json'),sha(pkg/'policy.json')],cwd=pkg,
                         stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    messages = queue.Queue()
    def exact(n):
        chunks = bytearray()
        while len(chunks)<n:
            data=p.stdout.read(n-len(chunks))
            if not data: raise EOFError('Native worker closed its pipe')
            chunks.extend(data)
        return bytes(chunks)
    def reader():
        try:
            while True:
                size=struct.unpack('<I',exact(4))[0]
                assert 1<=size<=1024*1024
                messages.put(json.loads(exact(size)))
        except Exception as e: messages.put(e)
    threading.Thread(target=reader,daemon=True).start()
    sequence=0
    def receive():
        reply=messages.get(timeout=60)
        if isinstance(reply,Exception): raise reply
        return reply
    def send(operation):
        nonlocal sequence
        sequence+=1;data=json.dumps(dict(protocol='brass-local-native/1.0.0',id=sequence,request=operation),separators=(',',':')).encode()
        p.stdin.write(struct.pack('<I',len(data))+data);p.stdin.flush()
        r=receive();assert r['id']==sequence
        return r
    def request(focus,index):
        root=copy.deepcopy(template);i=['neutral','cotton','manufacturer','pottery','brewery','iron','coal'].index(focus)
        root['context']['profile_id']=f'alpha-test-g890-{focus}/1.0.0'
        root['context']['style']=[float(1<=i<=3),0.,float(i>=4),0.]
        root['context']['intent']=[float(i>0 and j==i-1) for j in range(8)]
        serialized=json.dumps(root,separators=(',',':'))
        return dict(op='search',ticket=dict(game_instance=f'focus-{index}',game_id=f'focus-{index}',generation=1,
                    revision=root['view']['context']['revision'],actor=root['view']['seat'],profile=root['context']['profile_id'],
                    root_sha256=hashlib.sha256(serialized.encode()).hexdigest(),source_sha256=read(pkg/'build.json')['source_sha256']),
                    allowance_ms=4000,remaining_ms=4000,input_json=serialized)
    rows=[]
    try:
        ready=receive();assert ready['status']=='ready'
        focuses=read(OUT/'build-manifest.json')['industry_focuses']
        assert ready['data']['industry_focuses']==focuses
        for index,focus in enumerate(focuses):
            r=send(request(focus,index));assert r['status']=='ok',r
            d=r['data'];assert d['industry_focus']==focus and d['completed']>0 and d['sessions']==2 and d['threads_per_session']==1
            assert d['response_ready_ns']<d['deadline_ns'] and d['requested_k']==16 and d['retired']
            rows.append(dict(focus=focus,completed=d['completed'],actions=d['actions'],profile=d['ticket']['profile']))
            print(focus,d['completed'],'completed simulations',flush=True)
        invalid=request('cotton',7);root=json.loads(invalid['input_json']);root['context']['intent']=[0.]*8
        invalid['input_json']=json.dumps(root,separators=(',',':'));invalid['ticket']['root_sha256']=hashlib.sha256(invalid['input_json'].encode()).hexdigest()
        assert send(invalid)['status']=='error','mismatched focus encoding must be rejected'
        invalid=request('cotton',8);invalid['ticket']['profile']='alpha-test-g890-coal/1.0.0'
        assert send(invalid)['status']=='error','mismatched ticket profile must be rejected'
        assert send(dict(op='shutdown'))['status']=='stopped'
        p.wait(timeout=10);assert p.returncode==0
        write(destination,dict(passed=True,search_ms=4000,search_k=16,rows=rows,negative_checks=2,
              build_manifest_sha256=sha(OUT/'build-manifest.json')))
    finally:
        if p.poll() is None:p.kill();p.wait(timeout=10)


def qualify():
    assert not (OUT/'qualification.json').exists()
    m = read(OUT/'build-manifest.json'); old = read(BASE/'build-manifest.json')
    before = {p['path']:p['sha256'] for p in old['sourcePins']}
    assert {p['path'] for p in m['sourcePins'] if before.get(p['path'])!=p['sha256']}==set(SCOPE)
    for p in m['sourcePins']: assert sha(OUT/'runtime'/p['path'])==p['sha256']
    assert pins(OUT/'BrassNative')==m['nativeFiles']
    for p in m['files']:
        assert sha(OUT/'Managed'/p['name'])==p['sha256']
        if p['name']!='BrassLegacyAI.dll': assert p in old['files']
    evidence = []
    matrix=read(STUDY/'protocol-smoke.json')
    assert matrix['passed'] and matrix['build_manifest_sha256']==sha(OUT/'build-manifest.json')
    assert [r['focus'] for r in matrix['rows']]==m['industry_focuses']
    for label in ('ui-write','ui-read','live-neutral','live-cotton','live-coal'):
        d=STUDY/label;t=read(d/'test-build.json');u=read(d/'unity.json')
        assert u['passed'] and not u['nativeErrors']
        assert t['candidateManifestSha256']==sha(OUT/'build-manifest.json') and t['testDllSha256']==sha(d/'LocalAiUnityTests.dll')
        for p in t['sources']: assert sha(d/'test-source'/p['path'])==p['sha256']
        evidence.append(dict(label=label,checks=u['checks'],unity_sha256=sha(d/'unity.json'),test_build_sha256=sha(d/'test-build.json')))
    write(OUT/'qualification.json',dict(format='brass-nn-industry-focus-release/1.0.0',passed=True,version=m['version'],
          build_manifest_sha256=sha(OUT/'build-manifest.json'),native_build_receipt_sha256=sha(NATIVE/'native-build.json'),
          evidence=evidence,protocol_smoke_sha256=sha(STUDY/'protocol-smoke.json'),search_method='search_v1',v205_enabled=False,model='installed-g890-test',
          industry_focuses=m['industry_focuses'],files=pins(OUT),limitations=['Short functional smokes; industry specialization/strength is not measured.']))
    print('Qualified local industry tag patch',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('native','package','protocol','ui-write','ui-read','live-neutral','live-cotton','live-coal','qualify'))
    stage=parser.parse_args().stage
    STUDY.mkdir(parents=True,exist_ok=True)
    if stage.startswith('ui-'): unity(stage,phase=stage[3:])
    elif stage.startswith('live-'): unity(stage,focus=stage[5:])
    else: globals()[stage]()
