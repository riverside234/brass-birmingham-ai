"""L14.5: bounded, immutable CPU-provider qualification groups.

No installation, tournament, global Python packages or historical receipts are
changed. Each measured subprocess has a 90-second watchdog.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import winreg
import re
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/rust-backend/20261026-l14-5/providers'
MODEL = ROOT / 'artifacts/rust-backend/20261025-l09-qualified-v2/model'
BASE = ROOT / 'artifacts/rust-backend/20261026-l14-smoke/candidate-native'
BIN = OUT / 'native-experiment'
SUPPORT = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pin(path):
    path = Path(path)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size)


def freeze():
    cpu = winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
        r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'), 'ProcessorNameString')[0]
    write(OUT / 'protocol.json', dict(
        version='brass-providers/1.0.0', recorded_before_measurement=True,
        cpu=cpu, short_groups=True, watchdog_seconds=90, repetitions=2,
        sessions=2, threads_per_session=1, cap=1, pending_limit=1, flush_us=0,
        precision='FP32', gelu='exact erf; no approximation', allowance_ms=4000,
        controls=dict(A='L14 v5 ORT 1.22.1 / ort rc.10 CPUExecutionProvider',
            A2='ORT 1.31.0 / ort rc.13 CPUExecutionProvider',
            B='same A2 stack; OpenVINO plugin 1.8.1 CPU'),
        primary=dict(metric='per-input median total warm native evaluation nanoseconds, both owners active',
            aggregation='geometric mean of paired A2/B and A/B speed ratios across inputs; AB then BA',
            minimum_ratio=1.05, minimum_improving_fraction=0.75,
            held_out='separate dense Rail/staged inputs; each group must improve in aggregate'),
        end_to_end=dict(median_regression_max_fraction=0.05, individual_regression_max_ms=250,
            useful_work_min_ratio=0.95, no_supported_plan_coverage_loss=True,
            no_new_deadline_or_correctness_failures=True),
        parity=dict(atol=0.0002, rtol=0.0002, heads=5, masks_and_order='exact'),
        scope='Short smoke evidence; no population tails, playing strength or L15 release claim.',
        baseline=[pin(BASE / n) for n in ['brass-ai-local.exe','gumbel-pair.exe']],
        model=[pin(MODEL / n) for n in ['manifest.json','model.onnx','model.json','onnxruntime.dll']],
        historical=pin(ROOT / 'artifacts/rust-backend/20261026-l14-smoke/final-checks.json'),
        installed=[pin(p) for p in sorted((ROOT / 'BrassNative/alpha-g890').rglob('*')) if p.is_file()],
        managed=[pin(ROOT/'BrassBirminghamUnity_Data/Managed'/n) for n in
                 ['Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll']]))
    print('Frozen protocol before candidate measurements', flush=True)


def fetch():
    """Explicit download step; never installs into the user's Python environment."""
    for package in read(ROOT/'configs/references/cpu-providers.lock.json')['packages']:
        path=OUT/package['filename']
        if not path.exists():
            data=urllib.request.urlopen(package['url'],timeout=60).read()
            assert len(data)==package['bytes'] and hashlib.sha256(data).hexdigest()==package['sha256']
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        assert sha(path)==package['sha256']
        record=OUT/(package['name']+'-download.json')
        if not record.exists():write(record,dict(filename=package['filename'],url=package['url'],size=package['bytes'],digests={'sha256':package['sha256']}))
        destination=OUT/package['name']
        if not destination.exists():
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    assert (destination/name).resolve().is_relative_to(destination.resolve())
                archive.extractall(destination)
        print('Pinned package available:',package['filename'],flush=True)


def run(command, folder, *, env=None):
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sys.path[:0]=[str(ROOT/'artifacts/benchmark-deps'),str(ROOT/'artifacts/r2-pythondeps')]
    import psutil
    peak_rss=0;cpu_seconds=0;first_threads=None;last_threads={};samples=0
    with (folder / 'run.log').open('w', encoding='utf-8') as stream:
        process = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        monitor=psutil.Process(process.pid)
        try:
            while process.poll() is None:
                if time.monotonic()-started>90:raise subprocess.TimeoutExpired(command,90)
                try:
                    peak_rss=max(peak_rss,monitor.memory_info().rss)
                    cpu=monitor.cpu_times();cpu_seconds=max(cpu_seconds,cpu.user+cpu.system)
                    log=(folder/'run.log').read_bytes()
                    if b'MEASURE\n' in log or b'MEASURE\r\n' in log:
                        current={str(t.id):t.user_time+t.system_time for t in monitor.threads()}
                        if first_threads is None:first_threads=current
                        if b'MEASURE_DONE' not in log:last_threads=current;samples+=1
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                try:process.wait(timeout=.025)
                except subprocess.TimeoutExpired:pass
            code=process.returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                stdout=stream, stderr=stream, timeout=10, check=False)
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            write(folder/'receipt.json', dict(timeout=True, wall_seconds=time.monotonic()-started))
            raise
    write(folder/'receipt.json', dict(command=[str(x) for x in command], exit_code=code,
        wall_seconds=time.monotonic()-started, watchdog_seconds=90,sampled_peak_rss_bytes=peak_rss,
        sampled_process_cpu_seconds=cpu_seconds,steady_state_samples=samples,
        steady_thread_cpu_seconds={k:max(0,v-(first_threads or {}).get(k,0)) for k,v in last_threads.items()}))
    if code:
        raise subprocess.CalledProcessError(code, command)


def prepare():
    assert (OUT/'protocol.json').exists()
    plugin = OUT/'onnxruntime-ep-openvino/onnxruntime_ep_openvino'
    config = '{"CPU":{"INFERENCE_PRECISION_HINT":"f32","EXECUTION_MODE_HINT":"ACCURACY","INFERENCE_NUM_THREADS":1,"NUM_STREAMS":1,"ENABLE_CPU_PINNING":false}}'
    for arm in ('A2', 'B'):
        target = OUT/'models'/arm
        target.mkdir(parents=True, exist_ok=False)
        for name in ('model.onnx','model.json'):
            shutil.copyfile(MODEL/name,target/name)
        shutil.copyfile(OUT/'onnxruntime/onnxruntime/capi/onnxruntime.dll',target/'onnxruntime.dll')
        m = read(MODEL/'manifest.json')
        m.update(runtime_version='1.31.0',binding_version='2.0.0-rc.13',runtime_sha256=sha(target/'onnxruntime.dll'))
        if arm == 'B':
            files=[]
            for p in sorted(plugin.iterdir()):
                if p.suffix not in ('.dll','.manifest','.json'): continue
                destination=target/'openvino'/p.name; destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(p,destination)
                files.append(dict(path=destination.relative_to(target).as_posix(),sha256=sha(destination)))
            m.update(provider='OpenVINOExecutionProvider', provider_runtime=dict(plugin_version='1.8.1',
                openvino_version='2026.4.1-22982-e213a147257',library='openvino/onnxruntime_providers_openvino_plugin.dll',
                files=files,options={'load_config':config},allow_cpu_fallback=False))
        write(target/'manifest.json',m)
    # Real recorded roots and first staged leaves. Keep held-out roots separate.
    capture=ROOT/'artifacts/rust-backend/20261025-l10-capture'
    cases=[]
    for i in (0,3,8,26,30,41,13,28):
        c=read(capture/f'brass-{i:03d}-request.json')['cases'][0]
        cases.append(dict(name=f'root-{i:03d}',held_out=i in (13,28),keys=[c['replies'][0]['info']['key']]))
    c=read(capture/'brass-003-request.json')['cases'][0]
    staged=next(r['info']['key'] for r in c['replies'] if r['info']['key']['information']['observation']['view']['context']['phase']!='action')
    cases.append(dict(name='held-staged',held_out=True,keys=[staged]))
    cases.append(dict(name='mixed-2',held_out=False,keys=[cases[1]['keys'][0],cases[-1]['keys'][0]]))
    cases.append(dict(name='mixed-4',held_out=False,keys=[cases[i]['keys'][0] for i in (0,2,4,5)]))
    write(OUT/'corpus.json',cases)
    write(OUT/'runtime-pins.json',dict(packages=[read(OUT/(n+'-download.json')) for n in
        ('onnxruntime','onnxruntime-ep-openvino')],models={arm:pin(OUT/'models'/arm/'manifest.json') for arm in ('A2','B')}))
    print('Prepared pinned runtime bundles and representative/held-out input corpus',flush=True)


def micro(arm, label):
    manifest=MODEL/'manifest.json' if arm=='A' else OUT/'models'/arm/'manifest.json'
    request=OUT/(label+'-input.json')
    write(request,dict(manifest=str(manifest),cases=read(OUT/'corpus.json'),iterations=32))
    binary=(OUT/'native-baseline' if arm=='A' else BIN)/'provider_probe.exe'
    run([binary,request,OUT/label/'output.json'],OUT/label)
    result=read(OUT/label/'output.json')
    assert len(result['owners'])==2
    for owner in result['owners']:
        assert owner['receipt']['provider']=='CPUExecutionProvider'
        assert owner['receipt'].get('provider_selection',{}).get('fallback_reason') is None
        assert len(owner['rows'])==11
    write(OUT/label/'inputs.json',dict(binary=pin(binary),request=pin(request),arm=arm,
        scope='CPU control only, not an OpenVINO performance result. A uses the unchanged rc.10 / 1.22.1 stack.'))
    print(label,'passed',flush=True)


def fallback(label):
    import struct
    folder=OUT/label;folder.mkdir(exist_ok=False)
    original=ROOT/'artifacts/rust-backend/20261026-l14-candidate-v5/BrassNative/alpha-g890'
    outcomes=[]
    for name,allow,override,missing,corrupt in [
        ('strict',False,False,False,False),('fallback',True,False,False,False),
        ('override',False,True,False,False),('missing',True,False,True,False),
        ('tampered-pin',True,False,False,True)]:
        package=folder/name;package.mkdir();model=package/'model';model.mkdir()
        for p in (OUT/'models/B').rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(OUT/'models/B');target=model/rel;target.parent.mkdir(exist_ok=True,parents=True)
            if missing and p.name=='onnxruntime_providers_openvino_plugin.dll':continue
            shutil.copyfile(p,target)
        m=read(model/'manifest.json');m['provider_runtime']['allow_cpu_fallback']=allow
        if corrupt:m['provider_runtime']['files'][0]['sha256']='0'*64
        # JSON files above are copies, never hard links to the pinned source.
        (model/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
        shutil.copyfile(original/'warmup.json',package/'warmup.json')
        policy=read(original/'policy.json');policy['model_manifest_sha256']=sha(model/'manifest.json')
        write(package/'policy.json',policy)
        shutil.copyfile(BIN/'brass-ai-local.exe',package/'brass-ai-local.exe')
        env=dict(os.environ,PATH=str(package),PYTHONPATH='');env.pop('BRASS_NN_PROVIDER',None)
        if override:env['BRASS_NN_PROVIDER']='cpu'
        start=time.monotonic()
        r=subprocess.run([str(package/'brass-ai-local.exe'),str(package/'policy.json'),sha(package/'policy.json')],
            input=b'',capture_output=True,env=env,cwd=package,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        (package/'stderr.log').write_bytes(r.stderr)
        (package/'stdout.bin').write_bytes(r.stdout)
        hello=None
        if r.stdout:
            n=struct.unpack('<I',r.stdout[:4])[0];hello=json.loads(r.stdout[4:4+n])
        if name in ('strict','tampered-pin'):
            assert r.returncode and hello is None
            message=r.stderr.replace(b'\x00',b'').decode(errors='replace')
            assert ('binding' in message.lower() if name=='strict' else 'hash mismatch' in message.lower()),message
        else:
            assert r.returncode==0 and hello['status']=='ready',r.stderr[-2000:]
            for receipt in hello['data']['provider_receipts']:
                p=receipt['provider_selection'];assert p['requested']=='OpenVINOExecutionProvider' and p['effective']=='CPUExecutionProvider'
                assert (p['override_reason']=='BRASS_NN_PROVIDER=cpu' and p['fallback_reason'] is None) if override else p['fallback_reason']
        outcomes.append(dict(case=name,passed=True,exit_code=r.returncode,seconds=time.monotonic()-start,hello=hello))
        print(name,'passed',flush=True)
    write(folder/'result.json',dict(passed=True,clean_path=True,cases=outcomes))


def build_probes(label=None):
    """Compile isolated controls; rejected provider code never changes production."""
    archive=ROOT/'artifacts/rust-backend/20261026-l14-candidate-v5/BrassNative/alpha-g890/native-source.zip'
    patch=ROOT/'tests/RustBackendInference/openvino-candidate.patch'
    env,cargo=SUPPORT['tool_environment']();env['CARGO_TARGET_DIR']=str(ROOT/'artifacts/rust-native-target')
    env['CARGO_BUILD_JOBS']='2'
    for arm in ('baseline','experiment'):
        destination=OUT/('native-'+arm)
        if (destination/'build.json').exists():
            for entry in read(destination/'build.json')['files']:assert sha(ROOT/entry['path'])==entry['sha256']
            print('Retained built',arm,'control',flush=True);continue
        source=OUT/((label if arm=='experiment' and label else arm+'-source'));source.mkdir(exist_ok=False)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():assert (source/name).resolve().is_relative_to(source.resolve())
            z.extractall(source)
        if arm=='experiment':
            for name in re.findall(r'^\+\+\+ b/(.+)$',patch.read_text(),re.MULTILINE):
                target=source/name
                if target.exists():target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
            subprocess.run(['git','apply','--directory='+source.relative_to(ROOT).as_posix(),str(patch)],cwd=ROOT,check=True)
        probe=ROOT/'native/brass-ai-inference/examples/provider_probe.rs'
        shutil.copyfile(probe,source/'native/brass-ai-inference/examples/provider_probe.rs')
        destination.mkdir(exist_ok=False)
        commands=[['build','--release','--example','provider_probe']]
        if arm=='experiment':commands[0]+=['--bin','brass-ai-local','--bin','gumbel-pair']
        commands.append(['test','--release','-p','brass-ai-inference','--test','runtime','--test','contracts','--no-run'])
        for i,args in enumerate(commands):
            with (destination/f'build-{i}.log').open('w') as log:
                command=[str(cargo),args[0],'--offline','--locked',*args[1:],'--target','x86_64-pc-windows-gnu']
                result=subprocess.run(command,cwd=source/'native',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            assert result.returncode==0,f'{arm} build {i}: see log'
        release=ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release'
        shutil.copyfile(release/'examples/provider_probe.exe',destination/'provider_probe.exe')
        if arm=='experiment':
            for n in ('brass-ai-local.exe','gumbel-pair.exe'):shutil.copyfile(release/n,destination/n)
        log=(destination/'build-1.log').read_text()
        for kind in ('runtime','contracts'):
            line=next(x for x in log.splitlines() if 'Executable tests' in x and f'{kind}.rs' in x)
            path=Path(line[line.index('(')+1:line.rindex(')')]);shutil.copyfile(path,destination/(kind+'-tests.exe'))
        write(destination/'build.json',dict(arm=arm,base_archive=pin(archive),patch=pin(patch) if arm=='experiment' else None,
            probe=pin(probe),lock=pin(source/'native/Cargo.lock'),files=[pin(p) for p in destination.glob('*.exe')]))
        print('Built isolated',arm,'control',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['freeze','fetch','prepare','build','micro','fallback'])
    parser.add_argument('--arm',choices=['A','A2','B']);parser.add_argument('--label')
    parser.add_argument('--out',type=Path,default=OUT,help='Fresh evidence directory for reproduction')
    args = parser.parse_args()
    OUT=args.out.resolve();BIN=OUT/'native-experiment'
    if args.step=='micro':micro(args.arm,args.label)
    elif args.step=='fallback':fallback(args.label)
    elif args.step=='build':build_probes(args.label)
    else:{'freeze':freeze,'fetch':fetch,'prepare':prepare}[args.step]()
