"""L14 Windows qualification. Raw measurements and failed gates are immutable.

Run --help for independent, resumable stages. This never installs a candidate.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
L13 = ROOT / 'artifacts/rust-backend/20261026-l13-qualified'
L10 = ROOT / 'artifacts/rust-backend/20261025-l10-capture'
MODEL = ROOT / 'artifacts/rust-backend/20261025-l09-qualified-v2/model'
SUPPORT = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
read, write, sha, receipt = (SUPPORT[k] for k in ('read', 'write', 'sha', 'receipt'))


def freeze(out):
    assert not out.exists(), 'Choose a new run; protocols cannot be overwritten'
    out.mkdir(parents=True)
    manifest = read(L13 / 'build-manifest.json')
    for p in manifest['sourcePins']:
        assert sha(L13 / 'runtime' / p['path']) == p['sha256']
    for p in manifest['files']:
        assert sha(L13 / 'Managed' / p['name']) == p['sha256']
    protocol = dict(
        format='brass-l14-study/1.0.0', created_utc=datetime.now(timezone.utc).isoformat(),
        baseline=receipt(L13 / 'qualification.json'), baseline_build=receipt(L13 / 'build-manifest.json'),
        model=receipt(MODEL / 'manifest.json'), sessions=2, threads_per_session=1,
        whole_turn_ms=4000, matched_repetitions=5, fixed_reply_repetitions=30,
        validation_material=dict(p95_main_thread_ms=8, median_per_turn_ms=10, operator='either'),
        validation_benefit=dict(minimum_main_stall_reduction_ms=4, minimum_main_stall_reduction_fraction=.30,
            end_to_end_max_regression_ms=20, cpu_max_regression_fraction=.05,
            acceptance_rejection_events_state='exact equivalence', coverage_regression_allowed=False),
        performance=dict(rules_ratio=2, scripted_gumbel_ratio=2, real_gumbel_ratio=1.25,
            critical_group_max_slowdown=.05, response_p95_ms=4050, response_max_ms=4150),
        memory=dict(process_peak_bytes=2*1024**3, post_retirement_growth_bytes=32*1024**2,
            source='Conservative two-owner g890 local process ceiling; measure peak and retained growth separately'),
        search_policy=dict(k=read(L13/'BrassNative/alpha-g890/policy.json')['config']['k'], c1=50., c2=.1, cap=1, pending_limit=1, flush_us=0,
            tuning='Frozen L12/L13 singleton policy. Alternative settings are separate studies, never silently substituted.'),
        quality=dict(minimum_games_per_control=16, seeds=[92781,92782,92783,92784], seats=[0,1,2,3],
            controls=['legacy-neural','qualified-v4'], time_ms=4000,
            noninferiority_utility_margin=.05, confidence=.95, minimum_supported_plan_fraction=.95,
            invalid_actions_allowed=0, interpretation='Seat/setup-block bootstrap; no throughput surrogate for strength'),
        validation_development=[3,26,13,17], validation_heldout=[0,8,20,23],
        diagnostic_phases=['action','payment','continuation','bonus','debt'],
        promotion='Every applicable gate must pass. Preserve failed measurements and retain baseline on failed benefit/correctness.',
        baseline_sources=[receipt(L13/'runtime'/p['path']) for p in manifest['sourcePins']],
        baseline_binaries=[receipt(L13/'Managed'/p['name']) for p in manifest['files']])
    write(out/'protocol.json', protocol)
    evidence = read(L13/'transport-replay/verification.json')
    rows=[]
    for i in protocol['validation_development']+protocol['validation_heldout']:
        item=read(L10/f'brass-{i:03d}-request.json')['cases'][0]
        reply=next(r for r in evidence['cases'] if r['name']==f'brass-{i:03d}')
        rows.append(dict(name=reply['name'],partition='development' if i in protocol['validation_development'] else 'heldout',
            input=item['input'],actions=reply['response']['data']['actions']))
    write(out/'validation-fixtures.json', rows)
    write(out/'freeze.json',dict(protocol=receipt(out/'protocol.json'),fixtures=receipt(out/'validation-fixtures.json'),
        installed_receipt=receipt(ROOT/'artifacts/patch-installation.json'),
        baseline_validation_calls_per_ordinary_action=3))
    print('Frozen L14 protocol and exact L13 baseline', flush=True)


def checked(out):
    frozen=read(out/'freeze.json')
    assert sha(out/'protocol.json')==frozen['protocol']['sha256']
    assert sha(out/'validation-fixtures.json')==frozen['fixtures']['sha256']
    return read(out/'protocol.json')


def compile_probe(out, build, name='ValidationBench', unity=False, candidate=False):
    folder=out/name;folder.mkdir(exist_ok=True)
    managed=build/'Managed'
    for p in managed.glob('*.dll'):
        if not any(s in p.name for s in ('Tests','Regression')):shutil.copyfile(p,folder/p.name)
    csc=ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    target=folder/('LocalAiUnityTests.dll' if unity else name+'.exe')
    args=['/nologo','/target:'+('library' if unity else 'exe'),'/langversion:7.3','/nostdlib+',
          '/deterministic+','/optimize+',f'/out:"{target}"']
    if unity:args+=['/define:UNITY_PROBE'+(';VALIDATION_CANDIDATE' if candidate else '')]
    args += [f'/reference:"{p}"' for p in folder.glob('*.dll') if p!=target]
    args += [f'"{ROOT}/tests/RustBackendLocal/ValidationBench.cs"']
    if candidate:args += [f'"{ROOT}/tests/RustBackendLocal/ValidationCandidateBench.cs"']
    rsp=folder/'probe.rsp';rsp.write_text('\n'.join(args),encoding='utf-8-sig')
    subprocess.run([str(csc),'@'+str(rsp)],check=True,cwd=ROOT)
    return target


def validation(out, build, label, unity=False, candidate=False):
    protocol=checked(out)
    target=out/(label+'-validation.json')
    assert not target.exists()
    binary=compile_probe(out,build,name=label+'-probe',unity=unity,candidate=candidate)
    if unity:
        player=ROOT/'artifacts/card-tracker-player'
        managed=player/'BrassBirminghamUnity_Data/Managed'
        for p in read(build/'build-manifest.json')['files']:
            shutil.copyfile(build/'Managed'/p['name'],managed/p['name'])
        shutil.copyfile(binary,player/'artifacts/patch-build/Managed/LocalAiUnityTests.dll')
        env=dict(os.environ,BRASS_L14_FIXTURES=str(out/'validation-fixtures.json'),BRASS_L14_RESULT=str(target),
                 BRASS_L14_REPETITIONS=str(protocol['fixed_reply_repetitions']))
        command=[str(player/'BrassBirminghamUnity.exe'),'-batchmode','-nographics','-logFile',str(out/(label+'-unity.log')),'--brass-local-ai-test']
        subprocess.run(command,cwd=player,env=env,check=True,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    else:
        subprocess.run([str(binary),str(out/'validation-fixtures.json'),str(target),str(protocol['fixed_reply_repetitions'])],check=True,
                       timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    data=read(target)
    assert data['passed'],data
    write(out/(label+'-validation-receipt.json'),dict(binary=receipt(binary),build=receipt(build/'build-manifest.json'),
        sources=[receipt(ROOT/'tests/RustBackendLocal'/name) for name in
                 (['ValidationBench.cs','ValidationCandidateBench.cs'] if candidate else ['ValidationBench.cs'])],
        measurement=receipt(target),runtime='Unity Mono' if unity else '.NET Framework'))
    print(label, data.get('summary',dict(checks=data.get('checks'),rows=len(data['rows']))), flush=True)


def profile_build(out, build, label, validate_study=True):
    """Instrument copies only. The installed/L13/candidate sources stay intact."""
    if validate_study: checked(out)
    # Other isolated studies may supply their own frozen protocol, but every
    # source/binary still has to match the candidate's sealed manifest.
    for pin in read(build/'build-manifest.json')['sourcePins']:
        assert sha(build/'runtime'/pin['path'])==pin['sha256'].lower()
    for pin in read(build/'build-manifest.json')['files']:
        assert sha(build/'Managed'/pin['name'])==pin['sha256'].lower()
    dest=out/(label+'-profile-build');assert not dest.exists()
    shutil.copytree(build/'runtime',dest/'runtime')
    shutil.copytree(build/'Managed',dest/'Managed')
    shutil.copytree(build/'BrassNative',dest/'BrassNative')
    manifest=read(build/'build-manifest.json')
    changes=[]
    def wrap(name, signature, stage, context='null'):
        path=dest/'runtime'/name;text=path.read_text(encoding='utf-8-sig')
        start=text.index('{',text.index(signature));at=start+1;depth=1;mode=None
        # C# lexical brace scan; strings/comments may contain braces.
        while depth:
            ch=text[at];nxt=text[at:at+2]
            if mode=='line':
                if ch=='\n':mode=None
            elif mode=='comment':
                if nxt=='*/':mode=None;at+=1
            elif mode in ('"',"'"):
                if ch=='\\':at+=1
                elif ch==mode:mode=None
            elif nxt=='//':mode='line';at+=1
            elif nxt=='/*':mode='comment';at+=1
            elif ch in ('"',"'"):mode=ch
            elif ch=='{':depth+=1
            elif ch=='}':depth-=1
            at+=1
        code=text[:start+1]+f' using(BrassLegacyAI.Native.NativeStageTrace.Measure("{stage}",{context})) {{'+text[start+1:at-1]+'} '+text[at-1:]
        path.write_text(code,encoding='utf-8');changes.append(dict(path=name,method=signature,stage=stage))
    base='src/BrassLocalAI/'
    for name,signature,stage in [
        ('Policy/LiveHost.cs','public static Host Capture(','snapshot.live_capture'),
        ('Policy/LiveHost.cs','public void ValidateLiveAction(','validation.full'),
        ('Policy/LiveHost.cs','public void LiveCommit(','commit.live_with_validation'),
        ('Host/Observation.cs','public object Observe(','snapshot.observe'),
        ('Policy/Belief.cs','public static JObject Setup(','validation.belief'),
        ('Policy/HostExtensions.cs','public void SearchApply(','rules.semantic_step'),
        ('Native/NativeInput.cs','public static JObject Capture(','snapshot.search_input'),
        ('Native/NativeInput.cs','public static JObject FromView(','snapshot.search_copy'),
        ('Native/NativeInput.cs','public static JObject Request(','ipc.request_preparation'),
        ('Native/NativeProcess.cs','JObject Read(','ipc.read_wait_and_parse'),
        ('Native/NativeProcess.cs','long Write(','ipc.write'),
        ('Unity/LocalTurnBridge.cs',' Events()','snapshot.events'),
        ('Policy/LocalGameState.cs','public static void CommitIntent(','save.intent')]:
        wrap(base+name,signature,stage)
    for name,signature,stage in [
        ('PublicHistory.cs','public static Host CapturePersonality(','snapshot.public_projection'),
        ('PublicHistory.cs','public JObject ObservePersonality(','snapshot.nn_observe'),
        ('LiveChoices.cs','public void ValidatePersonalityAction(','validation.personality'),
        ('LiveChoices.cs','public void CommitPersonalityAction(','commit.personality')]:wrap('src/BrassPersonalityGame/'+name,signature,stage)
    wrap(base+'Native/NativeProcess.cs','public JObject Search(','native.roundtrip','request["ticket"].ToString(Newtonsoft.Json.Formatting.None)')
    wrap(base+'Native/NativeProcess.cs','public Task ExecuteAsync(','native.job','request["ticket"].ToString(Newtonsoft.Json.Formatting.None)')
    wrap(base+'Unity/LocalTurnBridge.cs','void Update()','unity.update','instance+":"+(startedGeneration+1)+":"+planningRequests+":"+revision+":"+actor')
    wrap(base+'Unity/LocalTurnBridge.cs','void LaunchNative(','unity.launch','instance+":"+(startedGeneration+1)+":"+planningRequests+":"+revision+":"+actor')
    p=dest/'runtime'/base/'Native/NativeValidation.cs'
    if p.exists():
        wrap(base+'Native/NativeValidation.cs','public Task ExecuteAsync(','validation.detached','binding')
        wrap(base+'Native/NativeValidation.cs','public void Commit(','commit.native_guard_and_apply','binding')
    if 'public static bool ContinuationMatches(' in (dest/'runtime'/base/'Native/NativeInput.cs').read_text(encoding='utf-8-sig'):
        wrap(base+'Native/NativeInput.cs','public static bool ContinuationMatches(','validation.projected_successor')
    p=dest/'runtime'/base/'Native/NativeProcess.cs';text=p.read_text(encoding='utf-8')
    text=text.replace('JObject.Parse(new UTF8Encoding(false,true).GetString(Exact(stream,length)))','NativeStageTrace.Parse(new UTF8Encoding(false,true).GetString(Exact(stream,length)))')
    text=text.replace('Encoding.UTF8.GetBytes(request.ToString(Formatting.None))','NativeStageTrace.Encode(request)')
    text=text.replace('stream.Write(header,0,header.Length);stream.Write(bytes,0,bytes.Length);stream.Flush();return id;',
        'NativeStageTrace.Point("ipc.write_begin",id,bytes.Length);stream.Write(header,0,header.Length);stream.Write(bytes,0,bytes.Length);stream.Flush();NativeStageTrace.Point("ipc.write_end",id,bytes.Length);return id;')
    p.write_text(text,encoding='utf-8')
    shutil.copyfile(ROOT/'tests/RustBackendLocal/NativeStageTrace.cs',dest/'runtime'/base/'Native/NativeStageTrace.cs')
    response_file=build/'BrassLegacyAI.dll.rsp'
    if not response_file.exists():response_file=build/'legacy.rsp'
    original=response_file.read_text(encoding='utf-8-sig')
    original=original.replace(str(build),str(dest))+'\n"'+str(dest/'runtime'/base/'Native/NativeStageTrace.cs')+'"\n'
    rsp=dest/'BrassLegacyAI.dll.rsp';rsp.write_text(original,encoding='utf-8-sig')
    compiler=ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    subprocess.run([str(compiler),'@'+str(rsp)],check=True,cwd=ROOT)
    for item in manifest['files']:item['sha256']=sha(dest/'Managed'/item['name'])
    manifest['sourcePins']=[dict(path=p.relative_to(dest/'runtime').as_posix(),sha256=sha(p))
                            for p in sorted((dest/'runtime').rglob('*')) if p.is_file()]
    manifest.update(diagnostic_only=True,instrumentation=changes,profiled_build=receipt(build/'build-manifest.json'))
    write(dest/'build-manifest.json',manifest)
    print('Instrumented isolated build',dest,flush=True)


def live_fixtures(out):
    return [dict(name=f'live-{seed}-{era}-{round}',seed=seed,era=era,round=round,
                 partition='development' if seed==92781 else 'heldout')
            for seed in (92781,92782) for era in ('CanalEra','RailEra') for round in (2,7)]


def live(out, build, label, quick=False):
    checked(out)
    inputs=out/(label+'-live-input.json');result=out/(label+'-live.json')
    assert not result.exists()
    rows=live_fixtures(out);write(inputs,rows[:1] if quick else rows)
    env=dict(os.environ,BRASS_L14_LIVE_FIXTURES=str(inputs),BRASS_L14_LIVE_RESULT=str(result))
    cache=out/'trusted-live-fixtures';cache.mkdir(exist_ok=True)
    env['BRASS_L14_LIVE_CACHE']=str(cache)
    if read(build/'build-manifest.json').get('diagnostic_only'):
        env['BRASS_L14_TRACE']=str(out/(label+'-stage-trace.json'))
    else:env.pop('BRASS_L14_TRACE',None)
    # Never let the integration runner write results into the sealed L13 build.
    if (build/'qualification.json').exists():
        clone=out/(label+'-live-build');assert not clone.exists()
        for name in ('runtime','Managed','BrassNative'):shutil.copytree(build/name,clone/name)
        shutil.copyfile(build/'build-manifest.json',clone/'build-manifest.json');build=clone
    pwsh=shutil.which('pwsh');assert pwsh
    log=out/(label+'-live.log')
    with log.open('w',encoding='utf-8') as stream:
        command=[pwsh,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'tests/RustBackendLocal/run-unity.ps1'),
                 '-BuildDirectory',build.relative_to(ROOT).as_posix(),'-Matched']
        if quick:command+=['-MaxSeconds','90']
        done=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,timeout=120 if quick else 600)
    for name in ('unity.json','unity.log'):
        if (build/name).exists():shutil.copyfile(build/name,out/(label+'-'+name))
    assert done.returncode==0,log
    data=read(result);assert data['complete'] and data['passed']
    write(out/(label+'-live-receipt.json'),dict(build=receipt(build/'build-manifest.json'),input=receipt(inputs),result=receipt(result),
        log=receipt(log),sources=[receipt(ROOT/'tests/RustBackendLocal/NativeMatchedUnityTests.cs'),receipt(ROOT/'tests/BrassLocalAI/LocalAiUnityTests.cs')]))
    print(label,'live',len(data['rows']),'passed',flush=True)


def matched(out, candidate, prefix='matched'):
    protocol=checked(out)
    baseline=out/'baseline-matched-build'
    if not baseline.exists():
        for name in ('runtime','Managed','BrassNative'):shutil.copytree(L13/name,baseline/name)
        shutil.copyfile(L13/'build-manifest.json',baseline/'build-manifest.json')
    for rep in range(protocol['matched_repetitions']):
        for arm in (['baseline','candidate'] if rep%2==0 else ['candidate','baseline']):
            label=f'{prefix}-r{rep}-{arm}'
            if (out/(label+'-live-receipt.json')).exists():
                prior=read(out/(label+'-live-receipt.json'))
                assert sha(out/(label+'-live.json'))==prior['result']['sha256']
                assert sha((baseline if arm=='baseline' else candidate)/'build-manifest.json')==prior['build']['sha256']
                for source in prior['sources']:assert sha(ROOT/source['path'])==source['sha256'],source['path']
                continue
            live(out,baseline if arm=='baseline' else candidate,label)


def lifecycle(out,build,label):
    checked(out)
    target=out/(label+'-lifecycle-unity.json');assert not target.exists()
    assert not (build/'qualification.json').exists(), 'Never overwrite a sealed qualification'
    env=dict(os.environ)
    for key in list(env):
        if key.startswith('BRASS_L14_'):del env[key]
    command=[shutil.which('pwsh'),'-NoProfile','-File',str(ROOT/'tests/RustBackendLocal/run-unity.ps1'),
             '-BuildDirectory',build.relative_to(ROOT).as_posix()]
    log=out/(label+'-lifecycle.log')
    with log.open('w',encoding='utf-8') as stream:
        done=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    for name in ('unity.json','unity.log','unity-turns.json'):
        if (build/name).exists():shutil.copyfile(build/name,out/(label+'-lifecycle-'+name))
    assert done.returncode==0,log
    data=read(target);assert data['passed'] and not data['nativeErrors']
    sources=[ROOT/'tests/BrassLocalAI'/n for n in ('LocalAiUnityTests.cs','AiExecutionResourceTests.cs')]
    sources += [ROOT/'tests/RustBackendLocal'/n for n in
                ('NativeLocalUnityTests.cs','NativeMatchedUnityTests.cs','NativeValidationLifecycleTests.cs','run-unity.ps1')]
    write(out/(label+'-lifecycle-receipt.json'),dict(build=receipt(build/'build-manifest.json'),
        evidence=receipt(target),log=receipt(log),sources=[receipt(p) for p in sources]))
    print(label,'lifecycle',data['checks'],'passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','validation','profile-build','live','matched','lifecycle'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path,default=L13)
    p.add_argument('--label',default='baseline')
    p.add_argument('--unity',action='store_true')
    p.add_argument('--candidate',action='store_true')
    p.add_argument('--quick',action='store_true')
    a=p.parse_args();out=a.out.resolve()
    assert out.is_relative_to(ROOT/'artifacts/rust-backend')
    if a.stage=='freeze':freeze(out)
    elif a.stage=='validation':validation(out,a.build.resolve(),a.label,a.unity,a.candidate)
    elif a.stage=='profile-build':profile_build(out,a.build.resolve(),a.label)
    elif a.stage=='live':live(out,a.build.resolve(),a.label,a.quick)
    elif a.stage=='matched':matched(out,a.build.resolve(),'matched' if a.label=='baseline' else a.label)
    elif a.stage=='lifecycle':lifecycle(out,a.build.resolve(),a.label)
