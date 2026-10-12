"""Small isolated RPC/save/cancellation/install/rollback fixture smoke."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from brass_ai.learning.contracts import require
from brass_ai.bank import sha
from brass_ai.personality.laptop import install,sampling
from brass_ai.personality.laptop.contracts import read,write,seal,PROTOCOL
from brass_ai.personality.laptop.package import validate


def run(package,build,cases,out,debt_cases=None):
    package,build,out=[Path(p).resolve() for p in (package,build,out)]
    require(not out.exists(),'Use a fresh pipeline smoke directory');out.mkdir(parents=True)
    p=validate(package);require(p['fixture'],'This smoke is fixture-only')
    env=dict(os.environ);deps=[x for x in env.get('PYTHONPATH','').split(os.pathsep) if 'artifacts' in x]
    env['PYTHONPATH']=os.pathsep.join([str(package/'runtime/src')]+deps);env['PYTHONDONTWRITEBYTECODE']='1'
    proc=subprocess.Popen([sys.executable,'-m','brass_ai.personality.laptop.cli','serve','--package',str(package),'--fixture'],
        cwd=package,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def send(i,op,body=None):proc.stdin.write(json.dumps(dict(protocol=PROTOCOL,id=i,op=op,body=body))+'\n');proc.stdin.flush()
    def receive(i):
        line=proc.stdout.readline()
        if not line:raise ValueError('Standalone runtime failed: '+proc.stderr.read())
        r=json.loads(line);require(r['id']==i and r['ok'],str(r));return r['data']
    timings=[];case=read(cases)[0];seat=case['view']['seat'];assign=[dict(seat=seat,archetype='random',tier='easy')]
    try:
        send('hello','hello');hello=receive('hello');require(hello['fixture'] and hello['package_id']==p['id'],'Wrong standalone package')
        send('init','initialize',dict(game_id='p6-native-fixture',seed='736',assignments=assign));save=receive('init')
        require(save==json.loads(json.dumps(sampling.new_save(p,'p6-native-fixture','736',assign))),'Packaged sampler differs')
        write(out/'initial-save.json',save)
        for i,mode in enumerate(('play-now','cancel')):
            request_id='turn-'+str(i);body=dict(protocol=PROTOCOL,request_id=request_id,view=case['view'],choices=case['choices'],
                save=save,insolvencies=[0]*4)
            start=time.monotonic();send(request_id,'plan',body);time.sleep(.2);send(request_id,mode)
            r=receive(request_id);timings.append(dict(mode=mode,seconds=time.monotonic()-start,metrics=r['metrics']))
            if mode=='cancel':require(r['status']=='cancelled' and not r['paths'] and r['save']==save,'Cancellation published work or advanced saved RNG')
            else:
                require(r['status']=='ok' and r['paths'] and r['metrics']['deadline_resets']==0,'Play Now failed to return a legal complete incumbent')
                send('restore','restore',dict(save=r['save'],game_id='p6-native-fixture'));require(receive('restore')==r['save'],'Saved state changed')
            # A stale button event must not poison the next service reply.
            send('retired-request','play-now')
        if debt_cases:
            case=next(r for r in read(debt_cases)['records'] if r['kind']=='debt');seat=case['view']['seat']
            debt_save=sampling.new_save(p,'p6-debt-fixture','91',[dict(seat=seat,archetype='random',tier='hard')])
            debt_save['pending']=dict(key=[seat,'CanalEra',2,1,0],remaining_seconds=0.)
            debt_save=seal({k:v for k,v in debt_save.items() if k!='id'})
            body=dict(protocol=PROTOCOL,request_id='debt',view=case['view'],choices=case['choices'],save=debt_save,insolvencies=case['insolvencies'])
            start=time.monotonic();send('debt','plan',body);r=receive('debt')
            require(r['status']=='ok' and len(r['paths'])==1 and len(r['paths'][0])==1
                and r['paths'][0][0] in [c['id'] for c in case['choices']],'Debt did not select a legal industry')
            require(r['metrics']['searches']==0 and r['metrics']['qualified_samples']==0 and r['save']['pending']==debt_save['pending'],
                'Debt changed search budget or used a different search policy')
            timings.append(dict(mode='mandatory-debt',seconds=time.monotonic()-start,metrics=r['metrics']))
        send('close','close');proc.stdin.close();require(proc.wait(timeout=30)==0,'Runtime did not retire cleanly')
    finally:
        if proc.poll() is None:
            proc.stdin.close()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait();raise
    if debt_cases:
        from brass_ai.recovery.inputs import InputBoundary
        from brass_ai.searchlab.spiel import BrassSpielGame
        boundary=InputBoundary();game=BrassSpielGame(root=package/'runtime')
        try:
            for case in read(debt_cases)['records']:
                view=case['view'];boundary.check(view,case['choices'])
                state=game.from_observation(view,819,[])
                require(state.views[view['seat']]==view and state.legal==case['choices'],'Live/rules adapter reconstruction mismatch: '+case['kind'])
        finally:game.close()
    sandbox=out/'isolated-game';managed=sandbox/'BrassBirminghamUnity_Data/Managed';managed.mkdir(parents=True)
    (sandbox/'.p6-fixture-sandbox').write_text('Fixture only, not the installed game',encoding='utf-8')
    base=ROOT/'artifacts/patch-releases'/read(build/'build.json')['base_version']/'Managed'
    for name in ('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'):shutil.copyfile(base/name,managed/name)
    before={p.name:sha(p) for p in managed.glob('*.dll')}
    dry=install.install(package,build,sandbox,python=sys.executable,sandbox_fixture=True)
    require(not dry['installed'] and before=={p.name:sha(p) for p in managed.glob('*.dll')},'Dry run wrote game files')
    real=install.install(package,build,sandbox,python=sys.executable,sandbox_fixture=True,apply=True)
    require(real['installed'],'Sandbox install failed')
    require(sha(managed/'Assembly-CSharp.dll')==before['Assembly-CSharp.dll'],'Game assembly changed')
    require(all(sha(managed/n)==sha(build/'Managed'/n) for n in install.OWNED),'Staged turn/debt dispatch helpers differ')
    rolled=install.rollback(real['journal'],apply=True,sandbox_fixture=True)
    require(rolled['rolled_back'] and before=={p.name:sha(p) for p in managed.glob('*.dll')},'Rollback did not restore exact bytes')
    report=dict(passed=True,fixture=True,release_qualified=False,standalone_rpc=hello,turns=timings,sandbox_install=True,rollback=True,installed_game_modified=False,
        native_reconstruction_cases=len(read(debt_cases)['records']) if debt_cases else 0)
    write(out/'report.json',report);print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('package','build','cases','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--debt-cases',type=Path)
    a=p.parse_args();run(a.package,a.build,a.cases,a.output,a.debt_cases)
