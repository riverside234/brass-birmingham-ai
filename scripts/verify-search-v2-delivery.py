"""Independent C# replay and actual-successor guard; no installed mutations."""
import argparse
import runpy
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
T=runpy.run_path(str(ROOT/'scripts/search-v2-tools.py'))
read,write,receipt=(T[k] for k in ('read','write','receipt'))


def main(label, protocol=None, build=None):
    study=T['OUT']/'study';out=T['OUT']/label;out.mkdir(exist_ok=False)
    fixtures=[]
    sources=[study]
    if (T['OUT']/'fixture-correction-v2/protocol.json').exists():sources.append(T['OUT']/'fixture-correction-v2')
    if protocol: sources=[protocol.resolve().parent]
    for study,batch in ((s,b) for s in sources for b in read(s/'protocol.json')['batches']):
        path=study/(batch['batch']+'-output.json')
        if not path.exists():continue
        inputs=read(study/(batch['batch']+'-input.json'))['cases']
        for row in read(path)['results']:
            if row['name'].endswith('-warmup'):continue
            case=next(c for c in inputs if c['name']==row['name'])
            d=row['result']['decision'];response=dict(actions=d['result']['action_ids'],ticket=dict(revision=case['input']['view']['context']['revision']))
            if d.get('focus'):response['search_revision']=dict(version='brass-local-focus/2.0.0',focus=d['focus'])
            fixtures.append(dict(name=row['name'],input=case['input'],response=response))
    assert fixtures
    source=out/'fixtures.json';write(source,fixtures)
    build=build.resolve() if build else T['OUT']/'local-candidate';folder=out/'probe';folder.mkdir()
    for p in (build/'Managed').glob('*.dll'):
        if not any(s in p.name for s in ('Tests','Regression')):shutil.copyfile(p,folder/p.name)
    target=folder/'SearchV2ValidationTests.exe'
    args=['/nologo','/target:exe','/main:SearchV2ValidationTests','/langversion:7.3','/nostdlib+',
          '/deterministic+','/optimize+',f'/out:"{target}"']
    args += [f'/reference:"{p}"' for p in folder.glob('*.dll')]
    for name in ('ValidationBench.cs','SearchV2ValidationTests.cs'):
        p=ROOT/'tests/RustBackendLocal'/name;shutil.copyfile(p,folder/name);args.append(f'"{folder/name}"')
    rsp=folder/'probe.rsp';rsp.write_text('\n'.join(args),encoding='utf-8-sig')
    subprocess.run([str(ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'),'@'+str(rsp)],check=True)
    result=subprocess.run([str(target),str(source),str(out/'result.json')],timeout=60,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    write(out/'receipt.json',dict(binary=receipt(target),fixtures=receipt(source),
          build=receipt(build/'build-manifest.json'),result=receipt(out/'result.json')))
    data=read(out/'result.json')
    print(dict(passed=data['passed'],checks=data['checks'],plans=len(data['rows']),error=data.get('error')))
    assert result.returncode==0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--protocol',type=Path);p.add_argument('--build',type=Path)
    a=p.parse_args();main(a.label,a.protocol,a.build)
