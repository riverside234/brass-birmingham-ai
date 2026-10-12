"""Build additive authored throughput workers; qualified/game runtimes stay intact."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import hashlib

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def source_sha(path):return hashlib.sha256(path.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest()
def replace(source,old,new):
    assert source.count(old)==1, 'Training host hook changed: '+old
    return source.replace(old,new)

def build(root=ROOT,mono=None):
    base=root/'artifacts/r1-host/guarded'
    pinned=json.loads((base/'runtime.json').read_text(encoding='utf-8-sig'))
    for entry in pinned['files']:assert sha(base/entry['name'])==entry['sha256'],'Changed qualified runtime'
    spec=importlib.util.spec_from_file_location('search_builder',root/'scripts/build-search-host.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    reports=[]
    for search in (False,True):
        kind='search' if search else 'ordinary'
        out=root/'artifacts/training-host'/kind;out.mkdir(parents=True,exist_ok=True)
        for entry in pinned['files']:
            if entry['name']!='BrassHost.exe':shutil.copyfile(base/entry['name'],out/entry['name'])
        originals=sorted((root/'src/BrassHost').glob('*.cs'))
        originals+=sorted((root/'src/BrassTrainingHost').glob('*.cs'))
        if search:originals+=sorted((root/'src/BrassSearchHost').glob('*.cs'))
        sources=[]
        for path in originals:
            if search and path==root/'src/BrassHost/Program.cs':continue
            text=path.read_text(encoding='utf-8-sig')
            if path.name=='Host.cs':
                if search:text=builder.search_source(text)
                text=replace(text,'var before=Checkpoint();','var before=RollbackMark();')
                text=replace(text,'Adopt(Restore(before));','Adopt(Restore(before()));')
                text=replace(text,'void Adopt(Host other)\n        {','void Adopt(Host other)\n        {\n            InvalidateFastCaches();')
                text=replace(text,'public void Advance()\n        {','public void Advance()\n        {\n            InvalidateFastCaches();')
            elif path.name=='Choices.cs':text=replace(text,'public List<Choice> Choices()','List<Choice> UncachedChoices()')
            elif path.name=='Observation.cs':text=replace(text,'string[] KnownWild(int seat)','string[] UncachedKnownWild(int seat)')
            elif path.name=='Program.cs':
                text='using System.Diagnostics;\n'+text
                text=replace(text,'string id=null;','string id=null;var elapsed=Stopwatch.StartNew();var cpu=Process.GetCurrentProcess().TotalProcessorTime;')
                var='r' if search else 'req'
                hook='case "context":'
                extra='case "capabilities":data=new{fast="training-host/1.0.0"};break;\n'
                if search:
                    extra+='case "search_snapshot":data=host.SearchSnapshot();break;\n'
                    extra+='case "search_step":\n'+f'if((string){var}["game_id"]!=host.GameId)throw new InvalidOperationException("stale_game");\n'
                    extra+=f'if(host.ChancePending){{if((int){var}["revision"]!=host.Revision)throw new InvalidOperationException("stale_revision");host.ResolveChance((string){var}["card"]);}}\n'
                    extra+=f'else host.Apply((int){var}["revision"],(string){var}["choice"]);data=host.SearchSnapshot();break;\n'
                else:
                    extra+='case "decision":data=host.Decision();break;\n'
                    extra+=f'case "step_decision":if((string){var}["game_id"]!=host.GameId)throw new InvalidOperationException("stale_game");data=host.StepDecision((int){var}["revision"],(string){var}["choice"],(int?){var}["forced_limit"]??0);break;\n'
                text=replace(text,hook,extra+hook)
                text=replace(text,'Console.WriteLine(Wire.Json(new{id,ok=true,data}));','FastReply.Write(id,data,elapsed,cpu,host.ChoiceCacheHits,host.ChoiceCacheMisses);')
            target=out/(('Search' if search and path.parent.name=='BrassSearchHost' else '')+path.name)
            target.write_text(text,encoding='utf-8',newline='\n');sources.append(target)
        args=['/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',f'/out:"{out / "BrassHost.exe"}"']
        if search:args+=['/define:TRAINING_SEARCH']
        args += [f'/reference:"{p}"' for p in sorted(out.glob('*.dll'))]+[f'"{p}"' for p in sources]
        rsp=out/'build.rsp';rsp.write_text('\n'.join(args),encoding='utf-8-sig')
        compiler=root/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
        subprocess.run(([str(compiler)] if os.name=='nt' else [mono or 'mono',str(compiler)])+['@'+str(rsp)],check=True)
        # The generated C# also depends on the explicit-chance transformation
        # helper. Pin it alongside the source inputs so a Python-only change
        # cannot leave a stale compiled search worker admitted on the lab.
        authored=originals+[root/'scripts/build-search-host.py']
        manifest=dict(pinned,optimization='training-host/1.0.0',kind=kind,
            baseline_host_sha256=sha(base/'BrassHost.exe'),
            authored_sources={p.relative_to(root).as_posix():source_sha(p) for p in authored},
            builder_sha256=source_sha(Path(__file__)),files=[dict(name=p.name,sha256=sha(p)) for p in sorted(out.glob('*.dll'))]+[dict(name='BrassHost.exe',sha256=sha(out/'BrassHost.exe'))])
        target=root/'training/runtime/throughput'/kind;target.mkdir(parents=True,exist_ok=True)
        (out/'runtime.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        for name in ('runtime.json','BrassHost.exe'):shutil.copyfile(out/name,target/name)
        reports.append(dict(kind=kind,sha256=sha(out/'BrassHost.exe')))
    return reports

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--mono');a=p.parse_args()
    print(json.dumps(build(a.root.resolve(),a.mono),indent=2))
