"""Build/freeze the v5 candidate in isolation; never install or select a pool."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import below,digest,freeze,load_bank,read,require,sha
from brass_ai.search.legacy_runtime import source_paths

BASE='bank-3dc65253a610191d66a5'
STYLES=('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def build(root,mono=None):
    root=Path(root).resolve();base=load_bank(root,BASE)['payload']
    sources=source_paths(root)
    require(not any(p.name=='NetworkAndHand.cs' for p in sources) and
            all('NetworkOptions.' not in p.read_text(encoding='utf-8-sig') for p in sources if p.name!='NetworkOptions.cs'),
            'V5 starts from official v4; the archived Network helper must remain uncalled')
    names=[style+'-v5-masked-industry' for style in STYLES]
    for style,name in zip(STYLES,names):
        profile=read(root/'configs/legacy-ai'/(name+'.json'))
        original=dict(base['profiles'][style+'-v3-masked-economy'])
        require({k:v for k,v in profile.items() if k not in ('id','description')}==
                {k:v for k,v in original.items() if k not in ('id','description')},'Preserve v4 search budgets and style parameters')
    pins=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in sources]
    out=root/'artifacts/v5-development'/digest(pins)[:20]/'runtime'
    dependencies=[f for f in base['runtime_files'] if f['name']!='BrassLegacyAI.exe']
    compiler=root/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    require(compiler.is_file() and sha(compiler)==base['compiler_sha256'],'Use the pinned Roslyn compiler')
    require(os.name=='nt' or shutil.which(mono or 'mono'),'Mono is required on Linux')
    if not (out/'runtime.json').exists():
        out.mkdir(parents=True,exist_ok=True)
        for pin in dependencies:
            source=root/'artifacts/legacy-ai/bank'/BASE/'runtime'/pin['name']
            require(sha(source)==pin['sha256'],'Private dependency changed')
            shutil.copyfile(source,out/pin['name'])
        exe=out/'BrassLegacyAI.exe'
        lines=['/nologo','/target:exe','/main:BrassLegacyAI.Program','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',f'/out:"{exe}"']
        lines += [f'/reference:"{out/f["name"]}"' for f in dependencies]+[f'"{p}"' for p in sources]
        rsp=out/'build.rsp';rsp.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
        cmd=[str(compiler)] if os.name=='nt' else [mono or 'mono',str(compiler)]
        subprocess.run(cmd+['@'+str(rsp)],check=True,cwd=root,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        require(all(sha(root/p['path'])==p['sha256'] for p in pins),'Source changed during build')
        write(out/'runtime.json',dict(format='legacy-runtime/0.1.0',player_count=4,rules=base['rules'],protocol=base['wire'],
            sources=pins,files=dependencies+[dict(name=exe.name,sha256=sha(exe))]))
    bank=freeze(root,names,runtime_directory=out)
    record=dict(format='funded-industry-v5/1.0.0',generation='v5',bank=bank,previous_bank=BASE,
        profiles=names,manifest_sha256=sha(root/'training/opponents/bank'/bank/'manifest.json'),
        runtime_directory='artifacts/legacy-ai/bank/'+bank+'/runtime',strength_promoted=False,
        installed=False,selected=False,maximum_removals=6,development_reserve=4,
        note='Official v4 strategy ancestry plus funded higher-tier targets; shared search optimizations retained. Candidate, not a strength qualification.')
    write(root/'training/opponents/candidates/industry-v5.json',record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--mono')
    args=parser.parse_args();print(json.dumps(build(args.root,args.mono),indent=2))
