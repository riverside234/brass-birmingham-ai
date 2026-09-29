"""Rebuild a frozen worker from original source and separately supplied pinned DLLs.

Run under Python on Windows, or with --mono on the lab. All game dependencies
remain private. A compiler/runtime mismatch fails instead of changing the identity.
"""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT, below, load_bank, sha, require
from brass_ai.qualification import write_new

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--bank',required=True)
    p.add_argument('--managed',required=True);p.add_argument('--compiler',default='artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe');p.add_argument('--mono')
    a=p.parse_args();root=a.root.resolve();payload=load_bank(root,a.bank)['payload']
    compiler=below(root,a.compiler);managed=below(root,a.managed)
    require(sha(compiler)==payload['compiler_sha256'],'Frozen compiler changed')
    output=below(root,'artifacts/legacy-ai/bank/'+a.bank+'/runtime')
    require(not (output/'runtime.json').exists(),'Frozen runtime already exists; verify/use it without rebuilding')
    output.mkdir(parents=True,exist_ok=True)
    for f in payload['runtime_files']:
        if f['name']=='BrassLegacyAI.exe':continue
        source=below(managed,f['name']);require(sha(source)==f['sha256'],'Private dependency mismatch: '+f['name'])
        shutil.copyfile(source,output/f['name'])
    snapshot=below(root,'training/opponents/bank/'+a.bank+'/source')
    sources=[below(snapshot,path) for path in payload['runtime_source_paths']]
    exe=output/'BrassLegacyAI.exe';rsp=output/'build.rsp'
    lines=['/nologo','/target:exe','/main:BrassLegacyAI.Program','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',f'/out:"{exe}"']
    lines += [f'/reference:"{output/f["name"]}"' for f in payload['runtime_files'] if f['name'].endswith('.dll')]
    lines += [f'"{path}"' for path in sources];rsp.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    cmd=[str(compiler),'@'+str(rsp)] if os.name=='nt' else [a.mono or 'mono',str(compiler),'@'+str(rsp)]
    subprocess.run(cmd,check=True,cwd=root)
    expected=next(f['sha256'] for f in payload['runtime_files'] if f['name']=='BrassLegacyAI.exe')
    require(sha(exe)==expected,'Build differs from frozen binary; do not use this as the qualified opponent')
    runtime=dict(format='legacy-runtime/0.1.0',player_count=4,rules=payload['rules'],protocol=payload['wire'],files=payload['runtime_files'],
                 sources=[dict(path=(snapshot/path).relative_to(root).as_posix(),sha256=payload['source_files'][path]) for path in payload['runtime_source_paths']])
    write_new(output/'runtime.json',runtime);print('Verified '+a.bank)

if __name__=='__main__':main()
