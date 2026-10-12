"""Build an isolated explicit-chance host; never overwrite the R1/Step 5 runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def search_source(source):
    edits=[
        ('Game=other.Game;', 'AdoptChance(other); Game=other.Game;'),
        ('if(Terminal){Phase="terminal";return;}', 'if(ChancePending)return; if(Terminal){Phase="terminal";return;}'),
        ('Game.GameFlow.EndRound();', 'if(TryPauseRail())return; Game.GameFlow.EndRound();'),
        ('Game.GameFlow.EndTurn();', 'if(TryPauseDraw())return; Game.GameFlow.EndTurn();'),
        ('["format"]="brass-trusted-checkpoint",', '["explicit_chance"]=ExplicitChance,["format"]="brass-trusted-checkpoint",'),
        ('h.GameId=(string)cp["game_id"];', 'h.GameId=(string)cp["game_id"]; h.ExplicitChance=(bool?)cp["explicit_chance"]??false;'),
        ('var c=h.Choices().SingleOrDefault', 'if(cmd["chance"]!=null){h.ResolveChance((string)cmd["chance"]);continue;} var c=h.Choices().SingleOrDefault'),
        ('Wire.Hash(Canonical())', 'FullHash()'),
        ('Wire.Hash(h.Canonical())', 'h.FullHash()')]
    for old,new in edits:
        assert source.count(old)==1, 'R1 hook changed: '+old
        source=source.replace(old,new)
    return source

def build(root=ROOT, mono=None):
    base=root/'artifacts/r1-host/guarded'
    out=root/'artifacts/r4-host';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((base/'runtime.json').read_text(encoding='utf-8-sig'))
    for f in manifest['files']:
        assert sha(base/f['name'])==f['sha256'], 'Changed R1 runtime'
        if f['name']!='BrassHost.exe':shutil.copyfile(base/f['name'],out/f['name'])
    source=search_source((root/'src/BrassHost/Host.cs').read_text(encoding='utf-8-sig'))
    generated=out/'Host.cs';generated.write_text(source,encoding='utf-8',newline='\n')
    sources=[generated]+[root/'src/BrassHost'/n for n in ('Choices.cs','Model.cs','Observation.cs','Position.cs')]
    sources+=sorted((root/'src/BrassSearchHost').glob('*.cs'))
    compiler=root/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    args=['/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',f'/out:"{out / "BrassHost.exe"}"']
    args += [f'/reference:"{p}"' for p in sorted(out.glob('*.dll'))]+[f'"{p}"' for p in sources]
    rsp=out/'build.rsp';rsp.write_text('\n'.join(args),encoding='utf-8-sig')
    subprocess.run(([str(compiler)] if os.name=='nt' else [mono or 'mono',str(compiler)])+['@'+str(rsp)],check=True)
    manifest['sources']=[dict(name=p.relative_to(root).as_posix(),sha256=sha(p)) for p in sources]
    manifest['extension']='brass-search-chance/1.0.0'
    manifest['files']=[dict(name=p.name,sha256=sha(p)) for p in sorted(out.glob('*.dll'))]+[dict(name='BrassHost.exe',sha256=sha(out/'BrassHost.exe'))]
    (out/'runtime.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    target=root/'training/runtime/search';target.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(out/'BrassHost.exe',target/'BrassHost.exe')
    shutil.copyfile(out/'runtime.json',target/'runtime.json')
    return dict(runtime=str(out),host_sha256=sha(out/'BrassHost.exe'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--mono')
    a=p.parse_args();print(json.dumps(build(a.root.resolve(),a.mono),indent=2))
