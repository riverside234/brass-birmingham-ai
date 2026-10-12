"""Compile/run the staged game bridge in a private native fixture directory."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from brass_ai.learning.contracts import require


def run(build,out,save=None):
    out=Path(out).resolve();require(not out.exists(),'New test directory required');out.mkdir(parents=True)
    for p in (ROOT/'BrassBirminghamUnity_Data/Managed').glob('*.dll'):shutil.copyfile(p,out/p.name)
    for p in (Path(build)/'Managed').glob('*.dll'):shutil.copyfile(p,out/p.name)
    args=['/nologo','/target:exe','/langversion:7.3','/nostdlib+',f'/out:"{out}/NativeLaptopChecks.exe"']
    args += [f'/reference:"{p}"' for p in out.glob('*.dll') if 'Tests' not in p.name and 'Regression' not in p.name]
    args += [f'"{ROOT}/tests/Personality/NativeLaptopChecks.cs"']
    rsp=out/'test.rsp';rsp.write_text('\n'.join(args),encoding='utf-8-sig')
    compiler=ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    subprocess.run([str(compiler),'@'+str(rsp)],check=True)
    subprocess.run([str(out/'NativeLaptopChecks.exe'),str(out/'report.json'),'--brass-local-ai-test']+([str(Path(save).resolve())] if save else []),check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--save',type=Path);a=p.parse_args();run(a.build,a.output,a.save)
