"""Short saved-trace regression against the installed build and a detached fix.

No game installation, save mutation, search session or training work is run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/search-delivery-grace/20261029/release-1.4.45'
CSC=ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
HOST=ROOT/'artifacts/rust-backend/20261026-local-release/allocation-final/Managed/allocation-host.exe'
SOURCE=Path('src/BrassPersonalityGame/PublicHistory.cs')


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def compile_assembly(rsp):
    result=subprocess.run([str(CSC),'@'+str(rsp)],cwd=ROOT,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    rsp.with_suffix('.log').write_text(result.stdout,encoding='utf-8')
    if result.returncode: raise RuntimeError(result.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',type=Path,required=True,help='Read-only source events.storage with the reported bonus sale')
    a=p.parse_args();original=sha(a.events)
    out=ROOT/'artifacts/public-history-bonus'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    installed_before={name:sha(ROOT/'BrassBirminghamUnity_Data/Managed'/name)
        for name in ('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')}
    report=dict(passed=False,installed=False,events_sha256=original,source_sha256=sha(ROOT/SOURCE),results={})
    try:
        baseline=read(BASE/'build-manifest.json')
        for item in baseline['files']:
            assert sha(BASE/'Managed'/item['name'])==item['sha256']
            assert sha(ROOT/'BrassBirminghamUnity_Data/Managed'/item['name'])==item['sha256'],'Installed managed baseline differs'
        source_pin=next(item for item in baseline['sourcePins'] if item['path']==SOURCE.as_posix())
        assert sha(BASE/'runtime'/SOURCE)==source_pin['sha256']
        assert sha(HOST)==read(HOST.parent.parent/'receipt.json')['host']['sha256']
        shutil.copyfile(a.events,out/'events.storage')
        for arm in ('baseline','candidate'):
            work=out/arm;work.mkdir()
            managed=BASE/'Managed'
            if arm=='candidate':
                managed=work/'Managed';shutil.copytree(BASE/'Managed',managed)
                # Reuse the exact installed sources/resources; replace only the
                # corrected history parser in this isolated compilation.
                rsp=(BASE/'legacy.rsp').read_text(encoding='utf-8-sig')
                rsp=rsp.replace(str(BASE/'Managed'),str(managed))
                rsp=rsp.replace(str(BASE/'runtime'/SOURCE),str(ROOT/SOURCE))
                assert str(ROOT/SOURCE) in rsp
                (work/'legacy.rsp').write_text(rsp,encoding='utf-8-sig')
                compile_assembly(work/'legacy.rsp')
            exe=work/'PublicHistoryBonusChecks.exe'
            opts=['/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',f'/out:"{exe}"']
            opts += [f'/reference:"{dll}"' for dll in managed.glob('*.dll') if not any(n in dll.name for n in ('Tests','Regression'))]
            opts += [f'"{ROOT}/tests/Personality/PublicHistoryBonusChecks.cs"']
            (work/'tests.rsp').write_text('\n'.join(opts),encoding='utf-8-sig')
            compile_assembly(work/'tests.rsp')
            command=[str(HOST),str(ROOT/'MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll'),
                str(managed),str(ROOT/'MonoBleedingEdge/etc'),str(exe),str(out/'events.storage'),arm,str(work/'result.json')]
            with (work/'test.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
            result=read(work/'result.json');assert result['passed'],result
            report['results'][arm]=result
            print(arm,result['checks'],'checks passed',flush=True)
        report['passed']=True
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        report['source_events_unchanged']=sha(a.events)==original
        report['installed_managed_unchanged']=all(sha(ROOT/'BrassBirminghamUnity_Data/Managed'/name)==digest for name,digest in installed_before.items())
        (out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(out/'qualification.json',flush=True)
        assert report['source_events_unchanged'] and report['installed_managed_unchanged']


if __name__=='__main__': main()
