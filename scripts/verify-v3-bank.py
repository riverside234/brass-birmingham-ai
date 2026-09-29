"""Bind v3 conversion/cash fixtures and a reproducible build to a frozen bank.

Run after check-frozen-bank.py and both deployment/research C# fixture runners.
This produces correctness evidence, never playing-strength admission.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT, below, load_bank, read, require, sha
from brass_ai.qualification import STYLES, write_new


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=LAB_ROOT)
    parser.add_argument('--bank',required=True)
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--mono')
    args=parser.parse_args();root=args.root.resolve()
    directory=below(root,'training/opponents/evaluations/'+args.evaluation)
    spec=read(directory/'spec.json')
    require(spec['payload']['bank']==args.bank,'Evaluation belongs to another bank')
    manifest=load_bank(root,args.bank);payload=manifest['payload']
    require(set(payload['profiles'])=={s+'-v3' for s in STYLES},'Six v3 profiles required')
    for profile in payload['profiles'].values():
        require(profile['cash_policy']==profile['conversion_policy']==1,'Missing v3 policies')
    platform=read(directory/'platform-check.json')
    require(platform['passed'] and platform['bank']==args.bank and platform['tests']>=34,
            'Frozen-worker correctness suite incomplete')
    build=root/'artifacts/patch-build'
    candidate=read(build/'build-manifest.json')
    require(candidate['version']=='1.4.1','Wrong deployment package')
    evidence={}
    def pin(path):evidence[path.relative_to(root).as_posix()]=sha(path)
    pairs=[]
    for suite,count in [('conversion',54),('cash-beer',12),('credit',12)]:
        local=read(build/f'local-ai-{suite}-0.json')
        research=read(build/f'local-ai-{suite}-1.json')
        require(local['passed'] and research['passed'] and local['cases']==research['cases']
                and len(local['cases'])==count,'Incomplete or divergent '+suite+' fixtures')
        for report,files in [(local,candidate['files']),(research,payload['runtime_files'])]:
            actual={f['name']:f['sha256'].lower() for f in report['testedFiles']}
            require(all(actual.get(f['name'])==f['sha256'].lower() for f in files),
                    'Fixture runtime fingerprint mismatch')
        for variant in (0,1):pin(build/f'local-ai-{suite}-{variant}.json')
        pairs.append(dict(suite=suite,checks_each=local['checks'],identical_scenarios=count))
    local=read(build/'local-ai-demand-0.json');research=read(build/'local-ai-demand-1.json')
    require(local['passed'] and research['passed'] and local['checks']==research['checks']
            and local['checks']>=19 and local['signature']==research['signature']
            and local['cases']==research['cases'] and {c['work'] for c in local['cases']}=={64,240},'Demand regression failed')
    for report,files in [(local,candidate['files']),(research,payload['runtime_files'])]:
        actual={f['name']:f['sha256'].lower() for f in report['testedFiles']}
        require(all(actual.get(f['name'])==f['sha256'].lower() for f in files),'Demand runtime mismatch')
    for variant in (0,1):pin(build/f'local-ai-demand-{variant}.json')
    pairs.append(dict(suite='demand',checks_each=local['checks'],identical_scenarios=2))
    local=read(build/'local-ai-route-0.json');research=read(build/'local-ai-route-1.json')
    require(local['passed'] and research['passed'] and local['checks']==research['checks']
            and local['checks']>=24 and local['cases']==research['cases']
            and {c['work'] for c in local['cases']}=={64,240},'Merchant-route regression failed')
    for report,files in [(local,candidate['files']),(research,payload['runtime_files'])]:
        actual={f['name']:f['sha256'].lower() for f in report['testedFiles']}
        require(all(actual.get(f['name'])==f['sha256'].lower() for f in files),'Route runtime mismatch')
    for variant in (0,1):pin(build/f'local-ai-route-{variant}.json')
    pairs.append(dict(suite='route',checks_each=local['checks'],identical_scenarios=2))
    snapshot=root/'training/opponents/bank'/args.bank/'source'
    runtime=root/'artifacts/legacy-ai/bank'/args.bank/'runtime'
    output=root/'artifacts/legacy-ai'/args.bank/'rebuild-check'
    output.mkdir(parents=True,exist_ok=True)
    compiler=root/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    require(sha(compiler)==payload['compiler_sha256'],'Frozen compiler changed')
    for f in payload['runtime_files']:
        require(sha(runtime/f['name'])==f['sha256'],'Frozen runtime changed')
        if f['name'].endswith('.dll'):shutil.copyfile(runtime/f['name'],output/f['name'])
    executable=output/'BrassLegacyAI.exe'
    options=['/nologo','/target:exe','/main:BrassLegacyAI.Program','/langversion:7.3',
             '/nostdlib+','/deterministic+','/optimize+',f'/out:"{executable}"']
    options += [f'/reference:"{output/f["name"]}"' for f in payload['runtime_files'] if f['name'].endswith('.dll')]
    options += [f'"{snapshot/p}"' for p in payload['runtime_source_paths']]
    response=output/'build.rsp';response.write_text('\n'.join(options)+'\n',encoding='utf-8-sig')
    command=([args.mono] if args.mono else [])+[str(compiler),'@'+str(response)]
    subprocess.run(command,check=True,cwd=root)
    require(sha(executable)==sha(runtime/'BrassLegacyAI.exe'),'Frozen rebuild differs')
    for path in ['tests/BrassLocalAI/ConversionTests.cs','tests/BrassLocalAI/CashBeerTests.cs','tests/BrassLocalAI/DemandTests.cs','tests/BrassLocalAI/CreditTests.cs',
                 'tests/BrassLocalAI/RouteTests.cs','tests/Qualification/test_economic_admission.py','tests/legacy-ai/test_v3.py','tests/legacy-ai/test_bank.py','scripts/check-frozen-bank.py',
                 'scripts/amend-v3-qualification.py','scripts/promote-v3-bank.py',
                 'scripts/verify-v3-bank.py','src/brass_ai/qualification.py']:
        pin(root/path)
    pin(directory/'platform-check.json')
    environment=dict(os.environ,PYTHONPATH=str(root/'src'))
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/Qualification','-v'],
                   cwd=root,env=environment,check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/legacy-ai','-p','test_bank.py','-v'],
                   cwd=root,env=environment,check=True)
    report=dict(format='legacy-v3-correctness/1.0.0',passed=True,bank=args.bank,
                bank_manifest_sha256=sha(snapshot.parent/'manifest.json'),tests=platform['tests'],
                fixtures=pairs,worker_sha256=sha(executable),identical_snapshot_rebuild=True,
                economic_admission_regressions=2,bank_admission_regressions=5,
                platform=sys.platform,evidence=evidence,
                scope='Frozen contracts and identical deadline/merchant/cash decisions; no strength admission.')
    write_new(directory/'checks.json',report)
    print(report)


if __name__=='__main__':main()
