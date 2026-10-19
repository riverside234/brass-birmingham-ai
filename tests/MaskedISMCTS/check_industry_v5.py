"""Focused v5 checks and four bounded legal-turn smoke probes, no installation."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from benchmark_parallel import compile_cs,game_checks
from brass_ai.bank import load_bank,read,sha
from brass_ai.host import RulesHost
from brass_ai.legacy import LegacyWorker
from brass_ai.search.legacy_runtime import source_paths


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--current-source',action='store_true',help='Validate shared source in isolation before freezing a new candidate')
    args=parser.parse_args()
    out=args.output.resolve();assert out.is_relative_to(ROOT/'artifacts') and not out.exists();out.mkdir(parents=True)
    candidate=read(ROOT/'training/opponents/candidates/industry-v5.json');bank=candidate['bank']
    load_bank(ROOT,bank);runtime=ROOT/candidate['runtime_directory']
    for pin in read(runtime/'runtime.json')['files']:shutil.copyfile(runtime/pin['name'],out/pin['name'])
    if args.current_source:
        sources=source_paths(ROOT);dependencies=[p for p in read(runtime/'runtime.json')['files'] if p['name']!='BrassLegacyAI.exe']
        compile_cs(out/'BrassLegacyAI.exe',sources,list(out.glob('*.dll')),main='BrassLegacyAI.Program')
        (out/'runtime.json').write_text(json.dumps(dict(sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sources],
            files=dependencies+[dict(name='BrassLegacyAI.exe',sha256=sha(out/'BrassLegacyAI.exe'))]),indent=2)+'\n',encoding='utf-8')
        runtime=out;bank=None
    refs=list(out.glob('*.dll'))+[out/'BrassLegacyAI.exe']
    resources=[(p,'BrassLegacyAI.Profiles.'+p.name) for p in (ROOT/'configs/legacy-ai').glob('*.json') if '-v3-' in p.name]
    reports={}
    for name in ('IndustryV5Checks',):
        sources=[ROOT/'tests/MaskedISMCTS'/(name+'.cs')]
        compile_cs(out/(name+'.exe'),sources,refs,main=name,resources=resources)
        run=subprocess.run([str(out/(name+'.exe')),str(out/(name+'.json')),str(ROOT)],cwd=ROOT,timeout=60)
        report=read(out/(name+'.json'));reports[name]=report
        print(json.dumps(dict(test=name,**report)),flush=True)
        assert run.returncode==0 and report['passed']
    for name,report in game_checks(out/'game').items():
        if isinstance(report,str):
            assert report.startswith('PASS '),report
            report=dict(passed=True,checks=int(report.split()[1]),output=report)
        reports[name]=report
    # The existing industry regressions run against the real game host, whose
    # configuration also supports two/three-player tables.
    game=out/'game'
    for name in ('IndustryChecks','EconomyChecks'):
        compile_cs(game/(name+'.exe'),[ROOT/'tests/MaskedISMCTS'/(name+'.cs')],list(game.glob('*.dll')),main=name)
        subprocess.run([str(game/(name+'.exe')),str(game/(name+'.json')),str(ROOT)],cwd=ROOT,timeout=60,check=True)
        reports[name]=read(game/(name+'.json'))
    print(json.dumps(dict(game_checks={k:r.get('checks') for k,r in reports.items()})),flush=True)
    protected=[ROOT/'training/opponents/current.json',ROOT/'training/opponents/lab/current.json',ROOT/'training/opponents/pool.json',
               ROOT/'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll']
    pins={str(p):sha(p) for p in protected if p.is_file()};rows=[]
    if args.smoke:
        positions=read(ROOT/'artifacts/v4-search-vs-fast-2s2w-20261017/final/positions.json')
        with ExitStack() as stack:
            worker=stack.enter_context(LegacyWorker(ROOT,runtime_directory=runtime,timeout=25))
            host=stack.enter_context(RulesHost(ROOT/'artifacts/r1-host/guarded'))
            for index in (0,3,5,7):
                position=positions[index];profile=worker.profile('balanced-v5-masked-industry')
                profile.update(milliseconds=2000.,simulations=100000,work_limit=1000000)
                start=time.monotonic()
                result=worker.call('plan_forest',observation=position['view'],choices=position['choices'],profile=profile,
                                   search_workers=1,sampling_seed=845510+index,intent=None)
                elapsed=time.monotonic()-start
                assert result['failures']==0 and result['search']['completed']>0
                context=host.call('restore',checkpoint=position['checkpoint'])
                for action in result['selected']['actions']:
                    for identity in action:
                        choice=next(c for c in host.choices() if c['id']==identity)
                        context=host.apply(context,choice)
                assert context['terminal'] or context['turn_owner']!=position['context']['turn_owner']
                row=dict(era=position['era'],round=position['round'],seconds=elapsed,whole_turn_legal=True,result=result)
                rows.append(row);print(json.dumps(dict(phase=row['era'],round=row['round'],seconds=elapsed,completed=result['search']['completed'])),flush=True)
    assert all(sha(Path(p))==value for p,value in pins.items())
    summary=dict(passed=True,bank=bank,current_source=args.current_source,checks=sum(x['checks'] for x in reports.values()),focused=reports,smoke=rows,
                 protected_hashes=pins,scope='Focused fixtures and four 2s single-worker roots; no playing-strength conclusion')
    (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
