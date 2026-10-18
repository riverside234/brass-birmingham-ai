"""Bounded paired search check; builds only below a fresh artifacts directory."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from brass_ai.legacy import LegacyWorker
from brass_ai.host import RulesHost

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_cs(output, sources, references, *, main=None, resources=()):
    compiler = ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    args = ['/nologo', '/langversion:7.3', '/nostdlib+', '/deterministic+', '/optimize+',
            '/target:'+('exe' if main else 'library'), '/out:"'+str(output)+'"']
    if main:
        args.append('/main:'+main)
    args += ['/reference:"'+str(p)+'"' for p in references]
    args += ['/resource:"'+str(p)+'",'+name for p, name in resources]
    args += ['"'+str(p)+'"' for p in sources]
    rsp = output.with_suffix('.rsp'); rsp.write_text('\n'.join(args), encoding='utf-8-sig')
    run=subprocess.run([str(compiler), '@'+str(rsp)], capture_output=True, text=True, timeout=60, cwd=ROOT,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if run.returncode:
        raise RuntimeError(run.stdout+'\n'+run.stderr)


def build_native(out, source):
    out.mkdir()
    frozen = ROOT/'artifacts/legacy-ai/bank/bank-3dc65253a610191d66a5/runtime'
    deps = read(ROOT/'configs/rules/runtime-dependencies-1.1.0.json')['files']
    for pin in deps:
        p = frozen/pin['name']; assert sha(p) == pin['sha256']
        shutil.copy2(p, out/p.name)
    sources = sorted(p for folder in ('BrassHost', 'BrassLegacyAI', 'BrassMaskedSearch')
                     for p in (source/folder).glob('*.cs')
                     if folder != 'BrassHost' or p.name != 'Program.cs')
    compile_cs(out/'BrassLegacyAI.exe', sources, sorted(out.glob('*.dll')), main='BrassLegacyAI.Program')
    write(out/'runtime.json', dict(sources=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in sources],
        files=deps+[dict(name='BrassLegacyAI.exe', sha256=sha(out/'BrassLegacyAI.exe'))]))


def game_checks(out, *, include_candidates=False):
    """Compile the shared search against the actual three/four-player game copy."""
    out.mkdir()
    staged = ROOT/'artifacts/patch-build/Managed'
    for p in staged.glob('*.dll'):
        if p.name != 'BrassLegacyAI.dll' and not any(x in p.name for x in ('Tests', 'Regression')):
            shutil.copy2(p, out/p.name)
    sources = sorted((ROOT/'src/BrassLocalAI').rglob('*.cs'))+sorted((ROOT/'src/BrassMaskedSearch').glob('*.cs'))
    resources = []
    for style in ('balanced', 'rail-tempo', 'cotton', 'goods-beer', 'supplier', 'pottery-flex'):
        for suffix in ('v2', 'v3', 'v3-masked', 'v3-masked-strength', 'v3-masked-era',
                       'v3-masked-industry', 'v3-masked-economy'):
            name=style+'-'+suffix+'.json'
            resources.append((ROOT/'configs/legacy-ai'/name, 'BrassLegacyAI.Profiles.'+name))
        resources.append((ROOT/'configs/legacy-ai/archive/1.4.0'/(style+'-v3.json'),
                          'BrassLegacyAI.Profiles.'+style+'-v3-1.4.0.json'))
    compile_cs(out/'BrassLegacyAI.dll', sources, sorted(out.glob('*.dll')), resources=resources)
    results = {}
    # Network is an archived, unselected candidate, not the official v4 path.
    names=['ParallelChecks','RuntimeChecks']
    if include_candidates:names.append('NetworkOptionsChecks')
    for name in names:
        exe=out/(name+'.exe')
        compile_cs(exe, [ROOT/'tests/MaskedISMCTS'/(name+'.cs')], sorted(out.glob('*.dll')), main=name)
        report=out/(name+'.json')
        run=subprocess.run([str(exe), str(report), str(ROOT)], capture_output=True, text=True, check=True, timeout=60, cwd=ROOT,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        results[name]=read(report) if report.exists() else run.stdout.strip()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    args=parser.parse_args(); out=args.output.resolve(); baseline=args.baseline_source.resolve()
    assert out.is_relative_to(ROOT/'artifacts') and not out.exists()
    assert baseline.is_relative_to(ROOT/'artifacts') and (baseline/'BrassMaskedSearch/MaskedSearch.cs').exists()
    out.mkdir(parents=True); began=time.monotonic()
    build_native(out/'baseline', baseline); build_native(out/'candidate', ROOT/'src')
    checks=game_checks(out/'game-checks'); write(out/'checks.json', checks)
    print(json.dumps(dict(event='compiled_and_checked', checks=list(checks))), flush=True)
    fixtures=read(ROOT/'artifacts/v4-search-vs-fast-2s2w-20261017/final/positions.json')
    # One predetermined root from each phase, alternating matched execution order.
    positions=[fixtures[i] for i in (0, 3, 5, 7)]; records=[]; fixed=[]
    protected=[ROOT/'training/opponents/current.json', ROOT/'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll']
    pins={str(p):sha(p) for p in protected if p.exists()}
    with ExitStack() as stack:
        workers={name:stack.enter_context(LegacyWorker(ROOT, runtime_directory=out/name, timeout=20))
                 for name in ('baseline', 'candidate')}
        host=stack.enter_context(RulesHost(ROOT/'artifacts/r1-host/guarded'))
        def validate(position, result):
            assert result['failures']==0 and result['search']['opponent_nodes']==0
            context=host.call('restore', checkpoint=position['checkpoint'])
            for action in result['selected']['actions']:
                for identity in action:
                    choice=next(c for c in host.choices() if c['id']==identity)
                    context=host.apply(context, choice)
            fields=('era', 'round', 'turn', 'turn_owner')
            assert context['terminal'] or tuple(context[k] for k in fields)!=tuple(position['context'][k] for k in fields)
        for index, position in enumerate(positions):
            profile=read(
                ROOT/'configs/legacy-ai'/('balanced-v3-masked-economy.json' if index%2==0 else 'rail-tempo-v3-masked-economy.json'))
            profile.update(milliseconds=2000., work_limit=1000000, simulations=8)
            results={name:w.call('plan', observation=position['view'], choices=position['choices'], profile=profile,
                                fixed_work=True, sampling_seed=717100+index, intent=None) for name,w in workers.items()}
            for result in results.values():validate(position,result)
            a,b=results.values()
            assert a['selected']==b['selected'] and a['search']['root']==b['search']['root']
            assert a['search']['plans']==b['search']['plans'] and a['search']['completed']==b['search']['completed']==8
            assert b['search']['deadline_stops']==0
            fixed.append(dict(position=index, completed=8, exact_fixed_work_parity=True))
            profile['simulations']=100000
            for repeat in range(2):
                assert time.monotonic()-began<180, 'Bounded smoke deadline'
                order=list(workers) if (index+repeat)%2==0 else list(reversed(workers))
                for name in order:
                    start=time.monotonic()
                    result=workers[name].call('plan_forest', observation=position['view'], choices=position['choices'],
                        profile=profile, search_workers=2, sampling_seed=717100+index, intent=None)
                    elapsed=time.monotonic()-start;validate(position,result);search=result['search']
                    record=dict(position=index, phase=('canal_early','canal_late','rail_early','rail_late')[index],
                        repeat=repeat, variant=name, seconds=elapsed, result=result, whole_turn_legal=True)
                    records.append(record)
                    print(json.dumps(dict(position=index, repeat=repeat, variant=name, completed=search['completed'],
                        interrupted=search['interrupted'], seconds=elapsed, tail_stops=search.get('deadline_stops',0))), flush=True)
    summaries={}
    for name in workers:
        rows=[r for r in records if r['variant']==name]
        summaries[name]=dict(decisions=len(rows), completed=sum(r['result']['search']['completed'] for r in rows),
            interrupted=sum(r['result']['search']['interrupted'] for r in rows),
            deadline_stops=sum(r['result']['search'].get('deadline_stops',0) for r in rows),
            mean_seconds=statistics.mean(r['seconds'] for r in rows),
            root_alternatives=sum(len(r['result']['search']['root']) for r in rows),
            supported_plans=sum(r['result']['search']['revisited_plans'] for r in rows))
    assert all(sha(Path(p))==value for p,value in pins.items())
    write(out/'report.json', dict(scope='Four roots, two repeats; current source before/after; no strength evaluation',
        summaries=summaries, fixed_work=fixed, records=records, protected_hashes=pins,
        total_seconds=time.monotonic()-began))
    print(json.dumps(summaries), flush=True)


if __name__=='__main__':main()
