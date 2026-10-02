"""Matched four-player games; both frozen controls must lose on every metric."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT, load_bank, read, sha
from brass_ai.learning.contracts import project_path, require
from brass_ai.recovery.economy_bank import CONTROLS
from brass_ai.step5.league import write_json

spec = importlib.util.spec_from_file_location('_industry_screen', Path(__file__).with_name('smoke-industry-v3.py'))
screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=LAB_ROOT); p.add_argument('--bank')
    p.add_argument('--output',default='artifacts/economy-v3/smoke-1'); p.add_argument('--tables',type=int,default=4)
    p.add_argument('--seed',type=int,default=843200); p.add_argument('--mono'); a=p.parse_args(); root=a.root.resolve()
    bank=a.bank or read(root/'artifacts/economy-v3/candidate-bank.json')['bank']; candidate=load_bank(root,bank)
    require(1<=a.tables<=24,'Invalid table count'); out=project_path(root,a.output)
    require(not out.exists(),'Preserve earlier evidence'); out.mkdir(parents=True)
    for control in CONTROLS: load_bank(root,control); (out/control).mkdir()
    rows=[]; failures=[]
    with ThreadPoolExecutor(max_workers=a.tables) as pool:
        jobs={pool.submit(screen.play,root,out,bank,c,block*4+r,a.seed+block,r,a.mono):(c,block*4+r)
              for c in CONTROLS for block in range(3) for r in range(4)}
        for future in as_completed(jobs):
            try: rows.append(future.result())
            except Exception as e: failures.append(dict(control=jobs[future][0],index=jobs[future][1],error=str(e)))
            print(dict(completed=len(rows),failures=failures,total=24),flush=True)
    comparisons={}
    for control in CONTROLS:
        group=[r for r in rows if r['control']==control]; policies={role:screen.summarize(group,role) for role in ('new','old')}
        n,o=policies['new'],policies['old']; wins=bool(n and o and n['win_share']>o['win_share'])
        all_metrics=bool(wins and n['first_places']>o['first_places'] and n['mean_vp']>o['mean_vp'] and n['mean_rank']<o['mean_rank'])
        comparisons[control]=dict(games=len(group),runtime_files=load_bank(root,control)['payload']['runtime_files'],
            policies=policies,more_wins=wins,all_metrics_improved=all_metrics,
            eligible=len(group)==12 and all_metrics)
    passed=len(rows)==24 and not failures; promote=passed and all(c['eligible'] for c in comparisons.values())
    report=dict(format='masked-economy-smoke/1.3.0',passed=passed,promote=promote,games=len(rows),failures=failures,
        new_bank=bank,new_runtime_files=candidate['payload']['runtime_files'],comparisons=comparisons,
        seconds=1.,search_workers=4,search_backend='masked_ismcts',local_tables=a.tables,lab_tables=24,seed=a.seed,
        logical_cpus=os.cpu_count(),winner='new' if promote else 'not-promoted',
        selection_rule='Previous pre-network revision AND economic control: strictly more first places/official wins, higher mean VP AND lower mean numeric rank against EACH. All 24 games complete without errors.',
        evidence={r['control']+f'/game-{r["index"]:02d}.json':sha(out/r['control']/f'game-{r["index"]:02d}.json') for r in rows},
        runner_inputs={Path(f).name:sha(Path(f)) for f in (__file__,screen.__file__)},
        limitation='Three fresh setups with four seat rotations per control. Small directional screen, not a statistically established win-rate advantage.')
    write_json(out/'report.json',report)
    print(dict(passed=passed,promote=promote,games=len(rows),bank=bank,
        comparisons={b:{k:v for k,v in c.items() if k!='runtime_files'} for b,c in comparisons.items()}),flush=True)
    require(passed,'Smoke failed')


if __name__=='__main__': main()
