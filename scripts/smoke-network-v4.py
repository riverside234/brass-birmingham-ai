"""Bounded v4 network extension vs selected v4 and economic v3, equal budgets."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import os
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT, load_bank, read, sha
from brass_ai.learning.contracts import project_path, require
from brass_ai.recovery.network_v4_bank import CONTROLS, ADMISSION
from brass_ai.step5.league import write_json

spec=importlib.util.spec_from_file_location('_network_screen',Path(__file__).with_name('smoke-industry-v3.py'))
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)


def throughput(rows,role):
    turns=[t for r in rows for t in r['turns'] if r['lineup'][t['seat']][0]==role]
    if not turns:return {}
    seconds=sum(t['seconds'] for t in turns);completed=sum(t['completed'] for t in turns)
    return dict(decisions=len(turns),mean_nodes=statistics.mean(t['nodes'] for t in turns),
        mean_completed=completed/len(turns),completed_per_second=completed/seconds,
        mean_turn_seconds=seconds/len(turns),p95_turn_seconds=sorted(t['seconds'] for t in turns)[int(.95*(len(turns)-1))],
        network_attempted=sum(t['network']['network_candidates'] for t in turns),
        network_completed=sum(t['network']['network_completed'] for t in turns),
        network_rejected=sum(t['network']['network_rejected'] for t in turns),
        mean_extra_root_worker_ms=sum(t['search_timing_ms'].get('network_candidate',0) for t in turns)/(4*len(turns)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--bank')
    p.add_argument('--output',default='artifacts/v4-network/smoke-1');p.add_argument('--tables',type=int,default=2)
    p.add_argument('--seed',type=int,default=844000);p.add_argument('--mono');a=p.parse_args();root=a.root.resolve()
    bank=a.bank or read(root/'artifacts/v4-network/candidate-bank.json')['bank'];candidate=load_bank(root,bank)
    require(1<=a.tables<=24,'Invalid table count');out=project_path(root,a.output);require(not out.exists(),'Preserve prior screen');out.mkdir(parents=True)
    for control in CONTROLS:load_bank(root,control);(out/control).mkdir()
    rows=[];failures=[]
    with ThreadPoolExecutor(max_workers=a.tables) as pool:
        jobs={pool.submit(screen.play,root,out,bank,c,block*4+r,a.seed+block,r,a.mono):(c,block*4+r)
              for c in CONTROLS for block in range(3) for r in range(4)}
        for future in as_completed(jobs):
            try:rows.append(future.result())
            except Exception as error:failures.append(dict(control=jobs[future][0],index=jobs[future][1],error=str(error)))
            print(dict(completed=len(rows),failures=failures,total=24),flush=True)
    comparisons={}
    for control in CONTROLS:
        group=[r for r in rows if r['control']==control];policies={role:screen.summarize(group,role) for role in ('new','old')}
        n,o=policies['new'],policies['old'];improved=bool(n and o and n['win_share']>o['win_share'] and n['first_places']>o['first_places'] and n['mean_vp']>o['mean_vp'] and n['mean_rank']<o['mean_rank'])
        speeds={role:throughput(group,role) for role in ('new','old')}
        old=speeds['old'].get('completed_per_second',0)
        regression=(1-speeds['new']['completed_per_second']/old)*100 if old else None
        comparisons[control]=dict(games=len(group),runtime_files=load_bank(root,control)['payload']['runtime_files'],policies=policies,
            all_metrics_improved=improved,eligible=len(group)==12 and improved,throughput=speeds,completed_per_second_regression_percent=regression)
    passed=len(rows)==24 and not failures;promote=passed and all(c['eligible'] for c in comparisons.values())
    report=dict(format=ADMISSION,passed=passed,promote=promote,games=len(rows),failures=failures,new_bank=bank,
        new_runtime_files=candidate['payload']['runtime_files'],comparisons=comparisons,seconds=1.,search_workers=4,
        local_tables=a.tables,lab_tables=24,seed=a.seed,logical_cpus=os.cpu_count(),
        selection_rule='Against EACH current-v4/economic-v3 control: strictly more official wins/first places, higher mean VP AND lower mean numeric rank. All 24 games complete without errors.',
        evidence={r['control']+f'/game-{r["index"]:02d}.json':sha(out/r['control']/f'game-{r["index"]:02d}.json') for r in rows},
        runner_inputs={Path(f).name:sha(Path(f)) for f in (__file__,screen.__file__)},
        limitation='Three predetermined setups and four seat rotations per control; small directional screen, not established human/tournament strength.')
    write_json(out/'report.json',report);print(dict(passed=passed,promote=promote,games=len(rows),bank=bank,comparisons=comparisons),flush=True)
    require(passed,'Screen failed')


if __name__=='__main__':main()
