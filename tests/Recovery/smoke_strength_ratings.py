"""Three isolated native held-out games; source training state/checkpoints remain read-only."""
import argparse
from pathlib import Path
import time
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import ratings
from brass_ai.recovery.plackett_luce import fit
from brass_ai.step5.league import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run=project_path(SOURCE_ROOT,a.run);out=project_path(SOURCE_ROOT,a.output)
    require(not out.exists() and not out.is_relative_to(run),'Use a new isolated output')
    protected={f:sha(f) for f in run.rglob('*') if f.is_file()}
    state=read(run/'state.json');c=state['signature']['config'];torch.set_num_threads(2)
    spec=ratings.slot_spec(SOURCE_ROOT,run,state['ratings'],state['history'][-1],c)
    spec['lineups']=[next(row for row in spec['lineups'] if row['kind']==kind) for kind in ('rating','v4-panel','il-panel')]
    spec['scope']='Bounded native integration smoke: one main, one 1-vs-3 v4, one 2-vs-2 IL game; not a rating slot'
    out.mkdir(parents=True);write_json(out/'spec.json',spec);rng=torch.get_rng_state().clone();began=time.monotonic()
    with torch.random.fork_rng(devices=[]):rows=ratings.collect_slot(SOURCE_ROOT,out,spec,c,'cpu',None,None)
    require(len(rows)==3,'Incomplete native smoke');require(torch.equal(rng,torch.get_rng_state()),'Evaluation advanced training RNG')
    ratings.append_games(out/'games.jsonl',rows)
    # A fit is executable on the single main game, but has no useful precision.
    result=fit([r for r in rows if r['kind']=='rating'],spec['agents'],c['rating'])
    panels=ratings.panel_report(rows,spec['candidate'],None,c['rating']['seed'])
    require(all(sha(path)==value for path,value in protected.items()),'Source training artifacts changed')
    report=dict(passed=True,seconds=time.monotonic()-began,games=3,
        kinds=[r['kind'] for r in rows],fit=result,panels=panels,source_files_unchanged=True,rng_unchanged=True,
        scope=spec['scope']+'; no playing-strength or confidence-interval claim')
    write_json(out/'report.json',report);print(report,flush=True)


if __name__=='__main__':main()
