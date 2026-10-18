"""One ordinary v4 table and four scheduled-panel rotations; no optimizer work."""
import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT, project_path, require
from brass_ai.recovery.config import latest_model_config, table_kind
from brass_ai.recovery.execution import ActorPool
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.legacy_search import legacy_spec
from brass_ai.recovery.ordinary import plan_collection
from brass_ai.recovery import evaluation
from brass_ai.step5.league import write_json, implementation_id


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='Existing small diagnostic run, read only')
    p.add_argument('--output',required=True)
    a=p.parse_args();run=project_path(SOURCE_ROOT,a.run);out=project_path(SOURCE_ROOT,a.output)
    require(not out.exists() and not out.is_relative_to(run),'Use a new isolated output')
    protected={path:sha(path) for path in run.rglob('*') if path.is_file()}
    state=read(run/'state.json');c=latest_model_config(state['signature']['config'],state['signature']['imitation_source'])
    c.update(games_per_iteration=1,threads=2);c['retention']['enabled']=False
    c['execution'].update(processes=1,hosts_per_process=2,heavy_processes=1)
    out.mkdir(parents=True);write_json(out/'config.json',c)
    torch.set_num_threads(2);rng=torch.get_rng_state().clone();began=time.monotonic();records=[]
    def audit(batch):
        for r in batch:
            for row in r['search'].values():
                if 'search' not in row:continue
                s=row['search'];require(s['search_free'],'A v4 seat enabled search')
                require(s['turns']>0,'No actual v4 turns')
                require(all(s[k]==0 for k in ('completed','attempted_simulations','allocated_simulations')),
                        'A v4 seat performed MCTS')
        records.extend(batch)
    class AuditedPool(ActorPool):
        def collect(self,jobs,**kwargs):
            batch=super().collect(jobs,**kwargs);audit(batch);return batch
    with torch.random.fork_rng(devices=[]):
        ordinary_state=copy.deepcopy(state)
        ordinary_state['games_done']=next(i for i in range(100) if table_kind(i,c['curriculum'])=='legacy')
        profiles=sorted(read(SOURCE_ROOT/'training/opponents/pool.json')['training'])
        fast={p:legacy_spec(SOURCE_ROOT,p,c['legacy_budget'],search_free=True,
                          work_budget=c['legacy_opponent_work_budget']) for p in profiles}
        slow={p:legacy_spec(SOURCE_ROOT,p,c['legacy_budget'],
                          work_budget=c['legacy_opponent_work_budget']) for p in profiles}
        current=state['history'][-1]
        with InferenceService(c['execution'],'cpu') as service:
            service.register_checkpoint(project_path(SOURCE_ROOT,current['path']))
            jobs=plan_collection(SOURCE_ROOT,out/'ordinary',ordinary_state,c,dict(config=c),service,
                current['model_id'],profiles,slow,fast,None)['jobs']
            require(jobs[0]['provenance']['legacy_mode']=='fast','Ordinary plan did not use fast v4')
            with AuditedPool(SOURCE_ROOT,c['host'],service,c['execution']) as actor:actor.collect(jobs)
        print('Ordinary v4 table completed with zero MCTS simulations',flush=True)
        with patch.object(evaluation,'ActorPool',AuditedPool):
            panel=evaluation.evaluate(SOURCE_ROOT,out/'evaluation',current['path'],
                [state['evaluation_anchor']['path']],c,setups=1,seed=91831,search=False,lineups=['mixed-v3'])
    require(torch.equal(rng,torch.get_rng_state()),'Diagnostic advanced caller RNG')
    require(all(sha(path)==value for path,value in protected.items()),'Source run changed')
    require(len(records)==5 and panel['games']==4,'Smoke exceeded its five-game bound')
    telemetry=[v['search'] for r in records for v in r['search'].values() if 'search' in v]
    require(len(telemetry)==14,'Missing v4 seat telemetry')
    result=dict(passed=True,implementation=implementation_id(),games=5,ordinary_games=1,evaluation_games=4,
        v4_seat_records=len(telemetry),allocated_simulations=sum(s['allocated_simulations'] for s in telemetry),
        attempted_simulations=sum(s['attempted_simulations'] for s in telemetry),
        completed_simulations=sum(s['completed'] for s in telemetry),source_files_unchanged=True,rng_unchanged=True,
        optimizer_steps=0,seconds=time.monotonic()-began,
        scope='Native search-disable integration only; no throughput comparison or playing-strength claim')
    write_json(out/'records.json',records);write_json(out/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
