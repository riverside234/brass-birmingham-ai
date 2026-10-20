"""Recover an isolated partial queued collection with a fresh worker pool."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import time

import torch
from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery.execution import ActorPool
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.ingress import StreamingCollection,reader_pool
from brass_ai.recovery.replay import read_shards,cache_receipt
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.step5.league import write_json
from smoke_branch_scheduling import signature,compare
from test_exploration_revision import config


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve()
    require(source.is_relative_to(SOURCE_ROOT/'artifacts') and
        out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use isolated artifacts')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(73043);began=time.monotonic()
    cfg=config();model=FourSeatNet(ModelConfig.preset('medium'))
    optimizer=torch.optim.AdamW(model.parameters());migrate(model,optimizer,reward_spec(cfg));model.eval()
    jobs=[];expected={};children={};files={};omitted=None;ex=None
    for path in sorted((source/'queued').glob('*.jsonl.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as stream:header=json.loads(next(stream))
        recipe=header['provenance']['recipe'];ex=recipe['execution'];job=recipe['jobs'][0]
        job=dict(job,path=str(out/'collection'/f'game-{job["index"]}.jsonl.gz'));jobs.append(job)
        games=read_shards([path],receipts={str(path):cache_receipt(path)});require(len(games)==1,'One family per shard')
        g=games[0];expected[g['game_id']]=signature(g)
        for child in g.get('branches',[]):children[child['game_id']]=signature(child)
        files[job['index']]=path
        if omitted is None and g.get('branches'):omitted=job['index']
    require(omitted is not None,'Need a branch family to recover')
    directory=out/'collection';directory.mkdir()
    retained={}
    for index,path in files.items():
        if index==omitted:continue
        for item in path.parent.glob(path.name+'*'):
            shutil.copyfile(item,directory/item.name)
        retained[path.name]=sha(path)
    jobs.sort(key=lambda j:j['index'])
    with InferenceService(ex,'cpu') as service,reader_pool(2,admission=service.cpu) as readers:
        identity=service.register(model)
        require(all(p['model_id']==identity for j in jobs for p in j['policies']),'Frozen model changed')
        policy_id=jobs[0]['policies'][0]['policy_id']
        from brass_ai.learning.contracts import digest
        kinds={digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=j['seed'])):'selfplay' for j in jobs}
        with ActorPool(SOURCE_ROOT,cfg['host'],service,ex) as pool:
            with StreamingCollection(readers,2,policy_id,kinds,cfg.get('insolvency'),reward_spec(cfg)) as prepared:
                records=pool.collect(jobs,on_records=prepared.submit)
                ingress={}
                games,data,diagnostics=prepared.finish(records,ingress)
            scheduling=pool.last_collection_report
    actual={};actual_children={}
    for record in records:
        for g in read_shards([record['path']],receipts={record['path']:record.get('features')}):
            actual[g['game_id']]=signature(g)
            for child in g.get('branches',[]):actual_children[child['game_id']]=signature(child)
    require(sum(not r['cached'] for r in records)==1,'Recovered complete families ran again')
    require(all(sha(directory/name)==value for name,value in retained.items()),'Completed family replay modified')
    require(diagnostics['health_counts']['tables']==len(jobs),'Branches entered ordinary health')
    require(scheduling['branch_games']==1,'Missing or duplicated recovered branch')
    write_json(out/'report.json',dict(passed=True,seconds=time.monotonic()-began,cached_families=len(retained),
        recollected_parent=omitted,recollected_branches=1,scheduling=scheduling,
        parent_probability_max_error=compare(expected,actual),branch_probability_max_error=compare(children,actual_children),
        prepared_games=len(games),prepared_rows=len(data),ordinary_health_games=diagnostics['health_counts']['tables'],
        ingress=ingress,
        original_family_bytes_preserved=True,
        scope='Fresh worker/inference processes; copied complete native families, omitted one branched family, streamed real replay preparation. No training or live checkpoint changes.'))


if __name__=='__main__':main()
