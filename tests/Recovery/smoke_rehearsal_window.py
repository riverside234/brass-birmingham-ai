"""Two one-game CPU iterations in a new run, with bounded train rehearsal."""
import argparse
from pathlib import Path
import time

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery.config import latest_model_config,validate_config
from brass_ai.recovery.demonstrations import completed_imitation
from brass_ai.recovery.league import train
from brass_ai.step5.league import write_json,implementation_id


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imitation-run',required=True)
    parser.add_argument('--probe-from-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=SOURCE_ROOT;out=project_path(root,args.output)
    require(not out.exists(),'Smoke requires a new output directory')
    _,origin=completed_imitation(root,args.imitation_run)
    cfg=latest_model_config(read(root/'configs/step5/joint7-latest.json'),origin)
    cfg.update(games_per_iteration=1,threads=2,audit_workers=1,profile_steps=0,seed=18570000,
               rehearsal_manifest=origin['manifest'])
    cfg['execution'].update(processes=1,hosts_per_process=2,heavy_processes=1,worker_threads=1,
        batch_size=64,max_candidate_cells=131072,shared_slot_bytes=2097152)
    cfg['update'].update(effective_batch_size=512,critic_batch_size=64,microbatch_size=16,
                         max_candidate_cells=8192,epochs=1)
    cfg['rehearsal_sampling'].update(rows_per_iteration=64,shard_draws=2,max_cache_bytes=64*1024**2)
    cfg['retention']['enabled']=False;cfg['rating']['enabled']=False
    validate_config(cfg)
    donor=project_path(root,args.probe_from_run);donor_state=read(donor/'state.json')
    donor_probe=read(donor/'il-kl/manifest.json')
    protected=[donor/'state.json',donor/'il-kl/manifest.json',project_path(root,donor_probe['cache']['path']),
               project_path(root,donor_state['history'][-1]['path']),project_path(root,origin['path']),
               project_path(root,origin['manifest'])]
    before={str(p):sha(p) for p in protected};source=implementation_id()
    write_json(out/'config.json',cfg);started=time.perf_counter()
    first=train(root,out/'run',cfg,iterations=1,device='cpu',imitation_run=args.imitation_run,
        fresh=True,il_kl_probe_from_run=args.probe_from_run)
    first_checkpoint=read(out/'run/state.json')['history'][-1]
    second=train(root,out/'run',cfg,iterations=1,device='cpu',imitation_run=args.imitation_run,expected_generation=1)
    state=read(out/'run/state.json');iterations=[read(out/'run'/f'iteration-{i:04d}'/'report.json') for i in range(2)]
    sampling=[r['actor']['rehearsal_sampling'] for r in iterations]
    require(state['iteration']==2 and state['games_done']==2,'Smoke continuation progress differs')
    require(all(r['cache_deleted'] and r['used_presentations']>0 for r in sampling) and
            not (out/'run/.rehearsal-cache').exists(),'Rehearsal samples unused or temporary cache retained')
    require(sampling[0]['selection_sha256']!=sampling[1]['selection_sha256'],'Window did not rotate')
    require(state['il_kl_limit']['probe']['probe_hash']==donor_state['il_kl_limit']['probe']['probe_hash'],
            'Fixed validation probe changed')
    require(sha(project_path(root,first_checkpoint['path']))==first_checkpoint['sha256'],'Continuation changed prior checkpoint')
    require(before=={p:sha(Path(p)) for p in before},'Existing IL or donor artifacts changed')
    require(source==implementation_id(),'Implementation changed during smoke')
    report=dict(passed=True,seconds=time.perf_counter()-started,implementation=source,iterations=2,games=2,
        rehearsal=sampling,probe_reused=True,probe_hash=state['il_kl_limit']['probe']['probe_hash'],
        source_files_unchanged=True,previous_checkpoint_unchanged=True,
        optimizer_steps=[r['actor']['optimizer_steps'] for r in iterations],
        scope='Two one-game CPU iterations, fresh start and same-run resume; diagnostic IL, 64-row windows, no teacher/evaluations; not a strength or production-throughput test')
    write_json(out/'report.json',report);print(out/'report.json',flush=True)


if __name__=='__main__':main()
