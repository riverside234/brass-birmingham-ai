"""Short Linux/CUDA B02 coverage with independent C# progression/scoring oracle."""
from __future__ import annotations
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import queue
import threading

import numpy as np
import torch
from brass_ai.learning.contracts import SOURCE_ROOT,read_json,require,digest
from brass_ai.learning.features import Encoder
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.replay import check_game
from brass_ai.recovery.updates import prepared
from brass_ai.recovery.utility import reward_spec
from .lab_collection import collect_lane,frozen_id,route
from .lab_transport import Collector,DecisionEncoder,Pipe,CONTRACT
from .lab_families import FamilyStore,collect_families
from .lab_contract import METHOD,pin,sha


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def require_coverage(kinds, branches=None):
    require(set(kinds)=={'selfplay','historical','legacy'},
            'Incomplete B02 table coverage in the next frozen batch; do not resample or change the run recipe')
    if branches is not None:
        require(branches>0,'B02 needs a complete branch family; a branch-disabled recipe cannot qualify this gate')


def reference_part(root,job,endpoint,execution,mono):
    """Exact whole-table fallback. No native label is attached to these rows."""
    from brass_ai.host import RulesHost
    from brass_ai.recovery.execution import run_game,make_policy
    from brass_ai.recovery.host_runtime import local_scoring_host,training_host
    from brass_ai.recovery.telemetry import Timings
    with ExitStack() as stack:
        directory=local_scoring_host(root) if 'local_scoring' in job['provenance'] else training_host(root)
        host=stack.enter_context(RulesHost(directory,mono=mono));raw={}
        policies=[make_policy(p,root,endpoint,execution,mono) for p in job['policies']]
        for p in policies:
            stack.callback(p.close)
            if hasattr(p,'evaluator'):
                predict=p.evaluator.predict
                def record(requests,p=p,predict=predict):
                    replies=predict(requests)
                    require(len(replies)==len(requests),'Reference inference batch incomplete')
                    for r,v in zip(requests,replies,strict=True):raw[(p.policy_id,digest(r.view))]=list(v['probabilities'])
                    return replies
                p.evaluator.predict=record
        # Stable IDs are additive execution bindings, not a new policy draw.
        job=copy.deepcopy(job);job.setdefault('stable_game_id',digest(dict(job=frozen_id(job),seed=job['seed'])))
        game=run_game(host,job,policies,Timings(),execution['max_decisions']);evidence=[]
        require(not game.get('branches'),'Reference-routed branch tables need separate qualification')
        for row in game['records']:
            spec=job['policies'][(row['seat']-job.get('rotation',0))%4]
            evidence.append(dict(kind='ordinary',model_id=spec.get('model_id'),policy_id=spec['policy_id'],
                raw_probabilities=([1.] if len(row['choices'])==1 else raw[(spec['policy_id'],digest(row['view']))]) if spec['kind']=='neural' else None,
                semantic_id=row['choices'][row['chosen']]['id']))
        return dict(game=dict(game),evidence=evidence,scoring=None,branch_expected=False,
            producer=dict(contract='brass-reference-collection/1.0.0',runtime=pin(directory/'runtime.json')),timings={})


def compare_targets(candidate,reference,config):
    kwargs=dict(insolvency=config.get('insolvency'),reward=reward_spec(config))
    a=prepared([check_game(candidate)],**kwargs);b=prepared([check_game(reference)],**kwargs)
    require(len(a)==len(b),'Target row count differs')
    for (ra,ta,ia,ea),(rb,tb,ib,eb) in zip(a,b,strict=True):
        require(ta==tb and ia==ib,'Terminal targets or trajectory attribution differ')
        require({k:v for k,v in ra.items() if k.startswith('_')}=={k:v for k,v in rb.items() if k.startswith('_')},'Causal reward/branch target differs')
        for key in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(ea,key),getattr(eb,key))
    return len(a)


def fixture_checks(executable,fixtures,model_id):
    oracle=Encoder();coverage=[]
    with Collector(executable) as lane:
        for root in fixtures['roots']:
            d=root['expected']['state']
            lane.begin('fixture-'+root['id'],dict(state=d,rail_keys=list(range(64))),2)
            decision=lane.current['decision'];require(decision['view']==root['view'] and decision['choices']==root['expected']['choices'],'B02 fixture projection differs')
            py=oracle.encode(decision['view'],decision['choices'],[0.]*4,[0.]*8)
            rs=DecisionEncoder(lane,model_id).encode(decision['view'],decision['choices'],[0.]*4,[0.]*8)
            for key in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(py,key),getattr(rs,key))
            coverage.append(dict(id=root['id'],era=d['context']['era'],phase=d['context']['phase'],seat=d['context']['actor'],candidates=len(decision['choices'])))
            lane.release()
    return coverage


def run(args):
    root=SOURCE_ROOT;out=args.out;build=read_json(args.build);snapshot=read_json(args.state)
    for item in (build['collector'],build['bootstrap']['binary'],*build['bootstrap']['dependencies']):
        require(pin(item['path'])==item,'B02 runtime changed')
    require(torch.device(args.device).type=='cuda' and torch.cuda.is_available(),'B02 requires real CUDA')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,raw=load_checkpoint(snapshot['checkpoint']['path'],device='cpu');model.eval()
    execution=snapshot['execution'];config=snapshot['state']['signature']['config']
    fixtures=read_json(Path(build['b01']['path']).parent/'inputs/fixtures.json')
    coverage=fixture_checks(build['collector']['path'],fixtures,snapshot['model_id'])
    # Match each actual frozen job. Only select a bounded diagnostic subset;
    # preserve original indices, policies, rotations, seed and branch settings.
    with InferenceService(execution,args.device) as service:
        from .lab_jobs import plan
        planned=plan(root,args.run,snapshot,model,raw,service,out/'diagnostic-jobs')
        jobs=[];seen=set()
        for job in planned['jobs']:
            kind=job['provenance']['kind']
            if kind not in seen and kind in ('selfplay','historical','legacy'):
                jobs.append(job);seen.add(kind)
        require_coverage(seen)  # Check kinds before starting any full diagnostic games.
        require(any(j.get('branching') and route(j)[0]=='rust' for j in jobs),
                'Next frozen batch lacks a native branch-enabled self-play table; B02 remains Pending')
        require(all(route(j)[0]=='rust' or j['provenance']['legacy_mode']=='fast' for j in jobs),'B02 does not qualify enabled search jobs')
        write(out/'frozen-jobs.json',dict(jobs=jobs,full_plan_sha256=digest(planned['jobs']),
            full_plan_composition=planned['composition'],diagnostic_subset=True,table_mix_changed=False))
        lanes=min(2,execution['processes']*execution['hosts_per_process'])
        leases=service.channels(lanes);endpoints=queue.Queue()
        owned=[leases.acquire() for _ in range(lanes)]
        for endpoint in owned: endpoint.timeout=20.;endpoints.put(endpoint)
        stop=threading.Event();oracle_checkpoints={};lock=threading.Lock();metrics=[]
        bootstrap=[args.mono,build['bootstrap']['binary']['path']]
        if os.name=='nt':bootstrap=bootstrap[1:]
        def native(job,continuation,submit):
            endpoint=endpoints.get(timeout=20)
            try:
                with Pipe(bootstrap,binary=False) as reference,Pipe(bootstrap,binary=False) as draws:
                    origin=reference.call('init',seed=job['seed']);reference_records=[]
                    if continuation is not None:
                        with lock:cp=oracle_checkpoints.pop((frozen_id(job),continuation['metadata']['seed']))
                        reference.call('restore',checkpoint=cp,shuffle_seed=int(continuation['metadata']['seed'][:8],16)&0x7fffffff if job['branching']['reshuffle_hidden_deck'] else None)
                    def compare(row,encoded,lane):
                        ref=reference.call('decision')
                        require(row['view']==ref['view'] and row['choices']==ref['choices'],'Reference/native decision differs at '+str(row['step']))
                        if encoded is not None:
                            py=Encoder().encode(ref['view'],ref['choices'],row['style'],row['intent'])
                            for key in ('state','actions','stable_to_relative'):np.testing.assert_array_equal(getattr(py,key),getattr(encoded,key))
                        reference_records.append(dict(row,view=ref['view'],choices=ref['choices']))
                        reference.call('step',choice=row['choices'][row['chosen']]['id'])
                    def fork(packet):
                        with lock:oracle_checkpoints[(frozen_id(job),packet['metadata']['seed'])]=reference.call('checkpoint')
                        submit(packet)
                    part=collect_lane(root,build['collector']['path'],draws,job,endpoint,execution,origin=origin,
                        continuation=continuation,submit_branch=fork,cancel=stop,compare=compare,
                        diagnostic_branch_probe=continuation is None and bool(job.get('branching')))
                    final=reference.call('decision');require(final['context']['terminal'] and part['game']['result']==final['result'],'Final result differs')
                    scoring=reference.call('scoring')
                    require({k:v for k,v in scoring.items() if k!='producer'}=={k:v for k,v in part['scoring'].items() if k!='producer'},'Native scoring receipt differs')
                    reference_game=dict(part['game'],records=reference_records,result=final['result'])
                    targets=compare_targets(part['game'],reference_game,config)
                    with lock:metrics.append(dict(job=job['index'],branch=continuation is not None,rows=len(reference_records),targets=targets,
                        timings=part['timings'],native_cost=part['native_receipt']['cost'],buffer_bytes=part['native_receipt']['buffer_bytes']))
                    return part
            except BaseException:stop.set();raise
            finally:endpoints.put(endpoint)
        def fallback(job,continuation,submit):
            require(continuation is None,'Reference continuation unsupported')
            endpoint=endpoints.get(timeout=20)
            try:return reference_part(root,job,endpoint,execution,args.mono)
            except BaseException:stop.set();raise
            finally:endpoints.put(endpoint)
        producer=dict(build_sha256=sha(args.build),b01_sha256=sha(build['b01']['path']),signature=snapshot['signature_sha256'],profile=METHOD,v205_enabled=False)
        store=FamilyStore(out/'families',producer)
        try:
            families,scheduler=collect_families(jobs,store,native,fallback,lanes=lanes,cancel=stop,model_cohorts=execution.get('model_cohorts',False))
            _,cached=collect_families(jobs,store,lambda *a:require(False,'Cached family recollected'),lambda *a:require(False,'Cached reference recollected'),lanes=lanes)
            require(cached['counts']['cached']==len(jobs) and not oracle_checkpoints,'Retry/fork leak')
        finally:
            for endpoint in owned:endpoint.close()
        report=service.report();require(report['error'] is None,'Inference service failure')
    branches=sum(f['child'] is not None for f in families)
    require_coverage(seen, branches)
    return dict(passed=True,scope='bounded diagnostic subset of frozen ordinary jobs; native neural tables, whole-table reference legacy route; no training admission',
        frozen_jobs=pin(out/'frozen-jobs.json'),families=[pin(store.path(job)) for job in jobs],
        roots=coverage,metrics=metrics,scheduler=scheduler,cached=cached,service=report,branch_families=branches,
        missing_table_kinds=sorted({'selfplay','historical','legacy'}-seen),training_started=False,lab_deployed=False,
        full_native_coverage=False,method=METHOD,search_enabled=False,v205_enabled=False,diagnostic_branch_trigger='first eligible decision; alternative draws and fork semantics unchanged',
        branch_exit_qualified=branches>0)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','state','build','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--mono',default='mono');a=p.parse_args()
    try:value=run(a)
    except Exception as e:
        write(a.out/'collection-smoke.json',dict(passed=False,error=f'{type(e).__name__}: {e}'));raise
    write(a.out/'collection-smoke.json',value)
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
