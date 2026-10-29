"""V205 follow-up: 4s/K8 only, sequential cache-only/reuse smoke pairs.

Reuse the pinned runner/model/fixtures. Each pair has a 60-second watchdog;
there is no long game, parameter sweep, or production change.
"""
import argparse
import copy
from pathlib import Path
import runpy
import shutil
import statistics
import zipfile

ROOT=Path(__file__).resolve().parents[1]
B=runpy.run_path(str(ROOT/'scripts/benchmark-search-v2.py'))
read,write,receipt,sha=(B[k] for k in ('read','write','receipt','sha'))
OUT=B['OUT'];FOLDER=OUT/'v205';CONFIG=ROOT/'configs/search/search-v205-smoke.json'


def freeze():
    FOLDER.mkdir(exist_ok=False)
    original=read(OUT/'study/protocol.json');corrected=read(OUT/'fixture-correction-v2/protocol.json')
    lookup={r['id']:r for r in original['roots']}
    lookup.update({r['id']:r for r in corrected['roots']})
    names=['brass-003','loan-build','dense-canal','rail-double','dense-rail','multi-sale']
    roots=[]
    for name in names:
        row=copy.deepcopy(lookup[name]);row['partition']='previously inspected coverage root'
        roots.append(row)
    batches=[]
    for i,name in enumerate(names):
        order=['search_v2_focus_only','search_v2']
        if i%2:order.reverse()
        batches.append(dict(batch=name+'-4s_k8',root=name,setting='4s_k8',allowance_ms=4000,k=8,order=order))
    # Predeclared diagnostic second seed/order; no adaptive seed hunting.
    for name in ['brass-003','dense-rail']:
        row=copy.deepcopy(lookup[name]);row.update(id=name+'-repeat',search_seed=19,world_seed=891,
            partition='predeclared diagnostic second seed')
        roots.append(row)
        batches.append(dict(batch=row['id']+'-4s_k8',root=row['id'],setting='4s_k8',allowance_ms=4000,k=8,
            order=['search_v2','search_v2_focus_only']))
    binary=ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/gumbel-pair.exe'
    shutil.copyfile(binary,FOLDER/binary.name)
    protocol=dict(version='search-v205-smoke/1.0.0',folder='v205',roots=roots,batches=batches,
        settings=[['4s_k8',4000,8]],sessions=2,threads_per_session=1,
        preparation_ns=20_000_000,minimum_support=2,subtree_reuse=True,production_switch=False,
        startup=original['startup'],model=original['model'],binary=receipt(FOLDER/binary.name),
        batch_watchdog_seconds=60,maximum_trials=16,search_allowance_seconds=64,
        warmup=original['warmup'],native_source_sha256=B['TOOLS']['BASE']['native_fingerprint'](),
        frozen_control_candidate=receipt(OUT/'final-candidate/build-manifest.json'),
        reuse_policy='descendant-reuse/1.0.0; first-expansion descendant forest; completed-watermark rollback; fresh root counts/support/draws; independent certified sampler/ledger per owner',
        interpretation='Old edge visits are not newly completed trajectories. Different first paths or focus checkpoints are not pure reuse speed comparisons.',
        decision='Keep only if correctness and bounded delivery pass and retention gives useful work/latency or tactical benefit against cache-only, including the predeclared second seeds. If overhead or deeper-path cost dominates without independent quality evidence, retain cache-only and leave reuse disabled.',
        expectations='Complete independently legal actions; same causal resources/ledger. No invented optimal-action oracle. Previously observed multi-sale development/sale choice is audited, not forced.',
        repeat_scope='Run the two predeclared diagnostic second-seed pairs once after the six primary pairs. No added timing trials.')
    write(CONFIG,protocol);write(FOLDER/'protocol.json',protocol)
    shutil.copyfile(__file__,FOLDER/Path(__file__).name)
    with zipfile.ZipFile(FOLDER/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['native','tests/RustBackend','tests/GumbelISMCTS','scripts']:
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and p.suffix in ('.rs','.toml','.lock','.json','.py') and '__pycache__' not in p.parts:
                    z.write(p,p.relative_to(ROOT))
    print('Frozen six 4s/K8 pairs plus two diagnostic second-seed pairs; at most 64s total admitted search allowances.')


def summarize():
    protocol=read(CONFIG);rows=[];pairs=[]
    for batch in protocol['batches']:
        path=FOLDER/(batch['batch']+'-output.json')
        if not path.exists():continue
        receipts={r['method']:r for r in read(FOLDER/(batch['batch']+'-receipt.json'))}
        arms={}
        for raw in read(path)['results']:
            if raw['name'].endswith('-warmup'):continue
            d=raw['result']['decision'];f=d['focus'];method=f['method'];owners=d.get('focused_lanes') or []
            all_owners=d['lanes']+owners
            reuse=[o['outcome'].get('reuse') for o in owners]
            row=dict(batch=batch['batch'],method=method,switched=f['switched'],reason=f['switch_reason'],
                first_path=d['result']['action_ids'][0],actions=d['result']['action_ids'],
                completed=sum(o['outcome']['completed'] for o in all_owners),
                phase2_completed=sum(o['outcome']['completed'] for o in owners),
                watermarks=[o['outcome']['snapshot']['support_watermark'] if o['outcome']['snapshot'] else 0 for o in d['lanes']],
                phase2_effective_k=[o['outcome']['effective_k'] for o in owners],
                response_ms=d['response_ready_ns']/1e6,process_cpu_seconds=receipts[method]['cpu_seconds'],
                peak_mib=receipts[method]['peak_working_set_bytes']/1048576,
                reused_selections=sum(o['outcome']['schedule']['reused_selections'] for o in owners),
                unique_leaves=sum(o['outcome']['schedule']['unique_leaves'] for o in all_owners),
                phase2_unique_leaves=sum(o['outcome']['schedule']['unique_leaves'] for o in owners),
                nn_forwards=sum(o['neural']['batches'] for o in all_owners),
                phase2_nn_forwards=sum(o['neural']['batches'] for o in owners),
                inference_ms=sum(o['neural']['inference_ns'] for o in all_owners)/1e6,
                encoding_ms=sum(o['neural']['encoding_ns'] for o in all_owners)/1e6,
                phase2_traversal_ms=sum(sum(o['outcome']['schedule'].get(k,0) for k in ['sampling_ns','information_ns','lookup_ns','selection_ns','application_ns']) for o in owners)/1e6,
                cache_hits=sum(o['outcome']['cache_hits'] for o in all_owners),reuse=reuse,
                fallback=f['fallback_used'],retirement_ns=(f['retirement_finished_ns'] or 0)-(f['retirement_started_ns'] or 0),
                action_count=len(d['result']['action_ids']))
            rows.append(row);arms[method]=row
        if len(arms)==2:
            a,b=(arms[m] for m in ('search_v2_focus_only','search_v2'))
            pairs.append(dict(batch=batch['batch'],same_first=a['first_path']==b['first_path'],same_plan=a['actions']==b['actions'],
                same_watermarks=a['watermarks']==b['watermarks'],
                completed_gain_percent=100*(b['completed']/a['completed']-1),
                phase2_gain_percent=100*(b['phase2_completed']/a['phase2_completed']-1) if a['phase2_completed'] else None,
                response_delta_ms=b['response_ms']-a['response_ms'],
                cpu_gain_percent=100*(b['process_cpu_seconds']/a['process_cpu_seconds']-1),
                peak_delta_mib=b['peak_mib']-a['peak_mib']))
    def stats(group):
        return {k:statistics.median(p[k] for p in group if p[k] is not None) for k in
            ['completed_gain_percent','phase2_gain_percent','response_delta_ms','cpu_gain_percent','peak_delta_mib']} if group else {}
    summary=dict(version='search-v205-analysis/1',rows=rows,pairs=pairs,
        primary=stats([p for p in pairs if '-repeat-' not in p['batch']]),
        repeats=stats([p for p in pairs if '-repeat-' in p['batch']]),
        all_actions_complete=all(r['action_count']==2 for r in rows),
        policy=read(CONFIG)['reuse_policy'],scope='4s/K8 only; small smoke evidence, no playing-strength or population-tail claim')
    reused=[m for r in rows if r['method']=='search_v2' for m in r['reuse'] if m]
    summary['retention']=dict(owners=len(reused),hits=sum(m['hit'] for m in reused),
        matching_ms_range=[min(m['matching_ns'] for m in reused)/1e6,max(m['matching_ns'] for m in reused)/1e6] if reused else [],
        retained_nodes_range=[min(m['counts']['retained_nodes'] for m in reused),max(m['counts']['retained_nodes'] for m in reused)] if reused else [],
        inherited_edge_visits_range=[min(m['counts']['inherited_edge_visits'] for m in reused),max(m['counts']['inherited_edge_visits'] for m in reused)] if reused else [])
    write(FOLDER/'analysis.json',summary)
    print({k:v for k,v in summary.items() if k not in ('rows','policy')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','summarize']);p.add_argument('--batch');a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='summarize':summarize()
    else:
        B['run'].__globals__['CONFIG']=CONFIG
        B['run'](a.batch)
