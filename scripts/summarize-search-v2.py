"""Summarize immutable study rows. Counts exclude inherited work (always zero)."""
import json
import runpy
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
T=runpy.run_path(str(ROOT/'scripts/search-v2-tools.py'));read,write=(T[k] for k in ('read','write'))
folder=T['OUT']/'study';protocol=read(folder/'protocol.json')
rows=[]
amended=T['OUT']/'fixture-correction-v2/protocol.json'
sources=[(folder,protocol)]+([(amended.parent,read(amended))] if amended.exists() else [])
for folder,batch in ((f,b) for f,p in sources for b in p['batches']):
    path=folder/(batch['batch']+'-output.json')
    if not path.exists():continue
    receipts=read(folder/(batch['batch']+'-receipt.json'))
    for row in read(path)['results']:
        if row['name'].endswith('-warmup'):continue
        method=row['name'].removeprefix(batch['batch']+'-');d=row['result']['decision'];f=d.get('focus') or {}
        owners=d['lanes'];focused=d.get('focused_lanes') or [];all_owners=owners+focused
        process=next(r for r in receipts if r['method']==method)
        out=d['result'];actions=out['action_ids']
        rows.append(dict(root=batch['root'],setting=batch['setting'],method=method,switched=f.get('switched',False),
            switch_reason=f.get('switch_reason'),fallback=f.get('fallback_used',False),
            actions=actions,complete_actions=len(actions),response_ms=d['response_ready_ns']/1e6,
            harness_elapsed_ms=row['elapsed_ns']/1e6,
            newly_completed=sum(o['outcome']['completed'] for o in all_owners),
            second_root_new=sum(o['outcome']['completed'] for o in focused),inherited_visits=0,
            partial_completed=sum(o['outcome']['partial_completed'] for o in all_owners),
            eligible_work=sum(o['outcome']['snapshot']['support_watermark'] if o['outcome']['snapshot'] else 0 for o in all_owners),
            forwards=sum(o['neural']['batches'] for o in all_owners),cache_hits=sum(o['outcome']['cache_hits'] for o in all_owners),
            inference_ms=sum(o['neural']['inference_ns'] for o in all_owners)/1e6,
            encoding_ms=sum(o['neural']['encoding_ns'] for o in all_owners)/1e6,
            extraction_ms=sum(f.get('extraction_ns',[0,0]))/1e6,projection_ms=f.get('preparation_ns',0)/1e6,
            retirement_ms=((f.get('retirement_finished_ns') or 0)-(f.get('retirement_started_ns') or 0))/1e6,
            owner_root_k=[o['outcome']['effective_k'] for o in owners],second_root_k=[o['outcome']['effective_k'] for o in focused],
            process_cpu_seconds=process['cpu_seconds'],peak_mib=process['peak_working_set_bytes']/2**20,
            raw=str(path.relative_to(ROOT))))
pairs=[]
for candidate in (r for r in rows if r['method']=='search_v2_focus_only'):
    control=next(r for r in rows if r['root']==candidate['root'] and r['setting']==candidate['setting'] and r['method']=='search_v1')
    pairs.append(dict(root=candidate['root'],setting=candidate['setting'],switched=candidate['switched'],
        work_ratio=candidate['newly_completed']/max(1,control['newly_completed']),
        response_delta_ms=candidate['response_ms']-control['response_ms'],
        cpu_ratio=candidate['process_cpu_seconds']/control['process_cpu_seconds'],
        peak_ratio=candidate['peak_mib']/control['peak_mib'],
        same_first=candidate['actions'][0]==control['actions'][0],same_plan=candidate['actions']==control['actions']))
summary=[]
for setting,_,_ in protocol['settings']:
    selected=[p for p in pairs if p['setting']==setting]
    if not selected:continue
    summary.append(dict(setting=setting,roots=len(selected),switches=sum(p['switched'] for p in selected),
        changed_first=sum(not p['same_first'] for p in selected),changed_plan=sum(not p['same_plan'] for p in selected),
        median_work_ratio=statistics.median(p['work_ratio'] for p in selected),
        median_cpu_ratio=statistics.median(p['cpu_ratio'] for p in selected),
        median_response_delta_ms=statistics.median(p['response_delta_ms'] for p in selected),
        max_response_ms=max(r['response_ms'] for r in rows if r['setting']==setting),
        max_peak_mib=max(r['peak_mib'] for r in rows if r['setting']==setting)))
write(T['OUT']/'analysis.json',dict(rows=rows,pairs=pairs,summary=summary,
    limitations=['One paired observation per root/setting; no win-rate conclusion.',
    'Process CPU/peak include startup and count-8 warmup. Not per-session CPU.',
    'Native response excludes Unity; partial one-action plans are not whole-turn completion.',
    'First 4s development pair overlapped final package source/hash preparation; retain as flagged evidence.']))
print(json.dumps(summary,indent=2));print('rows',len(rows),'two-action plans',sum(r['complete_actions']==2 for r in rows))
