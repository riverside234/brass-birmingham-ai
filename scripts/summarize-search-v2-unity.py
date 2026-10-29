"""Separate diagnostic host spans without counting nested stages twice."""
import runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
T=runpy.run_path(str(ROOT/'scripts/search-v2-tools.py'));read,write,receipt=(T[k] for k in ('read','write','receipt'))
rows=[]
for name in ('unity-control-4s-profile','unity-candidate-4s-profile',
    'unity-search_v2_focus_only-8s-k8','unity-search_v1-8s-k8',
    'unity-search_v1-8s-k16','unity-search_v2_focus_only-8s-k16'):
    folder=T['OUT']/name
    if not (folder/'result.json').exists():continue
    result=read(folder/'result.json');trace=read(folder/'stage-trace.json');meta=read(folder/'receipt.json')
    instance=result['replies'][0]['ticket']['game_instance']
    # Events not attached to the live turn (cold start/warmup/UI setup) are
    # excluded. Each sum is one named stage; nested columns are not additive.
    samples=[s for s in trace['samples'] if instance in (s.get('request') or '')]
    totals={}
    for sample in samples:
        if 'wall_ms' not in sample:continue
        stage=sample['stage'];totals[stage]=totals.get(stage,0)+sample['wall_ms']
    rows.append(dict(method=meta['method'],setting=meta['setting'],delivery_ms=result['delivery_ms'],
        initial_snapshot_ms=result['initial_snapshot_ms'],cold_warmup_ms=result['cold_warmup_ms'],
        unity_cpu_ms=result['unity_cpu_ms'],native_cpu_ms=result['native_cpu_ms'],
        requests=result['requests'],focused=result['focused'],commits=result['commits'],
        stage_totals_ms=totals,source=receipt(folder/'result.json'),trace=receipt(folder/'stage-trace.json')))
lifecycle=[]
for name in ('unity-cancel','unity-load'):
    p=T['OUT']/name/'result.json'
    if p.exists():
        r=read(p);lifecycle.append(dict(scenario=r['scenario'],passed=r['passed'],focused=r['focused'],source=receipt(p)))
write(T['OUT']/'unity-analysis.json',dict(rows=rows,lifecycle=lifecycle,
    limitations=['One paired Unity root per setting; diagnostic instrumentation adds overhead.',
    'Stage sums are nested; do not sum columns as a disjoint latency decomposition.',
    'Native roundtrip/read wait includes search; IPC serialization/parse are separate.',
    'Native per-session CPU and Mono native peak working set unavailable.']))
for row in rows:
    t=row['stage_totals_ms'];print(row['setting'],row['method'],'delivery',round(row['delivery_ms'],2),
        'snapshot',round(t.get('snapshot.search_input',0),2),'detached',round(t.get('validation.detached',0),2),
        'duplicate full',round(t.get('validation.full',0),2),'commit',round(t.get('commit.native_guard_and_apply',0),2),
        'successor guard',round(t.get('validation.projected_successor',0),2),
        'IPC serde',round(t.get('ipc.serialize',0)+t.get('ipc.deserialize',0),2))
print('lifecycle',lifecycle)
