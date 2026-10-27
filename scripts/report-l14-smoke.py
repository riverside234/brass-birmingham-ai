"""Audit bounded L14 evidence without promoting missing qualification gates."""
import copy
import json
from pathlib import Path
import runpy
import statistics

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'artifacts/rust-backend/20261026-l14-smoke'
OLD=ROOT/'artifacts/rust-backend/20261026-l14-study-v2'
BASE=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
read,write,receipt,sha=(BASE[n] for n in ('read','write','receipt','sha'))
compare=runpy.run_path(str(ROOT/'scripts/verify-rust-search.py'))['compare']


def decision(folder,file):return read(S/folder/file)['results']


def main():
    assert not (S/'assessment.json').exists(),'Preserve completed evidence'
    traces=read(S/'trace-equivalence-03/results.json');assert traces['passed']
    fields=('actions','action_ids','gumbels','effective_k','snapshot','completed','attempted','partial_completed','plan_events','eligible_plans')
    count=[]
    for rep in range(2):
        a=decision(f'count-before-{rep}',f'count-before-{rep}-output.json')
        b=decision(f'count-after-{rep}',f'count-after-{rep}-output.json')
        for x,y in zip(a,b):
            dx,dy=x['result']['decision'],y['result']['decision']
            assert dx['selection']==dy['selection']
            for left,right in zip(dx['lanes'],dy['lanes']):
                for field in fields:assert left['outcome'][field]==right['outcome'][field],field
            count.append(dict(name=x['name'],repeat=rep,before_ms=x['elapsed_ns']/1e6,after_ms=y['elapsed_ns']/1e6,
                completed=sum(l['outcome']['completed'] for l in dx['lanes']),fraction_faster=1-y['elapsed_ns']/x['elapsed_ns']))
    allocation=[]
    a=read(S/'alloc-before/results.json')['rows'];b=read(S/'alloc-after/results.json')['rows']
    for x,y in zip(a,b):
        old,new=x['allocations'],y['allocations']
        allocation.append(dict(name=x['name'],before=old,after=new,
            allocated_byte_reduction=1-new['bytes_allocated']/old['bytes_allocated'],
            allocation_count_reduction=1-new['allocations']/old['allocations']))
    timed=[]
    a=decision('timed-before-0','timed-before-0-output.json');b=decision('timed-after-0','timed-after-0-output.json')
    for x,y in zip(a,b):
        dx,dy=x['result']['decision'],y['result']['decision']
        assert x['allowance_ms']==y['allowance_ms']==4000
        assert dy['sessions']==2 and dy['threads_per_session']==1 and dy['response_ready_ns']<4_000_000_000
        timed.append(dict(name=x['name'],before_ms=dx['response_ready_ns']/1e6,after_ms=dy['response_ready_ns']/1e6,
            before_completed=sum(l['outcome']['completed'] for l in dx['lanes']),after_completed=sum(l['outcome']['completed'] for l in dy['lanes']),
            before_provenance=dx['selection']['provenance'],after_provenance=dy['selection']['provenance']))
    py=read(S/'gpy-whole-count/results.json')
    ort=decision('external-ort-count','ort-output.json');torch=decision('external-torch-count-02','torch-output.json')
    parity=[]
    for x,y,z in zip(py['rows'],ort,torch):
        d=x['result']
        for control in (y,z):
            nd=control['result']['decision']
            for a,b in zip(d['lanes'],nd['lanes']):
                for field in ('gumbels','effective_k','snapshot','completed','partial_completed','eligible_plans'):
                    compare(a[field],b['outcome'][field],tol=2e-4)
            for field in ('actions','source_session','proposal','provenance'):compare(d['selection'][field],nd['selection'][field],tol=2e-4)
        assert x['whole_turn_complete'] and x['original_deadline_preserved']
        parity.append(dict(name=x['name'],owners=2,torch_and_ort_snapshot_and_plan_parity=True,action_two=True))
    py_timed=read(S/'gpy-whole-timed-torch/results.json')
    for x in py_timed['rows']:
        assert x['whole_turn_complete'] and x['original_deadline_preserved'] and x['elapsed_ns']<4_000_000_000
    # Correlate IDs, then subtract durations only. Process clock origins differ.
    unity=read(OLD/'smoke-v5-profile-stage-trace.json');samples=unity['samples']
    native=[json.loads(line) for line in (S/'native-service-trace.jsonl').read_text().splitlines()]
    ipc=[]
    for request in sorted({r['request'] for r in native if r['request']>0}):
        nr={r['stage']:r for r in native if r['request']==request}
        cr={r['stage']:r for r in samples if r.get('native_id')==request}
        if 'ipc.write_begin' not in cr or 'ipc.reply_received' not in cr:continue
        caller=(cr['ipc.reply_received']['at']-cr['ipc.write_begin']['at'])*1000/unity['frequency']
        service=(nr['reply_write_begin']['at_ns']-nr['pipe_read_complete']['at_ns'])/1e6
        assert caller>=service
        span=lambda start,end:(nr[end]['at_ns']-nr[start]['at_ns'])/1e6 if start in nr and end in nr else None
        ipc.append(dict(request=request,caller_frame_roundtrip_ms=caller,native_contained_service_ms=service,
            transport_and_caller_dispatch_residual_ms=caller-service,
            native_queue_ms=span('pipe_read_complete','dequeued'),native_parse_ms=span('dequeued','parsed'),
            admission_ms=span('parsed','input_validated'),pair_prepare_ms=span('input_validated','pair_started'),
            pair_search_and_retirement_ms=span('pair_started','pair_retired'),
            delivery_prepare_ms=span('pair_retired','reply_serialize_begin'),serialize_ms=span('reply_serialize_begin','reply_write_begin'),
            native_write_ms=span('reply_write_begin','reply_write_end'),request_bytes=nr['pipe_read_complete']['bytes'],reply_bytes=nr['reply_write_begin']['bytes']))
    assert ipc
    rules=read(S/'rules-api-01/results.json');assert rules['passed']
    quality=read(S/'quality-smoke-02/results.json');assert quality['passed']
    unity_test=read(ROOT/'artifacts/rust-backend/20261026-l14-candidate-v5/unity.json');assert unity_test['passed']
    assert sha(S/'gpy-whole-count/reference.py')==sha(ROOT/'tests/RustBackend/gumbel_session_reference.py')
    assert sha(S/'gpy-whole-timed-torch/reference.py')==sha(ROOT/'tests/RustBackend/gumbel_session_reference.py')
    assert read(S/'candidate-native/receipt.json')['source_sha256']==BASE['native_fingerprint']()
    result=dict(format='brass-l14-smoke-assessment/1.0.0',validation_optimization_retained=True,native_decision_reuse_retained=True,
        scope='Bounded smoke qualification; no full-game tournament or statistical strength/tail claim.',
        full_l14_qualification='open',installed=False,source_fingerprint=BASE['native_fingerprint'](),
        trace_parity=traces,count_pairs=count,allocation_pairs=allocation,timed_pairs=timed,python_reference_parity=parity,
        summary=dict(count_pair_median_reduction=statistics.median(x['fraction_faster'] for x in count if x['completed']>0),
            allocated_bytes_reduction=1-sum(x['after']['bytes_allocated'] for x in allocation)/sum(x['before']['bytes_allocated'] for x in allocation),
            allocation_count_reduction=1-sum(x['after']['allocations'] for x in allocation)/sum(x['before']['allocations'] for x in allocation)),
        native_ipc=ipc,unity_allocation_counter=dict(available=unity['allocation_counter_available'],reason=unity['allocation_source'],
            self_test=[x for x in samples if x['stage']=='allocation.self_test']),
        unity_checks=unity_test['checks'],rules_api=rules,quality=quality,
        open_gates=[dict(gate='Unity managed allocation totals',reason='Installed Mono has no per-thread allocation API; native callback self-test reports zero. Requires a compatible allocation-enabled profiling runtime.'),
                    dict(gate='Statistical strength and latency tails',reason='Deferred under the user instruction to use short smoke tests. Historical thresholds and failures remain visible; no full release/training promotion.')],
        evidence=[receipt(S/p) for p in ('protocol.json','baseline-native/receipt.json','candidate-native/receipt.json','gpy-whole-count/results.json',
            'gpy-whole-timed-torch/results.json','external-ort-timed/results.json','external-torch-timed/results.json','native-service-trace.jsonl')])
    write(S/'assessment.json',result)
    print(json.dumps(dict(summary=result['summary'],ipc=ipc,checks=unity_test['checks']),indent=2))


if __name__=='__main__':main()
