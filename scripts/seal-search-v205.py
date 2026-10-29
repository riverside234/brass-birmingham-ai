"""Verify the bounded V205 evidence and record the cache-only selection."""
from pathlib import Path
import re
import runpy

ROOT=Path(__file__).resolve().parents[1]
T=runpy.run_path(str(ROOT/'scripts/search-v2-tools.py'))
read,write,sha,receipt=(T[k] for k in ('read','write','sha','receipt'))
OUT=T['OUT'];study=OUT/'v205';protocol=read(study/'protocol.json')
assert protocol==read(ROOT/'configs/search/search-v205-smoke.json')
assert protocol['settings']==[['4s_k8',4000,8]]
assert sha(ROOT/protocol['binary']['path'])==protocol['binary']['sha256']
assert T['BASE']['native_fingerprint']()==protocol['native_source_sha256']
assert read(OUT/'v205-cache-candidate/BrassNative/alpha-g890/policy.json')['method']=='search_v2_focus_only'
checks=0
for batch in protocol['batches']:
    raw=read(study/(batch['batch']+'-output.json'))
    rows=[r for r in raw['results'] if not r['name'].endswith('-warmup')]
    assert len(rows)==2
    for row in rows:
        assert row['result']['status']=='complete'
        d=row['result']['decision'];assert d['sessions']==2 and d['threads_per_session']==1
        assert len(d['result']['action_ids'])==2 and d['response_ready_ns']<4_000_000_000
        assert d['focus']['switched'] and not d['focus']['fallback_used']
        assert d['focus']['composite']['joint_support'] is None
        for lane in d['lanes']+d['focused_lanes']:
            o=lane['outcome'];assert o['budget']['deadline_ns']<=4_000_000_000
            assert o['last_completion_ns']<o['budget']['work_deadline_ns']
            assert sum(o['snapshot']['stats']['count'])==o['snapshot']['support_watermark']<=o['completed']
            checks+=3
        for n,lane in enumerate(d['focused_lanes']):
            metrics=lane['outcome'].get('reuse')
            if d['focus']['method']=='search_v2':
                assert metrics['hit'] and metrics['counts']['watermark']==d['lanes'][n]['outcome']['snapshot']['support_watermark']
                assert metrics['counts']['retained_nodes']>1
            else:assert metrics is None
            checks+=1
        checks+=5
native=read(OUT/'v205-native-checks/result.json');delivery=read(OUT/'v205-delivery/result.json')
assert native['passed'] and delivery['passed'] and len(delivery['rows'])==16
unity=[]
for label in ['v205-unity-normal','v205-unity-cache-normal','v205-unity-cancel','v205-unity-load']:
    row=read(OUT/label/'result.json');assert row['passed'] and row['focused'] and row['allowance_ms']==4000
    unity.append(receipt(OUT/label/'result.json'))
control=read(OUT/'control/manifest.json');pins=control['managed']+control['package']
for pin in pins:assert sha(ROOT/pin['path'])==pin['sha256'],pin['path']
assert sha(ROOT/'artifacts/patch-installation.json')==sha(OUT/'control/installation.json')
write(study/'installed-unchanged.json',dict(passed=True,files=len(pins),control=receipt(OUT/'control/manifest.json'),
    installation=receipt(ROOT/'artifacts/patch-installation.json')))
test_log=OUT/'v205-final-tests.log';test_text=test_log.read_text()
assert 'test result: FAILED' not in test_text and 'Doc-tests brass_ai_worker' in test_text
test_count=sum(map(int,re.findall(r'test result: ok\. (\d+) passed',test_text)))
decision=dict(version='search-v205-decision/1',setting='4s_k8',v205_implemented=True,correctness_passed=True,
    decision='retain_cache_only; reject_statistical_reuse_for_selected_candidate',
    rationale='No overall native/Unity latency or decision improvement; slightly fewer new completions and consistently more peak memory. One Canal latency gain repeats, but does not justify enabling the more complex method across this setting.',
    experimental_reuse_available=True,selected_method='search_v2_focus_only',production_method='search_v1',
    production_changed=False,model_final=False,playing_strength='not established',
    study=receipt(study/'protocol.json'),analysis=receipt(study/'analysis.json'),
    candidate=receipt(OUT/'v205-candidate/build-manifest.json'),selected_candidate=receipt(OUT/'v205-cache-candidate/build-manifest.json'),
    native_test_executions=test_count,native_tests=receipt(test_log),envelope_assertions=checks,
    parity_cancellation=receipt(OUT/'v205-native-checks/result.json'),validation=receipt(OUT/'v205-delivery/result.json'),
    unity=unity,installation=receipt(study/'installed-unchanged.json'),source=receipt(study/'source.zip'))
write(study/'decision.json',decision)
print(dict(decision=decision['decision'],native_test_executions=test_count,envelope_assertions=checks,
    independent_plans=len(delivery['rows']),installed_files_unchanged=len(pins)))
