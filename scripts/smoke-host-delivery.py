"""Bounded, isolated 4s/K16+K32 Unity comparison; never installs a build."""
from pathlib import Path
import argparse
import json
import os
import runpy
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
B = runpy.run_path(str(ROOT / 'scripts/build-host-delivery.py'))
STUDY, BASE, read, write, sha = (B[k] for k in ('STUDY', 'BASE', 'read', 'write', 'sha'))
CASES = {
    'canal-k16': ('live-92781-CanalEra-2.json', 16, ('baseline', 'candidate')),
    'canal-k32': ('live-92781-CanalEra-2.json', 32, ('candidate', 'baseline')),
    'rail-k16': ('live-92782-RailEra-7.json', 16, ('candidate', 'baseline')),
    'rail-k32': ('live-92782-RailEra-7.json', 32, ('baseline', 'candidate')),
}


def protocol():
    path = STUDY / 'protocol.json'
    data = dict(scope='short-host-delivery-smoke', allowance_ms=4000, temperature=1.0,
        sessions=2, threads_per_session=1, search_method='search_v1', v205_enabled=False,
        search_k=[16,32], matched_cases=CASES, unchanged_native_files=read(BASE / 'build-manifest.json')['nativeFiles'],
        baseline_manifest_sha256=sha(STUDY / 'baseline/build-manifest.json'),
        candidate_manifest_sha256=sha(STUDY / 'candidate/build-manifest.json'),
        decisions=['Retain only with semantic/event/identity parity and useful measured benefit.',
            'Keep the native 4s allowance and both K-specific reserves unchanged.',
            'Two roots per K, one paired sample each; no strength or population-tail claim.'])
    if path.exists():
        assert read(path) == json.loads(json.dumps(data))
    else:
        write(path, data)


def invoke(build, label, fixture, k, scenario='normal', profile=False):
    out = STUDY / label
    assert not out.exists(), 'Preserve old test output'
    fixture_path = ROOT / 'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures' / fixture
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', BRASS_SEARCH_V2_RESULT=str(out / 'result.json'),
        BRASS_SEARCH_V2_EVENTS=str(fixture_path), BRASS_SEARCH_V2_MS='4000',
        BRASS_SEARCH_V2_SCENARIO=scenario, BRASS_SEARCH_K=str(k), BRASS_NN_TEMPERATURE='1.0')
    if profile:
        env['BRASS_L14_TRACE'] = str(out / 'stage-trace.json')
    else:
        env.pop('BRASS_L14_TRACE', None)
    command = [shutil.which('pwsh'), '-NoProfile', '-File', str(ROOT / 'tests/RustBackendLocal/run-unity.ps1'),
        '-BuildDirectory', str(build.relative_to(ROOT)), '-EvidenceDirectory', str(out.relative_to(ROOT)), '-SearchV2', '-MaxSeconds', '60']
    with (STUDY / (label + '.log')).open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=85)
    result, unity = read(out / 'result.json'), read(out / 'unity.json')
    assert result['passed'] and unity['passed'], result
    assert result['search_k'] == k and result['allowance_ms'] == 4000 and result['search_method'] == 'search_v1' and not result['focused']
    assert all(r['requested_k'] == k and r['sessions'] == 2 and r['threads_per_session'] == 1
        and r['policy_temperature'] == 1.0 and r['response_ready_ns'] < r['deadline_ns'] for r in result['replies'])
    if scenario == 'normal':
        assert len(result['commits']) == 2 and result['commits'][-1]['remaining'] == 0
    else:
        assert len(result['commits']) == 1 and result['commits'][-1]['remaining'] == 1
    write(out / 'receipt.json', dict(passed=True, scope='isolated-player/4s', fixture_sha256=sha(fixture_path),
        build_manifest_sha256=sha(build / 'build-manifest.json'), result_sha256=sha(out / 'result.json'),
        unity_sha256=sha(out / 'unity.json'), diagnostic_only=profile,
        checks=unity['checks'], k=k, scenario=scenario, delivery_ms=result['delivery_ms']))
    print(label, 'checks', unity['checks'], 'delivery_ms', round(result['delivery_ms'], 2), flush=True)


def profiles():
    module = runpy.run_path(str(ROOT / 'scripts/benchmark-rust-backend.py'))
    for arm in ('baseline', 'candidate'):
        module['profile_build'](STUDY, STUDY / arm, arm, validate_study=False)
        invoke(STUDY / (arm + '-profile-build'), 'profile-' + arm, CASES['canal-k32'][0], 32, profile=True)


def report():
    assert not (STUDY / 'assessment.json').exists(), 'Preserve sealed assessment'
    rows = []
    for case, (_, k, _) in CASES.items():
        pair = {a: read(STUDY / (case + '-' + a) / 'result.json') for a in ('baseline', 'candidate')}
        assert pair['baseline']['input'] == pair['candidate']['input'], 'Permitted root differs'
        row = dict(case=case, k=k)
        for arm, data in pair.items():
            ready = sum(r['response_ready_ns']/1e6 for r in data['replies'])
            row[arm] = dict(delivery_ms=data['delivery_ms'], native_ready_ms=ready,
                outside_native_timer_ms=data['delivery_ms']-ready, unity_cpu_ms=data['unity_cpu_ms'],
                requests=data['requests'], actions=data['replies'][0]['actions'])
        row['same_selected_actions'] = row['baseline']['actions'] == row['candidate']['actions']
        row['outside_native_reduction_ms'] = row['baseline']['outside_native_timer_ms'] - row['candidate']['outside_native_timer_ms']
        rows.append(row)
    stages = {}
    for arm in ('baseline', 'candidate'):
        data = read(STUDY / ('profile-' + arm) / 'result.json')
        trace = read(STUDY / ('profile-' + arm) / 'stage-trace.json')
        instance = data['replies'][0]['ticket']['game_instance']
        samples = [s for s in trace['samples'] if instance in (s.get('request') or '')]
        totals = {}
        for s in samples:
            if 'wall_ms' in s:
                entry = totals.setdefault(s['stage'], dict(count=0, wall_ms=0))
                entry['count'] += 1
                entry['wall_ms'] += s['wall_ms']
        stages[arm] = dict(stages=totals, delivery_ms=data['delivery_ms'],
            outside_native_roundtrip_ms=data['delivery_ms']-totals['native.roundtrip']['wall_ms'])
    fixed = read(STUDY / 'host-assessment.json')
    assert fixed['passed']
    edge = {a:read(STUDY / ('edge-v3-' + a) / 'result.json') for a in ('baseline','candidate')}
    assert edge['baseline'] == edge['candidate'] and edge['candidate']['passed']
    lifecycle = {s:read(STUDY / ('lifecycle-' + s) / 'receipt.json') for s in ('cancel','load')}
    assert all(r['passed'] for r in lifecycle.values())
    broad = read(STUDY / 'validation-lifecycle-v2/unity.json')
    assert broad['passed'] and broad['checks'] >= 830
    contracts_path = ROOT / 'artifacts/rust-backend/20261026-l14-study-v2/host-delivery-contracts-20261029'
    contracts = {n:read(contracts_path / n) for n in ('validation-events.json','checkpoint-tests.json')}
    assert all(r['passed'] for r in contracts.values())
    assert all(r['outside_native_reduction_ms'] > 0 for r in rows), 'Investigate non-improving live case'
    installed = read(ROOT / 'artifacts/patch-installation.json')
    assert installed['version'] == '1.4.43-rust-local'
    assert sha(ROOT / 'artifacts/patch-installation.json') == read(STUDY / 'baseline/build-manifest.json')['installed_baseline_receipt_sha256']
    for pin in installed['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / pin['name']) == pin['sha256'].lower()
    candidate = read(STUDY / 'candidate/build-manifest.json')
    old = {p['path']:p['sha256'] for p in read(BASE / 'build-manifest.json')['sourcePins']}
    changed = []
    for pin in candidate['sourcePins']:
        assert sha(STUDY / 'candidate/runtime' / pin['path']) == pin['sha256']
        if old[pin['path']] != pin['sha256']:
            changed.append(pin['path'])
            assert sha(ROOT / pin['path']) == pin['sha256']
    assert set(changed) == set(B['SCOPE'])
    assert B['pins'](STUDY / 'candidate/BrassNative') == candidate['nativeFiles'] == B['pins'](ROOT / 'BrassNative')
    for pin in candidate['files']:
        assert sha(STUDY / 'candidate/Managed' / pin['name']) == pin['sha256']
        if pin['name'] != 'BrassLegacyAI.dll':
            assert pin['sha256'] == next(p['sha256'] for p in installed['files'] if p['name'] == pin['name'])
    output = dict(passed=True, installed=False, matched_smokes=rows, diagnostic_stages=stages,
        fixed_reply_assessment=fixed, search_method='search_v1', v205_enabled=False,
        changed_sources=changed, lifecycle=lifecycle, broad_lifecycle_checks=broad['checks'],
        edge_checks=edge['candidate']['checks'], event_identity_checks=contracts['validation-events.json']['checks'],
        checkpoint_checks=contracts['checkpoint-tests.json']['checks'],
        preserved_failures=[
            dict(path='edge-candidate/result.json',reason='Extra test assumed a recorded root owned an industry; corrected synthetic debt perspective conserves hand count.'),
            dict(path='edge-v2-candidate/result.json',reason='Extra test assumed the late Canal fixture avoided debt; corrected test completes its real legal liquidation.'),
            dict(path='validation-lifecycle/unity.json',reason='Old test fixture reloaded the human checkpoint after writing a newer AI checkpoint; corrected to load the latest completed turn.')],
        scope='four paired live turns, short lifecycle checks and fixed-reply oracles',
        limitations=['Native response-ready timers have a different start point; their residual includes preparation and dispatch.',
            'Diagnostic stages are nested; do not add their totals together.',
            'One paired sample per K/root; no strength or population-tail claim.'])
    write(STUDY / 'assessment.json', output)
    report_path = ROOT / 'ai-research/HOST_DELIVERY_OPTIMIZATION_REPORT.md'
    assert not report_path.exists()
    table = '\n'.join('| '+r['case']+' | '+f"{r['baseline']['outside_native_timer_ms']:.1f} ms | {r['candidate']['outside_native_timer_ms']:.1f} ms | {100*(1-r['candidate']['outside_native_timer_ms']/r['baseline']['outside_native_timer_ms']):.1f}% |" for r in rows)
    text = '''# Host delivery optimization — qualified local candidate

Date: 2026-10-29 (America/Chicago). Implemented and smoke-tested in the isolated Unity player. **This task does not install the candidate:** the game remains the verified 1.4.43 release. The frozen candidate is `artifacts/host-delivery/20261029/candidate/`.

The installed `search_v1` binary, V205-off setting, g890 temporary NN, ONNX Runtime, two independent one-thread sessions, all policies and both K-specific deadline reserves are unchanged. All matched live turns use a four-second search allowance and temperature 1.0. K16 retains its original reserve; K32 retains 1.2 seconds. These results do not establish playing strength or latency tails.

## Implementation

- After ordinary action one, `search_v1` captures only the clock-journal fields instead of replaying public history into a second complete NN input. The projected-successor path retains its full observation and comparison when that contract is present.
- Event history is reused only within one bridge, rules object and revision. Changing the revision/object reloads it; loading or moving history cancels the bridge. NN turns skip legacy intent-history arguments when there is no intent. Two ordinary actions now require two storage reads instead of eight; the profiler still sees six cheap-or-real `Events()` calls.
- Allowlisted observations build JSON tokens directly. General `Wire.Token` parsing semantics are unchanged. Board-object indexes bind the rules object and era; mutable tile/owner values remain live. The selected-choice resolver shares the canonical enumeration predicates, filters to the requested branch, and rejects missing or ambiguous selected IDs. Canonical complete enumeration stays available and supplies the oracle.
- Only detached validation delivery uses a 60 FPS cap. Native search remains at 30 FPS, concurrent search takes priority, and an existing lower software cap is respected. Optional-work suspension, cancellation ownership and exact frame/VSync restoration remain active.

Identity, revision, actor, cancellation, exact observation/composition/path equality, single-use receipt and current semantic legality checks still precede main-thread mutation. Action two is captured after the actual first action. The original turn clock is never replenished.

## Short matched Unity results

Each row is one fresh baseline/candidate pair with the same permitted root, native package, settings and selected actions. Execution order is counterbalanced. The displayed difference is total delivery minus native response-ready duration; it includes preparation and dispatch because those timers start at different boundaries.

| Root / setting | Baseline outside native timer | Candidate outside native timer | Reduction |
| --- | ---: | ---: | ---: |
'''+table+'''

Both actions and completed-turn save/load passed in all eight runs. Native response-ready timing varied between processes: the K16 late-Rail candidate took 3.701 seconds overall versus 2.432 seconds for baseline, while its outside-native portion still improved from 463.5 to 215.7 ms. The unchanged native binary took 3.485 versus 1.968 seconds in that pair. Therefore this smoke supports a host-side improvement, not a claim that every total turn becomes faster.

A separate K32 diagnostic pair measured C# time outside its containing native round trip at **338.8 → 149.1 ms**. Full NN captures fell from two to one; detached validation spans totaled **107.6 → 34.2 ms**, guarded commits **58.2 → 7.3 ms**, and event-history method spans **51.7 → 23.3 ms**. These spans are nested and must not be added together. Instrumentation changes timing; raw uninstrumented results remain separate.

## Fixed replies and correctness

The exact shipped Mono runtime checked 40 ordinary/staged states and 15 complete recorded actions. Both builds passed 3,719 checks and produced exactly equal observations, complete legal sets, committed states, semantic event payload/order/revisions and clock journals. Only the journal creation timestamp is excluded from the cross-process comparison. A separate 300-check oracle covers bonus skip/develop, explicit debt membership and a real seven-turn Canal-to-Rail transition with two legal liquidations; baseline/candidate results are identical.

Across eight matched actions, retained-path managed object allocations fell **383,082,424 → 110,887,688 bytes (71.1%)**. The small allocation-instrumented fixed-action timing sample had median **149.1 → 38.6 ms (74.1% lower)**. These are managed allocation traffic and instrumented isolated timings, not whole-Unity allocation totals, retained memory or an uninstrumented latency guarantee.

The actual Unity lifecycle suite passed **839 checks**, including queued/retiring work, stale actor/revision, same-revision load, fresh bridge after load, cancellation, receipt reuse boundaries, action-two validation, family/difficulty setup and worker failure. Separate post-action-one K16 cancellation and K32 loading smokes passed. Another **45 event/identity + 79 checkpoint/liquidation checks** passed. Saves remain once after completed human/AI End, with no per-action AI save.

## Preserved failed attempts

Two supplemental test attempts had incorrect fixture assumptions: no recorded ordinary root owned collateral, and the late-Canal replay entered real debt. The corrected test uses an explicit conserved debt perspective and executes the actual liquidation. Their failed evidence remains in `edge-candidate/` and `edge-v2-candidate/`.

The first broader lifecycle run found an older test-helper error: it returned the human checkpoint object after already saving a newer AI-turn checkpoint. The helper now reloads the newest checkpoint. The original failure remains in `validation-lifecycle/`; the corrected suite is `validation-lifecycle-v2/`. No production save behavior was changed to satisfy the test.

## Evidence and reproduction

See [assessment](../artifacts/host-delivery/20261029/assessment.json), [fixed-reply assessment](../artifacts/host-delivery/20261029/host-assessment.json), [protocol](../artifacts/host-delivery/20261029/protocol.json), and [candidate qualification](../artifacts/host-delivery/20261029/candidate/qualification.json).

The scoped builder is `scripts/build-host-delivery.py`. `scripts/test-host-delivery.py` runs the exact-Mono oracles/allocation probes; `scripts/check-host-delivery-edges.ps1` runs supplemental branches; `scripts/smoke-host-delivery.py` runs individual short matched pairs, diagnostic profiles, lifecycle smokes and sealing. Existing output directories are protected. Use a fresh study label when reproducing; retain all failed attempts. Native/lab packages, installed managed files, user preferences and gameplay saves are unchanged.
'''
    report_path.write_text(text, encoding='utf-8')
    write(STUDY / 'candidate/qualification.json', dict(passed=True, installed=False,
        scope='local-host-delivery/K16+K32/4s', assessment_sha256=sha(STUDY / 'assessment.json'),
        report_sha256=sha(report_path), build_manifest_sha256=sha(STUDY / 'candidate/build-manifest.json'),
        source_scope=changed, native_unchanged=True, files=B['pins'](STUDY / 'candidate'),
        evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(STUDY.rglob('*.json')) if p.relative_to(STUDY).parts[0] != 'candidate' and p.name in ('receipt.json','host-assessment.json','assessment.json','protocol.json','unity.json')]))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=tuple(CASES) + ('profiles','lifecycle','report'))
    args = parser.parse_args()
    protocol()
    if args.group in CASES:
        fixture, k, order = CASES[args.group]
        for arm in order:
            invoke(STUDY / arm, args.group + '-' + arm, fixture, k)
    elif args.group == 'profiles':
        profiles()
    elif args.group == 'lifecycle':
        for scenario, k in [('cancel',16),('load',32)]:
            invoke(STUDY / 'candidate', 'lifecycle-' + scenario, CASES['canal-k16'][0], k, scenario)
    else:
        report()
