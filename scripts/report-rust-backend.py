"""Evaluate L14 evidence against its frozen protocol; never promote missing data.

Observed quantiles are descriptive screens, not population tail guarantees.
Diagnostic fixed-count traces cannot satisfy timed whole-turn attribution gates.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
import runpy
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / 'scripts/benchmark-rust-backend.py'))
read, write, sha, receipt = (BASE[k] for k in ('read', 'write', 'sha', 'receipt'))


def describe(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values),
                p95=values[math.ceil(.95 * len(values)) - 1], maximum=values[-1])


def benefit(baseline, candidate, policy):
    reduction = baseline - candidate
    return reduction >= policy['minimum_main_stall_reduction_ms'] and (
        baseline > 0 and reduction / baseline >= policy['minimum_main_stall_reduction_fraction'])


def bootstrap(blocks, count=5000):
    """Resample fixture blocks, not individual frames or correlated actions."""
    if not blocks:
        return None
    rng = random.Random(92781)
    estimates = sorted(statistics.mean(rng.choices(blocks, k=len(blocks))) for _ in range(count))
    return dict(mean=statistics.mean(blocks), low=estimates[int(.025 * count)],
                high=estimates[int(.975 * count)], blocks=len(blocks), resamples=count,
                method='paired fixture-block percentile bootstrap; exploratory with eight fixtures')


def check_receipt(pin,out=None):
    path = (ROOT / pin['path']).resolve()
    assert path.is_relative_to(ROOT),pin['path']
    if path.is_file() and sha(path)==pin['sha256']:return path
    if out is not None:
        archived=out/'source-archive'/pin['sha256']/path.name
        if archived.is_file() and sha(archived)==pin['sha256']:return archived
    raise AssertionError('Missing or changed receipt: '+pin['path'])


def fixed_report(path):
    data = read(path)
    assert data['passed'] and data['checks'] > 0 and data['rows']
    paired = defaultdict(dict)
    groups = defaultdict(list)
    for row in data['rows']:
        key = (row['fixture'], row['repeat'], row['action'])
        assert row['arm'] not in paired[key]
        paired[key][row['arm']] = row
        groups[row['arm']].append(row)
    assert all(set(p) == {'baseline', 'candidate'} for p in paired.values())
    metrics = ['snapshot_ms', 'queue_ms', 'background_ms', 'main_commit_ms', 'total_ms', 'cpu_ms']
    summary = {arm: {k: describe([r[k] for r in rows]) for k in metrics} for arm, rows in groups.items()}
    for arm, rows in groups.items():
        summary[arm]['main_thread_ms'] = describe([r['snapshot_ms'] + r['main_commit_ms'] for r in rows])
    return dict(checks=data['checks'], pairs=len(paired), summary=summary,
                allocations=data.get('allocation_measurement'), evidence=receipt(path))


def live_report(out, prefix, repetitions, build):
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    evidence = []
    for rep in range(repetitions):
        for arm in ('baseline', 'candidate'):
            pin = read(out / f'{prefix}-r{rep}-{arm}-live-receipt.json')
            expected=BASE['L13'] if arm=='baseline' else build
            assert pin['build']['sha256']==sha(expected/'build-manifest.json'), 'Mixed build identities in a paired arm'
            for name in ('build', 'input', 'result', 'log'):
                check_receipt(pin[name])
            for source in pin['sources']:
                check_receipt(source,out)
            data = read(check_receipt(pin['result']))
            assert data['passed'] and data['complete']
            for row in data['rows']:
                assert row['complete'] and row['whole_turn_ms'] == 4000
                key = (row['fixture'], rep)
                assert arm not in pairs[key]
                pairs[key][arm] = row
                groups[arm].append(row)
            evidence.append(receipt(out / f'{prefix}-r{rep}-{arm}-live-receipt.json'))
    assert all(set(p) == {'baseline', 'candidate'} for p in pairs.values())
    assert all(p['baseline']['root_sha256'] == p['candidate']['root_sha256'] for p in pairs.values())
    metrics = ['delivery_ms', 'ordinary_clock_ms', 'preparation_and_dispatch_ms', 'max_frame_ms',
               'main_process_cpu_ms', 'native_process_cpu_ms', 'total_cpu_ms', 'native_peak_bytes']
    summary = {arm: {k: describe([r[k] for r in rows]) for k in metrics} for arm, rows in groups.items()}
    fixtures = []
    for name in sorted({k[0] for k in pairs}):
        rows = [p for k, p in pairs.items() if k[0] == name]
        assert len(rows) == repetitions
        item = dict(fixture=name, era=rows[0]['candidate']['era'], partition=rows[0]['candidate']['partition'])
        for metric in metrics:
            a = statistics.median(r['baseline'][metric] for r in rows)
            b = statistics.median(r['candidate'][metric] for r in rows)
            item[metric] = dict(baseline=a, candidate=b, delta=b-a, ratio=b/a if a else None)
        fixtures.append(item)
    growth = []
    for rep in range(repetitions):
        data = read(out / f'{prefix}-r{rep}-candidate-live.json')['rows']
        growth.append(data[-1]['native_retired_bytes_after'] - data[0]['native_retired_bytes_before'])
    return dict(pairs=len(pairs), summary=summary, fixtures=fixtures,
                delivery_delta_ci=bootstrap([r['delivery_ms']['delta'] for r in fixtures]),
                frame_delta_ci=bootstrap([r['max_frame_ms']['delta'] for r in fixtures]),
                retained_growth_bytes=describe(growth), evidence=evidence)


def stage_report(path):
    data = read(path)
    spans = [r for r in data['samples'] if 'wall_ms' in r]
    children = defaultdict(list)
    for row in spans:
        children[row['parent']].append(row)
    stages = defaultdict(list)
    for row in spans:
        # Union child intervals on THIS execution thread. Cross-thread work is
        # concurrent and must not be subtracted from a main-thread frame.
        intervals = sorted((max(row['started'], c['started']), min(row['end'], c['end']))
                           for c in children[row['id']] if c['thread'] == row['thread'])
        covered = 0
        end = row['started']
        for start, stop in intervals:
            covered += max(0, stop - max(start, end))
            end = max(end, stop)
        item = dict(row, exclusive_ms=max(0, row['wall_ms'] - covered*1000/data['frequency']))
        stages[row['stage']].append(item)
    summary = {name: dict(wall_ms=describe([r['wall_ms'] for r in rows]),
                         exclusive_ms=describe([r['exclusive_ms'] for r in rows]),
                         cpu_ms=describe([r['cpu_ms'] for r in rows]),
                         threads=sorted({r['thread'] for r in rows})) for name, rows in stages.items()}
    sizes = defaultdict(list)
    for r in data['samples']:
        if 'bytes' in r:
            sizes[r['stage']].append(r['bytes'])
    return dict(evidence=receipt(path), stages=summary, bytes={k: describe(v) for k, v in sizes.items()},
                allocation_counter_available=data['allocation_counter_available'],
                limitations=['Pipe-read wait overlaps native computation; it is not pure transport overhead.',
                             'Native ready timestamps have their own clock origin; no guessed cross-process subtraction.',
                             'Unity Mono has no allocated-byte counter. Heap deltas and GC counts are not allocations.',
                             'Inclusive parent and child times must not be summed.'])


def control_report(out,legacy_label):
    result={}
    for label in ('fixed-controls',legacy_label,'native-components'):
        folder=out/label;path=folder/'results.json'
        if not path.exists():
            result[label]=dict(status='pending',reason='No completed result; partial trials are not a passing control.')
            continue
        data=read(path)
        for pin in data.get('artifacts',[]):check_receipt(pin,out)
        protocol=folder/'protocol.json'
        item=dict(status='passed' if data['passed'] else 'failed',evidence=receipt(path),protocol=receipt(protocol),
                  timed_gumbel_gate_eligible=False,strength_gate_eligible=False)
        runner=folder/'runner-source.json'
        if runner.exists():
            check_receipt(read(runner),out);item['runner']=receipt(runner)
        if label=='fixed-controls':
            item.update(cases=len(data['records']),scope=data['attribution'])
        elif label=='native-components':
            item.update(summary=data['summary'],scope='Native-only component and four-second controls; no matched implementation ratio.')
        else:
            check_receipt(data['raw'],out)
            fixtures={f['id']:f for f in read(protocol)['fixtures']}
            def boundary(row):
                fixture=fixtures[row['fixture']]
                phase=fixture.get('phase') or next(t.removeprefix('phase:') for t in fixture['tags'] if t.startswith('phase:'))
                return 'ordinary' if phase=='action' else 'staged'
            item.update(turns=len(data['rows']),scope=data['scope'],
                        completed_by_boundary={kind:{arm:describe([r['completed'] for r in data['rows'] if r['arm']==arm and boundary(r)==kind])
                            for arm in ('A','B')} for kind in ('ordinary','staged')})
        result[label]=item
    return result


def report(out, build, prefix, validation_label, name,legacy_label='legacy-controls-v4'):
    protocol = BASE['checked'](out)
    destination = out / (name + '.json')
    assert not destination.exists(), 'Reports are immutable; choose another --name'
    fixed_pin=read(out/(validation_label+'-validation-receipt.json'))
    for key in ('binary','build','measurement'):check_receipt(fixed_pin[key])
    for source in fixed_pin['sources']:check_receipt(source,out)
    assert fixed_pin['build']['sha256']==sha(build/'build-manifest.json')
    fixed = fixed_report(out / (validation_label + '-validation.json'))
    live = live_report(out, prefix, protocol['matched_repetitions'],build)
    policy = protocol['validation_benefit']
    lifecycle=read(out/(prefix+'-lifecycle-receipt.json'))
    check_receipt(lifecycle['build'])
    for source in lifecycle['sources']:check_receipt(source,out)
    unity_path=check_receipt(lifecycle['evidence'])
    unity = read(unity_path)
    preserved = read(build / 'preserved-runtime.json')
    events=read(out/'contract-checks/validation-events.json')
    checkpoints=read(out/'contract-checks/checkpoint-tests.json')
    before = fixed['summary']['baseline']['main_thread_ms']
    after = fixed['summary']['candidate']['main_thread_ms']
    frames = {a: live['summary'][a]['max_frame_ms']['median'] for a in ('baseline', 'candidate')}
    gates = {}
    def gate(name, passed, reason):
        gates[name] = dict(status='passed' if passed else 'failed', reason=reason)
    gate('validation_materiality', before['p95'] >= protocol['validation_material']['p95_main_thread_ms'] or
         before['median'] >= protocol['validation_material']['median_per_turn_ms'], 'Frozen millisecond thresholds; fixed replies.')
    gate('validation_correctness', unity['passed'] and not unity['nativeErrors'] and preserved['passed'] and events['passed'] and checkpoints['passed'],
         'Fixed legal/invalid-path parity and actual Unity identity/cancellation/action-two/checkpoint lifecycle.')
    gate('validation_main_thread_benefit', benefit(before['median'], after['median'], policy) and
         benefit(frames['baseline'], frames['candidate'], policy), 'Both fixed-reply main-thread work and uninstrumented worst-frame medians.')
    gate('validation_delivery_regression', all(r['delivery_ms']['delta'] <= policy['end_to_end_max_regression_ms'] for r in live['fixtures']),
         'Every fixture median must remain within the frozen 20 ms regression tolerance.')
    gate('validation_cpu_regression', all(r['total_cpu_ms']['ratio'] <= 1 + policy['cpu_max_regression_fraction'] for r in live['fixtures']),
         'Unity plus native process CPU, measured over the same delivery interval; each fixture median.')
    gate('live_delivery_screen', live['summary']['candidate']['ordinary_clock_ms']['p95'] <= protocol['performance']['response_p95_ms'] and
         live['summary']['candidate']['ordinary_clock_ms']['maximum'] <= protocol['performance']['response_max_ms'],
         'Conservative full live delivery from the ordinary clock, including final checks/commit; observed screen, not a tail guarantee.')
    memory_path=out/'native-memory/results.json'
    memory_evidence=None
    if memory_path.exists():
        memory=read(memory_path)
        memory_protocol=read(memory_path.parent/'protocol.json')
        for key in ('model','package'):check_receipt(memory_protocol[key],out)
        assert sha(BASE['L13']/'BrassNative/alpha-g890/brass-ai-local.exe')==sha(build/'BrassNative/alpha-g890/brass-ai-local.exe')
        memory_evidence=dict(result=receipt(memory_path),protocol=receipt(memory_path.parent/'protocol.json'),
            peak_bytes=memory['peak_bytes'],retained_growth_bytes=memory['retained_growth_bytes'],turns=memory['turns'],scope=memory['scope'])
        gate('memory_screen', memory['peak_bytes']>0 and memory['peak_bytes']<=protocol['memory']['process_peak_bytes'] and
             memory['retained_growth_bytes']<=protocol['memory']['post_retirement_growth_bytes'],
             'Win32 counters on the same production native binary after warmup; repeated heterogeneous turns. Long-game leak qualification remains separate.')
    else:
        gates['memory_screen']=dict(status='pending',reason='Unity Mono returned zero memory counters; zero is unavailable, not a passing measurement.')
    for key, reason in {
        'equivalent_rules_throughput': 'Requires an identical staged C#/Rust transition trace measured on this candidate.',
        'equivalent_timed_gumbel': 'Existing G_py is fixed-count only; L11/L12 timed/complete-plan extensions and external-evaluator G_ref are not yet matched.',
        'strength': 'Requires the frozen 16 seat/setup-balanced complete games per control and confidence/plan-support gates.',
        'allocation_profile': 'Unity allocation counts are unavailable; obtain an allocation profiler trace before completing this requirement.',
        'native_ipc_attribution': 'Add diagnostic per-request native service timestamps to separate transport from overlapping native work exactly.'
    }.items():
        gates[key] = dict(status='pending', reason=reason)
    traces = {arm: stage_report(out / f'{prefix}-{arm}-stage-trace.json') for arm in ('baseline', 'candidate')
              if (out / f'{prefix}-{arm}-stage-trace.json').exists()}
    if len(traces) != 2:
        gates['stage_breakdown'] = dict(status='pending', reason='Both final diagnostic profiles are required.')
    else:
        gates['stage_breakdown'] = dict(status='passed', reason='Both traced stage reports are retained; attribution limitations are separate open gates.')
    validation_pass = all(gates[k]['status'] == 'passed' for k in gates if k.startswith('validation_'))
    acceptance=out/'optimization-retention-decision.json'
    retained=validation_pass or (acceptance.exists() and read(acceptance)['correctness_waived'] is False and
                                gates['validation_correctness']['status']=='passed' and gates['validation_main_thread_benefit']['status']=='passed')
    result = dict(format='brass-l14-assessment/1.0.0', step='L14', status='qualified' if all(g['status']=='passed' for g in gates.values()) else 'open',
                  installed=False, validation_optimization_passes_original_screens=validation_pass,
                  validation_optimization_retained=retained,retention_decision=receipt(acceptance) if acceptance.exists() else None,
                  gates=gates, protocol=receipt(out/'protocol.json'),
                  build=receipt(build/'build-manifest.json'), unity=receipt(unity_path), preservation=receipt(build/'preserved-runtime.json'),
                  fixed_reply=fixed, matched_live=live, diagnostic_profiles=traces,
                  native_memory=memory_evidence,search_controls=control_report(out,legacy_label),
                  event_identity_tests=receipt(out/'contract-checks/validation-events.json'),checkpoint_tests=receipt(out/'contract-checks/checkpoint-tests.json'),
                  source_pins=[receipt(ROOT/'scripts/report-rust-backend.py')],
                  interpretation='A/B Masked and fixed-count Gumbel diagnostics cannot pass matched timed Gumbel or strength gates. No missing evidence is treated as a pass.')
    write(destination, result)
    print(result['status'], {k:g['status'] for k,g in gates.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--prefix', default='final')
    parser.add_argument('--validation-label', default='final-paired')
    parser.add_argument('--name', default='assessment')
    parser.add_argument('--legacy-label',default='legacy-controls-v4')
    args = parser.parse_args()
    report(args.out.resolve(), args.build.resolve(), args.prefix, args.validation_label, args.name,args.legacy_label)
