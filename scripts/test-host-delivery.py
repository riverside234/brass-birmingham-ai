"""Short exact-Mono host parity, event/identity and allocation probes."""
from pathlib import Path
import argparse
import json
import runpy
import shutil
import subprocess
import statistics

ROOT = Path(__file__).resolve().parents[1]
B = runpy.run_path(str(ROOT / 'scripts/build-host-delivery.py'))
STUDY, read, write, sha = (B[k] for k in ('STUDY', 'read', 'write', 'sha'))
COMPILER = ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
FIXTURES = ROOT / 'artifacts/rust-backend/20261026-l14-study-v2/validation-fixtures.json'
HOST = ROOT / 'artifacts/rust-backend/20261026-local-release/allocation-final/Managed/allocation-host.exe'


def probe(arm):
    build = STUDY / arm
    managed = build / 'Managed'
    out = STUDY / ('probe-' + arm)
    out.mkdir(exist_ok=False)
    pinned_host = read(HOST.parent.parent / 'receipt.json')['host']['sha256']
    assert sha(HOST) == pinned_host
    shutil.copyfile(HOST, out / HOST.name)
    rows = read(FIXTURES)
    write(out / 'allocation-fixtures.json', [next(r for r in rows if r['input']['view']['context']['era'] == e)
        for e in ('CanalEra', 'RailEra')])
    sources = [ROOT / 'tests/RustBackendLocal' / n for n in
        ('HostDeliveryChecks.cs', 'ValidationBench.cs', 'AllocationBench.cs', 'NativeValidationContractChecks.cs')]
    for source in sources:
        shutil.copyfile(source, out / source.name)
    for main, fixture, result in [('HostDeliveryChecks', FIXTURES, 'host.json'),
        ('AllocationBench', out / 'allocation-fixtures.json', 'allocation.json'),
        ('NativeValidationContractChecks', FIXTURES, 'contracts.json')]:
        exe = out / (main + '.exe')
        rsp = out / (main + '.rsp')
        args = ['/nologo', '/target:exe', '/main:' + main, '/nostdlib+', '/langversion:7.3', '/optimize+', '/deterministic+', f'/out:"{exe}"']
        args += [f'/reference:"{p}"' for p in managed.glob('*.dll') if not any(n in p.name for n in ('Tests', 'Regression'))]
        args += [f'"{out / p.name}"' for p in sources]
        rsp.write_text('\n'.join(args), encoding='utf-8-sig')
        subprocess.run([str(COMPILER), '@' + str(rsp)], check=True)
        command = [str(out / HOST.name), str(ROOT / 'MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll'),
            str(managed), str(ROOT / 'MonoBleedingEdge/etc'), str(exe), str(fixture), str(out / result)]
        with (out / (main + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        data = read(out / result)
        assert data['passed'], data.get('error')
        print(arm, main, data.get('checks', 'passed'), flush=True)
    write(out / 'receipt.json', dict(passed=True, arm=arm, build_manifest_sha256=sha(build / 'build-manifest.json'),
        sources=[dict(path=p.name, sha256=sha(out / p.name)) for p in sources],
        results={n:sha(out / n) for n in ('host.json', 'allocation.json', 'contracts.json')},
        allocation_host_sha256=sha(out / HOST.name), runtime='exact-shipped-Mono/allocation-enabled-isolated-host'))


def compare():
    reports = {a: read(STUDY / ('probe-' + a) / 'host.json') for a in ('baseline', 'candidate')}
    # The journal's creation timestamp is intentionally volatile. Nothing else
    # in the observation, complete action, event or journal contract is removed.
    for data in reports.values():
        for row in data['journals']:
            row.pop('created_utc')
    for key in ('roots', 'results', 'journals'):
        assert reports['baseline'][key] == reports['candidate'][key], key + ' changed'
    assert not reports['baseline']['optimized_resolver'] and reports['candidate']['optimized_resolver']
    allocation = {a: read(STUDY / ('probe-' + a) / 'allocation.json') for a in reports}
    totals = {a: sum(sum(r['bytes'].values()) for r in d['rows'] if r['arm'] == 'candidate') for a, d in allocation.items()}
    times = {a: statistics.median(r['total_ms'] for r in d['timings']) for a, d in reports.items()}
    result = dict(passed=True, exact_observation_legal_sets_event_order_and_payloads=True,
        roots=len(reports['candidate']['roots']), full_actions=len(reports['candidate']['results']),
        checks={a:d['checks'] for a,d in reports.items()},
        eight_action_managed_object_bytes=totals, allocation_reduction_fraction=1-totals['candidate']/totals['baseline'],
        fixed_reply_median_action_ms=times, fixed_reply_reduction_fraction=1-times['candidate']/times['baseline'],
        limitations=['Bounded recorded roots; exact Mono allocation counters add overhead to these isolated timing samples.',
            'Live Unity timing is measured separately; allocation traffic is not retained memory or whole-player allocation.'])
    assert totals['candidate'] < totals['baseline'], 'No allocation improvement'
    write(STUDY / 'host-assessment.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=('baseline', 'candidate', 'compare'))
    arm = parser.parse_args().arm
    compare() if arm == 'compare' else probe(arm)
