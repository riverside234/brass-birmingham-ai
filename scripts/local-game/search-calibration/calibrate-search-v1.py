"""Re-calibrate the local search_v1 startup estimate (policy `startup_ns`) on the installed engine-speed packages.

The timed local search sizes a turn's first search from this estimate: quota = available time / estimate, per lane,
and answers once the quota is done. The installed packages still carry the L12 estimate (7.26 ms per simulation),
measured on the engine before the decision-reuse change, so they now stop after about half their time.

Method, as at L12: a singleton (cap-one) search of about 64 trajectories per owner, timed from the request's receipt
to that owner's last completed trajectory, so root preparation is included; cost = that time / completed
trajectories. Every package, K 16 and K 8, every captured position that searches (forced and bypassed roots
complete none), two interleaved repetitions. The estimate is the maximum. A measuring copy of each package lives
under tmp/; the installed packages are not touched.

Stage 2 (`full <stage-1 estimate>`) times the same lanes at the quota a turn actually runs, where root preparation
is spread over several hundred trajectories, and recommends that maximum (rounded up to a microsecond). `verify <estimate_ns>` runs
every position at the recommended estimate (K 16 and K 8 at 4 s, K 16 at 12 s); every search must complete its quota.
"""
import json
import runpy
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEED = runpy.run_path(str(ROOT / 'scripts/local-game/engine-speed/qualify-engine-speed.py'))
sha, read, write, position = (SPEED[key] for key in ('sha', 'read', 'write', 'position'))
PACKAGES = list(SPEED['PACKAGES'])
STUDY = ROOT / 'artifacts/search-calibration'
SCRATCH = ROOT / 'tmp/search-calibration'
CAPTURE = SPEED['CAPTURE']
POSITIONS = (None, *range(0, 44))  # the warm-up root and every captured position
MEASURE_NS = 46_000_000  # about 64 trajectories per owner in a 4 s allowance
ALLOWANCE_MS = 4000
MARGIN = 1.  # the calibration maximum itself, as at L12 (measured under sustained load)
SOURCE_VERSION = '1.4.61-rust-local'


def copy_package(model, startup_ns, label, k):
    """A measuring copy of the installed package with one extra policy: `startup_ns`, K `k` at every allowance."""
    source, target = ROOT / 'BrassNative' / model, SCRATCH / model
    if not target.exists():
        shutil.copytree(source, target)
    policy = read(source / ('policy.json' if k == 16 else f'policy-k{k}.json'))
    policy.update(startup_ns=startup_ns, startup_source=label)
    name = f'policy-measure-k{k}.json'
    write(target / name, policy)
    return target, name


def worker(package, name):
    Worker = runpy.run_path(str(ROOT / 'tests/RustBackendLocal/test_local.py'))['Worker']
    w = Worker(package, policy=name)
    assert w.hello['data']['measured_readiness']['warmup_passed'], w.hello
    return w


def search(w, package, model, index, instance, allowance=ALLOWANCE_MS):
    inp = position(package, index)
    if model == 'alpha-g890':
        inp['context']['profile_id'] = 'alpha-test-g890-neutral/1.0.0'
    ready = w.hello['data']
    # A fresh game instance per search: every search is a turn's first, sized by the startup estimate.
    ticket = dict(game_instance=instance, game_id='calibration', generation=2, revision=inp['view']['context']['revision'],
                  actor=inp['view']['seat'], profile=inp['context']['profile_id'], source_sha256=ready['build']['source_sha256'])
    request_id = w.send(dict(op='search', input=inp, ticket=ticket, allowance_ms=allowance, remaining_ms=allowance))
    reply = w.recv(timeout=allowance / 1000 + 30)
    assert reply['id'] == request_id, reply
    data = reply['data']
    if reply['status'] != 'ok':
        return dict(position='warmup' if index is None else index, error=data.get('error'), owners=[])
    # A forced, debt or bypassed root has no quota and completes no trajectory.
    owners = [dict(completed=o['completed'], quota=o['quota']['simulations'], available_ns=o['quota']['available_ns'],
                   estimate_ns=(o['quota'].get('estimate') or {}).get('nanoseconds'),
                   last_completion_ns=o['last_completion_ns'], work_stopped_ns=o['work_stopped_ns'])
              for o in data['replay']['owners'] if o.get('quota') is not None]
    return dict(position='warmup' if index is None else index, owners=owners, response_ready_ns=data['response_ready_ns'],
                deadline_ns=data['deadline_ns'], provenance=data['provenance'])


def sweep(estimate_ns, label, ks, reps, positions=POSITIONS, models=PACKAGES, allowance=ALLOWANCE_MS):
    """Timed searches at `estimate_ns`, interleaved over models and K; roots that search nothing are left out."""
    workers = {}
    for model in models:
        for k in ks:
            package, name = copy_package(model, estimate_ns, label, k)
            workers[model, k] = (worker(package, name), package)
    rows = []
    try:
        for rep in range(reps):
            for index in positions:
                for model in (models if rep % 2 == 0 else models[::-1]):
                    for k in (ks if rep % 2 == 0 else ks[::-1]):
                        w, package = workers[model, k]
                        result = search(w, package, model, index, f'{label}-{model}-{k}-{index}-{rep}-{allowance}', allowance)
                        if not any(o['completed'] for o in result['owners']):
                            continue  # forced, debt or bypassed root: no trajectory to time
                        owners = result['owners']
                        result.update(model=model, k=k, rep=rep, allowance_ms=allowance,
                                      cost_ns=[o['last_completion_ns'] / o['completed'] for o in owners if o['completed']],
                                      full=all(o['completed'] == o['quota'] for o in owners),
                                      used=max(o['last_completion_ns'] / o['available_ns'] for o in owners))
                        rows.append(result)
            print(label, 'repetition', rep, 'done:', len(rows), 'searches', flush=True)
    finally:
        for w, _ in workers.values():
            w.close()
    return rows


def summarize(rows):
    costs = sorted(c for r in rows for c in r['cost_ns'])
    by = lambda key: {str(v): max(c for r in rows if r[key] == v for c in r['cost_ns']) for v in sorted({r[key] for r in rows}, key=str)}
    return dict(maximum_ns=costs[-1], p99_ns=costs[int(.99 * (len(costs) - 1))], median_ns=costs[len(costs) // 2],
                lanes=len(costs), searches=len(rows), full_quota=sum(r['full'] for r in rows),
                max_used=max(r['used'] for r in rows), by_model=by('model'), by_k=by('k'), by_position=by('position'))


def save(name, **record):
    STUDY.mkdir(parents=True, exist_ok=True)
    out = STUDY / name
    write(out, dict(format='brass-search-v1-startup-calibration/1.0.0', engine=SOURCE_VERSION,
                    source_sha256=read(ROOT / 'BrassNative/alpha-g890/build.json')['source_sha256'], **record))
    print(json.dumps(record['summary'], indent=1), flush=True)
    print(name, sha(out), flush=True)


def measure():
    """Stage 1, the L12 method: about 64 trajectories per owner, root preparation included."""
    installed = read(ROOT / 'artifacts/patch-installation.json')
    assert installed['version'] == SOURCE_VERSION, installed['version']
    rows = sweep(MEASURE_NS, 'calibration', (16, 8), 2)
    save('calibration.json', method='singleton search, about 64 trajectories per owner, receipt to last completed '
         'trajectory / completed', previous=dict(startup_ns=7262519, source='L12 singleton calibration maximum'),
         summary=summarize(rows), searches=rows)


def full(estimate_ns):
    """Stage 2: the same cost at the quota a turn actually runs (timed searches sized by stage 1's estimate).

    Root preparation is then spread over the real quota instead of 64 trajectories; a lane that ran out of time
    counts its actual cost (available time / completed). The recommended estimate is the maximum lane cost, rounded
    up to a whole microsecond."""
    rows = sweep(estimate_ns, 'full-scale', (16, 8), 1)
    summary = summarize(rows)
    summary['recommended_ns'] = int(-(-summary['maximum_ns'] * MARGIN // 1000) * 1000)
    save('full-scale.json', method=f'timed searches at {estimate_ns} ns, receipt to last completed trajectory / '
         f'completed; recommended = maximum x {MARGIN}', provisional_ns=estimate_ns,
         stage1_sha256=sha(STUDY / 'calibration.json'), summary=summary, searches=rows)


def verify(estimate_ns):
    """Every searching position, every package, K 16 and K 8 at 4 s, and K 16 at 12 s: each must complete its quota."""
    rows = sweep(estimate_ns, 'verification', (16, 8), 1)
    rows += sweep(estimate_ns, 'verification', (16,), 1, positions=(None, 0, 8, 17), allowance=12000)
    summary = summarize(rows)
    save(f'verification-{estimate_ns}.json', estimate_ns=estimate_ns, summary=summary, searches=rows)
    print('full quota', summary['full_quota'], 'of', summary['searches'], '; max share of available time used',
          round(summary['max_used'], 3), flush=True)


if __name__ == '__main__':
    {'measure': measure, 'full': lambda: full(int(sys.argv[2])), 'verify': lambda: verify(int(sys.argv[2]))}[sys.argv[1]]()
