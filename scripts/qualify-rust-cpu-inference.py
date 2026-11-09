#!/usr/bin/env python3
"""Complete lab qualification of in-worker CPU inference. The run is never changed.

Each stage writes its own evidence file and the qualification carries on after
a failure, so one run shows every problem. A retry on the same code, build,
checkpoint and settings skips stages that already passed; --fresh repeats them.
A crashed iteration's unfinished work is read around, left in place and checked
afterwards, so the normal recovery still finds it.

Stages:

  environment  host, CPU layout, memory, Mono, package versions, pinned checkpoint
  regressions  the CPU test suite against the prepared Linux workers
  bundles      every model the league can seat: parity, utility and search heads
  parity       collector outputs against PyTorch on real positions, per model kind
  teacher      real tables at the production budgets, blended and unblended
  ordinary     real frozen jobs at full width, matched against the GPU path
  rating       a real rated table on CPU and on the service, with the anchor
  startup      per-lane Mono bootstrap, startup phases and peak memory

Prepare the workers first:

  python scripts/prepare-rust-cpu-inference.py --expected-generation 1290
  python scripts/qualify-rust-cpu-inference.py --expected-generation 1290
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
CONTRACT = 'brass-lab-cpu-qualification/1.0.0'
TEST_PATHS = ('src', 'scripts', 'tests/R2', 'tests/Recovery', 'tests/FeatureEngineering', 'tests/RustBackendLab')
LOGIT_TOLERANCE = 2e-4
PROBABILITY_RTOL, PROBABILITY_ATOL = 3e-3, 3e-5
# Independent float32 rounding can flip a sampled action on rare positions.
MAX_DIVERGED_GAMES = .15
MAX_DEADLINE_FRACTION = .02
MAX_REJECTED_LABELS = .05
BLENDS = (0., .25, 1.)
EXPECTED_SKIPS = 2
MINIMUM_TESTS = 60
# What prepare could not run: it has no bootstrap, so these need one here.
REMAINING_MODULES = ('test_cpu_inference.RealLane', 'test_cpu_qualification')
REMAINING_MINIMUM_TESTS = 22
PREPARE_MAXIMUM_SKIPS = 3
PREPARE_MINIMUM_TESTS = 40
OK_LINE = re.compile(r'^OK(?: \(skipped=(\d+)\))?$', re.M)
RAN_LINE = re.compile(r'^Ran (\d+) tests?', re.M)
# Quick host checks and the bundle receipts other stages need always run.
REUSABLE = ('regressions', 'parity', 'teacher', 'ordinary', 'rating')


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def spread(values):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    return dict(samples=len(values), min=values[0], p50=values[len(values)//2], p95=values[min(len(values)-1, int(len(values)*.95))],
                max=values[-1], mean=sum(values)/len(values))


def expect(checks, name, passed, detail=None):
    """Record one named check; a stage passes only if all of its checks pass."""
    checks[name] = dict(passed=bool(passed), detail=detail)
    return bool(passed)


def finish(checks, **fields):
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks, **fields)


# ------------------------------------------------------------------ pure checks

def search_summary(tables, budget):
    """Quotas actually searched, per profile, from real table metrics."""
    from collections import Counter
    budgets = dict(full=(budget['full'], budget['full_k']), fast=(budget['fast'], budget['fast_k']))
    result = {}
    for name, (quota, k) in budgets.items():
        rows = [s for table in tables for s in table['metrics']['searches'] if s['frozen_n'] > 0 and s['profile'] == name]
        result[name] = dict(searches=len(rows), expected_n=quota, requested_k=k,
            frozen_n=dict(Counter(str(s['frozen_n']) for s in rows)), effective_k=dict(Counter(str(s['effective_k']) for s in rows)),
            completed_all=sum(s['completed'] == s['frozen_n'] for s in rows), deadline_hits=sum(bool(s['deadline_hit']) for s in rows),
            elapsed_seconds=spread(s['elapsed_ns']/1e9 for s in rows))
    return result


def check_searches(checks, summary):
    for name, row in summary.items():
        expect(checks, name+'_searches_ran', row['searches'] > 0, row['searches'])
        expect(checks, name+'_uses_the_budget', set(row['frozen_n']) == {str(row['expected_n'])}, row['frozen_n'])
        expect(checks, name+'_stays_within_k', all(int(k) <= row['requested_k'] for k in row['effective_k']), row['effective_k'])
        expect(checks, name+'_reaches_k', str(row['requested_k']) in row['effective_k'], row['effective_k'])
        fraction = row['deadline_hits']/max(1, row['searches'])
        expect(checks, name+'_rarely_hits_the_safety_cap', fraction <= MAX_DEADLINE_FRACTION, fraction)


def family(game):
    return [game, *game.get('branches', [])]


def match_games(gpu, cpu):
    """CPU against GPU on the same frozen jobs.

    Identical until the first diverged sample. Probabilities are compared only
    up to that point; a game that diverges is counted, never ignored.
    """
    games = diverged = rows = forced = 0
    worst_absolute = worst_relative = 0.
    failures = []
    for left, right in zip(gpu, cpu, strict=True):
        for a, b in zip(family(left), family(right)):
            games += 1
            same = True
            for x, y in zip(a['records'], b['records']):
                if x['step'] != y['step'] or x['seat'] != y['seat'] or [c['id'] for c in x['choices']] != [c['id'] for c in y['choices']]:
                    same = False
                    break
                p, q = x['behavior']['probability'], y['behavior']['probability']
                if len(x['choices']) == 1:
                    forced += 1
                else:
                    rows += 1
                    absolute = abs(p-q)
                    relative = absolute/max(abs(p), abs(q), 1e-30)
                    worst_absolute, worst_relative = max(worst_absolute, absolute), max(worst_relative, relative)
                    if absolute > PROBABILITY_ATOL and relative > PROBABILITY_RTOL:
                        failures.append(dict(game=a['game_id'], step=x['step'], gpu=p, cpu=q))
                if x['chosen'] != y['chosen']:
                    same = False
                    break
            if not same or len(a['records']) != len(b['records']):
                diverged += 1
    return dict(games=games, diverged=diverged, rows=rows, forced_rows=forced, max_absolute=worst_absolute,
                max_relative=worst_relative, tolerance_failures=failures[:32], tolerance_failure_count=len(failures))


def blend_error(model, row, weight):
    """Largest difference between a search root value and PyTorch's blend."""
    import numpy as np
    import torch
    from brass_ai.learning.search_value import correction
    states = torch.as_tensor(row['state'], dtype=torch.float32)[None]
    mapping = torch.as_tensor(row['mapping'])[None]
    remaining = torch.as_tensor(row['remaining'], dtype=torch.float32)[None]
    with torch.inference_mode():
        encoded = model.encode_state(states)
        utility = model.heads_from_encoded(encoded, mapping, remaining)[3].float()
        expected = utility
        if weight:
            delta = correction(model, encoded, mapping, remaining).float()
            expected = utility+torch.tensor(weight, dtype=torch.float32)*delta
    values = np.asarray(row['v'], dtype=np.float64)
    reference = expected[0].double().numpy()
    return float(np.max(np.abs(values-reference)/np.maximum(np.abs(reference), 1.)))


def pending_inventory(run, generation):
    """Size and modification time of every file a crashed iteration left behind."""
    run = Path(run)
    roots = [run/f'iteration-{generation:04d}', *(run/'teacher-labels').glob(f'generation-{generation:06d}-*'),
             *(run/'search-values').glob(f'{generation:06d}-*')]
    files = {}
    for root in roots:
        if not root.exists():
            continue
        for path in ([root] if root.is_file() else sorted(root.rglob('*'))):
            if path.is_file():
                info = path.stat()
                files[path.relative_to(run).as_posix()] = (info.st_size, info.st_mtime_ns)
    return files


def only_unfinished_iteration(run, generation):
    """An unfinished checkpoint is a different situation: recover that first."""
    from brass_ai.learning.contracts import require
    from brass_ai.rust_backend import lab_snapshot as snapshots
    pending = snapshots.pending_paths(Path(run), generation)
    require(all(Path(p).name == f'iteration-{generation:04d}' for p in pending),
            'An unfinished checkpoint exists; recover it before qualifying: '+', '.join(pending))
    return pending


def capture_run(run, device):
    """Capture a run whose current iteration crashed before it committed."""
    from brass_ai.rust_backend import lab_snapshot as snapshots
    generation = json.loads((Path(run)/'state.json').read_text(encoding='utf-8-sig'))['iteration']
    paths = only_unfinished_iteration(run, generation)
    with patch.object(snapshots, 'pending_paths', return_value=[]):
        snapshot, model, raw = snapshots.capture(ROOT, run, devices=[device])
    snapshot['pending_work'] = paths
    return snapshot, model, raw, dict(paths=paths, generation=generation, files=pending_inventory(run, generation))


def confirm_unchanged(snapshot, pending, run):
    from brass_ai.learning.contracts import require
    from brass_ai.rust_backend import lab_snapshot as snapshots
    with patch.object(snapshots, 'pending_paths', return_value=[]):
        snapshots.verify_unchanged(snapshot)
    only_unfinished_iteration(run, pending['generation'])
    require(pending_inventory(run, pending['generation']) == pending['files'], 'The unfinished iteration changed during qualification')


def passed_stages(base, current, identity):
    """Stages that passed on this exact code, build, checkpoint and settings.

    The newest attempt at each stage decides, so a stage that last failed runs again.
    """
    seen, passed = set(), {}
    for directory in sorted(Path(base).glob('*-cpu-qualification-*'), reverse=True):
        path = directory/'qualification.json'
        if directory == Path(current) or not path.is_file():
            continue
        try:
            earlier = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if earlier.get('identity') != identity:
            continue
        for name, stage in earlier.get('stages', {}).items():
            if name in seen or 'skipped' in stage:
                continue
            seen.add(name)
            evidence = directory/(name+'.json')
            if name in REUSABLE and stage.get('passed') is True and evidence.is_file():
                result = json.loads(evidence.read_text(encoding='utf-8'))
                if result.get('passed') is True:
                    passed[name] = dict(directory=directory.name, result=result)
    return passed


def prepared_suite(value):
    """The CPU suite's receipt from prepare, if it passed on this exact build."""
    from brass_ai.rust_backend.lab_contract import pin
    receipt = value.get('regressions')
    if not receipt or not Path(receipt['path']).is_file() or pin(receipt['path']) != receipt:
        return None
    text = Path(receipt['path']).read_text(errors='replace')
    ok, ran = OK_LINE.search(text), RAN_LINE.search(text)
    # A native-worker skip would count every test in its class, so this proves they ran.
    if ok is None or ran is None or int(ok.group(1) or 0) > PREPARE_MAXIMUM_SKIPS or int(ran.group(1)) < PREPARE_MINIMUM_TESTS:
        return None
    return receipt


# ----------------------------------------------------------------------- stages

class Context:
    pass


def environment(c):
    import onnxruntime
    import torch
    checks = {}
    expect(checks, 'linux', platform.system() == 'Linux', platform.platform())
    expect(checks, 'cuda', torch.cuda.is_available(), c.args.device)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count()))
    sockets = {}
    for cpu in cpus:
        try:
            socket = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id').read_text().strip()
        except OSError:
            socket = '?'
        sockets[socket] = sockets.get(socket, 0)+1
    expect(checks, 'enough_cpus', len(cpus) >= c.args.workers, dict(usable=len(cpus), lanes=c.args.workers, sockets=sockets))
    memory = {}
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        memory = {k.strip(): v.strip() for k, v in (line.split(':', 1) for line in meminfo.read_text().splitlines())
                  if k.strip() in ('MemTotal', 'MemAvailable')}
    version = subprocess.run([c.args.mono, '--version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    expect(checks, 'mono', version.returncode == 0, version.stdout.splitlines()[0] if version.stdout else '')
    expect(checks, 'onnxruntime', onnxruntime.__version__ == '1.22.1', onnxruntime.__version__)
    expect(checks, 'generation', c.snapshot['generation'] == c.args.expected_generation, c.snapshot['generation'])
    limit = None
    try:
        import resource
        limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except ImportError:
        pass
    return finish(checks, python=sys.version, torch=torch.__version__, memory=memory, open_file_limit=limit,
                  checkpoint=c.snapshot['checkpoint'], cpu_build=c.cpu_value['receipt'], execution=c.execution)


def regressions(c):
    value = c.cpu_value
    evidence = Path(value['tests']['path']).parent
    native = evidence/'teacher/checkout/native'
    checks = {}
    if not expect(checks, 'build_evidence_retained', native.is_dir(), str(native)):
        return finish(checks)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in TEST_PATHS),
        BRASS_CPU_TEACHER=value['teacher']['path'], BRASS_CPU_COLLECTOR=value['collector']['path'], BRASS_CPU_RUNTIME=value['runtime']['path'],
        BRASS_CPU_FIXTURES=str(c.fixtures), BRASS_CPU_CORE=value['teacher_core'], BRASS_CPU_NATIVE=str(native),
        BRASS_CPU_BOOTSTRAP=c.gpu_build['bootstrap']['binary']['path'], BRASS_CPU_MONO=c.args.mono, CUDA_VISIBLE_DEVICES='')
    # Prepare already ran the main suite on this exact build; only finish what it could not.
    prepared = prepared_suite(value)
    modules, skips, minimum = ((list(REMAINING_MODULES), 0, REMAINING_MINIMUM_TESTS) if prepared else
                               (['test_cpu_inference', 'test_cpu_qualification'], EXPECTED_SKIPS, MINIMUM_TESTS))
    log = c.out/'regressions.log'
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', *modules, '-v'], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=2400)
    text = log.read_text(errors='replace')
    tail = text.splitlines()[-6:]
    expect(checks, 'suite', result.returncode == 0, tail)
    ok = OK_LINE.search(text)
    # With the full suite, only the timed-budget test and the local-host rating game may skip.
    expect(checks, 'only_expected_skips', ok is not None and int(ok.group(1) or 0) == skips, ok.group(0) if ok else None)
    ran = RAN_LINE.search(text)
    expect(checks, 'native_suites_ran', ran is not None and int(ran.group(1)) >= minimum, ran.group(0) if ran else None)
    return finish(checks, log_tail=tail, modules=modules, reused_prepare_suite=prepared)


def league_entries(c):
    """Every checkpoint the league can seat at this generation."""
    state = c.snapshot['state']
    entries = {state['history'][-1]['model_id']: state['history'][-1]}
    for item in [*state.get('quality_pool', {}).get('members', ()), state.get('evaluation_anchor'), state.get('champion'),
                 *state.get('population', ())]:
        if item is not None:
            entries.setdefault(item['model_id'], item)
    return entries


def bundles(c):
    from brass_ai.bank import sha
    from brass_ai.learning.contracts import project_path, require
    from brass_ai.rust_backend.lab_cpu_models import ensure
    checks = {}
    entries = league_entries(c)
    began = time.perf_counter()
    for identity, entry in entries.items():
        path = project_path(ROOT, entry['path'])
        require(sha(path) == entry['sha256'], 'League checkpoint changed: '+str(path))
        if identity not in c.service.models:
            c.service.register_checkpoint(path)
    registered = time.perf_counter()-began
    began = time.perf_counter()
    receipts, exported = ensure(ROOT, c.service.models, list(entries))
    seconds = time.perf_counter()-began
    models = {}
    worst = 0.
    for identity, receipt in receipts.items():
        manifest = json.loads(Path(receipt['manifest']).read_text(encoding='utf-8'))
        error = max(manifest['parity']['errors'].values())
        worst = max(worst, error)
        models[identity] = dict(utility=manifest.get('utility', True), search_delta=manifest['search_delta'], parity_error=error,
                                reward_id=manifest['reward_id'], shapes=len(manifest['parity']['shapes']))
    c.bundles, c.league = receipts, models
    expect(checks, 'every_league_model_has_a_bundle', set(models) == set(entries), len(models))
    expect(checks, 'an_anchor_without_utility_is_covered', any(not m['utility'] for m in models.values()),
           sum(not m['utility'] for m in models.values()))
    expect(checks, 'learner_has_the_search_head', models[c.snapshot['model_id']]['search_delta'], c.snapshot['model_id'])
    expect(checks, 'parity_within_bound', worst < LOGIT_TOLERANCE, worst)
    expect(checks, 'six_shapes_each', all(m['shapes'] == 6 for m in models.values()), None)
    return finish(checks, registered_seconds=registered, export_seconds=seconds, newly_exported=exported, models=models)


def representatives(c):
    """One model of each kind: search head, utility only, and no utility."""
    kinds = dict(search_head=None, utility=None, anchor=None)
    for identity, model in c.league.items():
        kind = 'search_head' if model['search_delta'] else ('utility' if model['utility'] else 'anchor')
        if kinds[kind] is None or identity == c.snapshot['model_id']:
            kinds[kind] = identity
    return {k: v for k, v in kinds.items() if v is not None}


def parity(c):
    import numpy as np
    import torch
    from brass_ai.learning.batching import collate
    from brass_ai.rust_backend.lab_teacher_inference import predict
    from brass_ai.rust_backend.lab_transport import Collector, DecisionEncoder, feature_context
    from brass_ai.recovery import collector_runtime as R
    checks = {}
    fixtures = json.loads(Path(c.fixtures).read_text(encoding='utf-8'))
    roots = [r for r in fixtures['roots'] if r['view']['context']['phase'] == 'action' and len(r['expected']['choices']) > 1]
    wide = max(roots, key=lambda r: len(r['expected']['choices']))
    chosen = [roots[0], roots[len(roots)//2], roots[-1], wide]
    runtime = dict(path=c.cpu_value['runtime']['path'], sha256=c.cpu_value['runtime']['sha256'])
    results = {}
    kinds = representatives(c)
    expect(checks, 'covers_the_learner_and_an_anchor', {'search_head', 'anchor'} <= set(kinds), sorted(kinds))
    with Collector(c.cpu_value['collector']['path']) as lane:
        expect(checks, 'collector_contract', lane.hello.get('cpu_inference') == R.CPU_CONTRACT, lane.hello.get('cpu_inference'))
        lane.call('configure_cpu', contract=R.CPU_CONTRACT, runtime=runtime, model_cache=8)
        style, intent = [0.]*4, [0.]*8
        for kind, identity in kinds.items():
            model = c.service.models[identity]
            worst = positions = 0
            for root in chosen:
                lane.begin(root['id'], dict(state=root['expected']['state'], rail_keys=list(range(64))), 5000)
                lane.call('load_model', model_id=identity, bundle=c.bundles[identity])
                d = lane.current['decision']
                encoded = DecisionEncoder(lane, identity).encode(d['view'], d['choices'], style, intent)
                page = lane.call('infer', token=lane.token, revision=d['view']['context']['revision'],
                                 context=feature_context(identity, style, intent))
                n = len(d['choices'])
                values = np.frombuffer(page['_arrays'], '<f4')
                with torch.inference_mode():
                    if c.league[identity]['utility']:
                        cap = model.config.utility['insolvency']['max_cost']
                        arrays = predict(model, [encoded], 'cpu', 8192, torch.full((1, 4), cap, dtype=torch.float32))
                        logits, heads = arrays[-1][0, :n], [arrays[i][0] for i in range(1, 5)]
                    else:
                        out = model(*collate([encoded], 'cpu'))
                        logits, heads = out[0][0].numpy(), [h[0].numpy() for h in out[1:]]+[np.zeros(4, np.float32)]
                errors = [np.abs(values[:n]-logits).max()]
                for index, head in enumerate(heads):
                    errors.append(np.abs(values[n+4*index:n+4*index+4]-head).max())
                worst = max(worst, float(max(errors)))
                positions += 1
                lane.release()
            results[kind] = dict(model_id=identity, positions=positions, worst_absolute_error=worst, candidates=len(wide['expected']['choices']))
            expect(checks, kind+'_matches_pytorch', worst <= 2*LOGIT_TOLERANCE, worst)
    return finish(checks, kinds=results)


def teacher_jobs(c, weight, count, offset):
    """Production-shaped tables. Stage B's schedule adds .05 before the first use."""
    from brass_ai.recovery.teacher_phase import plan_jobs
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    state = copy.deepcopy(c.snapshot['state'])
    # Table identities carry the seed, not the blend, so each blend needs its own seeds.
    state['teacher_seed_cursor'] = state.get('teacher_seed_cursor', 0)+offset
    if weight:
        schedule = 'schedule' in c.config['teacher_phase']['search_value']
        state['search_value'] = dict(stage=dict(stage='B', w=round(weight-(.05 if schedule else 0.), 2)))
    jobs, _ = plan_jobs(c.config, state, model_id=c.snapshot['model_id'], export_hash=c.snapshot['checkpoint']['sha256'], epoch=1, ready=True,
                        native_source=source_fingerprint())
    jobs = [j for j in jobs if j['split'] == 'train'][:count]
    for job in jobs:
        job['readiness'] = c.readiness
    return jobs


def teacher(c):
    from brass_ai.recovery import collector_runtime as R
    from brass_ai.recovery.teacher_phase import readiness
    from brass_ai.learning.contracts import digest, require
    from brass_ai.rust_backend.lab_cpu_runtime import core_allocation, teacher_plan
    from brass_ai.rust_backend.lab_shadow import teacher_build
    from brass_ai.rust_backend.lab_teacher import TeacherCollector, count_calibration
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    checks = {}
    state = c.snapshot['state']
    build = teacher_build(state)
    reward = digest(c.model.config.utility)
    receipt, ready = readiness(state.get('teacher_phase', {}).get('readiness'), c.raw['training'].get('utility_calibration'), reward)
    if not expect(checks, 'teacher_readiness', ready, receipt):
        return finish(checks)
    c.readiness = receipt
    jobs = []
    for number, weight in enumerate(BLENDS):
        jobs += teacher_jobs(c, weight, 8 if weight == 0 else 4, 100000*number)
    expect(checks, 'blend_weights', sorted({j['search_value']['w'] for j in jobs}) == sorted(BLENDS), sorted({j['search_value']['w'] for j in jobs}))
    bootstrap = [build['bootstrap']['binary']['path']]
    if os.name != 'nt':
        bootstrap.insert(0, c.args.mono)
    lanes = min(len(jobs), c.block['workers'])
    cores = core_allocation(lanes) if c.block['pinning'] == 'one-lane-per-cpu' else []
    positions_before = c.service.metrics['positions']
    with R.scope(c.cpu_settings):
        plan = teacher_plan(ROOT, build, c.service, jobs, c.block)
        calibration = count_calibration(build['core_source'], jobs[0])
        require(calibration['source'] == source_fingerprint(), 'Calibration source differs')
        collector = TeacherCollector(plan['binary'], build['core_source'], bootstrap, c.service, calibration, lanes=lanes, model=c.model,
                                     history=state['history'], inference=plan['inference'], cores=cores)
        began = time.perf_counter()
        try:
            tables = collector.collect(jobs)
        except BaseException:
            write(c.out/'teacher-failures.json', dict(failures=collector.failures))
            raise
        wall = time.perf_counter()-began
    budget = c.block['teacher_budget']
    summary = search_summary(tables, budget)
    check_searches(checks, summary)
    accepted = rejected = rows = 0
    labels_seen = 0
    for table, job in zip(tables, jobs, strict=True):
        good, bad = collector.verifier.take(table, job, c.model)
        accepted += len(good)
        rejected += sum(bad.values())
        rows += len(table['rows'])
        labels_seen += table['metrics']['labels']
    expect(checks, 'labels_produced', accepted > 0, accepted)
    expect(checks, 'labels_accepted', rejected <= MAX_REJECTED_LABELS*max(1, rows), dict(accepted=accepted, rejected=rejected, rows=rows))
    errors = []
    unblended = {}
    for table, job in zip(tables, jobs, strict=True):
        weight = job['search_value']['w']
        for row in table['value_rows'][:4]:
            errors.append(blend_error(c.model, row, weight))
        if weight:
            unblended.setdefault(weight, []).append(max(abs(a-b) for a, b in zip(
                table['value_rows'][0]['v'], blend_values(c.model, table['value_rows'][0]))))
    expect(checks, 'value_rows_match_pytorch_blend', errors and max(errors) <= 2*LOGIT_TOLERANCE, max(errors) if errors else None)
    expect(checks, 'every_table_used_cpu', all(t['metrics']['cpu_inference']['backend'] == 'cpu-ort' for t in tables), None)
    expect(checks, 'no_gpu_positions', c.service.metrics['positions'] == positions_before, c.service.metrics['positions']-positions_before)
    return finish(checks, tables=len(tables), lanes=lanes, wall_seconds=wall, searches=summary, value_error=spread(errors),
                  blend_effect={str(k): spread(v) for k, v in unblended.items()},
                  table_seconds=spread(t['metrics']['elapsed_ns']/1e9 for t in tables),
                  mono_bootstrap_seconds=spread(t['metrics']['mono_bootstrap_seconds'] for t in tables),
                  forward=dict(positions=sum(t['metrics']['cpu_inference']['positions'] for t in tables),
                               forward_ns=sum(t['metrics']['cpu_inference']['forward_ns'] for t in tables)))


def blend_values(model, row):
    """Root value with no blend, for showing that blending changed something."""
    import torch
    states = torch.as_tensor(row['state'], dtype=torch.float32)[None]
    mapping = torch.as_tensor(row['mapping'])[None]
    remaining = torch.as_tensor(row['remaining'], dtype=torch.float32)[None]
    with torch.inference_mode():
        encoded = model.encode_state(states)
        return model.heads_from_encoded(encoded, mapping, remaining)[3].float()[0].double().tolist()


def planned_jobs(c):
    from brass_ai.learning.contracts import require, digest
    from brass_ai.rust_backend.lab_collection import frozen_id, route
    from brass_ai.rust_backend.lab_jobs import plan
    planned = plan(ROOT, c.run, c.snapshot, c.model, c.raw, c.service, c.out/'payload/ordinary-plan')
    jobs = [j for j in planned['jobs'] if j['provenance']['legacy_mode'] != 'searched']
    require(jobs and all(route(j)[0] == 'rust' for j in jobs), 'Real ordinary jobs need the reference route')
    for job in jobs:
        job.setdefault('stable_game_id', digest(dict(job=frozen_id(job), seed=job['seed'])))
    return jobs


def native_arm(c, jobs, name, *, cpu):
    from brass_ai.rust_backend import lab_admission as admission
    from brass_ai.rust_backend.lab_pool import NativePool
    output = c.out/'payload/ordinary'/name
    frozen = copy.deepcopy(jobs)
    for job in frozen:
        job['path'] = str(output/'plan'/(f"job-{job['index']}.jsonl.gz"))
    (output/'plan').mkdir(parents=True)
    build = c.cpu_build if cpu else c.gpu_build
    execution = c.cpu_execution if cpu else c.execution
    signature = frozen[0]['provenance']['spec']
    with NativePool(ROOT, c.service, execution, build, c.config, signature, mono=c.args.mono, scope='diagnostic', cpu=cpu) as pool:
        began = time.perf_counter()
        records = pool.collect(frozen)
        elapsed = time.perf_counter()-began
        report = pool.last_collection_report
    games = [admission.read(r['path'], r['native_admission'], allow_diagnostic=True) for r in records]
    return games, dict(collection_seconds=elapsed, records=len(records), report=report)


def ordinary(c):
    from brass_ai.recovery import collector_runtime as R
    from brass_ai.rust_backend.lab_cpu_runtime import ordinary_plan, raise_open_file_limit
    from rust_lab_b06_metrics import Meter
    checks = {}
    raise_open_file_limit()
    jobs = planned_jobs(c)
    anchors = {i for i, m in c.league.items() if not m['utility']}
    seats_anchor = lambda j: any(p['kind'] == 'neural' and p['model_id'] in anchors for p in j['policies'])
    # Tables that seat an anchor come first: that is the shape that stopped the first run.
    chosen = sorted(jobs, key=lambda j: not seats_anchor(j))[:c.args.ordinary_games]
    matched = chosen[:c.args.matched_games]
    expect(checks, 'real_jobs', len(chosen) >= min(c.args.ordinary_games, len(jobs)), dict(chosen=len(chosen), planned=len(jobs)))
    expect(checks, 'jobs_seat_an_anchor', any(seats_anchor(j) for j in chosen) or not any(seats_anchor(j) for j in jobs),
           sum(seats_anchor(j) for j in chosen))
    print('Qualification: GPU arm', flush=True)
    gpu_games, gpu_report = native_arm(c, matched, 'gpu', cpu=None)
    positions_before = c.service.metrics['positions']
    with R.scope(c.cpu_settings):
        c.cpu_build, settings = ordinary_plan(ROOT, c.gpu_build, c.service, chosen, c.block)
        print('Qualification: CPU arm at full width', flush=True)
        with Meter('cpu') as meter:
            with meter.phase('collection'):
                games, report = native_arm(c, chosen, 'cpu', cpu=settings)
            usage = meter.report()
    expect(checks, 'cpu_arm_finished_every_job', len(games) == len(chosen), len(games))
    expect(checks, 'no_gpu_positions', c.service.metrics['positions'] == positions_before, c.service.metrics['positions']-positions_before)
    by_id = {g['game_id']: g for g in games}
    comparison = match_games(gpu_games, [by_id[g['game_id']] for g in gpu_games])
    expect(checks, 'probabilities_within_bound', comparison['tolerance_failure_count'] == 0, comparison['tolerance_failure_count'])
    fraction = comparison['diverged']/max(1, comparison['games'])
    expect(checks, 'matched_games_stay_together', fraction <= MAX_DIVERGED_GAMES, comparison['diverged'])
    c.ordinary_report, c.ordinary_usage = report, usage
    return finish(checks, jobs=len(chosen), matched=len(matched), comparison=comparison, gpu=gpu_report, cpu=report, usage=usage)


def rating(c):
    from brass_ai.recovery import collector_runtime as R
    from brass_ai.recovery.execution import ActorPool, neural_spec
    from brass_ai.recovery.replay import read_shards
    from brass_ai.learning.contracts import digest
    from brass_ai.rust_backend.lab_cpu_runtime import rating_settings
    from brass_ai.step5.league import seeds_for
    checks = {}
    learner = c.snapshot['model_id']
    anchor = next((i for i, m in c.league.items() if not m['utility']), learner)
    specs = [neural_spec(learner, 0., 'neutral'), neural_spec(anchor, 0., 'neutral')]
    policies = [specs[0], specs[1], specs[0], specs[1]]
    seeds = seeds_for(22517000, c.args.rating_games, 'validation')[0]
    host = c.config['host']
    execution = dict(c.cpu_execution, processes=2, hosts_per_process=1, heavy_processes=1)
    def play(name, cpu):
        directory = c.out/'payload/rating'/name
        directory.mkdir(parents=True)
        jobs = [dict(index=i, seed=seed, rotation=0, sampling_seed=99017000+i, policies=policies, path=str(directory/f'game-{i}.jsonl.gz'),
                     provenance=dict(producer='plackett-luce-rating/1.0.0', purpose='evaluation', kind='rating', seed=seed,
                                     sampling_seed=99017000+i, rotation=0, spec=digest(policies))) for i, seed in enumerate(seeds)]
        before = c.service.metrics['positions']
        with ActorPool(ROOT, host, c.service, execution, mono=c.args.mono, cpu_inference=cpu) as pool:
            records = pool.collect(jobs)
        games = read_shards([r['path'] for r in records], partition='validation', receipts={r['path']: r.get('features') for r in records})
        return sorted(games, key=lambda g: g['game_id']), c.service.metrics['positions']-before
    service_games, used = play('service', None)
    expect(checks, 'service_used_the_gpu', used > 0, used)
    with R.scope(c.cpu_settings):
        settings = rating_settings(ROOT, c.block, c.service, [learner, anchor])
        cpu_games, used = play('cpu', settings)
    expect(checks, 'cpu_never_used_the_service', used == 0, used)
    comparison = match_games(service_games, cpu_games)
    expect(checks, 'probabilities_within_bound', comparison['tolerance_failure_count'] == 0, comparison['tolerance_failure_count'])
    expect(checks, 'games_stay_together', comparison['diverged'] <= max(1, len(seeds)//4), comparison['diverged'])
    expect(checks, 'anchor_seated', anchor != learner, anchor)
    return finish(checks, games=len(seeds), comparison=comparison)


def startup(c):
    checks = {}
    report = getattr(c, 'ordinary_report', None)
    if not expect(checks, 'ordinary_stage_measured_startup', report is not None and report.get('startup_breakdown') is not None, None):
        return finish(checks)
    breakdown = report['startup_breakdown']
    expect(checks, 'every_lane_reported', breakdown['lanes_reported'] == breakdown['lanes'], (breakdown['lanes_reported'], breakdown['lanes']))
    phases = breakdown['phases']
    expect(checks, 'mono_bootstrap_measured', phases.get('mono_bootstrap_seconds', {}).get('lanes', 0) == breakdown['lanes_reported'], None)
    expect(checks, 'startup_measured', 'total_seconds' in phases, sorted(phases))
    expect(checks, 'lanes_do_not_wait_for_each_other', not breakdown['waits_for_all_lanes'], None)
    expect(checks, 'no_reply_channels_on_the_cpu_path', breakdown['reply_channel_seconds'] < .01, breakdown['reply_channel_seconds'])
    return finish(checks, first_lane_ready_seconds=breakdown['first_lane_ready_seconds'], last_lane_ready_seconds=breakdown['last_lane_ready_seconds'],
                  mono_bootstrap=phases.get('mono_bootstrap_seconds'), python_import=phases.get('python_import_seconds'), total=phases.get('total_seconds'),
                  process_start_seconds=breakdown['process_start_seconds'], usage=getattr(c, 'ordinary_usage', None),
                  note='Peak RSS is the whole qualification process tree during the CPU arm, including this parent.')


STAGES = [('environment', environment), ('regressions', regressions), ('bundles', bundles), ('parity', parity),
          ('teacher', teacher), ('ordinary', ordinary), ('rating', rating), ('startup', startup)]
# A stage that cannot run without these results is skipped, never reported as passed.
NEEDS = dict(parity=('bundles',), teacher=('bundles',), ordinary=('bundles',), rating=('bundles',), startup=('ordinary',))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mono', default='mono')
    p.add_argument('--workers', type=int, default=64)
    p.add_argument('--ordinary-games', type=int, default=128, help='Real frozen jobs on the CPU arm')
    p.add_argument('--matched-games', type=int, default=16, help='Of those, also run on the GPU arm')
    p.add_argument('--rating-games', type=int, default=2)
    p.add_argument('--skip-regressions', action='store_true', help='Only when the suite already ran on this exact build')
    p.add_argument('--fresh', action='store_true', help='Repeat stages that already passed on this exact code, build and checkpoint')
    args = p.parse_args()
    import torch
    from brass_ai.learning.contracts import project_path, require
    from brass_ai.recovery import collector_runtime as R
    from brass_ai.recovery.inference import InferenceService
    from brass_ai.recovery.retention import run_lock
    from brass_ai.rust_backend.lab_backend import check_build, validate_state
    from brass_ai.rust_backend.lab_cpu_runtime import resolve
    from brass_ai.rust_backend.lab_perf_runtime import resolve as resolve_ordinary
    from brass_ai.rust_backend.lab_replay_compat import execution as execution_source
    from brass_ai.rust_backend.lab_shadow import teacher_build
    from rust_lab_results import export
    out = ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-cpu-qualification-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    report = dict(contract=CONTRACT, passed=False, production_modified=False, expected_generation=args.expected_generation, stages={},
                  scope='Lab qualification of in-worker CPU inference; the run is read only')
    snapshot = None
    try:
        report['execution_source'] = execution_source()
        run = project_path(ROOT, args.run)
        with run_lock(run):
            snapshot, model, raw, pending = capture_run(run, args.device)
            report['pending_work'] = dict(paths=pending['paths'], files=len(pending['files']), left_in_place=True)
            require(snapshot['generation'] == args.expected_generation, 'Run is not at the expected generation')
            try:
                backend = validate_state(snapshot['state'])
                require(backend is not None and backend['mode'] == 'rust', 'CPU qualification needs a Rust backend run')
                c = Context()
                c.args, c.out, c.run, c.snapshot, c.model, c.raw = args, out, run, snapshot, model.eval(), raw
                c.config = snapshot['state']['signature']['config']
                c.gpu_build = resolve_ordinary(ROOT, check_build(backend['build']))
                c.cpu_value = resolve(ROOT, teacher_core=teacher_build(snapshot['state'])['core_source'])
                c.fixtures = Path(c.gpu_build['b01']['path']).parent/'inputs/fixtures.json'
                c.block = R.cpu_block(workers=args.workers)
                R.validate_cpu(c.block)
                c.execution = snapshot['execution']
                c.cpu_settings = dict(R.from_config(c.config), native_inference=c.block)
                with R.scope(c.cpu_settings):
                    c.cpu_execution = R.effective_execution(c.execution)
                report['budget'] = c.block['teacher_budget']
                report['identity'] = dict(contract=CONTRACT, implementation=report['execution_source']['implementation'],
                    cpu_build=c.cpu_value['receipt']['sha256'], checkpoint=snapshot['checkpoint']['sha256'], workers=args.workers,
                    budget=c.block['teacher_budget'], ordinary_games=args.ordinary_games, matched_games=args.matched_games,
                    rating_games=args.rating_games)
                earlier = {} if args.fresh else passed_stages(ROOT/'artifacts/rust-backend', out, report['identity'])
                torch.set_num_threads(2)
                with InferenceService(c.cpu_execution, args.device) as c.service:
                    require(c.service.register(model) == snapshot['model_id'], 'Learner identity changed')
                    finished = {}
                    for name, stage in STAGES:
                        if name == 'regressions' and args.skip_regressions:
                            report['stages'][name] = dict(passed=None, skipped='--skip-regressions')
                            continue
                        if name in earlier:
                            result = dict(earlier[name]['result'], reused_from=earlier[name]['directory'])
                            write(out/(name+'.json'), result)
                            finished[name] = True
                            if name == 'ordinary':
                                c.ordinary_report, c.ordinary_usage = result['cpu']['report'], result['usage']
                            report['stages'][name] = dict(passed=True, reused_from=result['reused_from'], failed_checks=[])
                            print(f"Qualification stage: {name}: already passed in {result['reused_from']}", flush=True)
                            write(out/'qualification.json', report)
                            continue
                        missing = [need for need in NEEDS.get(name, ()) if not finished.get(need)]
                        if missing:
                            report['stages'][name] = dict(passed=False, skipped='Needs '+', '.join(missing))
                            write(out/'qualification.json', report)
                            continue
                        print('Qualification stage: '+name, flush=True)
                        began = time.perf_counter()
                        try:
                            result = stage(c)
                        except Exception as error:
                            result = dict(passed=False, error=str(error), traceback=traceback.format_exc())
                        result['seconds'] = time.perf_counter()-began
                        write(out/(name+'.json'), result)
                        finished[name] = result['passed']
                        report['stages'][name] = {k: result.get(k) for k in ('passed', 'seconds', 'error') if k in result}
                        report['stages'][name]['failed_checks'] = [k for k, v in result.get('checks', {}).items() if not v['passed']]
                        print(f"  {name}: {'passed' if result['passed'] else 'FAILED'} in {result['seconds']:.1f}s", flush=True)
                        write(out/'qualification.json', report)
                    require(c.service.error is None, 'Inference service failed: '+str(c.service.error))
                report['passed'] = all(s.get('passed') is not False for s in report['stages'].values())
            finally:
                confirm_unchanged(snapshot, pending, run)
    except BaseException as error:
        report.update(passed=False, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write(out/'qualification.json', report)
        print('Git-visible results: '+str(export(out)), flush=True)
        print('Original lab evidence: '+str(out), flush=True)
        print('CPU qualification: '+('PASSED' if report['passed'] else 'FAILED'), flush=True)
    if not report['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
