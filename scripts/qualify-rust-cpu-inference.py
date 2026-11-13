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
  teacher      real tables at the production budgets, blended and unblended; from g1491 under that recipe
               (long and short labels, visit-count exploration), with a profile of label verification
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
TABLES_PER_BLEND = {0.: 8, .25: 4, 1.: 4}
TABLES_PER_BLEND_REVISED = {0.: 36, .25: 6, 1.: 6}
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
# Reviewed source changes that cannot affect a stage that already passed.
REVIEWED_CHANGES = {'54350dbd05bb9abd992bc5a818f77d488cb916d2a892f3408e0345cc181a906a': dict(
    previous='38c3b310e6ee8a9798f4b593922f48074bfcf5e52a2eacd6a2b1c31fedaefccd',  # Only how training hands the pool its recipe and worker settings.
    stages=REUSABLE)}
# Starting a process should not wait for it; the old bug measured about 0.7 s each.
MAX_PROCESS_START_SECONDS = .2


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

def search_summary(tables, budget, *, jobs=None):
    """Search quotas and scales observed for each profile."""
    from collections import Counter
    from brass_ai.recovery.collector_runtime import teacher_c_scale, teacher_job_budget
    from brass_ai.recovery.teacher_cotton import enabled
    groups = [('full', 'full', tables, budget), ('fast', 'fast', tables, budget)]
    if 'cotton_c_scale' in budget:
        by_group = dict(standard=[], cotton=[])
        for table, job in zip(tables, jobs, strict=True):
            by_group['cotton' if enabled(job) else 'standard'].append((table, job))
        groups = [('fast', 'fast', tables, budget)]
        for group, pairs in by_group.items():
            effective = teacher_job_budget(budget, pairs[0][1]) if pairs else budget
            groups.append((f'full_{group}', 'full', [table for table, _ in pairs], effective))
    result = {}
    for name, profile, selected, effective in groups:
        rows = [s for table in selected for s in table['metrics']['searches'] if s['frozen_n'] > 0 and s['profile'] == profile]
        result[name] = dict(searches=len(rows), expected_n=budget[profile], requested_k=budget[profile+'_k'],
            frozen_n=dict(Counter(str(s['frozen_n']) for s in rows)), effective_k=dict(Counter(str(s['effective_k']) for s in rows)),
            completed_all=sum(s['completed'] == s['frozen_n'] for s in rows), deadline_hits=sum(bool(s['deadline_hit']) for s in rows),
            elapsed_seconds=spread(s['elapsed_ns']/1e9 for s in rows))
        if 'fast_c_scale' in budget:
            result[name].update(expected_c_scale=teacher_c_scale(effective, fast=profile=='fast'),
                c_scales=dict(Counter(str(s.get('c_scale')) for s in rows)))
    return result


def check_searches(checks, summary):
    for name, row in summary.items():
        expect(checks, name+'_searches_ran', row['searches'] > 0, row['searches'])
        expect(checks, name+'_uses_the_budget', set(row['frozen_n']) == {str(row['expected_n'])}, row['frozen_n'])
        expect(checks, name+'_stays_within_k', all(int(k) <= row['requested_k'] for k in row['effective_k']), row['effective_k'])
        expect(checks, name+'_reaches_k', str(row['requested_k']) in row['effective_k'], row['effective_k'])
        if 'expected_c_scale' in row:
            expect(checks, name+'_uses_c_scale', set(row['c_scales']) == {str(row['expected_c_scale'])}, row['c_scales'])
        fraction = row['deadline_hits']/max(1, row['searches'])
        expect(checks, name+'_rarely_hits_the_safety_cap', fraction <= MAX_DEADLINE_FRACTION, fraction)


def label_summary(tables):
    """Labels by search profile and exploration, plus what the workers say they sampled."""
    rows = [row['teacher'] for table in tables for row in table['rows']]
    summary = {}
    for name in ('full', 'fast'):
        mine = [label for label in rows if label.get('profile', 'full') == name]
        summary[name] = dict(labels=len(mine), explored=sum(label.get('explored') is True for label in mine))
    receipts = [table['metrics']['cpu_inference'].get('revision') for table in tables]
    if receipts and all(receipts):
        summary['worker'] = {key: {profile: sum(r[key][profile] for r in receipts) for profile in ('long', 'short')}
                             for key in ('decisions', 'sampled', 'changed')}
        summary['worker']['short_labels'] = sum(r['short_labels'] for r in receipts)
    return summary


def check_revision(checks, jobs, tables, summary):
    """Check the revision's requested labels, exploration, and worker receipt."""
    expect(checks, 'worker_echoed_the_revision', all(t['metrics']['cpu_inference'].get('revision', {}).get('job') == j['teacher_revision']
           for t, j in zip(tables, jobs, strict=True)), None)
    expect(checks, 'long_labels_produced', summary['full']['labels'] > 0, summary['full'])
    expect(checks, 'every_long_label_is_explored', summary['full']['explored'] == summary['full']['labels'], summary['full'])
    if any(j['teacher_revision']['short_labels'] for j in jobs):
        expect(checks, 'short_labels_produced', summary['fast']['labels'] > 0, summary['fast'])
        # Short searches sample only in each seat's opening.
        expect(checks, 'short_labels_explore_only_openings', 0 < summary['fast']['explored'] < summary['fast']['labels'], summary['fast'])
    else:
        expect(checks, 'long_only_policy_labels', summary['fast']['labels'] == 0, summary['fast'])
    worker = summary.get('worker')
    expect(checks, 'worker_counts_match_the_labels', worker is not None and worker['short_labels'] == summary['fast']['labels'] and
           worker['sampled']['long'] >= summary['full']['labels'] and worker['decisions']['short'] > worker['sampled']['short'], worker)


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


def scaled_error(actual, expected):
    """Largest difference as a fraction of the export's relative-or-absolute bound."""
    import numpy as np
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(actual-expected)/(LOGIT_TOLERANCE+LOGIT_TOLERANCE*np.abs(expected))))


def restore_admission(service, execution):
    """A failed collection aborts the shared CPU gate; later stages need a working one."""
    from brass_ai.recovery.resources import CPUAdmission
    if service.cpu.aborted.value:
        service.cpu = CPUAdmission(service.mp, execution.get('cpu_budget', 64), execution.get('cpu_reserve', 4))


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


def reusable_stages(base, current, identity):
    """Passed stages on this identity, plus reviewed ones from just before a source change."""
    passed = passed_stages(base, current, identity)
    review = REVIEWED_CHANGES.get(identity['implementation'])
    if review is None:
        return passed
    older = passed_stages(base, current, dict(identity, implementation=review['previous']))
    return {**{name: stage for name, stage in older.items() if name in review['stages']}, **passed}


def by_setup(games):
    """Rated games have random IDs, so pair the two arms by their setup."""
    return sorted(games, key=lambda game: game['setup_group'])


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


def cpu_view():
    """The CPUs this process can use, seen from every side.

    A login shell's ``nproc`` can differ from what a launched job sees (a scheduler's cpuset, ``taskset``, a
    container), so the evidence records each view next to the count the lanes depend on.
    """
    view = dict(cpu_count=os.cpu_count(), affinity=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'), omp_num_threads=os.environ.get('OMP_NUM_THREADS'))
    for name, path in (('online', '/sys/devices/system/cpu/online'), ('present', '/sys/devices/system/cpu/present'),
                       ('smt_active', '/sys/devices/system/cpu/smt/active'), ('cgroup_cpuset', '/sys/fs/cgroup/cpuset.cpus.effective'),
                       ('cgroup_cpu_max', '/sys/fs/cgroup/cpu.max')):
        try:
            view[name] = Path(path).read_text().strip()
        except OSError:
            view[name] = None
    try:
        status = Path('/proc/self/status').read_text()
        view['allowed_list'] = next((line.split(':', 1)[1].strip() for line in status.splitlines() if line.startswith('Cpus_allowed_list')), None)
    except OSError:
        view['allowed_list'] = None
    for name, command in (('nproc', ['nproc']), ('nproc_all', ['nproc', '--all'])):
        try:
            view[name] = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            view[name] = None
    return view


def teacher_recipe(config, state, choice):
    """Which recipe the teacher tables are planned under: the committed one, or g1491 applied to a copy of it.

    The run is never changed. Before the revision is adopted the copy is what the first revised iteration would use.
    """
    from brass_ai.recovery import training_revision_g1491 as g1491
    from brass_ai.recovery.collector_runtime import recipe
    revised = choice == 'g1491' or (choice == 'auto' and g1491.due(state))
    if not revised:
        return None, config
    if g1491.applied(config):
        return g1491.CONTRACT, config
    return g1491.CONTRACT, dict(g1491.configure(recipe(copy.deepcopy(config)), state), execution=config.get('execution', {}))


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
    # One lane per usable CPU: pinning wraps around, so extra lanes would share a CPU and distort every timing below.
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
    return finish(checks, python=sys.version, torch=torch.__version__, memory=memory, open_file_limit=limit, cpu_view=cpu_view(),
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
        BRASS_CPU_FIXTURES=str(c.fixtures), BRASS_CPU_CORE=value['teacher_execution_core'], BRASS_CPU_NATIVE=str(native),
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
    """Every checkpoint the league can seat. Frozen v4 bot members have no file."""
    state = c.snapshot['state']
    entries = {state['history'][-1]['model_id']: state['history'][-1]}
    for item in [*state.get('quality_pool', {}).get('members', ()), state.get('evaluation_anchor'), state.get('champion'),
                 *state.get('population', ())]:
        if item is not None and item.get('path'):
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
                                parity_passed=manifest['parity']['passed'], reward_id=manifest['reward_id'],
                                shapes=len(manifest['parity']['shapes']))
    c.bundles, c.league = receipts, models
    expect(checks, 'every_league_model_has_a_bundle', set(models) == set(entries), len(models))
    expect(checks, 'an_anchor_without_utility_is_covered', any(not m['utility'] for m in models.values()),
           sum(not m['utility'] for m in models.values()))
    expect(checks, 'learner_has_the_search_head', models[c.snapshot['model_id']]['search_delta'], c.snapshot['model_id'])
    # The export itself enforces the relative-or-absolute bound; the VP head alone reaches 100+.
    expect(checks, 'parity_passed', all(m['parity_passed'] for m in models.values()), worst)
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
            worst = scaled = positions = 0
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
                pairs = [(values[:n], logits), *((values[n+4*index:n+4*index+4], head) for index, head in enumerate(heads))]
                worst = max(worst, *(float(np.abs(a-b).max()) for a, b in pairs))
                scaled = max(scaled, *(scaled_error(a, b) for a, b in pairs))
                positions += 1
                lane.release()
            results[kind] = dict(model_id=identity, positions=positions, worst_absolute_error=worst, worst_scaled_error=scaled,
                                 candidates=len(wide['expected']['choices']))
            expect(checks, kind+'_matches_pytorch', scaled <= 1., dict(scaled=scaled, absolute=worst))
    return finish(checks, kinds=results)


def teacher_jobs(c, weight, count, offset):
    """Production-shaped tables at the requested blend, regardless of saved progress."""
    from brass_ai.learning.contracts import require
    from brass_ai.learning.search_value import binding
    from brass_ai.recovery import training_revision_g1491 as g1491
    from brass_ai.recovery.teacher_phase import plan_jobs
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    config = getattr(c, 'teacher_config', c.config)
    state = dict(c.snapshot['state'])
    state['teacher_seed_cursor'] = state.get('teacher_seed_cursor', 0)+offset
    if g1491.teacher_search_head_weight(config) is not None:
        require(weight in (0, 1), 'Mixed teacher games need weight 0 or 1')
        config = g1491.configure_teacher_games(config, int(weight))
    jobs, _ = plan_jobs(config, state, model_id=c.snapshot['model_id'], export_hash=c.snapshot['checkpoint']['sha256'], epoch=1, ready=True,
                        native_source=source_fingerprint())
    jobs = [j for j in jobs if j['split'] == 'train'][:count]
    for job in jobs:
        job['search_value'] = binding(weight)
        job['readiness'] = c.readiness
    return jobs


def teacher(c):
    from brass_ai.recovery import collector_runtime as R
    from brass_ai.recovery import training_revision_g1491 as g1491
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
    # A g1491 iteration runs 48 tables at once, so its qualification does too (36+6+6 over the three blends):
    # the verification and lock contention that stretched g1490's tables from 43 s of search to 76 s grows with the table count.
    counts = TABLES_PER_BLEND_REVISED if getattr(c, 'revision', None) else TABLES_PER_BLEND
    blends = (0., 1.) if g1491.teacher_search_head_weight(getattr(c, 'teacher_config', c.config)) is not None else BLENDS
    for number, weight in enumerate(blends):
        jobs += teacher_jobs(c, weight, counts[weight], 100000*number)
    actual_blends = sorted({j['search_value']['w'] for j in jobs})
    expect(checks, 'blend_weights', actual_blends == list(blends), actual_blends)
    bootstrap = [build['bootstrap']['binary']['path']]
    if os.name != 'nt':
        bootstrap.insert(0, c.args.mono)
    lanes = min(len(jobs), c.block['workers'])
    cores = core_allocation(lanes) if c.block['pinning'] == 'one-lane-per-cpu' else []
    positions_before = c.service.metrics['positions']
    with R.scope(c.cpu_settings):
        plan = teacher_plan(ROOT, build, c.service, jobs, c.block)
        calibration = count_calibration(plan['core_source'], jobs[0])
        require(calibration['source'] == source_fingerprint(), 'Calibration source differs')
        collector = TeacherCollector(plan['binary'], plan['core_source'], bootstrap, c.service, calibration, lanes=lanes, model=c.model,
                                     history=state['history'], inference=plan['inference'], cores=cores)
        began = time.perf_counter()
        try:
            tables = collector.collect(jobs)
        except BaseException:
            write(c.out/'teacher-failures.json', dict(failures=collector.failures))
            raise
        wall = time.perf_counter()-began
    budget = c.block['teacher_budget']
    summary = search_summary(tables, budget, jobs=jobs)
    check_searches(checks, summary)
    accepted = rejected = rows = 0
    labels_seen = 0
    handoff = []
    for table, job in zip(tables, jobs, strict=True):
        began_take = time.perf_counter()
        good, bad = collector.verifier.take(table, job, c.model)
        handoff.append(time.perf_counter()-began_take)
        accepted += len(good)
        rejected += sum(bad.values())
        rows += len(table['rows'])
        labels_seen += table['metrics']['labels']
    expect(checks, 'labels_produced', accepted > 0, accepted)
    expect(checks, 'labels_accepted', rejected <= MAX_REJECTED_LABELS*max(1, rows), dict(accepted=accepted, rejected=rejected, rows=rows))
    labels = label_summary(tables)
    revised = any('teacher_revision' in job for job in jobs)
    if revised:
        check_revision(checks, jobs, tables, labels)
    # Where a labelled turn's time goes: sampled per-turn checks and the one batched model check at hand-off.
    samples = [t for table in tables for t in table['metrics'].get('verification', [])]
    verification = dict(
        handoff_seconds=spread(handoff),
        deferred_logit_check=spread(t['metrics']['deferred_logit_check']['seconds'] for t in tables if 'deferred_logit_check' in t['metrics']),
        deferred_rows=sum(t['metrics'].get('deferred_logit_check', {}).get('rows', 0) for t in tables),
        **{name: spread(t['verification_seconds'] for t in samples if t['profile'] == name and t['rows']) for name in ('full', 'fast')})
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
    return finish(checks, tables=len(tables), lanes=lanes, wall_seconds=wall, searches=summary, labels=labels, verification=verification,
                  budget=budget, revision=jobs[0].get('teacher_revision'), value_error=spread(errors),
                  # Time a table spent beyond its own searches: the wait on verification and on the shared lock.
                  search_seconds=spread(sum(s['elapsed_ns'] for s in t['metrics']['searches'])/1e9 for t in tables),
                  stall_seconds=spread((t['metrics']['elapsed_ns']-sum(s['elapsed_ns'] for s in t['metrics']['searches']))/1e9 for t in tables),
                  labels_per_table=spread(len(t['rows']) for t in tables),
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


def native_arm(c, jobs, name, *, cpu, core_plan=None):
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
    entered = time.perf_counter()
    with NativePool(ROOT, c.service, execution, build, c.config, signature, mono=c.args.mono, scope='diagnostic', cpu=cpu, core_plan=core_plan) as pool:
        began = time.perf_counter()
        records = pool.collect(frozen)
        elapsed = time.perf_counter()-began
        report = pool.last_collection_report
        left = time.perf_counter()
    closed = time.perf_counter()
    games = [admission.read(r['path'], r['native_admission'], allow_diagnostic=True) for r in records]
    # Where the stage's time goes besides collection: starting and closing the pool, and this test's own full re-read of every game.
    outside = dict(pool_start_seconds=began-entered, pool_close_seconds=closed-left, game_read_seconds=time.perf_counter()-closed)
    return games, dict(collection_seconds=elapsed, records=len(records), report=report, outside_collection=outside)


def ordinary(c):
    from brass_ai.recovery import collector_runtime as R
    from brass_ai.rust_backend.lab_cpu_runtime import ordinary_plan
    from rust_lab_b06_metrics import Meter
    checks = {}
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
    c.ordinary_report, c.ordinary_usage = report['report'], usage
    return finish(checks, jobs=len(chosen), matched=len(matched), comparison=comparison, gpu=gpu_report, cpu=report, usage=usage)


def overlap(c):
    """Teacher tables beside ordinary games, on the CPUs the training run would give them."""
    from brass_ai.learning.contracts import digest
    from brass_ai.recovery import collector_runtime as R
    from brass_ai.recovery.teacher_phase import readiness
    from brass_ai.rust_backend.lab_cpu_runtime import CorePlan, core_allocation, ordinary_plan, teacher_plan
    from brass_ai.rust_backend.lab_shadow import TeacherRun, teacher_build
    from brass_ai.rust_backend.lab_teacher import TeacherCollector, count_calibration
    checks = {}
    state = c.snapshot['state']
    build = teacher_build(state)
    receipt, ready = readiness(state.get('teacher_phase', {}).get('readiness'), c.raw['training'].get('utility_calibration'),
                               digest(c.model.config.utility))
    if not expect(checks, 'teacher_readiness', ready, receipt):
        return finish(checks)
    c.readiness = receipt
    jobs = teacher_jobs(c, 0., c.args.overlap_tables, 800000)
    games = planned_jobs(c)[:c.args.overlap_games]
    expect(checks, 'work_planned', len(jobs) == c.args.overlap_tables and len(games) > 0, dict(tables=len(jobs), games=len(games)))
    bootstrap = [build['bootstrap']['binary']['path']]
    if os.name != 'nt':
        bootstrap.insert(0, c.args.mono)
    lanes = min(len(jobs), c.block['workers'])
    cores = core_allocation(lanes) if c.block['pinning'] == 'one-lane-per-cpu' else []
    done = []
    with R.scope(c.cpu_settings):
        plan = teacher_plan(ROOT, build, c.service, jobs, c.block)
        calibration = count_calibration(plan['core_source'], jobs[0])
        c.cpu_build, settings = ordinary_plan(ROOT, c.gpu_build, c.service, games, c.block)
        core_plan = CorePlan(cores)
        collector = TeacherCollector(plan['binary'], plan['core_source'], bootstrap, c.service, calibration, lanes=lanes, model=c.model,
                                     history=state['history'], inference=plan['inference'], cores=cores, core_plan=core_plan)
        began = time.perf_counter()
        run = TeacherRun(collector, core_plan, lambda: collector.collect(jobs), done.extend)
        try:
            played, report = native_arm(c, games, 'overlap', cpu=settings, core_plan=core_plan)
        except BaseException:
            run.abort()
            raise
        ordinary_seconds = time.perf_counter()-began
        run.finish()
        wall = time.perf_counter()-began
    expect(checks, 'all_tables_finished', len(done) == len(jobs), len(done))
    expect(checks, 'all_games_finished', len(played) == len(games), len(played))
    budget = c.block['teacher_budget']
    scales = [R.teacher_job_budget(budget, job) for job in jobs]
    expect(checks, 'labels_use_c_scale', len(done) == len(scales) and all(
           row['teacher']['c2'] == R.teacher_c_scale(effective, fast=row['teacher'].get('profile') == 'fast')
           for table, effective in zip(done, scales) for row in table['rows']),
           sorted({R.teacher_c_scale(effective) for effective in scales}))
    accepted = rejected = 0
    for table, job in zip(done, jobs, strict=True):
        good, bad = collector.verifier.take(table, job, c.model)
        accepted += len(good)
        rejected += sum(bad.values())
    expect(checks, 'labels_accepted', accepted > 0 and rejected == 0, dict(accepted=accepted, rejected=rejected))
    if cores:
        expect(checks, 'lanes_waited_for_tables', report['report']['overlap']['parked_lanes'] == len(cores), report['report']['overlap'])
    expect(checks, 'tables_and_games_overlapped', ordinary_seconds < wall+1e-6 and wall < 2*ordinary_seconds,
           dict(wall=wall, ordinary_seconds=ordinary_seconds))
    return finish(checks, wall_seconds=wall, ordinary_seconds=ordinary_seconds, tables=len(jobs), games=len(games), cpu=report)


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
        return by_setup(games), c.service.metrics['positions']-before
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
    each = breakdown['process_start_seconds']/max(1, breakdown['processes'])
    expect(checks, 'processes_start_without_waiting', each <= MAX_PROCESS_START_SECONDS, each)
    return finish(checks, first_lane_ready_seconds=breakdown['first_lane_ready_seconds'], last_lane_ready_seconds=breakdown['last_lane_ready_seconds'],
                  mono_bootstrap=phases.get('mono_bootstrap_seconds'), python_import=phases.get('python_import_seconds'), total=phases.get('total_seconds'),
                  process_start_seconds=breakdown['process_start_seconds'], usage=getattr(c, 'ordinary_usage', None),
                  note='Peak RSS is the whole qualification process tree during the CPU arm, including this parent.')


STAGES = [('environment', environment), ('regressions', regressions), ('bundles', bundles), ('parity', parity),
          ('teacher', teacher), ('ordinary', ordinary), ('overlap', overlap), ('rating', rating), ('startup', startup)]
# A stage that cannot run without these results is skipped, never reported as passed.
NEEDS = dict(parity=('bundles',), teacher=('bundles',), ordinary=('bundles',), overlap=('bundles',), rating=('bundles',), startup=('ordinary',))


def main():
    from brass_ai.recovery import collector_runtime as R, training_revision_g1491 as g1491

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mono', default='mono')
    p.add_argument('--workers', type=int, default=64)
    p.add_argument('--revision', choices=('auto', 'g1491', 'none'), default='auto',
                   help='Plan teacher tables under the g1491 recipe (long and short labels, visit-count exploration); auto does so from g1491')
    p.add_argument('--teacher-full', type=int, default=640, help='Long-search simulations per turn')
    p.add_argument('--teacher-full-k', type=int, choices=R.FULL_K, help='Long-search actions (committed K or 16 by default)')
    p.add_argument('--c-scale', type=float, choices=R.C_SCALES, help='Gumbel c_scale (committed value or 0.1 by default)')
    p.add_argument('--teacher-fast', type=int, default=None, help='Short-search simulations per turn (default 128 under g1491, else 96)')
    p.add_argument('--teacher-fast-k', type=int, default=8)
    p.add_argument('--ordinary-games', type=int, default=128, help='Real frozen jobs on the CPU arm')
    p.add_argument('--matched-games', type=int, default=16, help='Of those, also run on the GPU arm')
    p.add_argument('--rating-games', type=int, default=2)
    p.add_argument('--overlap-tables', type=int, default=8, help='Teacher tables in the overlap stage')
    p.add_argument('--overlap-games', type=int, default=64, help='Ordinary games beside those tables')
    p.add_argument('--skip-regressions', action='store_true', help='Only when the suite already ran on this exact build')
    p.add_argument('--fresh', action='store_true', help='Repeat stages that already passed on this exact code, build and checkpoint')
    args = p.parse_args()
    import torch
    from brass_ai.learning.contracts import project_path, require
    from brass_ai.recovery.inference import InferenceService
    from brass_ai.recovery.retention import run_lock
    from brass_ai.rust_backend.lab_backend import check_build, validate_state
    from brass_ai.rust_backend.lab_cpu_runtime import raise_open_file_limit, resolve
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
        raise_open_file_limit()
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
                c.revision, c.teacher_config = teacher_recipe(c.config, snapshot['state'], args.revision)
                profile = g1491.count_search(c.config) or {}
                budget = dict(profile, mode='count', full=args.teacher_full, full_k=args.teacher_full_k or profile.get('full_k', 16),
                              fast=args.teacher_fast or (96 if profile or not c.revision else 128), fast_k=args.teacher_fast_k,
                              c_scale=args.c_scale or profile.get('c_scale', R.DEFAULT_C_SCALE))
                c.gpu_build = resolve_ordinary(ROOT, check_build(backend['build']))
                c.cpu_value = resolve(ROOT, teacher_core=teacher_build(snapshot['state'])['core_source'])
                c.fixtures = Path(c.gpu_build['b01']['path']).parent/'inputs/fixtures.json'
                c.block = R.cpu_block(workers=args.workers, teacher_budget=budget)
                R.validate_cpu(c.block)
                c.execution = snapshot['execution']
                c.cpu_settings = dict(R.from_config(c.config), native_inference=c.block)
                with R.scope(c.cpu_settings):
                    c.cpu_execution = R.effective_execution(c.execution)
                report['budget'], report['revision'] = c.block['teacher_budget'], c.revision
                report['identity'] = dict(contract=CONTRACT, implementation=report['execution_source']['implementation'],
                    cpu_build=c.cpu_value['receipt']['sha256'], checkpoint=snapshot['checkpoint']['sha256'], workers=args.workers,
                    budget=c.block['teacher_budget'], ordinary_games=args.ordinary_games, matched_games=args.matched_games,
                    rating_games=args.rating_games, revision=c.revision)
                earlier = {} if args.fresh else reusable_stages(ROOT/'artifacts/rust-backend', out, report['identity'])
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
                            restore_admission(c.service, c.cpu_execution)
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
