"""Shared inputs, native probes, and reports for the offline C_scale studies."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
from itertools import zip_longest
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.learning.search_value import binding
from brass_ai.recovery.teacher_phase import plan_jobs, readiness, reserved_rounds
from brass_ai.rust_backend import lab_cpu_models, lab_cpu_runtime, lab_teacher
from brass_ai.rust_backend.lab_paths import runtime_path
from brass_ai.step5.league import write_json

RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-3'
WEIGHT = 1.0
K = 8
SHORT_N = 96
LONG_N = 640
REF_N = 2560
GROUPS = ('total', 'normal', 'cotton')


def generation_entry(state, generation):
    rows = [row for row in state['history'] if row['generation'] == generation]
    require(len(rows) == 1, f'Run history must identify exactly one g{generation}')
    return rows[0]


def source_roots(run, incoming, saved, reward_id, limit):
    generation = incoming['generation']
    report = read_json(run / f'iteration-{generation:04d}/teacher-shadow/collection.json')
    tables = {}
    for job, table, metrics in zip(report['post_admission_per_table_seconds'],
                                   report['teacher_actions']['per_table'], report['collection_metrics'], strict=True):
        exploration = metrics['cpu_inference']['revision']['job']['exploration']
        require(table['cotton_exploration'] == ('cotton' in exploration), 'Teacher cotton assignment differs')
        group = 'cotton' if table['cotton_exploration'] else 'normal'
        tables['teacher/' + job] = (table['split'], group)
    require(len(tables) == len(report['teacher_actions']['per_table']), 'Duplicate teacher table')

    receipts = [receipt for receipt in saved['buffer']['receipts']
                if receipt['generation'] == generation and receipt['reward_id'] == reward_id]
    require(receipts, f'The checkpoint needs retained g{generation} search-value shards for the selected reward')
    zero_spent = set()
    for receipt in receipts:
        file = run / 'search-values' / receipt['file']
        require(file.is_file() and sha(file) == receipt['sha256'], f'Missing or changed value shard: {file}')
        with gzip.open(file, 'rt', encoding='utf-8') as stream:
            zero_spent.update((row['game_id'], row['revision']) for row in json.load(stream)
                              if row['model_id'] == incoming['model_id'] and not any(row['spent']))

    groups = {group: [] for group in GROUPS[1:]}
    for split in ('train', 'heldout'):
        if not any(table_split == split for table_split, _ in tables.values()):
            continue
        file = run / 'teacher-labels' / f'generation-{generation:06d}-{split}.jsonl.gz'
        receipt = read_json(file.with_name(file.name + '.receipt.json'))
        require(receipt['generation'] == generation and receipt['split'] == split
                and receipt['model_id'] == incoming['model_id'] and receipt['reward_id'] == reward_id
                and file.is_file() and sha(file) == receipt['sha256'], f'Missing or changed teacher shard: {file}')
        with gzip.open(file, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                label = row['teacher']
                if (label['kind'] == 'root' and len(label['legal_ids']) > 1
                        and label['model_id'] == incoming['model_id'] and label['export_hash'] == incoming['sha256']
                        and label['reward_id'] == reward_id and label['requested_k'] == K
                        and label['frozen_n'] == label['completed_trajectories'] == LONG_N
                        and (label['game_id'], label['revision']) in zero_spent):
                    table_split, group = tables[label['game_id']]
                    require(table_split == split, 'Teacher table split differs')
                    groups[group].append((group, row))

    rng = random.Random(generation)
    require(all(groups.values()), f'g{generation} has no eligible zero-spend roots in one or both groups')
    for rows in groups.values():
        rng.shuffle(rows)
    # Interleave the groups so a limited run samples both.
    selection = [case for pair in zip_longest(*groups.values()) for case in pair if case is not None]
    return selection[:limit] if limit else selection


def prepare_origins(sampler, rows):
    from brass_ai.native.protocol import NativeWorker

    with NativeWorker(sampler, timeout=60) as native:
        for group, row in rows:
            view, label = row['view'], row['teacher']
            recall = [dict(revision=label['revision'], hand=view['own_hand'])]
            draws = native.simulation(0, 'prepare', view=view, recall=recall)['draw_count']
            sample = native.simulation(0, 'sample', random=dict(kind='recorded', indices=[0] * draws))
            handle, snapshot = sample['handle'], sample['snapshot']
            try:
                require(snapshot['information'][view['seat']]['observation']['view'] == view
                        and {choice['id'] for choice in snapshot['choices']} == set(label['legal_ids']),
                        'Reconstructed root differs from the archived position')
                state = native.simulation(0, 'export', handle=handle)['checkpoint']['world']['state']
            finally:
                native.simulation(0, 'release', handle=handle)
            yield group, row, dict(state=state, rail_keys=list(range(64)))


def completed_q(label):
    return np.where(np.asarray(label['counts']) > 0, label['mean_returns'], label['v_mix'])


class DiagnosticTurn:
    def __init__(self, job, profile):
        self.job, self.profile = job, profile

    def verify(self, frame, _received):
        require(frame['job'] == self.job and frame['diagnostic'] is True
                and frame['profile'] == self.profile, 'Mismatched diagnostic turn')


class Probes:
    def __init__(self, binary, core, job, inference, ref_scale, stop):
        self.binary, self.core, self.job, self.inference = binary, core, job, inference
        self.ref_scale, self.stop = ref_scale, stop
        calibration = lab_teacher.count_calibration(core, job)
        self.timing = dict(estimate_ns=calibration['estimate_ns'], calibration_id=digest(calibration))

    def trial(self, label):
        token = digest(['policy-c-scale', self.job['expected']['model_id'], label['game_id'], label['revision']])
        seed = int(token[:16], 16)
        expected = dict(self.job['expected'], job_id=token, game_id='teacher/' + token)
        return dict(self.job, expected=expected, sampling_seed=seed, reservations={
            str(seat): reserved_rounds(self.job['seed'], seed, seat) for seat in range(4)
        })

    def run(self, trial, origin, row, n, scale, *, fast=False):
        require(not self.stop.is_set(), 'C_scale study cancelled')
        archived = row['teacher']
        profile = 'fast' if fast else 'full'
        budget = dict(mode='count', full=REF_N if fast else n, full_k=K, fast=SHORT_N, fast_k=K,
                      c_scale=self.ref_scale if fast else scale)
        if fast:
            budget['fast_c_scale'] = scale
        request = dict(contract=lab_teacher.CONTRACT, op='turn_probe', job=trial, origin=origin,
                       core_source=self.core, probe_profile=profile,
                       inference=dict(self.inference, budget=budget), **self.timing)
        expected = trial['expected']
        result = lab_teacher.Worker(self.binary, request, None, self.stop).collect(
            expected, self.core, verifier=DiagnosticTurn(expected, profile), turn_probe=True)
        revision = archived['revision']
        searches = [m for m in result['metrics']['searches'] if m['revision'] == revision]
        labels = [r for r in result['rows'] if r['teacher']['revision'] == revision]
        if not searches:
            return None
        require(len(searches) == 1 and searches[0]['profile'] == profile
                and searches[0]['frozen_n'] == n and searches[0]['c_scale'] == scale, 'Wrong search profile, quota or scale')
        if searches[0]['completed'] != n:
            return None
        require(len(labels) == 1, 'Complete diagnostic search did not export one root label')
        label = labels[0]['teacher']
        require(labels[0]['view'] == row['view'] and labels[0]['diagnostic'] is True
                and label['legal_ids'] == archived['legal_ids'] and label['search_value'] == binding(WEIGHT)
                and label['requested_k'] == K and label['frozen_n'] == label['completed_trajectories'] == n
                and label['c2'] == scale and not label.get('explored', False), 'Probe root or settings differ')
        return label


def build_binary(manifest, target, name, toolchain, *, teacher=False):
    print(f'Building isolated {name}', flush=True)
    command = ['cargo', '+' + toolchain, 'build', '--locked', '--release',
               '--manifest-path', str(manifest), '--target-dir', str(target), '--bin', name]
    if teacher:
        command.extend(('--features', 'teacher'))
    # Each build has an explicit output directory and uses the machine's host target.
    env = dict(os.environ)
    env.pop('CARGO_BUILD_TARGET', None)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return target / 'release' / (name + ('.exe' if os.name == 'nt' else ''))


def print_and_save(kind, context, groups):
    lines = []
    for group, summary in groups.items():
        lines.extend(('', f"{group}: {summary['valid']}/{summary['roots']} valid positions", summary['header']))
        lines.extend(summary['lines'])
    table = '\n'.join(lines).lstrip() + '\n'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = project_path(ROOT, f'results/policy-c-scale/g{context["generation"]}-{kind}-{stamp}')
    require(not output.is_relative_to(Path(context['run'])), 'Study output must be outside the training run')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'table.txt').write_text(table, encoding='utf-8')
    write_json(output / 'summary.json', dict(context, kind=kind, groups={
        group: {key: value for key, value in summary.items() if key not in ('header', 'lines')}
        for group, summary in groups.items()
    }))
    print(table, end='')
    print(f'Results: {output}')


def main(kind, default_scales, compare, summarize, study_worker=None):
    parser = argparse.ArgumentParser(description=f'Offline g2240 {kind} search C_scale study')
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--generation', type=int, default=int(os.getenv('GEN', '2240')))
    parser.add_argument('--collected-generation', type=int, help='Source of archived positions (default: generation - 1)')
    parser.add_argument('--lanes', type=int, default=int(os.getenv('LANES', '48')))
    parser.add_argument('--ref-scale', type=float, choices=(0.1, 1.0), default=float(os.getenv('REF_SCALE', '0.1')))
    parser.add_argument('--limit', type=int, default=int(os.getenv('LIMIT', '200')), help='Maximum positions, or 0 for all')
    parser.add_argument('--scales', type=float, nargs='+', default=default_scales)
    parser.add_argument('--sampler', default=os.getenv('BRASS_NATIVE_WORKER'))
    parser.add_argument('--toolchain', default='1.99.1')
    args = parser.parse_args()
    referenced = args.generation - 1 if args.collected_generation is None else args.collected_generation
    if not (0 <= referenced < args.generation and args.lanes > 0 and args.limit >= 0
            and args.ref_scale in (0.1, 1.0)
            and len(set(args.scales)) == len(args.scales) and all(0 < scale <= 1 for scale in args.scales)):
        parser.error('Use committed source positions, a positive lane count, and distinct C_scale values in (0, 1].')

    import torch
    from brass_ai.learning.checkpoint import load_checkpoint

    run = project_path(ROOT, args.run)
    state = read_json(run / 'state.json')
    entry = generation_entry(state, args.generation)
    incoming = generation_entry(state, referenced)
    checkpoint = project_path(ROOT, entry['path'])
    require(checkpoint.is_file() and sha(checkpoint) == entry['sha256'],
            f'Missing or changed checkpoint g{args.generation}: {checkpoint}')
    torch.set_num_threads(2)
    model, raw = load_checkpoint(checkpoint)
    model.eval()
    require(raw['step'] == args.generation and raw['model_id'] == entry['model_id']
            and hasattr(model, 'search_value_head'), 'Expected checkpoint with a search value head')
    saved = raw['training']['search_value']
    jobs, _ = plan_jobs(state['signature']['config'], dict(state, iteration=args.generation, search_value=saved),
                        model_id=entry['model_id'], export_hash=entry['sha256'], epoch=args.generation,
                        ready=True, native_source=lab_teacher.source_fingerprint())
    job = dict(next(j for j in jobs if not j.get('shadow_probe')), split='heldout', search_value=binding(WEIGHT))
    # Use the archived table groups, with identical search seeds and no cotton or
    # visit-count action sampling in the new probes.
    job.pop('teacher_revision', None)
    reward_id = job['expected']['reward_id']
    job['readiness'], _ = readiness(None, raw['training'].get('utility_calibration'), reward_id, skip=True)
    lab_teacher.validate_job(job)
    rows = source_roots(run, incoming, saved, reward_id, args.limit)
    build = lab_cpu_runtime.resolve(ROOT)
    core = build['teacher_execution_core']
    lab_cpu_runtime.raise_open_file_limit()
    cap_seconds = REF_N * lab_cpu_runtime.COUNT_CAP_PER_TRAJECTORY_NS / 1e9
    lab_teacher.COUNT_TURN_SECONDS = max(lab_teacher.COUNT_TURN_SECONDS, cap_seconds + 30)
    print(f'g{args.generation} | g{referenced} positions | {len(rows)} roots {dict(Counter(g for g, _ in rows))} | '
          f'{args.lanes} lanes | w={WEIGHT:g} | REF_SCALE={args.ref_scale:g}', flush=True)

    with TemporaryDirectory(prefix='brass-policy-c-scale-', dir=ROOT / 'artifacts',
                            ignore_cleanup_errors=os.name == 'nt') as directory:
        scratch = Path(directory)
        sampler = Path(args.sampler).resolve() if args.sampler else ROOT / 'native/target/release' / (
            'brass-ai-worker.exe' if os.name == 'nt' else 'brass-ai-worker')
        if not sampler.is_file() and not args.sampler:
            sampler = build_binary(ROOT / 'native/Cargo.toml', scratch / 'sampler', 'brass-ai-worker', args.toolchain)
        require(sampler.is_file(), f'Missing native position sampler: {sampler}')
        cases = list(prepare_origins(sampler, rows))
        manifest = lab_cpu_models.bundle_root(ROOT) / entry['model_id'] / 'manifest.json'
        if not manifest.is_file():
            manifest = scratch / 'model/manifest.json'
            lab_cpu_models.export_bundle(model, manifest.parent)
        binary = runtime_path(build['teacher']['path'])
        if study_worker is not None:
            binary = study_worker(binary, scratch, args.toolchain, core)
        inference = dict(contract=lab_cpu_runtime.CONTRACT,
                         runtime=dict(path=str(runtime_path(build['runtime']['path'])), sha256=build['runtime']['sha256']),
                         bundle=dict(manifest=str(manifest), sha256=sha(manifest)))
        del raw, model
        stop = Event()
        probes = Probes(binary, core, job, inference, args.ref_scale, stop)
        results = [None] * len(cases)
        with ThreadPoolExecutor(max_workers=args.lanes) as pool:
            work = {pool.submit(compare, probes, case, args.scales): index for index, case in enumerate(cases)}
            try:
                for done, task in enumerate(as_completed(work), 1):
                    results[work[task]] = task.result()
                    if done == 1 or done % 25 == 0 or done == len(cases):
                        print(f'Finished {done}/{len(cases)} positions', flush=True)
            except BaseException:
                stop.set()
                for task in work:
                    task.cancel()
                raise
        worker_sha = sha(binary)

    groups = {}
    for group in GROUPS:
        pairs = [result for (name, _, _), result in zip(cases, results, strict=True)
                 if group == 'total' or name == group]
        valid = [result for result in pairs if result is not None]
        groups[group] = dict(roots=len(pairs), valid=len(valid), **summarize(valid, args.scales))
    print_and_save(kind, dict(run=str(run), generation=args.generation, collected_generation=referenced,
                              checkpoint=entry, w=WEIGHT, k=K, ref_scale=args.ref_scale, ref_n=REF_N,
                              candidate_n=SHORT_N if study_worker is not None else LONG_N,
                              method='native-scale-sweep' if study_worker is not None else 'rescore-root-statistics',
                              lanes=args.lanes, scales=args.scales, worker_sha256=worker_sha, core_sha256=core),
                   groups)
    require(groups['total']['valid'], 'No complete, supported reference pairs')
