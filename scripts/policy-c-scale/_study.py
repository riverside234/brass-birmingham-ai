"""Shared inputs, native probes, and reports for the offline C_scale studies."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
from itertools import islice, zip_longest
import json
import os
from pathlib import Path
import random
import re
import shutil
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
from brass_ai.recovery.training_revision_g1491 import EXPLORATION, GAME_OPENING, _job_block
from brass_ai.rust_backend import lab_cpu_models, lab_cpu_runtime, lab_teacher
from brass_ai.rust_backend.lab_contract import native_fingerprint
from brass_ai.rust_backend.lab_paths import runtime_path
from brass_ai.step5.league import write_json

RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-3'
DEFAULT_GENERATION = 2400
DEFAULT_LANES = 60
WEIGHT = 0.0
K = 8
SHORT_N = 96
LONG_N = 640
REF_N = 3000
GROUPS = ('total', 'normal', 'cotton')
TARGETS = {'normal': 200, 'cotton': 50}


def generation_entry(state, generation):
    rows = [row for row in state['history'] if row['generation'] == generation]
    require(len(rows) == 1, f'Run history must identify exactly one g{generation}')
    return rows[0]


def source_roots(run, incoming, saved, reward_id):
    generation = incoming['generation']
    report_path = run / f'iteration-{generation:04d}/teacher-shadow/collection.json'
    require(report_path.is_file(), f'Missing source collection report: {report_path}')
    report = read_json(report_path)
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
        sidecar = file.with_name(file.name + '.receipt.json')
        require(file.is_file() and sidecar.is_file(), f'Missing source teacher labels or receipt: {file}')
        receipt = read_json(sidecar)
        require(receipt['generation'] == generation and receipt['split'] == split
                and receipt['model_id'] == incoming['model_id'] and receipt['reward_id'] == reward_id
                and sha(file) == receipt['sha256'], f'Changed teacher shard: {file}')
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
    return groups


def pooled_roots(run, entries, saved, reward_id):
    groups = {group: [] for group in GROUPS[1:]}
    source_counts = {}
    for entry in entries:
        rows = source_roots(run, entry, saved, reward_id)
        source_counts[entry['generation']] = {group: len(values) for group, values in rows.items()}
        for group, values in rows.items():
            groups[group].extend(values)
    if len(entries) > 1:
        # Shuffle the whole pool so a filled quota does not favor the first age.
        rng = random.Random(digest(['policy-c-scale-sources', [entry['generation'] for entry in entries]]))
        for values in groups.values():
            rng.shuffle(values)
    return groups, source_counts


def interleave(groups):
    return (case for pair in zip_longest(*groups) for case in pair if case is not None)


def sample_targets(rows, limit, targets=TARGETS):
    if limit == 0:
        return {group: len(positions) for group, positions in rows.items()}
    if limit is None or limit >= sum(targets.values()):
        return targets.copy()
    # An explicit small smoke run sees both groups before either quota fills.
    groups = ([group] * target for group, target in targets.items())
    selected = Counter(islice(interleave(groups), limit))
    return {group: selected[group] for group in targets}


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


def diagnostic_job(config, state, entry, saved, utility_calibration):
    generation = entry['generation']
    jobs, _ = plan_jobs(config, dict(state, iteration=generation, search_value=saved),
                        model_id=entry['model_id'], export_hash=entry['sha256'], epoch=generation,
                        ready=True, native_source=lab_teacher.source_fingerprint())
    job = dict(next(j for j in jobs if not j.get('shadow_probe')), split='heldout', search_value=binding(WEIGHT))
    # The diagnostic binary uses the native visit sampler only inside game_opening.
    # All source groups use the same job recipe and an empty set of cotton seats.
    job['teacher_revision'] = _job_block(dict(EXPLORATION, game_opening=GAME_OPENING),
                                         short_labels=False, search_head_weight=int(WEIGHT), seats=[])
    job['readiness'], _ = readiness(None, utility_calibration, job['expected']['reward_id'], skip=True)
    lab_teacher.validate_job(job)
    return job


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
        probe_row = labels[0]
        label = probe_row['teacher']
        require(probe_row['view'] == row['view'] and probe_row['diagnostic'] is True
                and label['legal_ids'] == archived['legal_ids'] and label['search_value'] == binding(WEIGHT)
                and label['requested_k'] == K and label['frozen_n'] == label['completed_trajectories'] == n
                and label['c2'] == scale, 'Probe root or settings differ')
        context = row['view']['context']
        opening = context['era'] == 'CanalEra' and context['round'] <= 2
        require(label.get('explored', False) == opening
                and label.get('sampling_temperature') == (GAME_OPENING['temperature'] if opening else None)
                and 'cotton_exploration' not in label, 'Probe opening sampling differs')
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


def study_worker(binary, scratch, toolchain, core):
    # The CPU preparation script puts target/<host>/release beside checkout.
    source = binary.parents[3] / 'checkout'
    require((source / 'lab-native/src/cpu_eval.rs').is_file() and native_fingerprint(source / 'native') == core,
            f'Expected the pinned CPU teacher checkout beside the build: {source}')
    destination = scratch / 'teacher-probe'
    for name in ('native', 'lab-native'):
        shutil.copytree(source / name, destination / name,
                        ignore=shutil.ignore_patterns('target', '.git', '__pycache__'))

    def patch(file, replacements, *, patterns=()):
        path = destination / 'lab-native/src' / file
        code = path.read_text(encoding='utf-8')
        for before, after in patterns:
            code, count = re.subn(before, after, code)
            require(count == 1, f'CPU teacher diagnostic edit did not match {path.name}: {before}')
        for before, after in replacements:
            require(code.count(before) == 1, f'CPU teacher diagnostic edit did not match {path.name}: {before}')
            code = code.replace(before, after, 1)
        path.write_text(code, encoding='utf-8', newline='\n')

    patch('cpu_eval.rs', [
        ('        let index = state.action(actual.actor, c.era, c.round, c.actions_remaining);\n'
         '        let window = index <= revision.opening_actions || !fast && revision.long_search;',
         '        state.action(actual.actor, c.era, c.round, c.actions_remaining);\n'
         '        let window = revision.tempered_opening && game_opening(c.era, c.round);'),
        ('        check(!fast || revision.short_labels, "short-search label without short labels")?;\n', ''),
    ], patterns=[(r'(?m)^        \[DEFAULT_C_SCALE,[^\r\n]*\]\.contains\(&c_scale\),$',
                   '        c_scale.is_finite() && c_scale > 0. && c_scale <= 1.,')])
    patch('teacher.rs', [
        ('    let job = &request["job"];',
         '    check(request["op"] == "turn_probe", "research worker only accepts turn probes")?;\n'
         '    let job = &request["job"];'),
        ('(!turn_work.fast || crate::cpu_eval::teacher_short_labels())',
         '(!turn_work.fast || probe || crate::cpu_eval::teacher_short_labels())'),
    ])
    return build_binary(destination / 'lab-native/Cargo.toml', destination / 'target',
                        'brass-lab-teacher', toolchain, teacher=True)


def print_and_save(kind, context, groups):
    lines = []
    for group, summary in groups.items():
        lines.extend(('', f"{group}: {summary['valid']}/{summary['requested']} valid positions "
                          f"({summary['roots']} attempted)", summary['header']))
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


def compare_roots(sampler, rows, probes, targets, compare, scales, lanes):
    positions = {group: iter(values) for group, values in rows.items()}
    valid = {group: [] for group in rows}
    attempts = Counter()
    with ThreadPoolExecutor(max_workers=lanes) as pool:
        while True:
            pending = [islice(positions[group], targets[group] - len(valid[group])) for group in rows]
            batch = list(interleave(pending))
            if not batch:
                break
            cases = list(prepare_origins(sampler, batch))
            work = {pool.submit(compare, probes, case, scales): index for index, case in enumerate(cases)}
            previous = sum(attempts.values())
            try:
                results = [None] * len(cases)
                for done, task in enumerate(as_completed(work), 1):
                    results[work[task]] = task.result()
                    if done == 1 or done % 25 == 0 or done == len(cases):
                        print(f'Finished {previous + done} positions', flush=True)
            except BaseException:
                probes.stop.set()
                for task in work:
                    task.cancel()
                raise
            for (group, _, _), result in zip(cases, results, strict=True):
                attempts[group] += 1
                if result is not None:
                    valid[group].append(result)
    return valid, attempts


def main(kind, default_scales, compare, summarize, *, candidate_n, default_targets=TARGETS):
    parser = argparse.ArgumentParser(description=f'Offline {kind} search C_scale study')
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--generation', type=int, default=int(os.getenv('GEN', str(DEFAULT_GENERATION))),
                        help=f'Frozen checkpoint (default: g{DEFAULT_GENERATION} or GEN)')
    parser.add_argument('--collected-generation', '--collected-generations', type=int, nargs='+',
                        help='One or more archived generations to pool (default: generation - 1)')
    parser.add_argument('--lanes', type=int, default=int(os.getenv('LANES', str(DEFAULT_LANES))),
                        help=f'Concurrent CPU workers (default: {DEFAULT_LANES} or LANES)')
    parser.add_argument('--ref-scale', type=float, choices=(0.1, 1.0), default=float(os.getenv('REF_SCALE', '0.1')))
    parser.add_argument('--normal-positions', type=int, default=default_targets['normal'],
                        help=f'Normal position target (default: {default_targets["normal"]})')
    parser.add_argument('--cotton-positions', type=int, default=default_targets['cotton'],
                        help=f'Cotton position target (default: {default_targets["cotton"]})')
    parser.add_argument('--limit', type=int, default=os.getenv('LIMIT'),
                        help='Cap the combined normal/cotton position targets, or 0 for all available positions')
    parser.add_argument('--scales', type=float, nargs='+', default=default_scales)
    parser.add_argument('--sampler', default=os.getenv('BRASS_NATIVE_WORKER'))
    parser.add_argument('--toolchain', default='1.99.1')
    args = parser.parse_args()
    referenced = [args.generation - 1] if args.collected_generation is None else sorted(args.collected_generation)
    if not (len(set(referenced)) == len(referenced) and all(0 <= g < args.generation for g in referenced)
            and args.lanes > 0 and (args.limit is None or args.limit >= 0)
            and args.normal_positions > 0 and args.cotton_positions > 0
            and args.ref_scale in (0.1, 1.0)
            and len(set(args.scales)) == len(args.scales) and all(0 < scale <= 1 for scale in args.scales)):
        parser.error('Use distinct committed source generations, positive lane/position counts, and distinct C_scale values in (0, 1].')

    import torch
    from brass_ai.learning.checkpoint import load_checkpoint

    run = project_path(ROOT, args.run)
    state = read_json(run / 'state.json')
    entry = generation_entry(state, args.generation)
    incoming = [generation_entry(state, generation) for generation in referenced]
    checkpoint = project_path(ROOT, entry['path'])
    require(checkpoint.is_file() and sha(checkpoint) == entry['sha256'],
            f'Missing or changed checkpoint g{args.generation}: {checkpoint}')
    torch.set_num_threads(2)
    model, raw = load_checkpoint(checkpoint)
    model.eval()
    require(raw['step'] == args.generation and raw['model_id'] == entry['model_id']
            and hasattr(model, 'search_value_head'), 'Expected checkpoint with a search value head')
    saved = raw['training']['search_value']
    job = diagnostic_job(state['signature']['config'], state, entry, saved, raw['training'].get('utility_calibration'))
    rows, source_counts = pooled_roots(run, incoming, saved, job['expected']['reward_id'])
    targets = sample_targets(rows, args.limit, dict(normal=args.normal_positions, cotton=args.cotton_positions))
    build = lab_cpu_runtime.resolve(ROOT)
    core = build['teacher_execution_core']
    lab_cpu_runtime.raise_open_file_limit()
    cap_seconds = REF_N * lab_cpu_runtime.COUNT_CAP_PER_TRAJECTORY_NS / 1e9
    lab_teacher.COUNT_TURN_SECONDS = max(lab_teacher.COUNT_TURN_SECONDS, cap_seconds + 30)
    sources = ', '.join(f'g{generation}' for generation in referenced)
    print(f'g{args.generation} | {sources} positions | targets {targets} | '
          f'{args.lanes} lanes | w={WEIGHT:g} | reference K{K}/{REF_N} C_scale={args.ref_scale:g} | '
          f'opening T={GAME_OPENING["temperature"]:g}',
          flush=True)

    with TemporaryDirectory(prefix='brass-policy-c-scale-', dir=ROOT / 'artifacts',
                            ignore_cleanup_errors=os.name == 'nt') as directory:
        scratch = Path(directory)
        sampler = Path(args.sampler).resolve() if args.sampler else ROOT / 'native/target/release' / (
            'brass-ai-worker.exe' if os.name == 'nt' else 'brass-ai-worker')
        if not sampler.is_file() and not args.sampler:
            sampler = build_binary(ROOT / 'native/Cargo.toml', scratch / 'sampler', 'brass-ai-worker', args.toolchain)
        require(sampler.is_file(), f'Missing native position sampler: {sampler}')
        manifest = lab_cpu_models.bundle_root(ROOT) / entry['model_id'] / 'manifest.json'
        if not manifest.is_file():
            manifest = scratch / 'model/manifest.json'
            lab_cpu_models.export_bundle(model, manifest.parent)
        binary = study_worker(runtime_path(build['teacher']['path']), scratch, args.toolchain, core)
        inference = dict(contract=lab_cpu_runtime.CONTRACT,
                         runtime=dict(path=str(runtime_path(build['runtime']['path'])), sha256=build['runtime']['sha256']),
                         bundle=dict(manifest=str(manifest), sha256=sha(manifest)))
        del raw, model
        probes = Probes(binary, core, job, inference, args.ref_scale, Event())
        results, attempts = compare_roots(sampler, rows, probes, targets, compare, args.scales, args.lanes)
        worker_sha = sha(binary)

    results['total'] = [*results['normal'], *results['cotton']]
    attempts['total'] = sum(attempts.values())
    targets['total'] = sum(targets.values())
    groups = {}
    for group in GROUPS:
        valid = results[group]
        groups[group] = dict(roots=attempts[group], valid=len(valid), requested=targets[group], **summarize(valid, args.scales))
        if len(valid) < targets[group]:
            print(f'{group}: {len(valid)}/{targets[group]} requested valid positions after exhausting the archived roots', flush=True)
    source_context = dict(collected_generations=referenced, source_positions=source_counts)
    if len(referenced) == 1:
        source_context['collected_generation'] = referenced[0]
    print_and_save(kind, dict(run=str(run), generation=args.generation, **source_context,
                              checkpoint=entry, w=WEIGHT, k=K, ref_scale=args.ref_scale, ref_n=REF_N,
                              candidate_n=candidate_n, method='native-scale-sweep', game_opening=GAME_OPENING,
                              lanes=args.lanes, scales=args.scales, worker_sha256=worker_sha, core_sha256=core),
                   groups)
    require(groups['total']['valid'], 'No complete, supported reference pairs')
