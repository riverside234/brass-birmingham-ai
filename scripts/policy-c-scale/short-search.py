#!/usr/bin/env python3
"""Run K8/96 searches at each scale against a K8/2560 reference."""

import shutil
from statistics import mean

import numpy as np

from _study import REF_N, SHORT_N, build_binary, completed_q, main
from brass_ai.learning.contracts import require
from brass_ai.rust_backend.lab_contract import native_fingerprint

SCALES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HEADER = 'short_scale  match_long_move  differs_from_policy  mean_Q_gap  miss_Q_gap  Q_supported_misses'


def study_worker(binary, scratch, toolchain, core):
    # Use the exact sources from the prepared CPU worker. Compile a separate
    # turn-probe executable so evaluation-only scales cannot enter training.
    # prepare-rust-cpu-inference.py places target/<host>/release next to checkout.
    source = binary.parents[3] / 'checkout'
    require((source / 'lab-native/src/cpu_eval.rs').is_file() and
            native_fingerprint(source / 'native') == core,
            f'Expected the pinned CPU teacher checkout beside the build: {source}')
    destination = scratch / 'short-probe'
    for name in ('native', 'lab-native'):
        shutil.copytree(source / name, destination / name,
                        ignore=shutil.ignore_patterns('target', '.git', '__pycache__'))

    def patch(file, replacements):
        path = destination / 'lab-native/src' / file
        code = path.read_text(encoding='utf-8')
        for before, after in replacements:
            require(code.count(before) == 1, f'CPU teacher diagnostic edit did not match {path.name}: {before}')
            code = code.replace(before, after, 1)
        path.write_text(code, encoding='utf-8', newline='\n')

    patch('cpu_eval.rs', [
        ('[DEFAULT_C_SCALE, 0.15, 0.20, 0.5, 1.0].contains(&c_scale)',
         'c_scale.is_finite() && c_scale > 0. && c_scale <= 1.'),
    ])
    patch('teacher.rs', [
        ('    let job = &request["job"];',
         '    check(request["op"] == "turn_probe", "research worker only accepts turn probes")?;\n'
         '    let job = &request["job"];'),
        ('(!turn_work.fast || crate::cpu_eval::teacher_short_labels())',
         '(!turn_work.fast || probe || crate::cpu_eval::teacher_short_labels())'),
        ('                    crate::cpu_eval::teacher_mark_label(&mut row, turn_work.fast, explored)?;',
         '                    if probe && turn_work.fast && job.get("teacher_revision").is_none() {\n'
         '                        row["teacher"]["profile"] = json!("fast");\n'
         '                        row["teacher"]["requested_k"] = json!(crate::cpu_eval::teacher_k(true, false, 4));\n'
         '                    } else {\n'
         '                        crate::cpu_eval::teacher_mark_label(&mut row, turn_work.fast, explored)?;\n'
         '                    }'),
    ])
    return build_binary(destination / 'lab-native/Cargo.toml', destination / 'target',
                        'brass-lab-teacher', toolchain, teacher=True)


def compare(probes, case, scales):
    _, row, origin = case
    trial = probes.trial(row['teacher'])
    reference = probes.run(trial, origin, row, REF_N, probes.ref_scale)
    if reference is None or reference['fallback_played']:
        return None
    ids = reference['legal_ids']
    move = ids.index(reference['played_action'])
    policy = int(np.argmax(reference['logits']))
    picks = [policy]
    for scale in scales:
        short = probes.run(trial, origin, row, SHORT_N, scale, fast=True)
        if short is None:
            return None
        picks.append(ids.index(short['played_action']))
    q = completed_q(reference)
    counts = reference['counts']
    # Completed Q includes the mixed-value estimate for unvisited actions.
    # A supported miss requires direct visits to both actions.
    return [(pick == move, pick != policy, float(q[move] - q[pick]),
             pick != move and counts[move] > 0 and counts[pick] > 0) for pick in picks]


def summarize(valid, scales):
    rows, lines = [], []
    if valid:
        names = ('policy', *(f'{scale:.2f}' for scale in scales))
        for name, results in zip(names, zip(*valid, strict=True), strict=True):
            hits, changed, gaps, support = zip(*results, strict=True)
            misses = len(hits) - sum(hits)
            supported_gaps = [gap for gap, used in zip(gaps, support, strict=True) if used]
            supported = len(supported_gaps)
            match, change, overall = mean(hits), mean(changed), mean(gaps)
            miss = mean(supported_gaps) if supported else None
            miss_display = 'n/a' if miss is None else f'{miss:+.5f}'
            rows.append(dict(short_scale=name, match_long_move=match, differs_from_policy=change,
                             mean_Q_gap=overall, miss_Q_gap=miss,
                             Q_supported_misses=dict(supported=supported, total=misses)))
            lines.append(f'{name:>11}  {match:>15.1%}  {change:>19.1%}  {overall:>+10.5f}  '
                         f'{miss_display:>10}  {supported}/{misses}')
    return dict(header=HEADER, lines=lines, rows=rows)


if __name__ == '__main__':
    main('short', SCALES, compare, summarize, study_worker=study_worker)
