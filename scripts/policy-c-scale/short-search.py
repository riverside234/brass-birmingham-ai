#!/usr/bin/env python3
"""Run K8/96 searches at each scale against a K8/2560 reference."""

from statistics import mean

import numpy as np

from _study import REF_N, SHORT_N, completed_q, main

SCALES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HEADER = 'short_scale  match_long_move  differs_from_policy  mean_Q_gap  miss_Q_gap  Q_supported_misses'


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
    main('short', SCALES, compare, summarize, candidate_n=SHORT_N)
