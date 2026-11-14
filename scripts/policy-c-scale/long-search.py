#!/usr/bin/env python3
"""Run K8/640 searches at each scale against a K8/2560 reference."""

import numpy as np

from _study import LONG_N, REF_N, completed_q, main

SCALES = (0.10, 0.15, 0.20, 0.25, 0.30)
HEADER = 'short_scale  match_long_move  match_long_label  move_NLL'


def compare(probes, case, scales):
    _, row, origin = case
    trial = probes.trial(row['teacher'])
    reference = probes.run(trial, origin, row, REF_N, probes.ref_scale)
    if reference is None or reference['fallback_played']:
        return None
    ids = reference['legal_ids']
    move = ids.index(reference['played_action'])
    label = int(np.argmax(reference['target']))
    logits = np.asarray(reference['logits'], dtype=np.float64)
    policy = int(np.argmax(logits))
    comparisons = [(policy == move, policy == label, float(np.logaddexp.reduce(logits) - logits[move]))]
    for scale in scales:
        candidate = probes.run(trial, origin, row, LONG_N, scale)
        if candidate is None:
            return None
        # Keep NLL in log space using this candidate's own search statistics.
        multiplier = candidate['c2'] * (candidate['c1'] + max(candidate['counts']))
        scores = np.asarray(candidate['logits'], dtype=np.float64) + multiplier * completed_q(candidate)
        comparisons.append((candidate['played_action'] == reference['played_action'],
                            int(np.argmax(candidate['target'])) == label,
                            float(np.logaddexp.reduce(scores) - scores[move])))
    return comparisons


def summarize(valid, scales):
    rows, lines = [], []
    if valid:
        names = ('policy', *(f'{scale:.2f}' for scale in scales))
        for name, (move, label, nll) in zip(names, np.mean(valid, axis=0).tolist(), strict=True):
            rows.append(dict(short_scale=name, match_long_move=move, match_long_label=label, move_NLL=nll))
            lines.append(f'{name:>11}  {move:>15.1%}  {label:>16.1%}  {nll:>8.4f}')
    return dict(header=HEADER, lines=lines, rows=rows)


if __name__ == '__main__':
    main('long', SCALES, compare, summarize, candidate_n=LONG_N)
