#!/usr/bin/env python3
"""Compare re-scored K8/640 policies against a K8/2560 reference move and label."""

import numpy as np

from _study import LONG_N, REF_N, completed_q, main

SCALES = (0.10, 0.15, 0.20, 0.25, 0.30)
HEADER = 'short_scale  match_long_move  match_long_label  move_NLL'


def compare(probes, case, scales):
    _, row, origin = case
    archived = row['teacher']
    trial = probes.trial(archived)
    reference = probes.run(trial, origin, row, REF_N, probes.ref_scale)
    if reference is None or reference['fallback_played']:
        return None
    base = probes.run(trial, origin, row, LONG_N, archived['c2'])
    if base is None:
        return None
    move = reference['legal_ids'].index(reference['played_action'])
    label = int(np.argmax(reference['target']))
    logits = np.asarray(base['logits'], dtype=np.float64)
    adjustment = (base['c1'] + max(base['counts'])) * completed_q(base)
    return [
        (int(np.argmax(scores)) == move, int(np.argmax(scores)) == label,
         float(np.logaddexp.reduce(scores) - scores[move]))
        for scores in (logits, *(logits + scale * adjustment for scale in scales))
    ]


def summarize(valid, scales):
    rows, lines = [], []
    if valid:
        names = ('policy', *(f'{scale:.2f}' for scale in scales))
        for name, (move, label, nll) in zip(names, np.mean(valid, axis=0).tolist(), strict=True):
            rows.append(dict(short_scale=name, match_long_move=move, match_long_label=label, move_NLL=nll))
            lines.append(f'{name:>11}  {move:>15.1%}  {label:>16.1%}  {nll:>8.4f}')
    return dict(header=HEADER, lines=lines, rows=rows)


if __name__ == '__main__':
    main('long', SCALES, compare, summarize)
