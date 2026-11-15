#!/usr/bin/env python3
"""Run K8/640 searches at each scale against a K8/3000 reference."""

import numpy as np

from _study import LONG_N, REF_N, completed_q, main

SCALES = (0.10, 0.15, 0.20, 0.25, 0.30)
TARGETS = {'normal': 700, 'cotton': 200}
HEADER = 'short_scale  target_KL  match_long_move  match_long_label  move_NLL'


def target_kl(reference, log_probabilities):
    positive = reference > 0
    weights = reference[positive]
    return max(float(np.dot(weights, np.log(weights) - log_probabilities[positive])), 0.)


def compare(probes, case, scales):
    _, row, origin = case
    trial = probes.trial(row['teacher'])
    reference = probes.run(trial, origin, row, REF_N, probes.ref_scale)
    if reference is None or reference['fallback_played']:
        return None
    ids = reference['legal_ids']
    move = ids.index(reference['played_action'])
    reference_target = np.asarray(reference['target'], dtype=np.float64)
    reference_target = reference_target / reference_target.sum()
    label = int(np.argmax(reference_target))
    logits = np.asarray(reference['logits'], dtype=np.float64)
    policy = int(np.argmax(logits))
    log_policy = logits - np.logaddexp.reduce(logits)
    comparisons = [(target_kl(reference_target, log_policy), policy == move, policy == label, float(-log_policy[move]))]
    for scale in scales:
        candidate = probes.run(trial, origin, row, LONG_N, scale)
        if candidate is None:
            return None
        # Keep NLL in log space using this candidate's own search statistics.
        multiplier = candidate['c2'] * (candidate['c1'] + max(candidate['counts']))
        scores = np.asarray(candidate['logits'], dtype=np.float64) + multiplier * completed_q(candidate)
        log_scores = scores - np.logaddexp.reduce(scores)
        target = np.asarray(candidate['target'], dtype=np.float64)
        # Softmax gives every action mass. Use the log score when a tiny probability rounds to zero on export.
        log_target = np.log(target / target.sum(), out=log_scores.copy(), where=target > 0)
        comparisons.append((target_kl(reference_target, log_target),
                            candidate['played_action'] == reference['played_action'],
                            int(np.argmax(target)) == label, float(-log_scores[move])))
    return comparisons


def summarize(valid, scales):
    rows, lines = [], []
    if valid:
        names = ('policy', *(f'{scale:.2f}' for scale in scales))
        for name, (kl, move, label, nll) in zip(names, np.mean(valid, axis=0).tolist(), strict=True):
            rows.append(dict(short_scale=name, target_KL=kl, match_long_move=move, match_long_label=label, move_NLL=nll))
            lines.append(f'{name:>11}  {kl:>9.6f}  {move:>15.1%}  {label:>16.1%}  {nll:>8.4f}')
    return dict(header=HEADER, lines=lines, rows=rows)


if __name__ == '__main__':
    main('long', SCALES, compare, summarize, candidate_n=LONG_N, default_targets=TARGETS)
