"""Search-v1 completed-trajectory labels, independent of played-move support.

B04 supplies these records. No legacy visit target can be relabelled here.
The caller supplies the frozen job identity, never identity from the reply.
"""
import copy
from dataclasses import dataclass

import numpy as np

from brass_ai.learning.contracts import digest, require
from .teacher_phase_config import ALGORITHM, SESSION, LABEL_SOURCE

PHASES = ('canal_early', 'canal_late', 'rail_early', 'rail_late')
KINDS = ('root', 'continuation', 'replanned_root')
IDENTITY = ('job_id', 'game_id', 'generation', 'model_id', 'export_hash', 'reward_id',
            'source_fingerprint', 'epoch')


def vector(value, size, name):
    require(isinstance(value, (list, tuple, np.ndarray)), 'Invalid teacher ' + name)
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == (size,) and np.isfinite(result).all(), 'Non-finite/invalid teacher ' + name)
    return result


def softmax(values):
    a = np.asarray(values, dtype=np.float64)
    e = np.exp(a - np.max(a))
    return e / e.sum()


def improved_policy(logits, counts, means, value, c1=50, c2=.1):
    """Float64 oracle; visited-prior weighting, raw utility, no Gumbel draws."""
    size = len(logits)
    require(size > 0, 'Empty legal teacher set')
    l, n, q = (vector(x, size, name) for x, name in
               ((logits, 'logits'), (counts, 'counts'), (means, 'means')))
    require(np.equal(n, np.floor(n)).all() and (n >= 0).all(), 'Invalid teacher counts')
    require(type(value) in (int, float) and np.isfinite(value), 'Non-finite node value')
    require(c1 == 50 and c2 == .1 and type(c1) is not bool and type(c2) is not bool,
            'Teacher search constants differ')
    visited = n > 0; total = n.sum()
    # Restricted softmax also handles underflow when the visited prior mass is tiny.
    mixed = (value + total * np.dot(softmax(l[visited]), q[visited])) / (1 + total) if total else value
    completed = np.where(visited, q, mixed)
    target = softmax(l + c2 * (c1 + n.max()) * completed)
    return target, float(mixed)


def validate_improved_policy(label, choices):
    """Validate mathematical/ordered-legal contract; return None for no evidence."""
    require(isinstance(label, dict) and label.get('source') == LABEL_SOURCE
            and label.get('method') == 'search_v1' and label.get('algorithm') == ALGORITHM
            and label.get('session') == SESSION and label.get('v205_enabled') is False,
            'Unknown teacher label/search source')
    ids = [c['id'] for c in choices]
    require(ids and len(set(ids)) == len(ids) and label['legal_ids'] == ids, 'Teacher legal order differs')
    require(label['kind'] in KINDS and label['phase'] in PHASES, 'Unknown teacher decision kind/phase')
    require(all(type(n) is int and n >= 0 for n in label['counts']), 'Invalid teacher integer counts')
    target, mixed = improved_policy(label['logits'], label['counts'], label['mean_returns'],
                                   label['node_value'], label['c1'], label['c2'])
    exported = vector(label['target'], len(ids), 'target')
    require((exported >= 0).all() and abs(exported.sum() - 1) <= 1e-6,
            'Teacher target is not a probability distribution')
    require(np.max(np.abs(target - exported)) <= 1e-5, 'Teacher target recomputation differs')
    require(np.isfinite(label['v_mix']) and abs(label['v_mix'] - mixed) <= 1e-5,
            'Teacher mixed value differs')
    require(label['max_n'] == max(label['counts']) and type(label['max_n']) is int,
            'Teacher max count differs')
    for key in ('requested_k', 'effective_k', 'frozen_n', 'completed_trajectories',
                'node_trajectories', 'completed_rounds'):
        require(type(label[key]) is int and label[key] >= 0, 'Invalid teacher search counter ' + key)
    root_width = label.get('root_legal_count',len(ids)) if label['kind'] == 'continuation' else len(ids)
    require(type(root_width) is int and root_width >= 1 and
            label['requested_k'] == 16 and label['effective_k'] in (1, 2, 4, 8, 16)
            and label['effective_k'] <= root_width,
            'Teacher K differs')
    require(sum(label['counts']) == label['node_trajectories'] <= label['completed_trajectories'],
            'Teacher completed watermark differs')
    require(label['completed_trajectories'] <= label['frozen_n'] <= 1_000_000,
            'Teacher evidence exceeds its frozen quota')
    require(type(label['overrun_seconds']) in (int, float) and np.isfinite(label['overrun_seconds'])
            and label['overrun_seconds'] >= 0, 'Invalid teacher overrun')
    for key in ('deadline_hit', 'plan_supported', 'fallback_played', 'cancelled'):
        require(type(label[key]) is bool, 'Invalid teacher provenance ' + key)
    require(label['played_action'] in ids and isinstance(label['gumbel_draw_id'], str)
            and bool(label['gumbel_draw_id']), 'Missing teacher played-action provenance')
    return None if len(ids) == 1 or not sum(label['counts']) or label['cancelled'] else target


@dataclass(frozen=True)
class LabelRow:
    row: dict
    encoded: object

    @property
    def label(self): return self.row['teacher']

    @property
    def identity(self):
        return digest({k: self.label[k] for k in (*IDENTITY, 'seat', 'turn_key', 'revision', 'kind')})


def validate_row(row, expected, history, admitted_sources):
    """Bind a reply to an independently frozen job and actual observer/decision."""
    label, view = row['teacher'], row['view']; context = view['context']
    require(digest({k: label.get(k) for k in IDENTITY}) == digest({k: expected[k] for k in IDENTITY}),
            'Stale/wrong teacher job identity')
    require(any(e['generation'] == label['generation'] and e['model_id'] == label['model_id']
                for e in history), 'Teacher collection model not in run lineage')
    require(label['source_fingerprint'] in admitted_sources, 'Unadmitted teacher source fingerprint')
    require(label['seat'] == row['seat'] == view['seat'] == context['actor'], 'Teacher observer mismatch')
    require(type(label['seat']) is int and 0 <= label['seat'] < 4 and context['phase'] == 'action'
            and not context['terminal'], 'Teacher row is not an acting action-stage decision')
    require(type(label['generation']) is int and label['generation'] >= 0
            and type(label['revision']) is int and label['revision'] >= 0, 'Invalid teacher generation/revision')
    require(label['kind'] == 'root' or context['actions_remaining'] == 1,
            'Teacher second-action label has wrong successor')
    require(label['revision'] == context['revision'] and label['turn_key'] ==
            [context[k] for k in ('era', 'round', 'turn', 'turn_owner')], 'Teacher decision revision differs')
    phase = ('canal_' + ('early' if context['round'] <= 4 else 'late') if context['era'] == 'CanalEra'
             else 'rail_' + ('early' if context['round'] <= 6 else 'late'))
    require(label['phase'] == phase, 'Teacher phase differs from observation')
    require(row.get('style', [0.] * 4) == [0.] * 4 and row.get('intent', [0.] * 8) == [0.] * 8,
            'Teacher policy requires neutral context')
    return validate_improved_policy(label, row['choices'])


def admit_rows(rows, *, model, expected, history, admitted_sources):
    """New-generation rows only; old sealed rows are checked by buffer receipts."""
    from brass_ai.learning.batching import model_id
    from .encoding import PreparedEncoder
    from .teacher_term import policy_logits
    require(model_id(model) == expected['model_id'], 'Teacher admission model differs')
    encoder = PreparedEncoder(); accepted = []; seen = set(); rejected = dict(no_evidence=0, forced=0, cancelled=0)
    for raw in rows:
        row = copy.deepcopy(raw)
        target = validate_row(row, expected, history, admitted_sources)
        if target is None:
            reason = 'cancelled' if row['teacher']['cancelled'] else 'forced' if len(row['choices']) == 1 else 'no_evidence'
            rejected[reason] += 1
            continue
        value = LabelRow(row, encoder.encode(row['view'], row['choices'], [0.] * 4, [0.] * 8))
        require(value.identity not in seen, 'Duplicate teacher decision')
        seen.add(value.identity); accepted.append(value)
    actual = policy_logits(model, accepted)
    for row, logits in zip(accepted, actual):
        require(np.allclose(logits, row.label['logits'], atol=2e-5, rtol=1e-5),
                'Teacher logits parity failed')
    return accepted, rejected
