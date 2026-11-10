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
# c2 is the Gumbel c_scale. The qualified GPU binaries only ever produce .1.
C_SCALES = (.1, 1.)
# A short-search label carries profile "fast"; long labels have no profile field.
SHORT_K = (2, 4, 8, 16)


def label_profile(label):
    """'full' (long search) or 'fast' (short search)."""
    profile = label.get('profile', 'full')
    require(profile in ('full', 'fast') and ('profile' not in label or profile == 'fast'),
            'Unknown teacher label profile')
    return profile


def check_turn(label, row_round, job):
    """Long labels belong to the reserved turns; short labels only to the other turns of a g1491 job."""
    reserved = row_round == job['reservations'][str(label['seat'])][label['phase']]
    if label.get('cotton_exploration') is not None:
        from .teacher_cotton import enabled
        require(job.get('split')=='train' and enabled(job), 'Cotton exploration outside the training recipe')
    if label_profile(label) == 'fast':
        require((job.get('teacher_revision') or {}).get('short_labels') is True,
                'Short-search labels need the g1491 teacher revision')
        require(not reserved, 'Short-search label on a reserved long-search turn')
    else:
        require(reserved, 'Teacher label outside reserved turn')


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
    require(c1 == 50 and c2 in C_SCALES and type(c1) is not bool and type(c2) is not bool,
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
    # Long searches always ask for K=16. A short-search label (g1491) records its own K.
    profile = label_profile(label)
    requested = 16 if profile == 'full' else label['requested_k']
    root_width = label.get('root_legal_count',len(ids)) if label['kind'] == 'continuation' else len(ids)
    require(type(root_width) is int and root_width >= 1 and requested in SHORT_K and
            label['requested_k'] == requested and label['effective_k'] in (1, 2, 4, 8, 16)
            and label['effective_k'] <= min(root_width, requested),
            'Teacher K differs')
    require(type(label.get('explored', False)) is bool, 'Invalid teacher exploration flag')
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
    if 'cotton_exploration' in label:
        from .teacher_cotton import validate_label
        validate_label(label, choices)
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
    require(model_id(model) == expected['model_id'], 'Teacher admission model differs')
    accepted, rejected = prepare_rows(rows, expected=expected, history=history,
                                      admitted_sources=admitted_sources, encoder=PreparedEncoder())
    check_logits(model, accepted)
    return accepted, rejected


def prepare_rows(rows, *, expected, history, admitted_sources, encoder):
    """Independent row/target/schema/feature admission, without model mutation."""
    accepted = []; seen = set(); rejected = dict(no_evidence=0, forced=0, cancelled=0)
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
    return accepted, rejected


def logit_tolerance(label):
    """ONNX Runtime and PyTorch differ by up to about 5e-5 at float32.

    CPU labels use the established cross-runtime export tolerance. Labels from
    the GPU service keep their original bound.
    """
    backend = label.get('inference_backend')
    require(backend in (None, 'cpu-ort'), 'Unknown teacher inference backend')
    return (2e-4, 2e-4) if backend == 'cpu-ort' else (2e-5, 1e-5)


def check_logits(model, accepted):
    from .teacher_term import policy_logits
    actual = policy_logits(model, accepted) if accepted else []
    for row, logits in zip(accepted, actual):
        atol, rtol = logit_tolerance(row.label)
        require(np.allclose(logits, row.label['logits'], atol=atol, rtol=rtol),
                'Teacher logits parity failed')
