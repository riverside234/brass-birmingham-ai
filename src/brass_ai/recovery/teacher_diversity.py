"""Immutable contracts for the optional DOMiNO search teacher.

Dynamic evaluator identity never participates in the four-age buffer key.
This module owns no optimizer, ordinary RNG, or mutable diversity controller.
"""
import copy
import random

import numpy as np

from brass_ai.learning.contracts import digest, require
from .domino_config import ENCODING, mapping

CONTRACT = 'teacher-phase/2.0.0'
MIGRATION = 'teacher-domino-adoption/1.0.0'
LABEL = 'search-v1-domino-improved-policy/1.0.0'
ALGORITHM = 'brass-gumbel-is-mcts/domino-count-v1'
SESSION = 'brass-gumbel-is-mcts/domino-session-v1'
EVALUATOR = 'teacher-domino-evaluator/1.0.0'
ASSIGNMENT = 'teacher-uniform-seven-seat-game/1.0.0'
NORMALIZATION = 'four-player-phase-actions-7-8-12-4/1.0.0'
COUNTS = (7, 8, 12, 4)


def options():
    return dict(contract='teacher-domino-search/1.0.0', assignment=ASSIGNMENT,
        context_encoding=ENCODING, objective='domino-retained-fit-direction',
        coefficient_source='current-committed-domino', opponent_model='root-latent-self-play',
        reward_owner='root-only', intrinsic_leaf='bounded-root-owner-bootstrap/1.0.0',
        descriptor_input='known-root-self-unknown-task', normalization=NORMALIZATION,
        missing_readiness='defer-collection')


def enabled(config):
    return config.get('teacher_phase', {}).get('contract') == CONTRACT


def adopt(config):
    from .teacher_phase_config import defaults, validate
    from .utility import reward_spec
    from .domino_config import validate as validate_domino
    result = copy.deepcopy(config)
    require('diversity' in result, 'Diversity teacher requires existing DOMiNO training')
    validate_domino(result['diversity'], result)
    require(result['diversity']['mode'] == 'apply', 'Diversity teacher requires DOMiNO apply')
    require(result.get('teacher_phase', {}).get('mode', 'off') == 'off',
            'Adopt diversity teacher only from off at a committed boundary')
    if enabled(result):
        validate(result)
        return result
    result['teacher_phase'] = defaults(reward_spec(result), diversity=True)
    from .config import teacher_value_config
    result['teacher_value'] = dict(teacher_value_config(result.get('teacher_value')), enabled=False)
    validate(result)
    return result


def assignments(seed_identity, *, run_seed, model_id, export_hash):
    """Independent physical-seat draws; no ordinary-game cursor or global RNG."""
    result = []
    for seat in range(4):
        key = dict(contract=ASSIGNMENT, run_seed=run_seed, seed_identity=seed_identity, seat=seat)
        z = random.Random(digest(key)).randrange(7)
        context = mapping()[z]
        value = dict(contract=ASSIGNMENT, scope='teacher-only', seed=digest(key),
            strategy_id=z, seat=seat, style=context['style'], intent=context['intent'],
            context_encoding=ENCODING, model_id=model_id, export_hash=export_hash, probability=1/7)
        result.append(dict(value, identity=digest(value)))
    return result


def check_context(receipt):
    require(type(receipt) is dict and receipt.get('contract') == ASSIGNMENT
            and receipt.get('scope') == 'teacher-only'
            and receipt.get('context_encoding') == ENCODING
            and type(receipt.get('strategy_id')) is int and 0 <= receipt['strategy_id'] < 7
            and type(receipt.get('seat')) is int and 0 <= receipt['seat'] < 4,
            'Invalid teacher context receipt')
    c = mapping()[receipt['strategy_id']]
    require(digest(receipt['style']) == digest(c['style']) and digest(receipt['intent']) == digest(c['intent'])
            and receipt['probability'] == 1/7 and receipt['identity'] == digest({k:v for k,v in receipt.items() if k != 'identity'}),
            'Changed teacher assignment/context')
    return receipt


def validate_bundle(bundle):
    from .domino_features import manifest, width
    from .domino_search_fit import validate as validate_fit
    require(bundle['contract'] == EVALUATOR and bundle['identity'] == digest({k:v for k,v in bundle.items() if k != 'identity'}),
            'Changed diversity evaluator identity')
    require(bundle['normalization'] == NORMALIZATION and bundle['descriptor'] == digest(manifest())
            and bundle['context_encoding'] == ENCODING, 'Diversity evaluator contracts differ')
    fit = validate_fit(bundle['fit'])
    require(fit['regime'] == bundle['regime'] and fit['descriptor'] == bundle['descriptor']
            and fit['normalization'] == bundle['normalization'], 'Intrinsic fitting regime differs')
    d = np.asarray(bundle['directions'], dtype=np.float64)
    c = np.asarray(bundle['coefficients'], dtype=np.float64)
    require(d.shape == (7, 4*width()) and np.isfinite(d).all() and np.array_equal(d, fit['directions'])
            and (d[0] == 0).all(), 'Search direction is not the fitted direction')
    require(c.shape == (7,) and np.isfinite(c).all() and c[0] == 1 and ((c > 0)&(c <= 1)).all(),
            'Invalid diversity coefficients')
    require(len(bundle['fingerprints']) == 7, 'Missing root-self fingerprints')
    from brass_ai.learning.seat_descriptors import descriptor
    for z, fp in enumerate(bundle['fingerprints']):
        require(fp['context_identity']['model_id'] == bundle['model_id']
                and fp['context_identity']['strategy_id'] == z, 'Fingerprint belongs to another model/context')
        descriptor('learner', 0, fp['values'])
    return bundle


def freeze(model, diversity, fingerprints, *, export_hash, generation, checkpoint=None):
    """Return (bundle, readiness); missing optional old fit defers, corruption fails."""
    from brass_ai.learning.batching import model_id
    from . import domino_state as S
    from .domino_search_fit import validate as validate_fit
    from .local_scoring_auxiliary import bundle_id
    if diversity is None:
        raise ValueError('Missing DOMiNO companion for diversity teacher')
    diversity.validate_tensors()
    require(diversity.pending is None and diversity.state['generation'] == generation,
            'Freeze teacher evaluator only at committed DOMiNO boundary')
    if diversity.config['mode'] != 'apply' or S.readiness(diversity.state, diversity.config) != 'ready':
        return None, dict(ready=False, reason='domino_evidence_not_ready')
    fit = diversity.search_fit
    if fit is None:
        return None, dict(ready=False, reason='missing_retained_intrinsic_fit', generation=generation)
    validate_fit(fit)
    require(fit['checkpoint_generation']<=generation,'Intrinsic fit is from a future checkpoint')
    if not fit['normalization_proven']:
        return None, dict(ready=False,reason='unsupported_fit_phase_normalization')
    if fit['regime'] != diversity.state['regime']:
        return None, dict(ready=False, reason='retained_fit_regime_changed')
    missing = [z for z in range(1,7) if fit['support'].get(str(z), {}).get('rows', 0) == 0]
    if missing:
        return None, dict(ready=False, reason='retained_fit_strategy_support', missing_strategies=missing)
    require(fingerprints is not None, 'Missing frozen current-model fingerprints')
    mid = model_id(model)
    if checkpoint is not None:
        require(checkpoint['model_id']==mid and checkpoint['sha256']==export_hash
                and checkpoint['generation']==generation,'Evaluator checkpoint binding differs')
    fps = [copy.deepcopy(fingerprints.get(mid,z)) for z in range(7)]
    direction, nearest, _ = S.directions(diversity.state['centroids'])
    fitted_direction=np.asarray(fit['directions']);fitted_norm=np.linalg.norm(fitted_direction,axis=1)
    current_norm=np.linalg.norm(direction,axis=1)
    cosine=np.divide((direction*fitted_direction).sum(1),current_norm*fitted_norm,
                     out=np.zeros(7),where=current_norm*fitted_norm>0)
    from brass_ai.learning.seat_descriptors import descriptor
    drift={str(z):[float(np.linalg.norm(np.asarray(old)-descriptor('learner',0,fps[z]['values'])))
                   for old in fit['support'].get(str(z),{}).get('input_descriptors',[])] for z in range(7)}
    value = dict(contract=EVALUATOR, generation=generation, model_id=mid, export_hash=export_hash,
        head_id=bundle_id(diversity.head.state_dict()), fit=copy.deepcopy(fit), directions=copy.deepcopy(fit['directions']),
        coefficients=S.coefficients(diversity.state,diversity.config).tolist(),
        descriptor=diversity.config['descriptor_manifest_sha256'], context_encoding=ENCODING,
        normalization=NORMALIZATION, regime=diversity.state['regime'], fingerprints=fps,
        fit_lag=generation-fit['checkpoint_generation'], current_directions=direction.tolist(), current_nearest=nearest.tolist(),
        direction_diagnostics=dict(fitted_norm=fitted_norm.tolist(),current_norm=current_norm.tolist(),
                                   angle_radians=np.arccos(np.clip(cosine,-1,1)).tolist()),
        self_descriptor_drift_l2=drift,
        heldout=copy.deepcopy(diversity.fit_diagnostics),checkpoint=copy.deepcopy(checkpoint))
    bundle = dict(value, identity=digest(value))
    validate_bundle(bundle)
    return bundle, dict(ready=True, reason='retained_fit_available', bundle_id=bundle['identity'],
                       fit_lag=bundle['fit_lag'], quality='Estimator quality still requires bounded lab qualification')


def validate_channels(label):
    n = len(label['counts']); c = label['coefficient']
    require(label['normalization']==NORMALIZATION and type(label['intrinsic_prefix']) in (float,int)
            and np.isfinite(label['intrinsic_prefix']), 'Invalid intrinsic prefix/normalization')
    require(type(c) in (int,float) and np.isfinite(c) and 0 < c <= 1, 'Invalid teacher mixing coefficient')
    require(label['root_owner'] == label['seat'] and label['opponent_model'] == 'root-latent-self-play',
            'Wrong diversity reward owner/opponent model')
    require(label['channel_counts'] == label['counts'] and label['channel_watermark'] == label['completed_trajectories'],
            'Incomplete multi-channel teacher evidence')
    for task, intrinsic, combined, size in (
        (label['task_mean_returns'],label['intrinsic_mean_returns'],label['mean_returns'],n),
        ([label['task_node_value']],[label['intrinsic_node_value']],[label['node_value']],1)):
        a,b,y = (np.asarray(v,dtype=np.float64) for v in (task,intrinsic,combined))
        require(all(v.shape == (size,) and np.isfinite(v).all() for v in (a,b,y))
                and np.allclose(c*a+(1-c)*b,y,atol=1e-10,rtol=1e-10), 'Teacher return channels do not recombine')
    if label['strategy_id'] == 0:
        require(c == 1 and label['intrinsic_node_value'] == 0 and not any(label['intrinsic_mean_returns']),
                'Neutral teacher must bypass intrinsic work')


def buffer_diagnostics(aged_rows, receipts, current):
    """Describe frozen objective drift without altering admission or targets."""
    from .teacher_term import summary
    bundles={k:v for r in receipts for k,v in r.get('evaluators',{}).items()}
    out={}
    for age in sorted({a for a,_ in aged_rows}):
        by_strategy={}
        for z in range(7):
            rows=[r.label for a,r in aged_rows if a==age and r.label.get('strategy_id')==z]
            if not rows:continue
            old=[bundles[r['bundle_id']] for r in rows]
            angles=[];norms=[];coefficients=[]
            if current is not None:
                v=np.asarray(current['directions'][z]);nv=np.linalg.norm(v)
                for b in old:
                    u=np.asarray(b['directions'][z]);nu=np.linalg.norm(u)
                    angles.append(float(np.arccos(np.clip(np.dot(u,v)/(nu*nv),-1,1))) if nu*nv>0 else None)
                    norms.append(float(nv-nu));coefficients.append(current['coefficients'][z]-b['coefficients'][z])
            by_strategy[str(z)]=dict(rows=len(rows),unique_games=len({r['game_id'] for r in rows}),
                fit_lag=summary([b['fit_lag'] for b in old]),direction_angle=summary(angles),
                direction_norm_change=summary(norms),coefficient_change=summary(coefficients))
        out[str(age)]=by_strategy
    return out
