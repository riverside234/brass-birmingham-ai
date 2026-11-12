"""Iteration-level teacher gradient balance using the existing sampled norms."""
import math
import statistics

from brass_ai.learning.contracts import require
from .teacher_gradient_qualification import fixed_beta, full_tables

KEY = 'adaptive_beta'
REMEASUREMENT = 'beta_remeasurement'
RHO = .9
ETA = .1
EPSILON = 1e-12


def settings():
    return dict(contract='adaptive-teacher-beta/1.0.0', rho=RHO, eta=ETA, epsilon=EPSILON)


def pending(weight):
    request = weight.get(REMEASUREMENT)
    return request is not None and 'measured_generation' not in request


def initial(beta, basis):
    return dict(beta=beta, actor_ema=None, teacher_ema=None, updates=0, basis=basis)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def restore(saved, beta, basis, *, prior_basis=None):
    if saved is None or (saved['basis'] != basis and
                         (prior_basis is None or saved['basis'] != prior_basis)):
        return initial(beta, basis)
    require(set(saved) == set(initial(beta, basis)) and positive(saved['beta']) and .01 <= saved['beta'] <= 10
            and type(saved['updates']) is int and saved['updates'] >= 0 and (
                positive(saved['actor_ema']) and positive(saved['teacher_ema']) if saved['updates'] else
                saved['actor_ema'] is None and saved['teacher_ema'] is None), 'Invalid adaptive teacher beta state')
    return saved if saved['basis'] == basis else dict(saved, basis=basis)


def paired_norms(steps):
    pairs = [(s.get('actor_norm'), s.get('raw_teacher_norm')) for s in steps]
    return [(a, t) for a, t in pairs if positive(a) and positive(t)]


def advance(state, actor_mean, teacher_mean, target):
    if not (positive(actor_mean) and positive(teacher_mean)):
        return state
    actor, teacher = actor_mean, teacher_mean
    if state['updates']:
        actor = RHO * state['actor_ema'] + (1 - RHO) * actor
        teacher = RHO * state['teacher_ema'] + (1 - RHO) * teacher
    desired = fixed_beta(target, (teacher + EPSILON) / actor)
    beta = (1 - ETA) * state['beta'] + ETA * desired
    return dict(state, beta=max(.01, min(10., beta)), actor_ema=actor, teacher_ema=teacher, updates=state['updates'] + 1)


def remeasure(term, pairs, request):
    """Accept one full teacher-on observation. The historical activation stays valid."""
    if not pairs or not full_tables(term):
        return request
    target = term.config['term']['share_target']
    raw_median = statistics.median(t / a for a, t in pairs)
    context = term.qualification_context
    return dict(request, measured_generation=term.generation, valid_steps=len(pairs), raw_median=raw_median,
                target=target, beta=fixed_beta(target, raw_median), model_id=context.get('model_id'),
                export_hash=context.get('export_hash'))


def accepted(qualification, request, target):
    if request is not None and 'measured_generation' in request:
        require(positive(request['raw_median']) and request['beta'] == fixed_beta(request['target'], request['raw_median']),
                'Invalid teacher-on beta measurement')
        # An explicit share-target edit uses the same measured ratio at the new target.
        return fixed_beta(target, request['raw_median'])
    return qualification
