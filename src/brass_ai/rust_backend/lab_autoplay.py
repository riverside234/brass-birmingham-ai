"""Build step 9: the CPU collector plays a whole ordinary game, sampler included.

The worker replays ``NeuralPolicy.sample_training`` (exploration version 3)
with each seat's own ``random.Random`` stream. Python hands the four Mersenne
Twister states over once and takes them back at the end, so no RNG draw moves.
Rows, features and seals still come from the collector's append path, and the
publisher replays the draws of audited seats exactly as for the Python loop:
cotton seats on every game, all four seats on the semantic audit games.

A game keeps the Python loop when the worker does not reproduce it: branching,
non-neural or non-v3 seats, GPU inference, strict validation or diagnostic
hooks. ``BRASS_ORDINARY_AUTOPLAY=0`` keeps every game on the Python loop.
"""
import os
import time

from brass_ai.learning.contracts import require

CONTRACT = 'ordinary-autoplay/1.0.0'
SWITCH = 'BRASS_ORDINARY_AUTOPLAY'
# A whole game per call; cancellation still kills the collector at once.
TIMEOUT = 600.


def enabled():
    value = os.environ.get(SWITCH, '1')
    require(value in ('0', '1'), 'Invalid '+SWITCH)
    return value == '1'


def _opening(opening):
    if opening is None:
        return None
    require(opening['action_decisions_per_seat'] >= 0 and opening['temperature'] > 0, 'Invalid opening temperature')
    return dict(temperature=float(opening['temperature']), action_decisions_per_seat=int(opening['action_decisions_per_seat']))


def request(job, assigned, rngs, cotton, lane, *, continuation, compare, raw_export, validation, diagnostic):
    """Seat settings for the worker, or (None, reason) when the game keeps the Python loop."""
    from brass_ai.recovery import ordinary_cotton
    from brass_ai.step5.policy import NeuralPolicy
    from .lab_cpu_inference import NativeCpuEvaluator
    if not enabled():
        return None, 'disabled'
    if (getattr(lane, 'hello', None) or {}).get('autoplay') != CONTRACT:
        return None, 'collector'
    if continuation is not None or job.get('branching') or compare is not None or not raw_export or diagnostic:
        return None, 'route'
    if validation is None or validation.get('mode') == 'strict':
        return None, 'strict'
    seats = []
    for p, rng in zip(assigned, rngs, strict=True):
        if (type(p) is not NeuralPolicy or p.exploration_version != 3 or type(p.evaluator) is not NativeCpuEvaluator
                or p.evaluator.lane is not lane or p.request_deadline is not None or rng in p.action_counts):
            return None, 'policy'
        if p.focus == 'mixed' and rng not in p.contexts:
            return None, 'context'  # The mixed cycle would draw its offset from this RNG.
        style, intent = p.context_features(None, rng)  # Preset or fixed focus: no draw.
        version, state, gauss = rng.getstate()
        require(version == 3 and len(state) == 625 and gauss is None, 'Unexpected seat RNG state')
        seats.append(dict(model_id=p.evaluator.model_id, policy_id=p.policy_id, exploration=float(p.exploration),
                          temperature=float(p.temperature), opening=_opening(p.opening),
                          style=[float(x) for x in style], intent=[float(x) for x in intent],
                          rng=list(state), action_count=0))
    table = None
    if cotton.seats:
        table = dict(contract=job['provenance'][ordinary_cotton.KEY]['contract'], seats=list(cotton.seats),
                     probability=float(cotton.probability))
    return dict(contract=CONTRACT, seats=seats, cotton=table), None


def play(lane, settings, rngs, timings):
    """Run the game in the worker and restore each seat's RNG to where the loop would leave it."""
    began = time.monotonic()
    reply = lane.autoplay(settings, timeout=TIMEOUT)
    elapsed = time.monotonic()-began
    require(reply['wire'] == CONTRACT and reply['token'] == lane.token and type(reply['rows']) is int and
            reply['rows'] > 0 and len(reply['rng']) == len(rngs) == 4 and
            0 <= reply['forced_rows'] <= reply['rows'] and reply['inferences'] == reply['rows']-reply['forced_rows'],
            'Autoplay reply differs')
    for rng, state in zip(rngs, reply['rng'], strict=True):
        require(type(state) is list and len(state) == 625, 'Autoplay RNG state differs')
        rng.setstate((3, tuple(state), None))
    worker = reply['elapsed_ns']/1e9; policy = reply['policy_ns']/1e9
    timings['autoplay_games'] += 1
    timings['autoplay_seconds'] += elapsed
    timings['autoplay_worker_seconds'] += worker
    timings['autoplay_decisions'] += reply['rows']
    timings['autoplay_forced_decisions'] += reply['forced_rows']
    timings['autoplay_inferences'] += reply['inferences']
    timings['policy_seconds'] += policy
    timings['native_step_seconds'] += max(0., worker-policy)
    lane.metrics['cpu_forward_ns'] += reply['forward_ns']
    return reply
