"""Snapshot-critic weight evidence only. GAE keeps its existing EV gates."""
import copy

from brass_ai.learning.contracts import digest, require
from .ev_evidence import TABLES, finite, validate_statistics

CONTRACT = 'snapshot-weight-r2/2.0.0'
WINDOW = 20
MIN_ITERATIONS = 10
MIN_GAMES = 8
MIN_TABLE_GAMES = 2
HEALTHY_ITERATIONS = 5
WEIGHTS = (.5, .75, 1.)


def pool(parts):
    """Merge target central moments and *uncentered* squared errors in float64."""
    n = 0
    mean = m2 = error_sum = error_square_sum = 0.
    ids = set()
    for part in parts:
        validate_statistics(part)
        require(ids.isdisjoint(part['game_ids']), 'Repeated game in snapshot R2 evidence')
        require(not any('/branch/' in g for g in part['game_ids']), 'Branch in snapshot R2 evidence')
        ids.update(part['game_ids'])
        count = part['rows']
        total = n + count
        delta = part['target_mean'] - mean
        m2 += part['target_m2'] + delta * delta * n * count / total
        mean += delta * count / total
        error_sum += part['error_mean'] * count
        error_square_sum += part['error_m2'] + count * part['error_mean'] ** 2
        n = total
    variance = m2 / n if n else None
    mse = error_square_sum / n if n else None
    bias = error_sum / n if n else None
    r2 = 1 - mse / variance if n and variance > 1e-12 else None
    require(all(v is None or finite(v) for v in (variance, mse, bias, r2)),
            'Nonfinite snapshot R2 aggregate')
    return dict(rows=n, games=len(ids), r2=r2, learner_mse=mse,
                prediction_bias=bias, target_variance=variance)


def regime(value, update, exploration=None):
    """Effective training/measurement regime, excluding counters and changing LR."""
    split = value.get('split', {})
    holdout = {k: copy.deepcopy(split[k]) for k in
               ('contract', 'fraction', 'minimum_games_per_table', 'reference_games', 'games_per_table')
               if k in split}
    gae = update.get('gae', {})
    exploration = exploration or {}
    return dict(source='ppg' if value.get('enabled', True) else 'ppo',
        value_contract=value.get('contract'), holdout=holdout,
        ppg={k: value.get(k) for k in ('configured_learning_rate', 'kl_cap', 'acceptance_steps')}
            if value.get('enabled', True) else None,
        lambda_by_table=copy.deepcopy(gae.get('lambda_by_table', {'selfplay': gae.get('lambda', 1.)})),
        reward=copy.deepcopy(update.get('reward', {}).get('spec', update.get('reward_spec'))),
        tables={k: v['games'] for k, v in split.get('by_table', {}).items()},
        exploration={k: copy.deepcopy(exploration[k]) for k in
                     ('version', 'opening', 'branching', 'branching_status') if k in exploration})


def entry(generation, report, *, healthy, trained):
    """Only the three ordinary learner groups; branches/snapshot groups never enter."""
    groups = report.get('heldout_after', {}).get('groups', {})
    parts = {kind: copy.deepcopy(groups.get('table:' + kind + '/role:learner', {}).get('ev_evidence'))
             for kind in TABLES}
    split = report.get('split', {})
    minima = max(MIN_TABLE_GAMES, split.get('minimum_games_per_table', MIN_TABLE_GAMES))
    supported = True
    for kind, stats in parts.items():
        if stats is not None:
            validate_statistics(stats)
            # Evidence must be from actually held-out whole games.
            require(not any('/branch/' in g for g in stats['game_ids']), 'Branch in snapshot R2 evidence')
            selection = split.get('selection')
            if selection is not None:
                require(set(stats['game_ids']) <= set(selection['heldout_games']) and
                    all(selection['games'].get(g) == kind for g in stats['game_ids']),
                    'Snapshot R2 game is outside its whole-game holdout')
        supported &= stats is not None and len(stats['game_ids']) >= minima
    aggregate = pool([p for p in parts.values() if p is not None])
    # `trained` is bound to retained PPO snapshot rows or accepted PPG rows.
    # A rejected/empty PPG phase must not erase real PPO snapshot work.
    qualified = bool(trained and supported
                     and aggregate['games'] >= MIN_GAMES and finite(aggregate['r2']))
    return dict(generation=generation, healthy=bool(healthy), qualified=qualified, tables=parts)


def summarize(entries):
    selected = [e for e in entries if e['qualified']]
    result = pool([p for e in selected for p in e['tables'].values()])
    result.update(contributing_generations=[e['generation'] for e in selected],
        qualifying_iterations=len(selected),
        games_by_table={k: sum(len(e['tables'][k]['game_ids']) for e in selected) for k in TABLES})
    result['supported'] = (len(selected) >= MIN_ITERATIONS and result['games'] >= MIN_GAMES
                           and all(n >= MIN_TABLE_GAMES for n in result['games_by_table'].values())
                           and finite(result['r2']))
    return result


def initial(generation, weight, legacy_ev=None):
    return dict(contract=CONTRACT, origin=generation, healthy_streak=0,
        last_change_generation=None, stage_origin=generation, stage_weight=weight,
        regime=None, entries=[], reference=None, legacy_ev_baseline=legacy_ev)


def validate(r, generation, weight):
    require(type(r) is dict and set(r) == set(initial(0, .5)) and r['contract'] == CONTRACT,
            'Invalid snapshot R2 controller checkpoint')
    for key in ('origin', 'stage_origin'):
        require(type(r[key]) is int and 0 <= r[key] <= generation, 'Invalid snapshot R2 origin')
    require(r['stage_weight'] == weight and weight in WEIGHTS, 'Snapshot R2 weight stage differs')
    require(r['regime'] is None or type(r['regime']) is dict, 'Invalid snapshot R2 regime')
    require(r['legacy_ev_baseline'] is None or finite(r['legacy_ev_baseline']), 'Invalid archived snapshot EV')
    last = r['last_change_generation']
    require(last is None or type(last) is int and 0 <= last <= r['stage_origin'],
            'Invalid snapshot R2 weight change age')
    start = max(r['stage_origin'] + 1, generation - WINDOW + 1)
    entries = r['entries']
    require(type(entries) is list and [e.get('generation') for e in entries] == list(range(start, generation + 1)),
            'Snapshot R2 committed evidence progress differs')
    all_parts = []
    for e in entries:
        require(set(e) == {'generation', 'healthy', 'qualified', 'tables'} and
                type(e['generation']) is int and type(e['healthy']) is bool and type(e['qualified']) is bool
                and type(e['tables']) is dict and set(e['tables']) == set(TABLES),
                'Invalid snapshot R2 evidence entry')
        parts = [p for p in e['tables'].values() if p is not None]
        all_parts.extend(parts)
        if e['qualified']:
            require(len(parts) == len(TABLES) and all(len(p['game_ids']) >= MIN_TABLE_GAMES for p in parts)
                    and pool(parts)['games'] >= MIN_GAMES and finite(pool(parts)['r2']),
                    'Unsupported qualifying snapshot R2 iteration')
    pool(all_parts)  # Reject repeated physical games even across unqualified entries.
    ref = r['reference']
    if ref is not None:
        require(type(ref) is dict and set(ref) == {'generation', 'weight', 'regime', 'evidence'}
                and type(ref['generation']) is int and 0 <= ref['generation'] <= generation
                and ref['weight'] in WEIGHTS and type(ref['regime']) is dict,
                'Invalid frozen snapshot R2 reference')
        a = ref['evidence']
        require(a['supported'] is True and MIN_ITERATIONS <= a['qualifying_iterations'] <= WINDOW
                and len(a['contributing_generations']) == a['qualifying_iterations']
                and a['contributing_generations'] == sorted(set(a['contributing_generations']))
                and all(type(g) is int and 0 < g <= ref['generation'] for g in a['contributing_generations'])
                and a['games'] >= MIN_GAMES and a['rows'] >= a['games']
                and finite(a['r2']) and finite(a['learner_mse']) and a['learner_mse'] >= 0
                and finite(a['prediction_bias']) and finite(a['target_variance']) and a['target_variance'] > 1e-12
                and abs(a['r2'] - (1 - a['learner_mse'] / a['target_variance'])) < 1e-12,
                'Invalid frozen snapshot R2 moments/support')
    streak = r['healthy_streak']
    require(type(streak) is int and 0 <= streak < HEALTHY_ITERATIONS
            and streak <= generation - r['stage_origin']
            and (not streak or ref is not None and weight < 1. and len(entries) >= streak
                 and all(e['healthy'] and e['qualified'] for e in entries[-streak:])),
            'Invalid snapshot R2 healthy streak')
    return r


def migrate(state, config, generation):
    from .critic_data import validate_state
    validate_state(state, config, generation)
    result = copy.deepcopy(state)
    if result['weight_recovery']['contract'] == CONTRACT:
        return result, None
    old = result['learner_ev_baseline']
    result['weight_recovery'] = initial(generation, result['snapshot_weight'], old)
    result['learner_ev_baseline'] = None
    return result, dict(contract=CONTRACT, generation=generation, archived_ev_baseline=old,
        baseline_status='pending_ten_qualifying_iterations',
        scope='Snapshot controller only; old EV is archived, never used as R2. Weight, GAE, both Adams and RNG preserved.')


def reference(evidence, generation, weight, current_regime):
    require(evidence['supported'], 'Cannot freeze unsupported snapshot R2 baseline')
    return dict(generation=generation, weight=weight, regime=copy.deepcopy(current_regime),
                evidence=copy.deepcopy(evidence))


def advance(r, weight, *, generation, observation, current_regime, active, tolerance):
    r = copy.deepcopy(validate(r, generation - 1, weight))
    require(observation['generation'] == generation, 'Snapshot R2 observation generation differs')
    regime_changed = r['regime'] is not None and r['regime'] != current_regime
    if regime_changed:
        # Do not pool incompatible targets/holdouts/training modes. A frozen
        # reference survives this reset and cannot ratchet down on poor results.
        r.update(stage_origin=generation - 1, entries=[], healthy_streak=0)
    r['regime'] = copy.deepcopy(current_regime)
    stage_origin_used = r['stage_origin']
    before = r['healthy_streak']
    r['entries'] = (r['entries'] + [copy.deepcopy(observation)])[-WINDOW:]
    evidence = summarize(r['entries'])
    ref_before = copy.deepcopy(r['reference'])
    threshold = ref_before['evidence']['r2'] if ref_before else None
    healthy = observation['healthy']
    fresh = observation['qualified']
    recovered = threshold is not None and evidence['supported'] and evidence['r2'] >= threshold
    eligible = bool(active and fresh and healthy and recovered)
    r['healthy_streak'] = 0
    decision, reason = 'hold', 'snapshot_r2_insufficient_support'
    next_weight = weight
    used_streak = 0
    if not active:
        reason = 'snapshots_inactive'
    elif not evidence['supported']:
        pass
    elif ref_before is None:
        # One initial calibration; never refresh merely because R2 has fallen.
        if healthy and fresh:
            recent = [e for e in r['entries'] if e['qualified']][-MIN_ITERATIONS:]
            r['reference'] = reference(summarize(recent), generation, weight, current_regime)
            reason = 'snapshot_r2_initial_reference_frozen'
        else:
            reason = 'snapshot_r2_calibration_health_not_ready'
    elif fresh and evidence['r2'] < threshold - tolerance and weight > .5:
        next_weight = .5
        decision, reason = 'reduce_snapshot_weight', 'learner_rolling_r2_below_frozen_reference'
    elif not fresh:
        reason = 'no_new_qualifying_snapshot_iteration'
    elif not healthy:
        reason = 'snapshot_recovery_health_not_ready'
    elif not recovered:
        reason = 'snapshot_recovery_r2_below_frozen_reference'
    elif weight < 1.:
        used_streak = before + 1
        if used_streak >= HEALTHY_ITERATIONS:
            next_weight = WEIGHTS[WEIGHTS.index(weight) + 1]
            decision, reason = 'increase_snapshot_weight', 'sustained_healthy_rolling_r2_recovery'
            # Freeze PRE-INCREASE performance. No new-stage sample can change it.
            r['reference'] = reference(evidence, generation, weight, current_regime)
        else:
            r['healthy_streak'] = used_streak
            reason = 'snapshot_recovery_streak'
    else:
        reason = 'snapshots_active'
    if next_weight != weight:
        r.update(stage_origin=generation, stage_weight=next_weight, entries=[], healthy_streak=0,
                 last_change_generation=generation)
    validate(r, generation, next_weight)
    report = dict(contract=CONTRACT, origin=r['origin'], window_iterations=WINDOW,
        minimum_iterations=MIN_ITERATIONS, minimum_games=MIN_GAMES,
        minimum_games_per_table=MIN_TABLE_GAMES, required_iterations=HEALTHY_ITERATIONS,
        stage_origin_used=stage_origin_used,
        stage_origin_next=r['stage_origin'], regime_changed=regime_changed,
        healthy_streak_before=before, evidence_streak=used_streak, healthy_streak_next=r['healthy_streak'],
        eligible=eligible, health_ok=healthy, r2_recovered=bool(recovered), r2_threshold=threshold,
        reduce_below=threshold - tolerance if threshold is not None else None,
        learner=evidence, current=pool([p for p in observation['tables'].values() if p is not None]),
        current_qualifying=fresh, reference_before=ref_before, reference_next=copy.deepcopy(r['reference']),
        next_weight=WEIGHTS[WEIGHTS.index(next_weight) + 1] if next_weight < 1. else None,
        last_change_generation=r['last_change_generation'],
        scope='Ordinary held-out learner rows only; row-pooled MSE includes bias. Twenty committed iterations, ten qualifying; five new consecutive healthy commits, not independent overlapping windows. New weight stages start empty; reference never lowered on deterioration.')
    return r, next_weight, decision, reason, report
