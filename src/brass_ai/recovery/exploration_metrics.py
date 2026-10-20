"""Bounded sufficient counts from ordinary replays; branches reported separately."""
from collections import Counter

from brass_ai.learning.contracts import digest
from brass_ai.step5.policy import FAMILY_WEIGHTS


def collect(games):
    families=Counter();forced=Counter();actions=Counter();unknown=Counter();opening=0;seat_games=0;first={str(s):set() for s in range(4)}
    eligible=0;branches=[];extra_actor=extra_critic=0
    pass_only=pass_with_alternative=0
    for game in games:
        seat_games+=sum(any(r['seat']==s and r['behavior'].get('exploration_version')==3 for r in game['records']) for s in range(4))
        eligible+=game.get('branch_statistics',{}).get('eligible_decisions',0)
        for r in game['records']:
            b=r['behavior']
            if b.get('exploration_version')!=3:continue
            if r['view']['context']['phase']=='action':
                kind=r['choices'][r['chosen']]['kind']
                if len(r['choices'])>1:actions[kind]+=1
                if b['component']=='exploration':
                    (families if len(r['choices'])>1 else forced)[kind]+=1
                    if kind=='pass':
                        if all(c['kind']=='pass' for c in r['choices']):pass_only+=1
                        else:pass_with_alternative+=1
                unknown.update({c['kind'] for c in r['choices'] if c['kind'] not in FAMILY_WEIGHTS and c['kind']!='pass'})
                if b['action_decision']==0:first[str(r['seat'])].add(digest(r['choices'][r['chosen']]))
            opening+=int(b['temperature']!=1.)
        for child in game.get('branches',[]):
            extra_critic+=len(child['records'])-1
            extra_actor+=sum(len(r['choices'])>1 for r in child['records'][1:])
            branches.append(child)
    from .diagnostics import health_counts
    return dict(contract='exploration-metrics/1.0.0',seat_games=seat_games,exploration_picks=dict(families),
        forced_exploration_picks=dict(forced),sampled_action_families=dict(actions),
        pass_only_exploration_picks=pass_only,pass_with_alternative_exploration_picks=pass_with_alternative,
        unknown_action_kinds=dict(unknown),tempered_decisions=opening,
        first_actions={k:sorted(v) for k,v in first.items()},
        branching=dict(games=len(branches),eligible_decisions=eligible,extra_actor_rows=extra_actor,
            extra_critic_rows=extra_critic,components=dict(Counter(g['branch']['component'] for g in branches)),
            fraction_remaining_sum=sum(g['branch']['fraction_remaining'] for g in branches),
            seconds=sum(g['branch']['elapsed_seconds'] for g in branches),
            economy=health_counts(branches,None) if branches else {}))


def merge(parts):
    parts=[p for p in parts if p]
    if not parts:return {}
    def counts(values):
        c=Counter()
        for v in values:c.update(v)
        return dict(c)
    result=dict(contract=parts[0]['contract'],
        seat_games=sum(p['seat_games'] for p in parts),
        tempered_decisions=sum(p['tempered_decisions'] for p in parts),
        exploration_picks=counts(p['exploration_picks'] for p in parts),
        forced_exploration_picks=counts(p['forced_exploration_picks'] for p in parts),
        sampled_action_families=counts(p['sampled_action_families'] for p in parts),
        pass_only_exploration_picks=sum(p['pass_only_exploration_picks'] for p in parts),
        pass_with_alternative_exploration_picks=sum(p['pass_with_alternative_exploration_picks'] for p in parts),
        unknown_action_kinds=counts(p['unknown_action_kinds'] for p in parts),
        distinct_first_actions_per_seat={str(s):len({a for p in parts for a in p['first_actions'][str(s)]}) for s in range(4)})
    total=sum(result['exploration_picks'].values())
    result['family_shares']={k:v/max(1,total) for k,v in result['exploration_picks'].items()}
    result['random_picks_per_neural_seat_game']={k:v/max(1,result['seat_games']) for k,v in result['exploration_picks'].items()}
    branch={k:sum(p['branching'][k] for p in parts) for k in ('games','eligible_decisions','extra_actor_rows','extra_critic_rows','seconds','fraction_remaining_sum')}
    branch['components']=counts(p['branching']['components'] for p in parts)
    branch['economy']=counts(p['branching']['economy'] for p in parts)
    branch['mean_fraction_remaining']=branch.pop('fraction_remaining_sum')/branch['games'] if branch['games'] else None
    branch['scope']='Continuation outcomes/actions only; excluded from ordinary health, PFSP, ratings and progress'
    result['branching']=branch
    result['scope']='Actual latent exploration picks in nonforced ordinary action-phase decisions; learner and neural snapshot seats. Single-choice picks and pass-only positions reported separately (a pass-only position may have several discard choices).'
    return result
