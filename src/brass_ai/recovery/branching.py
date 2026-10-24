"""Trusted ordinary-worker forks; alternative move is never a training row."""
import copy
import random
import time
from collections import Counter

import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.step5.policy import exploration_distribution,NeuralPolicy
from .exploration import BRANCH,LEGACY_BRANCH,OPENING,REVISED_BRANCH,REVISED_OPENING


def seed_for(parent,decision,contract=LEGACY_BRANCH['contract']):
    require(contract in (LEGACY_BRANCH['contract'],BRANCH['contract'],REVISED_BRANCH['contract']),'Unknown branch contract')
    return digest(dict(contract=contract,parent=parent,decision=decision))


def eligible(view,choices,action_count):
    c=view['context']
    return (c['phase']=='action' and action_count>=2 and len(choices)>1 and
            not (c['era']=='RailEra' and view['deck_count']==0 and len(view['own_hand'])<=4))


def alternative(raw,q,played,rng):
    """Exclude first, normalize each component; skip a zero-support component.

    A q-only alternative can be empty when the other move is pass. Reweight
    the available components, recording both conditional and marginal mass.
    """
    candidates=[]
    for name,weight,values in (('temperature_1',.7,raw),('temperature_2',.25,np.sqrt(raw)),('exploration_q',.05,q)):
        p=np.array(values,dtype=np.float64,copy=True);p[played]=0
        if p.sum()>0:
            p/=p.sum();candidates.append((name,weight,p))
    if not candidates:return None
    selected=rng.choices(range(len(candidates)),weights=[c[1] for c in candidates],k=1)[0]
    name,weight,p=candidates[selected]
    action=rng.choices(range(len(p)),weights=p,k=1)[0]
    total=sum(c[1] for c in candidates)
    return dict(chosen=action,component=name,probability=float(p[action]),
        component_probability=weight/total,marginal_probability=float(sum(w/total*v[action] for _,w,v in candidates)))


def capture(host,game,row,assigned,rngs,*,portable=False,branching=LEGACY_BRANCH):
    step=row['step'];seat=row['seat'];p=assigned[seat];seed=seed_for(game['game_id'],step,branching['contract'])
    q=exploration_distribution(row['choices'],row['intent'],3,'action')
    move=alternative(p.last_raw[rngs[seat]],q,row['chosen'],random.Random(seed))
    if move is None:return None
    require(all(type(p) is NeuralPolicy and p.exploration_version==3 for p in assigned),
        'Branches require four current neural learners')
    snapshot=host.call('fork_save',export=portable)
    from .insolvency import events
    prior=Counter(e['seat'] for e in events(game['records']))
    return dict(**(dict(checkpoint=snapshot['checkpoint']) if portable else {}),
        row=copy.deepcopy(row),contexts=[copy.deepcopy(p.contexts[rngs[s]]) for s,p in enumerate(assigned)],
        counts=[p.action_counts.get(rngs[s],0) for s,p in enumerate(assigned)],
        metadata=dict(contract=branching['contract'],parent_game_id=game['game_id'],decision=step,
            played=row['chosen'],seed=seed,prior_events=[prior[s] for s in range(4)],
            raw_probabilities=p.last_raw[rngs[seat]].tolist(),**move))


def continue_branch(host,job,parent,packet,assigned,timings,max_decisions,encoder,lease):
    from .execution import _run_turns
    from .replay import CheckedGame
    started=time.monotonic();meta=packet['metadata'];seed=meta['seed']
    require(meta['contract']==job['branching']['contract'],'Branch collection contract differs')
    reset=meta['contract'] in (BRANCH['contract'],REVISED_BRANCH['contract'])
    if reset:
        opening=REVISED_OPENING if meta['contract']==REVISED_BRANCH['contract'] else OPENING
        require(job['branching']['reset_opening_at_fork'] is True and
                all(p.opening==opening for p in assigned),'Branch opening policy differs')
    shuffle=int(seed[:8],16)&0x7fffffff if job['branching']['reshuffle_hidden_deck'] else None
    if 'checkpoint' in packet:
        context=host.call('fork_restore',checkpoint=packet['checkpoint'],shuffle_seed=shuffle)
        if hasattr(host,'enable_fastpath') and not host.fast_checked:host.enable_fastpath()
    else:context=host.call('fork_resume',shuffle_seed=shuffle)
    if 'local_scoring' in job['provenance']:host.call('local_scoring_enable',fresh=True)
    row=copy.deepcopy(packet['row'])
    require({k:v for k,v in context.items() if k!='game_id'}==row['view']['context'],
        'Restored branch context differs')
    if 'checkpoint' in packet:
        require(context['game_id']==packet['checkpoint']['game_id'],'Restored branch request identity differs')
    row['step']=0;row['chosen']=meta['chosen']
    row['behavior']['probability']=meta['marginal_probability']
    row['behavior']['component']='branch_move'
    # Fork action remains in the authoritative replay for causal reward
    # accounting, but prepared() excludes it from actor AND critic.
    game=dict(game_id=parent['game_id']+'/branch/'+seed,setup_group=parent['setup_group'],
        players=4,split=parent['split'],opponents=list(parent['opponents']),result=None,records=[row],branch=meta)
    encoded=[encoder.encode(row['view'],row['choices'],row['style'],row['intent'],validated=True)]
    rngs=[random.Random(digest(dict(branch=seed,seat=s))) for s in range(4)]
    for s,p in enumerate(assigned):
        p.contexts[rngs[s]]=packet['contexts'][s]
        # The forced fork action is excluded. Start two fresh action decisions
        # per seat after it; payment/bonus/continuation choices consume none.
        p.action_counts[rngs[s]]=0 if reset else packet['counts'][s]
    context=host.apply(context,row['choices'][row['chosen']])
    result=_run_turns(host,{k:v for k,v in job.items() if k!='branching'},assigned,timings,
        max_decisions,context,game,rngs,encoded,encoder,lease)
    result['branch']['elapsed_seconds']=time.monotonic()-started
    result['branch']['fraction_remaining']=len(result['records'])/(len(result['records'])+meta['decision'])
    return CheckedGame(result).bind(encoded)


def validate_child(parent,child):
    meta=child['branch']
    require(meta['contract'] in (LEGACY_BRANCH['contract'],BRANCH['contract'],REVISED_BRANCH['contract']),'Unknown branch contract')
    require(meta['parent_game_id']==parent['game_id'] and meta['seed']==seed_for(parent['game_id'],meta['decision'],meta['contract']),
        'Foreign or changed branch seed')
    require(child['game_id']==parent['game_id']+'/branch/'+meta['seed'] and
        child['setup_group']==parent['setup_group'] and child['split']==parent['split'] and
        child['opponents']==parent['opponents'], 'Branch lineage/seat/split differs')
    source=parent['records'][meta['decision']];row=child['records'][0]
    require(source['view']==row['view'] and source['choices']==row['choices'] and
        meta['played']==source['chosen'] and row['chosen']==meta['chosen']!=meta['played'] and
        row['style']==source['style'] and row['intent']==source['intent'], 'Branch point differs from parent')
    require(0<meta['probability']<=1 and 0<meta['marginal_probability']<=1 and
        meta['component'] in BRANCH['move_distribution'],'Invalid branch sampling evidence')
    require(row['behavior']['component']=='branch_move' and
        row['behavior']['probability']==meta['marginal_probability'], 'Branch behavior probability differs')
    require(eligible(source['view'],source['choices'],source['behavior']['action_decision']),'Ineligible branch point')
    raw=np.asarray(meta['raw_probabilities'],dtype=np.float64)
    require(raw.shape==(len(source['choices']),) and np.isfinite(raw).all() and (raw>=0).all() and
        abs(raw.sum()-1)<1e-6,'Invalid branch policy distribution')
    q=exploration_distribution(source['choices'],source['intent'],3,'action')
    replay=alternative(raw,q,meta['played'],random.Random(meta['seed']))
    require(replay is not None and all(meta[k]==v for k,v in replay.items()),'Branch move/probability differs')
    epsilon=source['behavior']['exploration']
    require(abs(((1-epsilon)*raw+epsilon*q)[meta['played']]-source['behavior']['probability'])<3e-6,
        'Branch policy differs from parent behavior')
    from .insolvency import events
    counts=Counter(e['seat'] for e in events(parent['records'][:meta['decision']+1]))
    require(meta['prior_events']==[counts[s] for s in range(4)],'Branch cost budget differs')
    if meta['contract'] in (BRANCH['contract'],REVISED_BRANCH['contract']):
        validate_opening_rows(child['records'][1:],REVISED_OPENING if meta['contract']==REVISED_BRANCH['contract'] else OPENING)


def validate_opening_rows(rows,opening=OPENING):
    counts=[0]*4
    for row in rows:
        action=row['view']['context']['phase']=='action';seat=row['seat'];b=row['behavior']
        expected=opening['temperature'] if action and counts[seat]<opening['action_decisions_per_seat'] else 1.
        require(b['action_decision']==(counts[seat] if action else None) and b['temperature']==expected,
                'Branch opening counters/temperature differ from fresh per-seat decisions')
        if action:counts[seat]+=1


def expand(games):
    result=[]
    for game in games:
        result.append(game)
        for child in game.get('branches',[]):
            validate_child(game,child)
            if hasattr(game,'critic_table'):
                child.critic_table=game.critic_table;child.table_kind=game.table_kind
            if hasattr(game,'ppo_opening_objective'):
                child.ppo_opening_objective=game.ppo_opening_objective
            result.append(child)
    return result


def family(game_id):
    return game_id.split('/branch/',1)[0]
