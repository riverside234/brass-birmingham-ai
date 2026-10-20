"""Economic and population diagnostics from permitted replay information."""
from collections import Counter, defaultdict
from brass_ai.learning.runner import match_report
from brass_ai.step5.league import pottery_metrics
from brass_ai.learning.replay import terminal_targets


def learner_seats(games,learner_ids):
    result={}
    for game in games:
        targets=terminal_targets(game['result'])
        for seat,policy in enumerate(game['opponents']):
            if learner_ids is not None and policy not in learner_ids:continue
            row=result.setdefault(str(seat),dict(seat_games=0,win_share_sum=0.,placement_sum=0.,vp_sum=0.,vp_square_sum=0.))
            row['seat_games']+=1
            for key,field in (('win_share_sum','win_share'),('placement_sum','placement'),('vp_sum','vp')):
                row[key]+=targets[field][seat]
            row['vp_square_sum']+=targets['vp'][seat]**2
    return result


def health_counts(games,learner_ids):
    """Exact complete-game learner seat counts, before replay IPC compaction."""
    from .insolvency import events
    out=dict(tables=len(games),all_learner_tables=0,seat_games=0,vp_sum=0.,sells=0,
             passes_with_alternative=0,zero_vp_seats=0,insolvent_seats=0,insolvency_events=0)
    for game in games:
        seats={s for s,p in enumerate(game['opponents']) if learner_ids is None or p in learner_ids}
        vp=terminal_targets(game['result'])['vp']
        out['all_learner_tables']+=int(len(seats)==4)
        out['seat_games']+=len(seats);out['vp_sum']+=sum(vp[s] for s in seats)
        out['zero_vp_seats']+=sum(vp[s]==0 for s in seats)
        found=[e for e in events(game['records']) if e['seat'] in seats]
        out['insolvency_events']+=len(found);out['insolvent_seats']+=len({e['seat'] for e in found})
        for row in game['records']:
            if row['seat'] not in seats:continue
            kind=row['choices'][row['chosen']]['kind']
            out['sells']+=int(kind=='sell')
            out['passes_with_alternative']+=int(kind=='pass' and any(c['kind']!='pass' for c in row['choices']))
    return out


def economy(games,learner_ids=None):
    actions=Counter();phases=Counter();builds=Counter();flipped=Counter();merchant=Counter()
    cash=[];income=[];forced=0;debt=0;feasible_sales=0;selected_sales=0;seen_tiles=set();tile_states={}
    by_era=Counter();levels=Counter();passes_with_alternative=sale_starts=sale_continuations=0
    for game in games:
        relevant={s for s,p in enumerate(game['opponents']) if learner_ids is None or p in learner_ids}
        for row in game['records']:
            view=row['view']
            for slot in view['slots']:
                tile=slot.get('tile')
                if tile and tile['seat'] in relevant:
                    key=(game['game_id'],tile['id'])
                    if tile['flipped'] and key not in seen_tiles:
                        flipped[tile['type']]+=1;seen_tiles.add(key)
                    tile_states[key]=dict(era=view['context']['era'],type=tile['type'],
                        level=tile['level'],flipped=tile['flipped'],resource=tile['resource'])
            if row['seat'] not in relevant: continue
            choice=row['choices'][row['chosen']];actor=view['players'][row['seat']]
            actions[choice['kind']]+=1;phases[view['context']['phase']]+=1
            by_era[view['context']['era']+'|'+choice['kind']]+=1
            passes_with_alternative+=int(choice['kind']=='pass' and any(c['kind']!='pass' for c in row['choices']))
            sale_starts+=int(choice['kind']=='sell' and view['context']['phase']=='action')
            sale_continuations+=int(choice['kind']=='sell' and view['context']['phase']=='continuation')
            forced+=int(len(row['choices'])==1);debt+=int(view['context']['phase']=='debt')
            cash.append(actor['cash']);income.append(actor['income'])
            if choice['kind']=='build':
                builds[choice['industry']]+=1
                mat=next((m['tiles'] for m in actor['mat'] if m['type']==choice['industry']),[])
                if mat:levels[f"{view['context']['era']}|{choice['industry']}|{mat[0]['level']}"]+=1
            feasible_sales+=int(any(c['kind']=='sell' for c in row['choices']))
            selected_sales+=int(choice['kind']=='sell')
            if choice['kind']=='beer' and str(choice.get('target','')).startswith('merchant:'):
                identity=choice['target'].split(':',1)[1]
                bonus=next((m['bonus'] for m in view['merchants'] if m['id']==identity),'unknown')
                merchant[bonus]+=1
    unflipped=Counter(v['type'] for v in tile_states.values() if not v['flipped'])
    return dict(actions=dict(actions),phases=dict(phases),forced_decisions=forced,
        actions_by_era=dict(by_era),build_levels=dict(levels),
        passes_with_nonpass_alternative=passes_with_alternative,
        sale_actions_started=sale_starts,sale_continuations=sale_continuations,
        builds=dict(builds),observed_flips=dict(flipped),last_seen_unflipped=dict(unflipped),
        minimum_observed_cash=min(cash) if cash else None,
        mean_observed_cash=sum(cash)/len(cash) if cash else None,
        mean_observed_income=sum(income)/len(income) if income else None,
        debt_decisions=debt,feasible_sale_decisions=feasible_sales,selected_sales=selected_sales,
        merchant_beer_by_bonus=dict(merchant),
        interpretation='Counts use public predecision views; build level is the next public mat tile. A legal alternative to Pass need not be good. Sale starts and continuation tile choices are separate. Last-seen unflipped tiles are not exact end-of-era waste; scoring may follow the last view.')


def summarize(games,learner_ids,kind_by_group=None):
    kinds=defaultdict(list)
    for game in games:
        kinds[(kind_by_group or {}).get(game['setup_group'],'unknown')].append(game)
    # Derive the overall diagnostic from disjoint composition summaries. The
    # previous path walked every board/history again for the same totals.
    parts=[]
    for kind,group in kinds.items():
        summary=dict(matches=match_report(group),economy=economy(group,learner_ids),
                     diversity=pottery_metrics(group,learner_ids),learner_by_seat=learner_seats(group,learner_ids),
                     health_counts=health_counts(group,learner_ids))
        parts.append(dict(**summary,by_composition={kind:summary},
            measured_policy='frozen incoming collection policy, before this update'))
    return merge_summaries(parts)


def merge_summaries(parts):
    """Merge disjoint complete-game shards without transporting board history."""
    def counters(items):
        total=Counter()
        for item in items:
            for key,value in item.items():total[key]+=value
        return dict(total)
    def matches(items):
        policies={}
        for item in items:
            for key,row in item['policies'].items():
                out=policies.setdefault(key,dict(seat_games=0,win_share=0.,placement=0.,vp=0.,seat_counts=[0]*4))
                out['seat_games']+=row['seat_games']
                for metric in ('win_share','placement','vp'):out[metric]+=row[metric]*row['seat_games']
                out['seat_counts']=[a+b for a,b in zip(out['seat_counts'],row['seat_counts'])]
        for row in policies.values():
            for metric in ('win_share','placement','vp'):row[metric]/=row['seat_games']
        return dict(items[0],games=sum(p['games'] for p in items),policies=policies)
    def economics(items):
        out=dict(items[0]);counts=[sum(p['actions'].values()) for p in items];total=sum(counts)
        for key in ('actions','phases','builds','observed_flips','last_seen_unflipped','merchant_beer_by_bonus',
                    'actions_by_era','build_levels'):
            out[key]=counters([p[key] for p in items])
        for key in ('forced_decisions','debt_decisions','feasible_sale_decisions','selected_sales',
                    'passes_with_nonpass_alternative','sale_actions_started','sale_continuations'):
            out[key]=sum(p[key] for p in items)
        minima=[p['minimum_observed_cash'] for p in items if p['minimum_observed_cash'] is not None]
        out['minimum_observed_cash']=min(minima) if minima else None
        for key in ('mean_observed_cash','mean_observed_income'):
            out[key]=sum(p[key]*n for p,n in zip(items,counts) if n)/total if total else None
        return out
    def diversity(items):
        out=dict(items[0])
        for key in ('builds','legal_build_opportunities','builds_matching_assigned_focus','seat_trajectory_contexts'):
            out[key]=counters([p[key] for p in items])
        out['pottery_warning']=out['legal_build_opportunities']['Pottery']>=30 and out['builds']['Pottery']==0
        return out
    def combine(items):
        return dict(matches=matches([p['matches'] for p in items]),economy=economics([p['economy'] for p in items]),
                    diversity=diversity([p['diversity'] for p in items]),
                    health_counts=counters([p.get('health_counts',{}) for p in items]),
                    learner_by_seat={str(s):counters([p.get('learner_by_seat',{}).get(str(s),{}) for p in items]) for s in range(4)})
    grouped=defaultdict(list)
    for part in parts:
        for key,value in part['by_composition'].items():grouped[key].append(value)
    from .exploration_metrics import merge
    return dict(**combine(parts),exploration_metrics=merge([p.get('exploration_metrics') for p in parts]),
                by_composition={k:combine(v) for k,v in grouped.items()},
                measured_policy=parts[0]['measured_policy'])
