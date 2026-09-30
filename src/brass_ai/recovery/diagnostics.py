"""Economic and population diagnostics from permitted replay information."""
from collections import Counter, defaultdict
from brass_ai.learning.runner import match_report
from brass_ai.step5.league import pottery_metrics


def economy(games,learner_ids=None):
    actions=Counter();phases=Counter();builds=Counter();flipped=Counter();merchant=Counter()
    cash=[];income=[];forced=0;debt=0;feasible_sales=0;selected_sales=0;seen_tiles=set();tile_states={}
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
            forced+=int(len(row['choices'])==1);debt+=int(view['context']['phase']=='debt')
            cash.append(actor['cash']);income.append(actor['income'])
            if choice['kind']=='build': builds[choice['industry']]+=1
            feasible_sales+=int(any(c['kind']=='sell' for c in row['choices']))
            selected_sales+=int(choice['kind']=='sell')
            if choice['kind']=='beer' and str(choice.get('target','')).startswith('merchant:'):
                identity=choice['target'].split(':',1)[1]
                bonus=next((m['bonus'] for m in view['merchants'] if m['id']==identity),'unknown')
                merchant[bonus]+=1
    unflipped=Counter(v['type'] for v in tile_states.values() if not v['flipped'])
    return dict(actions=dict(actions),phases=dict(phases),forced_decisions=forced,
        builds=dict(builds),observed_flips=dict(flipped),last_seen_unflipped=dict(unflipped),
        minimum_observed_cash=min(cash) if cash else None,
        mean_observed_cash=sum(cash)/len(cash) if cash else None,
        mean_observed_income=sum(income)/len(income) if income else None,
        debt_decisions=debt,feasible_sale_decisions=feasible_sales,selected_sales=selected_sales,
        merchant_beer_by_bonus=dict(merchant),
        interpretation='Counts use public predecision views. Last-seen unflipped tiles are diagnostics, not exact end-of-era wasted/scored tiles; scoring may follow the last view.')


def summarize(games,learner_ids,kind_by_group=None):
    kinds=defaultdict(list)
    for game in games:
        kinds[(kind_by_group or {}).get(game['setup_group'],'unknown')].append(game)
    return dict(matches=match_report(games),by_composition={k:dict(matches=match_report(v),
        economy=economy(v,learner_ids),diversity=pottery_metrics(v,learner_ids)) for k,v in kinds.items()},
        economy=economy(games,learner_ids),diversity=pottery_metrics(games,learner_ids),
        measured_policy='frozen incoming collection policy, before this update')
