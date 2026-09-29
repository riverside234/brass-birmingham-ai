"""Real-engine gates: explicit chance parity, clones and both planners' privacy."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import time

import torch

from brass_ai.host import RulesHost
from brass_ai.beliefs import sample_world_setup
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import require,read_json
from brass_ai.learning.runner import RandomPolicy
from brass_ai.step5.search import SearchConfig
from brass_ai.searchlab.spiel import BrassSpielGame,action_id
from brass_ai.searchlab.ismcts import OpenSpielPolicy,SpielSearchConfig
from brass_ai.searchlab.s3 import S3Policy,TreeConfig


def without_revision(view):
    view=copy.deepcopy(view);view['context'].pop('revision',None);return view


def full_game_parity(root,seed,mono=None):
    rng=random.Random(seed);policy=RandomPolicy();game=BrassSpielGame(root=root,mono=mono)
    counters=Counter();comparisons=clones=private_checks=0;chance_clones=set()
    try:
        with RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as oracle:
            ctx=oracle.reset(seed);setup=oracle.call('checkpoint')['setup'];state=game.new_initial_state()
            # Follow a known random tape in this trusted parity test only. Real
            # policies never receive the oracle or its seed/deck/checkpoint.
            tape={'order':list(map(str,setup['order'])),
                'merchants':[json.dumps(m.get('types') or []) for m in setup['merchants']],
                'cards':sum(setup['cards']['hands'],[])+sum(setup['cards']['discards'],[])+setup['cards']['deck']}
            for stage in ('order','merchants','cards'):
                for item in tape[stage]:
                    require(abs(sum(p for _,p in state.chance_outcomes())-1)<1e-10,'Unnormalized setup chance')
                    state.apply_action(action_id('chance:'+stage+':'+item));counters[stage]+=1
            for step in range(2000):
                if state.is_terminal():break
                if step and step%100==0:print(json.dumps(dict(gate='parity-progress',decisions=step)),flush=True)
                require(state.current_player()==ctx['actor'],'Scheduler/actor mismatch')
                for s in range(4):
                    expected,actual=without_revision(oracle.observe(s)),without_revision(state.views[s])
                    require(expected==actual,'R1/R4 observed transition differs: '+json.dumps(dict(step=step,seat=s,
                        differences={k:[expected[k],actual[k]] for k in expected if expected[k]!=actual[k]})))
                    comparisons+=1
                choices=oracle.choices();require(choices==state.legal,'Legal choice completeness differs')
                selected=policy.select_batch([(oracle.observe(ctx['actor']),choices,rng)])[0][0]
                if step%47==0:
                    key=state.information_state_string();clone=state.clone()
                    clone.apply_action(action_id(choices[selected]['id']))
                    require(state.information_state_string()==key,'Clone changed original permitted state');clones+=1
                old_actor=ctx['actor'];old_hand=Counter(state.views[old_actor]['own_hand'])
                # Choice is commonly finish; for all other choices no random transition occurs.
                ctx=oracle.apply(ctx,choices[selected]);state.apply_action(action_id(choices[selected]['id']))
                while state.pending:
                    kind=state.pending['kind'];after=oracle.call('checkpoint')['commands'][-1]['cards_after']
                    if kind=='draw':tape_cards=list((Counter(after['hands'][old_actor])-old_hand).elements())
                    else:
                        order=oracle.observe(0)['order']
                        tape_cards=sum([after['hands'][s] for s in order],[])+after['deck']
                    require(len(tape_cards)==state.pending['remaining'],'Chance draw count differs')
                    if kind not in chance_clones:
                        saved=state.serialize();clone=state.clone()
                        clone.apply_action(state.chance_outcomes()[0][0])
                        require(state.serialize()==saved,'Private chance clone changed original')
                        require(game.deserialize_state(saved).serialize()==saved,'Serialization did not preserve pending chance')
                        chance_clones.add(kind)
                    for index,card in enumerate(tape_cards):
                        probs=state.chance_outcomes();require(abs(sum(p for _,p in probs)-1)<1e-10,'Unnormalized chance')
                        before=[state.observation_string(s) for s in range(4)]
                        state.apply_action(action_id('chance:card:'+card));counters[kind]+=1
                        if state.pending and index<len(tape_cards)-1:
                            require(before==[state.observation_string(s) for s in range(4)],'Private chance callback leaked a card')
                            private_checks+=1
            require(state.is_terminal() and ctx['terminal'],'Incomplete parity game')
            values=[p['utility'] for p in oracle.call('result')['seats']]
            require(values==state.returns() and abs(sum(values)-1)<1e-9,'Four-seat terminal/tie return mismatch')
        return dict(seed=seed,decisions=step,seat_observation_comparisons=comparisons,
            independent_clones=clones,pending_chance_clone_kinds=sorted(chance_clones),
            private_chance_checks=private_checks,chance=dict(counters),returns=state.returns())
    finally:game.close()


def privacy(root,mono=None,device='cpu'):
    model,_=load_checkpoint(root/'training/models/bootstrap/r2-fresh-medium.pt',device=device)
    composition=read_json(root/'configs/contracts/v0.2.0/card-composition.json')
    rows=[]
    with RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as a, RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as b:
        for seat in range(4):
            seed=1
            while a.reset(seed)['actor']!=seat:seed+=1
            view=a.observe(seat)
            view['players'][seat]['cash']=0
            view['deck_count']+=len(view['own_hand'])-2
            view['own_hand']=view['own_hand'][:2];view['players'][seat]['hand_count']=2
            ca=a.reset(1,sample_world_setup(view,composition,581+seat))
            cb=b.reset(1,sample_world_setup(view,composition,781+seat))
            require(a.call('canonical_hash')!=b.call('canonical_hash'),'Counterfactual hidden worlds must differ')
            va,vb=a.observe(seat),b.observe(seat);choices=a.choices()
            require(va==vb and choices==b.choices(),'Counterfactual allowed input differs')
            for variant in ('r4','s3'):
                if variant=='r4':
                    policy=OpenSpielPolicy(model,root,device,config=SpielSearchConfig(seconds=120,simulations=8,proposals=2),mono=mono)
                else:
                    policy=S3Policy(model,root/'artifacts/r1-host/guarded',device,config=SearchConfig(seconds=120,
                        proposals=2,finalists=2,pilot_samples=1,confirmation_samples=2,turn_interval=1),
                        tree=TreeConfig(simulations=2,future_turns=1),focus='neutral',mono=mono)
                try:
                    outputs=[];statistics=[]
                    for allowed in (va,vb):
                        rng=random.Random(730+seat)
                        if variant=='r4':out=policy.decide(allowed,choices,rng,time.monotonic()+120)
                        else:out=policy.search(allowed,choices,rng,time.monotonic()+120)
                        outputs.append(out);statistics.append(copy.deepcopy(policy.last_statistics if variant=='r4' else policy.last_tree))
                    require(outputs[0]==outputs[1] and statistics[0]==statistics[1],'Hidden state changed search or selection statistics')
                    require(policy.telemetry['incomplete']==0,'Fixed-work gate hit a timeout')
                    if variant=='r4':
                        require(statistics[0]['opponent_nodes']==0,'Root-conditioned opponent statistics')
                        require(policy.telemetry['opponent_choices']>0,'Gate did not exercise opponent replies')
                    else:require(policy.telemetry['tree_nodes']>0 and policy.telemetry['future_root_turns']>0,'S3 did not traverse future observations')
                    rows.append(dict(seat=seat,variant=variant,passed=True,telemetry=policy.telemetry))
                    print(json.dumps(dict(gate='privacy',seat=seat,variant=variant,passed=True)),flush=True)
                finally:policy.close()
    return rows


def depth_and_deadline(root,mono=None,device='cpu'):
    """Exercise the second future turn and rejection of expired teacher work."""
    model,_=load_checkpoint(root/'training/models/bootstrap/r2-fresh-medium.pt',device=device)
    composition=read_json(root/'configs/contracts/v0.2.0/card-composition.json')
    with RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as live:
        ctx=live.reset(51);view=live.observe(ctx['actor']);seat=view['seat']
        view['players'][seat]['cash']=0;view['deck_count']+=len(view['own_hand'])-2
        view['own_hand']=view['own_hand'][:2];view['players'][seat]['hand_count']=2
        live.reset(1,sample_world_setup(view,composition,819));view=live.observe(seat);choices=live.choices()
        policy=S3Policy(model,root/'artifacts/r1-host/guarded',device,
            config=SearchConfig(seconds=180,proposals=2,finalists=2,pilot_samples=1,confirmation_samples=2,turn_interval=1),
            tree=TreeConfig(simulations=2,future_turns=2),mono=mono,focus='neutral')
        try:
            out=policy.search(view,choices,random.Random(184),time.monotonic()+180)
            require(policy.telemetry['confirmed']==1 and policy.telemetry['leaf_depths'].get('2',0)>0,
                    'Two-future-turn S3 horizon did not complete')
            from brass_ai.step5.search import SearchIncomplete
            try:policy.search(view,choices,random.Random(184),time.monotonic()-1)
            except SearchIncomplete:pass
            else:raise ValueError('Expired S3 work must not produce teacher targets')
            return dict(two_future_turns=True,expired_search_rejected=True,
                fresh_confirmation=True,telemetry=policy.telemetry)
        finally:policy.close()


def run(root,mono=None,device='cpu'):
    torch.set_num_threads(2)
    games=[full_game_parity(root,9211,mono)]
    print(json.dumps(dict(gate='full-game-explicit-chance-parity',result=games[0])),flush=True)
    privacy_rows=privacy(root,mono,device)
    depth=depth_and_deadline(root,mono,device)
    print(json.dumps(dict(gate='horizon-and-deadline',passed=True)),flush=True)
    return dict(full_games=games,privacy=privacy_rows,depth_and_deadline=depth)
