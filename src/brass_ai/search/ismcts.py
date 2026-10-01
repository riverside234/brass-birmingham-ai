# ISMCTS/PUCT mechanics adapted from OpenSpiel, Copyright 2019 DeepMind.
# Apache-2.0; see LICENSE-OpenSpiel and NOTICE.md. Project-specific masking,
# pending-path batching, progressive expansion and transactional four-seat backup.
"""Batched single-observer Masked ISMCTS over the explicit-chance rules adapter.

The cooperative generators own detached hypothetical states. Rules access is
serialized by the adapter; no real hidden checkpoint or optimized opponent node
is accepted. NN predictions are cached separately from completed tree evidence.
"""
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import math
import json
import random
import time
import weakref
from threading import Event

import numpy as np

from brass_ai.learning.contracts import digest, require
from brass_ai.step5.policy import NeuralPolicy, inference_deadline
from brass_ai.searchlab.spiel import BrassSpielGame, action_id
from .common import SearchConfig, SearchIncomplete, expansion_indices, turn_key


def information_key(view,choices,recall):
    # Match the adapter's permitted information; add exact semantic legality.
    return digest(dict(observation=json.dumps(dict(view=view,setup=False,chance=None),sort_keys=True),
                       recall=recall,legal=[c['id'] for c in choices]))


def node_key(state,observer):
    return information_key(state.views[observer],state.legal,state.recall[observer])


@dataclass
class Edge:
    visits: int = 0
    pending: int = 0
    available: int = 0
    returns: np.ndarray = field(default_factory=lambda:np.zeros(4,dtype=np.float64))


@dataclass
class Node:
    visits: int = 0
    priors: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)


class MaskedISMCTS:
    def __init__(self,game,base,opponents,config,root,context,rng,deadline,recall=None,tree=None,cache=None,lanes=None,cancelled=None):
        self.game,self.base,self.opponents,self.config=game,base,opponents,config
        self.root,self.context,self.rng,self.deadline,self.recall=root,context,rng,deadline,recall
        self.tree=tree if tree is not None else {}
        self.lanes=lanes or [game]
        self.cancelled=cancelled
        self.cache=cache if cache is not None else OrderedDict()
        self.completed=self.interrupted=self.opponent_choices=self.max_pending=0
        self.root_key=None;self.continuations={};self.opponent_statistics={}
        self.metrics=dict(cache_hits=0,requests=0,coalesced=0,batches=0,positions=0,max_batch=0,inference_seconds=0.)

    def check(self):
        if self.cancelled is not None and self.cancelled.is_set():raise SearchIncomplete('cancelled')
        if time.monotonic()>=self.deadline:raise SearchIncomplete('whole_turn_deadline')

    def request(self,policy,state,rng):
        actor=state.current_player()
        require(actor>=0,'No policy evaluation of a chance or terminal node')
        context=policy.context_features(state.views[actor],rng)
        key=digest(dict(view=state.views[actor],choices=state.legal,context=context,
                        recall=state.recall[actor],model=policy.evaluator.model_id,policy=policy.policy_id))
        if key in self.cache:
            self.metrics['cache_hits']+=1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.metrics['requests']+=1
        result=yield (key,policy,state.views[actor],state.legal,rng)
        self.check()
        self.cache[key]=result
        if len(self.cache)>1024:self.cache.popitem(last=False)
        return result

    def trajectory(self,view,choices,game=None):
        path=[];trace=[];seed=self.rng.randrange(2**63)
        local=random.Random(seed)
        actor_rngs=[random.Random(digest(dict(seed=seed,seat=s))) for s in range(4)]
        assigned=[self.opponents[local.randrange(len(self.opponents))] for _ in range(4)]
        self.base.contexts[actor_rngs[self.root]]=self.context
        try:
            self.check()
            state=(game or self.game).from_observation(view,seed,self.recall)
            require([c['id'] for c in state.legal]==[c['id'] for c in choices], 'Belief changed current root legality')
            key=node_key(state,self.root)
            if self.root_key is None:self.root_key=key
            require(key==self.root_key,'Belief changed root permitted information/recall')
            initial_turn=last_turn=turn_key(view['context']);depth=0
            for step in range(self.config.max_choices):
                self.check()
                if state.is_terminal():
                    values=np.asarray(state.returns(),dtype=float);break
                if state.is_chance_node():
                    actions,weights=zip(*state.chance_outcomes())
                    state.apply_action(local.choices(actions,weights=weights)[0]);continue
                actor=state.current_player()
                if actor!=self.root:
                    if len(state.legal)==1:chosen=0
                    else:
                        p,_=yield from self.request(assigned[actor],state,actor_rngs[actor])
                        chosen=local.choices(range(len(state.legal)),weights=p)[0]
                    state.apply_action(action_id(state.legal[chosen]['id']));self.opponent_choices+=1
                    continue
                current=turn_key(state.context)
                if state.context['phase']=='action' and current!=last_turn:
                    depth+=1;last_turn=current
                if depth>=self.config.future_turns and state.context['phase']=='action':
                    _,response=yield from self.request(self.base,state,actor_rngs[actor])
                    values=np.asarray(response['win_share'],dtype=float);break
                if len(state.legal)==1:p=[1.]
                else:p,_=yield from self.request(self.base,state,actor_rngs[actor])
                key=node_key(state,self.root)
                if key not in self.tree and len(self.tree)>=16384:raise SearchIncomplete('tree_memory_cap')
                node=self.tree.setdefault(key,Node())
                ids=[c['id'] for c in state.legal]
                if node.priors:
                    require(set(node.priors)==set(ids),'Information node legality changed')
                else:node.priors=dict(zip(ids,map(float,p)))
                priors=[node.priors[c] for c in ids]
                active=expansion_indices(state.legal,priors,node.visits,node.edges,local,
                    root=(key==self.root_key),config=self.config)
                unseen=[i for i in active if ids[i] not in node.edges]
                if unseen:
                    width=(self.config.root_top_k if key==self.root_key else self.config.interior_top_k)+int(self.config.widening*math.sqrt(node.visits))
                    top=set(sorted(range(len(ids)),key=lambda i:(-priors[i],ids[i]))[:width])
                    tail=[i for i in unseen if i not in top]
                    pool=tail if tail and (not node.edges or len(node.edges)%4==0) else unseen
                    chosen=local.choices(pool,weights=[.9*priors[i]+.1/len(ids) for i in pool])[0]
                    node.edges[ids[chosen]]=Edge(available=node.visits)
                else:
                    def ucb(i):
                        edge=node.edges[ids[i]];n=edge.visits+edge.pending
                        mean=float(edge.returns[self.root]/edge.visits) if edge.visits else .25
                        return mean+self.config.exploration*priors[i]*math.sqrt(max(1,node.visits))/(1+n)
                    chosen=max(active,key=lambda i:(ucb(i),-i))
                identity=ids[chosen];edge=node.edges[identity];edge.pending+=1
                path.append((node,edge))
                if current==initial_turn:
                    trace.append((key,identity))
                state.apply_action(action_id(identity))
            else:raise SearchIncomplete('simulation_choice_cap')
            self.check()
            require(values.shape==(4,) and np.isfinite(values).all() and (values>=0).all() and (values<=1).all(), 'Invalid four-seat values')
            for node,edge in path:
                node.visits+=1;edge.visits+=1;edge.returns+=values
                for available in node.edges.values():available.available=node.visits
            if trace:
                self.continuations[trace[0][1]]=trace
            self.completed+=1
            return values
        finally:
            for _,edge in path:
                edge.pending-=1
                require(edge.pending>=0,'Reservation underflow')

    def run(self,view,choices):
        slots=[];started=0;search_began=time.monotonic()
        def advance(generator,result=None,first=False):
            try:return next(generator) if first else generator.send(result)
            except StopIteration:return None
            except SearchIncomplete:self.interrupted+=1;return None
        try:
            capacity=min(self.config.pending_paths,len(self.lanes))
            free=list(range(capacity))
            while time.monotonic()<self.deadline and (started<self.config.simulations or slots):
                while free and len(slots)<capacity and started<self.config.simulations and time.monotonic()<self.deadline:
                    lane=free.pop();g=self.trajectory(view,choices,self.lanes[lane]);started+=1
                    waiting=advance(g,first=True)
                    if waiting is not None:slots.append((g,waiting,lane))
                    else:free.append(lane)
                self.max_pending=max(self.max_pending,len(slots))
                if not slots:break
                # One allowed request can be shared, without inventing extra paths.
                unique={request[0]:request for _,request,_ in slots};groups=defaultdict(list)
                self.metrics['coalesced']+=len(slots)-len(unique)
                predicted={}
                for key,policy,v,c,rng in unique.values():groups[policy].append((key,v,c,rng))
                for policy,group in groups.items():
                    for start in range(0,len(group),self.config.batch_size):
                        self.check();part=group[start:start+self.config.batch_size]
                        began=time.monotonic();outputs=policy.predict([(v,c,rng) for _,v,c,rng in part])
                        self.metrics['inference_seconds']+=time.monotonic()-began
                        self.metrics['batches']+=1;self.metrics['positions']+=len(part);self.metrics['max_batch']=max(self.metrics['max_batch'],len(part))
                        self.check();predicted.update((row[0],out) for row,out in zip(part,outputs))
                next_slots=[]
                for generator,request,lane in slots:
                    waiting=advance(generator,predicted[request[0]])
                    if waiting is not None:next_slots.append((generator,waiting,lane))
                    else:free.append(lane)
                slots=next_slots
        except SearchIncomplete:self.interrupted+=max(1,len(slots))
        finally:
            for generator,_,_ in slots:generator.close()
        self.metrics['elapsed_seconds']=time.monotonic()-search_began
        self.metrics['completed_per_second']=self.completed/max(1e-9,self.metrics['elapsed_seconds'])
        return self.statistics(choices)

    def statistics(self,choices):
        node=self.tree.get(self.root_key,Node())
        return dict(completed=self.completed,interrupted=self.interrupted,opponent_nodes=0,
            max_pending=self.max_pending,opponent_choices=self.opponent_choices,
            root={c['id']:dict(visits=e.visits,returns=e.returns.tolist(),available=e.available)
                  for c in choices if (e:=node.edges.get(c['id'])) is not None and e.visits>0},
            expansion=[c['id'] in node.edges for c in choices],batching=self.metrics,
            continuations=self.continuations,distinct_root_alternatives=sum(e.visits>0 for e in node.edges.values()))


def visit_teacher(choices,stats,priors,*,model,observer,config):
    ids=[c['id'] for c in choices]
    visits=[stats['root'].get(i,{}).get('visits',0) for i in ids]
    if sum(n>0 for n in visits)<2:return None
    count=sum(visits);prior=np.asarray(priors,dtype=float);prior/=prior.sum()
    target=(1-config.prior_mix)*np.asarray(visits,dtype=float)/count+config.prior_mix*prior
    from brass_ai.step5.league import implementation_id
    source=implementation_id()
    return dict(source='ismcts-root-visits/1.0.0',probabilities=target.tolist(),coverage=[True]*len(ids),
        confidence=1.,samples=count,horizon='settled-current-turn/actual-next-root-information',
        legal_ids=ids,visits=visits,priors=prior.tolist(),expansion=stats['expansion'],model_id=model,
        observer=observer,worlds=count,prior_mix=config.prior_mix,source_fingerprint=source,
        search=config.to_dict(),fingerprint=digest(dict(model=model,config=config.to_dict(),contract='masked-ismcts/1.0.0',source=source)))


class MaskedSearchPolicy:
    kind='external'
    def __init__(self,model,root,device='cpu',*,config=None,mono=None,focus='neutral',opponents=None,evaluator=None):
        self.config=(config or SearchConfig()).validate()
        self.base=NeuralPolicy(model,device,exploration=.1,focus=focus,evaluator=evaluator)
        self.rollouts=opponents or [NeuralPolicy(model,device,exploration=.05,focus='neutral',evaluator=evaluator)]
        require(all(isinstance(p,NeuralPolicy) and p.focus=='neutral' for p in self.rollouts),'Fixed masked rollout policies required')
        self.game=BrassSpielGame(root=root,mono=mono)
        # Persistent bounded lanes, not a new process per leaf. A pending path
        # advances in its own warm engine; NN batching never swaps live contexts.
        self.simulation_games=[]
        try:
            for _ in range(self.config.pending_paths):self.simulation_games.append(BrassSpielGame(root=root,mono=mono))
        except BaseException:
            self.game.close()
            for game in self.simulation_games:game.close()
            raise
        self.contexts=weakref.WeakKeyDictionary()
        self.policy_id='masked-ismcts/1.0.0/'+digest(dict(model=self.base.policy_id,config=self.config.to_dict(),opponents=[p.policy_id for p in self.rollouts]))
        self.telemetry=dict(searches=0,simulations=0,incomplete=0,max_pending=0,opponent_choices=0)
        self.last_statistics=None
        self.last_continuation={}
        self.cancelled=Event()

    def cancel(self):self.cancelled.set()

    def close(self):
        self.cancel()
        self.game.close()
        for game in self.simulation_games:game.close()
        for policy in [self.base]+self.rollouts:policy.close()
        self.contexts.clear()
    def context_features(self,view,rng):return self.base.context_features(view,rng)

    def incumbent(self,view,choices,probabilities,rng,recall):
        """Bounded complete current-turn fallback from permitted choices only.

        The first choice uses the full NN prior; cheap legal completion then
        avoids additional model calls. Ordinary future draws are a stop boundary.
        Every execution lookup checks its newly permitted information/legality.
        """
        state=self.game.from_observation(view,rng.randrange(2**63),recall)
        turn=turn_key(view['context']);trace={}
        for step in range(96):
            if state.is_terminal() or state.is_chance_node() or state.current_player()!=view['seat'] or turn_key(state.context)!=turn:break
            legal=state.legal
            require(legal,'Empty incumbent legality')
            if step==0:
                require([c['id'] for c in legal]==[c['id'] for c in choices],'Incumbent root legality changed')
                index=int(np.argmax(probabilities))
            else:
                # Prefer a settled action; merchant beer remains an explicit source.
                index=max(range(len(legal)),key=lambda i:(legal[i]['kind']=='finish',legal[i]['kind']=='beer' and bool(legal[i].get('merchant')),-i))
            identity=legal[index]['id'];trace[node_key(state,view['seat'])]=identity
            state.apply_action(action_id(identity))
        else:raise ValueError('Incumbent completion cap exceeded')
        return trace

    def decide(self,view,choices,rng,deadline,recall=None,tree=None,cache=None):
        # This legal policy incumbent is available even when no tree path finishes.
        try:
            with inference_deadline([self.base],deadline):probabilities,_=self.base.predict([(view,choices,rng)])[0]
        except SearchIncomplete:
            probabilities=np.full(len(choices),1./len(choices))
        fallback=int(np.argmax(probabilities))
        self.last_continuation=self.incumbent(view,choices,probabilities,rng,recall)
        if len(choices)==1 or time.monotonic()>=deadline or self.cancelled.is_set():return fallback,1.,None
        bot=MaskedISMCTS(self.game,self.base,self.rollouts,self.config,view['seat'],
            self.context_features(view,rng),rng,deadline,recall,tree,cache,self.simulation_games,self.cancelled)
        self.telemetry['searches']+=1
        with inference_deadline([self.base]+self.rollouts,deadline):stats=bot.run(view,choices)
        self.last_statistics=stats
        for name,source in [('simulations','completed'),('incomplete','interrupted'),('opponent_choices','opponent_choices')]:self.telemetry[name]+=stats[source]
        self.telemetry['max_pending']=max(self.telemetry['max_pending'],stats['max_pending'])
        selected=max(range(len(choices)),key=lambda i:(stats['root'].get(choices[i]['id'],{}).get('visits',0),probabilities[i],-i))
        teacher=visit_teacher(choices,stats,probabilities,model=self.base.evaluator.model_id,observer=view['seat'],config=self.config)
        if stats['root']:
            self.last_continuation=dict(stats['continuations'].get(choices[selected]['id'],[]))
        return selected if stats['root'] else fallback,1.,teacher

    def select_batch(self,inputs):
        output=[]
        for view,choices,rng in inputs:
            key=turn_key(view['context']);state=self.contexts.get(rng)
            if state is None or view['context']['revision']<=state['revision']:
                state=dict(turn=None,revision=-1,recall=[],turns=0,searches=0,tree={},cache=OrderedDict(),continuation={})
            if state['turn']!=key:
                due=(state['turns'])%self.config.turn_interval==0 and (not self.config.max_searches_per_seat or state['searches']<self.config.max_searches_per_seat)
                state.update(turn=key,deadline=time.monotonic()+self.config.seconds,turns=state['turns']+1,tree={},cache=OrderedDict(),continuation={},due=due)
                if due:state['searches']+=1
            if not state['recall'] or state['recall'][-1]['hand']!=view['own_hand']:
                state['recall'].append(dict(revision=view['context']['revision'],hand=list(view['own_hand'])))
            due=state['due']
            identity=state['continuation'].get(information_key(view,choices,state['recall']))
            if due and time.monotonic()<state['deadline'] and len(choices)>1:
                before=self.telemetry['searches']
                result=self.decide(view,choices,rng,state['deadline'],state['recall'],state['tree'],state['cache'])
                state['continuation']=dict(self.last_continuation)
            elif identity in [c['id'] for c in choices]:result=[c['id'] for c in choices].index(identity),1.,None
            else:
                selected,_,_=self.base.select_batch([(view,choices,rng)])[0];result=selected,1.,None
            state['revision']=view['context']['revision'];self.contexts[rng]=state;output.append(result)
        return output
