# Copyright 2019 DeepMind Technologies Limited (OpenSpiel ISMCTS mechanics).
# Project modifications: masked opponent transitions, transactional backups,
# deadlines, bounded horizons and permitted-input policy wrapper.
# Licensed under the Apache License, Version 2.0; see NOTICE.md and LICENSE-OpenSpiel.
"""OpenSpiel ISMCTS comparison with fixed, independently masked opponents.

Unrestricted opponent-node optimization is deliberately not used: even masked
network inputs would let those node statistics condition on root-private cards.
OpenSpiel's node/PUCT/prior/resampling interfaces are used for the acting root.
"""
from dataclasses import asdict, dataclass
import math
import random
import time
import weakref

import numpy as np
from open_spiel.python.algorithms import ismcts

from brass_ai.learning.contracts import digest, require
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.step5.search import SearchIncomplete, first_choices, turn_key
from .spiel import BrassSpielGame, action_id


@dataclass(frozen=True)
class SpielSearchConfig:
    seconds: float = 3.
    simulations: int = 32
    future_turns: int = 1
    proposals: int = 12
    max_choices: int = 512
    exploration: float = 1.4
    turn_interval: int = 1

    def validate(self):
        require(self.seconds>0 and self.simulations>=2 and self.future_turns in (1,2)
                and self.proposals>=2 and self.max_choices>0 and self.exploration>0
                and self.turn_interval>0, 'Invalid OpenSpiel search config')


class PermittedEvaluator:
    def __init__(self, policy, context, rng):
        self.policy,self.context,self.rng=policy,context,rng

    def predict(self,state):
        actor=state.current_player()
        require(actor>=0,'No neural evaluation of chance/full-state data')
        self.policy.contexts[self.rng]=self.context
        return self.policy.predict([(state.views[actor],state.legal,self.rng)])[0]

    def prior(self,state):
        probabilities,_=self.predict(state)
        return [(action_id(c['id']),float(p)) for c,p in zip(state.legal,probabilities)]

    def evaluate(self,state): return self.predict(state)[1]['win_share']


class MaskedISMCTS(ismcts.ISMCTSBot):
    def __init__(self,game,evaluator,root,rollouts,config,rng,deadline):
        super().__init__(game,evaluator,config.exploration,config.simulations,
            random_state=np.random.RandomState(rng.randrange(2**31)),
            child_selection_policy=ismcts.ChildSelectionPolicy.PUCT,
            final_policy_type=ismcts.ISMCTSFinalPolicyType.MAX_VISIT_COUNT)
        self.root,self.rollouts,self.config,self.deadline=root,rollouts,config,deadline
        self.rng=rng;self.completed=0;self.opponent_choices=0
        # OpenSpiel's default UniformProbabilitySampler owns a separate RNG.
        # Bind all belief draws to this experiment's recorded search RNG.
        self.set_resampler(lambda state,player:state.resample_from_infostate(player,self.random_number))
        self.opponent_statistics={}  # Kept empty: only masked fixed policies act there.

    def check(self):
        if time.monotonic()>=self.deadline:raise SearchIncomplete('whole_turn_deadline')

    def run_search(self,state):
        self.reset();self._root_node=self.create_new_node(state)
        key=self.get_state_key(state)
        for _ in range(self.config.simulations):
            self.check()
            sample=self.sample_root_state(state)
            require(key==self.get_state_key(sample),'Resampling changed root information')
            seed=self.rng.randrange(2**63)
            self.rngs=[random.Random(digest(dict(seed=seed,seat=s))) for s in range(4)]
            self.assigned=[self.rollouts[random.Random(f'{seed}:{s}:opponent').randrange(len(self.rollouts))] for s in range(4)]
            self.simulate(sample,turn_key(sample.context),0,0)
            self.completed+=1
        return self.safe_policy(state)

    def safe_policy(self,state):
        children={a:c for a,c in self._root_node.child_info.items() if c.visits>0}
        if not children:return []
        maximum=max(c.visits for c in children.values())
        selected=[a for a,c in children.items() if c.visits==maximum]
        # Same visit count: use the root's own return, never a pooled bot score.
        best=max(c.value() for a,c in children.items() if a in selected)
        selected=[a for a in selected if children[a].value()>=best-1e-9]
        return [(a,1/len(selected) if a in selected else 0.) for a in state.legal_actions()]

    def simulate(self,state,last_root_turn,depth,steps):
        self.check()
        if state.is_terminal():return state.returns()
        if steps>=self.config.max_choices:raise SearchIncomplete('simulation_choice_cap')
        if state.is_chance_node():
            actions,probabilities=zip(*state.chance_outcomes())
            state.apply_action(int(self._random_state.choice(actions,p=probabilities)))
            return self.simulate(state,last_root_turn,depth,steps+1)
        actor=state.current_player()
        if actor!=self.root:
            view,choices=state.views[actor],state.legal
            chosen=self.assigned[actor].select_batch([(view,choices,self.rngs[actor])])[0][0]
            state.apply_action(action_id(choices[chosen]['id']));self.opponent_choices+=1
            return self.simulate(state,last_root_turn,depth,steps+1)
        current=turn_key(state.context)
        if current!=last_root_turn:depth+=1;last_root_turn=current
        if depth>self.config.future_turns:
            values=self._evaluator.evaluate(state);self.check();return values
        node=self.lookup_or_create_node(state)
        if node.total_visits==ismcts.UNEXPANDED_VISIT_COUNT:
            prior=dict(self._evaluator.prior(state));value=self._evaluator.evaluate(state)
            self.check();node.prior_map=prior;node.total_visits=0
            return value
        probabilities=[node.prior_map[action_id(c['id'])] for c in state.legal]
        options=[action_id(state.legal[i]['id']) for i in first_choices(state.legal,probabilities,self.config.proposals,self.rngs[actor])]
        unseen=[a for a in options if a not in node.child_info or node.child_info[a].visits==0]
        if unseen:
            action=max(unseen,key=lambda a:(node.prior_map[a],-a))
        else:
            action=max(options,key=lambda a:self._action_value(node,node.child_info[a]))
        self.expand_if_necessary(state,node,action)
        state.apply_action(action)
        values=self.simulate(state,last_root_turn,depth,steps+1)
        self.check()
        require(len(values)==4 and np.isfinite(values).all(),'Four-seat backup required')
        # Commit only a fully completed simulation. An interrupted path cannot
        # inflate visits or inject an artificial zero-return loss.
        node.total_visits+=1;child=node.child_info[action]
        child.visits+=1;child.return_sum+=values[actor]
        return values


class OpenSpielPolicy:
    kind='external'
    def __init__(self,model,root,device='cpu',*,config=None,mono=None,focus='neutral',opponents=None):
        self.config=config or SpielSearchConfig();self.config.validate()
        self.base=NeuralPolicy(model,device,exploration=.1,focus=focus)
        self.rollouts=opponents or [NeuralPolicy(model,device,exploration=.05,focus='neutral')]
        require(all(isinstance(p,NeuralPolicy) and p.focus=='neutral' for p in self.rollouts),'Frozen masked neutral rollout models required')
        self.game=BrassSpielGame(root=root,mono=mono)
        self.contexts=weakref.WeakKeyDictionary()
        self.policy_id='r4-openspiel-ismcts/1.0.0/'+digest(dict(model=self.base.policy_id,
            config=asdict(self.config),opponents=[p.policy_id for p in self.rollouts]))
        self.telemetry=dict(searches=0,simulations=0,incomplete=0,opponent_choices=0)
        self.last_statistics=None

    def close(self):self.game.close()
    def context_features(self,view,rng):return self.base.context_features(view,rng)

    def decide(self,view,choices,rng,deadline,recall=None):
        fallback=self.base.select_batch([(view,choices,rng)])[0]
        if len(choices)==1 or time.monotonic()>=deadline:return fallback
        state=self.game.from_observation(view,rng.randrange(2**63),recall)
        evaluator=PermittedEvaluator(self.base,self.context_features(view,rng),random.Random(rng.randrange(2**63)))
        bot=MaskedISMCTS(self.game,evaluator,view['seat'],self.rollouts,self.config,rng,deadline)
        self.telemetry['searches']+=1
        try:policy=bot.run_search(state)
        except SearchIncomplete:
            self.telemetry['incomplete']+=1
            policy=bot.safe_policy(state) if hasattr(bot,'_root_node') else []
        self.telemetry['simulations']+=bot.completed;self.telemetry['opponent_choices']+=bot.opponent_choices
        self.last_statistics=dict(completed=bot.completed,root={str(a):dict(visits=c.visits,returns=c.return_sum)
            for a,c in sorted(bot._root_node.child_info.items()) if c.visits},opponent_nodes=len(bot.opponent_statistics))
        if not policy:return fallback
        actions,weights=zip(*policy);selected=rng.choices(actions,weights=weights)[0]
        index=next(i for i,c in enumerate(choices) if action_id(c['id'])==selected)
        return index,1.,None

    def select_batch(self,inputs):
        output=[]
        for view,choices,rng in inputs:
            key=turn_key(view['context']);state=self.contexts.get(rng)
            if state is None or view['context']['revision']<=state['revision']:
                state=dict(turn=None,revision=-1,recall=[],turns=0)
            if state['turn']!=key:
                state.update(turn=key,deadline=time.monotonic()+self.config.seconds,turns=state['turns']+1)
            if not state['recall'] or state['recall'][-1]['hand']!=view['own_hand']:
                state['recall'].append(dict(revision=view['context']['revision'],hand=list(view['own_hand'])))
            if (state['turns']-1)%self.config.turn_interval:
                result=self.base.select_batch([(view,choices,rng)])[0]
            else:result=self.decide(view,choices,rng,state['deadline'],state['recall'])
            state['revision']=view['context']['revision'];self.contexts[rng]=state;output.append(result)
        return output
