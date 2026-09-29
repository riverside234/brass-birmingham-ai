"""POMCP-inspired root sampling and action/observation trees of complete turns.

Opponents are independently motivated, fixed masked policies, not tree branches
optimized using root-private information. No full live state is accepted. This
is a finite-horizon neural extension, not a claim of POMCP convergence for Brass.
"""
from dataclasses import asdict, dataclass
import math
import random

import numpy as np

from brass_ai.learning.contracts import digest, require
from brass_ai.step5.search import SearchPolicy, SearchIncomplete, turn_key


@dataclass(frozen=True)
class TreeConfig:
    simulations: int = 32
    future_turns: int = 1
    exploration: float = 1.4
    particle_limit: int = 32

    def validate(self):
        require(self.simulations>=2 and self.future_turns in (1,2) and self.exploration>0
                and self.particle_limit>0, 'Invalid S3 tree configuration')


def history_key(view, actions):
    # The full permitted view contains own hand and public action recall.
    return digest(dict(view=view, root_turn_plans=actions))


class TurnNode:
    def __init__(self, plans):
        self.plans=plans
        self.counts=np.zeros(len(plans),dtype=np.int64)
        self.sums=np.zeros((len(plans),4))
        self.particles=[]

    def choose(self, seat, exploration, frozen=False):
        if not frozen:
            unseen=np.flatnonzero(self.counts==0)
            if len(unseen):return int(unseen[0])
        means=np.divide(self.sums[:,seat],self.counts,out=np.full(len(self.plans),-np.inf),where=self.counts>0)
        if not frozen:
            means+=exploration*np.sqrt(math.log(max(2,int(self.counts.sum())))/np.maximum(1,self.counts))
        return int(np.argmax(means))

    def backup(self, index, returns):
        require(len(returns)==4 and np.isfinite(returns).all(), 'Four stable-seat values required')
        self.counts[index]+=1;self.sums[index]+=returns


class S3Policy(SearchPolicy):
    def __init__(self, *args, tree=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.tree_config=tree or TreeConfig();self.tree_config.validate()
        self.policy_id='s3-root-observation/1.0.0/'+digest(dict(base=self.policy_id,tree=asdict(self.tree_config)))
        self.telemetry.update(tree_simulations=0,tree_nodes=0,future_root_turns=0,
                              leaf_depths={},unconfirmed_incumbents=0)
        self.last_tree=None

    def play_plan(self, context, plan, deadline):
        for identity in plan:
            self.check(deadline)
            choice=next((c for c in self.host.choices() if c['id']==identity),None)
            require(choice is not None,'Fixed root plan is inconsistent with its observation')
            context=self.host.apply(context,choice)
        return context

    def next_root(self, context, root, rngs, assigned, deadline):
        for _ in range(self.config.max_reply_choices):
            self.check(deadline)
            if context['terminal'] or context['actor']==root:return context
            actor=context['actor'];allowed=self.host.observe(actor);legal=self.host.choices()
            index=assigned[actor].select_batch([(allowed,legal,rngs[actor])])[0][0]
            context=self.host.apply(context,legal[index]);self.telemetry['reply_choices']+=1
        raise SearchIncomplete('next_root_horizon_cap')

    def leaf(self, context, root, root_context, rng, depth):
        depths=self.telemetry['leaf_depths'];depths[str(depth)]=depths.get(str(depth),0)+1
        if context['terminal']:
            return np.array([p['utility'] for p in self.host.call('result')['seats']])
        require(context['actor']==root,'Nonterminal value must be a real root decision')
        self.base.contexts[rng]=root_context
        return np.array(self.base.predict([(self.host.observe(root),self.host.choices(),rng)])[0][1]['win_share'])

    def neural_turn(self, context, root, root_context, rng, deadline):
        key=turn_key(context);self.base.contexts[rng]=root_context
        for _ in range(self.config.max_plan_choices):
            self.check(deadline)
            view=self.host.observe(root);choices=self.host.choices()
            index=self.base.select_batch([(view,choices,rng)])[0][0]
            context=self.host.apply(context,choices[index])
            if context['terminal'] or context['actor']!=root or turn_key(context)!=key:return context
        raise SearchIncomplete('future_root_plan_cap')

    def continuation(self, context, root, root_context, rngs, assigned, tree, path, depth, seed, deadline, frozen):
        context=self.next_root(context,root,rngs,assigned,deadline)
        if context['terminal'] or depth>=self.tree_config.future_turns:
            return self.leaf(context,root,root_context,rngs[root],depth)
        view=self.host.observe(root);choices=self.host.choices();key=history_key(view,path)
        node=tree.get(key)
        if node is None and not frozen:
            # Proposals use a reconstructed world from this allowed observation.
            # Restore the current sampled continuation afterwards.
            cp=self.host.call('checkpoint')
            local=random.Random(digest(dict(history=key,seed=seed)))
            self.base.contexts[local]=root_context
            plans=self.propose(view,choices,local,deadline)
            context=self.host.call('restore',checkpoint=cp)
            node=tree[key]=TurnNode(plans);self.telemetry['tree_nodes']+=1
        if node is None:
            # Freeze confirmation policy: unseen future observations use the same
            # masked neural fallback, never optimize after seeing a hidden world.
            context=self.neural_turn(context,root,root_context,rngs[root],deadline)
            return self.continuation(context,root,root_context,rngs,assigned,tree,path+['neural'],
                                     depth+1,seed,deadline,True)
        index=node.choose(root,self.tree_config.exploration,frozen)
        context=self.play_plan(context,node.plans[index],deadline)
        value=self.continuation(context,root,root_context,rngs,assigned,tree,
            path+[node.plans[index]],depth+1,seed,deadline,frozen)
        if not frozen:
            node.backup(index,value)
            if len(node.particles)<self.tree_config.particle_limit:node.particles.append(seed)
        self.telemetry['future_root_turns']+=1
        return value

    def trajectory(self, view, plan, seed, root_context, tree, deadline, frozen=False):
        context=self.reset_world(view,seed,deadline)
        context=self.play_plan(context,plan,deadline)
        rngs=[random.Random(digest(dict(sample=seed,seat=s))) for s in range(4)]
        assigned=[self.rollouts[random.Random(f'{seed}:{s}:opponent').randrange(len(self.rollouts))] for s in range(4)]
        return self.continuation(context,view['seat'],root_context,rngs,assigned,tree,[plan],0,seed,deadline,frozen)

    def search(self, view, choices, rng, deadline):
        self.telemetry['searches']+=1
        plans=self.propose(view,choices,rng,deadline)
        root=TurnNode(plans);tree={};context=self.context_features(view,rng)
        used=set()
        try:
            for _ in range(self.tree_config.simulations):
                index=root.choose(view['seat'],self.tree_config.exploration)
                seed=rng.randrange(2**63);used.add(seed)
                value=self.trajectory(view,plans[index],seed,context,tree,deadline)
                self.check(deadline)
                root.backup(index,value);self.telemetry['tree_simulations']+=1
            indices=sorted((i for i,n in enumerate(root.counts) if n),
                           key=lambda i:-root.sums[i,view['seat']]/root.counts[i])[:self.config.finalists]
            require(indices,'No completed root simulations')
            finalists=[plans[i] for i in indices]
            seeds=[]
            while len(seeds)<self.config.confirmation_samples:
                seed=rng.randrange(2**63)
                if seed not in used and seed not in seeds:seeds.append(seed)
            samples=[[self.trajectory(view,p,s,context,tree,deadline,True)[view['seat']] for s in seeds] for p in finalists]
            self.check(deadline)
            means=list(map(float,np.mean(samples,axis=1)))
            order=sorted(range(len(means)),key=lambda i:-means[i]);confidence=0.
            if len(order)>1:
                delta=np.asarray(samples[order[0]])-samples[order[1]]
                gap=max(0.,float(delta.mean()));error=float(delta.std(ddof=1)/math.sqrt(len(seeds)))
                confidence=min(.9,gap/(gap+2*error+.01))
            self.telemetry['confirmed']+=1;self.telemetry['fresh_worlds']+=len(seeds)
            return finalists[order[0]],finalists,means,confidence
        except SearchIncomplete:
            self.telemetry['incomplete']+=1;self.telemetry['unconfirmed_incumbents']+=1
            index=root.choose(view['seat'],self.tree_config.exploration,True) if root.counts.sum() else 0
            return plans[index],[plans[index]],[0.],0.
        finally:
            self.last_tree=dict(root_counts=root.counts.tolist(),root_sums=root.sums.tolist(),
                nodes=len(tree),observation_keys=sorted(tree),
                particle_counts={key:len(node.particles) for key,node in tree.items()})

    def select_batch(self, inputs):
        output=super().select_batch(inputs)
        for _,_,teacher in output:
            if teacher is not None:
                teacher['horizon']=f's3/complete-current-turn+{self.tree_config.future_turns}-future-root-turns/observations-v1'
        return output
