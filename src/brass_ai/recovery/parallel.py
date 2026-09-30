"""Persistent private search replicas; CPU allowance is distinct from game count.

S2 distributes fixed root proposals and matched scenario batches. S3 distributes
four-seat UCT simulations with private observation trees, then confirms finalists
on fresh matched worlds. R4 merges public-root visits only. Neither raw visits nor
partial batches become distillation labels. All work uses one inherited turn clock.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import random
import time

import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.step5.policy import inference_deadline
from brass_ai.step5.search import SearchPolicy,SearchIncomplete,first_choices
from brass_ai.searchlab.s3 import S3Policy,TurnNode


def confidence(samples):
    means=list(map(float,np.mean(samples,axis=1)));order=sorted(range(len(means)),key=lambda i:(-means[i],i))
    value=0.
    if len(order)>1:
        delta=np.asarray(samples[order[0]])-samples[order[1]]
        gap=max(0.,float(delta.mean()));error=float(delta.std(ddof=1)/math.sqrt(len(delta)))
        value=min(.9,gap/(gap+2*error+.01))
    return means,order,value


class _Parallel:
    def configure(self,workers,role,factory):
        require(type(workers) is int and 2<=workers<=32 and role in ('lab','laptop','reference'),'Invalid search CPU role')
        self.replicas=[]
        try:
            for i in range(workers): self.replicas.append(factory(i))
        except BaseException:
            for p in self.replicas: p.close()
            super().close();raise
        self.parallel_pool=ThreadPoolExecutor(max_workers=workers)
        self.policy_id='parallel-search/1.0.0/'+digest(dict(serial=self.policy_id,workers=workers,role=role))
        self.telemetry.update(search_workers=workers,role=role,parallel_batches=0)
        return self

    def map_private(self,items,function,deadline):
        """One bounded lane per replica; drain all lanes even after a timeout."""
        items=list(items);pending=[]
        def lane(index):
            worker=self.replicas[index];out=[]
            with inference_deadline([worker.base]+worker.rollouts,deadline):
                for number in range(index,len(items),len(self.replicas)):
                    SearchPolicy.check(deadline);out.append((number,function(worker,items[number])))
            return out
        for i in range(min(len(items),len(self.replicas))): pending.append(self.parallel_pool.submit(lane,i))
        output={};error=None
        for future in pending:
            try: output.update(future.result())
            except BaseException as failure:
                if error is None: error=failure
        if error is not None: raise error
        SearchPolicy.check(deadline);self.telemetry['parallel_batches']+=1
        return [output[i] for i in range(len(items))]

    def close(self):
        self.parallel_pool.shutdown(wait=True)
        for p in self.replicas: p.close()
        super().close()

    def matrix(self,view,plans,seeds,context,deadline,*,trees=None):
        def batch(worker,job):
            index,seed=job
            if trees is None: return [worker.scenario(view,p,seed,context,deadline) for p in plans]
            # Every plan in a world uses the SAME frozen continuation tree.
            tree=trees[index%len(self.replicas)]
            return [worker.trajectory(view,p,seed,context,tree,deadline,True)[view['seat']] for p in plans]
        values=self.map_private(enumerate(seeds),batch,deadline)
        return np.asarray(values).T.tolist()


class ParallelS2(_Parallel,SearchPolicy):
    def propose(self,view,choices,rng,deadline):
        self.check(deadline);p,_=self.base.predict([(view,choices,rng)])[0]
        indices=first_choices(choices,p,self.config.proposals,rng);setup_seed=rng.randrange(2**31)
        context=self.context_features(view,rng)
        items=[(i,rng.randrange(2**63)) for i in indices]
        plans=self.map_private(items,lambda w,j:w.propose_one(view,choices,j[0],setup_seed,j[1],context,deadline),deadline)
        plans=list({tuple(p):p for p in plans}.values());self.telemetry['proposed']+=len(plans)
        return plans

    def search(self,view,choices,rng,deadline):
        self.telemetry['searches']+=1;plans=self.propose(view,choices,rng,deadline)
        require(plans,'No complete root plan');context=self.context_features(view,rng)
        pilot=[rng.randrange(2**63) for _ in range(self.config.pilot_samples)]
        scores=np.mean(self.matrix(view,plans,pilot,context,deadline),axis=1)
        indices=sorted(range(len(plans)),key=lambda i:(-scores[i],plans[i]))[:self.config.finalists]
        finalists=[plans[i] for i in indices];fresh=[]
        while len(fresh)<self.config.confirmation_samples:
            seed=rng.randrange(2**63)
            if seed not in pilot and seed not in fresh: fresh.append(seed)
        samples=self.matrix(view,finalists,fresh,context,deadline);means,order,value=confidence(samples)
        self.telemetry['pilot_worlds']+=len(pilot);self.telemetry['fresh_worlds']+=len(fresh);self.telemetry['confirmed']+=1
        return finalists[order[0]],finalists,means,value


class ParallelS3(_Parallel,S3Policy):
    propose=ParallelS2.propose

    def search(self,view,choices,rng,deadline):
        self.telemetry['searches']+=1;plans=self.propose(view,choices,rng,deadline)
        require(plans,'No complete root plan');root=TurnNode(plans);trees=[{} for _ in self.replicas]
        context=self.context_features(view,rng);used=set();completed=0
        # Explicitly reserve confirmation time; partial exploration stays private.
        explore_deadline=time.monotonic()+max(0.,deadline-time.monotonic())*.65
        try:
            while completed<self.tree_config.simulations and time.monotonic()<explore_deadline:
                jobs=[]
                for i in range(min(len(self.replicas),self.tree_config.simulations-completed)):
                    index=root.choose(view['seat'],self.tree_config.exploration)
                    seed=rng.randrange(2**63);used.add(seed);jobs.append((i,index,seed))
                    root.counts[index]+=1  # Virtual reservations; never a value backup.
                try:
                    values=self.map_private(jobs,lambda w,j:w.trajectory(view,plans[j[1]],j[2],context,trees[j[0]],explore_deadline),explore_deadline)
                except SearchIncomplete:
                    for _,index,_ in jobs: root.counts[index]-=1
                    break
                for (_,index,_),value in zip(jobs,values): root.counts[index]-=1;root.backup(index,value)
                completed+=len(jobs)
            self.telemetry['tree_simulations']+=completed
            indices=sorted((i for i,n in enumerate(root.counts) if n),
                key=lambda i:(-root.sums[i,view['seat']]/root.counts[i],plans[i]))[:self.config.finalists]
            if not indices: raise SearchIncomplete('no_completed_root_simulation')
            finalists=[plans[i] for i in indices];fresh=[]
            while len(fresh)<self.config.confirmation_samples:
                seed=rng.randrange(2**63)
                if seed not in used and seed not in fresh: fresh.append(seed)
            samples=self.matrix(view,finalists,fresh,context,deadline,trees=trees)
            means,order,value=confidence(samples)
            self.telemetry['confirmed']+=1;self.telemetry['fresh_worlds']+=len(fresh)
            return finalists[order[0]],finalists,means,value
        except SearchIncomplete:
            self.telemetry['incomplete']+=1;self.telemetry['unconfirmed_incumbents']+=1
            index=root.choose(view['seat'],self.tree_config.exploration,True) if root.counts.sum() else 0
            return plans[index],[plans[index]],[0.],0.
        finally:
            self.last_tree=dict(root_counts=root.counts.tolist(),root_sums=root.sums.tolist(),
                nodes=sum(map(len,trees)),private_observation_trees=True,search_workers=len(self.replicas),
                particle_counts=[{k:len(n.particles) for k,n in t.items()} for t in trees])


def parallelize(policy,kind,workers,role,factory):
    if workers==1: return policy
    if kind=='s2': policy.__class__=ParallelS2
    elif kind=='s3': policy.__class__=ParallelS3
    elif kind=='r4':
        from brass_ai.searchlab.ismcts import OpenSpielPolicy
        from brass_ai.searchlab.spiel import action_id
        class ParallelR4(_Parallel,OpenSpielPolicy):
            def decide(self,view,choices,rng,deadline,recall=None):
                fallback=self.base.select_batch([(view,choices,rng)])[0]
                if len(choices)==1 or time.monotonic()>=deadline: return fallback
                seeds=[rng.randrange(2**63) for _ in self.replicas];context=self.context_features(view,rng)
                for worker in self.replicas: worker.last_statistics=None
                def run(worker,seed):
                    private=random.Random(seed);worker.base.contexts[private]=context;worker.last_statistics=None
                    worker.decide(view,choices,private,deadline,recall)
                    return worker.last_statistics
                # Individual ISMCTS workers can stop with completed root visits.
                try: results=self.map_private(seeds,run,deadline)
                except SearchIncomplete:
                    results=[p.last_statistics for p in self.replicas]
                root={};completed=0
                for item in results:
                    if not item: continue
                    require(item['opponent_nodes']==0,'Private opponent nodes cannot be merged')
                    completed+=item['completed']
                    for action,stats in item['root'].items():
                        value=root.setdefault(action,dict(visits=0,returns=0.))
                        value['visits']+=stats['visits'];value['returns']+=stats['returns']
                self.last_statistics=dict(completed=completed,root=root,opponent_nodes=0)
                self.telemetry['searches']+=1;self.telemetry['simulations']+=completed
                available=[(i,root[str(action_id(c['id']))]) for i,c in enumerate(choices) if str(action_id(c['id'])) in root]
                if not available: return fallback
                selected=max(available,key=lambda p:(p[1]['visits'],p[1]['returns']/p[1]['visits'],-p[0]))[0]
                return selected,1.,None
        policy.__class__=ParallelR4
    else: raise ValueError('Unknown parallel search')
    return policy.configure(workers,role,factory)
