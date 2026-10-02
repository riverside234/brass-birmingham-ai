"""Private Masked ISMCTS root forests; merge only completed four-seat evidence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import RLock
import random
import numpy as np
from brass_ai.learning.contracts import require
from brass_ai.search.ismcts import MaskedSearchPolicy,visit_teacher,supported_continuation

def merge_statistics(items,choices):
    merged=dict(completed=0,interrupted=0,opponent_nodes=0,max_pending=0,opponent_choices=0,
                root={},plans={},expansion=[False]*len(choices))
    allowed={c['id'] for c in choices}
    for item in items:
        if item is None:continue
        require(item['opponent_nodes']==0 and set(item['root'])<=allowed,'Unsafe root forest merge')
        for field in ('completed','interrupted','opponent_choices'):merged[field]+=item[field]
        merged['max_pending']=max(merged['max_pending'],item['max_pending'])
        require(len(item['expansion'])==len(choices),'Forest legal order differs')
        merged['expansion']=[a or b for a,b in zip(merged['expansion'],item['expansion'])]
        for identity,e in item['root'].items():
            require(len(e['returns'])==4 and e['visits']>0,'Incomplete or mis-seated edge')
            target=merged['root'].setdefault(identity,dict(visits=0,returns=np.zeros(4),available=0))
            target['visits']+=e['visits'];target['returns']+=np.asarray(e['returns']);target['available']+=e['available']
        for signature,e in item.get('plans',{}).items():
            require(e['first'] in allowed and e['visits']>0 and len(e['returns'])==4,'Invalid completed continuation')
            from brass_ai.learning.contracts import digest
            require(signature==e['signature']==digest(e['trace']) and e['trace'][0][1]==e['first'],'Continuation identity changed')
            target=merged['plans'].setdefault(signature,dict(signature=signature,first=e['first'],trace=e['trace'],visits=0,returns=np.zeros(4)))
            target['visits']+=e['visits'];target['returns']+=np.asarray(e['returns'])
    for group in ('root','plans'):
        for e in merged[group].values():e['returns']=e['returns'].tolist()
    return merged

class ParallelMasked(MaskedSearchPolicy):
    def configure(self,workers,role,factory):
        require(1<=workers<=16 and role in ('lab','laptop','reference'),'Invalid CPU root forest')
        self.replicas=[]
        try:
            for i in range(workers):
                replica=factory(i);replica.partition_index=i;self.replicas.append(replica)
        except BaseException:
            for p in self.replicas:p.close()
            raise
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='masked-private-root')
        self.decision_lock=RLock()
        from brass_ai.learning.contracts import digest
        self.config=replace(self.config,workers=workers,cpu_role=role)
        self.policy_id='masked-ismcts-forest/1.1.0/'+digest(dict(model=self.base.policy_id,config=self.config.to_dict()))
        self.telemetry.update(search_workers=workers,cpu_role=role)
        return self

    def decide(self,view,choices,rng,deadline,recall=None,tree=None,cache=None):
        with self.decision_lock:
            return self._decide(view,choices,rng,deadline,recall,tree,cache)

    def _decide(self,view,choices,rng,deadline,recall=None,tree=None,cache=None):
        from brass_ai.step5.policy import inference_deadline
        from brass_ai.search.common import SearchIncomplete
        try:
            with inference_deadline([self.base],deadline):probabilities,_=self.base.predict([(view,choices,rng)])[0]
        except SearchIncomplete:
            probabilities=np.full(len(choices),1./len(choices))
        self.last_continuation=self.incumbent(view,choices,probabilities,rng,recall)
        seeds=[rng.randrange(2**63) for _ in self.replicas];context=self.context_features(view,rng)
        private_trees=(tree if tree is not None else {}).setdefault('__private_forest__',{})
        private_caches=(cache if cache is not None else {}).setdefault('__private_forest__',{})
        def run(index):
            from collections import OrderedDict
            worker=self.replicas[index];private=random.Random(seeds[index]);worker.base.contexts[private]=context
            worker.last_statistics=None
            worker.decide(view,choices,private,deadline,recall,private_trees.setdefault(index,{}),private_caches.setdefault(index,OrderedDict()))
            return worker.last_statistics
        # Await actual retirement of ALL workers, including exceptions. No worker
        # continues to use a reply channel after the actor submits its next request.
        pending=[self.pool.submit(run,i) for i in range(len(self.replicas))]
        results=[];failure=None
        for future in pending:
            try:results.append(future.result())
            except BaseException as e:failure=failure or e
        if failure:raise failure
        stats=merge_statistics(results,choices);self.last_statistics=stats
        self.telemetry['searches']+=1;self.telemetry['simulations']+=stats['completed'];self.telemetry['incomplete']+=stats['interrupted']
        self.telemetry['max_pending']=max(self.telemetry['max_pending'],stats['max_pending'])
        chosen=max(range(len(choices)),key=lambda i:(stats['root'].get(choices[i]['id'],{}).get('visits',0),probabilities[i],-i))
        continuation=supported_continuation(stats,choices[chosen]['id'],view['seat'],self.config.minimum_plan_visits)
        if continuation is not None:self.last_continuation=continuation
        else:chosen=int(np.argmax(probabilities))
        return chosen,1.,visit_teacher(choices,stats,probabilities,model=self.base.evaluator.model_id,observer=view['seat'],config=self.config)

    def close(self):
        self.cancel()
        for worker in self.replicas:worker.cancel()
        self.pool.shutdown(wait=True)
        for worker in self.replicas:worker.close()
        super().close()

def parallelize(policy,workers,role,factory):
    if workers==1:return policy
    policy.__class__=ParallelMasked
    return policy.configure(workers,role,factory)
