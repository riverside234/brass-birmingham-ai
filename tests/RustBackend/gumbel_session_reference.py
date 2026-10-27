"""Independent, scalar G_py diagnostic for timed two-owner decision roots.

Uses the admitted simulator protocol and Python feature encoder. It deliberately
does not call Rust's search, scheduler or merge. Cap=pending=1, flush=0 only.
This is a reference implementation, never a deployed/training backend. The
pair retains its original deadline and checks exact action-two preconditions.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import math
import runpy
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
gp=runpy.run_path(str(ROOT/'tests/RustBackend/gumbel_brass_reference.py'))
ref=gp['ref']
information,key,canonical=(gp[k] for k in ('information','key','canonical'))


class Draws:
    """ChaCha12 with the pinned rand_core 0.6 PCG seed expansion.

    Verified against native common perturbations and complete fixed-count
    snapshots; equal integer seeds alone are not evidence of equivalent RNGs.
    """
    def __init__(self,seed):
        mask=(1<<32)-1; words=[]
        for _ in range(8):
            seed=(seed*6364136223846793005+11634580027462260723)&((1<<64)-1)
            x=(((seed>>18)^seed)>>27)&mask; n=seed>>59
            words.append(((x>>n)|(x<<((-n)&31)))&mask)
        self.state=[0x61707865,0x3320646e,0x79622d32,0x6b206574,*words,0,0,0,0]
        self.buffer=[]

    def word(self):
        if not self.buffer:
            x=self.state.copy(); mask=(1<<32)-1
            def qr(a,b,c,d):
                def rot(v,n):return ((v<<n)|(v>>(32-n)))&mask
                x[a]=(x[a]+x[b])&mask;x[d]=rot(x[d]^x[a],16)
                x[c]=(x[c]+x[d])&mask;x[b]=rot(x[b]^x[c],12)
                x[a]=(x[a]+x[b])&mask;x[d]=rot(x[d]^x[a],8)
                x[c]=(x[c]+x[d])&mask;x[b]=rot(x[b]^x[c],7)
            for _ in range(6):
                for q in ((0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15),
                          (0,5,10,15),(1,6,11,12),(2,7,8,13),(3,4,9,14)):qr(*q)
            self.buffer=list(reversed([(a+b)&mask for a,b in zip(x,self.state)]))
            self.state[12]=(self.state[12]+1)&mask
            if self.state[12]==0:self.state[13]=(self.state[13]+1)&mask
        return self.buffer.pop()

    def uniform(self):
        n=self.word()+(self.word()<<32)
        return ((n>>12)+.5)/4503599627370496.

    def categorical(self,outcomes):
        target=self.uniform()*sum(p for _,p in outcomes);c=0
        for action,p in outcomes:
            c+=p
            if target<c:return action
        return next(a for a,p in reversed(outcomes) if p>0)


class Expired(Exception):pass


def turn(info):
    c=info['key']['information']['observation']['view']['context']
    return tuple(c[k] for k in ('era','round','turn','turn_owner'))+(info['actor'],c['phase']=='debt')


class Session:
    def __init__(self,worker,lane,evaluator):
        self.worker,self.lane,self.evaluator=worker,lane,evaluator
        self.cache={};self.costs={};self.epoch=0;self.continuation=[]

    def call(self,kind,**kw):return self.worker.simulation(self.lane,kind,**kw)

    def check(self,delivery=False):
        if self.cancel.is_set() or time.perf_counter_ns()>=(self.deadline if delivery else self.work_deadline):raise Expired()

    def evaluate(self,info):
        self.check();k=key(info)
        if k in self.cache:return self.cache[k]
        answer=self.evaluator(info);self.check()
        if len(self.cache)==128:self.cache.pop(next(iter(self.cache)))
        self.cache[k]=answer
        return answer

    def sample(self,zero=False):
        random=(dict(kind='recorded',indices=[0]*self.draw_count) if zero else
                dict(kind='chacha12',seed=self.input['sample_seed'],stream=self.samples))
        row=self.call('sample',random=random)
        if not zero:self.samples+=1
        return row['handle'],row['snapshot']

    def start(self,input,config,deadline,cancel):
        from brass_ai.learning.reward import CostLedger
        self.input=input;self.config=config;self.deadline=deadline;self.cancel=cancel
        # The caller supplies one original turn deadline, never a fresh budget.
        self.work_deadline=deadline-min(self.allowance//10,1_200_000_000)
        self.check(True)
        self.draw_count=self.call('prepare',view=input['view'],recall=input['recall'])['draw_count']
        self.ledger=CostLedger(input['reward'],**input['ledger']);self.ledger.observe(input['view'])
        assert list(self.ledger.remaining)==input['context']['remaining']
        self.samples=0
        h,s=self.sample(True)
        try:self.root=information(s,input['context'],self.ledger)
        finally:self.call('release',handle=h)
        assert self.root['key']['information']['observation']['view']==input['view']
        self.check(True)

    def completion(self,first,actions=None):
        """Independent semantic replay, including every action-two precondition."""
        from brass_ai.learning.reward import CostLedger
        h,s=self.sample(True);ledger=CostLedger(self.input['reward'],**self.ledger.snapshot())
        result=[];steps=[];initial=turn(self.root)
        try:
            supplied=[st for a in actions for st in a['action']['steps']] if actions else None
            for n in range(192 if supplied else 96):
                self.check(True)
                ledger.observe(s['information'][self.input['view']['seat']]['observation']['view'])
                info=information(s,self.input['context'],ledger);assert turn(info)==initial
                choices=info['key']['choices']
                if supplied:
                    st=supplied[n];assert st['precondition']==info
                    choice=next(c for c in choices if c['id']==st['choice'])
                elif n==0:choice=next(c for c in choices if c['id']==first)
                else:choice=min(choices,key=lambda c:(c['kind']!='finish',not(c['kind']=='beer' and c.get('merchant') is not None),c['id']))
                closes=choice['kind']=='finish' or (initial[-1] and choice['kind']=='liquidate')
                steps.append(dict(precondition=info,choice=choice['id']))
                ledger.chosen(info['key']['information']['observation']['view'],choice)
                s=self.call('step',handle=h,revision=s['context']['revision'],semantic=choice['id'])['snapshot']
                self.check(True)
                if closes:
                    result.append(dict(action=dict(steps=steps),support=0,returns=[0.]*4));steps=[]
                    if not supplied or n+1==len(supplied):return result
            raise AssertionError('bounded completion did not close')
        finally:self.call('release',handle=h)

    def decide(self,shared,mode):
        from brass_ai.learning.reward import CostLedger,terminal_utility
        cfg=self.config;info=self.root;self.epoch+=1
        for i,a in enumerate(self.continuation):
            for j,st in enumerate(a['action']['steps']):
                if st['precondition']==info:
                    actions=copy.deepcopy(self.continuation[i:]);actions[0]['action']['steps']=actions[0]['action']['steps'][j:]
                    self.completion(None,actions);return dict(actions=actions,completed=0,provenance='continuation')
        self.continuation=[]
        if len(info['actions'])==1 or info['key']['information']['observation']['view']['context']['phase']=='debt':
            return dict(actions=self.completion(info['actions'][0]),completed=0,provenance='forced_or_debt',snapshot=None,eligible_plans=[])
        if shared['stats'] is None:
            return dict(actions=self.completion(info['actions'][0]),completed=0,provenance='bounded_legal_fallback',snapshot=None,eligible_plans=[])
        logits,value=shared['stats'];g=shared['gumbels'];draws=Draws(cfg['seed']+self.epoch-1)
        for _ in info['actions']:draws.uniform()
        def node(info,stats):return ref.Node(ref.Info(info['actor'],(),(),tuple(info['actions'])),*stats,4)
        root=node(info,(logits,value));nodes=[root];infos=[info];lookup={key(info):0}
        bucket=(info['key']['information']['observation']['view']['context']['phase'],str(info['key']['context']))
        costs=self.costs.setdefault(bucket,[]);estimate=(mode.get('startup') or {}).get('estimate',{}).get('nanoseconds')
        if len(costs)>=4:estimate=sorted(costs)[(len(costs)-1)//4]
        available=max(0,self.work_deadline-time.perf_counter_ns())
        budget=cfg['simulations'] if mode['kind']=='count' else (available//estimate if estimate else 0)
        k=ref.effective_k(cfg['k'],len(logits),budget)
        ranking=lambda choices,s:sorted(choices,key=lambda i:(-s[i]-g[i],i))
        survivors=sorted(ranking(range(len(logits)),logits)[:k]);completed=attempted=round_done=rounds=remainder=0
        snapshot=None;entries={}
        while completed<budget:
            if time.perf_counter_ns()>=self.work_deadline or self.cancel.is_set():break
            if mode['kind']=='timed' and estimate and self.work_deadline-time.perf_counter_ns()<estimate:break
            edge=survivors[round_done%len(survivors)];began=time.perf_counter_ns();attempted+=1
            h,s=self.sample();ledger=CostLedger(self.input['reward'],**self.ledger.snapshot())
            path=[];spent=[];prefix=[];steps=[];fresh=None
            try:
                assert information(s,self.input['context'],ledger)==info
                for _ in range(cfg['max_depth']):
                    self.check()
                    if not s['chance']:ledger.observe(s['information'][self.input['view']['seat']]['observation']['view'])
                    if s['result']:value=terminal_utility(s['result'],self.input['reward']).tolist();break
                    if s['chance']:
                        action=draws.categorical(sorted(('chance:card:'+c['card'],c['probability']) for c in s['chance']['outcomes']))
                    else:
                        current=information(s,self.input['context'],ledger);ident=key(current)
                        if ident not in lookup:
                            answer=self.evaluate(current);fresh=(current,answer);value=list(answer[1]);break
                        ni=lookup[ident];n=nodes[ni];e=edge if ni==0 else n.select(cfg['c1'],cfg['c2'])
                        path.append((ni,e));spent.append(ledger.spent.copy());action=current['actions'][e]
                        choice=current['key']['choices'][e]
                        # Only consecutive complete prefixes in the root turn.
                        if len(prefix)<2 and steps is not None:
                            if turn(current)!=turn(info) or choice['kind']=='liquidate':steps=None
                            else:
                                steps.append(dict(precondition=current,choice=action))
                                if choice['kind']=='finish':prefix.append(dict(steps=steps));steps=[]
                        ledger.chosen(current['key']['information']['observation']['view'],choice)
                    s=self.call('step',handle=h,revision=s['context']['revision'],semantic=action)['snapshot']
                else:raise AssertionError('depth guard')
                self.check()
                returns=[[value[seat]-(ledger.spent[seat]-prior[seat]) for seat in range(4)] for prior in spent]
                # No late evaluation/tree/ledger publication.
                self.check()
                old={ni:(nodes[ni].count.copy(),copy.deepcopy(nodes[ni].returns)) for ni,_ in path}
                if fresh:
                    assert len(nodes)<cfg['max_nodes'] and sum(len(n.count) for n in nodes)+len(fresh[0]['actions'])<=cfg['max_edges']
                    lookup[key(fresh[0])]=len(nodes);infos.append(fresh[0]);nodes.append(node(*fresh))
                for (ni,e),v in zip(path,returns):
                    nodes[ni].count[e]+=1
                    for seat in range(4):nodes[ni].returns[e][seat]+=v[seat]
                try:self.check()
                except Expired:
                    for ni,(counts,totals) in old.items():nodes[ni].count=counts;nodes[ni].returns=totals
                    if fresh:del lookup[key(fresh[0])];nodes.pop();infos.pop()
                    raise
                completed+=1;round_done+=1
                for depth in range(1,len(prefix)+1):
                    p=prefix[:depth];ident=str(p)
                    row=entries.setdefault(ident,dict(prefix=p,contributions=[]));row['contributions'].append((completed,returns[0]))
                balanced=False
                if len(survivors)>1:
                    quota=budget//(len(survivors)*(k.bit_length()-1))
                    if round_done==len(survivors)*quota:
                        survivors=sorted(ranking(survivors,root.scores(cfg['c1'],cfg['c2']))[:len(survivors)//2])
                        rounds+=1;round_done=0;balanced=True
                else:remainder+=1
                if completed==budget or k==1 or balanced:
                    snapshot=dict(stats=dict(logits=list(root.logits),value=list(root.value),count=root.count.copy(),returns=copy.deepcopy(root.returns)),
                        survivors=survivors.copy(),completed_rounds=rounds,support_watermark=completed,remainder_simulations=remainder,
                        reason='quota_complete' if completed==budget else 'single_candidate' if k==1 else 'balanced_round')
                costs.append(time.perf_counter_ns()-began);costs[:]=costs[-8:]
            except Expired:break
            finally:self.call('release',handle=h)
        if snapshot:
            rn=node(info,(logits,shared['stats'][1]));rn.count=snapshot['stats']['count'];rn.returns=snapshot['stats']['returns']
            rank=ranking(snapshot['survivors'],rn.scores(cfg['c1'],cfg['c2']))
        else:rank=ranking(range(len(logits)),logits)
        plans=[]
        for edge in rank:
            chosen=[];actions=[]
            for depth in (1,2):
                options=[]
                for row in entries.values():
                    p=row['prefix']
                    if len(p)!=depth or p[:-1]!=chosen or p[0]['steps'][0]['choice']!=info['actions'][edge]:continue
                    vals=[v for serial,v in row['contributions'] if serial<=snapshot['support_watermark']]
                    if len(vals)<2:continue
                    total=[sum(v[s] for v in vals) for s in range(4)]
                    options.append((p,len(vals),total))
                if not options:break
                p,n,v=min(options,key=lambda x:(-x[1],-x[2][info['actor']]/x[1],[st['choice'] for a in x[0] for st in a['steps']]))
                chosen=p;actions.append(dict(action=p[-1],support=n,returns=v))
            if actions:plans.append(actions)
        return dict(root=info,snapshot=snapshot,gumbels=g,effective_k=k,quota=budget,completed=completed,attempted=attempted,
                    partial_completed=completed-(snapshot['support_watermark'] if snapshot else 0),eligible_plans=plans,
                    actions=plans[0] if plans else self.completion(info['actions'][rank[0]]))


def merge(root,stats,gumbels,lanes,config):
    n=ref.Node(ref.Info(root['actor'],(),(),tuple(root['actions'])),*stats,4);survivors=set()
    for lane in lanes:
        s=lane.get('snapshot')
        if s:
            survivors.update(s['survivors'])
            for i in range(len(n.count)):
                n.count[i]+=s['stats']['count'][i]
                for seat in range(4):n.returns[i][seat]+=s['stats']['returns'][i][seat]
    scores=n.scores(config['c1'],config['c2']) if survivors else n.logits
    rank=sorted(survivors or range(len(n.logits)),key=lambda i:(-scores[i]-gumbels[i],i))
    for edge in rank:
        options=[(i,plan) for i,lane in enumerate(lanes) for plan in lane['eligible_plans'] if plan[0]['action']['steps'][0]['choice']==root['actions'][edge]]
        if options:
            i,actions=min(options,key=lambda x:(-x[1][0]['support'],-x[1][0]['returns'][root['actor']]/x[1][0]['support'],
                         [s['choice'] for a in x[1] for s in a['action']['steps']],x[0]))
            return dict(actions=actions,source_session=i,proposal=root['actions'][rank[0]],provenance='pooled_search_supported')
    return dict(actions=None,source_session=None,proposal=root['actions'][rank[0]],provenance='pooled_bounded_completion' if survivors else 'common_gumbel_prior_fallback')


class Pair:
    def __init__(self,worker,evaluator_factory):
        self.pools=[ThreadPoolExecutor(1,thread_name_prefix=f'g_py-{i}') for i in range(2)]
        self.owners=[p.submit(lambda i=i:Session(worker,i,evaluator_factory())).result() for i,p in enumerate(self.pools)]
        self.turn=None;self.deadline=None

    def decide(self,input,config,mode,allowance_ms=4000,deadline=None,cancel=None):
        cancel=cancel or threading.Event();began=time.perf_counter_ns()
        c=input['view']['context'];current=tuple(c[k] for k in ('era','round','turn','turn_owner','actor'))+(c['phase']=='debt',)
        proposed=deadline or began+allowance_ms*1_000_000
        if self.turn!=current:
            self.turn=current;self.deadline=proposed
            for s in self.owners:s.epoch=0;s.cache={};s.costs={};s.continuation=[]
        else:self.deadline=min(self.deadline,proposed)
        deadline=self.deadline
        ready=threading.Event();shared={};configs=[copy.deepcopy(config),copy.deepcopy(config)];inputs=[copy.deepcopy(input),copy.deepcopy(input)]
        configs[1]['seed']^=0xd1b54a32d192ed03;inputs[1]['sample_seed']^=0x9e3779b97f4a7c15
        def owner(i):
            s=self.owners[i];s.allowance=allowance_ms*1_000_000
            try:
                s.start(inputs[i],configs[i],deadline,cancel)
                if i==0:
                    bypass=time.perf_counter_ns()>=s.work_deadline or len(s.root['actions'])==1 or input['view']['context']['phase']=='debt' or any(st['precondition']==s.root for a in s.continuation for st in a['action']['steps'])
                    shared['stats']=None if bypass else s.evaluate(s.root)
                    draws=Draws(config['seed']+s.epoch);shared['gumbels']=[-math.log(-math.log(draws.uniform())) for _ in s.root['actions']]
                    shared['root']=s.root;ready.set()
                else:
                    if not ready.wait(max(0,(s.work_deadline-time.perf_counter_ns())/1e9)):raise Expired()
                    if 'error' in shared:raise RuntimeError(shared['error'])
                    assert shared['root']==s.root
                return s.decide(shared,mode)
            except BaseException as e:
                shared['error']=str(e);ready.set();raise
        futures=[p.submit(owner,i) for i,p in enumerate(self.pools)];lanes=[f.result() for f in futures]
        if shared['stats'] is None:selection=dict(actions=lanes[0]['actions'],provenance=lanes[0]['provenance'])
        else:
            selection=merge(shared['root'],shared['stats'],shared['gumbels'],lanes,config)
            if selection['actions'] is None:selection['actions']=self.pools[0].submit(self.owners[0].completion,selection['proposal']).result()
        for p,s in zip(self.pools,self.owners):
            p.submit(s.completion,None,selection['actions']).result();s.continuation=copy.deepcopy(selection['actions'])
        return dict(lanes=lanes,selection=selection,response_ready_ns=time.perf_counter_ns()-began,
                    sessions=2,threads_per_session=1,cost_samples=[list(s.costs.values()) for s in self.owners])

    def close(self):
        for p in self.pools:p.shutdown(wait=True)
