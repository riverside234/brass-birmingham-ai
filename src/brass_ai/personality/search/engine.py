"""Goal-scoped extension of the existing private Masked ISMCTS forests."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import random
from threading import Event,RLock
import time

import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.learning.replay import validate_teacher
from brass_ai.recovery.inference import RemoteEvaluator,share_encoder_cache
from brass_ai.recovery.parallel import merge_statistics
from brass_ai.recovery.resources import TurnLease
from brass_ai.recovery.teacher_schedule import reserved_rounds,counters,phase_for,finalization_reserve,TeacherTurns
from brass_ai.recovery.telemetry import Timings
from brass_ai.search.common import SearchIncomplete,turn_key
from brass_ai.search.deadline import configure_clock
from brass_ai.search.ismcts import MaskedISMCTS,MaskedSearchPolicy,information_key,node_key,visit_teacher,supported_continuation
from brass_ai.step5.policy import NeuralPolicy,inference_deadline
from .contracts import TEACHER,identity
from .inference import PersonalityRemote,PersonalityPolicy
from .intent import IntentState,alignment,biased_prior,candidate_order


class RootMasked(MaskedISMCTS):
    def bind(self,scope):self.scope=scope;return self

    def request(self,policy,state,rng,ledger=None):
        p,response=yield from super().request(policy,state,rng,ledger)
        # Only the initial permitted information root. Never an opponent node,
        # a payment continuation, or a future private-world goal prediction.
        if policy is self.base and node_key(state,self.root)==self.scope['information'] and self.scope['inject']:
            q=biased_prior(p,response['probabilities'],state.legal,self.scope['alignment'],
                self.scope['profile'],self.scope['confidence'])
            node=self.tree.get(self.root_key)
            if node is not None and node.ranked is None:
                node.ranked=candidate_order(state.legal,p,q,self.config.root_top_k,self.scope['profile'])
            return q,response
        return p,response


class Worker(MaskedSearchPolicy):
    def __init__(self,root,config,base,opponents,mono,cancelled,index):
        # Fused native snapshots advance rules and return checkpoint, permitted
        # views and choices in one reply. Older bundles retain the checked
        # capability fallback; each simulation still owns its private host.
        super().__init__(None,root,config=config,mono=mono,evaluator=base.evaluator,opponents=opponents,fast_host=True)
        self.base=base;self.cancelled=cancelled;self.partition_index=index;self.scope=None

    def decide(self,view,choices,rng,deadline,recall=None,tree=None,cache=None):
        self.last_statistics=None
        bot=RootMasked(self.game,self.base,self.rollouts,self.config,view['seat'],self.context_features(view,rng),rng,
            deadline,recall,tree,cache,self.simulation_games,self.cancelled,self.partition_index,self.reward_context).bind(self.scope)
        configure_clock(bot,self)
        with inference_deadline([self.base]+self.rollouts,deadline):stats=bot.run(view,choices)
        stats['context_id']=self.scope['id'];self.last_statistics=stats
        return stats


def teacher(choices,stats,base_priors,search_priors,*,model,seat,config,scope,remaining,selected):
    target=visit_teacher(choices,stats,base_priors,model=model,observer=seat,config=config)
    if target is None or supported_continuation(stats,choices[selected]['id'],seat,config.minimum_plan_visits) is None:return None
    payload=dict(version=TEACHER,contract=identity(),target=target,personality=scope['context'],goal=scope['goal'],
        profile=scope['profile'].to_dict(),bias_applied=scope['inject'],base_priors=target['priors'],
        search_priors=list(map(float,search_priors)),remaining=list(remaining),
        root_information=scope['information'],root_view=scope['view_hash'],selected=selected,selected_plan=supported_continuation(stats,choices[selected]['id'],seat,config.minimum_plan_visits),
        selected_plan_support=max(p['visits'] for p in stats['plans'].values() if p['first']==choices[selected]['id']),
        root_context_id=scope['id'],calibration_id=scope['calibration_id'],reward_id=digest(config.reward))
    payload['fingerprint']=digest(payload)
    return payload


def validate_personality_teacher(value,row,model,reward):
    from brass_ai.personality.contracts import context_vector
    require(value['version']==TEACHER and value['contract']==identity(),'Old/mismatched personality teacher')
    require(value['fingerprint']==digest({k:v for k,v in value.items() if k!='fingerprint'}),'Teacher context changed')
    t=value['target'];validate_teacher(t,len(row['choices']))
    require(t['model_id']==model and t['search']['reward']==reward and value['reward_id']==digest(reward),'Teacher model/reward mismatch')
    require(t['observer']==row['seat'] and t['legal_ids']==[c['id'] for c in row['choices']] and len(row['choices'])>1,'Teacher observation/legality differs')
    require(value['root_view']==digest(row['view']),'Teacher root observation differs')
    require(row['chosen']==value['selected'] and row['teacher'] is None and row['behavior']['kind']=='external','Teacher cannot be relabeled as on-policy PPO')
    require(value['base_priors']==t['priors'] and len(value['search_priors'])==len(row['choices'])
        and np.isfinite(value['search_priors']).all() and (np.asarray(value['search_priors'])>=0).all() and abs(sum(value['search_priors'])-1)<1e-6,'Invalid teacher priors')
    require(len(value['remaining'])==4 and all(0<=x<=reward['insolvency']['max_cost']+1e-8 for x in value['remaining']),'Invalid teacher cost budget')
    require(sum(n>0 for n in t['visits'])>=2 and value['selected_plan'] and value['selected_plan_support']>=t['search']['minimum_plan_visits'],
            'Insufficient completed search evidence')
    require(value['selected_plan'].get(value['root_information'])==row['choices'][row['chosen']]['id'],'Selected completed plan differs')
    context_vector(value['personality'])
    return value


class SearchSession:
    """One persistent table forest; four independent seat states, one CPU lease.

    Tree ownership stays within each worker; shared queues use private locked
    reply lanes. The coordinator drains every worker before releasing the lease.
    """
    cheap_choice=staticmethod(TeacherTurns.cheap_choice)
    def __init__(self,root,service,channels,model_id,parent_ids,config,profiles,calibration,*,mono=None,search_enabled=True):
        self.config=config.validate();self.profiles=profiles;self.calibration=calibration
        self.search_enabled=search_enabled
        self.cancelled=Event();self.lock=RLock();self.service=service;self.timings=Timings()
        self.lease=TurnLease(service.cpu,self.timings);self.model_id=model_id
        self.intents=[IntentState(s,profiles[s]) for s in range(4)];self.recall=[[] for _ in range(4)]
        self.rounds=None;self.phases=[{} for _ in range(4)];self.turn=None;self.workers=[]
        self.pool=ThreadPoolExecutor(max_workers=config.workers,thread_name_prefix='personality-private-root')
        try:
            count=config.workers if search_enabled else 0
            endpoints=[channels.acquire() for _ in range(1+count*(1+len(parent_ids)))]
            # The coordinator warms the root before the private forests start.
            # Reuse completed, immutable features across every lane/model in
            # this table, while keeping mutable builders and reply pipes private.
            # Preserve the former aggregate bound of 256 entries per lane.
            self.encoding_cache=share_encoder_cache(endpoints)
            lanes=iter(endpoints)
            self.base=PersonalityPolicy(PersonalityRemote(next(lanes),model_id),exploration=.1,cancelled=self.cancelled)
            for i in range(count):
                base=PersonalityPolicy(PersonalityRemote(next(lanes),model_id),exploration=.1,cancelled=self.cancelled)
                opponents=[NeuralPolicy(None,evaluator=RemoteEvaluator(next(lanes),p),focus='neutral',exploration=.05) for p in parent_ids]
                self.workers.append(Worker(root,config,base,opponents,mono,self.cancelled,i))
        except BaseException:self.close();raise
        self.game=self.workers[0].game if self.workers else None
        self.telemetry=dict(decisions=0,searches=0,completed=0,interrupted=0,fallbacks=0,seconds=0.,prior_seconds=0.,overruns=0.,
            attempted=0,deadline_stops=0,deadline_unused_worker_seconds=0.)

    def begin(self,setup_seed,sampling_seed,*,teacher_game=False):
        require(self.turn is None,'Close the previous game first')
        self.intents=[IntentState(s,self.profiles[s]) for s in range(4)];self.recall=[[] for _ in range(4)]
        self.rounds=[reserved_rounds(setup_seed,sampling_seed,s) for s in range(4)] if teacher_game else None
        self.phases=[{p:counters() for p in self.rounds[s]} if self.rounds else {} for s in range(4)]

    def finish_turn(self):
        if self.turn is not None and self.turn['bucket'] is not None:
            b=self.turn['bucket'];b['successful_turns']+=int(self.turn['qualified']>0);b['failed_turns']+=int(self.turn['qualified']==0)
            b['turn_elapsed_seconds']+=time.monotonic()-self.turn['started']
        self.turn=None;self.lease.close()

    def snapshot(self):
        require(self.turn is None,'Save only after the active turn retires')
        return dict(contract=identity(),model=self.model_id,profiles=[p.to_dict() for p in self.profiles],
            search=self.config.to_dict(),search_enabled=self.search_enabled,calibration_id=digest(self.calibration),
            intents=[x.snapshot() for x in self.intents],recall=copy.deepcopy(self.recall),rounds=self.rounds,phases=self.phases)

    def restore(self,state):
        require(self.turn is None and state['contract']==identity() and state['model']==self.model_id
            and state['profiles']==[p.to_dict() for p in self.profiles] and state['search']==self.config.to_dict()
            and state['search_enabled']==self.search_enabled and state['calibration_id']==digest(self.calibration),'Search restore differs')
        for target,value in zip(self.intents,state['intents']):target.restore(value)
        self.recall=copy.deepcopy(state['recall']);self.rounds=state['rounds'];self.phases=copy.deepcopy(state['phases'])

    def close(self):
        self.cancelled.set();self.pool.shutdown(wait=True)
        for w in self.workers:w.close()
        self.finish_turn()

    def select(self,view,choices,rng,ledger):
        with self.lock:return self._select(view,choices,rng,ledger)

    def _select(self,view,choices,rng,ledger):
        seat=view['seat'];key=(turn_key(view['context']),seat,view['context']['phase']=='debt')
        if not self.recall[seat] or self.recall[seat][-1]['hand']!=view['own_hand']:
            self.recall[seat].append(dict(revision=view['context']['revision'],hand=list(view['own_hand'])))
        if self.turn is None or self.turn['key']!=key:
            self.finish_turn();phase=phase_for(view['context']);b=None
            selected=self.search_enabled and view['context']['phase']!='debt' and (self.rounds is None or (phase in self.rounds[seat] and
                view['context']['round']==self.rounds[seat][phase] and not self.phases[seat][phase]['reached_turns']))
            self.lease.update(key,self.config.workers if selected else 1)
            started=time.monotonic()  # CPU admission is outside the clock.
            ctx=self.intents[seat].before_turn(view) if view['context']['phase']!='debt' else context_for_debt(self.profiles[seat])
            if selected and self.rounds is not None:
                b=self.phases[seat][phase];b['reached_turns']=1;b['unreached_turns']=0
            self.turn=dict(key=key,started=started,deadline=started+self.config.seconds,
                work_deadline=started+self.config.seconds-finalization_reserve(self.config.seconds),due=selected,
                bucket=b,qualified=0,attempts=0,disabled=False,context=ctx,goal=None,proposed=False,injected=False,
                trees=[{} for _ in self.workers],caches=[OrderedDict() for _ in self.workers],continuation={})
        turn=self.turn;self.base.set_context(turn['context']);self.base.utility_remaining[rng]=tuple(ledger.remaining)
        self.telemetry['decisions']+=1
        info=information_key(view,choices,self.recall[seat]);saved=turn['continuation'].get(info)
        if len(choices)==1:
            if turn['bucket'] is not None:turn['bucket']['forced_choices']+=1
            if view['context']['phase']=='action' and not turn['proposed']:
                # Goal supervision starts at the first action of an own turn.
                # Keep a surviving commitment, but do not move a skipped fresh
                # proposal into payments or the unsupervised second action.
                turn['goal']=copy.deepcopy(turn['context']['saved_intent']);turn['proposed']=True
            return 0,None,dict(context=turn['context'],goal=turn['goal'],searched=False)
        if not turn['due']:
            with inference_deadline([self.base],time.monotonic()+120):p,response=self.base.predict([(view,choices,rng)])[0]
            if not turn['proposed'] and view['context']['phase']=='action':
                turn['goal']=self.intents[seat].commit(response,view,self.calibration);turn['proposed']=True
            chosen=rng.choices(range(len(p)),weights=p,k=1)[0]
            return chosen,None,dict(context=turn['context'],goal=turn['goal'],searched=False)
        if turn['disabled'] or self.cancelled.is_set() or time.monotonic()>=turn['work_deadline']:
            if turn['bucket'] is not None:turn['bucket']['budget_exhausted_choices']+=1
            chosen=[c['id'] for c in choices].index(saved) if saved in [c['id'] for c in choices] else TeacherTurns.cheap_choice(choices)
            return chosen,None,dict(context=turn['context'],goal=turn['goal'],searched=False)
        began=time.monotonic();bucket=turn['bucket'];turn['attempts']+=1
        if bucket is not None:bucket['actual_searches']+=1;bucket['searched_turns']=1
        selected=TeacherTurns.cheap_choice(choices);label=None;stats=None;reason=None
        try:
            with inference_deadline([self.base],turn['work_deadline']):base,response=self.base.predict([(view,choices,rng)])[0]
            # Goal labels/calibration cover ordinary action roots. A forced
            # root must not move proposal generation into an untrained payment
            # subchoice.
            if not turn['proposed'] and view['context']['phase']=='action':
                turn['goal']=self.intents[seat].commit(response,view,self.calibration);turn['proposed']=True
            inject=not turn['injected'] and view['context']['phase']=='action'
            prior_started=time.perf_counter();g=alignment(view,choices,turn['goal']) if inject else np.zeros(len(choices))
            profile=self.profiles[seat];confidence=turn['goal']['confidence'] if turn['goal'] else 0.
            priors=biased_prior(base,response['probabilities'],choices,g,profile,confidence)
            scope=dict(context=copy.deepcopy(turn['context']),goal=copy.deepcopy(turn['goal']),profile=profile,calibration_id=digest(self.calibration),
                information=info,view_hash=digest(view),inject=inject,alignment=g,confidence=confidence)
            scope['id']=digest(dict(context=scope['context'],goal=scope['goal'],profile=profile.to_dict(),information=info,inject=inject,
                calibration_id=scope['calibration_id'],contract=identity()))
            self.telemetry['prior_seconds']+=time.perf_counter()-prior_started
            self.teacher_work_deadline=turn['work_deadline'];self.teacher_replica=False
            self.last_continuation=TeacherTurns.incumbent(self,view,choices,base,rng,self.recall[seat])
            turn['continuation'].update(self.last_continuation)
            seeds=[rng.randrange(2**63) for _ in self.workers]
            def run(i):
                w=self.workers[i];w.base.set_context(scope['context']);w.scope=scope;w.reward_context=ledger.snapshot()
                return w.decide(view,choices,random.Random(seeds[i]),turn['work_deadline'],self.recall[seat],turn['trees'][i],turn['caches'][i])
            pending=[self.pool.submit(run,i) for i in range(len(self.workers))];items=[];error=None
            for future in pending:
                try:items.append(future.result())
                except BaseException as e:error=error or e
            if error:raise error
            require(all(x['context_id']==scope['id'] for x in items),'Cannot merge incompatible private roots')
            stats=merge_statistics(items,choices);turn['injected']|=inject
            self.telemetry['attempted']+=sum(x['batching']['attempted'] for x in items)
            self.telemetry['deadline_stops']+=sum(x['batching'].get('deadline_stops',0) for x in items)
            # Sum of unused worker clocks, not additional whole-turn time.
            self.telemetry['deadline_unused_worker_seconds']+=sum(x['batching'].get('deadline_remaining_seconds',0.) for x in items)
            selected=max(range(len(choices)),key=lambda i:(stats['root'].get(choices[i]['id'],{}).get('visits',0),priors[i],-i))
            plan=supported_continuation(stats,choices[selected]['id'],seat,self.config.minimum_plan_visits)
            if plan is None:
                selected=int(np.argmax(base));reason='fallbacks'
            else:turn['continuation'].update(plan)
            if reason is None:label=teacher(choices,stats,base,priors,model=self.model_id,seat=seat,config=self.config,
                scope=scope,remaining=ledger.remaining,selected=selected)
            if reason is None and label is None:reason='insufficient_coverage'
        except (SearchIncomplete,TimeoutError):reason='cancelled' if self.cancelled.is_set() else 'timeouts'
        ended=time.monotonic();elapsed=ended-began
        if self.cancelled.is_set():reason='cancelled'
        if ended>turn['deadline']:reason='timeouts'
        self.timings.observe('search.decision',elapsed)
        self.telemetry['searches']+=1;self.telemetry['seconds']+=elapsed;self.telemetry['overruns']+=max(0.,ended-turn['deadline'])
        if stats:
            self.telemetry['completed']+=stats['completed'];self.telemetry['interrupted']+=stats['interrupted']
        if bucket is not None:
            bucket['elapsed_seconds']+=elapsed;bucket['deadline_overrun_seconds']+=max(0.,ended-turn['deadline'])
            bucket['completed_rollouts']+=stats['completed'] if stats else 0;bucket['interrupted_rollouts']+=stats['interrupted'] if stats else 0
        if reason:
            label=None;turn['disabled']=True;self.telemetry['fallbacks']+=1
            if bucket is not None:bucket[reason]+=1;bucket['failures']+=1
            # Keep a completed legal current-information plan when retiring.
            saved=turn['continuation'].get(info)
            if saved in [c['id'] for c in choices]:selected=[c['id'] for c in choices].index(saved)
        else:
            turn['qualified']+=1
            if bucket is not None:bucket['qualified_samples']+=1
        return selected,label,dict(context=turn['context'],goal=turn['goal'],searched=True,reason=reason,
            completed=stats['completed'] if stats else 0,seconds=elapsed,deadline_seconds=self.config.seconds,
            supported_alternatives=sum(x['visits']>0 for x in stats['root'].values()) if stats else 0)


def context_for_debt(profile):
    from brass_ai.personality.contracts import context
    return context(profile.style)
