"""Reserved phase turns for training teachers only; replay/label schemas unchanged."""
from collections import OrderedDict
import copy
from dataclasses import replace
import random
import time
import weakref

import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.learning.replay import validate_teacher
from brass_ai.search.common import SearchIncomplete,turn_key
from brass_ai.search.ismcts import MaskedSearchPolicy,information_key,node_key,supported_continuation
from brass_ai.searchlab.spiel import action_id
from brass_ai.step5.policy import inference_deadline
from .parallel import ParallelMasked
from .config import (SEARCH_WALL_FRACTION,teacher_search_config,teacher_work_budget_config,
                     nn_work_budget_config)


CONTRACT='four-phase-teacher-turns/1.0.0'
PHASES=(('canal_early','CanalEra',1,4),('canal_late','CanalEra',5,8),
        ('rail_early','RailEra',1,6),('rail_late','RailEra',7,8))
WORK_COUNTERS=frozenset(('allocated_simulations','attempted_simulations','work_exhausted_turns',
                        'deadline_limited_turns','work_budget_exhausted_choices'))


def phase_for(context):
    return next((name for name,era,first,last in PHASES
                 if context['era']==era and first<=context['round']<=last),None)


def reserved_rounds(setup_seed,sampling_seed,seat):
    require(type(seat) is int and 0<=seat<4,'Invalid teacher seat')
    # Independent phase streams: board actions, worker order and failed searches
    # cannot redraw a reservation or spend the later phase's allocation.
    return {name:random.Random(digest(dict(contract=CONTRACT,setup_seed=setup_seed,
                sampling_seed=sampling_seed,seat=seat,phase=name))).randint(first,last)
            for name,_,first,last in PHASES}


def counters():
    return dict(scheduled_turns=1,reached_turns=0,searched_turns=0,actual_searches=0,
        qualified_samples=0,successful_turns=0,failed_turns=0,unreached_turns=1,
        forced_choices=0,failures=0,timeouts=0,fallbacks=0,insufficient_coverage=0,
        cancelled=0,continuation_choices=0,budget_exhausted_choices=0,
        allocated_simulations=0,attempted_simulations=0,work_exhausted_turns=0,
        deadline_limited_turns=0,work_budget_exhausted_choices=0,
        completed_rollouts=0,interrupted_rollouts=0,elapsed_seconds=0.,
        turn_elapsed_seconds=0.,deadline_overrun_seconds=0.)


def search_allowance(ordinary_seconds,search_seconds,fraction=SEARCH_WALL_FRACTION):
    require(0<fraction<=SEARCH_WALL_FRACTION and min(ordinary_seconds,search_seconds)>=0,'Invalid search wall allowance')
    return max(0.,ordinary_seconds*fraction/(1-fraction)-search_seconds)


def estimate_collection(config,ordinary_game_seconds):
    # Four seats x four reserved TURNS. Workers/actions never multiply seconds.
    return config['search_games']*(4*len(PHASES)*teacher_search_config(config['search']).seconds+ordinary_game_seconds)


def finalization_reserve(seconds):
    # The 31-iteration lab run lost completed evidence to 0.15 s retirement
    # margins (individual overruns up to 0.89 s in its last collection).
    # Keep the whole-turn clock fixed; reserve 10%, capped at 1.2 s, for private
    # workers to finish an in-flight host RPC, merge and validate their labels.
    return min(1.2, seconds * .10)


def turn_seconds(policy):
    """Fixed work has an exceptional watchdog, never a normal think clock.

    Explicit old work/1.0 receipts keep their original capped-time semantics.
    Their configs/checkpoints are not silently relabelled as count-primary.
    """
    work=getattr(policy,'teacher_work_budget',None)
    return work['watchdog_seconds'] if work and work['contract']=='teacher-turn-work/1.1.0' else policy.config.seconds


def new_turn(policy,key,phase):
    started=time.monotonic();seconds=turn_seconds(policy);deadline=started+seconds
    return dict(key=key,phase=phase,started=started,deadline=deadline,
        work_deadline=deadline-finalization_reserve(seconds),attempts=0,disabled=False,
        tree={},cache=OrderedDict(),continuation={},qualified=0,work_allocated=0)


class TurnSearch:
    def decide(self,view,choices,rng,deadline,recall=None,tree=None,cache=None):
        # Replicas use this guard too, even though only the coordinator schedules
        # real turns. Never start another incumbent/tree after the common clock.
        self.teacher_work_deadline=deadline
        if time.monotonic()>=deadline or self.cancelled.is_set():raise SearchIncomplete('teacher_deadline_or_cancel')
        quota=getattr(self,'teacher_simulation_allowance',None)
        original=None
        # A forest divides its own shared allowance. A single private tree can
        # directly use the existing bounded simulation loop.
        if quota is not None and not isinstance(self,ParallelMasked):
            require(quota<=self.config.simulations,'Teacher work exceeds existing simulation cap')
            original=self.config;self.config=replace(original,simulations=quota)
        try:return super().decide(view,choices,rng,deadline,recall,tree,cache)
        finally:
            if original is not None:self.config=original

    def incumbent(self,view,choices,probabilities,rng,recall):
        """Bound the existing legal incumbent by the SAME search work deadline."""
        # A private replica's returned choice is discarded by the coordinator;
        # only its completed search statistics are merged. One real incumbent
        # at the coordinator is enough and leaves replicas time for tree work.
        if getattr(self,'teacher_replica',False):return {}
        deadline=self.teacher_work_deadline
        if time.monotonic()>=deadline:return {}
        state=self.game.from_observation(view,rng.randrange(2**63),recall)
        turn=turn_key(view['context']);trace={}
        for step in range(96):
            if time.monotonic()>=deadline or self.cancelled.is_set():break
            if state.is_terminal() or state.is_chance_node() or state.current_player()!=view['seat'] or turn_key(state.context)!=turn:break
            legal=state.legal;require(legal,'Empty incumbent legality')
            if step==0:
                require([c['id'] for c in legal]==[c['id'] for c in choices],'Incumbent root legality changed')
                index=int(np.argmax(probabilities))
            else:index=self.cheap_choice(legal)
            trace[node_key(state,view['seat'])]=legal[index]['id']
            state.apply_action(action_id(legal[index]['id']))
        return trace

    @staticmethod
    def cheap_choice(choices):
        return max(range(len(choices)),key=lambda i:(choices[i]['kind']=='finish',
            choices[i]['kind']=='beer' and bool(choices[i].get('merchant')),-i))

    def _complete(self,view,choices,rng,state,current):
        if current is not None:
            identity=current['continuation'].get(information_key(view,choices,state['recall']))
            if identity in [c['id'] for c in choices]:
                state['phases'][current['phase']]['continuation_choices']+=1
                return [c['id'] for c in choices].index(identity),1.,None
            if time.monotonic()>=current['deadline'] or self.cancelled.is_set():
                return self.cheap_choice(choices),1.,None
            try:
                with inference_deadline([self.base],current['deadline']):
                    selected,_,_=self.base.select_batch([(view,choices,rng)])[0]
                return selected,1.,None
            except (SearchIncomplete,TimeoutError):return self.cheap_choice(choices),1.,None
        selected,_,_=self.base.select_batch([(view,choices,rng)])[0]
        return selected,1.,None

    def select_batch(self,inputs):
        output=[]
        for view,choices,rng in inputs:
            require(rng in self.teacher_states,'Teacher game must declare its seeded schedule')
            state=self.teacher_states[rng];current=state['current']
            if not state['recall'] or state['recall'][-1]['hand']!=view['own_hand']:
                state['recall'].append(dict(revision=view['context']['revision'],hand=list(view['own_hand'])))
            if current is None or current['key']!=turn_key(view['context']) or view['context']['phase']=='debt':
                output.append(self._complete(view,choices,rng,state,None));continue
            bucket=state['phases'][current['phase']]
            if len(choices)==1:
                bucket['forced_choices']+=1;output.append((0,1.,None));continue
            work_budget=getattr(self,'teacher_work_budget',None)
            if work_budget is not None and current['work_allocated']:
                # Completed trajectories already cover both actions and their
                # subchoices. Follow the supported plan; never replenish work
                # at a new action, a forced choice, or an unsuccessful root.
                bucket['work_budget_exhausted_choices']+=1
                output.append(self._complete(view,choices,rng,state,current));continue
            if current['disabled'] or time.monotonic()>=current['work_deadline']:
                bucket['budget_exhausted_choices']+=int(not current['disabled'])
                if not current['disabled'] and current['attempts']==0:
                    bucket['timeouts']+=1;bucket['failures']+=1;current['disabled']=True
                    bucket['deadline_limited_turns']+=1
                output.append(self._complete(view,choices,rng,state,current));continue
            bucket['searched_turns']+=int(current['attempts']==0)
            current['attempts']+=1;bucket['actual_searches']+=1
            self.last_statistics=None;self.last_continuation={};began=time.monotonic();reason=None;teacher=None
            if work_budget is not None:
                current['work_allocated']=work_budget['simulations_per_phase'][current['phase']]
                bucket['allocated_simulations']+=current['work_allocated']
                self.teacher_simulation_allowance=current['work_allocated']
            try:
                selected,_,teacher=self.decide(view,choices,rng,current['work_deadline'],state['recall'],current['tree'],current['cache'])
                current['continuation']=dict(self.last_continuation)
            except (SearchIncomplete,TimeoutError):
                selected=self.cheap_choice(choices);reason='cancelled' if self.cancelled.is_set() else 'timeouts'
            finally:
                self.teacher_simulation_allowance=None
            stats=self.last_statistics or {};ended=time.monotonic()
            bucket['elapsed_seconds']+=ended-began
            bucket['completed_rollouts']+=stats.get('completed',0)
            bucket['interrupted_rollouts']+=stats.get('interrupted',0)
            attempted=stats.get('batching',{}).get('attempted',0)
            bucket['attempted_simulations']+=attempted
            if work_budget is not None:
                require(attempted<=current['work_allocated'],'Teacher exceeded its shared simulation allowance')
                bucket['deadline_limited_turns']+=int(ended>=current['work_deadline'])
                bucket['work_exhausted_turns']+=int(attempted==current['work_allocated'] and not self.cancelled.is_set())
                if work_budget['contract']=='teacher-turn-work/1.1.0':
                    # Count-primary teaching requires its allotted attempts.
                    # A watchdog-limited partial result remains a legal playing
                    # fallback but must never be called successful teaching.
                    if ended>=current['work_deadline']:reason='timeouts'
                    elif reason is None and attempted<current['work_allocated']:reason='insufficient_coverage'
            if ended>current['deadline']:reason='timeouts'
            if self.cancelled.is_set():reason='cancelled'
            if reason is None and (teacher is None or teacher['confidence']<=0):reason='insufficient_coverage'
            if reason is None:
                validate_teacher(teacher,len(choices))
                require(teacher['legal_ids']==[c['id'] for c in choices] and teacher['observer']==view['seat'],
                        'Misattributed teacher output')
                leading=max(range(len(choices)),key=lambda i:(stats.get('root',{}).get(choices[i]['id'],{}).get('visits',0),teacher['priors'][i],-i))
                if selected!=leading or supported_continuation(stats,choices[selected]['id'],view['seat'],self.config.minimum_plan_visits) is None:
                    reason='fallbacks'
                if time.monotonic()>current['deadline']:reason='timeouts'
            finalized=time.monotonic()
            bucket['elapsed_seconds']+=finalized-ended
            bucket['deadline_overrun_seconds']+=max(0.,finalized-current['deadline'])
            if finalized>current['deadline']:reason='timeouts'
            if reason is not None:
                bucket[reason]+=1;bucket['failures']+=1;current['disabled']=True;teacher=None
                if reason=='timeouts' and (work_budget is None or ended<current['work_deadline']):
                    bucket['deadline_limited_turns']+=1
            else:
                bucket['qualified_samples']+=1;current['qualified']+=1
            output.append((selected,1.,teacher))
        return output


class TeacherTurns(TurnSearch):
    def begin_teacher_game(self,rng,seat,setup_seed,sampling_seed):
        self.teacher_states[rng]=dict(seat=seat,rounds=reserved_rounds(setup_seed,sampling_seed,seat),
            phases={name:counters() for name,_,_,_ in PHASES},consumed=set(),current=None,recall=[])

    def teacher_workers(self,context,rng):
        state=self.teacher_states[rng];phase=phase_for(context)
        selected=(context['turn_owner']==state['seat'] and context['phase']!='debt'
                  and phase is not None and context['round']==state['rounds'][phase])
        return self.config.workers if selected else 1

    def admit_teacher_turn(self,context,rng):
        """Called immediately AFTER the actor's CPU lease has been acquired."""
        state=self.teacher_states[rng];key=turn_key(context)
        if state['current'] is not None and state['current']['key']==key:return
        if context['turn_owner']!=state['seat'] or context['phase']!='action':return
        self.finish_teacher_turn(rng)
        phase=phase_for(context)
        if phase is None or phase in state['consumed'] or context['round']!=state['rounds'][phase]:return
        state['consumed'].add(phase);bucket=state['phases'][phase]
        bucket['reached_turns']+=1;bucket['unreached_turns']=0
        # Fixed work normally stops at the path count. The separate watchdog
        # only bounds stalled hosts/inference; legacy recipes retain their clock.
        state['current']=new_turn(self,key,phase)

    def finish_teacher_turn(self,rng):
        state=self.teacher_states[rng];current=state['current']
        if current is None:return
        bucket=state['phases'][current['phase']]
        bucket['turn_elapsed_seconds']+=max(0.,time.monotonic()-current['started'])
        bucket['successful_turns']+=int(current['qualified']>0)
        bucket['failed_turns']+=int(current['qualified']==0)
        state['current']=None

    def teacher_report(self,rng):
        self.finish_teacher_turn(rng);state=self.teacher_states[rng]
        return dict(contract=CONTRACT,seat=state['seat'],reserved_rounds=dict(state['rounds']),
                    seconds_per_turn=self.config.seconds,workers=self.config.workers,
                    watchdog_seconds=turn_seconds(self),
                    work_budget=copy.deepcopy(getattr(self,'teacher_work_budget',None)),phases=copy.deepcopy(state['phases']))



class ScheduledMasked(TeacherTurns,MaskedSearchPolicy):pass
class ScheduledParallelMasked(TeacherTurns,ParallelMasked):pass


class WorkTurns(TurnSearch):
    """Fixed work on the existing NN search cadence, separate from teachers."""
    def begin_work_game(self,rng,seat):
        self.teacher_states[rng]=dict(seat=seat,recall=[],current=None,last_turn=None,turns=0,searches=0,due=False,
            phases={name:dict(counters(),scheduled_turns=0,unreached_turns=0) for name,_,_,_ in PHASES})

    def work_due(self,context,state):
        if context['turn_owner']!=state['seat'] or context['phase']=='debt':return False
        if state['last_turn']==turn_key(context):return state['due']
        if context['phase']!='action':return False
        return state['turns']%self.config.turn_interval==0 and (not self.config.max_searches_per_seat or
            state['searches']<self.config.max_searches_per_seat)

    def work_workers(self,context,rng):
        return self.config.workers if self.work_due(context,self.teacher_states[rng]) else 1

    def admit_work_turn(self,context,rng):
        """Called after CPU admission, before observing or encoding the turn."""
        state=self.teacher_states[rng];key=turn_key(context)
        if state['last_turn']==key:return
        if context['turn_owner']!=state['seat'] or context['phase']!='action':return
        self.finish_work_turn(rng);due=self.work_due(context,state)
        state.update(last_turn=key,turns=state['turns']+1,due=due)
        if not due:return
        state['searches']+=1;phase=phase_for(context)
        require(phase is not None,'NN work turn outside four-player era rounds')
        state['phases'][phase]['scheduled_turns']+=1
        state['phases'][phase]['reached_turns']+=1
        state['current']=new_turn(self,key,phase)

    finish_work_turn=TeacherTurns.finish_teacher_turn

    def work_report(self,rng):
        self.finish_work_turn(rng);state=self.teacher_states[rng]
        return dict(contract=self.nn_work_budget['contract'],seat=state['seat'],workers=self.config.workers,
                    work_budget=copy.deepcopy(self.nn_work_budget),turns=state['turns'],search_turns=state['searches'],
                    turn_interval=self.config.turn_interval,max_searches_per_seat=self.config.max_searches_per_seat,
                    phases=copy.deepcopy(state['phases']))

    def select_batch(self,inputs):
        # Evaluations/experiments must not accidentally produce PPO teacher
        # labels. The same legal-plan and evidence gates still choose the move.
        return [(choice,probability,None) for choice,probability,_ in super().select_batch(inputs)]


class WorkMasked(WorkTurns,MaskedSearchPolicy):pass
class WorkParallelMasked(WorkTurns,ParallelMasked):pass


def work_search(policy,work_budget):
    require(isinstance(policy,MaskedSearchPolicy),'NN work policy must use Masked ISMCTS')
    work=nn_work_budget_config(work_budget)
    require(work is not None,'Missing NN whole-turn work budget')
    require(work['simulations_per_turn']<=policy.config.workers*policy.config.simulations,
            'NN work budget exceeds the private-tree simulation caps')
    policy.__class__=WorkParallelMasked if isinstance(policy,ParallelMasked) else WorkMasked
    policy.teacher_states=weakref.WeakKeyDictionary();policy.teacher_replica=False
    policy.nn_work_budget=copy.deepcopy(work)
    policy.teacher_work_budget=dict(contract='teacher-turn-work/1.1.0',mode='simulations',
        simulations_per_phase={phase:work['simulations_per_turn'] for phase,_,_,_ in PHASES},
        watchdog_seconds=work['watchdog_seconds'])
    policy.policy_id='step5-work-search/'+digest(dict(parent=policy.policy_id,work_budget=work))
    # Replicas need the bounded incumbent and quota plumbing, not independent
    # real-turn schedules. Only the coordinator owns the whole-turn allocation.
    for worker in getattr(policy,'replicas',[]):training_teacher(worker,replica=True)
    return policy


def training_teacher(policy,work_budget=None,*,replica=False):
    require(isinstance(policy,MaskedSearchPolicy),'Training teacher must use Masked ISMCTS + NN')
    work_budget=teacher_work_budget_config(work_budget)
    policy.__class__=ScheduledParallelMasked if isinstance(policy,ParallelMasked) else ScheduledMasked
    policy.teacher_states=weakref.WeakKeyDictionary()
    policy.teacher_replica=replica
    policy.teacher_work_budget=copy.deepcopy(work_budget)
    identity=dict(parent=policy.policy_id,contract=CONTRACT)
    if work_budget is not None:identity['work_budget']=work_budget
    policy.policy_id='step5-phase-teacher/'+digest(identity)
    for worker in getattr(policy,'replicas',[]):training_teacher(worker,replica=True)
    return policy


def collection_report(records,games):
    phases={name:{key:0 for key in counters()} for name,_,_,_ in PHASES}
    receipts={}
    for record in records:
        reports=record.get('teacher_phases')
        require(reports and len(reports)==4,'Missing four-seat teacher scheduling receipts')
        require({r['seat'] for r in reports}==set(range(4)),'Duplicate/missing teacher seat')
        require(record['game_id'] not in receipts,'Duplicate teacher game receipt')
        receipts[record['game_id']]={r['seat']:r for r in reports}
        for report in reports:
            require(report['contract']==CONTRACT,'Unknown teacher schedule receipt')
            require(set(report['phases'])==set(phases) and set(report['reserved_rounds'])==set(phases),'Missing teacher phase')
            work_budget=teacher_work_budget_config(report.get('work_budget'))
            for name,_,first,last in PHASES:
                require(first<=report['reserved_rounds'][name]<=last,'Reservation outside teacher phase')
                counts=report['phases'][name]
                require(counts['scheduled_turns']==1 and counts['reached_turns'] in (0,1) and
                        counts['reached_turns']+counts['unreached_turns']==1,'Teacher turn allocation exceeded')
                if work_budget is not None:
                    require(WORK_COUNTERS<=set(counts),'Missing teacher work counters')
                    require(counts['actual_searches'] in (0,1) and
                            counts['allocated_simulations']==counts['actual_searches']*work_budget['simulations_per_phase'][name] and
                            0<=counts['completed_rollouts']<=counts['attempted_simulations']<=counts['allocated_simulations'],
                            'Teacher work receipt exceeds shared turn allowance')
            for phase in phases:
                for key in phases[phase]:
                    counts=report['phases'][phase]
                    phases[phase][key]+=counts.get(key,0) if key in WORK_COUNTERS else counts[key]
    require({g['game_id'] for g in games}==set(receipts),'Teacher receipts differ from collected games')
    actual={name:0 for name in phases}
    for game in games:
        for row in game['records']:
            if row['teacher'] is not None:
                require(len(row['choices'])>1,'Forced teaching sample')
                phase=phase_for(row['view']['context']);report=receipts[game['game_id']][row['seat']]
                require(phase is not None and row['view']['context']['round']==report['reserved_rounds'][phase],
                        'Teaching sample outside reserved round')
                actual[phase]+=1
    require(all(actual[p]==phases[p]['qualified_samples'] for p in phases),'Teacher sample counts differ from replay')
    return dict(contract=CONTRACT,phases=phases,
        note='Search calls and qualified decision labels are distinct from reserved turns; simulation allocations are shared across workers/actions. Elapsed search includes retirement, turn elapsed includes host/subchoice work.')
