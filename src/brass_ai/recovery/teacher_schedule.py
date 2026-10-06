"""Reserved phase turns for training teachers only; replay/label schemas unchanged."""
from collections import OrderedDict
import copy
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
from .config import SEARCH_WALL_FRACTION,teacher_search_config


CONTRACT='four-phase-teacher-turns/1.0.0'
PHASES=(('canal_early','CanalEra',1,4),('canal_late','CanalEra',5,8),
        ('rail_early','RailEra',1,6),('rail_late','RailEra',7,8))


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
        completed_rollouts=0,interrupted_rollouts=0,elapsed_seconds=0.,
        turn_elapsed_seconds=0.,deadline_overrun_seconds=0.)


def search_allowance(ordinary_seconds,search_seconds,fraction=SEARCH_WALL_FRACTION):
    require(0<fraction<=.50 and min(ordinary_seconds,search_seconds)>=0,'Invalid search wall allowance')
    return max(0.,ordinary_seconds*fraction/(1-fraction)-search_seconds)


def estimate_collection(config,ordinary_game_seconds):
    # Four seats x four reserved TURNS. Workers/actions never multiply seconds.
    return config['search_games']*(4*len(PHASES)*teacher_search_config(config['search']).seconds+ordinary_game_seconds)


class TeacherTurns:
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
        started=time.monotonic();deadline=started+self.config.seconds
        # All workers/actions use this one deadline. Stop launching tree work
        # slightly earlier to leave bounded time for merging and label checks.
        state['current']=dict(key=key,phase=phase,started=started,deadline=deadline,
            work_deadline=deadline-min(.15,self.config.seconds*.05),attempts=0,disabled=False,
            tree={},cache=OrderedDict(),continuation={},qualified=0)

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
                    seconds_per_turn=self.config.seconds,workers=self.config.workers,phases=copy.deepcopy(state['phases']))

    def decide(self,view,choices,rng,deadline,recall=None,tree=None,cache=None):
        # Replicas use this guard too, even though only the coordinator schedules
        # real turns. Never start another incumbent/tree after the common clock.
        self.teacher_work_deadline=deadline
        if time.monotonic()>=deadline or self.cancelled.is_set():raise SearchIncomplete('teacher_deadline_or_cancel')
        return super().decide(view,choices,rng,deadline,recall,tree,cache)

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
            if current['disabled'] or time.monotonic()>=current['work_deadline']:
                bucket['budget_exhausted_choices']+=int(not current['disabled'])
                if not current['disabled'] and current['attempts']==0:
                    bucket['timeouts']+=1;bucket['failures']+=1;current['disabled']=True
                output.append(self._complete(view,choices,rng,state,current));continue
            bucket['searched_turns']+=int(current['attempts']==0)
            current['attempts']+=1;bucket['actual_searches']+=1
            self.last_statistics=None;self.last_continuation={};began=time.monotonic();reason=None;teacher=None
            try:
                selected,_,teacher=self.decide(view,choices,rng,current['work_deadline'],state['recall'],current['tree'],current['cache'])
                current['continuation']=dict(self.last_continuation)
            except (SearchIncomplete,TimeoutError):
                selected=self.cheap_choice(choices);reason='cancelled' if self.cancelled.is_set() else 'timeouts'
            stats=self.last_statistics or {};ended=time.monotonic()
            bucket['elapsed_seconds']+=ended-began
            bucket['completed_rollouts']+=stats.get('completed',0)
            bucket['interrupted_rollouts']+=stats.get('interrupted',0)
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
            else:
                bucket['qualified_samples']+=1;current['qualified']+=1
            output.append((selected,1.,teacher))
        return output


class ScheduledMasked(TeacherTurns,MaskedSearchPolicy):pass
class ScheduledParallelMasked(TeacherTurns,ParallelMasked):pass


def training_teacher(policy,*,replica=False):
    require(isinstance(policy,MaskedSearchPolicy),'Training teacher must use Masked ISMCTS + NN')
    policy.__class__=ScheduledParallelMasked if isinstance(policy,ParallelMasked) else ScheduledMasked
    policy.teacher_states=weakref.WeakKeyDictionary()
    policy.teacher_replica=replica
    policy.policy_id='step5-phase-teacher/'+digest(dict(parent=policy.policy_id,contract=CONTRACT))
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
            for name,_,first,last in PHASES:
                require(first<=report['reserved_rounds'][name]<=last,'Reservation outside teacher phase')
                counts=report['phases'][name]
                require(counts['scheduled_turns']==1 and counts['reached_turns'] in (0,1) and
                        counts['reached_turns']+counts['unreached_turns']==1,'Teacher turn allocation exceeded')
            for phase in phases:
                for key in phases[phase]:phases[phase][key]+=report['phases'][phase][key]
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
        note='Search calls and qualified decision labels are distinct from reserved turns; elapsed search includes retirement, turn elapsed includes host/subchoice work.')
