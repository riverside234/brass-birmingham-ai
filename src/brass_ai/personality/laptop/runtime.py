"""Permitted-observation game boundary, shared exports and private seat forests.

No vendor save/checkpoint or shuffle seed enters this process. Private saved
seat recall stays in the trusted coordinator; each policy sees only its seat.
The native bridge validates every complete action again against the live game.
"""
import copy
from dataclasses import replace
from pathlib import Path
import random
from threading import Event,RLock
import time

from brass_ai.learning.contracts import require,digest
from brass_ai.learning.reward import CostLedger
from brass_ai.personality.search.contracts import Profile
from brass_ai.personality.search.engine import SearchSession
from brass_ai.personality.difficulty.collection import TrackedChannels
from brass_ai.personality.difficulty.ui import IntentDisplay,PublicAction,PUBLIC_TEXT
from brass_ai.recovery.inputs import InputBoundary
from brass_ai.search.common import SearchConfig,turn_key
from brass_ai.searchlab.spiel import BrassSpielGame,action_id
from brass_ai.search.ismcts import supported_continuation
from brass_ai.recovery.parallel import merge_statistics
from . import sampling
from .contracts import PROTOCOL,seal
from .inference import PortableInference
from .package import validate


def public_key(view):
    return [view['seat'],*turn_key(view['context'])]


def completed_paths(session,view,choices,selected):
    """Reuse the ordered complete trace the existing planner already evaluated.

    Never concatenate different information-world branches from the merged
    continuation dictionary. Use one supported trace, or its complete incumbent.
    Unity still independently validates each full action before committing it.
    """
    first=choices[selected]['id'];trace=None
    items=[w.last_statistics for w in session.workers if w.last_statistics is not None]
    if items and len({s['context_id'] for s in items})==1:
        stats=merge_statistics(items,choices)
        trace=supported_continuation(stats,first,view['seat'],session.config.minimum_plan_visits)
    if trace is None:trace=getattr(session,'last_continuation',{})
    ids=list(trace.values());paths=[];path=[]
    if not ids or ids[0]!=first:return None
    for ident in ids:
        path.append(ident)
        if ident.split('|',1)[0]=='finish':paths.append(path);path=[]
    if path or len(paths)!=view['context']['actions_remaining'] or any(len(p)>128 for p in paths):return None
    return paths


def validate_paths(game,view,paths,recall,seed):
    """Detached revalidation: a plan stops at a chance/turn boundary, never reads it."""
    state=game.from_observation(view,seed,recall);seat=view['seat'];key=public_key(view)
    require(1<=len(paths)<=2 and all(1<=len(p)<=128 for p in paths),'Invalid complete-plan size')
    for i,path in enumerate(paths):
        require(not state.is_terminal() and not state.is_chance_node() and state.current_player()==seat
                and public_key(state.views[seat])==key,'Plan crosses a turn/chance boundary')
        for j,choice in enumerate(path):
            require(not state.is_chance_node() and choice in {c['id'] for c in state.legal},'Illegal complete plan')
            c=next(c for c in state.legal if c['id']==choice)
            require((c['kind']=='finish')==(j==len(path)-1),'Action must finish exactly once')
            state.apply_action(action_id(choice))
    return True


class Runtime:
    def __init__(self,directory,backend='ort-cpu',*,fixture=False,mono=None):
        self.directory=Path(directory).resolve();self.package=validate(directory,allow_fixture=fixture)
        self.root=self.directory/'runtime';self.mono=mono;self.lock=RLock();self.interrupted=Event();self.cancelled=Event()
        self.sessions={};self.channels={};self.active=None;self.closed=False;self.boundary=InputBoundary()
        self.service=PortableInference(self.directory,self.package['models'],backend)
        self.service.__enter__()
        try:self.game=BrassSpielGame(root=self.root,mono=mono,fast=True)
        except BaseException:self.service.__exit__();raise

    def close(self):
        self.interrupt(cancel=True)
        with self.lock:
            if self.closed:return
            self.closed=True
            try:
                for session in self.sessions.values():session.close()
            finally:
                try:
                    for channels in self.channels.values():channels.close()
                finally:
                    try:self.game.close()
                    finally:self.service.__exit__()

    def interrupt(self,*,cancel=False):
        # No lock that a planning thread owns: Play Now/cancel must be immediate.
        self.interrupted.set()
        if cancel:self.cancelled.set()
        active=self.active
        if active is not None:active.cancelled.set()

    def session(self,seat,tier,profile):
        # Serialized game jobs reuse the same native workers. begin() clears all
        # trees/recall, and only this seat's pinned state is then restored.
        key='one-table';p=self.package
        if key not in self.sessions:
            config=SearchConfig(**p['searches'][tier]).validate()
            parents=[m['model_id'] for m in p['models'][1:]]
            channels=None
            try:
                channels=TrackedChannels(self.service.channels(1+config.workers*(1+len(parents))))
                s=SearchSession(self.root,self.service,channels,p['models'][0]['model_id'],parents,config,
                    [profile if i==seat else Profile(alpha=0.) for i in range(4)],p['calibration'],mono=self.mono)
            except BaseException:
                # The persistent service owns the other pipe ends, shared slots
                # and lease queue. Retire it too; retry must start a fresh runtime.
                try:
                    if channels is not None:channels.close()
                finally:self.close()
                raise
            self.sessions[key]=s;self.channels[key]=channels
        return self.sessions[key]

    def plan(self,request):
        with self.lock:
            require(not self.closed,'Runtime has retired')
            try:return self._plan(request)
            finally:
                if self.active is not None:self.active.finish_turn();self.active.cancelled.clear()
                self.active=None;self.interrupted.clear();self.cancelled.clear()

    def _plan(self,r):
        require(set(r)=={'protocol','request_id','view','choices','save','insolvencies'},'Unexpected game request fields')
        require(r['protocol']==PROTOCOL and isinstance(r['request_id'],str),'Game protocol differs')
        view=copy.deepcopy(r['view']);choices=r['choices'];self.boundary.check(view,choices)
        require(view['context']['phase'] in ('action','debt') and not view['context']['terminal'],'Completed action or mandatory debt boundary required')
        debt=view['context']['phase']=='debt'
        require(len(r['insolvencies'])==4 and all(type(n) is int and 0<=n<=16 for n in r['insolvencies']), 'Invalid public insolvency counts')
        p=self.package;save=sampling.restore(p,r['save'],r['save']['game_id']);seat=view['seat']
        own=next((s for s in save['seats'] if s['selection']['seat']==seat),None);require(own is not None,'Unassigned neural seat')
        # Native transport game_id is independent of the sampled game's engine ID.
        selected=own['selection'];member=next(m for m in p['roster']['members'] if m['id']==selected['member'])
        profile=Profile.parse(member['profile']);rng=random.Random();rng.setstate(sampling.tuples(own['rng']))
        session=self.session(seat,selected['tier'],profile)
        session.profiles[seat]=profile;session.begin(0,0)
        session.last_continuation={}
        for w in session.workers:w.last_statistics=None
        session.intents[seat].restore(own['intent']);session.recall[seat]=copy.deepcopy(own['recall'])
        session.cancelled.clear();self.active=session
        if self.interrupted.is_set():session.cancelled.set()
        start=time.monotonic();before=dict(session.telemetry);paths=[];path=[];kinds=[];qualified=0;deadlines=[];reason=None
        reward=p['reward'];cost=reward['insolvency'];spent=[min(cost['max_cost'],n*cost['event_cost']) for n in r['insolvencies']]
        ledger=CostLedger(reward,spent=spent)
        old_budget=save.get('pending');key=public_key(view);seconds=p['profiles']['tiers'][selected['tier']]['seconds']
        if not debt and old_budget and old_budget['key']==key:seconds=min(seconds,old_budget['remaining_seconds'])
        # Re-entering after Scout may reveal a new own card. Preserve the SAME
        # allowance, even across save/load; a zero budget means legal completion.
        session.config=replace(SearchConfig(**p['searches'][selected['tier']]),seconds=max(.001,seconds))
        if seconds<=0:session.cancelled.set()
        for w in session.workers:w.config=session.config
        source_seed=rng.randrange(2**63);state=self.game.from_observation(view,source_seed,session.recall[seat])
        require(state.legal==choices and state.views[seat]==view,'Live/simulator permitted observation differs')
        try:
            for _ in range(256):
                if self.cancelled.is_set():reason='cancelled';break
                if state.is_terminal() or state.is_chance_node() or state.current_player()!=seat or public_key(state.views[seat])!=key:break
                current=state.views[seat];legal=state.legal
                chosen,label,info=session.select(current,legal,rng,ledger)
                if session.turn is not None:deadlines.append(session.turn['deadline'])
                require(type(chosen) is int and 0<=chosen<len(legal),'Illegal policy result')
                qualified+=int(label is not None)
                if debt:
                    require(legal[chosen]['kind']=='liquidate' and not info['searched'],'Debt must use the existing policy-only legal liquidation path')
                    paths=[[legal[chosen]['id']]];kinds=['liquidate'];break
                if not paths and not path and info['searched']:
                    finished=completed_paths(session,current,legal,chosen)
                    if finished is not None:
                        paths=finished;kinds=[p[0].split('|',1)[0] for p in paths];break
                choice=legal[chosen];path.append(choice['id']);ledger.chosen(current,choice)
                if len(path)==1:kinds.append(choice['kind'])
                state.apply_action(action_id(choice['id']))
                if choice['kind']=='finish':paths.append(path);path=[]
                if not state.is_terminal() and not state.is_chance_node():ledger.observe(state.views[seat])
            require(reason=='cancelled' or (paths and not path),'No complete legal action; stop visibly')
            require(len(set(deadlines))<=1,'Whole-turn deadline was reset')
            if self.cancelled.is_set():reason='cancelled'
            started=session.turn['started'] if session.turn else time.monotonic()
            admitted_seconds=time.monotonic()-started
            # Each step above already passed exact native legality in this
            # detached observation world. Do not replay/serialize it a second
            # time here; the Unity bridge MUST revalidate against the live board.
            after=dict(session.telemetry);metrics={k:after[k]-before[k] for k in before}
            metrics.update(qualified_samples=qualified,total_seconds=time.monotonic()-start,
                admitted_seconds=admitted_seconds,budget_seconds=seconds,workers=2,
                turn_overrun_seconds=max(0.,admitted_seconds-seconds),
                deadline_resets=max(0,len(set(deadlines))-1),play_now=self.interrupted.is_set(),cancelled=reason=='cancelled')
            if reason=='cancelled':return dict(protocol=PROTOCOL,request_id=r['request_id'],status='cancelled',paths=[],save=r['save'],metrics=metrics)
            own.update(rng=rng.getstate(),intent=session.intents[seat].snapshot(),recall=copy.deepcopy(session.recall[seat]))
            display=IntentDisplay(seat)
            if own['ui'] is not None:
                display.__dict__.update(own['ui'])
                if display.last_turn is not None:display.last_turn=tuple(display.last_turn)
            group=next(g for g in p['roster']['archetypes'] if g['id']==selected['archetype'])
            public=display.update(era=view['context']['era'],turn=view['context']['round']*4+view['context']['turn'],
                description=group['description'],mode=p['profiles']['hint_mode'],
                action=PublicAction(kinds[0]) if kinds[0] in PUBLIC_TEXT else None,own_intent=own['intent'])
            own['ui']=copy.deepcopy(display.__dict__)
            if not debt:save['pending']=dict(key=key,remaining_seconds=max(0.,seconds-(time.monotonic()-started)))
            save=seal({k:v for k,v in save.items() if k!='id'})
            return dict(protocol=PROTOCOL,request_id=r['request_id'],status='ok',paths=paths,save=save,
                # The bridge displays this only after committing the action.
                public=public,metrics=metrics)
        finally:
            session.finish_turn();self.active=None
            # Workers/device requests have retired before either event is reset.
            session.cancelled.clear();self.interrupted.clear();self.cancelled.clear()
