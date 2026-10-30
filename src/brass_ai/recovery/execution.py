"""Persistent spawned CPU actors, private host/planner lanes, asynchronous replay writes."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
import gzip
import json
import multiprocessing
from pathlib import Path
import queue
import random
import threading
import time
import traceback
import uuid

import torch

from brass_ai.bank import sha
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import CONFIG_DIR, digest, read_json, require, project_path
from brass_ai.learning.replay import split_for, terminal_targets
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.step5.policy import NeuralPolicy, context_for
from brass_ai.search.ismcts import MaskedSearchPolicy
from brass_ai.search.common import SearchConfig
from .inference import RemoteEvaluator, endpoint_encoder, share_encoder_cache
from .telemetry import Timings, instrument_host
from .inputs import BOUNDARY
from .encoding import PreparedEncoder
from .replay import CheckedGame,SerializedGame,check_game,write_shard,read_shards,cache_receipt,write_cache


class _ActorSerializedGame(SerializedGame):
    """Reuse only the live actor's sealed input-validation proof.

    Keep the replay reader/validator and its derived-cache version unchanged:
    existing audited datasets must not need another schema/encoding pass.
    """
    def __init__(self,game):
        if not isinstance(game,CheckedGame) or not getattr(game,'actor_validated',False):
            super().__init__(game)
            return
        game.assert_metadata()
        encoder=PreparedEncoder()
        arrays=[game.encoded(row,encoder) for row in game['records']]
        # Seals catch input, label and context mutations; check_game still
        # verifies headers, decisions, choices and targets before publication.
        self.game=check_game(game,arrays,validated_inputs=True)
        self.line=json.dumps(self.game,separators=(',',':'),allow_nan=False)+'\n'
        self.size=len(self.line.encode('utf-8'))


def neural_spec(identity,exploration=.05,focus='neutral',*,version=2,opening=None,temperature=1.):
    from brass_ai.step5.policy import policy_identity
    policy_id=policy_identity(identity,exploration,temperature,focus,version,opening)
    extra=dict(exploration_version=version,opening=opening) if version==3 else {}
    if temperature!=1.:extra['temperature']=temperature
    return dict(kind='neural',model_id=identity,exploration=exploration,focus=focus,policy_id=policy_id,**extra)


def _collection_recipe(jobs,config):
    # Provenance describes the caller's intent, but older callers omit actual
    # model/opponent IDs or mixed-policy focuses. Bind all execution inputs;
    # output paths are transport locations and may change on relocation.
    for job in jobs:
        if any(p.get('frozen_rating') for p in job['policies']):
            provenance=job.get('provenance',{})
            require(provenance.get('producer')=='plackett-luce-rating/1.0.0' and
                    provenance.get('purpose')=='evaluation' and
                    provenance.get('kind') in ('rating','il-panel','v4-panel'),
                    'Frozen rating runtime is restricted to held-out rating jobs')
    return dict(format='actor-collection/1.0.0',execution=config,
        jobs=[dict(rotation=0,rng_index=0,focuses={})|{k:v for k,v in job.items() if k!='path'}
              for job in jobs])


def make_policy(spec,root,endpoint,config,mono,*,_replica=False):
    if spec['kind']=='lab-legacy':
        from .legacy_search import LabLegacyPolicy
        policy=LabLegacyPolicy(root,spec['parent'],spec['budget'],mono=mono,purpose=spec.get('purpose','training'),
            search_free=spec.get('search_free',False),work_budget=spec.get('work_budget'),search_runtime=spec.get('search_runtime',False),
            frozen_rating=spec.get('frozen_rating',False))
        require(policy.policy_id==spec['policy_id'],'Lab teacher identity changed')
        return policy
    if spec['kind']=='legacy':
        return LegacyPolicy(root,spec['policy_id'],purpose=spec.get('purpose','training'),mono=mono)
    endpoints=endpoint if isinstance(endpoint,list) else [endpoint]
    # A heavy actor can alternate legacy/ordinary searches and teacher games.
    # Reserve enough channels for either role, but only create the requested
    # role's forest. Teacher widening must not change evaluation opponents.
    from .config import search_workers_per_turn
    workers=search_workers_per_turn(config,teacher=bool(spec.get('teacher_schedule')))
    endpoints=endpoints[:1+workers if workers>1 else 1]
    evaluator=RemoteEvaluator(endpoints[0],spec['model_id'],config['batch_size'],config['max_candidate_cells'])
    if spec['kind']=='neural':
        policy=NeuralPolicy(None,exploration=spec['exploration'],focus=spec['focus'],evaluator=evaluator,
            exploration_version=spec.get('exploration_version',2),opening=spec.get('opening'),temperature=spec.get('temperature',1.))
        require(policy.policy_id==spec['policy_id'],'Neural policy identity changed')
        return policy
    opponents=[NeuralPolicy(None,exploration=.05,focus='neutral',
        evaluator=RemoteEvaluator(endpoints[0],p,config['batch_size'],config['max_candidate_cells']))
        for p in spec.get('opponents',[spec['model_id']])]
    from dataclasses import replace
    from .config import teacher_search_config,time_budget,TEACHER_TIME_BUDGET,NN_TIME_BUDGET,teacher_time_seconds
    search_config=teacher_search_config(spec['search']) if spec.get('teacher_schedule') else SearchConfig(**spec['search'])
    clock=(spec.get('teacher_work_budget') or TEACHER_TIME_BUDGET) if spec.get('teacher_schedule') else (spec.get('nn_work_budget') or NN_TIME_BUDGET)
    if not _replica and time_budget(clock):
        # The coordinator binds each phase at admission. This maximum only
        # initializes the policy; replicas inherit the same absolute deadline.
        seconds=teacher_time_seconds(clock) if spec.get('teacher_schedule') else clock['seconds_per_turn']
        search_config=replace(search_config,seconds=seconds)
    common=dict(config=replace(search_config,cpu_role=spec.get('cpu_role','reference')),mono=mono,focus=spec.get('focus','neutral'),
        opponents=opponents,evaluator=evaluator,fast_host=True)
    require(spec['kind']=='masked_ismcts','Only Masked ISMCTS is an active search policy')
    policy=MaskedSearchPolicy(None,root,**common)
    if len(endpoints)>1:
        from .parallel import parallelize
        replica_spec={k:v for k,v in spec.items() if k not in ('teacher_schedule','teacher_work_budget','nn_work_budget')}
        # Carry resolved training defaults into each private replica; only the
        # coordinator schedules real turns; replicas receive no new turn clocks.
        replica_spec['search']=search_config.to_dict()
        policy=parallelize(policy,len(endpoints)-1,spec.get('cpu_role','lab'),
            lambda i:make_policy(replica_spec,root,endpoints[i+1],config,mono,_replica=True))
    if spec.get('teacher_schedule'):
        require(not spec.get('nn_work_budget'),'Teacher phases and generic NN work cannot share a policy')
        from .teacher_schedule import CONTRACT,training_teacher
        from .config import TEACHER_TIME_BUDGET,current_work_budget
        require(spec['teacher_schedule']==CONTRACT,'Unknown training teacher schedule')
        policy=training_teacher(policy,work_budget=current_work_budget(spec.get('teacher_work_budget'),TEACHER_TIME_BUDGET))
    elif not _replica:
        from .teacher_schedule import work_search
        from .config import NN_TIME_BUDGET
        # This factory is the lab actor boundary (not installed-game or
        # personality deployment). Explicit experiment/evaluation specs bind
        # their budget; preflight/profiling callers receive the same shared clock.
        # Recursive private trees must not acquire independent turn budgets.
        policy=work_search(policy,spec.get('nn_work_budget') or NN_TIME_BUDGET)
    instrument_host(policy.game.host)
    for game in policy.simulation_games:instrument_host(game.host)
    return policy


def policy_encoder(policy,fallback=None):
    evaluator=getattr(policy,'evaluator',None) or getattr(getattr(policy,'base',None),'evaluator',None)
    return getattr(evaluator,'encoder',fallback)


def run_game(host,job,policies,timings,max_decisions,admission=None,encoder=None,submit_branch=None):
    pins=read_json(CONFIG_DIR/'host-1.0.0.json')
    with timings.phase('host.reset'): context=host.reset(job['seed'])
    if 'local_scoring' in job['provenance']:
        from .local_scoring_targets import CONTRACT
        producer=dict(runtime_sha256=sha(host.runtime_directory/'runtime.json'),host_sha256=sha(host.runtime_directory/'BrassHost.exe'))
        require(producer==job['provenance']['local_scoring']['producer'],'Local scoring runtime differs from collection provenance')
        require(host.call('capabilities').get('local_scoring')==CONTRACT,'Local scoring recorder required')
        host.call('local_scoring_enable')
    if hasattr(host,'enable_fastpath') and not host.fast_checked:host.enable_fastpath()
    assigned=[policies[(seat-job.get('rotation',0))%4] for seat in range(4)]
    group=digest(dict(rules=pins['rules'],setup_seed=job['seed']))
    game=dict(game_id=job.get('stable_game_id') or uuid.uuid4().hex,setup_group=group,players=4,split=split_for(group),
        opponents=[p.policy_id for p in assigned],result=None,records=[])
    rngs=[random.Random(digest(dict(sampling_seed=job['sampling_seed'],game=job.get('rng_index',0),seat=s))) for s in range(4)]
    for seat,p in enumerate(assigned):
        if hasattr(p,'begin_teacher_game'):
            require(job['provenance'].get('purpose')=='training' and job['provenance'].get('kind')=='masked-ismcts-teacher',
                    'Reserved search schedule is only for training teacher games')
            p.begin_teacher_game(rngs[seat],seat,job['seed'],job['sampling_seed'])
        if hasattr(p,'begin_work_game'):p.begin_work_game(rngs[seat],seat)
        if isinstance(p,NeuralPolicy) and p.focus=='mixed':
            p.contexts[rngs[seat]]=context_for(job.get('focuses',{}).get(str(seat),'neutral'))
    encoded=[]
    from .domino_sampling import install as install_strategies
    install_strategies(job,assigned,rngs)
    if encoder is None:
        encoder=next((e for p in assigned if (e:=policy_encoder(p)) is not None),None)
        if encoder is None:encoder=PreparedEncoder()
    from .resources import TurnLease
    lease=TurnLease(admission,timings)
    with ExitStack() as cleanup:
        cleanup.callback(lease.close)
        result=_run_turns(host,job,assigned,timings,max_decisions,context,game,rngs,encoded,encoder,lease,submit_branch)
        packet=getattr(result,'branch_packet',None)
        if packet is not None and submit_branch is None:
            from .branching import continue_branch
            child=continue_branch(host,job,result,packet,assigned,timings,max_decisions,encoder,lease)
            result['branches']=[child]
            result.bind(encoded)
        result.branch_deferred=packet is not None and submit_branch is not None
        result.branch_packet=None  # private checkpoint is owned by the queue
        teachers=[p.teacher_report(rngs[seat]) for seat,p in enumerate(assigned) if hasattr(p,'teacher_report')]
        if teachers:result.teacher_phases=teachers
        work=[p.work_report(rngs[seat]) for seat,p in enumerate(assigned) if hasattr(p,'work_report')]
        if work:result.search_work=work
        return result


def _run_turns(host,job,assigned,timings,max_decisions,context,game,rngs,encoded,encoder,lease,submit_branch=None):
    rewards=[p.config.reward for p in assigned if isinstance(p,MaskedSearchPolicy) and p.config.reward is not None]
    from brass_ai.learning.reward import CostLedger
    ledger=CostLedger(rewards[0]) if rewards else None
    require(not rewards or all(r==rewards[0] for r in rewards),'Mixed search reward contracts in one table')
    from collections import deque
    buffered=deque();decision=None;branch_packet=None;eligible_count=0
    for step in range(len(game['records']),max_decisions):
        if context['terminal']: break
        seat=context['actor'];p=assigned[seat]
        from brass_ai.search.common import turn_key
        workers=p.config.workers if isinstance(p,MaskedSearchPolicy) else getattr(p,'cpu_workers',1)
        if hasattr(p,'teacher_workers'):workers=p.teacher_workers(context,rngs[seat])
        if hasattr(p,'work_workers'):workers=p.work_workers(context,rngs[seat])
        lease.update(turn_key(context)+(seat,),workers)
        if hasattr(p,'admit_teacher_turn'):p.admit_teacher_turn(context,rngs[seat])
        if hasattr(p,'admit_work_turn'):p.admit_work_turn(context,rngs[seat])
        if hasattr(p,'admit_legacy_turn'):p.admit_legacy_turn(context,rngs[seat])
        after_forced=None
        if buffered:
            entry=buffered.popleft();current=entry['before'];after_forced=entry['after']
            require(current['context']==context,'Forced trace context changed')
        elif getattr(host,'fast',False):
            with timings.phase('host.decision'):current=decision or host.decision()
            require(current['context']==context,'Fused decision context changed')
        else:current=None
        if current is not None:view,choices=current['view'],current['choices']
        else:
            with timings.phase('host.observe'): view=host.observe(seat)
            with timings.phase('host.choices'): choices=host.choices()
        with timings.phase('inputs.validate'): BOUNDARY.check(view,choices)
        require(view['seat']==seat and view['context']['revision']==context['revision'],'Stale actor observation')
        if ledger is not None:
            ledger.observe(view)
            if isinstance(p,MaskedSearchPolicy) and p.config.reward is not None:p.set_reward_context(ledger)
        began=time.monotonic()
        with timings.phase('policy.'+p.kind):
            if type(p) is NeuralPolicy and len(choices)==1:
                _,intent=p.context_features(view,rngs[seat])
                # Match rng.choices' draw so future nonforced sampling is unchanged.
                if p.exploration_version==3:
                    import numpy as np
                    chosen,probability,teacher=p.sample_training(view,choices,rngs[seat],np.ones(1),intent)
                else:
                    rngs[seat].choices([0],weights=[1.0],k=1)
                    chosen,probability,teacher=0,1.0,None
            else: chosen,probability,teacher=p.select_batch([(view,choices,rngs[seat])])[0]
        require(type(chosen) is int and 0<=chosen<len(choices),'Actor returned an illegal choice')
        style,intent=p.context_features(view,rngs[seat]) if hasattr(p,'context_features') else ([0.0]*4,[0.0]*8)
        game['records'].append(dict(step=step,seat=seat,view=view,choices=choices,chosen=chosen,
            teacher=teacher,style=style,intent=intent,seconds=time.monotonic()-began,
            behavior=dict(dict(kind=p.kind,policy_id=p.policy_id,probability=probability,temperature=getattr(p,'temperature',1.0)),
                **(p.last_behavior[rngs[seat]] if type(p) is NeuralPolicy and p.exploration_version==3 else {}))))
        if job.get('branching'):
            from .branching import eligible,capture,seed_for
            row=game['records'][-1]
            if eligible(view,choices,row['behavior'].get('action_decision',0),job['branching']['skip_opening_decisions']):
                eligible_count+=1
                # Selection uses an independent stream; parent actions and
                # training RNG do not change when branching is enabled.
                draw=random.Random('select:'+seed_for(game['game_id'],step,job['branching']['contract'])).random()
                if branch_packet is None and draw<job['branching']['rate_per_decision']:
                    branch_packet=capture(host,game,row,assigned,rngs,portable=submit_branch is not None,branching=job['branching'])
                    if branch_packet is not None and submit_branch is not None:
                        submit_branch(branch_packet,{k:v for k,v in game.items() if k not in ('records','result')})
        e=policy_encoder(p,encoder)
        with timings.phase('inputs.encode'): encoded.append(e.encode(view,choices,style,intent,validated=True))
        if ledger is not None:ledger.chosen(view,choices[chosen])
        with timings.phase('host.apply'):
            if after_forced is not None:
                require(type(p) is NeuralPolicy and len(choices)==1 and chosen==0,'Unsafe forced trace')
                context=after_forced
            elif getattr(host,'fast',False):
                limit=min(16,max_decisions-step-1) if type(p) is NeuralPolicy else 0
                reply=host.step_decision(context,choices[chosen],limit)
                decision=reply['decision'];buffered.extend(reply['forced'])
                context=buffered[0]['before']['context'] if buffered else decision['context']
            else:context=host.apply(context,choices[chosen])
        if hasattr(p,'finish_teacher_turn') and (context['terminal'] or context['actor']!=seat or
                turn_key(context)!=turn_key(view['context']) or context['phase']=='debt'):
            p.finish_teacher_turn(rngs[seat])
        if hasattr(p,'finish_work_turn') and (context['terminal'] or context['actor']!=seat or
                turn_key(context)!=turn_key(view['context']) or context['phase']=='debt'):
            p.finish_work_turn(rngs[seat])
    require(context['terminal'],'Decision cap reached; incomplete game excluded')
    with timings.phase('host.result'): game['result']=host.call('result')
    terminal_targets(game['result'])
    if 'local_scoring' in job['provenance']:
        from .local_scoring_targets import bind,FIELD
        with timings.phase('host.local_scoring'):
            game[FIELD]=bind(host.call('local_scoring_targets'),game,job['provenance']['local_scoring']['producer'])
    if job.get('branching'):
        game['branch_statistics']=dict(eligible_decisions=eligible_count)
    result=CheckedGame(game).bind(encoded)
    result.branch_packet=branch_packet
    result.actor_validated=True  # every row crossed BOUNDARY.check above
    return result


def _actor(root,host_path,jobs,results,lease,config,mono,lanes,search_channels,admission,actor_index,inboxes):
    root=Path(root);torch.set_num_threads(config['worker_threads'])
    stop=threading.Event();slots=threading.Semaphore(2*lanes)
    with ThreadPoolExecutor(max_workers=1) as writer:
        def submit_publication(completed,packet,child=None):
            slots.acquire()
            def publish():
                try:
                    began=time.monotonic()
                    if child is not None:
                        from .branch_scheduler import attach_child
                        require(len(completed)==1,'Branch family must have one parent')
                        attach_child(completed[0][0],child)
                    parts=[];part=[];size=0;serialized={}
                    for item in completed:
                        value=_ActorSerializedGame(item[0]);serialized[item[0]['game_id']]=value;weight=value.size
                        if part and size+weight>config.get('shard_bytes',268435456):parts.append(part);part=[];size=0
                        part.append(item);size+=weight
                    if part:parts.append(part)
                    records=[]
                    for number,part in enumerate(parts,packet['part_start']):
                        path=Path(packet['prefix']+f'-part-{number:04d}.jsonl.gz')
                        provenance=dict(producer='recovery/2.1.0',purpose=part[0][1]['provenance']['purpose'],
                            collection=packet['identity'],recipe=packet['recipe'],
                            jobs=[dict(index=j['index'],seed=j['seed'],provenance=j['provenance'],
                                **({'teacher_phases':g.teacher_phases} if hasattr(g,'teacher_phases') else {}),
                                **({'search_work':g.search_work} if hasattr(g,'search_work') else {})) for g,j,_,_ in part])
                        write_shard(path,[serialized[g['game_id']] for g,_,_,_ in part],provenance,cache=config.get('feature_cache',False))
                        receipt=cache_receipt(path);replay_hash=receipt['replay_sha256'] if receipt else sha(path)
                        for g,j,metrics,telemetry in part:
                            records.append(dict(job=j['index'],game_id=g['game_id'],path=str(path),sha256=replay_hash,cached=False,
                                features=receipt,metrics=metrics,search=telemetry,
                                worker_seconds=g.actor_elapsed_seconds,
                                actor_python_cpu_seconds=g.actor_python_cpu_seconds,
                                **({'branch_execution':g.branch_execution} if hasattr(g,'branch_execution') else {}),
                                **({'teacher_phases':g.teacher_phases} if hasattr(g,'teacher_phases') else {}),
                                **({'search_work':g.search_work} if hasattr(g,'search_work') else {})))
                    elapsed=time.monotonic()-began
                    for record in records:record.update(replay_write_seconds=elapsed/len(completed),replay_write_scope='amortized packet')
                    results.put(dict(records=records))
                except BaseException:
                    stop.set();results.put(dict(error=traceback.format_exc()))
                finally:slots.release()
            writer.submit(publish)

        from .branch_scheduler import FamilyMailbox,pack_child
        families=FamilyMailbox(inboxes[actor_index],results,stop,submit_publication)
        def lane():
            endpoint=[lease.acquire() for _ in range(search_channels)];timings=Timings()
            encoding_cache=share_encoder_cache(endpoint)
            cache=OrderedDict()
            try:
                with ExitStack() as stack:
                    with timings.phase('host.startup'):
                        from .host_runtime import training_host
                        selected=project_path(root,host_path)
                        if selected==root/'artifacts/r1-host/guarded':selected=training_host(root)
                        host=stack.enter_context(RulesHost(selected,mono=mono))
                    instrument_host(host,timings)
                    while not stop.is_set():
                        packet=jobs.get()
                        if packet is None: break
                        completed=[]
                        for job in packet['jobs']:
                            policies=[];needed={digest(s) for s in job['policies']}
                            for key in list(cache):
                                # Search owns persistent private simulator lanes.
                                # Retire an unused model generation promptly;
                                # the light NN cache can retain its normal bound.
                                if key not in needed and (isinstance(cache[key],MaskedSearchPolicy) or len(cache)>=config.get('policy_cache',16)):
                                    old=cache.pop(key)
                                    if hasattr(old,'close'): old.close()
                            for spec in job['policies']:
                                key=digest(spec)
                                if key not in cache: cache[key]=make_policy(spec,root,endpoint,config,mono)
                                cache.move_to_end(key);policies.append(cache[key])
                            cpu_started=time.process_time()
                            encoding_before=encoding_cache.report()
                            started=time.monotonic()
                            if 'continuation' in packet:
                                from .branching import continue_branch
                                from .resources import TurnLease
                                assigned=[policies[(s-job.get('rotation',0))%4] for s in range(4)]
                                turn_lease=TurnLease(admission,timings)
                                try:
                                    game=continue_branch(host,job,packet['parent'],packet['continuation'],assigned,
                                        timings,config['max_decisions'],endpoint_encoder(endpoint[0]),turn_lease)
                                finally:turn_lease.close()
                            else:
                                def fork(continuation,parent,packet=packet):
                                    results.put(dict(fork=dict(packet,continuation=continuation,parent=parent,
                                        origin_actor=actor_index,queued=time.monotonic())))
                                game=run_game(host,job,policies,timings,config['max_decisions'],admission,
                                    encoder=endpoint_encoder(endpoint[0]),submit_branch=fork if packet.get('schedule_token') else None)
                            finished=time.monotonic()
                            # process_time counts actual Python-process CPU,
                            # including its threads. Multiple simultaneous host
                            # lanes would overlap this interval, so omit them.
                            game.actor_python_cpu_seconds=time.process_time()-cpu_started if lanes==1 else None
                            game.actor_elapsed_seconds=finished-started
                            after=encoding_cache.report()
                            telemetry={'encoding_cache':dict(
                                **{k:after[k]-encoding_before[k] for k in ('warm_hit','miss','evictions')},
                                retained=after['retained'],capacity=after['capacity'],
                                scope='one game; shared coordinator/private-worker input cache')}
                            for i,p in enumerate(policies):
                                entry={}
                                if hasattr(p,'telemetry'): entry['search']=dict(p.telemetry)
                                evaluator=getattr(p,'evaluator',getattr(getattr(p,'base',None),'evaluator',None))
                                if evaluator is not None: entry['inference']=dict(evaluator.telemetry)
                                if hasattr(getattr(p,'host',None),'recovery_metrics'):
                                    entry['host']=p.host.recovery_metrics.report()
                                if entry: telemetry[str(i)]=entry
                            completed.append((game,job,timings.report(),telemetry))
                            if packet.get('schedule_token'):
                                results.put(dict(work_complete=dict(token=packet['schedule_token'],job=job['index'],
                                    kind='branch' if 'continuation' in packet else 'parent',
                                    table_kind=job['provenance']['kind'],started=started,finished=finished,
                                    queued=packet.get('queued',started))))
                            timings=Timings();instrument_host(host,timings)
                        if 'continuation' in packet:
                            slots.acquire()
                            def return_child(completed=completed,packet=packet):
                                try:
                                    g,j,metrics,telemetry=completed[0]
                                    inboxes[packet['origin_actor']].put(dict(key=FamilyMailbox.key(packet),payload=pack_child(g),
                                        execution=dict(metrics=metrics,search=telemetry,worker_seconds=g.actor_elapsed_seconds,
                                            actor_python_cpu_seconds=g.actor_python_cpu_seconds)))
                                except BaseException:
                                    stop.set();results.put(dict(error=traceback.format_exc()))
                                finally:slots.release()
                            writer.submit(return_child)
                        elif any(getattr(g,'branch_deferred',False) for g,_,_,_ in completed):
                            families.parent(completed,packet)
                        else:submit_publication(completed,packet)
            except BaseException:
                stop.set();results.put(dict(error=traceback.format_exc()))
            finally:
                for p in cache.values():
                    if hasattr(p,'close'): p.close()
                for e in endpoint:e.close()
        try:
            with ThreadPoolExecutor(max_workers=lanes) as actors:
                futures=[actors.submit(lane) for _ in range(lanes)]
                for future in futures:future.result()
        finally:families.close()


class ActorPool:
    def __init__(self,root,host,service,config,*,mono=None,heavy=False):
        self.root,self.host,self.service,self.config=root,host,service,config
        self.mono,self.heavy=mono,heavy
        self.processes=[];self.mp=multiprocessing.get_context('spawn')
        self.jobs=self.mp.Queue();self.results=self.mp.Queue()
        self.width=config['heavy_processes'] if heavy else config['processes']
        self.lanes=1 if heavy else config['hosts_per_process']
        self.inboxes=[self.mp.Queue() for _ in range(self.width)]
        self.last_collection_report=None;self.duration_estimates={}

    def start(self,target=None):
        target=min(self.width,target or self.width)
        if len(self.processes)>=target: return
        from .host_runtime import training_host
        training_host(self.root)
        if self.heavy:training_host(self.root,'search')
        from .config import search_workers_per_turn
        workers=max(search_workers_per_turn(self.config),search_workers_per_turn(self.config,teacher=True))
        search_channels=1+workers if self.heavy and workers>1 else 1
        lease=self.service.channels((target-len(self.processes))*self.lanes*search_channels)
        for _ in range(target-len(self.processes)):
            p=self.mp.Process(target=_actor,args=(str(self.root),str(self.host),self.jobs,self.results,
                lease,self.config,self.mono,self.lanes,search_channels,self.service.cpu,len(self.processes),self.inboxes))
            p.start();self.processes.append(p)

    def collect(self,jobs,*,on_records=None):
        try:
            return self._collect(jobs,on_records=on_records)
        except BaseException as error:
            # Readers and other actor pools share this admission queue. A
            # failed actor may own slots or a queued ticket that will never be
            # released once its process is terminated; wake every waiter before
            # any enclosing executor starts waiting for its workers to exit.
            self.service.cpu.abort()
            # Another actor may report only the resulting admission abort.
            # Surface the service's original failure while retaining that
            # actor traceback and preserving explicit user interruptions.
            if isinstance(error,Exception) and self.service.error is not None:
                raise ValueError('Inference service failed: '+str(self.service.error)) from error
            raise

    def _collect(self,jobs,*,on_records=None):
        require(jobs and len({j['index'] for j in jobs})==len(jobs),'Missing/duplicate actor jobs')
        scheduled=self._branch_scheduling(jobs)
        self.last_collection_report=None
        records=[];pending=[]
        buckets=defaultdict(list)
        for job in jobs:
            group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
            buckets[(str(Path(job['path']).parent),split_for(group))].append(job)
        for bucket in buckets.values():
            size=min(self.config.get('shard_games',1),max(1,len(bucket)//(self.width*self.lanes)))
            # One schedulable parent per packet: no long sequential packet can
            # strand ready continuations or hold a complete family unpublished.
            if scheduled:size=1
            for start in range(0,len(bucket),size):
                group=bucket[start:start+size]
                recipe=_collection_recipe(group,self.config);identity=digest(recipe)
                prefix=str(Path(group[0]['path']).parent/('games-'+identity[:16]))
                expected={j['index']:j for j in group};existing=sorted(Path(prefix).parent.glob(Path(prefix).name+'-part-*.jsonl.gz'))
                seen=set()
                for path in existing:
                    with gzip.open(path,'rt',encoding='utf-8') as stream: header=json.loads(next(stream))
                    proof=header['provenance'];require(proof.get('collection')==identity,'Cached collection differs')
                    require(proof.get('recipe')==recipe,'Cached actor execution recipe differs')
                    games=read_shards([path]);require(len(games)==len(proof['jobs']),'Cached shard membership differs')
                    # A crash may leave an uncommitted derived cache. Recreate it
                    # from authoritative validated records before issuing a receipt.
                    if self.config.get('feature_cache'): write_cache(path,games)
                    loaded=[]
                    for item,game in zip(proof['jobs'],games):
                        index=item['index'];require(index in expected and index not in seen,'Duplicate/foreign cached job')
                        require(item['provenance']==expected[index]['provenance'],'Cached actor game differs')
                        require(game['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=expected[index]['seed'])),'Cached setup differs')
                        seen.add(index);loaded.append(dict(job=index,game_id=game['game_id'],path=str(path),sha256=sha(path),
                            cached=True,features=cache_receipt(path),
                            **({'teacher_phases':item['teacher_phases']} if 'teacher_phases' in item else {}),
                            **({'search_work':item['search_work']} if 'search_work' in item else {})))
                    records.extend(loaded)
                    if on_records is not None:on_records(loaded)
                missing=[j for j in group if j['index'] not in seen]
                if missing: pending.append(dict(jobs=missing,prefix=prefix,identity=identity,recipe=recipe,part_start=len(existing)))
        if pending:
            self.start(len(pending))
            if scheduled:
                return self._collect_branches(jobs,pending,records,on_records)
            if not self.heavy and self.config.get('model_cohorts',False):
                from .branch_scheduler import model_cohort
                pending.sort(key=lambda p:(model_cohort(p),p['jobs'][0]['index']))
            for packet in pending: self.jobs.put(packet)
            outstanding={j['index'] for packet in pending for j in packet['jobs']}
            while outstanding:
                try: record=self.results.get(timeout=1)
                except queue.Empty:
                    self.service.cpu.check()
                    require(all(p.is_alive() for p in self.processes),'Actor process exited before completing its games')
                    require(self.service.error is None,'Inference service failed')
                    continue
                require('error' not in record,record.get('error','Actor failed'))
                for item in record['records']:
                    require(item['job'] in outstanding,'Duplicate/misrouted actor result')
                    outstanding.remove(item['job']);records.append(item)
                if on_records is not None:on_records(record['records'])
                print(json.dumps(dict(phase='collection',complete=len(records),of=len(jobs),heavy=self.heavy)),flush=True)
        elif scheduled:
            from .branch_scheduler import report
            now=time.monotonic()
            self.last_collection_report=report([],now,now,0,len(records))
        return sorted(records,key=lambda r:r['job'])

    def _branch_scheduling(self,jobs):
        return any(j.get('branching') for j in jobs)

    def _collect_branches(self,jobs,pending,records,on_records):
        if on_records is None:return self._dispatch_branches(jobs,pending,records,None)
        # Reader submission can wait on StreamingCollection's bounded slots.
        # Keep it off the dispatcher so a full reader queue cannot postpone a
        # newly captured fork or leave ordinary lanes idle. Receipts are small
        # and bounded by this collection's finite parent-job count.
        with ThreadPoolExecutor(max_workers=1) as feeder:
            futures=[]
            def submit(items):
                for future in futures:
                    if future.done():future.result()
                futures.append(feeder.submit(on_records,items))
            try:
                result=self._dispatch_branches(jobs,pending,records,submit)
                began=time.monotonic()
                for future in futures:future.result()
                elapsed=time.monotonic()-began
                self.last_collection_report['ingress_submission_join_seconds']=elapsed
                self.last_collection_report['collection_wall_seconds']+=elapsed
                self.last_collection_report['publication_tail_seconds']+=elapsed
                return result
            except BaseException:
                # Release reader admission before executor shutdown waits for
                # a submission already blocked on reader backpressure.
                self.service.cpu.abort()
                for future in futures:future.cancel()
                raise

    def _dispatch_branches(self,jobs,pending,records,on_records):
        from .branch_scheduler import ReadyWork,report
        started=time.monotonic();token=uuid.uuid4().hex
        pending=[dict(p,schedule_token=token) for p in pending]
        ready=ReadyWork(pending,getattr(self,'duration_estimates',{}),getattr(self,'config',{}).get('model_cohorts',False))
        outstanding={p['jobs'][0]['index'] for p in pending}
        workers=len(self.processes)*self.lanes
        require(workers>0,'Branch scheduling needs live ordinary workers')
        running=set();forked=set();events=[];cached=len(records)
        while outstanding or running:
            while ready and len(running)<workers:
                packet=ready.pop();key=('branch' if 'continuation' in packet else 'parent',packet['jobs'][0]['index'])
                require(key not in running,'Duplicate scheduled job')
                running.add(key);self.jobs.put(packet)
            try:message=self.results.get(timeout=1)
            except queue.Empty:
                self.service.cpu.check()
                require(all(p.is_alive() for p in self.processes),'Actor process exited before completing its games')
                require(self.service.error is None,'Inference service failed')
                continue
            require('error' not in message,message.get('error','Actor failed'))
            if 'fork' in message:
                packet=message['fork'];index=packet['jobs'][0]['index']
                require(packet['schedule_token']==token and ('parent',index) in running and index not in forked,
                    'Duplicate, late or foreign branch request')
                forked.add(index);ready.add_branch(packet)
            elif 'work_complete' in message:
                event=message['work_complete'];key=(event['kind'],event['job'])
                require(event['token']==token and key in running,'Duplicate/foreign completion timing')
                running.remove(key);events.append(event)
            else:
                for item in message['records']:
                    require(item['job'] in outstanding,'Duplicate/misrouted actor result')
                    outstanding.remove(item['job']);records.append(item)
                if on_records is not None:on_records(message['records'])
                print(json.dumps(dict(phase='collection',complete=len(records),of=len(jobs),heavy=self.heavy)),flush=True)
        require(not ready and len(forked)==sum(e['kind']=='branch' for e in events),'Unfinished branch work')
        self.last_collection_report=report(events,started,time.monotonic(),workers,cached)
        durations=defaultdict(list)
        for e in events:
            if e['kind']=='parent':durations[e['table_kind']].append(e['finished']-e['started'])
        self.duration_estimates={k:sum(v)/len(v) for k,v in durations.items()}
        return sorted(records,key=lambda r:r['job'])

    def close(self,failed=False):
        if failed:
            self.service.cpu.abort()
            for p in self.processes:
                if p.is_alive(): p.terminate()
        else:
            for _ in range(len(self.processes)*self.lanes): self.jobs.put(None)
        for p in self.processes:
            p.join(timeout=30)
            if p.is_alive(): p.terminate();p.join(timeout=5)
        # No pending input messages should be flushed into a failed consumer.
        self.jobs.cancel_join_thread();self.results.cancel_join_thread()
        self.jobs.close();self.results.close()
        for inbox in getattr(self,'inboxes',()):
            inbox.cancel_join_thread();inbox.close()

    def __enter__(self): return self
    def __exit__(self,kind,*_): self.close(failed=kind is not None)
