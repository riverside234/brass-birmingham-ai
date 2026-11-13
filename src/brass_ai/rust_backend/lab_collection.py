"""Opt-in B02 frozen ordinary jobs. No trainer default/source admission changes.

The trusted adapter deliberately retains Python's exact policy/branch RNG and
versioned probability code. Rust owns states, legal sets, features and buffers.
"""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
import random
import time

import numpy as np
from brass_ai.learning.contracts import digest, require
from brass_ai.learning.replay import split_for
from brass_ai.step5.policy import NeuralPolicy, context_for
from brass_ai.recovery.branching import eligible, capture, seed_for, RESET_OPENINGS
from brass_ai.recovery.inputs import BOUNDARY
from .lab_transport import CONTRACT, VERSION, DecisionEncoder


def frozen_id(job):
    return digest({k:v for k,v in job.items() if k != 'path'})


def branch_opening(job, parent_opening):
    """Use the reference branch window without changing the frozen policy ID."""
    from brass_ai.recovery.exploration import FORK_T15_BRANCH, THREE_OPENING
    config=job['branching'];contract=config['contract']
    if contract not in RESET_OPENINGS:return parent_opening
    opening=RESET_OPENINGS[contract]
    expected=THREE_OPENING if contract==FORK_T15_BRANCH['contract'] else opening
    require(config['reset_opening_at_fork'] is True and parent_opening==expected
            and config.get('continuation_opening',opening)==opening, 'Branch opening policy changed')
    return copy.deepcopy(opening)


def route(job):
    """Never redraw a table or hide unsupported coverage behind a heuristic."""
    require(len(job['policies']) == 4, 'Frozen ordinary job needs four policies')
    if any(p['kind'] not in ('neural', 'legacy', 'lab-legacy') for p in job['policies']):
        return 'reference', 'enabled search/unsupported policy is outside B02'
    if any(p['kind'] != 'neural' and not p.get('search_free',False) for p in job['policies']):
        return 'reference', 'searched legacy policy is outside ordinary collection'
    if job.get('branching'):
        from brass_ai.recovery.branching import CONTRACTS
        require(job['branching']['contract'] in CONTRACTS and job['branching']['max_per_game'] == 1,
                'Unsupported branch contract')
        require(all(p.get('exploration_version') == 3 for p in job['policies']), 'Branch policies differ')
    return 'rust', None


class ForkCapture:
    def __init__(self, lane): self.lane = lane
    def call(self, op, **fields):
        require(op == 'fork_save' and fields == {'export': True}, 'Invalid trusted fork operation')
        return dict(checkpoint=self.lane.checkpoint(self.lane.current['decision']['view']['context']['revision']))


def _causal(row):
    """Only public fields used by insolvency.events; no features/private checkpoint."""
    v = row['view']
    return dict(chosen=row['chosen'], choices=row['choices'], view={k:v[k] for k in ('context','players','slots')})


def collect_lane(root, executable, bootstrap, job, endpoint, execution, *, origin=None,
                 continuation=None, submit_branch=None, cancel=None, compare=None, diagnostic_branch_probe=False,
                 session=None, mono=None, raw_export=False, validation=None):
    require(route(job)[0] == 'rust', 'Unsupported whole job must use reference route')
    start = time.monotonic(); timings = Counter(); job_id = frozen_id(job)
    with ExitStack() as cleanup:
        from brass_ai.recovery.replay_binding import scope
        binding_counts=cleanup.enter_context(scope())
        from .lab_session import OrdinarySession
        if session is None:
            session=cleanup.enter_context(OrdinarySession(root,executable,endpoint,execution,mono))
        lane,policies=cleanup.enter_context(session.game(job,timings,cancel,compare))
        assigned = [policies[(s-job.get('rotation',0))%4] for s in range(4)]
        from brass_ai.recovery import ordinary_cotton
        proof = job['provenance']
        cotton_seats = ordinary_cotton.validate_assignment(proof) if ordinary_cotton.KEY in proof else ()
        group = digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
        game_id = job.get('stable_game_id') or digest(dict(job=job_id, seed=job['seed']))
        game = dict(game_id=game_id,setup_group=group,players=4,split=split_for(group),
                    opponents=[p.policy_id for p in assigned],result=None,records=[])
        cap = execution['max_decisions']; causal = []; forked = False; eligible_count = 0
        if continuation is None:
            t = time.monotonic(); origin = origin or bootstrap.call('init',seed=job['seed'])
            timings['bootstrap_seconds'] += time.monotonic()-t
            lane.begin(job_id, origin, cap)
            rngs = [random.Random(digest(dict(sampling_seed=job['sampling_seed'],game=job.get('rng_index',0),seat=s))) for s in range(4)]
            for s,p in enumerate(assigned):
                if getattr(p,'focus',None) == 'mixed': p.contexts[rngs[s]] = context_for(job.get('focuses',{}).get(str(s),'neutral'))
            from brass_ai.recovery.domino_sampling import install
            install(job,assigned,rngs)
        else:
            packet = continuation; meta = packet['metadata']; seed = meta['seed']; forked = True
            require(packet.get('sha256')==digest({k:v for k,v in packet.items() if k!='sha256'}),'Branch packet changed in transit')
            require(packet['job_id'] == job_id and meta['parent_game_id'] == game_id, 'Foreign branch packet')
            require(meta['contract'] == job['branching']['contract'], 'Branch contract changed')
            cp = packet['checkpoint']; count = len(cp['payload']['state']['cards']['deck'])
            indices = bootstrap.call('shuffle',count=count,seed=int(seed[:8],16)&0x7fffffff) if job['branching']['reshuffle_hidden_deck'] else list(range(count-1,0,-1))
            lane.begin(job_id, {}, cap, checkpoint=cp, indices=indices)
            rngs = [random.Random(digest(dict(branch=seed,seat=s))) for s in range(4)]
            for s,p in enumerate(assigned):
                p.contexts[rngs[s]] = copy.deepcopy(packet['contexts'][s])
                reset = meta['contract'] in RESET_OPENINGS
                p.opening = branch_opening(job,p.opening)
                p.action_counts[rngs[s]] = 0 if reset else packet['counts'][s]
            row = copy.deepcopy(packet['row']); row['step'] = 0; row['chosen'] = meta['chosen']
            row['behavior'].update(probability=meta['marginal_probability'],component='branch_move')
            require(lane.current['decision']['view'] == row['view'] and lane.current['decision']['choices'] == row['choices'], 'Fork origin differs')
            encoded=assigned[row['seat']].evaluator.encoder.encode(row['view'],row['choices'],row['style'],row['intent'])
            if compare: compare(row, encoded, lane)
            lane.append(row, dict(kind='excluded-fork',metadata=meta))
            game.update(game_id=game_id+'/branch/'+seed,branch=copy.deepcopy(meta))
        while not lane.current.get('terminal'):
            require(cancel is None or not cancel.is_set(), 'Collection canceled; incomplete family discarded')
            decision = lane.current['decision']; view, choices = decision['view'],decision['choices']
            # Same immutable admitted frame as the reference actor: policy and
            # encoder reuse its exact input identity instead of hashing it again.
            seat=view['seat']; p=assigned[seat]; rng=rngs[seat]
            neural=type(p) is NeuralPolicy
            if validation is not None and not validation['semantic']:
                from .lab_perf import admit
                view,choices=admit(view,choices)
                decision.update(view=view,choices=choices)
            elif neural and len(choices)>1:
                view,choices=BOUNDARY.admit(view,choices)
                # Same immutable admitted objects flow through policy, encoder
                # and append. Avoid comparing each history against a duplicate
                # mutable JSON tree at both inner-loop boundaries.
                decision.update(view=view,choices=choices)
            else:BOUNDARY.check(view,choices)
            t=time.monotonic()
            if hasattr(p,'admit_legacy_turn'):p.admit_legacy_turn(view['context'],rng)
            if neural and len(choices)==1:
                _,intent=p.context_features(view,rng)
                if p.exploration_version==3: chosen,probability,teacher=p.sample_training(view,choices,rng,np.ones(1),intent)
                else: rng.choices([0],weights=[1.],k=1); chosen,probability,teacher=0,1.,None
            elif seat in cotton_seats:
                chosen,probability,teacher=p.select_batch([(view,choices,rng)],cotton=True)[0]
            else: chosen,probability,teacher=p.select_batch([(view,choices,rng)])[0]
            style,intent=p.context_features(view,rng)
            behavior=dict(kind=p.kind,policy_id=p.policy_id,probability=probability,temperature=getattr(p,'temperature',1.))
            if neural and p.exploration_version==3: behavior.update(p.last_behavior[rng])
            row=dict(step=lane.current['step'],seat=seat,view=view,choices=choices,chosen=chosen,teacher=teacher,
                     style=style,intent=intent,seconds=time.monotonic()-t,behavior=behavior)
            timings['policy_seconds']+=row['seconds']
            t=time.monotonic()
            encoder=p.evaluator.encoder if neural else DecisionEncoder(lane,'ordinary-external')
            encoded=encoder.encode(view,choices,style,intent,validated=True) if compare or (neural and len(choices)>1) else None
            timings['encoding_seconds']+=time.monotonic()-t
            evidence=dict(kind='ordinary',model_id=p.evaluator.model_id if neural else None,policy_id=p.policy_id,forced=len(choices)==1,
                raw_probabilities=(np.ones(1) if len(choices)==1 else p.native_raw).tolist() if neural else None,
                feature_version=VERSION,semantic_id=choices[chosen]['id'])
            if job.get('branching') and continuation is None:
                if not forked: causal.append(_causal(row))
                if eligible(view,choices,behavior.get('action_decision',0),job['branching']['skip_opening_decisions']):
                    eligible_count+=1
                    draw=random.Random('select:'+seed_for(game_id,row['step'],job['branching']['contract'])).random()
                    if not forked and (draw<job['branching']['rate_per_decision'] or diagnostic_branch_probe):
                        packet=capture(ForkCapture(lane),dict(game,records=causal),row,assigned,rngs,portable=True,branching=job['branching'])
                        if packet is not None:
                            packet['job_id']=job_id; forked=True; causal.clear()
                            packet['sha256']=digest(packet)
                            require(submit_branch is not None, 'Native fork needs an asynchronous family scheduler')
                            submit_branch(packet)
            if compare: compare(row,encoded,lane)
            t=time.monotonic(); lane.append(row,evidence,allow_forced=neural); timings['native_step_seconds']+=time.monotonic()-t
        if raw_export:
            t=time.monotonic();receipt,pages,features=lane.completed_export()
            timings['combined_completion_export_seconds']+=time.monotonic()-t
        else:receipt=lane.call('finish')
        records=[]; evidence=[]
        require(receipt['contract']==CONTRACT and receipt['job']==job_id and receipt['token']==lane.token,
                'Foreign native completion receipt')
        if raw_export:
            # No complete JSON tree is decoded/re-encoded in the actor. The
            # independently admitted publisher checks exact canonical seals.
            from .lab_shared import Export
            require(len(features)==receipt['rows'],'Incomplete raw features')
            game['result']=receipt['result'];game.pop('records')
            if continuation is None and job.get('branching'):game['branch_statistics']=dict(eligible_decisions=eligible_count)
            if continuation is not None:
                game['branch'].update(elapsed_seconds=time.monotonic()-start,
                    fraction_remaining=receipt['rows']/(receipt['rows']+game['branch']['decision']))
            timings.update({'ipc_'+k:v for k,v in lane.metrics.items()})
            producer=dict(contract=CONTRACT,core=lane.hello['core'])
            lane.release()
            timings.update({'binding_'+k:v for k,v in binding_counts.items()})
            return Export(dict(game=game,scoring=receipt['scoring'],native_receipt=receipt,
                origin_counts=continuation['counts'] if continuation is not None else [0]*4,
                diagnostic_branch_probe=diagnostic_branch_probe,
                branch_expected=forked and continuation is None,
                actor_validated=validation is None or validation['semantic'],
                timings=dict(timings,total_seconds=time.monotonic()-start),producer=producer),pages,features)
        record_hash=hashlib.sha256(b'[');evidence_hash=hashlib.sha256(b'[')
        t=time.monotonic()
        for index,(row_bytes,evidence_bytes) in enumerate(lane.records(receipt,drain=True)):
            require(cancel is None or not cancel.is_set(), 'Collection canceled during export')
            for values,hasher,raw_bytes in ((records,record_hash,row_bytes),(evidence,evidence_hash,evidence_bytes)):
                if index:hasher.update(b',')
                hasher.update(raw_bytes);values.append(json.loads(bytes(raw_bytes)))
        record_hash.update(b']');evidence_hash.update(b']')
        require(record_hash.hexdigest()==receipt['records_sha256'] and evidence_hash.hexdigest()==receipt['evidence_sha256'],
                'Native completion data seal differs')
        timings['drain_seconds']+=time.monotonic()-t
        game.update(result=receipt['result'],records=records)
        if continuation is None and job.get('branching'): game['branch_statistics']=dict(eligible_decisions=eligible_count)
        if continuation is not None:
            game['branch'].update(elapsed_seconds=time.monotonic()-start,
                fraction_remaining=len(records)/(len(records)+game['branch']['decision']))
        t=time.monotonic()
        from brass_ai.recovery.replay import CheckedGame
        arrays=list(lane.training_features(receipt,records,drain=True))
        lane.release()
        game=CheckedGame(game).bind(arrays)
        game.actor_validated=True  # Each original frame crossed BOUNDARY above.
        timings['training_feature_export_seconds']+=time.monotonic()-t
        timings.update({'ipc_'+k:v for k,v in lane.metrics.items()})
        timings.update({'binding_'+k:v for k,v in binding_counts.items()})
        return dict(game=game,scoring=receipt['scoring'],evidence=evidence,native_receipt=receipt,
            origin_counts=continuation['counts'] if continuation is not None else [0]*4,
            diagnostic_branch_probe=diagnostic_branch_probe,
            branch_expected=forked and continuation is None,timings=dict(timings,total_seconds=time.monotonic()-start),
            producer=dict(contract=CONTRACT,core=lane.hello['core']))
