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
from pathlib import Path
import random
import time

import numpy as np
from brass_ai.learning.contracts import digest, require
from brass_ai.learning.replay import split_for
from brass_ai.step5.policy import NeuralPolicy, context_for
from brass_ai.recovery.execution import make_policy
from brass_ai.recovery.branching import eligible, capture, seed_for, RESET_OPENINGS
from brass_ai.recovery.inputs import BOUNDARY
from .lab_transport import CONTRACT, DecisionEncoder, Collector


def frozen_id(job):
    return digest({k:v for k,v in job.items() if k != 'path'})


def route(job):
    """Never redraw a table or hide unsupported coverage behind a heuristic."""
    require(len(job['policies']) == 4, 'Frozen ordinary job needs four policies')
    if any(p['kind'] not in ('neural', 'legacy', 'lab-legacy') for p in job['policies']):
        return 'reference', 'enabled search/unsupported policy is outside B02'
    if any(p['kind'] in ('legacy', 'lab-legacy') for p in job['policies']):
        # Existing frozen V4 CPU adapter remains exact, but its full environment
        # integration is reference-routed as one job, never seat-by-seat fallback.
        return 'reference', 'frozen legacy/IL rule adapter retained as a complete reference table'
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
                 continuation=None, submit_branch=None, cancel=None, compare=None, diagnostic_branch_probe=False):
    require(route(job)[0] == 'rust', 'Unsupported whole job must use reference route')
    start = time.monotonic(); timings = Counter(); job_id = frozen_id(job)
    with ExitStack() as cleanup:
        lane = cleanup.enter_context(Collector(executable))
        policies = [make_policy(p, root, endpoint, execution, None) for p in job['policies']]
        for p in policies:
            cleanup.callback(p.close)
            p.evaluator.encoder = DecisionEncoder(lane, p.evaluator.model_id)
            predict=p.evaluator.predict
            def recorded(requests, p=p, predict=predict):
                began=time.monotonic()
                responses=predict(requests)
                timings['inference_envelope_seconds']+=time.monotonic()-began
                require(len(responses)==len(requests),'Incomplete ordinary inference batch')
                p.native_raw=np.asarray(responses[0]['probabilities'],dtype=np.float64)
                return responses
            p.evaluator.predict=recorded
        assigned = [policies[(s-job.get('rotation',0))%4] for s in range(4)]
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
                if p.focus == 'mixed': p.contexts[rngs[s]] = context_for(job.get('focuses',{}).get(str(s),'neutral'))
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
                require(not reset or (job['branching']['reset_opening_at_fork'] is True and p.opening == RESET_OPENINGS[meta['contract']]), 'Branch opening policy changed')
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
            BOUNDARY.check(view,choices); seat=view['seat']; p=assigned[seat]; rng=rngs[seat]
            t=time.monotonic()
            if len(choices)==1:
                _,intent=p.context_features(view,rng)
                if p.exploration_version==3: chosen,probability,teacher=p.sample_training(view,choices,rng,np.ones(1),intent)
                else: rng.choices([0],weights=[1.],k=1); chosen,probability,teacher=0,1.,None
            else: chosen,probability,teacher=p.select_batch([(view,choices,rng)])[0]
            style,intent=p.context_features(view,rng)
            behavior=dict(kind=p.kind,policy_id=p.policy_id,probability=probability,temperature=p.temperature)
            if p.exploration_version==3: behavior.update(p.last_behavior[rng])
            row=dict(step=lane.current['step'],seat=seat,view=view,choices=choices,chosen=chosen,teacher=teacher,
                     style=style,intent=intent,seconds=time.monotonic()-t,behavior=behavior)
            timings['policy_seconds']+=row['seconds']
            t=time.monotonic(); encoded=p.evaluator.encoder.encode(view,choices,style,intent,validated=True)
            timings['encoding_seconds']+=time.monotonic()-t
            evidence=dict(kind='ordinary',model_id=p.evaluator.model_id,policy_id=p.policy_id,forced=len(choices)==1,
                raw_probabilities=(np.ones(1) if len(choices)==1 else p.native_raw).tolist(),
                feature_version=encoded.feature_version,semantic_id=choices[chosen]['id'])
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
            t=time.monotonic(); lane.append(row,evidence); timings['native_step_seconds']+=time.monotonic()-t
        receipt=lane.call('finish'); records=[]; evidence=[]
        require(receipt['contract']==CONTRACT and receipt['job']==job_id and receipt['token']==lane.token,
                'Foreign native completion receipt')
        record_hash=hashlib.sha256(b'[');evidence_hash=hashlib.sha256(b'[')
        t=time.monotonic()
        for index in range(receipt['rows']):
            reply=lane.call('row',index=index)
            for values,hasher,key in ((records,record_hash,'row_json'),(evidence,evidence_hash,'evidence_json')):
                if index:hasher.update(b',')
                hasher.update(reply[key].encode('utf-8'));values.append(json.loads(reply[key]))
        record_hash.update(b']');evidence_hash.update(b']')
        require(record_hash.hexdigest()==receipt['records_sha256'] and evidence_hash.hexdigest()==receipt['evidence_sha256'],
                'Native completion data seal differs')
        timings['drain_seconds']+=time.monotonic()-t
        game.update(result=receipt['result'],records=records)
        if continuation is None and job.get('branching'): game['branch_statistics']=dict(eligible_decisions=eligible_count)
        if continuation is not None:
            game['branch'].update(elapsed_seconds=time.monotonic()-start,
                fraction_remaining=len(records)/(len(records)+game['branch']['decision']))
        lane.release()
        return dict(game=game,scoring=receipt['scoring'],evidence=evidence,native_receipt=receipt,
            origin_counts=continuation['counts'] if continuation is not None else [0]*4,
            diagnostic_branch_probe=diagnostic_branch_probe,
            branch_expected=forked and continuation is None,timings=dict(timings,total_seconds=time.monotonic()-start),
            producer=dict(contract=CONTRACT,core=lane.hello['core']))
