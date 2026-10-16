"""Opt-in mixed-policy outcome learning inside the existing teacher steps.

Teacher outcomes are on-policy for the recorded search/direct mixture, not for
the ordinary PPO actor. Selection is deterministic and precedes encoding.
"""
from collections import Counter,defaultdict
import gzip
import json
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest,require
from brass_ai.learning.replay import terminal_targets,validate_teacher
from brass_ai.learning.reward import replay_returns
from brass_ai.step5.league import implementation_id
from .config import teacher_value_config
from .encoding import PreparedEncoder
from .teacher_schedule import PHASES,phase_for

SOURCE='completed-teacher-mixed-search-direct/1.0.0'
PHASE_NAMES=tuple(p for p,_,_,_ in PHASES)


def provenance(identity,reward,options):
    return dict(contract=SOURCE,model_id=identity,implementation=implementation_id(),
                reward_id=digest(reward),config_id=digest({k:v for k,v in options.items() if k not in ('diagnostics','probe_positions')}))


def admit_collection(games,records,jobs,expected):
    """Bind even direct/forced/unlabelled rows to this actual collection.

    Policy-label checks cannot establish freshness for a game with no labels.
    Actor recipes and raw shard hashes supply that independent admission proof.
    """
    require(len(records)==len(games)==len(jobs),'Incomplete teacher outcome collection')
    wanted={j['index']:j for j in jobs};by_game={r['game_id']:r for r in records}
    require(len(wanted)==len(by_game)==len(games),'Duplicate teacher outcome receipt')
    headers={};bindings={}
    for game in games:
        record=by_game.get(game['game_id']);require(record is not None,'Foreign teacher outcome game')
        job=wanted.get(record['job']);require(job is not None,'Foreign teacher outcome job')
        require(job['provenance'].get('teacher_value')==expected,'Stale teacher outcome request')
        path=record['path']
        if path not in headers:
            require(sha(path)==record['sha256'],'Teacher outcome replay changed')
            with gzip.open(path,'rt',encoding='utf-8') as stream:headers[path]=json.loads(next(stream))['provenance']
        proof=headers[path];recipe=proof.get('recipe',{})
        require(proof.get('collection')==digest(recipe),'Teacher outcome collection identity differs')
        matches=[j for j in proof.get('jobs',[]) if j['index']==job['index']]
        require(len(matches)==1 and matches[0]['provenance']==job['provenance'],'Stale teacher outcome provenance')
        recipes=[j for j in recipe.get('jobs',[]) if j['index']==job['index']]
        require(len(recipes)==1 and recipes[0]['policies']==job['policies'],'Teacher outcome policy recipe differs')
        require(all(p['model_id']==expected['model_id'] and p['search']['reward'] is not None and
                    digest(p['search']['reward'])==expected['reward_id'] and p.get('teacher_schedule')
                    for p in recipes[0]['policies']),'Teacher outcome model/reward differs')
        require(game['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed'])),
                'Teacher outcome setup differs')
        bindings[game['game_id']]=dict(expected,game_digest=digest(game))
    return bindings


def balanced(rows,key,limit):
    """Round-robin strata with evenly spaced members; no training RNG draws."""
    groups=defaultdict(list)
    for item in rows:groups[key(item)].append(item)
    groups={k:v for k,v in sorted(groups.items())}
    quotas={k:0 for k in groups};remaining=min(limit,len(rows))
    while remaining:
        for k,items in groups.items():
            if quotas[k]<len(items) and remaining:quotas[k]+=1;remaining-=1
    selected=[]
    for k,items in groups.items():
        n=quotas[k]
        selected.extend(items[(2*j+1)*len(items)//(2*n)] for j in range(n))
    return selected


def position_key(entry):
    row=entry[0]
    return entry[2][0],phase_for(row['view']['context'])


class OutcomeProbe:
    """At most 128 fixed positions; batched diagnostics never touch gradients."""
    def __init__(self,aux):
        self.aux=aux
        self.indices=aux.probe_indices
        self.stages={};self.seconds=0.;self.forwards=0;self.positions=0;self.reused=0
        self.before=None;self.after=None;self.cached={};self.reused_fit=0
        key=lambda d:(d[2][0],d[0]['seat'],d[0]['view']['context']['revision'])
        self.fit_keys={key(aux.teacher.data[i]):i for i in self.indices
            if aux.teacher.rows[i]['_reward_remaining']==[aux.teacher.rows[i]['_reward_remaining'][0]]*4
            and aux.teacher.rows[i]['_reward_remaining'][0]==aux.reward_cap}

    def start_fit(self):self.cached={}

    def observe_fit(self,part,data,outputs):
        # Existing policy-fit forwards omit the budget input, hence only rows
        # with the full four-seat remaining budget have matching utility output.
        # Other rows receive a bounded value forward with their actual budget.
        began=time.perf_counter();matches=[]
        for j,i in enumerate(part):
            row=data[i][0];key=(data[i][2][0],row['seat'],row['view']['context']['revision'])
            if key in self.fit_keys:matches.append((j,self.fit_keys[key]))
        if matches:
            predictions=torch.stack(outputs[1:],1).detach()
            for j,i in matches:self.cached[i]=predictions[j]
            self.reused_fit+=len(matches)
        self.seconds+=time.perf_counter()-began

    def capture(self,model,stage):
        from .updates import autocast
        ds=self.aux.teacher;config=self.aux.update
        began=time.perf_counter();was_training=model.training;model.eval()
        missing=[i for i in self.indices if i not in self.cached]
        try:
            with torch.no_grad():
                # Preserve index order for fixed targets and prediction deltas.
                parts=ds.microbatches(missing,config['microbatch_size'],config['max_candidate_cells'],False)
                for part,ix,inputs in ds.value_batches(parts,model):
                    with autocast(config,ds.device):out=model(*inputs,remaining=ds.remaining[ix])
                    predictions=torch.stack(out[1:],1).detach()
                    for i,prediction in zip(part,predictions.unbind()):self.cached[i]=prediction
                    self.forwards+=1;self.positions+=len(part)
                values=torch.stack([self.cached[i] for i in self.indices]).float().cpu()
        finally:model.train(was_training)
        self.record(stage,values)
        self.cached={}
        self.seconds+=time.perf_counter()-began

    def record(self,stage,values):
        ds=self.aux.teacher
        targets=torch.stack((*ds.targets,ds.utility),1)[self.indices].detach().float().cpu()
        diff=values-targets
        metrics=dict(win_share_brier=float(diff[:,0].square().sum(-1).mean()),
            rank_mae=float(diff[:,1].abs().mean()),vp_mae=float(diff[:,2].abs().mean()),
            utility_mae=float(diff[:,3].abs().mean()),utility_mse=float(diff[:,3].square().mean()))
        changes=None
        if self.before is None:self.before=values.clone()
        else:
            delta=values-self.before
            changes={name:dict(mean_absolute=float(delta[:,i].abs().mean()),
                maximum_absolute=float(delta[:,i].abs().max())) for i,name in enumerate(('win_share','rank','vp','utility'))}
        self.stages[stage]=dict(available=True,errors=metrics,prediction_change_from_before=changes)
        if stage=='post_update':self.after=values.clone()

    def after_calibration(self,predictions):
        # Calibration is head-only. Official predictions at post_update remain
        # exact; utility outputs come from its existing final fit/error pass.
        if self.after is None:return
        began=time.perf_counter();values=self.after.clone()
        values[:,3]=predictions[self.indices].detach().float().cpu()
        self.record('post_calibration',values);self.reused+=len(self.indices)
        self.seconds+=time.perf_counter()-began

    def report(self):
        data=self.aux.teacher.data
        return dict(enabled=True,positions=len(self.indices),games=len({data[i][2][0] for i in self.indices}),
            phases=dict(Counter(phase_for(data[i][0]['view']['context']) for i in self.indices)),
            sampling='Deterministic game/phase balanced subset of bounded teacher value candidates, selected before encoding; fixed through all stages',
            scope='Training-data outcomes; correlated positions are not independent games or playing-strength evidence',
            stages={s:self.stages.get(s,dict(available=False,reason='No matching update/calibration stage'))
                    for s in ('before','post_update','post_calibration')},
            elapsed_seconds=self.seconds,extra_forward_passes=self.forwards,extra_forward_positions=self.positions,
            reused_fit_positions=self.reused_fit,reused_calibration_positions=self.reused,
            prediction_precision='Configured fit/probe precision; post-calibration utility comes from FP32 calibration',
            iteration_overhead_target='<=1% median full iteration; unverified on representative training')


class TeacherValue:
    def __init__(self,model,games,ordinary,options,update,bindings,*,ordinary_policy_id):
        from .updates import Dataset
        from .utility import calibration_ready
        self.options=teacher_value_config(options);self.update=update
        require(self.options['enabled'],'Teacher value is disabled')
        reward=model.config.utility
        require(reward is not None and calibration_ready(model),'Teacher value needs a calibrated current utility model')
        expected=provenance(model_id(model),reward,self.options)
        require(games and len({g['game_id'] for g in games})==len(games) and
                len({g['setup_group'] for g in games})==len(games),'Duplicate/empty teacher outcome games')
        candidates=[];self.valid=Counter();self.seen=Counter();self.ordinary_seen=Counter();self.calibration_seen=Counter()
        self.training_seconds=0.;self.preparation_seconds=0.;began=time.perf_counter()
        encoder=PreparedEncoder();prepared=[];targets_by_game={}
        for game in sorted(games,key=lambda g:g['game_id']):
            require(game['split']=='train','Held-out teacher value data')
            require(bindings.get(game['game_id'])==dict(expected,game_digest=digest(game)),
                    'Teacher value requires fresh current-model/current-source complete-game provenance')
            if hasattr(game,'assert_metadata'):game.assert_metadata()
            require({r['seat'] for r in game['records']}==set(range(4)),'Incomplete teacher seat trajectories')
            target=terminal_targets(game['result']);labels,summary=replay_returns(game['records'],game['result'],reward)
            targets_by_game[game['game_id']]=(target,labels,summary)
            local=[];positions=set()
            for i,row in enumerate(game['records']):
                require(row['step']==i and row['seat']==row['view']['seat'],'Invalid teacher value position')
                phase=phase_for(row['view']['context']);require(phase is not None,'Unknown teacher value phase')
                key=(row['seat'],row['view']['context']['revision'])
                require(key not in positions,'Duplicate teacher value position');positions.add(key)
                teacher=row['teacher']
                if teacher is not None:
                    validate_teacher(teacher,len(row['choices']))
                    require(teacher['source']=='ismcts-root-visits/1.2.0' and
                        teacher['model_id']==expected['model_id'] and teacher['source_fingerprint']==expected['implementation'] and
                        teacher['search']['reward']==reward,'Stale teacher value label source')
                # Validity is independent of search coverage, confidence and
                # forced-action filtering. Terminal outcome applies to all.
                require(row['choices'] and not row['view']['context'].get('terminal',False),'Invalid teacher value decision')
                local.append((game,i,phase));self.valid[phase]+=1
            candidates+=balanced(local,lambda t:t[2],self.options['samples_per_game'])
        candidates=balanced(candidates,lambda t:(t[0]['game_id'],t[2]),self.options['max_samples'])
        self.outcome_targets={key:values[0] for key,values in targets_by_game.items()}
        self.probe_indices=balanced(list(range(len(candidates))),
            lambda i:(candidates[i][0]['game_id'],candidates[i][2]),self.options['probe_positions'])
        # Both the bounded training set and its diagnostic subset are chosen
        # from raw metadata before asking for any encoded input.
        for game,i,_ in candidates:
            row=game['records'][i];target,labels,summary=targets_by_game[game['game_id']]
            r=dict(row,_reward_return=labels[i]['utility'],_reward_remaining=labels[i]['remaining'],_reward_id=summary['reward_id'])
            encoded=game.encoded(row,encoder) if hasattr(game,'encoded') else encoder.encode(row['view'],row['choices'],row['style'],row['intent'])
            prepared.append((r,target,(game['game_id'],row['seat']),encoded))
        require(ordinary_policy_id and ordinary and all(d[0]['teacher'] is None and
                d[0]['behavior']['kind']=='model' and d[0]['behavior']['policy_id']==ordinary_policy_id and
                d[0].get('_reward_id')==digest(reward) for d in ordinary),
                'Teacher value requires fresh ordinary learner targets with the same reward')
        require(not ({d[2][0] for d in ordinary}&set(targets_by_game)),'Teacher games cannot enter ordinary critic/actor data')
        selected=balanced(list(range(len(ordinary))),lambda i:position_key(ordinary[i]),len(prepared))
        device=next(model.parameters()).device
        self.teacher=Dataset(prepared,device,policy=False)
        self.ordinary=Dataset([ordinary[i] for i in selected],device,policy=False)
        self.loss_totals=torch.zeros(2,device=device);self.loss_steps=0
        self.reward_id=digest(reward);self.games=len(games);self.binding=expected
        self.reward_cap=reward['insolvency']['max_cost']
        self.preparation_seconds=time.perf_counter()-began
        self.probe=OutcomeProbe(self) if self.options['diagnostics'] else None

    def begin(self,model):
        if self.probe:self.probe.capture(model,'before')

    def backward(self,model,step,width):
        """Accumulate gradients before the existing policy optimizer.step()."""
        from .updates import value_losses,autocast
        began=time.perf_counter();count=len(self.teacher.data);width=min(width,count)
        ids=list(range(step*width,min((step+1)*width,count*self.options['max_reuse'])))
        ids=[i%count for i in ids]
        if not ids:return False
        ordinary=[(step*width+i)%len(self.ordinary.data) for i in range(min(len(ids),len(self.ordinary.data)))]
        mix=self.options['teacher_fraction'];weight=self.options['loss_weight']
        for group,(ds,indices,fraction) in enumerate(((self.ordinary,ordinary,1-mix),(self.teacher,ids,mix))):
            parts=ds.microbatches(indices,self.update['microbatch_size'],self.update['max_candidate_cells'],self.update['bucket'])
            for part,ix,inputs in ds.value_batches(parts,model):
                with autocast(self.update,ds.device):out=model(*inputs,remaining=ds.remaining[ix])
                value=value_losses(out,tuple(t[ix] for t in ds.targets))[0]
                value=value+(out[4].float()-ds.utility[ix]).square().mean(-1)
                (weight*fraction*value.sum()/len(indices)).backward()
                self.loss_totals[group]+=value.detach().sum()/len(indices)
        self.seen.update(ids);self.ordinary_seen.update(ordinary)
        self.loss_steps+=1
        # Synchronize only at the aggregate phase boundary, never per row.
        if self.teacher.device.type=='cuda':torch.cuda.synchronize(self.teacher.device)
        self.training_seconds+=time.perf_counter()-began
        return True

    def post_update(self,model):
        if self.probe:self.probe.capture(model,'post_update')

    def report(self):
        ordinary_loss,teacher_loss=(self.loss_totals/max(1,self.loss_steps)).cpu().tolist()
        total=self.seen+self.calibration_seen
        used=Counter(phase_for(self.teacher.rows[i]['view']['context']) for i in total)
        presentations=Counter()
        for i,n in total.items():presentations[phase_for(self.teacher.rows[i]['view']['context'])]+=n
        return dict(enabled=True,contract=self.options['contract'],target_source=SOURCE,reward_id=self.reward_id,
            provenance=self.binding,distinct_completed_games=self.games,valid_value_positions=sum(self.valid.values()),
            selected_positions=len(self.teacher.data),distinct_positions_trained=len(total),
            unused_positions=sum(self.valid.values())-len(total),sample_presentations=sum(total.values()),
            auxiliary_distinct_positions=len(self.seen),auxiliary_presentations=sum(self.seen.values()),
            calibration_distinct_positions=len(self.calibration_seen),calibration_presentations=sum(self.calibration_seen.values()),
            max_auxiliary_reuse=max(self.seen.values(),default=0),ordinary_positions_trained=len(self.ordinary_seen),
            ordinary_sample_presentations=sum(self.ordinary_seen.values()),
            phases={p:dict(valid=self.valid[p],trained=used[p],unused=self.valid[p]-used[p],presentations=presentations[p]) for p in PHASE_NAMES},
            loss_weight=self.options['loss_weight'],ordinary_fraction=1-self.options['teacher_fraction'],
            teacher_fraction=self.options['teacher_fraction'],
            ordinary_mean_loss=ordinary_loss,teacher_mean_loss=teacher_loss,
            weighted_auxiliary_loss=self.options['loss_weight']*((1-self.options['teacher_fraction'])*ordinary_loss+
                                                                self.options['teacher_fraction']*teacher_loss),
            weighting_scope='PPO critic remains ordinary-only; each existing distillation step adds loss_weight * ((1-teacher_fraction)*ordinary_mean + teacher_fraction*teacher_mean). Calibration uses the same fractions.',
            preparation_seconds=self.preparation_seconds,auxiliary_training_seconds=self.training_seconds,
            diagnostics=self.probe.report() if self.probe else dict(enabled=False,reason='Opt-in telemetry disabled; full-iteration overhead target unverified'),
            policy_kl='See existing teacher_policy_drift; shared policy+auxiliary update, before head-only calibration',
            skip_reason=None if self.seen else 'No eligible policy-distillation optimizer steps; value validity does not bypass policy gates')
