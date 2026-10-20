"""Bounded PPG critic fitting with guarded steps and held-out prefix acceptance."""
import copy
import hashlib
import math
import time
from collections import defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import digest,require

LEGACY_CONTRACT='ppg-value-phase/2.0.0'
LEGACY_DEFAULT=dict(contract=LEGACY_CONTRACT,enabled=True,buffer_iterations=2,max_epochs=3,batch_size=4096,
    clone_rows=8192,clone_batch_size=512,beta_clone=1.,beta_clone_max=8.,kl_cap=.002,
    learning_rate=6e-6,holdout_fraction=.05)
STRATIFIED_CONTRACT='ppg-value-phase/3.0.0'
CONTRACT='ppg-value-phase/4.0.0'
from .critic_holdout import DEFAULT as HOLDOUT_DEFAULT,LEGACY_DEFAULT as OLD_HOLDOUT
STRATIFIED_DEFAULT=dict(LEGACY_DEFAULT,contract=STRATIFIED_CONTRACT,learning_rate=3e-6,holdout=copy.deepcopy(OLD_HOLDOUT))
DEFAULT=dict(STRATIFIED_DEFAULT,contract=CONTRACT,acceptance_steps=16,enabled=False,holdout=copy.deepcopy(HOLDOUT_DEFAULT))
CONFIG_MIGRATION='step5-ppg-refinement/1.0.0'
INTERVAL_MIGRATION='step5-ppg-acceptance-intervals/1.0.0'
DISABLE_MIGRATION='step5-ppo-only-critic/1.0.0'


def disabled_options(options):
    """Exact reviewed switch; retain the holdout and every unrelated setting."""
    result=copy.deepcopy(options)
    validate(result['value_phase'])
    result['value_phase']['enabled']=False
    return result


def critic_work(report):
    """Actual critic optimizer work; never label PPO steps as accepted PPG."""
    if report.get('enabled',True):
        return dict(source='ppg',optimizer_steps=report.get('optimizer_steps',0))
    work=report.get('critic_training',dict(source='ppo',optimizer_steps=0))
    require(work.get('source')=='ppo' and type(work.get('optimizer_steps')) is int
        and work['optimizer_steps']>=0,'Invalid PPO critic evidence')
    return work


def stratified(config):return config.get('contract') in (STRATIFIED_CONTRACT,CONTRACT)


def interval_options(options):
    result=copy.deepcopy(options);value=result.get('value_phase')
    if value is not None and value.get('contract')==STRATIFIED_CONTRACT:
        validate(value);value.update(contract=CONTRACT,acceptance_steps=16)
    return result


def refined_options(options):
    """Exact opt-in upgrade; preserve every unrelated value-phase option."""
    result=copy.deepcopy(options);value=result.get('value_phase')
    if value is not None and value.get('contract')==LEGACY_CONTRACT:
        validate(value)
        value.update(contract=STRATIFIED_CONTRACT,learning_rate=value['learning_rate']*.5,holdout=copy.deepcopy(OLD_HOLDOUT))
    return result


def validate(config):
    current=type(config) is dict and stratified(config)
    intervals=type(config) is dict and config.get('contract')==CONTRACT
    require(type(config) is dict and set(config)==set(DEFAULT if intervals else STRATIFIED_DEFAULT if current else LEGACY_DEFAULT) and
            (config['contract'] in (CONTRACT,STRATIFIED_CONTRACT,LEGACY_CONTRACT) or
             (config['contract']=='ppg-value-phase/1.0.0' and config['buffer_iterations']==1)),
            'Invalid value phase contract/fields')
    if intervals:
        require(type(config['acceptance_steps']) is int and 1<=config['acceptance_steps']<=16,
            'PPG acceptance interval must be 1 to 16 optimizer steps')
    if current:
        from .critic_holdout import validate as validate_holdout
        validate_holdout(config['holdout'])
    require(type(config['enabled']) is bool,'Invalid value phase enabled flag')
    for key in ('buffer_iterations','max_epochs','batch_size','clone_rows','clone_batch_size'):
        require(type(config[key]) is int and config[key]>0,'Invalid value phase '+key)
    for key in ('beta_clone','beta_clone_max','kl_cap','learning_rate','holdout_fraction'):
        require(type(config[key]) in (int,float) and math.isfinite(config[key]) and config[key]>0,
                'Invalid value phase '+key)
    require(config['buffer_iterations'] in (1,2) and config['max_epochs']<=3 and
            config['clone_batch_size']<=config['clone_rows']<=8192 and
            config['beta_clone']<=config['beta_clone_max']<=8 and config['kl_cap']<=.002 and
            config['holdout_fraction']<.5,'Invalid value phase bounds; buffer_iterations must be 1 or 2')
    return config


def heldout(game_id,fraction):
    require(type(game_id) is str and bool(game_id),'Critic holdout requires a game id')
    from .branching import family
    game_id=family(game_id)
    value=int.from_bytes(hashlib.sha256(('ppg-critic-holdout/1.0.0:'+game_id).encode()).digest()[:8],'big')
    return value < int(fraction*2**64)


def split(data,fraction,holdout_options=None):
    if holdout_options is not None:
        from .critic_holdout import partition
        return partition(data,fraction,holdout_options)
    held={row[2][0] for row in data if heldout(row[2][0],fraction)}
    train=[i for i,d in enumerate(data) if d[2][0] not in held]
    validation=[i for i,d in enumerate(data) if d[2][0] in held]
    return train,validation,dict(contract='ppg-critic-holdout/1.0.0',fraction=fraction,
        train_rows=len(train),heldout_rows=len(validation),train_games=len({data[i][2][0] for i in train}),
        heldout_games=len(held),heldout_games_digest=digest(sorted(held)),
        scope='Hash of complete game id; every seat and value head excluded from critic fitting; actor still uses these games')


def value_parts(dataset,ids,config,model):
    from .updates import state_value_model
    if state_value_model(model):
        ids=list(ids);size=min(config['microbatch_size'],config['max_candidate_cells'])
        return (ids[i:i+size] for i in range(0,len(ids),size))
    return dataset.microbatches(ids,config['microbatch_size'],config['max_candidate_cells'],config['bucket'])


def metrics(model,dataset,ids,config,*,unknown=False,evidence=False):
    """All four heads' loss/calibration; EV is the observing learner seat's utility."""
    from .updates import value_losses,value_output
    from .teacher_schedule import phase_for
    if not ids:return dict(rows=0,games=0,loss=None,explained_variance=None,utility_mse=None)
    arrays=[];order=[];model.eval()
    with torch.no_grad():
        for part,ix,inputs in dataset.value_batches(value_parts(dataset,ids,config,model),model):
            out=value_output(model,inputs,dataset,ix,unknown=unknown)
            value=value_losses(out,tuple(t[ix] for t in dataset.targets))[0]
            error=out[4].float()-dataset.utility[ix]
            seats=dataset.seats[ix];positions=torch.arange(len(ix),device=ix.device)
            arrays.append(torch.cat((torch.stack((value+error.square().mean(-1),
                error.square().mean(-1),out[4][positions,seats].float(),dataset.utility[ix,seats]),-1),
                out[1].float(),dataset.targets[0][ix]),-1).cpu().numpy());order.extend(part)
    all_values=np.concatenate(arrays).astype(np.float64)
    require(np.isfinite(all_values).all(),'Nonfinite critic validation metrics')
    def summarize(a):
        variance=np.var(a[:,3]);error=a[:,2]-a[:,3]
        return dict(rows=len(a),loss=float(a[:,0].mean()),utility_mse=float(a[:,1].mean()),
            explained_variance=float(1-np.var(error)/variance) if variance>1e-12 else None,
            learner_utility_mse=float(np.square(error).mean()))
    groups=defaultdict(list);game_ids=[]
    for position,i in enumerate(order):
        game_ids.append(dataset.data[i][2][0])
        branch='/branch/' in dataset.data[i][2][0]
        groups['source:'+('branch' if branch else 'ordinary')].append(position)
        context=dataset.rows[i]['view']['context'];era=context['era']
        groups['era:'+era].append(position);groups['phase:'+era+'/'+context['phase']].append(position)
        table=dataset.tables[i];role=dataset.roles[i]
        if not branch:
            groups['table:'+table].append(position);groups['role:'+role].append(position)
            groups['table:'+table+'/role:'+role].append(position)
        name=phase_for(context) if type(context.get('round')) is int else None
        groups['period:'+(name or 'unassigned')].append(position)
    probabilities=all_values[:,4:8].reshape(-1);targets=all_values[:,8:12].reshape(-1)
    buckets=np.minimum((probabilities*10).astype(int),9)
    calibration=[]
    for bucket in range(10):
        selected=buckets==bucket;n=int(selected.sum())
        calibration.append(dict(lower=bucket/10,upper=(bucket+1)/10,count=n,
            predicted=float(probabilities[selected].mean()) if n else None,
            observed=float(targets[selected].mean()) if n else None))
    summaries={key:dict(summarize(all_values[group]),
        games=len({game_ids[p] for p in group})) for key,group in sorted(groups.items())}
    if evidence:
        from .ev_evidence import statistics,TABLES
        for kind in TABLES:
            key='table:'+kind+'/role:learner';group=groups.get(key)
            if group:
                a=all_values[group]
                summaries[key]['ev_evidence']=statistics(a[:,3],a[:,2],[game_ids[p] for p in group])
    summary=summarize(all_values)
    if 'source:ordinary' in summaries:
        summary['explained_variance']=summaries['source:ordinary']['explained_variance']
    return dict(summary,games=len(set(game_ids)),groups=summaries,
        win_calibration=calibration,
        scope='MC loss over all heldout heads; gate EV and table/role evidence use ordinary games. Branch/ordinary EV reported separately; correlated rows/heads')


class RawPolicyReference:
    """Frozen post-PPO legal distributions, on CPU; no exploration mixture."""
    def __init__(self,model,dataset,ids,config):
        self.dataset,self.config=dataset,config;self.ids=list(ids);self.rows={}
        model.eval()
        with torch.no_grad():
            for part in dataset.microbatches(ids,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                _,inputs,_=dataset.batch(part)
                p=torch.softmax(model(*inputs)[0].float(),-1).cpu()
                for j,i in enumerate(part):self.rows[i]=p[j,:len(dataset.rows[i]['choices'])].clone()

    def loss(self,part,inputs,model):
        logits=model(*inputs)[0].float()
        old=torch.zeros_like(logits)
        for j,i in enumerate(part):old[j,:len(self.rows[i])]=self.rows[i].to(logits.device)
        logp=torch.log_softmax(logits,-1).masked_fill(~inputs[2],0.)
        return (old*(old.clamp_min(1e-30).log()-logp)).sum(-1)

    def measure(self,model,ids=None):
        ids=self.ids if ids is None else list(ids);total=torch.zeros((),device=self.dataset.device)
        model.eval()
        with torch.no_grad():
            for part in self.dataset.microbatches(ids,self.config['microbatch_size'],self.config['max_candidate_cells'],self.config['bucket']):
                _,inputs,_=self.dataset.batch(part);total+=self.loss(part,inputs,model).sum()
        result=float(total)/max(1,len(ids))
        require(math.isfinite(result),'Nonfinite PPG raw policy KL')
        return max(0.,result)


def clone_indices(dataset,actor_ids,limit):
    # Bounded, reproducible subset independent of minibatch/PPO RNG. Include a
    # row's ordinal within its seat chain, not its shard's completion order.
    counts=defaultdict(int);ranked=[]
    for i in actor_ids:
        identity=dataset.data[i][2];ordinal=counts[identity];counts[identity]+=1
        ranked.append((hashlib.sha256(repr((identity,ordinal)).encode()).digest(),i))
    return [i for _,i in sorted(ranked)[:limit]]


def fit(model,optimizer,dataset,train_ids,heldout_ids,actor_ids,probe_ids,config,update,*,beta,
        value_dataset=None,value_train_ids=None,validation_dataset=None,current_count=None,dropout=None):
    from .updates import autocast,value_losses,value_output
    began=time.perf_counter();model.eval()
    value_dataset=dataset if value_dataset is None else value_dataset
    value_train_ids=list(train_ids) if value_train_ids is None else list(value_train_ids)
    current_count=len(dataset.data) if current_count is None else current_count
    validation_dataset=dataset if validation_dataset is None else validation_dataset
    report=dict(contract=config['contract'],enabled=config['enabled'],optimizer_steps=0,attempted_steps=0,
        epochs=0,completed_epochs=0,partial_epochs=0,attempted_epochs=0,early_stop_reason=None,kl_guard_tripped=False,
        acceptance='kl-safe-interval/1.0.0' if 'acceptance_steps' in config else 'kl-safe-prefix/1.0.0',
        acceptance_steps=config.get('acceptance_steps'),accepted_intervals=0,rejected_intervals=0,interval_reports=[],
        kl_cap=config['kl_cap'],
        beta_clone=beta,learning_rate=config['learning_rate'] if config['enabled'] else 0.,
        configured_learning_rate=config['learning_rate'],train_rows=len(value_train_ids),heldout_rows=len(heldout_ids),
        current_train_rows=len(train_ids),previous_train_rows=len(value_train_ids)-len(train_ids),
        distinct_value_rows_used=0,value_presentations=0,clone_presentations=0,
        branch_distinct_rows_used=0,branch_presentations=0,
        attempted_value_presentations=0,attempted_clone_presentations=0,
        distinct_current_rows_used=0,distinct_previous_rows_used=0,current_presentations=0,previous_presentations=0,
        scope='Critic: current plus configured previous iteration; validation/cloning/probes: current only; separate Adam/configured value LR, no PPO counters; KL-safe prefixes require non-increasing held-out loss')
    if not config['enabled']:report['early_stop_reason']='disabled'
    elif not value_train_ids or not heldout_ids:report['early_stop_reason']='insufficient_critic_split'
    elif not actor_ids or not probe_ids:report['early_stop_reason']='no_policy_reference'
    if report['early_stop_reason']:
        # PPO has moved the shared trunk. Neither a skipped value fit nor
        # read-only holdout diagnostics can certify its old calibration receipt.
        model.utility_calibration=None
        report.update(heldout_after=metrics(model,validation_dataset,heldout_ids,update,evidence=True),
            descriptor_gain=dict(learner_ev_gain=None),role_usage={role:dict(distinct_used=0,presentations=0) for role in ('learner','snapshot')})
        if not config['enabled']:
            # Keep the same whole-game holdout for PPO and adoption gates.
            # Inference only: no clones, cached rows, optimizer or RNG draws.
            unknown=metrics(model,validation_dataset,heldout_ids,update,unknown=True)
            from .critic_data import learner_ev
            true_ev=learner_ev(report['heldout_after']);unknown_ev=learner_ev(unknown)
            report.update(heldout_unknown=unknown,
                descriptor_gain=dict(learner_ev_gain=true_ev-unknown_ev
                    if true_ev is not None and unknown_ev is not None else None),
                calibration_receipt=None,
                scope='PPG disabled; current post-PPO whole-game critic holdout diagnostics only; no value optimizer steps or previous rows')
        report['seconds']=time.perf_counter()-began;return report
    clones=clone_indices(dataset,actor_ids,config['clone_rows'])
    reference=RawPolicyReference(model,dataset,sorted(set(clones)|set(probe_ids)),update)
    train_before=metrics(model,value_dataset,value_train_ids,update);before=metrics(model,validation_dataset,heldout_ids,update,evidence=True)
    current=before;epochs=[];used=set();presentations=0;clone_presentations=0
    max_kl=0.;accepted_kl=0.;last_attempt=None;previous_presentations=0;role_presentations=defaultdict(int);branch_presentations=0
    for epoch in range(config['max_epochs']):
        order=[value_train_ids[j] for j in torch.randperm(len(value_train_ids)).tolist()]
        batches=[order[start:start+config['batch_size']] for start in range(0,len(order),config['batch_size'])]
        interval=config.get('acceptance_steps',len(batches))
        epoch_before=current['loss'];epoch_steps=0;epoch_safe=0;epoch_accepted=0;rejection_reason=None
        report['attempted_epochs']+=1
        for segment in range(0,len(batches),interval):
            # A rejected segment restores BOTH weights and Adam to the last
            # accepted boundary. PPO state is untouched; draw RNG is consumed.
            saved_model=copy.deepcopy(model.state_dict());saved_adam=copy.deepcopy(optimizer.state_dict())
            attempted=0;safe_steps=0;segment_used=[];segment_clones=0;safe_kl=accepted_kl
            for batch in batches[segment:segment+interval]:
                step_model=copy.deepcopy(model.state_dict());step_adam=copy.deepcopy(optimizer.state_dict())
                optimizer.zero_grad(set_to_none=True);model.train()
                for part,ix,inputs in value_dataset.value_batches(value_parts(value_dataset,batch,update,model),model):
                    drop=None if dropout is None else (dropout[0],epoch,dropout[1],'value')
                    with autocast(update,value_dataset.device):out=value_output(model,inputs,value_dataset,ix,dropout=drop)
                    loss=value_losses(out,tuple(t[ix] for t in value_dataset.targets))[0]
                    loss=loss+(out[4].float()-value_dataset.utility[ix]).square().mean(-1)
                    ((loss*value_dataset.critic_weights[ix]).sum()/len(batch)).backward()
                sample=[clones[j] for j in torch.randint(len(clones),(config['clone_batch_size'],)).tolist()]
                for part in dataset.microbatches(sample,update['microbatch_size'],update['max_candidate_cells'],update['bucket']):
                    _,inputs,_=dataset.batch(part)
                    with autocast(update,dataset.device):loss=reference.loss(part,inputs,model)
                    (beta*loss.sum()/len(sample)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),update['grad_clip'],error_if_nonfinite=True)
                optimizer.step();attempted+=1
                report['attempted_value_presentations']+=len(batch);report['attempted_clone_presentations']+=len(sample)
                measured=reference.measure(model,probe_ids);max_kl=max(max_kl,measured)
                if measured>config['kl_cap']:
                    model.load_state_dict(step_model);optimizer.load_state_dict(step_adam)
                    report.update(early_stop_reason='policy_kl_cap',kl_guard_tripped=True)
                else:
                    safe_steps+=1;safe_kl=measured;segment_used.extend(batch);segment_clones+=len(sample)
                del step_model,step_adam
                if report['kl_guard_tripped']:break
            report['attempted_steps']+=attempted;epoch_steps+=attempted;epoch_safe+=safe_steps
            # Evaluate only the safe prefix. The KL reference remains fixed at
            # post-PPO weights for the entire phase, across all accepted segments.
            last_attempt=metrics(model,validation_dataset,heldout_ids,update,evidence=True)
            loss_increased=last_attempt['loss']>current['loss']
            rejected=not safe_steps or loss_increased
            rejection_reason='heldout_loss_increased' if loss_increased else 'no_safe_steps' if rejected else None
            if report['early_stop_reason'] is None and loss_increased:
                report['early_stop_reason']='heldout_loss_increased'
            report['interval_reports'].append(dict(epoch=epoch+1,start_step=segment+1,
                steps=attempted,safe_steps=safe_steps,accepted=not rejected,
                accepted_steps=0 if rejected else safe_steps,rejection_reason=rejection_reason,
                heldout_loss_before=current['loss'],heldout_loss_attempted=last_attempt['loss'],
                probe_kl=safe_kl,attempted_probe_kl=measured))
            if rejected:
                report['rejected_intervals']+=1
                model.load_state_dict(saved_model);optimizer.load_state_dict(saved_adam)
            else:
                report['accepted_intervals']+=1;epoch_accepted+=safe_steps
                current=last_attempt;accepted_kl=safe_kl;report['optimizer_steps']+=safe_steps
                used.update(segment_used);presentations+=len(segment_used);clone_presentations+=segment_clones
                previous_presentations+=sum(i>=current_count for i in segment_used)
                branch_presentations+=sum('/branch/' in value_dataset.data[i][2][0] for i in segment_used)
                for i in segment_used:role_presentations[value_dataset.roles[i]]+=1
            del saved_model,saved_adam
            if rejected or report['kl_guard_tripped']:break
        partial=epoch_accepted<len(batches)
        epochs.append(dict(epoch=epoch+1,steps=epoch_steps,accepted=bool(epoch_accepted),
            accepted_steps=epoch_accepted,safe_steps=epoch_safe,partial=partial,
            rejection_reason=rejection_reason,heldout_loss_before=epoch_before,
            heldout_loss_attempted=last_attempt['loss'],probe_kl=accepted_kl,attempted_probe_kl=measured))
        if epoch_accepted:
            report['epochs']+=1;report['partial_epochs' if partial else 'completed_epochs']+=1
        if report['early_stop_reason'] is not None:break
    optimizer.zero_grad(set_to_none=True);model.eval()
    # One fresh receipt for the final accepted weights. No head refit on the
    # holdout, and no stale pre-value-phase receipt after trunk movement.
    # A fully rejected phase restored the exact post-PPO weights. Reuse its
    # already measured train metrics instead of scanning the whole buffer again.
    train_after=metrics(model,value_dataset,value_train_ids,update) if report['optimizer_steps'] else copy.deepcopy(train_before)
    if report['optimizer_steps']:
        ready=train_after['utility_mse']<=train_before['utility_mse']+1e-8
        model.utility_calibration=dict(version=config['contract'],reward_id=digest(model.config.utility),ready=ready,
            decisions=len(value_train_ids),games=train_after['games'],steps=report['optimizer_steps'],
            before=dict(mse=train_before['utility_mse']),after=dict(mse=train_after['utility_mse']),
            scope='Final accepted value-phase weights; ordinary critic-train games only; heldout games excluded')
    else:
        # PPO already moved the trunk. A skipped/rejected value fit cannot
        # reissue its old pre-PPO receipt as evidence about these weights.
        model.utility_calibration=None
    report.update(train_before=train_before,train_after=train_after,heldout_before=before,heldout_after=current,
        last_attempted_heldout=last_attempt,heldout_attempt_scope='Safe interval/prefix after any violating step is restored',
        clone_rows=len(clones),clone_kl=reference.measure(model,clones),
        probe_rows=len(probe_ids),probe_kl=reference.measure(model,probe_ids),maximum_attempted_probe_kl=max_kl,
        distinct_value_rows_used=len(used),value_presentations=presentations,clone_presentations=clone_presentations,
        branch_distinct_rows_used=sum('/branch/' in value_dataset.data[i][2][0] for i in used),
        branch_presentations=branch_presentations,
        distinct_current_rows_used=sum(i<current_count for i in used),
        distinct_previous_rows_used=sum(i>=current_count for i in used),
        current_presentations=presentations-previous_presentations,previous_presentations=previous_presentations,
        epoch_reports=epochs,calibration_receipt=copy.deepcopy(model.utility_calibration),
        train_metrics_reused=not bool(report['optimizer_steps']),
        seconds=time.perf_counter()-began)
    unknown_metrics=metrics(model,validation_dataset,heldout_ids,update,unknown=True)
    from .critic_data import learner_ev
    true_ev=learner_ev(current);unknown_ev=learner_ev(unknown_metrics)
    report.update(heldout_unknown=unknown_metrics,
        descriptor_gain=dict(learner_ev_gain=true_ev-unknown_ev if true_ev is not None and unknown_ev is not None else None),
        role_usage={role:dict(distinct_used=sum(value_dataset.roles[i]==role for i in used),presentations=role_presentations[role]) for role in ('learner','snapshot')})
    report['seconds']=time.perf_counter()-began
    require(report['probe_kl']<=config['kl_cap']+1e-7,'Accepted value phase exceeds policy KL cap')
    return report
