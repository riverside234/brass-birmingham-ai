"""Uniform cached epochs, paired game acceptance, and frozen raw-policy guards."""
import copy
import math
import time
from collections import defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import digest,require
from . import ppg_config,ppg_sampling,value_phase
from .updates import autocast,value_losses,value_output


def memory(device):
    """RSS reporting also works on lab Linux without optional psutil."""
    from .telemetry import memory as base_memory
    result=base_memory(device)
    try:
        import resource,sys
        result['process_lifetime_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if sys.platform.startswith('linux'):
            from pathlib import Path
            fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
            result['host_resident_bytes']=int(fields['VmRSS'].split()[0])*1024
    except ImportError:
        try:
            import psutil
            result['process_lifetime_peak_rss_bytes']=psutil.Process().memory_info().peak_wset
        except (ImportError,AttributeError):
            import os
            if os.name=='nt':
                import ctypes
                from ctypes import wintypes
                class Counters(ctypes.Structure):
                    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                        (n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                        'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
                kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
                api=ctypes.WinDLL('psapi');api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
                info=Counters();info.cb=ctypes.sizeof(info)
                if api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
                    result.update(host_resident_bytes=info.WorkingSetSize,process_lifetime_peak_rss_bytes=info.PeakWorkingSetSize)
    except (OSError,KeyError):pass  # optional current-RSS telemetry; keep OS peak if available
    return result


def paired(before,after,*,standard_errors=1.):
    require(before and before.keys()==after.keys(),'Paired PPG validation games differ')
    require(standard_errors in (1.,1.5),'Invalid paired PPG standard-error allowance')
    changes=np.array([after[k]-before[k] for k in sorted(before)],dtype=np.float64)
    require(np.isfinite(changes).all(),'Nonfinite paired PPG validation')
    mean=float(changes.mean());se=float(changes.std(ddof=1)/math.sqrt(len(changes))) if len(changes)>1 else 0.
    return dict(games=len(changes),mean_change=mean,standard_error=se,
        standard_errors=standard_errors,maximum_increase=standard_errors*se,reject=mean>standard_errors*se,
        improved=mean<0.,scope='Equally weighted ordinary held-out learner games; paired loss difference from best retained weights')


def losses(model,dataset,ids,update):
    result={};model.eval()
    with torch.no_grad():
        for part,ix,inputs in dataset.value_batches(value_phase.value_parts(dataset,ids,update,model),model):
            out=value_output(model,inputs,dataset,ix)
            loss=value_losses(out,tuple(t[ix] for t in dataset.targets))[0]+(out[4].float()-dataset.utility[ix]).square().mean(-1)
            values=loss.double().cpu().tolist()
            require(all(math.isfinite(x) for x in values),'Nonfinite PPG held-out loss')
            result.update(zip(part,values))
    return result


def by_game(dataset,values):
    groups=defaultdict(list)
    for i,v in values.items():groups[dataset.data[i][2][0]].append(v)
    return {k:float(np.mean(v)) for k,v in groups.items()}


def age_metrics(dataset,values,ages):
    groups=defaultdict(dict)
    for i,v in values.items():
        row=dataset.rows[i];age=str(ages[i]);table=row.get('_table_kind','selfplay');era=row['view']['context']['era']
        for key in (age,age+'/'+table,age+'/'+table+'/'+era):groups[key][i]=v
    return {k:dict(rows=len(v),games=len(by_game(dataset,v)),loss=float(np.mean(list(v.values()))),
        per_game_loss=float(np.mean(list(by_game(dataset,v).values())))) for k,v in sorted(groups.items())}


def layout(dataset,receipts,config,snapshot_weight):
    train={};held=[];validation=[];ages=[];offset=0;support={}
    for age,r in enumerate(receipts):
        held_games=set(r['holdout_selection']['heldout_games']);ids=[];games=set()
        for i in range(offset,offset+r['rows']):
            row=dataset.rows[i];game=dataset.data[i][2][0]
            if game in held_games:
                # Independent ordinary learner evidence, never branch families
                # or snapshot/bot outcomes, controls retention and diagnostics.
                if '/branch/' not in game and dataset.roles[i]=='learner':
                    held.append(i);games.add(game)
                    if age<config['validation_iterations']:validation.append(i)
            else:ids.append(i)
        train[age]=ids;ages.extend([age]*r['rows']);offset+=r['rows']
        support[str(age)]=dict(generation=r['generation'],train_rows=len(ids),ordinary_heldout_games=len(games),weight=ppg_config.weight(age,config))
    dataset.critic_weights=torch.tensor([snapshot_weight if r=='snapshot' else 1. for r in dataset.roles],device=dataset.device)
    return train,held,validation,ages,support


def epoch_order(train,config):
    require(config['weighting']=='uniform','PPG requires uniform replay weights')
    order=[];counts={}
    for age,ids in sorted(train.items()):
        counts[str(age)]=len(ids);order.extend(ids)
    order=[order[j] for j in torch.randperm(len(order)).tolist()]
    return order,len(order),counts


def intervals(train,config,epochs):
    """Shuffle one epoch at a time; flush validation at its final short batch."""
    for epoch in range(epochs):
        order,_,_=epoch_order(train,config)
        batches=[order[i:i+config['batch_size']] for i in range(0,len(order),config['batch_size'])]
        for start in range(0,len(batches),config['acceptance_steps']):
            block=batches[start:start+config['acceptance_steps']]
            yield epoch,start,block,start+len(block)==len(batches)


class Reference(value_phase.RawPolicyReference):
    def __init__(self,model,dataset,receipts,config,update):
        self.clone_ids=[];self.clone_weights=[];self.probe_ids=[];self.probe_weights=[];self.ages={};self.groups={};self.support={}
        offset=0;age_mass=[ppg_config.weight(age,config)*sum(r['policy_population'].values()) for age,r in enumerate(receipts)]
        total=sum(age_mass)
        for age,r in enumerate(receipts):
            self.support[str(age)]={}
            for split,limit in (('clone',config['clone_rows_per_age']),('probe',config['current_probe_rows'] if age==0 else config['probe_rows_per_age'])):
                ids,weights,support=ppg_sampling.phase_indices(r['policy_entries'],r['policy_population'],limit,split)
                getattr(self,split+'_ids').extend(offset+i for i in ids)
                getattr(self,split+'_weights').extend(w*age_mass[age]/total for w in weights)
                self.support[str(age)][split]=support
            for i,e in enumerate(r['policy_entries']):self.ages[offset+i]=age;self.groups[offset+i]=e['stratum']
            offset+=len(r['policy_entries'])
        super().__init__(model,dataset,sorted(set(self.clone_ids)|set(self.probe_ids)),update)
        require(set(self.clone_ids).isdisjoint(self.probe_ids),'PPG cloning/probe overlap')

    def probes(self,model):
        groups=defaultdict(lambda:[0.,0.,0]);model.eval();weights=dict(zip(self.probe_ids,self.probe_weights))
        with torch.no_grad():
            for part in self.microbatches(self.probe_ids):
                inputs=self.inputs(part)
                values=self.loss(part,inputs,model).double().cpu().tolist()
                for i,v in zip(part,values):
                    require(math.isfinite(v),'Nonfinite PPG policy KL')
                    table,decision=self.groups[i].split('/',1)
                    for key in ('overall','age:'+str(self.ages[i]),'group:'+self.groups[i],
                                'decision_group:'+decision,'table:'+table,
                                'age:'+str(self.ages[i])+'/decision_group:'+decision,
                                'age:'+str(self.ages[i])+'/group:'+self.groups[i]):
                        g=groups[key];g[0]+=weights[i]*v;g[1]+=weights[i];g[2]+=1
        report={k:dict(kl=max(0.,a/b) if b else None,rows=n,weight_mass=b) for k,(a,b,n) in sorted(groups.items())}
        require('overall' in report and 'age:0' in report,'PPG KL safeguards need age-0 and overall probe support')
        return dict(overall=report['overall']['kl'],age0=report['age:0']['kl'],groups=report)


def diagnostics(model,current,heldout,config,update):
    after=value_phase.metrics(model,current,heldout,update,evidence=True)
    unknown=value_phase.metrics(model,current,heldout,update,unknown=True)
    from .critic_data import learner_ev
    ev=learner_ev(after);other=learner_ev(unknown)
    return dict(heldout_after=after,heldout_unknown=unknown,
        descriptor_gain=dict(learner_ev_gain=ev-other if ev is not None and other is not None else None))


def fit(model,optimizer,buffer,current,train_ids,heldout_ids,config,update,*,beta,snapshot_weight,dropout,run_phase,ppo_work,epoch_cap=None,companion=None,ppo_peak=None):
    began=time.perf_counter()
    from . import ppg_phasic,dcpg_config
    delayed=dcpg_config.active(config)
    peak_rate=config['learning_rate']
    if ppg_config.linked(config) and ppo_peak is not None:
        ppg_phasic.validate_rate(ppo_peak,buffer.generation+1)
        require(ppo_peak['generation']==buffer.generation+1 and ppo_peak['source']=='ppo-last-applied',
            'PPG peak must come from this iteration PPO, before its next-LR controller decision')
        peak_rate=ppo_peak['peak_lr']
        if delayed:dcpg_config.validate_rate(ppo_peak,buffer.generation+1)
    epoch_cap=config['max_epochs'] if epoch_cap is None else epoch_cap
    require(type(epoch_cap) is int and (epoch_cap==5 if ppg_config.fixed(config) else
        config['epoch_controller']['minimum']<=epoch_cap<=config['epoch_controller']['maximum']),
        'Invalid effective PPG epoch budget')
    require(math.isfinite(beta) and config['beta_clone']<=beta<=config['beta_clone_max'],
        'Invalid effective PPG clone beta')
    report=dict(contract=config['contract'],enabled=True,scheduled=run_phase,optimizer_steps=0,attempted_steps=0,
        kl_guard_tripped=False,kl_cap=config['kl_cap'],acceptance_steps=config['acceptance_steps'],
        kl_check_steps=config.get('kl_check_steps',1),policy_guard_checks=0,
        configured_learning_rate=config['learning_rate'],learning_rate=peak_rate if run_phase else 0.,
        learning_rate_source=copy.deepcopy(ppo_peak) if ppg_config.linked(config) else dict(source='fixed-config',peak_lr=peak_rate),
        peak_learning_rate=peak_rate,phase_interval=config['phase_interval'],buffer_iterations=config['buffer_iterations'],
        beta_clone=beta,beta_mode='adaptive-up-down' if dcpg_config.beta_feedback(config) else 'adaptive-up',
        early_stop_reason=None if run_phase else 'not_due',gate_critic_training=copy.deepcopy(ppo_work),
        interval_reports=[],accepted_intervals=0,rejected_intervals=0,
        distinct_current_rows_used=0,distinct_previous_rows_used=0,
        role_usage={r:dict(distinct_used=0,presentations=0) for r in ('learner','snapshot')},
        value_presentations=0,clone_presentations=0,distinct_value_rows_used=0,branch_distinct_rows_used=0,
        attempted_value_presentations=0,attempted_clone_presentations=0,distinct_clone_rows_used=0,
        epoch_cap_used=epoch_cap,epochs_started=0,epochs_completed=0,best_budget_step=0,
        budget_steps_attempted=0,beta_clone_final=beta,beta_retries=0,
        adam_reset=dict(reset=False),applied_lr_min=None,applied_lr_max=None,last_attempted_lr=None,
        scope=f'Auxiliary phase every {config["phase_interval"]} committed iterations with eight uniform ages. GAE retains PPO work evidence plus auxiliary KL safeguards; current-only holdout gates never recount older games.')
    report.update(policy_guard_seconds=0.,heldout_evaluation_seconds=0.,core_optimizer_seconds=0.,core_gradient_norm_max=0.,
        policy_reference_seconds=0.,critic_training_seconds=0.,cloning_training_seconds=0.)
    report.update(heldout_checks=0,scheduled_policy_checks=0,retention_policy_checks=0,
        rejected_unverified_selections=0,retention_checks=[])
    if delayed:report['scope']='Eight committed iterations between phases; eight uniform replay ages. Fresh-target GAE work comes only from accepted PPG; current-only holdouts exclude branches/snapshots.'
    report['timing_scope']='Host wall time per operation, including input packing; CUDA execution can overlap until a guard/report synchronizes. Not isolated GPU kernel times.'
    report['heldout_standard_errors']=config.get('selection',{}).get('standard_errors',1.)
    def compare(before,after):
        if report['heldout_standard_errors']==1.:return paired(before,after)
        return paired(before,after,standard_errors=report['heldout_standard_errors'])
    def measured_losses(ds,ids):
        start=time.perf_counter();result=losses(model,ds,ids,update)
        report['heldout_evaluation_seconds']+=time.perf_counter()-start;return result
    def measured_probe():
        start=time.perf_counter();result=reference.probes(model)
        report['policy_guard_seconds']+=time.perf_counter()-start;report['policy_guard_checks']+=1;return result
    if not run_phase:
        model.utility_calibration=None
        report.update(diagnostics(model,current,heldout_ids,config,update),seconds=time.perf_counter()-began)
        return report
    if ppg_config.linked(config) and ppo_peak is None:
        model.utility_calibration=None
        report.update(early_stop_reason='no_ppo_steps_for_linked_rate',learning_rate=0.,
            **diagnostics(model,current,heldout_ids,config,update),seconds=time.perf_counter()-began)
        return report
    peak=memory(current.device)
    def sample_memory():
        for k,v in memory(current.device).items():
            if v is not None:peak[k]=max(peak.get(k) or 0,v)
    with buffer.datasets(current.device) as (ds,policy,receipts):
        train,held,validation,ages,support=layout(ds,receipts,config,snapshot_weight)
        if companion is not None and companion.active:companion.inventory(ds,[i for group in train.values() for i in group])
        report['age_support']=support
        if len(receipts)<config['buffer_iterations'] or not validation:
            report['early_stop_reason']='insufficient_fresh_holdouts'
        else:
            reference_start=time.perf_counter()
            reference=Reference(model,policy,receipts,config,update)
            report['policy_reference_seconds']=time.perf_counter()-reference_start
            report['policy_support']=reference.support
            if (not reference.clone_ids or not reference.probe_ids or
                    not any(reference.ages[i]==0 for i in reference.probe_ids)):
                report['early_stop_reason']='no_policy_reference'
        if report['early_stop_reason']:
            model.utility_calibration=None;report.update(diagnostics(model,current,heldout_ids,config,update))
            report.update(seconds=time.perf_counter()-began,memory=peak);return report
        before_losses=measured_losses(ds,held);best_games=by_game(ds,{i:before_losses[i] for i in validation})
        report['heldout_before']=value_phase.metrics(model,current,heldout_ids,update,evidence=True)
        report['heldout_loss_by_age_before']=age_metrics(ds,before_losses,ages)
        report['acceptance_games']=len(best_games);report['acceptance_generations']=[r['generation'] for r in receipts[:config['validation_iterations']]]
        calibration_before=value_phase.metrics(model,current,train_ids,update)
        previous_steps=[int(s.get('step',0)) for s in optimizer.state.values()]
        report['adam_reset']=dict(reset=True,parameter_states=len(optimizer.state),
            previous_max_step=max(previous_steps,default=0),generation=receipts[0]['generation'],
            scope='Reset value Adam moments/step counters once at phase start; PPO Adam unchanged. Best-state rollback stays within this phase.')
        optimizer.state.clear();optimizer.zero_grad(set_to_none=True)
        if companion is not None:companion.begin('ppg',reset=True)
        for group in optimizer.param_groups:
            group['lr']=peak_rate
            if ppg_config.linked(config):group[ppg_phasic.RATE_KEY]=copy.deepcopy(ppo_peak)
        best_model=copy.deepcopy(model.state_dict());best_adam=copy.deepcopy(optimizer.state_dict())
        best_aux=companion.snapshot() if companion is not None else None
        selection=None
        if ppg_config.coverage(config):
            from .ppg_selection import Selection
            selection=Selection(model,optimizer,best_games,companion=companion,config=config['selection'])
        # Loss-selected states at steps 16,48,... are provisional. The separate
        # certified selection contains only states with their OWN exact KL check.
        # A later endpoint's KL never certifies an earlier weight snapshot.
        certified=Selection(model,optimizer,best_games,companion=companion,config=config['selection']) if delayed else None
        counts={str(a):len(ids) for a,ids in train.items()};rows=sum(counts.values())
        steps_per_epoch=math.ceil(rows/config['batch_size'])
        report.update(weighted_rows=rows,rows_per_epoch=rows,planned_presentations=rows*epoch_cap,
            sampled_rows_by_age={a:n*epoch_cap for a,n in counts.items()},
            planned_steps=steps_per_epoch*epoch_cap,steps_per_epoch=steps_per_epoch,weighting='uniform',
            weighting_scope='Each eligible critic row once per epoch; all eight age weights are 1. Policy strata are population-corrected once; existing snapshot critic loss weight is retained.',
            clone_rows=len(reference.clone_ids),probe_rows=len(reference.probe_ids))
        if report['planned_steps']:
            report['rate_schedule']=dict(config['rate_schedule'],warmup_steps=min(config['acceptance_steps'],report['planned_steps']),
                planned_steps=report['planned_steps'],peak_lr=peak_rate,
                first_lr=ppg_config.phase_rate(config,1,report['planned_steps'],peak=peak_rate),
                final_lr=ppg_config.phase_rate(config,report['planned_steps'],report['planned_steps'],peak=peak_rate),
                scope='Linear warmup over the first acceptance interval, then linear decay to 0.3x at the planned epoch cap; retries reuse their original planned-step rate.')
            if delayed:report['rate_schedule'].update(warmup_steps=0,
                scope='Constant 0.35 times this iteration actual last PPO LR, including retries; no warmup or decay.')
        clone_weights=torch.tensor(reference.clone_weights,dtype=torch.float64)
        working=[];best_used=[];best_clones=[];working_clones=[];checks_without_best=0;working_steps=0
        maximum=0.;maximum_age0=0.;accepted_checks=0;rejected_checks=0
        probe=None;checked_position=0
        def safe_probe(result):return result['overall']<=config['kl_cap'] and result['age0']<=config['kl_cap']
        def verify_restored():
            nonlocal maximum,maximum_age0
            result=measured_probe();report['retention_policy_checks']+=1
            maximum=max(maximum,result['overall']);maximum_age0=max(maximum_age0,result['age0'])
            report['retention_checks'].append(dict(overall=result['overall'],age0=result['age0'],safe=safe_probe(result)))
            return safe_probe(result)
        for epoch,start,block,end_epoch in intervals(train,config,epoch_cap):
            report['epochs_started']=epoch+1
            active_dropout=(dropout[0],epoch,*dropout[2:]) if dropout is not None else None
            # Reuse the same cloning draws if this interval needs a higher beta.
            # They already correct stratum quotas; there is no second weighting.
            clone_batches=[[reference.clone_ids[j] for j in torch.multinomial(clone_weights,
                config['clone_batch_size'],replacement=True).tolist()] for _ in block]
            while True:
                attempted=0;trip=False
                for index,(batch,sample) in enumerate(zip(block,clone_batches)):
                    position=epoch*steps_per_epoch+start+index+1
                    rate=ppg_config.phase_rate(config,position,report['planned_steps'],peak=peak_rate)
                    for group in optimizer.param_groups:group['lr']=rate
                    report['applied_lr_min']=min(report['applied_lr_min'] or rate,rate)
                    report['applied_lr_max']=max(report['applied_lr_max'] or rate,rate)
                    report['last_attempted_lr']=rate
                    optimizer.zero_grad(set_to_none=True);model.eval()
                    if companion is not None:companion.begin('ppg',rate_factor=rate/peak_rate)
                    train_start=time.perf_counter()
                    for part,ix,inputs in ds.value_batches(value_phase.value_parts(ds,batch,update,model),model):
                        with autocast(update,ds.device):
                            if companion is not None:out,local_loss=companion.forward(model,inputs,ds,ix,dropout=active_dropout)
                            else:out,local_loss=value_output(model,inputs,ds,ix,dropout=active_dropout),None
                        loss=value_losses(out,tuple(t[ix] for t in ds.targets))[0]+(out[4].float()-ds.utility[ix]).square().mean(-1)
                        if local_loss is not None:
                            loss=loss+local_loss
                            companion.account_loss(local_loss,ds.critic_weights[ix],len(batch),'ppg')
                        ((loss*ds.critic_weights[ix]).sum()/len(batch)).backward()
                    report['critic_training_seconds']+=time.perf_counter()-train_start
                    # Eval mode freezes stochastic layers; gradients remain on.
                    clone_start=time.perf_counter()
                    for part in reference.microbatches(sample):
                        inputs=reference.inputs(part)
                        with autocast(update,policy.device):loss=reference.loss(part,inputs,model)
                        (beta*loss.sum()/len(sample)).backward()
                    report['cloning_training_seconds']+=time.perf_counter()-clone_start
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),update['grad_clip'],error_if_nonfinite=True)
                    report['core_gradient_norm_max']=max(report['core_gradient_norm_max'],float(norm))
                    start_optimizer=time.perf_counter();optimizer.step();report['core_optimizer_seconds']+=time.perf_counter()-start_optimizer
                    attempted+=1;report['attempted_steps']+=1
                    if companion is not None:companion.step('ppg')
                    report['budget_steps_attempted']=max(report['budget_steps_attempted'],position)
                    report['attempted_value_presentations']+=len(batch)
                    report['attempted_clone_presentations']+=len(sample)
                    # v10 separates provisional loss selection at 16 from KL
                    # at 32 and epoch tails. Older recipes check every interval.
                    # Final retention independently verifies selected weights.
                    check=(position%report['kl_check_steps']==0 or (end_epoch and index+1==len(block))
                        if delayed else (index+1)%report['kl_check_steps']==0 or index+1==len(block))
                    if check:
                        probe=measured_probe();maximum=max(maximum,probe['overall']);maximum_age0=max(maximum_age0,probe['age0'])
                        checked_position=position;report['scheduled_policy_checks']+=1
                        sample_memory()
                        if probe['overall']>config['kl_cap'] or probe['age0']>config['kl_cap']:
                            report['kl_guard_tripped']=True;trip=True;break
                    working.extend(batch);working_clones.extend(sample);working_steps+=1
                if not trip:break
                # Unsafe weights are diagnostic only. More cloning pressure is
                # justified only if their paired held-out value loss improved.
                unsafe=compare(best_games,by_game(ds,measured_losses(ds,validation)))
                retry=unsafe['improved'] and not unsafe['reject'] and beta<config['beta_clone_max']
                before_beta=beta
                if retry:
                    beta=min(config['beta_clone_max'],2*beta);report['beta_retries']+=1
                model.load_state_dict(best_model);optimizer.load_state_dict(best_adam)
                if companion is not None:companion.restore(best_aux)
                if delayed and not verify_restored():
                    # The minimum-loss state could be an unchecked 16-step
                    # state. Reject it, including its Adam and companion state.
                    report['rejected_unverified_selections']+=1
                    certified.restore(model,optimizer)
                    selection=copy.copy(certified)
                    best_model=copy.deepcopy(model.state_dict());best_adam=copy.deepcopy(optimizer.state_dict())
                    best_aux=companion.snapshot() if companion is not None else None
                    best_games=copy.deepcopy(certified.games);best_used=list(certified.rows);best_clones=list(certified.clones)
                    report['optimizer_steps']=certified.steps;report['best_budget_step']=certified.budget_step
                working=list(best_used);working_clones=list(best_clones);working_steps=report['optimizer_steps']
                report['interval_reports'].append(dict(epoch=epoch+1,attempted_steps=attempted,accepted=False,
                    reason='policy_kl_cap',probe=probe,unsafe_paired=unsafe,retry=retry,beta_before=before_beta,beta_after=beta,last_lr=rate))
                rejected_checks+=1;report['beta_clone_final']=beta
                if not retry:
                    report['early_stop_reason']='policy_kl_cap_beta_max' if beta>=config['beta_clone_max'] else 'policy_kl_cap_no_value_improvement'
                    break
            if trip:break
            if end_epoch:report['epochs_completed']=epoch+1
            now=by_game(ds,measured_losses(ds,validation));comparison=compare(best_games,now)
            report['heldout_checks']+=1
            improved=comparison['improved'] and not comparison['reject']
            retained=False
            if selection is not None:
                retained=selection.consider(model,optimizer,now,working,working_clones,working_steps,
                    epoch*steps_per_epoch+start+len(block))
            if delayed and checked_position==position:
                certified.consider(model,optimizer,now,working,working_clones,working_steps,position)
            report['interval_reports'].append(dict(epoch=epoch+1,budget_step=epoch*steps_per_epoch+start+len(block),
                attempted_steps=attempted,accepted=not comparison['reject'],beta_clone=beta,last_lr=rate,
                retained_as_best=improved,paired=comparison,probe=probe if not delayed or checked_position==position else None,
                policy_checked=not delayed or checked_position==position))
            if selection is not None:
                report['interval_reports'][-1].update(retained_for_coverage=retained,
                    minimum_reference_step=selection.minimum_budget_step,
                    selected_budget_step=selection.budget_step,
                    outcome='improved' if improved else 'failed' if comparison['reject'] else 'neutral')
            if improved:
                best_model=copy.deepcopy(model.state_dict());best_adam=copy.deepcopy(optimizer.state_dict());best_games=now
                best_aux=companion.snapshot() if companion is not None else None
                best_used=list(working);best_clones=list(working_clones);report['optimizer_steps']=working_steps
                report['best_budget_step']=epoch*steps_per_epoch+start+len(block)
                checks_without_best=0;accepted_checks+=1
            else:
                # v7: a tolerance-safe plateau is neutral, not a failed check.
                # Archived v5/v6 retain their strict keep-best/patience semantics.
                if selection is None or comparison['reject']:checks_without_best+=1
                if selection is not None and not comparison['reject']:accepted_checks+=1
                if comparison['reject']:
                    rejected_checks+=1;model.load_state_dict(best_model);optimizer.load_state_dict(best_adam)
                    if companion is not None:companion.restore(best_aux)
                    working=list(best_used);working_clones=list(best_clones);working_steps=report['optimizer_steps']
                if checks_without_best>=config['patience']:
                    report['early_stop_reason']='validation_patience';break
        if selection is None:
            model.load_state_dict(best_model);optimizer.load_state_dict(best_adam)
            if companion is not None:companion.restore(best_aux)
        else:
            selection.restore(model,optimizer)
            if delayed and not verify_restored():
                report['rejected_unverified_selections']+=1
                report['kl_guard_tripped']=True
                report['early_stop_reason']='selected_state_policy_kl_cap'
                certified.restore(model,optimizer);selection=certified
            best_used=selection.rows;best_clones=selection.clones
            report['optimizer_steps']=selection.steps;report['selection']=selection.report()
        optimizer.zero_grad(set_to_none=True);model.eval()
        final_probe=measured_probe()
        report['retention_policy_checks']+=1
        require(final_probe['overall']<=config['kl_cap'] and final_probe['age0']<=config['kl_cap'],'Retained PPG checkpoint exceeds KL guard')
        after_losses=measured_losses(ds,held);used=set(best_used)
        calibration_after=value_phase.metrics(model,current,train_ids,update) if used else calibration_before
        model.utility_calibration=(dict(version=config['contract'],reward_id=digest(model.config.utility),
            ready=calibration_after['utility_mse']<=calibration_before['utility_mse']+1e-8,
            decisions=len(train_ids),games=calibration_after['games'],steps=report['optimizer_steps'],
            before=dict(mse=calibration_before['utility_mse']),after=dict(mse=calibration_after['utility_mse']),
            scope='Current ordinary/branch critic-train MC targets; whole-game holdouts excluded; final retained weights') if used else None)
        report.update(heldout_loss_by_age_after=age_metrics(ds,after_losses,ages),probe_kl=final_probe['overall'],
            best_epoch_position=report['best_budget_step']/steps_per_epoch if steps_per_epoch else 0.,
            beta_clone_final=beta,
            learning_rate=optimizer.param_groups[0]['lr'] if used else 0.,
            age0_probe_kl=final_probe['age0'],probe_details=final_probe,maximum_attempted_probe_kl=maximum,
            maximum_attempted_age0_kl=maximum_age0,accepted_intervals=accepted_checks,rejected_intervals=rejected_checks,
            distinct_value_rows_used=len(used),value_presentations=len(best_used),clone_presentations=len(best_clones),
            distinct_clone_rows_used=len(set(best_clones)),
            clone_usage_by_age={str(a):dict(distinct_used=sum(reference.ages[i]==a for i in set(best_clones)),
                presentations=sum(reference.ages[i]==a for i in best_clones)) for a in train},
            branch_distinct_rows_used=sum('/branch/' in ds.data[i][2][0] for i in used),
            distinct_current_rows_used=sum(ages[i]==0 for i in used),distinct_previous_rows_used=sum(ages[i]>0 for i in used),
            used_rows_by_age={str(a):sum(ages[i]==a for i in used) for a in train},
            role_usage={r:dict(distinct_used=sum(ds.roles[i]==r for i in used),presentations=sum(ds.roles[i]==r for i in best_used)) for r in ('learner','snapshot')},
            calibration_receipt=copy.deepcopy(model.utility_calibration))
        report.update(diagnostics(model,current,heldout_ids,config,update))
        report['policy_input_cache']=reference.cache.report();sample_memory()
    report.update(memory=peak,memory_scope='Peak sampled phase RSS, OS process-lifetime RSS high-water, and CUDA allocator high-water; disk peak reported separately',seconds=time.perf_counter()-began)
    return report
