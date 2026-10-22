"""Bounded policy measurements; no RNG draws or optimizer writes."""
import math
import numpy as np
import torch

from brass_ai.learning.contracts import require


PPO_KL_GUARD = 'ppo-categorical-kl-guard/1.0.0'


def mixed_ev_status(tables):
    """Current bounded evidence, also after admission; no gate-state changes."""
    result={}
    for kind,table in tables.items():
        evidence=table.get('evidence')
        if not evidence:continue
        mixed=evidence.get('mixed',{});reference=evidence.get('selfplay',{})
        a=mixed.get('explained_variance');b=reference.get('explained_variance')
        finite=all(type(x) in (int,float) and math.isfinite(x) for x in (a,b))
        supported=evidence.get('supported') is True and finite
        status=('ok' if evidence.get('ev_ok') is True else 'below_tolerance') if supported else 'insufficient_support'
        result[kind]=dict(status=status,enabled=table.get('enabled',False),
            lambda_used=table.get('lambda_used'),lambda_next=table.get('lambda_next'),
            pooled_ev=a,selfplay_ev=b,ev_gap=a-b if finite else None,
            games=mixed.get('games'),selfplay_games=reference.get('games'),
            contributing_iterations=evidence.get('contributing_iterations'),
            window_start=evidence.get('window_start'),window_end=evidence.get('window_end'),
            fresh=evidence.get('fresh'),reasons=list(evidence.get('reasons',[])),
            scope='Existing pooled EV admission tolerance; review signal only, no automatic rollback')
    return result


def mixed_ev_console(diagnostics):
    def number(x):return f'{x:.3f}' if type(x) in (int,float) and math.isfinite(x) else 'na'
    rows=diagnostics.get('mixed_ev_status',{})
    return (' mixedEV='+','.join(f'{k}:{number(v["pooled_ev"])}vs{number(v["selfplay_ev"])}[{v["status"]}]'
        for k,v in rows.items())) if rows else ''


def gate_diagnostics(actor,value=None,limit=None,quality=None):
    """Expose coupled stalls from existing measurements; never rewrite gates."""
    value=value or {};limit=limit or {};quality=quality or {}
    health=limit.get('health',{})
    planning=quality.get('planning',{});attempts=value.get('attempted_steps',0)
    before=value.get('heldout_before',{}).get('loss');after=value.get('heldout_after',{}).get('loss')
    groups=value.get('heldout_after',{}).get('groups',{})
    tables=actor.get('gae',{}).get('gate',{}).get('table_gate',{})
    mixed=mixed_ev_status(tables)
    warnings=[]
    if limit.get('alarm'):warnings.append('il_distance_alarm')
    if planning.get('N')==0 and limit.get('baseline') is not None:
        warnings.append('history_pool_il_only')
    if attempts and not value.get('optimizer_steps'):warnings.append('value_fit_rejected')
    warnings.extend('pool_regression:'+row['model_id'] for row in quality.get('alarms',[]))
    for kind,row in mixed.items():
        if row['enabled'] and row['lambda_next'] is not None and row['lambda_next']<1. and row['status']=='below_tolerance':
            warnings.append('mixed_ev_below_tolerance:'+kind)
    return dict(contract='selfplay-gate-diagnostics/2.2.0',warnings=warnings,
        health_ok=health.get('ok'),health_reasons=list(health.get('reasons',[])),
        lr_blockers=list(limit.get('lr_gate',{}).get('reasons',[])),
        scored_history_members_used=planning.get('N'),
        history_il_only=planning.get('N')==0 if 'N' in planning else None,
        history_admission_reason=quality.get('admission',{}).get('reason'),
        pool_regression_alarms=quality.get('alarms',[]),
        value_accepted_steps=value.get('optimizer_steps'),value_attempted_steps=value.get('attempted_steps'),
        value_acceptance_interval=value.get('acceptance_steps'),
        value_accepted_intervals=value.get('accepted_intervals'),
        value_accepted_fraction=value.get('optimizer_steps',0)/attempts if attempts else None,
        value_stop=value.get('early_stop_reason'),beta_reason=value.get('beta_reason'),
        critic_training=value.get('critic_training'),
        heldout_loss_delta=after-before if before is not None and after is not None else None,
        heldout_by_table={k:dict(rows=groups.get('table:'+k+'/role:learner',{}).get('rows',0),
            games=groups.get('table:'+k+'/role:learner',{}).get('games'),
            ev=groups.get('table:'+k+'/role:learner',{}).get('explained_variance'))
            for k in ('selfplay','historical','legacy')},
        gae_reason=actor.get('gae',{}).get('gate',{}).get('reason'),
        gae_stage_evidence=actor.get('gae',{}).get('gate',{}).get('stage_evidence'),
        mixed_ev_evidence=tables,mixed_ev_status=mixed,
        scope='Existing evidence only; missing groups are unavailable, not zero EV; no automatic threshold changes')


def categorical_kl(old, current):
    """Sum over legal actions; masked probabilities are exactly zero."""
    return (old*(old.clamp_min(1e-30).log()-current.clamp_min(1e-30).log())).sum()


def distribution(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), p05=float(np.quantile(values, .05)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                max=float(values.max()))


def reward_distribution(trajectories, spec):
    weights = dict(win=spec['win'], placement=spec['placement'], vp=spec['vp'], cost=-1.)
    names = list(weights)
    values = np.asarray([[t[k]*weights[k] for k in names] for t in trajectories])
    centered = values-values.mean(0)
    covariance = centered.T @ centered / len(values)
    return dict(weighted={k:distribution(values[:, i]) for i, k in enumerate(names)},
                utility=distribution([t['utility'] for t in trajectories]),
                covariance={a:{b:float(covariance[i,j]) for j,b in enumerate(names)} for i,a in enumerate(names)},
                capped_trajectories=sum(t['cost'] >= spec['insolvency']['max_cost']-1e-8 for t in trajectories),
                scope='One complete-game observation per learner seat; population moments, not independent decisions')


class CategoricalProbe:
    """Exact action-distribution KL on the existing bounded nonforced probe.

    Old probabilities are captured once before PPO and remain fixed across all
    steps/epochs. The same probe drives early stopping and post-update reporting.
    Tensor snapshots use a bounded device/host budget and live for this probe
    only. Probability checks still run after every optimizer step.
    """
    def __init__(self, model, dataset, indices, behavior, config):
        from .updates import row_probabilities
        from .batch_cache import PROBE_CACHE_BYTES, probe_device_limit, tensor_bytes
        self.parts = []; self.batches = []; self.seals = []
        self.decoupled='opening_objective' in config
        self.raw_parts=[];self.opening_masks=[];self.opening_positions=0
        self.dataset, self.behavior = dataset, behavior
        self.positions = len(indices)
        self.device_limit = probe_device_limit(dataset.device)
        self.host_limit = PROBE_CACHE_BYTES if torch.device(dataset.device).type == 'cuda' else 0
        self.device_bytes = self.host_bytes = self.cache_hits = self.cache_misses = 0
        with torch.no_grad():
            for part in dataset.microbatches(indices, config['microbatch_size'],
                                            config['max_candidate_cells'], config['bucket']):
                ix, inputs, q = dataset.batch(part, behavior.exploration)
                logits=model(*inputs)[0]
                p = row_probabilities(dataset,part,logits,q,behavior)
                tensors = (ix,*inputs,q)
                before = p.detach().clone()
                raw=logits.float().softmax(-1).detach() if self.decoupled else None
                size = tensor_bytes((*tensors,before,*((raw,) if raw is not None else ())))
                if size <= self.device_limit-self.device_bytes:
                    self.device_bytes += size
                elif size <= self.host_limit-self.host_bytes:
                    tensors = tuple(t.cpu().pin_memory() for t in tensors)
                    before = before.cpu().pin_memory()
                    if raw is not None:raw=raw.cpu().pin_memory()
                    self.host_bytes += size
                else:
                    tensors = None
                    before = before.cpu()
                    if raw is not None:raw=raw.cpu()
                self.parts.append((part,before))
                self.raw_parts.append(raw)
                mask=dataset.temperatures[ix,0]==2. if self.decoupled else None
                self.opening_masks.append(mask)
                if mask is not None:self.opening_positions+=int(mask.sum())
                self.batches.append(tensors)
                self.seals.append(dataset.input_seal(part,behavior.exploration))

    def _batch(self, index):
        part, before = self.parts[index]
        require(self.dataset.input_seal(part,self.behavior.exploration)==self.seals[index],
                'Fixed PPO probe inputs changed after admission')
        tensors = self.batches[index]
        if tensors is None:
            self.cache_misses += 1
            ix,inputs,q = self.dataset.batch(part,self.behavior.exploration)
        else:
            self.cache_hits += 1
            tensors = tuple(t.to(self.dataset.device,non_blocking=True) for t in tensors)
            ix,inputs,q = tensors[0],tensors[1:-1],tensors[-1]
        return part,before.to(self.dataset.device,non_blocking=True),ix,inputs,q

    def cache_report(self):
        device = torch.device(self.dataset.device).type
        return dict(device=device,resident_bytes=self.device_bytes,limit_bytes=self.device_limit,
                    host_fallback_bytes=self.host_bytes,host_fallback_limit_bytes=self.host_limit,
                    cached_parts=sum(batch is not None for batch in self.batches),parts=len(self.parts),
                    hits=self.cache_hits,uncached_calls=self.cache_misses,
                    scope='Immutable packed inputs and old probabilities; one PPO iteration, bounded, content checked')

    def measure_step(self, model, *, executor=None):
        """Exact and sampled KL from ONE forward pass per cached microbatch.

        Reuse the bounded CPU snapshots and optional pinned-input pipeline.
        Accumulate on-device, then synchronize once; do not reevaluate the IL
        anchor, resample positions or advance training RNG inside the guard.
        """
        from .updates import row_probabilities
        totals = torch.zeros(4, device=self.dataset.device)
        with torch.no_grad():
            for index in range(len(self.parts)):
                _,old,ix,inputs,q = self._batch(index)
                logits=model(*inputs)[0]
                p = row_probabilities(self.dataset,ix,logits,q,self.behavior)
                selected = p.gather(1, self.dataset.chosen[ix, None]).squeeze(1).clamp_min(1e-30)
                ratio = selected/self.dataset.old[ix]
                raw_kl=opening_kl=torch.zeros((),device=self.dataset.device)
                if self.decoupled:
                    raw=logits.float().softmax(-1);before=self.raw_parts[index].to(raw.device,non_blocking=True)
                    mask=self.opening_masks[index]
                    raw_kl=categorical_kl(before,raw);opening_kl=categorical_kl(before[mask],raw[mask])
                totals += torch.stack((categorical_kl(old, p), ((ratio-1)-ratio.log()).sum(),raw_kl,opening_kl))
        exact,sampled,raw,opening=totals.cpu().tolist()
        require(all(math.isfinite(x) for x in (exact,sampled,raw,opening)), 'Nonfinite PPO KL probe; refusing further updates')
        result=dict(positions=self.positions,categorical_kl=(raw if self.decoupled else exact)/max(1,self.positions),
                    sampled_kl=sampled/max(1,self.positions))
        if self.decoupled:result.update(self._movement(exact,raw,opening))
        return result

    def _movement(self,behavior,raw,opening):
        return dict(behavior_categorical_kl=behavior/max(1,self.positions),
            temperature1_categorical_kl=raw/max(1,self.positions),
            opening_temperature1_categorical_kl=opening/self.opening_positions if self.opening_positions else None,
            opening_positions=self.opening_positions,
            kl_scope='categorical_kl is raw T1 movement and the LR signal. Raw T1 and actual-behavior exact KL each retain the PPO per-step cap. Opening-only KL is a diagnostic; sampled KL remains behavior-policy movement.')

    def measure(self, model, anchor=None):
        from .updates import row_probabilities
        kl = entropy = relative_entropy = anchor_kl = raw_entropy = raw_relative = raw_kl = opening_kl = 0.
        groups={};unassigned=0
        if anchor is not None:
            # Reuse the existing round boundaries, not the action/payment phase
            # labels used to stratify the probe. Do not resample or reweight it.
            from .teacher_schedule import PHASES,phase_for
            groups={name:dict(positions=0,total=0.) for name,_,_,_ in PHASES}
        with torch.no_grad():
            for index in range(len(self.parts)):
                part,old,ix,inputs,q = self._batch(index)
                logits=model(*inputs)[0]
                p = row_probabilities(self.dataset,part,logits,q,self.behavior)
                kl += float(categorical_kl(old, p))
                ent = -(p*p.clamp_min(1e-30).log()).sum(-1)
                legal = inputs[2].sum(-1).float()
                entropy += float(ent.sum())
                relative_entropy += float((ent/legal.log().clamp_min(1e-8)).sum())
                raw=torch.softmax(logits.float(),-1)
                if self.decoupled:
                    before=self.raw_parts[index].to(raw.device,non_blocking=True);mask=self.opening_masks[index]
                    raw_kl+=float(categorical_kl(before,raw))
                    opening_kl+=float(categorical_kl(before[mask],raw[mask]))
                raw_ent=-(raw*raw.clamp_min(1e-30).log()).sum(-1)
                raw_entropy+=float(raw_ent.sum())
                raw_relative+=float((raw_ent/legal.log().clamp_min(1e-8)).sum())
                if anchor is not None:
                    reference=row_probabilities(self.dataset,part,anchor(*inputs)[0],q,self.behavior)
                    distances=(reference*(reference.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum(-1).cpu().tolist()
                    anchor_kl+=sum(distances)
                    for row_index,distance in zip(part,distances):
                        context=self.dataset.rows[row_index].get('view',{}).get('context',{})
                        name=phase_for(context) if 'era' in context and type(context.get('round')) is int else None
                        if name is None:unassigned+=1
                        else:
                            groups[name]['positions']+=1;groups[name]['total']+=distance
        group_means={name:dict(positions=row['positions'],
            anchor_distance=row['total']/row['positions'] if row['positions'] and np.isfinite(row['total']) else None)
            for name,row in groups.items()}
        return dict(positions=self.positions, categorical_kl=(raw_kl if self.decoupled else kl)/max(1,self.positions),
                    **(self._movement(kl,raw_kl,opening_kl) if self.decoupled else {}),
                    raw_entropy=raw_entropy/max(1,self.positions),
                    raw_entropy_over_log_legal=raw_relative/max(1,self.positions),
                    entropy=entropy/max(1,self.positions),
                    entropy_over_log_legal=relative_entropy/max(1,self.positions),
                    **(dict(anchor_distance=anchor_kl/self.positions if self.positions else None,
                        anchor_groups=group_means,anchor_unassigned_positions=unassigned,
                        anchor_group_scope='Four existing era/round groups; per-position means on the unchanged nonforced probe',
                        anchor_scope='Exact KL(fixed IL behavior || post-PPO behavior), same probe and exploration mixture') if anchor is not None else {}),
                    scope=('Exact KL(old raw T1 || new raw T1) plus separate behavior-policy KL' if self.decoupled else
                        'Exact categorical KL(old behavior || new behavior)')+
                        ' on the proportional era/phase nonforced probe; PPO per-step early stopping and optional LR feedback')


def teacher_drift_probe(model, data, behavior, config):
    """Snapshot a bounded post-PPO policy probe without advancing training RNG.

    Only the selected rows are tensorized. The resulting KL isolates subsequent
    teaching from PPO; it is not the difference between two total KL values.
    """
    from types import SimpleNamespace
    from .updates import Dataset, phase_probe
    rows = [entry[0] for entry in data]
    meaningful = [i for i, row in enumerate(rows) if len(row['choices']) > 1]
    with torch.random.fork_rng(devices=[]):
        selected = phase_probe(SimpleNamespace(rows=rows), meaningful, config['kl_probe_rows'])
    if not selected:
        return None
    dataset = Dataset([data[i] for i in selected], next(model.parameters()).device)
    model.eval()
    return CategoricalProbe(model, dataset, list(range(len(selected))), behavior, config)
