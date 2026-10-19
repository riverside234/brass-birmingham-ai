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
    """Expose coupled stalls from existing measurements; no new thresholds.

    This is reporting only. It must not turn healthy economy averages into an
    override for failed per-position behavior checks or rewrite gate state.
    """
    value=value or {};limit=limit or {};quality=quality or {}
    health=limit.get('health',{});behavior=limit.get('measurement',{}).get('behavior',{})
    passing=behavior.get('passing') or {}
    planning=quality.get('planning',{});attempts=value.get('attempted_steps',0)
    before=value.get('heldout_before',{}).get('loss');after=value.get('heldout_after',{}).get('loss')
    groups=value.get('heldout_after',{}).get('groups',{})
    tables=actor.get('gae',{}).get('gate',{}).get('table_gate',{})
    mixed=mixed_ev_status(tables)
    warnings=[]
    if health.get('ok') is False and 'behavior_check_failed' in health.get('reasons',[]):
        warnings.append('il_behavior_check_failed')
    if limit.get('alarm'):warnings.append('il_distance_alarm')
    if planning.get('N')==0 and limit.get('baseline') is not None:
        warnings.append('history_pool_il_only')
    if attempts and not value.get('optimizer_steps'):warnings.append('value_fit_rejected')
    for kind,row in mixed.items():
        if row['enabled'] and row['lambda_next'] is not None and row['lambda_next']<1. and row['status']=='below_tolerance':
            warnings.append('mixed_ev_below_tolerance:'+kind)
    return dict(contract='selfplay-gate-diagnostics/2.1.0',warnings=warnings,
        health_ok=health.get('ok'),health_reasons=list(health.get('reasons',[])),
        pass_failures=passing.get('failures'),pass_positions=passing.get('positions'),
        lr_blockers=list(limit.get('lr_gate',{}).get('reasons',[])),
        scored_history_members_used=planning.get('N'),
        history_il_only=planning.get('N')==0 if 'N' in planning else None,
        history_admission_reason=quality.get('admission',{}).get('reason'),
        value_accepted_steps=value.get('optimizer_steps'),value_attempted_steps=value.get('attempted_steps'),
        value_accepted_fraction=value.get('optimizer_steps',0)/attempts if attempts else None,
        value_stop=value.get('early_stop_reason'),beta_reason=value.get('beta_reason'),
        heldout_loss_delta=after-before if before is not None and after is not None else None,
        heldout_by_table={k:dict(rows=groups.get('table:'+k+'/role:learner',{}).get('rows',0),
            games=groups.get('table:'+k+'/role:learner',{}).get('games'),
            ev=groups.get('table:'+k+'/role:learner',{}).get('explained_variance'))
            for k in ('selfplay','historical','legacy')},
        gae_reason=actor.get('gae',{}).get('gate',{}).get('reason'),
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
    Each stored CPU tensor is bounded by the ordinary candidate-cell limit.
    """
    def __init__(self, model, dataset, indices, behavior, config):
        from .updates import probabilities
        self.parts = []
        self.dataset, self.behavior = dataset, behavior
        self.positions = len(indices)
        with torch.no_grad():
            for part in dataset.microbatches(indices, config['microbatch_size'],
                                            config['max_candidate_cells'], config['bucket']):
                _, inputs, q = dataset.batch(part, behavior.exploration)
                p = probabilities(model(*inputs)[0], q, behavior.exploration, behavior.temperature)
                self.parts.append((part, p.detach().cpu()))

    def measure_step(self, model, *, executor=None):
        """Exact and sampled KL from ONE forward pass per cached microbatch.

        Reuse the bounded CPU snapshots and optional pinned-input pipeline.
        Accumulate on-device, then synchronize once; do not reevaluate the IL
        anchor, resample positions or advance training RNG inside the guard.
        """
        from .updates import probabilities
        totals = torch.zeros(2, device=self.dataset.device)
        parts = (part for part, _ in self.parts)
        batches = self.dataset.policy_batches(parts, self.behavior.exploration, executor)
        with torch.no_grad():
            for (_, before), (_, ix, inputs, q) in zip(self.parts, batches):
                p = probabilities(model(*inputs)[0], q, self.behavior.exploration, self.behavior.temperature)
                old = before.to(p.device)
                selected = p.gather(1, self.dataset.chosen[ix, None]).squeeze(1).clamp_min(1e-30)
                ratio = selected/self.dataset.old[ix]
                totals += torch.stack((categorical_kl(old, p), ((ratio-1)-ratio.log()).sum()))
        exact, sampled = (value/max(1, self.positions) for value in totals.cpu().tolist())
        require(math.isfinite(exact) and math.isfinite(sampled), 'Nonfinite PPO KL probe; refusing further updates')
        return dict(positions=self.positions, categorical_kl=exact, sampled_kl=sampled)

    def measure(self, model, anchor=None):
        from .updates import probabilities
        kl = entropy = relative_entropy = anchor_kl = 0.
        groups={};unassigned=0
        if anchor is not None:
            # Reuse the existing round boundaries, not the action/payment phase
            # labels used to stratify the probe. Do not resample or reweight it.
            from .teacher_schedule import PHASES,phase_for
            groups={name:dict(positions=0,total=0.) for name,_,_,_ in PHASES}
        with torch.no_grad():
            for part, before in self.parts:
                _, inputs, q = self.dataset.batch(part, self.behavior.exploration)
                p = probabilities(model(*inputs)[0], q, self.behavior.exploration, self.behavior.temperature)
                old = before.to(p.device)
                kl += float(categorical_kl(old, p))
                ent = -(p*p.clamp_min(1e-30).log()).sum(-1)
                legal = inputs[2].sum(-1).float()
                entropy += float(ent.sum())
                relative_entropy += float((ent/legal.log().clamp_min(1e-8)).sum())
                if anchor is not None:
                    reference=probabilities(anchor(*inputs)[0],q,self.behavior.exploration,self.behavior.temperature)
                    distances=(reference*(reference.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum(-1).cpu().tolist()
                    anchor_kl+=sum(distances)
                    for index,distance in zip(part,distances):
                        context=self.dataset.rows[index].get('view',{}).get('context',{})
                        name=phase_for(context) if 'era' in context and type(context.get('round')) is int else None
                        if name is None:unassigned+=1
                        else:
                            groups[name]['positions']+=1;groups[name]['total']+=distance
        group_means={name:dict(positions=row['positions'],
            anchor_distance=row['total']/row['positions'] if row['positions'] and np.isfinite(row['total']) else None)
            for name,row in groups.items()}
        return dict(positions=self.positions, categorical_kl=kl/max(1,self.positions),
                    entropy=entropy/max(1,self.positions),
                    entropy_over_log_legal=relative_entropy/max(1,self.positions),
                    **(dict(anchor_distance=anchor_kl/self.positions if self.positions else None,
                        anchor_groups=group_means,anchor_unassigned_positions=unassigned,
                        anchor_group_scope='Four existing era/round groups; per-position means on the unchanged nonforced probe',
                        anchor_scope='Exact KL(fixed IL behavior || post-PPO behavior), same probe and exploration mixture') if anchor is not None else {}),
                    scope='Exact categorical KL(old behavior || new behavior) on the proportional era/phase nonforced probe; PPO per-step early stopping and optional LR feedback')


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
