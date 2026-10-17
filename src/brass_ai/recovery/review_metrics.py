"""Bounded diagnostics; no reward normalization, RNG draws or optimizer writes."""
import numpy as np
import torch


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

    Old probabilities are captured before any update. This observes drift but
    does not alter the existing sampled-action early-stop rule or RNG stream.
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
                kl += float((old*(old.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum())
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
                    scope='Exact categorical KL(old behavior || new behavior) on the proportional era/phase nonforced probe; optional PPO LR feedback, unchanged sampled early-stop guard')


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
