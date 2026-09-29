"""Clipped on-policy updates for learner seats in independent four-seat games."""
from __future__ import annotations

import torch

from brass_ai.learning.batching import collate, model_id
from brass_ai.learning.contracts import require
from brass_ai.learning.training import prepared, losses
from .policy import exploration_distribution


def behavior_probabilities(logits, rows, behavior):
    probabilities = torch.softmax(logits / behavior.temperature, -1)
    q = torch.zeros_like(probabilities)
    for i, row in enumerate(rows):
        q[i, :len(row['choices'])] = torch.as_tensor(
            exploration_distribution(row['choices'], row['intent']), device=logits.device,
            dtype=logits.dtype)
    return (1 - behavior.exploration) * probabilities + behavior.exploration * q


def ppo_update(model, optimizer, games, behavior, *, epochs=3, batch_size=32,
               clip=.2, entropy_weight=.01):
    """One frozen collection, several clipped epochs; only learner actor rows.

    Terminal rewards are each seat's own win share. No team reward, rule-policy
    imitation, opponent gradients, strategy reward shaping or reused stale actor
    data. Gradients accumulate over a complete epoch before each optimizer step.
    """
    require(all(g['split'] == 'train' for g in games), 'Held-out games cannot update weights')
    require(model_id(model) == behavior.evaluator.model_id, 'Collection weights differ from learner')
    require(epochs > 0 and batch_size > 0 and 0 < clip < 1, 'Invalid PPO configuration')
    data = [e for e in prepared(games) if e[0]['behavior']['policy_id'] == behavior.policy_id]
    require(data and all(e[0]['behavior']['kind'] == 'model' for e in data), 'No on-policy learner seats')
    trajectories = len({e[2] for e in data})
    device = next(model.parameters()).device
    advantages = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            rows = [e[0] for e in batch]
            output = model(*collate([e[3] for e in batch], device))
            p = behavior_probabilities(output[0], rows, behavior)
            chosen = torch.tensor([r['chosen'] for r in rows], device=device)
            actual = p.gather(1, chosen[:, None]).squeeze(1)
            old = torch.tensor([r['behavior']['probability'] for r in rows], device=device)
            require(torch.allclose(actual, old, rtol=3e-3, atol=3e-6), 'Wrong/stale behavior probabilities')
            for i, (row, target, _, _) in enumerate(batch):
                advantages.append(target['win_share'][row['seat']] - float(output[1][i, row['seat']]))
    metrics = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        totals = dict(actor=0.0, value=0.0, entropy=0.0, approx_kl=0.0)
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            rows, targets = [e[0] for e in batch], [e[1] for e in batch]
            output = model(*collate([e[3] for e in batch], device))
            p = behavior_probabilities(output[0], rows, behavior)
            chosen = torch.tensor([r['chosen'] for r in rows], device=device)
            new = p.gather(1, chosen[:, None]).squeeze(1).clamp_min(1e-30)
            old = torch.tensor([r['behavior']['probability'] for r in rows], device=device)
            ratio = new / old
            advantage = torch.tensor(advantages[start:start + len(batch)], device=device)
            actor = -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).sum() / trajectories
            value = losses(output, rows, targets)['value'].sum() / len(data)
            entropy = -(p * p.clamp_min(1e-30).log()).sum() / len(data)
            (actor + value - entropy_weight * entropy).backward()
            totals['actor'] += float(actor.detach())
            totals['value'] += float(value.detach())
            totals['entropy'] += float(entropy.detach())
            totals['approx_kl'] += float(((ratio - 1) - ratio.log()).detach().sum()) / len(data)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        metrics.append(dict(epoch=epoch, **totals, gradient_norm=float(norm)))
    return dict(epochs=metrics, learner_decisions=len(data), learner_trajectories=trajectories)


def distill_update(model, optimizer, games, *, steps=4, batch_size=32, weight=.25):
    require(all(g['split'] == 'train' for g in games), 'Evaluation teacher data is held out')
    require(0 < weight <= 1 and steps > 0, 'Invalid distillation configuration')
    data = [e for e in prepared(games) if e[0]['teacher'] and e[0]['teacher']['confidence'] > 0]
    if not data:
        return dict(updated=False, reason='No completed fresh-confirmation teacher labels')
    device = next(model.parameters()).device
    metrics = []
    model.train()
    for _ in range(steps):
        batch = [data[i] for i in torch.randint(len(data), (min(batch_size, len(data)),)).tolist()]
        output = model(*collate([e[3] for e in batch], device))
        terms = losses(output, [e[0] for e in batch], [e[1] for e in batch])
        total = weight * terms['teacher'].mean() + terms['value'].mean()
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        metrics.append(float(total.detach()))
    return dict(updated=True, teacher_rows=len(data), losses=metrics)


def reptile_step(model, adapted_models, fraction):
    """Move initialization toward mean task-adapted weights, with no query gradient."""
    require(adapted_models and 0 < fraction <= 1, 'Invalid Reptile step')
    base = model.state_dict()
    require(all(m.config == model.config for m in adapted_models), 'Task architecture changed')
    with torch.no_grad():
        for key, target in base.items():
            mean = torch.stack([m.state_dict()[key].to(target.device) for m in adapted_models]).mean(0)
            require(torch.isfinite(mean).all(), 'Non-finite task adaptation')
            target.lerp_(mean, fraction)
