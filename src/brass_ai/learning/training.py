"""Outcome/teacher learning and an optional on-policy terminal REINFORCE step.

This is a scaffold for the later search/population trainer, not an implementation
of two-player AlphaZero. Valid losses are kept; no strategy-based data filters.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .batching import collate, model_id
from .contracts import require
from .features import Encoder
from .replay import examples


def losses(outputs, rows, targets):
    logits, win, placement, vp = outputs
    device = logits.device
    win_target = torch.tensor([t['win_share'] for t in targets], device=device)
    rank_target = torch.tensor([t['placement'] for t in targets], device=device, dtype=torch.float32)
    vp_target = torch.tensor([t['vp'] for t in targets], device=device, dtype=torch.float32)
    win_loss = -(win_target * win.clamp_min(1e-9).log()).sum(-1)
    rank_loss = F.smooth_l1_loss((placement - 1) / 3, (rank_target - 1) / 3, reduction='none').mean(-1)
    vp_loss = F.smooth_l1_loss(vp / 100, vp_target / 100, reduction='none').mean(-1)
    # Unexamined candidates are excluded from the denominator, not assigned
    # zero target probability under a full-candidate softmax.
    policy_terms = []
    for i, row in enumerate(rows):
        teacher = row['teacher']
        if teacher is None:
            policy_terms.append(logits[i].sum() * 0)
            continue
        coverage = torch.tensor(teacher['coverage'], device=device, dtype=torch.bool)
        target = torch.tensor(teacher['probabilities'], device=device)
        log_p = F.log_softmax(logits[i, :len(coverage)].masked_fill(~coverage, -1e9), dim=-1)
        policy_terms.append(-(target * log_p).sum() * teacher['confidence'])
    policy_loss = torch.stack(policy_terms)
    value_loss = win_loss + 0.25 * rank_loss + 0.1 * vp_loss
    return {'value': value_loss, 'teacher': policy_loss, 'win': win_loss,
            'placement': rank_loss, 'vp': vp_loss}


def prepared(games):
    encoder = Encoder()
    return [(row, targets, trajectory, encoder.encode(row['view'], row['choices'], row['style'], row['intent']))
            for row, targets, trajectory in examples(games)]


def supervised_updates(model, optimizer, data, steps, batch_size=32, *, mode='value'):
    require(mode in ('value', 'teacher'), 'Unknown supervised mode')
    require(steps > 0 and batch_size > 0 and data, 'Invalid training request')
    if mode == 'teacher':
        require(any(r[0]['teacher'] is not None and r[0]['teacher']['confidence'] > 0 for r in data),
                'Teacher mode requires confirmed teacher targets')
    model.train()
    device = next(model.parameters()).device
    metrics = []
    for _ in range(steps):
        indices = torch.randint(len(data), (min(batch_size, len(data)),)).tolist()
        batch = [data[i] for i in indices]
        output = model(*collate([item[3] for item in batch], device))
        terms = losses(output, [item[0] for item in batch], [item[1] for item in batch])
        total = terms['value'].mean()
        if mode == 'teacher':
            total = total + terms['teacher'].mean()
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        metrics.append({**{k: float(v.detach().mean()) for k, v in terms.items()},
                        'total': float(total.detach()), 'gradient_norm': float(norm)})
    return metrics


def on_policy_update(model, optimizer, games, batch_size=32, entropy_weight=0.01):
    """One accumulated update per frozen batch of complete four-seat games.

    Each seat gets its OWN terminal win share and allowed-view value baseline.
    Loss sums decisions within a seat trajectory, then averages trajectories.
    No alternating sign, shared bot reward, bootstrapped truncation or PPO claim.
    """
    data = prepared(games)
    identity = model_id(model)
    require(data and batch_size > 0, 'No on-policy data')
    for row, _, _, _ in data:
        behavior = row['behavior']
        require(behavior['kind'] == 'model' and behavior['policy_id'] == identity and
                behavior['temperature'] == 1.0, 'On-policy update rejects random, stale or tempered behavior')
    trajectories = len({entry[2] for entry in data})
    require(trajectories == len(games) * 4, 'Four complete seat trajectories per game required')
    model.train()
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    total_value = total_actor = total_entropy = 0.0
    try:
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            rows, targets = [e[0] for e in batch], [e[1] for e in batch]
            output = model(*collate([e[3] for e in batch], device))
            logits, win, _, _ = output
            log_p = F.log_softmax(logits, -1)
            chosen = torch.tensor([r['chosen'] for r in rows], device=device)
            seat = torch.tensor([r['seat'] for r in rows], device=device)
            actual = log_p.gather(1, chosen[:, None]).squeeze(1)
            behavior = torch.tensor([r['behavior']['probability'] for r in rows], device=device)
            require(torch.allclose(actual.detach().exp(), behavior, rtol=2e-3, atol=2e-6),
                    'Recorded probabilities do not match the frozen learner')
            reward = torch.tensor([t['win_share'][r['seat']] for r, t in zip(rows, targets)], device=device)
            baseline = win.gather(1, seat[:, None]).squeeze(1).detach()
            actor = -(actual * (reward - baseline)).sum() / trajectories
            value = losses(output, rows, targets)['value'].sum() / len(data)
            entropy = -(log_p.exp() * log_p).sum() / len(data)
            (actor + value - entropy_weight * entropy).backward()
            total_value += float(value.detach())
            total_actor += float(actor.detach())
            total_entropy += float(entropy.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {'value': total_value, 'actor': total_actor, 'entropy': total_entropy,
            'gradient_norm': float(norm), 'games': len(games), 'seat_trajectories': trajectories}


def evaluate_values(model, games, batch_size=32):
    data = prepared(games)
    model.eval()
    device = next(model.parameters()).device
    total = {k: 0.0 for k in ('win', 'placement', 'vp')}
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            output = model(*collate([e[3] for e in batch], device))
            terms = losses(output, [e[0] for e in batch], [e[1] for e in batch])
            for key in total:
                total[key] += float(terms[key].sum()) / len(data)
    return total
