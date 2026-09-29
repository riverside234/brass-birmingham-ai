"""Strict, weights-only loading and complete optimizer/RNG resume metadata."""
from __future__ import annotations

from pathlib import Path
import torch

from .batching import model_id
from .contracts import assert_compatible, compatibility, keys, require
from .model import FourSeatNet, ModelConfig


def save_checkpoint(path, model, optimizer=None, *, step=0, training=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'four-seat-checkpoint/1.0.0', 'compatibility': compatibility(),
        'model_config': model.config.to_dict(), 'model_id': model_id(model),
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict() if optimizer else None,
        'step': step, 'training': training or {}, 'rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload['model_id']


def load_checkpoint(path, *, device='cpu'):
    # No fallback to arbitrary-pickle loading for old or foreign checkpoints.
    raw = torch.load(path, map_location='cpu', weights_only=True)
    keys(raw, ('format', 'compatibility', 'model_config', 'model_id', 'state_dict',
               'optimizer', 'step', 'training', 'rng', 'cuda_rng'))
    require(raw['format'] == 'four-seat-checkpoint/1.0.0', 'Unsupported checkpoint format')
    assert_compatible(raw['compatibility'])
    require(type(raw['step']) is int and raw['step'] >= 0, 'Invalid training step')
    model = FourSeatNet(ModelConfig.from_dict(raw['model_config']))
    model.load_state_dict(raw['state_dict'], strict=True)
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()), 'Non-finite weights')
    require(raw['model_id'] == model_id(model), 'Checkpoint weights/metadata digest mismatch')
    return model.to(device), raw


def resume_optimizer(raw, optimizer, *, restore_rng=True):
    require(raw['optimizer'] is not None, 'Checkpoint has no optimizer state')
    optimizer.load_state_dict(raw['optimizer'])
    if restore_rng:
        torch.set_rng_state(raw['rng'])
        if raw['cuda_rng'] and torch.cuda.is_available():
            require(len(raw['cuda_rng']) == torch.cuda.device_count(), 'CUDA device topology differs on resume')
            torch.cuda.set_rng_state_all(raw['cuda_rng'])
