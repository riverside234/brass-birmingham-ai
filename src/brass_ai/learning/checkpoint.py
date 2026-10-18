"""Strict, weights-only loading and complete optimizer/RNG resume metadata."""
from __future__ import annotations

from pathlib import Path
import torch

from .batching import model_id
from .contracts import assert_compatible, compatibility, keys, require
from .model import FourSeatNet, ModelConfig


def save_checkpoint(path, model, optimizer=None, *, step=0, training=None):
    model.validate_parameter_count()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'four-seat-checkpoint/'+('2.0.0' if getattr(model.config,'utility',None) is not None else '1.0.0'), 'compatibility': compatibility(),
        'model_config': model.config.to_dict(), 'model_id': model_id(model),
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict() if optimizer else None,
        'step': step, 'training': training or {}, 'rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if getattr(model.config,'utility',None) is not None:
        payload['training']=dict(payload['training'],utility_calibration=model.utility_calibration)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload['model_id']


def load_checkpoint(path, *, device='cpu', _legacy_il_upgrade=False):
    # No fallback to arbitrary-pickle loading for old or foreign checkpoints.
    raw = torch.load(path, map_location='cpu', weights_only=True)
    keys(raw, ('format', 'compatibility', 'model_config', 'model_id', 'state_dict',
               'optimizer', 'step', 'training', 'rng', 'cuda_rng'))
    require(raw['format'] in ('four-seat-checkpoint/1.0.0','four-seat-checkpoint/2.0.0'), 'Unsupported checkpoint format')
    # Reject a different tensor layout before the generic source-fingerprint
    # check, so old grouped checkpoints explain that fresh weights are needed.
    model_type,config_type=FourSeatNet,ModelConfig
    if raw['format']=='four-seat-checkpoint/2.0.0':
        from .utility_model import UtilityNet,UtilityConfig
        model_type,config_type=UtilityNet,UtilityConfig
    config=config_type.from_dict(raw['model_config'],allow_legacy=_legacy_il_upgrade)
    if _legacy_il_upgrade:
        from .model import LEGACY_ARCHITECTURE
        from .contracts import assert_heuristic_data_compatible
        require(raw['format']=='four-seat-checkpoint/1.0.0' and config.architecture==LEGACY_ARCHITECTURE
            and raw['compatibility']['architecture_sha256']=='fb085f74f1387258ba6fe8a66fdf40deaf6c6010e513f0c3c310e15129a75302',
            'Only the explicitly pinned completed v11.1 base IL architecture can upgrade')
        assert_heuristic_data_compatible(raw['compatibility'])
    else:assert_compatible(raw['compatibility'])
    require(type(raw['step']) is int and raw['step'] >= 0, 'Invalid training step')
    model=model_type(config)
    require((getattr(model.config,'utility',None) is not None)==(raw['format']=='four-seat-checkpoint/2.0.0'),'Checkpoint utility version differs')
    model.utility_calibration=raw['training'].get('utility_calibration')
    model.load_state_dict(raw['state_dict'], strict=True)
    model.validate_parameter_count()
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()), 'Non-finite weights')
    require(raw['model_id'] == model_id(model,contracts=raw['compatibility']), 'Checkpoint weights/metadata digest mismatch')
    return model.to(device), raw


def resume_optimizer(raw, optimizer, *, restore_rng=True, allow_cuda_topology_change=False):
    require(raw['optimizer'] is not None, 'Checkpoint has no optimizer state')
    optimizer.load_state_dict(raw['optimizer'])
    if restore_rng:
        torch.set_rng_state(raw['rng'])
        if raw['cuda_rng'] and torch.cuda.is_available():
            saved=len(raw['cuda_rng']);visible=torch.cuda.device_count()
            require(allow_cuda_topology_change or saved==visible, 'CUDA device topology differs on resume')
            # Step 5 keeps its learner on logical cuda:0. Additional visible
            # devices only serve deterministic frozen-model inference. Keep
            # existing logical streams; new inference devices retain their
            # initialized stream instead of addressing missing CUDA indices.
            restored=min(saved,visible)
            for index in range(restored):torch.cuda.set_rng_state(raw['cuda_rng'][index],index)
            return dict(saved_devices=saved,visible_devices=visible,restored_devices=restored)
