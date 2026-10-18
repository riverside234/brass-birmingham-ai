"""Export-only rewrites and explicit CPU graph optimization.

Checkpoint tensors and the trainable model are never rewritten. The portable
file stays at opset 17; exact GELU fusion is checked in ORT's execution graph.
"""
from collections import Counter
from dataclasses import dataclass
import copy
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import require
from .local_entities import LocalEntityEncoder, LocalState


@dataclass
class ExportCandidates:
    context: torch.Tensor
    banks: torch.Tensor


class SelectorMatmulExport(nn.Module):
    """Frozen v11.1 execution: static action weights + one dynamic local MatMul.

    Keep W_act as an initializer so ORT can prepack it. Only the 143 x 160
    projected bank is position-dependent, with no absent rows or candidate
    gathers. This is deliberately export-only; training retains index gathers.
    """
    def __init__(self, source):
        super().__init__()
        self.source = source
        # Derived from the action feature manifest at construction, not from
        # candidate values. Static slices export with dynamic batch/row axes.
        self.reference_slices = source.reference_slices

    @property
    def config(self):return self.source.config

    @property
    def state_encoder(self):return self.source.state_encoder

    @property
    def entity_encoder(self):return self.source.entity_encoder

    def encode_state(self, states, *, context=None):
        return self.source.encode_state(states, context=context)

    def prepare_candidates(self, encoded):
        banks = torch.cat((self.source.entity_projections['target'](encoded.targets),
                           self.source.entity_projections['merchant'](encoded.merchants),
                           self.source.entity_projections['industry'](encoded.industries)), dim=1)
        return ExportCandidates(self.source.project_policy_context(encoded.context), banks)

    def score_prepared(self, actions, prepared):
        selectors = torch.cat([actions[..., start:stop] for start, stop in self.reference_slices], dim=-1)
        local = torch.matmul(selectors, prepared.banks)
        encoder = self.source.action_encoder
        pre = encoder[0](actions) + local
        candidates = encoder[2](encoder[1](pre))
        return self.source.score_candidates(candidates, prepared.context)

    def heads_from_encoded(self, encoded, mapping, remaining=None, descriptors=None):
        return self.source.heads_from_encoded(encoded, mapping, remaining,descriptors)

    def forward(self, states, actions, legal_mask, stable_to_relative, remaining=None, descriptors=None):
        encoded = self.encode_state(states)
        logits = self.score_prepared(actions, self.prepare_candidates(encoded))
        return (logits.masked_fill(~legal_mask, -torch.inf),) + self.heads_from_encoded(encoded, stable_to_relative, remaining,descriptors)


def descriptor_default(path):
    """An overridable ONNX initializer: unknown seats, broadcasts over positions."""
    import onnx
    from onnx import numpy_helper
    from .seat_descriptors import unknown_numpy
    model=onnx.load(str(path))
    require('descriptors' in {v.name for v in model.graph.input},'Missing ONNX critic descriptor input')
    require('descriptors' not in {v.name for v in model.graph.initializer},'Duplicate descriptor default')
    model.graph.initializer.append(numpy_helper.from_array(unknown_numpy()[None],name='descriptors'))
    onnx.checker.check_model(model);onnx.save(model,str(path))


class BatchedSeatEncoder(nn.Module):
    """One grouped MatMul for four seats, retaining both learned encoders."""
    def __init__(self, source):
        super().__init__()
        self.source = source
        own, other = source.encoders['own_player'], source.encoders['other_player']
        require(own[2].eps == other[2].eps, 'Seat normalization epsilons differ')
        self.eps = own[2].eps
        padding = other[0].in_features - own[0].in_features
        require(padding >= 0, 'Own-seat padding would truncate inputs')
        self.register_buffer('seat_indices', torch.cat((F.pad(source.own_player_indices,
            (0, padding), value=-1), source.other_player_indices), dim=0))
        # Export a fixed copy; keep independently learned own/other weights.
        self.register_buffer('seat_weights', torch.stack((F.pad(own[0].weight,
            (0, padding)), *(other[0].weight for _ in range(3)))).detach().transpose(1, 2).contiguous())
        for name, values in (
            ('seat_bias', (own[0].bias, other[0].bias)),
            ('seat_scale', (own[2].weight, other[2].weight)),
            ('seat_shift', (own[2].bias, other[2].bias)),
        ):
            self.register_buffer(name, torch.stack((values[0], *(values[1] for _ in range(3)))).detach().clone())

    def forward(self, states, context):
        banks = {name: self.source.encoders[name](self.source.gather(states, name))
                 for name in ('slot', 'link', 'merchant', 'industry', 'market')}
        indices = self.seat_indices
        players = states.index_select(1, indices.clamp_min(0).flatten()).reshape(-1, 4, indices.shape[1])
        players = players.masked_fill((indices < 0).unsqueeze(0), 0)
        # [4,B,287] @ [4,287,64]; no shared-weight approximation or new flag.
        players = torch.matmul(players.transpose(0, 1), self.seat_weights).transpose(0, 1) + self.seat_bias
        players = F.layer_norm(F.gelu(players), (self.seat_scale.shape[-1],), eps=self.eps)
        players = players * self.seat_scale + self.seat_shift
        targets = torch.cat([banks[name] for name in ('slot', 'link', 'merchant', 'market')], dim=1)
        return LocalState(context, targets, banks['merchant'], banks['industry'],
                          players + self.source.player_identity.weight.unsqueeze(0))


def prepare_export(model, *, local_selectors=True):
    """Independent export copy; `local_selectors=False` is a profiling reference."""
    frozen = copy.deepcopy(model).cpu().eval()
    replaced = 0
    for parent in list(frozen.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LocalEntityEncoder):
                setattr(parent, name, BatchedSeatEncoder(child))
                replaced += 1
    require(replaced == 1, 'Expected one local entity encoder in the export')
    if local_selectors:
        if hasattr(frozen, 'core'):
            frozen.core = SelectorMatmulExport(frozen.core)
        else:
            frozen = SelectorMatmulExport(frozen)
    return frozen


def candidate_export_metadata():
    from .local_entities import entity_layout
    return dict(form='static-action-linear-plus-selector-matmul/1.0.0',
        activation='LayerNorm(GELU(action_linear + selected_local))',
        action_weights='static initializer; eligible for runtime prepacking',
        selector_fields=entity_layout()['action_fields'],
        selector_width=143, dynamic_bank_shape=[143, 160], dynamic_values_per_position=22880,
        candidate_gathers=0, absent_rows=0,
        reuse='prepare once per position; share across score_prepared candidate chunks')


def cpu_options(threads=1, *, profile_prefix=None, optimized_path=None):
    import onnxruntime as ort
    require(type(threads) is int and threads > 0, 'Invalid ONNX CPU thread count')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    # Preserve exact erf GELU; do not enable FastGelu approximations.
    options.add_session_config_entry('optimization.enable_gelu_approximation', '0')
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    if optimized_path is not None:
        options.optimized_model_filepath = str(optimized_path)
    return options


def operator_counts(model):
    return dict(Counter((node.domain + '::' if node.domain else '') + node.op_type
                        for node in model.graph.node))


def simplify_export(path):
    """Simplify without specializing dynamic axes or generating invalid inputs.

    Numerical checks use the exporter's real categorical/masked parity corpus,
    against the original PyTorch model, after this structural pass.
    """
    import onnx
    import onnxsim
    path = Path(path)
    original = onnx.load(str(path))
    inputs = [(v.name, v.type.SerializeToString()) for v in original.graph.input]
    outputs = [v.name for v in original.graph.output]
    simplified, checked = onnxsim.simplify(original, check_n=0, certify=False)
    require(checked, 'ONNX simplifier structural check failed')
    require(inputs == [(v.name, v.type.SerializeToString()) for v in simplified.graph.input]
            and outputs == [v.name for v in simplified.graph.output],
            'Simplification changed the public input/output contract')
    onnx.checker.check_model(simplified)
    onnx.save(simplified, str(path))
    return dict(simplifier='onnxsim', version=onnxsim.__version__, structural_check=True,
        before=operator_counts(original), after=operator_counts(simplified),
        opset=17, dynamic_axes_preserved=True)


def inspect_runtime_fusions(path):
    """Inspect the optimized execution graph without deploying host-specific ops."""
    import onnx
    import onnxruntime as ort
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix='ort-fusion-', dir=path.parent) as directory:
        optimized = Path(directory) / 'execution.onnx'
        session = ort.InferenceSession(str(path), cpu_options(optimized_path=optimized),
                                       providers=['CPUExecutionProvider'])
        counts = operator_counts(onnx.load(str(optimized)))
        del session
    gelu = sum(value for name, value in counts.items()
               if name.rsplit('::', 1)[-1] in ('Gelu', 'BiasGelu'))
    norm = sum(value for name, value in counts.items()
               if name.rsplit('::', 1)[-1] in ('LayerNormalization', 'SkipLayerNormalization'))
    require(gelu > 0 and norm > 0 and counts.get('Erf', 0) == 0,
            'ORT execution graph did not fuse exact GELU and LayerNorm')
    return dict(runtime=ort.__version__, graph_optimization='ORT_ENABLE_ALL', operators=counts,
                fused_gelu=gelu, fused_layernorm=norm, unfused_erf=0,
                gelu_approximation=False, host_specific_graph_deployed=False)
