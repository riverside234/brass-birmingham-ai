"""g3140 local-scoring auxiliary: era-separated heads with a per-entity trunk slice.

Replaces the 2.0.0 companion once, at a committed generation, with fresh heads
(src/brass_ai/learning/local_scoring_heads_v2.py). Differences from 2.0.0:

- Canal rows and Rail rows have their own owner/VP outputs, loss weights and
  report sums; a third pair predicts the game-end (Rail-end) map from Canal rows.
- VP targets are divided by 5 (was 20) and summed over the four seats (was a
  mean), with the owner term's weight.
- Head steps are kept when the value phase rejects an interval: the heads are
  not exported, so a head step is harmless when the model's step is undone.
  Only the trunk is restored. The warm-up counts every attempted PPG head step.
  With the g3318 ownership stop (recovery/ownership_stop.py) the heads instead keep
  only the 16-step intervals whose held-out owner CE did not increase, and keep
  fitting with the trunk frozen after the value phase's own stop.
- Training sums and the held-out check are reported by era and by open versus
  settled entities (settled: a flipped tile or a built link at the decision).
- The held-out check takes one learner row per game, era and quarter of the era
  from every held-out game, and reports an in-sample copy-the-board rule beside
  the heads. Report only; it never gates fitting.
"""
import copy
import hashlib
import math
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from brass_ai.learning.contracts import digest, keys, require
from brass_ai.learning.local_scoring_heads_v2 import CONTRACT as HEAD, EraScoringHeads, output_names
from . import local_scoring_targets as targets

CONTRACT = 'local-scoring-training/3.0.0'
PREVIOUS = 'local-scoring-training/2.0.0'
LADDER = (.05, .15, .5)
VP_SCALE = 5.
REPORT_CONTRACT = 'local-scoring-report/3.0.0'
ERAS = ('CanalEra', 'RailEra')


def default(generation, previous=None, *, weight=.05, rail_end=0., mode='apply'):
    """`previous` is the 2.0.0 config these heads replace; it is kept so the recipe can be traced back exactly."""
    if previous is None:
        from .local_scoring_auxiliary import DEFAULT, fresh_config
        previous = dict(fresh_config(DEFAULT), mode=mode)
    return dict(contract=CONTRACT, head_contract=HEAD, target=targets.identity(), mode=mode, vp_scale=VP_SCALE,
                weights=dict(canal=weight, rail=weight, rail_end=rail_end),
                optimizer=dict(kind='Adam', learning_rate=1e-4, weight_decay=0., grad_clip=1.),
                head_only_iterations=2, minimum_fitting_steps=32, ramp_iterations=5,
                replacement=dict(generation=generation, from_contract=PREVIOUS, previous=copy.deepcopy(previous)))


def is_v3(config): return type(config) is dict and config.get('contract') == CONTRACT


def head_identity():
    from brass_ai.learning import local_scoring_heads_v2
    return hashlib.sha256(Path(local_scoring_heads_v2.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def validate_config(config):
    keys(config, tuple(default(0)), ('rail_end_removed', 'rail_end_restored'))
    restored = 'rail_end_restored' in config
    if 'rail_end_removed' in config:  # g3528 drop-rail-end: the Rail-end outputs are gone from that generation on
        require(type(config['rail_end_removed']) is int and config['rail_end_removed'] >= config['replacement']['generation']
                and (restored or config['weights']['rail_end'] == 0), 'Invalid Rail-end head removal')
    if restored:  # g4770: fresh Rail-end outputs again from that generation on (training_revision_g4770)
        require('rail_end_removed' in config and type(config['rail_end_restored']) is int
                and config['rail_end_restored'] > config['rail_end_removed'], 'Invalid Rail-end head restoration')
    require(config['contract'] == CONTRACT and config['head_contract'] == HEAD and config['target'] == targets.identity()
            and config['mode'] in ('off', 'shadow', 'apply') and config['vp_scale'] == VP_SCALE,
            'Local scoring 3.0.0 contract differs')
    w = config['weights']; keys(w, ('canal', 'rail', 'rail_end'))
    require(all(type(v) in (int, float) and math.isfinite(v) for v in w.values()) and w['canal'] in LADDER
            and w['rail'] == w['canal'] and w['rail_end'] in (RESTORED_LADDER if restored else (0., w['canal'] / 2)),
            'Invalid local scoring 3.0.0 weights')
    o = config['optimizer']; keys(o, ('kind', 'learning_rate', 'weight_decay', 'grad_clip'))
    require(o == dict(kind='Adam', learning_rate=1e-4, weight_decay=0., grad_clip=1.), 'Invalid local scoring 3.0.0 optimizer')
    require((config['head_only_iterations'], config['minimum_fitting_steps'], config['ramp_iterations']) == (2, 32, 5),
            'Local scoring 3.0.0 warm-up differs')
    r = config['replacement']; keys(r, ('generation', 'from_contract', 'previous'))
    from .local_scoring_auxiliary import accepted_warmup, validate_config as validate_previous
    require(type(r['generation']) is int and r['generation'] > 0 and r['from_contract'] == PREVIOUS
            and type(r['previous']) is dict and accepted_warmup(r['previous']), 'Invalid local scoring 3.0.0 replacement')
    validate_previous(r['previous'])
    return config


def weights_forward(old, new):
    """Only up the ladder; the Rail-end target only switches on (at half the Canal weight, or from g4770 on
    the restored ladder .05 -> .25) and never moves down."""
    if old == new: return True
    if LADDER.index(new['canal']) < LADDER.index(old['canal']): return False
    return new['rail_end'] >= old['rail_end']


def config_forward(old, new):
    """A reviewed 3.0.0 -> 3.0.0 change: mode, the weight ladder, (g3528) the Rail-end heads' removal and
    (g4770) their restoration only."""
    if not (is_v3(old) and is_v3(new)): return False
    validate_config(old); validate_config(new)
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    a.pop('mode'); b.pop('mode'); wa, wb = a.pop('weights'), b.pop('weights')
    ra, rb = a.pop('rail_end_removed', None), b.pop('rail_end_removed', None)
    if ra is not None and ra != rb: return False  # a removal is kept
    ea, eb = a.pop('rail_end_restored', None), b.pop('rail_end_restored', None)
    if ea is not None and ea != eb: return False  # a restoration is kept
    if eb is not None and ea is None and ra is None: return False  # only removed heads are restored
    return a == b and weights_forward(wa, wb)


# ---- g3528 drop-rail-end: era heads without the Rail-end outputs ----------------------------------------

RAIL_END = ('rail_end_slots', 'rail_end_links')


def rail_end_removed(config): return is_v3(config) and 'rail_end_removed' in config


# ---- g4770: the Rail-end heads restored ----------------------------------------------------------------
# Fresh Rail-end owner and VP outputs (zero-initialised, as the original heads) on the trained shared, slice and
# hidden layers, trained only in PPG, in the same batches and steps as the Canal and Rail outputs. PPO neither
# snapshots nor preserves them, and the ownership stop reports their CE without deciding on it: a fresh head's
# falling CE would otherwise accept steps that cost the trained Canal and Rail outputs.

RESTORED_LADDER = (.05, .25)


def rail_end_restored(config): return is_v3(config) and 'rail_end_restored' in config


def two_era(config):
    """True when the heads have no Rail-end outputs (removed at g3528 and not restored)."""
    return rail_end_removed(config) and not rail_end_restored(config)


class TwoEraHeads(EraScoringHeads):
    """The era heads with the Rail-end owner and VP outputs removed. Built as the full heads, then those
    outputs dropped, so the canal and rail layers draw the same initial values."""
    def __init__(self, context_width=512):
        super().__init__(context_width)
        self.full_names = [n for n, _ in self.named_parameters()]  # the full heads' parameter order
        for name in RAIL_END:
            del self.owner[name]; del self.vp[name]

    def forward(self, encoded, descriptors, scale=1.):
        from brass_ai.learning.local_scoring_heads import scale_gradient
        from brass_ai.learning.local_scoring_heads_v2 import KINDS, LINKS, SLICE_WIDTH, SLOTS
        context = scale_gradient(torch.cat((encoded.context, descriptors.flatten(1)), -1), scale)
        entities = scale_gradient(encoded.targets[:, :SLOTS + LINKS], scale)
        shared = F.gelu(self.shared(context))
        slices = self.slices(shared).view(len(context), SLOTS + LINKS, SLICE_WIDTH)
        result = {}
        for kind, lo, hi in KINDS:
            hidden = F.gelu(self.hidden[kind](torch.cat((entities[:, lo:hi], slices[:, lo:hi]), -1)))
            for horizon in ('canal', 'rail'):
                name = f'{horizon}_{kind}'
                result[name] = (self.owner[name](hidden), self.vp[name](hidden))
        return result


def drop_rail_end(heads, optimizers, full_names, kept_names):
    """A full-heads checkpoint (state dict and the two Adams) without the Rail-end parameters, and a record."""
    from collections import OrderedDict as Ordered
    removed = [n for n in full_names if n not in kept_names]
    require(sorted(heads) == sorted(full_names) and all(n.split('.')[1] in RAIL_END for n in removed),
            'Era-head checkpoint layout differs')
    index = [full_names.index(n) for n in kept_names]
    result = {}
    for phase, state in optimizers.items():
        groups = state['param_groups']
        require(len(groups) == 1 and list(groups[0]['params']) == list(range(len(full_names))), 'Era-head Adam layout differs')
        result[phase] = dict(state={new: copy.deepcopy(state['state'][old]) for new, old in enumerate(index) if old in state['state']},
                             param_groups=[dict(copy.deepcopy(groups[0]), params=list(range(len(index))))])
    record = dict(contract='rail-end-head-removal/1.0.0', parameters=int(sum(heads[n].numel() for n in removed)),
                  max_abs=float(max(heads[n].detach().abs().max() for n in removed)))
    return Ordered((n, heads[n]) for n in kept_names), result, record


def add_rail_end(heads, optimizers, fresh, kept_names):
    """g4770: a two-era checkpoint (state dict and the two Adams) with the full heads' fresh Rail-end parameters
    (`fresh`, the full heads' state dict) added, and a record. The Rail-end slots start with no Adam state."""
    from collections import OrderedDict as Ordered
    full_names = list(fresh)
    added = [n for n in full_names if n not in kept_names]
    require(sorted(heads) == sorted(kept_names) and [n for n in full_names if n in kept_names] == list(kept_names)
            and all(n.split('.')[1] in RAIL_END for n in added), 'Era-head checkpoint layout differs')
    index = [full_names.index(n) for n in kept_names]
    result = {}
    for phase, state in optimizers.items():
        groups = state['param_groups']
        require(len(groups) == 1 and list(groups[0]['params']) == list(range(len(kept_names))), 'Era-head Adam layout differs')
        result[phase] = dict(state={index[old]: copy.deepcopy(value) for old, value in state['state'].items()},
                             param_groups=[dict(copy.deepcopy(groups[0]), params=list(range(len(full_names))))])
    record = dict(contract='rail-end-head-restoration/1.0.0', parameters=int(sum(fresh[n].numel() for n in added)),
                  max_abs=float(max(fresh[n].detach().abs().max() for n in added)))
    return Ordered((n, heads[n] if n in heads else fresh[n]) for n in full_names), result, record


class StateColumns:
    """Indices into the encoded state for the board as seen at the decision."""
    def __init__(self):
        from brass_ai.learning.board_topology import topology
        from brass_ai.learning.contracts import CONFIG_DIR, FEATURE_SCHEMA, read_json
        from brass_ai.learning.state_layout import state_layout
        schema = read_json(CONFIG_DIR / FEATURE_SCHEMA); board = topology()
        require(schema['slots'] == board['slot_order'] and schema['links'] == board['link_order'],
                'Local scoring entity order differs from the encoder')
        segments = {s.name: s for s in state_layout().segments}
        def span(name, size):
            s = segments[name]; require(s.stop - s.start == size, 'Unexpected local scoring state segment')
            return list(range(s.start, s.stop))
        self.slot_owner = [i for name in schema['slots'] for i in span('slot.' + name + '.owner', 4)]
        tile = segments['slot.' + schema['slots'][0] + '.tile']
        flipped = list(tile.components).index('flipped'); vp = list(tile.components).index('vp')
        self.slot_flipped = [segments['slot.' + name + '.tile'].start + flipped for name in schema['slots']]
        self.slot_vp = [segments['slot.' + name + '.tile'].start + vp for name in schema['slots']]
        self.vp_scale = float(schema['board_tile_fields']['vp'])
        self.link_owner = [i for name in schema['links'] for i in span('link.' + name, 4)]
        self.devices = {}

    def tensors(self, device):
        key = str(device)
        if key not in self.devices:
            self.devices[key] = {k: torch.tensor(getattr(self, k), dtype=torch.int64, device=device)
                                 for k in ('slot_owner', 'slot_flipped', 'slot_vp', 'link_owner')}
        return self.devices[key]

    def board(self, states):
        """Current relative owner class (0 none, 1..4) and settled mask per entity, for slots and links."""
        t = self.tensors(states.device); n = len(states)
        def owners(index, count):
            onehot = states.index_select(1, index).view(n, count, 4)
            present = onehot.sum(-1) > .5
            return torch.where(present, onehot.argmax(-1) + 1, torch.zeros_like(present, dtype=torch.int64)), present
        slot_class, slot_present = owners(t['slot_owner'], 49)
        link_class, link_present = owners(t['link_owner'], 69)
        flipped = states.index_select(1, t['slot_flipped']) > .5
        printed = states.index_select(1, t['slot_vp']).float() * self.vp_scale
        return dict(slots=(slot_class, slot_present & flipped, printed), links=(link_class, link_present, None))


_COLUMNS = None


def columns():
    global _COLUMNS
    if _COLUMNS is None: _COLUMNS = StateColumns()
    return _COLUMNS


def end_map(envelope):
    """The game-end (Rail-era) scoring map of a sealed envelope, if the game reached it."""
    return next((e for e in envelope['receipt']['eras'] if e['era'] == 'RailEra'), None)


def envelopes(ds, ids):
    if hasattr(ds, 'scoring_envelopes'): return ds.scoring_envelopes(ids)
    result = []
    for i in ids:
        value = ds.rows[i].get('_local_scoring')
        result.append(None if value is None else value['envelope'])
    return result


def make(model, config, *, raw=None, generation=0, adopting=False):
    from .local_scoring_auxiliary import Companion
    return Companion(model, config, raw=raw, generation=generation, adopting=adopting)


def build(base):
    """CompanionV3 subclasses the 2.0.0 Companion so every call site stays the same."""

    class CompanionV3(base):
        def __init__(self, model, config, *, raw=None, generation=0, adopting=False):
            from . import local_scoring_warmup as warmup
            from .local_scoring_auxiliary import KEY, bundle_id
            self.config = copy.deepcopy(validate_config(config)); self.generation = generation
            saved = (raw or {}).get('training', {}).get(KEY)
            replacement = self.config['replacement']; self.replacement = None
            if saved is not None and saved.get('contract') == CONTRACT:
                origin = copy.deepcopy(saved['origin'])
            else:
                require(generation == replacement['generation'] and (saved is None or saved.get('contract') == replacement['from_contract']),
                        'Local scoring 3.0.0 heads replace the 2.0.0 heads only at their committed generation')
                origin = dict(generation=generation, model_id=(raw or {}).get('model_id', 'fresh'))
                if saved is not None:
                    require(saved['identity'] == bundle_id({k: v for k, v in saved.items() if k != 'identity'}),
                            'Corrupt replaced companion bundle identity')
                    self.replacement = dict(contract='local-scoring-head-replacement/1.0.0', generation=generation,
                        previous=dict(contract=saved['contract'], identity=saved['identity'], age=saved['age'],
                                      retained_steps=copy.deepcopy(saved['retained_steps']),
                                      retained_samples=copy.deepcopy(saved['retained_samples'])),
                        scope='Fresh era heads with a per-entity trunk slice; 2.0.0 heads and Adams retired; '
                              'the trunk ramps in again through the 2.0.0 warm-up rule.')
                saved = None
            self.origin = origin; self.age = generation - origin['generation']
            require(self.age >= 0, 'Auxiliary activation origin is in the future')
            self.seed = int(digest(dict(origin=origin, head=HEAD))[:15], 16)
            removed = two_era(self.config); self.rail_end_removal = None; self.rail_end_restoration = None
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(self.seed); self.heads = (TwoEraHeads if removed else EraScoringHeads)(model.config.hidden)
            self.heads.to(next(model.parameters()).device)
            lr = self.config['optimizer']['learning_rate']
            self.optimizers = {p: torch.optim.Adam(self.heads.parameters(), lr=lr, weight_decay=0.) for p in ('ppo', 'ppg')}
            self.retained_steps = dict(ppo=0, ppg=0); self.retained_samples = dict(rows=0, games=0, era_maps=0)
            self.work_warmup = True
            self.warmup = dict(warmup.initial(), generation=generation)
            if saved is not None:
                keys(saved, ('contract', 'config', 'config_sha256', 'head_sha256', 'origin', 'seed', 'generation', 'age', 'heads',
                             'optimizers', 'retained_steps', 'retained_samples', 'warmup', 'replacement', 'identity'))
                require(saved['identity'] == bundle_id({k: v for k, v in saved.items() if k != 'identity'}), 'Corrupt companion bundle identity')
                require(saved['config_sha256'] == digest(saved['config']) and saved['head_sha256'] == head_identity(),
                        'Companion architecture/config hash differs')
                require(config_forward(saved['config'], self.config) and saved['seed'] == self.seed and
                        saved['generation'] == generation and saved['age'] == self.age, 'Companion checkpoint/config/progress differs')
                heads, optimizers = saved['heads'], saved['optimizers']
                if rail_end_restored(self.config) and two_era(saved['config']):
                    # g4770, once at its generation: the two-era checkpoint gains fresh Rail-end outputs (zero, as the
                    # heads' own initialisation) with no Adam state; every other parameter and Adam slot carries over.
                    require(self.config['rail_end_restored'] == generation,
                            'The Rail-end heads are restored at their committed generation')
                    kept = [n for n in self.heads.state_dict() if n.split('.')[1] not in RAIL_END]
                    heads, optimizers, self.rail_end_restoration = add_rail_end(heads, optimizers, self.heads.state_dict(), kept)
                    self.rail_end_restoration['generation'] = generation
                if removed and not rail_end_removed(saved['config']):
                    # g3528 drop-rail-end, once at its generation: the full heads' checkpoint loses the Rail-end
                    # outputs, which trained at weight 0 (no loss or preservation term reached them).
                    require(self.config['rail_end_removed'] == generation and saved['config']['weights']['rail_end'] == 0,
                            'The Rail-end heads are removed untrained, at their committed generation')
                    heads, optimizers, self.rail_end_removal = drop_rail_end(
                        heads, optimizers, self.heads.full_names, [n for n, _ in self.heads.named_parameters()])
                    self.rail_end_removal['generation'] = generation
                self.heads.load_state_dict(heads, strict=True)
                for p, o in self.optimizers.items(): o.load_state_dict(optimizers[p])
                self.retained_steps = copy.deepcopy(saved['retained_steps'])
                self.retained_samples = copy.deepcopy(saved['retained_samples'])
                self.warmup = copy.deepcopy(warmup.validate(saved['warmup'], self.config, generation))
                require(self.warmup['fitting_steps'] == self.retained_steps['ppg'], 'Auxiliary fitting counters differ from kept head steps')
                self.replacement_record = copy.deepcopy(saved['replacement'])
            else:
                warmup.validate(self.warmup, self.config, generation)
                self.replacement_record = copy.deepcopy(self.replacement) if self.replacement else dict(
                    contract='local-scoring-head-replacement/1.0.0', generation=generation, previous=None)
            for phase, o in self.optimizers.items():
                require(len(o.param_groups) == 1 and o.param_groups[0]['weight_decay'] == 0 and math.isfinite(o.param_groups[0]['lr'])
                        and 0 < o.param_groups[0]['lr'] <= lr * (1 + 1e-12), 'Invalid companion Adam settings')
            self.retained = set(); self.reset_report()
            self.active = self.config['mode'] != 'off'; self.pending = False
            self.target_cache = OrderedDict(); self.board = columns()

        # ---- weights and outputs -------------------------------------------------------------

        def horizon_weight(self, horizon):
            return float(self.config['weights'][horizon])

        def preservation_weights(self):
            # g4770: restored Rail-end outputs train only in PPG; PPO neither snapshots nor preserves them.
            ppg_only = RAIL_END if rail_end_restored(self.config) else ()
            return {name: (self.horizon_weight(name.rsplit('_', 1)[0]),) * 2 for name in output_names()
                    if name in self.heads.owner and name not in ppg_only}

        @property
        def rail_end_decides_stop(self):
            """g4770: a restored Rail-end head's CE is reported by the ownership stop but does not decide it."""
            return not rail_end_restored(self.config)

        @property
        def scale(self):
            if self.config['mode'] != 'apply': return 0.
            from .local_scoring_warmup import scale
            return scale(self.warmup, self.config)

        def reset_report(self):
            super().reset_report()
            self.era_stats = {}; self.search_stats = {}  # search_stats: g3423 search-table rows, kept apart
            self.stopped = False; self.stop_record = None  # ownership-stop: set per value phase

        def era_accumulate(self, key, values):
            if key not in self.era_stats: self.era_stats[key] = torch.zeros(len(values), dtype=torch.float64, device=values[0].device)
            self.era_stats[key].add_(torch.stack([v.detach().double() for v in values]))

        def target_arrays(self, metadata, kind, count):
            unique = {}; values = []; inverse = []
            for target, key, _, mapping in metadata:
                token = (key, kind)
                if token not in unique:
                    unique[token] = len(values)
                    if target is None:
                        value = (np.zeros(count, np.int64), np.zeros((count, 4), np.float32), np.zeros(count, np.bool_))
                    else:
                        value = self.target_cache.get(token)
                        if value is None:
                            stable = np.array([r['owner'] - 1 for r in target[kind]])
                            relative = mapping[stable.clip(0)]; present = stable >= 0
                            cls = np.where(present, relative + 1, 0); vp = np.zeros((count, 4), np.float32)
                            vp[np.arange(count)[present], relative[present]] = np.array([r['vp'] for r in target[kind]])[present] / VP_SCALE
                            value = (cls, vp, np.array([r['valid'] for r in target[kind]], np.bool_))
                            self.target_cache[token] = value; self.stats['target_cache_misses'] += 1
                            if len(self.target_cache) > 65536: self.target_cache.popitem(last=False)
                        else: self.stats['target_cache_hits'] += 1
                        self.target_cache.move_to_end(token)
                    values.append(value)
                inverse.append(unique[token])
            return tuple(np.stack([v[j] for v in values])[inverse] for j in range(3))

        def end_metadata(self, ds, ids, metadata):
            result = []
            for (target, key, token, mapping), envelope in zip(metadata, envelopes(ds, ids)):
                end = end_map(envelope) if target is not None and envelope is not None and target['era'] == 'CanalEra' else None
                result.append((end, (key[0], 'RailEra', key[2], key[3]) if end is not None else None, token, mapping))
            return result

        def _terms(self, owner, vp, classes, values, mask, device, correction=None):
            """`correction` (g3471 search-rows): [B, n, 9] share-scaled lambda-return corrections of the owner
            probabilities and local VP; the loss then uses the blended target, while the returned CE and VP error
            (the by-era report) stay those against the actual map."""
            m = torch.as_tensor(mask, device=device); cls = torch.as_tensor(classes, device=device)
            truth = torch.as_tensor(values, device=device)
            ce = F.cross_entropy(owner.float().transpose(1, 2), cls, reduction='none')
            error = F.smooth_l1_loss(vp.float(), truth, reduction='none').sum(-1)
            den = m.sum(-1).clamp_min(1)
            if correction is None:
                return m, cls, truth, ce, error, ((ce + error) * m).sum(-1) / den
            target = (F.one_hot(cls, 5).float() + correction[..., :5]).clamp_min(0)
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
            soft = -(target * F.log_softmax(owner.float(), -1)).sum(-1)
            soft_error = F.smooth_l1_loss(vp.float(), truth + correction[..., 5:], reduction='none').sum(-1)
            return m, cls, truth, ce, error, ((soft + soft_error) * m).sum(-1) / den

        def forward(self, model, inputs, ds, ix, *, dropout=None, detach=False, capture_encoded=None):
            from .updates import value_output
            ids = ix.detach().cpu().tolist() if isinstance(ix, torch.Tensor) else list(ix)
            # A stopped ownership head (training_revision_g3140 ownership-stop) adds no loss for the rest of the phase.
            if not self.active or self.stopped:
                return value_output(model, inputs, ds, ix, dropout=dropout, detach=detach, capture_encoded=capture_encoded), None
            metadata = self.metadata(ds, ids)
            if not any(x[0] is not None for x in metadata):
                return value_output(model, inputs, ds, ix, dropout=dropout, detach=detach, capture_encoded=capture_encoded), None
            require(self.phase == 'ppg', 'Fresh auxiliary target fitting is restricted to PPG')
            from brass_ai.learning.seat_descriptors import critic_inputs
            start = time.perf_counter()
            with torch.set_grad_enabled(torch.is_grad_enabled() and not detach): encoded = model.encode_state(inputs[0])
            if detach:
                from .ppg_phasic import stop_value_gradient
                encoded = stop_value_gradient(encoded)
            embeddings = model.descriptor_embeddings(encoded, critic_inputs(ds, ids, dropout=dropout))
            if capture_encoded is not None: capture_encoded(encoded)
            heads = model.official_heads(encoded, inputs[3], embeddings)
            out = (inputs[1].new_zeros(inputs[1].shape[:2]), *heads)
            if hasattr(model, 'predict_utility'):
                out += (model.predict_utility(encoded.context, inputs[3], heads, None if ds.remaining is None else ds.remaining[ix], embeddings),)
            loss = self._auxiliary(encoded, embeddings, inputs, ds, ix, ids, metadata)
            self.stats['forward_seconds'] += time.perf_counter() - start
            return out, loss

        def head_loss(self, model, inputs, ds, ix, *, dropout=None):
            """The same auxiliary loss with the trunk frozen: only the heads get gradients (ownership-stop)."""
            ids = ix.detach().cpu().tolist() if isinstance(ix, torch.Tensor) else list(ix)
            if not self.active or self.stopped: return None
            metadata = self.metadata(ds, ids)
            if not any(x[0] is not None for x in metadata): return None
            require(self.phase == 'ppg', 'Fresh auxiliary target fitting is restricted to PPG')
            from brass_ai.learning.seat_descriptors import critic_inputs
            start = time.perf_counter()
            with torch.no_grad():
                encoded = model.encode_state(inputs[0])
                embeddings = model.descriptor_embeddings(encoded, critic_inputs(ds, ids, dropout=dropout))
            loss = self._auxiliary(encoded, embeddings, inputs, ds, ix, ids, metadata)
            self.stats['head_only_seconds'] += time.perf_counter() - start
            return loss

        def search_loss(self, model, ds, ids, *, frozen=False):
            """Per-row auxiliary loss of search-table roots (g3423 search-aux): the era heads' ownership and
            local-VP terms (and Rail-end when on) only, no value terms; trunk scale as for ordinary rows.
            ``frozen``: a heads-only step, so the trunk gets no gradient (as head_loss).

            ``ds`` is a search_auxiliary.SearchRows. Its rows are counted apart: the ordinary attempted,
            retained and fitting counters, map support and era sums stay those of the ordinary rows.
            """
            if not self.active or self.stopped: return None
            require(self.phase == 'ppg', 'Fresh auxiliary target fitting is restricted to PPG')
            from brass_ai.learning.seat_descriptors import critic_inputs
            start = time.perf_counter()
            index = torch.as_tensor(ids, device=ds.device); states = ds.states[index]
            with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
                encoded = model.encode_state(states)
                embeddings = model.descriptor_embeddings(encoded, critic_inputs(ds, ids))  # no descriptors: unknown seats
            kept = (self.era_stats, self.batch_fitting, self.batch_used, self.attempted, self.map_support, self.map_masked,
                    self.pending)
            self.era_stats, self.batch_fitting, self.batch_used = self.search_stats, set(), set()
            self.attempted, self.map_support, self.map_masked = set(), {}, {}
            try:
                loss = self._auxiliary(encoded, embeddings, (states,), ds, index, ids, ds.scoring_metadata(ids))
                used = bool(self.batch_used)
            finally:
                self.search_stats = self.era_stats
                (self.era_stats, self.batch_fitting, self.batch_used, self.attempted, self.map_support, self.map_masked,
                 pending) = kept
            self.pending = pending or used
            self.stats['search_seconds'] += time.perf_counter() - start
            return loss

        def search_report(self):
            """Era sums of the search rows' attempted presentations, as the ordinary `by_era`."""
            return self._by_era(self.search_stats)

        def _by_era(self, stats):
            names = list(stats)
            totals = dict(zip(names, torch.stack([stats[k] for k in names]).cpu().tolist())) if names else {}
            by_era = {}
            for (era, kind, state), (count, ce, none, vp) in totals.items():
                by_era.setdefault(era, {}).setdefault(kind, {})[state] = dict(entities=int(count),
                    owner_ce=ce / count if count else None, none_share=none / count if count else None,
                    vp_abs_error_per_entity=vp / count if count else None)
            return by_era

        def _auxiliary(self, encoded, embeddings, inputs, ds, ix, ids, metadata):
            predictions = self.heads(encoded, embeddings, self.scale)
            device = inputs[0].device
            canal = torch.tensor([t is not None and t['era'] == 'CanalEra' for t, _, _, _ in metadata], device=device)
            present = torch.tensor([t is not None for t, _, _, _ in metadata], device=device)
            board = self.board.board(inputs[0])
            loss = torch.zeros(len(ids), device=device, dtype=torch.float32)
            wc, wr, we = (self.horizon_weight(h) for h in ('canal', 'rail', 'rail_end'))
            weight = torch.where(canal, torch.full_like(loss, wc), torch.full_like(loss, wr)) * present
            end_meta = self.end_metadata(ds, ids, metadata) if we > 0 else None
            # g3471 search-rows: lambda-return corrections of the own-era targets (ordinary rows .5, search rows 1);
            # g3528: channels 9..17, when present, the Canal rows' Rail-end corrections.
            source = getattr(ds, 'local_bootstrap', None)
            corrections = source(ix) if source is not None else None
            ends = corrections[..., 9:] if corrections is not None and corrections.shape[-1] > 9 else None
            if corrections is not None: corrections = corrections[..., :9]
            for kind in ('slots', 'links'):
                (co, cv), (ro, rv) = predictions['canal_' + kind], predictions['rail_' + kind]
                owner = torch.where(canal[:, None, None], co, ro); vp = torch.where(canal[:, None, None], cv, rv)
                classes, values, mask = self.target_arrays(metadata, kind, owner.shape[1])
                entities = slice(0, 49) if kind == 'slots' else slice(49, None)
                part = None if corrections is None else corrections[:, entities]
                m, cls, truth, ce, error, term = self._terms(owner, vp, classes, values, mask, device, part)
                loss = loss + weight * term
                current, settled, _ = board[kind]
                for era_index, era in enumerate(ERAS):
                    rows = (canal if era_index == 0 else (~canal & present))[:, None]
                    for state, selected in (('open', ~settled), ('settled', settled)):
                        s = m & rows & selected
                        self.era_accumulate((era, kind, state), (s.sum(), (ce.detach() * s).sum(), ((cls == 0) & s).sum(),
                            ((vp.detach().float() - truth).abs().sum(-1) * s).sum() * VP_SCALE))
                if end_meta is not None:
                    eo, ev = predictions['rail_end_' + kind]
                    classes, values, mask = self.target_arrays(end_meta, kind, eo.shape[1])
                    rows = torch.tensor([t is not None for t, _, _, _ in end_meta], device=device)
                    m, cls, truth, ce, error, term = self._terms(eo, ev, classes, values, mask, device,
                                                                 None if ends is None else ends[:, entities])
                    loss = loss + we * rows * term
                    s = m & rows[:, None]
                    self.era_accumulate(('CanalEra->RailEnd', kind, 'all'), (s.sum(), (ce.detach() * s).sum(), ((cls == 0) & s).sum(),
                        ((ev.detach().float() - truth).abs().sum(-1) * s).sum() * VP_SCALE))
            for (target, _, token, _), w in zip(metadata, ds.critic_weights[ix].detach().cpu().tolist()):
                if target is None: continue
                game = token[0]
                valid = any(any(r['valid'] for r in target[k]) for k in ('slots', 'links')) and w > 0
                if valid:
                    self.batch_fitting.add(token); self.batch_used.add(token); self.attempted.add(token)
                if (game, target['era']) not in self.map_support:
                    self.map_support[(game, target['era'])] = sum(int(r['valid']) for k in ('slots', 'links') for r in target[k])
                    self.map_masked[(game, target['era'])] = sum(int(not r['valid']) for k in ('slots', 'links') for r in target[k])
            self.pending = bool(self.batch_used)
            return loss

        # ---- head steps survive a rejected interval --------------------------------------------
        # The value phase never restores the heads; without ownership-stop they keep every
        # attempted step, with it the stop's own held-out CE decides (stop_snapshot/stop_restore).

        def snapshot(self):
            return dict(kept=True)

        def restore(self, saved):
            require(saved == dict(kept=True), 'Local scoring 3.0.0 snapshot differs')
            self.optimizers['ppg'].zero_grad(set_to_none=True); self.pending = False; self.restorations += 1

        def stop_snapshot(self):
            """Heads, PPG Adam and the kept-work counters at an ownership-stop boundary."""
            return dict(heads={k: v.detach().clone() for k, v in self.heads.state_dict().items()},
                        adam=copy.deepcopy(self.optimizers['ppg'].state_dict()),
                        retained=set(self.retained), retained_steps=dict(self.retained_steps), steps=dict(self.steps),
                        fitting_steps=self.fitting_steps, fitting_used=set(self.fitting_used))

        def stop_restore(self, saved):
            """Undo every head step since `saved`: they no longer count as kept or fitting work."""
            self.heads.load_state_dict(saved['heads']); self.optimizers['ppg'].load_state_dict(saved['adam'])
            self.retained = set(saved['retained']); self.retained_steps = dict(saved['retained_steps'])
            self.steps = dict(saved['steps']); self.fitting_steps = saved['fitting_steps']
            self.fitting_used = set(saved['fitting_used'])
            self.optimizers['ppg'].zero_grad(set_to_none=True); self.pending = False
            self.restorations += 1; self.rollbacks += 1

        # ---- checkpoint ---------------------------------------------------------------------

        def payload(self, generation):
            from .local_scoring_auxiliary import bundle_id
            require(generation in (self.generation, self.generation + 1), 'Noncommitted companion progress jump')
            advancing = generation == self.generation + 1
            result = dict(contract=CONTRACT, config=self.config, config_sha256=digest(self.config), head_sha256=head_identity(),
                origin=self.origin, seed=self.seed, generation=generation, age=generation - self.origin['generation'],
                heads=self.heads.state_dict(), optimizers={p: o.state_dict() for p, o in self.optimizers.items()},
                retained_steps=self.retained_steps,
                retained_samples={k: v + (self.usage()[k] if advancing else 0) for k, v in self.retained_samples.items()},
                warmup=self.next_warmup() if advancing else copy.deepcopy(self.warmup), replacement=self.replacement_record)
            require(result['warmup']['fitting_steps'] == self.retained_steps['ppg'],
                    'Auxiliary warm-up must account for exactly the kept head steps')
            return dict(result, identity=bundle_id(result))

        def committed(self, generation):
            super().committed(generation)
            self.replacement = None

        # ---- reports ------------------------------------------------------------------------

        def report(self):
            result = super().report()
            names = list(self.era_stats)
            totals = dict(zip(names, torch.stack([self.era_stats[k] for k in names]).cpu().tolist())) if names else {}
            by_era = {}
            for (era, kind, state), (count, ce, none, vp) in totals.items():
                by_era.setdefault(era, {}).setdefault(kind, {})[state] = dict(entities=int(count),
                    owner_ce=ce / count if count else None, none_share=none / count if count else None,
                    vp_abs_error_per_entity=vp / count if count else None)
            stop = self.stop_record
            result.update(contract=CONTRACT, report_contract=REPORT_CONTRACT, weights=copy.deepcopy(self.config['weights']),
                vp_scale=VP_SCALE, by_era=by_era, replacement=copy.deepcopy(self.replacement),
                head_steps_kept_on_rejection=stop is None,
                by_era_scope=('Attempted PPG presentations (' + ('the ownership stop decides which head steps stay'
                              if stop is not None else 'heads keep rejected steps') + '). VP error is summed over the four seats, in VP. '
                             'Settled: a flipped tile or built link at the decision; open: everything else.'))
            if stop is not None: result['ownership_stop'] = copy.deepcopy(stop)
            if rail_end_removed(self.config):
                result['rail_end_heads'] = dict(removed_at=self.config['rail_end_removed'], migration=copy.deepcopy(self.rail_end_removal))
            if rail_end_restored(self.config):
                result['rail_end_heads'].update(restored_at=self.config['rail_end_restored'], weight=self.horizon_weight('rail_end'),
                    restoration=copy.deepcopy(self.rail_end_restoration),
                    scope='Restored Rail-end outputs: PPG only (same batches and steps as Canal/Rail), no PPO snapshot '
                          'or preservation; PPO-game Canal rows on the actual game-end map, search rows on lambda '
                          'returns after the warm-up; their CE is reported by the ownership stop, not decisive.')
            return result

        def diagnostics(self, model, dataset, heldout, *, maximum=None):
            return heldout_check(self, model, dataset, heldout)

    return CompanionV3


def _quarter(context):
    return min(3, max(0, (int(context['round']) - 1) // 2))


def heldout_check(companion, model, dataset, heldout):
    """One learner row per game, era and quarter of the era, from every held-out game."""
    from brass_ai.learning.seat_descriptors import critic_inputs
    began = time.perf_counter(); chosen = {}
    for i in heldout:
        row = dataset.rows[i]
        if row.get('_branch') or dataset.roles[i] != 'learner' or targets.map_for(row) is None: continue
        context = row['view']['context']; game = dataset.data[i][2][0]
        group = (game, context['era'], _quarter(context))
        rank = hashlib.sha256(repr((REPORT_CONTRACT, group, int(context.get('revision', i)), int(row['seat']))).encode()).digest()
        if group not in chosen or rank < chosen[group][0]: chosen[group] = (rank, i)
    ids = [i for _, (_, i) in sorted(chosen.items())]
    cells = defaultdict(lambda: defaultdict(float)); reference = defaultdict(lambda: defaultdict(int))
    was_training = model.training; heads_training = companion.heads.training
    model.eval(); companion.heads.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(ids), 64):
                part = ids[start:start + 64]; ix, inputs, _ = dataset.value_batch(part, model)
                encoded = model.encode_state(inputs[0])
                embeddings = model.descriptor_embeddings(encoded, critic_inputs(dataset, part))
                predictions = companion.heads(encoded, embeddings, 0.)
                board = companion.board.board(inputs[0])
                metadata = companion.metadata(dataset, part)
                for kind in ('slots', 'links'):
                    current, settled, printed = (x.cpu().numpy() if x is not None else None for x in board[kind])
                    outputs = {h: (predictions[h + '_' + kind][0].float().softmax(-1).cpu().numpy(),
                                   predictions[h + '_' + kind][1].float().cpu().numpy() * VP_SCALE) for h in ('canal', 'rail')}
                    classes, values, mask = companion.target_arrays(metadata, kind, outputs['canal'][0].shape[1])
                    for b, i in enumerate(part):
                        context = dataset.rows[i]['view']['context']; era = context['era']; q = _quarter(context)
                        probs, vp = outputs['canal' if era == 'CanalEra' else 'rail']
                        for e in np.flatnonzero(mask[b]):
                            state = 'settled' if settled[b, e] else 'open'; c = int(classes[b, e])
                            cell = cells[(era, kind, state)]
                            cell['entities'] += 1; cell['owner_ce'] += -math.log(max(float(probs[b, e, c]), 1e-30))
                            reference[(era, kind, state, q, int(current[b, e]))][c] += 1
                            if c:
                                truth = float(values[b, e, c - 1]) * VP_SCALE
                                cell['scoring'] += 1; cell['wrong_seat'] += int(int(probs[b, e].argmax()) != c)
                                cell['vp_error'] += abs(float(vp[b, e, c - 1]) - truth); cell['vp_zero_error'] += truth
                                if kind == 'slots' and settled[b, e]:
                                    copy_vp = float(printed[b, e]) if int(current[b, e]) == c else 0.
                                    cell['copy_vp_error'] += abs(copy_vp - truth); cell['copy_count'] += 1
                            else:
                                cell['none'] += 1
    finally:
        model.train(was_training); companion.heads.train(heads_training)
    entropy = defaultdict(float)
    for (era, kind, state, _, _), counts in reference.items():
        total = sum(counts.values())
        entropy[(era, kind, state)] += sum(-n * math.log(n / total) for n in counts.values())
    groups = {}
    for (era, kind, state), c in sorted(cells.items()):
        n, s = c['entities'], c['scoring']
        groups[f'{era}/{kind}/{state}'] = dict(entities=int(n), none_share=c['none'] / n if n else None,
            owner_ce=c['owner_ce'] / n if n else None, reference_owner_ce=entropy[(era, kind, state)] / n if n else None,
            scoring_entities=int(s), wrong_seat_rate=c['wrong_seat'] / s if s else None,
            vp_error_scoring_seat=c['vp_error'] / s if s else None, vp_error_always_zero=c['vp_zero_error'] / s if s else None,
            copy_vp_error=c['copy_vp_error'] / c['copy_count'] if c['copy_count'] else None)
    return dict(contract=REPORT_CONTRACT, rows=len(ids), games=len({dataset.data[i][2][0] for i in ids}), groups=groups,
        seconds=time.perf_counter() - began,
        scope='One learner row per held-out game, era and quarter (rounds 1-2, 3-4, 5-6, 7-8). Settled: a flipped tile or built link at '
              'the decision. reference_owner_ce: in-sample copy-the-board rule (base rate given era, kind, state, quarter and the '
              'current owner), slightly optimistic. VP errors are in VP on the seat that scores the entity; copy_vp_error credits '
              'the printed VP of a flipped tile to its current owner. Report only; no gate or fitting change.')
