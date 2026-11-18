"""g3318 ownership stop: the era heads' own held-out cross-entropy early stop.

Adopted with `--value-revision ownership-stop` (training_revision_g3140). Without it the
3.0.0 heads keep every attempted PPG step, and the value phase's utility-loss stop is the
only thing that ends their fitting, so they always kept the first 16 steps.

With it, at every 16-step boundary of the value phase:

1. the value phase first decides the trunk as before (kept, or restored on a held-out
   value-loss increase);
2. the stop then scores the owner cross-entropy of the resulting state on the value
   phase's validation games (every held-out row with a scoring map, critic weights);
3. if that CE is above the last kept boundary's, the heads and their PPG Adam return
   to that boundary (the interval keeps 0 head steps) and head fitting ends for this
   phase: the auxiliary loss no longer reaches the trunk either.

When the value phase's own stop ends the trunk's phase first, the heads keep fitting on
the remaining planned batches with the trunk frozen, still checked every 16 steps, so
the utility stop no longer limits them. Head steps never exceed the phase's planned
steps. The criterion is the owner-CE part of the training loss with its era weights
(divided by the Canal weight, so it reads as CE per row: slots mean plus links mean).
"""
import time

import torch
from torch.nn import functional as F

from brass_ai.learning.contracts import require

CONTRACT = 'ownership-ce-early-stop/1.0.0'


def era_companion(companion):
    """The 3.0.0 local-scoring companion inside `companion` (the DOMiNO wrapper holds it as .local)."""
    from .local_scoring_v3 import CONTRACT as V3
    local = getattr(companion, 'local', companion)
    require(local is not None and getattr(local, 'config', {}).get('contract') == V3,
            'The ownership stop needs the 3.0.0 era heads')
    return local


class OwnershipStop:
    def __init__(self, companion, settings, model, dataset, heldout_ids, update):
        from .value_phase import value_parts
        require(settings is not None and settings.get('contract') == CONTRACT, 'Unknown ownership stop settings')
        self.local = era_companion(companion); self.settings = dict(settings)
        self.interval = settings['interval_steps']; self.model = model; self.dataset = dataset
        began = time.perf_counter()
        local = self.local; local.stopped = False
        hits, misses = local.stats.get('target_cache_hits', 0.), local.stats.get('target_cache_misses', 0.)
        found = {}
        ids = list(heldout_ids)
        for start in range(0, len(ids), 4096):
            part = ids[start:start + 4096]
            found.update((i, m) for i, m in zip(part, local.metadata(dataset, part)) if m[0] is not None)
        chosen = [i for i in ids if i in found]
        self.parts = [list(p) for p in value_parts(dataset, chosen, update, model)]
        device = dataset.device; we = local.horizon_weight('rail_end'); self.cache = []
        games = {m[2][0] for m in found.values()}
        for part in self.parts:
            metadata = [found[i] for i in part]
            entry = dict(part=part, index=torch.as_tensor(part, device=device),
                         canal=torch.tensor([m[0]['era'] == 'CanalEra' for m in metadata], device=device))
            for kind, count in (('slots', 49), ('links', 69)):
                classes, _, mask = local.target_arrays(metadata, kind, count)
                entry[kind] = (torch.as_tensor(classes, device=device), torch.as_tensor(mask, device=device))
            if we > 0:
                end = local.end_metadata(dataset, part, metadata)
                entry['end_rows'] = torch.tensor([m[0] is not None for m in end], device=device)
                for kind, count in (('slots', 49), ('links', 69)):
                    classes, _, mask = local.target_arrays(end, kind, count)
                    entry['end_' + kind] = (torch.as_tensor(classes, device=device), torch.as_tensor(mask, device=device))
            self.cache.append(entry)
        local.stats['target_cache_hits'], local.stats['target_cache_misses'] = hits, misses  # keep training counters
        self.rows = len(chosen); self.games = len(games)
        self.running = bool(chosen); self.reason = None if chosen else 'no_heldout_targets'
        self.intervals = []; self.measure_seconds = 0.; self.head_only_seconds = 0.; self.head_only_steps = 0
        self.attempted = 0; self.kept = 0; self.last_step = 0
        self.precompute_seconds = time.perf_counter() - began
        self.before = self.current = self.measure() if chosen else None
        self.saved = local.stop_snapshot()
        if not chosen: local.stopped = True

    # ---- the criterion --------------------------------------------------------------------

    def measure(self):
        """Critic-weighted mean over held-out rows of the training loss's owner-CE part, per Canal weight."""
        from brass_ai.learning.seat_descriptors import critic_inputs
        began = time.perf_counter(); local, model, ds = self.local, self.model, self.dataset
        wc, wr, we = (local.horizon_weight(h) for h in ('canal', 'rail', 'rail_end'))
        sums = None
        was, heads_was = model.training, local.heads.training
        model.eval(); local.heads.eval()
        try:
            with torch.no_grad():
                for entry, (part, ix, inputs) in zip(self.cache, ds.value_batches(iter(self.parts), model)):
                    require(list(part) == entry['part'], 'Ownership stop rows changed order')
                    encoded = model.encode_state(inputs[0])
                    predictions = local.heads(encoded, model.descriptor_embeddings(encoded, critic_inputs(ds, part)), 0.)
                    canal = entry['canal']; era = torch.zeros(len(part), dtype=torch.float64, device=canal.device)
                    end = torch.zeros_like(era)
                    for kind in ('slots', 'links'):
                        owner = torch.where(canal[:, None, None], predictions['canal_' + kind][0], predictions['rail_' + kind][0])
                        era += _mean_ce(owner, *entry[kind])
                        if we > 0:
                            end += _mean_ce(predictions['rail_end_' + kind][0], *entry['end_' + kind]) * entry['end_rows']
                    weight = ds.critic_weights[entry['index']].double()
                    scale = torch.where(canal, torch.full_like(era, wc / wc), torch.full_like(era, wr / wc))
                    row = scale * era + (we / wc) * end
                    values = torch.stack(((weight * row).sum(), weight.sum(),
                                          (weight * era * canal).sum(), (weight * canal).sum(),
                                          (weight * era * ~canal).sum(), (weight * ~canal).sum(),
                                          (weight * end * entry.get('end_rows', canal)).sum(),
                                          (weight * entry.get('end_rows', canal)).sum()))
                    sums = values if sums is None else sums + values
        finally:
            model.train(was); local.heads.train(heads_was)
        s = sums.cpu().tolist(); self.measure_seconds += time.perf_counter() - began
        ratio = lambda a, b: a / b if b > 0 else None
        return dict(owner_ce=ratio(s[0], s[1]), canal_owner_ce=ratio(s[2], s[3]), rail_owner_ce=ratio(s[4], s[5]),
                    rail_end_owner_ce=ratio(s[6], s[7]) if we > 0 else None)

    # ---- one 16-step boundary ---------------------------------------------------------------

    def boundary(self, step, trunk):
        """`trunk`: 'kept'/'restored' (the value phase's decision at this boundary) or 'frozen' (heads only)."""
        if not self.running: return
        local = self.local
        head_steps = local.retained_steps['ppg'] - self.saved['retained_steps']['ppg']
        attempt = self.measure()
        increased = attempt['owner_ce'] > self.current['owner_ce']
        self.intervals.append(dict(start_step=self.last_step + 1, end_step=step, head_steps=head_steps, trunk=trunk,
            owner_ce_before=self.current['owner_ce'], owner_ce_attempted=attempt['owner_ce'],
            canal_owner_ce_attempted=attempt['canal_owner_ce'], rail_owner_ce_attempted=attempt['rail_owner_ce'],
            accepted=not increased, kept_head_steps=0 if increased else head_steps))
        self.attempted += head_steps; self.last_step = step
        if increased:
            local.stop_restore(self.saved); local.stopped = True
            self.running = False; self.reason = 'heldout_ce_increased'
            # With a kept trunk the restored heads now sit on a moved trunk: score that state once for the report.
            self.final = self.measure() if trunk == 'kept' else self.current
        else:
            self.current = attempt; self.kept += head_steps; self.saved = local.stop_snapshot()

    def rebase(self):
        """The trunk moved without a check (a deferred KL rollback): restart the baseline from here."""
        if not self.running: return
        self.current = self.measure(); self.saved = self.local.stop_snapshot()

    def continue_heads(self, batches, step, head_step):
        """Heads-only fitting on `batches` ((epoch, rows) pairs) after the value phase stopped, from
        attempted step `step`; a boundary every `interval` head steps and at the end."""
        if not self.running: return
        began = time.perf_counter(); pending = 0
        for epoch, batch in batches:
            step += 1; head_step(batch, step, epoch); self.head_only_steps += 1; pending += 1
            if pending == self.interval:
                pending = 0; self.boundary(step, 'frozen')
                if not self.running: break
        if pending and self.running: self.boundary(step, 'frozen')
        self.head_only_seconds += time.perf_counter() - began

    def finish(self):
        local = self.local
        if self.running: self.reason = self.reason or 'phase_steps_exhausted'; self.final = self.current
        local.stopped = False
        record = dict(contract=CONTRACT, interval_steps=self.interval, rows=self.rows, games=self.games,
            stop_reason=self.reason, attempted_head_steps=self.attempted, kept_head_steps=self.kept,
            head_only_steps=self.head_only_steps, before=self.before, after=getattr(self, 'final', self.before),
            intervals=self.intervals,
            seconds=dict(precompute=self.precompute_seconds, measure=self.measure_seconds, head_only=self.head_only_seconds),
            scope='Owner CE of the era heads on the value phase\'s validation rows (every held-out row with a scoring map, '
                  'critic weights; slots mean plus links mean per row, era weights relative to Canal). Checked after the '
                  'value phase decides the trunk at each 16-step boundary; an increase restores the heads and their Adam to '
                  'the last kept boundary and ends head fitting. After the value phase stops, heads fit alone on the '
                  'remaining planned batches with the trunk frozen.')
        local.stop_record = record
        return record


def _mean_ce(logits, classes, mask):
    """Per row: mean owner CE over valid entities (0 for a row with none), as in the training loss."""
    ce = F.cross_entropy(logits.float().transpose(1, 2), classes, reduction='none').double()
    return (ce * mask).sum(-1) / mask.sum(-1).clamp_min(1)
