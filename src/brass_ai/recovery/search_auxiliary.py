"""g3423 search auxiliary: completed search-teacher tables train the era heads' ownership and local-VP outputs.

Adopted with `--value-revision search-aux` (training_revision_g3140; needs `heads`, `ownership-stop` and the
g3270 teacher revision). Doc: ai-research/G3423_SEARCH_AUXILIARY.md.

Maps.  The CPU teacher already feeds a native scoring recorder at every transition; its finish message now
carries the table's two era-end scoring maps once (lab_cpu_overlay). Admission audits them exactly like an
ordinary game's (local_scoring_targets.audit_eras), binds them to the table's own result and stores them,
sealed and packed, on the table's first root row in its search-value shard, so a held-out table still has
them when it rotates out of the held-out set eight iterations later.

Rows.  Every played root of a table (64 per all-seat table) is a row: its state is the acting seat's
encoded state, its seat mapping the root's `mapping` (stable seat -> relative, as the encoder's
stable_to_relative) and its era `turn_key[0]`; the target is that era's end map (the Rail-end map for
Canal rows too, when rail-end is on). Labels, masks, VP scaling and loss weights are the era heads' own.

Training (2.0.0).  The search pool is the search-value phase's own training tables admitted in the last four
iterations: the train tables collected at G-3..G, and the held-out tables that left its latest 32 in
those four iterations (they join its training set as they leave). Every PPG epoch spreads a fresh
permutation of the pool evenly over that epoch's batches, so each batch the heads train on (joint steps
and heads-only steps) carries its share, at the ordinary rows' own rate: each search root once per
epoch. Search rows contribute only the era heads' ownership and local-VP losses (and Rail-end when on),
each weighted as one ordinary row of its step; the ordinary rows, their order and their accounting are
unchanged (the permutation has its own generator). The heads' trunk-gradient scale, the PPG KL checks,
the value check, its rollback and the ownership stop apply to every step as before.

Held-out.  The search-value validation set (the latest 32 held-out tables) with maps: 4 tables in the
first iteration after adoption, 32 after eight. Their roots are pooled with the ordinary validation rows
into the ownership stop's one owner-CE criterion, each root weighted as one ordinary row.
"""
import copy
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import keys, require
from . import local_scoring_targets as targets

CONTRACT = 'search-table-auxiliary/2.0.0'
FIELD = 'local_scoring'
SOURCE = 'search-teacher-table'
ERAS = ('CanalEra', 'RailEra')
CHUNK = 512
SEED = 3423


def chunk_rows():
    """g6547: rows per search-row forward, value_phase.JOINED_ROWS (a step's whole share in one); CHUNK when off."""
    from .value_phase import JOINED_ROWS
    return JOINED_ROWS or CHUNK  # the search permutation's own generator: (SEED, generation, epoch)
POOL_STREAM = 4770  # g4770: the pool sample's own generator: (SEED, POOL_STREAM, generation)


# ---- the table's maps -----------------------------------------------------------------------

def table_envelope(receipt, result, job):
    """The finish message's receipt, audited and sealed for its table, packed for the shard."""
    targets.validate_table_receipt(receipt, result)
    envelope = dict(targets.identity(), game_id=job['expected']['game_id'], source=SOURCE, receipt=copy.deepcopy(receipt))
    envelope['sha256'] = targets.seal(envelope)
    return targets.pack(envelope)


def read_envelope(packed, game_id):
    envelope = targets.unpack(packed)
    keys(envelope, (*targets.identity(), 'game_id', 'source', 'receipt', 'sha256'))
    require(all(envelope[k] == v for k, v in targets.identity().items()) and envelope['game_id'] == game_id
            and envelope['source'] == SOURCE, 'Teacher scoring map differs from its table')
    return envelope


# ---- which tables -----------------------------------------------------------------------------

def latest(held, generation, games):
    """The latest `games` held-out games collected by `generation`, in search_value_retention.rotating_holdout's
    order (collection generation, then game id)."""
    order = sorted((g for g, rows in held.items() if rows[0]['generation'] <= generation),
                   key=lambda g: (held[g][0]['generation'], g))
    return order[-games:]


def tables(buffer, generation, *, heldout_games=32, iterations=4, pool=None):
    """Search training and held-out tables for the era heads, read without changing the buffer.

    train     the search-value training tables admitted in the last `iterations` iterations: the train
              tables collected at generation-iterations+1 .. generation, and the held-out tables that left
              the latest `heldout_games` in those iterations (rotating_holdout moves them to training);
    heldout   the latest `heldout_games` held-out tables: the search-value validation set.
    Tables without a stored map (collected before adoption or by a binary that does not send them) are
    counted and left out, so the held-out set grows from 4 to `heldout_games` after adoption.

    `pool` (g4770, training_revision_g4770.search_pool): the train tables are instead a uniform sample, without
    replacement, of the tables admitted in the last `window_iterations` iterations, as many as the last
    `sampled_iterations` admitted, drawn by a generator of its own seeded with the generation.
    """
    window = iterations if pool is None else pool['window_iterations']
    size_iterations = iterations if pool is None else pool['sampled_iterations']
    require(1 <= size_iterations <= window, 'Invalid search pool window')
    groups = dict(train=defaultdict(list), heldout=defaultdict(list))
    for receipt in buffer.receipts:
        if receipt['reward_id'] != buffer.reward_id: continue
        age = generation - receipt['generation']
        if not 0 <= age < buffer.windows[receipt['split']]: continue
        if receipt['split'] == 'train' and age >= window: continue
        for row in buffer._read(receipt): groups[receipt['split']][row['game_id']].append(row)
    held = groups['heldout']; first = lambda rows: rows[0]['generation']
    now = latest(held, generation, heldout_games); current = set(now)
    train = sorted(groups['train'], key=lambda g: (first(groups['train'][g]), g))
    rotated = [g for g in latest(held, generation - window, heldout_games) if g not in current]
    sampled = None
    if pool is not None:
        # As many tables as the last `sampled_iterations` admitted: their train tables and the held-out tables
        # that left the latest set in those iterations.
        recent = set(latest(held, generation - size_iterations, heldout_games)) - current
        size = (sum(generation - first(groups['train'][g]) < size_iterations for g in train)
                + sum(g in recent for g in rotated))
        candidates = [('train', g) for g in train] + [('rotated', g) for g in rotated]
        size = min(size, len(candidates))
        picked = sorted(np.random.default_rng((SEED, POOL_STREAM, generation)).choice(len(candidates), size, replace=False).tolist())
        train = [candidates[i][1] for i in picked if candidates[i][0] == 'train']
        rotated = [candidates[i][1] for i in picked if candidates[i][0] == 'rotated']
        sampled = dict(window_iterations=window, sampled_iterations=size_iterations, candidates=len(candidates), size=size)
    selected = dict(train=train, rotated=rotated, heldout=now)
    source = dict(train=groups['train'], rotated=held, heldout=held)
    chosen = {name: [] for name in selected}; missing = Counter()
    for name, games in selected.items():
        for game in games:
            rows = sorted(source[name][game], key=lambda r: r['index'])
            packed = next((r[FIELD] for r in rows if FIELD in r), None)
            if packed is None:
                missing[name] += 1; continue
            chosen[name].append((game, rows, read_envelope(packed, game)))
    generations = lambda name: sorted({first(source[name][g]) for g, _, _ in chosen[name]})
    report = dict(generation=generation, iterations=iterations, train_games=len(chosen['train']),
                  rotated_games=len(chosen['rotated']), heldout_games=len(chosen['heldout']),
                  train_generations=generations('train'), rotated_generations=generations('rotated'),
                  heldout_generations=generations('heldout'), without_maps=dict(missing),
                  candidates={k: len(v) for k, v in selected.items()})
    if sampled is not None:
        report['sampled'] = dict(sampled, by_generation={str(k): v for k, v in sorted(Counter(
            first(source[name][g]) for name in ('train', 'rotated') for g, _, _ in chosen[name]).items())})
    return dict(train=chosen['train'] + chosen['rotated'], heldout=chosen['heldout'], report=report)


# ---- rows on the training device ---------------------------------------------------------------

class SearchRows:
    """Played roots (and, from g3471, a table's extra decision states) with their era map and seat mapping;
    the companion reads them like critic rows."""
    def __init__(self, games, device):
        from .search_rows import kind, counts
        self.device = device; meta, envelopes, states = [], [], []; self.tables = []
        for game, rows, envelope in games:
            maps = {e['era']: e for e in envelope['receipt']['eras']}
            for r in rows:
                era = r['turn_key'][0]; seat = int(r['seat'])
                require(era in ERAS and era in maps and r['game_id'] == game, 'Search root era/map differs')
                mapping = np.asarray(r['mapping'], dtype=np.int64)
                require(mapping.tolist() == [(s - seat) % 4 for s in range(4)], 'Search root seat mapping differs')
                position = int(r['index']) if kind(r) == 'turn' else (kind(r), int(r['revision']))
                meta.append((maps[era], (envelope['sha256'], era, seat, tuple(mapping.tolist())),
                             (game, era, position, seat), mapping))
                envelopes.append(envelope); states.append(np.asarray(r['state'], dtype=np.float32))
            self.tables.append((rows, maps))
        self.meta, self.envelopes = meta, envelopes
        self.states = torch.as_tensor(np.stack(states), device=device) if states else None
        self.critic_weights = torch.ones(len(meta), device=device)
        self.games = len(games); self.rows = len(meta)
        self.eras = dict(Counter(m[0]['era'] for m in meta))
        self.kinds = counts([r for rows, _ in self.tables for r in rows])
        self.local_bootstrap = None  # g3471: the era heads' lambda-return corrections (prepare)

    def prepare(self, model, local, settings, *, rail_end=False):
        """g3471: freeze the era heads' predictions at every turn-start root and turn them into each row's
        lambda-return correction on the table's turn clock (search_rows.search_corrections).
        g3528 `rail_end`: the Canal roots' Rail-end predictions too, for the Rail-end corrections."""
        from brass_ai.learning.seat_descriptors import critic_inputs
        from .search_rows import Corrections, ENTITIES, WIDTH, kind, search_corrections
        began = time.perf_counter(); positions = []; offset = 0
        for rows, _ in self.tables:
            positions.extend(offset + j for j, r in enumerate(rows) if kind(r) == 'turn'); offset += len(rows)
        frozen, ends = {}, {}
        was, heads_were = model.training, local.heads.training
        model.eval(); local.heads.eval()
        size = chunk_rows()
        try:
            with torch.no_grad():
                for start in range(0, len(positions), size):
                    part = positions[start:start + size]; index = torch.as_tensor(part, device=self.device)
                    encoded = model.encode_state(self.states[index])
                    output = local.heads(encoded, model.descriptor_embeddings(encoded, critic_inputs(self, part)), 0.)
                    canal = torch.tensor([self.meta[j][0]['era'] == 'CanalEra' for j in part], device=self.device)
                    pieces = []
                    for name in ('slots', 'links'):
                        (co, cv), (ro, rv) = output['canal_' + name], output['rail_' + name]
                        owner = torch.where(canal[:, None, None], co, ro).float().softmax(-1)
                        pieces.append(torch.cat((owner, torch.where(canal[:, None, None], cv, rv).float()), -1))
                    values = torch.cat(pieces, 1).cpu().numpy()
                    require(values.shape[1:] == (ENTITIES, WIDTH), 'Era-head outputs differ from the search-row targets')
                    frozen.update(zip(part, values))
                    if rail_end:
                        end = torch.cat([torch.cat((output['rail_end_' + name][0].float().softmax(-1),
                                                    output['rail_end_' + name][1].float()), -1) for name in ('slots', 'links')], 1)
                        ends.update((j, v) for j, v, c in zip(part, end.cpu().numpy(), canal.tolist()) if c)
        finally:
            model.train(was); local.heads.train(heads_were)
        lam = settings['auxiliary']['lam']; tables = []; offset = 0
        for rows, maps in self.tables:
            roots = [j for j, r in enumerate(rows) if kind(r) == 'turn']
            tables.append((rows, maps, {j: frozen[offset + j] for j in roots},
                           {j: ends[offset + j] for j in roots if offset + j in ends})); offset += len(rows)
        corrections = search_corrections(tables, lam, rail_end=rail_end)
        self.corrections = torch.as_tensor(corrections, device=self.device)
        # g5466: the tensor's own indexing, not a closure over self: a self-reference made every SearchRows (its rows,
        # era maps and device tensors) cyclic garbage that waited for the iteration's full GC pass.
        self.local_bootstrap = Corrections(self.corrections.__getitem__, settings['auxiliary']['search_share'], self.device)
        report = dict(roots=len(positions), rows=self.rows, lam=lam, share=settings['auxiliary']['search_share'],
                      mean_owner_l1=float(np.abs(corrections[..., :5]).sum(-1).mean()),
                      mean_vp_l1=float(np.abs(corrections[..., 5:WIDTH]).sum(-1).mean() * 5), seconds=time.perf_counter() - began)
        if rail_end:
            canal = np.array([m[0]['era'] == 'CanalEra' for m in self.meta], dtype=np.bool_)
            end = corrections[canal, :, WIDTH:]
            report['rail_end'] = dict(rows=int(canal.sum()), canal_roots=len(ends),
                                      mean_owner_l1=float(np.abs(end[..., :5]).sum(-1).mean()) if len(end) else None,
                                      mean_vp_l1=float(np.abs(end[..., 5:]).sum(-1).mean() * 5) if len(end) else None)
        return report

    def __len__(self): return self.rows

    def parts(self, size=CHUNK):
        return [list(range(start, min(start + size, self.rows))) for start in range(0, self.rows, size)]

    def scoring_metadata(self, ids): return [self.meta[i] for i in ids]

    def scoring_envelopes(self, ids): return [self.envelopes[i] for i in ids]


# ---- search rows mixed into every PPG batch -------------------------------------------------------

class SearchAuxiliary:
    def __init__(self, selection, device, settings, *, batches):
        """`batches`: ordinary batches per PPG epoch; each gets an even share of the search pool."""
        began = time.perf_counter()
        self.settings = copy.deepcopy(settings); self.selection = copy.deepcopy(selection['report'])
        self.train = SearchRows(selection['train'], device); self.heldout = SearchRows(selection['heldout'], device)
        self.batches = int(batches); self.generation = int(self.selection['generation'])
        self.orders = {}; self.steps = []; self.loss = None; self.reason = None
        self.ready = bool(self.train.rows and self.heldout.rows and self.batches > 0)
        if not self.ready: self.reason = 'no_search_maps' if not self.train.rows else 'no_heldout_search_maps'
        self.seconds = dict(rows=time.perf_counter() - began, train=0.)
        self.targets = None  # g3471: the training rows' lambda-return targets (prepare)

    def prepare(self, model, local, settings, *, rail_end=False):
        """g3471 search-rows: the training rows train on lambda returns of the heads' own frozen predictions at
        the turn-start roots; the held-out rows keep the actual maps. g3528 `rail_end`: the Rail-end head too."""
        if self.ready and settings is not None: self.targets = self.train.prepare(model, local, settings, rail_end=rail_end)

    def share(self, epoch, position):
        """Search rows for ordinary batch `position` of `epoch`: a slice of that epoch's own permutation."""
        require(0 <= position < self.batches, 'Search rows: batch position outside the epoch')
        order = self.orders.get(epoch)
        if order is None:  # its own generator: the ordinary batches and clone draws are untouched
            order = self.orders[epoch] = np.random.default_rng((SEED, self.generation, epoch)).permutation(self.train.rows).tolist()
        n = self.train.rows
        return order[position * n // self.batches:(position + 1) * n // self.batches]

    def add(self, model, local, denominator, update, step, epoch, position, *, frozen=False):
        """This batch's search rows into the step's gradients, each weighted as one ordinary row of the step.
        `frozen`: a heads-only step (the trunk gets no gradient, as for its ordinary rows)."""
        if not self.ready: return
        from .updates import autocast
        began = time.perf_counter(); ids = self.share(epoch, position); total = None; size = chunk_rows()
        for start in range(0, len(ids), size):
            with autocast(update, self.train.device):
                loss = local.search_loss(model, self.train, ids[start:start + size], frozen=frozen)
            if loss is None: break  # the heads are stopped
            value = loss.sum() / denominator
            value.backward()
            total = value.detach() if total is None else total + value.detach()
        if total is not None:
            local.accumulate('ppg_search_weighted_auxiliary_loss_sum', total)
            self.loss = total if self.loss is None else self.loss + total
            self.steps.append(dict(step=step, epoch=epoch + 1, position=position, rows=len(ids), frozen=frozen))
        self.seconds['train'] += time.perf_counter() - began

    def finish(self, local, stop_record, report):
        """Where the search rows went, and what the pooled stop and the search validation tables say."""
        stop_record = stop_record or {}
        kept = lambda s: any(i['accepted'] and i['start_step'] <= s['step'] <= i['end_step'] for i in stop_record.get('intervals', ()))
        # The value check numbers steps within their epoch; heads-only steps leave the trunk frozen.
        trunk = lambda s: not s['frozen'] and any(i['accepted'] and i['epoch'] == s['epoch'] and
            i['start_step'] <= s['position'] + 1 < i['start_step'] + max(i['steps'], 1) for i in report.get('interval_reports', ()))
        steps = self.steps; rows = lambda chosen: sum(s['rows'] for s in chosen)
        if self.ready and not steps: self.reason = 'heads_stopped'
        per_step = [s['rows'] for s in steps]
        search = stop_record.get('search') or {}; criterion = stop_record.get('criterion') or {}
        return dict(contract=CONTRACT, settings=self.settings, ready=self.ready, reason=self.reason,
            selection=self.selection,
            train=dict(rows=self.train.rows, games=self.train.games, eras=self.train.eras, kinds=self.train.kinds),
            heldout=dict(rows=self.heldout.rows, games=self.heldout.games, eras=self.heldout.eras, kinds=self.heldout.kinds),
            lambda_targets=self.targets,
            presented=dict(rows=rows(steps), steps=len(steps), joint_rows=rows([s for s in steps if not s['frozen']]),
                           heads_only_rows=rows([s for s in steps if s['frozen']]),
                           rows_per_step=dict(min=min(per_step), max=max(per_step)) if per_step else None,
                           batches_per_epoch=self.batches),
            kept=dict(heads_rows=rows([s for s in steps if kept(s)]), trunk_rows=rows([s for s in steps if trunk(s)])),
            weighted_loss=float(self.loss) if self.loss is not None else None,
            heldout_owner_ce=dict(before=search.get('before'), after=search.get('after')),
            criterion=dict(before=criterion.get('before'), after=criterion.get('after'), search_share=criterion.get('search_share')),
            by_era=local.search_report() if local is not None else None, seconds=self.seconds,
            scope='Search-value training tables of the last four iterations (from g4770 a uniform sample of that size from '
                  'the last eight: selection.sampled), mixed into every PPG batch the heads train '
                  'on (each root once per epoch, at the ordinary rows\' rate): era-head ownership and local-VP losses only, '
                  'each row weighted as one ordinary row of its step. kept.heads_rows: rows in intervals the ownership stop '
                  'kept; kept.trunk_rows: rows in joint steps the value check kept. Held-out: the search-value validation '
                  'tables with maps, pooled with the ordinary validation rows into the stop\'s owner-CE criterion.')
