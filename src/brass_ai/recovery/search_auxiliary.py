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

Training.  Once per iteration one batch of the 40 search training tables -- the 36 train tables of this
iteration and the 4 held-out tables leaving the 32-game search-value held-out set -- joins the first step
of the PPG value phase. Its rows contribute only the era heads' ownership and local-VP losses (and
Rail-end when on), each row weighted as one ordinary row of that step; the step's ordinary rows are kept
unchanged. The heads' trunk-gradient scale, the PPG KL checks, the value check, its rollback and the
ownership stop all apply to that step as to any other.

Held-out.  This iteration's 4 search validation tables are scored by the ownership stop beside its
ordinary validation games: at the boundary closing the interval that holds the search batch, an increase
of either owner CE restores the heads (existing rollback) and ends their fitting. Their CE is reported at
every boundary.
"""
import copy
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import keys, require
from . import local_scoring_targets as targets

CONTRACT = 'search-table-auxiliary/1.0.0'
FIELD = 'local_scoring'
SOURCE = 'search-teacher-table'
ERAS = ('CanalEra', 'RailEra')
CHUNK = 512


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

def tables(buffer, generation, *, heldout_games=32):
    """Train and held-out tables for this iteration's search batch, read without changing the buffer.

    train     the `generation` train tables plus the held-out tables leaving the latest `heldout_games`
              (the 32-game rotating search-value held-out set) at this generation;
    heldout   the `generation` held-out (validation) tables.
    Tables without a stored map (collected before adoption or by a binary that does not send them) are
    counted and left out.
    """
    groups = dict(train=defaultdict(list), heldout=defaultdict(list))
    for receipt in buffer.receipts:
        if receipt['reward_id'] != buffer.reward_id: continue
        if not 0 <= generation - receipt['generation'] < buffer.windows[receipt['split']]: continue
        for row in buffer._read(receipt): groups[receipt['split']][row['game_id']].append(row)
    first = lambda rows: rows[0]['generation']
    held = groups['heldout']
    order = sorted(held, key=lambda game: (first(held[game]), game))
    now = set(order[-heldout_games:])
    before = [game for game in order if first(held[game]) < generation][-heldout_games:]
    selected = dict(current_train=sorted(g for g, rows in groups['train'].items() if first(rows) == generation),
                    rotated=[g for g in before if g not in now],
                    heldout=[g for g in order if first(held[g]) == generation])
    source = dict(current_train=groups['train'], rotated=held, heldout=held)
    chosen = {name: [] for name in selected}; missing = Counter()
    for name, games in selected.items():
        for game in games:
            rows = sorted(source[name][game], key=lambda r: r['index'])
            packed = next((r[FIELD] for r in rows if FIELD in r), None)
            if packed is None:
                missing[name] += 1; continue
            chosen[name].append((game, rows, read_envelope(packed, game)))
    report = dict(generation=generation, current_train_games=len(chosen['current_train']), rotated_games=len(chosen['rotated']),
                  rotated_generations=sorted({first(held[g]) for g in selected['rotated']}),
                  heldout_games=len(chosen['heldout']), without_maps=dict(missing),
                  candidates={k: len(v) for k, v in selected.items()})
    return dict(train=chosen['current_train'] + chosen['rotated'], heldout=chosen['heldout'], report=report)


# ---- rows on the training device ---------------------------------------------------------------

class SearchRows:
    """Played roots with their era map and seat mapping; the companion reads them like critic rows."""
    def __init__(self, games, device):
        self.device = device; meta, envelopes, states = [], [], []
        for game, rows, envelope in games:
            maps = {e['era']: e for e in envelope['receipt']['eras']}
            for r in rows:
                era = r['turn_key'][0]; seat = int(r['seat'])
                require(era in ERAS and era in maps and r['game_id'] == game, 'Search root era/map differs')
                mapping = np.asarray(r['mapping'], dtype=np.int64)
                require(mapping.tolist() == [(s - seat) % 4 for s in range(4)], 'Search root seat mapping differs')
                meta.append((maps[era], (envelope['sha256'], era, seat, tuple(mapping.tolist())),
                             (game, era, int(r['index']), seat), mapping))
                envelopes.append(envelope); states.append(np.asarray(r['state'], dtype=np.float32))
        self.meta, self.envelopes = meta, envelopes
        self.states = torch.as_tensor(np.stack(states), device=device) if states else None
        self.critic_weights = torch.ones(len(meta), device=device)
        self.games = len(games); self.rows = len(meta)
        self.eras = dict(Counter(m[0]['era'] for m in meta))

    def __len__(self): return self.rows

    def parts(self, size=CHUNK):
        return [list(range(start, min(start + size, self.rows))) for start in range(0, self.rows, size)]

    def scoring_metadata(self, ids): return [self.meta[i] for i in ids]

    def scoring_envelopes(self, ids): return [self.envelopes[i] for i in ids]


# ---- the once-per-iteration batch ---------------------------------------------------------------

class SearchAuxiliary:
    def __init__(self, selection, device, settings):
        began = time.perf_counter()
        self.settings = copy.deepcopy(settings); self.selection = copy.deepcopy(selection['report'])
        self.train = SearchRows(selection['train'], device); self.heldout = SearchRows(selection['heldout'], device)
        self.step = None; self.loss = None; self.reason = None
        self.ready = bool(self.train.rows and self.heldout.rows)
        if not self.ready: self.reason = 'no_search_maps' if not self.train.rows else 'no_heldout_search_maps'
        self.seconds = dict(rows=time.perf_counter() - began, train=0.)

    def due(self, step):
        return self.ready and self.step is None and step == self.settings['step']

    def add(self, model, local, denominator, update, step):
        """The search rows' auxiliary loss into this step's gradients; each row weighs as one ordinary row."""
        from .updates import autocast
        began = time.perf_counter(); total = None
        for part in self.train.parts():
            with autocast(update, self.train.device):
                loss = local.search_loss(model, self.train, part)
            if loss is None: continue
            value = loss.sum() / denominator
            value.backward()
            total = value.detach() if total is None else total + value.detach()
        self.step = step
        if total is not None:
            local.accumulate('ppg_search_weighted_auxiliary_loss_sum', total)
            self.loss = total
        else:
            self.reason = 'heads_stopped'
        self.seconds['train'] += time.perf_counter() - began

    def finish(self, local, stop_record, report):
        """The batch, where it went, and what the held-out search tables say."""
        step = self.step
        # The ownership stop's interval and the value check's interval that hold the search step
        # (the value check numbers steps within its epoch; the search step is in the first epoch).
        heads = next((i for i in (stop_record or {}).get('intervals', ()) if i['start_step'] <= step <= i['end_step']), None) \
            if step is not None else None
        trunk = next((i for i in report.get('interval_reports', ()) if i['epoch'] == 1 and
                      i['start_step'] <= step < i['start_step'] + max(i['steps'], 1)), None) if step is not None else None
        search = (stop_record or {}).get('search') or {}
        return dict(contract=CONTRACT, settings=self.settings, ready=self.ready, reason=self.reason, step=self.step,
            selection=self.selection,
            train=dict(rows=self.train.rows, games=self.train.games, eras=self.train.eras),
            heldout=dict(rows=self.heldout.rows, games=self.heldout.games, eras=self.heldout.eras),
            weighted_loss=float(self.loss) if self.loss is not None else None,
            heads_kept=heads['accepted'] if heads else None, trunk_kept=trunk['accepted'] if trunk else None,
            stopped_by=heads.get('stopped_by') if heads else None,
            heldout_owner_ce=dict(before=search.get('before'), after_search_interval=heads.get('search_attempted') if heads else None,
                                  after=search.get('after')),
            by_era=local.search_report() if local is not None else None, seconds=self.seconds,
            scope='One batch of played roots from the search training tables in the first PPG step: era-head ownership '
                  'and local-VP losses only, each row weighted as one ordinary row of that step. heads_kept: the ownership '
                  'stop kept that interval (ordinary and search held-out owner CE did not increase); trunk_kept: the value '
                  'check kept it. Held-out: this iteration\'s search validation tables, unweighted roots.')
