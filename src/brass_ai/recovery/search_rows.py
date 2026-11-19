"""g3471 expanded search rows: more search-teacher states for the search-value head and the era heads.

Adopted with `--value-revision search-rows` (training_revision_g3140; needs `search-aux` and `targets`).
Doc: ai-research/G3471_SEARCH_ROWS.md.

Rows.  A search-teacher table keeps its 64 turn-start roots (kind 'turn') and the CPU teacher adds two row
types (lab-native-cpu/src/cpu_capture.rs), with no new search:
  second   the state before every played second main action of a turn (60 in a 4-seat game);
  sampled  in each era, a uniform sample of up to 24 other played decisions with two or more legal choices
           (a develop's second tile, a sale, a payment source, a bonus, a liquidation): the 24 decisions with
           the smallest keys of a splitmix64 stream keyed by the job's sampling seed and the decision's
           revision, a stream of its own. The finish message lists every candidate, so admission recomputes
           the sample, and each decision is captured once.
An extra row keeps the index of its turn's root (the turn clock), its own revision, seat mapping, encoded
state, remaining insolvency budget and spent costs. Tables collected before adoption keep their 64 roots.

Search value (lambda .95, once per turn).  The roots' targets are unchanged. An extra row x in turn t
bootstraps at the next original turn start:

    G_x = -(spent_{t+1} - spent_x) + (1 - lam) r_{t+1} + lam G_{t+1},

with r the next root's search value and G its own target, so x's own costs from x onward count once and the
last turn ends in the terminal utility (the NN and search value is the full-game utility, so no reward falls
between). lam = 1 gives x's realized utility-to-go u_x. Held-out tables score every row against u.

Era heads (lambda .95).  A target is the actual era-end map Z (owner class as one-hot probabilities, each
seat's local VP / 5) plus a correction C = G - Z from the heads' own frozen predictions P (ownership
probabilities and per-seat local VP of the head of that era):

    G_i = (1 - lam) P_next + lam G_next,   G = Z at the era's last row (its actual scoring horizon).

Ordinary PPO rows follow each seat's chain of critic rows, as the utility blend does, with P frozen before
PPO; they train on Z + .5 C. Search rows follow the table's turn clock: P at the next turn-start root of the
same era (an extra row in turn t takes root t+1, so it shares its turn root's correction), aligned to the
row's seat through the stable seats and frozen at the start of the value phase; they train on Z + C. The
Rail-end head (Canal rows predicting the game-end map; weight 0 in this run) keeps its actual map. The
ownership stop's held-out criterion keeps scoring the actual maps.
"""
import copy
from collections import defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import keys, require

CONTRACT = 'expanded-search-rows/1.0.0'
JOB = dict(contract=CONTRACT, sampled_per_era=24)
KINDS = ('turn', 'second', 'sampled')
DECISIONS = ('action', 'payment', 'continuation', 'bonus', 'debt')
ERAS = ('CanalEra', 'RailEra')
FIELDS = ('kind', 'index', 'seat', 'revision', 'decision', 'phase', 'legal', 'actions_remaining', 'state', 'mapping',
          'remaining', 'spent', 'turn_key')
ENTITIES = 49 + 69  # slots then links, in the era heads' order
WIDTH = 9           # owner probabilities (none, four relative seats) then the four seats' local VP / 5
STREAM = 0x34715EA200000000
MASK = (1 << 64) - 1


def kind(row): return row.get('kind', 'turn')


def roots(rows): return [r for r in rows if kind(r) == 'turn']


def row_key(row):
    """Unique within a table: a root by its turn index, an extra row by its decision's revision."""
    return (row['game_id'], kind(row), row['index'] if kind(row) == 'turn' else row['revision'])


def counts(rows):
    """Rows by kind."""
    result = defaultdict(int)
    for r in rows: result[kind(r)] += 1
    return {k: result[k] for k in KINDS if result[k]}


# ---- the sample stream (cpu_capture.rs twin) ------------------------------------------------------

def _mix(z):
    z = (z + 0x9e3779b97f4a7c15) & MASK
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK
    return z ^ (z >> 31)


def sample_key(seed, revision): return _mix(seed ^ _mix(revision ^ STREAM))


def sample(seed, candidates, size):
    """The revisions a table keeps from `candidates`: the `size` smallest (key, revision), in revision order."""
    return sorted(sorted(candidates, key=lambda r: (sample_key(seed, r), r))[:size])


# ---- admission ---------------------------------------------------------------------------------

def game_phase(era, round_):
    if era == 'CanalEra': return 'canal_early' if round_ <= 4 else 'canal_late'
    return 'rail_early' if round_ <= 6 else 'rail_late'


def validate(block, turn, job):
    """A table's extra rows from its finish message, checked against its 64 validated roots and its job;
    returned with the table's lineage, ready for targets."""
    from .search_value_rows import state_width
    require(type(block) is dict, 'Missing expanded search rows')
    keys(block, ('contract', 'sampled_per_era', 'second', 'sampled', 'candidates'))
    require(block['contract'] == CONTRACT and block['sampled_per_era'] == JOB['sampled_per_era']
            and type(block['second']) is list and type(block['sampled']) is list
            and type(block['candidates']) is dict and set(block['candidates']) == set(ERAS), 'Expanded search rows differ')
    require(len(turn) == 64 and [r['index'] for r in turn] == list(range(1, 65)), 'Expanded rows need the 64 roots')
    cap = job['search']['reward']['insolvency']['max_cost']; width = state_width()
    bounds = [int(r['revision']) for r in turn]
    seen = set(bounds); second_turns = set(); result = []
    by_era = {era: [] for era in ERAS}
    for name in ('second', 'sampled'):
        for x in block[name]:
            keys(x, FIELDS)
            require(x['kind'] == name and type(x['index']) is int and 1 <= x['index'] <= 64
                    and type(x['seat']) is int and 0 <= x['seat'] < 4 and type(x['revision']) is int
                    and type(x['legal']) is int and x['legal'] >= 1 and type(x['actions_remaining']) is int
                    and x['decision'] in DECISIONS, 'Invalid expanded search row')
            require(x['mapping'] == [(s - x['seat']) % 4 for s in range(4)], 'Expanded row seat mapping differs')
            t = x['index']; root = turn[t - 1]
            require(bounds[t - 1] < x['revision'] and (t == 64 or x['revision'] < bounds[t]) and x['revision'] not in seen,
                    'Expanded row outside its turn or repeated')
            seen.add(x['revision'])
            turn_key = x['turn_key']
            require(type(turn_key) is list and len(turn_key) == 4 and turn_key[0] in ERAS and type(turn_key[1]) is int
                    and x['phase'] == game_phase(turn_key[0], turn_key[1]), 'Expanded row turn/phase differs')
            state = np.asarray(x['state'], dtype=np.float32)
            require(state.shape == (width,) and np.isfinite(state).all(), 'Expanded row state dimensions differ')
            spent, remaining = (np.asarray(x[k], dtype=float) for k in ('spent', 'remaining'))
            require(spent.shape == remaining.shape == (4,) and np.isfinite(spent).all() and np.isfinite(remaining).all()
                    and np.min(spent) >= 0 and np.allclose(spent + remaining, cap, atol=1e-9)
                    and np.all(spent >= np.asarray(root['spent']) - 1e-10)
                    and (t == 64 or np.all(spent <= np.asarray(turn[t]['spent']) + 1e-10)), 'Expanded row costs differ')
            if name == 'second':
                require(x['decision'] == 'action' and x['seat'] == root['seat'] and turn_key == list(root['turn_key'])
                        and t not in second_turns and not (turn_key[0] == 'CanalEra' and turn_key[1] == 1),
                        'Second-action row differs from its turn')
                second_turns.add(t)
            else:
                require(x['legal'] >= 2, 'A sampled decision needs two or more choices')
                by_era[turn_key[0]].append(x['revision'])
            result.append(dict(x, generation=root['generation'], model_id=root['model_id'], reward_id=root['reward_id'],
                               game_id=root['game_id'], split=root['split'], profile=root['profile']))
    seconds = {x['revision'] for x in block['second']}
    for era in ERAS:
        candidates = block['candidates'][era]
        require(type(candidates) is list and all(type(r) is int for r in candidates)
                and all(a < b for a, b in zip(candidates, candidates[1:]))
                and not set(candidates) & (set(bounds) | seconds), 'Expanded row candidates differ')
        require(sorted(by_era[era]) == sample(job['sampling_seed'], candidates, JOB['sampled_per_era']),
                'Sampled decisions differ from their stream')
    return result


def complete(turn, extras, result, final_spent, reward, lam):
    """Targets and realized outcomes of a table's extra rows, from its completed roots (search_value_rows.targets)."""
    from brass_ai.learning.reward import terminal_utility
    from .search_value_rows import copy_row, targets
    if not extras: return []
    g, _ = targets(turn, result, final_spent, reward, lam)
    final = terminal_utility(result, reward); last = np.asarray(final_spent, dtype=float)
    credit = turn[0]['vp_payments']['canal'] if 'vp_payments' in turn[0] else None
    output = []
    for x in extras:
        t = x['index']; spent = np.asarray(x['spent'], dtype=float)
        following = final if t == 64 else g[t]
        boot = final if t == 64 else np.asarray(turn[t]['r'])
        upcoming = last if t == 64 else np.asarray(turn[t]['spent'], dtype=float)
        target = -(upcoming - spent) + (1 - lam) * boot + lam * following
        row = dict(copy_row(x), target=target.tolist(), u=(final - (last - spent)).tolist(), target_lambda=lam)
        if credit is not None:
            paid = list(credit) if row['turn_key'][0] == 'RailEra' else [0.] * 4
            row.update(vp_paid=paid, future_u=(np.asarray(row['u']) - paid).tolist())
        output.append(row)
    return output


def turn_targets(game, lam):
    """One table's rows with targets at `lam` (search_value_phase._targets): the 64 roots exactly as before,
    then every extra row from its turn's next root."""
    turn = sorted(roots(game), key=lambda r: r['index'])
    following = np.asarray(turn[-1]['terminal_utility'])
    targets, upcoming = [], {}
    for i in reversed(range(len(turn))):
        next_u = np.asarray(turn[i + 1]['u'] if i + 1 < len(turn) else turn[i]['terminal_utility'])
        cost = next_u - turn[i]['u']
        boot = np.asarray(turn[i + 1]['r'] if i + 1 < len(turn) else turn[i]['terminal_utility'])
        upcoming[turn[i]['index']] = (next_u, boot, following)
        following = -cost + (1 - lam) * boot + lam * following
        targets.append(dict(turn[i], target=following.tolist(), target_lambda=lam))
    result = list(reversed(targets))
    for x in game:
        if kind(x) == 'turn': continue
        next_u, boot, after = upcoming[x['index']]
        value = -(next_u - np.asarray(x['u'])) + (1 - lam) * boot + lam * after
        result.append(dict(x, target=value.tolist(), target_lambda=lam))
    return result


# ---- era-head targets: lambda-returns of the heads' own frozen predictions -------------------------

def map_arrays(target, mapping):
    """[118, 9] actual era map in a row's relative frame (one-hot owner, then local VP / 5) and its [118] mask."""
    from .local_scoring_v3 import VP_SCALE
    owner = np.array([r['owner'] - 1 for k in ('slots', 'links') for r in target[k]])
    vp = np.array([r['vp'] for k in ('slots', 'links') for r in target[k]], dtype=np.float64)
    valid = np.array([r['valid'] for k in ('slots', 'links') for r in target[k]], dtype=np.bool_)
    present = owner >= 0; relative = np.asarray(mapping)[owner.clip(0)]
    z = np.zeros((ENTITIES, WIDTH), dtype=np.float64)
    z[np.arange(ENTITIES), np.where(present, relative + 1, 0)] = 1.
    z[np.flatnonzero(present), 5 + relative[present]] = vp[present] / VP_SCALE
    return z, valid


def head_outputs(values, rows, canal):
    """[n, 118, 9] frozen predictions (owner probabilities, then local VP) of each row's own era head, from
    per-output arrays named as value_preservation.outputs names them ('canal_slots_owner', ...)."""
    rows = np.asarray(rows, dtype=np.int64); canal = np.asarray(canal, dtype=np.bool_)
    result = np.empty((len(rows), ENTITIES, WIDTH), dtype=np.float32)
    for horizon, chosen in (('canal', canal), ('rail', ~canal)):
        where = np.flatnonzero(chosen)
        if not len(where): continue
        ix = torch.as_tensor(rows[where])
        parts = [torch.cat((torch.as_tensor(values[f'{horizon}_{k}_owner'])[ix].float(),
                            torch.as_tensor(values[f'{horizon}_{k}_vp'])[ix].float()), -1) for k in ('slots', 'links')]
        result[where] = torch.cat(parts, 1).cpu().numpy()
    return result


def ordinary_corrections(dataset, frozen, lam):
    """g3471: ([rows, 118, 9] float16 corrections of the critic rows' era-head targets, report).

    Each seat's chain of critic rows (game-record order, as the utility blend) is cut wherever its scoring map
    or seat frame changes, so each run ends at its era's actual map; rows without a map get zeros."""
    from .local_scoring_targets import map_for
    require(0 < lam <= 1, 'Invalid auxiliary lambda')
    rows = len(dataset.data)
    require(all(len(v) == rows for v in frozen.values()), 'Frozen era-head outputs do not match the critic rows')
    out = np.zeros((rows, ENTITIES, WIDTH), dtype=np.float16)
    runs, tokens, current = [], [], {}
    for i in range(rows):
        row = dataset.rows[i]; label = row.get('_local_scoring')
        if label is None: continue
        target = map_for(row)
        token = (label['envelope']['sha256'], target['era'], tuple(int(x) for x in dataset.data[i][3].stable_to_relative))
        chain = dataset.data[i][2]; last = current.get(chain)
        if last is None or last[0] != token:
            current[chain] = (token, len(runs)); runs.append([]); tokens.append((token, target))
        runs[current[chain][1]].append(i)
    if not runs: return out, dict(rows=0, runs=0)
    maps = {}
    for token, target in tokens:
        if token not in maps: maps[token] = map_arrays(target, token[2])[0]
    z = np.stack([maps[token] for token, _ in tokens]).astype(np.float32)
    canal = np.array([token[1] == 'CanalEra' for token, _ in tokens])
    longest = max(map(len, runs))
    index = np.full((len(runs), longest), -1, dtype=np.int64)
    for k, run in enumerate(runs): index[k, :len(run)] = run
    c = np.zeros((len(runs), ENTITIES, WIDTH), dtype=np.float32)
    following = np.full(len(runs), -1, dtype=np.int64)
    owner = vp = 0.
    for t in range(longest - 1, -1, -1):
        now = index[:, t]; live = now >= 0
        step = np.flatnonzero(live & (following >= 0))
        if len(step):
            c[step] = (1 - lam) * (head_outputs(frozen, following[step], canal[step]) - z[step]) + lam * c[step]
            owner += float(np.abs(c[step, :, :5]).sum()); vp += float(np.abs(c[step, :, 5:]).sum())
        c[live & (following < 0)] = 0.
        out[now[live]] = c[live]
        following = np.where(live, now, following)
    require(np.isfinite(out).all(), 'Nonfinite era-head corrections')
    used = sum(map(len, runs)) * ENTITIES
    return out, dict(rows=sum(map(len, runs)), runs=len(runs), lam=lam, mean_owner_l1=owner / used, mean_vp_l1=vp * 5 / used,
                     scope='Mean over critic rows with a map and entities: L1 of the owner-probability correction and of the '
                           'local-VP correction (in VP, summed over seats), before the .5 share')


def to_stable(a, mapping):
    """A [..., 118, 9] relative-frame array in the stable-seat frame (mapping: stable seat -> relative seat)."""
    m = np.asarray(mapping); result = a.copy()
    result[..., 1:5] = a[..., 1 + m]; result[..., 5:9] = a[..., 5 + m]
    return result


def to_relative(a, mapping):
    m = np.asarray(mapping); result = a.copy()
    result[..., 1 + m] = a[..., 1:5]; result[..., 5 + m] = a[..., 5:9]
    return result


def search_corrections(tables, lam):
    """g3471: [rows, 118, 9] float32 corrections of search rows on each table's turn clock.

    `tables`: per table (rows in SearchRows order, era maps, frozen), with `frozen[j]` the [118, 9] prediction of
    the root at row position j by its own era head, in its own frame. Within an era the roots form the chain (the
    era's last root has C = 0); an extra row in turn t bootstraps at root t+1 when that root is in its era, else it
    is at its horizon (C = 0). Frames meet in the stable seats."""
    require(0 < lam <= 1, 'Invalid auxiliary lambda')
    total = sum(len(rows) for rows, _, _ in tables)
    out = np.zeros((total, ENTITIES, WIDTH), dtype=np.float32)
    offset = 0
    for rows, maps, frozen in tables:
        at = {r['index']: j for j, r in enumerate(rows) if kind(r) == 'turn'}
        stable = {era: map_arrays(target, [0, 1, 2, 3])[0] for era, target in maps.items()}
        era_of = lambda t: rows[at[t]]['turn_key'][0]
        chain = {}
        for era in maps:
            following = None
            for t in sorted((t for t in at if era_of(t) == era), reverse=True):
                chain[t] = (np.zeros((ENTITIES, WIDTH)) if following is None else
                            (1 - lam) * (to_stable(frozen[at[following]], rows[at[following]]['mapping']) - stable[era])
                            + lam * chain[following])
                following = t
        for j, r in enumerate(rows):
            era = r['turn_key'][0]
            if era not in maps: continue
            if kind(r) == 'turn':
                c = chain[r['index']]
            else:
                t = r['index'] + 1
                c = ((1 - lam) * (to_stable(frozen[at[t]], rows[at[t]]['mapping']) - stable[era]) + lam * chain[t]
                     if t in at and era_of(t) == era else None)
            if c is not None: out[offset + j] = to_relative(c, r['mapping'])
        offset += len(rows)
    require(np.isfinite(out).all(), 'Nonfinite search-row corrections')
    return out


class Corrections:
    """What the era heads' loss reads (local_scoring_v3): `(ds, ix) -> [B, 118, 9] share-scaled corrections or None`."""
    def __init__(self, source, share, device):
        self.source, self.share, self.device = source, float(share), device

    def __call__(self, ix):
        values = self.source(ix)
        if values is None: return None
        return torch.as_tensor(values, device=self.device).float() * self.share


def summary(settings): return copy.deepcopy(settings)
