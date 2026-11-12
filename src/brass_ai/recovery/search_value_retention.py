"""A reproducible game-level holdout for the Stage-C value update only."""
from collections import Counter, defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import digest, require
from brass_ai.learning.search_value import correction
from .search_value_schedule import HELDOUT_GAMES, HELDOUT_ITERATIONS


def reserve_games(rows, generation, seed):
    """Draw eight whole games from the live eight-age return buffer.

    These games may have been used in earlier phases. They are excluded from
    *all* fitting in this phase. The independent periodic four-game diagnostic
    remains separate. Neither the sampling nor recovery uses a global RNG.
    """
    groups = defaultdict(list)
    for row in rows:
        require(row['split'] == 'train' and 0 <= generation-row['generation'] < 8,
                'Stage-C holdout must come from the current eight-age training buffer')
        groups[row['game_id']].append(row)
    for game in groups.values():
        require(len(game) == 64 and len({r['index'] for r in game}) == 64
                and len({r['generation'] for r in game}) == 1, 'Incomplete/duplicate Stage-C value game')
    keys = sorted(groups)
    draw_seed = int(digest(['stage-c-heldout/1', seed, generation, keys])[:16], 16)
    report = dict(contract='stage-c-heldout/1', seed=draw_seed, requested_games=8,
                  available_games=len(keys), window_iterations=8, games=[], rows=0,
                  scope='Eight whole buffered games excluded from this Stage-C update; not an unseen-across-history test')
    if len(keys) <= 8:
        return [], [], dict(report, ready=False, reason='Need eight held-out games and at least one training game')
    rng = np.random.default_rng(draw_seed)
    selected = sorted(keys[int(i)] for i in rng.choice(len(keys), 8, replace=False))
    held = set(selected)
    train = [r for r in rows if r['game_id'] not in held]
    validation = [dict(r, split='heldout', source_split='train') for r in rows if r['game_id'] in held]
    return train, validation, dict(report, ready=True, games=selected, rows=len(validation),
        games_digest=digest(selected), generations=sorted({r['generation'] for r in validation}), train_rows=len(train))


def recent_holdout(rows, generation):
    """Latest 16 complete dedicated games in the trailing 20 collections."""
    groups = defaultdict(list)
    for row in rows:
        require(row['split'] == 'heldout' and 0 <= generation - row['generation'] < HELDOUT_ITERATIONS,
                'Early stop requires dedicated held-out games in the current window')
        groups[row['game_id']].append(row)
    for game in groups.values():
        require(len(game) == 64 and {r['index'] for r in game} == set(range(1, 65)) and
                len({r['generation'] for r in game}) == 1, 'Incomplete/duplicate held-out search game')
    ready = len(groups) >= HELDOUT_GAMES
    latest = sorted(groups, key=lambda game: (groups[game][0]['generation'], game))[-HELDOUT_GAMES:]
    # Score on the realized utility-to-go. Older held-out games may have been
    # collected with the retired head, so their root predictions are not targets.
    validation = [dict(r, target=r['u']) for game in latest for r in groups[game]] if ready else []
    return validation, dict(contract='dedicated-search-heldout/1', ready=ready, requested_games=HELDOUT_GAMES,
        available_games=len(groups), window_iterations=HELDOUT_ITERATIONS, games=latest if ready else [],
        rows=len(validation), generations=sorted({r['generation'] for r in validation}),
        evaluation_target='final-utility-to-go')


class HeldoutLoss:
    """Pack validation positions once and recompute predictions per interval."""
    def __init__(self, model, rows):
        games = Counter(r['game_id'] for r in rows)
        require(len(games) in (8, HELDOUT_GAMES) and all(n == 64 for n in games.values()) and
                all(r['split'] == 'heldout' for r in rows), 'Search early stop needs complete held-out games')
        device = next(model.parameters()).device
        self.model = model
        self.inputs = tuple(torch.as_tensor(np.asarray([r[key] for r in rows]), dtype=dtype, device=device)
            for key, dtype in (('state', torch.float32), ('mapping', torch.long), ('remaining', torch.float32)))
        self.targets = torch.as_tensor(np.asarray([r['target'] for r in rows]), dtype=torch.float32, device=device)
        require(bool(torch.isfinite(self.targets).all()), 'Nonfinite Stage-C held-out target')

    def __call__(self):
        state, mapping, remaining = self.inputs
        count = len(self.targets)
        losses = []
        with torch.no_grad():
            for start in range(0, count, 512):
                batch = slice(start, start + 512)
                encoded = self.model.encode_state(state[batch])
                main = self.model.heads_from_encoded(encoded, mapping[batch], remaining[batch])[-1]
                delta = correction(self.model, encoded, mapping[batch], remaining[batch])
                losses.append(torch.nn.functional.mse_loss(main + delta, self.targets[batch]))
        value = float(losses[0] if len(losses) == 1 else torch.stack(losses).mean())
        require(np.isfinite(value), 'Nonfinite Stage-C held-out loss')
        return value
