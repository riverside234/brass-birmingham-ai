"""g3140 step 7: lambda-returns for the win and utility heads from the critic frozen before PPO.

For each seat's own chain of critic rows (ingress keeps game-record order, as in
GAE), with Monte Carlo targets Z and frozen predictions V of the same 4-seat
quantity:

    G_t = Z_t + C_t,   C_t = (1 - lam) (V_{t+1} - Z_{t+1}) + lam C_{t+1},   C_last = 0.

This is the usual TD(lambda) return: rewards between decisions are Z_t - Z_{t+1}
and the last row ends in the real result. For win shares Z is constant within a
game, so G_t stays a distribution. Training blends G with the outcome; the
held-out early stop keeps scoring real outcomes.
"""
from collections import defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import require

CONTRACT = 'critic-lambda-returns/1.0.0'


def lambda_returns(dataset, frozen, lam):
    """[rows, 2, 4] float32: lambda-return targets for the win shares and the utility, by critic row."""
    require(0 < lam < 1, 'Invalid bootstrap lambda')
    rows = len(dataset.data)
    z_win = dataset.targets[0].detach().double().cpu().numpy()
    z_utility = dataset.utility.detach().double().cpu().numpy()
    v_win = torch.as_tensor(frozen['win']).double().numpy()
    v_utility = torch.as_tensor(frozen['utility']).double().numpy()
    require(z_win.shape == z_utility.shape == v_win.shape == v_utility.shape == (rows, 4)
            and np.isfinite(v_win).all() and np.isfinite(v_utility).all(), 'Frozen critic outputs do not match the critic rows')
    chains = defaultdict(list)
    for i in range(rows): chains[dataset.data[i][2]].append(i)
    longest = max(len(c) for c in chains.values())
    # Pad every chain to one length (-1 = no row) and sweep all chains backwards at once.
    index = np.full((len(chains), longest), -1, dtype=np.int64)
    for c, chain in enumerate(chains.values()): index[c, :len(chain)] = chain
    z = np.stack((z_win, z_utility), 1); v = np.stack((v_win, v_utility), 1)
    gap = v - z
    result = np.empty((rows, 2, 4), dtype=np.float64)
    correction = np.zeros((len(chains), 2, 4)); following = np.full(len(chains), -1, dtype=np.int64)
    for t in range(longest - 1, -1, -1):
        current = index[:, t]; live = current >= 0
        has_next = live & (following >= 0)
        correction[has_next] = (1 - lam) * gap[following[has_next]] + lam * correction[has_next]
        correction[live & (following < 0)] = 0.
        result[current[live]] = z[current[live]] + correction[live]
        following = np.where(live, current, following)
    require(np.isfinite(result).all(), 'Nonfinite bootstrap targets')
    return result.astype(np.float32)


def report(dataset, targets):
    z_win = dataset.targets[0].detach().cpu().numpy(); z_utility = dataset.utility.detach().cpu().numpy()
    return dict(contract=CONTRACT, rows=int(len(targets)),
        win_mean_abs_shift=float(np.abs(targets[:, 0] - z_win).mean()),
        utility_mean_abs_shift=float(np.abs(targets[:, 1] - z_utility).mean()),
        utility_variance_outcome=float(z_utility.var()), utility_variance_bootstrap=float(targets[:, 1].var()))
