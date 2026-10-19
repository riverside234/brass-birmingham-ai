"""PFSP (EMA outrank) vs OpenAI Five quality-score sampler on the 2+2 history tables.

Model: each seat's finishing utility is theta + Gumbel noise, so the pairwise outrank
probability is sigmoid(theta_L - theta_j) (Plackett-Luce). 144 history tables per
iteration, 2 learner seats + 2 snapshot seats. Each snapshot seat goes to IL with
probability 1/(k+1), where k = models in the pool including IL (every seat while IL is
alone); otherwise it is an independent draw from the sampler.

Run:  python3 pfsp_sim.py              # IL share 1/(k+1)
      python3 pfsp_sim.py --il-fixed   # old fixed 20% IL share, for comparison
"""
import sys
import numpy as np

TABLES = 144
FLOOR = 0.2
IL_RULE = 'fixed' if '--il-fixed' in sys.argv else 'reciprocal'


def il_share(n_scored, rule=None):
    rule = rule or IL_RULE
    if n_scored == 0:
        return 1.0
    return 0.2 if rule == 'fixed' else 1.0 / (n_scored + 2)   # k = n_scored + 1


def logit(x):
    return np.log(x / (1 - x))


class PFSP:
    def __init__(self, half_life=5.0, prior_n=10.0, power=2.0, min_count=0.0):
        self.d = 0.5 ** (1.0 / half_life)
        self.prior_n = prior_n
        self.power = power
        self.min_count = min_count
        self.S, self.C, self.ids, self.age = [], [], [], []

    def admit(self, mid):
        self.S.append(0.5 * self.prior_n); self.C.append(self.prior_n)
        self.ids.append(mid); self.age.append(0)

    def x(self):
        return np.array(self.S) / np.array(self.C)

    def probs(self):
        x = self.x(); w = (1 - x) ** self.power; n = len(x)
        tot = w.sum()
        core = w / tot if tot > 0 else np.full(n, 1.0 / n)
        return (1 - FLOOR) * core + FLOOR / n

    def update(self, wins, counts):
        self.S = list(np.array(self.S) * self.d + wins)
        self.C = list(np.array(self.C) * self.d + counts)
        self.age = [a + 1 for a in self.age]

    def evict(self, protect_newest=2, min_age=20):
        n = len(self.ids)
        x = self.x(); C = np.array(self.C)
        order = np.argsort(self.age)  # youngest first
        protected = set(order[:protect_newest].tolist())
        elig = [i for i in range(n) if i not in protected and self.age[i] >= min_age and C[i] >= self.min_count]
        if not elig:
            return None
        k = max(elig, key=lambda i: x[i])
        for arr in (self.S, self.C, self.ids, self.age):
            del arr[k]
        return k


class Quality:
    def __init__(self, eta=0.01):
        self.eta = eta
        self.q, self.ids, self.age = [], [], []

    def admit(self, mid):
        self.q.append(max(self.q) if self.q else 0.0); self.ids.append(mid); self.age.append(0)

    def probs(self):
        q = np.array(self.q); e = np.exp(q - q.max()); n = len(q)
        return (1 - FLOOR) * e / e.sum() + FLOOR / n

    def update_with(self, member_idx, outcomes, p_used):
        n = len(self.q); q = np.array(self.q)
        np.add.at(q, member_idx, -self.eta * outcomes / (n * p_used[member_idx]))
        q -= q.max(); self.q = list(q)
        self.age = [a + 1 for a in self.age]

    def evict(self, protect_newest=2, min_age=20):
        n = len(self.ids)
        order = np.argsort(self.age)
        protected = set(order[:protect_newest].tolist())
        elig = [i for i in range(n) if i not in protected and self.age[i] >= min_age]
        if not elig:
            return None
        k = min(elig, key=lambda i: self.q[i])
        for arr in (self.q, self.ids, self.age):
            del arr[k]
        return k


def play_iteration(rng, p, theta_members, theta_learner, theta_il, il_p):
    """Return per-comparison (member index, or -1 for IL; outcome) arrays."""
    n = len(p)
    seats = rng.random((TABLES, 2)) < il_p
    draws = rng.choice(n, size=(TABLES, 2), p=p)
    member = np.where(seats, -1, draws)
    th = np.where(member >= 0, theta_members[np.clip(member, 0, None)], theta_il)
    uS = th + rng.gumbel(size=(TABLES, 2))
    uL = theta_learner + rng.gumbel(size=(TABLES, 2))
    out = (uL[:, :, None] > uS[:, None, :]).astype(float)  # [table, learner seat, snapshot seat]
    mem = np.broadcast_to(member[:, None, :], out.shape)
    return mem.reshape(-1), out.reshape(-1)


def static_precision(N, iters=400, burn=60, seed=0, half_life=5.0):
    rng = np.random.default_rng(seed)
    x_true = np.linspace(0.55, 0.92, N)
    theta_m = -logit(x_true)
    s = PFSP(half_life=half_life)
    for i in range(N):
        s.admit(i)
    errs, shares, counts, ncomp, il_n = [], [], [], [], []
    for t in range(iters):
        p = s.probs()
        mem, out = play_iteration(rng, p, theta_m, 0.0, -logit(0.95), il_share(N))
        sel = mem >= 0
        wins = np.bincount(mem[sel], weights=out[sel], minlength=N)
        cnt = np.bincount(mem[sel], minlength=N).astype(float)
        s.update(wins, cnt)
        if t >= burn:
            errs.append(s.x() - x_true); shares.append(p); counts.append(np.array(s.C)); ncomp.append(cnt)
            il_n.append((~sel).sum())
    errs, shares, counts, ncomp = map(np.array, (errs, shares, counts, ncomp))
    rmse = np.sqrt((errs ** 2).mean(0))
    d = s.d
    pred = np.sqrt(x_true * (1 - x_true) / ncomp.mean(0) * (1 - d) / (1 + d))
    return dict(N=N, x_true=x_true, rmse=rmse, pred=pred, share=shares.mean(0), C=counts.mean(0),
                Cmin=counts.min(0), n_iter=ncomp.mean(0), eff=(1.0 / (shares ** 2).sum(1)).mean(),
                max_share=shares.max(1).mean(), scored=ncomp.sum(1).mean(), il=np.mean(il_n))


def dynamic(method, cap=16, iters=400, seed=1, kappa=0.035, forget_at=250, forget_age=60, half_life=5.0):
    """Learner improves linearly; members frozen at admission. One old member is 'forgotten'."""
    rng = np.random.default_rng(seed)
    s = PFSP(half_life=half_life) if method == 'pfsp' else Quality()
    theta_of, next_id, forgotten, log = {}, 0, None, []
    for t in range(iters):
        theta_L = kappa * t
        if t >= 15 and (t - 15) % 5 == 0:          # one admission every 5 iterations from 15
            if len(s.ids) >= cap:
                s.evict()
            if len(s.ids) < cap:
                theta_of[next_id] = theta_L
                s.admit(next_id); next_id += 1
        if not s.ids:
            continue
        if t == forget_at:
            k = int(np.argmin(np.abs(np.array(s.age) - forget_age)))
            forgotten = s.ids[k]
        if forgotten is not None and t >= forget_at:
            theta_of[forgotten] = theta_L               # learner can no longer beat it (x = 0.5)
        th = np.array([theta_of[i] for i in s.ids])
        p = s.probs()
        il_p = il_share(len(s.ids))
        mem, out = play_iteration(rng, p, th, theta_L, theta_L - logit(0.95), il_p)
        sel = mem >= 0
        n = len(s.ids)
        cnt = np.bincount(mem[sel], minlength=n).astype(float)
        if method == 'pfsp':
            wins = np.bincount(mem[sel], weights=out[sel], minlength=n)
            s.update(wins, cnt)
        else:
            s.update_with(mem[sel], out[sel], p)
        x_true = 1 / (1 + np.exp(-(theta_L - th)))
        f_idx = s.ids.index(forgotten) if (forgotten is not None and forgotten in s.ids) else None
        log.append(dict(t=t, p=p, ages=np.array(s.age), x_true=x_true, cnt=cnt, il_p=il_p,
                        f_share=p[f_idx] if f_idx is not None else None,
                        f_in_pool=(f_idx is not None) if forgotten is not None else None,
                        C=np.array(s.C) if method == 'pfsp' else None))
    return log


def summarize(method, cap, seeds):
    rows = []
    for seed in seeds:
        log = dynamic(method, cap=cap, seed=seed)
        steady = [r for r in log if 150 <= r['t'] < 250]
        post = [r for r in log if r['t'] >= 250]
        sh = np.array([r['f_share'] if r['f_in_pool'] else np.nan for r in post])
        tail = sh[30:80]                     # NaN once the forgotten member has been evicted
        start, plateau = sh[0], (np.nanmean(tail) if np.isfinite(tail).any() else np.nan)
        frac = (sh - start) / (plateau - start)
        sm = np.convolve(np.nan_to_num(frac), np.ones(3) / 3, mode='valid')   # 3-iteration smoothing
        first = [r['cnt'][r['ages'] == 1] for r in log if r['t'] >= 100 and (r['ages'] == 1).any()]
        rows.append(dict(
            young=np.mean([r['p'][r['ages'] <= 20].sum() for r in steady]),
            old=np.mean([r['p'][r['ages'] > 50].sum() for r in steady]),
            eff=np.mean([1 / (r['p'] ** 2).sum() for r in steady]),
            mx=np.mean([r['p'].max() for r in steady]),
            traj=np.nan_to_num(sh), evicted=bool(np.isnan(sh).any()),
            half=int(np.argmax(sm >= 0.5)) + 1, ninety=int(np.argmax(sm >= 0.9)) + 1,
            first=np.concatenate(first) if first else np.array([]),
            floor_cnt=np.mean([r['cnt'][r['ages'] >= 20].min() for r in steady if (r['ages'] >= 20).any()]),
            Cmin=(np.mean([r['C'][r['ages'] >= 20].min() for r in steady if (r['ages'] >= 20).any()])
                  if method == 'pfsp' else np.nan),
            il_p=steady[-1]['il_p']))
    return rows


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    print(f"IL rule: {IL_RULE}")
    for N in (16, 23):
        r = static_precision(N)
        de = (r['rmse'] / r['pred']) ** 2
        print(f"\n=== static pool, {N} scored (IL share {il_share(N):.3f}): scored comparisons/iter {r['scored']:.0f}, "
              f"IL comparisons/iter {r['il']:.0f}, eff members {r['eff']:.1f}, mean max share {r['max_share']:.3f}, "
              f"design effect median {np.median(de):.2f}")
        for i in [0, N // 2, N - 2, N - 1]:
            print(f"  x={r['x_true'][i]:.3f} share={r['share'][i]:.3f} n/iter={r['n_iter'][i]:5.1f} "
                  f"C={r['C'][i]:6.1f} (min {r['Cmin'][i]:5.1f}) rmse={r['rmse'][i]:.3f}")
    for hl in (3, 5, 8):
        r = static_precision(16, half_life=hl)
        print(f"half-life {hl}: median rmse {np.median(r['rmse']):.3f}")
    for cap in (16, 23):
        for method, seeds in (('pfsp', range(16)), ('quality', range(16))):
            rows = summarize(method, cap, seeds)
            m = lambda k: np.mean([x[k] for x in rows])
            traj = np.mean([np.pad(x['traj'], (0, 150 - len(x['traj'])))[:150] for x in rows], axis=0)
            first = np.concatenate([x['first'] for x in rows])
            print(f"\n=== dynamic, cap {cap}, {method} (16 seeds; steady-state IL share {rows[0]['il_p']:.3f}): "
                  f"share age<=20 {m('young'):.2f}, age>50 {m('old'):.2f}, eff {m('eff'):.1f}, max {m('mx'):.2f}")
            print("  forgotten-member share at event, +5, +10, +20:", np.round(traj[[0, 5, 10, 20]], 3),
                  f"| evicted in {sum(x['evicted'] for x in rows)}/16 runs")
            if method == 'pfsp':
                h = sorted(x['half'] for x in rows); n9 = sorted(x['ninety'] for x in rows)
                print(f"  halfway iters {h[0]}-{h[-1]} (median {np.median(h):.0f}); 90% iters {n9[0]}-{n9[-1]} "
                      f"(median {np.median(n9):.0f})")
                print(f"  new member first-iteration comparisons: mean {first.mean():.0f}, min {first.min():.0f}; "
                      f"least-sampled member comparisons/iter {m('floor_cnt'):.1f}; "
                      f"min decayed count among age>=20: {m('Cmin'):.1f}")