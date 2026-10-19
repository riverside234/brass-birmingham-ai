"""PFSP (EMA outrank) vs OpenAI Five quality-score sampler on the 2+2 history tables.

Model: each seat's finishing utility is theta + Gumbel noise, so the pairwise outrank
probability is sigmoid(theta_L - theta_j) (Plackett-Luce). 144 history tables per
iteration, 2 learner seats + 2 snapshot seats. Each snapshot seat goes to IL with
probability 1/(k+1), where k = models in the pool including IL (every seat while IL is
alone); otherwise it is an independent draw from the sampler.

Derived from the supplied script, archived byte-for-byte in
ai-research/simulations/pfsp_sim_original.py. See PFSP_SIMULATION.md there.
This remains a synthetic sampler comparison, not a full trainer simulation.

Run: python scripts/pfsp_sim.py --history-share 0.45 --output results/pfsp-45.json
     python scripts/pfsp_sim.py --il-fixed --min-count 0
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

TABLES = 144
FLOOR = 0.2
IL_RULE = 'reciprocal'


def il_share(n_scored, rule=None):
    rule = rule or IL_RULE
    if n_scored == 0:
        return 1.0
    return 0.2 if rule == 'fixed' else 1.0 / (n_scored + 2)   # k = n_scored + 1


def logit(x):
    return np.log(x / (1 - x))


class PFSP:
    def __init__(self, half_life=5.0, prior_n=10.0, power=2.0, min_count=15.0):
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
        if not n:
            return np.array([], dtype=float)
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
        q = np.array(self.q); n = len(q)
        if not n:
            return np.array([], dtype=float)
        e = np.exp(q - q.max())
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


def play_iteration(rng, p, theta_members, theta_learner, theta_il, il_p, *, tables=TABLES):
    """Return per-comparison (member index, or -1 for IL; outcome) arrays."""
    n = len(p)
    if not n:
        if il_p != 1.0:
            raise ValueError('An empty scored pool must give every snapshot seat to IL')
        member = np.full((tables, 2), -1, dtype=int)
        th = np.full((tables, 2), theta_il)
    else:
        seats = rng.random((tables, 2)) < il_p
        draws = rng.choice(n, size=(tables, 2), p=p)
        member = np.where(seats, -1, draws)
        th = np.where(member >= 0, theta_members[np.clip(member, 0, None)], theta_il)
    uS = th + rng.gumbel(size=(tables, 2))
    uL = theta_learner + rng.gumbel(size=(tables, 2))
    out = (uL[:, :, None] > uS[:, None, :]).astype(float)  # [table, learner seat, snapshot seat]
    mem = np.broadcast_to(member[:, None, :], out.shape)
    return mem.reshape(-1), out.reshape(-1)


def static_precision(N, iters=400, burn=60, seed=0, half_life=5.0, *, tables=TABLES, il_rule=IL_RULE):
    rng = np.random.default_rng(seed)
    x_true = np.linspace(0.55, 0.92, N)
    theta_m = -logit(x_true)
    s = PFSP(half_life=half_life)
    for i in range(N):
        s.admit(i)
    errs, shares, counts, ncomp, il_n = [], [], [], [], []
    for t in range(iters):
        p = s.probs()
        mem, out = play_iteration(rng, p, theta_m, 0.0, -logit(0.95), il_share(N, il_rule), tables=tables)
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
    # Very small table budgets may leave a member unobserved in the measurement
    # window. Report its precision prediction as unavailable, not infinity.
    mean_n = ncomp.mean(0)
    pred = np.sqrt(np.divide(x_true * (1 - x_true), mean_n,
        out=np.full(N, np.nan), where=mean_n > 0) * (1 - d) / (1 + d))
    return dict(N=N, x_true=x_true, rmse=rmse, pred=pred, share=shares.mean(0), C=counts.mean(0),
                Cmin=counts.min(0), n_iter=ncomp.mean(0), eff=(1.0 / (shares ** 2).sum(1)).mean(),
                max_share=shares.max(1).mean(), scored=ncomp.sum(1).mean(), il=np.mean(il_n))


def dynamic(method, cap=16, iters=400, seed=1, kappa=0.035, forget_at=250, forget_age=60, half_life=5.0,
            *, tables=TABLES, il_rule=IL_RULE, min_count=15.0):
    """Learner improves linearly; members frozen at admission. One old member is 'forgotten'."""
    rng = np.random.default_rng(seed)
    if method not in ('pfsp', 'quality'):
        raise ValueError('Unknown sampler')
    s = PFSP(half_life=half_life, min_count=min_count) if method == 'pfsp' else Quality()
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
            # Preserve the original RNG sequence. Startup is outside this
            # scored-member experiment; no claim about IL-only games is made.
            continue
        if t == forget_at:
            k = int(np.argmin(np.abs(np.array(s.age) - forget_age)))
            forgotten = s.ids[k]
        if forgotten is not None and t >= forget_at:
            theta_of[forgotten] = theta_L               # learner can no longer beat it (x = 0.5)
        th = np.array([theta_of[i] for i in s.ids])
        p = s.probs()
        il_p = il_share(len(s.ids), il_rule)
        mem, out = play_iteration(rng, p, th, theta_L, theta_L - logit(0.95), il_p, tables=tables)
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


def response_times(shares):
    """First trailing three-sample crossing, in elapsed iterations from event.

    None means unreached/undefined, including no positive plateau shift. Do not
    turn eviction/missing data into zero progress or argmax(False) into a hit.
    The event is index 0; a window over indices 0..2 is known only at index 2.
    """
    sh = np.asarray(shares, dtype=float)
    tail = sh[30:80]
    if not len(sh) or not np.isfinite(sh[0]) or not np.isfinite(tail).any():
        return None, None
    delta = np.nanmean(tail) - sh[0]
    if delta <= 0:
        return None, None
    frac = (sh - sh[0]) / delta
    sm = np.convolve(frac, np.ones(3) / 3, mode='valid')
    def first(threshold):
        hits = np.flatnonzero(np.isfinite(sm) & (sm >= threshold))
        return int(hits[0]) + 2 if len(hits) else None
    return first(.5), first(.9)


def summarize(method, cap, seeds, **options):
    rows = []
    for seed in seeds:
        log = dynamic(method, cap=cap, seed=seed, **options)
        steady = [r for r in log if 150 <= r['t'] < 250]
        post = [r for r in log if r['t'] >= 250]
        sh = np.array([r['f_share'] if r['f_in_pool'] else np.nan for r in post])
        half, ninety = response_times(sh)
        first = [r['cnt'][r['ages'] == 1] for r in log if r['t'] >= 100 and (r['ages'] == 1).any()]
        rows.append(dict(
            young=np.mean([r['p'][r['ages'] <= 20].sum() for r in steady]),
            old=np.mean([r['p'][r['ages'] > 50].sum() for r in steady]),
            eff=np.mean([1 / (r['p'] ** 2).sum() for r in steady]),
            mx=np.mean([r['p'].max() for r in steady]),
            traj=np.nan_to_num(sh), evicted=bool(np.isnan(sh).any()),
            half=half, ninety=ninety,
            first=np.concatenate(first) if first else np.array([]),
            floor_cnt=np.mean([r['cnt'][r['ages'] >= 20].min() for r in steady if (r['ages'] >= 20).any()]),
            Cmin=(np.mean([r['C'][r['ages'] >= 20].min() for r in steady if (r['ages'] >= 20).any()])
                  if method == 'pfsp' else np.nan),
            il_p=steady[-1]['il_p']))
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--games', type=int, default=320, help='Ordinary games per iteration')
    parser.add_argument('--history-share', type=float, default=.45, help='History table fraction, e.g. 0.30')
    parser.add_argument('--tables', type=int, help='Exact history tables per iteration; overrides games * share')
    parser.add_argument('--caps', type=int, nargs='+', default=[16, 23], help='Scored members, excluding IL')
    parser.add_argument('--seeds', type=int, default=16, help='Dynamic seeds 0 through seeds-1')
    parser.add_argument('--half-life', type=float, default=5.)
    parser.add_argument('--min-count', type=float, default=15., help='PFSP eviction support; original script used 0')
    parser.add_argument('--il-fixed', action='store_true', help='Use the old fixed 20% share (IL-only still 100%)')
    parser.add_argument('--output', type=Path, help='New JSON result file; never overwritten')
    args = parser.parse_args(argv)
    if args.games <= 0 or not 0 < args.history_share <= 1:
        parser.error('games must be positive and history-share must be in (0, 1]')
    if args.tables is None:
        tables = args.games * args.history_share
        if not np.isclose(tables, round(tables), rtol=0, atol=1e-9):
            parser.error('games * history-share must be integral; pass --tables with the actual planned count')
        args.tables = round(tables)
    if not 0 < args.tables <= args.games or args.seeds <= 0 or any(n < 2 or n > 23 for n in args.caps):
        parser.error('tables must be in 1..games; seeds positive; scored caps between 2 and 23')
    if not np.isfinite(args.half_life) or args.half_life <= 0 or not np.isfinite(args.min_count) or args.min_count < 0:
        parser.error('half-life must be finite and positive; min-count finite and nonnegative')
    if args.output is not None and args.output.exists():
        parser.error('Use a new output filename')
    return args


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(v) for key, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    np.set_printoptions(precision=3, suppress=True)
    rule = 'fixed' if args.il_fixed else 'reciprocal'
    common = dict(tables=args.tables, il_rule=rule, half_life=args.half_life)
    original = Path(__file__).resolve().parents[1] / 'ai-research/simulations/pfsp_sim_original.py'
    result = dict(contract='pfsp-simulation/1.0.0', scope='Synthetic correlated PL-seat experiment; no training or strength evaluation. Simplified admission/eviction and moving IL strength; see PFSP_SIMULATION.md.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(), numpy_version=np.__version__,
        settings=dict(games=args.games, requested_history_share=args.history_share, history_tables=args.tables,
            effective_history_share=args.tables/args.games, scored_caps=args.caps, seeds=args.seeds,
            il_rule=rule, half_life=args.half_life, min_decayed_eviction_count=args.min_count,
            uniform_floor=FLOOR, prior_comparisons=10, power=2, iterations=400, burn=60,
            steady_window=[150, 250], forget_at=250, forget_age=60, kappa=.035),
        static=[], sensitivity=[], dynamic=[])
    print(f"IL rule: {rule}; history tables/iteration: {args.tables}; eviction C >= {args.min_count:g}", flush=True)
    for N in args.caps:
        r = static_precision(N, **common)
        result['static'].append(r)
        de = (r['rmse'] / r['pred']) ** 2
        print(f"\n=== static pool, {N} scored (IL share {il_share(N, rule):.3f}): scored comparisons/iter {r['scored']:.0f}, "
              f"IL comparisons/iter {r['il']:.0f}, eff members {r['eff']:.1f}, mean max share {r['max_share']:.3f}, "
              f"design effect median {np.median(de):.2f}")
        for i in [0, N // 2, N - 2, N - 1]:
            print(f"  x={r['x_true'][i]:.3f} share={r['share'][i]:.3f} n/iter={r['n_iter'][i]:5.1f} "
                  f"C={r['C'][i]:6.1f} (min {r['Cmin'][i]:5.1f}) rmse={r['rmse'][i]:.3f}")
    for hl in (3, 5, 8):
        r = static_precision(16, **dict(common, half_life=hl))
        result['sensitivity'].append(dict(half_life=hl, **r))
        print(f"half-life {hl}: median rmse {np.median(r['rmse']):.3f}")
    for cap in args.caps:
        for method in ('pfsp', 'quality'):
            rows = summarize(method, cap, range(args.seeds), **common, min_count=args.min_count)
            result['dynamic'].append(dict(method=method, scored_cap=cap,
                runs=[dict(seed=seed, **row) for seed, row in enumerate(rows)]))
            m = lambda k: np.mean([x[k] for x in rows])
            traj = np.mean([np.pad(x['traj'], (0, 150 - len(x['traj'])))[:150] for x in rows], axis=0)
            first = np.concatenate([x['first'] for x in rows])
            print(f"\n=== dynamic, cap {cap}, {method} ({args.seeds} seeds; steady-state IL share {rows[0]['il_p']:.3f}): "
                  f"share age<=20 {m('young'):.2f}, age>50 {m('old'):.2f}, eff {m('eff'):.1f}, max {m('mx'):.2f}")
            print("  forgotten-member share at event, +5, +10, +20:", np.round(traj[[0, 5, 10, 20]], 3),
                  f"| evicted in {sum(x['evicted'] for x in rows)}/{args.seeds} runs")
            if method == 'pfsp':
                for key, label in (('half', 'halfway'), ('ninety', '90%')):
                    hits = sorted(x[key] for x in rows if x[key] is not None)
                    summary = f'{hits[0]}-{hits[-1]} (median {np.median(hits):.0f})' if hits else 'unavailable'
                    print(f'  {label} elapsed iters: {summary}; unreached/undefined {len(rows)-len(hits)}/{len(rows)}')
                first_summary = f'mean {first.mean():.0f}, min {first.min():.0f}' if len(first) else 'unavailable'
                print(f"  new member first-iteration comparisons: {first_summary}; "
                      f"least-sampled member comparisons/iter {m('floor_cnt'):.1f}; "
                      f"min decayed count among age>=20: {m('Cmin'):.1f}")
    result['elapsed_seconds'] = time.perf_counter() - started
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(json_ready(result), handle, indent=2, allow_nan=False)
            handle.write('\n')
        print(f'Saved {args.output}', flush=True)
    return result


if __name__ == '__main__':
    main()
