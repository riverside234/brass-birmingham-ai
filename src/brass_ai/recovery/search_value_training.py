"""Isolated search residual optimisation, deterministic gates and PPG rollback.

Production invokes this lifecycle only after B05/B07 adoption. The qualification
uses the same objects against an isolated copy and never the ordinary datasets.
"""
import copy
import time
from collections import defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import digest, require
from brass_ai.learning.search_value import CONTRACT, correction, install
from .local_scoring_auxiliary import bundle_id


def clustered_gate(rows, main, delta, *, seed, use_root=False):
    require(len(rows) == len(main) == len(delta), 'Gate prediction count differs')
    groups = defaultdict(list)
    phases = defaultdict(list)
    phase_games = defaultdict(lambda: defaultdict(list))
    generations = defaultdict(set)
    for row, v, d in zip(rows, main, delta, strict=True):
        require(row['split'] == 'heldout', 'Gate data must be held out')
        u = np.asarray(row['u'])
        candidate = np.asarray(row['r']) if use_root else np.asarray(v) + np.asarray(d)
        difference = float(np.mean((u-v)**2 - (u-candidate)**2))
        require(np.isfinite(difference), 'Nonfinite gate error')
        groups[row['game_id']].append(difference)
        phases[row['phase']].append(difference)
        phase_games[row['phase']][row['game_id']].append(difference)
        generations[row['generation']].add(row['game_id'])
    ages = sorted(generations)
    full = (len(ages) == 8 and ages == list(range(ages[0], ages[0]+8)) and
            all(len(g) == 4 for g in generations.values()) and len(groups) == 32 and
            all(len(values) == 64 for values in groups.values())) if ages else False
    means = np.asarray([np.mean(groups[k]) for k in sorted(groups)], dtype=float)
    if not full:
        return dict(passed=False, full_window=False, games=len(groups), mean=float(means.mean()) if len(means) else None,
                    lower=None, upper=None, by_phase={p: float(np.mean(v)) for p, v in phases.items()})
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(len(means), size=(200, len(means)))].mean(1)
    lower, upper = np.quantile(boot, [.025, .975])
    by_phase = {}
    for phase, games in sorted(phase_games.items()):
        values = np.asarray([np.mean(games[k]) for k in sorted(games)])
        resamples = values[rng.integers(len(values),size=(200,len(values)))].mean(1)
        lo, hi = np.quantile(resamples,[.025,.975])
        by_phase[phase] = dict(mean=float(values.mean()),lower=float(lo),upper=float(hi))
    return dict(passed=bool(lower > 0), full_window=True, games=32, mean=float(means.mean()),
                lower=float(lower), upper=float(upper), by_phase=by_phase)


def h2h_verdict(comparisons):
    require(len(comparisons) == 256 and all(np.isfinite(x) and 0 <= x <= 1 for x in comparisons), 'Head-to-head needs 256 games')
    rng = np.random.default_rng(4504)
    values = np.asarray(comparisons)
    # Four rotations share a setup; keep those correlated games together.
    clusters = values.reshape(64,4).mean(1)
    boot = clusters[rng.integers(64, size=(2000, 64))].mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(passed=bool(values.mean() >= .5 and hi >= .5), mean=float(values.mean()),
                lower=float(lo), upper=float(hi), games=256)


class StageMachine:
    def __init__(self, *, saved=None, startup='immediate', options=None):
        self.state = copy.deepcopy(saved or dict(contract=CONTRACT, stage='A', w=0., scale=0.,
            generation=-1, lam=None, startup=startup, g0_passed=startup == 'immediate',
            history=[], ordinary=[], baseline=None, full_scale_generation=None, h2h=None))
        require(self.state['contract'] == CONTRACT and self.state['stage'] in ('off','A','B','head_to_head','C'), 'Invalid stage checkpoint')
        from brass_ai.learning.search_value import binding
        binding(self.state['w'])
        require(0 <= self.state['scale'] <= 1 and self.state['lam'] in (None,.5,.8,.95,1.), 'Invalid resumed stage parameters')
        if options is not None and 'schedule' in options:
            from .search_value_schedule import initialize
            initialize(self.state,options)

    def choose_lambda(self, value):
        require(value in (.5,.8,.95,1.) and self.state['lam'] in (None,value), 'Lambda changes require reviewed buffer retirement')
        self.state['lam'] = value

    def manual(self, stage):
        order = ['off', 'A', 'B', 'head_to_head', 'C']
        require(stage in order and order.index(stage) <= order.index(self.state['stage']), 'Manual advancement prohibited')
        self.state.update(stage=stage, scale=0., full_scale_generation=None)
        if stage != 'C': self.state['baseline'] = None
        if stage in ('off','A'): self.state['w'] = 0.

    def advance(self, generation, gate, *, ordinary=None, h2h=None, annealing=False, pause=False):
        s = self.state
        from .search_value_schedule import CONTRACT as SCHEDULE,advance
        if s.get('schedule')==SCHEDULE:return advance(s,generation,gate,pause=pause)
        require(type(generation) is int and generation > s['generation'], 'Stage transition already committed/out of order')
        s['generation'] = generation
        if ordinary is not None:
            require(ordinary and all(np.isfinite(v) for v in ordinary.values()), 'Invalid ordinary guard metrics')
            s['ordinary'] = (s['ordinary'] + [dict(generation=generation, values=copy.deepcopy(ordinary))])[-40:]
        s['history'] = (s['history'] + [dict(generation=generation, gate=copy.deepcopy(gate))])[-40:]
        if annealing or s['stage'] == 'off': return copy.deepcopy(s)
        if not s['g0_passed']:
            s['g0_passed'] = bool(gate['passed'])
            return copy.deepcopy(s)
        stage = s['stage']
        if stage == 'A' and gate['passed']: s['stage'] = 'B'
        elif stage in ('B','head_to_head'):
            if gate.get('upper') is not None and gate['upper'] < 0:
                s.update(stage='B', w=0., scale=0.)
            elif stage == 'B' and gate['passed']:
                s['w'] = min(1., round(s['w']+.05, 2))
                if s['startup'] == 'shadow-until-search-informative': s['scale'] = min(1., round(s['scale']+.1, 2))
                if s['w'] == 1: s['stage'] = 'head_to_head'
            elif stage == 'head_to_head' and h2h is not None:
                require(h2h.get('games') == 256, 'Incomplete head-to-head')
                s['h2h'] = copy.deepcopy(h2h)
                if h2h['upper'] < .5: s.update(stage='B', w=0., scale=0.)
                elif h2h['passed'] and gate['passed'] and len(s['ordinary']) >= 20:
                    baseline = s['ordinary'][-20:]
                    names = set(baseline[0]['values'])
                    require(all(set(r['values']) == names for r in baseline), 'Ordinary guard metric set changed')
                    s.update(stage='C', scale=0., full_scale_generation=None, baseline=dict(g1=gate['mean'],
                        bounds={k:[min(r['values'][k] for r in baseline),max(r['values'][k] for r in baseline)] for k in names}))
                elif not h2h['passed']: s['stage'] = 'B'
        elif stage == 'C':
            s['scale'] = min(1., round(s['scale']+.1, 2))
            if s['scale'] == 1 and s['full_scale_generation'] is None: s['full_scale_generation'] = generation
            if s['full_scale_generation'] is not None:
                age = generation-s['full_scale_generation']
                bounds = s['baseline']['bounds']
                guard = ordinary is not None and set(ordinary) == set(bounds) and all(lo <= ordinary[k] <= hi for k,(lo,hi) in bounds.items())
                g2 = age < 20 or (gate.get('mean') is not None and gate['mean'] >= s['baseline']['g1'])
                if (age <= 20 and not guard) or not g2:
                    s.update(stage='B', scale=0., full_scale_generation=None)
        return copy.deepcopy(s)


def _batch(model, rows):
    device = next(model.parameters()).device
    state = torch.as_tensor(np.asarray([r['state'] for r in rows]), dtype=torch.float32, device=device)
    mapping = torch.as_tensor(np.asarray([r['mapping'] for r in rows]), dtype=torch.long, device=device)
    remaining = torch.as_tensor(np.asarray([r['remaining'] for r in rows]), dtype=torch.float32, device=device)
    return state, mapping, remaining


def predict_rows(model, rows):
    main, residual = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 512):
            state, mapping, remaining = _batch(model, rows[start:start+512])
            encoded = model.encode_state(state)
            main.extend(model.heads_from_encoded(encoded,mapping,remaining)[-1].float().cpu().tolist())
            residual.extend(correction(model,encoded,mapping,remaining).float().cpu().tolist())
    return np.asarray(main), np.asarray(residual)


class Companion:
    def __init__(self, model, *, saved=None, seed=4504):
        install(model, seed=seed)
        self.model, self.seed = model, seed
        self.optimizer = torch.optim.Adam(model.search_value_head.parameters(), lr=1e-4, weight_decay=1e-4)
        self.steps = 0
        if saved is not None: self.restore(saved)

    def snapshot(self):
        result = dict(contract=CONTRACT, seed=self.seed, steps=self.steps,
                      optimizer=copy.deepcopy(self.optimizer.state_dict()))
        result['identity'] = bundle_id(result)
        return result

    def restore(self, saved):
        require(saved['contract'] == CONTRACT and saved['seed'] == self.seed and
                saved['identity'] == bundle_id({k:v for k,v in saved.items() if k != 'identity'}), 'Companion checkpoint changed')
        self.optimizer.load_state_dict(saved['optimizer'])
        self.steps = saved['steps']
        require(type(self.steps) is int and self.steps >= 0 and len(self.optimizer.param_groups)==1 and
                self.optimizer.param_groups[0]['lr']==1e-4 and self.optimizer.param_groups[0]['weight_decay']==1e-4,
                'Invalid search companion Adam settings')
        states=list(self.optimizer.state.values())
        require(len(states) == (len(list(self.model.search_value_head.parameters())) if self.steps else 0)
                and all(set(s) == {'step','exp_avg','exp_avg_sq'} and int(s['step']) == self.steps
                        and all(torch.isfinite(v).all() for v in s.values()) for s in states),
                'Missing/invalid search companion Adam moments')
        require(all(s['exp_avg'].shape == s['exp_avg_sq'].shape == p.shape
                    for p,s in self.optimizer.state.items()), 'Search companion Adam moment shape differs')

    def head_only(self, rows, *, generation, steps=32):
        began=time.perf_counter()
        require(rows and all(r['split'] == 'train' for r in rows) and 1 <= steps <= 32, 'Invalid head-only training data')
        model = self.model
        # Encode once per pass, then run all 32 head steps on cached features.
        # No repeated trunk work, policy candidates, replay collation or GAE.
        inputs, targets, mains, mappings = [], [], [], []
        with torch.no_grad():
            for start in range(0,len(rows),512):
                batch = rows[start:start+512]
                state, mapping, remaining = _batch(model,batch)
                encoded = model.encode_state(state)
                emb = model.descriptor_embeddings(encoded,None).mean(1)
                inputs.append(torch.cat((encoded.context,remaining.gather(1,mapping.argsort(1)),emb),-1))
                mains.append(model.heads_from_encoded(encoded,mapping,remaining)[-1])
                mappings.append(mapping)
                targets.append(state.new_tensor([r['target'] for r in batch]))
        x, base, mapping, target = map(torch.cat,(inputs,mains,mappings,targets))
        encoded_at=time.perf_counter()
        rng = np.random.default_rng(int(digest([self.seed,generation,self.steps])[:16],16))
        losses = []
        model.zero_grad(set_to_none=True)
        for _ in range(steps):
            ix = torch.as_tensor(rng.choice(len(rows),min(512,len(rows)),replace=False),device=x.device)
            self.optimizer.zero_grad(set_to_none=True)
            delta = model.search_value_head(x[ix]).gather(1,mapping[ix])
            loss = torch.nn.functional.mse_loss(base[ix]+delta,target[ix])
            loss.backward()
            require(all(p.grad is None for n,p in model.named_parameters() if not n.startswith('search_value_head.')), 'Search loss reached ordinary parameters')
            torch.nn.utils.clip_grad_norm_(model.search_value_head.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step(); self.steps += 1; losses.append(loss.detach())
        self.optimizer.zero_grad(set_to_none=True)
        # Preserve every observed loss without 32 separate device-to-host reads.
        losses=torch.stack(losses).float().cpu().tolist()
        return dict(steps=steps, rows=len(rows), loss_before=losses[0], loss_after=losses[-1], trunk_gradient=0., policy_kl=0.,
            timings=dict(input_and_frozen_encoder_seconds=encoded_at-began,head_update_seconds=time.perf_counter()-encoded_at,
                total_seconds=time.perf_counter()-began))


def guarded_pass(companion, rows, heldout, value_optimizer, *, scale, measure_ordinary,
                 cloning_loss, measure_joint_kl, beta, rate_for_step, kl_cap=.002):
    """After ordinary PPG; callbacks use its existing frozen probe/cache.

    measure_joint_kl compares to the *pre-PPG* policy, never a reset reference.
    The current g1250 cap is .003; B04.5 still applies min(.002, ordinary cap).
    """
    from .ppg_selection import Selection
    require(rows and heldout and all(r['split']=='train' for r in rows) and
            all(r['split']=='heldout' for r in heldout) and 0 < scale <= 1, 'Invalid guarded search pass')
    model = companion.model
    head_ids = {id(p) for p in model.search_value_head.parameters()}
    require(not head_ids & {id(p) for g in value_optimizer.param_groups for p in g['params']}, 'Search head has two Adam owners')
    cap = min(.002, kl_cap)
    if measure_joint_kl() >= cap: return dict(intervals=[], reason='No shared PPG KL budget')
    def held_loss():
        main, delta = predict_rows(model,heldout)
        return float(np.mean((main+delta-np.asarray([r['target'] for r in heldout]))**2))
    reference = measure_ordinary()
    reports = []
    rng = np.random.default_rng(companion.seed+companion.steps)
    for interval in range(2):
        before = held_loss()
        rollback = Selection(model,value_optimizer,{'search':before},companion=companion)
        try:
            for step in range(16):
                selected = [rows[i] for i in rng.choice(len(rows),min(512,len(rows)),replace=False)]
                state, mapping, remaining = _batch(model,selected)
                encoded = model.encode_state(state)
                with torch.no_grad(): main = model.heads_from_encoded(encoded,mapping,remaining)[-1]
                delta = correction(model,encoded,mapping,remaining,scale=scale)
                loss = torch.nn.functional.mse_loss(main+delta,state.new_tensor([r['target'] for r in selected]))
                value_optimizer.zero_grad(set_to_none=True); companion.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                require(all(p.grad is None for n,p in model.named_parameters() if n.startswith(('outcomes.','utility_head.'))), 'Correction reached main value head')
                (beta*cloning_loss()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                for group in value_optimizer.param_groups: group['lr'] = rate_for_step(interval*16+step+1,32)
                value_optimizer.step(); companion.optimizer.step(); companion.steps += 1
            after, ordinary, kl = held_loss(), measure_ordinary(), measure_joint_kl()
            accepted = bool(np.isfinite([after,ordinary,kl]).all() and after < before and ordinary <= reference+.013 and kl <= cap)
            reports.append(dict(accepted=accepted,search_before=before,search_after=after,ordinary_delta=ordinary-reference,joint_kl=kl,cap=cap))
            if not accepted:
                rollback.restore(model,value_optimizer); break
        except BaseException:
            rollback.restore(model,value_optimizer); raise
        finally:
            model.zero_grad(set_to_none=True)
    return dict(intervals=reports)


def unconditional_pass(companion,rows,value_optimizer,*,scale,cloning_loss,measure_joint_kl,
                       beta,rate_for_step,kl_cap=.003,heldout=None,**unused):
    """Unconditional stage schedule, with interval-level KL/loss retention.

    A rejected interval restores both Adams and weights. Stage counters never
    roll back. Exceptions retain the normal transactional recovery.
    """
    from .ppg_selection import Selection
    require(rows and all(r['split']=='train' for r in rows) and 0<scale<=1 and kl_cap==.003,
        'Invalid unconditional trunk pass')
    model=companion.model;head_ids={id(p) for p in model.search_value_head.parameters()}
    require(not head_ids & {id(p) for g in value_optimizer.param_groups for p in g['params']},'Search head has two Adam owners')
    initial=measure_joint_kl()
    require(np.isfinite(initial) and initial>=0,'Nonfinite initial search/PPG joint KL')
    if initial>=kl_cap:return dict(intervals=[],steps=0,reason='No shared PPG KL budget',cap=kl_cap)
    held_loss = None
    if heldout is not None:
        from .search_value_retention import HeldoutLoss
        require({r['game_id'] for r in rows}.isdisjoint(r['game_id'] for r in heldout),
            'Stage-C training/held-out game leakage')
        held_loss = HeldoutLoss(model,heldout)
    current_loss = initial_loss = held_loss() if held_loss is not None else None
    # Inputs are fixed, but trunk encodings must be recomputed after each
    # optimizer step. Pack numeric rows once, retaining exact sample order/RNG.
    states=np.asarray([r['state'] for r in rows],dtype=np.float32)
    mappings=np.asarray([r['mapping'] for r in rows],dtype=np.int64)
    remaining_costs=np.asarray([r['remaining'] for r in rows],dtype=np.float32)
    targets=np.asarray([r['target'] for r in rows],dtype=np.float32)
    device=next(model.parameters()).device
    rng=np.random.default_rng(companion.seed+companion.steps);reports=[];steps=0
    for interval in range(2):
        rollback=Selection(model,value_optimizer,{'search':0.},companion=companion)
        try:
            for step in range(16):
                selected=rng.choice(len(rows),min(512,len(rows)),replace=False)
                state,mapping,remaining=(torch.as_tensor(a[selected],device=device) for a in (states,mappings,remaining_costs))
                encoded=model.encode_state(state)
                with torch.no_grad():main=model.heads_from_encoded(encoded,mapping,remaining)[-1]
                delta=correction(model,encoded,mapping,remaining,scale=scale)
                loss=torch.nn.functional.mse_loss(main+delta,torch.as_tensor(targets[selected],device=device))
                value_optimizer.zero_grad(set_to_none=True);companion.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                require(all(p.grad is None for n,p in model.named_parameters() if n.startswith(('outcomes.','utility_head.'))),
                    'Correction reached main value head')
                (beta*cloning_loss()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                for group in value_optimizer.param_groups:group['lr']=rate_for_step(interval*16+step+1,32)
                value_optimizer.step();companion.optimizer.step();companion.steps+=1
            kl=measure_joint_kl();require(np.isfinite(kl) and kl>=0,'Nonfinite search/PPG joint KL')
            attempted_loss=held_loss() if held_loss is not None and kl<=kl_cap else None
            loss_increased=attempted_loss is not None and attempted_loss>current_loss
            accepted=kl<=kl_cap and not loss_increased
            reason='joint-kl-cap' if kl>kl_cap else 'heldout_loss_increased' if loss_increased else 'within-joint-kl-cap'
            reports.append(dict(accepted=accepted,joint_kl=kl,cap=kl_cap,scale=scale,
                reason=reason,heldout_loss_before=current_loss,heldout_loss_attempted=attempted_loss))
            if not accepted:rollback.restore(model,value_optimizer);break
            if attempted_loss is not None:current_loss=attempted_loss
            steps+=16
        except BaseException:
            rollback.restore(model,value_optimizer);raise
        finally:model.zero_grad(set_to_none=True)
    return dict(intervals=reports,steps=steps,cap=kl_cap,validation_gates=held_loss is not None,
        heldout_loss_before=initial_loss,heldout_loss_after=current_loss,
        early_stop_reason=reports[-1]['reason'] if reports and not reports[-1]['accepted'] else None)
