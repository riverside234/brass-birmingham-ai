"""B04.5 lifecycle kept outside ordinary replay, PPO and the ordinary PPG pass."""
import copy
import json
import math
import time
from collections import defaultdict

import numpy as np

from brass_ai.learning.contracts import digest, require
from .search_value_config import validate
from .search_value_rows import ValueBuffer, complete_game, copy_row
from .search_value_training import Companion, StageMachine, clustered_gate, predict_rows, guarded_pass


def label_snapshot(ages):
    """Detach JSON labels once at publication, rejecting nonfinite values."""
    return json.loads(json.dumps(ages, allow_nan=False, separators=(',', ':')))


class SearchValuePhase:
    def __init__(self, directory, model, options, reward_id, *, saved=None, startup='immediate', trainable=True, pending_files=(), pending_directory=None, defer_install=False):
        self.options = copy.deepcopy(validate(options))
        self.model, self.reward_id = model, reward_id
        saved = saved or {}
        self.machine = StageMachine(saved=saved.get('stage'), startup=startup,options=options)
        if options['stage'] == 'off': self.machine.manual('off')
        self.companion = (Companion(model, saved=saved.get('companion'))
                          if (trainable and options['stage'] != 'off' and not defer_install) or hasattr(model,'search_value_head') else None)
        self.trainable=trainable
        self.buffer = ValueBuffer(directory, reward_id, receipts=saved.get('buffer', {}).get('receipts', ()),
                                  retired=saved.get('buffer', {}).get('retired', ()),pending_files=pending_files,pending_directory=pending_directory)
        # Revised return fitting uses validated numeric value rows directly.
        # Policy staleness still belongs to TeacherTerm; duplicating its full
        # label views here both starved the head and inflated checkpoints.
        self.label_ages = [] if 'schedule' in options else copy.deepcopy(saved.get('label_ages', []))
        self.report = {}

    def ensure_companion(self):
        if self.companion is None and self.trainable and self.options['stage']!='off':
            self.companion=Companion(self.model)

    def admit(self, results, jobs, generation, *, post_spent=0.):
        import time
        began = time.monotonic()
        by_id = {r['job_id']: r for r in results}
        require(len(by_id) == len(results) == len(jobs) and
                set(by_id) == {j['expected']['job_id'] for j in jobs}, 'Missing/duplicate value table')
        spent = post_spent if isinstance(post_spent,dict) else {key:post_spent for key in by_id}
        require(set(spent)==set(by_id) and all(type(v) in (float,int) and math.isfinite(v) and v>=0
                for v in spent.values()), 'Invalid per-table post-search accounting')
        counts = defaultdict(int)
        timings, labels = {}, []
        self.report.update(rows=counts,value_admission_per_table_seconds=timings)
        for job in jobs:
            table_began = time.monotonic()
            result = by_id[job['expected']['job_id']]
            rows = complete_game(result['value_rows'], result['result'], result['final_spent'], job,
                                 lam=self.machine.state['lam'] or 1.)
            if rows[0].get('vp_payments'):
                self.report.setdefault('vp_payments_per_table',{})[job['expected']['game_id']]=rows[0]['vp_payments']
            # Keep the final utility so all eight ages can be retargeted exactly
            # once lambda is selected. No targets are inferred from policy labels.
            from brass_ai.learning.reward import terminal_utility
            terminal = terminal_utility(result['result'],job['search']['reward']).tolist()
            for row in rows: row['terminal_utility'] = terminal
            # Seal each complete table independently. Charging the entire batch
            # to every final turn makes identical work fail as lane count grows.
            self.buffer.publish(generation,job['split'],rows,game_id=job['expected']['game_id'])
            if job['split'] == 'train' and 'schedule' not in self.options: labels.extend(copy.deepcopy(result['rows']))
            counts[job['split']] += len(rows)
            elapsed = time.monotonic()-table_began
            timings[job['expected']['job_id']] = elapsed
            # Keep attribution and finite accounting, without a per-turn
            # deadline on independently checked label/shard publication.
        self.label_ages = [a for a in self.label_ages if 0 <= generation-a['generation'] < 8 and a['reward_id'] == self.reward_id]
        if 'schedule' not in self.options:
            self.label_ages.append(dict(generation=generation,reward_id=self.reward_id,rows=labels))
        self.report['rows'] = dict(counts)
        self.report['value_admission_seconds'] = time.monotonic()-began
        self.report['value_admission_per_table_seconds'] = timings
        self.report['verification_limit_seconds'] = None

    def stale_ages(self, generation):
        """Same per-age KL rule as the label buffer, including value ages 4–7."""
        from .teacher_labels import LabelRow
        from .encoding import PreparedEncoder
        from .teacher_term import policy_logits, kl, softmax
        stale, report = [], {}
        encoder = PreparedEncoder()
        for age in self.label_ages:
            rows = [LabelRow(r,encoder.encode(r['view'],r['choices'],[0.]*4,[0.]*8)) for r in age['rows']]
            if not rows:
                stale.append(generation-age['generation']); continue
            logits = policy_logits(self.model,rows)
            improve = np.mean([kl(r.label['target'],softmax(r.label['logits'])) for r in rows])
            drift = np.mean([kl(softmax(r.label['logits']),softmax(v)) for r,v in zip(rows,logits,strict=True)])
            n = generation-age['generation']
            if drift > improve: stale.append(n)
            report[str(n)] = dict(improvement=float(improve),drift=float(drift),skipped=n in stale)
        self.report['staleness'] = report
        return stale

    def _targets(self, rows):
        lam = self.machine.state['lam'] or 1.
        groups = defaultdict(list)
        for row in rows: groups[row['game_id']].append(copy_row(row))
        result = []
        for game in groups.values():
            game.sort(key=lambda r:r['index'])
            following = np.asarray(game[-1]['terminal_utility'])
            for i in reversed(range(len(game))):
                next_u = np.asarray(game[i+1]['u'] if i+1 < len(game) else game[i]['terminal_utility'])
                cost = next_u - game[i]['u']
                boot = np.asarray(game[i+1]['r'] if i+1 < len(game) else game[i]['terminal_utility'])
                following = -cost + (1-lam)*boot + lam*following
                game[i].update(target=following.tolist(),target_lambda=lam)
            result.extend(game)
        return result

    def after_ppg(self, generation, *, mode, ordinary=None, h2h=None, annealing=False, guarded=None):
        require(mode in ('off','shadow','on'), 'Invalid search phase mode')
        if mode != 'on' or self.options['stage']=='off':
            return dict(updated=False,reason='Search value off/shadow; no training')
        self.ensure_companion()
        if 'schedule' in self.options:return self._scheduled_after_ppg(generation,guarded=guarded)
        self.report.pop('update',None)
        state = self.machine.state
        heldout = self.buffer.rows(generation,'heldout')
        train = self._targets(self.buffer.rows(generation,'train',stale_ages=self.stale_ages(generation)))
        if train and state['g0_passed'] and not annealing:
            if state['scale'] > 0:
                require(guarded is not None, 'Trunk pass requires ordinary PPG probes and optimizer')
                self.report['update'] = guarded_pass(self.companion,train,self._targets(heldout),scale=state['scale'],**guarded)
            else:
                self.report['update'] = self.companion.head_only(train,generation=generation)
        main, delta = predict_rows(self.model,heldout)
        gate = clustered_gate(heldout,main,delta,seed=generation,use_root=not state['g0_passed'])
        if gate['full_window'] and state['lam'] is None:
            # On an unbiased final-outcome objective, lambda=1 is the exact
            # candidate target. Report every error, including this zero minimum.
            errors = {}
            for lam in (.5,.8,.95,1.):
                state['lam'] = lam
                candidate = self._targets(heldout)
                errors[str(lam)] = float(np.mean([(np.asarray(r['target'])-r['u'])**2 for r in candidate]))
            state['lam'] = None
            self.machine.choose_lambda(min((.5,.8,.95,1.),key=lambda x:errors[str(x)]))
            self.report['lambda_selection'] = errors
        self.report.update(gate=gate,stage=self.machine.advance(generation,gate,ordinary=ordinary,h2h=h2h,annealing=annealing),
            delta=dict(mean=float(np.abs(delta).mean()) if len(delta) else None,p90=float(np.quantile(np.abs(delta),.9)) if len(delta) else None))
        self.report['updated'] = bool(self.report.get('update',{}).get('steps') or
                                      any(i['accepted'] for i in self.report.get('update',{}).get('intervals',[])))
        return copy.deepcopy(self.report)

    def trunk_scale(self):
        if 'schedule' in self.options:
            from .search_value_schedule import trunk_scale
            return trunk_scale(self.machine.state)
        return self.machine.state['scale']

    def _scheduled_after_ppg(self,generation,*,guarded):
        from .search_value_schedule import diagnostics
        from .search_value_training import unconditional_pass
        began=time.perf_counter()
        if self.machine.state['lam'] is None:self.machine.choose_lambda(1.)
        # No policy-drift filter or validation gate on residual return fitting.
        # In particular current-G data must remain usable after its PPO update.
        buffered=self.buffer.rows(generation,'train')
        scale=self.trunk_scale();self.report.pop('update',None)
        rebuild=self.machine.state.get('buffer_rebuild')
        paused=False
        if rebuild is not None and rebuild['ready_generation'] is None:
            require(rebuild['required_ages']==8 and generation>=rebuild['origin_generation'], 'Invalid return-buffer rebuild')
            ages=sorted({r['generation'] for r in buffered if r['generation']>=rebuild['origin_generation']})
            paused=ages!=list(range(generation-7,generation+1))
            if not paused:rebuild['ready_generation']=generation
            self.report['buffer_rebuild']=dict(rebuild,fresh_ages=len(ages),paused=paused)
        # Rebuilding phases admit/verify shards but need no copied targets or
        # encoder preparation until auxiliary fitting can actually resume.
        train=[] if paused else self._targets(buffered)
        prepared=time.perf_counter()
        validation = None
        if train and scale>0 and not paused:
            from .search_value_retention import reserve_games
            train,validation,self.report['retention_holdout']=reserve_games(train,generation,self.companion.seed)
        if paused:self.report['update']=dict(steps=0,reason='Rebuilding eight fresh return-buffer ages; stage counter paused')
        elif train:
            if scale>0:
                if guarded is not None:
                    self.report['update']=unconditional_pass(self.companion,train,scale=scale,heldout=validation,
                        intervals=self.options['trunk']['intervals'],**guarded)
                else:self.report['update']=dict(steps=0,reason='No ordinary PPG probe; trunk update deferred')
            else:self.report['update']=self.companion.head_only(train,generation=generation,steps=self.options['steps_per_iteration'])
        else:self.report['update']=dict(steps=0,reason='No admitted training value rows')
        updated=time.perf_counter()
        # Never reuse an older validation as this iteration's observation.
        heldout=[r for r in self.buffer.rows(generation,'heldout') if r['generation']==generation]
        main,delta=predict_rows(self.model,heldout) if heldout else ([],[])
        diagnostic=diagnostics(heldout,np.asarray(main),np.asarray(delta))
        self.report.update(validation=diagnostic,stage=self.machine.advance(generation,diagnostic,pause=paused),
            updated=bool(self.report['update'].get('steps')),training_scale=scale,
            delta=dict(mean=float(np.abs(delta).mean()) if len(delta) else None,
                p90=float(np.quantile(np.abs(delta),.9)) if len(delta) else None))
        self.report['timings']=dict(value_read_and_target_seconds=prepared-began,update_seconds=updated-prepared,
            validation_seconds=time.perf_counter()-updated,total_seconds=time.perf_counter()-began)
        return copy.deepcopy(self.report)

    def payload(self):
        # Live labels may be FrozenDict/FrozenList. deepcopy deliberately shares
        # those objects; their pickle reducers require constructor globals that
        # the strict weights-only loader rejects. Persist only plain JSON trees,
        # stripping process-local admission identity without changing row values.
        # Keep this conversion at checkpoint publication, outside search/sealing.
        labels = label_snapshot(self.label_ages)
        return dict(stage=copy.deepcopy(self.machine.state),companion=self.companion.snapshot() if self.companion else None,
                    buffer=self.buffer.payload(),label_ages=labels)

    def job_state(self):
        """Small committed scheduling state; Adam stays in the checkpoint only."""
        return copy.deepcopy(self.machine.state)
