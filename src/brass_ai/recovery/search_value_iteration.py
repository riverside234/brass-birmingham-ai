"""Committed search-value lifecycle. Production enablement remains a B07 gate."""
import copy
import math

from brass_ai.learning.contracts import digest, require
from .local_scoring_auxiliary import bundle_id
from .search_value_phase import SearchValuePhase

CONTRACT = 'search-value-iteration/1.0.0'


def ordinary_guard(update, health):
    probe = update.get('categorical_probe', {})
    if not health or not health.get('seat_games'): return None
    values = dict(ppo_kl=probe.get('categorical_kl'),entropy=probe.get('raw_entropy'))
    for key in ('vp_sum','sells','passes_with_alternative','zero_vp_seats','insolvent_seats','insolvency_events'):
        values[key] = health[key]/health['seat_games']
    return values if all(v is not None and math.isfinite(v) for v in values.values()) else None


def stamp(payload):
    return dict(contract=CONTRACT, generation=payload['generation'],
                identity=bundle_id(payload), stage=copy.deepcopy(payload['stage']))


class SearchValueIteration:
    def __init__(self, run, config, state, model, raw=None):
        self.mode = config['teacher_phase']['mode']
        self.config, self.model = config, model
        saved = (raw or {}).get('training', {}).get('search_value')
        summary = state.get('search_value')
        require((saved is None and summary is None) or
                (saved is not None and summary == stamp(saved) and saved['generation'] == state['iteration']),
                'Search value checkpoint/state differs')
        if saved is not None:
            require(saved['contract'] == CONTRACT and saved['reward_id'] == digest(model.config.utility),
                    'Search value reward/contract differs; reviewed retirement required')
            require((saved['companion'] is not None) == hasattr(model,'search_value_head'),
                    'Missing search head/companion optimizer')
        options = config['teacher_phase']['search_value']
        if saved is None:
            require(options['stage'] in ('off','A') and options['blend']['w'] == options['trunk']['scale'] == 0,
                    'Search value adoption cannot skip gates')
        pending_files=[]
        from brass_ai.rust_backend.lab_shadow_config import is_active
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        if self.mode in ('shadow','on') and is_active(config):
            from brass_ai.learning.batching import model_id
            from brass_ai.recovery.teacher_phase import plan_jobs
            from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
            jobs,_=plan_jobs(config,state,model_id=model_id(model),export_hash=state['history'][-1]['sha256'],
                epoch=state['iteration'],ready=True,native_source=source_fingerprint())
            pending_files=[f"{state['iteration']:06d}-{j['split']}-{digest(j['expected']['game_id'])}.json.gz" for j in jobs]
        # These names permit staging only. No pending byte is read/admitted or
        # included in payload until complete cached tables are independently
        # validated and normal publication compares the exact recomputed bytes.
        self.phase = SearchValuePhase(run/'search-values', model, options, digest(model.config.utility),
            saved=saved, startup=config['teacher_phase']['startup'], trainable=self.mode == 'on',pending_files=pending_files,
            pending_directory=run/f"iteration-{state['iteration']:04d}"/'teacher-shadow/value-partials' if pending_files else None,
            defer_install=is_on(config) and not hasattr(model,'search_value_head'))
        self.generation = state['iteration']
        self.report = None
        self.pending_generation = None
        self._publication = None

    def _require_mutable(self):
        require(self._publication is None, 'Search value checkpoint publication is pending')

    def admit(self, tables, jobs, *, post_spent=0.):
        self._require_mutable()
        if jobs: self.phase.admit(tables,jobs,self.generation,post_spent=post_spent)

    def prepare_ppg(self, phases, generation, *, ordinary=None, annealing=False):
        self._require_mutable()
        require(generation == self.generation+1, 'Search value iteration out of order')
        self.pending_generation, self.report = generation, None
        self.ordinary, self.annealing = ordinary, annealing
        if phases is not None:
            phases.post_value_fit = self.post_fit if self.mode == 'on' else None

    def post_fit(self, *, model, optimizer, reference, clones, probe_ids, dataset,
                 heldout_ids, config, update, beta, report):
        """Called before PPG releases its mmap/preparation owners. No new probe."""
        from .value_phase import metrics
        from .ppg_g210 import rate_factor, scheduled
        require(model is self.model and self.pending_generation is not None, 'Unbound post-PPG pass')
        offset = 0
        def clone_loss():
            nonlocal offset
            count = min(len(clones),config['clone_batch_size'])
            require(count > 0, 'No ordinary cloning rows')
            sample = [clones[(offset+i)%len(clones)] for i in range(count)]
            offset += count
            return sum(reference.loss(part,reference.inputs(part),model).sum()/count
                       for part in reference.microbatches(sample))
        guard = dict(value_optimizer=optimizer,
            measure_ordinary=lambda:metrics(model,dataset,heldout_ids,update)['loss'],
            cloning_loss=clone_loss,measure_joint_kl=lambda:reference.measure(model,probe_ids),
            beta=beta,rate_for_step=lambda step,total:config['learning_rate'] *
                (rate_factor(step,total,config) if scheduled(config) else 1.),kl_cap=config['kl_cap'])
        ev = report['heldout_after'].get('explained_variance')
        self.ordinary = (dict(self.ordinary,critic_ev=ev) if self.ordinary is not None and
                         ev is not None and math.isfinite(ev) else None)
        self._finish(guard)

    def _finish(self, guarded=None):
        self._require_mutable()
        from .ppg_selection import Selection
        phase = self.phase
        # Keep collection on the exact committed model identity. A newly
        # enabled residual is attached only at the post-PPG training boundary;
        # existing companions restore eagerly from the full checkpoint.
        if self.mode=='on':phase.ensure_companion()
        old_stage = copy.deepcopy(phase.machine.state)
        scale=phase.trunk_scale()
        rollback = (Selection(self.model,guarded['value_optimizer'],{'search':0.},companion=phase.companion)
                    if guarded is not None and scale > 0 else None)
        head_rollback = ((copy.deepcopy(self.model.search_value_head.state_dict()),phase.companion.snapshot())
                         if rollback is None and self.mode == 'on' and phase.companion is not None else None)
        try:
            # Jobs/labels belong to the collection checkpoint G. The durable
            # checkpoint is G+1; aging at G+1 here would discard the eighth age
            # early and make the full eight-generation gate unreachable.
            result = phase.after_ppg(self.generation,mode=self.mode,ordinary=self.ordinary,
                                    annealing=self.annealing,guarded=guarded)
            # A stage-C guard rejection also rolls back intervals that passed
            # their local loss/KL tests. Keep the fallback stage, not its weights.
            if rollback is not None and old_stage['stage']=='C' and phase.machine.state['stage']!='C':
                rollback.restore(self.model,guarded['value_optimizer'])
                result.update(updated=False,stage_guard_rollback=True)
            if result.get('updated') and scale > 0:
                self.model.utility_calibration = None
            self.report = result
        except BaseException:
            if rollback is not None: rollback.restore(self.model,guarded['value_optimizer'])
            if head_rollback is not None:
                self.model.search_value_head.load_state_dict(head_rollback[0])
                phase.companion.restore(head_rollback[1])
            if rollback is not None or head_rollback is not None: self.model.zero_grad(set_to_none=True)
            phase.machine.state = old_stage
            raise

    def finish_ppg(self):
        require(self.pending_generation is not None, 'Search value PPG boundary missing')
        if self.report is None:
            if self.mode == 'on' and self.phase.trunk_scale() > 0 and 'schedule' not in self.phase.options:
                self.report = dict(updated=False,reason='No ordinary PPG probe; guarded pass deferred')
            else: self._finish()
        return self.report

    def payload(self, generation):
        require(generation == self.generation+1, 'Search value commit generation differs')
        if self._publication is None:
            # Own exactly the snapshot handed to save_checkpoint. Never copy or
            # convert label histories again when verifying the durable commit.
            payload = dict(self.phase.payload(),contract=CONTRACT,generation=generation,reward_id=self.phase.reward_id)
            self._publication = (payload, stamp(payload))
        else:
            require(stamp(self._publication[0]) == self._publication[1], 'Search value publication snapshot changed')
        return self._publication[0]

    def committed(self, state):
        require(self._publication is not None and state['iteration'] == self.generation+1,
                'Search value cleanup requires a prepared next-generation snapshot')
        payload, expected = self._publication
        # Rehash the same detached snapshot to detect caller mutation, including
        # companion tensors and receipts. Do not stamp a second live snapshot.
        require(state.get('search_value') == expected == stamp(payload),
                'Search value cleanup requires durable checkpoint/state commit')
        self.generation = state['iteration']
        self._publication = None
        # Retired shards remain rollback dependencies. B07 retention may delete
        # them only after inventorying every supported checkpoint.
