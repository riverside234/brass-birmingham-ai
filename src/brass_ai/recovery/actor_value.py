"""Iteration lifecycle and checkpoint binding for GAE and the separate value Adam."""
import copy
import math

import torch

from brass_ai.learning.contracts import require
from . import gae,value_phase,value_buffer,critic_data

KEY='actor_value_phases'
CONTRACT='actor-value-phases/1.0.0'


class TrainingPhases:
    def __init__(self,model,config,*,raw=None,generation=0,root=None,run=None):
        self.options={k:copy.deepcopy(config[k]) for k in ('advantage','value_phase')}
        self.options.update({k:copy.deepcopy(config[k]) for k in ('critic_data','seat_descriptors') if k in config})
        self.snapshot_data=[];self.fingerprint=None;self.critic_state=None
        self.config=config;self.dataset=None;self.probe_ids=None
        gae.validate(self.options['advantage'],combined='reward' in config)
        value_phase.validate(self.options['value_phase'])
        self.optimizer=torch.optim.Adam(model.parameters(),lr=config['value_phase']['learning_rate'])
        buffered=value_buffer.enabled(config)
        descriptors='critic_data' in config
        previous_contract='actor-value-phases/3.0.0' if descriptors else 'actor-value-phases/2.0.0' if buffered else CONTRACT
        evidence_contract='actor-value-phases/4.0.0' if gae.windowed(config['advantage']) else previous_contract
        base_contract='actor-value-phases/5.0.0' if descriptors else evidence_contract
        self.contract='actor-value-phases/6.0.0' if value_phase.stratified(config['value_phase']) else base_contract
        self.evidence_migration=None;self.critic_migration=None;self.refinement_migration=None
        saved=(raw or {}).get('training',{}).get(KEY)
        previous_value=(saved or {}).get('options',{}).get('value_phase')
        if saved is None:
            require(generation==0,'Missing GAE/value optimizer checkpoint; use a fresh IL run')
            self.state=gae.initial(config['advantage']);self.beta=config['value_phase']['beta_clone'];self.trips=[]
            if descriptors:self.critic_state=critic_data.initial(config['critic_data'])
        else:
            require(type(saved) is dict,'Invalid GAE/value checkpoint')
            migration_options=copy.deepcopy(self.options)
            if self.contract!=base_contract and previous_value and previous_value.get('contract')==value_phase.LEGACY_CONTRACT:
                # Older reviewed EV/weight-recovery upgrades compose before
                # the value-only refinement, without bypassing their checks.
                migration_options['value_phase']=copy.deepcopy(previous_value)
            # The league verifies the signed source/config transition first.
            # This exact upgrade also handles a restart after that transition
            # was committed but before its first new generation checkpoint.
            if (saved.get('contract') in (previous_contract,base_contract) and gae.windowed(config['advantage'])
                    and not gae.windowed(saved.get('options',{}).get('advantage',{}))):
                require(gae.evidence_config(dict(saved['options'],reward=config.get('reward',{})))==
                        dict(migration_options,reward=config.get('reward',{})),
                        'GAE/value checkpoint configuration differs during evidence upgrade')
                saved=copy.deepcopy(saved)
                saved['gae']=gae.migrate_state(saved['gae'],saved['options']['advantage'],config['advantage'],generation)
                saved.update(contract=(base_contract if saved['contract']==base_contract else evidence_contract),
                    options=copy.deepcopy(migration_options))
                self.evidence_migration=dict(contract=gae.CONFIG_MIGRATION,origin=generation,
                    scope='Empty EV window and new health streaks; active lambdas/table origins, both Adams, beta and critic state preserved')
            if descriptors and saved.get('contract')==evidence_contract:
                require(saved.get('options')==migration_options,'GAE/value checkpoint configuration differs during critic upgrade')
                saved=copy.deepcopy(saved)
                saved['critic_data']=critic_data.migrate_state(saved['critic_data'],config['critic_data'],generation)
                saved['contract']=base_contract
                self.critic_migration=dict(contract=critic_data.RECOVERY_CONTRACT,origin=generation,
                    scope='Empty weight-recovery streak; saved weight, baseline, activation, fingerprint, both Adams and GAE state preserved')
            if self.contract!=base_contract and saved.get('contract')==base_contract:
                require(value_phase.refined_options(saved.get('options',{}))==self.options,
                    'GAE/value checkpoint configuration differs during PPG refinement')
                saved=copy.deepcopy(saved);groups=saved['optimizer']['param_groups']
                old_rate=saved['options']['value_phase']['learning_rate']
                require(len(groups)==1 and groups[0]['lr']==old_rate,'Previous value optimizer LR differs')
                groups[0]['lr']=config['value_phase']['learning_rate']
                saved.update(contract=self.contract,options=copy.deepcopy(self.options))
                # Do not combine evidence gathered under different selection
                # protocols into a new qualification/recovery streak.
                state=saved['gae'];state.update(ev_streak=0,healthy_streak=0)
                if 'table_streaks' in state:state['table_streaks']={k:0 for k in state['table_streaks']}
                if 'table_evidence' in state:
                    from .ev_evidence import initial
                    state['table_evidence']=initial(generation)
                if descriptors:saved['critic_data']['weight_recovery']['healthy_streak']=0
                self.refinement_migration=dict(contract=value_phase.CONFIG_MIGRATION,origin=generation,
                    learning_rate_before=old_rate,learning_rate_after=config['value_phase']['learning_rate'],
                    scope='Value LR only changes in Adam; all moments/steps, weights and RNG preserved. New holdout evidence starts here; lambdas, activation/origins, snapshot weight, fixed EV baseline and training counters retained.')
            fields={'contract','options','gae','beta_clone','kl_trips','optimizer'}|({'buffer'} if buffered else set())|({'critic_data'} if descriptors else set())
            require(type(saved) is dict and set(saved)==fields
                    and saved['contract']==self.contract and saved['options']==self.options,
                    'GAE/value checkpoint configuration differs')
            self.state=copy.deepcopy(gae.validate_state(saved['gae'],config['advantage'],generation))
            if descriptors:self.critic_state=copy.deepcopy(critic_data.validate_state(saved['critic_data'],config['critic_data'],generation))
            self.beta=saved['beta_clone'];self.trips=copy.deepcopy(saved['kl_trips'])
            require(type(self.beta) in (int,float) and math.isfinite(self.beta) and
                    config['value_phase']['beta_clone']<=self.beta<=config['value_phase']['beta_clone_max'],
                    'Invalid persisted cloning coefficient')
            require(type(self.trips) is list and len(self.trips)<=5 and all(type(t) is bool for t in self.trips),
                    'Invalid persisted value-phase KL history')
            self.optimizer.load_state_dict(saved['optimizer'])
            require(len(self.optimizer.param_groups)==1 and
                    self.optimizer.param_groups[0]['lr']==config['value_phase']['learning_rate'] and
                    self.optimizer.param_groups[0]['weight_decay']==0,'Value optimizer settings differ')
        self.buffer=None;self.collection_source=None
        if buffered:
            require(root is not None and run is not None,'Two-iteration PPG requires an owned run disk cache')
            self.buffer=value_buffer.DiskBuffer(root,run,config,generation=generation,
                receipt=saved['buffer'] if saved is not None else None,
                previous_reward=(raw or {}).get('model_config',{}).get('utility'),previous_value=previous_value)

    def collection(self,records,policy_id,kinds,model_id):
        if self.buffer:
            self.collection_source=self.buffer.source(records,policy_id,kinds,model_id)
            self.collection_source['snapshot_rows']=bool(self.critic_state and self.critic_state['snapshots_active'])

    def dropout(self,epoch,phase):
        if self.critic_state is None:return None
        return self.state['generation'],epoch,self.config['seat_descriptors']['dropout_games'],phase

    def prepare(self,dataset):
        from .updates import Dataset
        self.dataset=dataset
        self.critic_dataset=Dataset(dataset.data+self.snapshot_data,dataset.device,policy=False) if self.snapshot_data else dataset
        critic=self.critic_dataset
        require(all(role in ('learner','snapshot') for role in critic.roles),'Excluded opponent in critic data')
        if self.critic_state:
            for i,role in enumerate(critic.roles):
                if role=='snapshot':critic.critic_weights[i]=self.critic_state['snapshot_weight']
        if self.options['value_phase']['enabled']:
            self.train_ids,self.heldout_ids,self.split=value_phase.split(critic.data,self.options['value_phase']['holdout_fraction'],
                self.options['value_phase'].get('holdout'))
            if self.critic_state and not self.critic_state['snapshots_active']:
                self.train_ids=[i for i in self.train_ids if critic.roles[i]=='learner']
        else:
            self.train_ids=list(range(len(dataset.data)));self.heldout_ids=[];self.split=None
        self.learner_train_ids=[i for i in self.train_ids if i<len(dataset.data)]
        if self.split:self.split['eligible_train_rows']=len(self.train_ids)
        return self.train_ids

    def advantages(self,mc,actor_ids):
        ds=self.dataset
        returns=(ds.utility if ds.utility is not None else ds.targets[0])[
            torch.arange(len(ds.data),device=ds.device),ds.seats]
        by_table=None
        if 'table_enabled' in self.state:
            by_table=dict(selfplay=self.state['active_lambda'],**{k:self.state['active_lambda'] if v else 1. for k,v in self.state['table_enabled'].items()})
        actual,report=gae.compute(ds.data,actor_ids,mc,returns,self.state['active_lambda'],self.options['advantage']['shadow_lambdas'],by_table)
        return actual,report

    def after_ppo(self,model,actor_ids,probe_ids,update):
        self.actor_ids=list(actor_ids);self.probe_ids=list(probe_ids)
        # The PPO report/probes stay bound to the post-PPO weights. A separate
        # final probe is retained for checkpoint metrics after value/teaching.
        self.ppo_probe=update

    def fit(self,model):
        require(self.dataset is not None and self.probe_ids is not None,'Value phase must follow PPO')
        def fit(dataset=None,indices=None):
            return value_phase.fit(model,self.optimizer,self.dataset,self.train_ids,self.heldout_ids,
                self.actor_ids,self.probe_ids,self.options['value_phase'],self.config['update'],beta=self.beta,
                value_dataset=dataset or self.critic_dataset,value_train_ids=indices,
                validation_dataset=self.critic_dataset,current_count=len(self.critic_dataset.data),
                dropout=(self.state['generation'],self.config['seat_descriptors']['dropout_games']) if self.critic_state else None)
        if self.buffer:
            require(self.collection_source is not None,'PPG disk cache requires ordinary collection provenance')
            cache=self.buffer.stage(self.critic_dataset,self.train_ids,self.collection_source,model)
            with self.buffer.previous() as previous:
                if previous is None:self.value_report=fit()
                else:
                    ds=value_buffer.ValueDataset(self.critic_dataset,previous)
                    if self.critic_state:
                        ds.critic_weights=torch.tensor([self.critic_state['snapshot_weight'] if r=='snapshot' else 1. for r in ds.roles],device=ds.device)
                    ids=self.train_ids+list(range(len(self.critic_dataset.data),len(ds.data)))
                    self.value_report=fit(ds,ids)
            cache['rebuilt_from_replay']=self.buffer.rebuilt
            self.value_report['buffer']=cache
        else:self.value_report=fit()
        self.value_report['split']=self.split
        if self.refinement_migration:self.value_report['refinement_migration']=self.refinement_migration
        return self.value_report

    def finish(self,*,generation,update,anchor_gate,rating):
        value=self.value_report;before=self.beta
        self.trips=(self.trips+[value['kl_guard_tripped']])[-5:]
        # A clean iteration must not re-escalate on the same old cap hits.
        # Keep the rolling evidence on resume, but require a new trip to act.
        if value['kl_guard_tripped'] and sum(self.trips)>=3:
            self.beta=min(self.options['value_phase']['beta_clone_max'],2*self.beta)
        value.update(beta_clone_next=self.beta,beta_decision='up' if self.beta>before else 'hold',
            beta_reason=('new_trip_in_three_of_five' if self.beta>before else
                         'at_cap' if value['kl_guard_tripped'] and sum(self.trips)>=3 else
                         'no_new_kl_trip' if not value['kl_guard_tripped'] else 'fewer_than_three_trips'),
            kl_trip_window=list(self.trips))
        kl=update['categorical_probe']['categorical_kl']
        # The calibrated IL distance budget/latch controls LR increases only.
        # GAE and critic adoption still require the underlying economy AND
        # behavior checks, plus their own PPO/value KL and evidence gates.
        training_health=(anchor_gate.get('health',anchor_gate) if anchor_gate else None)
        healthy=bool(training_health and training_health.get('ok') is True and update['controller_warmup_done'] and
            update['optimizer_steps']>0 and not update['kl_early_stop'] and gae.finite(kl) and
            kl<=1.5*self.config['update']['target_kl'])
        self.state,report=gae.advance(self.state,self.options['advantage'],generation=generation,
            value_report=value,health_ok=healthy,rating=rating)
        report['anchor_gate']=copy.deepcopy(anchor_gate)
        report['training_health']=copy.deepcopy(training_health)
        report['health_scope']='Economy/behavior health and PPO/value guards; calibrated IL distance/latch gates LR increases only'
        if self.evidence_migration:report['evidence_migration']=copy.deepcopy(self.evidence_migration)
        if self.critic_state is not None:
            self.critic_state,value['critic_rollout']=critic_data.advance(self.critic_state,self.config['critic_data'],
                generation=generation,report=value,healthy=healthy and not value.get('kl_guard_tripped',False),fingerprint_now=self.fingerprint,
                ppo_snapshot_rows=update.get('critic_rows',{}).get('snapshot_distinct_used',0) if update['optimizer_steps']>0 else 0)
            if self.critic_migration:value['critic_rollout']['recovery_migration']=copy.deepcopy(self.critic_migration)
        update['gae']['gate']=report
        return report

    def payload(self):
        if self.buffer:require(self.buffer.pending is not None,'PPG checkpoint needs the staged cache receipt')
        return dict(contract=self.contract,options=copy.deepcopy(self.options),gae=copy.deepcopy(self.state),
            **({'critic_data':copy.deepcopy(self.critic_state)} if self.critic_state is not None else {}),
            beta_clone=self.beta,kl_trips=copy.deepcopy(self.trips),optimizer=self.optimizer.state_dict(),
            **({'buffer':copy.deepcopy(self.buffer.pending)} if self.buffer else {}))

    def committed(self,generation):
        if self.buffer:
            self.value_report['buffer']['committed_cleanup']=self.buffer.committed(generation)
        self.collection_source=None
        self.refinement_migration=None

    def release(self):
        self.dataset=None;self.critic_dataset=None;self.snapshot_data=[];self.ppo_probe=None;self.probe_ids=None
