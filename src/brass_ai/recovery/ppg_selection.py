"""Versioned PPG checkpoint selection, separate from paired-loss stopping."""
import copy
import math

from brass_ai.learning.contracts import require

DEFAULT=dict(contract='ppg-one-se-coverage/1.0.0',standard_errors=1.,
    reference='minimum-mean-game-loss',preference='distinct-critic-rows',
    ties='lower-loss-then-later',neutral='continue-without-failure')
RELAXED_DEFAULT=dict(DEFAULT,contract='ppg-paired-se-coverage/2.0.0',standard_errors=1.5)
MINIMUM_DEFAULT=dict(RELAXED_DEFAULT,contract='ppg-minimum-heldout-loss/1.0.0',
    preference='minimum-mean-game-loss',ties='earlier-checkpoint')
LATEST_DEFAULT=dict(RELAXED_DEFAULT,contract='ppg-last-accepted-loss/1.0.0',
    reference='previous-accepted-game-loss',preference='latest-accepted-checkpoint',ties='later-checkpoint')


def minimum(config):return config.get('contract')==MINIMUM_DEFAULT['contract']


def latest(config):return config.get('contract')==LATEST_DEFAULT['contract']


def mean_loss(games):return math.fsum(games.values())/len(games)


class Selection:
    def __init__(self,model,optimizer,games,companion=None,config=None):
        self.companion=companion
        self.config=copy.deepcopy(DEFAULT if config is None else config)
        require(self.config in (DEFAULT,RELAXED_DEFAULT,MINIMUM_DEFAULT,LATEST_DEFAULT),'Invalid paired-loss selection tolerance')
        self.minimum_games=copy.deepcopy(games);self.minimum_budget_step=0
        self.minimum_rows=0;self.initial_loss=self._loss(games)
        self.neutral_intervals=0;self.failed_intervals=0;self.improvements=0
        self._save(model,optimizer,games,[],[],0,0)

    def _loss(self,games):
        return mean_loss(games) if minimum(self.config) else sum(games.values())/len(games)

    def _save(self,model,optimizer,games,rows,clones,steps,budget_step):
        self.weights=copy.deepcopy(model.state_dict());self.adam=copy.deepcopy(optimizer.state_dict())
        self.auxiliary=self.companion.snapshot() if self.companion is not None else None
        self.games=copy.deepcopy(games);self.rows=list(rows);self.clones=list(clones)
        self.steps=steps;self.budget_step=budget_step

    def consider(self,model,optimizer,games,rows,clones,steps,budget_step):
        comparison=self.paired(self.games if latest(self.config) else self.minimum_games,games)
        if latest(self.config):
            if mean_loss(games)<mean_loss(self.minimum_games):
                self.minimum_games=copy.deepcopy(games);self.minimum_budget_step=budget_step
                self.minimum_rows=len(set(rows));self.improvements+=1
            if comparison['reject']:
                self.failed_intervals+=1;return False
            if not comparison['improved']:self.neutral_intervals+=1
            self._save(model,optimizer,games,rows,clones,steps,budget_step)
            return True
        improved=(mean_loss(games)<mean_loss(self.minimum_games) if minimum(self.config) else comparison['improved'])
        if improved:
            self.minimum_games=copy.deepcopy(games);self.minimum_budget_step=budget_step
            self.minimum_rows=len(set(rows));self.improvements+=1
        elif comparison['reject']:self.failed_intervals+=1
        else:self.neutral_intervals+=1
        if minimum(self.config):
            # Capture the exact weights, value Adam and companion together at
            # every new minimum. More rows, later steps and SE tolerance never
            # override this decision. Step zero remains eligible throughout.
            if improved:self._save(model,optimizer,games,rows,clones,steps,budget_step)
            return improved
        # A later minimum can disqualify a previously acceptable selection.
        # Never move the reference upward merely to retain more training.
        eligible=not self.paired(self.minimum_games,games)['reject']
        previous_eligible=not self.paired(self.minimum_games,self.games)['reject']
        score=(len(set(rows)),-sum(games.values())/len(games),budget_step)
        previous=(len(set(self.rows)),-sum(self.games.values())/len(self.games),self.budget_step)
        selected=eligible and (not previous_eligible or score>previous)
        if selected:self._save(model,optimizer,games,rows,clones,steps,budget_step)
        return selected

    def restore(self,model,optimizer):
        if minimum(self.config):
            require(self.games==self.minimum_games and self.budget_step==self.minimum_budget_step
                and mean_loss(self.games)<=self.initial_loss,'Selected PPG state is not its minimum held-out loss')
        elif not latest(self.config):
            require(not self.paired(self.minimum_games,self.games)['reject'],
                'Selected PPG state exceeds its configured paired-SE loss allowance')
        model.load_state_dict(self.weights);optimizer.load_state_dict(self.adam)
        if self.companion is not None:self.companion.restore(self.auxiliary)

    def report(self):
        return dict(config=copy.deepcopy(self.config),initial_loss=self.initial_loss,
            minimum_loss=self._loss(self.minimum_games),
            retained_loss=self._loss(self.games),
            minimum_budget_step=self.minimum_budget_step,retained_budget_step=self.budget_step,
            minimum_distinct_rows=self.minimum_rows,retained_distinct_rows=len(set(self.rows)),
            minimum_loss_improvements=self.improvements,neutral_intervals=self.neutral_intervals,
            failed_intervals=self.failed_intervals,retained_vs_minimum=self.paired(self.minimum_games,self.games),
            scope=('Compare against the previous accepted mean ordinary-learner game loss. Retain every interval within 1.5 paired SE, including neutral/worse intervals; reject and restore the last accepted weights, value Adam and companion together. Minimum loss is diagnostic only; gradual drift within successive allowances is possible.' if latest(self.config) else
                'Minimum mean ordinary-learner held-out game loss within this phase, including its post-PPO starting state. Exact ties retain the earlier checkpoint; weights, value Adam and companion state match. SE allowance controls stopping only; no cross-phase minimum because the held-out games change.' if minimum(self.config) else
                'Paired ordinary learner parent games over all configured validation ages. Maximum distinct retained critic rows within the configured SE allowance of minimum mean loss; ties prefer lower loss then later budget position. Policy control and epoch budgets follow the separately versioned phase configuration.'))

    def paired(self,before,after):
        from .ppg_auxiliary import paired
        if self.config['standard_errors']==1.:return paired(before,after)
        return paired(before,after,standard_errors=self.config['standard_errors'])
