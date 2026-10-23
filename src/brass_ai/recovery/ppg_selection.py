"""One-SE PPG retention with a separate, monotonically improving loss reference."""
import copy

from brass_ai.learning.contracts import require

DEFAULT=dict(contract='ppg-one-se-coverage/1.0.0',standard_errors=1.,
    reference='minimum-mean-game-loss',preference='distinct-critic-rows',
    ties='lower-loss-then-later',neutral='continue-without-failure')
RELAXED_DEFAULT=dict(DEFAULT,contract='ppg-paired-se-coverage/2.0.0',standard_errors=1.5)


class Selection:
    def __init__(self,model,optimizer,games,companion=None,config=None):
        self.companion=companion
        self.config=copy.deepcopy(DEFAULT if config is None else config)
        require(self.config in (DEFAULT,RELAXED_DEFAULT),'Invalid paired-loss selection tolerance')
        self.minimum_games=copy.deepcopy(games);self.minimum_budget_step=0
        self.minimum_rows=0;self.initial_loss=sum(games.values())/len(games)
        self.neutral_intervals=0;self.failed_intervals=0;self.improvements=0
        self._save(model,optimizer,games,[],[],0,0)

    def _save(self,model,optimizer,games,rows,clones,steps,budget_step):
        self.weights=copy.deepcopy(model.state_dict());self.adam=copy.deepcopy(optimizer.state_dict())
        self.auxiliary=self.companion.snapshot() if self.companion is not None else None
        self.games=copy.deepcopy(games);self.rows=list(rows);self.clones=list(clones)
        self.steps=steps;self.budget_step=budget_step

    def consider(self,model,optimizer,games,rows,clones,steps,budget_step):
        comparison=self.paired(self.minimum_games,games)
        if comparison['improved']:
            self.minimum_games=copy.deepcopy(games);self.minimum_budget_step=budget_step
            self.minimum_rows=len(set(rows));self.improvements+=1
        elif comparison['reject']:self.failed_intervals+=1
        else:self.neutral_intervals+=1
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
        require(not self.paired(self.minimum_games,self.games)['reject'],
            'Selected PPG state exceeds its configured paired-SE loss allowance')
        model.load_state_dict(self.weights);optimizer.load_state_dict(self.adam)
        if self.companion is not None:self.companion.restore(self.auxiliary)

    def report(self):
        return dict(config=copy.deepcopy(self.config),initial_loss=self.initial_loss,
            minimum_loss=sum(self.minimum_games.values())/len(self.minimum_games),
            retained_loss=sum(self.games.values())/len(self.games),
            minimum_budget_step=self.minimum_budget_step,retained_budget_step=self.budget_step,
            minimum_distinct_rows=self.minimum_rows,retained_distinct_rows=len(set(self.rows)),
            minimum_loss_improvements=self.improvements,neutral_intervals=self.neutral_intervals,
            failed_intervals=self.failed_intervals,retained_vs_minimum=self.paired(self.minimum_games,self.games),
            scope='Paired ordinary learner parent games over all eight ages. Maximum distinct retained critic rows within the configured SE allowance of minimum mean loss; ties prefer lower loss then later budget position. KL guards are independent. Epoch budgets follow the separately versioned phase configuration.')

    def paired(self,before,after):
        from .ppg_auxiliary import paired
        if self.config['standard_errors']==1.:return paired(before,after)
        return paired(before,after,standard_errors=self.config['standard_errors'])
