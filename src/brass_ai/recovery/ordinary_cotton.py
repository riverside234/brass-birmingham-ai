"""The cotton subset of the existing 70% ordinary self-play quota."""
from brass_ai.learning.contracts import require
from brass_ai.step5.policy import (COTTON_DEVELOP_ONLY, COTTON_EXPLORATION, COTTON_PROBABILITY, LIMITED_COTTON_PROBABILITY,
                                   cotton_build_available, cotton_candidates, cotton_seats, has_unflipped_cotton)

KEY = COTTON_EXPLORATION
CONTRACT = 'ordinary-cotton-exploration/1.0.0'
LIMITED_CONTRACT = 'ordinary-cotton-exploration/2.0.0'
CAP = 3
PROBABILITIES = (COTTON_PROBABILITY, .10)
SETTINGS = dict(contract=CONTRACT, probability=COTTON_PROBABILITY, standard_selfplay=50, cotton_selfplay=20)
LIMITED = dict(contract=LIMITED_CONTRACT, probability=LIMITED_COTTON_PROBABILITY, standard_selfplay=60, cotton_selfplay=10,
               max_per_seat_era=CAP, pause_unflipped=True, minimum_build_level=2)


def enabled(config):
    return KEY in config.get('table_plan', {})


def configure(config, *, probability=COTTON_PROBABILITY, limited=False):
    settings = dict(LIMITED) if limited else dict(SETTINGS, probability=probability)
    return dict(config, table_plan=dict(config['table_plan'], **{KEY: settings}))


def previous(config):
    if not enabled(config):
        return config
    return dict(config, table_plan={k: v for k, v in config['table_plan'].items() if k != KEY})


def assignment(index, sampling_seed, *, probability=COTTON_PROBABILITY, contract=CONTRACT):
    # Use the same interleaved quota points as table_kind.
    settings = LIMITED if contract == LIMITED_CONTRACT else SETTINGS
    point = (index * 35) % 100
    start = settings['standard_selfplay']
    if start <= point < start + settings['cotton_selfplay']:
        count = 1 if contract == LIMITED_CONTRACT else 1 + ((point - start) // 5) % 2
        return dict(contract=contract, probability=probability, seats=cotton_seats(sampling_seed, count, contract))
    return None


def planned_assignment(settings, index, seed):
    return assignment(index, seed, probability=settings['probability'], contract=settings['contract'])


def validate_assignment(proof):
    if KEY not in proof:
        return (), 0.
    value = proof[KEY]
    require(proof['kind'] == 'selfplay' and proof['purpose'] == 'training' and isinstance(value, dict)
            and value.get('contract') in (CONTRACT, LIMITED_CONTRACT)
            and value.get('probability') in ((LIMITED['probability'],) if value['contract'] == LIMITED_CONTRACT else PROBABILITIES)
            and value == planned_assignment(value, proof['ordinary_game_index'], proof['sampling_seed'])
            and all(type(seat) is int for seat in value['seats']), 'Ordinary cotton seat assignment differs')
    return tuple(value['seats']), value['probability']


def check_row(row, probability, *, builds=True):
    """Bind replay's active mixture to this seat's current legal choices."""
    b = row['behavior']
    phase = row['view']['context']['phase']
    candidates = cotton_candidates(row['choices'], phase, builds=builds) if probability else []
    require(b.get(KEY) == (probability if candidates else None) and
            (b.get(COTTON_DEVELOP_ONLY) is True if candidates and not builds else COTTON_DEVELOP_ONLY not in b) and
            (not candidates or b.get('exploration_version') == 3) and
            (b.get('component') != 'cotton' or row['chosen'] in candidates),
            'Ordinary cotton behavior eligibility differs')
    return candidates


class TableExploration:
    """Per-table allowance shared by actors and chronological replay validation."""

    def __init__(self, proof):
        self.seats, self.probability = validate_assignment(proof)
        self.limited = proof.get(KEY, {}).get('contract') == LIMITED_CONTRACT
        self.used = {'CanalEra': 0, 'RailEra': 0} if self.limited else None

    def options(self, view, choices):
        if not self.limited:
            return self.probability, True
        context = view['context']
        phase = context['phase']
        if len(choices) <= 1 or phase not in ('action', 'continuation') or self.used[context['era']] >= CAP:
            return 0., True
        seat = view['seat']
        builds = phase != 'action' or cotton_build_available(view, seat)
        if not has_unflipped_cotton(view, seat):
            return self.probability, builds
        return 0., True

    def commit(self, view, behavior):
        if self.limited and behavior.get('component') == 'cotton':
            self.used[view['context']['era']] += 1

    def check(self, row):
        view, choices = row['view'], row['choices']
        rate, builds = self.options(view, choices) if row['seat'] in self.seats else (0., True)
        candidates = check_row(row, rate, builds=builds)
        self.commit(view, row['behavior'])
        return candidates
