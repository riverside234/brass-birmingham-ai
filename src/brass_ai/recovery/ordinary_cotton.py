"""The cotton subset of the existing 70% ordinary self-play quota."""
from brass_ai.learning.contracts import require
from brass_ai.step5.policy import (BUILD_ONLY_COTTON_ACTIONS, COTTON_CANDIDATES, COTTON_DEVELOP_ONLY, COTTON_EXPLORATION,
                                   COTTON_PROBABILITY, COTTON_TIMINGS, LIMITED_COTTON_PROBABILITY, QUIET_COTTON_ACTIONS,
                                   RETIMED_NO_DEVELOP_COTTON_ACTIONS, RETIMED_NO_NETWORK_COTTON_ACTIONS,
                                   RETIMED_QUIET_COTTON_ACTIONS, cotton_build_available, cotton_candidates, cotton_permissions,
                                   cotton_seats, has_unflipped_cotton, network_cotton_candidates)

KEY = COTTON_EXPLORATION
CONTRACT = 'ordinary-cotton-exploration/1.0.0'
LIMITED_CONTRACT = 'ordinary-cotton-exploration/2.0.0'
NETWORK_CONTRACT = 'ordinary-cotton-exploration/3.0.0'
# g2654: the network recipe with Rail-era cotton builds targeted only from level three.
RAIL_NETWORK_CONTRACT = 'ordinary-cotton-exploration/4.0.0'
# g3471: the g2654 recipe, timed within each era (step5.policy.cotton_timing): no cotton exploration at a seat's
# first and last Canal action and first two and last Rail action; cotton builds only in each era's last two.
TIMED_CONTRACT = 'ordinary-cotton-exploration/5.0.0'
# g3528: the g3471 rule retimed (step5.policy.COTTON_TIMINGS['g3528']): the first Canal action explores all cotton
# targets again, and Canal action 14 cotton develops too (no cotton-city network).
RETIMED_CONTRACT = 'ordinary-cotton-exploration/6.0.0'
NETWORK_CONTRACTS = (NETWORK_CONTRACT, RAIL_NETWORK_CONTRACT, TIMED_CONTRACT, RETIMED_CONTRACT)
TIMINGS = {TIMED_CONTRACT: COTTON_TIMINGS['g3471'], RETIMED_CONTRACT: COTTON_TIMINGS['g3528']}
LIMITED_CONTRACTS = (LIMITED_CONTRACT, *NETWORK_CONTRACTS)
RAIL_MINIMUM_BUILD_LEVEL = 3
CAP = 3
PROBABILITIES = (COTTON_PROBABILITY, .10)
SETTINGS = dict(contract=CONTRACT, probability=COTTON_PROBABILITY, standard_selfplay=50, cotton_selfplay=20)
LIMITED = dict(contract=LIMITED_CONTRACT, probability=LIMITED_COTTON_PROBABILITY, standard_selfplay=60, cotton_selfplay=10,
               max_per_seat_era=CAP, pause_unflipped=True, minimum_build_level=2)
NETWORK = dict(LIMITED, contract=NETWORK_CONTRACT, network_max_per_seat_era=1)
RAIL_NETWORK = dict(NETWORK, contract=RAIL_NETWORK_CONTRACT, rail_minimum_build_level=RAIL_MINIMUM_BUILD_LEVEL)
TIMED = dict(RAIL_NETWORK, contract=TIMED_CONTRACT, quiet_actions=QUIET_COTTON_ACTIONS,
             build_only_actions=BUILD_ONLY_COTTON_ACTIONS)
RETIMED = dict(RAIL_NETWORK, contract=RETIMED_CONTRACT, quiet_actions=RETIMED_QUIET_COTTON_ACTIONS,
               no_develop_actions=RETIMED_NO_DEVELOP_COTTON_ACTIONS, no_network_actions=RETIMED_NO_NETWORK_COTTON_ACTIONS)


def enabled(config):
    return KEY in config.get('table_plan', {})


def selfplay_settings(config):
    """The cotton recipe PPO self-play tables run, or None when every self-play table is a standard game.

    g4551: the table revision keeps the committed table_plan block (its contract still names the teacher's
    cotton rule) but assigns no self-play cotton seats (training_revision_g4551)."""
    from .training_revision_g4551 import selfplay_cotton_off
    return None if selfplay_cotton_off(config) else config.get('table_plan', {}).get(KEY)


def configure(config, *, probability=COTTON_PROBABILITY, limited=False):
    settings = dict(LIMITED) if limited else dict(SETTINGS, probability=probability)
    return dict(config, table_plan=dict(config['table_plan'], **{KEY: settings}))


def previous(config):
    if not enabled(config):
        return config
    return dict(config, table_plan={k: v for k, v in config['table_plan'].items() if k != KEY})


def assignment(index, sampling_seed, *, probability=COTTON_PROBABILITY, contract=CONTRACT):
    # Use the same interleaved quota points as table_kind.
    settings = LIMITED if contract in LIMITED_CONTRACTS else SETTINGS
    point = (index * 35) % 100
    start = settings['standard_selfplay']
    if start <= point < start + settings['cotton_selfplay']:
        count = 1 if contract in LIMITED_CONTRACTS else 1 + ((point - start) // 5) % 2
        return dict(contract=contract, probability=probability, seats=cotton_seats(sampling_seed, count, contract))
    return None


def planned_assignment(settings, index, seed):
    return assignment(index, seed, probability=settings['probability'], contract=settings['contract'])


def validate_assignment(proof):
    if KEY not in proof:
        return (), 0.
    value = proof[KEY]
    require(proof['kind'] == 'selfplay' and proof['purpose'] == 'training' and isinstance(value, dict)
            and value.get('contract') in (CONTRACT, *LIMITED_CONTRACTS)
            and value.get('probability') in ((LIMITED['probability'],) if value['contract'] != CONTRACT else PROBABILITIES)
            and value == planned_assignment(value, proof['ordinary_game_index'], proof['sampling_seed'])
            and all(type(seat) is int for seat in value['seats']), 'Ordinary cotton seat assignment differs')
    return tuple(value['seats']), value['probability']


def check_row(row, probability, *, builds=True, targets=None):
    """Bind replay's active mixture to this seat's current legal choices."""
    b = row['behavior']
    phase = row['view']['context']['phase']
    candidates = (cotton_candidates(row['choices'], phase, builds=builds) if targets is None else targets) if probability else []
    require(b.get(KEY) == (probability if candidates else None) and
            (b.get(COTTON_DEVELOP_ONLY) is True if candidates and not builds and targets is None else COTTON_DEVELOP_ONLY not in b) and
            (b.get(COTTON_CANDIDATES) == candidates if candidates and targets is not None else COTTON_CANDIDATES not in b) and
            (not candidates or b.get('exploration_version') == 3) and
            (b.get('component') != 'cotton' or row['chosen'] in candidates),
            'Ordinary cotton behavior eligibility differs')
    return candidates


class TableExploration:
    """Per-table allowance shared by actors and chronological replay validation."""

    def __init__(self, proof):
        self.seats, self.probability = validate_assignment(proof)
        contract = proof.get(KEY, {}).get('contract')
        self.network = contract in NETWORK_CONTRACTS
        self.limited = self.network or contract == LIMITED_CONTRACT
        self.rail_build_level = RAIL_MINIMUM_BUILD_LEVEL if contract in (RAIL_NETWORK_CONTRACT, TIMED_CONTRACT, RETIMED_CONTRACT) else 2
        self.timing = TIMINGS.get(contract)  # g3471 / g3528: cotton exploration timed within each era
        self.used = {'CanalEra': 0, 'RailEra': 0} if self.limited else None
        self.network_used = {'CanalEra': 0, 'RailEra': 0} if self.network else None

    def targets(self, view, choices):
        if not self.network:
            return None
        era = view['context']['era']
        if view['seat'] not in self.seats or has_unflipped_cotton(view, view['seat']):
            return []
        targets, develop, network = cotton_permissions(view['context'], self.timing) if self.timing else (True, True, True)
        if not targets:
            return []
        return network_cotton_candidates(view, choices, industry=self.used[era] < CAP,
                                         network=network and self.network_used[era] < 1,
                                         minimum_build_level=self.rail_build_level if era == 'RailEra' else 2, develop=develop)

    def options(self, view, choices):
        if self.network:
            return self.probability, True
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

    def commit(self, view, behavior, choice=None):
        if self.limited and behavior.get('component') == 'cotton':
            counts = self.network_used if self.network and choice['kind'] == 'network' else self.used
            counts[view['context']['era']] += 1

    def check(self, row):
        view, choices = row['view'], row['choices']
        rate, builds = self.options(view, choices) if row['seat'] in self.seats else (0., True)
        candidates = check_row(row, rate, builds=builds, targets=self.targets(view, choices))
        self.commit(view, row['behavior'], choices[row['chosen']])
        return candidates
