"""The cotton subset of the existing 70% ordinary self-play quota."""
from brass_ai.learning.contracts import require
from brass_ai.step5.policy import COTTON_EXPLORATION, COTTON_PROBABILITY, cotton_candidates, cotton_seats

KEY = COTTON_EXPLORATION
CONTRACT = 'ordinary-cotton-exploration/1.0.0'
SETTINGS = dict(contract=CONTRACT, probability=COTTON_PROBABILITY, standard_selfplay=50, cotton_selfplay=20)


def enabled(config):
    return KEY in config.get('table_plan', {})


def configure(config):
    return dict(config, table_plan=dict(config['table_plan'], **{KEY: dict(SETTINGS)}))


def previous(config):
    if not enabled(config):
        return config
    return dict(config, table_plan={k: v for k, v in config['table_plan'].items() if k != KEY})


def assignment(index, sampling_seed):
    # Use the same interleaved quota points as table_kind. Alternate
    # one and two seats across the four cotton points of each cycle.
    point = (index * 35) % 100
    start = SETTINGS['standard_selfplay']
    if start <= point < start + SETTINGS['cotton_selfplay']:
        return dict(contract=CONTRACT, probability=COTTON_PROBABILITY,
                    seats=cotton_seats(sampling_seed, 1 + ((point - start) // 5) % 2, CONTRACT))
    return None


def validate_assignment(proof):
    value = proof[KEY]
    require(proof['kind'] == 'selfplay' and proof['purpose'] == 'training' and isinstance(value, dict)
            and value == assignment(proof['ordinary_game_index'], proof['sampling_seed'])
            and all(type(seat) is int for seat in value['seats']), 'Ordinary cotton seat assignment differs')
    return tuple(value['seats'])


def check_row(row, selected_seat):
    """Bind replay's active mixture to this seat's current legal choices."""
    b = row['behavior']
    phase = row['view']['context']['phase']
    candidates = cotton_candidates(row['choices'], phase) if selected_seat else []
    require(b.get(KEY) == (COTTON_PROBABILITY if candidates else None) and
            (not candidates or b.get('exploration_version') == 3) and
            (b.get('component') != 'cotton' or row['chosen'] in candidates),
            'Ordinary cotton behavior eligibility differs')
    return candidates
