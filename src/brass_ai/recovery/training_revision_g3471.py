"""g3471: cotton exploration timed within each era (the user's rule), on ordinary PPO and teacher cotton tables.

- No cotton exploration at a seat's first and last action of the Canal era, nor at its first two and last action
  of the Rail era.
- In each era's last two actions only cotton builds are targeted: no cotton develop and no cotton-city network.

A seat's action in its era runs 1..15 in the Canal era (one action in round 1) and 1..16 in the Rail era
(step5.policy.era_action); a payment or continuation decision keeps its action's index. Seats, table quotas, rates,
the three build-and-develop targets per era, the one cotton network per era and the Rail-era build level three are
the g2654 recipe's. Adopted once with --update-cotton-timing at a committed boundary, then kept. Labels, buffers,
models and optimizers are kept: older rows and labels validate under the contract they were collected with.
"""
import copy

from brass_ai.learning.contracts import require
from brass_ai.step5.policy import BUILD_ONLY_COTTON_ACTIONS, QUIET_COTTON_ACTIONS
from . import ordinary_cotton, teacher_cotton, training_revision_g1491 as teacher


def applied(config):
    return (config.get('table_plan', {}).get(ordinary_cotton.KEY) == ordinary_cotton.TIMED and
            teacher.cotton_settings(config).get('contract') == teacher_cotton.TIMED_NETWORK_CONTRACT)


def configure(config):
    if applied(config):
        return config
    require(config.get('table_plan', {}).get(ordinary_cotton.KEY) == ordinary_cotton.RAIL_NETWORK and
            teacher.cotton_settings(config).get('contract') == teacher_cotton.RAIL_NETWORK_CONTRACT,
            'Cotton timing extends the g2654 Rail-era cotton recipe (--update-rail-cotton)')
    phase = config['teacher_phase']
    return dict(config, table_plan=dict(config['table_plan'], **{ordinary_cotton.KEY: copy.deepcopy(ordinary_cotton.TIMED)}),
                teacher_phase=dict(phase, **{teacher.KEY: teacher.timed_cotton_marker(phase[teacher.KEY])}))


def summary(config):
    return dict(quiet_actions=copy.deepcopy(QUIET_COTTON_ACTIONS), build_only_actions=copy.deepcopy(BUILD_ONLY_COTTON_ACTIONS),
                ppo_contract=config['table_plan'][ordinary_cotton.KEY]['contract'],
                teacher_contract=teacher.cotton_settings(config)['contract'])
