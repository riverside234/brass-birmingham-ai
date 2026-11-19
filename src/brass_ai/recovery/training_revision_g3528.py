"""g3528: cotton exploration retimed within each era (the user's revision of the g3471 rule), on ordinary PPO and
teacher cotton tables.

By a seat's action in its era (step5.policy.era_action: Canal 1..15, Rail 1..16):

| action | Canal | Rail |
| --- | --- | --- |
| no cotton exploration | 15 | 1, 2, 16 |
| no cotton develop | 15 | 15, 16 |
| no cotton-city network | 14, 15 | 15, 16 |

Against g3471: Canal action 1 explores every cotton target again, and Canal action 14 targets cotton develops as
well as builds. The Rail era is unchanged. Seats (drawn from the new contracts' streams), quotas, rates, the three
build-and-develop targets per era, the one cotton network per era and the Rail-era build level three are unchanged.
Adopted with --update-cotton-timing at a committed boundary (from the g3471 or the g2654 recipe), then kept. Labels,
buffers, models and optimizers are kept: older rows and labels validate under the contract they were collected with.
"""
import copy

from brass_ai.step5.policy import COTTON_TIMINGS
from . import ordinary_cotton, teacher_cotton, training_revision_g1491 as teacher, training_revision_g3471 as g3471


def applied(config):
    return (config.get('table_plan', {}).get(ordinary_cotton.KEY) == ordinary_cotton.RETIMED and
            teacher.cotton_settings(config).get('contract') == teacher_cotton.RETIMED_NETWORK_CONTRACT)


def configure(config):
    if applied(config):
        return config
    timed = config if g3471.applied(config) else g3471.configure(config)
    phase = timed['teacher_phase']
    return dict(timed, table_plan=dict(timed['table_plan'], **{ordinary_cotton.KEY: copy.deepcopy(ordinary_cotton.RETIMED)}),
                teacher_phase=dict(phase, **{teacher.KEY: teacher.retimed_cotton_marker(phase[teacher.KEY])}))


def summary(config):
    timing = COTTON_TIMINGS['g3528']
    return dict(quiet_actions=copy.deepcopy(timing['quiet']), no_develop_actions=copy.deepcopy(timing['no_develop']),
                no_network_actions=copy.deepcopy(timing['no_network']),
                ppo_contract=config['table_plan'][ordinary_cotton.KEY]['contract'],
                teacher_contract=teacher.cotton_settings(config)['contract'])
