"""g2654: Rail-era cotton builds below level three are not targeted by cotton exploration.

Applies to the cotton seats of ordinary PPO cotton tables and of teacher cotton tables.
Level-one builds stay excluded in both eras; Canal-era builds still need level two.
Develop and network targets, allowances and table quotas are unchanged. Labels, buffers,
models, optimizers and the search-head stage are kept: older rows and labels carry
their own contract and validate under the rule they were collected with.
"""
from brass_ai.learning.contracts import require
from . import ordinary_cotton, teacher_cotton, training_revision_g1491 as teacher, training_revision_g2600 as g2600


def applied(config):
    return (config.get('table_plan', {}).get(ordinary_cotton.KEY) == ordinary_cotton.RAIL_NETWORK and
            teacher.cotton_settings(config).get('contract') == teacher_cotton.RAIL_NETWORK_CONTRACT)


def configure(config):
    if applied(config):
        return config
    require(g2600.applied(config) and config['table_plan'].get(ordinary_cotton.KEY) == ordinary_cotton.NETWORK and
            teacher.cotton_settings(config).get('contract') == teacher_cotton.NETWORK_CONTRACT,
            'The Rail-era cotton build rule requires the g2600 cotton-network recipe')
    phase = config['teacher_phase']
    return dict(config, table_plan=dict(config['table_plan'], **{ordinary_cotton.KEY: dict(ordinary_cotton.RAIL_NETWORK)}),
                teacher_phase=dict(phase, **{teacher.KEY: teacher.rail_cotton_marker(phase[teacher.KEY])}))


def summary(config):
    return dict(minimum_cotton_build_level=dict(CanalEra=2, RailEra=ordinary_cotton.RAIL_MINIMUM_BUILD_LEVEL),
                ppo_contract=config['table_plan'][ordinary_cotton.KEY]['contract'],
                teacher_contract=teacher.cotton_settings(config)['contract'])
