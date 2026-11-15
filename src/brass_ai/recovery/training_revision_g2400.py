"""Cotton network allowance and two-epoch ordinary PPG at a committed boundary."""
from brass_ai.learning.contracts import require
from . import ordinary_cotton, teacher_cotton, training_revision_g1491 as teacher


def applied(config):
    return teacher.cotton_settings(config).get('contract') == teacher_cotton.NETWORK_CONTRACT


def configure(config):
    if applied(config):
        return config
    from .training_revision_g2090 import applied as opening
    require(opening(config) and teacher.cotton_only(config) and config['teacher_phase']['mode']=='on' and
            config['value_phase']['max_epochs']==1 and
            config['table_plan'].get(ordinary_cotton.KEY)==ordinary_cotton.LIMITED,
            'The cotton network update requires the single-seat teacher-on recipe')
    phase = config['teacher_phase']
    return dict(config, value_phase=dict(config['value_phase'], max_epochs=2),
                table_plan=dict(config['table_plan'], **{ordinary_cotton.KEY: dict(ordinary_cotton.NETWORK)}),
                teacher_phase=dict(phase, **{teacher.KEY: teacher.cotton_only_marker(phase[teacher.KEY], network=True)}))
