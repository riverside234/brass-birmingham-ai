"""Adopt the single-seat cotton recipe at a committed teacher-on boundary."""
from brass_ai.learning.contracts import require
from . import ordinary_cotton, training_revision_g1491 as teacher, training_revision_g2090 as opening


def applied(config):
    return teacher.cotton_only(config)


def configure(config):
    if applied(config):
        return config
    from brass_ai.rust_backend.lab_teacher_activation import is_on
    require(opening.applied(config) and teacher.seated_cotton(config) and is_on(config) and
            config['table_plan'].get(ordinary_cotton.KEY) == dict(ordinary_cotton.SETTINGS, probability=.10),
            'The cotton sampling update requires the g2090 seated teacher-on recipe')
    phase = config['teacher_phase']
    return ordinary_cotton.configure(dict(config, teacher_phase=dict(phase, **{
        teacher.KEY: teacher.cotton_only_marker(phase[teacher.KEY])})), limited=True)
