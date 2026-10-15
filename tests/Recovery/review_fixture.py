"""Synthetic same-feature-version receipts for unit tests, never production approvals."""
from unittest.mock import patch
from brass_ai.learning.contracts import compatibility,digest
from brass_ai.step5.league import implementation_id


def install(test,transitions):
    entries=[dict(entry,learning_compatibility=digest(compatibility()),reason='Authored unit-test receipt only')
             for entry in transitions]
    fixture=patch('brass_ai.recovery.resume.expand_source_reviews',return_value=dict(
        format='selfplay-resume-sources/1.0.0',transitions=entries))
    fixture.start();test.addCleanup(fixture.stop)


def phase_clocks(test):
    from brass_ai.recovery.config import TIME_CONFIG_MIGRATION,ARCHIVED_PHASE_TIME_CONFIG_MIGRATION,ARCHIVED_EARLY_RAIL_TIME_CONFIG_MIGRATION
    uniform='3db61ed932779f2cd8d8e149509ab0c1b7232af63102213f17d5666c2d338130'
    first='48831bc83887a8d6beb563fd908dff18ee6b180dd8d7da6824724f4c53d9e596'
    second='201a2b00d5e5c1c8485cfa26fd2d3e062375d510a4bbae74257dba99e54b6e9d'
    rows=[dict(from_implementation=old,to_implementation=implementation_id(),config_migration=TIME_CONFIG_MIGRATION)
          for old in (uniform,first,second)]
    rows += [dict(from_implementation=a,to_implementation=b,config_migration=m) for a,b,m in
        ((uniform,first,ARCHIVED_PHASE_TIME_CONFIG_MIGRATION),
         (uniform,second,ARCHIVED_EARLY_RAIL_TIME_CONFIG_MIGRATION),
         (first,second,ARCHIVED_EARLY_RAIL_TIME_CONFIG_MIGRATION))]
    install(test,rows)
