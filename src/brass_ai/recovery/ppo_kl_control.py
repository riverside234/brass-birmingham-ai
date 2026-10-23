"""Reviewed PPO control based only on the fixed exact categorical probe."""
import copy

from brass_ai.learning.contracts import require

CONTRACT='ppo-categorical-kl-guard/3.0.0'
MIGRATION='ppo-exact-probe-only-control/1.0.0'
DEFAULT=dict(contract=MIGRATION,probe_rows=2048,sampled_kl='diagnostic-only')


def validate(update):
    block=update.get('kl_control')
    require(type(block) is dict and block==DEFAULT and type(block['probe_rows']) is int,
        'Invalid exact-probe PPO KL control')
    require(update.get('actor_nonforced') is True and update.get('kl_probe_rows')==2048,
        'Exact-only PPO control requires the 2048-row nonforced probe')
    return block


def active(update):
    if 'kl_control' not in update:return False
    validate(update);return True


def configure(config):
    result=copy.deepcopy(config)
    if 'kl_control' in result['update']:validate(result['update'])
    result['update']['kl_control']=copy.deepcopy(DEFAULT)
    validate(result['update'])
    return result
