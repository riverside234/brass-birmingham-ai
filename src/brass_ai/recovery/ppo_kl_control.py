"""Reviewed PPO control based only on the fixed exact categorical probe."""
import copy

from brass_ai.learning.contracts import require

CONTRACT='ppo-categorical-kl-guard/3.0.0'
MIGRATION='ppo-exact-probe-only-control/1.0.0'
DEFAULT=dict(contract=MIGRATION,probe_rows=2048,sampled_kl='diagnostic-only')
SPARSE_CONTRACT='ppo-categorical-kl-guard/4.0.0'
SPARSE_MIGRATION='ppo-sparse-probe-control/1.0.0'
SPARSE_DEFAULT=dict(DEFAULT,contract=SPARSE_MIGRATION,probe_check_steps=4,full_dataset_kl=False)


def validate(update):
    block=update.get('kl_control')
    expected=SPARSE_DEFAULT if isinstance(block,dict) and block.get('contract')==SPARSE_MIGRATION else DEFAULT
    require(type(block) is dict and block==expected and all(type(block[k]) is type(v) for k,v in expected.items()),
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


def configure_sparse(config):
    """Only the reviewed PPO observation cadence changes, never collection/PPG."""
    result=copy.deepcopy(config)
    require(active(result['update']),'Sparse checks require exact-probe-only PPO control')
    result['update']['kl_control']=copy.deepcopy(SPARSE_DEFAULT)
    validate(result['update'])
    return result


def guard_contract(update):
    return SPARSE_CONTRACT if validate(update)['contract']==SPARSE_MIGRATION else CONTRACT


def probe_interval(update):
    return validate(update).get('probe_check_steps',1) if active(update) else 1


def full_dataset_enabled(update):
    return validate(update).get('full_dataset_kl',True) if active(update) else True


def probe_due(step,interval,epoch_end):
    # Always observe the first update and the final weights of every epoch.
    return step==1 or step%interval==0 or epoch_end


def sampled_console(update):
    """Never present the bounded sampled probe as a full-dataset measurement."""
    full=update['epochs'][-1]['post_update_kl']
    if full is not None:return f'KLsample={full:.5f} '
    return f'KLsample_probe={update["epochs"][-1]["probe_kl"]:.5f} KLfull=disabled '
