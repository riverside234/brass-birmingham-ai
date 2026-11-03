"""Exact g370 phase restoration across its already reviewed config adoption."""
import copy
from brass_ai.learning.contracts import require


def phase_payload(raw, config, generation):
    saved = copy.deepcopy(raw['training']['actor_value_phases'])
    target = {k: config[k] for k in ('advantage','value_phase','critic_data','seat_descriptors') if k in config}
    if saved['options'] == target:
        return saved, None
    from brass_ai.recovery import gae, ppg_g210 as P, trained_startup
    require(trained_startup.enabled(config) and config['advantage']['contract'] == gae.DIRECT_095_CONTRACT,
            'Finish unrelated recipe adoption before backend qualification')
    before = copy.deepcopy(saved['options']); evidence = {}
    if saved['options']['advantage'] != target['advantage']:
        saved['gae'], evidence['gae'] = gae.migrate_direct_095(saved['gae'], saved['options']['advantage'],
                                                            target['advantage'], generation)
        saved['options']['advantage'] = copy.deepcopy(target['advantage'])
    if saved['options']['value_phase'] != target['value_phase']:
        require(saved['options']['value_phase']['contract'] == P.KL32_CONTRACT
                and P.beta1_options(saved['options']) == target, 'Unreviewed PPG adoption')
        beta = saved['beta_clone']
        require(type(beta) in (int,float) and .5 <= beta <= 32, 'Invalid prior beta')
        saved['options'] = copy.deepcopy(target)
        saved['adoption'] = dict(contract=P.BETA1_MIGRATION,origin=generation,
            beta_before=beta,beta_after=max(1.,beta),previous_adoption=copy.deepcopy(saved['adoption']),
            scope='Raise beta only if below the new floor; preserve trip history, cache, Adam/RNG, schedule and progress.')
        saved['beta_clone'] = max(1.,beta)
        evidence['ppg'] = copy.deepcopy(saved['adoption'])
    require(saved['options'] == target, 'Unexpected continuation phase change')
    return saved, dict(contract='g370-diagnostic-restoration/1.0.0',generation=generation,
        previous_options=before,effective_options=target,changes=evidence,
        scope='Replay the previously reviewed g370 config transition in the isolated fork only; no source-run commit or update')
