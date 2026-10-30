"""Read-only use of the existing frozen ordinary planner at a committed boundary."""
from pathlib import Path
import copy

from brass_ai.learning.contracts import require
from brass_ai.recovery.ordinary import plan_collection


def plan(root, run, snapshot, model, raw, service, output):
    from brass_ai.recovery import exploration
    from brass_ai.recovery.config import v4_search_enabled
    from brass_ai.recovery.legacy_search import legacy_spec
    from brass_ai.recovery.curriculum import boost_window
    from brass_ai.recovery.critic_data import fingerprint
    from brass_ai.recovery.il_kl_probe import load as load_probe
    root,run=Path(root),Path(run)
    state=copy.deepcopy(snapshot['state']);signature=state['signature'];config=signature['config']
    settings=exploration.effective(config,exploration.restore(config,state,raw['training'].get('exploration')),state['iteration'])
    # Use the committed pool, never a later mutable pool selection.
    profiles=sorted(signature['legacy_pool'])
    specs={p:legacy_spec(root,p,config.get('legacy_budget'),search_free=not v4_search_enabled(config),
        work_budget=config.get('legacy_opponent_work_budget')) for p in profiles}
    fast={p:legacy_spec(root,p,config['legacy_budget'],search_free=True,
        work_budget=config['legacy_opponent_work_budget']) for p in profiles}
    fingerprints=None
    if 'critic_data' in config:
        require((run/'il-kl/manifest.json').is_file(),'Committed descriptor probe is missing; B02 never creates one')
        origin=signature.get('imitation_source')
        require(origin is not None and state.get('il_kl_limit',{}).get('probe'),'Missing committed descriptor probe binding')
        probe,_=load_probe(root,run,origin,None,config['il_kl_limit'],config['update'],config['audit_workers'],
                          receipt=state['il_kl_limit']['probe'])
        fingerprints=dict(learner=fingerprint(probe,probe.probabilities(model,config['update'])),rule_bot=fingerprint(probe,rule_bot=True))
        from brass_ai.recovery.domino_config import enabled
        if enabled(config):
            from brass_ai.recovery.domino_fingerprints import Fingerprints
            fingerprints['contexts']=Fingerprints(root,output,probe,model,config['update'],
                {m['model_id']:m for m in state['quality_pool']['members']})
    current=service.register(model)
    require(current==snapshot['model_id'],'Incoming learner identity changed')
    result=plan_collection(root,output,state,config,signature,service,current,profiles,specs,fast,
        boost_window(state,config),fingerprints=fingerprints,exploration=settings)
    return result
