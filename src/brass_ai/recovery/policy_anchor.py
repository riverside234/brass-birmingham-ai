"""Qualified frozen KL references, with checkpoint-bound calibration transitions."""
import copy
import json
import math
import numpy as np
import torch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import digest, project_path, require
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.step5.league import write_json
from .plackett_luce import IL, V4
from . import il_kl_limit, panel_gates, rating_coverage

CONTRACT='qualified-policy-anchor/1.0.0'
MIGRATION='qualified-anchor-competitive-rating-epsilon02/1.0.0'
DEFAULT=dict(contract=CONTRACT,qualification='retired-or-active-il-panel-ge90',
    selection='highest-rated-qualified-newer',promotion='paired-setup-bootstrap-ci95-lower-gt0',
    initial_limit=.15,initial_group_limit=.30)


def configure(config):
    result=panel_gates.configure(config)
    result.update(policy_anchor=copy.deepcopy(DEFAULT),rating_coverage=copy.deepcopy(rating_coverage.DEFAULT),exploration=.02)
    return result


def validate(config):
    require(config.get('policy_anchor')==DEFAULT and 'panel_gates' in config and 'rating_coverage' in config and
        config.get('il_kl_limit',{}).get('initial_limit')==.15 and
        config.get('il_kl_limit',{}).get('group_factor')==2. and config.get('rating',{}).get('enabled') is True,
        'Qualified policy anchor requires IL-panel qualification and fixed .15/.30 calibration')
    rating_coverage.validate(config['rating_coverage'],config['rating'])


def initial(generation):
    return dict(contract=CONTRACT,origin=generation,active=None,last_fit=None,last_decision=None,events=[],pending=None)


def token(controller):
    active=(controller or {}).get('active')
    return digest(active) if active else None


def checkpoint_state(state):
    controller=copy.deepcopy(state.get('policy_anchor'))
    if controller is not None:controller['pending']=None
    return controller


def validate_state(controller, generation):
    require(type(controller) is dict and set(controller)==set(initial(0)) and controller['contract']==CONTRACT and
        type(controller['origin']) is int and 0<=controller['origin']<=generation and
        type(controller['events']) is list,'Invalid policy-anchor state')
    active=controller['active']
    if active:
        require(active['agent']==f'generation-{active["checkpoint"]["generation"]:06d}' and
            0<active['checkpoint']['generation']<=active['selected_generation']<=generation and
            active['evidence']['points_delta_ci95'][0]>0 and active['qualification']['win_share']>=.9,
            'Invalid qualified policy anchor')
    if controller['last_fit'] is not None:
        require(type(controller['last_fit']['generation']) is int and
            controller['last_fit']['generation']<=generation,'Policy anchor considered future rating evidence')
    return controller


def paired_difference(report, samples, names, candidate, reference, scale):
    require(candidate in names and reference in names and candidate!=reference,'Missing paired rating identity')
    ci=np.quantile(samples[:,names.index(candidate)]-samples[:,names.index(reference)],[.025,.975])*scale
    delta=report['ratings'][candidate]['points']-report['ratings'][reference]['points']
    matches=[d for d in report['differences'] if {d['a'],d['b']}=={candidate,reference}]
    require(len(matches)==1,'Missing/ambiguous paired rating difference')
    saved=matches[0];old_ci=saved['points_delta_ci95'];old_delta=saved['points_delta']
    if saved['b']!=candidate:old_ci=[-old_ci[1],-old_ci[0]];old_delta=-old_delta
    require(np.allclose(old_ci,ci,rtol=1e-10,atol=1e-10) and math.isclose(old_delta,delta,rel_tol=1e-10,abs_tol=1e-10),
        'Paired rating interval does not reproduce shared bootstrap samples')
    return dict(candidate=candidate,reference=reference,points_delta=delta,points_delta_ci95=ci.tolist(),
        method='same-fit paired setup-bootstrap percentile 95%',bootstrap_refits=len(samples))


def load_fit(root, rating, fit):
    for key in ('report','bootstrap','spec'):
        require(sha(project_path(root,fit[key]['path']))==fit[key]['sha256'],'Policy anchor rating evidence changed')
    report=read(project_path(root,fit['report']['path']));spec=read(project_path(root,fit['spec']['path']))
    require(report['generation']==spec['generation']==fit['generation'] and spec['config']==rating['config'] and
        report['fit']['converged'] is True and report['contract']==rating['contract'],
        'Policy anchor requires a valid current PL fit')
    with np.load(project_path(root,fit['bootstrap']['path']),allow_pickle=False) as data:
        samples=data['theta'].copy();names=data['agents'].tolist()
    require(len(set(names))==len(names) and set(names)==set(report['ratings']) and IL in names and
        samples.shape==(rating['config']['bootstrap'],len(names)) and np.isfinite(samples).all() and
        np.all(samples[:,names.index(IL)]==0.) and report['ratings'][IL]['theta']==0.,
        'Invalid shared setup-bootstrap samples or fixed IL origin')
    return report,samples,names


def select(root, state, controller):
    rating=state['ratings']
    if rating['latest_report'] is None:return None,None,'no_completed_rating'
    slot=next((s for s in rating['slots'] if s['report']==rating['latest_report']),None)
    require(slot is not None and slot['generation']<=state['iteration'],'Missing current committed rating fit')
    fit={k:copy.deepcopy(slot[k]) for k in ('generation','report','bootstrap','spec')}
    if controller['last_fit']==fit:return None,fit,'same_completed_fit'
    if controller['last_fit'] is not None:
        require(fit['generation']>controller['last_fit']['generation'],'Policy anchor rating fit went backwards or changed')
    report,samples,names=load_fit(root,rating,fit)
    current=controller['active'];reference=current['agent'] if current else IL
    generation=current['checkpoint']['generation'] if current else 0
    candidates=sorted((a for a in names if a not in (IL,V4) and a in rating['agents'] and
        rating['agents'][a]['checkpoint']['generation']>generation),
        key=lambda a:(-report['ratings'][a]['points'],-rating['agents'][a]['checkpoint']['generation'],a))
    for agent in candidates:
        checkpoint=rating['agents'][agent]['checkpoint']
        if checkpoint['model_id']==rating['agents'][reference]['checkpoint']['model_id']:continue
        evidence=paired_difference(report,samples,names,agent,reference,rating['config']['elo_scale'])
        if evidence['points_delta_ci95'][0]<=0:continue
        path=project_path(root,checkpoint['path'])
        if not path.is_file():continue  # Explicitly select among available evaluated checkpoints.
        require(sha(path)==checkpoint['sha256'],'Qualified policy reference checkpoint changed')
        return dict(agent=agent,checkpoint=copy.deepcopy(checkpoint),fit=fit,evidence=evidence),fit,'qualified_improvement'
    return None,fit,'no_available_newer_qualified_checkpoint'


def cache(root, run, selected, probe, update, device):
    """Separate immutable cache; never overwrite IL or pool probability arrays."""
    from .retention import safe
    key=digest(dict(model_id=selected['checkpoint']['model_id'],probe_hash=probe.metadata['probe_hash']))
    directory=safe(run,run/'policy-anchor'/'probabilities');directory.mkdir(parents=True,exist_ok=True)
    manifest=safe(run,directory/(key+'.json'))
    binding=dict(contract=CONTRACT,model_id=selected['checkpoint']['model_id'],probe_hash=probe.metadata['probe_hash'])
    if manifest.exists():
        metadata=read(manifest);require(metadata['binding']==binding,'Policy reference cache binding changed')
        return dict(path=manifest.relative_to(root).as_posix(),sha256=sha(manifest))
    with torch.random.fork_rng(devices=[]):
        model,raw=load_checkpoint(project_path(root,selected['checkpoint']['path']),device=device)
        require(raw['model_id']==selected['checkpoint']['model_id'],'Policy reference model identity differs')
        values=probe.probabilities(model,update)
    path=safe(run,directory/(key+'.npz'));temp=path.with_suffix('.tmp')
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            require(np.array_equal(values,old['reference']),'Uncommitted policy reference cache differs')
    else:
        with temp.open('wb') as handle:np.savez(handle,reference=values)
        temp.replace(path)
    from .ratings import write_once,receipt
    write_once(manifest,dict(binding=binding,probabilities=receipt(root,path)))
    return receipt(root,manifest)


def values(root, active, probe):
    if active is None:return None
    receipt=active['cache'];path=project_path(root,receipt['path'])
    require(sha(path)==receipt['sha256'],'Policy anchor cache manifest changed')
    metadata=read(path)
    require(metadata['binding']==dict(contract=CONTRACT,model_id=active['checkpoint']['model_id'],
        probe_hash=probe.metadata['probe_hash']),'Policy anchor cache/probe identity differs')
    from .quality_pool import cached
    return cached(root,dict(probe=dict(metadata['probabilities'],probe_hash=probe.metadata['probe_hash'])),probe)


def verify_active(root,state,controller):
    active=controller['active']
    if active is None:return
    checkpoint=active['checkpoint'];rating=state['ratings']
    require(active['agent'] in rating['agents'] and rating['agents'][active['agent']]['checkpoint']['model_id']==checkpoint['model_id']
        and sha(project_path(root,checkpoint['path']))==checkpoint['sha256'],'Active policy anchor checkpoint changed')
    panel_gates.verify_evidence(root,rating,'il-panel',active['qualification'])
    report,samples,names=load_fit(root,rating,active['fit'])
    evidence=paired_difference(report,samples,names,active['agent'],active['evidence']['reference'],rating['config']['elo_scale'])
    require(evidence==active['evidence'] and evidence['points_delta_ci95'][0]>0,'Active anchor promotion evidence differs')


def verify_checkpoint(state,raw):
    """A pre-PPO hotfix is the sole permitted calibration/checkpoint overlay."""
    controller=validate_state(state['policy_anchor'],state['iteration'])
    pending=controller['pending'];saved=raw['training'].get('il_kl_limit')
    current=state['il_kl_limit'];previous=raw['training'].get('policy_anchor')
    if pending is None:
        if current!=saved and token(controller) is None and token(previous) is None:
            from .g110_hotfix import verify_checkpoint as verify_g110
            verify_g110(state,raw)
        else:require(current==saved,'Anchor calibration differs from committed checkpoint')
        require(token(controller)==token(previous),'Anchor state differs from committed checkpoint')
        return
    require(pending['checkpoint']==state['history'][-1] and raw['step']==state['iteration'] and
        raw['model_id']==pending['checkpoint']['model_id'] and digest(saved)==pending['before_limit'] and
        pending['previous_anchor']==token(previous) and pending['next_anchor']==token(controller),
        'Policy anchor transition is not bound to the committed checkpoint')
    expected,report=il_kl_limit.reset_reference(saved,pending['measurement'],generation=state['iteration'])
    require(expected==current and digest(current)==pending['after_limit'] and report==pending['reset_report'],
        'Policy anchor calibration transition cannot be reproduced')


def prepare(root,run,state,config,probe,model):
    """Apply at committed progress before collection; atomic state only, no PPO."""
    if 'policy_anchor' not in config:return None,None
    validate(config);controller=copy.deepcopy(state.get('policy_anchor',initial(state['iteration'])))
    validate_state(controller,state['iteration']);verify_active(root,state,controller)
    if controller['pending'] is not None:
        return values(root,controller['active'],probe),dict(decision='hold',reason='pending_reference_restored',generation=state['iteration'])
    qualification=panel_gates.prepare(root,state,config)['retired']['il-panel']
    selected=fit=None;reason='il_panel_not_qualified'
    if qualification is not None:selected,fit,reason=select(root,state,controller)
    if reason=='same_completed_fit':
        return values(root,controller['active'],probe),dict(decision='hold',reason=reason,generation=state['iteration'])
    decision=dict(decision='hold',reason=reason,generation=state['iteration'])
    new_limit=state['il_kl_limit']
    if selected:
        selected.update(selected_generation=state['iteration'],qualification=copy.deepcopy(qualification))
        selected['cache']=cache(root,run,selected,probe,config['update'],next(model.parameters()).device)
        reference=values(root,selected,probe)
        measurement=probe.measure(model,config['update'],reference=reference)
        new_limit,reset=il_kl_limit.reset_reference(state['il_kl_limit'],measurement,generation=state['iteration'])
        previous=token(controller);controller['active']=selected
        controller['pending']=dict(checkpoint=copy.deepcopy(state['history'][-1]),previous_anchor=previous,
            next_anchor=token(controller),before_limit=digest(state['il_kl_limit']),after_limit=digest(new_limit),
            measurement=measurement,reset_report=reset)
        decision.update(decision='promote',reference_generation=selected['checkpoint']['generation'],
            evidence=copy.deepcopy(selected['evidence']),calibration=reset)
        controller['events'].append(copy.deepcopy(decision))
    controller.update(last_fit=fit,last_decision=decision)
    committed=dict(state,policy_anchor=controller,il_kl_limit=new_limit)
    require(read(run/'state.json')==json.loads(json.dumps(state,allow_nan=False)),
        'Run advanced during policy anchor transition')
    write_json(run/'state.json',committed)
    state.update(policy_anchor=controller,il_kl_limit=new_limit)
    return values(root,controller['active'],probe),decision


def annotate(report,state):
    active=(state.get('policy_anchor') or {}).get('active')
    report['anchor_reference']=copy.deepcopy(active['checkpoint'] if active else state['evaluation_anchor'])
    report['anchor_promotion']=copy.deepcopy(active['evidence']) if active else None
    report['original_il_measurement']=copy.deepcopy(report['measurement'].get('original_il',report['measurement']))
    report['lr_increases_permitted_by_anchor']=report['lr_gate']['ok']
    report['lr_gate']['scope']='Active frozen policy reference on fixed IL probe; independently anchored economic health'
    return report


def console(decision):
    if not decision or decision['decision']!='promote':return
    r=decision['calibration'];e=decision['evidence']
    print(f'KL reference -> g{decision["reference_generation"]}: paired delta={e["points_delta"]:.2f} '
        f'CI95={e["points_delta_ci95"]}; AnchorKL={r["measurement"]["kl_mean"]:.5f}/0.15000 '
        f'ILKL={r["measurement"]["original_il"]["kl_mean"]:.5f} group_limit=0.30000 '
        f'anchor/econ permits LR increases={r["lr_anchor_health_permits"]}',flush=True)
