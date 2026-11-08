"""Two full shadow iterations qualify a fixed shared-encoder teacher weight."""
import copy
import math
import statistics

from brass_ai.learning.contracts import digest,require

CONTRACT='teacher-fixed-beta-qualification/1.0.0'
TARGETS=(.25,.4,.5)


def seal(receipt):
    result={k:copy.deepcopy(v) for k,v in receipt.items() if k!='sha256'}
    result['sha256']=digest(result);return result


def fixed_beta(target,raw):return min(10.,max(.01,target/raw))


def restore(receipt,target):
    if receipt is None:return None
    require(receipt==seal(receipt) and receipt['contract']==CONTRACT and receipt['target'] in TARGETS,
        'Teacher gradient qualification changed')
    result=copy.deepcopy(receipt)
    if result['passed']:
        require(len(result['observations'])==2 and result['raw_median']>0 and
            result['beta']==fixed_beta(result['target'],result['raw_median']), 'Invalid fixed teacher beta')
    require(target in TARGETS,'Fixed teacher beta requires target 0.25, 0.4 or 0.5')
    if target!=result['target']:
        # A manual target edit is the only way to change a qualified weight.
        result['target']=target
        if result['passed']:result['beta']=fixed_beta(target,result['raw_median'])
        result=seal(result)
    return result


def finish(term,state):
    if term.config['contract']!='teacher-phase/2.0.0':return
    target=term.config['term']['share_target']
    if target not in TARGETS:return  # Retain historical diagnostic-only recipes.
    q=restore(term.qualification,target)
    if q is not None and q['passed']:
        term.beta=q['beta'];term.calibrated=True
    elif term.mode=='shadow':
        observations=[] if q is None else q['observations']
        context=getattr(term,'qualification_context',{})
        require(all(s.get('raw_share') is None or math.isfinite(s['raw_share']) for s in term.steps),
            'Nonfinite shadow gradient qualification')
        valid=[s for s in term.steps if s.get('raw_share') is not None and s['raw_share']>0 and
            all(math.isfinite(s.get(k,float('nan'))) for k in ('raw_share','cosine','actor_norm','teacher_norm')) and
            s['actor_norm']>0]
        phase_rows=context.get('labels_by_phase',{})
        full=(context.get('tables')=={'train':28,'heldout':4} and
            all(phase_rows.get(p,0)>0 for p in ('canal_early','canal_late','rail_early','rail_late')))
        if valid and full:
            observation=dict(generation=term.generation,steps=len(term.steps),valid_steps=len(valid),
                raw_shares=[s['raw_share'] for s in valid],cosines=[s['cosine'] for s in valid],
                actor_norms=[s['actor_norm'] for s in valid],teacher_norms=[s['teacher_norm'] for s in valid],
                context=copy.deepcopy(context))
            require(not observations or term.generation>observations[-1]['generation'],
                'Duplicate/out-of-order teacher gradient qualification')
            observations=(observations+[observation])[-2:]
        consecutive=len(observations)==2 and observations[1]['generation']==observations[0]['generation']+1
        q=dict(contract=CONTRACT,passed=consecutive,target=target,observations=observations,
            reason='Qualified two complete consecutive shadow iterations' if consecutive else
                'Need two consecutive full 28+4 shadow iterations with finite nonzero actor/teacher measurements',
            scope='Pre-clipping PPO actor without entropy; shared state/entity encoders; shadow gradients discarded',
            beta=None,raw_median=None)
        if consecutive:
            q['raw_median']=statistics.median(v for o in observations for v in o['raw_shares'])
            q['beta']=fixed_beta(target,q['raw_median']);q['qualified_generation']=term.generation+1
            q['clamped']=q['beta']!=target/q['raw_median']
            term.beta=q['beta'];term.calibrated=True
        q=seal(q)
    term.qualification=q
    if q is not None:state['gradient_qualification']=copy.deepcopy(q)
    state.update(beta=term.beta,calibrated=term.calibrated)
