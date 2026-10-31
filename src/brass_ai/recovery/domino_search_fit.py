"""Provenance of an actually retained ordinary PPG intrinsic fit."""
import copy
from collections import defaultdict

import numpy as np
from brass_ai.learning.contracts import digest, require

CONTRACT = 'domino-search-fit/1.0.0'


def validate(receipt):
    from .domino_features import width, manifest
    from .teacher_diversity import NORMALIZATION
    require(receipt['contract'] == CONTRACT and receipt['identity'] == digest({k:v for k,v in receipt.items() if k != 'identity'}),
            'Corrupted retained intrinsic fit receipt')
    d = np.asarray(receipt['directions'], dtype=np.float64)
    require(d.shape == (7,4*width()) and np.isfinite(d).all() and not d[0].any()
            and receipt['descriptor'] == digest(manifest()) and receipt['normalization'] == NORMALIZATION,
            'Intrinsic fit descriptor/target differs')
    require(type(receipt['retained_steps']) is int and receipt['retained_steps'] > 0
            and receipt['checkpoint_generation'] == receipt['collection_generation']+1
            and receipt['direction_generation'] == receipt['collection_generation'], 'Invalid retained fit generation/steps')
    require(receipt['target']=='unweighted-episodic-intrinsic-reward-to-go/1.0.0'
            and type(receipt['normalization_proven']) is bool and len(receipt['nearest'])==7,
            'Unknown intrinsic fit target')
    for z,s in receipt['support'].items():
        require(z in tuple(map(str,range(7))) and type(s['rows']) is int and type(s['games']) is int
                and 0<s['games']<=s['rows'] and sum(s['phases'].values())==s['rows'],'Invalid intrinsic fit support')
    return receipt


def retain(diversity, model, dataset, retained_rows, retained_steps):
    """Called after keep-best has restored its selected weights/Adam/row set.

    Zero-step intervals leave the old receipt intact. A partial new fit records
    its actual support; it cannot borrow missing support from a different target.
    """
    if not retained_steps:
        return
    from . import domino_state as S
    from .domino_auxiliary import row_key
    from .local_scoring_auxiliary import bundle_id
    from .teacher_diversity import NORMALIZATION
    from .teacher_schedule import phase_for
    require(diversity.active, 'Cannot retain an inactive intrinsic fit')
    support = defaultdict(lambda: dict(rows=0,games=set(),phases=defaultdict(int),receipts=set(),descriptors=set()))
    ordered=sorted(retained_rows)
    for start in range(0,len(ordered),4096):
        ids=ordered[start:start+4096]
        if hasattr(dataset,'intrinsic_fit_metadata'):
            metadata=dataset.intrinsic_fit_metadata(ids)
        else:
            descriptors=(dataset.descriptors[ids,0].detach().cpu().tolist()
                         if hasattr(dataset,'descriptors') else [None]*len(ids))
            metadata=[(row_key(dataset,i),dataset.rows[i]['_domino'],
                       dataset.rows[i]['view']['context'],descriptor) for i,descriptor in zip(ids,descriptors)]
        for key,assignment,context,descriptor in metadata:
            require(key in diversity.train_keys and assignment is not None
                    and assignment['receipt']==diversity.target_receipts[key],
                    'Retained intrinsic row outside current ordinary fitting set')
            s=support[str(assignment['strategy_id'])]
            s['rows']+=1; s['games'].add(key[0]); s['phases'][phase_for(context)]+=1
            s['receipts'].add(digest(assignment))
            if descriptor is not None:s['descriptors'].add(tuple(descriptor))
    require(support, 'Retained intrinsic steps have no fitted rows')
    d,nearest,_ = S.directions(diversity.state['centroids'])
    value = dict(contract=CONTRACT, target='unweighted-episodic-intrinsic-reward-to-go/1.0.0',
        descriptor=diversity.config['descriptor_manifest_sha256'], normalization=NORMALIZATION,
        normalization_proven=diversity.search_normalization_proven,
        regime=diversity.state['regime'], directions=d.tolist(), nearest=nearest.tolist(),
        direction_generation=diversity.state['generation'], collection_generation=diversity.state['generation'],
        checkpoint_generation=diversity.state['generation']+1, centroid_hash=digest(diversity.state['centroids']),
        retained_steps=retained_steps, head_id=bundle_id(diversity.head.state_dict()), encoder_id=bundle_id(model.state_dict()),
        descriptor_input='acting-observer-relative-seat-zero/ordinary-critic-descriptors',
        support={z:dict(rows=s['rows'],games=len(s['games']),phases=dict(s['phases']),
                       context_receipts=sorted(s['receipts']),input_descriptors=[list(v) for v in sorted(s['descriptors'])]) for z,s in support.items()})
    diversity.search_fit = validate(dict(value,identity=digest(value)))
    diversity.report['search_fit'] = copy.deepcopy(diversity.search_fit)
