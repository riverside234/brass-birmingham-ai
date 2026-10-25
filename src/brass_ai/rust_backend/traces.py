"""Freeze permitted NN replies and separate trusted sampled-world outcomes."""
from pathlib import Path
import random
import time

import numpy as np

from brass_ai.learning.contracts import digest,read_json,require
from .contracts import ScriptedReplies,permitted_request
from .fixtures import file_sha,read_gzip,write_gzip,write_json,harness_receipt


def capture_traces(root,out,baseline):
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.learning.batching import Evaluator,Request
    from brass_ai.learning.reward import CostLedger
    from brass_ai.searchlab.spiel import BrassSpielGame
    out=Path(out);directory=out/'local/fixtures';corpus=read_json(directory/'corpus.json')
    model,_=load_checkpoint(Path(root)/baseline['model']['path']);model.eval()
    evaluator=Evaluator(model,device='cpu',batch_size=1)
    engine=BrassSpielGame(root=root,fast=True);entries=[];worlds=[];seen=set()
    try:
        for item in corpus['roots']:
            if item['partition']!='development':continue
            fixture=read_gzip(directory/item['path']);p=fixture['projection'];c=p['context']
            if c['terminal'] or p['chance'] is not None:continue
            key=(c['era'],c['phase'],c['round']//3)
            if key in seen:continue
            seen.add(key);seat=c['actor'];view=p['views'][seat];legal=p['choices']
            request=permitted_request(view,legal,fixture['recall'][seat],request_id=item['id'],
                model_id=baseline['model']['model_id'],reward_context=fixture['ledger'])
            ledger=CostLedger(model.config.utility,**fixture['ledger'])
            response=evaluator.predict([Request(item['id'],evaluator.model_id,view,legal,time.monotonic()+60,
                                                remaining=tuple(ledger.remaining))])[0]
            require(response['status']=='ok','Reference NN reply failed')
            reply={k:response[k] for k in ('probabilities','win_share','placement','vp','utility','reward_id')}
            require(all(np.isfinite(reply[k]).all() for k in ('probabilities','win_share','placement','vp','utility')),
                    'Nonfinite reference NN reply')
            require(abs(sum(reply['probabilities'])-1)<1e-5,'Invalid NN probabilities')
            entry=dict(request=request,request_digest=digest(request),reply=reply,
                       scope='Frozen test-model reply for deterministic adapter replay, not a strength label')
            scripted=ScriptedReplies([entry]);require(scripted.predict(request)==reply,'Scripted reply changed');scripted.assert_consumed()
            entries.append(entry)
            for seed in (910001+len(entries)*2,910002+len(entries)*2):
                state=engine.from_observation(view,seed,fixture['recall'][seat])
                require(state.views[seat]==view and state.legal==legal,'Sampled world changed root information')
                worlds.append(dict(request_digest=digest(request),reference_seed=seed,
                    replay_kind='explicit sampled outcome; seed equality is not cross-language RNG equivalence',
                    setup=state.checkpoint['setup'],checkpoint_hash=state.checkpoint['hash']))
    finally:evaluator.close();engine.close()
    write_gzip(directory/'scripted-nn.json.gz',dict(format='rust-scripted-nn/1.0.0',model=baseline['model'],entries=entries))
    write_gzip(directory/'trusted-sampled-worlds.json.gz',dict(format='rust-sampled-world-tape/1.0.0',worlds=worlds))
    result=dict(passed=True,scripted_replies=len(entries),sampled_worlds=len(worlds),
        harness=harness_receipt('src/brass_ai/rust_backend/traces.py'),
        model_sha256=baseline['model']['sha256'],weights_unchanged=file_sha(Path(root)/baseline['model']['path'])==baseline['model']['sha256'],
        scripted_nn_sha256=file_sha(directory/'scripted-nn.json.gz'),
        trusted_worlds_sha256=file_sha(directory/'trusted-sampled-worlds.json.gz'),
        scope='Replay scaffolding and root sampling parity; full MaskedISMCTS selection/backup traces are L10')
    require(result['weights_unchanged'],'Checkpoint bytes changed')
    write_json(directory/'trace-qualification.json',result)
    return result
