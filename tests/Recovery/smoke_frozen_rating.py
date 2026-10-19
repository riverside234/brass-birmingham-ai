"""Two native opening decisions with the exact older reference from g55."""
import argparse
import copy
import random
import time

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery.execution import make_policy
from brass_ai.recovery.legacy_search import frozen_rating_runtime
from brass_ai.search.legacy_runtime import reference,runtime_directory
from brass_ai.step5.league import implementation_id,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated output directory')
    out.mkdir(parents=True);began=time.perf_counter();source=implementation_id()
    state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0055-20261019T053031361382Z/state.json')
    original=copy.deepcopy(state['ratings']['agents']['v4']['policy']);ref=original['search_runtime']
    current=reference(SOURCE_ROOT,ref['recipe']['bank'])
    reproduced=False
    try:runtime_directory(SOURCE_ROOT,ref)
    except ValueError as error:
        require('source changed' in str(error),'Unexpected original runtime error');reproduced=True
    # Restoring the production v4 strategy can make this reference current
    # again. Unit tests exercise source divergence; the native smoke verifies
    # the actual pinned binary without inventing a newer policy.
    runtime,pin=frozen_rating_runtime(SOURCE_ROOT,ref,original['parent'])
    files={p:sha(p) for p in runtime.iterdir() if p.is_file()}
    policy=make_policy(dict(original,frozen_rating=True),SOURCE_ROOT,None,None,None)
    rows=[]
    try:
        for seed in (910517,910518):
            worker=policy.opponent.worker;context=worker.call('reset',seed=seed)
            view=worker.call('observe',seat=context['actor']);choices=worker.call('choices')
            index,probability,_=policy.select_batch([(view,choices,random.Random(seed))])[0]
            require(0<=index<len(choices) and probability==1.,'Invalid frozen reference action')
            rows.append(dict(seed=seed,choices=len(choices),selected=choices[index]))
        require(policy.policy_id==original['policy_id'] and policy.search_free and policy.cpu_workers==1,
                'Rated policy identity/settings changed')
        require(policy.telemetry['completed']==0 and policy.telemetry['failures']==0,'Search ran or the reference failed')
        telemetry=copy.deepcopy(policy.telemetry)
    finally:policy.close()
    require(files=={p:sha(p) for p in files},'Frozen runtime files were modified')
    require(source==implementation_id(),'Source changed during smoke')
    write_json(out/'report.json',dict(passed=True,implementation=source,seconds=time.perf_counter()-began,
        source_mismatch=reproduced,rated_runtime=ref['id'],ordinary_runtime=current['id'],
        policy_id=original['policy_id'],runtime_manifest_sha256=pin,runtime_unchanged=True,
        openings=rows,telemetry=telemetry,
        scope='Two native opening decisions, no complete evaluation games, PPO, checkpoint edits or strength claim'))
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
