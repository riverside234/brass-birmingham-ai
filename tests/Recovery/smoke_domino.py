"""Bounded authored DOMiNO bootstrap/active updates and exact full restart."""
import argparse
import copy
from pathlib import Path
import shutil
import time
import torch
from brass_ai.bank import sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.domino_auxiliary import Diversity,KEY as DKEY
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT as AUX,KEY as AUX_KEY
from brass_ai.recovery import updates
from brass_ai.recovery.utility import reward_spec
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.telemetry import memory
from brass_ai.step5.league import write_json
from test_domino import fixture,collect
from test_local_scoring import annotate
from test_gae_value_phase import finish
from test_value_buffer import commit


def advance(model,opt,c,p,run,g):
    b,rows=collect(model,p,g);annotate(rows);annotate(p.snapshot_data)
    reward=reward_spec(c)
    actor=updates.ppo_update(model,opt,[dict(split='train')],b,c['update'],learner_data=rows,
        insolvency=reward['insolvency'],reward=reward,phases=p)
    before=copy.deepcopy(opt.state_dict());value=p.fit(model)
    require(_same_payload(before,opt.state_dict()),'PPG changed PPO Adam')
    require(value['probe_kl']<=.002+1e-7,'Unsafe retained PPG policy')
    require(value['acceptance_generations']==list(range(max(1,g-1),g+1)),'Changed PPG cadence/window')
    finish(p,actor,g);path=run/f'generation-{g:04d}.pt'
    save_checkpoint(path,model,opt,step=g,training={KEY:p.payload(),DKEY:p.diversity.payload(),AUX_KEY:p.local_scoring.payload(g)})
    evidence=copy.deepcopy(p.diversity.report)
    commit(run,p,g);p.diversity.committed(g);p.local_scoring.committed(g)
    return path,dict(generation=g,actor=actor,value=value,diversity=evidence,memory=memory('cpu'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args()
    out=project_path(SOURCE_ROOT,args.output);require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use new isolated output')
    out.mkdir(parents=True);torch.set_num_threads(2);started=time.perf_counter()
    model,opt,c=fixture();c[AUX_KEY]=copy.deepcopy(AUX)
    run=out/'continuous';p=TrainingPhases(model,c,root=SOURCE_ROOT,run=run)
    p.local_scoring=Companion(model,AUX);p.diversity=Diversity(model,c['diversity'],regime='authored-complete-games')
    reports=[]
    for g in (1,2):path,r=advance(model,opt,c,p,run,g);reports.append(r)
    require(reports[0]['diversity']['readiness']=='centroids_incomplete' and reports[1]['diversity']['readiness']=='ready','Bootstrap did not activate from actual authored support')
    checksum=sha(path);restart=out/'restart';shutil.copytree(run,restart)
    rng=torch.get_rng_state().clone();other,raw=load_checkpoint(restart/path.name);torch.set_rng_state(rng)
    _,r=advance(model,opt,c,p,run,3);reports.append(r)
    expected=copy.deepcopy((model.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,
        p.diversity.state,p.diversity.snapshot(),p.local_scoring.payload(3),torch.get_rng_state()))
    other_opt=torch.optim.AdamW(other.parameters());resume_optimizer(raw,other_opt)
    q=TrainingPhases(other,c,raw=raw,generation=2,root=SOURCE_ROOT,run=restart)
    q.local_scoring=Companion(other,AUX,raw=raw,generation=2)
    q.diversity=Diversity(other,c['diversity'],raw=raw,generation=2)
    _,repeat=advance(other,other_opt,c,q,restart,3)
    actual=(other.state_dict(),other_opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,
        q.diversity.state,q.diversity.snapshot(),q.local_scoring.payload(3),torch.get_rng_state())
    require(_same_payload(expected,actual),'DOMiNO full restart changed tensors, statistics, optimizers or RNG')
    require(sha(path)==checksum,'Input checkpoint changed')
    require(not p.buffer.path(1,'critic.npy').exists(),'Expired cache survived new commit')
    result=dict(passed=True,seconds=time.perf_counter()-started,device='cpu',cuda_available=torch.cuda.is_available(),
        parameters=model.parameter_count,intrinsic_parameters=sum(v.numel() for v in p.diversity.head.parameters()),
        exact_restart=True,iterations=reports,
        scope='Three bounded authored iterations (one repeated after restart), 126 synthetic complete tables per collection, 1 PPO and up to 6 PPG steps per iteration; no game engines, strength claims or production checkpoints')
    write_json(out/'report.json',result)
    print({k:v for k,v in result.items() if k!='iterations'},flush=True)


if __name__=='__main__':main()
