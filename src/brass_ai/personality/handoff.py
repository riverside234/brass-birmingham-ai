"""P0: verify and freeze an accepted parent, its evidence and eligible traces."""
import gzip
import shutil
from pathlib import Path

import torch

from brass_ai.bank import sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import (assert_compatible, decode_json, digest, project_path,
    read_json, require)
from brass_ai.learning.reward import specification
from brass_ai.recovery.champion import CONTRACT as CHAMPION, decision
from brass_ai.recovery.retention import owned, safe, run_lock, active
from brass_ai.recovery.utility import calibration_ready, reward_spec
from brass_ai.step5.league import write_json
from .contracts import HANDOFF,contract,local_file,schema,accepted_contract


def selection_evidence(state,best,report,gate_spec):
    """The same selection proof applies to a stopped run and its portable copy."""
    require(state.get('format')=='step5-recovery-league/2.2.0','Unsupported Step 5 run')
    require(best.get('format')==CHAMPION and best['checkpoint']==state.get('champion'),
            'Best receipt differs from selected incumbent')
    entry=best['checkpoint'];gate=state.get('champion_gate',{}).get('accepted_gate')
    require(entry['generation']>0 and gate and gate.get('accepted') and best['gate']==gate,
            'No accepted post-IL PPO parent; initial/IL incumbent is not eligible')
    require(best['selected_games']==state['champion_gate']['accepted_games']==gate['games_done']
        and gate['candidate_generation']==entry['generation'],'Selected generation/gate differs')
    require(report['format']==CHAMPION and digest(gate_spec)==report['spec_id'],'Gate specification differs')
    require(gate_spec['format']==CHAMPION and set(gate_spec['seeds'])=={r['seed'] for r in report['outcomes']},'Gate setup identities differ')
    require(gate_spec['candidate']['model_id']==entry['model_id'] and
            gate_spec['candidate']['generation']==entry['generation'],'Gate selected another model')
    outcome=decision(report['outcomes'],gate_spec['config'])
    require(outcome==report['result'] and outcome['accepted'] and report['failed_games']==0
        and all(gate.get(k)==v for k,v in outcome.items()),'Incomplete/unaccepted gate evidence')
    return entry,gate


def inspect(root,source,*,_owner=False):
    root,run=owned(root,source)
    require(not active(run,allow_owner=_owner),'Step 5 is still active; wait for the run to stop before handoff')
    require(all((run/n).is_file() for n in ('state.json','best.json','best.pt')),'Completed Step 5 selection files are not available yet')
    state=read_json(run/'state.json');best=read_json(run/'best.json')
    gate=state.get('champion_gate',{}).get('accepted_gate')
    require(gate,'No accepted post-IL PPO parent; initial/IL incumbent is not eligible')
    report_path=safe(run,project_path(root,gate['report']))
    require(sha(report_path)==gate['report_sha256'],'Accepted gate report changed')
    report=read_json(report_path);spec_path=report_path.parent/'spec.json';gate_spec=read_json(spec_path)
    entry,gate=selection_evidence(state,best,report,gate_spec)
    path=safe(run,project_path(root,entry['path']))
    require(sha(path)==entry['sha256']==sha(run/'best.pt'),'Selected parent bytes changed')
    signature=state['signature'];assert_compatible(signature['compatibility'])
    reward=reward_spec(signature['config'])
    require(reward==specification(),'Parent uses another reward; explicit personality migration required')
    with torch.random.fork_rng(devices=[]):model,raw=load_checkpoint(path)
    require(raw['format']=='four-seat-checkpoint/2.0.0' and raw['model_id']==entry['model_id']
        and raw['training'].get('mode')=='step5-population-ppo'
        and raw['training'].get('fresh_run')==run.relative_to(root).as_posix()
        and raw['step']==entry['generation'],'Parent is not this run\'s trained PPO checkpoint')
    require(model.config.utility==reward and calibration_ready(model),'Parent utility reward/calibration is not ready')
    return dict(root=root,run=run,state=state,best=best,entry=entry,model=model,raw=raw,
                report_path=report_path,spec_path=spec_path,reward=reward)


def _traces(root,run,state,limit):
    """Use authoritative ordinary-game receipts, never gate/evaluation/search targets."""
    from brass_ai.recovery.replay import read_shards
    selected=[];seen=set()
    for report_path in sorted(run.glob('iteration-*/report.json'),reverse=True):
        report=read_json(report_path)
        for record in report.get('execution',[]):
            path=safe(run,project_path(root,record['path']))
            if path in seen or not path.is_file():continue
            seen.add(path)
            require(sha(path)==record['sha256'],'Recorded training shard changed')
            with gzip.open(path,'rt',encoding='utf-8') as f:header=decode_json(next(f))
            provenance=header['provenance']
            jobs=[j['provenance'] for j in provenance.get('jobs',[])] or [provenance]
            require(provenance.get('purpose')=='training' and all(j.get('purpose')=='training' and
                j.get('kind') in ('selfplay','historical','legacy') and j.get('spec')==digest(state['signature'])
                for j in jobs),'Not a current ordinary training shard')
            games=read_shards([path])
            require(all(g['split']=='train' and all(r['teacher'] is None for r in g['records']) for g in games),
                    'Evaluation or searched replay cannot be an ordinary handoff trace')
            selected.append(dict(source=path,sha256=record['sha256'],report=report_path,
                games=[dict(game_id=g['game_id'],setup_group=g['setup_group'],split=g['split']) for g in games]))
            if len(selected)>=limit:return selected
    return selected


def create(root,source,output,*,trace_shards=4,reference_count=3):
    root,run=owned(root,source);_,out=owned(root,output)
    require(not out.is_relative_to(run) and not run.is_relative_to(out),'Handoff must be separate from the Step 5 run')
    require(not out.exists(),'Use a new immutable handoff directory')
    require(type(trace_shards) is int and 0<=trace_shards<=64 and
            type(reference_count) is int and 0<=reference_count<=16,'Invalid bounded handoff size')
    # This is the same lock used by training and cleanup. Inspection is repeated
    # inside the lease; no model may change while copying its receipts.
    require(not active(run),'Step 5 is still active')
    with run_lock(run):
        # inspect() recognizes only a stopped source; allow our own temporary lease.
        data=_inspect_owned(root,run)
        state=data['state'];parent=data['entry'];fresh=data['raw']['training']['fresh_run']
        references=[];ids={parent['model_id']}
        candidates=[state['evaluation_anchor']]+list(state.get('population',[]))
        # Stable age spread, not merely the newest snapshots.
        unique={e['model_id']:e for e in candidates if e['model_id'] not in ids}
        ordered=sorted(unique.values(),key=lambda e:e['generation'])
        count=min(reference_count,len(ordered))
        indexes=sorted({round(i*(len(ordered)-1)/max(1,count-1)) for i in range(count)})
        for i in indexes:
            entry=ordered[i];path=safe(run,project_path(root,entry['path']))
            require(sha(path)==entry['sha256'],'Reference checkpoint changed')
            with torch.random.fork_rng(devices=[]):_,raw=load_checkpoint(path)
            require(raw['model_id']==entry['model_id'] and raw['training'].get('fresh_run')==fresh,'Foreign reference lineage')
            references.append((entry,path))
        traces=_traces(root,run,state,trace_shards) if trace_shards else []
        sources=[project_path(root,parent['path']),data['report_path'],data['spec_path']]
        sources.extend(p for _,p in references)
        sources.extend(t[k] for t in traces for k in ('source','report'))
        out.mkdir(parents=True);files=[]
        # Retention discovers pins across all owned artifact directories. Keep
        # this handoff's source dependencies here so the completed training run
        # (including any existing pins) stays unchanged, even if copying fails.
        source_pins=[p.relative_to(root).as_posix() for p in sources]
        write_json(out/'pins.json',dict(paths=sorted(set(source_pins))))
        def copy_file(path,destination):
            target=out/destination;target.parent.mkdir(parents=True,exist_ok=True)
            before=sha(path);shutil.copyfile(path,target)
            require(sha(path)==before==sha(target),'File changed during handoff')
            item=dict(path=destination,sha256=before);files.append(item);return item
        checkpoint=copy_file(project_path(root,parent['path']),'parent.pt')
        refs=[dict(**copy_file(p,f'references/{e["model_id"]}.pt'),model_id=e['model_id'],generation=e['generation']) for e,p in references]
        receipts={p:copy_file(p,f'evidence/trace-receipt-{i:04d}.snapshot')
                  for i,p in enumerate(dict.fromkeys(t['report'] for t in traces))}
        copied_traces=[dict(**copy_file(t['source'],f'traces/{i:04d}.jsonl.gz'),games=t['games'],receipt=receipts[t['report']],
            context='Legacy style/intent preserved; new personality context absent; auxiliary-only') for i,t in enumerate(traces)]
        # Snapshot suffix keeps retention's recursive JSON dependency scan from
        # pinning every historical checkpoint path inside the full old state.
        copy_file(run/'state.json','evidence/step5-state.snapshot')
        copy_file(run/'best.json','evidence/best.snapshot')
        copy_file(data['report_path'],'evidence/accepted-report.snapshot')
        copy_file(data['spec_path'],'evidence/accepted-spec.snapshot')
        metadata=('configs/contracts/v0.2.0/manifest.json','configs/contracts/v0.2.0/wire.schema.json',
            'configs/learning/features-1.0.0.json','configs/learning/features-2.0.0.json','configs/learning/features-3.0.0.json','configs/learning/host-1.0.0.json',
            'training/opponents/current.json','training/opponents/pool.json','training/runtime/distribution.json')
        for i,name in enumerate(metadata):copy_file(root/name,f'evidence/runtime-{i}.snapshot')
        current=read_json(root/'training/opponents/current.json')
        require(state['signature']['legacy_pool']==read_json(root/'training/opponents/pool.json'),'Selected training bank/pool differs')
        copy_file(project_path(root,f'training/opponents/bank/{current["bank"]}/manifest.json'),'evidence/bank.snapshot')
        copy_file(project_path(root,current['qualification']),'evidence/bank-selection.snapshot')
        copy_file(project_path(root,state['signature']['config']['host'])/'runtime.json','evidence/host-runtime.snapshot')
        manifest=dict(format=HANDOFF,contract=contract(),parent=dict(**checkpoint,model_id=parent['model_id'],
            generation=parent['generation'],fresh_run=fresh,parameters=data['model'].parameter_count,
            training_signature=data['raw']['training'].get('signature')),
            reward=data['reward'],reward_id=digest(data['reward']),utility_calibration=data['model'].utility_calibration,
            model_config=data['raw']['model_config'],signature=state['signature'],origin=state['origin'],
            references=refs,traces=copied_traces,files=files,goal_schema=schema(),
            selection=dict(gate=data['best']['gate'],qualification='Training selection only; not personality release'),
            trace_status='available' if traces else 'No retained ordinary traces; collect fresh P1 discovery games',
            source_run=run.relative_to(root).as_posix())
        write_json(out/'manifest.json',manifest)
        write_json(out/'pins.json',dict(paths=sorted(set(source_pins+[
            (out/f['path']).relative_to(root).as_posix() for f in files]))))
        load(out)
    return manifest


def _inspect_owned(root,run):
    # No lock removal or admission bypass: inspect permits this process's lease.
    return inspect(root,run, _owner=True)


def load(directory):
    directory=Path(directory);manifest=read_json(directory/'manifest.json')
    require(manifest.get('format')==HANDOFF,'Handoff format differs')
    accepted_contract(manifest['contract'])
    files={item['path']:item['sha256'] for item in manifest['files']}
    require(len(files)==len(manifest['files']),'Duplicate handoff evidence files')
    for item in manifest['files']:
        require(sha(local_file(directory,item['path']))==item['sha256'],'Handoff dependency changed: '+item['path'])
    def included(item):
        require(files.get(item['path'])==item['sha256'],'Unlisted handoff dependency evidence')
    included(manifest['parent'])
    for item in manifest['references']:included(item)
    for item in manifest['traces']:
        included(item);included(item['receipt'])
    required=('step5-state','best','accepted-report','accepted-spec')
    require(all('evidence/'+name+'.snapshot' in files for name in required),'Missing authoritative handoff evidence')
    state,best,report,gate_spec=[read_json(local_file(directory,'evidence/'+name+'.snapshot')) for name in required]
    entry,gate=selection_evidence(state,best,report,gate_spec)
    require(files['evidence/accepted-report.snapshot']==gate['report_sha256']
        and manifest['signature']==state['signature'] and manifest['origin']==state['origin']
        and manifest['selection']['gate']==gate and manifest['parent']['generation']==entry['generation']
        and manifest['parent']['model_id']==entry['model_id'] and manifest['parent']['sha256']==entry['sha256'],
        'Handoff claims differ from authoritative source evidence')
    available={e['model_id']:e for e in [state['evaluation_anchor']]+list(state.get('population',[]))}
    require(len({r['model_id'] for r in manifest['references']})==len(manifest['references'])
        and all(r['model_id']!=entry['model_id'] and r['model_id'] in available
            and all(r[k]==available[r['model_id']][k] for k in ('sha256','generation'))
            for r in manifest['references']), 'Handoff reference evidence differs')
    assert_compatible(manifest['signature']['compatibility'])
    require(manifest['reward_id']==digest(manifest['reward']) and manifest['reward']==specification(),'Handoff reward differs')
    with torch.random.fork_rng(devices=[]): model,raw=load_checkpoint(local_file(directory,manifest['parent']['path']))
    require(raw['model_id']==manifest['parent']['model_id'] and model.config.utility==manifest['reward']
        and model.utility_calibration==manifest['utility_calibration'] and calibration_ready(model),'Handoff model/calibration differs')
    require(raw['format']=='four-seat-checkpoint/2.0.0' and raw['training'].get('mode')=='step5-population-ppo'
        and raw['step']==manifest['parent']['generation']
        and raw['training'].get('fresh_run')==manifest['parent']['fresh_run']==manifest['source_run']
        and raw['training'].get('signature')==manifest['parent']['training_signature']
        and raw['model_config']==manifest['model_config'] and model.parameter_count==manifest['parent']['parameters']
        and manifest['goal_schema']==schema() and reward_spec(manifest['signature']['config'])==manifest['reward'],
        'Handoff model/provenance evidence differs')
    return manifest,model
