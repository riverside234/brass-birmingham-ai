#!/usr/bin/env python3
"""Restore an exported boundary in this project, preserve the newer run, and
continue with revised rewards, empty replay buffers and all current stages.

The project ROOT never changes. Only its owned run directory is retired to an
ignored, reversible backup. Existing transfer/checkpoint migration code owns
all reward, optimizer, RNG and buffer-reset semantics.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def transfer_module(root):
    spec=importlib.util.spec_from_file_location('revised_reward_transfer',root/'scripts/step5-transfer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare(root,bundle,run_path,generation,preserve_generation):
    from brass_ai.learning.contracts import digest,require,project_path
    from brass_ai.learning.post_transfer_reward import REVISION,revise
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.recovery.transfer_resume import install_instruction,instruction,MARKER,_publish
    transfer=transfer_module(root);root,run=owned(root,run_path)
    manifest=transfer.inspect_bundle(bundle)  # Reject a bad/wrong archive before moving the current run.
    require(manifest['iteration']==generation and manifest['run']==run.relative_to(root).as_posix(),
            'Archive must contain the selected generation and the original run name')
    require(manifest.get('buffer_rebuild',{}).get('reward_revision')==REVISION,
            'Use the g1284 fresh-buffer archive containing the complete reward revision')
    require(transfer.source_identity(root,recipe=manifest.get('source_recipe',1))==manifest['implementation'],
            'Pull the matching current training code before restoring; the complete Stage-C implementation is source-bound')
    key=digest([manifest['id'],manifest['run'],generation])[:20]
    journal=project_path(root,'.training-rollback/restore-'+key+'.json')
    record=transfer.read(journal) if journal.exists() else None
    if record is not None:
        require(record.get('contract')=='step5-inplace-export-revision/1.0.0' and
            record.get('archive_id')==manifest['id'] and record.get('run')==manifest['run'] and
            record.get('restore_generation')==generation and record.get('preserved_generation')==preserve_generation and
            record.get('id')==digest({k:v for k,v in record.items() if k!='id'}),
            'Existing rollback journal belongs to another operation')
    with run_lock(run):
        owner=transfer.read(run/'.active-run.json')
        def clear_own_marker(folder):
            file=folder/'.active-run.json'
            if file.is_file() and transfer.read(file).get('token')==owner['token']:file.unlink()
        backup=None
        try:
            if record is None:
                state=transfer.read(run/'state.json');entry=state['history'][-1]
                require(state['iteration']==entry['generation']==preserve_generation,
                    'Current committed generation differs; nothing moved. Set --preserve-current-generation to its actual value')
                checkpoint=project_path(root,entry['path'])
                require(checkpoint.is_relative_to(run) and transfer.sha(checkpoint)==entry['sha256'],
                        'Current checkpoint is not an intact checkpoint of this run')
                backup=project_path(root,'.training-rollback/'+run.name+'-g'+str(preserve_generation)+'-'+uuid.uuid4().hex[:12])
                record=dict(contract='step5-inplace-export-revision/1.0.0',archive_id=manifest['id'],run=manifest['run'],
                    restore_generation=generation,preserved_generation=preserve_generation,
                    backup=backup.relative_to(root).as_posix(),state_sha256=transfer.sha(run/'state.json'),
                    checkpoint=entry,checkpoint_suffix=checkpoint.relative_to(run).as_posix())
                record['id']=digest(record);_publish(journal,transfer.encode(record))
            backup=project_path(root,record['backup'])
            require(backup.parent==project_path(root,'.training-rollback') and backup!=run,
                    'Rollback backup escaped its dedicated directory')
            checkpoint_in_run=project_path(root,manifest['run']+'/'+record['checkpoint_suffix'])
            checkpoint_in_backup=project_path(root,record['backup']+'/'+record['checkpoint_suffix'])
            require(checkpoint_in_run.is_relative_to(run) and checkpoint_in_backup.is_relative_to(backup),
                    'Rollback checkpoint escaped its owned directory')
            if not backup.exists():
                require(transfer.sha(run/'state.json')==record['state_sha256'] and
                    transfer.sha(checkpoint_in_run)==record['checkpoint']['sha256'],
                    'Current run changed since rollback planning; nothing moved')
                backup.parent.mkdir(parents=True,exist_ok=True)
                run.rename(backup)
            require(backup.is_dir() and transfer.sha(backup/'state.json')==record['state_sha256'] and
                transfer.sha(checkpoint_in_backup)==record['checkpoint']['sha256'],
                'Preserved newer run changed; refusing to continue rollback')
            # Keep the OS-level run guard held while the transfer tool takes its
            # own short marker lease. No trainer can acquire this run meanwhile.
            clear_own_marker(backup);clear_own_marker(run)
            marker=instruction(root,run) if (run/MARKER).is_file() else None
            if marker is None:
                transfer.restore(root,bundle,relocate=True,use_checkout_code=True)
                # Explicit user-authorized same-machine reset. An ordinary
                # in-place archive verification must still never enable this.
                install_instruction(root,manifest)
                marker=instruction(root,run)
            require(marker['archive_id']==manifest['id'] and marker['origin_generation']==generation and
                marker.get('reward_revision')==REVISION,'Restored run uses a different buffer/reward instruction')
            state=transfer.read(run/'state.json')
            require(state['iteration']>=generation and marker['checkpoint'] in state['history'],
                    'Restored checkpoint lineage differs')
            expected=revise(state['signature']['config'])
            require(expected['reward']['placement']==.05 and expected['insolvency']['event_cost']==0.,
                    'Restored reward revision differs')
            return dict(root=str(root),run=manifest['run'],generation=state['iteration'],
                backup=str(backup),archive_id=manifest['id'],restore_generation=generation,
                preserved_generation=preserve_generation,reward=expected['reward'])
        finally:
            if backup is not None:clear_own_marker(backup)
            clear_own_marker(run)


def revision_summary(state,share_target):
    """Check the entire requested recipe, not just the reward toggle."""
    from brass_ai.learning.contracts import require
    from brass_ai.learning.post_transfer_reward import revise,AFTER,ZERO_COST
    from brass_ai.rust_backend.lab_teacher_activation import configure
    config=configure(revise(state['signature']['config']),share_target,revise_stages=True)
    phase=config['teacher_phase'];value=phase['search_value'];schedule=value['schedule']
    require(config['reward']==AFTER and config['insolvency']==ZERO_COST and
        config['branching']['rate_per_decision']==0. and phase['mode']=='on',
        'Requested reward/branching/teacher changes are incomplete')
    require((schedule['stage_a_iterations'],schedule['stage_b_iterations'],schedule['stage_c_iterations'])==(10,20,10) and
        schedule['validation_interval']==10 and schedule['validation_during_beta_qualification'] is True and
        value['head_to_head']['games']==0 and value['trunk']['kl']==.003 and value['buffer']['iterations']==8 and
        config['value_phase']['beta_clone_min']==.25,'Requested auxiliary/PPG schedule changes are incomplete')
    require(phase['search']['method']=='search_v1' and phase['search']['v205_enabled'] is False and
        phase['search']['allowance_seconds']==4. and phase['search']['post_search_seconds']==1. and
        phase['fast_search']['post_search_seconds']==.5,'Requested search profile differs')
    require(config.get('diversity',{}).get('mode','off')=='off','DOMiNO must remain disabled')
    # These implementations are included in the full recipe-4 source identity
    # verified before rollback; no checkpoint/config can substitute older code.
    from brass_ai.recovery.search_value_retention import reserve_games,HeldoutLoss
    from brass_ai.recovery.search_value_training import unconditional_pass
    import inspect
    require(callable(reserve_games) and callable(HeldoutLoss) and
        'heldout' in inspect.signature(unconditional_pass).parameters,'Stage-C held-out stopping implementation is missing')
    return dict(reward=config['reward'],insolvency=config['insolvency'],branching=0.,teacher='on',
        fixed_beta=state['teacher_phase']['weight']['gradient_qualification']['beta'],
        stages=dict(A=10,B=20,C=10,trunk_increment=.1,kl_cap=.003,head_to_head_games=0),
        stage_c_holdout=dict(games=8,buffer_iterations=8,stop_on_loss_increase=True,interval_steps=16),
        diagnostic_holdout=dict(games=4,every_iterations=10,beta_qualification_every_iteration=True),
        ppg_beta_floor=.25,return_buffer_rebuild_ages=8,stage_counter_paused_during_rebuild=True,
        search='search_v1',v205_enabled=False,domino_enabled=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,default=1284)
    p.add_argument('--preserve-current-generation',type=int,default=1286)
    p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--share-target',type=float,choices=(.25,.4,.5),default=.25)
    p.add_argument('--device',default='cuda');p.add_argument('--mono',default='mono')
    p.add_argument('--prepare-only',action='store_true',help='Restore/authorize the revision; do not start training')
    args=p.parse_args()
    if args.iterations<=0 or args.preserve_current_generation<args.expected_generation:p.error('Invalid generation/iteration boundary')
    result=prepare(ROOT,args.bundle,args.run,args.expected_generation,args.preserve_current_generation)
    state=json.loads((ROOT/result['run']/'state.json').read_text(encoding='utf-8-sig'))
    result['revision']=revision_summary(state,args.share_target)
    print(json.dumps(result,indent=2),flush=True)
    if args.prepare_only:return
    def run(argv):
        print(shlex.join(argv),flush=True);subprocess.run(argv,check=True,cwd=ROOT)
    # Reuses the existing prepared teacher; only a missing variant needs a build.
    if result['generation']==args.expected_generation:
        run([sys.executable,str(ROOT/'scripts/prepare-rust-post-transfer-reward.py'),'--run',args.run,
             '--expected-generation',str(args.expected_generation)])
    run([sys.executable,str(ROOT/'scripts/step5-transfer.py'),'repair-runtimes','--root',str(ROOT),'--run',args.run])
    print('Full revision: placement=.05; zero insolvency; canal/final-delta VP; fresh buffers; teacher on; branching=0%.',flush=True)
    print('Stages A10/B20/C10; no 256-game gate; Stage C: eight buffered held-out games, loss-increase stop, hard KL=.003; PPG floor=.25.',flush=True)
    print('Four diagnostic held-out games every tenth iteration; beta-qualification override retained. Stage counter pauses for eight fresh return ages.',flush=True)
    run([sys.executable,str(ROOT/'scripts/continue-rust-teacher.py'),'--run',args.run,
        '--expected-generation',str(args.expected_generation),'--iterations',str(args.iterations),
        '--share-target',str(args.share_target),'--device',args.device,'--mono',args.mono])


if __name__=='__main__':main()
