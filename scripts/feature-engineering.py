"""Inspect the compact grouped model, explicitly re-audit raw data, or bind fresh IL."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from brass_ai.learning.contracts import compatibility,project_path,read_json,require
    from brass_ai.learning.model import FourSeatNet,ModelConfig,dense_multiply_adds
    from brass_ai.step5.league import write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('inspect')
    conversion=sub.add_parser('convert')
    conversion.add_argument('--manifest',required=True);conversion.add_argument('--output',required=True)
    conversion.add_argument('--mono')
    conversion.add_argument('--workers',type=int,
        help='CPU processes (1-48); default: up to 48, capped by available CPUs and replay shards')
    config=sub.add_parser('configure')
    config.add_argument('--imitation-run',required=True);config.add_argument('--output',required=True)
    config.add_argument('--base',default='configs/step5/joint7-latest.json')
    portable=sub.add_parser('export',help='Export the current utility checkpoint with CPU parity')
    portable.add_argument('--checkpoint',required=True);portable.add_argument('--output',required=True)
    portable.add_argument('--include-personality',action='store_true',
        help='Also export a newly initialized goal extension for integration diagnostics; not a qualified personality policy')
    a=parser.parse_args();root=a.root.resolve()
    if a.command=='convert':
        from brass_ai.feature_conversion import convert
        result=convert(root,a.manifest,a.output,mono=a.mono,workers=a.workers)
    elif a.command=='configure':
        from brass_ai.recovery.demonstrations import completed_imitation
        from brass_ai.recovery.config import latest_model_config,time_budget_config,champion_mix_config
        model,origin=completed_imitation(root,a.imitation_run)
        require(model.config==ModelConfig.preset('medium'),'Production grouped recipe requires width 304 and the pinned layout')
        path=project_path(root,a.output);require(not path.exists(),'Use a new feature-revision config path')
        base=read_json(project_path(root,a.base))
        # Fresh IL owns both initialization and rehearsal. Keeping the old
        # recipe's dataset would reject the handoff or admit incompatible data.
        base['rehearsal_manifest']=origin['manifest']
        result=champion_mix_config(time_budget_config(latest_model_config(base,origin)))
        write_json(path,result)
    elif a.command=='export':
        import torch
        from brass_ai.bank import sha
        from brass_ai.learning.checkpoint import load_checkpoint
        from brass_ai.learning.utility_model import UtilityNet
        from brass_ai.learning.state_layout import state_layout,LAYOUT_FILE
        from brass_ai.personality.model import import_parent
        from brass_ai.personality.checkpoint import save
        from brass_ai.personality.laptop.export import export
        from brass_ai.step5.league import implementation_id
        checkpoint=project_path(root,a.checkpoint);out=project_path(root,a.output)
        require(not out.exists(),'Use a new immutable export directory')
        model,raw=load_checkpoint(checkpoint)
        require(type(model) is UtilityNet,'Export requires the target self-play utility checkpoint')
        model_config={k:v for k,v in model.config.to_dict().items() if k!='utility'}
        require(ModelConfig.from_dict(model_config)==ModelConfig.preset('medium'),'Export the width-304 target preset')
        torch.set_num_threads(2);out.mkdir(parents=True)
        meta=export(model,out/'utility.onnx')
        models=dict(utility=dict(path='utility.onnx',model_id=meta['model_id'],actor=False,metadata=meta))
        if a.include_personality:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(661);actor,import_receipt=import_parent(model)
            save(out/'actor.pt',actor,training=dict(diagnostic=True,parent_model_id=raw['model_id'],seed=661))
            meta=export(actor,out/'actor.onnx')
            models['actor']=dict(path='actor.onnx',model_id=meta['model_id'],actor=True,metadata=meta,
                checkpoint='actor.pt',import_receipt=import_receipt,goals_trained=False)
        write_json(out/LAYOUT_FILE,state_layout().specification())
        result=dict(format='grouped-target-export/1.0.0',implementation=implementation_id(),
            checkpoint=dict(path=checkpoint.relative_to(root).as_posix(),sha256=sha(checkpoint),model_id=raw['model_id']),
            layout_file=LAYOUT_FILE,layout_sha256=state_layout().identity,models=models,
            scope='Current target only; float32 CPU parity. An optional initialized goal branch is diagnostic and not release-qualified.')
        write_json(out/'receipt.json',result)
    else:
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.learning.state_layout import state_layout,LAYOUT_VERSION,LAYOUT_FILE
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.personality.model import PersonalityNet,PersonalityConfig
        from brass_ai.feature_engineering.contracts import STATE_INPUTS,ACTION_INPUTS,STATE_OMISSIONS,ACTION_OMISSIONS
        model=FourSeatNet(ModelConfig.preset('medium'))
        utility=UtilityNet(utility_config(model.config,reward_spec(read_json(root/'configs/step5/joint7-latest.json'))))
        goals=PersonalityNet(PersonalityConfig(utility.config.to_dict()))
        result=dict(compatibility=compatibility(),model=model.config.to_dict(),
            state_projection=dict(layout_version=LAYOUT_VERSION,layout_file=LAYOUT_FILE,
                fusion_inputs=sum(g.width for g in state_layout().groups),context_passthrough=12,
                groups=[dict(name=g.name,inputs=len(g.indices),outputs=g.width) for g in state_layout().groups]),
            neural_fields=dict(state_facts=len(STATE_INPUTS),state_components=sum(len(f.components) for f in STATE_INPUTS),
                action_facts=len(ACTION_INPUTS),action_components=sum(len(f.components) for f in ACTION_INPUTS),
                omitted_state=STATE_OMISSIONS,omitted_actions=ACTION_OMISSIONS),
            parameters=dict(state_encoder=sum(p.numel() for p in model.state_encoder.parameters()),
                base=model.parameter_count,utility=utility.parameter_count,with_goals=goals.parameter_count),
            multiply_adds=dict(base=dense_multiply_adds(model),utility=dense_multiply_adds(utility),
                              with_goals=dense_multiply_adds(goals)),requires_fresh_IL=True)
    import json
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
