"""Inspect six-group features and five-block models, re-audit raw data, or bind IL."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from brass_ai.learning.contracts import compatibility,project_path,read_json,require
    from brass_ai.learning.model import FourSeatNet,ModelConfig
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
    a=parser.parse_args();root=a.root.resolve()
    if a.command=='convert':
        from brass_ai.feature_conversion import convert
        result=convert(root,a.manifest,a.output,mono=a.mono,workers=a.workers)
    elif a.command=='configure':
        from brass_ai.recovery.demonstrations import completed_imitation
        from brass_ai.recovery.config import latest_model_config,time_budget_config,champion_mix_config
        model,origin=completed_imitation(root,a.imitation_run)
        require(model.config==ModelConfig.preset('medium'),'Production five-block recipe requires width 304')
        path=project_path(root,a.output);require(not path.exists(),'Use a new feature-revision config path')
        base=read_json(project_path(root,a.base))
        # Fresh IL owns both initialization and rehearsal. Keeping the old
        # recipe's dataset would reject the handoff or admit incompatible data.
        base['rehearsal_manifest']=origin['manifest']
        result=champion_mix_config(time_budget_config(latest_model_config(base,origin)))
        write_json(path,result)
    else:
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.personality.model import PersonalityNet,PersonalityConfig
        model=FourSeatNet(ModelConfig.preset('medium'))
        utility=UtilityNet(utility_config(model.config,reward_spec(read_json(root/'configs/step5/joint7-latest.json'))))
        goals=PersonalityNet(PersonalityConfig(utility.config.to_dict()))
        result=dict(compatibility=compatibility(),model=model.config.to_dict(),
            parameters=dict(state_encoder=sum(p.numel() for p in model.state_encoder.parameters()),
                base=model.parameter_count,utility=utility.parameter_count,with_goals=goals.parameter_count),
            requires_fresh_IL=True)
    import json
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
