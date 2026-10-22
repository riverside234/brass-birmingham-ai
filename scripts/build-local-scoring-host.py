"""Build the separately admitted passive scoring recorder, never the pinned host."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def recorder_source(text, name, replace):
    if name=='Host.cs':
        text=replace(text,'Game=other.Game;', 'localRecorder=other.localRecorder; Game=other.Game;')
        text=replace(text,'["format"]="brass-trusted-checkpoint",', '["local_scoring"]=LocalScoringCheckpoint(),["format"]="brass-trusted-checkpoint",')
        text=replace(text,'h.GameId=(string)cp["game_id"];', 'h.GameId=(string)cp["game_id"]; if(cp["local_scoring"] is JObject)h.EnableLocalScoring();')
        text=replace(text,'h.Revision++; h.commands.Add((JObject)cmd.DeepClone());',
            'h.Revision++; h.commands.Add((JObject)cmd.DeepClone()); if(cp["local_scoring"] is JObject && h.Revision==(int)cp["local_scoring"]["start"])h.localRecorder.Begin(h.Revision);')
    elif name=='FastHost.cs':
        text=replace(text,'var setup=initial;', 'var setup=initial;var localMark=LocalScoringCheckpoint();')
        text=replace(text,'var cp=new JObject{', 'var cp=new JObject{["local_scoring"]=localMark,')
    elif name=='Program.cs':
        text=replace(text,'fast="training-host/1.0.0"', 'fast="training-host/1.0.0",local_scoring="local-scoring-targets/1.0.0"')
        text=replace(text,'case "context":',
            'case "local_scoring_enable":data=host.EnableLocalScoring((bool?)req["fresh"]??false);break;\n'
            'case "local_scoring_targets":data=host.LocalScoringTargets();break;\ncase "context":')
    return text


def build(root=ROOT,mono=None):
    # Reuse the existing authored throughput transformations verbatim. This
    # variant has independent outputs, builder/source pins and admission.
    text=(root/'scripts/build-training-host.py').read_text(encoding='utf-8-sig')
    spec=importlib.util.spec_from_loader('local_scoring_builder',loader=None)
    module=importlib.util.module_from_spec(spec)
    module.__file__=str(Path(__file__).resolve())
    text=text.replace('for search in (False,True):','for search in (False,):')
    text=text.replace("root/'artifacts/training-host'", "root/'artifacts/local-scoring-host'")
    text=text.replace("root/'training/runtime/throughput'", "root/'training/runtime/local-scoring'")
    text=text.replace("optimization='training-host/1.0.0'", "optimization='local-scoring-host/1.0.0'")
    text=text.replace("if search:originals+=", "originals+=sorted((root/'src/BrassLocalScoringHost').glob('*.cs'))\n        if search:originals+=")
    text=text.replace("target=out/", "text=recorder_source(text,path.name,replace)\n            target=out/",1)
    text=text.replace("authored=originals+", "authored=originals+[root/'scripts/build-training-host.py']+")
    module.__dict__['recorder_source']=recorder_source
    exec(compile(text,str(root/'scripts/build-training-host.py'),'exec'),module.__dict__)
    return module.build(root,mono)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--mono');a=p.parse_args()
    print(json.dumps(build(a.root.resolve(),a.mono),indent=2))
