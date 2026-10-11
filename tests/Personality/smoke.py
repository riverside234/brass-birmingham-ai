"""Three native games with a fixture parent; no strength or release claim."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'src',ROOT/'tests/R2',ROOT/'tests/Personality'):
    sys.path.insert(0,str(path))

import torch
from personality_fixtures import qualified_run
from brass_ai.bank import sha
from brass_ai.learning.contracts import read_json,require
from brass_ai.recovery.utility import calibration_ready
from brass_ai.step5.league import implementation_id,write_json
from brass_ai.personality import handoff,checkpoint
from brass_ai.personality.training import execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mono')
    a=parser.parse_args();out=a.output.resolve()
    require(out.is_relative_to((ROOT/'artifacts').resolve()) and not out.exists(),'Use a new artifacts smoke directory')
    torch.set_num_threads(2);torch.manual_seed(10203);began=time.monotonic();before=implementation_id()
    source=qualified_run(out/'fixture-source');source_hash=sha(source/'state.json')
    handoff.create(ROOT,source,out/'handoff',trace_shards=1,reference_count=1)
    common=dict(root=ROOT,bundle=out/'handoff',run=out/'pilot',epochs=1,device='cpu',mono=a.mono)
    warm=execute(**common,mode='warmup')
    joint=execute(**common,mode='joint',games=2,workers=2)
    identity=sha(out/'pilot/model.pt')
    validation=execute(**common,mode='validate',games=1,workers=1)
    require(sha(out/'pilot/model.pt')==identity,'Held-out diagnostics changed the model')
    model,raw=checkpoint.load(out/'pilot/model.pt')
    require(not calibration_ready(model.core),'Joint update cannot inherit parent search readiness')
    require(checkpoint.model_id(model)==read_json(out/'pilot/state.json')['model_id'],'Resume identity differs')
    require(before==implementation_id() and source_hash==sha(source/'state.json'),'Step 5 source/run changed')
    require(raw['optimizer'] is not None,'Missing P1 optimizer state')
    report=dict(scope='Authored parent/evidence fixture; three real host games, not trained-parent or playing-strength qualification',
        games=3,workers=[2,1],seconds=time.monotonic()-began,step5_implementation=before,
        warmup=warm,joint=joint,validation=validation,restored=True,parent_unchanged=True)
    write_json(out/'report.json',report)
    print(f'P0/P1 native smoke passed: 3 games in {report["seconds"]:.1f}s; report: {out / "report.json"}',flush=True)


if __name__=='__main__':main()
