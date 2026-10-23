"""Twelve tiny authored iterations; two eight-age phases and exact restarts."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for p in ('src','tests/Recovery','tests/FeatureEngineering','tests/R2'):sys.path.insert(0,str(ROOT/p))


def main():
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.step5.league import write_json,implementation_id
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=project_path(ROOT,a.output)
    require(out.is_relative_to(ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    out.mkdir(parents=True);began=time.perf_counter()
    spec=importlib.util.spec_from_file_location('local_smoke',ROOT/'scripts/smoke-local-scoring.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rows=helper.training(out,phasic=True)
    result=dict(passed=True,source=implementation_id(),seconds=time.perf_counter()-began,training=rows,
        scope='Small CPU authored fixtures; no production checkpoints, native evaluations or strength claims')
    write_json(out/'report.json',result);print(str(out/'report.json'),flush=True)


if __name__=='__main__':main()
