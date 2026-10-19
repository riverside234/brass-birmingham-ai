"""Write a PFSP/60-iteration-rehearsal config without touching run or IL state."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from brass_ai.learning.contracts import project_path,read_json,require
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.quality_config import sampling_config,pfsp
from brass_ai.step5.league import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();source=project_path(ROOT,args.config);output=project_path(ROOT,args.output)
    require(not output.exists(),'Choose a new config filename; existing configs are never overwritten')
    before=read_json(source);validate_config(before)
    after=validate_config(sampling_config(before))
    require(pfsp(after.get('opponent_pool',{})),'Only the recognized quality-pool default can migrate automatically')
    require(before!=after,'Config already has these settings; use it directly')
    write_json(output,after)
    print(f'Wrote {output}; run, IL and checkpoint files unchanged. Switch at a committed generation divisible by 5 with --resume-reviewed-source.')


if __name__=='__main__':main()
