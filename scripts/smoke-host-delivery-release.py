"""One short isolated turn for the reduced-reserve release candidate."""
from pathlib import Path
import argparse
import runpy

ROOT=Path(__file__).resolve().parents[1]
S=runpy.run_path(str(ROOT/'scripts/smoke-host-delivery.py'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('case',choices=('canal-k32','rail-k32','rail-branching-k32','canal-k16'))
    case=parser.parse_args().case
    fixture,k,_=S['CASES'][case] if case!='rail-branching-k32' else ('live-92781-RailEra-2.json',32,())
    S['invoke'](S['STUDY']/'release-1.4.44','reserve08-'+case,fixture,k)
