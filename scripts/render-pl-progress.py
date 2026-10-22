"""Restyle an existing PL report as progress.svg, without refitting or training.

Example:
    python scripts/render-pl-progress.py --report results/.../ratings/slots/generation-000210/report.json
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from brass_ai.recovery.rating_plot import render, DEFAULT_MAX_POINTS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='Defaults to progress.svg beside the report')
    parser.add_argument('--preview',type=Path,help='Optional PNG for visual inspection')
    parser.add_argument('--max-points',type=int,default=DEFAULT_MAX_POINTS,
        help='Full-history display cap (3–30, default 30); average adjacent history above this cap, keep first/latest exact')
    args=parser.parse_args()
    output=args.output or args.report.with_name('progress.svg')
    if output.suffix.lower()!='.svg':raise ValueError('Output must be an SVG file')
    render(json.loads(args.report.read_text(encoding='utf-8-sig')),output,preview=args.preview,max_points=args.max_points)
    print(output)


if __name__=='__main__':main()
