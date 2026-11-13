"""Render progress.svg, progress_vp.svg, and progress_teacher_vp.svg when teacher VP is available.

Example:
    python scripts/render-pl-progress.py --report results/.../ratings/slots/generation-000210/report.json
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from brass_ai.recovery.rating_plot import render, render_progress, DEFAULT_MAX_POINTS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='Defaults to progress.svg beside the report')
    parser.add_argument('--preview',type=Path,help='Optional PNG for visual inspection')
    parser.add_argument('--vp-preview',type=Path,help='Optional VP PNG for visual inspection')
    parser.add_argument('--run',type=Path,
        help='Run or complete export bundle containing state.json; otherwise inferred from report ancestors')
    parser.add_argument('--history-source',type=Path,action='append',default=[],
        help='Optional older complete bundle from the same run to fill retained-history gaps (repeatable)')
    parser.add_argument('--max-points',type=int,default=DEFAULT_MAX_POINTS,
        help='Full-history display cap (3–30, default 30); average adjacent history above this cap, keep first/latest exact')
    args=parser.parse_args()
    output=args.output or args.report.with_name('progress.svg')
    if output.suffix.lower()!='.svg':raise ValueError('Output must be an SVG file')
    run=args.run or next((p for p in args.report.resolve().parents if (p/'state.json').is_file()),None)
    if args.history_source and run is None:raise ValueError('--history-source requires a run or complete export bundle')
    vp=render_progress(json.loads(args.report.read_text(encoding='utf-8-sig')),output,run=run,
        history_sources=args.history_source,preview=args.preview,vp_preview=args.vp_preview,max_points=args.max_points)
    print(output)
    print(vp)
    teacher_vp=output.with_name('progress_teacher_vp.svg')
    if teacher_vp.is_file():print(teacher_vp)


if __name__=='__main__':main()
