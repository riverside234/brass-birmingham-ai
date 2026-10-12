"""P5-only commands; every large evaluation is planned separately from execution."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json

from brass_ai.learning.contracts import read_json, project_path, require
from brass_ai.personality.deployment import roster
from brass_ai.recovery.retention import owned
from brass_ai.step5.league import write_json
from . import collection, evaluation, native, panels, ui
from .contracts import REPORT, Gates, read_sealed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/not_backed_up/yxu209/ai-training/'))
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--mono')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('build-host', help='Build the isolated original-Hard evaluation bridge')
    p = commands.add_parser('plan', help='Freeze all jobs/recipes; does not run games')
    p.add_argument('--roster', required=True); p.add_argument('--output', required=True)
    p.add_argument('--profiles'); p.add_argument('--gates'); p.add_argument('--setups', type=int, default=32)
    p.add_argument('--repetitions', type=int, default=2); p.add_argument('--seed', type=int, default=151000000)
    p.add_argument('--stage', choices=('calibration','release'), default='calibration'); p.add_argument('--calibration')
    p.add_argument('--hardware-label', default='lab-preview')
    p.add_argument('--tables', type=int, default=24)
    p.add_argument('--fixture', action='store_true', help='Unqualified fixture/preview only; permanently ineligible for certification')
    p = commands.add_parser('collect', help='Run one bounded immutable slice; failures remain in evidence')
    p.add_argument('--panel', required=True); p.add_argument('--output', required=True)
    p.add_argument('--start', type=int, default=0); p.add_argument('--games', type=int, default=16)
    p.add_argument('--tables', type=int, default=24)
    p.add_argument('--fixture-direct', action='store_true', help='Small fixture smoke only, never difficulty evidence')
    p = commands.add_parser('check')
    p.add_argument('--panel', required=True); p.add_argument('--chunks', nargs='+', required=True)
    p.add_argument('--output', required=True); p.add_argument('--review')
    p = commands.add_parser('menu', help='Export public selection DTO only, not a Unity deployment')
    p.add_argument('--roster', required=True); p.add_argument('--report'); p.add_argument('--output', required=True)
    p.add_argument('--fixture', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    if args.command == 'build-host': result = native.build(root, args.mono)
    elif args.command == 'plan':
        value = panels.plan(root, args.roster, args.output,
            profiles=read_json(project_path(root,args.profiles)) if args.profiles else None,
            gates=Gates(**read_json(project_path(root,args.gates))) if args.gates else None,
            setups=args.setups, repetitions=args.repetitions, seed=args.seed, stage=args.stage,
            fixture=args.fixture, calibration=args.calibration, hardware_label=args.hardware_label,tables=args.tables)
        result = dict(panel_id=value['id'], jobs=len(value['jobs']), fixture=value['fixture'],
                      profiles=value['profiles'], gates=value['gates'], ran_games=False)
    elif args.command == 'collect':
        import torch
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        value = collection.execute(root,args.panel,args.output,start=args.start,games=args.games,
            tables=args.tables,device=device,mono=args.mono,fixture_direct=args.fixture_direct)
        result = {k:value[k] for k in ('id','fixture','seconds','inference')}; result['games']=len(value['records'])
    elif args.command == 'check':
        value = evaluation.check(root,args.panel,args.chunks,args.output,review=args.review)
        result = {k:value[k] for k in ('status','games','expected_games','checks','p5_qualified','release_qualified','timing')}
    else:
        _, out = owned(root,args.output); require(not out.exists(), 'Use a new public-interface directory')
        value = roster.load(project_path(root,args.roster))
        report = evaluation.load_report(root,args.report) if args.report else None
        roster.validate_sources(root,value)
        if report: require(report['roster_id']==value['id'], 'Difficulty report belongs to another roster')
        result = ui.menu(value,report,fixture=args.fixture)
        out.mkdir(parents=True); write_json(out/'menu.json',result)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__': main()
