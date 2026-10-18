"""Refit committed held-out ratings into a separate review directory; no games or training."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery import ratings
    from brass_ai.recovery.plackett_luce import estimate
    from brass_ai.step5.league import write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='artifacts/step5/joint7-v11-1-grouped-lr')
    parser.add_argument('--output',required=True,help='New directory under artifacts/; leaves the run unchanged')
    args=parser.parse_args();run=project_path(ROOT,args.run);output=project_path(ROOT,args.output)
    require(output.is_relative_to(ROOT/'artifacts') and not output.is_relative_to(run) and not output.exists(),
            'Choose a new artifacts/ directory outside the training run')
    path=run/'state.json';before=sha(path);state=read(path)
    require('ratings' in state,'This run has no Plackett-Luce rating state')
    saved=ratings.validate_state(ROOT,state['ratings'],state['signature']['config']['rating'])
    require(saved['slots'],'No completed rating slot yet; the default first slot is generation 10')
    games=[g for g in ratings.read_games(ROOT,saved) if g['kind']=='rating']
    report,_=estimate(games,saved['agents'],saved['config'])
    latest=saved['slots'][-1];panels=read(project_path(ROOT,latest['panels']['path']))
    previous=ratings.panel_history(ROOT,saved)[:-1]
    report.update(generation=latest['generation'],alerts=ratings.alerts(report,saved,panels,previous),
        panels=latest['panels'],source_state=dict(path=path.relative_to(ROOT).as_posix(),sha256=before))
    require(sha(path)==before,'Run advanced during refit; retry after the iteration finishes')
    output.mkdir(parents=True);write_json(output/'report.json',report)
    ratings.plot_report(output/'progress.svg',report,saved['agents'])
    print(f'Refitted {len(games)} held-out games; report and curve: {output}')
    print('Deployment qualification is pending separate panel review. No checkpoint was published.')


if __name__=='__main__':main()
