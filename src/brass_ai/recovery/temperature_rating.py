"""Paired temperature comparison, separate from the main PL training evidence."""
import copy
import json

from brass_ai.bank import read
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.step5.league import write_json
from . import ratings
from .plackett_luce import estimate


def specification(base):
    spec=copy.deepcopy(base);original=base['candidate'];candidate=original+'-temperature-0.7'
    entry=copy.deepcopy(spec['agents'].pop(original));entry['temperature']=.7
    spec['agents'][candidate]=entry;spec['candidate']=candidate
    for row in spec['lineups']:
        row['agents']=[candidate if a==original else a for a in row['agents']]
    spec['temperature_comparison']=dict(reference=original,temperature=.7)
    return spec


def schedule(root,run,state,config,*,device,mono=None,inference_devices=None):
    if 'entropy_control' not in config or not config['rating']['enabled']:return None
    interval=config['entropy_control']['deployment_temperature_check']['every_iterations']
    origin=state.get('exploration',{}).get('origin',state['iteration'])
    completed={r['generation'] for r in state.get('temperature_ratings',[])}
    result=None
    for slot in state['ratings']['slots']:
        generation=slot['generation']
        if generation<=origin or generation%interval or generation in completed:continue
        base=read(project_path(root,slot['spec']['path']));spec=specification(base)
        directory=run/'ratings'/'temperature'/f'generation-{generation:06d}'
        ratings.write_once(directory/'spec.json',spec)
        path=directory/'games.jsonl';manifest_path=directory/'games-manifest.json'
        if path.exists() and manifest_path.exists():
            require(read(manifest_path)==dict(spec_digest=digest(spec),records=ratings.receipt(root,path)),
                'Temperature comparison replay changed')
            rows=[json.loads(r) for r in path.read_text(encoding='utf-8').splitlines()]
        else:
            rows=ratings.collect_slot(root,directory,spec,config,device,mono,inference_devices)
            ratings.append_games(path,rows)
            ratings.write_once(manifest_path,dict(spec_digest=digest(spec),records=ratings.receipt(root,path)))
        require(len(rows)==len(spec['lineups']) and all(r['spec_digest']==digest(spec) for r in rows),
            'Incomplete temperature comparison')
        agents=dict(state['ratings']['agents'],**{spec['candidate']:spec['agents'][spec['candidate']]})
        ordinary=ratings.read_games(root,state['ratings'])
        fit,samples=estimate([r for r in ordinary+rows if r['kind']=='rating'],agents,config['rating'])
        a=base['candidate'];b=spec['candidate']
        # Paired setup bootstrap uses exactly the main estimator's setup clusters.
        names=list(fit['ratings']);scale=config['rating']['elo_scale']
        differences=(samples[:,names.index(b)]-samples[:,names.index(a)])*scale
        import numpy as np
        ci=np.quantile(differences,[.025,.975]).tolist()
        result=dict(contract='deployment-temperature-comparison/1.0.0',generation=generation,
            temperatures=[1.,.7],reference=fit['ratings'][a],alternative=fit['ratings'][b],
            difference_points=fit['ratings'][b]['points']-fit['ratings'][a]['points'],
            difference_points_ci95=ci,
            evidence_favors=(.7 if ci[0]>0 else 1. if ci[1]<0 else None),
            deployment_qualified=None,reason='Temperature evidence only; deployment still requires panel review',
            panels=ratings.panel_report(rows,b,None,config['rating']['seed']),
            spec=ratings.receipt(root,directory/'spec.json'),records=ratings.receipt(root,path),
            scope='Same checkpoint, setups, seat rotations and opponents; epsilon zero, no openings or branches. Separate fit; main rating history and GAE slope unchanged.')
        ratings.write_once(directory/'report.json',result)
        state.setdefault('temperature_ratings',[]).append(dict(generation=generation,
            report=ratings.receipt(root,directory/'report.json')))
        write_json(run/'state.json',state)
    return result
