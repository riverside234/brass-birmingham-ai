"""Synchronous report-only panel lifecycle, with sealed checkpoints and retries."""
import copy
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from brass_ai.bank import read, sha
from brass_ai.learning.contracts import digest, project_path, require
from brass_ai.step5.league import write_json
from .ratings import write_once, receipt, append_games
from .context_evaluation_config import CONTRACT,enabled,validate
from .context_evaluation_spec import specification_for,validate_runtime,jobs_for
from .context_evaluation_collect import summary,save_summary,load_summaries,collect

MAX_ATTEMPTS=3


@contextmanager
def isolated_rng(model=None):
    """Error paths restore Python, NumPy, CPU/CUDA RNGs and an optional mode."""
    python=random.getstate();numpy=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mode=model.training if model is not None else None
    try:
        if model is not None:model.eval()
        yield
    finally:
        random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        if model is not None:model.train(mode)


def directory_for(run,generation):
    from .retention import safe
    return safe(run,run/'ratings'/'contexts'/f'generation-{generation:06d}')


def prepare(root,run,state,config):
    if not enabled(config):return None
    c=validate(config['context_evaluation'],config)
    if 'context_evaluation' not in state:
        activation=state['iteration'];interval=c['every_iterations'];main=config['rating']['every_iterations']
        state['context_evaluation']=dict(contract=CONTRACT,config=copy.deepcopy(c),activation_generation=activation,
            next_due_generation=((activation+interval+main-1)//main)*main,
            schedule_origin=activation,scheduled_interval=interval,pending=None,panels=[],
            timing=dict(normal_seconds=0.,added_seconds=0.,first_generation=activation+1,last_generation=activation,
                complete=True),latest_visualization=None)
        write_json(run/'state.json',state)
    saved=state['context_evaluation']
    require(saved['contract']==CONTRACT and type(saved['activation_generation']) is int and
        saved['activation_generation']<=state['iteration'] and type(saved['next_due_generation']) is int and
        saved['next_due_generation']==((saved['schedule_origin']+saved['scheduled_interval']+config['rating']['every_iterations']-1)//
            config['rating']['every_iterations'])*config['rating']['every_iterations'],
        'Invalid context evaluation schedule')
    generations=[p['generation'] for p in saved['panels']]
    require(generations==sorted(set(generations)) and all(saved['activation_generation']<g<=state['iteration'] for g in generations),
        'Invalid context evaluation history')
    if saved['pending'] is not None:
        require(saved['pending']['generation']==saved['next_due_generation'],'Pending context generation differs from schedule')
    # Reviewed cadence changes affect subsequent slots, never restart activation
    # or reinterpret an already sealed pending panel.
    if saved['config']!=c:
        saved['config']=copy.deepcopy(c);write_json(run/'state.json',state)
    return saved


def seal(root,run,state,config,parent,parent_path):
    saved=state.get('context_evaluation')
    if not enabled(config) or saved is None:return None
    if parent['generation']!=saved['next_due_generation']:return None
    if saved['pending'] is not None:
        require(saved['pending']['generation']==parent['generation'],'Only one pending context panel allowed')
        return saved['pending']
    spec=specification_for(root,parent,config);directory=directory_for(run,parent['generation'])
    path=directory/'spec.json';write_once(path,spec)
    diversity=next((r.get('diversity') for r in reversed(state.get('recent_training',[]))
        if r.get('iteration',-1)+1==parent['generation']),None)
    pending=dict(generation=parent['generation'],spec=receipt(root,path),parent_spec=receipt(root,parent_path),
        dependencies=[r['checkpoint'] for r in parent['agents'].values() if r['kind']=='model'],
        origin_diversity=copy.deepcopy(diversity),attempts=0,status='waiting_for_neutral')
    saved['pending']=pending;write_json(run/'state.json',state)
    write_json(directory/'status.json',dict(status='waiting_for_neutral',generation=parent['generation'],attempts=[],
        neutral_extraction_seconds=0.,additional_seconds=0.,timing_complete=True))
    return pending


def neutral_sink(root,run,state,parent):
    pending=state.get('context_evaluation',{}).get('pending')
    if pending is None or pending['generation']!=parent['generation']:return None
    path=project_path(root,pending['spec']['path']);require(sha(path)==pending['spec']['sha256'],'Context spec changed')
    spec=read(path);directory=path.parent;indices={i for s in spec['selection']['setups'] for i in s['indices']}
    require(digest(parent)==spec['parent_digest'],'Context neutral parent changed')
    status=read(directory/'status.json')
    if status.get('neutral_running'):status['timing_complete']=False
    status['neutral_running']=True;write_json(directory/'status.json',status)
    elapsed=0.;errors=[]
    def sink(game,job,record):
        nonlocal elapsed
        if job['index'] not in indices:return
        began=time.perf_counter()
        try:
            value=summary(game,job,spec,0,job['index'],receipt(root,Path(record['path'])))
            save_summary(directory,value)
        except Exception as error:
            # Diagnostic extraction must not discard valid main PL evidence.
            errors.append(str(error))
        finally:elapsed+=time.perf_counter()-began
    def finish():
        nonlocal elapsed
        status=read(directory/'status.json');status['neutral_extraction_seconds']+=elapsed
        status['additional_seconds']+=elapsed
        status['neutral_running']=False;elapsed=0.
        if errors:status['neutral_errors']=errors[:8]
        write_json(directory/'status.json',status)
    sink.finish=finish
    return sink


def _verified(root,reference):
    path=project_path(root,reference['path'])
    require(path.is_file() and sha(path)==reference['sha256'],'Missing/corrupt required context receipt: '+reference['path'])
    return path


def admit_neutral(root,state,pending,spec,directory):
    parent=_verified(root,pending['parent_spec']);require(digest(read(parent))==spec['parent_digest'],'Neutral spec mismatch')
    slot=next((s for s in state['ratings']['slots'] if s['generation']==spec['generation']),None)
    require(slot is not None and slot['spec']==pending['parent_spec'],'Matching committed main PL slot unavailable')
    records=[json.loads(line) for line in _verified(root,slot['records']).read_text(encoding='utf-8').splitlines()]
    require(all(r['spec_digest']==spec['parent_digest'] for r in records),'Main PL receipt spec differs')
    values=load_summaries(directory,spec,neutral_only=True)
    require(len(values)==spec['reused_games'],'Missing exact neutral summaries; replacement neutral games are forbidden')
    by={(r['setup_seed'],r['candidate_seat']):r for r in records if r['kind']=='rating'}
    for v in values:
        r=by[v['setup_seed'],v['seat']];s=v['seat']
        require(v['game_id']==r['game_id'] and v['setup_id']==r['setup_id'] and
            v['vp']==r['vp'][s] and v['win_share']==r['win_share'][s] and v['placement']==r['official_placement'][s],
            'Neutral summary differs from committed PL outcome')
    return values,dict(spec=pending['parent_spec'],records=slot['records'])


def _disk(directory):
    return sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())


def partial_support(spec,rows):
    by_context=Counter(r['context'] for r in rows);by_setup=Counter(str(r['setup_seed']) for r in rows)
    return dict(validated_games=len(rows),expected_games=spec['new_games']+spec['reused_games'],
        new_completed=sum(r['context_id']!=0 for r in rows),neutral_reused=sum(r['context_id']==0 for r in rows),
        per_context={c['name']:by_context[c['name']] for c in spec['contexts']},
        per_setup={str(s['seed']):by_setup[str(s['seed'])] for s in spec['selection']['setups']})


def _history(root,saved,current):
    history=[]
    for entry in saved['panels']:
        if entry['status']!='complete' or entry['generation']>=current['generation']:continue
        r=read(_verified(root,entry['report']))
        cost=read(_verified(root,entry['final_cost'])) if entry.get('final_cost') else r['cost']
        history.append(dict(generation=r['generation'],protocol_id=r['protocol_id'],report=entry['report'],
            cost=entry.get('final_cost'),mixture_vp=r['contexts'][-1]['paired']['vp'],incremental_wall_fraction=cost['incremental_wall_fraction'],
            additional_seconds=cost['additional_seconds']))
    return (history+[dict(generation=current['generation'],protocol_id=current['protocol_id'],
        mixture_vp=current['contexts'][-1]['paired']['vp'],incremental_wall_fraction=current['cost']['incremental_wall_fraction'],
        additional_seconds=current['cost']['additional_seconds'])])[-12:]


def render_panel(root,run,state,entry):
    """Retry only the view, from frozen report/history; no games or resampling."""
    if entry.get('render_failures',0)>=MAX_ATTEMPTS:return 0.
    directory=directory_for(run,entry['generation']);began=time.perf_counter()
    try:
        from .context_evaluation_plot import render,VERSION
        report=read(_verified(root,entry['report']))
        if entry.get('final_cost'):
            cost=read(_verified(root,entry['final_cost']))
            report['cost'].update(cost)
            for h in report['history']:
                if h['generation']==report['generation']:
                    h.update(incremental_wall_fraction=cost['incremental_wall_fraction'],additional_seconds=cost['additional_seconds'])
        # Preserve the earlier immutable SVG while final timing becomes available.
        # A failed final redraw must not invalidate the previous alias receipt.
        suffix='-final' if entry.get('final_cost') else ''
        svg_path=directory/f'progress{suffix}.svg'
        visual=render(report,svg_path)
        visual.update(status='complete',generation=entry['generation'],report=entry['report'],cost=entry.get('final_cost'),
            svg=receipt(root,svg_path))
        visual_path=directory/f'visualization{suffix}.json';write_json(visual_path,visual)
        entry['visualization']=receipt(root,visual_path)
        entry.update(render_pending=False,render_failures=0)
        alias=run/'ratings'/'contexts'/'progress.svg';temp=alias.with_suffix('.tmp')
        temp.write_bytes(svg_path.read_bytes());temp.replace(alias)
        state['context_evaluation']['latest_visualization']=dict(generation=entry['generation'],svg=receipt(root,alias),
            source=visual['svg'],report=entry['report'],renderer=VERSION)
    except Exception as error:
        path=directory/'visualization-failed.json'
        write_json(path,dict(status='failed',generation=entry['generation'],report=entry['report'],reason=str(error)))
        entry['visualization_failure']=receipt(root,path)
        entry.setdefault('visualization',entry['visualization_failure'])
        entry.update(render_pending=True,render_failures=entry.get('render_failures',0)+1)
        print('Context visualization pending: '+str(error),flush=True)
    finally:
        elapsed=time.perf_counter()-began
        state['context_evaluation']['timing']['added_seconds']+=elapsed
        entry['render_seconds']=entry.get('render_seconds',0.)+elapsed
        write_json(run/'state.json',state)
        _write_index(run,state['context_evaluation'])
    return elapsed


def prune(root,run,state):
    """Only owned raw-panel replays, after complete report/index publication."""
    from .retention import safe
    saved=state['context_evaluation'];complete=[p for p in saved['panels'] if p['status']=='complete']
    expired=complete[:-saved['config']['raw_replay_panels']]+[p for p in saved['panels'] if p['status']=='failed'];count=0;size=0
    for entry in expired:
        directory=directory_for(run,entry['generation'])
        _verified(root,entry['report'] if entry['status']=='complete' else entry['status_receipt'])
        raw=safe(run,directory/'raw')
        for path in raw.glob('*.jsonl.gz'):
            path=safe(run,path);require(path.parent==raw,'Context retention escaped owned raw directory')
            size+=path.stat().st_size;path.unlink();count+=1
    return dict(files=count,bytes=size)


def _advance(root,run,state,entry):
    saved=state['context_evaluation']
    if not any(p['generation']==entry['generation'] for p in saved['panels']):saved['panels'].append(entry)
    saved['pending']=None
    saved['schedule_origin']=entry['generation'];saved['scheduled_interval']=saved['config']['every_iterations']
    saved['next_due_generation']=entry['generation']+saved['config']['every_iterations']
    write_json(run/'state.json',state)
    _write_index(run,saved)


def _write_index(run,saved):
    write_json(run/'ratings'/'contexts'/'index.json',dict(contract=CONTRACT,
        activation_generation=saved['activation_generation'],next_due_generation=saved['next_due_generation'],panels=saved['panels'],
        latest_visualization=saved['latest_visualization']))


def run_pending(root,run,state,config,*,device='cpu',mono=None,inference_devices=None):
    saved=state.get('context_evaluation')
    if not enabled(config) or saved is None:return None
    pending=saved['pending']
    if pending is None:
        # A renderer retry cannot trigger a collection or a new PL fit.
        complete=[p for p in saved['panels'] if p['status']=='complete']
        if complete:
            last=complete[-1];v=last.get('visualization')
            if last.get('render_pending') or v is None or read(_verified(root,v)).get('status')!='complete':
                render_panel(root,run,state,last)
        return None
    directory=directory_for(run,pending['generation']);status=read(directory/'status.json')
    # A hard kill loses an unknown tail. Do not pretend the overhead is complete.
    if status.get('status')=='running':status['timing_complete']=False
    began=time.perf_counter();before=copy.deepcopy({k:v for k,v in state.items() if k!='context_evaluation'})
    status['status']='running';pending['attempts']+=1;pending['status']='running'
    write_json(directory/'status.json',status);write_json(run/'state.json',state)
    result=None;failure=None;interrupted=False;metrics={};spec=None;rows=None
    try:
        with isolated_rng():
            spec=read(_verified(root,pending['spec']));validate_runtime(root,spec)
            neutral,parent_receipts=admit_neutral(root,state,pending,spec,directory)
            require(not status.get('neutral_errors'),'Neutral summary admission previously failed')
            # If report publication finished but the state commit did not, reuse
            # its receipts. Never rerun games to retry a report or a plot.
            if (directory/'report.json').exists():
                result=read(directory/'report.json')
                require(result['panel_digest']==digest(spec),'Uncommitted context report differs')
                for key in ('spec','summaries','games'):_verified(root,result['evidence'][key])
            else:
                metrics=collect(root,directory,spec,device=device,mono=mono,inference_devices=inference_devices)
                rows=load_summaries(directory,spec)
                from .context_evaluation_analysis import analyze,markdown
                started=time.perf_counter();result=analyze(spec,rows);metrics['analysis_seconds']=time.perf_counter()-started
                append_games(directory/'summaries.jsonl',rows)
                append_games(directory/'games.jsonl',[{k:r[k] for k in ('game_id','context','context_id','setup_seed','seat','vp',
                    'placement','win_share','task_return','insolvency_events','insolvent','zero_vp','parent_digest','parent_index')} for r in rows])
                result['evidence']=dict(spec=pending['spec'],parent=parent_receipts,summaries=receipt(root,directory/'summaries.jsonl'),
                    games=receipt(root,directory/'games.jsonl'))
                result['origin_diversity']=pending['origin_diversity']
                for r in result.pop('representatives'):
                    write_once(directory/'representatives'/(r['context']+'.json'),r)
                result['representatives']=[receipt(root,p) for p in sorted((directory/'representatives').glob('*.json'))]
                elapsed=status['additional_seconds']+time.perf_counter()-began
                timing=saved['timing'];normal=timing['normal_seconds']
                # Current generation's normal work is not finalized yet. Publish
                # the same-interval denominator in cost.json, never a proxy here.
                fraction=None
                result['cost']=dict(**metrics,additional_seconds=elapsed,incremental_wall_fraction=fraction,
                    review_fraction=.05,review_only=True,completed_live_panels=1+sum(p['status']=='complete' for p in saved['panels']),
                    timing_complete=timing['complete'] and status['timing_complete'],normal_trainer_seconds=normal,
                    normal_interval=[timing['first_generation'],timing['last_generation']],
                    scope='Cumulative since activation; current iteration normal time and rendering/cleanup finalize in cost.json',
                    attempted_game_upper_bound=pending['attempts']*spec['new_games'],unique_new_games=spec['new_games'],
                    reused_neutral_games=spec['reused_games'],failed_attempts=len([a for a in status['attempts'] if a['status']!='complete']),
                    disk_bytes=_disk(directory))
                result['history']=_history(root,saved,result)
                write_once(directory/'report.json',result)
                (directory/'report.md').write_text(markdown(result),encoding='utf-8')
    except KeyboardInterrupt:
        failure='Interrupted by user; exact pending jobs retained';interrupted=True
    except Exception as error:failure=str(error)
    finally:
        if spec is not None:
            try:
                status['support']=partial_support(spec,rows if rows is not None else load_summaries(directory,spec))
            except Exception as error:
                status['support']=dict(validated_games=None,reason='Support admission failed: '+str(error))
        elapsed=time.perf_counter()-began;status['additional_seconds']+=elapsed
        status['attempts'].append(dict(attempt=pending['attempts'],seconds=elapsed,status='interrupted' if interrupted else 'failed' if failure else 'complete',
            reason=failure,metrics=metrics))
        # Neutral extraction is accounted once, along with all retry attempts.
        saved['timing']['added_seconds']+=elapsed+(status['neutral_extraction_seconds'] if not status.get('neutral_accounted') else 0.)
        status['neutral_accounted']=True
        saved['timing']['complete'] &= status['timing_complete']
        status['status']='interrupted' if interrupted else 'retryable' if failure else 'complete'
        pending['status']=status['status'];write_json(directory/'status.json',status)
        require(before=={k:v for k,v in state.items() if k!='context_evaluation'},'Context panel modified training state')
        write_json(run/'state.json',state)
    if interrupted:raise KeyboardInterrupt
    if failure:
        print(f'Context panel g{pending["generation"]} incomplete: {failure}',flush=True)
        # Missing source/neutral/runtime receipts cannot be repaired by recollecting.
        permanent=any(w in failure.lower() for w in ('neutral','receipt','changed','mismatch','differs','missing','corrupt'))
        if permanent or pending['attempts']>=MAX_ATTEMPTS:
            status['status']='failed';write_json(directory/'status.json',status)
            _advance(root,run,state,dict(generation=pending['generation'],status='failed',spec=pending['spec'],
                status_receipt=receipt(root,directory/'status.json'),reason=failure))
            cleanup_at=time.perf_counter();prune(root,run,state)
            saved['timing']['added_seconds']+=time.perf_counter()-cleanup_at
            write_json(run/'state.json',state)
        return dict(generation=pending['generation'],status=status['status'],reason=failure)
    entry=dict(generation=pending['generation'],status='complete',spec=pending['spec'],report=receipt(root,directory/'report.json'),
        summaries=receipt(root,directory/'summaries.jsonl'),games=receipt(root,directory/'games.jsonl'),
        status_receipt=receipt(root,directory/'status.json'))
    _advance(root,run,state,entry)
    render_panel(root,run,state,entry)
    cleanup_at=time.perf_counter();cleanup=prune(root,run,state);cleanup_seconds=time.perf_counter()-cleanup_at
    saved['timing']['added_seconds']+=cleanup_seconds
    entry.update(cleanup=cleanup,cleanup_seconds=cleanup_seconds,additional_seconds=status['additional_seconds']+entry.get('render_seconds',0.)+cleanup_seconds)
    _advance(root,run,state,entry)
    print(f'Context panel g{entry["generation"]}: {result["support"]["new_games"]} new + {result["support"]["reused_games"]} reused; '
        f'{entry["additional_seconds"]:.1f}s; report-only, no additional PL fit',flush=True)
    return dict(generation=entry['generation'],status='complete',report=entry['report'],additional_seconds=entry['additional_seconds'])


def record_iteration_time(root,run,state,report):
    saved=state.get('context_evaluation')
    if saved is None:return
    timing=saved['timing'];g=report['iteration']+1
    if g<=timing['last_generation']:return
    if g!=timing['last_generation']+1:timing['complete']=False
    added=report.get('timings',{}).get('context_evaluation_seconds',0.)
    timing['normal_seconds']+=max(0.,report['timings']['iteration_seconds']-added)
    timing['last_generation']=g
    panels=[p for p in saved['panels'] if p['generation']==g and p['status']=='complete']
    if panels:
        entry=panels[-1];directory=directory_for(run,g)
        fraction=timing['added_seconds']/timing['normal_seconds'] if timing['complete'] and timing['normal_seconds']>0 else None
        cost=dict(generation=g,additional_seconds=entry['additional_seconds'],cumulative_added_seconds=timing['added_seconds'],
            normal_trainer_seconds=timing['normal_seconds'],incremental_wall_fraction=fraction,
            interval=[timing['first_generation'],g],timing_complete=timing['complete'],review_fraction=.05,
            review_due=sum(p['status']=='complete' for p in saved['panels'])>=2,
            advisory_exceeded=fraction>.05 if fraction is not None else None,
            disk_bytes=_disk(directory),scope='Includes rendering, extraction, retries and owned cleanup; no automatic cadence change')
        write_once(directory/'cost.json',cost);entry['final_cost']=receipt(root,directory/'cost.json')
        # A cost companion avoids an impossible self-referential render timer.
        # Redraw reads these committed values; its own time is separately logged
        # in visualization.json and cumulative accounting for the next interval.
        render_panel(root,run,state,entry)
        write_json(run/'ratings'/'contexts'/'index.json',dict(contract=CONTRACT,
            activation_generation=saved['activation_generation'],next_due_generation=saved['next_due_generation'],panels=saved['panels'],
            latest_visualization=saved['latest_visualization']))
    write_json(run/'state.json',state)
