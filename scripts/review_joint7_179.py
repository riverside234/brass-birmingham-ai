"""Audit the generation-179 export without altering training or source receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from review_joint7_150 import get, normalize, read, reward_moments


EXPORT = 'results/v4-selfplay-joint-g0179-20261014T200020968013Z'
WINDOWS = ((140,149),(150,159),(160,169),(170,178),(160,178),(177,178))


def mean_present(rows, field):
    values = [get(row,field) for row in rows if get(row,field) is not None]
    return dict(observed=len(values),mean=statistics.mean(values) if values else None)


def summarize(rows, lo, hi):
    selected = [rows[i] for i in range(lo,hi+1)]
    debt = [r['actor']['insolvency'] for r in selected]
    n = sum(d['learner_trajectories'] for d in debt)
    events = sum(d['events'] for d in debt)
    affected = sum(d['affected_trajectories'] for d in debt)
    reward = reward_moments(selected)
    composition = {}
    for kind in ('selfplay','historical','legacy'):
        values = [r['by_composition'][kind][r['incoming_policy_id']] for r in selected]
        count = sum(v['seat_games'] for v in values)
        composition[kind] = dict(learner_games=count,
            **{k:sum(v[k]*v['seat_games'] for v in values)/count for k in ('win_share','placement','vp')})
    phases = {}
    for phase in ('canal_early','canal_late','rail_early','rail_late'):
        values = [r['search'].get('phases',{}).get(phase,{}) for r in selected]
        phases[phase] = {}
        for key in ('scheduled_turns','searched_turns','actual_searches','completed_searches',
                    'qualified_samples','insufficient_coverage','timeouts','attempted_simulations',
                    'completed_rollouts','elapsed_seconds','turn_elapsed_seconds','allocated_seconds',
                    'legal_root_actions','visited_root_actions','supported_plans'):
            observed = [v[key] for v in values if key in v]
            phases[phase][key] = dict(observed=len(observed),total=sum(observed) if observed else None)
    usage = {}
    for key in ('teacher_rows','distinct_labels_used','unused_qualified_labels','sample_presentations','optimizer_steps'):
        observed = [r['search'][key] for r in selected if key in r['search']]
        usage[key] = dict(observed=len(observed),total=sum(observed) if observed else None)
    fields = ('actor.categorical_probe.categorical_kl',
              'search.teaching_only_categorical_probe.categorical_kl','search.post_teacher_sampled_kl',
              'timings.collection_seconds','timings.search_and_distill_seconds','timings.iteration_seconds')
    economy = {key:sum(r['economy']['actions'].get(key,0) for r in selected)/n
               for key in ('loan','pass','liquidate','build','network','sell')}
    return dict(iterations=[lo,hi],learner_games=n,insolvency_events=events,
        events_per_learner_game=events/n,affected_games=affected,affected_fraction=affected/n,
        capped_games=reward['cap_count'],capped_fraction=reward['cap_count']/n,
        mean_penalty=-reward['weighted_means']['cost'],composition=composition,
        economy_actions_per_learner_game=economy,reward=reward,teacher_usage=usage,
        phases=phases,means={field:mean_present(selected,field) for field in fields})


def audit(root):
    folder = root/EXPORT
    state, manifest, retained = [read(folder/name) for name in
        ('state.json','export-manifest.json','retention-history.json')]
    checks = []
    for entry in manifest['files']:
        path = folder/entry['path']
        assert path.resolve().is_relative_to(folder.resolve()), entry['path']
        raw = path.read_bytes(); normalized = raw.replace(b'\r\n',b'\n')
        raw_sha, normalized_sha = [hashlib.sha256(v).hexdigest() for v in (raw,normalized)]
        assert entry['sha256'] in (raw_sha,normalized_sha), entry['path']
        checks.append(dict(path=entry['path'],expected_sha256=entry['sha256'],
            match='raw' if entry['sha256']==raw_sha else 'LF-normalized'))
    assert read(folder/'recorded-config.json')==state['signature']['config']
    rows = {v['summary']['iteration']:normalize(v['summary']) for k,v in retained.items()
            if k.startswith('iteration-')}
    full = {}
    for path in folder.glob('iteration-*/report.json'):
        report = read(path); full[report['iteration']] = report
        rows[report['iteration']] = normalize(report)
    assert state['iteration']==179 and state['games_done']==57280
    for i in range(140,179):
        assert rows[i]['games_done']==320*(i+1)
        assert rows[i]['incoming_model_id']==state['history'][i]['model_id']
        assert rows[i]['post_update_model_id']==state['history'][i+1]['model_id']
        who=rows[i]['incoming_policy_id']
        assert {k:v[who]['seat_games'] for k,v in rows[i]['by_composition'].items()}==dict(selfplay=224,historical=80,legacy=16)
        assert rows[i]['selfplay_incumbent']['generation']==(0 if i<160 else 160)
    gate = read(folder/'evaluations/champion-00051200/report.json')
    assert gate['result']['accepted'] and gate['games_done']==51200
    assert gate['spec']['candidate']['model_id']==state['champion']['model_id']
    assert len(gate['outcomes'])==432 and len({r['seed'] for r in gate['outcomes']})==72
    assert state['champion']['generation']==160
    assert gate['spec_id']==hashlib.sha256(json.dumps(gate['spec'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    evaluations = {}
    for path in sorted((folder/'evaluations').glob('*/report.json')):
        if path.parent.name.startswith('champion-'):continue
        report=read(path); spec=read(path.with_name('spec.json'))
        evaluations[path.parent.name]=dict(spec=spec,
            panels={k:{f:v.get(f) for f in ('games','setups','win_share','mean_rank','mean_vp',
                         'win_share_setup_bootstrap_95')} for k,v in report.get('results',{}).items()})
    latest = {}
    for i,report in sorted(full.items()):
        who=report['incoming_policy_id']
        latest[str(i)]=dict(insolvency=report['actor']['insolvency'],
            teacher_phases=report['search']['teaching']['phases'],
            teacher_only_drift=report['search'].get('teaching_only_categorical_probe'),
            teacher_collections=[{k:b.get(k) for k in ('teacher_rows','distinct_labels_used','unused_qualified_labels',
                'optimizer_steps','sample_presentations','teacher_policy_drift')} for b in report['search']['collections']],
            compositions={k:dict(learner=v['matches']['policies'][who],
                other_policies={p:m for p,m in v['matches']['policies'].items() if p!=who} if k=='selfplay' else None,
                economy=v['economy']) for k,v in report['by_composition'].items()})
    series = []
    for i in range(140,179):
        r=rows[i]; d=r['actor']['insolvency']; who=r['incoming_policy_id']
        series.append(dict(iteration=i,generation_incoming=i,generation_saved=i+1,
            incumbent_generation=r['selfplay_incumbent']['generation'],
            insolvency_events=d['events'],affected_games=d['affected_trajectories'],
            learner_games=d['learner_trajectories'],mean_cost=d['mean_cost_per_trajectory'],
            capped_games=r['actor']['reward_distribution']['capped_trajectories'],
            composition={k:v[who] for k,v in r['by_composition'].items()},
            teacher_labels=r['search']['teacher_rows'],distinct_labels_used=r['search'].get('distinct_labels_used'),
            ppo_kl=get(r,'actor.categorical_probe.categorical_kl'),
            teaching_only_kl=get(r,'search.teaching_only_categorical_probe.categorical_kl')))
    return dict(format='joint7-generation179-audit/1.0.0',input=EXPORT,
        manifest_checks=checks,manifest_missing=manifest['missing'],manifest_warnings=manifest['warnings'],
        source=state['signature']['implementation'],recorded_config=state['signature']['config'],
        state=dict(generation=179,games_done=57280,champion_generation=160,next_gate_games=57600),
        verified_continuity=dict(iterations=[140,178],model_and_progress=True,composition=True,incumbent_boundary=True),
        source_transitions=[dict(iteration=t['iteration'],from_implementation=t['from_signature']['implementation'],
            config_changes=t.get('config_changes',{})) for t in state.get('source_transitions',[])],
        champion_result=gate['result'],windows=[summarize(rows,*w) for w in WINDOWS],
        iterations=series,evaluations=evaluations,latest=latest,
        limitations=[
            'Before/after comparisons use different setup seeds and an adapting learner; they do not establish causality.',
            'Historical opponent pools evolve; historical/legacy results are descriptive controls, not randomized controls.',
            'Only iterations 177 and 178 have full reports in this export. No raw game replays or model checkpoints are included.',
            'Retained summaries lack insolvency counts by opponent composition and the complete event-count distribution.',
            'Economic flip counters are observed flips, not exact end-of-era utilization or causal traces.',
            'No frozen generation-179 strength evaluation is included. Champion selection is relative to generation 0 in a two-versus-two lineup.',
            'Missing historical reports are declared by the export; all 37 included files match after LF normalization.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('results/reviews/joint7-g0179/metrics.json'))
    args=parser.parse_args(); data=audit(args.root)
    output=args.root/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes((json.dumps(data,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    print(output)
    print('Verified included-file checksums, recorded configuration, model lineage and the promotion boundary.')
    for w in data['windows'][:4]:
        print(f"Iterations {w['iterations']}: events/game={w['events_per_learner_game']:.3f}, "
              f"affected={100*w['affected_fraction']:.2f}%, capped={100*w['capped_fraction']:.2f}%, "
              f"self-play VP={w['composition']['selfplay']['vp']:.2f}")


if __name__=='__main__':main()
