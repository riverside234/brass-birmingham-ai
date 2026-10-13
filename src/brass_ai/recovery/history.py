"""Bounded diagnostic summaries, never replay/restore evidence or run state."""
from pathlib import Path
import re

from brass_ai.bank import read
from brass_ai.step5.league import write_json


def scalars(value,keys):
    if not isinstance(value,dict):return {}
    return {k:value[k] for k in keys if k in value and (value[k] is None or type(value[k]) in (int,float,bool,str))}


def calibration(value):
    if not isinstance(value,dict):return None
    out=scalars(value,('required','ready','version','reward_id','decisions','games','steps','reason','scope'))
    for key in ('before','after'):
        if isinstance(value.get(key),dict):out[key]=scalars(value[key],('mse','mae'))
    if isinstance(value.get('receipt'),dict):out['receipt']=calibration(value['receipt'])
    return out


def evaluation_evidence(report):
    """Keep comparable setup/seat evidence when full reports are retired."""
    spec=report.get('spec',{})
    out=dict(spec={k:spec[k] for k in ('format','implementation','config','seeds','lineups','entries','candidate','incumbent',
        'legacy_pool','teachers','held') if k in spec})
    def panel(row):
        result=scalars(row,('games','setups','win_share','mean_rank','mean_vp','accepted','candidate_win_share',
            'candidate_vp','incumbent_vp','candidate_rank','incumbent_rank','threshold','scope',
            'stage','protocol','planned_setups','min_ci_lower'))
        for key in ('win_share_setup_bootstrap_95','setup_bootstrap_95','seat_counts','screen','confirmation'):
            if key in row:result[key]=row[key]
        outcomes=row.get('outcomes',[])
        result.update(outcomes=outcomes[:1024],outcomes_total=len(outcomes),outcomes_truncated=len(outcomes)>1024)
        return result
    if 'outcomes' in report:
        out['champion']=panel(dict(report.get('result',{}),outcomes=report['outcomes']))
    out['panels']={k:panel(v) for k,v in list(report.get('results',{}).items())[:4]}
    return out


def compact(report):
    # Old compact summaries cannot recover fields that were already discarded.
    # Preserve their version and missingness instead of relabelling as v4.
    if report.get('summary_version') in (2,3,4):return report
    out=scalars(report,('format','iteration','games_done','cumulative_games','incoming_model_id','post_update_model_id',
                       'incoming_policy_id','elapsed_seconds','games','failed_games','promotion_eligible'))
    out['summary_version']=4
    if report.get('format') in ('step5-evaluation-report/1.0.0','four-player-champion/1.0.0'):
        out['evaluation_evidence']=evaluation_evidence(report)
    out['timings']=scalars(report.get('timings',{}),('collection_seconds','replay_load_seconds','ppo_seconds',
        'search_and_distill_seconds','checkpoint_seconds','evaluation_seconds','cleanup_seconds','iteration_seconds'))
    actor=report.get('actor',{})
    out['actor']=scalars(actor,('optimizer_steps','learner_decisions','meaningful_decisions','forced_decisions',
        'learner_trajectories','kl_early_stop','rehearsal_weight','gradient_clip_steps','baseline_explained_variance',
        'baseline_scope','kl_estimator','kl_probe_rows','kl_probe_sampling'))
    out['actor']['learning_rates']=actor.get('learning_rates',[])[:8]
    for key in ('advantage','advantage_distribution','return_distribution','legal_actions','categorical_probe','gradient_norms','phases'):
        out['actor'][key]=scalars(actor.get(key,{}),actor.get(key,{}))
    epochs=actor.get('epochs',[])
    if epochs:
        out['actor']['epochs']=[dict(scalars(epoch,('epoch','actor','value','win_loss','rank_loss','vp_loss',
            'entropy','post_update_kl','probe_kl','clip_fraction','gradient_norm','actor_positions','critic_positions')),
            utility_prediction=scalars(epoch.get('utility_prediction',{}),('mse','mae'))) for epoch in epochs[-8:]]
        out['actor']['last_epoch']=out['actor']['epochs'][-1]
        out['actor']['utility_prediction']=scalars(epochs[-1].get('utility_prediction',{}),('mse','mae'))
    out['actor']['insolvency']=scalars(actor.get('insolvency',{}),('events','affected_trajectories','learner_trajectories','mean_cost_per_trajectory'))
    reward=actor.get('reward',{})
    out['actor']['reward_components']=scalars(reward.get('components',{}),('win','placement','vp','cost','utility'))
    out['actor']['reward_spec']=reward.get('spec')
    out['actor']['reward_distribution']=reward.get('distribution')
    out['actor']['utility_calibration']=calibration(actor.get('utility_calibration'))
    search=report.get('search',{})
    out['search']=scalars(search,('updated','reason','collection_count','optimizer_steps','teacher_rows','teacher_setups',
        'sample_presentations','max_observed_reuse','distinct_labels_used','unused_qualified_labels',
        'post_teacher_sampled_kl','post_teacher_kl_scope'))
    for key in ('fit_before','fit_after','budget'):
        out['search'][key]=scalars(search.get(key,{}),search.get(key,{}))
    out['search']['teaching_only_categorical_probe']=search.get('teaching_only_categorical_probe')
    for key in ('samples_by_phase','used_samples_by_phase','unused_samples_by_phase','sample_presentations_by_phase'):
        if key in search:out['search'][key]=search[key]
    out['search']['collection_summaries']=[dict(
        scalars(batch,('teacher_model_id','due_games','elapsed_seconds','teacher_rows','optimizer_steps',
            'sample_presentations','max_observed_reuse','distinct_labels_used','unused_qualified_labels',
            'distill_batch_size','sample_presentation_cap','utility_calibration_invalidated','post_distill_calibration_seconds')),
        teacher_policy_drift=batch.get('teacher_policy_drift'),
        sample_usage={key:batch[key] for key in ('samples_by_phase','used_samples_by_phase','unused_samples_by_phase',
            'sample_presentations_by_phase') if key in batch},
        phases=batch.get('teaching',{}).get('phases',{}),
        rollout_generations=batch.get('rollout_generations',[])[:65],learning_rates=batch.get('learning_rates',[])[:8],
        post_distill_calibration=calibration(batch.get('post_distill_calibration')))
        for batch in search.get('collections',[])[:8]]
    out['search']['phases']={name:scalars(row,row)
        for name,row in list(search.get('teaching',{}).get('phases',{}).items())[:4]}
    out['results']={name:scalars(row,('games','win_share','mean_rank','mean_vp'))
                    for name,row in list(report.get('results',{}).items())[:4]}
    out['champion_gate']=scalars(report.get('result',report.get('champion_gate')),
        ('accepted','candidate_win_share','candidate_vp','incumbent_vp','candidate_rank','incumbent_rank',
         'games','setups','stage','protocol','planned_setups','threshold','min_ci_lower'))
    gate=report.get('result',report.get('champion_gate')) or {}
    for key in ('setup_bootstrap_95','screen','confirmation'):
        if key in gate:out['champion_gate'][key]=gate[key]
    # Report only learner metrics for new runs; older summaries retain at most
    # sixteen policy rows per composition, rather than arbitrary telemetry trees.
    out['by_composition']={}
    for name,row in list(report.get('by_composition',{}).items())[:3]:
        policies=row.get('matches',{}).get('policies',{})
        who=report.get('incoming_policy_id')
        selected={who:policies[who]} if who in policies else dict(list(policies.items())[:16])
        out['by_composition'][name]={key:dict(scalars(v,('seat_games','win_share','placement','vp')),
            seat_counts=v.get('seat_counts')) for key,v in selected.items()}
    out['learner_by_seat']=report.get('learner_by_seat',{})
    out['learner_by_composition_seat']={k:v.get('learner_by_seat',{}) for k,v in list(report.get('by_composition',{}).items())[:3]}
    out['economy']={key:report.get('economy',{}).get(key) for key in
        ('actions','actions_by_era','builds','build_levels','observed_flips','last_seen_unflipped',
         'passes_with_nonpass_alternative','sale_actions_started','sale_continuations','merchant_beer_by_bonus')}
    utilization=report.get('utilization',{})
    out['utilization']={key:utilization.get(key) for key in ('games_per_second','mean_inference_batch',
        'inference_batches','inference_positions','padding_fraction','actor_phase_seconds',
        'inference_phase_deltas','replay_ingress','scope')}
    out['selfplay_incumbent']=report.get('selfplay_incumbent')
    out['historical_opponents']=report.get('historical_opponents')
    out['optimizer_resume']=report.get('optimizer_resume')
    return out


def compact_history(run,archived=(),limit=128):
    """Migrate existing large archives once; compact even when nothing retires."""
    path=Path(run)/'retention-history.json'
    before=read(path) if path.exists() else {}
    if not before and not archived:return
    history={key:dict(sha256=value.get('sha256'),summary=compact(value.get('summary',{}))) for key,value in before.items()}
    for key,checksum,report in archived:history[key]=dict(sha256=checksum,summary=compact(report))
    def order(key):
        values=re.findall(r'\d+',key)
        summary=history[key]['summary']
        return (summary.get('games_done',summary.get('iteration',int(values[-1]) if values else -1)),key)
    # Prefer recent additions. Numeric ordering handles iteration-10000 safely;
    # evaluation reports have their own bounded on-disk selection receipts.
    history=dict(sorted(history.items(),key=lambda item:order(item[0]))[-limit:])
    if history!=before:write_json(path,history)
