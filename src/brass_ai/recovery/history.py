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
    out=scalars(value,('required','ready','version','reward_id','decisions','games','steps','reason','scope',
        'ordinary_fraction','teacher_fraction','teacher_decisions','teacher_games','teacher_seconds',
        'teacher_sample_presentations','teacher_optimizer_steps','readiness_gate'))
    for key in ('before','after','ordinary_before','ordinary_after','teacher_before','teacher_after'):
        if isinstance(value.get(key),dict):out[key]=scalars(value[key],('mse','mae'))
    if isinstance(value.get('receipt'),dict):out['receipt']=calibration(value['receipt'])
    return out


def teacher_value(value):
    """Bounded totals/four-phase probes only, never per-position arrays."""
    if not isinstance(value,dict):return None
    out=scalars(value,('enabled','contract','target_source','reward_id','distinct_completed_games','valid_value_positions',
        'selected_positions','distinct_positions_trained','unused_positions','sample_presentations','max_auxiliary_reuse',
        'ordinary_positions_trained','ordinary_sample_presentations','auxiliary_distinct_positions','auxiliary_presentations',
        'calibration_distinct_positions','calibration_presentations','loss_weight','ordinary_fraction','teacher_fraction',
        'weighting_scope','preparation_seconds','auxiliary_training_seconds','policy_kl','skip_reason','reason','collections','scope',
        'ordinary_mean_loss','teacher_mean_loss','weighted_auxiliary_loss'))
    out['provenance']=scalars(value.get('provenance'),('contract','model_id','implementation','reward_id','config_id'))
    out['phases']={p:scalars(v,('valid','trained','unused','presentations')) for p,v in list(value.get('phases',{}).items())[:4]}
    probe=value.get('diagnostics',{})
    out['diagnostics']=scalars(probe,('enabled','reason','positions','games','sampling','scope','elapsed_seconds',
        'extra_forward_passes','extra_forward_positions','reused_fit_positions','reused_calibration_positions',
        'prediction_precision','iteration_overhead_target'))
    out['diagnostics']['phases']=scalars(probe.get('phases'),('canal_early','canal_late','rail_early','rail_late'))
    out['diagnostics']['stages']={}
    for stage in ('before','post_update','post_calibration'):
        data=probe.get('stages',{}).get(stage)
        if not isinstance(data,dict):continue
        row=scalars(data,('available','reason'))
        row['errors']=scalars(data.get('errors'),('win_share_brier','rank_mae','vp_mae','utility_mae','utility_mse'))
        changes=data.get('prediction_change_from_before') or {}
        row['prediction_change_from_before']={k:scalars(changes[k],('mean_absolute','maximum_absolute'))
            for k in ('win_share','rank','vp','utility') if k in changes}
        out['diagnostics']['stages'][stage]=row
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
    # Preserve their version and missingness instead of relabelling as v5.
    if report.get('summary_version') in (2,3,4,5,6):return report
    out=scalars(report,('format','iteration','games_done','cumulative_games','incoming_model_id','post_update_model_id',
                       'incoming_policy_id','elapsed_seconds','games','failed_games','promotion_eligible'))
    out['summary_version']=6 if report.get('value_phase') is not None or 'gae' in report.get('actor',{}) else 5
    for key in ('il_kl_limit','rehearsal_schedule','health_counts','quality_pool','strength_rating','value_phase','seat_fingerprints','gate_diagnostics'):
        if key in report:out[key]=report[key]
    out['health_by_composition']={k:v['health_counts'] for k,v in report.get('by_composition',{}).items()
                                  if 'health_counts' in v}
    if report.get('format') in ('step5-evaluation-report/1.0.0','four-player-champion/1.0.0'):
        out['evaluation_evidence']=evaluation_evidence(report)
    out['timings']=scalars(report.get('timings',{}),('collection_seconds','replay_load_seconds','ppo_seconds',
        'sampling_seconds','rehearsal_prepare_seconds','rehearsal_cleanup_seconds',
        'post_ppo_diagnostics_seconds','final_diagnostics_seconds',
        'value_phase_seconds','search_and_distill_seconds','checkpoint_seconds','evaluation_seconds','cleanup_seconds','iteration_seconds'))
    actor=report.get('actor',{})
    out['actor']=scalars(actor,('optimizer_steps','learner_decisions','meaningful_decisions','forced_decisions',
        'learner_trajectories','kl_early_stop','rehearsal_weight','gradient_clip_steps','baseline_explained_variance',
        'baseline_scope','kl_estimator','kl_probe_rows','kl_probe_sampling'))
    out['actor']['learning_rates']=actor.get('learning_rates',[])[:8]
    out['actor']['lr_controller']=actor.get('lr_controller')
    out['actor']['optimizer_steps_planned']=actor.get('optimizer_steps_planned')
    out['actor']['optimizer_steps_fraction']=actor.get('optimizer_steps_fraction')
    schedule=actor.get('learning_rate_schedule')
    out['actor']['learning_rate_schedule']=scalars(schedule,('contract','kind','base_lr','steps','start_factor',
        'completed_steps','effective_lr','next_lr')) if schedule else None
    out['actor']['sample_usage']=actor.get('sample_usage')
    for key in ('gae','critic_split','critic_rows','final_categorical_probe'):
        if key in actor:out['actor'][key]=actor[key]
    sampling=actor.get('rehearsal_sampling')
    if isinstance(sampling,dict):out['actor']['rehearsal_sampling']=scalars(sampling,(
        'contract','iteration','source_rows','source_games','source_shards','selected_shards','selected_rows',
        'distinct_rows_selected','nonforced_rows_selected','used_presentations','distinct_rows_used',
        'selection_sha256','manifest_sha256','max_cache_bytes','peak_cache_bytes','preparation_seconds','cache_deleted','scope',
        'workers_requested','workers','reused_reader_pool','worker_read_prepare_seconds','worker_write_seconds',
        'worker_cpu_seconds','dense_ipc_bytes'))
    if 'kl_guard' in actor:out['actor']['kl_guard']=dict(actor['kl_guard'])
    for key in ('advantage','advantage_distribution','return_distribution','legal_actions','categorical_probe','gradient_norms','phases'):
        out['actor'][key]=scalars(actor.get(key,{}),actor.get(key,{}))
    epochs=actor.get('epochs',[])
    if epochs:
        out['actor']['epochs']=[dict(scalars(epoch,('epoch','actor','value','win_loss','rank_loss','vp_loss',
            'entropy','post_update_kl','probe_kl','probe_categorical_kl','clip_fraction','gradient_norm','actor_positions','critic_positions')),
            utility_prediction=scalars(epoch.get('utility_prediction',{}),('mse','mae'))) for epoch in epochs[-8:]]
        out['actor']['last_epoch']=out['actor']['epochs'][-1]
        out['actor']['utility_prediction']=scalars(epochs[-1].get('utility_prediction',{}),('mse','mae'))
    out['actor']['insolvency']=scalars(actor.get('insolvency',{}),('events','affected_trajectories','learner_trajectories','mean_cost_per_trajectory'))
    reward=actor.get('reward') or {}
    out['actor']['reward_components']=scalars(reward.get('components',{}),('win','placement','vp','cost','utility'))
    out['actor']['reward_spec']=reward.get('spec')
    out['actor']['reward_distribution']=reward.get('distribution')
    out['actor']['utility_calibration']=calibration(actor.get('utility_calibration'))
    search=report.get('search',{})
    out['search']=scalars(search,('enabled','updated','reason','collection_count','optimizer_steps','teacher_rows','teacher_setups',
        'sample_presentations','max_observed_reuse','distinct_labels_used','unused_qualified_labels',
        'post_teacher_sampled_kl','post_teacher_kl_scope'))
    for key in ('fit_before','fit_after','budget','timings'):
        out['search'][key]=scalars(search.get(key,{}),search.get(key,{}))
    out['search']['teaching_only_categorical_probe']=search.get('teaching_only_categorical_probe')
    out['search']['teacher_value']=teacher_value(search.get('teacher_value'))
    for key in ('samples_by_phase','used_samples_by_phase','unused_samples_by_phase','sample_presentations_by_phase'):
        if key in search:out['search'][key]=search[key]
    out['search']['collection_summaries']=[dict(
        scalars(batch,('teacher_model_id','due_games','elapsed_seconds','teacher_rows','optimizer_steps',
            'sample_presentations','max_observed_reuse','distinct_labels_used','unused_qualified_labels',
            'distill_batch_size','sample_presentation_cap','utility_calibration_invalidated','post_distill_calibration_seconds')),
        teacher_policy_drift=batch.get('teacher_policy_drift'),
        teacher_value=teacher_value(batch.get('teacher_value')),
        timings=scalars(batch.get('timings',{}),batch.get('timings',{})),
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
    out['selfplay_opponents']=report.get('selfplay_opponents')
    out['historical_opponents']=report.get('historical_opponents')
    out['optimizer_resume']=report.get('optimizer_resume')
    for key in ('composition','sampling','curriculum'):
        if key in report:out[key]=report[key]
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
