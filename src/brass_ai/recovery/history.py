"""Bounded diagnostic summaries, never replay/restore evidence or run state."""
from pathlib import Path
import re

from brass_ai.bank import read
from brass_ai.step5.league import write_json


def scalars(value,keys):
    if not isinstance(value,dict):return {}
    return {k:value[k] for k in keys if k in value and (value[k] is None or type(value[k]) in (int,float,bool,str))}


def compact(report):
    if report.get('summary_version')==2:return report
    out=scalars(report,('format','iteration','games_done','cumulative_games','incoming_model_id','post_update_model_id',
                       'incoming_policy_id','elapsed_seconds','games','failed_games','promotion_eligible'))
    out['summary_version']=2
    out['timings']=scalars(report.get('timings',{}),('collection_seconds','replay_load_seconds','ppo_seconds',
        'search_and_distill_seconds','checkpoint_seconds','evaluation_seconds','cleanup_seconds','iteration_seconds'))
    actor=report.get('actor',{})
    out['actor']=scalars(actor,('optimizer_steps','actor_rows','critic_rows','learning_rate','kl_early_stop','rehearsal_weight'))
    epochs=actor.get('epochs',[])
    if epochs:
        out['actor']['last_epoch']=scalars(epochs[-1],('policy','value','entropy','post_update_kl','clip_fraction','loss'))
        out['actor']['utility_prediction']=scalars(epochs[-1].get('utility_prediction',{}),('mse','mae'))
    out['actor']['insolvency']=scalars(actor.get('insolvency',{}),('events','affected_trajectories','learner_trajectories','mean_cost_per_trajectory'))
    reward=actor.get('reward',{})
    out['actor']['reward_components']=scalars(reward.get('components',{}),('win','placement','vp','cost','utility'))
    search=report.get('search',{})
    out['search']=scalars(search,('updated','reason','collection_count','optimizer_steps','teacher_rows','teacher_setups'))
    out['search']['phases']={name:scalars(row,('scheduled_turns','actual_searches','qualified_samples','failures',
        'elapsed_seconds','completed_rollouts','failed_turns','timeouts','insufficient_coverage'))
        for name,row in list(search.get('teaching',{}).get('phases',{}).items())[:4]}
    out['results']={name:scalars(row,('games','win_share','mean_rank','mean_vp'))
                    for name,row in list(report.get('results',{}).items())[:4]}
    out['champion_gate']=scalars(report.get('result',report.get('champion_gate')),
        ('accepted','candidate_win_share','candidate_vp','incumbent_vp','candidate_rank','incumbent_rank'))
    # Report only learner metrics for new runs; older summaries retain at most
    # sixteen policy rows per composition, rather than arbitrary telemetry trees.
    out['by_composition']={}
    for name,row in list(report.get('by_composition',{}).items())[:3]:
        policies=row.get('matches',{}).get('policies',{})
        who=report.get('incoming_policy_id')
        selected={who:policies[who]} if who in policies else dict(list(policies.items())[:16])
        out['by_composition'][name]={key:scalars(v,('seat_games','win_share','placement','vp')) for key,v in selected.items()}
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
