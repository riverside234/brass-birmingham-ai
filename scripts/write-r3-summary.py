"""Render the completed qualification record into the authored R3 exit document."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT,below,read,require

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--report',required=True)
a=p.parse_args();root=a.root.resolve();report=read(below(root,a.report))
require(report['format']=='legacy-qualification/1.0.0','Wrong report type')
profiles=report['profiles'];training=[n for n,v in profiles.items() if v['training_eligible']];evaluation=[n for n,v in profiles.items() if v['evaluation_eligible']]
lines=[f"All **{report['total_games']} prescribed games** finished processing; **{report['failed_games']} failed games**. "
       'Failures are not credited as wins. Bank source/configuration was frozen throughout the study.',
       '',f"Admitted training profiles: {', '.join('`'+n+'`' for n in training) or 'none'}. "
       f"Held-out evaluation profiles: {', '.join('`'+n+'`' for n in evaluation) or 'none'}.",
       '',f"The [complete qualification record](../{Path(a.report).as_posix()}) and [disjoint pools](../training/opponents/pool.json) preserve the exact identities and evidence hashes.",
       '', '| Configuration | Guarded-stock games | First-place share | Setup-bootstrap 95% interval | Mean rank | Mean VP | All-table turn p95 |',
       '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
for name,v in profiles.items():
    g=v['guarded']
    if not g['games']:continue
    interval=g['win_share_setup_bootstrap_95'];ci=f'{interval[0]:.1%}–{interval[1]:.1%}' if interval else 'insufficient groups'
    lines.append(f"| {name} | {g['games']} | {g['win_share']:.1%} | {ci} | {g['mean_rank']:.2f} | {g['mean_vp']:.1f} | {v['headless_whole_turn_ms']['p95']/1000:.2f} s |")
pair=report['reply_comparison'];interval=pair['setup_bootstrap_95']
ci=f'{100*interval[0]:+.2f} to {100*interval[1]:+.2f} percentage points' if interval else 'unavailable'
delta=f"{100*pair['win_share_delta']:+.2f} percentage points" if pair['win_share_delta'] is not None else 'unavailable'
confirmed=sum(v['reply_status'].get('confirmed',0) for v in profiles.values())
fallback=sum(v['reply_status'].get('no_fresh_batch_s1_incumbent',0) for v in profiles.values())
reply_failures=sum(v['reply_failures'] for v in profiles.values())
lines += ['', 'These are exploratory empirical intervals over only four setup clusters per configuration. '
          'A narrow or degenerate bootstrap interval is not certainty about unseen games. Identical and mixed tables are included in failure and latency reporting; '
          'the win/rank/VP columns above are against three guarded stock opponents.',
          '',f"**H4 equal-time gate:** {pair['paired_games']} paired seat/setup results, both at {pair['milliseconds']/1000:g} seconds per turn. "
          f"S2 minus S1 first-place share is {delta}; grouped interval {ci}. " +
          ('The prespecified benefit check passes.' if pair['benefit_demonstrated'] else 'The prespecified benefit check does not pass; S2 is implemented, but it is not promoted as a stronger replacement for S1.'),
          '', f"Across all S2 tables, {confirmed} decisions completed fresh confirmation and {fallback} kept the S1 incumbent because a fresh matched batch did not finish. There were {reply_failures} rollout failures. These counts cover guarded-stock and identical-profile S2 matches.",
          '', 'Specialist S2 configurations were evaluated only in their declared identical tables and lack the full individual admission matrix. '
          'They remain diagnostic versions. The 10-second S1 control also lacks the required mixed/identical-table coverage and is not admitted. No profile passes the general improved-Hard or live-deployment gate in this study.']
path=root/'docs/R3_EXIT_CHECKS.md';text=path.read_text(encoding='utf-8')
start=text.index('<!-- FINAL_TOURNAMENT_RESULTS -->');end=text.index('<!-- END_FINAL_TOURNAMENT_RESULTS -->')+len('<!-- END_FINAL_TOURNAMENT_RESULTS -->')
text=text[:start]+'\n\n'.join(['<!-- FINAL_TOURNAMENT_RESULTS -->','\n'.join(lines),'<!-- END_FINAL_TOURNAMENT_RESULTS -->'])+text[end:]
text=text.replace('**Correctness and integration checks pass. The preregistered 156-game tournament is still running; pool admission and the final strength gates will be recorded here after it finishes.**',
    '**Implementation, correctness and integration checks pass. The 156-game study is complete, with version-specific admission below. Full R3 playing-strength/deployment qualification remains open wherever its gates are not met.**')
path.write_text(text,encoding='utf-8')
print({'training_profiles':training,'evaluation_profiles':evaluation,'s2_benefit_demonstrated':pair['benefit_demonstrated']})
