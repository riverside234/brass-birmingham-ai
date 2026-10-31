"""Headless view of committed context evidence. No collection or statistics."""
from pathlib import Path
import time
from brass_ai.learning.contracts import require

VERSION='domino-context-dashboard/1.0.0'
COLORS=('#6B7780','#167D8D','#7768A6','#BC735C','#8D912E','#4177AB','#865E52')
LABELS=('Neutral','Cotton mill','Manufacturer','Pottery','Brewery','Iron works','Coal mine')


def render(report,output,*,preview=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    import numpy as np
    began=time.perf_counter();generation=report['generation'];output=Path(output)
    ink='#203B43';muted='#5D737C';teal='#167D8D';grid='#E5ECEE';amber='#9C753F'
    history=report.get('history',[])
    require(len(history)<=12 and all(h['generation']<=generation for h in history),'Invalid frozen panel history')
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,'axes.labelcolor':muted,
        'xtick.color':muted,'ytick.color':muted,'svg.fonttype':'none','svg.hashsalt':VERSION,
        'axes.unicode_minus':False}):
        fig=plt.figure(figsize=(18,18),facecolor='#F6F8F7')
        try:
            fig.text(.045,.976,'BRASS BIRMINGHAM / CONTEXT EVALUATION',fontsize=11,color=muted,weight='bold')
            fig.text(.045,.943,'Strategy quality and diversity',fontsize=24,weight='bold')
            fig.text(.955,.974,f'GENERATION g{generation:,}',ha='right',fontsize=13,weight='bold')
            fig.text(.955,.946,'T=1 / epsilon=0',ha='right',fontsize=12,color=muted)
            if report.get('test_only'):
                fig.text(.045,.918,'TEST-ONLY EVIDENCE — statistically insufficient; not a live strength result',color=amber,fontsize=11,weight='bold')
            elif report.get('synthetic'):
                fig.text(.045,.918,'SYNTHETIC FIXTURE — authored visual QA, not live results',color=amber,fontsize=11,weight='bold')
            if report.get('status')!='complete':
                fig.text(.5,.6,'INCOMPLETE PANEL',ha='center',fontsize=30,color=amber)
                fig.text(.5,.54,'No seven-context comparison published. See the status receipt.',ha='center',fontsize=14)
            else:
                s=report['support'];rows=report['contexts'];mixture=rows[-1]['paired']['vp'];d=report['descriptors']
                cost=report.get('cost',{});fraction=cost.get('incremental_wall_fraction')
                cards=[('SUPPORT',f'{s["setups"]} paired setups × 4',
                    f'{s["games_per_context"]} games/context\n{s["new_games"]} new + {s["reused_games"]} reused'),
                    ('MIXTURE VS NEUTRAL',f'{mixture["mean"]:+.2f} VP',
                    f'95% [{mixture["ci95"][0]:+.2f}, {mixture["ci95"][1]:+.2f}]\n'+('Difference uncertain' if mixture['ci95'][0]<=0<=mixture['ci95'][1] else 'Within-checkpoint comparison')),
                    ('DIVERSITY · EXPLORATORY',f'{d["without_industry"]["separation"]:.5f}',
                    f'Without industry families\nNull mean {d["without_industry"]["null"]["mean"]:.5f}'),
                    ('ADDITIONAL COST',f'{cost["additional_seconds"]/60:.2f} min' if 'additional_seconds' in cost else 'Not yet available',
                    (f'Measured overhead {fraction:.2%}' if fraction is not None else 'Overhead not yet available')+'\n5% review budget · advisory')]
                for x,(title,value,note) in zip((.035,.275,.515,.755),cards):
                    fig.add_artist(FancyBboxPatch((x,.814),.21,.086,boxstyle='round,pad=.008,rounding_size=.008',
                        facecolor='white',edgecolor=grid,transform=fig.transFigure,zorder=0))
                    fig.text(x+.008,.879,title,fontsize=10,weight='bold',color=muted)
                    fig.text(x+.008,.850,value,fontsize=18,weight='bold',color=teal)
                    fig.text(x+.008,.821,note,fontsize=10,color=muted,linespacing=1.45)

                def axes(rect,title,subtitle):
                    ax=fig.add_axes(rect,facecolor='white')
                    ax.set_title(title,loc='left',fontsize=14,pad=32,weight='bold')
                    ax.text(0,1.05,subtitle,transform=ax.transAxes,fontsize=10,color=muted)
                    for spine in ax.spines.values():spine.set_visible(False)
                    return ax

                forest=axes([.115,.629,.335,.134],'Paired VP difference from neutral','Same checkpoint · marginal 95% setup intervals')
                endpoints=[0.]
                for i,row in enumerate(rows[1:]):
                    value=row['paired']['vp'];mean=value['mean'];lo,hi=value['ci95'];endpoints.extend((mean,lo,hi))
                    color=COLORS[i+1] if i<6 else '#203B43'
                    forest.plot([lo,hi],[6-i,6-i],color=color,lw=2)
                    forest.scatter([mean],[6-i],marker='D' if i==6 else 'o',s=48,color=color,zorder=3)
                forest.axvline(0,color=muted,lw=1,linestyle='--');forest.grid(axis='x',color=grid)
                span=max(endpoints)-min(endpoints);pad=max(1.,span*.12)
                forest.set_xlim(min(endpoints)-pad,max(endpoints)+pad);forest.set_ylim(-.6,6.6)
                forest.set_yticks(range(7),list(reversed((*LABELS[1:],'Uniform mixture ◇'))),fontsize=11)
                forest.set_xlabel('VP difference');forest.xaxis.set_major_locator(MaxNLocator(5))
                forest.text(0,-.29,'Neutral is the reference. Mixture is derived, not extra games.',transform=forest.transAxes,fontsize=10,color=muted)

                table=axes([.52,.61,.44,.156],'Candidate outcomes','Win interval / VP / quality margin / insolvent seats')
                table.axis('off')
                data=[]
                for i,row in enumerate(rows):
                    lo,hi=row['ci95']['win_share'];counts=row['counts']
                    affected=f'{counts["insolvent"]}/{row["games"]}' if counts else f'{row["means"]["insolvent"]:.1%}*'
                    data.append([LABELS[i] if i<7 else 'Mixture ◇',f'{100*row["means"]["win_share"]:.1f} [{100*lo:.1f}, {100*hi:.1f}]',
                        f'{row["means"]["vp"]:.1f}',f'{row["task_quality_margin"]:+.3f}',affected])
                artist=table.table(cellText=data,colLabels=['Context','Win % [95%]','VP','J − .95 J₀','Insolvent'],
                    cellLoc='left',colLoc='left',colWidths=[.24,.29,.12,.17,.18],bbox=[0,0,1,1])
                artist.auto_set_font_size(False);artist.set_fontsize(10)
                for (r,c),cell in artist.get_celld().items():
                    cell.set_edgecolor(grid);cell.set_linewidth(.5)
                    cell.set_facecolor('#EDF3F2' if r in (0,8) else 'white')
                    if r==0:cell.set_text_props(weight='bold',color=ink)
                table.text(0,-.12,'* Derived average rate; no mixture event count or extra support.',transform=table.transAxes,fontsize=10,color=muted)

                vmax=float(d['heatmap_limits'][1]);vmax=vmax if vmax>0 else 1.e-12
                for x,key,title in ((.10,'full','Full descriptor distance'),(.59,'without_industry','Distance without industry families')):
                    matrix=d[key];ax=axes([x,.354,.31,.184],title,'Euclidean distance · same numeric color scale in both panels')
                    image=ax.imshow(matrix['distance_matrix'],vmin=0,vmax=vmax,cmap='YlGnBu',aspect='equal')
                    ax.set_xticks(range(7),LABELS,rotation=30,ha='right',fontsize=10)
                    ax.set_yticks(range(7),LABELS,fontsize=10)
                    for i,row in enumerate(matrix['distance_matrix']):
                        for j,value in enumerate(row):
                            ax.text(j,i,f'{value:.3f}',ha='center',va='center',fontsize=10,
                                color='white' if value>vmax*.55 else ink)
                    bar=fig.colorbar(image,ax=ax,fraction=.035,pad=.035)
                    bar.ax.tick_params(labelsize=10);bar.set_label('Distance',fontsize=10)
                    ax.text(0,-.40,f'Within-context RMS: {matrix["mean_within_context_rms"]:.4f}\n'
                        f'Separation {matrix["separation"]:.5f}; null p={matrix["null"]["p_exploratory"]:.3f} (exploratory)',
                        transform=ax.transAxes,fontsize=10,color=muted,linespacing=1.5)

                quality=axes([.07,.108,.385,.102],'Recent paired quality history','Each point uses its own checkpoint and matched opponents')
                overhead=axes([.56,.108,.385,.102],'Actual evaluation overhead','Added diagnostic wall time / normal trainer time')
                # Protocol changes are gaps; neither pooling nor smoothing.
                compatible=[h for h in history if h.get('protocol_id')==report.get('protocol_id')]
                for i,h in enumerate(history):
                    if h not in compatible:continue
                    value=h['mixture_vp'];g=h['generation']
                    quality.plot([g,g],value['ci95'],color=teal,lw=1.8)
                    quality.scatter([g],[value['mean']],marker='D',color=teal,s=35)
                    rate=h.get('incremental_wall_fraction')
                    if rate is not None:overhead.scatter([g],[rate*100],color=teal,s=40)
                quality.axhline(0,color=muted,linestyle='--',lw=1)
                overhead.axhline(5,color=amber,linestyle='--',lw=1)
                for ax in (quality,overhead):
                    ax.grid(axis='y',color=grid);ax.set_xlabel('Committed generation');ax.xaxis.set_major_locator(MaxNLocator(5,integer=True))
                    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));ax.tick_params(axis='x',labelsize=10)
                    if len(compatible)==1:ax.set_xlim(generation-1,generation+1)
                quality.set_ylabel('Mixture − neutral VP');overhead.set_ylabel('Additional wall time (%)')
                if len(compatible)<=1:quality.text(.02,.95,'First panel',va='top',transform=quality.transAxes,fontsize=10,color=muted)
                if not any(h.get('incremental_wall_fraction') is not None for h in compatible):
                    overhead.text(.5,.45,'Not yet available\nMissing timing evidence is not zero',ha='center',transform=overhead.transAxes,fontsize=11,color=muted)
                fig.text(.045,.048,'T=1, epsilon=0; fixed candidate context vs neutral-context frozen opponents. All four rotations remain one setup cluster.',fontsize=10,color=muted)
                fig.text(.045,.032,'Paired stratified percentile bootstrap; marginal 95% intervals. Ablation removes investment + realization without rescaling. Diversity is exploratory.',fontsize=10,color=muted)
                fig.text(.045,.016,f'{VERSION}  |  report {report.get("panel_digest","unavailable")[:16]}  |  '+
                    ('SYNTHETIC FIXTURE' if report.get('synthetic') else 'TEST-ONLY REAL GAMES' if report.get('test_only') else 'COMPLETE REPORT-ONLY PANEL'),fontsize=10,color=muted)
            output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.svg.tmp')
            fig.savefig(temp,format='svg',metadata={'Date':None,'Title':f'DOMiNO context evaluation g{generation}',
                'Description':f'{report.get("status")} / {report.get("support",{})} / '+', '.join(r.get('context','') for r in report.get('contexts',[])),
                'Creator':VERSION})
            temp.replace(output)
            if preview:
                preview=Path(preview);preview.parent.mkdir(parents=True,exist_ok=True)
                fig.savefig(preview,dpi=1280/18,facecolor=fig.get_facecolor())
        finally:plt.close(fig)
    return dict(renderer=VERSION,seconds=time.perf_counter()-began,bytes=output.stat().st_size,
        history_receipts=[h.get('report') for h in history if h.get('report')])
