"""Restyle an existing PL report as progress.svg, without refitting or training.

Example:
    python scripts/render-pl-progress.py --report results/.../ratings/slots/generation-000210/report.json
"""
import argparse
import json
import math
from pathlib import Path


def render(report, output, *, preview=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch, Patch
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ratings=report['ratings']
    points=sorted((int(name.removeprefix('generation-')), name)
        for name in ratings if name.startswith('generation-') and int(name.removeprefix('generation-'))>0)
    if not points:
        raise ValueError('The report has no rated training checkpoints')
    def values(name):
        row=ratings[name];value=row['points'];low,high=row['points_ci95']
        if not all(math.isfinite(v) for v in (value,low,high)) or low>high:
            raise ValueError('Invalid PL point estimate or interval: '+name)
        return value,low,high
    data=[values(name) for _,name in points]
    x=[g for g,_ in points];y=[v for v,_,_ in data]
    lower=[lo for _,lo,_ in data];upper=[hi for _,_,hi in data]
    latest,latest_name=points[-1];score,low,high=data[-1]
    v4,v4low,v4high=values('v4')
    anchor=values('generation-0')[0]
    if anchor!=0:
        raise ValueError('Expected the fixed IL reference at zero PL points')

    ink='#19333F';muted='#6B7B83';blue='#177E89';blue_dark='#0D6470'
    amber='#B27E37';grid='#E6ECEE';paper='#F6F7F5'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,
            'text.color':ink,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':muted,
            'svg.fonttype':'none','svg.hashsalt':'brass-pl-progress-v3',
            'axes.unicode_minus':False,'lines.solid_capstyle':'round'}):
        fig=plt.figure(figsize=(13.2,8.2),facecolor=paper)
        def text(x,y,label,**kwargs):
            return fig.text(x,y,label,**kwargs)
        fig.add_artist(Line2D([.06,.089],[.956,.956],transform=fig.transFigure,
            color=amber,linewidth=2.5))
        text(.102,.951,'BRASS BIRMINGHAM  /  SELF-PLAY',fontsize=8.6,weight='bold',color=muted)
        text(.06,.906,'Self-play strength',fontsize=27,weight='bold')
        text(.06,.874,'Plackett–Luce rating progression',fontsize=11,color=muted)
        text(.938,.920,'GENERATION '+str(latest),ha='right',va='center',fontsize=9.5,
            weight='bold',color=blue_dark,bbox=dict(boxstyle='round,pad=.65,rounding_size=.25',
                facecolor='#E5F0EE',edgecolor='none'))
        text(.938,.877,'Common fit · IL anchored at 0',ha='right',fontsize=9,color=muted)

        # Compact, aligned summaries give the plot a larger share of the page.
        text(.06,.834,'LATEST PL SCORE',fontsize=8.4,weight='bold',color=muted)
        text(.06,.790,f'{score:,.1f}',fontsize=25,weight='bold',color=blue_dark)
        text(.184,.798,'points above IL',fontsize=9,color=muted)
        text(.377,.834,'95% BOOTSTRAP INTERVAL',fontsize=8.4,weight='bold',color=muted)
        text(.377,.792,f'{low:,.0f} – {high:,.0f}',fontsize=23,weight='bold')
        text(.722,.834,'EVALUATION COVERAGE',fontsize=8.4,weight='bold',color=muted)
        text(.722,.793,f'{len(points)}',fontsize=23,weight='bold')
        text(.770,.806,'rated checkpoints',fontsize=9,color=muted)
        text(.770,.786,f'{report.get("games",0):,} games · {report.get("setups",0):,} setups',
            fontsize=8.5,color=muted)
        for left in (.338,.682):
            fig.add_artist(Line2D([left,left],[.791,.838],transform=fig.transFigure,
                color='#DCE4E4',linewidth=.85))

        # A single quiet panel, with no decorative shadows or heavy borders.
        fig.add_artist(FancyBboxPatch((.044,.078),.912,.686,boxstyle='round,pad=0,rounding_size=.012',
            transform=fig.transFigure,facecolor='white',edgecolor='#E0E7E6',linewidth=.85,zorder=0))
        ax=fig.add_axes([.099,.184,.785,.502],facecolor='white')
        ax.set_axisbelow(True);ax.grid(axis='y',color=grid,linewidth=.8)
        for spine in ('left','right','top'):ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#D1DCDC');ax.spines['bottom'].set_linewidth(.8)
        ax.tick_params(axis='both',length=0,pad=9,labelsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8,steps=[1,2,2.5,5,10]))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        step=max(10,math.ceil(latest/70)*10)
        ticks=list(range(0,latest+1,step))
        if latest not in ticks:
            if len(ticks)>1 and latest-ticks[-1]<step*.55:ticks[-1]=latest
            else:ticks.append(latest)
        ax.set_xticks(ticks)
        ax.set_xlabel('Committed training generation',labelpad=15,fontsize=10)
        ax.text(0,1.025,'PL POINTS',transform=ax.transAxes,color=muted,fontsize=8.1,weight='bold')
        ax.set_xlim(-latest*.015,latest*1.10)
        bottom=min(anchor,v4low,*lower);top=max(v4high,*upper,score)
        span=max(100.,top-bottom)
        ax.set_ylim(bottom-span*.055,top+span*.085)

        # Preserve exact point estimates and percentile endpoints. A bootstrap
        # interval may exclude its MAP estimate; never clamp or recenter it.
        ax.axhspan(v4low,v4high,color=amber,alpha=.065,linewidth=0,zorder=1)
        ax.axhline(v4,color=amber,linewidth=1.2,linestyle=(0,(5,4)),zorder=2)
        ax.axhline(anchor,color='#9AA8AC',linewidth=1.05,linestyle=(0,(2,4)),zorder=2)
        ax.fill_between(x,lower,upper,color=blue,alpha=.10,linewidth=0,zorder=2)
        ax.vlines(x,lower,upper,color=blue,alpha=.20,linewidth=1.05,zorder=3)
        ax.plot(x,y,color=blue,linewidth=2.5,zorder=4)
        ax.scatter(x,y,s=25,facecolor='white',edgecolor=blue,linewidth=1.35,zorder=5)
        # Give the latest interval and estimate a deliberate visual hierarchy.
        ax.vlines([latest],[low],[high],color=blue_dark,alpha=.55,linewidth=1.5,zorder=5)
        ax.scatter([latest],[score],s=210,color=blue,alpha=.10,linewidth=0,zorder=6)
        ax.scatter([latest],[score],s=61,color=blue_dark,edgecolor='white',linewidth=1.8,zorder=7)
        ax.annotate(f'{score:,.1f}',xy=(latest,score),xytext=(12,4),textcoords='offset points',
            fontsize=11.5,color=blue_dark,weight='bold',va='center',annotation_clip=False)
        ax.annotate('LATEST',xy=(latest,score),xytext=(12,-11),textcoords='offset points',
            fontsize=7.2,color=muted,va='center',annotation_clip=False)
        for value,label,color in [(v4,f'v4  {v4:,.0f}',amber),(anchor,'IL  0','#829397')]:
            ax.text(1.007,(value-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]),label,
                transform=ax.transAxes,va='center',fontsize=8.7,weight='bold',color=color)
        handles=[Line2D([],[],color=blue,marker='o',markerfacecolor='white',linewidth=2,
                       markersize=5,label='Training checkpoint'),
                 Patch(facecolor=blue,alpha=.12,edgecolor='none',label='95% setup-bootstrap interval'),
                 Line2D([],[],color=amber,linewidth=1.3,linestyle=(0,(5,4)),label='v4 reference · no search')]
        fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.067,.730),ncol=3,
            frameon=False,fontsize=8.7,handlelength=2.2,columnspacing=2.5,labelcolor=muted)
        text(.070,.104,'PL points = 173.7 × θ',fontsize=8.2,color=muted)
        text(.929,.104,f'{report.get("bootstrap_refits",0):,} bootstrap refits · unsmoothed estimates',
            fontsize=8.2,color=muted,ha='right')
        text(.06,.034,'Pointwise 95% intervals show uncertainty. Use paired difference intervals to compare checkpoints.',
            fontsize=8.5,color=muted)
        metadata=dict(Title='Brass Birmingham — self-play PL strength',
            Description=f'Common-fit PL point estimates and pointwise 95% setup-bootstrap intervals through generation {latest}. Latest score {score:.3f}; interval [{low:.3f}, {high:.3f}]. IL anchor 0; v4 reference {v4:.3f}.',Date=None)
        fig.savefig(output,format='svg',facecolor=paper,metadata=metadata)
        # Matplotlib leaves spaces before newlines in SVG path attributes.
        # Newlines already separate their coordinates; keep the saved diff clean.
        output=Path(output)
        output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf-8').splitlines())+'\n',
            encoding='utf-8',newline='\n')
        if preview is not None:fig.savefig(preview,format='png',dpi=130,facecolor=paper)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='Defaults to progress.svg beside the report')
    parser.add_argument('--preview',type=Path,help='Optional PNG for visual inspection')
    args=parser.parse_args()
    output=args.output or args.report.with_name('progress.svg')
    if output.suffix.lower()!='.svg':raise ValueError('Output must be an SVG file')
    render(json.loads(args.report.read_text(encoding='utf-8-sig')),output,preview=args.preview)
    print(output)


if __name__=='__main__':main()
