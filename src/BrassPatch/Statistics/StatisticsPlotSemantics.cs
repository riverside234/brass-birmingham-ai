using System;
using System.Linq;
namespace BrassPatch.Statistics
{
    public static class StatisticsPlotSemantics
    {
        public static bool Horizontal(StatisticsPlot p) {return p.Kind==StatisticsPlotKind.StackedBars||p.Kind==StatisticsPlotKind.PairedBars||p.Kind==StatisticsPlotKind.Dumbbell||p.Kind==StatisticsPlotKind.Waterfall||p.Kind==StatisticsPlotKind.Lollipop||p.Kind==StatisticsPlotKind.DelayDots;}
        public static string Value(StatisticsPlot p,StatisticsPlotMark m)
        {
            if(m.Missing||Double.IsNaN(m.Y))return m.ValueLabel??"— · unavailable";
            if(m.ValueLabel!=null)return m.ValueLabel;
            if(p.Kind==StatisticsPlotKind.MiniMap)return m.Row+" "+p.YLabel.ToLowerInvariant();
            if(p.Percent){double total=p.Marks.Where(x=>x.Row==m.Row).Sum(x=>x.Y);return m.Y.ToString("0.##")+" of "+total.ToString("0.##")+" · "+StatisticsFormat.Percent(m.Y,total);}
            if(p.Kind==StatisticsPlotKind.Lifespans)return "Global turns "+m.X+" → "+m.End+" · duration "+m.Y+" global turns";
            if(p.Kind==StatisticsPlotKind.EventLanes)return "Global turn "+m.X+" · own turn "+m.Y;
            if(p.DiscreteHeat)return "Position "+m.Y.ToString("0");
            return StatisticsAxisLayout.Format(m.Y,Horizontal(p)?p.XUnit:p.YUnit,1,true)+" "+(Horizontal(p)?p.XLabel:p.YLabel);
        }
        public static string Other(StatisticsPlot p,StatisticsPlotMark m)
        {
            if(m.OtherLabel!=null)return m.OtherLabel;
            if(p.Kind==StatisticsPlotKind.Scatter)return StatisticsAxisLayout.Format(m.X,p.XUnit,1,true)+" · "+p.XLabel;
            if(p.Kind==StatisticsPlotKind.Step)return "Global turn "+m.X;
            if(p.Kind==StatisticsPlotKind.Waterfall)return "Balance "+StatisticsAxisLayout.Format(m.X,p.XUnit)+" → "+StatisticsAxisLayout.Format(m.X+m.Y,p.XUnit);
            return "";
        }
        public static string Description(StatisticsPlot p,StatisticsPlotMark m)
        {
            var s=p.Series.FirstOrDefault(x=>x.Id==m.Series);
            return m.Label+" · "+(s==null?m.Series:s.Label)+"\n"+Value(p,m)+(Other(p,m)==""?"":"\n"+Other(p,m))+"\n"+m.Detail+"\nExact data retains the recorded values and membership.";
        }
    }
}
