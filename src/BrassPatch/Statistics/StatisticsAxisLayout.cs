using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
namespace BrassPatch.Statistics
{
    public enum StatisticsAxisUnit { Count, Cash, Price, Ratio, Percent, Turn, Income }
    public sealed class StatisticsAxisTick { public double Value; public string Label; public float LabelWidth; }
    // The mesh, labels and hit geometry all consume these exact bounds/ticks.
    // Measurement is supplied by TMP at the current physical text preset.
    public sealed class StatisticsAxisLayout
    {
        public double Min,Max,Step;
        public readonly List<StatisticsAxisTick> Ticks = new List<StatisticsAxisTick>();
        public StatisticsAxisUnit Unit;
        public static StatisticsAxisLayout Create(double min,double max,StatisticsAxisUnit unit,float length,Func<string,float> measure,bool vertical=false)
        {
            if(Double.IsNaN(min)||Double.IsInfinity(min))min=0;
            if(Double.IsNaN(max)||Double.IsInfinity(max))max=0;
            if(max<min){double swap=min;min=max;max=swap;}
            min=Math.Min(0,min);max=Math.Max(0,max);
            bool integer=unit==StatisticsAxisUnit.Count||unit==StatisticsAxisUnit.Turn||unit==StatisticsAxisUnit.Income;
            if(max-min<1e-9)max=min+1;
            int target=vertical?6:length>=850?8:length>=570?7:5;
            if(unit==StatisticsAxisUnit.Percent){min=0;max=100;}
            double raw=(max-min)/Math.Max(1,target-1),power=Math.Pow(10,Math.Floor(Math.Log10(raw)));
            var candidates=new List<double>();
            for(int exponent=-1;exponent<=3;exponent++)foreach(double n in new[]{1.0,2,2.5,5,10}) {
                double step=n*power*Math.Pow(10,exponent);
                if(step>=raw*.8&&(!integer||step>=1&&Math.Abs(step-Math.Round(step))<1e-9))candidates.Add(step);
            }
            foreach(double step in candidates.Distinct().OrderBy(x=>x)) {
                var a=new StatisticsAxisLayout{Unit=unit,Step=step};
                a.Min=unit==StatisticsAxisUnit.Percent?0:Math.Floor(min/step)*step;
                a.Max=unit==StatisticsAxisUnit.Percent?100:Math.Ceiling(max/step)*step;
                if(a.Max<=a.Min)a.Max=a.Min+step;
                int count=(int)Math.Round((a.Max-a.Min)/step)+1;
                if(count>target+1||count>50)continue;
                for(int i=0;i<count;i++) {
                    double value=a.Min+i*step;string label=Format(value,unit,step);
                    a.Ticks.Add(new StatisticsAxisTick{Value=value,Label=label,LabelWidth=measure==null?label.Length*14:measure(label)});
                }
                if(a.Ticks.Count<2)continue;
                float gap=length/(a.Ticks.Count-1);
                bool fits=vertical?gap>=34:a.Ticks.Zip(a.Ticks.Skip(1),(left,right)=>(left.LabelWidth+right.LabelWidth)/2+18<=gap).All(x=>x);
                if(fits)return a;
            }
            // Very constrained views still have two distinct endpoint labels.
            var fallback=new StatisticsAxisLayout{Unit=unit,Min=integer?Math.Floor(min):min,Max=integer?Math.Ceiling(max):max,Step=max-min};
            foreach(double value in new[]{fallback.Min,fallback.Max}){string label=Format(value,unit,fallback.Step);fallback.Ticks.Add(new StatisticsAxisTick{Value=value,Label=label,LabelWidth=measure==null?label.Length*14:measure(label)});}return fallback;
        }
        public static string Format(double value,StatisticsAxisUnit unit,double step=1,bool exact=false)
        {
            if(Double.IsNaN(value)||Double.IsInfinity(value))return "—";
            int decimals=0;
            if(exact&&(unit==StatisticsAxisUnit.Ratio||unit==StatisticsAxisUnit.Price))decimals=2;
            else {while(decimals<6&&Math.Abs(step*Math.Pow(10,decimals)-Math.Round(step*Math.Pow(10,decimals)))>1e-8)decimals++;}
            if(unit==StatisticsAxisUnit.Price)decimals=Math.Max(2,decimals);
            if(Math.Abs(value)<Math.Pow(10,-decimals)*.5)value=0;
            string number=Math.Abs(value).ToString(decimals==0?"0":"0."+new string('0',decimals),CultureInfo.InvariantCulture);
            return (value<0?"−":"")+(unit==StatisticsAxisUnit.Cash||unit==StatisticsAxisUnit.Price?"£":"")+number+(unit==StatisticsAxisUnit.Percent?"%":"");
        }
    }
}
