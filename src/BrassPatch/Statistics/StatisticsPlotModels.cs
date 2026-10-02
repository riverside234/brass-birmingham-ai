using System;
using System.Collections.Generic;
using BoardGameRules.Entities.Players;

namespace BrassPatch.Statistics
{
    public enum StatisticsPlotKind { StackedBars, PairedBars, Dumbbell, Heatmap, Step, Waterfall, Lollipop, Scatter, EventLanes, Lifespans, MiniMap, DelayDots, Columns }
    public enum StatisticsColorRole { Player, Industry, Resource, Category, Era }
    public sealed class StatisticsPlotSeries
    {
        public string Id, Label;
        public StatisticsColorRole Role;
        public PlayerColor? Player;
    }
    public sealed class StatisticsPlotMark
    {
        public string Series, Label, Detail;
        public double X,Y,End;
        public int Row,Column;
        public StatisticsDrill Target;
    }
    public sealed class StatisticsPlot
    {
        public string Id, Title, Definition, XLabel, YLabel;
        public StatisticsPlotKind Kind;
        public bool Percent, WholeMatch;
        public readonly List<string> Rows=new List<string>(),Columns=new List<string>();
        public readonly List<StatisticsPlotSeries> Series=new List<StatisticsPlotSeries>();
        public readonly List<StatisticsPlotMark> Marks=new List<StatisticsPlotMark>();
        public readonly List<string> Context=new List<string>();
    }
    public sealed class StatisticsPageView
    {
        public string Context;
        public readonly List<string> Headlines=new List<string>();
        public readonly List<StatisticsPlot> Plots=new List<StatisticsPlot>();
        public readonly List<StatisticsSection> Ledgers=new List<StatisticsSection>();
    }
}
