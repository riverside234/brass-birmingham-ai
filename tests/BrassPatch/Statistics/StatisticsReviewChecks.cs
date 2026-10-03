using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Players;
using BrassPatch.Statistics;

public static class StatisticsReviewChecks
{
    static T Value<T>(params object[] values) where T:new()
    {var item=new T();for(int i=0;i<values.Length;i+=2)typeof(T).GetProperty((string)values[i],BindingFlags.Instance|BindingFlags.Public|BindingFlags.NonPublic).SetValue(item,values[i+1],null);return item;}
    static void Add<T>(CompletedMatchStatistics report,string list,params T[] values)
    {((List<T>)typeof(CompletedMatchStatistics).GetField(list,BindingFlags.Instance|BindingFlags.NonPublic).GetValue(report)).AddRange(values);}
    public static void Run(Action<bool,string,object> check)
    {
        var r=Value<CompletedMatchStatistics>("CompleteHistory",true,"Availability","Independent hand-built fixture");
        Add(r,"players",Value<StatisticsPlayer>("Color",PlayerColor.Blue,"Name","Alex","Opponent","Human","OpeningVP",2,"FinalVP",23,"Rank",1,"FinalCash",35),Value<StatisticsPlayer>("Color",PlayerColor.Red,"Name","Alex","Opponent","AI specialist","FinalVP",21,"Rank",2,"FinalCash",15));
        Add(r,"entries",Score(PlayerColor.Blue,1,"Industries",10),Score(PlayerColor.Blue,1,"Links",4),Score(PlayerColor.Blue,1,"Merchant",2),Score(PlayerColor.Blue,1,"Penalty",-3),Score(PlayerColor.Blue,2,"Industries",8),Score(PlayerColor.Red,1,"Industries",7),Score(PlayerColor.Red,1,"Links",5),Score(PlayerColor.Red,2,"Industries",9));
        var cotton=Value<StatisticsAsset>("Key","cotton","Player",PlayerColor.Blue,"City","Birmingham","RegionColor","Yellow","Type","CottonMill","Sellable",true,"Level",2,"Revision",5,"Era",1,"OwnTurn",1,"Turn",1,"FlipRevision",10,"FlipEra",1,"FlipOwnTurn",2,"FlipTurn",4,"BaseCost",12,"MarketCost",3,"CanalVP",10,"RailVP",8,"Carried",true,"PrintedVP",10);
        var wiped=Value<StatisticsAsset>("Key","wiped","Player",PlayerColor.Blue,"City","Birmingham","RegionColor","Yellow","Type","Manufacturer","Sellable",true,"Level",1,"Revision",8,"Era",1,"OwnTurn",1,"Turn",1,"RemovedRevision",51,"RemovedEra",1,"Removal","Canal cleanup","BaseCost",15,"MarketCost",5,"PrintedVP",6);
        var unfinished=Value<StatisticsAsset>("Key","unfinished","Player",PlayerColor.Blue,"City","Birmingham","RegionColor","Yellow","Type","Pottery","Sellable",true,"Level",3,"Revision",70,"Era",2,"OwnTurn",5,"Turn",14,"BaseCost",20,"MarketCost",4,"PrintedVP",10);
        var coal=Value<StatisticsAsset>("Key","coal","Player",PlayerColor.Blue,"City","Birmingham","RegionColor","Yellow","Type","CoalMine","Revision",6,"Era",1,"OwnTurn",1,"Turn",1,"BaseCost",5,"Produced",5,"Supplied",5,"FlipRevision",15,"FlipOwnTurn",2,"FlipTurn",5,"FlipEra",1,"RemovedRevision",45,"Removal","Overbuilt","FlipTrigger","Opponent consumption","FlipActor",(PlayerColor?)PlayerColor.Red);
        Add(r,"assets",cotton,wiped,unfinished,coal);
        Add(r,"boundaries",Value<StatisticsBoundary>("Name","Scoring","Revision",50,"Era",1,"Player",PlayerColor.Blue,"OwnTurn",4),Value<StatisticsBoundary>("Name","After cleanup","Revision",52,"Era",1,"Player",PlayerColor.Blue,"OwnTurn",4),Value<StatisticsBoundary>("Name","Rail entry","Revision",60,"Era",2,"Player",PlayerColor.Blue,"Cash",10,"Income",3),Value<StatisticsBoundary>("Name","Final","Revision",100,"Era",2,"Player",PlayerColor.Blue,"OwnTurn",7));
        var loan=Value<StatisticsAction>("Key","loan","Player",PlayerColor.Blue,"Kind","Loan","Revision",20,"EndRevision",24,"Era",1,"OwnTurn",2,"Borrowed",30,"IncomeDrop",3);
        Add(r,"actions",loan,Value<StatisticsAction>("Key","sell2","Player",PlayerColor.Blue,"Kind","Sell","Era",1,"Goods",2),Value<StatisticsAction>("Key","sell1","Player",PlayerColor.Blue,"Kind","Sell","Era",1,"Goods",1),Value<StatisticsAction>("Key","scout","Player",PlayerColor.Blue,"Kind","Scout","Era",1),Value<StatisticsAction>("Key","develop","Player",PlayerColor.Blue,"Kind","Develop","Era",1,"Developed",2),Value<StatisticsAction>("Key","network","Player",PlayerColor.Blue,"Kind","Network","Era",2,"Links",2,"Spending",18,"MarketCost",3),Value<StatisticsAction>("Key","build","Player",PlayerColor.Blue,"Kind","Build","Era",2,"Spending",15,"MarketCost",3));
        Add(r,"samples",Value<StatisticsSample>("Player",PlayerColor.Blue,"Revision",19,"Income",10,"OwnTurn",1),Value<StatisticsSample>("Player",PlayerColor.Blue,"Revision",24,"Income",7,"OwnTurn",2),Value<StatisticsSample>("Player",PlayerColor.Blue,"Revision",30,"Income",8,"OwnTurn",3),Value<StatisticsSample>("Player",PlayerColor.Blue,"Revision",40,"Income",10,"OwnTurn",4));
        Add(r,"entries",Use(PlayerColor.Blue,PlayerColor.Blue,2,11,"coal"),Use(PlayerColor.Red,PlayerColor.Blue,3,12,"coal"),Use(PlayerColor.Blue,PlayerColor.Red,1,13,"othercoal"),Use(PlayerColor.Blue,null,2,14,null,"Market"),Value<StatisticsEntry>("Kind","Market purchase","Resource","Coal","Player",PlayerColor.Blue,"Quantity",2,"TotalPrice",5,"Era",1,"Revision",14));
        var f=new StatisticsFilters();f.Reset(r);f.Players.Clear();f.Players.Add(PlayerColor.Blue);f.Resource="Coal";
        var allKinds=new HashSet<StatisticsPlotKind>();
        for(int page=0;page<10;page++){
            var v=StatisticsPlotPages.Build(page,r,f);check(v.Plots.Count>0,"hand-built page provides a plot",page);
            foreach(var p in v.Plots){allKinds.Add(p.Kind);check(p.Marks.All(m=>!Double.IsNaN(m.X)&&!Double.IsNaN(m.Y)&&!Double.IsInfinity(m.Y)),"finite exact plot values",p.Id);check(p.Series.Select(s=>s.Id).Distinct().Count()==p.Series.Count,"stable nonduplicate legend IDs",p.Id);check(p.Marks.All(m=>p.Series.Any(s=>s.Id==m.Series)),"every mark has a named legend",p.Id);}
        }
        check(allKinds.Count>=11,"varied plot families implemented",allKinds.Select(k=>k.ToString()).ToArray());
        var gap=StatisticsPlotPages.Build(1,r,f).Plots.First(p=>p.Id=="gap");check(gap.Marks.Where(m=>m.Series!="total").Sum(m=>m.Y)==2&&gap.Marks.Last().Y==2,"signed gap including opening and penalties reconciles to 23 − 21",null);
        f.Era=1;check(StatisticsPlotPages.Build(1,r,f).Plots.First(p=>p.Id=="gap").Marks.Last().Y==2,"summary gap stays whole match under era filter",null);f.Era=0;
        var conversion=StatisticsPlotPages.Build(2,r,f).Plots.First(p=>p.Id=="conversion");check(conversion.Marks.Where(m=>m.Series=="built").Sum(m=>m.Y)==4&&conversion.Marks.Where(m=>m.Series=="flipped").Sum(m=>m.Y)==2,"built/flipped counts are distinct tiles",null);
        var delays=StatisticsPlotPages.Build(2,r,f).Plots.First(p=>p.Id=="flip-delay");check(delays.Marks.Count==2&&delays.Marks.All(m=>m.Y==1),"own-turn delays differ from 3/4 global-turn delays",null);
        check(StatisticsDetailMetrics.Median(new[]{1,2,4,9})==3&&Double.IsNaN(StatisticsDetailMetrics.Median(new int[0])),"independent even median and no-sample result",null);
        var recovery=StatisticsMetrics.Loans(r,PlayerColor.Blue).Single();check(recovery.Baseline==10&&recovery.After==7&&recovery.RecoveredOwnTurn==4,"loan recovery uses pre-loan level, not marker or cash",null);
        check(StatisticsMetrics.At(wiped,r.Boundaries[0])&&!StatisticsMetrics.At(wiped,r.Boundaries[1]),"Canal scoring and after-cleanup occupancy differ",null);
        check(StatisticsMetrics.At(cotton,r.Boundaries.Last())&&!StatisticsMetrics.At(unfinished,r.Boundaries[0]),"historical birth/removal revisions control boundary status",null);
        check(StatisticsMetrics.Flow(r,f).Sum(e=>e.Quantity)==5,"incoming matrix includes own 2, opponents 1, market 2",null);
        var flow=StatisticsPlotPages.Build(5,r,f).Plots.First(p=>p.Id=="resource-flow");check(flow.Marks.Sum(m=>m.Y)==5&&flow.Columns.Count==1&&flow.Rows.Count==4,"incoming counterparts remain context with duplicate player names",null);
        var sourceMix=StatisticsPlotPages.Build(5,r,f).Plots.First(p=>p.Id=="sources");check(sourceMix.Marks.Single(m=>m.Series=="Own").Y==2&&sourceMix.Marks.Single(m=>m.Series=="Opponents").Y==1&&sourceMix.Marks.Single(m=>m.Series=="Market").Y==2,"incoming source bars partition the same five matrix units",null);
        f.Outgoing=true;flow=StatisticsPlotPages.Build(5,r,f).Plots.First(p=>p.Id=="resource-flow");check(flow.Marks.Sum(m=>m.Y)==5&&flow.Rows.Count==1&&flow.Columns.Count==2,"outgoing matrix includes selected supplier and both consumers",null);
        var outgoing=StatisticsPlotPages.Build(5,r,f);sourceMix=outgoing.Plots.First(p=>p.Id=="sources");check(sourceMix.Marks.Single(m=>m.Series=="Own").Y==2&&sourceMix.Marks.Single(m=>m.Series=="Opponents").Y==3&&sourceMix.Series.Count==2&&sourceMix.Marks.All(m=>m.Target.Outgoing==true),"outgoing source bars and evidence retain both consumers",null);
        check(outgoing.Headlines.Any(h=>h.StartsWith("3 units used by opponents"))&&outgoing.Ledgers.Single(s=>s.Title=="Resource supply records").Rows.Count==2,"outgoing headlines and source ledger match the exact flow",null);
        f.Players.Clear();f.Players.Add(PlayerColor.Red);outgoing=StatisticsPlotPages.Build(5,r,f);check(outgoing.Plots.Where(p=>p.Id=="sources"||p.Id=="resource-flow").All(p=>p.Marks.Sum(m=>m.Y)==1)&&outgoing.Plots.First(p=>p.Id=="sources").Marks.All(m=>m.Target.Player==PlayerColor.Red),"selected supplier uses one outgoing unit rather than three incoming units",null);f.Players.Clear();f.Players.Add(PlayerColor.Blue);
        f.Industry="CoalMine";f.Region="Yellow";check(StatisticsMetrics.Flow(r,f).Sum(e=>e.Quantity)==5,"source industry and region filters retain opponent consumers",null);
        var ownUse=r.Entries.First(e=>e.Kind=="Resource use"&&e.Revision==11);typeof(StatisticsEntry).GetProperty("ActionKey").SetValue(ownUse,"network",null);f.Action="Network";check(StatisticsMetrics.Flow(r,f).Sum(e=>e.Quantity)==2&&StatisticsPlotPages.Build(5,r,f).Plots.All(p=>p.Marks.Sum(m=>m.Y)==2),"consuming action filter applies to both resource plots",null);f.Action="";
        typeof(StatisticsEntry).GetProperty("Era").SetValue(ownUse,2,null);f.Era=2;check(StatisticsMetrics.Flow(r,f).Sum(e=>e.Quantity)==2,"Rail resource scope does not exclude a source built in Canal",null);typeof(StatisticsEntry).GetProperty("Era").SetValue(ownUse,1,null);f.Era=0;f.Industry=f.Region="";
        f.Evidence.Add("entry-12-Red");check(StatisticsMetrics.Flow(r,f).Sum(e=>e.Quantity)==3,"resource evidence uses stable event IDs",null);f.Evidence.Clear();f.Outgoing=false;
        f.EvidenceRestricted=true;check(StatisticsMetrics.Flow(r,f).Count()==0&&StatisticsMetrics.Assets(r,f).Count()==0,"explicit empty evidence stays empty instead of becoming the whole match",null);f.EvidenceRestricted=false;
        f.Evidence.Add("cotton");check(StatisticsMetrics.SupportingRecords(r,f).Rows.Any(row=>row.Key=="cotton"),"supporting-record ledger reaches the exact asset identity",null);f.Evidence.Clear();
        var prices=StatisticsMetrics.AdditionalLedgers(r,f,5).First(s=>s.Title.StartsWith("Recorded market")).Rows.Single();check(prices.Cells[5]=="2.50"&&prices.Cells[4]=="£5","typed £5 / 2 units gives £2.50 without parsing prose",null);
        var histogram=StatisticsPlotPages.Build(6,r,f).Plots.First(p=>p.Id=="sale-batches");check(histogram.Marks.Where(m=>m.Row==0).Sum(m=>m.Y)==1&&histogram.Marks.Where(m=>m.Row==1).Sum(m=>m.Y)==1,"single- and multi-tile sales occupy distinct bins",null);
        var share=StatisticsPlotPages.Build(6,r,f).Plots.First(p=>p.Id=="actions");check(share.Marks.Sum(m=>m.Y)==7,"double rail, Scout trades and Develop tiles are not extra actions",null);
        f.Industry="CottonMill";check(StatisticsMetrics.Assets(r,f).Count()==1&&StatisticsPlotPages.Build(2,r,f).Plots.First(p=>p.Id=="conversion").Rows.Count==1,"typed industry scope applies to counts and evidence",null);f.Industry="";
        check(StatisticsMetrics.Focus(r,f).Color==PlayerColor.Blue&&StatisticsMetrics.Reference(r,f).Color==PlayerColor.Red,"default sole human and runner-up reference identities",null);
        typeof(StatisticsPlayer).GetProperty("FinalVP").SetValue(r.Players[1],23,null);gap=StatisticsPlotPages.Build(1,r,f).Plots.First(p=>p.Id=="gap");check(gap.Marks.Last().Y==0,"VP tie remains zero despite ranked tie-break",null);
        typeof(CompletedMatchStatistics).GetProperty("Introductory").SetValue(r,true,null);check(StatisticsPlotPages.Build(0,r,f).Plots.All(p=>p.Id!="era-comparison"),"introductory overview suppresses a nonexistent Rail comparison",null);
        typeof(CompletedMatchStatistics).GetProperty("CompleteHistory").SetValue(r,false,null);check(StatisticsPlotPages.Build(2,r,f).Plots.Count==0,"missing history suppresses detailed charts and unsupported advice",null);
    }
    static StatisticsEntry Score(PlayerColor p,int era,string category,int vp){return Value<StatisticsEntry>("Player",p,"Era",era,"Kind","Balance","Category",category,"VP",vp);}
    static StatisticsEntry Use(PlayerColor consumer,PlayerColor? supplier,int amount,int revision,string asset,string source="Industry"){return Value<StatisticsEntry>("Player",consumer,"Supplier",supplier,"Kind","Resource use","Resource","Coal","Quantity",amount,"Revision",revision,"Era",1,"AssetKey",asset,"Source",source);}
    public static void Projection(CompletedMatchStatistics r,IEvent[] events,Action<bool,string,object> check)
    {
        foreach(var p in r.Players){
            check(r.Entries.Count(e=>e.Player==p.Color&&e.Kind=="Payday")==events.Count(e=>e.GetType().Name=="IncomeMoneyPaid"),"each player gets every actual payday, including zero payments",p.Color);
            foreach(var a in r.Actions.Where(a=>a.Player==p.Color)){
                int expected=events.Where(e=>e.Metadata.Revision>=a.Revision&&e.Metadata.Revision<=a.EndRevision).Sum(e=>e.GetType().Name=="MoneySpent"?Number(e,"Amount"):e.GetType().Name=="CoalBought"||e.GetType().Name=="IronBought"?Number(e,"Price"):0);
                check(a.Spending==expected,"action spending independently includes base and market payments once",new{a.Key,a.Spending,expected});
            }
            check(r.Boundaries.Any(b=>b.Player==p.Color&&b.Name=="Final"),"final boundary captured for every player",p.Color);
        }
        check(r.Contributions.Sum(c=>c.VP)==r.Links.Sum(l=>l.VP),"typed three-region link contributions reconcile independently",null);
        check(r.Assets.All(a=>a.OwnTurn>0)&&r.Actions.All(a=>a.OwnTurn>0),"explicit own-turn ordinals are populated",null);
        foreach(var m in r.MatTiles)check(m.Type!="None"&&m.Level>0,"mat snapshots contain actual industry tiles",m.Type);
        var f=new StatisticsFilters();f.Reset(r);foreach(var p in r.Players){f.Focus=p.Color;var gap=StatisticsPlotPages.Build(1,r,f).Plots.First(x=>x.Id=="gap");check(gap.Marks.Where(m=>m.Series!="total").Sum(m=>m.Y)==gap.Marks.Last().Y,"complete-match gap categories reconcile for each reference",p.Color);}
    }
    static int Number(object e,string property){return Convert.ToInt32(e.GetType().GetProperty(property).GetValue(e,null));}
}
