using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using BoardGameRules.Entities.Players;
using BrassPatch.Statistics;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public static class StatisticsLightChecks
{
    static T V<T>(params object[] properties)where T:new(){var result=new T();for(int i=0;i<properties.Length;i+=2)typeof(T).GetProperty((string)properties[i]).SetValue(result,properties[i+1],null);return result;}
    static void Add<T>(CompletedMatchStatistics r,string field,params T[] values){((List<T>)typeof(CompletedMatchStatistics).GetField(field,BindingFlags.NonPublic|BindingFlags.Instance).GetValue(r)).AddRange(values);}
    static StatisticsEntry E(string kind,int rev,string category="",string resource="",int quantity=0,string action=null,string asset=null,int cash=0,int vp=0){return V<StatisticsEntry>("Kind",kind,"Revision",rev,"Player",PlayerColor.Blue,"Era",1,"Category",category,"Resource",resource,"Quantity",quantity,"ActionKey",action,"AssetKey",asset,"Cash",cash,"VP",vp,"Label",resource,"Detail","Independent record");}
    public static void Run(Action<bool,string,object> check)
    {
        var r=V<CompletedMatchStatistics>("CompleteHistory",true,"Availability","Independent light-theme fixture");
        Add(r,"players",V<StatisticsPlayer>("Color",PlayerColor.Blue,"Name","Blue","FinalVP",20),V<StatisticsPlayer>("Color",PlayerColor.Red,"Name","Red","FinalVP",0));
        Add(r,"actions",V<StatisticsAction>("Key","build","Kind","Build","Era",1,"Player",PlayerColor.Blue,"Spending",12),V<StatisticsAction>("Key","network","Kind","Network","Era",1,"Player",PlayerColor.Blue),V<StatisticsAction>("Key","sell","Kind","Sell","Era",1,"Player",PlayerColor.Blue));
        Add(r,"entries",E("Balance",1,cash:-12),E("Balance",2,"Penalty",cash:-8,vp:-3),E("Balance",3,"Industries",vp:23));
        Add(r,"assets",V<StatisticsAsset>("Key","carry","Player",PlayerColor.Blue,"Era",1,"Carried",true,"Type","CoalMine","RegionColor","Yellow","City","Birmingham"));
        var f=new StatisticsFilters();f.Reset(r);f.Players.Remove(PlayerColor.Red);f.Era=2;
        check(StatisticsPlotPages.Build(8,r,f).Headlines.Any(x=>x.StartsWith("1 tiles carried")),"SD02 Rail carryover uses Rail-entry population, not Rail build cohort",null);
        f.Era=0;f.Action="Sell";var t=StatisticsChartQueries.Efficiency(r,f).Single();check(t.Actions==3&&t.VP==20&&t.Spending==12&&Math.Abs(t.VPPerAction-20.0/3)<1e-9&&Math.Abs(t.CostPerVP-.6)<1e-9,"SD03 detail actions do not narrow efficiency denominators",null);
        var view=StatisticsPlotPages.Build(8,r,f);check(view.Plots.First().Id=="player-efficiency"&&view.Ledgers.First(x=>x.Title=="Conversion and efficiency").Rows.Single().Cells[1]=="6.67","N7 is first and agrees with the shared ledger",null);
        f.AllCashPayments=true;t=StatisticsChartQueries.Efficiency(r,f).Single();check(t.Spending==20&&t.CostPerVP==1,"N7 cash basis includes negative payments once and excludes receipts",null);
        f.Era=2;t=StatisticsChartQueries.Efficiency(r,f).Single();check(Double.IsNaN(t.VPPerAction)&&Double.IsNaN(t.CostPerVP),"N7 missing action and nonpositive VP denominators stay unavailable",null);
        f.Era=1;f.Action="";f.AllCashPayments=false;
        Add(r,"entries",E("Development",10,"Develop action","Pottery",3,"develop"),E("Development",11,"Develop action","Pottery",1,"develop"),E("Development",12,"Merchant","Pottery",2,"sell"));
        Add(r,"actions",V<StatisticsAction>("Key","develop","Kind","Develop","Player",PlayerColor.Blue,"Era",1));
        var development=StatisticsPlotPages.Build(2,r,f).Plots.First(p=>p.Id=="development");check(development.Kind==StatisticsPlotKind.StackedColumns&&development.Marks.Sum(m=>m.Y)==3,"N1 levels 3,1,2 count three tiles, not six",null);f.DevelopmentSource="Merchant";check(StatisticsPlotPages.Build(2,r,f).Plots.First(p=>p.Id=="development").Marks.Sum(m=>m.Y)==1,"N1 merchant development counts one tile without another action",null);f.DevelopmentSource="All";
        var acquireL=E("Wildcard acquired",20,"Wild Location");typeof(StatisticsEntry).GetProperty("Label").SetValue(acquireL,"location-id",null);typeof(StatisticsEntry).GetProperty("AssetKey").SetValue(acquireL,"wild@Blue/20/location-id",null);
        var acquireI=E("Wildcard acquired",20,"Wild Industry");typeof(StatisticsEntry).GetProperty("Label").SetValue(acquireI,"industry-id",null);typeof(StatisticsEntry).GetProperty("AssetKey").SetValue(acquireI,"wild@Blue/20/industry-id",null);
        var useL=E("Card",21,"Build","Wild Location",1,"build",acquireL.AssetKey);var reacquire=E("Wildcard acquired",24,"Wild Location",asset:"wild@Blue/24/location-id");
        Add(r,"entries",acquireL,acquireI,useL,reacquire);
        var episodes=StatisticsDetailMetrics.Build(r,f,6).Single(s=>s.Title.StartsWith("Wildcard acquisition")).Rows;
        check(episodes.Count==3&&episodes.Select(e=>e.Key).Distinct().Count()==3&&episodes[0].Cells[5]=="No"&&episodes[1].Cells[5]=="Yes"&&episodes[2].Cells[5]=="Yes","SD01 same-Scout wildcard types and reacquisition have distinct use episodes",null);
        check(episodes.Select(e=>e.Key).Distinct().Count()==3&&new[]{acquireL,acquireI,reacquire}.Select(StatisticsChartQueries.RecordRowKey).Distinct().Count()==3,"SD01 supporting and timing rows preserve acquisition episode identities",null);
        Add(r,"entries",E("Card",25,"Non-Build payment","Industry",1,"sell"),E("Card",26,"Scout trade","Location",1,"scout"));
        check(StatisticsPlotPages.Build(6,r,f).Plots.First(p=>p.Id=="card-purpose").Marks.Sum(m=>m.Y)==3,"N4 payment, authorization and Scout extras partition played cards",null);
        var purchase1=E("Market purchase",30,resource:"Coal",quantity:1,action:"build");typeof(StatisticsEntry).GetProperty("TotalPrice").SetValue(purchase1,3,null);
        var purchase2=E("Market purchase",31,resource:"Coal",quantity:3,action:"build");typeof(StatisticsEntry).GetProperty("TotalPrice").SetValue(purchase2,3,null);
        var free=E("Resource use",32,"Network","Coal",2,"network","carry");typeof(StatisticsEntry).GetProperty("Supplier").SetValue(free,(PlayerColor?)PlayerColor.Blue,null);
        var paid=E("Resource use",33,"Build","Coal",4,"build");typeof(StatisticsEntry).GetProperty("Source").SetValue(paid,"Market",null);
        Add(r,"entries",purchase1,purchase2,free,paid);f.Resource="Coal";f.ResourceView="Market";
        var market=StatisticsPlotPages.Build(5,r,f);check(market.Plots.Single(p=>p.Id=="market-coal").Marks.Single().Y==1.5&&market.Plots.Single(p=>p.Id=="market-iron").Marks.Single().Missing,"N3 weighted six pounds / four units; no iron purchases is missing",null);
        f.ResourceView="Supply";f.Action="Network";var supply=StatisticsPlotPages.Build(5,r,f);var dependence=supply.Ledgers.Single(s=>s.Title=="Market dependence and paid unit prices").Rows.Single();check(supply.Plots.All(p=>p.Marks.Sum(m=>m.Y)==2)&&dependence.Cells[2]=="0 / 2"&&dependence.Cells[4]=="£0","SD04 resource charts and price/dependence ledger agree under action filters",null);f.Action="";
        Add(r,"links",V<StatisticsLink>("Key","A","Player",PlayerColor.Blue,"Era",1,"VP",4,"OwnVP",1,"OthersVP",2,"MerchantVP",1,"Scored",true,"Route","A"),V<StatisticsLink>("Key","B","Player",PlayerColor.Blue,"Era",1,"VP",10,"Scored",true,"Route","B"));
        Add(r,"contributions",V<StatisticsLinkContribution>("LinkKey","A","Owner",PlayerColor.Blue,"Era",1,"City","Birmingham","RegionColor","Yellow","VP",4),V<StatisticsLinkContribution>("LinkKey","B","Owner",PlayerColor.Blue,"Era",1,"City","Walsall","RegionColor","Yellow","VP",10));
        Add(r,"locations",V<StatisticsLocation>("City","Birmingham","RegionColor","Yellow","Neighbors",new string[0]),V<StatisticsLocation>("City","Walsall","RegionColor","Yellow","Neighbors",new string[0]));
        Add(r,"boundaries",V<StatisticsBoundary>("Name","Scoring","Player",PlayerColor.Blue,"Era",1,"Revision",80),V<StatisticsBoundary>("Name","Final","Player",PlayerColor.Blue,"Era",2,"Revision",100));
        f.MapMode="Link contributions";f.EvidenceRestricted=true;f.Evidence.Add("A");var map=StatisticsPlotPages.Build(3,r,f).Plots.Single(p=>p.Id=="geography");var city=map.Marks.Single(m=>m.Label=="Birmingham");check(city.Row==4&&map.Marks.Single(m=>m.Label=="Walsall").Row==0,"SD07 link evidence excludes unrelated map contributions",null);check(city.Target.Page==3&&city.Target.Keys.SetEquals(new[]{"A"}),"SD06 link map drills to actual contributing link records",null);
        check(StatisticsPlotSemantics.Value(map,city)=="4 VP","SD05 map tooltips use quantity, never normalized Y",null);
        check(StatisticsPlotPages.Build(3,r,f).Plots.Single(p=>p.Id=="link-sources").Marks.Sum(m=>m.Y)==4,"N2 own/others/merchant link VP partition",null);
        f.Evidence.Clear();f.EvidenceRestricted=false;f.ResourceView="Merchants";
        var sale=E("Goods sale",40,"CottonMill",quantity:1,action:"sell",asset:"carry");typeof(StatisticsEntry).GetProperty("Destination").SetValue(sale,"Unattributed",null);Add(r,"entries",sale);
        check(StatisticsPlotPages.Build(5,r,f).Plots.Single().Marks.Where(m=>m.Label=="Unattributed").Sum(m=>m.Y)==1,"N5 unattributed sale stays in its own destination",null);
        var round=V<StatisticsRound>("Era",1,"Round",1);((List<PlayerColor>)typeof(StatisticsRound).GetField("order",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(round)).Add(PlayerColor.Blue);Add(r,"rounds",round);f.Players.Add(PlayerColor.Red);
        var seats=StatisticsPlotPages.Build(7,r,f).Plots.Single(p=>p.Id=="position-share");check(seats.Series.Count==2&&seats.Marks.Where(m=>m.Label=="Blue").Sum(m=>m.Y)==1&&seats.Marks.Where(m=>m.Label=="Red").Sum(m=>m.Y)==0,"N6 nonexistent seats omitted; missing player not assigned 0th/first",null);
        var secondRound=V<StatisticsRound>("Era",1,"Round",2);((List<PlayerColor>)typeof(StatisticsRound).GetField("order",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(secondRound)).AddRange(new[]{PlayerColor.Red,PlayerColor.Blue});Add(r,"rounds",secondRound);
        f.EvidenceRestricted=true;f.Evidence.Add(StatisticsChartQueries.RoundKey(round));var selectedRound=StatisticsPlotPages.Build(7,r,f);var seatTable=StatisticsPages.Build(7,r,f).Single(s=>s.Title=="Round positions");
        check(selectedRound.Plots.Single(p=>p.Id=="position-share").Marks.Sum(m=>m.Y)==1&&seatTable.Rows.Sum(row=>Int32.Parse(row.Cells[2]))==1&&selectedRound.Plots.Single(p=>p.Id=="seats").Columns.Count==1&&StatisticsMetrics.SupportingRecords(r,f).Rows.Count==1,"N6 selected round shares, heatmap, table and supporting evidence reconcile",null);f.EvidenceRestricted=false;f.Evidence.Clear();
        var noOrder=V<CompletedMatchStatistics>("CompleteHistory",true);Add(noOrder,"players",V<StatisticsPlayer>("Color",PlayerColor.Blue,"Name","Blue"));Add(noOrder,"rounds",V<StatisticsRound>("Era",1,"Round",1),V<StatisticsRound>("Era",1,"Round",2));var noOrderFilters=new StatisticsFilters();noOrderFilters.Reset(noOrder);
        check(StatisticsPlotPages.Build(7,noOrder,noOrderFilters).Ledgers.Single(s=>s.Title=="Round position evidence").Rows.All(row=>row.Cells[2]=="Unavailable"&&row.Cells[4]=="No"),"N6 empty orders never fabricate a Blue back-to-back turn",null);
        f.ActionShare=false;var actionCounts=StatisticsPlotPages.Build(6,r,f).Plots.Single(p=>p.Id=="actions");check(!actionCounts.Percent&&actionCounts.Marks.Sum(m=>m.Y)==r.Actions.Count,"action count/share toggle retains the complete action denominator",null);f.ActionShare=true;
        LoanChecks(check);
        var percent=new StatisticsPlot{Kind=StatisticsPlotKind.StackedBars,Percent=true};percent.Marks.Add(new StatisticsPlotMark{Y=1});percent.Marks.Add(new StatisticsPlotMark{Y=3});check(StatisticsPlotSemantics.Value(percent,percent.Marks[0]).Contains("25.0%"),"SD05 percentage description states raw count and share",StatisticsPlotSemantics.Value(percent,percent.Marks[0]));
        var interval=new StatisticsPlot{Kind=StatisticsPlotKind.Lifespans};check(StatisticsPlotSemantics.Value(interval,new StatisticsPlotMark{X=2,End=8,Y=6}).Contains("duration 6 global turns"),"SD05 interval units include start/end/duration",null);
        var axis=StatisticsAxisLayout.Create(0,1,StatisticsAxisUnit.Count,950,s=>s.Length*15);check(axis.Ticks.All(x=>x.Value==Math.Round(x.Value))&&axis.Ticks.Select(x=>x.Label).Distinct().Count()==axis.Ticks.Count,"SD11 small count range uses unique integer ticks",null);
        foreach(var range in new[]{new[]{-13d,77},new[]{0d,.91},new[]{0d,146},new[]{0d,0},new[]{-.1d,.1},new[]{0d,123456}})foreach(var unit in new[]{StatisticsAxisUnit.Cash,StatisticsAxisUnit.Ratio,StatisticsAxisUnit.Count,StatisticsAxisUnit.Turn}){
            var a=StatisticsAxisLayout.Create(range[0],range[1],unit,950,s=>s.Length*15);check(a.Min<=range[0]&&a.Max>=range[1]&&a.Ticks.All(x=>!Double.IsNaN(x.Value))&&a.Ticks.Select(x=>x.Label).Distinct().Count()==a.Ticks.Count,"rounded axis covers signed/extreme/zero values with unique finite labels",new{range,unit});
        }
        check(StatisticsAxisLayout.Format(.25,StatisticsAxisUnit.Ratio,.25)=="0.25"&&StatisticsAxisLayout.Format(-.0001,StatisticsAxisUnit.Price,.25)=="£0.00","axis precision preserves quarter steps and avoids negative zero",null);
        check(StatisticsPages.Names.Length==10&&StatisticsPages.NormalizePage(10)==0&&StatisticsPages.NormalizePage(-1)==0&&StatisticsPages.NormalizePage(99)==0,"advice page retired and legacy/invalid targets normalize to Overview",null);
        Geometry(check);
    }
    static void Geometry(Action<bool,string,object> check)
    {
        var node=new GameObject("Independent statistics geometry",typeof(RectTransform));var g=node.AddComponent<StatisticsPlotGraphic>();g.Area=new Rect(0,0,900,300);g.RowHeight=50;g.XMin=0;g.XMax=1;g.YMin=0;g.YMax=1;
        var heat=new StatisticsPlot{Kind=StatisticsPlotKind.Heatmap};heat.Columns.AddRange(new[]{"A","B","C"});heat.Rows.Add("R");var cell=new StatisticsPlotMark{Row=0,Column=0,Y=1,Series="s"};heat.Marks.Add(cell);g.Plot=heat;check(g.Hits(new Vector2(10,275)).Contains(cell)&&g.Hits(new Vector2(288,260)).Contains(cell),"SD10 cell edges and corners use visible rectangle, not center radius",null);
        var columns=new StatisticsPlot{Kind=StatisticsPlotKind.Columns};columns.Columns.AddRange(new[]{"C1","C2"});columns.Series.Add(new StatisticsPlotSeries{Id="A"});columns.Series.Add(new StatisticsPlotSeries{Id="B"});g.Plot=columns;g.XMin=-.5;g.XMax=1.5;g.YMax=10;foreach(int category in new[]{0,1})foreach(string s in new[]{"A","B"}){var m=new StatisticsPlotMark{X=category,Y=5,Series=s};columns.Marks.Add(m);check(g.MarkBounds(m).xMin>=g.Area.xMin&&g.MarkBounds(m).xMax<=g.Area.xMax,"SD08 complete first/last grouped columns lie within category bands",null);}
        var scatter=new StatisticsPlot{Kind=StatisticsPlotKind.Scatter};g.Plot=scatter;g.XMin=g.YMin=0;g.XMax=g.YMax=10;for(int i=0;i<12;i++)scatter.Marks.Add(new StatisticsPlotMark{X=10,Y=10,Series="s",Label="overlap-"+i});check(scatter.Marks.All(m=>g.Area.Contains(g.Point(m)))&&g.Hits(g.Point(scatter.Marks[0])).Count==12&&scatter.Marks.All(m=>m.X==10&&m.Y==10),"SD09 bounded scatter cluster keeps all twelve exact records selectable",null);
        UnityEngine.Object.Destroy(node);
    }
    public static void Projection(CompletedMatchStatistics r,Action<bool,string,object> check)
    {
        var f=new StatisticsFilters();f.Reset(r);
        check(r.Entries.Select(StatisticsChartQueries.RecordRowKey).Distinct().Count()==r.Entries.Count,"SD01 collected supporting rows have distinct identities, including Scout extras",null);
        foreach(int era in r.Introductory?new[]{0,1}:new[]{0,1,2}){
            f.Era=era;var entries=r.Entries.Where(e=>era==0||e.Era==era).ToArray();
            var development=StatisticsPlotPages.Build(2,r,f).Plots.First(p=>p.Id=="development");check(development.Marks.Sum(m=>m.Y)==entries.Count(e=>e.Kind=="Development"),"collected N1 chart counts development records independent of tile levels",era);
            var sources=StatisticsPlotPages.Build(3,r,f).Plots.First(p=>p.Id=="link-sources");check(sources.Marks.Sum(m=>m.Y)==r.Links.Where(l=>l.Scored&&(era==0||l.Era==era)).Sum(l=>l.VP),"collected N2 link sources equal actual scored VP",era);
            var cards=StatisticsPlotPages.Build(6,r,f).Plots.First(p=>p.Id=="card-purpose");check(cards.Marks.Sum(m=>m.Y)==entries.Count(e=>e.Kind=="Card"),"collected N4 chart partitions every played card and excludes acquisitions",era);
            f.ResourceView="Merchants";var sales=StatisticsPlotPages.Build(5,r,f).Plots.Single();check(sales.Marks.Sum(m=>m.Y)==entries.Where(e=>e.Kind=="Goods sale").Sum(e=>e.Quantity),"collected N5 destinations reconcile to sold industry tiles",era);
            f.ResourceView="Market";var market=StatisticsPlotPages.Build(5,r,f);
            foreach(var player in r.Players)foreach(string resource in new[]{"Coal","Iron"}){var bought=entries.Where(e=>e.Player==player.Color&&e.Kind=="Market purchase"&&e.Resource==resource).ToArray();var mark=market.Plots.Single(p=>p.Id=="market-"+resource.ToLowerInvariant()).Marks.Single(m=>m.Series==player.Color.ToString());int quantity=bought.Sum(e=>e.Quantity);check(quantity==0?mark.Missing:Math.Abs(mark.Y-(double)bought.Sum(e=>e.TotalPrice)/quantity)<1e-9,"collected N3 uses independent weighted purchase totals",new{era,player=player.Color,resource});}
            f.ResourceView="Supply";
            foreach(bool all in new[]{false,true}){f.AllCashPayments=all;foreach(var t in StatisticsChartQueries.Efficiency(r,f)){var actions=r.Actions.Where(a=>a.Player==t.Player.Color&&(era==0||a.Era==era)).ToArray();int vp=era==0?t.Player.FinalVP:entries.Where(e=>e.Player==t.Player.Color&&e.Kind=="Balance").Sum(e=>e.VP);int spending=all?-entries.Where(e=>e.Player==t.Player.Color&&e.Kind=="Balance"&&e.Cash<0).Sum(e=>e.Cash):actions.Sum(a=>a.Spending);check(t.Actions==actions.Length&&t.Spending==spending&&t.VP==vp,"collected N7 action/score/cash totals reconcile in each era and cash basis",new{era,all,player=t.Player.Color});}}
        }
    }
    public static CompletedMatchStatistics Preview(bool partial,bool zero)
    {
        var r=V<CompletedMatchStatistics>("CompleteHistory",!partial,"Availability","History unavailable: independent partial-history preview");
        foreach(var color in new[]{PlayerColor.Blue,PlayerColor.Yellow}){
            Add(r,"players",V<StatisticsPlayer>("Color",color,"Name","Alexandra Birmingham Manufacturing Partnership","Rank",1,"FinalVP",zero?0:color==PlayerColor.Blue?21:23,"FinalCash",zero?0:color==PlayerColor.Blue?-13:77));
            if(!partial&&!zero){Add(r,"actions",V<StatisticsAction>("Key","pass-"+color,"Kind","Pass","Player",color,"Era",1,"Turn",1,"OwnTurn",1,"Spending",color==PlayerColor.Blue?12:20));
                Add(r,"entries",V<StatisticsEntry>("Kind","Balance","Category","Penalty","Player",color,"Era",1,"VP",-3,"Cash",-13,"Revision",1),V<StatisticsEntry>("Kind","Balance","Category","Industries","Player",color,"Era",2,"VP",color==PlayerColor.Blue?24:26,"Cash",77,"Revision",2));
                Add(r,"samples",V<StatisticsSample>("Player",color,"Era",1,"Turn",1,"Revision",1,"Cash",-13),V<StatisticsSample>("Player",color,"Era",2,"Turn",2,"Revision",2,"Cash",77));
            }
        }return r;
    }
    static void LoanChecks(Action<bool,string,object> check)
    {
        var r=V<CompletedMatchStatistics>("CompleteHistory",true);Add(r,"players",V<StatisticsPlayer>("Color",PlayerColor.Blue,"Name","Blue"));
        foreach(var item in new[]{new{Revision=10,End=11,Era=1,Own=2},new{Revision=15,End=16,Era=1,Own=3},new{Revision=40,End=41,Era=2,Own=8}})Add(r,"actions",V<StatisticsAction>("Kind","Loan","Player",PlayerColor.Blue,"Revision",item.Revision,"EndRevision",item.End,"Era",item.Era,"OwnTurn",item.Own));
        foreach(var item in new[]{new{Rev=0,Income=10,Own=1,Era=1},new{Rev=11,Income=7,Own=2,Era=1},new{Rev=14,Income=8,Own=3,Era=1},new{Rev=16,Income=5,Own=3,Era=1},new{Rev=30,Income=10,Own=6,Era=2},new{Rev=39,Income=11,Own=7,Era=2},new{Rev=41,Income=8,Own=8,Era=2}})Add(r,"samples",V<StatisticsSample>("Player",PlayerColor.Blue,"Revision",item.Rev,"Income",item.Income,"OwnTurn",item.Own,"Era",item.Era));
        var loans=StatisticsMetrics.Loans(r,PlayerColor.Blue);check(loans[0].RecoveredRevision==30&&loans[1].RecoveredRevision==30&&!loans[2].RecoveredRevision.HasValue,"loan episodes can overlap and recover in Rail; unrecovered episodes remain unavailable",null);
    }
}

public sealed partial class StatisticsDriver
{
    IEnumerator LightPanelChecks(StatisticsPanel panel,Camera camera)
    {
        Check(panel.GetComponent<StatisticsTheme>()!=null&&panel.PageButtons.Length==10,"Statistics opts in to isolated light palette and ten factual pages");
        panel.SelectPage(10);yield return null;yield return null;Check(panel.CurrentPage==0,"legacy page 10 request opens Overview safely");
        panel.Drill(new StatisticsDrill{Page=99});yield return null;yield return null;Check(panel.CurrentPage==0,"invalid typed drill normalizes before indexing page state");panel.ResetFilters();
        foreach(var resolution in resolutions){var target=new RenderTexture(resolution.x,resolution.y,24);target.Create();camera.targetTexture=target;yield return null;yield return null;
            while(panel.TextScale<1.14f){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Larger text").onClick.Invoke();yield return null;yield return null;}
            foreach(string localView in new[]{"Supply","Market","Merchants"})foreach(bool largest in new[]{false,true}){
                if(largest&&panel.TextScale<1.29f){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Larger text").onClick.Invoke();yield return null;yield return null;}
                panel.SelectPage(5);yield return null;yield return null;panel.GetComponentsInChildren<Button>().First(b=>b.name=="Resource view "+localView).onClick.Invoke();yield return null;yield return null;
                panel.BodyScroll.verticalNormalizedPosition=1;Canvas.ForceUpdateCanvases();Check(panel.Filters.ResourceView==localView,"Resources local view retains its own plot/cache scope",localView);Capture(target,camera,"statistics-light-"+resolution.x+"x"+resolution.y+"-resources-"+localView+(largest?"-largest":"-default"));
                foreach(var graphic in panel.GetComponentsInChildren<StatisticsPlotGraphic>())CheckAxisLabels(graphic,panel);
                if(largest){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Smaller text").onClick.Invoke();yield return null;yield return null;}
            }
            foreach(int era in new[]{0,1,2})foreach(bool all in new[]{false,true}){panel.ResetFilters();panel.Filters.Era=era;panel.Filters.AllCashPayments=all;panel.SelectPage(8);yield return null;yield return null;panel.BodyScroll.verticalNormalizedPosition=1;Canvas.ForceUpdateCanvases();var graphic=panel.GetComponentsInChildren<StatisticsPlotGraphic>().First();Check(graphic.Plot.Id=="player-efficiency-vp"&&panel.GetComponentsInChildren<Button>().Any(b=>b.name=="Cost denominator"),"N7 two explicit units and visible cost basis above the first plot",new{era,all});CheckAxisLabels(graphic,panel);Capture(target,camera,"statistics-light-"+resolution.x+"x"+resolution.y+"-efficiency-"+era+"-"+(all?"all":"actions"));}
            panel.ResetFilters();panel.SelectPage(5);yield return null;yield return null;
            var heat=panel.GetComponentsInChildren<StatisticsPlotGraphic>().First(g=>g.Plot.Kind==StatisticsPlotKind.Heatmap);var mark=heat.Plot.Marks.First(m=>m.Target!=null);var bounds=heat.MarkBounds(mark);var local=new Vector2(bounds.xMin+4,bounds.yMax-4);var point=RectTransformUtility.WorldToScreenPoint(camera,heat.transform.TransformPoint(local));var data=new PointerEventData(EventSystem.current){position=point,button=PointerEventData.InputButton.Left};var hits=new List<RaycastResult>();EventSystem.current.RaycastAll(data,hits);var hit=hits.FirstOrDefault(h=>h.gameObject.GetComponent<StatisticsPlotGraphic>()==heat);
            data.pointerPressRaycast=hit;data.pointerCurrentRaycast=hit;Check(hit.gameObject!=null,"SD10 native pointer raycast reaches heatmap edge without a center proxy",resolution.ToString());heat.OnPointerClick(data);yield return null;yield return null;Check(panel.Filters.EvidenceRestricted&&panel.Filters.Evidence.SetEquals(mark.Target.Keys),"SD10 actual native heatmap edge click opens represented records",null);
            foreach(var shape in new[]{new{Page=6,Id="card-purpose"},new{Page=9,Id="lifespans"}}){var shapeCheck=NativeShapeClick(panel,camera,shape.Page,shape.Id,resolution);while(shapeCheck.MoveNext())yield return shapeCheck.Current;}
            camera.targetTexture=null;target.Release();Destroy(target);
        }
        panel.ResetFilters();panel.SelectPage(0);yield return null;yield return null;
    }
    IEnumerator NativeShapeClick(StatisticsPanel panel,Camera camera,int page,string id,Vector2Int resolution)
    {
        panel.ResetFilters();panel.SelectPage(page);yield return null;yield return null;
        var graphic=panel.GetComponentsInChildren<StatisticsPlotGraphic>().Single(g=>g.Plot.Id==id);
        var mark=graphic.Plot.Marks.Where(m=>m.Target!=null&&graphic.MarkBounds(m).width>40*graphic.Unit).OrderByDescending(m=>graphic.MarkBounds(m).width).First();
        var rect=graphic.MarkBounds(mark);var local=new Vector2(rect.xMin+Math.Min(8*graphic.Unit,rect.width/4),rect.center.y);Vector2 screen=Vector2.zero;bool visible=false;
        for(int step=0;step<=100;step++){panel.BodyScroll.verticalNormalizedPosition=1-step/100f;Canvas.ForceUpdateCanvases();screen=RectTransformUtility.WorldToScreenPoint(camera,graphic.transform.TransformPoint(local));var viewportLocal=panel.BodyScroll.viewport.InverseTransformPoint(graphic.transform.TransformPoint(local));var visibleRect=panel.BodyScroll.viewport.rect;visibleRect.yMin+=20;visibleRect.yMax-=20;if(visibleRect.Contains(viewportLocal)){visible=true;break;}}
        Check(visible&&panel.BodyScroll.verticalNormalizedPosition<.995f,"SD10 bar/interval test reaches a visible body edge after scrolling",new{id,resolution});
        var data=new PointerEventData(EventSystem.current){position=screen,button=PointerEventData.InputButton.Left};var hits=new List<RaycastResult>();EventSystem.current.RaycastAll(data,hits);var hit=hits.FirstOrDefault(h=>h.gameObject==graphic.gameObject);data.pointerPressRaycast=hit;data.pointerCurrentRaycast=hit;
        Check(hit.gameObject!=null&&graphic.Hits(local).Contains(mark),"SD10 native bar/interval raycast uses full displayed shape",id);graphic.OnPointerClick(data);yield return null;yield return null;
        Check(panel.Filters.EvidenceRestricted&&panel.Filters.Evidence.SetEquals(mark.Target.Keys),"SD10 native bar/interval body click opens represented records after scrolling",id);
    }
    IEnumerator LightSpecialPreviews(StatisticsPanel original,Camera camera,IEnumerable<CompletedMatchStatistics> reports)
    {
        var typeface=original.GetComponentsInChildren<TextMeshProUGUI>().First().font;
        foreach(var resolution in resolutions){var target=new RenderTexture(resolution.x,resolution.y,24);target.Create();camera.targetTexture=target;yield return null;yield return null;
            int index=0;foreach(var report in reports.Concat(new[]{StatisticsLightChecks.Preview(true,false),StatisticsLightChecks.Preview(false,true),StatisticsLightChecks.Preview(false,false)})){
                var root=new GameObject("Special statistics preview",typeof(RectTransform),typeof(Canvas),typeof(GraphicRaycaster),typeof(CanvasGroup));root.transform.SetParent(original.transform.parent,false);var rect=(RectTransform)root.transform;rect.anchorMin=Vector2.zero;rect.anchorMax=Vector2.one;rect.offsetMin=rect.offsetMax=Vector2.zero;var canvas=root.GetComponent<Canvas>();canvas.overrideSorting=true;canvas.sortingOrder=501;
                var preview=root.AddComponent<StatisticsPanel>();preview.Initialize(report,typeface,null);preview.CreateShowViewImmediatelyAnimation();yield return null;yield return null;
                foreach(int page in report.CompleteHistory?new[]{0,4,8}:new[]{0}){preview.SelectPage(page);yield return null;yield return null;Canvas.ForceUpdateCanvases();foreach(var graphic in preview.GetComponentsInChildren<StatisticsPlotGraphic>())CheckAxisLabels(graphic,preview);Check(preview.IsOpen&&preview.PageButtons.Length==10,"native small-game/introductory/zero/partial/long-name preview opens factual pages",new{index,page,report.CompleteHistory,report.Introductory,players=report.Players.Count,resolution});Capture(target,camera,"statistics-special-"+resolution.x+"x"+resolution.y+"-"+index+"-"+page);}
                root.SetActive(false);Destroy(root);yield return null;index++;
            }
            camera.targetTexture=null;target.Release();Destroy(target);
        }
        original.SelectPage(0);yield return null;yield return null;
    }
    void CheckAxisLabels(StatisticsPlotGraphic graphic,StatisticsPanel panel)
    {
        foreach(var axis in new[]{graphic.XAxis,graphic.YAxis}.Where(a=>a!=null)){Check(axis.Ticks.Select(t=>t.Label).Distinct().Count()==axis.Ticks.Count,"native axis labels remain unique after unit formatting",graphic.Plot.Id);if(axis.Unit==StatisticsAxisUnit.Count||axis.Unit==StatisticsAxisUnit.Turn||axis.Unit==StatisticsAxisUnit.Income)Check(axis.Ticks.All(t=>t.Value==Math.Round(t.Value)),"native count/time axes have integer ticks",graphic.Plot.Id);}
        var labels=graphic.transform.parent.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name.StartsWith("Value tick")).ToArray();Check(labels.All(t=>t.overflowMode!=TextOverflowModes.Ellipsis&&t.GetPreferredValues(t.text,0,0).x<=t.rectTransform.rect.width+1),"native tick gutters fit measured numbers without ellipsis",graphic.Plot.Id);
        Check(labels.All(t=>t.fontSize*panel.GetComponentInParent<Canvas>().scaleFactor>=23.5f),"major axis numbers meet 24px default target",graphic.Plot.Id);
        var root=graphic.transform;while(root.parent!=null&&root.name!="Plot "+graphic.Plot.Id)root=root.parent;
        var legends=root.GetComponentsInChildren<Button>().Where(b=>b.name.StartsWith("Legend ")).Select(b=>b.GetComponentInChildren<TextMeshProUGUI>()).ToArray();
        Check(legends.All(t=>t.GetPreferredValues(t.text,t.rectTransform.rect.width,0).y<=t.rectTransform.rect.height+1),"native legends fit wrapped long/duplicate names without vertical overflow",graphic.Plot.Id);
        Check(panel.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="Final VP").All(t=>t.overflowMode!=TextOverflowModes.Ellipsis&&t.GetPreferredValues(t.text,0,0).x<=t.rectTransform.rect.width+1),"native final VP remains readable beside long player names",graphic.Plot.Id);
        var endpoints=graphic.transform.parent.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="Curve endpoint").Select(t=>new Rect(t.rectTransform.anchoredPosition-new Vector2(0,t.rectTransform.rect.height),t.rectTransform.rect.size)).ToArray();
        Check(endpoints.SelectMany((a,i)=>endpoints.Skip(i+1).Select(b=>!a.Overlaps(b))).All(p=>p),"native curve endpoint values avoid overlap, including tied zero curves",graphic.Plot.Id);
    }
}
