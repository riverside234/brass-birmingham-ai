using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoardGameRules.Entities.Players;
using BrassPatch.History;
using BrassPatch.Statistics;
using DG.Tweening;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public static class StatisticsEfficiencyChecks
{
    static T V<T>(params object[] pairs) where T:new(){var value=new T();for(int i=0;i<pairs.Length;i+=2)typeof(T).GetProperty((string)pairs[i]).SetValue(value,pairs[i+1],null);return value;}
    static void Add<T>(CompletedMatchStatistics r,string name,params T[] values){((List<T>)typeof(CompletedMatchStatistics).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic).GetValue(r)).AddRange(values);}
    public static void Data(Action<bool,string,object> check)
    {
        var r=V<CompletedMatchStatistics>("GameId","efficiency-fixture","CompleteHistory",true,"Availability","Complete","FinalRevision",100);
        Add(r,"players",V<StatisticsPlayer>("Color",PlayerColor.Blue,"Name","Same name","Rank",1,"FinalVP",44),V<StatisticsPlayer>("Color",PlayerColor.Red,"Name","Same name","Rank",2,"FinalVP",20));
        Add(r,"assets",
            V<StatisticsAsset>("Key","carry","Type","CottonMill","City","Birmingham","RegionColor","Yellow","Player",PlayerColor.Blue,"Era",1,"OwnTurn",1,"Turn",2,"BaseCost",10,"MarketCost",2,"CanalVP",0,"RailVP",12,"CanalPrinted",12,"RailPrinted",12,"IntroVP",3,"Carried",true,"FlipEra",2,"FlipOwnTurn",5,"FlipTurn",16,"FlipRevision",80),
            V<StatisticsAsset>("Key","zero","Type","CottonMill","City","Worcester","RegionColor","Yellow","Player",PlayerColor.Blue,"Era",1,"OwnTurn",2,"BaseCost",8,"CanalPrinted",8,"Carried",true,"Sellable",true),
            V<StatisticsAsset>("Key","same-turn","Type","CottonMill","City","Walsall","RegionColor","Yellow","Player",PlayerColor.Blue,"Era",1,"OwnTurn",2,"Turn",5,"FlipOwnTurn",2,"FlipTurn",5,"FlipRevision",21,"FlipEra",1,"CanalVP",5,"CanalPrinted",5,"BaseCost",0),
            V<StatisticsAsset>("Key","removed","Type","CottonMill","City","Dudley","RegionColor","Yellow","Player",PlayerColor.Blue,"Era",1,"BaseCost",5,"RemovedRevision",60,"Removal","Canal cleanup"),
            V<StatisticsAsset>("Key","supplier","Type","CoalMine","City","Dudley","RegionColor","Red","Player",PlayerColor.Blue,"Era",2,"BaseCost",7,"MarketCost",4,"Supplied",3,"Stock",1,"MarketReceipts",20),
            V<StatisticsAsset>("Key","rail-level2","Type","Brewery","City","Dudley","Player",PlayerColor.Blue,"Era",2,"Level",2,"RailVP",9,"BaseCost",10));
        Add(r,"links",V<StatisticsLink>("Key","rail-a","Route","A–B","Player",PlayerColor.Blue,"Era",2,"Scored",true,"VP",8),V<StatisticsLink>("Key","rail-b","Route","B–C","Player",PlayerColor.Blue,"Era",2,"Scored",true,"VP",0),V<StatisticsLink>("Key","not-scored","Route","D–E","Player",PlayerColor.Blue,"Era",2,"VP",99));
        Add(r,"contributions",V<StatisticsLinkContribution>("LinkKey","rail-a","RegionColor","Yellow"),V<StatisticsLinkContribution>("LinkKey","rail-a","RegionColor","Yellow"),V<StatisticsLinkContribution>("LinkKey","rail-b","RegionColor","Yellow"));
        var f=new StatisticsFilters();f.Reset(r);f.Players.Clear();f.Players.Add(PlayerColor.Blue);f.Era=1;
        var total=StatisticsChartQueries.Efficiency(r,f).Single();check(total.IndustryVP==5&&total.IntroVP==3&&total.Printed==25,"E26-DATA introductory bonus is separate from ordinary realization",null);
        var returns=StatisticsPlotPages.Build(8,r,f);var ratio=returns.Plots.Single(p=>p.Id=="construction-return").Marks.Single(m=>m.Series=="CottonMill");
        check(Math.Abs(ratio.Y-20.0/25)<1e-9&&ratio.Target.Keys.SetEquals(new[]{"carry","zero","same-turn","removed"}),"E26-DATA construction ratio of sums includes zero VP, removed tiles, lifetime carryover and one cost",ratio.Y);
        check(returns.Plots.Single(p=>p.Id=="construction-return").Marks.Single(m=>m.Series=="Pottery").Missing,"E26-DATA zero denominator is missing, never infinity",null);
        f.Era=2;f.Industry="Pottery";f.Action="Loan";var realized=StatisticsPlotPages.Build(8,r,f).Plots.Single(p=>p.Id=="industry-realization").Marks.Single();check(realized.Y==21&&realized.End==12&&realized.Target.Keys.Contains("carry"),"E26-DATA Rail realization retains earlier builds and declares detail filters inapplicable; no percentage clamp",realized.Y);
        f.Industry=f.Action="";f.Era=1;f.EfficiencyView="Conversion";var converted=StatisticsPlotPages.Build(8,r,f);var heat=converted.Plots.Single(p=>p.Id=="completion-speed");var cell=heat.Marks.Single(m=>m.Column==0);
        check(cell.Y==2&&cell.OtherLabel=="n = 2"&&!heat.HeatTotals&&cell.Target.Keys.Count==4,"E26-DATA median includes later-era flip and real zero, excludes unfinished, no summed medians",cell.Y);
        check(converted.Ledgers.Single(s=>s.Title=="Completion intervals by tile").Rows.Count==4&&heat.Marks.Single(m=>m.Column==1).Missing,"E26-DATA completion table and heatmap share cohort, no-flip cell unavailable",null);
        f.EfficiencyView="Transition & networks";f.Era=0;var transitions=StatisticsPlotPages.Build(8,r,f);var spend=transitions.Plots.Single(p=>p.Id=="unflipped-investment");check(spend.Marks.Sum(m=>m.Y)==29&&spend.Marks.Where(m=>m.Row==4).Sum(m=>m.Y)==11,"E26-DATA unflipped investment excludes removed and flipped; receipts do not reduce construction cost",spend.Marks.Sum(m=>m.Y));
        var carry=transitions.Plots.Single(p=>p.Id=="carryover-scoring");check(carry.Marks.Sum(m=>m.Y)==12&&carry.Marks.All(m=>m.Target.Keys.SetEquals(new[]{"carry","zero"})),"E26-DATA only actual carried cohort, including zero; excludes Rail level 2",null);
        var link=transitions.Plots.Single(p=>p.Id=="link-return");check(link.Marks.Single(m=>m.Series=="Rail").Y==4&&link.Marks.Single(m=>m.Series=="Canal").Missing,"E26-DATA VP per scored link includes zero-VP, excludes unscored; missing era is not zero",null);
        f.Region="Yellow";link=StatisticsPlotPages.Build(8,r,f).Plots.Single(p=>p.Id=="link-return");check(link.Marks.Single(m=>m.Series=="Rail").Y==4,"E26-DATA repeated endpoint contributions do not duplicate link denominator",null);
        f.Reset(r);check(StatisticsPlotPages.Build(8,r,f).Plots.Single(p=>p.Id=="industry-realization").Marks.Select(m=>m.Label).Distinct().Count()==2,"E26-DATA duplicate names are disambiguated in exact data",null);f.ByPlayer=true;returns=StatisticsPlotPages.Build(8,r,f);check(returns.Plots.Where(p=>p.Id.StartsWith("construction-return")).Select(p=>p.SharedMax).Distinct().Count()==1,"E26-DATA player facets share a scale",null);
        foreach(string view in new[]{"Returns","Conversion","Transition & networks"}){f.EfficiencyView=view;var p=StatisticsPlotPages.Build(8,r,f);check(p.Ledgers.Select(s=>s.Identity).Distinct().Count()==p.Ledgers.Count,"E26-DATA distinct table identities in "+view,null);check(p.Plots.SelectMany(x=>x.Marks).All(m=>!Double.IsInfinity(m.Y)&&(m.Missing||!Double.IsNaN(m.Y))),"E26-DATA all plotted nonmissing values finite in "+view,null);}
        typeof(CompletedMatchStatistics).GetProperty("Introductory").SetValue(r,true,null);f.EfficiencyView="Transition & networks";check(StatisticsPlotPages.Build(8,r,f).Plots.Single(p=>p.Id=="carryover-scoring").Marks.Count==0,"E26-DATA introductory carryover unavailable",null);
        f.Reset(r);check(f.EfficiencyView=="Returns","E26-DATA reset selects Returns",null);
    }
    public static IEnumerator UI(StatisticsPanel panel,Camera camera,Action<bool,string,object> check,Action<RenderTexture,Camera,string> capture,string prefix)
    {
        var previous=camera.targetTexture;var target=new RenderTexture(1920,1200,24);target.Create();camera.targetTexture=target;yield return null;yield return null;yield return null;
        panel.ResetFilters();panel.SelectPage(0);yield return null;yield return null;
        var ranked=panel.Report.Players.OrderBy(p=>p.Rank>0?p.Rank:Int32.MaxValue).ThenBy(p=>(int)p.Color).ToArray();
        for(int i=0;i<ranked.Length;i++){var card=panel.GetComponentsInChildren<StatisticsSurface>().Single(s=>s.name=="Final result "+ranked[i].Color);check(card.Alternate==((i/2)%2==1),"E26-OVERVIEW row-paired background",new{prefix,index=i});var medal=card.GetComponentInChildren<StatisticsMedal>();check(ranked[i].Rank==1||ranked[i].Rank==2?medal!=null&&medal.Rank==ranked[i].Rank:medal==null,"E26-OVERVIEW medal follows stored rank",new{prefix,rank=ranked[i].Rank});}
        panel.BodyScroll.verticalNormalizedPosition=1;Canvas.ForceUpdateCanvases();yield return null;capture(target,camera,"statistics-efficiency-"+prefix+"-overview-1200");
        foreach(int preset in new[]{1,2}) {
            while(panel.TextScale<(preset==1?1.14f:1.29f)){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Larger text").onClick.Invoke();yield return null;}
            while(panel.TextScale>(preset==1?1.16f:1.31f)){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Smaller text").onClick.Invoke();yield return null;}
            foreach(string view in new[]{"Returns","Conversion","Transition & networks"}) {
                panel.SelectPage(8);panel.SetEfficiencyView(view);yield return null;yield return null;
                var plots=panel.GetComponentsInChildren<StatisticsPlotGraphic>();check(plots.Length==(view=="Returns"?4:3),"E26-UI only active tab constructs charts",new{prefix,preset,view,count=plots.Length});
                check(panel.GetComponentsInChildren<Button>().Any(b=>b.name=="Cost denominator")== (view=="Returns"),"E26-UI cost control only on Returns",new{prefix,view});
                check(plots.SelectMany(p=>p.Plot.Marks).All(m=>m.Target==null||m.Target.Page!=8||m.Target.EfficiencyView==null||m.Target.EfficiencyView==view),"E26-UI drills retain the relevant view",new{prefix,view});
                check(panel.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>!String.IsNullOrWhiteSpace(t.text)).All(t=>t.fontSize*panel.GetComponentInParent<Canvas>().rootCanvas.scaleFactor>=17.9f),"E26-UI physical text readable",new{prefix,preset,view});
                foreach(var plot in plots){var rect=plot.rectTransform;Vector3 top=panel.BodyScroll.content.InverseTransformPoint(rect.TransformPoint(new Vector3(rect.rect.xMin,rect.rect.yMax)));float offset=Math.Max(0,-top.y-58*panel.TextScale);float range=panel.BodyScroll.content.rect.height-panel.BodyScroll.viewport.rect.height;panel.BodyScroll.verticalNormalizedPosition=range<=0?1:Mathf.Clamp01(1-offset/range);Canvas.ForceUpdateCanvases();yield return null;capture(target,camera,"statistics-efficiency-"+prefix+"-"+plot.Plot.Id+"-text"+preset+"-1200");}
                if(view=="Conversion")check(!panel.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.name=="Heat row total"),"E26-UI median heatmap has no sum totals",prefix);
            }
        }
        panel.SetEfficiencyView("Returns");yield return null;panel.BodyScroll.verticalNormalizedPosition=.35f;panel.SetEfficiencyView("Conversion");yield return null;panel.BodyScroll.verticalNormalizedPosition=.7f;panel.SetEfficiencyView("Returns");yield return null;check(Math.Abs(panel.BodyScroll.verticalNormalizedPosition-.35)<.01,"E26-UI per-tab scroll restoration",prefix);
        panel.ResetFilters();panel.SelectPage(8);yield return null;check(panel.Filters.EfficiencyView=="Returns","E26-UI Reset defaults Returns",prefix);
        var chosen=StatisticsPlotPages.Build(8,panel.Report,panel.Filters).Plots.Single(p=>p.Id=="construction-return").Marks.First(m=>!m.Missing);panel.Drill(chosen.Target);yield return null;yield return null;check(panel.Filters.EfficiencyView=="Returns"&&panel.GetComponentsInChildren<Button>().Count(b=>b.name=="Data Conversion and efficiency")<=1,"E26-UI evidence navigation and one efficiency ledger",prefix);
        panel.ResetFilters();panel.SelectPage(0);yield return null;camera.targetTexture=previous;target.Release();UnityEngine.Object.Destroy(target);yield return null;
    }
    public static IEnumerator Lease(Action<bool,string,object> check)
    {
        var root=new GameObject("Lease fixture");var under=new GameObject("Board presentation",typeof(RectTransform),typeof(Canvas),typeof(Frameworks.View.Common.SafeArea));under.transform.SetParent(root.transform);var panel=new GameObject("Statistics",typeof(RectTransform),typeof(Canvas));panel.transform.SetParent(root.transform);
        var off=new GameObject("Already disabled",typeof(RectTransform),typeof(Canvas));off.transform.SetParent(under.transform);off.GetComponent<Canvas>().enabled=false;
        var work=DOTween.To(()=>0f,x=>{},1,60).SetTarget(under.transform);var wasPaused=DOTween.To(()=>0f,x=>{},1,60).SetTarget(under.transform).Pause();var routing=DOTween.Sequence().AppendInterval(60);var ui=DOTween.To(()=>0f,x=>{},1,60).SetTarget(panel.transform);
        for(int i=0;i<30;i++){bool routerBefore=routing.IsPlaying();using(var lease=new StatisticsPresentationLease(root.transform,panel.transform)){check(!under.GetComponent<Canvas>().enabled&&under.activeSelf&&!work.IsPlaying()&&routing.IsPlaying()==routerBefore&&ui.IsPlaying()&&panel.GetComponent<Canvas>().enabled,"E26-LIFECYCLE scoped suspension preserves panel/router state",i);yield return null;}check(under.GetComponent<Canvas>().enabled&&!off.GetComponent<Canvas>().enabled&&work.IsPlaying()&&!wasPaused.IsPlaying(),"E26-LIFECYCLE exact initial enabled/paused states restored",i);}
        var protectedTransition=DOTween.To(()=>0f,x=>{},1,60).SetTarget(under.transform);using(var lease=new StatisticsPresentationLease(root.transform,panel.transform,new Tween[]{protectedTransition})){check(protectedTransition.IsPlaying(),"E26-LIFECYCLE owned router transitions are protected",null);}protectedTransition.Kill();
        using(var lease=new StatisticsPresentationLease(root.transform,panel.transform)){work.Kill();UnityEngine.Object.Destroy(off);yield return null;}
        check(!work.IsActive(),"E26-LIFECYCLE destroyed objects and killed tweens are not resurrected",null);ui.Kill();routing.Kill();wasPaused.Kill();UnityEngine.Object.Destroy(root);yield return null;
    }
}
