using System;
using System.Collections;
using System.Linq;
using BrassPatch.History;
using BrassPatch.Statistics;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// The same checks run through the live Results panel and the archive viewer.
public static class StatisticsHistoryReviewChecks
{
    public static IEnumerator Run(StatisticsPanel panel,Camera camera,Action<bool,string,object> check,Action<RenderTexture,Camera,string> capture,string prefix)
    {
        var efficiency=StatisticsEfficiencyChecks.UI(panel,camera,check,capture,prefix);while(efficiency.MoveNext())yield return efficiency.Current;var report=panel.Report;var f=new StatisticsFilters();
        foreach(var player in report.Players)foreach(int era in report.Introductory?new[]{0,1}:new[]{0,1,2}){
            f.Reset(report);f.Players.Clear();f.Players.Add(player.Color);f.Era=era;
            var actions=report.Actions.Where(a=>a.Player==player.Color&&(era==0||a.Era==era)).ToArray();
            var entries=report.Entries.Where(e=>e.Player==player.Color&&(era==0||e.Era==era)).ToArray();
            var score=StatisticsPlotPages.Build(0,report,f).Plots.First(p=>p.Id=="composition");
            check(score.Marks.Sum(m=>m.Y)==entries.Where(e=>e.Kind=="Balance").Sum(e=>e.VP),"S23-DATA selected player/era score reconciles",new{prefix,player.Color,era});
            var industry=StatisticsPlotPages.Build(1,report,f).Plots.First(p=>p.Id=="industry-vp");
            int vp=report.Assets.Where(a=>a.Player==player.Color).Sum(a=>(era==2?0:a.CanalVP+a.IntroVP)+(era==1?0:a.RailVP));
            check(industry.Marks.Sum(m=>m.Y)==vp,"S23-DATA industry heatmap includes the correct era and carryover",new{prefix,player.Color,era});
            check(industry.Marks.All(m=>StatisticsPlotSemantics.Value(industry,m).EndsWith(" VP")),"S23-LABEL industry heatmap exact values use VP, not the row-axis name",prefix);
            var mix=StatisticsPlotPages.Build(6,report,f).Plots.First(p=>p.Id=="actions");
            check(mix.Marks.Sum(m=>m.Y)==actions.Length,"S23-DATA selected action denominator includes Pass",new{prefix,player.Color,era});
            foreach(bool all in new[]{false,true}){
                f.AllCashPayments=all;var ratio=StatisticsChartQueries.Efficiency(report,f).Single();int spending=all?-entries.Where(e=>e.Kind=="Balance"&&e.Cash<0).Sum(e=>e.Cash):actions.Sum(a=>a.Spending);
                int final=era==0?player.FinalVP:entries.Where(e=>e.Kind=="Balance").Sum(e=>e.VP);
                check(ratio.Actions==actions.Length&&ratio.Spending==spending&&ratio.VP==final&&(final<=0?Double.IsNaN(ratio.CostPerVP):Math.Abs(ratio.CostPerVP-(double)spending/final)<1e-9),"S23-DATA efficiency cash and VP bases reconcile",new{prefix,era,all});
            }
            foreach(string resource in new[]{"Coal","Iron","Beer"})foreach(bool outgoing in new[]{false,true}){
                f.Resource=resource;f.Outgoing=outgoing;f.ResourceView="Supply";
                int quantity=report.Entries.Where(e=>e.Kind=="Resource use"&&e.Resource==resource&&(era==0||e.Era==era)&&(outgoing?e.Supplier==player.Color:e.Player==player.Color)).Sum(e=>e.Quantity);
                check(StatisticsPlotPages.Build(5,report,f).Plots.All(p=>p.Marks.Sum(m=>m.Y)==quantity),"S23-DATA supplier matrix and source mix share the selected units",new{prefix,era,resource,outgoing});
                var flow=StatisticsPlotPages.Build(5,report,f).Plots.First(p=>p.Id=="resource-flow");check(flow.Marks.All(m=>StatisticsPlotSemantics.Value(flow,m).EndsWith(" units")),"S23-LABEL resource matrix exact values use units, not Supplier",prefix);
            }
            f.Outgoing=false;f.ResourceView="Market";var market=StatisticsPlotPages.Build(5,report,f);
            foreach(string resource in new[]{"Coal","Iron"}){var purchases=entries.Where(e=>e.Kind=="Market purchase"&&e.Resource==resource).ToArray();int quantity=purchases.Sum(e=>e.Quantity);var mark=market.Plots.Single(p=>p.Id=="market-"+resource.ToLowerInvariant()).Marks.Single();check(quantity==0?mark.Missing:!mark.Missing&&Math.Abs(mark.Y-(double)purchases.Sum(e=>e.TotalPrice)/quantity)<1e-9,"S23-DATA weighted market price separates no purchases from zero",new{prefix,era,resource});}
        }
        panel.ResetFilters();panel.Filters.Players.Clear();panel.Filters.Players.Add(report.Players[0].Color);panel.Filters.Era=1;panel.SelectPage(6);yield return null;yield return null;
        var shown=panel.GetComponentsInChildren<StatisticsPlotGraphic>().First(g=>g.Plot.Id=="actions");
        check(shown.Plot.Marks.Sum(m=>m.Y)==report.Actions.Count(a=>a.Player==report.Players[0].Color&&a.Era==1),"S23-FILTER rendered player/era selection uses correct cached data",prefix);
        panel.Filters.Search="not-a-city-or-record";panel.SelectPage(6);yield return null;yield return null;
        check(panel.GetComponentsInChildren<StatisticsPlotGraphic>().First(g=>g.Plot.Id=="actions").Plot.Marks.Sum(m=>m.Y)==shown.Plot.Marks.Sum(m=>m.Y),"S23-FILTER search changes ledgers without changing plot totals",prefix);
        panel.ResetFilters();panel.SelectPage(5);yield return null;yield return null;
        foreach(string view in new[]{"Market","Merchants","Supply"}){
            panel.GetComponentsInChildren<Button>().First(b=>b.name=="Resource view "+view).onClick.Invoke();yield return null;yield return null;
            check(panel.Filters.ResourceView==view&&panel.GetComponentsInChildren<StatisticsPlotGraphic>().All(g=>view=="Market"?g.Plot.Id.StartsWith("market-"):view=="Merchants"?g.Plot.Id=="merchant-sales":g.Plot.Id=="resource-flow"||g.Plot.Id=="sources"),"S23-FILTER resource tabs replace plots without stale views",new{prefix,view});
        }
        var target=new RenderTexture(1920,1200,24);target.Create();camera.targetTexture=target;yield return null;yield return null;
        while(panel.TextScale>1.16f){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Smaller text").onClick.Invoke();yield return null;yield return null;}
        while(panel.TextScale<1.14f){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Larger text").onClick.Invoke();yield return null;yield return null;}
        for(int page=0;page<10;page++){
            panel.ResetFilters();panel.SelectPage(page);yield return null;yield return null;
            foreach(string view in page==5?new[]{"Supply","Market","Merchants"}:new[]{""}){
                if(view!=""){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Resource view "+view).onClick.Invoke();yield return null;yield return null;}
                foreach(var graphic in panel.GetComponentsInChildren<StatisticsPlotGraphic>()){
                    Geometry(graphic,check,prefix);
                    var root=graphic.transform;while(root.parent!=null&&root.name!="Plot "+graphic.Plot.Id)root=root.parent;
                    var corners=new Vector3[4];((RectTransform)root).GetWorldCorners(corners);
                    float delta=panel.BodyScroll.viewport.InverseTransformPoint(corners[1]).y-panel.BodyScroll.viewport.rect.yMax;
                    var pos=panel.BodyScroll.content.anchoredPosition;pos.y=Mathf.Clamp(pos.y-delta,0,Math.Max(0,panel.BodyScroll.content.rect.height-panel.BodyScroll.viewport.rect.height));panel.BodyScroll.content.anchoredPosition=pos;yield return null;yield return null;Canvas.ForceUpdateCanvases();
                    capture(target,camera,prefix+"-plot-1200-"+graphic.Plot.Id);
                }
            }
        }
        // Detached copy: exercise literal names and tall category labels without
        // modifying the game or its stored report.
        var detached=GameHistoryStatisticsCodec.Decode(GameHistoryStatisticsCodec.Encode(report));
        string name="<size=80><color=red>Alexandra Birmingham Manufacturing Partnership</color></size>";
        foreach(var p in detached.Players)typeof(StatisticsPlayer).GetProperty("Name").SetValue(p,name,null);
        var previewRoot=new GameObject("Literal name statistics preview",typeof(RectTransform),typeof(Canvas),typeof(GraphicRaycaster),typeof(CanvasGroup));previewRoot.transform.SetParent(panel.transform.parent,false);
        var rect=(RectTransform)previewRoot.transform;rect.anchorMin=Vector2.zero;rect.anchorMax=Vector2.one;rect.offsetMin=rect.offsetMax=Vector2.zero;var canvas=previewRoot.GetComponent<Canvas>();canvas.overrideSorting=true;canvas.sortingOrder=510;
        var preview=previewRoot.AddComponent<StatisticsPanel>();preview.Initialize(detached,panel.GetComponentsInChildren<TextMeshProUGUI>().First().font,null,panel.OpeningContext,"Regression preview");preview.CreateShowViewImmediatelyAnimation();preview.SelectPage(8);yield return null;yield return null;Canvas.ForceUpdateCanvases();
        var label=preview.GetComponentsInChildren<TextMeshProUGUI>().First(t=>t.text==name);label.ForceMeshUpdate();
        check(!label.richText&&label.GetParsedText().StartsWith("<size="),"S23-LABEL player names are literal, never formatting instructions",prefix);
        foreach(var graphic in preview.GetComponentsInChildren<StatisticsPlotGraphic>())Geometry(graphic,check,prefix+" long names");
        capture(target,camera,prefix+"-literal-names-1200");previewRoot.SetActive(false);UnityEngine.Object.Destroy(previewRoot);yield return null;
        camera.targetTexture=null;target.Release();UnityEngine.Object.Destroy(target);panel.ResetFilters();panel.SelectPage(0);yield return null;yield return null;
    }
    static void Geometry(StatisticsPlotGraphic graphic,Action<bool,string,object> check,string prefix)
    {
        var texts=graphic.transform.parent.GetComponentsInChildren<TextMeshProUGUI>();
        check(texts.Where(t=>t.name.StartsWith("Axis row ")).All(t=>t.GetPreferredValues(t.text,t.rectTransform.rect.width,0).y<=t.rectTransform.rect.height+1),"S23-LABEL wrapped row labels fit their bands",new{prefix,graphic.Plot.Id});
        if(graphic.Horizontal||graphic.Plot.Kind==StatisticsPlotKind.EventLanes||graphic.Plot.Kind==StatisticsPlotKind.Lifespans)
            check(texts.Count(t=>t.name.StartsWith("Value tick top "))==graphic.XAxis.Ticks.Count,"S23-VISUAL long plots show the same numeric scale above and below",new{prefix,graphic.Plot.Id});
        if(graphic.Plot.Kind==StatisticsPlotKind.Step)check(texts.Where(t=>t.name=="Curve endpoint").All(t=>t.rectTransform.anchoredPosition.x+t.rectTransform.rect.width<=graphic.rectTransform.rect.width+1),"S23-VISUAL curve labels fit the reserved right gutter",new{prefix,graphic.Plot.Id});
        if(graphic.Plot.Kind==StatisticsPlotKind.Heatmap)check(graphic.Plot.DiscreteHeat?texts.Any(t=>t.name=="Position label 1"):texts.Any(t=>t.name=="Heat row totals heading")&&graphic.transform.parent.GetComponentsInChildren<Image>().Any(i=>i.name=="Heat scale band"),"S23-VISUAL heatmap has an explicit color key and totals label",new{prefix,graphic.Plot.Id});
        if(graphic.Plot.DiscreteHeat){var scroll=graphic.GetComponentInParent<StatisticsTableScroll>();float visible=scroll==null?graphic.rectTransform.rect.width:scroll.viewport.rect.width;check(texts.Where(t=>t.name.StartsWith("Position label ")).All(t=>t.rectTransform.anchoredPosition.x+t.rectTransform.rect.width<=visible+1),"S23-VISUAL all position color keys fit the initial view",prefix);}
        if(graphic.Plot.Kind==StatisticsPlotKind.MiniMap){
            var boxes=texts.Where(t=>t.name.StartsWith("City ")).Select(t=>new Rect(t.rectTransform.anchoredPosition-new Vector2(0,t.rectTransform.rect.height),t.rectTransform.rect.size)).ToArray();
            check(boxes.SelectMany((a,i)=>boxes.Skip(i+1).Select(b=>!a.Overlaps(b))).All(ok=>ok),"S23-LABEL geography city labels do not overlap",prefix);
            check(boxes.All(box=>graphic.Plot.Marks.All(mark=>!box.Overlaps(graphic.MarkBounds(mark)))),"S23-VISUAL geography label backgrounds leave city symbols fully visible",prefix);
        }
    }
}
