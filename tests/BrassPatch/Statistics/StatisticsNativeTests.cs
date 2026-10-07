using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Game.FinishedGameReview;
using BrassApplication.UseCases.Game.Replay;
using BrassPatch;
using BrassPatch.Statistics;
using DG.Tweening;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.View.Common;
using Frameworks.View.Game;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.EndOfGame;
using InterfaceAdapters.Game.FinishedGameReview;
using InterfaceAdapters.Routing;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class LocalAiUnityTests
{
    public static void Begin(string output)
    {
        var driver=new GameObject("Statistics qualification").AddComponent<StatisticsDriver>();
        UnityEngine.Object.DontDestroyOnLoad(driver.gameObject);driver.Output=output;
    }
}
public sealed partial class StatisticsDriver : MonoBehaviour
{
    public string Output;
    readonly List<object> cases=new List<object>();
    readonly List<string> errors=new List<string>();
    readonly List<PerformanceSample> performance=new List<PerformanceSample>();
    sealed class PerformanceSample { public string Operation,Scope; public double Milliseconds; }
    void Measure(string operation,string scope,System.Diagnostics.Stopwatch clock)
    {performance.Add(new PerformanceSample{Operation=operation,Scope=scope,Milliseconds=clock.Elapsed.TotalMilliseconds});}
    static double PerformanceMedian(IEnumerable<double> source)
    {var values=source.OrderBy(v=>v).ToArray();return values.Length%2==1?values[values.Length/2]:(values[values.Length/2-1]+values[values.Length/2])/2;}
    int checks,failures,matches,pages;
    readonly List<CompletedMatchStatistics> previewReports=new List<CompletedMatchStatistics>();
    readonly Vector2Int[] resolutions={new Vector2Int(1920,1200),new Vector2Int(1920,1080)};
    void Check(bool pass,string name,object detail=null){checks++;if(!pass)failures++;cases.Add(new{name,passed=pass,detail});}
    void Log(string text,string stack,LogType kind)
    {if(text.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages"))return;if(kind==LogType.Error||kind==LogType.Exception)errors.Add(text);}
    IEnumerator Start()
    {
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;Application.logMessageReceived+=Log;
        Exception failure=null;var run=Run();
        while(true) {
            bool more=false;object current=null;
            try{more=run.MoveNext();if(more)current=run.Current;}catch(Exception e){failure=e;}
            if(failure!=null||!more)break;yield return current;
        }
        File.WriteAllText(Output,JsonConvert.SerializeObject(new{passed=failure==null&&failures==0&&errors.Count==0,checks,failures,matches,pages,cases,focused=Environment.GetEnvironmentVariable("BRASS_GEO_FOCUS")=="1",captureSkipped=Environment.GetEnvironmentVariable("BRASS_STATS_FAST")=="1",layouts=resolutions.Select(r=>new{width=r.x,height=r.y}).ToArray(),nativeErrors=errors,error=failure==null?null:failure.ToString(),performance=new{processor=SystemInfo.processorType,graphics=SystemInfo.graphicsDeviceName,definition="Native batch-player measurements; response samples include frame waits and layout, excluding PNG capture/encoding. Scroll samples are CPU layout/update time, not a frame-rate guarantee.",samples=performance,summary=performance.GroupBy(p=>p.Operation).Select(g=>new{operation=g.Key,samples=g.Count(),minimumMilliseconds=g.Min(p=>p.Milliseconds),medianMilliseconds=PerformanceMedian(g.Select(p=>p.Milliseconds)),maximumMilliseconds=g.Max(p=>p.Milliseconds)})}},Formatting.Indented));
        Application.Quit(failure==null&&failures==0&&errors.Count==0?0:1);
    }
    sealed class Match
    {
        public BrassApplicationGame App;
        public BaseRepository Repository;
        public CompletedMatchStatistics Report;
        public IEvent[] Events;
    }
    Match Finish(int count,bool intro,int seed,bool passOnly=false)
    {
        var config=new GameStartConfiguration((NumberOfPlayers)count){Seed=seed,IsIntroductoryGame=intro};
        var metadata=new GameMetadata(config,GameType.Local);
        foreach(var start in config.Players) {
            var player=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="statistics-"+start.Color,NickName="Player "+start.Color,AiType=AIType.Hard};
            player.AIBehaviour=new HardAiBehavior(player.PlayerNetworkId);metadata.JoinGame(player);
        }
        var repository=new BaseRepository(new InMemoryGameStorage(),new InMemorySnapshotStorage(),new BrassBirminghamFactory());
        var app=repository.Create(metadata);app.Game.StartGame();int turns=0;
        while(!app.Game.GameProgressInformation.IsGameOver) {
            if(++turns>200)throw new Exception("Match did not finish");
            var behavior=app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AIBehaviour;
            while(app.Game.TurnFlow.GetRemainingActionsCount()>0) {
                if(passOnly) {
                    var player=app.Game.GameProgressInformation.CurrentPlayer;
                    if(count==4&&turns==1){var scout=app.Game.TurnFlow.StartScoutAction();scout.DiscardCard(player.Hand[0],player);scout.Scout(player.Hand.Take(2).ToList(),player);}
                    else app.Game.TurnFlow.StartPassAction().Pass(player.Hand.FirstOrDefault(c=>c.GetType().Name=="WildLocationCard")??player.Hand[0],player);
                } else behavior.GetNextAIMove(app)();
                app.Game.TurnFlow.ConfirmCurrentAction();
            }
            app.Game.GameFlow.EndTurn();
            if(app.Game.GameFlow.IsLastTurnOver()) {
                if(app.Game.GameFlow.IsSellIndustryNecessary())
                    RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet()).Select(app).MakeAction(app,false)();
                app.Game.GameFlow.EndRound();
            }
        }
        var events=app.Game.GetUncommittedEvents().ToArray();
        string payload=JsonConvert.SerializeObject(events);int revision=app.Game.Revision;
        var bonus=DevelopMerchantBonus.Instance;
        var report=StatisticsCollector.Collect(app,new IEvent[0],events);matches++;previewReports.Add(report);
        performance.Add(new PerformanceSample{Operation="Detached collection",Scope=count+" players; "+(intro?"introductory":"full")+(passOnly?"; all Pass":""),Milliseconds=report.CollectionMilliseconds});
        if(report.CompleteHistory){StatisticsReviewChecks.Projection(report,events,Check);StatisticsLightChecks.Projection(report,Check);}
        Check(report.CompleteHistory,"complete match reconstruction",new{count,intro,seed,report.Availability,report.CollectionMilliseconds});
        Check(JsonConvert.SerializeObject(events)==payload&&app.Game.Revision==revision&&ReferenceEquals(bonus,DevelopMerchantBonus.Instance),"collection leaves live events, revision and bonus scope untouched");
        int actions=intro?(count==2?19:count==3?17:15):(count==2?39:count==3?35:31);
        foreach(var player in app.Game.PlayersTrack.Players) {
            var result=report.Players.First(p=>p.Color==player.Color);
            Check(result.FinalVP==player.VP&&result.FinalCash==player.Money&&result.FinalIncome==player.IncomeLevel,"final result reconciles",player.Color.ToString());
            Check(report.Actions.Count(a=>a.Player==player.Color)==actions,"confirmed action counts follow player-count/era rules",new{count,intro,color=player.Color.ToString(),actual=report.Actions.Count(a=>a.Player==player.Color),expected=actions});
            Check(result.OpeningCash+report.Entries.Where(e=>e.Player==player.Color&&e.Kind=="Balance").Sum(e=>e.Cash)==result.FinalCash,"cash reconciliation");
            Check(result.OpeningVP+report.Entries.Where(e=>e.Player==player.Color&&e.Kind=="Balance").Sum(e=>e.VP)==result.FinalVP,"VP reconciliation");
        }
        Check(report.Assets.Sum(a=>a.CanalVP+a.RailVP)==report.Entries.Where(e=>e.Kind=="Balance"&&e.Category=="Industries").Sum(e=>e.VP),"industry scoring reconciles");
        Check(report.Links.Sum(l=>l.VP)==report.Entries.Where(e=>e.Kind=="Balance"&&e.Category=="Links").Sum(e=>e.VP),"links reconcile including three-endpoint routes");
        foreach(var asset in report.Assets.Where(a=>a.Produced>0)) {
            int sold=report.Entries.Where(e=>e.Kind=="Market sale"&&e.AssetKey==asset.Key).Sum(e=>e.Quantity);
            Check(asset.Produced==asset.Supplied+sold+asset.Stock,"producer resource conservation",new{asset.Type,asset.Key,asset.Produced,asset.Supplied,sold,asset.Stock});
        }
        var filters=new StatisticsFilters();filters.Reset(report);
        for(int page=0;page<StatisticsPages.Names.Length;page++) {
            var sections=page==10?StatisticsPlotPages.Build(page,report,filters).Ledgers:StatisticsPages.Build(page,report,filters);
            Check(sections.Count>0&&sections.All(s=>s.Rows.All(r=>r.Cells.Length==s.Columns.Length)),"page schema has complete usable ledgers",page);
        }
        var curve=StatisticsPages.Build(4,report,filters).First(s=>s.Chart!=null).Chart;
        Check(curve.Series.SelectMany(s=>s.Points).Max(p=>p.X)==report.Samples.Max(s=>s.Turn),"chart horizontal units are turns, not event revisions");
        var overlapping=StatisticsCollector.Collect(app,events.Take(events.Length/2),events.Skip(events.Length/3));
        Check(overlapping.CompleteHistory&&overlapping.Actions.Count==report.Actions.Count,"persisted/uncommitted overlap de-duplicates revisions");
        var gap=StatisticsCollector.Collect(app,events.Where(e=>e.Metadata.Revision!=10),new IEvent[0]);
        Check(!gap.CompleteHistory&&gap.Players.All(p=>p.FinalVP==report.Players.First(q=>q.Color==p.Color).FinalVP)&&gap.Actions.Count==0,"missing history retains final result without partial totals");
        var absent=StatisticsCollector.Collect(app,events.Skip(1),new IEvent[0]);
        Check(!absent.CompleteHistory&&StatisticsPages.Build(1,absent,filters).All(s=>s.Rows.Count==0),"missing setup is unavailable, never fabricated zero");
        var wrong=new BoardGameRules.Entities.Actions.Events.ActionPassed(null,PlayerColor.Red);wrong.Metadata.Revision=events[10].Metadata.Revision;
        var conflict=StatisticsCollector.Collect(app,events,new IEvent[]{wrong});
        Check(!conflict.CompleteHistory&&conflict.Availability.Contains("Conflicting events"),"conflicting overlapping revisions are rejected");
        if(passOnly) {
            Check(report.Players.All(p=>p.Rank==1)&&report.Actions.All(a=>a.Kind=="Pass"||count==4&&a.Kind=="Scout"),"unresolved draw and Pass/Scout actions stay explicit");
            if(count==4){var episodes=report.Entries.Where(e=>e.Kind=="Wildcard acquired").ToArray();Check(episodes.Length==2&&episodes.Select(e=>e.AssetKey).Distinct().Count()==2&&report.Entries.Any(e=>e.Kind=="Card"&&e.Label.StartsWith("Wild"))&&report.Entries.Where(e=>e.Kind=="Card"&&e.Label.StartsWith("Wild")).All(use=>episodes.Any(e=>e.AssetKey==use.AssetKey&&e.Category==use.Label)),"SD01 collected Scout acquisitions keep both card-specific episode associations");}
            var table=StatisticsPages.Build(8,report,filters).First(s=>s.Title=="Conversion and efficiency");
            Check(table.Rows.All(r=>r.Cells[2]=="—"),"zero-VP denominators display unavailable ratio");
        }
        repository.Save(app);
        var history=new BrassGameHistory{Repository=repository};history.SetGame(app);
        Check(StatisticsCollector.Capture(history).CompleteHistory,"saved history plus current revision can be collected");
        return new Match{App=app,Repository=repository,Report=report,Events=events};
    }
    IEnumerator Run()
    {
        StatisticsReviewChecks.Run(Check);StatisticsLightChecks.Run(Check);StatisticsEfficiencyChecks.Data(Check);var leaseChecks=StatisticsEfficiencyChecks.Lease(Check);while(leaseChecks.MoveNext())yield return leaseChecks.Current;
        Match example=null;
        foreach(int count in new[]{2,3,4})foreach(bool intro in new[]{false,true}) {
            var match=Finish(count,intro,20260930+count+(intro?10:0));if(count==4&&!intro)example=match;yield return null;
        }
        foreach(int count in new[]{2,3,4}){Finish(count,false,20261010+count,true);yield return null;}
        if(example==null||!example.Report.CompleteHistory)throw new Exception("No complete report for native UI");
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(Output),"statistics-example.json"),JsonConvert.SerializeObject(example.Report,Formatting.Indented));
        var viewFactory=Resources.FindObjectsOfTypeAll<UnityViewFactory>().First(f=>f.ViewList!=null&&f.ViewList.Views.OfType<Frameworks.View.Game.FinishedGameReview.FinishedGameReviewView>().Any());
        var root=new GameObject("Statistics test canvas",typeof(RectTransform),typeof(Canvas),typeof(GraphicRaycaster),typeof(CanvasScaler));
        root.GetComponent<Canvas>().renderMode=RenderMode.ScreenSpaceOverlay;
        root.GetComponent<CanvasScaler>().uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;root.GetComponent<CanvasScaler>().referenceResolution=new Vector2(1920,1080);
        var factory=root.AddComponent<UnityViewFactory>();factory.ParentTransform=root.transform;factory.ViewList=viewFactory.ViewList;
        var cameraNode=new GameObject("Statistics render camera",typeof(Camera));var camera=cameraNode.GetComponent<Camera>();
        camera.enabled=false;camera.orthographic=true;camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=new Color(.065f,.06f,.05f);
        camera.transform.position=new Vector3(0,0,-100);camera.farClipPlane=1000;
        root.GetComponent<Canvas>().renderMode=RenderMode.ScreenSpaceCamera;root.GetComponent<Canvas>().worldCamera=camera;root.GetComponent<Canvas>().planeDistance=100;
        var input=new Input();var router=new Router(factory,input);var builder=new Builder();router.UseCaseBuilder=builder;Utils.Global.Instance.Router=router;
        var history=new BrassGameHistory{Repository=example.Repository};history.SetGame(example.App);
        var presenter=new Presenter(router);var owner=new FinishedGameReviewUseCase(presenter,history);builder.Owner=owner;
        owner.StartGameReview(example.App.Metadata.PlayerMetadata.First());
        Check(!root.GetComponentsInChildren<Button>(true).Any(b=>b.name=="Match statistics")&&!router.IsViewCached<StatisticsPanel>(),"entry remains hidden until final replay completes");
        owner.ReplayCompleted();
        Tick(1);DOTween.CompleteAll();yield return null;Canvas.ForceUpdateCanvases();
        var results=(EndOfGameView)router.GetView<IEndOfGameView>();
        var entry=results.transform.Find("SafeArea/Popup/Results statistics").GetComponent<Button>();
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(Output),"results-entry-geometry.json"),JsonConvert.SerializeObject(results.GetComponentsInChildren<RectTransform>(true).Select(r=>new{name=r.name,path=PathTo(r,results.transform),width=r.rect.width,height=r.rect.height,position=r.anchoredPosition.ToString(),screen=RectTransformUtility.WorldToScreenPoint(camera,r.TransformPoint(r.rect.center)).ToString()}),Formatting.Indented));
        Check(entry!=null,"statistics entry appears in original results");
        var response=System.Diagnostics.Stopwatch.StartNew();
        entry.onClick.Invoke();Tick(1);yield return null;Canvas.ForceUpdateCanvases();
        Measure("First page response","4-player full match; includes collection and initial UI",response);
        var panel=router.GetView<StatisticsPanel>();Check(panel.IsOpen&&!router.IsViewVisible<IEndOfGameView>(),"statistics replaces the active result modal");
        Check(Math.Abs(panel.TextScale-1.15f)<.001f,"statistics defaults to the second text size");
        if(Environment.GetEnvironmentVariable("BRASS_GEO_FOCUS")=="1"){var geo=StatisticsGeographyChecks.UI(panel,camera,Check,Capture,"live");while(geo.MoveNext())yield return geo.Current;yield break;}
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Smaller text").onClick.Invoke();yield return null;
        Check(panel.CurrentPage==0&&StatisticsPages.NavigationOrder.First()==0,"Overview opens first with ten stable factual page IDs");
        var entryCorners=new Vector3[4];((RectTransform)entry.transform).GetWorldCorners(entryCorners);
        float entryPixels=Vector2.Distance(RectTransformUtility.WorldToScreenPoint(camera,entryCorners[0]),RectTransformUtility.WorldToScreenPoint(camera,entryCorners[3]));
        Check(entryPixels>=239,"results Statistics entry is at least 240 physical pixels wide",entryPixels);
        var es=EventSystem.current;if(es==null)es=new GameObject("Statistics event system",typeof(EventSystem),typeof(StandaloneInputModule)).GetComponent<EventSystem>();
        foreach(var resolution in resolutions) {
            var target=new RenderTexture(resolution.x,resolution.y,24,RenderTextureFormat.ARGB32);target.Create();camera.targetTexture=target;
            Screen.SetResolution(resolution.x,resolution.y,false);yield return null;yield return null;yield return null;
            panel.Close(true);Tick(1);DOTween.CompleteAll();yield return null;yield return null;Canvas.ForceUpdateCanvases();
            results=(EndOfGameView)router.GetView<IEndOfGameView>();entry=results.transform.Find("SafeArea/Popup/Results statistics").GetComponent<Button>();
            var statsBounds=ScreenBounds((RectTransform)entry.transform,camera);
            var resultControls=results.GetComponentsInChildren<Button>().Where(b=>b.name=="ReviewButton"||b.name=="ExitGameButton").ToArray();
            Check(resultControls.Length==2&&resultControls.All(b=>!ScreenBounds((RectTransform)b.transform,camera).Overlaps(statsBounds)&&ScreenBounds((RectTransform)b.transform,camera).yMin>statsBounds.yMax)&&Math.Abs(statsBounds.center.x-resolution.x/2)<1,"statistics entry sits below original results controls without overlap",new{resolution.x,resolution.y,statsBounds.xMin,statsBounds.yMin,statsBounds.width,statsBounds.height});
            Check(statsBounds.yMin>=19&&statsBounds.yMax<=resolution.y&&statsBounds.xMin>=0&&statsBounds.xMax<=resolution.x&&Math.Abs(statsBounds.width-240)<1&&Math.Abs(statsBounds.height-60)<1,"light results entry stays within screen with readable fixed size",new{resolution.x,statsBounds.xMin,statsBounds.yMin,statsBounds.width,statsBounds.height});
            Check(entry.GetComponent<StatisticsEntrySurface>()!=null&&entry.GetComponentInChildren<TextMeshProUGUI>().color.r<.3f,"statistics entry uses light brass surface and dark readable text");
            Check(entry.IsInteractable()&&entry.targetGraphic.canvasRenderer.GetAlpha()>.95f,"results statistics control remains fully visible and interactive",new{resolution.y,entry.interactable,effective=entry.IsInteractable(),tint=entry.targetGraphic.canvasRenderer.GetColor().ToString(),groups=entry.GetComponentsInParent<CanvasGroup>().Select(g=>new{g.name,g.alpha,g.interactable,g.ignoreParentGroups}).ToArray()});
            var reviewEntry=((BaseView)router.GetView<IFinishedGameReviewView>()).GetComponentsInChildren<Button>().First(b=>b.name=="Match statistics");
            Check(reviewEntry.GetComponent<CanvasGroup>().alpha==0&&!reviewEntry.GetComponent<CanvasGroup>().blocksRaycasts,"board statistics button is hidden behind original results modal");
            Capture(target,camera,"statistics-results-entry-"+resolution.x+"x"+resolution.y);
            entry.onClick.Invoke();Tick(1);yield return null;yield return null;
            for(int page=0;page<StatisticsPages.Names.Length;page++) {
                response=System.Diagnostics.Stopwatch.StartNew();
                panel.SelectPage(page);yield return null;yield return null;Canvas.ForceUpdateCanvases();
                Measure("Page response",resolution+"; page "+page+"; Large text",response);
                Check(panel.PageButtons.Length==StatisticsPages.Names.Length&&panel.CurrentPage==page&&panel.BodyScroll.content.rect.height>=panel.BodyScroll.viewport.rect.height,"native page layout and scroll bounds",new{resolution=resolution.ToString(),page});
                Check(panel.GetComponentsInChildren<RectTransform>().Where(r=>r.GetComponent<Selectable>()!=null).All(r=>r.rect.width>0&&r.rect.height>0),"controls have positive geometry",new{resolution=resolution.ToString(),page});
                CheckReadability(panel,root.GetComponent<Canvas>(),resolution.x,page);
                var close=panel.GetComponentsInChildren<Button>().First(b=>b.name=="Close statistics");
                var hits=new List<RaycastResult>();var data=new PointerEventData(es){position=RectTransformUtility.WorldToScreenPoint(camera,close.transform.TransformPoint(((RectTransform)close.transform).rect.center))};
                es.RaycastAll(data,hits);
                Check(hits.Count>0&&hits[0].gameObject.transform.IsChildOf(panel.transform),"statistics input shield wins raycast ordering");
                if(resolution.x>0) {
                    Capture(target,camera,"statistics-"+resolution.x+"x"+resolution.y+"-"+page);
                }
            }
            if(resolution.x>0) {
                panel.SelectPage(4);yield return null;yield return null;panel.BodyScroll.verticalNormalizedPosition=.37f;
                for(int i=0;i<2;i++){
                    panel.GetComponentsInChildren<Button>().First(b=>b.name=="Larger text").onClick.Invoke();yield return null;yield return null;
                    for(int page=0;page<StatisticsPages.Names.Length;page++){
                        panel.SelectPage(page);yield return null;yield return null;CheckReadability(panel,root.GetComponent<Canvas>(),resolution.x,page);
                        Capture(target,camera,"statistics-"+resolution.x+"x"+resolution.y+"-"+page+"-"+(i==0?"larger":"largest"));
                    }
                    panel.SelectPage(4);yield return null;yield return null;panel.BodyScroll.verticalNormalizedPosition=.37f;
                }
                Check(Math.Abs(panel.TextScale-1.3f)<.001f&&Math.Abs(panel.BodyScroll.verticalNormalizedPosition-.37f)<.015f,"text enlargement retains the page and scroll position");
                for(int page=0;page<StatisticsPages.Names.Length;page++) {
                    response=System.Diagnostics.Stopwatch.StartNew();
                    panel.SelectPage(page);yield return null;yield return null;Canvas.ForceUpdateCanvases();
                    Measure("Cached page response",resolution+"; page "+page+"; Largest text",response);
                    CheckReadability(panel,root.GetComponent<Canvas>(),resolution.x,page);
                    if(page==0||page==4||page==8)Capture(target,camera,"statistics-"+resolution.x+"x"+resolution.y+"-"+page+"-largest");
                }
                for(int i=0;i<2;i++){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Smaller text").onClick.Invoke();yield return null;yield return null;}
                Check(Math.Abs(panel.TextScale-1f)<.001f,"text size can return to the first preset");
            }
            camera.targetTexture=null;target.Release();Destroy(target);
        }
        pages=StatisticsPages.Names.Length;
        // Capture later plots at an actual scroll position, rather than only page tops.
        var detailTarget=new RenderTexture(1920,1200,24,RenderTextureFormat.ARGB32);detailTarget.Create();camera.targetTexture=detailTarget;
        foreach(int page in new[]{0,2,3,4,5,6,7,8,9}){
            panel.SelectPage(page);yield return null;yield return null;
            foreach(var graphic in panel.GetComponentsInChildren<StatisticsPlotGraphic>()){
                var plotRoot=(RectTransform)graphic.transform.parent.parent;var corners=new Vector3[4];plotRoot.GetWorldCorners(corners);
                float delta=panel.BodyScroll.viewport.InverseTransformPoint(corners[1]).y-panel.BodyScroll.viewport.rect.yMax;
                var position=panel.BodyScroll.content.anchoredPosition;position.y=Mathf.Clamp(position.y-delta,0,panel.BodyScroll.content.rect.height-panel.BodyScroll.viewport.rect.height);panel.BodyScroll.content.anchoredPosition=position;
                Canvas.ForceUpdateCanvases();Capture(detailTarget,camera,"statistics-detail-"+graphic.Plot.Id);
            }
        }
        camera.targetTexture=null;detailTarget.Release();Destroy(detailTarget);
        panel.ResetFilters();panel.SelectPage(3);yield return null;yield return null;
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Map measure").onClick.Invoke();yield return null;yield return null;
        Check(panel.Filters.MapMode=="Industry VP"&&panel.GetComponentsInChildren<StatisticsMapView>().Any(),"map measure changes without moving live board objects");
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Map era 1").onClick.Invoke();yield return null;yield return null;
        Check(panel.Filters.MapEra==1,"boundary selector distinguishes Canal scoring from final occupancy");
        panel.ResetFilters();panel.SelectPage(5);yield return null;yield return null;
        response=System.Diagnostics.Stopwatch.StartNew();
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Resource Coal").onClick.Invoke();yield return null;yield return null;
        Canvas.ForceUpdateCanvases();Measure("Filter response","Coal; all players; incoming",response);
        response=System.Diagnostics.Stopwatch.StartNew();
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Flow direction").onClick.Invoke();yield return null;yield return null;
        Canvas.ForceUpdateCanvases();Measure("Filter response","Coal; all players; outgoing",response);
        Check(panel.Filters.Resource=="Coal"&&panel.Filters.Outgoing,"resource and direction controls use typed scopes");
        var light=LightPanelChecks(panel,camera);while(light.MoveNext())yield return light.Current;
        var previews=LightSpecialPreviews(panel,camera,new[]{previewReports[0],previewReports[3],previewReports[8]});while(previews.MoveNext())yield return previews.Current;
        var currentReview=StatisticsHistoryReviewChecks.Run(panel,camera,Check,Capture,"live");while(currentReview.MoveNext())yield return currentReview.Current;
        panel.ResetFilters();panel.SelectPage(0);yield return null;yield return null;
        for(int i=0;i<2;i++){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Larger text").onClick.Invoke();yield return null;yield return null;}
        panel.SelectPage(0);yield return null;yield return null;
        OpenData(panel);yield return null;yield return null;
        var retainedTable=panel.GetComponentsInChildren<StatisticsTableScroll>().First(s=>s.content.rect.width>s.viewport.rect.width+100);
        string retainedTableName=retainedTable.name;retainedTable.horizontalNormalizedPosition=.62f;
        panel.SelectPage(9);yield return null;yield return null;panel.SelectPage(0);yield return null;yield return null;
        Check(Math.Abs(panel.GetComponentsInChildren<StatisticsTableScroll>().First(s=>s.name==retainedTableName).horizontalNormalizedPosition-.62f)<.01f,"horizontal table position is stored under its original page");
        for(int i=0;i<2;i++){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Smaller text").onClick.Invoke();yield return null;yield return null;}
        panel.SelectPage(0);yield return null;yield return null;
        var wide=panel.GetComponentsInChildren<StatisticsTableScroll>().First();
        for(int step=0;step<12;step++){
            response=System.Diagnostics.Stopwatch.StartNew();
            panel.BodyScroll.verticalNormalizedPosition=1-step/12f;Canvas.ForceUpdateCanvases();
            Measure("Scroll layout update","Expanded Overview ledger; 1920x1080",response);
            yield return null;
        }
        panel.BodyScroll.verticalNormalizedPosition=1;float sidewaysBefore=wide.horizontalNormalizedPosition;
        wide.OnScroll(new PointerEventData(es){scrollDelta=new Vector2(0,-3)});
        Check(panel.BodyScroll.verticalNormalizedPosition<.99f&&Math.Abs(wide.horizontalNormalizedPosition-sidewaysBefore)<.001f,"wheel over a wide table scrolls the page without horizontal drift");
        panel.BodyScroll.verticalNormalizedPosition=1;
        var start=RectTransformUtility.WorldToScreenPoint(camera,panel.BodyScroll.viewport.transform.position);
        var drag=new PointerEventData(es){position=start,pressPosition=start,button=PointerEventData.InputButton.Left};
        wide.OnInitializePotentialDrag(drag);wide.OnBeginDrag(drag);drag.position=start+new Vector2(0,100);drag.delta=new Vector2(0,100);wide.OnDrag(drag);wide.OnEndDrag(drag);
        Check(panel.BodyScroll.verticalNormalizedPosition<.99f,"vertical drag over a table scrolls the page");
        panel.ResetFilters();panel.Filters.Players.Remove(example.Report.Players[0].Color);panel.Filters.Era=1;panel.SelectPage(2);yield return null;
        Check(StatisticsPages.Build(2,example.Report,panel.Filters).SelectMany(s=>s.Rows).All(r=>!r.Player.HasValue||panel.Filters.Players.Contains(r.Player.Value)),"player subset excludes unwanted ledger rows");
        panel.Filters.Search="CoalMine";panel.SelectPage(2);yield return null;
        Check(StatisticsPages.Build(2,example.Report,panel.Filters).SelectMany(s=>s.Rows).All(r=>(String.Join(" ",r.Cells)+" "+r.Detail).IndexOf("CoalMine",StringComparison.OrdinalIgnoreCase)>=0),"ledger search includes details");
        panel.ResetFilters();panel.SelectPage(4);yield return null;yield return null;
        OpenData(panel);yield return null;yield return null;
        string secondName=panel.Report.Players[1].Name;
        ReflectionAccess.Set(panel.Report.Players[1],"Name",panel.Report.Players[0].Name);
        panel.SelectPage(4);yield return null;yield return null;
        panel.GetComponentsInChildren<Button>().First(b=>b.name.StartsWith("Legend ")).onClick.Invoke();yield return null;yield return null;
        Check(panel.GetComponentsInChildren<StatisticsPlotGraphic>().First().Hidden.Count==1,"legend toggles use player identity even when names match");
        panel.ResetFilters();yield return null;yield return null;
        Check(panel.GetComponentsInChildren<StatisticsPlotGraphic>().All(c=>c.Hidden.Count==0),"reset restores hidden chart series as well as filters");
        ReflectionAccess.Set(panel.Report.Players[1],"Name",secondName);
        panel.SelectPage(4);yield return null;yield return null;
        var row=panel.GetComponentsInChildren<Button>().First(b=>b.name.StartsWith("Row "));
        row.onClick.Invoke();yield return null;yield return null;
        Check(panel.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.name.StartsWith("Expanded ")),"table row expands measured details");
        Check(panel.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name.StartsWith("Expanded ")).All(t=>t.GetPreferredValues(t.text,t.rectTransform.rect.width,0).y<=t.rectTransform.rect.height+1),"expanded details are fully measured without a fixed truncation cap");
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Collapse rows").onClick.Invoke();yield return null;yield return null;
        Check(!panel.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.name.StartsWith("Expanded ")),"collapse rows removes expanded details");
        panel.GetComponentsInChildren<TMP_InputField>().First().text="not-a-matching-city";yield return null;yield return null;
        Check(panel.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.text.StartsWith("No records match")),"empty search results include recovery guidance");
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Clear search").onClick.Invoke();yield return null;yield return null;
        Check(panel.Filters.Search==""&&panel.GetComponentsInChildren<Button>().Any(b=>b.name.StartsWith("Row ")),"clear search restores the ledger");
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Find category").onClick.Invoke();
        panel.HandleKeyDown(KeyEnum.Escape);yield return null;
        Check(panel.IsOpen&&panel.transform.Find("Statistics window/Category choices")==null,"escape dismisses category choices before closing the panel");
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Find category").onClick.Invoke();
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Find Loan").onClick.Invoke();yield return null;yield return null;
        Check(panel.IsOpen&&panel.Filters.Action=="Loan"&&panel.transform.Find("Statistics window/Category choices")==null,"choosing a category applies a typed action scope and closes its menu");
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Clear search").onClick.Invoke();yield return null;yield return null;
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Find category").onClick.Invoke();panel.SelectPage(9);yield return null;yield return null;
        Check(panel.transform.Find("Statistics window/Category choices")==null,"changing pages dismisses stale category choices");
        panel.SelectPage(4);yield return null;yield return null;
        panel.BodyScroll.verticalNormalizedPosition=0;yield return null;
        panel.GetComponentsInChildren<Button>().First(b=>b.name=="Back to top").onClick.Invoke();
        Check(panel.BodyScroll.verticalNormalizedPosition>.995f,"top button returns to the start of the current page");
        var navigation=panel.PageButtons[9].GetComponentInParent<ScrollRect>();navigation.verticalNormalizedPosition=1;
        es.SetSelectedGameObject(panel.PageButtons[9].gameObject);Canvas.ForceUpdateCanvases();
        var navCorners=new Vector3[4];((RectTransform)panel.PageButtons[9].transform).GetWorldCorners(navCorners);
        Check(navigation.viewport.InverseTransformPoint(navCorners[0]).y>=navigation.viewport.rect.yMin-1,"keyboard focus reveals an offscreen navigation item");
        var sort=panel.GetComponentsInChildren<Button>().First(b=>b.name.StartsWith("Sort "));sort.onClick.Invoke();yield return null;
        Check(panel.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.text.Contains(" ↑")),"column sorting is selectable");
        panel.BodyScroll.verticalNormalizedPosition=.37f;panel.SelectPage(9);yield return null;panel.SelectPage(4);yield return null;
        Check(Math.Abs(panel.BodyScroll.verticalNormalizedPosition-.37f)<.015f,"page scroll is retained");
        input.Held=KeyEnum.RightArrow;router.ProcessInput();Check(panel.IsOpen,"underlying board keys cannot escape modal");
        input.Down=KeyEnum.Return;router.ProcessInput();Check(panel.IsOpen,"return does not invoke old results review/exit");
        input.Down=KeyEnum.Escape;router.ProcessInput();Tick(1);yield return null;
        Check(!panel.IsOpen&&router.IsViewVisible<IEndOfGameView>(),"escape restores original results entry context");
        results=(EndOfGameView)router.GetView<IEndOfGameView>();results.transform.Find("SafeArea/Popup/Results statistics").GetComponent<Button>().onClick.Invoke();Tick(1);yield return null;
        panel.Close(false);Tick(1);yield return null;Check(!router.IsViewVisible<IEndOfGameView>(),"explicit board review returns to board");
        var review=(BaseView)router.GetView<IFinishedGameReviewView>();var boardButton=review.GetComponentsInChildren<Button>().First(b=>b.name=="Match statistics");
        var boardEntryBounds=ScreenBounds((RectTransform)boardButton.transform,camera);
        Check(boardButton.GetComponent<CanvasGroup>().alpha==1&&boardEntryBounds.yMin<30&&review.GetComponentsInChildren<Button>().Where(b=>b!=boardButton).All(b=>!ScreenBounds((RectTransform)b.transform,camera).Overlaps(boardEntryBounds)),"board statistics entry returns in the bottom row beside original summary button",new{boardEntryBounds.xMin,boardEntryBounds.yMin,boardEntryBounds.width,boardEntryBounds.height});
        for(int i=0;i<30;i++) {
            boardButton.onClick.Invoke();Tick(.1f);yield return null;panel.Close(false);Tick(.1f);yield return null;
        }
        Check(router.GetView<StatisticsPanel>()==panel&&root.GetComponentsInChildren<StatisticsPanel>(true).Length==1,"reopening reuses one panel and its state");
        Check(example.Repository.Load(example.App.Metadata.GameId)!=null,"panel closing never finishes review or deletes match");
        boardButton.onClick.Invoke();Tick(.1f);yield return null;
        var ownerTable=typeof(StatisticsEntryPoints).GetField("owners",System.Reflection.BindingFlags.Static|System.Reflection.BindingFlags.NonPublic).GetValue(null);var ownerArgs=new object[]{owner,null};ownerTable.GetType().GetMethod("TryGetValue").Invoke(ownerTable,ownerArgs);var reviewSession=ownerArgs[1];var leaseField=reviewSession.GetType().GetField("Presentation");
        Check(leaseField.GetValue(reviewSession)!=null,"E26-LIFECYCLE live owner acquires one presentation lease");
        router.HideView<StatisticsPanel>();Tick(.1f);yield return null;Check(leaseField.GetValue(reviewSession)==null,"E26-LIFECYCLE direct router hide restores background");
        typeof(StatisticsPanel).GetProperty("Filters").SetValue(panel,null,null);bool injected=false;bool samePanel=ReferenceEquals(reviewSession.GetType().GetField("Panel").GetValue(reviewSession),panel);bool wasOpen=panel.IsOpen;
        try{boardButton.onClick.Invoke();}catch(NullReferenceException){injected=true;}
        Check(injected&&leaseField.GetValue(reviewSession)==null&&!router.IsViewCached<StatisticsPanel>(),"E26-LIFECYCLE injected render failure restores presentation and removes broken panel",new{injected,samePanel,wasOpen,lease=leaseField.GetValue(reviewSession)!=null,cached=router.IsViewCached<StatisticsPanel>(),ready=reviewSession.GetType().GetField("Ready").GetValue(reviewSession)});yield return null;
        boardButton.onClick.Invoke();Tick(.1f);yield return null;panel=router.GetView<StatisticsPanel>();Check(panel.IsOpen,"E26-LIFECYCLE reopening after failed show recreates a working panel");panel.Close(false);Tick(.1f);yield return null;
        var staleClick=boardButton.onClick;owner.FinishGameReview();Tick(.1f);yield return null;
        Check(!router.IsViewCached<StatisticsPanel>()&&root.GetComponentsInChildren<StatisticsPanel>(true).Length==0,"review exit clears report UI and cached callbacks");
        staleClick.Invoke();Check(!router.IsViewCached<StatisticsPanel>(),"retired match callback cannot reopen a stale report");
        Destroy(root);Destroy(cameraNode);yield return null;
    }
    static void OpenData(StatisticsPanel panel) {foreach(var b in panel.GetComponentsInChildren<Button>().Where(b=>b.name.StartsWith("Data ")&&b.GetComponentInChildren<TextMeshProUGUI>().text.StartsWith("+ Show data")).ToArray())b.onClick.Invoke();}
    static string PathTo(Transform item,Transform root){return item==root?item.name:PathTo(item.parent,root)+"/"+item.name;}
    static Rect ScreenBounds(RectTransform rect,Camera camera){var c=new Vector3[4];rect.GetWorldCorners(c);var p=c.Select(v=>RectTransformUtility.WorldToScreenPoint(camera,v)).ToArray();return Rect.MinMaxRect(p.Min(v=>v.x),p.Min(v=>v.y),p.Max(v=>v.x),p.Max(v=>v.y));}
    void CheckReadability(StatisticsPanel panel,Canvas canvas,int width,int page)
    {
        Canvas.ForceUpdateCanvases();
        var labels=panel.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>!String.IsNullOrWhiteSpace(t.text)).ToArray();
        Check(labels.All(t=>t.fontSize*canvas.scaleFactor>=17.9f),"visible text uses a readable physical size",new{width,page,scale=panel.TextScale,min=labels.Min(t=>t.fontSize*canvas.scaleFactor)});
        Check(labels.Where(t=>t.transform.parent.name.StartsWith("Sort ")).All(t=>t.GetPreferredValues(t.text,t.rectTransform.rect.width,0).y<=t.rectTransform.rect.height+1),"wrapped column headings fit their height",new{width,page,scale=panel.TextScale});
        foreach(var plot in panel.GetComponentsInChildren<StatisticsPlotGraphic>())CheckAxisLabels(plot,panel);
        var window=(RectTransform)panel.transform.Find("Statistics window");
        Check(window.GetComponentsInChildren<Selectable>().Where(s=>s.transform.parent==window||s.transform.parent.name=="Find statistics").All(s=>{
            var corners=new Vector3[4];((RectTransform)s.transform).GetWorldCorners(corners);
            return corners.All(c=>{var p=window.InverseTransformPoint(c);return p.x>=window.rect.xMin-1&&p.x<=window.rect.xMax+1&&p.y>=window.rect.yMin-1&&p.y<=window.rect.yMax+1;});
        }),"header and footer controls stay inside the responsive window",new{width,page,scale=panel.TextScale});
    }
    void Capture(RenderTexture target,Camera camera,string name)
    {
        if(Environment.GetEnvironmentVariable("BRASS_STATS_FAST")=="1")return; // Diagnostic only; release runs always render captures.
        Canvas.ForceUpdateCanvases();camera.Render();var previousTarget=RenderTexture.active;RenderTexture.active=target;
        var image=new Texture2D(target.width,target.height,TextureFormat.RGB24,false);image.ReadPixels(new Rect(0,0,target.width,target.height),0,0);image.Apply();RenderTexture.active=previousTarget;
        File.WriteAllBytes(Path.Combine(Path.GetDirectoryName(Output),name+".png"),image.EncodeToPNG());Destroy(image);
    }
    static void Tick(float seconds){for(float t=0;t<seconds;t+=.05f)DOTween.ManualUpdate(.05f,.05f);}
    sealed class Input:IInputSource {
        public KeyEnum Down,Held;public bool GetKeyDown(KeyEnum k){return k==Down;}public bool GetKey(KeyEnum k){return k==Held;}public bool GetKeyUp(KeyEnum k){return false;}
    }
    sealed class Builder:IUseCaseBuilder {
        public FinishedGameReviewUseCase Owner;
        public void BuildUseCases(){}public void RestoreUseCases(){}public void RestoreUseCaseOfType(Type t){}
        public IController GetController(Type t){if(typeof(IFinishedGameReviewView).IsAssignableFrom(t))return new FinishedGameReviewController(Owner);if(typeof(IEndOfGameView).IsAssignableFrom(t))return new EndOfGameController();return null;}
        public T GetUseCase<T>()where T:IUseCase{throw new NotSupportedException();}
    }
    sealed class Presenter:IFinishedGameReviewOutput {
        readonly Router _router;readonly FinishedGameReviewPresenter original;
        public Presenter(Router router){_router=router;original=new FinishedGameReviewPresenter(router);}
        public void ReplayUnseenActions(PlayerMetadata player,IReplayDelegate callback){}
        public void ShowSkipToMyNextTurnView(){}public void HidePlayerActionView(){}
        public void ShowButtonToLaunchGameSummaryTable(){original.ShowButtonToLaunchGameSummaryTable();}
        public void ShowGameSummaryTable(List<(PlayerColor,PlayerAvatar,string,int,int,int,int,int,int)> scores){original.ShowGameSummaryTable(scores);}
        public void ShowIntroductoryGameSummaryTable(List<(PlayerColor,PlayerAvatar,string,int,int,int,int,int,int,int,int)> scores){original.ShowIntroductoryGameSummaryTable(scores);}
    }
}
